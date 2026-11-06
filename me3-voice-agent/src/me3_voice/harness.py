"""Pipecat input adapters; Session remains the conversation/task lifecycle owner."""

import asyncio
from dataclasses import dataclass, field

from pipecat.frames.frames import ControlFrame, EndFrame, Frame, InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner


@dataclass
class CaptureFrame(InputAudioRawFrame):
    sequence: int = 0
    completion: asyncio.Future | None = field(default=None, repr=False)


@dataclass
class InputEndFrame(ControlFrame):
    completion: asyncio.Future | None = field(default=None, repr=False)


class StreamingInputProcessor(FrameProcessor):
    """Ingest PCM without awaiting the generated answer on each endpoint.

    The upstream adapter allows one outstanding request, bounding Pipecat's
    otherwise unbounded input queues. EOF alone drains the final answer.
    """

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.failed = False
        self.frames = 0
        self.active = None

    async def ingest(self, frame):
        if self.failed:
            raise RuntimeError("audio pipeline failed; open a new input adapter")
        if isinstance(frame, CaptureFrame):
            pre = self.application.pre
            if frame.sample_rate != pre.sample_rate or frame.num_channels != pre.channels:
                raise ValueError("capture format changed; open a new input adapter")
            await self.application.feed(frame.audio, frame.sequence)
            self.frames += 1
            return None
        return await self.application.finish()

    async def abort(self):
        self.failed = True
        if self.active:
            self.active.cancel()
            await asyncio.gather(self.active, return_exceptions=True)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (CaptureFrame, InputEndFrame)
        ):
            try:
                self.active = asyncio.create_task(self.ingest(frame))
                result = await self.active
            except asyncio.CancelledError:
                self.failed = True
                if frame.completion and not frame.completion.done():
                    frame.completion.cancel()
                if asyncio.current_task().cancelling():
                    raise
            except Exception as exc:
                self.failed = True
                if frame.completion and not frame.completion.done():
                    frame.completion.set_exception(exc)
            else:
                if frame.completion and not frame.completion.done():
                    frame.completion.set_result(result)
            finally:
                self.active = None
            # Audio is consumed here. It must not also reach another ASR or
            # reasoning processor and execute the same turn twice.
            return
        await self.push_frame(frame, direction)


class PipecatAudioInput:
    """Replaceable native-PCM ingress, not a microphone or speaker transport.

    feed() acknowledges ingestion, never answer completion. Calls serialize
    with backpressure; failures/cancellation close this adapter to further input.
    Providers, Session, MCP and the playback sink remain independent of Pipecat.
    """

    def __init__(self, application, *, timeout=60):
        if not 0 < timeout <= 120:
            raise ValueError("input deadline must be within 0..120 seconds")
        self.application, self.timeout = application, timeout
        self.processor = StreamingInputProcessor(application)
        self.worker = PipelineWorker(
            Pipeline([self.processor]),
            params=PipelineParams(
                audio_in_sample_rate=application.pre.sample_rate, audio_out_sample_rate=16000
            ),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        self.runner = WorkerRunner(handle_sigint=False)
        self.task = None
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.closed = False
        self.finished = False
        self.faulted = False

        @self.worker.event_handler("on_pipeline_started")
        async def started(worker, frame):
            self.ready.set()

    async def __aenter__(self):
        if self.task or self.closed:
            raise RuntimeError("audio input adapter cannot be reopened")
        await self.runner.add_workers(self.worker)
        self.task = asyncio.create_task(self.runner.run())
        try:
            await self._wait(self.ready.wait())
        except BaseException:
            await self.__aexit__(True, None, None)
            raise
        return self

    async def _wait(self, awaitable):
        pending = asyncio.ensure_future(awaitable)
        try:
            async with asyncio.timeout(self.timeout):
                done, _ = await asyncio.wait(
                    (pending, self.task), return_when=asyncio.FIRST_COMPLETED
                )
                if pending in done:
                    return pending.result()
                self.task.result()
                raise RuntimeError("Pipecat audio pipeline stopped before acknowledgment")
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def _request(self, frame):
        if not self.task or self.closed or self.finished or self.faulted:
            raise RuntimeError("audio input adapter is not accepting input")
        frame.completion = asyncio.get_running_loop().create_future()
        try:
            await self.worker.queue_frame(frame)
            return await self._wait(frame.completion)
        except BaseException:
            self.faulted = True
            self.processor.failed = True
            raise

    async def feed(self, pcm, sequence):
        async with self.lock:
            pre = self.application.pre
            # Validate before queueing large/malformed objects in the framework.
            if (
                not isinstance(pcm, bytes)
                or not pcm
                or len(pcm) % (2 * pre.channels)
                or len(pcm) > pre.sample_rate * pre.channels * 2
                or type(sequence) is not int
                or sequence != pre.sequence
            ):
                raise ValueError("expected ordered complete PCM16 frames, at most one second")
            await self._request(
                CaptureFrame(
                    audio=pcm,
                    sample_rate=pre.sample_rate,
                    num_channels=pre.channels,
                    sequence=sequence,
                )
            )

    async def finish(self):
        async with self.lock:
            result = await self._request(InputEndFrame())
            self.finished = True
            return result

    async def __aexit__(self, exc_type, *exc):
        self.closed = True
        if self.task:
            await self.processor.abort()
            if exc_type or self.faulted or not self.finished:
                await self.application.session.interrupt()
            # Cancel the input worker before returning ownership to the caller.
            if self.finished and not exc_type and not self.faulted:
                await self.worker.stop_when_done()
            else:
                await self.runner.cancel()
            async with asyncio.timeout(10):
                await self.task


@dataclass
class UtteranceFrame(Frame):
    pcm: bytes = b""


class SessionProcessor(FrameProcessor):
    def __init__(self, session):
        super().__init__()
        self.session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UtteranceFrame) and direction == FrameDirection.DOWNSTREAM:
            task = await self.session.submit(frame.pcm)
            await task
        await self.push_frame(frame, direction)


async def run_utterances(session, utterances):
    """Ordered replay compatibility spike, not a live mic transport.

    Barge-in uses Session.interrupt directly, independent of queued data frames.
    A future device transport must wire its confirmed onset into that method.
    """
    worker = PipelineWorker(
        Pipeline([SessionProcessor(session)]),
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
    )
    await worker.queue_frames([*(UtteranceFrame(pcm=pcm) for pcm in utterances), EndFrame()])
    async with asyncio.timeout(60):
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await runner.run()
