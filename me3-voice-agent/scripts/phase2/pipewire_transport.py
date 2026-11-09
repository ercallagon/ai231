"""Isolated native PipeWire transport prototype. Explicit nodes; no device fallback.

Only start() opens devices. Unit tests substitute process boundaries. No AEC is
implemented here: select the reviewed AEC source/reference sink upstream.
"""

import asyncio

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import InputAudioRawFrame, InterruptionFrame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from me3_voice.pi_audio import reap


async def open_owned(factory, *args, **kwargs):
    opening = asyncio.create_task(factory(*args, **kwargs))
    try:
        return await asyncio.shield(opening)
    except asyncio.CancelledError:
        child = await opening
        await reap(child)
        raise


class PipeWireInput(BaseInputTransport):
    def __init__(self, node, *, allow_capture=False, spawn=asyncio.create_subprocess_exec):
        if not node or node.startswith("-"):
            raise ValueError("explicit capture node required")
        super().__init__(TransportParams(audio_in_enabled=True, audio_in_sample_rate=16000))
        self.node, self.allow_capture, self.spawn = node, allow_capture, spawn
        self.child = self.reader = None
        self.resampler = create_stream_resampler(clear_after_secs=None)
        self.failure = None

    async def start(self, frame):
        if not self.allow_capture:
            raise RuntimeError(
                "Capture disabled; use text/mock mode until explicit device approval"
            )
        await super().start(frame)
        self.child = await open_owned(
            self.spawn,
            "pw-record",
            f"--target={self.node}",
            "--raw",
            "--format=s16",
            "--rate=48000",
            "--channels=1",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self.set_transport_ready(frame)
        self.reader = asyncio.create_task(self.capture())

    async def capture(self):
        try:
            while True:
                pcm = await asyncio.wait_for(self.child.stdout.readexactly(1920), 3)
                # One persistent conversion; BaseInputTransport does not resample.
                normalized = await self.resampler.resample(pcm, 48000, 16000)
                if normalized:
                    if self._audio_in_queue.qsize() >= 4:
                        raise RuntimeError(
                            "capture admission backlog; fail instead of accumulating stale audio"
                        )
                    await self.push_audio_frame(InputAudioRawFrame(normalized, 16000, 1))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = type(exc).__name__
            await self.push_error(error_msg="PipeWire capture unavailable; degraded mode required")
        finally:
            await reap(self.child)

    async def close_capture(self):
        await reap(self.child)
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
            self.reader = None
        self.child = None

    async def stop(self, frame):
        await self.close_capture()
        await super().stop(frame)

    async def cancel(self, frame):
        await self.close_capture()
        await super().cancel(frame)

    async def cleanup(self):
        await self.close_capture()
        await super().cleanup()


class PipeWireOutput(BaseOutputTransport):
    def __init__(self, node, *, allow_playback=False, spawn=asyncio.create_subprocess_exec):
        if not node or node.startswith("-"):
            raise ValueError("explicit reference sink required")
        super().__init__(
            TransportParams(
                audio_out_enabled=True, audio_out_sample_rate=16000, audio_out_channels=1
            )
        )
        self.node, self.allow_playback, self.spawn = node, allow_playback, spawn
        self.child = None
        self.lock = asyncio.Lock()
        self.enabled = False
        self.generation = 0
        self.failure = None

    async def start(self, frame):
        if not self.allow_playback:
            raise RuntimeError(
                "Playback disabled; use text/mock mode until explicit device approval"
            )
        self.enabled = True
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        generation = self.generation
        async with self.lock:
            if not self.enabled or generation != self.generation:
                return False
            if self.child is None:
                self.child = await open_owned(
                    self.spawn,
                    "pw-play",
                    f"--target={self.node}",
                    "--raw",
                    "--format=s16",
                    "--rate=16000",
                    "--channels=1",
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            if not self.enabled or generation != self.generation:
                await reap(self.child)
                self.child = None
                return False
            try:
                self.child.stdin.write(frame.audio)
                await asyncio.wait_for(self.child.stdin.drain(), 2)
                return generation == self.generation
            except Exception as exc:
                self.failure = type(exc).__name__
                self.enabled = False
                await reap(self.child)
                self.child = None
                return False

    async def halt(self, *, disable=False):
        self.generation += 1
        if disable:
            self.enabled = False
        # Drop owned device output before awaiting normal queue cancellation.
        await reap(self.child)
        async with self.lock:
            await reap(self.child)
            self.child = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            await self.halt()
        await super().process_frame(frame, direction)

    async def stop(self, frame):
        await self.halt(disable=True)
        await super().stop(frame)

    async def cancel(self, frame):
        await self.halt(disable=True)
        await super().cancel(frame)

    async def cleanup(self):
        await self.halt(disable=True)
        await super().cleanup()
