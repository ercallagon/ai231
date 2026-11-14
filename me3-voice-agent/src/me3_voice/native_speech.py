"""Thin native Pipecat speech adapters; inference remains inside our MCP servers.

No conversation history, LLM/tool loop, audio queue or endpoint classifier here.
VAD supplies speech boundaries; the native user aggregator owns accepted turns.
"""

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import math
import re
import time
import uuid

from pipecat.frames.frames import (
    CancelFrame,
    ControlFrame,
    EndFrame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from .contracts import FRAME_BYTES, TurnKey, encode_pcm


def spoken_text(text):
    """Remove presentation syntax at the speech boundary; retain UI text as-is."""
    value = str(text or "")
    # Code is useful on screen but generally unreadable aloud. Keep a short
    # truthful cue instead of speaking fence and programming punctuation.
    value = re.sub(r"```[^\n]*\n.*?```", " Code is shown on screen. ", value, flags=re.S)
    value = re.sub(r"`([^`\n]+)`", r"\1", value)
    value = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s+", "", value, flags=re.M)
    value = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", value, flags=re.M)
    value = re.sub(r"(\*\*|__|\*|_|~~)(.*?)\1", r"\2", value, flags=re.S)
    value = value.replace("**", "").replace("__", "").replace("~~", "")
    value = re.sub(r"(?<=\d)\s*%", " percent", value)
    value = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


@dataclass
class _CommitSpeechFrame(ControlFrame):
    generation: int = 0


class McpSTTService(STTService):
    """Streaming append + cancellable finish, with operation identity fencing.

    A native ordered control frame commits audio already admitted before VAD
    stop. Finishing never blocks the SystemFrame/control path. Interruption from
    the current VAD onset preserves that new input; old input is invalidated at
    onset. Batch selection is opt-in; 32ms remains the conservative default.
    """

    def __init__(
        self,
        speech,
        *,
        session_id=None,
        batch_ms=32,
        poll_interval=0.1,
        on_event=None,
        on_partial=None,
    ):
        if batch_ms not in (32, 64, 96):
            raise ValueError("batch_ms must be 32,64,96")
        if not 0.02 <= poll_interval <= 1:
            raise ValueError("poll_interval must be 0.02..1 seconds")
        super().__init__(
            sample_rate=16000, settings=STTSettings(model="project-mcp", language=None)
        )
        self.speech = speech
        self.session_id = session_id or uuid.uuid4().hex
        self.batch_bytes = batch_ms * 32
        self.poll_interval = poll_interval
        self.on_event = on_event
        self.on_partial = on_partial
        self.operation = None
        self.key = None
        self.generation = self.turn = self.sequence = self.revision = 0
        self.speaking = self.closing = False
        self.pending = bytearray()
        self.finalizer = self.poller = None
        self.append_worker = None
        self.append_queue = None
        self.append_error = None
        self.append_exception = None
        self.queued_sequence = 0
        self.rejected_audio_bytes = 0
        # Eight pre-onset frames occupy capacity before the first append is
        # ready. Forty queued 32ms frames therefore preserve about one second
        # of post-VAD startup headroom (plus one in-flight append) while still
        # failing explicitly on longer provider stalls.
        self.append_queue_limit = 40
        self.io = asyncio.Lock()
        self.last_append = 0
        self.last_audio_at = None
        # Pipecat forwards frames before its VAD start confirmation (160 ms in
        # native_vad). Keep only a short bounded onset ring until VAD admits a
        # turn; it is submitted only after that event, never while asleep.
        self.onset_buffer_frames = 8  # 256 ms at 32 ms/frame.
        self.onset_buffer = deque(maxlen=self.onset_buffer_frames)
        self.provider_diagnostics_signature = None

    @staticmethod
    def stream_ref(identifier):
        return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]

    def event(self, name, **values):
        if self.on_event:
            values.setdefault("response_id", self.response_context_id)
            self.on_event({"event": name, "at": time.monotonic(), **values})

    def valid(self, identifier, generation):
        return not self.closing and self.operation == identifier and self.generation == generation

    async def release(self, identifier):
        # Operation fencing is immediate; unreachable MCP cleanup must not hold
        # native control routing indefinitely. Child exit is the final fence.
        try:
            async with asyncio.timeout(0.25):
                result = await self.speech.release("asr", identifier)
                if isinstance(result, dict):
                    self.emit_provider_diagnostics(result.get("provider_diagnostics"), identifier)
                self.event(
                    "asr_closed",
                    stream_id=self.stream_ref(identifier),
                    cancelled=bool(result.get("cancelled")) if isinstance(result, dict) else None,
                )
        except TimeoutError:
            self.event("asr_release_timeout", stream_id=self.stream_ref(identifier))

    async def abort(self, control_frame=None, direction=None):
        if self.on_partial:
            self.on_partial(None)
        self.generation += 1
        identifier, self.operation = self.operation, None
        queued = self.append_queue.qsize() if self.append_queue is not None else 0
        self.pending.clear()
        queue, self.append_queue = self.append_queue, None
        worker, self.append_worker = self.append_worker, None
        if queue is not None:
            while not queue.empty():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                    queue.task_done()
        if control_frame is not None:
            # Forward native control before awaiting any remote resource release.
            await super().process_frame(control_frame, direction)
        tasks = [
            task
            for task in (self.finalizer, self.poller, worker)
            if task is not None and task is not asyncio.current_task()
        ]
        self.finalizer = self.poller = None
        self.append_error = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if identifier:
            await self.release(identifier)
        self.event(
            "asr_abort",
            stream_id=self.stream_ref(identifier) if identifier else None,
            queued_frames=queued,
            invalidated_generation=self.generation,
        )

    def partial(self, result):
        if not result:
            return None
        if (
            result.get("final") is not False
            or type(result.get("revision")) is not int
            or result["revision"] <= self.revision
            or not isinstance(result.get("text"), str)
        ):
            raise ValueError("invalid native ASR partial")
        self.revision = result["revision"]
        self.event("asr_partial", revision=self.revision)
        return InterimTranscriptionFrame(result["text"], "user", "")

    async def emit_partial(self, row, identifier, generation):
        if not self.valid(identifier, generation):
            self.event("asr_stale_rejected")
            return
        self.speech.check_key(self.key, row)
        frame = self.partial(row.get("partial"))
        if frame:
            if self.on_partial:
                self.on_partial(frame.text)
            await self.push_frame(frame)

    async def append(self, pcm, identifier, generation):
        use_batch = len(pcm) > FRAME_BYTES
        if use_batch and self.speech.capabilities.get("asr", {}).get("contract_version", 1) < 2:
            raise ValueError("ASR batch requires speech MCP contract v2")
        name = "asr_append_batch" if use_batch else "asr_append"
        args = {"operation_id": identifier, "sequence": self.sequence, "pcm_b64": encode_pcm(pcm)}
        started = time.monotonic()
        self.event(
            "asr_append_rpc_started",
            stream_id=self.stream_ref(identifier),
            utterance_id=self.turn,
            sequence=self.sequence,
            bytes=len(pcm),
        )
        try:
            row = await self.speech.call("asr", name, args)
        except Exception as exc:
            if not self.valid(identifier, generation):
                self.event("asr_stale_rejected")
                return
            self.event(
                "asr_append_rpc_failed",
                stream_id=self.stream_ref(identifier),
                utterance_id=self.turn,
                sequence=self.sequence,
                seconds=time.monotonic() - started,
                error_code=getattr(exc, "code", "mcp_transport_error"),
                error_stage=getattr(exc, "error_stage", None) or "asr_append",
                error_boundary=getattr(exc, "error_boundary", None) or "speech_client",
                exception_type=type(exc).__name__,
            )
            raise
        if not self.valid(identifier, generation):
            self.event("asr_stale_rejected")
            return
        self.sequence += 1
        if row.get("next_sequence") != self.sequence:
            raise ValueError("native ASR sequence mismatch")
        self.last_append = time.monotonic()
        self.event(
            "asr_append",
            stream_id=self.stream_ref(identifier),
            utterance_id=self.turn,
            sequence=self.sequence - 1,
            bytes=len(pcm),
            seconds=self.last_append - started,
        )
        self.event(
            "asr_append_rpc_completed",
            stream_id=self.stream_ref(identifier),
            utterance_id=self.turn,
            sequence=self.sequence - 1,
            seconds=self.last_append - started,
        )
        self.emit_provider_diagnostics(row.get("provider_diagnostics"), identifier)
        await self.emit_partial(row, identifier, generation)

    def emit_provider_diagnostics(self, value, identifier):
        """Relay only the fixed, secret-free ASR transport snapshot schema."""
        if not isinstance(value, dict):
            return
        allowed_numbers = {
            "connection_attempt_monotonic",
            "provider_request_reserved_monotonic",
            "websocket_handshake_monotonic",
            "connection_ready_monotonic",
            "connection_failed_monotonic",
            "socket_closed_monotonic",
            "first_append_monotonic",
            "first_send_started_monotonic",
            "first_send_completed_monotonic",
        }
        allowed_counts = {
            "accepted_audio_bytes",
            "sent_audio_bytes",
            "queued_audio_bytes",
            "peak_queued_audio_bytes",
            "discarded_audio_bytes",
            "queue_limit_bytes",
        }
        state = value.get("connection_state")
        if not isinstance(state, str) or state not in {
            "not_started",
            "connecting",
            "ready",
            "failed",
            "closed",
        }:
            return
        row = {"connection_state": state}
        for name in allowed_numbers:
            number = value.get(name)
            if number is not None:
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    return
                if not math.isfinite(number) or number < 0:
                    return
                row[name] = number
        for name in allowed_counts:
            count = value.get(name)
            if type(count) is not int or count < 0:
                return
            row[name] = count
        stages = value.get("connection_stages")
        if isinstance(stages, dict):
            time_stages = {
                "dns_started",
                "dns_completed",
                "dns_cache_hit",
                "dns_cache_miss",
                "connection_pool_wait_started",
                "connection_pool_wait_completed",
                "connection_establishment_started",
                "connection_established",
                "websocket_upgrade_request_sent",
                "websocket_http_response_received",
                "websocket_http_status",
                "connection_failed_monotonic",
                "connection_cancelled_monotonic",
            }
            safe_stages = {}
            for name, stage_value in stages.items():
                if (
                    name in time_stages
                    and isinstance(stage_value, (int, float))
                    and not isinstance(stage_value, bool)
                ):
                    if math.isfinite(stage_value) and stage_value >= 0:
                        safe_stages[name] = stage_value
                elif name == "connection_terminal_exception" and isinstance(stage_value, str):
                    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", stage_value):
                        safe_stages[name] = stage_value
                elif name == "request_outcome" and stage_value in {
                    "success",
                    "failure",
                    "unknown",
                }:
                    safe_stages[name] = stage_value
                elif name == "request_error_code" and isinstance(stage_value, str):
                    if re.fullmatch(r"[a-z0-9_]{1,48}", stage_value):
                        safe_stages[name] = stage_value
            if safe_stages:
                row["connection_stages"] = safe_stages
        code = value.get("failure_code")
        if code is not None:
            if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,48}", code):
                return
            row["failure_code"] = code
        signature = (identifier, *sorted(row.items()))
        if signature == self.provider_diagnostics_signature:
            return
        self.provider_diagnostics_signature = signature
        self.event(
            "asr_provider_transport",
            stream_id=self.stream_ref(identifier),
            utterance_id=self.turn,
            **row,
        )

    async def run_stt(self, audio):
        if not self.speaking or self.closing:
            return
        if len(audio) % 2:
            raise ValueError("native ASR requires PCM16")
        self.last_audio_at = time.monotonic()
        # This method runs in Pipecat's audio path. It may create the MCP
        # operation once, but it must never take ``io``: that lock is held by
        # the append worker while MCP/network I/O is pending. Admission is
        # serialized by Pipecat's processor; remote operations are serialized
        # separately by the worker and poller.
        generation = self.generation
        if self.operation is None:
            self.turn += 1
            key = TurnKey(session_id=self.session_id, turn_id=self.turn)
            open_started = time.monotonic()
            self.event("asr_open_started", utterance_id=self.turn)
            opened = await self.speech.open_operation("asr", "asr_open", key.model_dump())
            identifier = opened["operation_id"]
            if generation != self.generation or self.closing:
                await self.release(identifier)
                return
            self.speech.check_key(key, opened)
            self.operation, self.key = identifier, key
            self.sequence = self.queued_sequence = self.revision = 0
            self.append_error = None
            self.append_exception = None
            self.rejected_audio_bytes = 0
            self.append_queue = asyncio.Queue(maxsize=self.append_queue_limit)
            self.event(
                "asr_opened",
                stream_id=self.stream_ref(identifier),
                utterance_id=self.turn,
                seconds=time.monotonic() - open_started,
            )
            self.emit_provider_diagnostics(opened.get("provider_diagnostics"), identifier)
            self.append_worker = asyncio.create_task(
                self._append_loop(identifier, generation, self.append_queue)
            )
            if "asr_events" in self.speech.tools.get("asr", set()):
                self.poller = asyncio.create_task(self.poll(identifier, generation))
        identifier = self.operation
        if self.append_error:
            self.rejected_audio_bytes += len(audio)
            self.event(
                "asr_audio_rejected_after_error",
                stream_id=self.stream_ref(identifier),
                utterance_id=self.turn,
                bytes=len(audio),
                error_code=self.append_error,
            )
            return
        self.pending.extend(audio)
        while len(self.pending) >= self.batch_bytes or (self.queued_sequence == 0 and self.pending):
            count = min(
                len(self.pending), FRAME_BYTES if self.queued_sequence == 0 else self.batch_bytes
            )
            pcm = bytes(self.pending[:count])
            del self.pending[:count]
            queue = self.append_queue
            try:
                queue.put_nowait(pcm)
                self.event(
                    "asr_append_queued",
                    stream_id=self.stream_ref(identifier),
                    utterance_id=self.turn,
                    sequence=self.queued_sequence,
                    bytes=len(pcm),
                    queue_depth=queue.qsize(),
                    queue_limit=queue.maxsize,
                )
                self.queued_sequence += 1
            except asyncio.QueueFull:
                self.append_error = "queue_saturated"
                self.append_exception = None
                self.rejected_audio_bytes += len(pcm) + len(self.pending)
                rejected = len(pcm) + len(self.pending)
                self.pending.clear()
                self.event(
                    "asr_append_queue_saturated",
                    stream_id=self.stream_ref(identifier),
                    utterance_id=self.turn,
                    depth=queue.qsize(),
                    queue_limit=queue.maxsize,
                    rejected_bytes=rejected,
                )
                return
        if False:
            yield  # Base STTService consumes an async generator; frames are pushed as events.

    async def _append_loop(self, identifier, generation, queue):
        try:
            while self.valid(identifier, generation):
                pcm = await queue.get()
                append_sequence = self.sequence
                started = time.monotonic()
                self.event(
                    "asr_append_started",
                    stream_id=self.stream_ref(identifier),
                    utterance_id=self.turn,
                    sequence=append_sequence,
                    bytes=len(pcm),
                    queue_depth=queue.qsize(),
                )
                try:
                    if self.valid(identifier, generation):
                        async with self.io:
                            await self.append(pcm, identifier, generation)
                finally:
                    self.event(
                        "asr_append_worker_finished",
                        stream_id=self.stream_ref(identifier),
                        utterance_id=self.turn,
                        sequence=append_sequence,
                        seconds=time.monotonic() - started,
                        queue_depth=max(0, queue.qsize() - 1),
                    )
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.append_error = getattr(exc, "code", None) or type(exc).__name__
            self.append_exception = exc
            cause = exc.__cause__ or exc.__context__
            self.event(
                "asr_append_failed",
                stream_id=self.stream_ref(identifier),
                utterance_id=self.turn,
                exception_type=type(exc).__name__,
                error_code=getattr(exc, "code", "append_failed"),
                error_stage=getattr(exc, "error_stage", None) or "mcp_append",
                error_boundary=getattr(exc, "error_boundary", None) or "speech_client",
                cause_type=getattr(exc, "remote_cause_type", None)
                or getattr(exc, "cause_type", None)
                or (type(cause).__name__ if cause else None),
                provider_exception_type=getattr(exc, "error_detail", None),
                remote_cause_type=getattr(exc, "remote_cause_type", None),
                queue_depth=queue.qsize(),
            )
            while not queue.empty():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                    queue.task_done()
            # Let the native VAD continue receiving frames. The turn is failed
            # at VAD stop/commit, so a transport error cannot suppress the
            # stop boundary and confuse an I/O failure with speech endpointing.

    async def poll(self, identifier, generation):
        try:
            while self.valid(identifier, generation):
                await asyncio.sleep(self.poll_interval)
                if self.finalizer or time.monotonic() - self.last_append < self.poll_interval:
                    continue
                async with self.io:
                    if self.valid(identifier, generation):
                        row = await self.speech.call(
                            "asr", "asr_events", {"operation_id": identifier}
                        )
                        self.emit_provider_diagnostics(row.get("provider_diagnostics"), identifier)
                        await self.emit_partial(row, identifier, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.valid(identifier, generation):
                self.append_error = getattr(exc, "code", "asr_event_stream_failed")
                self.append_exception = exc
                self.event(
                    "asr_events_failed",
                    stream_id=self.stream_ref(identifier),
                    utterance_id=self.turn,
                    exception_type=type(exc).__name__,
                    error_code=self.append_error,
                    error_stage=getattr(exc, "error_stage", None) or "asr_events",
                    error_boundary=getattr(exc, "error_boundary", None) or "speech_client",
                )

    async def commit(self, generation):
        if generation != self.generation or (self.finalizer and not self.finalizer.done()):
            return
        self.speaking = False
        self.onset_buffer.clear()
        identifier, key = self.operation, self.key
        if not identifier:
            return
        if self.poller:
            self.poller.cancel()
            await asyncio.gather(self.poller, return_exceptions=True)
            self.poller = None
        queue = self.append_queue
        if self.pending:
            try:
                queue.put_nowait(bytes(self.pending))
                self.pending.clear()
            except asyncio.QueueFull:
                self.append_error = "queue_saturated"
        if queue:
            flush_started = time.monotonic()
            self.event(
                "asr_flush_started",
                stream_id=self.stream_ref(identifier),
                queued_frames=queue.qsize(),
            )
            await queue.join()
            self.event(
                "asr_flush_completed",
                stream_id=self.stream_ref(identifier),
                seconds=time.monotonic() - flush_started,
                accepted_frames=self.queued_sequence,
                completed_frames=self.sequence,
            )
        worker, self.append_worker = self.append_worker, None
        if worker:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        self.append_queue = None
        if self.append_error:
            await self.release(identifier)
            self.operation = None
            self.event(
                "asr_turn_aborted",
                stream_id=self.stream_ref(identifier),
                utterance_id=key.turn_id,
                error_code=self.append_error,
                exception_type=(
                    type(self.append_exception).__name__ if self.append_exception else None
                ),
                error_stage=getattr(self.append_exception, "error_stage", None) or "asr_append",
                error_boundary=getattr(self.append_exception, "error_boundary", None),
                provider_exception_type=getattr(self.append_exception, "error_detail", None),
                accepted_frames=self.queued_sequence,
                completed_frames=self.sequence,
                rejected_audio_bytes=self.rejected_audio_bytes,
            )
            await self.push_error(
                error_msg="ASR append failed before finalization",
                exception=self.append_exception,
            )
            return
        self.event(
            "asr_finish_requested",
            stream_id=self.stream_ref(identifier),
            utterance_id=key.turn_id,
            accepted_frames=self.queued_sequence,
            completed_frames=self.sequence,
        )
        self.finalizer = asyncio.create_task(self.finish(identifier, key, generation))

    async def finish(self, identifier, key, generation):
        started = time.monotonic()
        try:
            row = await self.speech.call("asr", "asr_finish", {"operation_id": identifier})
            if not self.valid(identifier, generation):
                self.event("asr_stale_rejected")
                return
            self.speech.check_key(key, row)
            self.emit_provider_diagnostics(row.get("provider_diagnostics"), identifier)
            if (
                row.get("final") is not True
                or not isinstance(row.get("text"), str)
                or type(row.get("revision")) is not int
                or row["revision"] <= self.revision
            ):
                raise ValueError("invalid native ASR final")
            self.event(
                "asr_final",
                stream_id=self.stream_ref(identifier),
                utterance_id=key.turn_id,
                seconds=time.monotonic() - started,
            )
            if self.on_partial:
                self.on_partial(None)
            # Finalized is an ASR-completion marker, not a second turn endpoint.
            await self.push_frame(TranscriptionFrame(row["text"], "user", "", finalized=True))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.valid(identifier, generation):
                cause = exc.__cause__ or exc.__context__
                self.event(
                    "asr_error",
                    stream_id=self.stream_ref(identifier),
                    utterance_id=key.turn_id,
                    exception_type=type(exc).__name__,
                    error_code=getattr(exc, "code", "asr_finalize_failed"),
                    error_stage=getattr(exc, "error_stage", None) or "asr_finish",
                    error_boundary=getattr(exc, "error_boundary", None),
                    cause_type=getattr(exc, "remote_cause_type", None)
                    or getattr(exc, "cause_type", None)
                    or (type(cause).__name__ if cause else None),
                )
                await self.push_error(
                    error_msg="ASR finalization failed",
                    exception=exc if isinstance(exc, Exception) else None,
                )
        finally:
            await self.release(identifier)
            if self.operation == identifier:
                self.operation = None

    async def process_frame(self, frame, direction):
        if isinstance(frame, VADUserStartedSpeakingFrame):
            onset_audio = b"".join(self.onset_buffer)
            self.onset_buffer.clear()
            self.event(
                "vad_speech_started",
                utterance_id=self.turn + 1,
                previous_stream_id=(self.stream_ref(self.operation) if self.operation else None),
                onset_buffer_frames=len(onset_audio) // FRAME_BYTES,
            )
            self.speaking = True
            await self.abort(frame, direction)
            if onset_audio:
                self.event(
                    "asr_onset_audio_submitted",
                    utterance_id=self.turn + 1,
                    frames=len(onset_audio) // FRAME_BYTES,
                    bytes=len(onset_audio),
                )
                await self.process_generator(self.run_stt(onset_audio))
            return
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            idle_gap = time.monotonic() - self.last_audio_at if self.last_audio_at else None
            self.event(
                "vad_speech_stopped",
                utterance_id=self.turn,
                idle_gap_seconds=idle_gap,
                idle_watchdog_triggered=idle_gap is not None and idle_gap >= 1.0,
            )
            await self.queue_frame(_CommitSpeechFrame(self.generation), direction)
        elif isinstance(frame, _CommitSpeechFrame):
            await self.commit(frame.generation)
            return
        elif isinstance(frame, InterruptionFrame):
            if not self.speaking:
                self.onset_buffer.clear()
                await self.abort(frame, direction)
                return
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self.closing = True
            self.speaking = False
            self.onset_buffer.clear()
            await self.abort()
        await super().process_frame(frame, direction)

    async def process_audio_frame(self, frame, direction):
        # Capture at Pipecat's actual STT audio hook, before its default method
        # calls run_stt. Those frames precede VADUserStartedSpeakingFrame.
        if not self.speaking and not self.closing:
            if frame.audio and len(frame.audio) % 2 == 0:
                self.onset_buffer.append(bytes(frame.audio))
            return
        await super().process_audio_frame(frame, direction)

    async def cleanup(self):
        self.closing = True
        await self.abort()
        await super().cleanup()


class McpTTSService(TTSService):
    """Native text aggregation/output owns sequencing; MCP owns inference/cancel."""

    def __init__(self, speech, *, session_id=None, on_event=None):
        # Pipecat's 3 s empty-context timer can expire while a valid MCP
        # tts_next call is still waiting for Aura's first PCM. Let the MCP
        # call's configured deadline report the actual failure first.
        tts_deadline = getattr(speech, "timeout", 5)
        backends = getattr(speech, "backends", None)
        if backends and "tts" in backends:
            tts_deadline = min(tts_deadline, backends["tts"].timeout_seconds)
        super().__init__(
            sample_rate=16000,
            settings=TTSSettings(model="project-mcp", voice=None, language=None),
            stop_frame_timeout_s=max(3.0, tts_deadline + 1.0),
            # Pipecat reuses one context ID for sentences in an LLM response.
            # Let the base class open its audio context once, but close it only
            # from on_turn_context_completed below. Its idle-stop timer is not
            # a reliable response boundary for a streamed multi-sentence answer.
            # The downstream BaseOutputTransport has a separate three-second
            # empty-queue fallback which may emit BotStopped/BotStarted when
            # provider PCM is genuinely absent that long; that is an output
            # activity pause, not a TTS response-finalization frame.
            push_start_frame=True,
            push_stop_frames=False,
        )
        self.speech = speech
        self.session_id = session_id or uuid.uuid4().hex
        self.turn = 0
        self.on_event = on_event
        self.response_context_id = None
        self.response_has_audio = False
        self.response_stop_emitted = False

    def event(self, name, **values):
        if self.on_event:
            self.on_event({"event": name, "at": time.monotonic(), **values})

    async def on_turn_context_created(self, context_id):
        self.response_context_id = context_id
        self.response_has_audio = False
        self.response_stop_emitted = False
        self.event("tts_response_started", response_id=context_id)

    async def on_turn_context_completed(self):
        """Emit exactly one terminal stop at the Pipecat response boundary.

        `run_tts` is called once per sentence. A stop there makes the assistant
        appear to stop between sentences. Pipecat's automatic idle stop can
        also occur before `LLMFullResponseEndFrame`; disable that as the
        authority and close the reused turn context here instead.
        """
        context_id = self.response_context_id
        if context_id and self.response_has_audio and not self.response_stop_emitted:
            stop = TTSStoppedFrame(context_id=context_id)
            if self.audio_context_available(context_id):
                # Preserve ordering after the last PCM already queued in this
                # context. The base completion hook then drains/removes it.
                await self.append_to_audio_context(context_id, stop)
            else:
                # Pipecat may have retired an idle audio context during a long
                # generation pause. Its audio has drained by then, so direct
                # downstream delivery still places the response stop last.
                await self.push_frame(stop)
            self.response_stop_emitted = True
            self.event("tts_response_stop_queued")
        await super().on_turn_context_completed()
        self.response_context_id = None
        self.response_has_audio = False
        self.response_stop_emitted = False

    async def on_audio_context_interrupted(self, context_id):
        if context_id == self.response_context_id:
            self.event(
                "tts_response_interrupted", response_id=context_id, reason="pipecat_interruption"
            )
            self.response_context_id = None
            self.response_has_audio = False
            self.response_stop_emitted = False

    async def run_tts(self, text, context_id):
        text = spoken_text(text)
        if not any(char.isalnum() for char in text):
            # A malformed streamed answer can contain punctuation-only
            # sentences. They have no speakable content and previously opened
            # separate provider requests, delaying or stalling the response.
            self.event("tts_fragment_skipped", characters=len(text))
            return
        segment_started = time.monotonic()
        # Preserve all text while respecting the project MCP per-operation limit.
        for offset in range(0, len(text), 500):
            self.turn += 1
            segment_index = self.turn
            segment_text = text[offset : offset + 500]
            first_pcm_at = last_pcm_at = None
            pcm_bytes = pcm_chunks = 0
            status = "failed"
            self.event(
                "tts_start",
                segment_id=segment_index,
                characters=len(segment_text),
                context_id=context_id,
                submitted_text=segment_text,
            )
            key = TurnKey(session_id=self.session_id, turn_id=self.turn)
            stream = self.speech.synthesize(key, segment_text)
            try:
                async for pcm in stream:
                    now = time.monotonic()
                    if first_pcm_at is None:
                        first_pcm_at = now
                    last_pcm_at = now
                    pcm_chunks += 1
                    pcm_bytes += len(pcm)
                    self.response_has_audio = True
                    self.event("tts_pcm", bytes=len(pcm))
                    yield TTSAudioRawFrame(pcm, 16000, 1, context_id=context_id)
                status = "completed"
            except (asyncio.CancelledError, GeneratorExit):
                status = "cancelled"
                raise
            except Exception as exc:
                status = "failed"
                self.event(
                    "tts_segment_error",
                    segment_id=segment_index,
                    error_code=getattr(exc, "code", "tts_error"),
                    exception_type=type(exc).__name__,
                )
                raise
            finally:
                await stream.aclose()
                self.event("tts_released")
                diagnostics = getattr(self.speech, "last_tts_diagnostics", None)
                self.event(
                    "tts_segment_summary",
                    segment_id=segment_index,
                    status=status,
                    characters=len(segment_text),
                    wall_seconds=time.monotonic() - segment_started,
                    native_first_pcm_monotonic=first_pcm_at,
                    native_last_pcm_monotonic=last_pcm_at,
                    native_pcm_chunks=pcm_chunks,
                    native_pcm_bytes=pcm_bytes,
                    native_audio_seconds=pcm_bytes / 32000,
                    diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
                )
