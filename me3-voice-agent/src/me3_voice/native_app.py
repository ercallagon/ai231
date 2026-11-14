"""Application binding for the native pipeline: resource lifetime and I/O only.

Native context/aggregators/LLM/TTS/output own conversation and scheduling.
No Session, ConversationInput, AudioApplication or nested ingress pipeline.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import time
from urllib.parse import urlsplit
import uuid

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSTextFrame,
    LLMFullResponseEndFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from .configuration import load_backends, RuntimeOptions
from .contracts import TurnKey
from .conversation_tools import schemas, validate_arguments, READ_ONLY
from .conversation_context import runtime_time
from .research import ResearchClient, ResearchError, public_url
from .native_llm import NativeLLM
from .native_pipeline import make_native_pipeline
from .preprocessing import PCMPreprocessor, PCMFrames
from .providers import ToolCall
from .remote import LiveTransport
from .speech import SpeechClient
from .tools import MockTaskExecutor


NATIVE_REPLAY_SETTLE_TIMEOUT_SECONDS = 60
NATIVE_REPLAY_EVENT_HISTORY_LIMIT = 4096


class NativeState:
    """Bridge-lifetime resources and the ONE native context, not a turn controller.

    MCP enter/exit runs on one owning task; requests/context survive reconnects.
    Restart recovery and arbitrary durable background work remain out of scope.
    """

    def __init__(
        self,
        backends=None,
        credentials=None,
        *,
        speech_factory=None,
        research=None,
        task_runtime_factory=None,
        device_config=None,
        receipt_store=None,
        tasks=None,
    ):
        self.backends = backends or load_backends()
        self.credentials = credentials or {}
        self.context = LLMContext()
        self.research = research
        self.session_id = uuid.uuid4().hex
        self.ui_device_actions = {}
        self._owns_devices = self._owns_music = tasks is None
        if tasks is None:
            from .devices import load_device_service
            from .music import MusicController

            devices = load_device_service(device_config)
            music = MusicController.configured()
            self.devices, self.music = devices, music
            self.tasks = MockTaskExecutor(actions=devices, music=music)
        else:
            self.devices = self.music = None
            self.tasks = tasks
        if receipt_store:
            from .receipts import ReceiptStore

            self.tasks.receipt_store = ReceiptStore(receipt_store)
        self.task_runtime = None
        self.task_runtime_factory = task_runtime_factory
        self.budget = LiveTransport(self.backends["llm"], self.credentials)
        self.speech_factory = speech_factory or (
            lambda: SpeechClient(
                backends=self.backends,
                credentials=self.credentials,
                timeout=max(p.timeout_seconds for p in self.backends.values()),
            )
        )
        self.speech = None
        self.owner = None
        self.lock = asyncio.Lock()
        self.ready = None
        self.closed = asyncio.Event()

    async def start(self):
        async with self.lock:
            if self.owner is not None and self.owner.done() and self.speech is not None:
                raise RuntimeError(
                    "speech owner stopped; restart bridge rather than reset provider budgets"
                )
            if self.owner is not None and self.owner.done() and self.speech is None:
                await asyncio.gather(self.owner, return_exceptions=True)
                self.owner = None
            if self.owner is None:
                self.ready = asyncio.get_running_loop().create_future()
                self.owner = asyncio.create_task(self.own())
            return await asyncio.shield(self.ready)

    async def own(self):
        try:
            async with self.speech_factory() as speech:
                self.speech = speech
                self.ready.set_result(speech)
                await self.closed.wait()
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            raise

    async def close(self):
        self.closed.set()
        if self.owner:
            await asyncio.gather(self.owner, return_exceptions=True)
        try:
            if self.task_runtime:
                await self.task_runtime.close()
                self.task_runtime = None
            await self.tasks.close()
            if self._owns_devices:
                await self.tasks.actions.close()
            if self._owns_music:
                await self.music.close()
        finally:
            if self.research:
                await self.research.close()


@dataclass
class AdmittedAudio(InputAudioRawFrame):
    completion: asyncio.Future | None = field(default=None, repr=False)


class WakeGate(FrameProcessor):
    """ME3 wake admission only; native VAD and turn controller do the rest."""

    def __init__(self, wake=None, *, active=False, on_wake=None):
        super().__init__()
        self.wake = wake
        self.on_wake = on_wake
        self.active = active
        self.frames = PCMFrames(wake.frame_samples if wake else 1280)
        self.preroll = deque(maxlen=10)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InputAudioRawFrame)
            and not self.active
        ):
            self.preroll.append(frame.audio)
            for pcm in self.frames.push(frame.audio):
                result = self.wake.feed(pcm) if self.wake else None
                if result and result["triggered"]:
                    self.active = True
                    if self.on_wake:
                        self.on_wake(result)
            if not self.active:
                if isinstance(frame, AdmittedAudio) and not frame.completion.done():
                    frame.completion.set_result(None)
                return
            for pcm in list(self.preroll)[:-1]:
                await self.push_frame(InputAudioRawFrame(pcm, 16000, 1))
            self.preroll.clear()
        await self.push_frame(frame, direction)


def native_vad():
    return VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000, params=VADParams(confidence=0.7, start_secs=0.16, stop_secs=0.2)
        )
    )


class NativeOutput(BaseOutputTransport):
    """Framework output queue; existing sink owns only delivery/ACK backpressure."""

    def __init__(self, sink, event):
        super().__init__(
            TransportParams(
                audio_out_enabled=True, audio_out_sample_rate=16000, audio_out_channels=1
            )
        )
        self.sink, self.event = sink, event
        self.generation = 0

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        generation = self.generation
        response_id = getattr(frame, "context_id", None)
        try:
            send_audio = getattr(self.sink, "send_audio", None)
            delivered = (
                await send_audio(frame.audio, response_id=response_id)
                if send_audio
                else await self.sink(frame.audio)
            )
        except Exception as exc:
            await self.push_error(error_msg="Native delivery failed: " + type(exc).__name__)
            return False
        if generation != self.generation or delivered is False:
            return False
        self.event(
            "playback_packet",
            packet_id=delivered if type(delivered) is int else None,
            response_id=response_id,
            bytes=len(frame.audio),
            pending_packets=len(self.sink.pending) if hasattr(self.sink, "pending") else 0,
        )
        return True

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if (
            isinstance(frame, (BotStoppedSpeakingFrame, TTSTextFrame, LLMFullResponseEndFrame))
            and direction == FrameDirection.DOWNSTREAM
        ):
            generation = self.generation
            if hasattr(self.sink, "flush"):
                await self.sink.flush()
            if generation != self.generation:
                return
        await super().push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self.event("output_speaking_started")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self.event("output_speaking_stopped")

    async def stop_delivery(self):
        self.generation += 1
        if hasattr(self.sink, "stop"):
            await self.sink.stop()

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            await self.stop_delivery()
            self.event("playback_stopped_browser")
        await super().process_frame(frame, direction)


class NativeApplication:
    """Own a framework worker and bounded input admission; no duplicate history."""

    def __init__(
        self,
        state,
        sink,
        *,
        options=None,
        rate=16000,
        channels=1,
        wake=None,
        annotated=False,
        on_event=None,
        upstream=None,
        clock=None,
        conversation_mode="followup",
        followup_seconds=20,
        extended_seconds=20,
    ):
        self.resources = state
        self.conversation_mode = conversation_mode
        self.idle_seconds = (
            extended_seconds if conversation_mode == "extended" else followup_seconds
        )
        self.sink = sink
        self.rate, self.channels = rate, channels
        self.options = options or RuntimeOptions(engine="native")
        self.pre = PCMPreprocessor(rate, channels)
        self.frames = PCMFrames(512)
        self.annotated = annotated
        self.on_event = on_event
        self.upstream = upstream
        # Injectable only to make the real application context deterministic in
        # offline evaluations; production defaults to the runtime system clock.
        self.clock = clock
        # Keep startup/ASR/LLM events through a bounded streamed TTS response.
        # A 2,000-row ring discarded those events before the live replay failed.
        self.events = deque(maxlen=NATIVE_REPLAY_EVENT_HISTORY_LIMIT)
        self.event_count = 0
        self.fast_action_rows = []
        self.response_records = {}
        self.response_order = deque(maxlen=64)
        self.response_by_conversation_turn = {}
        self.active_response_id = None
        self.last_llm_request_id = None
        self.last_vad_started_at = None
        self.action_revision = 0
        self.turn_id = 0  # Telemetry projection of native user-turn events only.
        self.run_id = uuid.uuid4().hex
        self.projection = "listening" if annotated else "wake_listening"
        self.gate = WakeGate(
            wake,
            active=annotated,
            on_wake=lambda result: self.event(
                "wake_detected", backend=result.get("backend"), score=result.get("score")
            ),
        )
        self.pending = set()
        self.audio_frames = 0
        self.audio_delivery_sequence = 0
        self.last_audio_delivery_at = None
        self.admission = asyncio.Lock()
        self.settled = asyncio.Event()
        self.settled.set()
        self.followup_timer = None
        self.ready = asyncio.Event()
        self.worker = self.runner = self.job = self.llm = self.parts = None
        self.error = None
        self.terminal_failure_code = None
        # Interim ASR is a live UI projection only, never model history or journal text.
        self.partial_transcript = None

    def update_partial_transcript(self, text):
        if text is None:
            self.partial_transcript = None
        elif isinstance(text, str):
            self.partial_transcript = text[:8000] if text.strip() else None

    def event(self, name, **fields):
        if name == "turn_started":
            self.turn_id += 1
            previous = self.response_records.get(self.active_response_id)
            if previous and previous.get("response_state") == "interrupted":
                previous["response_state"] = "superseded"
                previous["superseded_by_turn_id"] = self.turn_id
        if name == "vad_speech_started":
            self.last_vad_started_at = time.monotonic()
        if name == "tts_response_started":
            response_id = fields.get("response_id") or fields.get("context_id")
            if isinstance(response_id, str):
                conversation_turn = sum(
                    message.get("role") == "user"
                    for message in self.resources.context.get_messages()
                    if isinstance(message, dict)
                )
                record = {
                    "response_id": response_id,
                    "turn_id": conversation_turn,
                    "generated_state": "streaming",
                    "response_state": "pending",
                    "tts_state": "pending",
                    "playback_state": "pending",
                    "superseded_by_turn_id": None,
                    "tts_segments": [],
                    "playback_packets": 0,
                    "playback_acks": 0,
                    "started_at": time.monotonic(),
                }
                self.response_records[response_id] = record
                if len(self.response_order) == self.response_order.maxlen:
                    expired = self.response_order[0]
                    self.response_records.pop(expired, None)
                    for turn, prior_id in tuple(self.response_by_conversation_turn.items()):
                        if prior_id == expired:
                            self.response_by_conversation_turn.pop(turn, None)
                self.response_order.append(response_id)
                self.response_by_conversation_turn[conversation_turn] = response_id
                self.active_response_id = response_id
        elif name == "llm_request":
            self.last_llm_request_id = fields.get("request_id")
        response_id = fields.get("response_id") or self.active_response_id
        record = self.response_records.get(response_id)
        submitted_text = fields.pop("submitted_text", None) if name == "tts_start" else None
        if record is not None:
            if name == "tts_start":
                record["tts_state"] = "submitted"
                record["tts_segments"].append(
                    {
                        "segment_id": fields.get("segment_id"),
                        "characters": len(submitted_text)
                        if isinstance(submitted_text, str)
                        else fields.get("characters", 0),
                        "text_sha256": hashlib.sha256(submitted_text.encode()).hexdigest()[:16]
                        if isinstance(submitted_text, str)
                        else None,
                        "state": "submitted",
                    }
                )
            elif name == "tts_segment_summary":
                segment_id = fields.get("segment_id")
                for segment in reversed(record["tts_segments"]):
                    if segment.get("segment_id") == segment_id:
                        segment["state"] = fields.get("status", "unknown")
                        break
                record["tts_state"] = (
                    "completed"
                    if record["tts_segments"]
                    and all(item["state"] == "completed" for item in record["tts_segments"])
                    else "failed"
                    if fields.get("status") == "failed"
                    else "interrupted"
                    if fields.get("status") == "cancelled"
                    else record["tts_state"]
                )
            elif name == "playback_packet":
                record["playback_packets"] += 1
                record["playback_state"] = "queued"
            elif name == "playback_ack":
                record["playback_acks"] += 1
                record["playback_state"] = "browser_acknowledged"
            elif name == "tts_response_interrupted":
                record["response_state"] = "interrupted"
                record["tts_state"] = "interrupted"
                record["playback_state"] = (
                    "partially_acknowledged" if record["playback_acks"] else "not_acknowledged"
                )
                recent_vad = (
                    self.last_vad_started_at is not None
                    and time.monotonic() - self.last_vad_started_at < 2.0
                )
                record["interruption_reason"] = (
                    "vad_barge_in" if recent_vad else fields.get("reason", "speech_interruption")
                )
            elif name == "turn_completed":
                record["response_state"] = "completed"
                record["tts_state"] = "completed"
                record["playback_state"] = (
                    "browser_acknowledged"
                    if record["playback_packets"] > 0
                    and record["playback_packets"] == record["playback_acks"]
                    else "partially_acknowledged"
                    if record["playback_acks"]
                    else "not_acknowledged"
                )
                record["completed_at"] = time.monotonic()
        if name in {"tool_started", "tool_completed", "tool_operation_blocked"}:
            fields.setdefault("operation_id", fields.get("tool_call_id"))
            fields.setdefault("llm_request_id", self.last_llm_request_id)
        value = {
            "event": name,
            "at": time.monotonic(),
            "run_id": self.run_id,
            "session_id": self.resources.session_id,
            "turn_id": self.turn_id,
            **fields,
        }
        self.event_count += 1
        value["sequence"] = self.event_count
        self.events.append(value)
        if self.on_event:
            self.on_event(value)
        return value

    def task_event(self, payload):
        """Project durable task events into the native event stream."""
        if not isinstance(payload, dict):
            self.event("task_update", detail="invalid_task_event")
            return
        name = payload.get("event")
        fields = {key: value for key, value in payload.items() if key != "event"}
        self.event(name if isinstance(name, str) else "task_update", **fields)

    def _fast_action_row(self, call, turn):
        action = call.arguments.get("action") if isinstance(call.arguments, dict) else None
        kind = action if call.name == "run_device_action" else call.name
        if call.name == "control_music":
            kind = {
                "play": "music_play",
                "play_playlist": "music_playlist",
                "pause": "music_pause",
                "resume": "music_resume",
                "stop": "music_stop",
                "next": "music_next",
                "previous": "music_previous",
                "volume_up": "music_volume_up",
                "volume_down": "music_volume_down",
                "volume_set": "music_volume_set",
            }.get(action, "agent")
        if kind not in {
            "music_play",
            "music_playlist",
            "music_pause",
            "music_resume",
            "music_stop",
            "music_next",
            "music_previous",
            "music_volume_up",
            "music_volume_down",
            "music_volume_set",
            "light_on",
            "light_off",
            "set_color",
            "set_brightness",
            "set_timer",
            "set_alarm",
            "cancel_schedule",
            "set_temperature",
            "create_reminder",
            "list_reminders",
            "get_time",
            "get_weather",
        }:
            kind = "agent"
        title = {
            "light_on": "Turn on the light",
            "light_off": "Turn off the light",
            "set_color": "Change light color",
            "set_brightness": "Change light brightness",
            "music_play": "Play music",
            "music_playlist": "Play playlist",
            "music_pause": "Pause music",
            "music_resume": "Resume music",
            "music_stop": "Stop music",
            "music_next": "Skip to next track",
            "music_previous": "Play previous track",
            "music_volume_set": "Set music volume",
            "music_volume_up": "Increase music volume",
            "music_volume_down": "Decrease music volume",
        }.get(kind, kind.replace("_", " ").capitalize())
        mode = "reported"
        executor = self.resources.tasks
        if kind.startswith("music_"):
            if getattr(getattr(executor, "music", None), "token", None) is None:
                mode = "mock"
        elif getattr(getattr(executor, "actions", None), "mode", None) == "mock":
            mode = "mock"
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "id": f"{turn}:{call.call_id}",
            "turn_id": turn,
            "call_id": call.call_id,
            "kind": kind,
            "status": "running",
            "mode": mode,
            "title": title[:80],
            "summary": "Working on this command.",
            "display": str(action or kind)[:32],
            "started_utc": now,
            "updated_utc": now,
        }
        self.fast_action_rows = [r for r in self.fast_action_rows if r["id"] != row["id"]]
        self.fast_action_rows.append(row)
        self.fast_action_rows = self.fast_action_rows[-30:]
        self.action_revision += 1
        return row

    def _finish_fast_action(self, row, *, result=None, error=None):
        now = datetime.now(timezone.utc).isoformat()
        row["updated_utc"] = now
        row["finished_utc"] = now
        if error:
            row["status"] = (
                "unknown" if error in {"cancelled_outcome_unknown", "outcome_unknown"} else "failed"
            )
            row["summary"] = (
                "The command was interrupted; check its outcome before retrying."
                if row["status"] == "unknown"
                else "The command could not be completed: " + str(error)[:120]
            )
            row["display"] = row["status"]
        else:
            result_status = result.get("result_status", "reported")
            row["status"] = (
                result_status
                if result_status
                in {"simulated", "accepted", "completed", "reported", "unavailable", "unknown"}
                else "reported"
            )
            row["mode"] = "mock" if result.get("mode") == "mock" else "reported"
            row["summary"] = str(result.get("summary") or "Command completed.")[:200]
            row["display"] = str(result.get("display") or result.get("action") or row["display"])[
                :32
            ]
            widget = result.get("widget")
            if isinstance(widget, dict):
                row["widget"] = widget
        self.action_revision += 1

    async def fast_action(self, context):
        """Use shared exact-command parsers inside the native turn lifecycle."""
        import json

        from .fast_commands import (
            direct_clock_query,
            direct_light_command,
            direct_me2_command,
            direct_music_control,
            direct_music_request,
            direct_native_music_control,
            direct_light_followup,
            direct_clarification,
        )

        messages = context.get_messages()
        history, tool_requests = [], {}
        history_turn = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                history_turn += 1
                if isinstance(message.get("content"), str):
                    history.append(
                        {"role": "user", "content": message["content"], "turn_id": history_turn}
                    )
            elif role == "assistant":
                for tool in message.get("tool_calls", ()) or ():
                    if not isinstance(tool, dict) or not isinstance(tool.get("id"), str):
                        continue
                    function = tool.get("function") or {}
                    try:
                        arguments = function.get("arguments", "{}")
                        arguments = (
                            json.loads(arguments) if isinstance(arguments, str) else arguments
                        )
                    except (TypeError, ValueError):
                        continue
                    if isinstance(arguments, dict) and isinstance(function.get("name"), str):
                        tool_requests[tool["id"]] = {
                            "name": function["name"],
                            "arguments": json.dumps(arguments, sort_keys=True),
                        }
            elif role == "tool" and isinstance(message.get("tool_call_id"), str):
                request = tool_requests.get(message["tool_call_id"])
                if request is not None:
                    history.append(
                        {
                            "role": "tool",
                            "content": message.get("content", "{}"),
                            "tool_request": request,
                            "turn_id": history_turn,
                        }
                    )
        text = next(
            (
                row.get("content")
                for row in reversed(history)
                if row.get("role") == "user" and isinstance(row.get("content"), str)
            ),
            "",
        )
        if not text:
            return None
        turn = max(1, self.turn_id)
        call = (
            direct_music_control(text, turn, history)
            or direct_native_music_control(text, turn)
            or direct_music_request(text, turn, history)
            or direct_clock_query(text, turn)
            or direct_light_command(text, turn)
            or direct_light_followup(text, turn, history, history_turn=history_turn)
            or direct_me2_command(text, turn)
        )
        if call is None:
            clarification = direct_clarification(text)
            if clarification:
                self.event(
                    "route_selected", route="clarification", conversation_turn_id=history_turn
                )
                self.event("clarification_requested", reason="missing_command_detail")
                return clarification
            return None
        self.event(
            "route_selected",
            route="fast_command",
            call_id=call.call_id,
            action=call.arguments.get("action"),
            conversation_turn_id=history_turn,
        )
        action_row = self._fast_action_row(call, turn)
        self.event(
            "fast_action_started",
            tool_name=call.name,
            action=call.arguments.get("action"),
            call_id=call.call_id,
        )

        def record_result(payload):
            # The action may finish before its spoken confirmation. If barge-in
            # cancels that speech, keep the committed effect paired with the
            # user command so the next LLM turn cannot mistake it for pending.
            context.add_message(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, sort_keys=True),
                            },
                        }
                    ],
                }
            )
            context.add_message(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                }
            )

        try:
            result = await self.resources.tasks.execute(
                TurnKey(session_id=self.resources.session_id, turn_id=turn), call
            )
        except asyncio.CancelledError:
            self._finish_fast_action(action_row, error="cancelled_outcome_unknown")
            raise
        except Exception as exc:
            code = getattr(exc, "code", "action_failed")
            self._finish_fast_action(action_row, error=code)
            self.event("fast_action_failed", error_code=code, call_id=call.call_id)
            record_result({"ok": False, "error": getattr(exc, "code", "action_failed")})
            return "I couldn't complete that device or music command. Please try again."
        summary = result.get("summary") if isinstance(result, dict) else None
        if not isinstance(summary, str) or not summary.strip():
            self._finish_fast_action(action_row, error="missing_action_result")
            self.event("fast_action_failed", error_code="missing_action_result")
            record_result({"ok": False, "error": "missing_action_result"})
            return "I couldn't confirm the result of that command."
        status = result.get("result_status")
        if result.get("ok") is False or status in {"unknown", "unavailable", "failed", "error"}:
            if status in {"failed", "error"} or result.get("ok") is False:
                code = status or result.get("error") or "action_failed"
                self._finish_fast_action(action_row, error=code)
            else:
                self._finish_fast_action(action_row, result=result)
            self.event(
                "fast_action_failed",
                error_code=status or result.get("error", "action_failed"),
                call_id=call.call_id,
            )
            record_result(
                {
                    "ok": False,
                    "result_status": status,
                    "summary": summary,
                    "action": call.arguments.get("action"),
                }
            )
            return (
                "That action failed. " + summary
                if status in {"failed", "error"} or result.get("ok") is False
                else "I couldn't confirm that action. " + summary
            )
        self.event(
            "fast_action_completed",
            result_status=status or "reported",
            mode=result.get("mode", "reported"),
            call_id=call.call_id,
        )
        self._finish_fast_action(action_row, result=result)
        record_result(
            {
                "ok": True,
                "result_status": status or "reported",
                "summary": summary,
                "action": call.arguments.get("action"),
                "mode": result.get("mode", "reported"),
            }
        )
        return summary

    async def __aenter__(self):
        try:
            return await self._start()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def _start(self):
        speech = await self.resources.start()
        for startup in getattr(speech, "startup_diagnostics", ()):
            self.event("speech_mcp_ready", **startup)
        if self.resources.research is None:
            self.resources.research = ResearchClient(
                self.options.research, self.resources.credentials
            )
        if self.resources.task_runtime is None:
            from .task_runtime import TaskRuntime

            factory = self.resources.task_runtime_factory or TaskRuntime
            self.resources.task_runtime = factory(
                self.resources.backends["llm"],
                self.resources.credentials,
                budget=self.resources.budget,
                research=self.resources.research,
            )
            await self.resources.task_runtime.start_runtime()
        self.resources.task_runtime.on_event = self.task_event
        self.llm = NativeLLM(
            self.resources.backends["llm"],
            credentials=self.resources.credentials,
            budget=self.resources.budget,
            upstream=self.upstream,
            clock=self.clock,
            source_ids=lambda: set(self.resources.research.sources),
            research_remaining=lambda: (
                max(
                    0,
                    self.resources.research.options.request_limit
                    - self.resources.research.requests,
                )
                if self.resources.research.options.allow_live
                else None
            ),
            on_event=self.event,
            max_model_calls_per_turn=self.options.max_model_calls_per_turn,
            max_tool_calls_per_turn=self.options.max_tool_calls_per_turn,
            max_repeated_tool_calls=self.options.max_repeated_tool_calls,
            fast_action=self.fast_action,
        )
        self.register_tools()
        ingress = [self.gate] + ([] if self.annotated else [native_vad()])
        self.output = NativeOutput(self.sink, self.event)
        self.parts = make_native_pipeline(
            speech,
            self.llm,
            self.output,
            context=self.resources.context,
            session_id=self.resources.session_id,
            ingress=ingress,
            silence_seconds=0 if self.annotated else self.options.silence_seconds,
            on_event=lambda e: self.event(e.pop("event"), **e),
            on_partial=self.update_partial_transcript,
            user_idle_seconds=0 if self.annotated else self.idle_seconds,
        )
        self.worker = PipelineWorker(
            self.parts.pipeline,
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )

        @self.worker.event_handler("on_pipeline_started")
        async def started(*_):
            self.ready.set()

        @self.worker.event_handler("on_pipeline_error")
        async def failed(_worker, frame):
            exception = frame.exception
            self.error = (
                exception
                if isinstance(exception, Exception)
                else RuntimeError("native pipeline failure")
            )
            # Native interruption owns stopping the failed stream and queued output.
            # Invalidate asynchronous deliveries immediately before forwarding it.
            self.llm.epoch += 1
            await self.worker.queue_frame(InterruptionFrame())
            self.projection = "listening"
            cause = self.error.__cause__ or self.error.__context__
            self.event(
                "turn_error",
                exception_type=type(self.error).__name__,
                error_code=getattr(self.error, "code", "pipeline_error"),
                error_stage=getattr(self.error, "error_stage", None),
                error_boundary=getattr(self.error, "error_boundary", None),
                provider_exception_type=getattr(self.error, "error_detail", None),
                remote_cause_type=getattr(self.error, "remote_cause_type", None),
                cause_type=getattr(self.error, "remote_cause_type", None)
                or getattr(self.error, "cause_type", None)
                or (type(cause).__name__ if cause else None),
            )
            for future in tuple(self.pending):
                if not future.done():
                    future.set_exception(self.error)
            self.settled.set()

        @self.parts.stt.event_handler("on_after_process_frame")
        async def admitted(_processor, frame):
            if isinstance(frame, AdmittedAudio) and not frame.completion.done():
                frame.completion.set_result(None)

        @self.parts.aggregators.user().event_handler("on_user_turn_started")
        async def user_started(*_):
            self._cancel_followup_timer()
            self.llm.new_turn()
            self.terminal_failure_code = None
            self.settled.clear()
            self.projection = "listening"
            self.event("turn_started")

        @self.parts.aggregators.user().event_handler("on_user_turn_stopped")
        async def user_stopped(*_):
            self.projection = "thinking"

        @self.parts.aggregators.assistant().event_handler("on_assistant_turn_started")
        async def bot_started(*_):
            self.projection = "speaking"
            self.event("assistant_turn_started", projection=self.projection)

        @self.parts.aggregators.assistant().event_handler("on_assistant_turn_stopped")
        async def bot_stopped(_aggregator, message):
            self.projection = (
                "listening" if message.interrupted or self.settled.is_set() else "thinking"
            )
            self.event(
                "assistant_turn_stopped",
                interrupted=bool(message.interrupted),
                projection=self.projection,
            )

        @self.parts.aggregators.assistant().event_handler("on_after_process_frame")
        async def response_finished(_aggregator, frame):
            if (
                isinstance(frame, LLMFullResponseEndFrame)
                and frame.metadata.get("me3_response_epoch") == self.llm.epoch
            ):
                conversation_turn = sum(
                    message.get("role") == "user"
                    for message in self.resources.context.get_messages()
                    if isinstance(message, dict)
                )
                response = self.response_records.get(
                    self.response_by_conversation_turn.get(conversation_turn)
                )
                if response is not None:
                    response["generated_state"] = "completed"
            if (
                isinstance(frame, LLMFullResponseEndFrame)
                and frame.metadata.get("me3_response_epoch") == self.llm.epoch
                and not frame.metadata.get("me3_response_has_tools", True)
                and not self.error
            ):
                self.projection = "listening"
                failure = frame.metadata.get("me3_failure_reason")
                if failure:
                    self.terminal_failure_code = failure
                    self.event("turn_failed", error_code=failure)
                else:
                    self.event("turn_completed")
                self.settled.set()
                if self.conversation_mode == "single" and not self.annotated:
                    self.standby()
                else:
                    self._arm_followup_timer()

        @self.parts.aggregators.user().event_handler("on_user_turn_idle")
        async def user_idle(*_):
            # Pipecat's input-idle callback can arrive while the assistant is
            # still thinking or speaking. The response-owned timer starts only
            # after the completed spoken turn.
            if self.settled.is_set() and self.projection == "listening" and not self.followup_timer:
                self.standby()

        self.runner = WorkerRunner(handle_sigint=False)
        await self.runner.add_workers(self.worker)
        self.job = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(self.ready.wait(), 15)
        return self

    def register_tools(self):
        from .task_runtime import conversation_schemas, validate_control_arguments

        task_tool_names = {"start_task", "task_status", "steer_task"}
        all_schemas = [*schemas(), *conversation_schemas()]
        research_names = {"web_search", "retrieve_source", "verify_evidence"}

        def research_available(name):
            research = self.resources.research
            if name == "verify_evidence":
                return bool(research.sources)
            return bool(
                research.options.allow_live and research.requests < research.options.request_limit
            )

        def refresh_tools(context):
            available = [
                schema
                for schema in all_schemas
                if schema.name not in research_names or research_available(schema.name)
            ]
            context.set_tools(ToolsSchema(standard_tools=available))

        refresh_tools(self.resources.context)
        self.llm.refresh_tools = refresh_tools
        for schema in all_schemas:

            async def execute(params):
                epoch = self.llm.dispatch_epoch.get()
                call = ToolCall(params.tool_call_id, params.function_name, params.arguments)
                action_row = (
                    self._fast_action_row(call, max(1, self.turn_id))
                    if call.name in {"run_device_action", "control_music"}
                    else None
                )
                if epoch is None or (epoch != self.llm.epoch and call.name in READ_ONLY):
                    self.event("tool_stale_rejected", tool_call_id=call.call_id)
                    if action_row:
                        self._finish_fast_action(action_row, error="cancelled_outcome_unknown")
                    return
                operation_hash, operation_count, blocked = self.llm.claim_tool_operation(
                    call.name, call.arguments
                )
                self.event(
                    "tool_started",
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    operation_hash=operation_hash,
                    identical_operation_count=operation_count,
                )
                source_origin = None
                source_url_hash = None
                if blocked:
                    result = {"ok": False, "error": blocked}
                    self.event(
                        "tool_operation_blocked",
                        tool_call_id=call.call_id,
                        tool_name=call.name,
                        operation_hash=operation_hash,
                        error_code=blocked,
                    )
                else:
                    try:
                        if call.name in research_names and not research_available(call.name):
                            result = {"ok": False, "error": "research_unavailable"}
                        elif call.name in task_tool_names:
                            args = validate_control_arguments(call.name, call.arguments)
                            result = await self.dispatch_task_tool(call, args)
                        else:
                            args = validate_arguments(call.name, call.arguments)
                        if call.name in task_tool_names:
                            pass
                        elif call.name == "get_runtime_time":
                            result = runtime_time(args.get("timezone", "UTC"))
                        elif call.name == "web_search":
                            result = await self.resources.research.search(
                                args["query"], args.get("max_results", 5)
                            )
                        elif call.name == "retrieve_source":
                            safe_url = public_url(args["url"])
                            parsed = urlsplit(safe_url)
                            source_origin = f"{parsed.scheme}://{parsed.hostname}"
                            source_url_hash = hashlib.sha256(safe_url.encode()).hexdigest()[:16]
                            source = await self.resources.research.retrieve(safe_url)
                            query = next(
                                (
                                    m.get("content", "")
                                    for m in reversed(self.resources.context.get_messages())
                                    if isinstance(m, dict)
                                    and m.get("role") == "user"
                                    and isinstance(m.get("content"), str)
                                ),
                                "",
                            )
                            result = self.resources.research.evidence_for_model(source, query)
                        elif call.name == "verify_evidence":
                            result = self.resources.research.verify(
                                args["source_id"], args["quote"]
                            )
                        else:
                            result = await self.resources.tasks.execute(
                                TurnKey(
                                    session_id=self.resources.session_id,
                                    turn_id=max(1, self.turn_id),
                                ),
                                call,
                            )
                    except (ResearchError, ValueError) as exc:
                        # Symbolic policy errors only; never return exception payloads from I/O.
                        result = (
                            exc.result()
                            if isinstance(exc, ResearchError)
                            else {"ok": False, "error": "invalid_arguments_or_tool_budget"}
                        )
                    except Exception:
                        result = {"ok": False, "error": "tool_service_unavailable"}
                if action_row:
                    if isinstance(result, dict) and (
                        result.get("ok") is False
                        or result.get("result_status") in {"failed", "error"}
                    ):
                        code = result.get("error") or result.get("result_status") or "tool_failed"
                        self._finish_fast_action(
                            action_row,
                            error="outcome_unknown"
                            if code in {"music_outcome_uncertain", "tool_service_unavailable"}
                            else code,
                        )
                    elif isinstance(result, dict):
                        self._finish_fast_action(action_row, result=result)
                    else:
                        self._finish_fast_action(action_row, error="invalid_action_result")
                if epoch != self.llm.epoch:
                    self.event("tool_stale_rejected", tool_call_id=call.call_id)
                    return
                failed = isinstance(result, dict) and result.get("ok") is False
                error_code = result.get("error") if failed else None
                result_kind = (
                    "search_leads"
                    if isinstance(result, dict) and result.get("kind") == "untrusted_search_leads"
                    else "retrieved_source"
                    if isinstance(result, dict) and isinstance(result.get("source_id"), str)
                    else "tool_error"
                    if failed
                    else "tool_result"
                )
                if call.name == "retrieve_source":
                    self.llm.record_retrieval_result(failed=failed, error_code=error_code)
                safe_diagnostics = {
                    key: result[key]
                    for key in (
                        "http_status",
                        "host_hash",
                        "stage",
                        "failure_class",
                        "redirect_count",
                        "local_extraction",
                        "fallback_status",
                        "fallback_error",
                    )
                    if isinstance(result, dict) and key in result
                }
                if result_kind == "retrieved_source":
                    safe_diagnostics.update(
                        extraction_method=result.get("extraction_method"),
                        evidence_status=result.get("evidence_status"),
                    )
                self.event(
                    "tool_completed",
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    operation_hash=operation_hash,
                    ok=not failed,
                    error_code=error_code,
                    result_kind=result_kind,
                    result_count=len(result.get("results", []))
                    if isinstance(result, dict) and isinstance(result.get("results"), list)
                    else None,
                    retained_source_count=len(self.resources.research.sources),
                    source_origin=source_origin,
                    source_url_hash=source_url_hash,
                    diagnostics=safe_diagnostics or None,
                )
                await params.result_callback(result)
                self.event(
                    "tool_result_callback_returned",
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    operation_hash=operation_hash,
                )

            self.llm.register_function(
                schema.name,
                execute,
                cancel_on_interruption=schema.name in READ_ONLY,
                timeout_secs=8,
            )

    async def dispatch_task_tool(self, call, args):
        runtime = self.resources.task_runtime
        if runtime is None:
            raise RuntimeError("task_runtime_unavailable")
        if call.name == "start_task":
            import hashlib

            request_id = (
                "voice_"
                + hashlib.sha256((self.resources.session_id + call.call_id).encode()).hexdigest()[
                    :40
                ]
            )
            call_id = call.call_id if len(call.call_id) <= 80 else "call_" + request_id[-40:]
            return await runtime.start(
                args["goal"], request_id=request_id, turn_id=max(1, self.turn_id), call_id=call_id
            )
        if call.name == "task_status":
            return await runtime.status(args["task_id"])
        if call.name == "steer_task":
            return await runtime.steer(args["task_id"], args["expected_revision"], args["delta"])
        raise ValueError("unknown_task_command")

    def standby(self):
        self._cancel_followup_timer()
        self.gate.active = False
        self.projection = "wake_listening"
        if self.gate.wake and hasattr(self.gate.wake, "end_session"):
            self.gate.wake.end_session()
        self.event("engagement_expired")

    def _cancel_followup_timer(self):
        task, self.followup_timer = self.followup_timer, None
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _arm_followup_timer(self):
        self._cancel_followup_timer()
        if self.annotated or self.conversation_mode == "single":
            return

        async def expire():
            await asyncio.sleep(self.idle_seconds)
            if (
                self.settled.is_set()
                and self.projection == "listening"
                and not self.parts.stt.speaking
            ):
                self.standby()

        self.followup_timer = asyncio.create_task(expire())

    async def feed(self, pcm, sequence):
        async with self.admission:
            if self.error:
                if self.pending:
                    error, self.error = self.error, None
                    raise error
                error, self.error = self.error, None
                # A completed turn failure has already invalidated LLM work and
                # sent InterruptionFrame. Reset only partial input framing, then
                # accept the next browser frame in the same live session.
                self.pre = PCMPreprocessor(self.rate, self.channels)
                self.pre.sequence = sequence
                self.frames = PCMFrames(512)
                await self.parts.stt.abort()
                self.projection = "listening"
                self.event(
                    "turn_recovered",
                    error_code=getattr(error, "code", "pipeline_error"),
                    error_stage=getattr(error, "error_stage", None),
                )
            for frame in self.frames.push(self.pre.push(pcm, sequence)):
                await self.queue_audio(frame)

    async def queue_audio(self, pcm):
        now = time.monotonic()
        self.audio_frames += 1
        previous = self.last_audio_delivery_at
        self.last_audio_delivery_at = now
        self.event(
            "audio_frame_delivered",
            audio_sequence=self.audio_delivery_sequence,
            bytes=len(pcm),
            inter_frame_gap_seconds=now - previous if previous is not None else None,
            pending_frames=len(self.pending),
        )
        self.audio_delivery_sequence += 1
        if len(self.pending) >= 16:
            raise RuntimeError("native ingress backlog exceeds512ms; reconnect required")
        future = asyncio.get_running_loop().create_future()
        self.pending.add(future)

        def complete(done):
            self.pending.discard(done)
            # Observe exceptions even if transport closes before drain().
            if not done.cancelled():
                done.exception()

        future.add_done_callback(complete)
        await self.worker.queue_frame(
            AdmittedAudio(audio=pcm, sample_rate=16000, num_channels=1, completion=future)
        )

    async def drain(self):
        if self.pending:
            await asyncio.wait_for(asyncio.gather(*tuple(self.pending)), 10)

    async def finish_input(self):
        for frame in self.frames.push(self.pre.finish()):
            await self.queue_audio(frame)
            await self.drain()
        tail = self.frames.finish()
        if tail:
            await self.queue_audio(tail + b"\0" * (1024 - len(tail)))
        await self.drain()

    async def utterance(self, pcm, *, wait=True):
        if not self.annotated:
            raise ValueError("explicit utterances are offline-only")
        self.settled.clear()
        self.error = None
        self.terminal_failure_code = None
        await self.worker.queue_frame(VADUserStartedSpeakingFrame())
        for offset in range(0, len(pcm), 1024):
            await self.queue_audio(pcm[offset : offset + 1024])
            await self.drain()
        self.event("endpoint_submitted")
        await self.worker.queue_frame(VADUserStoppedSpeakingFrame())
        if wait:
            await asyncio.wait_for(self.settled.wait(), 15)
            if self.error:
                raise self.error

    async def interrupt(self):
        self.event("interrupt_requested", reason="user_requested")
        self.error = None
        await self.worker.queue_frame(InterruptionFrame())
        # Reset current recognizer input explicitly; UI interruption is distinct
        # from a VAD onset, which preserves the new user's input in the adapter.
        self.parts.stt.speaking = False
        await self.parts.stt.abort()
        self.projection = "listening"
        self.settled.set()
        self._arm_followup_timer()

    async def __aexit__(self, *_):
        self._cancel_followup_timer()
        try:
            if getattr(self, "output", None):
                await self.output.stop_delivery()
            if self.runner:
                await self.runner.cancel()
            if self.job:
                await asyncio.wait_for(self.job, 10)
        finally:
            if self.resources.task_runtime:
                self.resources.task_runtime.on_event = None
            for future in tuple(self.pending):
                if not future.done():
                    future.cancel()
            if self.llm:
                await self.llm._client.close()


async def run_native_utterances(
    utterances,
    *,
    backends=None,
    transcripts=None,
    trace=None,
    transcript=False,
    speech_factory=None,
    options=None,
):
    """Actual CLI demo binding. Explicit file boundaries match legacy demo EOF."""
    from contextlib import AsyncExitStack
    import json
    import platform
    from .playback import MemorySink
    from .native_bridge import NativeTranscriptView
    from .transcript import TranscriptPublisher

    selected = backends or load_backends()
    state = NativeState(
        selected,
        speech_factory=speech_factory
        or (lambda: SpeechClient(backends=selected, transcript=transcripts or "Mock question")),
    )
    sink = MemorySink(capacity=10000)
    try:
        async with NativeApplication(state, sink, annotated=True, options=options) as application:
            ready_at = time.monotonic()
            async with AsyncExitStack() as stack:
                if transcript:
                    await stack.enter_async_context(
                        TranscriptPublisher(NativeTranscriptView(state, application))
                    )
                for pcm in utterances:
                    await application.utterance(pcm)
                result = {
                    "engine": "native",
                    "application_ready_at": ready_at,
                    "legacy_session_imported": __import__("sys").modules.get("me3_voice.session")
                    is not None,
                    "mode": "mock",
                    "python": platform.python_version(),
                    "protocols": dict(state.speech.protocols),
                    "history": list(state.context.get_messages()),
                    "run_id": application.run_id,
                    "session_id": state.session_id,
                    "sink_chunks": len(sink.chunks),
                    "events": list(application.events),
                    "turn_completion": "explicit-utterance boundary, no extra silence wait",
                    "limitations": "Mock upstream; memory PCM; no acoustic/provider claims",
                }
        if trace:
            with trace.open("x") as stream:
                for event in result["events"]:
                    stream.write(json.dumps(event) + "\n")
            result["trace_schema"] = "native-runtime-v1"
        return result
    finally:
        await state.close()


async def replay_native_application(args, options, *, upstream=None):
    """Actual file replay route: one preprocessor/wake gate/native VAD path."""
    import wave
    import os
    import time
    from .credentials import load_secrets
    from .playback import MemorySink
    from .readiness import build_wake
    from .native_bridge import NativeTranscriptView
    from .transcript import TranscriptPublisher
    from contextlib import AsyncExitStack
    from .configuration import ConfigurationError
    import json

    replay_started = time.monotonic()

    if (
        getattr(args, "onit_python", None)
        or getattr(args, "direct_clock", False)
        or args.turn_policy != "silence"
    ):
        raise ConfigurationError(
            "native replay does not bind Onit/direct-clock/SmartTurn; select legacy explicitly"
        )
    if args.processed_wav:
        raise ConfigurationError("native replay does not export processed input")
    if args.wav.resolve() == args.output.resolve():
        raise ConfigurationError("report cannot overwrite input")
    if args.trace and (
        args.trace.exists() or args.trace.resolve() in {args.wav.resolve(), args.output.resolve()}
    ):
        raise ConfigurationError("native trace requires fresh separate path")
    backends = load_backends(args.backend_config)
    credentials = (
        load_secrets(
            args.secrets_file, {p.auth_env for p in backends.values() if p.backend == "external"}
        )
        if args.secrets_file
        else {}
    )
    if args.llm_secrets_file:
        credentials.update(load_secrets(args.llm_secrets_file, {backends["llm"].auth_env}))
    for spec in backends.values():
        if spec.backend == "external":
            LiveTransport(spec, credentials).check_credentials()
    wake = build_wake(args)
    if wake is not None and args.wake_frame is not None:
        raise ConfigurationError("choose wake model or injected wake, not both")
    state = NativeState(
        backends,
        credentials,
        speech_factory=lambda: SpeechClient(
            backends=backends,
            credentials=credentials,
            timeout=max(p.timeout_seconds for p in backends.values()),
            transcript=getattr(args, "mock_transcripts", None) or "Mock question",
        ),
    )
    sink = MemorySink(capacity=10000)
    application = None
    replay_stage = "mcp_and_pipeline_startup"
    trace_written_by_replay = False
    try:
        with wave.open(str(args.wav), "rb") as wav:
            if (
                wav.getsampwidth() != 2
                or wav.getnchannels() not in (1, 2)
                or wav.getnframes() / wav.getframerate() > 30
            ):
                raise ConfigurationError("require PCM16 mono/stereo WAV <=30s")
            async with NativeApplication(
                state,
                sink,
                options=options,
                upstream=upstream,
                rate=wav.getframerate(),
                channels=wav.getnchannels(),
                wake=wake,
            ) as application:
                replay_stage = "audio_replay"
                async with AsyncExitStack() as stack:
                    if args.transcript:
                        await stack.enter_async_context(
                            TranscriptPublisher(NativeTranscriptView(state, application))
                        )
                    sequence = 0
                    while pcm := wav.readframes(max(1, wav.getframerate() * 32 // 1000)):
                        if (
                            args.wake_frame is not None
                            and application.audio_frames >= args.wake_frame
                        ):
                            application.gate.active = True
                            args.wake_frame = None
                        await application.feed(pcm, sequence)
                        sequence += 1
                        await application.drain()
                        if application.error:
                            raise application.error
                        if args.pace:
                            await asyncio.sleep(
                                len(pcm) / (wav.getframerate() * wav.getnchannels() * 2) * args.pace
                            )
                    await application.finish_input()
                    if application.error:
                        raise application.error
                        # Do not promote a truncated voiced tail to a complete turn.
                    unfinished = application.parts.stt.speaking
                    if unfinished:
                        if application.error:
                            raise application.error
                        await application.interrupt()
                    elif not application.settled.is_set():
                        # Replay settlement is a diagnostic allowance, not a
                        # user-facing latency target. Sequential live TTS MCP
                        # segments can exceed 30 seconds for long answers;
                        # keep a finite bound while allowing completion events
                        # and the final WAV to be captured.
                        await asyncio.wait_for(
                            application.settled.wait(), NATIVE_REPLAY_SETTLE_TIMEOUT_SECONDS
                        )
                    # Pipeline error callbacks may race the final source drain;
                    # a settled error must never be serialized as a successful
                    # replay result.
                    if application.error:
                        raise application.error
                    result = {
                        "engine": "native",
                        "agent_mode": "native-framework",
                        "trace_schema": "native-runtime-v1",
                        "utterances": sum(e["event"] == "asr_final" for e in application.events),
                        "unfinished_input": unfinished,
                        "protocols": dict(state.speech.protocols),
                        "events": list(application.events),
                        "history": list(state.context.get_messages()) if args.transcript else [],
                        "sink_chunks": len(sink.chunks),
                        "limitations": "Native PCM replay; no physical/acoustic/provider quality inference",
                    }
                    replay_stage = "report_generation"
        if args.trace:
            with args.trace.open("x") as handle:
                for event in result["events"]:
                    handle.write(json.dumps(event) + "\n")
            trace_written_by_replay = True
        playback = getattr(args, "playback_wav", None)
        if playback:
            if playback.exists() or playback.resolve() in {
                args.wav.resolve(),
                args.output.resolve(),
                args.trace.resolve() if args.trace else None,
            }:
                raise ConfigurationError("playback WAV requires fresh separate path")
            with wave.open(str(playback), "wb") as rendered:
                rendered.setnchannels(1)
                rendered.setsampwidth(2)
                rendered.setframerate(16000)
                for pcm in sink.chunks:
                    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
                        raise RuntimeError("native TTS output is not valid PCM16")
                    rendered.writeframesraw(pcm)
            with wave.open(str(playback), "rb") as rendered:
                wav_valid = (
                    rendered.getnchannels() == 1
                    and rendered.getsampwidth() == 2
                    and rendered.getframerate() == 16000
                    and rendered.getnframes() > 0
                )
            result["tts_audio"] = {
                "format": "wav/pcm_s16le/16000hz/mono",
                "chunks": len(sink.chunks),
                "pcm_bytes": sum(len(pcm) for pcm in sink.chunks),
                "pcm16_valid": bool(sink.chunks),
                "wav_valid": wav_valid,
                "path": str(playback),
            }
        if "tts_audio" not in result:
            result["tts_audio"] = {
                "format": "pcm_s16le/16000hz/mono",
                "chunks": len(sink.chunks),
                "pcm_bytes": sum(len(pcm) for pcm in sink.chunks),
                "pcm16_valid": bool(sink.chunks)
                and all(
                    isinstance(pcm, bytes) and len(pcm) > 0 and len(pcm) % 2 == 0
                    for pcm in sink.chunks
                ),
                "wav_valid": None,
            }
        from .remote import request_ledger_summary

        result["request_ledger"] = request_ledger_summary(os.environ.get("ME3_REQUEST_RUN_ID"))
        result["provider_budget"] = {
            role: {
                "backend": spec.backend,
                "provider": urlsplit(spec.endpoint).hostname if spec.endpoint else None,
                "model": spec.model or None,
                "protocol": spec.protocol or None,
                "allow_live": spec.allow_live,
                "request_limit": spec.request_limit,
            }
            for role, spec in backends.items()
        }
        result["replay_elapsed_seconds"] = time.monotonic() - replay_started
        transcript_texts = [
            message.get("content")
            for message in result["history"]
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        answer_texts = [
            message.get("content")
            for message in result["history"]
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ]
        result["asr_transcription"] = transcript_texts[-1] if transcript_texts else None
        result["llm_response"] = answer_texts[-1] if answer_texts else None
        result["completion_events"] = [
            event["event"]
            for event in result["events"]
            if event.get("event")
            in {
                "asr_final",
                "llm_request",
                "llm_first_token",
                "tts_start",
                "tts_segment_summary",
                "tts_pcm",
                "tts_response_stop_queued",
                "output_speaking_started",
                "output_speaking_stopped",
                "assistant_turn_started",
                "assistant_turn_stopped",
                "playback_packet",
                "turn_completed",
                "turn_failed",
                "turn_error",
            }
        ]
        marks = {}
        for name in (
            "turn_started",
            "asr_finish_requested",
            "asr_final",
            "llm_request",
            "llm_first_token",
            "tts_start",
            "tts_pcm",
            "playback_packet",
            "turn_completed",
        ):
            event = next(
                (
                    row
                    for row in result["events"]
                    if row.get("event") == name and isinstance(row.get("at"), (int, float))
                ),
                None,
            )
            if event is not None:
                marks[name] = event["at"]

        def elapsed(first, last):
            if first in marks and last in marks and marks[last] >= marks[first]:
                return marks[last] - marks[first]
            return None

        result["latency_seconds"] = {
            "speech_endpoint_to_asr_final": elapsed("asr_finish_requested", "asr_final"),
            "asr_final_to_llm_request": elapsed("asr_final", "llm_request"),
            "llm_request_to_first_token": elapsed("llm_request", "llm_first_token"),
            "first_token_to_tts_start": elapsed("llm_first_token", "tts_start"),
            "tts_start_to_first_pcm": elapsed("tts_start", "tts_pcm"),
            "first_pcm_to_first_output_packet": elapsed("tts_pcm", "playback_packet"),
            "speech_endpoint_to_first_output_packet": elapsed(
                "asr_finish_requested", "playback_packet"
            ),
            "turn_started_to_completed": elapsed("turn_started", "turn_completed"),
            "replay_including_startup": result["replay_elapsed_seconds"],
        }
        result["errors"] = [
            {"event": event.get("event"), "code": event.get("error_code")}
            for event in result["events"]
            if event.get("event") in {"turn_error", "turn_failed", "asr_error"}
        ]
        return result
    except BaseException as exc:
        # A failed replay is still useful evidence. Persist only event metadata
        # and exception classes/codes: never exception messages or payloads.
        prior = list(application.events) if application else []
        causal = next(
            (
                row
                for row in reversed(prior)
                if row.get("event")
                in {"asr_append_failed", "asr_events_failed", "asr_error", "asr_turn_aborted"}
            ),
            {},
        )
        failure_code = getattr(exc, "code", None) or causal.get("error_code")
        failure_stage = getattr(exc, "error_stage", None) or causal.get("error_stage")
        failure_boundary = getattr(exc, "error_boundary", None) or causal.get("error_boundary")
        provider_exception = getattr(exc, "error_detail", None) or causal.get(
            "provider_exception_type"
        )
        failure = {
            "event": "replay_failed",
            "at": time.monotonic(),
            "stage": replay_stage,
            "exception_type": type(exc).__name__,
            "failure_origin_event": causal.get("event"),
            "error_code": failure_code,
            "error_stage": failure_stage,
            "error_boundary": failure_boundary,
            "provider_exception_type": provider_exception,
            "remote_cause_type": getattr(exc, "remote_cause_type", None),
            "cause_type": (
                getattr(exc, "remote_cause_type", None)
                or getattr(exc, "cause_type", None)
                or causal.get("cause_type")
                or (
                    type(exc.__cause__ or exc.__context__).__name__
                    if exc.__cause__ or exc.__context__
                    else None
                )
            ),
            "audio_frames_delivered": application.audio_frames if application else 0,
            "asr_stream_id": (
                application.parts.stt.stream_ref(application.parts.stt.operation)
                if application
                and application.parts
                and application.parts.stt.operation
                and hasattr(application.parts.stt, "stream_ref")
                else None
            ),
            "asr_queue_depth": (
                application.parts.stt.append_queue.qsize()
                if application
                and application.parts
                and application.parts.stt.append_queue is not None
                else 0
            ),
        }
        from .remote import request_ledger_summary

        try:
            failure["provider_requests"] = request_ledger_summary(
                os.environ.get("ME3_REQUEST_RUN_ID")
            )
        except Exception:
            failure["provider_requests"] = {"available": False}
        events = prior + [failure]
        trace_path = args.trace or args.output.with_suffix(".partial-trace.jsonl")
        if trace_written_by_replay and trace_path.exists():
            with trace_path.open("a") as handle:
                handle.write(json.dumps(failure) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        elif not trace_path.exists():
            with trace_path.open("x") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
                handle.flush()
        raise
    finally:
        await state.close()
