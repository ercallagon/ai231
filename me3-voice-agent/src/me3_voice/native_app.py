"""Application binding for the native pipeline: resource lifetime and I/O only.

Native context/aggregators/LLM/TTS/output own conversation and scheduling.
No Session, ConversationInput, AudioApplication or nested ingress pipeline.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import re
import time
from urllib.parse import urlsplit
import uuid

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
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
from .conversation_context import LOCAL_OWNER_TURN, runtime_time
from .research import ResearchClient, ResearchError, public_url
from .native_llm import NativeLLM
from .native_pipeline import make_native_pipeline
from .preprocessing import PCMPreprocessor, PCMFrames
from .providers import ToolCall
from .remote import LiveTransport, ProviderError
from .speech import SpeechClient
from .native_speech import ProgressTTSSpeakFrame
from .tools import MockTaskExecutor


NATIVE_REPLAY_SETTLE_TIMEOUT_SECONDS = 60
NATIVE_REPLAY_EVENT_HISTORY_LIMIT = 4096
RESEARCH_ACK_DELAY_SECONDS = 0.65
RESEARCH_AUTO_SOURCE_SECONDS = 3.5
_TERMINAL_TURN_EVENTS = {"turn_completed", "turn_failed", "turn_error", "turn_interrupted"}


def _safe_tool_error_code(exc):
    code = getattr(exc, "code", None)
    if (
        isinstance(code, str)
        and 1 <= len(code) <= 64
        and all(char.isascii() and (char.isalnum() or char == "_") for char in code)
    ):
        return code
    return "tool_service_unavailable"


def _normalize_command_phrase(text):
    if not isinstance(text, str):
        return ""
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _normalize_approval_utterance(text):
    if not isinstance(text, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9']+", " ", text.casefold()).split())


def _github_publication_status(text):
    phrase = _normalize_approval_utterance(text)
    return bool(
        re.fullmatch(r"(?:is it live|did you push it|has it been published)(?: yet| already)?", phrase)
        or (
            re.search(r"\b(?:github|pages|(?:site|website) live)\b", phrase)
            and re.match(
                r"^(?:(?:but|so) )?(?:did|have|has|is|are|was|were|what|what's|where|where's|"
                r"how|how's|can you check|check|status|progress)\b", phrase
            )
        )
    )


def _pages_request(text):
    phrase = _normalize_approval_utterance(text)
    return bool(
        re.search(r"\b(?:push|publish|deploy|upload|put)\b.*\bgithub (?:io|pages)\b", phrase)
        or re.search(r"\b(?:create|build|generate)\b.*\bon github (?:io|pages)\b", phrase)
        or _make_site_live(phrase)
    )


def _make_site_live(phrase):
    return re.search(
        r"\b(?:make|get|put) (?:this|the|my|that) (?:web ?site|site|page) live\b", phrase
    )


_GITHUB_APPROVAL_ACTIONS = {"github_publish", "github_pages_deploy", "github_publish_and_deploy"}


def _github_approval_intent(text):
    phrase = _normalize_approval_utterance(text)
    generic_approve = {"approve", "approve it", "yes approve", "yes approve it", "confirm"}
    reject = {"cancel", "cancel it", "no", "no thanks", "reject", "decline"}
    branch_approve = {"yes push it", "yes push this", "yes publish it", "yes publish this"}
    pages_approve = {
        "yes enable pages",
        "yes publish on pages",
        "yes publish on github pages",
        "yes deploy pages",
        "yes deploy on github pages",
    }
    if phrase in generic_approve:
        return "approve", None
    if phrase in reject:
        return "reject", None
    if phrase in branch_approve:
        return "approve", "github_publish"
    if phrase in pages_approve:
        return "approve", "github_pages_deploy"
    return None


def _looks_like_github_approval_or_request(text):
    phrase = _normalize_approval_utterance(text)
    if phrase in {"approve", "approve it", "confirm", "cancel", "cancel it", "reject", "decline"}:
        return True
    return bool(
        re.match(r"^(?:yes|yeah|yep|no|approve|approved|confirm|cancel|reject|decline)\b", phrase)
        or (
            re.search(r"\b(?:push|publish|upload|put)\b", phrase)
            and re.search(r"\bgithub(?: pages)?\b", phrase)
        )
    )


def _voice_task_request_id(session_id, call_id):
    if not isinstance(session_id, str) or not isinstance(call_id, str):
        return None
    return "voice_" + hashlib.sha256((session_id + call_id).encode()).hexdigest()[:40]


def _requests_remaining(limit, used):
    return None if limit is None else max(0, limit - used)


def _request_limit_reached(limit, used):
    return limit is not None and used >= limit


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
        self._task_budget = None
        self.startup_errors = {}
        self.provider_probes = {}
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

    def task_model_configuration(self, options):
        task_spec = self.backends["llm"]
        if not options.task_model:
            return task_spec, self.budget
        task_spec = replace(
            task_spec,
            model=options.task_model,
            request_limit=options.task_request_limit,
            max_input_tokens_total=options.task_input_tokens_total,
        )
        if self._task_budget is None or self._task_budget.spec != task_spec:
            self._task_budget = LiveTransport(task_spec, self.credentials)
        return task_spec, self._task_budget

    async def warm_runtime(self, options):
        """Start reusable local clients/stores before the browser opens its mic."""
        self.startup_errors.clear()
        try:
            await self.start()
        except Exception as exc:
            self.startup_errors["speech"] = type(exc).__name__

        try:
            if self.research is None:
                self.research = ResearchClient(options.research, self.credentials)
            self.research.session()
        except Exception as exc:
            self.startup_errors["research"] = type(exc).__name__

        try:
            # Create reusable HTTP pools without opening provider sockets or
            # making billable warm-up requests.
            self.budget.session()
            task_spec, task_budget = self.task_model_configuration(options)
            if task_budget is not self.budget:
                task_budget.session()
            if self.task_runtime is None:
                from .task_runtime import TaskRuntime

                factory = self.task_runtime_factory or TaskRuntime
                runtime_args = {"budget": task_budget, "research": self.research}
                if options.task_model:
                    runtime_args["max_completion_tokens"] = options.task_max_output_tokens
                candidate = factory(task_spec, self.credentials, **runtime_args)
                try:
                    await candidate.start_runtime()
                except BaseException:
                    await candidate.close()
                    raise
                self.task_runtime = candidate
        except Exception as exc:
            self.startup_errors["tasks"] = type(exc).__name__

    async def start(self):
        async with self.lock:
            if self.closed.is_set():
                raise RuntimeError("speech runtime is closing")
            if self.owner is not None and self.owner.done():
                await asyncio.gather(self.owner, return_exceptions=True)
                self.owner = None
                self.speech = None
                self.ready = None
                self.closed = asyncio.Event()
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
            if self._task_budget and self._task_budget is not self.budget:
                await self._task_budget.aclose()
            await self.budget.aclose()


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
            sample_rate=16000, params=VADParams(confidence=0.7, start_secs=0.16, stop_secs=0.4)
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
        self.suppressed_response_ids = set()

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        generation = self.generation
        response_id = getattr(frame, "context_id", None)
        if response_id in self.suppressed_response_ids:
            return False
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

    async def stop_delivery(self, response_id=None, *, interruption_id=None):
        self.generation += 1
        if hasattr(self.sink, "stop"):
            sent = await self.sink.stop()
            if type(sent) is int:
                self.event(
                    "audio_stop_sent",
                    response_id=response_id,
                    stop_id=sent,
                    interruption_id=interruption_id,
                )
            elif sent is True:
                self.event("audio_stop_sent", response_id=response_id)

    async def stop_response(self, response_id):
        self.suppress_response(response_id)
        await self.stop_delivery(response_id)
        self.event("playback_stopped_browser", response_id=response_id)

    def suppress_response(self, response_id):
        if isinstance(response_id, str):
            self.suppressed_response_ids.add(response_id)
            if len(self.suppressed_response_ids) > 64:
                self.suppressed_response_ids.pop()

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            await self.stop_delivery(interruption_id=frame.id)
            self.event("playback_stopped_browser")
        await super().process_frame(frame, direction)


def _elapsed_ms(start, end):
    """Return an ordered monotonic span, or None when either mark is absent."""
    if (
        not isinstance(start, dict)
        or not isinstance(end, dict)
        or not isinstance(start.get("at"), (int, float))
        or not isinstance(end.get("at"), (int, float))
        or end["at"] < start["at"]
    ):
        return None
    return round((end["at"] - start["at"]) * 1000, 3)


def _voice_latency_for_turn(events, turn_id):
    """Correlate software latency marks without crossing turns or audio kinds."""
    rows = list(events)
    turn_start = next(
        (
            row
            for row in reversed(rows)
            if row.get("event") == "turn_started" and row.get("turn_id") == turn_id
        ),
        None,
    )
    turn_end = next(
        (
            row
            for row in reversed(rows)
            if row.get("event") == "turn_completed" and row.get("turn_id") == turn_id
        ),
        None,
    )
    if turn_start is None or turn_end is None:
        return None
    start_sequence = turn_start.get("sequence", -1)
    end_sequence = turn_end.get("sequence", float("inf"))
    turn_rows = [
        row
        for row in rows
        if row.get("turn_id") == turn_id
        and start_sequence <= row.get("sequence", -1) <= end_sequence
    ]

    # ASR/VAD use the stream's utterance ID and precede the conversation turn.
    previous_turn_start = next(
        (
            row
            for row in reversed(rows)
            if row.get("event") == "turn_started"
            and row.get("turn_id", turn_id) < turn_id
            and row.get("sequence", -1) < start_sequence
        ),
        None,
    )
    previous_start_sequence = (
        previous_turn_start.get("sequence", -1) if previous_turn_start else -1
    )
    prior_finals = [
        row
        for row in rows
        if row.get("event") == "asr_final"
        and previous_start_sequence < row.get("sequence", -1) < start_sequence
        and row.get("utterance_id") is not None
    ]
    asr_final = prior_finals[-1] if prior_finals else None
    utterance_id = asr_final.get("utterance_id") if asr_final else None
    speech_end = (
        next(
            (
                row
                for row in reversed(rows)
                if row.get("event") == "vad_speech_stopped"
                and row.get("utterance_id") == utterance_id
                and previous_start_sequence < row.get("sequence", -1) < start_sequence
            ),
            None,
        )
        if utterance_id is not None
        else None
    )

    llm_request = next(
        (row for row in turn_rows if row.get("event") == "llm_provider_request_started"), None
    )
    request_id = llm_request.get("request_id") if llm_request else None
    first_token = next(
        (
            row
            for row in turn_rows
            if row.get("event") == "llm_first_token"
            and request_id is not None
            and row.get("request_id") == request_id
        ),
        None,
    )
    final_response = next(
        (row for row in turn_rows if row.get("event") == "tts_response_started"), None
    )
    final_response_id = final_response.get("response_id") if final_response else None
    final_tts_start = next(
        (
            row
            for row in turn_rows
            if row.get("event") == "tts_start"
            and row.get("sequence", -1) > (final_response or {}).get("sequence", -1)
            and row.get("response_id", final_response_id) == final_response_id
        ),
        None,
    )
    final_pcm = next(
        (
            row
            for row in turn_rows
            if row.get("event") == "tts_pcm"
            and row.get("sequence", -1) >= (final_tts_start or {}).get("sequence", float("inf"))
            and row.get("response_id", final_response_id) == final_response_id
        ),
        None,
    )
    final_packet = next(
        (
            row
            for row in turn_rows
            if row.get("event") == "playback_packet"
            and row.get("sequence", -1) >= (final_pcm or {}).get("sequence", float("inf"))
            and row.get("response_id", final_response_id) == final_response_id
        ),
        None,
    )

    acknowledgements = []
    progress_starts = [
        (index, row)
        for index, row in enumerate(turn_rows)
        if row.get("event") == "tts_progress_started"
    ]
    for progress_index, (row_index, progress_start) in enumerate(progress_starts):
        response_id = progress_start.get("response_id")
        stop_index = (
            progress_starts[progress_index + 1][0]
            if progress_index + 1 < len(progress_starts)
            else len(turn_rows)
        )
        progress_rows = turn_rows[row_index + 1 : stop_index]
        progress_tts_start = next(
            (item for item in progress_rows if item.get("event") == "tts_progress_start"),
            None,
        )
        progress_pcm = next(
            (item for item in progress_rows if item.get("event") == "tts_progress_pcm"),
            None,
        )
        progress_packet = next(
            (
                item
                for item in turn_rows
                if item.get("event") == "playback_packet"
                and response_id is not None
                and item.get("response_id") == response_id
            ),
            None,
        )
        acknowledgements.append(
            {
                "response_kind": "acknowledgement",
                "tts_start_to_first_pcm_ms": _elapsed_ms(
                    progress_tts_start, progress_pcm
                ),
                "first_pcm_to_client_delivery_ms": _elapsed_ms(
                    progress_pcm, progress_packet
                ),
            }
        )

    return {
        "turn_id": turn_id,
        "asr_utterance_id": utterance_id,
        "response_kind": "final_answer",
        "vad_speech_end_to_asr_final_ms": _elapsed_ms(speech_end, asr_final),
        "llm_provider_request_to_first_token_ms": _elapsed_ms(llm_request, first_token),
        "tts_start_to_first_pcm_ms": _elapsed_ms(final_tts_start, final_pcm),
        "first_pcm_to_client_delivery_ms": _elapsed_ms(final_pcm, final_packet),
        "acknowledgement_audio": acknowledgements,
    }


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
        conversation_mode="continuous",
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
        self.turn_terminal_outcomes = {}
        self.fast_action_rows = []
        self.pending_communication_confirmation = None
        self.pending_communication_request = None
        self._approval_announcement_ids = set()
        self._approval_announcement_jobs = set()
        self.response_records = {}
        self.response_order = deque(maxlen=64)
        self.response_by_conversation_turn = {}
        self.active_response_id = None
        self.progress_response_ids = deque(maxlen=64)
        self.last_llm_request_id = None
        self.last_vad_started_at = None
        self.playback_busy_until = 0.0
        self.research_ack_turns = set()
        self.research_recovery_active = False
        self.action_revision = 0
        self.turn_id = 0  # Telemetry projection of native user-turn events only.
        self.tavily_quick_research_designated = False
        self.tavily_quick_research_turns = set()
        self.run_id = uuid.uuid4().hex
        self.projection = "listening" if annotated else "wake_listening"
        self.gate = WakeGate(
            wake,
            active=annotated,
            on_wake=self._on_wake,
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
        self.startup_stage = "not_started"
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
            previous_turn_id = self.turn_id
            if (
                previous_turn_id in self.tavily_quick_research_turns
                and previous_turn_id not in self.turn_terminal_outcomes
            ):
                self.tavily_quick_research_turns.discard(previous_turn_id)
                self.event(
                    "turn_interrupted",
                    owner_turn_id=previous_turn_id,
                    reason="recognized_user_transcript",
                )
            previous = self.response_records.get(self.active_response_id)
            if (
                previous_turn_id > 0
                and previous is not None
                and previous.get("turn_id") == previous_turn_id
                and previous.get("response_state") == "pending"
                and previous_turn_id not in self.turn_terminal_outcomes
            ):
                previous["response_state"] = "interrupted"
                previous["interruption_reason"] = "recognized_user_transcript"
                if previous.get("generated_state") == "streaming":
                    previous["generated_state"] = "partial"
                self.event(
                    "turn_interrupted",
                    owner_turn_id=previous_turn_id,
                    response_id=previous["response_id"],
                    reason="recognized_user_transcript",
                )
            self.turn_id += 1
            self.playback_busy_until = 0.0
            self.research_ack_turns.clear()
            previous = self.response_records.get(self.active_response_id)
            if previous and previous.get("response_state") in {"interrupted", "superseded"}:
                previous["superseded_by_turn_id"] = self.turn_id
        elif name == "interrupt_requested" and self.turn_id in self.tavily_quick_research_turns:
            self.tavily_quick_research_turns.discard(self.turn_id)
            self.event(
                "turn_interrupted",
                owner_turn_id=self.turn_id,
                reason=fields.get("reason", "user_requested"),
            )
        owner_turn_id = fields.pop("owner_turn_id", None)
        event_turn_id = self.turn_id
        if type(owner_turn_id) is int and owner_turn_id >= 0:
            event_turn_id = owner_turn_id
        if name in _TERMINAL_TURN_EVENTS:
            if event_turn_id in self.turn_terminal_outcomes:
                return None
            self.turn_terminal_outcomes[event_turn_id] = name
            self.tavily_quick_research_turns.discard(event_turn_id)
        if name == "tavily_quick_research_started":
            self.tavily_quick_research_turns.add(event_turn_id)
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
                    "generated_text": "",
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
        elif name == "tts_progress_started":
            response_id = fields.get("response_id") or fields.get("context_id")
            if isinstance(response_id, str) and response_id not in self.progress_response_ids:
                self.progress_response_ids.append(response_id)
        elif name == "llm_request":
            self.last_llm_request_id = fields.get("request_id")
        progress_event = name.startswith("tts_progress_")
        response_id = fields.get("response_id") or (
            self.progress_response_ids[-1]
            if progress_event and self.progress_response_ids
            else None
            if progress_event
            else self.active_response_id
        )
        record = None if progress_event else self.response_records.get(response_id)
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
                if fields.get("status") == "failed":
                    record["response_state"] = "failed"
                elif (
                    fields.get("status") == "cancelled"
                    and record.get("response_state") == "pending"
                ):
                    record["response_state"] = "interrupted"
            elif name == "playback_packet":
                record["playback_packets"] += 1
                record["playback_state"] = "queued"
            elif name == "playback_ack":
                record["playback_acks"] += 1
                record["playback_state"] = "browser_acknowledged"
                queued = fields.get("queued_audio_seconds")
                if (
                    response_id == self.active_response_id
                    and record.get("response_state") not in {"interrupted", "superseded"}
                    and type(queued) in (int, float)
                    and 0 <= queued <= 120
                ):
                    self.playback_busy_until = max(
                        self.playback_busy_until, time.monotonic() + queued
                    )
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
                response_turn_id = record.get("turn_id")
                if (
                    type(response_turn_id) is int
                    and response_turn_id not in self.turn_terminal_outcomes
                ):
                    self.event(
                        "turn_interrupted",
                        owner_turn_id=response_turn_id,
                        response_id=record["response_id"],
                        reason=record["interruption_reason"],
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
            elif name in {"turn_failed", "turn_error", "turn_interrupted"}:
                record["response_state"] = "interrupted" if name == "turn_interrupted" else "failed"
                if name == "turn_interrupted":
                    record["interruption_reason"] = fields.get("reason", "interrupted")
        if name == "interrupt_requested":
            active = self.response_records.get(self.active_response_id)
            if (
                active
                and active.get("response_state") == "pending"
                and active.get("turn_id") not in self.turn_terminal_outcomes
            ):
                active["response_state"] = "interrupted"
                active["interruption_reason"] = fields.get("reason", "user_requested")
                self.event(
                    "turn_interrupted",
                    owner_turn_id=active["turn_id"],
                    response_id=active["response_id"],
                    reason=active["interruption_reason"],
                )
        if name in {"tool_started", "tool_completed", "tool_operation_blocked"}:
            fields.setdefault("operation_id", fields.get("tool_call_id"))
            fields.setdefault("llm_request_id", self.last_llm_request_id)
        audio_event = name in {
            "tts_start",
            "tts_pcm",
            "tts_segment_summary",
            "tts_progress_start",
            "tts_progress_pcm",
            "tts_progress_segment_summary",
            "playback_packet",
        }
        if audio_event and isinstance(response_id, str):
            if not isinstance(fields.get("response_id"), str):
                fields["response_id"] = response_id
            if not isinstance(fields.get("response_kind"), str):
                fields["response_kind"] = (
                    "acknowledgement"
                    if response_id in self.progress_response_ids
                    else "final_answer"
                    if response_id in self.response_records
                    else "unclassified"
                )
        value = {
            "event": name,
            "at": time.monotonic(),
            "run_id": self.run_id,
            "session_id": self.resources.session_id,
            "turn_id": event_turn_id,
            **fields,
        }
        self.event_count += 1
        value["sequence"] = self.event_count
        self.events.append(value)
        if self.on_event:
            self.on_event(value)
        if name == "turn_completed":
            summary = _voice_latency_for_turn(self.events, event_turn_id)
            if summary is not None:
                self.event("voice_latency", **summary)
        return value

    def task_event(self, payload):
        """Project durable task events into the native event stream."""
        if not isinstance(payload, dict):
            self.event("task_update", detail="invalid_task_event")
            return
        name = payload.get("event")
        fields = {key: value for key, value in payload.items() if key != "event"}
        if name == "task_update" and fields.get("kind") == "approval_requested":
            self._queue_github_approval_announcement(fields)
        self.event(name if isinstance(name, str) else "task_update", **fields)

    def _voice_owned_task(self, task_id):
        runtime = self.resources.task_runtime
        if runtime is None or runtime.store is None:
            return None
        row = runtime.store.rows.get(task_id)
        if not isinstance(row, dict):
            return None
        expected = _voice_task_request_id(self.resources.session_id, row.get("call_id"))
        return row if expected and row.get("request_id") == expected else None

    def _pending_voice_github_approvals(self):
        runtime = self.resources.task_runtime
        store = getattr(runtime, "store", None) if runtime else None
        if store is None:
            return []
        rows = []
        for task_id, approval in store.pending_approvals.items():
            task = self._voice_owned_task(task_id)
            if (
                task is None
                or task.get("status") != "waiting_approval"
                or not isinstance(approval, dict)
                or approval.get("status") != "pending"
                or approval.get("action") not in _GITHUB_APPROVAL_ACTIONS
                or type(approval.get("expires")) not in (int, float)
                or approval["expires"] <= time.time()
            ):
                continue
            rows.append((task, approval))
        return rows

    def _queue_github_approval_announcement(self, fields):
        task_id = fields.get("task_id")
        payload = fields.get("payload")
        if not isinstance(task_id, str) or not self._voice_owned_task(task_id):
            return
        if not isinstance(payload, dict) or payload.get("action") not in _GITHUB_APPROVAL_ACTIONS:
            return
        approval_id = payload.get("approval_id")
        summary = payload.get("spoken_summary")
        if (
            not isinstance(approval_id, str)
            or not re.fullmatch(r"approval_[a-f0-9]{32}", approval_id)
            or approval_id in self._approval_announcement_ids
            or not isinstance(summary, str)
            or not 1 <= len(summary) <= 500
        ):
            return
        self._approval_announcement_ids.add(approval_id)
        try:
            job = asyncio.get_running_loop().create_task(
                self._speak_github_approval_summary(approval_id, summary),
                name="swift-github-approval-" + approval_id[-8:],
            )
        except RuntimeError:
            return
        self._approval_announcement_jobs.add(job)
        job.add_done_callback(self._approval_announcement_jobs.discard)

    async def _speak_github_approval_summary(self, approval_id, summary):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 15
        while loop.time() < deadline:
            parts = self.parts
            if (
                parts is not None
                and self.error is None
                and self.settled.is_set()
                and self.projection in {"listening", "wake_listening"}
                and not parts.stt.speaking
            ):
                self.event("github_approval_summary_queued", approval_id=approval_id)
                await parts.tts.queue_frame(
                    ProgressTTSSpeakFrame(
                        summary,
                        owner_id="github-approval:" + approval_id,
                        owner_turn_id=max(1, self.turn_id),
                    )
                )
                return
            await asyncio.sleep(0.1)
        self.event("github_approval_summary_deferred", approval_id=approval_id)

    async def _handle_github_voice_control(self, text):
        if not _looks_like_github_approval_or_request(text):
            return None
        runtime = self.resources.task_runtime
        pending = self._pending_voice_github_approvals()
        intent = _github_approval_intent(text)
        if not pending:
            phrase = _normalize_approval_utterance(text)
            if (intent is not None and intent[0] == "approve") or re.match(
                r"^(?:yes|yeah|yep|approve|approved|confirm)\b", phrase
            ):
                self.event("github_voice_approval_rejected", reason="no_pending_approval")
                return "There is no pending GitHub approval in this conversation. Nothing was changed."
            return None
        if len(pending) != 1:
            self.event("github_voice_approval_rejected", reason="multiple_pending_approvals")
            return "I found more than one pending GitHub approval. Use the approval buttons in the task card."
        task, approval = pending[0]
        if intent is None:
            self.event(
                "github_voice_approval_rejected",
                reason="ambiguous_or_unrelated_utterance",
                task_id=task["task_id"],
                approval_id=approval["approval_id"],
            )
            return "That did not clearly approve or cancel the pending GitHub action. Say approve or cancel."
        decision, required_action = intent
        if required_action is not None and required_action != approval["action"]:
            self.event(
                "github_voice_approval_rejected",
                reason="approval_action_mismatch",
                task_id=task["task_id"],
                approval_id=approval["approval_id"],
            )
            return "That confirmation does not match the pending GitHub action. Say approve or cancel."
        try:
            await runtime.control(
                "approve",
                {
                    "task_id": task["task_id"],
                    "approval_id": approval["approval_id"],
                    "expected_revision": approval["revision"],
                    "decision": decision,
                },
            )
        except Exception as exc:
            code = _safe_tool_error_code(exc)
            self.event(
                "github_voice_approval_rejected",
                reason=code,
                task_id=task["task_id"],
                approval_id=approval["approval_id"],
            )
            return "That GitHub approval is stale or expired. No action was approved; review the task card."
        self.event(
            "github_voice_approval_decided",
            decision=decision,
            task_id=task["task_id"],
            approval_id=approval["approval_id"],
            action=approval["action"],
        )
        if decision == "reject":
            if approval["action"] == "github_pages_deploy":
                return "Cancelled the Pages deployment. The previously approved branch is unchanged."
            return "Cancelled the GitHub branch push. No branch was pushed."
        return "Approval recorded for that exact GitHub action. Swift is proceeding and will report the result."

    async def _handle_github_publish_request(self, text):
        from .task_runtime import PAGES_REQUEST_NOTE

        phrase = _normalize_approval_utterance(text)
        if re.search(
            r"\b(?:do not|don't|never|without) (?:push|publish|deploy|upload|put|make)\b", phrase
        ):
            return None
        pages = _pages_request(text)
        repository_request = (
            re.search(r"\b(?:create|edit|update|modify|push|publish|upload|put)\b", text, re.I)
            and re.search(r"\bgithub(?:\s+pages)?\b", text, re.I)
        )
        if _github_publication_status(text) or not (pages or repository_request):
            return None
        runtime = self.resources.task_runtime
        if runtime is None or runtime.store is None or runtime.github_writer is None:
            return "GitHub publishing is not configured in this bridge. Nothing was changed."
        if self._pending_voice_github_approvals():
            return "That sounds like a new publishing request, not approval. Say approve or cancel for the pending action."
        active = [
            row
            for row in runtime.store.rows.values()
            if row.get("status") in {"queued", "running"}
            and self._voice_owned_task(row.get("task_id")) is not None
            and (
                not pages
                or re.search(r"\b(?:website|web page|site|portfolio)\b", row.get("goal", ""), re.I)
                or any(
                    path.rsplit("/", 1)[-1] == "index.html"
                    for path in row.get("_artifact_paths", ())
                )
            )
        ]
        new_site = bool(
            re.search(r"\b(?:create|build|generate|make)\b.*\b(?:website|web page|site)\b", text, re.I)
        )
        if pages and _make_site_live(phrase):
            new_site = False
        if pages and not new_site and not active:
            saved = [
                row
                for task_id in runtime.store.rows
                if (row := self._voice_owned_task(task_id)) is not None
                and row["status"] in {"succeeded", "completed_unverified"}
                and any(path.rsplit("/", 1)[-1] == "index.html" for path in row.get("_artifact_paths", ()))
            ]
            if len(saved) != 1:
                return "I need one saved website in this conversation to publish. Which website should I use?"
            task = saved[0]
            if task.get("_publication", {}).get("pages_url"):
                return "This website is already live on GitHub Pages. The verified link is in its task card."
            try:
                await runtime.publish_saved_site(task["task_id"], task["revision"])
            except Exception as exc:
                self.event("github_publish_request_not_prepared", reason=_safe_tool_error_code(exc))
                return "I couldn't prepare those saved files for publication."
            return "I'm preparing the saved website. I'll ask once to approve the push and public deployment."
        if len(active) == 0 or (pages and new_site):
            goal = text.strip()
            if pages:
                goal += "\n\n" + PAGES_REQUEST_NOTE + " Use publish_github_branch with deploy_pages=true."
            call = ToolCall(
                "voice-github-task-" + uuid.uuid4().hex,
                "start_task",
                {"goal": goal},
            )
            self.event("route_selected", route="fast_task_handoff", task_kind="github")
            try:
                result = await self.dispatch_task_tool(call, call.arguments)
            except Exception as exc:
                self.event(
                    "github_publish_request_not_prepared",
                    reason=_safe_tool_error_code(exc),
                )
                return "I couldn't start the GitHub task. No repository or Pages changes were made."
            if not isinstance(result, dict) or result.get("status") not in {"queued", "running"}:
                return "The GitHub task was not accepted, so no repository changes were made."
            self.event(
                "github_publish_task_started",
                task_id=result.get("task_id"),
                pages=pages,
            )
            return (
                "I've started the GitHub task. It will inspect the configured repository and prepare the requested files; "
                "nothing will be pushed until you approve the exact change."
            )
        if len(active) != 1:
            self.event(
                "github_publish_request_not_prepared",
                reason="no_unique_active_voice_task",
                active_tasks=len(active),
            )
            return "I need one active website task to prepare that publication. Include GitHub in the original website request or try again while its task is running."
        task = active[0]
        instruction = (
            "Prepare the requested GitHub file operation for this task's saved and verified artifacts. "
            "Use github_list_files and github_read_file to discover the configured repository paths and read existing files before edits. "
            "For a new site, use a new nested directory under me3-voice-agent/artifacts/ and operation=create. "
            "For an existing site, preserve its directory and unrelated files, read the current files, then use operation=update. "
            "For an ordinary file push, select only the requested artifact paths and explain whether paths are new or updates. "
            "Use the persistent swift-artifacts branch for generated files; keep each non-website project under "
            "projects/<stable-project-slug>/ and reuse that slug for a deliberate update. Never write to main. "
            "Do not perform any remote write until the existing approval control is explicitly "
            "approved. A Pages approval covers the branch push and public deployment. "
            "The Pages source must remain stable and separate from swift-artifacts."
        )
        if pages:
            instruction += " " + PAGES_REQUEST_NOTE + " Use publish_github_branch with deploy_pages=true."
        try:
            result = await runtime.steer(task["task_id"], task["revision"], instruction)
        except Exception as exc:
            self.event(
                "github_publish_request_not_prepared",
                reason=_safe_tool_error_code(exc),
                task_id=task["task_id"],
            )
            return "I couldn't prepare the GitHub request. No repository or Pages changes were made."
        self.event(
            "github_publish_request_prepared",
            task_id=task["task_id"],
            deploy_pages=pages,
            status=result.get("status") if isinstance(result, dict) else None,
        )
        if pages:
            return "I asked the website task to prepare one approval for the branch push and public Pages deployment."
        return "I asked the website task to prepare a branch push. It will pause for your approval before any repository change."

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
            "call_demo",
            "message_demo",
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
            "call_demo": f"Call {call.arguments.get('value') or 'contact'}",
            "message_demo": f"Message {call.arguments.get('value') or 'contact'}",
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
        import re

        from .fast_commands import (
            direct_clock_query,
            direct_light_command,
            direct_me2_command,
            direct_music_control,
            direct_music_request,
            direct_native_music_control,
            direct_light_followup,
            direct_clarification,
            direct_sms_edit,
            demo_owner_identity_query,
        )

        messages = context.get_messages()
        history, tool_requests = [], {}
        previous_user = current_user = None
        history_turn = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                history_turn += 1
                if isinstance(message.get("content"), str):
                    previous_user, current_user = current_user, message
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
        github_control = await self._handle_github_voice_control(text)
        if github_control is not None:
            return github_control
        github_publish_request = await self._handle_github_publish_request(text)
        if github_publish_request is not None:
            return github_publish_request
        # A clear website creation request belongs to the durable coding
        # runtime. Sending the same composite request through foreground
        # research first can consume the citation/evidence turn budget before
        # the model ever hands work to LangGraph. Let the task own both source
        # retrieval and artifact generation, while this voice turn only
        # confirms durable acceptance.
        website_creation = re.search(
            r"(?is)\b(?:create|build|make|develop|generate)\b.*\b(?:website|web\s*page|webpage|web\s*site)\b",
            text,
        )
        if website_creation and self.resources.task_runtime is not None:
            self.event("route_selected", route="fast_task_handoff", task_kind="website")
            goal = text.strip()
            personal_site = bool(
                re.search(
                    r"\bmy (?:personal )?(?:website|web\s*page|webpage|web\s*site|portfolio|bio)\b"
                    r"|\bpersonal (?:website|web\s*page|webpage|web\s*site|portfolio|bio)\b"
                    r"|\b(?:website|web\s*page|webpage|web\s*site|portfolio|bio)\s+(?:about|for) me\b"
                    r"|\b(?:create|build|make|develop|generate) me (?:a )?(?:personal )?"
                    r"(?:website|web\s*page|webpage|web\s*site|portfolio|bio)\b",
                    text,
                    re.I,
                )
            )
            profile = getattr(self.resources, "owner_profile", {})
            if personal_site and isinstance(profile, dict) and profile.get("display_name"):
                facts = ["Name: " + profile["display_name"]]
                if profile.get("current_role"):
                    facts.append("Current role: " + profile["current_role"])
                goal += (
                    "\n\nApproved private profile facts for this requested personal website: "
                    + "; ".join(facts)
                    + ". Use only these confirmed facts; do not infer employer, contact details, "
                    "location, or other personal information. Generate the local artifact only; "
                    "public publication still requires separate explicit approval."
                )
            call = ToolCall(
                "voice-fast-task-" + uuid.uuid4().hex,
                "start_task",
                {"goal": goal},
            )
            self.event(
                "tool_started",
                tool_call_id=call.call_id,
                tool_name=call.name,
                handoff="background_task",
            )
            try:
                result = await self.dispatch_task_tool(call, call.arguments)
            except Exception as exc:
                code = _safe_tool_error_code(exc)
                self.event(
                    "tool_completed",
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    ok=False,
                    error_code=code,
                )
                self.event("task_handoff_failed", error_code=code)
                return "I couldn't start the website task, so no background work was accepted."
            accepted = (
                isinstance(result, dict)
                and result.get("status") in {"queued", "running"}
                and isinstance(result.get("task_id"), str)
            )
            self.event(
                "tool_completed",
                tool_call_id=call.call_id,
                tool_name=call.name,
                ok=accepted,
                task_id=result.get("task_id") if isinstance(result, dict) else None,
                task_status=result.get("status") if isinstance(result, dict) else None,
            )
            if not accepted:
                self.event("task_handoff_failed", error_code="task_not_accepted")
                return "The website task was not accepted, so I haven't claimed it is running."
            self.event("task_handoff_accepted", task_id=result["task_id"])
            return "I've started the research and website task in the background. You can keep talking to me."
        repeat_identity = (
            previous_user is not None
            and previous_user.get(LOCAL_OWNER_TURN) is True
            and re.fullmatch(
                r"(?:please )?(?:repeat that|repeat your answer|say that again)",
                re.sub(r"\s+", " ", text.strip().casefold()).strip(" .!?"),
            )
        )
        if demo_owner_identity_query(text) or repeat_identity:
            current_user[LOCAL_OWNER_TURN] = True
            profile = getattr(self.resources, "owner_profile", {})
            name = (profile.get("display_name") if isinstance(profile, dict) else None) or getattr(
                self.resources, "owner_name", ""
            )
            role = profile.get("current_role") if isinstance(profile, dict) else None
            self.event("route_selected", route="local_demo_owner_identity")
            if name:
                detail = f"Your private demo profile lists {name}."
                detail += (
                    f" Your current role is {role}."
                    if role
                    else " Your current role is not confirmed in the profile."
                )
                return detail + " A spoken name does not authenticate anyone."
            return (
                "No private demo owner is configured. A spoken name does not authenticate anyone."
            )
        progress_question = re.fullmatch(
            r"(?i)(?:what(?:'s| is) the progress(?: on (?:the )?(?:website|task))?|"
            r"how(?:'s| is) (?:the )?(?:website )?task going|status of (?:the )?(?:website )?task)[?.! ]*",
            text.strip(),
        )
        github_status_pattern = (
            r"(?:what(?:'s| is) the )?(?:status of|progress on) "
            r"(?:the |my )?github(?: pages)? task"
        )
        phrase = text.casefold().replace("’", "'").strip(" ?!.")
        github_progress = bool(re.fullmatch(github_status_pattern, phrase)) or _github_publication_status(text)
        publication_question = _github_publication_status(text)
        if not github_progress and re.fullmatch(
            r"(?:(?:it was )?the task about (?:my )?(?:personal )?website"
            r"(?:[.,]? i don't have the task id)?|is it (?:done|finished)(?: yet)?)",
            phrase,
        ):
            previous = [
                row["content"].casefold().replace("’", "'").strip(" ?!.")
                for row in history
                if row["role"] == "user"
            ][-3:-1]
            if previous and previous[-1] == "it was the":
                previous.pop()
            github_progress = bool(previous and re.fullmatch(github_status_pattern, previous[-1]))
        if github_progress or progress_question:
            runtime = self.resources.task_runtime
            self.event("route_selected", route="fast_task_progress")
            if runtime is None or runtime.store is None:
                return "Background task status is unavailable right now."
            rows = [
                row
                for task_id in runtime.store.rows
                if (row := self._voice_owned_task(task_id)) is not None
            ]
            if github_progress:
                rows = [
                    row
                    for row in rows
                    if row["call_id"].startswith("voice-github-task-")
                    or re.search(r"\bgithub\b", row.get("goal") or "", re.I)
                    or row.get("_publication")
                ]
                if len(rows) > 1:
                    return "This conversation has more than one GitHub task. Which one do you mean?"
                if not rows:
                    return "I couldn't find a GitHub task in this conversation."
            active = [
                row
                for row in rows
                if row["status"]
                in {
                    "queued",
                    "running",
                    "pausing",
                    "paused",
                    "waiting_approval",
                    "cancel_requested",
                }
            ]
            if len(active) > 1:
                return "More than one background task is active. Which task do you mean?"
            selected = (
                rows[0]
                if len(rows) == 1
                else max(active or rows, key=lambda row: row["updated"], default=None)
            )
            if selected is None:
                return "There isn't a background task to report on."
            snapshot = await runtime.status(selected["task_id"])
            stage_row = next(
                (
                    event
                    for event in snapshot["progress"]
                    if event["kind"] == "stage_update" and isinstance(event.get("payload"), dict)
                ),
                None,
            )
            status = snapshot["status"]
            publication = snapshot.get("publication", {})
            if publication_question:
                self.event("github_publication_status_read", task_id=selected["task_id"], task_status=status)
                if publication.get("pages_url"):
                    return "The website is live on GitHub Pages. The verified public link is in the task card."
                if status == "waiting_approval":
                    return "The GitHub task is waiting for approval."
                if publication.get("repository_url"):
                    return "The branch was pushed to GitHub. There is no verified public Pages URL for this task."
            stage = stage_row["payload"].get("stage") if stage_row and status == "running" else None
            message = {
                "queued": "The task is queued.",
                "pausing": "The task is pausing.",
                "succeeded": "The task is complete.",
                "completed_unverified": "The task finished, but its result needs review.",
                "failed": "The task failed.",
                "cancelled": "The task was cancelled.",
                "paused": "The task is paused.",
                "waiting_approval": "The task is waiting for approval.",
                "cancel_requested": "The task is stopping.",
            }.get(status)
            if status == "running":
                stage_message = stage_row["payload"].get("message") if stage_row else None
                message = (
                    stage_message
                    if isinstance(stage_message, str) and stage_message.strip()
                    else ("The task is in progress.")
                )
            self.event(
                "task_progress_read",
                task_id=selected["task_id"],
                task_status=status,
                stage=stage,
            )
            return message or "I couldn't confirm the task's current state."
        # Exact explicit task control shares the existing local fast-action
        # boundary; no provider round trip is needed just to request cancellation.
        cancel = re.fullmatch(
            r"(?i)cancel (?:the |my )?(?:current |running )?(?:background )?task(?: (task_[a-f0-9]{24}))?(?: now)?[.!]?",
            text.strip(),
        )
        if cancel and self.resources.task_runtime is not None:
            runtime = self.resources.task_runtime
            task_id = cancel.group(1)
            if task_id is None:
                active = [
                    row
                    for row in runtime.store.rows.values()
                    if row["status"]
                    in {"queued", "running", "paused", "waiting_approval", "cancel_requested"}
                ]
                if len(active) != 1:
                    return "Which background task would you like me to cancel?"
                task_id = active[0]["task_id"]
            if task_id not in runtime.store.rows:
                return "I couldn't find that background task."
            result = await runtime.cancel(task_id)
            self.event("route_selected", route="fast_task_cancel")
            self.event("task_cancel_ack", status=result["status"], task_id=task_id)
            if result["status"] == "cancelled":
                return "The background task is cancelled."
            return "Cancellation requested for the background task."
        turn = max(1, self.turn_id)
        call = None
        pending_confirmation = self.pending_communication_confirmation
        confirmed_communication = False
        if pending_confirmation is not None:
            phrase = _normalize_command_phrase(text)
            phrase = re.sub(r"^hey swift\s+", "", phrase)
            pending_call = pending_confirmation["call"]
            if pending_call.arguments.get("action") == "message_demo":
                replacement = direct_sms_edit(text)
                if replacement is not None:
                    from .communications import validate_message_text

                    try:
                        pending_call.arguments["message"] = validate_message_text(replacement)
                    except ValueError:
                        return "That draft must be a single-line text of at most 1000 characters. The existing draft is unchanged and no message was sent."
                    self.pending_communication_confirmation = pending_confirmation
                    body = pending_call.arguments.get("message")
                    recipient = pending_call.arguments.get("value")
                    confirmation = pending_confirmation["confirmation"]
                    return f"I updated the text for {recipient}. Draft: ‘{body}’ Say ‘{confirmation}’ to send, or ‘cancel’ to stop."
            if phrase == pending_confirmation["confirmation"]:
                self.pending_communication_confirmation = None
                call = pending_confirmation["call"]
                confirmed_communication = True
                self.event(
                    "outgoing_action_confirmed",
                    action=call.arguments.get("action"),
                    contact=call.arguments.get("value"),
                    confirmation_turn=history_turn,
                )
            elif phrase in {"cancel", "cancel it", "never mind", "nevermind"}:
                self.pending_communication_confirmation = None
                self.event("outgoing_action_confirmation_cancelled")
                return "Cancelled. No call was placed and no message was sent."
            else:
                self.pending_communication_confirmation = None
                self.event("outgoing_action_confirmation_expired")
        pending_request = self.pending_communication_request
        if call is None and pending_request is not None:
            self.pending_communication_request = None
            phrase = _normalize_command_phrase(text)
            phrase = re.sub(r"^hey swift\s+", "", phrase)
            if phrase in {"cancel", "cancel it", "never mind", "nevermind"}:
                self.event("outgoing_action_confirmation_cancelled")
                return "Cancelled. No call was placed and no message was sent."
            followup = direct_me2_command(text, turn)
            action = pending_request["action"]
            if (
                followup is not None
                and followup.name == "run_device_action"
                and followup.arguments.get("action") == action
                and followup.arguments.get("value")
            ):
                call = followup
            elif followup is not None and followup.arguments.get("action") != action:
                self.event("outgoing_action_confirmation_expired")
            else:
                call = ToolCall(
                    f"local-contact-{turn}",
                    "run_device_action",
                    {"action": action, "value": text.strip()},
                )
            if call is not None and pending_request.get("message") is not None:
                call.arguments["message"] = pending_request["message"]
        if call is None:
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
        action = call.arguments.get("action")
        if (
            not confirmed_communication
            and call.name == "run_device_action"
            and action in {"call_demo", "message_demo"}
        ):
            task_actions = getattr(self.resources.tasks, "actions", None)
            communication = getattr(task_actions, "communications", None)
            query = call.arguments.get("value")
            if not query:
                self.pending_communication_request = {
                    "action": action,
                    "message": call.arguments.get("message"),
                }
                question = "call" if action == "call_demo" else "message"
                return f"Who would you like me to {question}?"
            if communication is None or not communication.contacts:
                return "The private phone contact list is not configured. No call was placed and no message was sent."
            if action == "message_demo" and call.arguments.get("message") is not None:
                from .communications import validate_message_text

                try:
                    call.arguments["message"] = validate_message_text(call.arguments["message"])
                except ValueError:
                    return "That draft must be a single-line text of at most 1000 characters. No message was sent."
            match = communication.match_contact(query)
            if match["status"] == "missing":
                return "I couldn't find that name or alias in the private contact list. No phone action was taken."
            if match["status"] == "ambiguous":
                self.pending_communication_request = {
                    "action": action,
                    "message": call.arguments.get("message"),
                }
                names = ", ".join(match["matches"])
                return f"I found more than one matching contact: {names}. Which one should I use?"
            contact_name = match["name"]
            call.arguments["value"] = contact_name
            action_word = "call" if action == "call_demo" else "text"
            confirmation = _normalize_command_phrase(
                f"repeat call to {contact_name.casefold()}"
                if action == "call_demo"
                else f"repeat message to {contact_name.casefold()}"
            )
            self.pending_communication_confirmation = {
                "call": call,
                "confirmation": confirmation,
            }
            self.event(
                "outgoing_action_confirmation_requested",
                action=action,
                contact=contact_name,
            )
            if action == "message_demo":
                from .communications import DEMO_MESSAGE

                body = call.arguments.get("message", DEMO_MESSAGE)
                return f"I can text {contact_name}. Draft: ‘{body}’ Say ‘{confirmation}’ to send, or ‘cancel’ to stop."
            return f"I can {action_word} {contact_name}. Say ‘{confirmation}’ to proceed, or ‘cancel’ to stop."
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
        spoken_summary = result.get("spoken_summary") if isinstance(result, dict) else None
        return (
            spoken_summary
            if isinstance(spoken_summary, str) and spoken_summary.strip()
            else summary
        )

    async def __aenter__(self):
        try:
            return await self._start()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def _start(self):
        self.startup_stage = "speech_startup"
        speech = await self.resources.start()
        for startup in getattr(speech, "startup_diagnostics", ()):
            self.event("speech_mcp_ready", **startup)
        if self.resources.research is None:
            self.resources.research = ResearchClient(
                self.options.research, self.resources.credentials
            )
        if self.resources.task_runtime is None:
            self.startup_stage = "task_runtime_startup"
            from .task_runtime import TaskRuntime

            factory = self.resources.task_runtime_factory or TaskRuntime
            task_spec, task_budget = self.resources.task_model_configuration(self.options)
            runtime_args = {
                "budget": task_budget,
                "research": self.resources.research,
            }
            if self.options.task_model:
                runtime_args["max_completion_tokens"] = self.options.task_max_output_tokens
            self.resources.task_runtime = factory(
                task_spec, self.resources.credentials, **runtime_args
            )
            await self.resources.task_runtime.start_runtime()
        self.resources.task_runtime.on_event = self.task_event
        self.startup_stage = "model_and_tool_setup"
        weather = getattr(self.resources.devices, "weather", None)
        self.llm = NativeLLM(
            self.resources.backends["llm"],
            credentials=self.resources.credentials,
            budget=self.resources.budget,
            upstream=self.upstream,
            clock=self.clock,
            timezone_name=getattr(weather, "timezone_name", "UTC"),
            source_ids=lambda: set(self.resources.research.sources),
            source_lookup=lambda source_id: self.resources.research.sources.get(source_id),
            evidence_passage_ids=self.resources.research.evidence_passage_ids,
            evidence_passage_lookup=self.resources.research.lookup_evidence_passage,
            research_remaining=lambda: (
                _requests_remaining(
                    self.resources.research.options.request_limit,
                    self.resources.research.requests,
                )
                if self.resources.research.options.allow_live
                else None
            ),
            research_alternatives=lambda: self.resources.research.alternative_leads(limit=1),
            on_event=self.event,
            max_output_tokens=self.options.llm_max_output_tokens,
            max_model_calls_per_turn=self.options.max_model_calls_per_turn,
            max_tool_calls_per_turn=self.options.max_tool_calls_per_turn,
            max_repeated_tool_calls=self.options.max_repeated_tool_calls,
            fast_action=self.fast_action,
            tavily_quick_research=self.tavily_quick_research,
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

        def progress_allowed(frame):
            if frame.owner_id.startswith("github-approval:"):
                return (
                    self.settled.is_set()
                    and self.projection in {"listening", "wake_listening"}
                    and not self.parts.stt.speaking
                    and self.error is None
                )
            response_id = self.response_by_conversation_turn.get(self.turn_id)
            response = self.response_records.get(response_id) if response_id else None
            # A tool-call response record exists before the task has spoken.
            # Permit its one acceptance cue while that response has produced
            # neither TTS segments nor playback packets; never interrupt speech.
            before_answer_audio = response is None or (
                not response.get("tts_segments") and response.get("playback_packets", 0) == 0
            )
            return (
                frame.owner_turn_id == max(1, self.turn_id)
                and not self.settled.is_set()
                and before_answer_audio
                and self.error is None
            )

        self.parts.tts.progress_allowed = progress_allowed
        self.startup_stage = "pipeline_initialization"
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
            # A pipeline error before on_pipeline_started must wake the
            # session initializer so it can report the real failure instead
            # of waiting for the generic readiness timeout.
            self.ready.set()
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
                error_role=getattr(self.error, "role", None),
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
        async def user_started(_aggregator, strategy):
            self._cancel_followup_timer()
            self.research_recovery_active = False
            start_evidence = getattr(strategy, "start_evidence", "asr_interim")
            previous = self.response_records.get(self.active_response_id)
            supersedes_pending = bool(previous and previous.get("response_state") == "pending")
            self.llm.new_turn()
            if supersedes_pending:
                response_id = previous.get("response_id")
                self.event(
                    "barge_in_decided",
                    response_id=response_id,
                    utterance_id=self.parts.stt.turn,
                    start_evidence=start_evidence,
                )
                self.output.suppress_response(response_id)
                # Stop only the old assistant branch. A framework broadcast
                # would travel upstream and abort the ASR stream carrying the
                # newly recognized user turn.
                interruption = InterruptionFrame()
                self.event(
                    "barge_in_cancel_dispatched",
                    response_id=response_id,
                    utterance_id=self.parts.stt.turn,
                    interruption_id=interruption.id,
                )
                await self.llm.process_frame(interruption, FrameDirection.DOWNSTREAM)
            if self.tavily_quick_research_designated:
                self.tavily_quick_research_designated = False
                self.llm.designate_tavily_quick_research()
            self.terminal_failure_code = None
            self.settled.clear()
            self.projection = "listening"
            self.event(
                "turn_started",
                start_evidence=start_evidence,
            )

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
            conversation_turn = sum(
                item.get("role") == "user"
                for item in self.resources.context.get_messages()
                if isinstance(item, dict)
            )
            response_id = self.response_by_conversation_turn.get(conversation_turn)
            response = self.response_records.get(response_id)
            if response is not None and isinstance(message.content, str) and message.content:
                response["generated_text"] = message.content[:12000]
                if message.interrupted:
                    response["generated_state"] = "partial"
                    if response.get("response_state") == "pending":
                        response["response_state"] = "interrupted"
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
                    # Pipecat's assistant aggregator may commit the text after
                    # this callback. Keep the generated response independently
                    # addressable for interruption/error archive projections.
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
            await self._handle_user_idle()

        self.runner = WorkerRunner(handle_sigint=False)
        await self.runner.add_workers(self.worker)
        self.job = asyncio.create_task(self.runner.run())
        self.startup_stage = "pipeline_readiness"
        await self._wait_for_pipeline_ready()
        self.startup_stage = "ready"
        return self

    async def _handle_user_idle(self):
        """Apply legacy wake-standby behavior only to timed conversation modes."""
        # Pipecat's input-idle callback can arrive while the assistant is
        # still thinking or speaking. The response-owned timer starts only
        # after the completed spoken turn.
        if (
            self.conversation_mode != "continuous"
            and self.settled.is_set()
            and self.projection == "listening"
            and not self.followup_timer
        ):
            self.standby()

    async def _wait_for_pipeline_ready(self, timeout=15):
        ready_wait = asyncio.create_task(self.ready.wait())
        try:
            done, _ = await asyncio.wait(
                {ready_wait, self.job},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                error = TimeoutError("native pipeline did not signal readiness")
                error.error_stage = "pipeline_startup"
                error.worker_state = "running" if not self.job.done() else "stopped"
                raise error
            if self.error is not None:
                raise self.error
            if self.job in done and not self.ready.is_set():
                if not self.job.cancelled():
                    exception = self.job.exception()
                    if exception is not None:
                        raise RuntimeError(
                            "native pipeline runner failed before readiness"
                        ) from exception
                raise RuntimeError("native pipeline runner stopped before readiness")
        finally:
            if not ready_wait.done():
                ready_wait.cancel()
                await asyncio.gather(ready_wait, return_exceptions=True)

    @staticmethod
    def registered_tool_schemas():
        """Return every Pipecat handler registered by the native application."""
        from .task_runtime import conversation_schemas

        return [*schemas(), *conversation_schemas()]

    @staticmethod
    def available_tool_schemas(
        research_options, research_requests=0, research_sources=(), *, research_recovery=False
    ):
        """Return tools available under the profile and active research turn."""
        result = []
        for schema in NativeApplication.registered_tool_schemas():
            if research_recovery and schema.name not in {
                "web_search",
                "retrieve_source",
                "verify_evidence",
            }:
                continue
            if schema.name in {"web_search", "retrieve_source"}:
                if not research_options.allow_live or _request_limit_reached(
                    research_options.request_limit, research_requests
                ):
                    continue
            elif schema.name == "verify_evidence" and not research_sources:
                continue
            result.append(schema)
        return result

    def _initial_research_lead(self, result):
        """Select the first eligible URL from this search's ranked results."""
        if (
            self.llm.research_searches != 1
            or self.llm.research_retrievals
            or len(self.llm.response_tool_indices) != 1
            or self.llm.calls >= self.llm.max_model_calls_per_turn
            or not isinstance(result, dict)
            or result.get("kind") != "untrusted_search_leads"
            or not isinstance(result.get("results"), list)
        ):
            return None
        research = self.resources.research
        eligible = {lead["url"] for lead in research.alternative_leads(limit=20)}
        return next(
            (
                row["url"]
                for row in result["results"]
                if isinstance(row, dict)
                and isinstance(row.get("url"), str)
                and row["url"] in eligible
            ),
            None,
        )

    def register_tools(self):
        from .task_runtime import validate_control_arguments

        task_tool_names = {"start_task", "task_status", "steer_task", "cancel_task"}
        research = self.resources.research
        all_schemas = self.registered_tool_schemas()

        def refresh_tools(context):
            refreshed = self.available_tool_schemas(
                research.options,
                research.requests,
                research.sources,
                research_recovery=self.research_recovery_active,
            )
            context.set_tools(ToolsSchema(standard_tools=refreshed))

        refresh_tools(self.resources.context)
        self.llm.refresh_tools = refresh_tools

        def register(schema):
            combined_search = (
                schema.name == "web_search" and self.llm._supports_structured_research_output()
            )

            async def execute(params, *, auto_fetch_url=None):
                epoch = self.llm.dispatch_epoch.get()
                call = (
                    ToolCall(
                        params.tool_call_id + "-fetch", "retrieve_source", {"url": auto_fetch_url}
                    )
                    if auto_fetch_url is not None
                    else ToolCall(params.tool_call_id, params.function_name, params.arguments)
                )
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
                if self.research_recovery_active and call.name not in {
                    "web_search",
                    "retrieve_source",
                    "verify_evidence",
                }:
                    blocked = "tool_unavailable_during_research"
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
                        hidden_research_tool = call.name in {
                            "web_search",
                            "retrieve_source",
                            "verify_evidence",
                        } and call.name not in {
                            schema.name
                            for schema in self.available_tool_schemas(
                                self.resources.research.options,
                                self.resources.research.requests,
                                self.resources.research.sources,
                            )
                        }
                        if hidden_research_tool:
                            result = {"ok": False, "error": "research_unavailable"}
                        elif call.name in task_tool_names:
                            args = validate_control_arguments(call.name, call.arguments)
                            result = await self.dispatch_task_tool(call, args)
                        else:
                            args = validate_arguments(call.name, call.arguments)
                        if hidden_research_tool:
                            # The call is intentionally rejected above. Do not fall
                            # through into its handler: research arguments were not
                            # validated and must never reach the provider client.
                            pass
                        elif call.name in task_tool_names:
                            pass
                        elif call.name == "run_device_action" and args.get("action") in {
                            "call_demo",
                            "message_demo",
                        }:
                            result = {"ok": False, "error": "explicit_confirmation_required"}
                            self.event(
                                "outgoing_action_blocked",
                                tool_call_id=call.call_id,
                                action=args["action"],
                                error_code="explicit_confirmation_required",
                            )
                        elif call.name == "get_runtime_time":
                            result = runtime_time(
                                args.get("timezone", self.llm.timezone_name), self.clock
                            )
                        elif call.name == "web_search":
                            self.research_recovery_active = True
                            alternatives = (
                                self.resources.research.alternative_leads()
                                if self.llm.retrieval_failures
                                else []
                            )
                            if alternatives:
                                result = {
                                    "kind": "untrusted_search_leads",
                                    "query": args["query"],
                                    "results": alternatives,
                                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                    "reused_existing_leads": True,
                                }
                            else:
                                operation = self.research_operation_with_ack(
                                    call,
                                    self.resources.research.search(
                                        args["query"], args.get("max_results", 5)
                                    ),
                                )
                                if combined_search:
                                    async with asyncio.timeout(8):
                                        result = await operation
                                else:
                                    result = await operation
                            if self.resources.research.options.answer_mode == "tavily":
                                # Keep the provider answer as a ranking hint only.
                                # GPT-OSS must compose from retained snippets so
                                # the normal citation/support validator runs.
                                result, _, _ = self.project_tavily_answer(args["query"], result)
                            else:
                                result = self.llm.project_search_snippets(
                                    self.resources.research, result, args["query"], limit=1
                                )
                            # Offer only the best-ranked snippet as answer evidence;
                            # keep other leads discoverable for retrieval. This avoids
                            # competing, partial snippets while preserving a fetch path.
                        elif call.name == "retrieve_source":
                            self.research_recovery_active = True
                            safe_url = public_url(args["url"])
                            parsed = urlsplit(safe_url)
                            source_origin = f"{parsed.scheme}://{parsed.hostname}"
                            source_url_hash = hashlib.sha256(safe_url.encode()).hexdigest()[:16]
                            research = self.resources.research
                            if safe_url in research.failed_retrieval_urls:
                                raise ResearchError(
                                    "source_previously_failed",
                                    host=parsed.hostname,
                                    stage="session_cache",
                                    failure_class="not_retried",
                                )
                            existing_alternatives = research.alternative_leads(
                                exclude_url=safe_url, limit=1
                            )
                            remaining_http = _requests_remaining(
                                research.options.request_limit, research.requests
                            )
                            # When only the direct fetch plus one alternate fit,
                            # leave the remaining HTTP allowance for the alternate.
                            allow_extract_fallback = not (
                                existing_alternatives
                                and remaining_http is not None
                                and remaining_http <= 2
                            )

                            async def fetch_source(url, *, allow_extract_fallback=True):
                                # Leave time for the existing alternate inside the
                                # eight-second automatic retrieval window.
                                timeout = RESEARCH_AUTO_SOURCE_SECONDS if auto_fetch_url else None
                                try:
                                    async with asyncio.timeout(timeout):
                                        return await research.retrieve(
                                            url, allow_extract_fallback=allow_extract_fallback
                                        )
                                except TimeoutError:
                                    research.failed_retrieval_urls.add(url)
                                    raise ResearchError(
                                        "research_timeout", host=urlsplit(url).hostname,
                                        stage="transport", failure_class="timeout",
                                    ) from None

                            try:
                                source = await self.research_operation_with_ack(
                                    call,
                                    fetch_source(
                                        safe_url,
                                        allow_extract_fallback=allow_extract_fallback,
                                    ),
                                )
                            except ResearchError as primary_error:
                                alternate_eligible = primary_error.code in {
                                    "research_http_failure",
                                    "research_timeout",
                                    "research_connect_failure",
                                    "research_connection_refused",
                                    "source_too_large",
                                    "insufficient_source_content",
                                    "unsupported_source_type",
                                }
                                alternatives = research.alternative_leads(
                                    exclude_url=safe_url, limit=1
                                )
                                if (
                                    not alternate_eligible
                                    or not alternatives
                                    or _request_limit_reached(
                                        research.options.request_limit, research.requests
                                    )
                                    or not self.llm.reserve_alternate_retrieval()
                                ):
                                    raise
                                alternate_url = public_url(alternatives[0]["url"])
                                alternate_hash = hashlib.sha256(alternate_url.encode()).hexdigest()[
                                    :16
                                ]
                                self.llm.record_retrieval_result(
                                    failed=True, error_code=primary_error.code
                                )
                                self.event(
                                    "research_alternate_source_started",
                                    primary_source_url_hash=source_url_hash,
                                    alternate_source_url_hash=alternate_hash,
                                    primary_error_code=primary_error.code,
                                    remaining_http_requests=_requests_remaining(
                                        research.options.request_limit, research.requests
                                    ),
                                )
                                try:
                                    source = await fetch_source(alternate_url)
                                except ResearchError as alternate_error:
                                    self.event(
                                        "research_alternate_source_failed",
                                        primary_source_url_hash=source_url_hash,
                                        alternate_source_url_hash=alternate_hash,
                                        error_code=alternate_error.code,
                                        http_status=alternate_error.status,
                                        host_hash=alternate_error.host_hash,
                                        stage=alternate_error.stage,
                                        failure_class=alternate_error.failure_class,
                                    )
                                    raise ResearchError(
                                        primary_error.code,
                                        status=primary_error.status,
                                        host=parsed.hostname,
                                        stage=primary_error.stage,
                                        failure_class=primary_error.failure_class,
                                        fallback_status=primary_error.fallback_status,
                                        fallback_error=primary_error.fallback_error
                                        or {"error": primary_error.code},
                                        alternate_source_attempt={
                                            "source_url_hash": alternate_hash,
                                            "error": alternate_error.code,
                                            "http_status": alternate_error.status,
                                            "host_hash": alternate_error.host_hash,
                                            "stage": alternate_error.stage,
                                            "failure_class": alternate_error.failure_class,
                                        },
                                    ) from None
                                source_origin = (
                                    f"{urlsplit(alternate_url).scheme}://"
                                    f"{urlsplit(alternate_url).hostname}"
                                )
                                source_url_hash = alternate_hash
                                self.event(
                                    "research_alternate_source_succeeded",
                                    primary_source_url_hash=hashlib.sha256(
                                        safe_url.encode()
                                    ).hexdigest()[:16],
                                    source_id=source.get("source_id"),
                                    source_url_hash=alternate_hash,
                                    extraction_method=source.get("extraction_method"),
                                )
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
                            result = self.llm.evidence_for_model(
                                self.resources.research, source, query
                            )
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
                                ToolCall(call.call_id, call.name, args),
                            )
                    except (ResearchError, ValueError) as exc:
                        # Symbolic policy errors only; never return exception payloads from I/O.
                        result = (
                            exc.result()
                            if isinstance(exc, ResearchError)
                            else {
                                "ok": False,
                                "error": (
                                    "research_evidence_budget_exhausted"
                                    if str(exc) == "research_evidence_budget_exhausted"
                                    else "invalid_arguments_or_tool_budget"
                                ),
                            }
                        )
                        if call.name == "retrieve_source" and isinstance(exc, ResearchError):
                            alternatives = self.resources.research.alternative_leads(
                                exclude_url=call.arguments.get("url")
                                if isinstance(call.arguments, dict)
                                else None
                            )
                            if alternatives:
                                result["alternative_leads"] = alternatives
                    except Exception as exc:
                        # Keep the provider/tool message private; the sanitized
                        # exception class is enough to localize a failed boundary.
                        failure_frame = exc.__traceback__
                        while failure_frame and failure_frame.tb_next:
                            failure_frame = failure_frame.tb_next
                        result = {
                            "ok": False,
                            "error": _safe_tool_error_code(exc),
                            "exception_type": type(exc).__name__,
                            "exception_function": (
                                failure_frame.tb_frame.f_code.co_name if failure_frame else None
                            ),
                            "exception_line": failure_frame.tb_lineno if failure_frame else None,
                        }
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
                    self.llm.record_retrieval_result(
                        failed=(failed and error_code != "source_previously_failed"),
                        error_code=error_code,
                    )
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
                        "alternate_source_attempt",
                        "reused_existing_leads",
                        "exception_type",
                        "exception_function",
                        "exception_line",
                    )
                    if isinstance(result, dict) and key in result
                }
                if result_kind == "retrieved_source":
                    evidence_text = result.get("text")
                    source_url_hash = hashlib.sha256(
                        str(result.get("url", "")).encode()
                    ).hexdigest()[:16]
                    safe_diagnostics.update(
                        extraction_method=result.get("extraction_method"),
                        evidence_status=result.get("evidence_status"),
                        source_id=result.get("source_id"),
                        source_url_hash=source_url_hash,
                        evidence_characters=len(evidence_text)
                        if isinstance(evidence_text, str)
                        else 0,
                        evidence_chunks=result.get("evidence_chunks"),
                    )
                elif result_kind == "search_leads":
                    search_evidence = [
                        row
                        for row in result.get("results", [])
                        if isinstance(row, dict)
                        and row.get("evidence_status") == "search_result_snippet"
                    ]
                    if search_evidence:
                        safe_diagnostics.update(
                            search_evidence_count=len(search_evidence),
                            source_ids=[row.get("source_id") for row in search_evidence],
                            extraction_method="tavily_search_snippet",
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
                if (
                    call.name == "start_task"
                    and isinstance(result, dict)
                    and result.get("task_id")
                    and result.get("status") in {"queued", "running"}
                ):
                    self.llm.pending_task_acceptance = (self.llm.epoch,)
                if auto_fetch_url is not None:
                    return result
                if call.name == "web_search" and not failed:
                    url = self._initial_research_lead(result) if combined_search else None
                    if url:
                        self.event("research_auto_retrieval_selected", tool_call_id=call.call_id)
                        try:
                            async with asyncio.timeout(8):
                                fetched = await execute(params, auto_fetch_url=url)
                        except TimeoutError:
                            self.llm.record_retrieval_result(
                                failed=True, error_code="research_timeout"
                            )
                            fetched = {"ok": False, "error": "research_timeout"}
                        if epoch != self.llm.epoch:
                            return
                        if isinstance(fetched, dict) and fetched.get("fetched") is True:
                            self.llm.automatic_research_result = (epoch, True)
                            result = {
                                **fetched,
                                "search_leads": [
                                    row
                                    for row in result["results"]
                                    if row.get("url") != fetched["url"]
                                ],
                            }
                        else:
                            self.llm.automatic_research_result = (epoch, False)
                            if isinstance(fetched, dict) and fetched.get("ok") is False:
                                result = {**result, "retrieval_error": fetched}
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
                timeout_secs=16 if combined_search else 8,
            )

        for schema in all_schemas:
            register(schema)

    async def dispatch_task_tool(self, call, args):
        runtime = self.resources.task_runtime
        if runtime is None:
            raise RuntimeError("task_runtime_unavailable")
        if call.name == "start_task":
            import hashlib
            import os
            import re

            request_id = (
                "voice_"
                + hashlib.sha256((self.resources.session_id + call.call_id).encode()).hexdigest()[
                    :40
                ]
            )
            call_id = call.call_id if len(call.call_id) <= 80 else "call_" + request_id[-40:]
            goal = args["goal"]
            demo_contract = " Swift local website task: materialize_demo is approved with push=false; use build_demo_page."
            if (
                os.environ.get("ME3_DEMO_OUTPUT")
                and re.search(r"\b(?:website|web\s+page|webpage|web\s+site)\b", goal, re.I)
                and "materialize_demo" not in goal
                and len(goal) + len(demo_contract) <= 4000
            ):
                goal += demo_contract
            return await runtime.start(
                goal, request_id=request_id, turn_id=max(1, self.turn_id), call_id=call_id
            )
        if call.name == "cancel_task":
            return await runtime.cancel(args["task_id"])
        if call.name == "task_status":
            return await runtime.status(args["task_id"])
        if call.name == "steer_task":
            return await runtime.steer(args["task_id"], args["expected_revision"], args["delta"])
        raise ValueError("unknown_task_command")

    async def research_operation_with_ack(self, call, operation):
        """Start retrieval now; queue one brief Pipecat cue only if it stays slow."""
        owner_turn = max(1, self.turn_id)
        if owner_turn in self.research_ack_turns:
            return await operation
        operation_task = asyncio.create_task(operation)

        async def acknowledge_if_still_running():
            await asyncio.sleep(RESEARCH_ACK_DELAY_SECONDS)
            if (
                operation_task.done()
                or owner_turn != max(1, self.turn_id)
                or self.error is not None
                or self.parts is None
            ):
                return
            # Claim before yielding so simultaneous research calls in one turn
            # cannot enqueue overlapping acknowledgments.
            self.research_ack_turns.add(owner_turn)
            self.event(
                "research_acknowledgment_queued",
                tool_call_id=call.call_id,
                owner_turn_id=owner_turn,
            )
            await self.parts.tts.queue_frame(
                ProgressTTSSpeakFrame(
                    "I'll check that.", owner_id=call.call_id, owner_turn_id=owner_turn
                )
            )

        acknowledgment = asyncio.create_task(acknowledge_if_still_running())
        try:
            return await operation_task
        finally:
            if not operation_task.done():
                operation_task.cancel()
            acknowledgment.cancel()
            await asyncio.gather(operation_task, acknowledgment, return_exceptions=True)

    def standby(self):
        self._cancel_followup_timer()
        self.playback_busy_until = 0.0
        self.gate.active = False
        self.projection = "wake_listening"
        if self.gate.wake and hasattr(self.gate.wake, "end_session"):
            self.gate.wake.end_session()
        self.event("engagement_expired")

    def _on_wake(self, result):
        self.projection = "listening"
        self.event("wake_detected", backend=result.get("backend"), score=result.get("score"))
        # Speech MCP children, the Groq client and the task runtime already
        # belong to this application. Start only the inactivity lease here;
        # opening a provider stream without speech would spend a request.
        self._arm_followup_timer()

    def _cancel_followup_timer(self):
        task, self.followup_timer = self.followup_timer, None
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _arm_followup_timer(self):
        self._cancel_followup_timer()
        if self.annotated or self.conversation_mode in {"single", "continuous"}:
            return

        async def expire():
            await asyncio.sleep(self.idle_seconds)
            remaining = self.playback_busy_until + self.idle_seconds - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            if (
                self.settled.is_set()
                and self.projection == "listening"
                and not self.parts.stt.speaking
            ):
                self.standby()

        self.followup_timer = asyncio.create_task(expire())

    async def feed(self, pcm, sequence):
        ingress_at = time.monotonic()
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
                await self.queue_audio(frame, ingress_at=ingress_at, ingress_sequence=sequence)

    async def queue_audio(self, pcm, *, ingress_at=None, ingress_sequence=None):
        now = time.monotonic()
        self.audio_frames += 1
        previous = self.last_audio_delivery_at
        self.last_audio_delivery_at = now
        self.event(
            "audio_frame_delivered",
            audio_sequence=self.audio_delivery_sequence,
            ingress_at=ingress_at,
            ingress_sequence=ingress_sequence,
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

    async def inject_test_text(self, text):
        """Explicit localhost qualification only; ASR is bypassed and labeled."""
        import os

        if os.environ.get("ME3_TEST_TEXT_INGRESS") != "1":
            raise ValueError("test_text_disabled")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("invalid_test_text")
        if self.projection in {"thinking", "speaking"}:
            await self.interrupt()
        self.settled.clear()
        self.event("test_text_injected", asr_bypassed=True)
        # Enter immediately after the recognizer, retaining the native user
        # aggregator, model/tool path, TTS and output/terminal lifecycle.
        await self.parts.stt.push_frame(VADUserStartedSpeakingFrame())
        await self.parts.stt.push_frame(
            TranscriptionFrame(text.strip(), "user", "", finalized=True)
        )
        await self.parts.stt.push_frame(VADUserStoppedSpeakingFrame())

    async def interrupt(self):
        self.event("interrupt_requested", reason="user_requested")
        self.playback_busy_until = 0.0
        self.error = None
        await self.worker.queue_frame(InterruptionFrame())
        # Reset current recognizer input explicitly; UI interruption is distinct
        # from a VAD onset, which preserves the new user's input in the adapter.
        self.parts.stt.speaking = False
        await self.parts.stt.abort()
        self.projection = "listening"
        self.settled.set()
        self._arm_followup_timer()

    def designate_next_tavily_quick_research(self):
        """Mark exactly the next user turn for direct Tavily quick research."""
        if self.options.research.answer_mode != "tavily":
            raise ValueError("tavily_quick_research_not_enabled")
        if self.tavily_quick_research_designated:
            raise ValueError("research_route_already_designated")
        self.tavily_quick_research_designated = True
        self.event("research_route_designated", route="tavily_quick_research")

    def project_tavily_answer(self, query, result):
        """Use the best returned snippet as the single citation for a Tavily answer."""
        answer = result.pop("answer", None)
        research = self.resources.research
        if isinstance(answer, str) and isinstance(result.get("results"), list):
            result["results"] = research.rank_answer_support(answer, result["results"])
        projected = self.llm.project_search_snippets(research, result, query, limit=1)
        source = next(
            (
                row
                for row in projected.get("results", [])
                if isinstance(row, dict)
                and row.get("evidence_status") == "search_result_snippet"
                and isinstance(row.get("source_id"), str)
            ),
            None,
        )
        return projected, answer, source.get("source_id") if source else None

    async def tavily_quick_research(self, question, epoch):
        """Run one Tavily answer search for a specifically designated native turn."""
        try:
            result = await self.resources.research.search(question)
        except asyncio.CancelledError:
            raise
        except ResearchError as exc:
            self.event("tavily_quick_research_failed", error_code=exc.code)
            return None, None
        except asyncio.TimeoutError:
            self.event("tavily_quick_research_failed", error_code="research_timeout")
            return None, None
        if epoch != self.llm.epoch:
            return None, None
        _, answer, source_id = self.project_tavily_answer(question, result)
        self.event(
            "tavily_quick_research_result",
            available=isinstance(answer, str) and bool(answer.strip()) and source_id is not None,
        )
        return answer, source_id

    async def __aexit__(self, *_):
        self._cancel_followup_timer()
        try:
            for announcement in tuple(self._approval_announcement_jobs):
                announcement.cancel()
            if self._approval_announcement_jobs:
                await asyncio.gather(
                    *tuple(self._approval_announcement_jobs), return_exceptions=True
                )
            self._approval_announcement_jobs.clear()
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
    selected_credentials = {p.auth_env for p in backends.values() if p.backend == "external"}
    if options.research.allow_live:
        selected_credentials.add("ME3_RESEARCH_TAVILY_API_KEY")
    credentials = load_secrets(args.secrets_file, selected_credentials) if args.secrets_file else {}
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
                    # Live conversation may recover to accept another utterance.
                    # A replay qualification must still retain any failed turn;
                    # later silence must not clear the failure into a PASS.
                    recorded_failure = next(
                        (
                            event
                            for event in reversed(application.events)
                            if event.get("event") == "turn_error"
                        ),
                        None,
                    )
                    if recorded_failure:
                        if recorded_failure.get("error_code") == "pipeline_error":
                            recorded_failure = next(
                                (
                                    event
                                    for event in reversed(application.events)
                                    if event.get("event") == "asr_turn_aborted"
                                    and event.get("turn_id") == recorded_failure.get("turn_id")
                                ),
                                recorded_failure,
                            )
                        raise ProviderError(
                            "native replay contained a failed turn",
                            code=recorded_failure.get("error_code") or "native_replay_turn_failed",
                            error_stage=recorded_failure.get("error_stage"),
                            error_boundary=recorded_failure.get("error_boundary"),
                        )
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
                "llm_provider_request_started",
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
                "voice_latency",
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
        completed_turn_ids = dict.fromkeys(
            event.get("turn_id")
            for event in result["events"]
            if event.get("event") == "turn_completed"
            and type(event.get("turn_id")) is int
        )
        result["voice_latency_by_turn"] = [
            metrics
            for turn_id in completed_turn_ids
            if (metrics := _voice_latency_for_turn(result["events"], turn_id)) is not None
        ]
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
