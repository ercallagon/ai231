"""Application-owned lifecycle. Playback confirmations are transport acknowledgments.

No acoustic claims are possible with the mock sink. Partial unacknowledged text
is omitted from history. Completed task effects would use a separate ledger.
"""

import asyncio
import time
import uuid
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from enum import Enum

from .contracts import TurnKey, FRAME_BYTES, MAX_AUDIO_BYTES
from .english import EnglishTextRejected, has_non_latin_letters, rejects_english_asr
from .providers import LLMProvider, ToolCall
from .tools import MOCK_ACTION_TITLES, MockTaskExecutor, task_title
from .music import MusicUnavailable


from .fast_commands import (
    direct_clarification,
    direct_clock_query,
    direct_light_command,
    direct_light_followup,
    direct_me2_command,
    direct_music_control,
    direct_music_request,
    direct_spelling_correction,
    music_selection_requested,
)
from .fast_commands import explicit_percentage as explicit_percentage


class State(str, Enum):
    IDLE = "idle"
    ACTIVATING = "activating"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Playback:
    def __init__(
        self, sink: Callable[[bytes], Awaitable[None]], valid, acknowledge, capacity=8, started=None
    ):
        self.queue = asyncio.Queue(maxsize=capacity)
        self.sink, self.valid, self.acknowledge = sink, valid, acknowledge
        self.inflight = None
        self.worker = None
        self.error = None
        self.started = started
        self.last_started_key = None

    def start(self):
        self.worker = asyncio.create_task(self.run())

    async def enqueue(self, key, pcm, text="", segment_id=None):
        if self.error:
            raise RuntimeError("playback sink failed") from self.error
        if self.valid(key):
            await self.queue.put((key, pcm, text, segment_id))

    async def run(self):
        while True:
            key, pcm, text, segment_id = await self.queue.get()
            try:
                if self.valid(key):
                    if pcm is None:
                        if not self.error:
                            try:
                                if hasattr(self.sink, "flush"):
                                    await self.sink.flush()
                            except Exception as exc:
                                self.error = exc
                            else:
                                if self.valid(key):
                                    self.acknowledge(key, text, segment_id)
                        continue
                    if self.started and self.last_started_key != key:
                        self.last_started_key = key
                        self.started(key)
                    self.inflight = asyncio.create_task(self.sink(pcm))
                    try:
                        await self.inflight
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
                    except Exception as exc:
                        self.error = exc
                    else:
                        if self.valid(key) and text:
                            self.acknowledge(key, text, segment_id)
            finally:
                self.inflight = None
                self.queue.task_done()

    async def stop(self):
        if self.inflight:
            self.inflight.cancel()
            with suppress(asyncio.CancelledError):
                await self.inflight
        if hasattr(self.sink, "stop"):
            await self.sink.stop()
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.error = None  # A stopped/drained turn must not poison the next reply.

    async def close(self):
        await self.stop()
        if self.worker:
            self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker
        if hasattr(self.sink, "close"):
            await self.sink.close()


class Session:
    def __init__(
        self,
        speech,
        llm: LLMProvider,
        sink,
        *,
        idle_seconds=30,
        turn_timeout=45,
        clock=time.monotonic,
        transcript=None,
        tasks=None,
        close_tasks=True,
        llm_timeout=5,
        execution_mode="mock",
        direct_clock_enabled=False,
        conversation_mode=None,
        followup_seconds=7,
        answer_seconds=18,
        extended_seconds=20,
    ):
        self.speech, self.llm, self.clock = speech, llm, clock
        self.transcript = transcript
        self.tasks = tasks or MockTaskExecutor()
        self.close_tasks = close_tasks
        self.llm_timeout = llm_timeout
        self.execution_mode = execution_mode
        self.direct_clock_enabled = direct_clock_enabled
        if conversation_mode not in {None, "single", "followup", "extended"}:
            raise ValueError("invalid conversation mode")
        if any(
            type(v) not in (int, float) or not 1 <= v <= 120
            for v in (followup_seconds, answer_seconds, extended_seconds)
        ):
            raise ValueError("invalid engagement window")
        self.conversation_mode = conversation_mode
        self.followup_seconds, self.answer_seconds = followup_seconds, answer_seconds
        self.extended_seconds = extended_seconds
        self.engagement_deadline = None
        self.candidate_started_at = None
        self.awaiting_answer = False

        self.asr_task = None
        self.input_queue = None
        self.input_bytes = 0
        self.input_open = False
        self.idle_seconds, self.turn_timeout = idle_seconds, turn_timeout
        self.state = State.IDLE
        self.session_id = uuid.uuid4().hex
        self.run_id = uuid.uuid4().hex
        self.turn = 0
        self.key = None
        self.history: list[dict] = []
        self.events: list[dict] = []
        self.ui_action_calls = {}
        self.ui_action_turns = set()
        self.on_event = None
        self.music_playing = False
        self.generation = None
        self.last_activity = clock()
        self.playback = Playback(
            sink,
            self.valid,
            self.acknowledge,
            started=self.playback_started,
        )

    def engagement_snapshot(self):
        remaining = (
            None
            if self.engagement_deadline is None
            else max(0, self.engagement_deadline - self.clock())
        )
        return {
            "mode": self.conversation_mode,
            "remaining_seconds": remaining,
            "awaiting_answer": self.awaiting_answer,
            "window_seconds": self.answer_seconds
            if self.awaiting_answer
            else self.extended_seconds
            if self.conversation_mode == "extended"
            else self.followup_seconds,
        }

    def open_followup(self):
        if self.conversation_mode:
            seconds = (
                self.answer_seconds
                if self.awaiting_answer
                else (
                    self.extended_seconds
                    if self.conversation_mode == "extended"
                    else self.followup_seconds
                )
            )
            self.engagement_deadline = self.clock() + seconds

    def playback_started(self, key):
        if self.valid(key):
            self.state = State.SPEAKING
            if self.transcript:
                self.transcript.set_state("speaking")
            self.event(f"sink_write_started_{self._sink_context()}")

    def _sink_context(self):
        delivery = getattr(self.playback.sink, "delivery", "")
        if delivery == "sink_confirmed":
            return "browser"
        if delivery == "benchmark_transport_acknowledged":
            return "benchmark"
        return "mock"

    async def __aenter__(self):
        self.playback.start()
        if hasattr(self.tasks, "prewarm"):
            self.tasks.prewarm()
        return self

    async def __aexit__(self, *exc):
        await self.exit()
        await self.playback.close()
        if self.close_tasks:
            await self.tasks.close()

    def event(self, name, **values):
        row = {
            "event": name,
            "monotonic": self.clock(),
            "session_id": self.session_id,
            "turn_id": self.turn,
            "mode": self.execution_mode,
            "schema_version": 1,
            "run_id": self.run_id,
            "clock_id": f"monotonic:{self.run_id}",
            "state": self.state.value,
            **values,
        }
        self.events.append(row)
        if self.on_event:
            self.on_event(row)

    def observe_music_playback(self, status, **fields):
        self.music_playing = status == "playing"
        self.event("music_playback_observed", status=status, **fields)
        if (
            not self.transcript
            or status not in {"paused", "not_playing"}
            or fields.get("source") not in {"web_playback_sdk", "spotify_api"}
        ):
            return
        task = next(
            (
                row
                for row in reversed(self.transcript.tasks)
                if row.get("turn_id") == self.turn
                and row.get("kind") in {"music_pause", "music_stop"}
                and row.get("status") == "unknown"
            ),
            None,
        )
        if task:
            source = "Spotify" if fields["source"] == "spotify_api" else "the browser player"
            state = "paused" if status == "paused" else "not playing"
            action = "stop" if task["kind"] == "music_stop" else "pause"
            self.transcript.task_observation(
                TurnKey(session_id=self.session_id, turn_id=self.turn),
                task["call_id"],
                f"Spotify could not confirm the {action} request; {source} now reports {state}. "
                "Whether the request caused this state remains unverified.",
            )

    def record_tool_outcome(self, key, call, value):
        if key.session_id != self.session_id:
            return
        row = next(
            (
                r
                for r in self.history
                if r.get("role") == "tool"
                and r.get("turn_id") == key.turn_id
                and r.get("tool_call_id") == call.call_id
            ),
            None,
        )
        if row is None:
            self.history.append(
                {
                    "role": "tool",
                    "content": json.dumps(value, sort_keys=True),
                    "tool_call_id": call.call_id,
                    "tool_request": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, sort_keys=True),
                    },
                    "turn_id": key.turn_id,
                }
            )
        else:
            row["content"] = json.dumps(value, sort_keys=True)

    def reconcile_task_result(self, key, call_id, value):
        """Settle matching uncertain action data independently of UI availability."""
        if key.session_id != self.session_id or not isinstance(value, dict):
            return False
        status = value.get("result_status", "reported" if value.get("mode") == "spotify" else None)
        if status not in {"simulated", "completed", "reported", "unavailable", "failed"}:
            return False
        changed = False
        for record in self.history:
            if (
                record.get("role") == "tool"
                and record.get("turn_id") == key.turn_id
                and record.get("tool_call_id") == call_id
            ):
                try:
                    prior = json.loads(record["content"])
                except (TypeError, ValueError):
                    continue
                if isinstance(prior, dict) and prior.get("result_status") == "unknown":
                    record["content"] = json.dumps(value, sort_keys=True)
                    changed = True
        if self.transcript:
            row = next(
                (
                    t
                    for t in self.transcript.tasks
                    if t["turn_id"] == key.turn_id and t["call_id"] == call_id
                ),
                None,
            )
            if row and row["status"] == "unknown":
                changed = (
                    self.transcript.task_finished(
                        key,
                        call_id,
                        status,
                        value.get("summary", "Adapter reported an outcome."),
                        value.get("display", ""),
                        mode=value.get("mode")
                        if value.get("mode") in {"mock", "reported", "Pi-verified"}
                        else "reported",
                        widget=value.get("widget"),
                        artifacts=value.get("artifacts"),
                        artifact_id=value.get("task_id"),
                        execution_target=value.get("execution_target"),
                        absolute_path=value.get("absolute_path"),
                        verification_level=value.get("verification_level"),
                    )
                    or changed
                )
        if changed:
            self.event("task_reconciled", turn_id=key.turn_id, tool_call_id=call_id)
        return changed

    def background_task_update(self, key, call_id, **update):
        """Project executor-owned progress without tying it to the speech turn."""
        if not self.transcript or key.session_id != self.session_id:
            return False
        status = update.get("status")
        if status in {"completed", "simulated", "reported", "unavailable", "failed", "cancelled"}:
            changed = self.transcript.task_finished(
                key,
                call_id,
                status,
                update.get("summary", ""),
                update.get("display", ""),
                artifacts=update.get("artifacts"),
                artifact_id=update.get("artifact_id"),
                execution_target=update.get("execution_target"),
                absolute_path=update.get("absolute_path"),
                verification_level=update.get("verification_level"),
                mode=update.get("mode"),
                widget=update.get("widget"),
                task_id=update.get("task_id"),
                phase=update.get("phase"),
                completed_steps=update.get("completed_steps"),
            )
        else:
            changed = self.transcript.task_progress(
                key,
                call_id,
                status=status,
                phase=update.get("phase"),
                completed_steps=update.get("completed_steps"),
                heartbeat=update.get("heartbeat", False),
                summary=update.get("summary"),
                task_id=update.get("task_id"),
            )
        if changed:
            fields = {
                "task_call_id": call_id,
                "task_status": status,
                "task_id": update.get("task_id"),
                "task_heartbeat": update.get("heartbeat", False),
            }
            phase = update.get("phase")
            if isinstance(phase, str):
                fields["task_phase"] = phase[:100]
            steps = update.get("completed_steps")
            if isinstance(steps, list):
                fields["task_completed_steps"] = len(steps)
            self.event("background_task_update", turn_id=key.turn_id, **fields)
        return changed

    def llm_budget_fields(self):
        transport = getattr(self.llm, "transport", None)
        spec = getattr(transport, "spec", None)
        used = getattr(transport, "requests", None)
        if spec is None or type(used) is not int:
            return {}
        return {"llm_requests_used": used, "llm_request_limit": spec.request_limit}

    def valid(self, key):
        return self.key == key and self.state != State.IDLE

    def activate(self, activation_event="wake_injected"):
        if self.state != State.IDLE:
            return
        self.state = State.ACTIVATING
        self.session_id = uuid.uuid4().hex
        self.turn = 0
        self.history.clear()
        if self.transcript:
            self.transcript.start_session(self.session_id)
        self.event(activation_event)
        self.last_activity = self.clock()
        self.awaiting_answer = False
        self.open_followup()
        self.state = State.LISTENING
        self.event(f"capture_ready_{self._sink_context()}")

    def acknowledge(self, key, text, segment_id=None):
        if self.valid(key):
            if self.transcript:
                self.transcript.confirm_segment(key, segment_id)
            self.history.append(
                {
                    "role": "assistant",
                    "content": text,
                    "turn_id": key.turn_id,
                    "delivery": getattr(self.playback.sink, "delivery", "mock_sink_confirmed"),
                }
            )
            self.event("segment_sink_confirmed")

    async def interrupt(self, *, for_new_input=False):
        interrupted_key = self.key
        if self.generation and not self.generation.done():
            self.event("interrupt_requested")
        if self.transcript and interrupted_key:
            self.transcript.interrupt(interrupted_key)
        # A UI command is queued on this turn before reasoning consumes it.
        # If speech/input cancels that turn first, discard the unsubmitted
        # command so it cannot leak into a later turn or execute stale work.
        if interrupted_key:
            self.ui_action_calls.pop(interrupted_key.turn_id, None)
            self.ui_action_turns.discard(interrupted_key.turn_id)
        if self.tasks and hasattr(self.tasks, "cancel_pending"):
            self.tasks.cancel_pending()
        self.key = None  # Fence before any await, including late provider returns.
        old, recognition = self.generation, self.asr_task
        if old is asyncio.current_task():
            old = None
        self.generation = self.asr_task = None
        self.input_open = False
        self.input_queue = None
        for task in (old, recognition):
            if task and not task.done():
                task.cancel()
        await self.playback.stop()
        self.event(
            "playback_stopped_browser"
            if getattr(self.playback.sink, "delivery", "") == "sink_confirmed"
            else "playback_stopped_mock",
            input_following=for_new_input,
        )
        for task in (old, recognition):
            if task:
                with suppress(asyncio.CancelledError, Exception):
                    await task
        if self.state != State.IDLE:
            self.state = State.LISTENING
        self.last_activity = self.clock()
        if not for_new_input and self.state != State.IDLE:
            self.open_followup()

    async def begin_input(self, *, started_at=None):
        if self.state == State.IDLE:
            raise ValueError("wake activation required")
        if (
            self.conversation_mode
            and self.engagement_deadline is not None
            and (self.clock() if started_at is None else started_at) >= self.engagement_deadline
        ):
            await self.exit()
            raise ValueError("engagement expired; wake activation required")
        self.candidate_started_at = None
        self.input_deadline = self.engagement_deadline
        self.engagement_deadline = None
        self.awaiting_answer = False
        await self.interrupt(for_new_input=True)
        self.turn += 1
        self.key = TurnKey(session_id=self.session_id, turn_id=self.turn)
        if self.transcript:
            self.transcript.begin_turn(self.key)
            self.transcript.set_state("listening")
        self.input_queue = asyncio.Queue(maxsize=32)
        self.input_bytes = 0
        self.input_open = True
        self.asr_task = asyncio.create_task(self.recognize(self.key, self.input_queue))
        return self.key

    async def push_input(self, pcm):
        if not self.input_open:
            raise ValueError("no open input turn")
        if self.input_bytes + len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("input turn exceeds 30 seconds")
        self.input_bytes += len(pcm)
        for start in range(0, len(pcm), FRAME_BYTES):
            if self.asr_task.done():
                self.asr_task.result()  # Surface errors rather than block a full queue.
                raise ValueError("ASR closed before input commit")
            put = asyncio.create_task(self.input_queue.put(pcm[start : start + FRAME_BYTES]))
            try:
                done, _ = await asyncio.wait(
                    (put, self.asr_task), return_when=asyncio.FIRST_COMPLETED
                )
                if self.asr_task in done:
                    self.asr_task.result()
                    raise ValueError("ASR closed before input commit")
                await put
            finally:
                if not put.done():
                    put.cancel()
                    await asyncio.gather(put, return_exceptions=True)
        self.last_activity = self.clock()

    async def commit_input(self):
        if not self.input_open:
            raise ValueError("no input turn to commit")
        self.input_open = False
        self.state = State.THINKING
        if self.transcript:
            self.transcript.set_state("thinking")
        # Count the PCM actually submitted to ASR, including any preserved pre-roll.
        self.event("endpoint_submitted", audio_samples=self.input_bytes // 2)
        # Queue may be full; race a failed recognizer instead of hanging forever.
        put = asyncio.create_task(self.input_queue.put(None))
        try:
            done, _ = await asyncio.wait((put, self.asr_task), return_when=asyncio.FIRST_COMPLETED)
            if self.asr_task in done:
                self.asr_task.result()
            await put
        finally:
            if not put.done():
                put.cancel()
                await asyncio.gather(put, return_exceptions=True)
        self.generation = asyncio.create_task(self.respond(self.key, self.asr_task))
        return self.generation

    async def submit(self, pcm: bytes):
        await self.begin_input()
        await self.push_input(pcm)
        return await self.commit_input()

    async def submit_ui_device_action(self, call):
        """Route a compact lamp control through the existing task and speech path."""
        from .devices import ACTION_VALUES, normalize_rgb_color, valid_action_value

        if self.state == State.IDLE or not self.tasks:
            raise ValueError("Start a voice conversation before using device controls")
        if (
            call.name != "run_device_action"
            or not re.fullmatch(r"ui-[a-f0-9]{32}", call.call_id)
            or set(call.arguments) not in ({"action"}, {"action", "value"})
        ):
            raise ValueError("invalid UI device action")
        action, value = call.arguments.get("action"), call.arguments.get("value")
        if action not in {"light_on", "light_off", "set_color", "set_brightness"}:
            raise ValueError("unsupported UI device action")
        if action == "set_color":
            value = normalize_rgb_color(value)
            if value is None:
                raise ValueError("unsupported UI device value")
        elif action in ACTION_VALUES and not valid_action_value(action, value):
            raise ValueError("unsupported UI device value")
        if action not in ACTION_VALUES and value is not None:
            raise ValueError("this UI device action takes no value")

        label = {
            "light_on": "Turn on the light.",
            "light_off": "Turn off the light.",
            "set_color": f"Set the light color to {value}.",
            "set_brightness": f"Set the light brightness to {value} percent.",
        }[action]
        if (
            self.conversation_mode
            and self.engagement_deadline is not None
            and self.clock() >= self.engagement_deadline
        ):
            await self.exit()
            raise ValueError("The conversation ended. Start a new conversation to control devices.")
        self.candidate_started_at = None
        self.input_deadline = self.engagement_deadline
        self.engagement_deadline = None
        self.awaiting_answer = False
        await self.interrupt(for_new_input=True)
        self.turn += 1
        key = TurnKey(session_id=self.session_id, turn_id=self.turn)
        self.key = key
        self.ui_action_calls[key.turn_id] = call
        self.ui_action_turns.add(key.turn_id)
        self.input_open = False
        self.input_queue = None
        self.state = State.THINKING
        if self.transcript:
            self.transcript.begin_turn(key)
            self.transcript.set_state("thinking")
        self.event("device_ui_action_submitted", action=action, tool_call_id=call.call_id)

        async def command_text():
            return label

        self.generation = asyncio.create_task(self.respond(key, command_text()))
        return key

    async def recognize(self, key, queue):
        try:
            return await self.recognize_stream(key, queue)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.valid(key):
                self.event(
                    "turn_error",
                    error=type(exc).__name__,
                    error_code=getattr(exc, "code", "speech_error"),
                    error_role="asr",
                    error_stage="asr",
                )
                if self.transcript:
                    self.transcript.interrupt(key, error=True)
                self.key = None
                self.input_open = False
                self.engagement_deadline = getattr(self, "input_deadline", self.engagement_deadline)
                self.state = State.LISTENING
                self.last_activity = self.clock()
            raise

    async def recognize_stream(self, key, queue):
        self.event("asr_started")

        async def chunks():
            while (chunk := await queue.get()) is not None:
                yield chunk

        if hasattr(self.speech, "transcribe_stream"):
            self.event("asr_stream_opened")
            stream = self.speech.transcribe_stream(key, chunks())
            final = None
            try:
                async for result in stream:
                    if not self.valid(key):
                        return ""
                    if rejects_english_asr(result.text):
                        if result.final:
                            raise EnglishTextRejected("asr")
                        # Do not publish a foreign-script interim hypothesis. A later
                        # final English result can still complete this turn normally.
                        continue
                    if self.transcript:
                        self.transcript.user_text(key, result.text, final=result.final)
                    self.event("asr_final" if result.final else "asr_partial")
                    if result.final:
                        if final is not None:
                            raise ValueError("duplicate ASR final")
                        final = result.text
                if final is None:
                    raise ValueError("ASR ended without final")
                return final
            finally:
                await stream.aclose()
        audio = bytearray()
        async for chunk in chunks():
            audio.extend(chunk)
        text = await self.speech.transcribe(key, bytes(audio))
        if self.valid(key):
            if rejects_english_asr(text):
                raise EnglishTextRejected("asr")
            self.event("asr_final")
        return text

    async def reasoning(self, key):
        latest_user = next(
            (row["content"] for row in reversed(self.history) if row["role"] == "user"),
            "",
        )
        spelling_ack = direct_spelling_correction(latest_user)
        if spelling_ack:
            yield spelling_ack
            return
        contextual_music = direct_music_control(latest_user, key.turn_id, self.history)
        clarification = (
            None
            if contextual_music
            else direct_clarification(
                latest_user,
                communication_enabled=getattr(self.tasks, "communication_enabled", False),
            )
        )
        if clarification:
            self.awaiting_answer = True
            self.event("clarification_requested")
            yield clarification
            return
        ui_action_call = self.ui_action_calls.pop(key.turn_id, None)
        local_music_call = direct_music_control(
            latest_user, key.turn_id, self.history
        ) or direct_music_request(latest_user, key.turn_id, self.history)
        local_clock_call = (
            direct_clock_query(latest_user, key.turn_id) if self.direct_clock_enabled else None
        )
        local_light_command = direct_light_command(latest_user, key.turn_id)
        local_light_followup = direct_light_followup(latest_user, key.turn_id, self.history)
        local_light_call = local_light_command or local_light_followup
        local_me2_call = direct_me2_command(latest_user, key.turn_id)
        local_tool_call = (
            ui_action_call
            or local_music_call
            or local_clock_call
            or local_light_call
            or local_me2_call
        )
        for request_round in range(8):
            if request_round:
                self.event("post_tool_reasoning_started")
            called = False
            emitted = False
            call_count = 0
            confirmations = []
            local_call = request_round == 0 and local_tool_call is not None
            request_finished = local_call
            if local_call:
                if ui_action_call:
                    self.event(
                        "device_ui_action_routed",
                        tool_call_id=ui_action_call.call_id,
                        action=ui_action_call.arguments["action"],
                    )
                elif local_music_call:
                    self.event("music_command_routed", tool_call_id=local_music_call.call_id)
                elif local_light_command:
                    self.event(
                        "local_light_command_routed",
                        tool_call_id=local_light_command.call_id,
                        action=local_light_command.arguments["action"],
                    )
                elif local_light_followup:
                    self.event("light_followup_routed", tool_call_id=local_light_followup.call_id)
                elif local_me2_call:
                    self.event("device_command_routed", tool_call_id=local_me2_call.call_id)
                else:
                    self.event("local_clock_routed", tool_call_id=local_tool_call.call_id)

                async def local_stream():
                    yield local_tool_call

                stream = local_stream()
            else:
                if getattr(self.llm, "task_router", False):
                    self.event("agent_route_dispatched")
                else:
                    self.event("llm_request_started")
                stream = self.llm.respond(tuple(dict(message) for message in self.history))
            try:
                finished = object()
                while True:
                    async with asyncio.timeout(self.llm_timeout):
                        result = await anext(stream, finished)
                    if result is finished:
                        break
                    if not self.valid(key):
                        return
                    if isinstance(result, ToolCall):
                        # The model's tool-call event is already available;
                        # subsequent task execution belongs to the tool span.
                        if not request_finished:
                            if not local_call and not getattr(self.llm, "task_router", False):
                                self.event("llm_request_finished", **self.llm_budget_fields())
                            request_finished = True
                        if not isinstance(result.call_id, str) or not re.fullmatch(
                            r"[a-zA-Z0-9_-]{1,80}", result.call_id
                        ):
                            raise ValueError("invalid tool call identity")
                        if (
                            result.name == "control_music"
                            and result.arguments.get("action") in {"play", "play_playlist"}
                            and not music_selection_requested(
                                latest_user, key.turn_id, self.history
                            )
                        ):
                            self.awaiting_answer = True
                            self.event(
                                "tool_intent_rejected",
                                tool_call_id=result.call_id,
                                reason="music_selection_not_requested",
                            )
                            yield "Would you like to play, pause, or stop the music?"
                            return
                        if not self.tasks:
                            raise ValueError("task executor unavailable")
                        called = True
                        call_count += 1
                        if self.transcript:
                            kind = (
                                "music_playlist"
                                if result.name == "control_music"
                                and result.arguments.get("action") == "play_playlist"
                                else "music_" + str(result.arguments.get("action"))
                                if result.name == "control_music"
                                else result.arguments.get("action")
                                if result.name in {"run_device_action", "run_mock_action"}
                                else "light_on"
                                if result.name == "set_mock_light"
                                and result.arguments.get("enabled")
                                else "light_off"
                                if result.name == "set_mock_light"
                                else "agent"
                            )
                            if kind not in MOCK_ACTION_TITLES and kind not in {
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
                            }:
                                kind = "agent"
                            self.transcript.task_started(
                                key,
                                result.call_id,
                                task_title(result),
                                kind,
                                mode=(
                                    getattr(self.tasks, "background_task_mode", "reported")
                                    if result.name == "run_agent_task"
                                    else "reported"
                                    if result.name == "control_music"
                                    else getattr(self.tasks, "task_mode", "mock")
                                ),
                            )
                        self.event("tool_started", tool_call_id=result.call_id)
                        try:
                            background = getattr(self.tasks, "execute_background", None)
                            if result.name == "run_agent_task" and callable(background):
                                value = await background(
                                    key,
                                    result,
                                    on_update=lambda call_id=result.call_id, **update: (
                                        self.background_task_update(key, call_id, **update)
                                    ),
                                )
                            else:
                                value = await self.tasks.execute(key, result)
                        except asyncio.CancelledError:
                            # Cancelling narration does not establish cancellation of
                            # an adapter's submitted side effect. Keep its identity.
                            if self.transcript:
                                self.transcript.task_finished(
                                    key,
                                    result.call_id,
                                    "unknown",
                                    "The action was pending when this turn ended. Its outcome is not confirmed; check before retrying.",
                                )
                            self.record_tool_outcome(
                                key,
                                result,
                                {
                                    "result_status": "unknown",
                                    "summary": "The submitted action was unconfirmed when the turn ended.",
                                },
                            )
                            watcher = getattr(self.tasks, "watch_result", None)
                            if callable(watcher):
                                watcher(
                                    key,
                                    result,
                                    lambda value, k=key, c=result.call_id: (
                                        self.reconcile_task_result(k, c, value)
                                    ),
                                )
                            raise
                        except Exception as exc:
                            with suppress(Exception):
                                exc.error_stage = "tool"
                            if self.transcript:
                                uncertain = (
                                    result.name == "control_music"
                                    and isinstance(exc, MusicUnavailable)
                                    and exc.outcome_uncertain
                                )
                                if result.name == "control_music":
                                    summary = (
                                        str(exc)[:200]
                                        if isinstance(exc, MusicUnavailable)
                                        else "Spotify rejected the volume request. Use a percentage from 0 to 100."
                                        if isinstance(exc, ValueError)
                                        and result.arguments.get("action")
                                        in {"volume_up", "volume_down", "volume_set"}
                                        else "Spotify could not confirm this command. Check the player before trying again."
                                    )
                                else:
                                    summary = (
                                        "The agent task could not be confirmed."
                                        if result.name == "run_agent_task"
                                        else "The device action could not be confirmed."
                                        if result.name in {"run_device_action", "run_mock_action"}
                                        else "The simulated action could not complete."
                                    )
                                task_summary = getattr(exc, "user_summary", None)
                                task_status = getattr(exc, "task_status", None)
                                if (
                                    not isinstance(task_summary, str)
                                    or not 0 < len(task_summary) <= 200
                                ):
                                    # Keep the more specific provider/task error above.
                                    # Replacing it with the generic simulation fallback
                                    # obscured Spotify uncertainty and recovery guidance.
                                    task_summary = summary
                                if task_status not in {"unavailable", "failed", "unknown"}:
                                    task_status = "unknown" if uncertain else "failed"
                                self.transcript.task_finished(
                                    key,
                                    result.call_id,
                                    task_status,
                                    task_summary,
                                )
                            self.record_tool_outcome(
                                key,
                                result,
                                {
                                    "result_status": task_status
                                    if self.transcript
                                    else getattr(exc, "task_status", "unknown"),
                                    "summary": task_summary
                                    if self.transcript
                                    else "The action outcome was not confirmed.",
                                },
                            )
                            raise
                        if self.transcript:
                            summary = value.get("summary")
                            if not isinstance(summary, str) or len(summary) > 200:
                                summary = "Task service reported an outcome; effect not independently verified."
                            display = value.get("display")
                            if not isinstance(display, str):
                                display = ""
                            status = value.get("result_status")
                            if status not in {
                                "simulated",
                                "accepted",
                                "awaiting_input",
                                "completed",
                                "reported",
                                "unavailable",
                                "failed",
                                "unknown",
                            }:
                                status = (
                                    "simulated"
                                    if value.get("mode") == "mock"
                                    and value.get("verified_in_memory") is True
                                    else "reported"
                                )
                            mode = value.get("mode")
                            if mode not in {"mock", "reported", "Pi-verified"}:
                                mode = "reported"
                            task_fields = {
                                "summary": summary,
                                "display": value["artifacts"][0]
                                if result.name == "run_agent_task"
                                and isinstance(value.get("artifacts"), list)
                                and value["artifacts"]
                                and isinstance(value["artifacts"][0], str)
                                and len(value["artifacts"][0]) <= 32
                                else display[:32],
                                "execution_target": value.get("execution_target"),
                                "absolute_path": value.get("absolute_path"),
                                "verification_level": value.get("verification_level"),
                                "mode": mode,
                                "widget": value.get("widget"),
                            }
                            if status in {"accepted", "awaiting_input", "running"}:
                                self.transcript.task_progress(
                                    key,
                                    result.call_id,
                                    status=status,
                                    phase=value.get("phase"),
                                    completed_steps=value.get("completed_steps"),
                                    summary=summary,
                                    task_id=value.get("task_id"),
                                )
                            else:
                                self.transcript.task_finished(
                                    key,
                                    result.call_id,
                                    status,
                                    task_fields.pop("summary"),
                                    task_fields.pop("display"),
                                    artifact_id=value.get("task_id")
                                    if value.get("mode") in {"local-artifact", "local-project-site"}
                                    else None,
                                    artifacts=value.get("artifacts")
                                    if value.get("mode") in {"local-artifact", "local-project-site"}
                                    else None,
                                    answer_only=value.get("mode") == "local-answer"
                                    and not value.get("artifacts")
                                    and not value.get("tavily_credits_used")
                                    and not value.get("github_reads_used"),
                                    **task_fields,
                                )
                            replaced = value.get("replaced_task")
                            if isinstance(replaced, dict):
                                self.transcript.task_transition(
                                    key,
                                    replaced.get("turn_id"),
                                    replaced.get("call_id"),
                                    "cancelled"
                                    if result.arguments.get("action") == "cancel_schedule"
                                    else "replaced",
                                    "Cancelled by request."
                                    if result.arguments.get("action") == "cancel_schedule"
                                    else "Replaced by a newer timer/alarm.",
                                )
                        if not self.valid(key):
                            return
                        if value.get("query_normalization") == "maroon_file_to_maroon_5":
                            self.event("music_query_normalized")
                        self.event("tool_completed", tool_call_id=result.call_id)
                        if (
                            value.get("mode") == "mock"
                            and value.get("verified_in_memory") is True
                            and value.get("result_status") != "unavailable"
                        ):
                            self.event("task_verified_mock", tool_call_id=result.call_id)
                        self.history.append(
                            {
                                "role": "tool",
                                "content": json.dumps(value, sort_keys=True),
                                "tool_call_id": result.call_id,
                                "tool_request": {
                                    "name": result.name,
                                    "arguments": json.dumps(result.arguments, sort_keys=True),
                                },
                                "turn_id": key.turn_id,
                            }
                        )
                        confirmation = self.task_confirmation(result, value)
                        if confirmation is not None:
                            confirmations.append(confirmation)
                        else:
                            confirmations.append(None)
                    elif isinstance(result, str) and 0 < len(result) <= 500:
                        emitted = True
                        yield result
                    else:
                        raise ValueError("invalid reasoning event")
            finally:
                await stream.aclose()
            if (
                self.valid(key)
                and not request_finished
                and not local_call
                and not getattr(self.llm, "task_router", False)
            ):
                self.event("llm_request_finished", **self.llm_budget_fields())
            if not called:
                if not emitted and self.valid(key):
                    yield (
                        "I couldn't produce a reply. Check the task status before trying again."
                        if request_round
                        else "I couldn't produce a reply. Please try again."
                    )
                if request_round:
                    self.event("post_tool_reasoning_finished")
                return
            if (
                call_count
                and len(confirmations) == call_count
                and all(c is not None for c in confirmations)
            ):
                # Structured task receipts are the authority for a concise
                # acknowledgment. A second model request here adds latency and
                # can consume the allowance needed for the user's next turn.
                for confirmation in confirmations:
                    if confirmation:
                        yield confirmation
                return
        raise ValueError("reasoning tool-round limit exceeded")

    @staticmethod
    def task_confirmation(call, value):
        """Return a bounded spoken confirmation only for trusted task receipts."""
        if not isinstance(value, dict):
            return None
        if value.get("recovered_receipt") is True:
            return ""  # Receipt recovery is visual-only; never replay narration.
        if call.name == "run_agent_task" and value.get("mode") == "local-answer":
            answer = value.get("answer")
            return (
                answer.strip()
                if isinstance(answer, str) and 0 < len(answer.strip()) <= 500
                else None
            )
        if (
            call.name == "run_agent_task"
            and value.get("background") is True
            and value.get("result_status") == "accepted"
        ):
            return "I’ve started that task. It may take a few minutes; you can keep talking while it runs."
        if value.get("mode") == "mock" and value.get("verified_in_memory") is True:
            if call.name in {"set_mock_light", "read_mock_light"}:
                return f"The simulated light is {'on' if value.get('enabled') is True else 'off'}."
            summary = value.get("summary")
            return summary.strip() if isinstance(summary, str) and 0 < len(summary) <= 200 else None
        if call.name in {"run_device_action", "run_mock_action"}:
            if call.arguments.get("action") == "get_time":
                widget = value.get("widget")
                if isinstance(widget, dict) and widget.get("type") == "time":
                    stamp, zone = widget.get("local_time"), widget.get("timezone")
                    if isinstance(stamp, str) and isinstance(zone, str):
                        place = {"Asia/Manila": "in the Philippines", "Asia/Tokyo": "in Japan"}.get(
                            zone, "local time"
                        )
                        return f"It's {stamp} {place}."
            if call.arguments.get("action") == "get_weather":
                widget = value.get("widget")
                if isinstance(widget, dict) and widget.get("type") == "weather":
                    return f"In {widget['location']}, it's {widget['condition']}, {widget['temperature_c']} degrees Celsius."
            summary = value.get("summary")
            return summary.strip() if isinstance(summary, str) and 0 < len(summary) <= 200 else None
        if call.name == "control_music" and value.get("mode") == "spotify":
            return value.get("summary")
        if value.get("mode") == "local-artifact":
            artifacts = value.get("artifacts")
            if (
                call.name == "run_agent_task"
                and isinstance(value.get("task_id"), str)
                and isinstance(artifacts, list)
                and 0 < len(artifacts) <= 8
                and all(
                    isinstance(name, str)
                    and re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)",
                        name,
                    )
                    for name in artifacts
                )
            ):
                count = len(artifacts)
                noun = "file" if count == 1 else "files"
                return f"The draft task is complete. {count} {noun} are ready to review in Activity and results."
        if value.get("mode") == "local-project-site" and call.name == "run_agent_task":
            projects = value.get("projects")
            artifacts = value.get("artifacts")
            if (
                isinstance(projects, list)
                and len(projects) == 1
                and isinstance(projects[0], str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", projects[0])
                and isinstance(artifacts, list)
                and 0 < len(artifacts) <= 8
                and all(
                    isinstance(name, str)
                    and re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)",
                        name,
                    )
                    for name in artifacts
                )
            ):
                return (
                    f"The static website {projects[0]} is ready with {len(artifacts)} files. "
                    "It has not been run or deployed."
                )
        if value.get("mode") == "local-project" and call.name == "run_agent_task":
            projects = value.get("projects")
            if (
                isinstance(projects, list)
                and len(projects) == 1
                and isinstance(projects[0], str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", projects[0])
            ):
                return f"I created the empty project directory {projects[0]}."
        return None

    async def respond(self, key, recognition):
        try:
            async with asyncio.timeout(self.turn_timeout):
                text = await recognition
                if not self.valid(key):
                    return
                if rejects_english_asr(text):
                    raise EnglishTextRejected("asr")
                if not text.strip():
                    self.engagement_deadline = getattr(self, "input_deadline", None)
                    if self.transcript:
                        self.transcript.complete(key)
                    self.event("turn_completed")
                    return
                if self.transcript and key.turn_id not in self.ui_action_turns:
                    self.transcript.user_text(key, text, final=True)
                self.ui_action_turns.discard(key.turn_id)
                if text.strip().lower().rstrip(".!?") in {
                    "end session",
                    "goodbye",
                    "end conversation",
                    "that's all",
                    "that’s all",
                }:
                    if self.transcript:
                        self.transcript.end_session()
                    self.key = None
                    self.state = State.IDLE
                    self.event("explicit_exit")
                    return
                self.history.append({"role": "user", "content": text, "turn_id": key.turn_id})
                if text.strip().casefold().rstrip(".!?") in {"stop speaking", "stop talking"}:
                    # begin_input already fenced the old turn and stopped playback.
                    # Do not ask a model to interpret a speech-only control.
                    if self.transcript:
                        self.transcript.complete(key)
                    self.event("speech_stop_requested")
                    self.event("turn_completed")
                    self.open_followup()
                    return
                local_route = (
                    self.ui_action_calls.get(key.turn_id)
                    or direct_spelling_correction(text)
                    or direct_music_control(text, key.turn_id, self.history)
                    or direct_music_request(text, key.turn_id, self.history)
                    or (
                        direct_clock_query(text, key.turn_id) if self.direct_clock_enabled else None
                    )
                    or direct_light_command(text, key.turn_id)
                    or direct_light_followup(text, key.turn_id, self.history)
                    or direct_me2_command(text, key.turn_id)
                    or direct_clarification(
                        text,
                        communication_enabled=getattr(self.tasks, "communication_enabled", False),
                    )
                )
                if local_route is None and not getattr(self.llm, "task_router", False):
                    self.event("llm_started")
                segment_id = 0
                # Reasoning may produce the next sentence while the current
                # sentence is synthesized and played. Bound pending text so a
                # slow TTS service cannot build an unbounded response backlog.
                segments = asyncio.Queue(maxsize=2)

                async def synthesize_segments():
                    while True:
                        item = await segments.get()
                        try:
                            if item is None:
                                return
                            current_id, current_text = item
                            if not self.valid(key):
                                return
                            if has_non_latin_letters(current_text):
                                raise EnglishTextRejected("tts")
                            chunks = 0
                            self.event("tts_started", segment_id=current_id)
                            stream = self.speech.synthesize(key, current_text)
                            try:
                                async for chunk in stream:
                                    if not self.valid(key):
                                        return
                                    if chunks == 0:
                                        self.event("tts_first_chunk", segment_id=current_id)
                                    chunks += 1
                                    await self.playback.enqueue(key, chunk)
                            except Exception as exc:
                                with suppress(Exception):
                                    exc.error_stage = "tts"
                                raise
                            finally:
                                await stream.aclose()
                            if self.valid(key):
                                self.event("tts_finished", segment_id=current_id)
                                if chunks:
                                    await self.playback.enqueue(key, None, current_text, current_id)
                        finally:
                            segments.task_done()

                async with asyncio.TaskGroup() as group:
                    group.create_task(synthesize_segments())
                    async for segment in self.reasoning(key):
                        if not self.valid(key):
                            break
                        if has_non_latin_letters(segment):
                            raise EnglishTextRejected("llm")
                        segment_id += 1
                        if self.transcript:
                            self.transcript.assistant_segment(key, segment_id, segment)
                        if (
                            segment_id == 1
                            and local_route is None
                            and not getattr(self.llm, "task_router", False)
                        ):
                            self.event("llm_first_segment")
                        self.event("llm_segment")
                        await segments.put((segment_id, segment))
                    if self.valid(key) and self.transcript:
                        if self.transcript.finalize_text(key):
                            self.event("llm_text_finalized")
                    await segments.put(None)
                if not self.valid(key):
                    return
                await self.playback.queue.join()
                if self.playback.error:
                    failure = RuntimeError("playback sink failed")
                    failure.error_stage = "playback"
                    raise failure from self.playback.error
                if self.playback.worker.done():
                    self.playback.worker.result()
                if self.valid(key):
                    if self.transcript:
                        self.transcript.complete(key)
                    self.event("turn_completed")
                    if self.conversation_mode == "single" and not self.awaiting_answer:
                        await self.exit()
                    else:
                        self.open_followup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.valid(key):
                cause = (
                    exc.exceptions[0]
                    if isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1
                    else exc
                )
                fields = {"error": type(cause).__name__}
                code = getattr(cause, "code", None)
                if isinstance(cause, TimeoutError):
                    code = "turn_deadline_exceeded"
                stage = getattr(cause, "error_stage", None)
                if not stage:
                    stage = {
                        "asr": "asr",
                        "tts": "tts",
                        "llm": "reasoning",
                    }.get(
                        getattr(cause, "role", None),
                        "deadline" if isinstance(cause, TimeoutError) else "reasoning",
                    )
                if stage in {
                    "asr",
                    "reasoning",
                    "response_parse",
                    "tool",
                    "tts",
                    "playback",
                    "deadline",
                }:
                    fields["error_stage"] = stage
                boundary = getattr(cause, "error_boundary", None)
                if boundary in {
                    "response_headers",
                    "first_sse_event",
                    "first_text_delta",
                    "sse_stream",
                }:
                    fields["error_boundary"] = boundary
                detail = getattr(cause, "error_detail", None)
                if detail in {
                    "invalid_sse_json",
                    "incomplete_sse",
                    "invalid_event_shape",
                    "event_not_object",
                    "provider_error_event",
                    "missing_choices",
                    "choices_not_array",
                    "unexpected_choice_count",
                    "choice_not_object",
                    "invalid_choice_index",
                    "invalid_delta_shape",
                    "invalid_tool_call_shape",
                    "invalid_tool_arguments",
                    "invalid_tool_identity",
                    "unadvertised_tool_name",
                    "duplicate_tool_identity",
                    "invalid_finish_reason",
                }:
                    fields["error_detail"] = detail
                if isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,48}", code):
                    fields["error_code"] = code
                role = getattr(cause, "role", "llm")
                fields["error_role"] = role if role in {"asr", "tts", "llm"} else "llm"
                fields.update(self.llm_budget_fields())
                self.event("turn_error", **fields)
                if self.transcript:
                    self.transcript.interrupt(key, error=True)
                self.key = None
                await self.playback.stop()
        finally:
            if self.valid(key):
                self.state = State.LISTENING
                self.last_activity = self.clock()
            elif self.state != State.IDLE and self.key is None:
                self.state = State.LISTENING
                # A failed request ends its turn, not the wake-word session.
                # Start the inactivity window when recovery returns to listening.
                self.last_activity = self.clock()
                if self.engagement_deadline is None:
                    self.open_followup()

    async def tick(self):
        if self.conversation_mode:
            if (
                self.candidate_started_at is not None
                and self.clock() - self.candidate_started_at < 30
            ):
                return
            if (
                self.state == State.LISTENING
                and not self.input_open
                and self.engagement_deadline is not None
                and self.clock() >= self.engagement_deadline
            ):
                await self.exit()
            return
        if (
            not self.input_open
            and self.state == State.LISTENING
            and self.clock() - self.last_activity >= self.idle_seconds
        ):
            await self.exit()

    async def exit(self):
        self.candidate_started_at = None
        self.engagement_deadline = None
        self.awaiting_answer = False
        await self.interrupt()
        self.engagement_deadline = None
        self.state = State.IDLE
        if self.transcript:
            self.transcript.end_session()
        self.event("session_exit")
