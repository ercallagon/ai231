"""Repeatable mock integration cases using the existing Session and real stdio MCP.

Controlled delays are scenario inputs, never acoustic/model benchmarks.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path

from .audio import ConversationInput, ReplayInput
from .harness import PipecatAudioInput
from .runtime import AudioApplication
from .wake import WakeDetection, WakeGate
from .contracts import FRAME_BYTES
from .playback import MemorySink
from .providers import MockLLM
from .session import Session, State
from .speech import SpeechClient
from .tools import MockTaskExecutor
from .providers import ToolCall
from .telemetry import summarize, write_trace

PCM = b"\1\0" * (FRAME_BYTES // 2)


async def until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


class BarrierLLM:
    def __init__(self):
        self.entered = asyncio.Event()

    async def respond(self, history):
        self.entered.set()
        await asyncio.Event().wait()
        yield "unreachable"


class BrokenLLM:
    async def respond(self, history):
        raise RuntimeError("controlled mock reasoning failure")
        yield "unreachable"


class MockBackgroundTaskExecutor(MockTaskExecutor):
    """Scenario-only asynchronous project worker using the real Session contract."""

    background_task_mode = "mock"

    def __init__(self, output_dir, *, duration_seconds=120, outcome="success"):
        if type(duration_seconds) is not int or not 1 <= duration_seconds <= 180:
            raise ValueError("background task duration must be 1..180 seconds")
        if outcome not in {"success", "failure"}:
            raise ValueError("mock background outcome must be success or failure")
        super().__init__()
        self.output_dir = Path(output_dir)
        self.duration_seconds = duration_seconds
        self.outcome = outcome
        self.operations = {}
        self.by_request = {}
        self.started_count = 0
        self.current_phase = "Accepted; worker starting"
        self.completed_steps = []
        self.task_id = None
        self.task_started_at = None

    async def execute_background(self, key, call, *, on_update):
        if call.name != "run_agent_task" or set(call.arguments) != {"task"}:
            raise ValueError("mock background worker accepts only run_agent_task")
        task = call.arguments["task"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError("mock project request must be non-empty")
        operation = f"{key.session_id}:{key.turn_id}:{call.call_id}"
        task_id = hashlib.sha256(operation.encode()).hexdigest()[:24]
        existing = self.operations.get(operation)
        if existing:
            return self._accepted(existing[0])
        normalized = " ".join(task.casefold().split())
        active = self.by_request.get(normalized)
        if active and not active[0].done():
            active[1].append((key, call.call_id, on_update))
            return self._accepted(active[2])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        listeners = [(key, call.call_id, on_update)]
        job = asyncio.create_task(self._run(listeners, task_id))
        self.operations[operation] = (task_id, job)
        self.by_request[normalized] = (job, listeners, task_id)
        self.started_count += 1
        self.task_id = task_id
        self.task_started_at = time.monotonic()
        return self._accepted(task_id)

    @staticmethod
    def _accepted(task_id):
        return {
            "mode": "mock",
            "result_status": "accepted",
            "background": True,
            "task_id": task_id,
            "summary": "Mock project task accepted.",
            "display": "project-plan.md",
            "execution_target": "ME3 mock workspace",
            "absolute_path": "var/results/016-background-task-ui/task-output/project-plan.md",
            "verification_level": "Mock worker only; output is confined to this experiment under var/.",
        }

    async def _run(self, listeners, task_id):
        start = time.monotonic()
        phases = (
            (0, "Planning the requested project files", []),
            (self.duration_seconds / 4, "Drafting the project structure", ["Scope captured"]),
            (
                self.duration_seconds / 2,
                "Writing setup instructions",
                ["Scope captured", "Structure outlined"],
            ),
            (
                self.duration_seconds * 3 / 4,
                "Checking the generated project plan",
                ["Scope captured", "Structure outlined", "Setup notes written"],
            ),
        )
        heartbeat_interval = min(25.0, max(0.25, self.duration_seconds / 5))
        next_heartbeat = heartbeat_interval
        phase_index = 0
        try:
            while True:
                elapsed = time.monotonic() - start
                while phase_index < len(phases) and elapsed >= phases[phase_index][0]:
                    _, self.current_phase, self.completed_steps = phases[phase_index]
                    for key, call_id, update in tuple(listeners):
                        update(
                            status="running",
                            phase=self.current_phase,
                            completed_steps=self.completed_steps,
                            task_id=task_id,
                        )
                    phase_index += 1
                if elapsed >= self.duration_seconds:
                    break
                if elapsed >= next_heartbeat:
                    for key, call_id, update in tuple(listeners):
                        update(status="running", heartbeat=True, task_id=task_id)
                    next_heartbeat += heartbeat_interval
                    continue
                next_phase = (
                    phases[phase_index][0] if phase_index < len(phases) else self.duration_seconds
                )
                await asyncio.sleep(
                    max(0.01, min(next_phase, next_heartbeat, self.duration_seconds) - elapsed)
                )
            if self.outcome == "failure":
                for key, call_id, update in tuple(listeners):
                    update(
                        status="failed",
                        summary="Controlled mock failure before output creation.",
                        mode="mock",
                        task_id=task_id,
                    )
                return
            self.completed_steps = [
                "Scope captured",
                "Structure outlined",
                "Setup notes written",
                "Draft checked",
            ]
            output = self.output_dir / "project-plan.md"
            output.write_text(
                "# Mock Swift project plan\n\n"
                "This bounded project outline was generated by the ME3 deterministic mock worker.\n",
                encoding="utf-8",
            )
            self.current_phase = "Project plan ready"
            for key, call_id, update in tuple(listeners):
                update(
                    status="completed",
                    phase=self.current_phase,
                    completed_steps=self.completed_steps,
                    summary="Mock project plan created and checked in the designated experiment output.",
                    display=output.name,
                    execution_target="ME3 mock workspace",
                    absolute_path=str(output.resolve()),
                    verification_level="Mock file exists; no general-purpose agent or external project was used.",
                    mode="mock",
                    task_id=task_id,
                )
        except asyncio.CancelledError:
            for key, call_id, update in tuple(listeners):
                update(
                    status="cancelled",
                    summary="Mock background task was cancelled during worker shutdown.",
                    mode="mock",
                    task_id=task_id,
                )
            raise
        except Exception:
            for key, call_id, update in tuple(listeners):
                update(
                    status="failed",
                    summary="Mock project task failed before creating its output.",
                    mode="mock",
                    task_id=task_id,
                )
            raise

    async def cancel_task(self, task_id):
        for existing_id, job in self.operations.values():
            if existing_id == task_id and not job.done():
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
                return True
        return False

    async def close(self):
        jobs = [job for _, job in self.operations.values() if not job.done()]
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        await super().close()


class MarkedVAD:
    def voiced(self, pcm):
        return any(pcm)


class ScenarioWake:
    frame_samples = 320
    backend = "mock-timed-wake-event"
    provenance = {}

    def __init__(self):
        self.samples = 0

    def detect(self, pcm):
        self.samples += 320
        return WakeDetection("HEY_SWIFT") if self.samples == 640 else None


class SimulatedDisconnect:
    """Fault after a real MCP partial, not a claim of a real hosted outage."""

    def __init__(self, speech):
        self.speech, self.fail = speech, True

    async def transcribe_stream(self, key, chunks):
        stream = self.speech.transcribe_stream(key, chunks)
        try:
            async for result in stream:
                yield result
                if self.fail:
                    self.fail = False
                    raise ConnectionError("controlled mock stream disconnect")
        finally:
            await stream.aclose()

    def synthesize(self, key, text):
        return self.speech.synthesize(key, text)


async def run_text_injected_stress(turns=25, concurrency=4):
    """Exercise post-ASR turns concurrently with bounded deterministic dependencies."""
    if type(turns) is not int or not 1 <= turns <= 100:
        raise ValueError("stress turns must be from 1 to 100")
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise ValueError("stress concurrency must be from 1 to 8")

    class TextInjectedSpeech:
        def __init__(self, client_index):
            self.client_index = client_index

        async def transcribe(self, key, pcm):
            await asyncio.sleep(0.0005)
            number = key.turn_id - 1
            return (
                "simulate light on"
                if number % 10 == 0
                else "simulate light off"
                if number % 10 == 1
                else f"reliability check client {self.client_index} follow-up {number}"
            )

        async def synthesize(self, key, text):
            for _ in range(4):
                await asyncio.sleep(0.0005)
                yield PCM

    events, completed, effects, queue_bounds = [], 0, 0, []

    async def sink(_pcm):
        await asyncio.sleep(0.0005)

    async def run_client(index):
        nonlocal completed, effects
        async with Session(
            TextInjectedSpeech(index), MockLLM(), sink, turn_timeout=5, llm_timeout=1
        ) as session:
            session.activate(activation_event="text_injected_activation")
            for number in range(turns):
                generation = await session.submit(PCM)
                await asyncio.wait_for(generation, timeout=3)
                if session.state != State.LISTENING or not session.playback.queue.empty():
                    raise RuntimeError("turn did not return to a drained listening state")
                if session.playback.queue.maxsize != 8:
                    raise RuntimeError("playback queue capacity changed unexpectedly")
                completed += 1
                queue_bounds.append(session.playback.queue.maxsize)
            expected_effects = (turns + 9) // 10 + (turns + 8) // 10
            if session.tasks.actions.effects != expected_effects:
                raise RuntimeError("mock task effects were duplicated or omitted")
            effects += session.tasks.actions.effects
            if len([row for row in session.history if row["role"] == "user"]) != turns:
                raise RuntimeError("conversation history lost a text-injected turn")
            if len([row for row in session.history if row["role"] == "assistant"]) != turns:
                raise RuntimeError("assistant reply was not acknowledged for every turn")
            events.extend(session.events)

    preexisting = set(asyncio.all_tasks())
    await asyncio.gather(*(run_client(index) for index in range(concurrency)))
    await asyncio.sleep(0)
    leaked = [task for task in asyncio.all_tasks() if task not in preexisting and not task.done()]
    if leaked:
        raise RuntimeError("ME3 session tasks remained after concurrent clients closed")

    diagnostics = summarize(events)
    metric_samples = {}
    for row in diagnostics["turns"]:
        for metric, values in row["durations_ms"].items():
            metric_samples.setdefault(metric, []).extend(values)
    import numpy as np

    latency_aggregates = [
        {
            "mode": "mock",
            "clock_domain": "same host monotonic process; run/session IDs remain in the trace",
            "metric": metric,
            "count": len(values),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": max(values),
        }
        for metric, values in sorted(metric_samples.items())
    ]
    return {
        "passed": completed == turns * concurrency and not leaked,
        "turns_per_session": turns,
        "concurrent_sessions": concurrency,
        "completed_turns": completed,
        "mock_task_effects": effects,
        "playback_queue_capacity": max(queue_bounds, default=0),
        "playback_queue_drained_each_turn": True,
        "leaked_me3_tasks": len(leaked),
        "latency_aggregates": latency_aggregates,
        "limitations": (
            "A deterministic scripted ASR maps a placeholder PCM frame to each test transcript; no recorded "
            "speech or microphone is used. ASR partial accuracy, model/service timing, browser playback and "
            "acoustic latency are not measured; replies use deterministic mock reasoning, TTS PCM and a memory sink."
        ),
        "_events": events,
    }


async def run_scenarios(trace=None, *, stress_turns=25, stress_concurrency=4):
    if trace and trace.exists():
        raise FileExistsError("preserve existing trace; use a fresh path")
    cases, events = [], []
    for name in (
        "immediate_wake_command",
        "natural_pause",
        "slow_speech_service",
        "simulated_service_disconnect",
        "normal_followup_exit",
        "interrupt_reasoning",
        "interrupt_synthesis",
        "task_playback_correction",
        "provider_failure",
    ):
        started = time.monotonic()
        controlled = {
            "asr_delay_seconds": 0.6 if name == "slow_speech_service" else 0,
            "rpc_timeout_seconds": 0.25 if name == "slow_speech_service" else 5,
            "disconnect_after_partial": name == "simulated_service_disconnect",
            "reasoning_barrier": name == "interrupt_reasoning",
            "sink_delay_seconds": 60 if name == "task_playback_correction" else 0,
            "tts_chunk_delay_seconds": 0.03 if name == "interrupt_synthesis" else 0,
        }
        checks = []
        observations = {}
        failure = None
        session = None
        try:
            script = ["hello", "follow-up", "goodbye"]
            if name == "task_playback_correction":
                script = ["set mock light on", "mock light status", "set mock light off"]
            sink = MemorySink(delay=controlled["sink_delay_seconds"])
            model = (
                BarrierLLM()
                if name == "interrupt_reasoning"
                else BrokenLLM()
                if name == "provider_failure"
                else MockLLM()
            )
            async with SpeechClient(
                transcript=script,
                tts_delay=controlled["tts_chunk_delay_seconds"],
                delay=controlled["asr_delay_seconds"],
                timeout=controlled["rpc_timeout_seconds"],
            ) as speech:
                async with Session(speech, model, sink) as session:
                    if name != "immediate_wake_command":
                        session.activate()
                    if name == "immediate_wake_command":
                        app = AudioApplication(session, WakeGate(ScenarioWake()), MarkedVAD())
                        async with PipecatAudioInput(app) as ingress:
                            await ingress.feed(PCM * 3 + bytes(FRAME_BYTES) * 20, 0)
                            await ingress.finish()
                        checks.append(app.monitor.activations == 1 and app.utterance_count == 1)
                        checks.append(session.input_bytes == FRAME_BYTES * 23)
                        observations["frontend"] = (
                            "Pipecat PCM ingress; scripted wake/VAD; no real-speaker claim"
                        )
                    elif name == "natural_pause":
                        source = ConversationInput(ReplayInput(MarkedVAD()), session)
                        source.activate()
                        results = []
                        for frame in (
                            [PCM] * 3
                            + [bytes(FRAME_BYTES)] * 10
                            + [PCM] * 3
                            + [bytes(FRAME_BYTES)] * 20
                        ):
                            if await source.feed(frame):
                                results.append(True)
                        await session.generation
                        checks.append(len(results) == 1 and session.turn == 1)
                        for endpoint_ms in (640, 1024):
                            policy = ReplayInput(MarkedVAD(), silence_frames=endpoint_ms // 32)
                            policy.activate()
                            count = 0
                            for frame in (
                                [PCM] * 3
                                + [bytes(FRAME_BYTES)] * 25
                                + [PCM] * 3
                                + [bytes(FRAME_BYTES)] * 32
                            ):
                                count += policy.feed(frame) is not None
                            observations[f"800ms_pause_endpoint_{endpoint_ms}ms_utterances"] = count
                        observations["relaxed_profile_extra_wait_ms"] = 384
                        checks.append(observations["800ms_pause_endpoint_640ms_utterances"] == 2)
                        checks.append(observations["800ms_pause_endpoint_1024ms_utterances"] == 1)
                    elif name == "simulated_service_disconnect":
                        session.speech = SimulatedDisconnect(speech)
                        source = ConversationInput(
                            ReplayInput(MarkedVAD(), silence_frames=2), session
                        )
                        source.activate()
                        await source.feed(PCM)
                        await until(lambda: any(e["event"] == "turn_error" for e in session.events))
                        await source.feed(PCM)
                        await source.feed(bytes(FRAME_BYTES))
                        await source.feed(bytes(FRAME_BYTES))
                        checks.append(session.state == State.LISTENING and not session.input_open)
                        for frame in [PCM, bytes(FRAME_BYTES), bytes(FRAME_BYTES)]:
                            await source.feed(frame)
                        await session.generation
                        checks.append(any(e["event"] == "turn_completed" for e in session.events))
                        checks.append(
                            not any(
                                m["role"] == "assistant" and m["turn_id"] == 1
                                for m in session.history
                            )
                        )
                    elif name == "slow_speech_service":
                        await (await session.submit(PCM))
                        checks.append(any(e["event"] == "turn_error" for e in session.events))
                        checks.append(not sink.chunks and session.state == State.LISTENING)
                        speech.timeout = 2
                        await (await session.submit(PCM))
                        checks.append(session.history[-1]["turn_id"] == 2)
                    elif name == "normal_followup_exit":
                        identifier = session.session_id
                        for _ in range(3):
                            await (await session.submit(PCM))
                        checks.append(
                            session.session_id == identifier and session.state == State.IDLE
                        )
                        checks.append(
                            sum(e["event"] == "turn_completed" for e in session.events) == 2
                        )
                    elif name == "interrupt_reasoning":
                        old = await session.submit(PCM)
                        await asyncio.wait_for(model.entered.wait(), 5)
                        await session.interrupt()
                        session.llm = MockLLM()
                        await (await session.submit(PCM))
                        checks.append(old.done() and session.history[-1]["turn_id"] == 2)
                        checks.append(
                            not any(
                                m["role"] == "assistant" and m["turn_id"] == 1
                                for m in session.history
                            )
                        )
                    elif name == "interrupt_synthesis":
                        old = await session.submit(PCM)
                        await until(
                            lambda: any(e["event"] == "tts_first_chunk" for e in session.events)
                        )
                        await session.interrupt()
                        await (await session.submit(PCM))
                        checks.append(old.done() and not session.playback.queue.qsize())
                        checks.append(
                            not any(
                                m["role"] == "assistant" and m["turn_id"] == 1
                                for m in session.history
                            )
                        )
                    elif name == "task_playback_correction":
                        old = await session.submit(PCM)
                        await asyncio.wait_for(sink.entered.wait(), 5)
                        checks.append(
                            session.tasks.actions.effects == 1 and session.tasks.actions.light
                        )
                        operation = f"{session.session_id}:1:light-change"
                        await session.interrupt()
                        await session.tasks.actions.set_light(operation, True)
                        checks.append(
                            old.done() and session.tasks.actions.effects == 1 and not sink.chunks
                        )
                        sink.delay = 0
                        await (await session.submit(PCM))
                        checks.append(session.tasks.actions.effects == 1)
                        await (await session.submit(PCM))
                        checks.append(
                            session.tasks.actions.effects == 2 and not session.tasks.actions.light
                        )
                    else:
                        await (await session.submit(PCM))
                        checks.append(any(e["event"] == "turn_error" for e in session.events))
                        checks.append(not sink.chunks and session.state == State.LISTENING)
                        session.llm = MockLLM()
                        await (await session.submit(PCM))
                        checks.append(session.history[-1]["turn_id"] == 2)
        except Exception as exc:
            failure = type(exc).__name__  # Never export provider exception bodies.
        if session:
            events.extend(session.events)
        cases.append(
            {
                "name": name,
                "passed": bool(checks) and all(checks) and failure is None,
                "checks": checks,
                "error": failure,
                "controlled": controlled,
                "observations": observations,
                "run_id": session.run_id if session else None,
                "elapsed_seconds": time.monotonic() - started,
                "diagnostics": summarize(session.events) if session else None,
            }
        )
    try:
        stress = await run_text_injected_stress(stress_turns, stress_concurrency)
    except Exception as exc:
        stress = {
            "passed": False,
            "error": type(exc).__name__,
            "turns_per_session": stress_turns,
            "concurrent_sessions": stress_concurrency,
            "limitations": "Text-injected mock stress; no audio hardware or live provider.",
            "_events": [],
        }
    stress_events = stress.pop("_events")
    events.extend(stress_events)
    cases.append({"name": "concurrent_text_injected_turns", **stress})
    if trace:
        write_trace(trace, events)
    return {
        "kind": "scenario-suite",
        "schema_version": 1,
        "mode": "mock",
        "scenarios": cases,
        "limitations": "Real MCP stdio/local software execution; scripted ASR/LLM, TTS tones and memory sink. Controlled barriers/delays are test inputs. No acoustic timings, Pi or real-speaker reliability.",
        "diagnostics": summarize(events),
    }


async def run_background_task_scenario(output_dir, *, duration_seconds=120):
    """Run the fixed multi-turn project-task story with text input and mock tools."""
    from .transcript import Transcript

    output_dir = Path(output_dir).resolve()
    project_var = (Path.cwd() / "var").resolve()
    if not output_dir.is_relative_to(project_var) or output_dir == project_var:
        raise ValueError("background task scenario output must be inside ME3 var/")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "live-transcript.json",
        "task-accepted.json",
        "task-interleaved.json",
        "task-interrupted.json",
        "task-completed.json",
        "event-timeline.json",
        "task-output/project-plan.md",
    )
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError(
            "preserve prior background scenario output; choose a fresh var/results path"
        )

    transcript = Transcript(mode="mock")
    worker = MockBackgroundTaskExecutor(
        output_dir / "task-output", duration_seconds=duration_seconds
    )
    progress_answer = ""
    progress_reply_confirmed = False
    interruption_stop_ms = None

    class TextSpeech:
        async def transcribe(self, key, pcm):
            return pcm.decode("utf-8")

        async def synthesize(self, key, text):
            for _ in range(24):
                await asyncio.sleep(0.002)
                yield PCM

    class ScenarioLLM:
        async def respond(self, history):
            nonlocal progress_answer
            latest = next((row for row in reversed(history) if row.get("role") == "user"), {})
            prompt = latest.get("content", "").casefold()
            if history and history[-1].get("role") == "tool":
                try:
                    receipt = json.loads(history[-1]["content"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    receipt = {}
                yield receipt.get("summary", "The simulated action has no summary.")
                return
            call_id = f"scenario-{latest.get('turn_id', 0)}"
            if "turn on the lights" in prompt:
                yield ToolCall(call_id, "run_device_action", {"action": "light_on"})
            elif "play music" in prompt:
                yield ToolCall(
                    call_id, "control_music", {"action": "play", "query": "Taylor Swift"}
                )
            elif "small project" in prompt or "project files" in prompt:
                yield ToolCall(
                    call_id,
                    "run_agent_task",
                    {"task": "Create a small project outline and setup checklist."},
                )
            elif "pause music" in prompt:
                yield ToolCall(call_id, "control_music", {"action": "pause"})
            elif "change the light color" in prompt:
                yield ToolCall(
                    call_id, "run_device_action", {"action": "set_color", "value": "blue"}
                )
            elif "how is that project going" in prompt:
                steps = ", ".join(worker.completed_steps) or "none yet"
                progress_answer = (
                    "The mock project worker is still running. Current step: "
                    f"{worker.current_phase}. Completed steps: {steps}."
                )
                yield progress_answer
            else:
                yield "I can continue while the mock project worker runs."

    async def sink(_pcm):
        await asyncio.sleep(0.08)

    def save_snapshot(name, *, replace=False):
        path = output_dir / name
        target = path.with_name(f".{path.name}.new") if replace else path
        with target.open("w" if replace else "x", encoding="utf-8") as stream:
            json.dump(transcript.snapshot(), stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.write("\n")
        if replace:
            target.replace(path)

    async with Session(
        TextSpeech(),
        ScenarioLLM(),
        sink,
        transcript=transcript,
        tasks=worker,
        turn_timeout=None,
        llm_timeout=2,
    ) as session:
        session.activate(activation_event="text_injected_scenario")
        save_snapshot("live-transcript.json", replace=True)

        async def speak_turn(text):
            generation = await session.submit(text.encode("utf-8"))
            await generation
            save_snapshot("live-transcript.json", replace=True)

        await speak_turn("Turn on the lights")
        await speak_turn("Play music")
        generation = await session.submit(b"Create a small project with a setup checklist")
        await generation
        task_key = session.key
        task_call = ToolCall(
            f"scenario-{task_key.turn_id}",
            "run_agent_task",
            {"task": "Create a small project outline and setup checklist."},
        )

        def publish_background_update(**update):
            session.background_task_update(task_key, task_call.call_id, **update)
            save_snapshot("live-transcript.json", replace=True)

        duplicate = await worker.execute_background(
            task_key,
            task_call,
            on_update=publish_background_update,
        )
        if (
            worker.started_count != 1
            or duplicate["task_id"] != transcript.snapshot()["tasks"][-1]["task_id"]
        ):
            raise RuntimeError("duplicate submission started or identified another task")
        save_snapshot("live-transcript.json", replace=True)
        save_snapshot("task-accepted.json")

        await speak_turn("Pause music")
        await speak_turn("Change the light color to blue")
        save_snapshot("task-interleaved.json")

        assert worker.task_started_at is not None
        wait_for_status = 35.0 - (time.monotonic() - worker.task_started_at)
        if wait_for_status > 0:
            await asyncio.sleep(wait_for_status)
        generation = await session.submit(b"How is that project going?")
        await until(lambda: session.state == State.SPEAKING, timeout=3)
        await asyncio.sleep(0.02)
        stop_started = time.monotonic()
        await session.interrupt()
        interruption_stop_ms = round((time.monotonic() - stop_started) * 1000, 3)
        if not generation.done():
            await asyncio.gather(generation, return_exceptions=True)
        progress_reply_confirmed = any(
            row["role"] == "assistant" and row["content"] == progress_answer
            for row in session.history
        )
        running = next(
            row for row in transcript.snapshot()["tasks"] if row["task_id"] == worker.task_id
        )
        if running["status"] != "running":
            raise RuntimeError("speech interruption changed the background task state")
        save_snapshot("task-interrupted.json")

        jobs = [job for _, job in worker.operations.values()]
        await asyncio.gather(*jobs)
        save_snapshot("task-completed.json")

    events = list(session.events)
    event_path = output_dir / "event-timeline.json"
    with event_path.open("x", encoding="utf-8") as stream:
        json.dump(events, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")
    task = next(row for row in transcript.snapshot()["tasks"] if row["task_id"] == worker.task_id)
    if (
        task["status"] != "completed"
        or worker.started_count != 1
        or not task.get("completed_steps")
        or task["absolute_path"] != str((output_dir / "task-output/project-plan.md").resolve())
    ):
        raise RuntimeError("background worker did not produce its one truthful completion receipt")
    task_events = [event for event in events if event.get("turn_id") == task["turn_id"]]
    tool_started = next(
        event["monotonic"]
        for event in task_events
        if event["event"] == "tool_started"
        and event.get("tool_call_id") == f"scenario-{task['turn_id']}"
    )
    first_audio = next(
        event["monotonic"] for event in task_events if event["event"] == "sink_write_started_mock"
    )
    return {
        "kind": "background-task-scenario",
        "schema_version": 1,
        "mode": "mock",
        "duration_seconds_requested": duration_seconds,
        "worker_elapsed_seconds": round(time.monotonic() - worker.task_started_at, 3),
        "task_id": worker.task_id,
        "worker_start_count": worker.started_count,
        "duplicate_submission_reused_id": duplicate["task_id"] == worker.task_id,
        "task_status": task["status"],
        "task_phase": task.get("phase"),
        "completed_steps": task.get("completed_steps", []),
        "progress_answer": progress_answer,
        "progress_reply_confirmed_at_mock_sink": progress_reply_confirmed,
        "progress_reply_delivery": "mock sink confirmed"
        if progress_reply_confirmed
        else "speech interrupted before mock sink confirmation",
        "interruption_to_stop_dispatch_ms": interruption_stop_ms,
        "accepted_ack_tool_to_memory_sink_ms": round((first_audio - tool_started) * 1000, 3),
        "mock_device_effects": worker.actions.effects,
        "music_operation_count": len(worker.music.operations),
        "output_path": task["absolute_path"],
        "output_verification": task["verification_level"],
        "snapshots": [name for name in names if (output_dir / name).is_file()],
        "limitations": (
            "Text injected after ASR; deterministic mock LLM, device/music adapters, background worker, "
            "TTS chunks and memory sink. The task file is only a mock artifact under var/results. "
            "No real Onit execution, provider timing, browser audio, acoustic latency, or Pi behavior."
        ),
    }


async def run_background_ack_comparison(scratch_dir, *, samples=5, worker_delay_seconds=0.2):
    """Matched mock comparison: synchronous task wait vs background acceptance.

    The delay belongs only to a deterministic mock worker. It is not provider or
    user-audible latency evidence.
    """
    from statistics import median

    from .transcript import Transcript

    if type(samples) is not int or not 3 <= samples <= 9:
        raise ValueError("acknowledgement comparison samples must be 3..9")
    if not 0.05 <= worker_delay_seconds <= 1:
        raise ValueError("controlled mock worker delay must be 0.05..1 second")
    scratch_dir = Path(scratch_dir).resolve()
    if not scratch_dir.is_relative_to((Path.cwd() / "var").resolve()):
        raise ValueError("acknowledgement comparison scratch must be under var/")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    class ComparisonSpeech:
        async def transcribe(self, key, pcm):
            return pcm.decode("utf-8")

        async def synthesize(self, key, text):
            yield PCM

    class ComparisonLLM:
        async def respond(self, history):
            latest = next(row for row in reversed(history) if row.get("role") == "user")
            yield ToolCall(
                f"compare-{latest['turn_id']}",
                "run_agent_task",
                {"task": "Create the same bounded mock project outline."},
            )

    class DelayedSynchronousWorker(MockTaskExecutor):
        task_mode = "mock"

        def __init__(self):
            super().__init__()
            self.started_count = 0

        async def execute(self, key, call):
            if call.name != "run_agent_task":
                raise ValueError("comparison accepts only the project task")
            self.started_count += 1
            await asyncio.sleep(worker_delay_seconds)
            return {
                "mode": "mock",
                "result_status": "completed",
                "verified_in_memory": True,
                "summary": "The bounded mock project task is complete.",
            }

    async def one_trial(path, style):
        transcript = Transcript(mode="mock")
        worker = (
            DelayedSynchronousWorker()
            if style == "synchronous_before"
            else MockBackgroundTaskExecutor(path, duration_seconds=30)
        )

        async def sink(_pcm):
            await asyncio.sleep(0.002)

        async with Session(
            ComparisonSpeech(),
            ComparisonLLM(),
            sink,
            transcript=transcript,
            tasks=worker,
            close_tasks=True,
            turn_timeout=None,
            llm_timeout=2,
        ) as session:
            session.activate(activation_event="text_injected_ack_comparison")
            generation = await session.submit(b"Start the bounded mock project task")
            await generation
            tool_start = next(
                event["monotonic"] for event in session.events if event["event"] == "tool_started"
            )
            first_audio = next(
                event["monotonic"]
                for event in session.events
                if event["event"] == "sink_write_started_mock"
            )
            endpoint = next(
                event["monotonic"]
                for event in session.events
                if event["event"] == "endpoint_submitted"
            )
            task = transcript.snapshot()["tasks"][0]
            if style == "synchronous_before":
                assert worker.started_count == 1 and task["status"] == "completed"
            else:
                assert worker.started_count == 1 and task["status"] in {"accepted", "running"}
                assert "few minutes" in session.history[-1]["content"]
            return {
                "tool_to_first_mock_audio_ms": round((first_audio - tool_start) * 1000, 3),
                "endpoint_to_first_mock_audio_ms": round((first_audio - endpoint) * 1000, 3),
                "worker_effect_count": worker.started_count,
                "task_status_at_ack": task["status"],
                "task_id": task.get("task_id"),
            }

    observations = {"synchronous_before": [], "background_candidate": []}
    for index in range(samples):
        order = ("synchronous_before", "background_candidate")
        if index % 2:
            order = tuple(reversed(order))
        for style in order:
            result = await one_trial(scratch_dir / f"trial-{index}-{style}", style)
            observations[style].append(result)

    def statistics(rows, field):
        values = [row[field] for row in rows]
        return {
            "samples": len(values),
            "median_ms": round(median(values), 3),
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
        }

    baseline = observations["synchronous_before"]
    candidate = observations["background_candidate"]
    baseline_median = median(row["tool_to_first_mock_audio_ms"] for row in baseline)
    candidate_median = median(row["tool_to_first_mock_audio_ms"] for row in candidate)
    reduction = round(baseline_median - candidate_median, 3)
    return {
        "kind": "background-task-ack-comparison",
        "schema_version": 1,
        "evidence": "controlled software mock; no provider or acoustic timing",
        "worker_delay_seconds": worker_delay_seconds,
        "samples_per_path": samples,
        "order": "alternated by sample",
        "baseline": {
            "path": "synchronous executor waits for worker receipt before acknowledgment",
            "tool_to_first_mock_audio": statistics(baseline, "tool_to_first_mock_audio_ms"),
            "endpoint_to_first_mock_audio": statistics(baseline, "endpoint_to_first_mock_audio_ms"),
        },
        "candidate": {
            "path": "background executor returns accepted receipt before worker completion",
            "tool_to_first_mock_audio": statistics(candidate, "tool_to_first_mock_audio_ms"),
            "endpoint_to_first_mock_audio": statistics(
                candidate, "endpoint_to_first_mock_audio_ms"
            ),
        },
        "median_mock_worker_wait_removed_ms": reduction,
        "task_effects_once_each_trial": all(
            row["worker_effect_count"] == 1 for row in baseline + candidate
        ),
        "stable_task_ids": all(bool(row["task_id"]) for row in candidate),
        "limitations": (
            "The fixed delay is a mock worker wait, not a live model/provider delay. Candidate "
            "acknowledges acceptance, not completion. Full completion and interruption correctness "
            "are covered by the separate wall-clock scenario and focused tests."
        ),
        "raw_samples": observations,
    }


async def run_background_terminal_scenarios(output_dir):
    """Save real Session/Transcript projections from mock failure and cancellation paths."""
    from .transcript import Transcript

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to((Path.cwd() / "var").resolve()):
        raise ValueError("terminal scenario output must be under var/")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {"failed": "task-failed.json", "cancelled": "task-cancelled.json"}
    if any((output_dir / name).exists() for name in names.values()):
        raise FileExistsError("preserve existing terminal snapshots; choose a fresh output folder")

    class TerminalSpeech:
        async def transcribe(self, key, pcm):
            return pcm.decode("utf-8")

        async def synthesize(self, key, text):
            yield PCM

    class StartTaskLLM:
        async def respond(self, history):
            latest = next(row for row in reversed(history) if row.get("role") == "user")
            yield ToolCall(
                f"terminal-{latest['turn_id']}",
                "run_agent_task",
                {"task": "Create a bounded test project outline."},
            )

    async def sink(_pcm):
        await asyncio.sleep(0.002)

    results = {}
    for outcome in ("failed", "cancelled"):
        transcript = Transcript(mode="mock")
        worker = MockBackgroundTaskExecutor(
            output_dir / outcome,
            duration_seconds=1 if outcome == "failed" else 60,
            outcome="failure" if outcome == "failed" else "success",
        )
        async with Session(
            TerminalSpeech(),
            StartTaskLLM(),
            sink,
            transcript=transcript,
            tasks=worker,
            close_tasks=True,
            turn_timeout=None,
            llm_timeout=2,
        ) as session:
            session.activate(activation_event="text_injected_terminal_scenario")
            await (await session.submit(b"Start the bounded project task"))
            task = transcript.snapshot()["tasks"][0]
            task_id = task["task_id"]
            if outcome == "failed":
                await asyncio.gather(*(job for _, job in worker.operations.values()))
            elif not await worker.cancel_task(task_id):
                raise RuntimeError("explicit mock task cancellation was not accepted")
            snapshot = transcript.snapshot()
            row = next(row for row in snapshot["tasks"] if row["task_id"] == task_id)
            expected = outcome
            if row["status"] != expected or worker.started_count != 1:
                raise RuntimeError(f"mock task {outcome} terminal receipt was not recorded once")
            with (output_dir / names[outcome]).open("x", encoding="utf-8") as stream:
                json.dump(snapshot, stream, ensure_ascii=True, allow_nan=False, indent=2)
                stream.write("\n")
            results[outcome] = {
                "task_id": task_id,
                "status": row["status"],
                "worker_start_count": worker.started_count,
                "mode": row["mode"],
            }
    return {
        "kind": "background-task-terminal-scenarios",
        "schema_version": 1,
        "mode": "mock",
        "results": results,
        "snapshots": list(names.values()),
        "limitations": "Actual Session lifecycle with deterministic mock worker; no Onit/provider or physical task was invoked.",
    }
