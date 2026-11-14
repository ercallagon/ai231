"""Attended native transport/VAD barge test; no ASR/LLM/TTS APIs.

Reference is an operator-supplied nonprivate PCM WAV. Logs are private under var/.
Does not certify AEC, speaker audibility or acoustic timing by itself.
"""

import argparse, asyncio, json, os, time, wave, hashlib
from collections import deque
from pathlib import Path
from pipecat.frames.frames import OutputAudioRawFrame, InterruptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipewire_transport import PipeWireInput, PipeWireOutput
from me3_voice.pi_audio import reap


class Observe(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.events = []
        self.playing = False
        self.interrupted = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and type(frame).__name__ in {
            "InterruptionFrame",
            "VADUserStartedSpeakingFrame",
            "VADUserStoppedSpeakingFrame",
            "UserStartedSpeakingFrame",
            "UserStoppedSpeakingFrame",
        }:
            self.events.append(
                {
                    "event": type(frame).__name__,
                    "at": time.monotonic(),
                    "during_reference": self.playing,
                }
            )
            if isinstance(frame, InterruptionFrame) and self.playing:
                self.interrupted.set()
        await self.push_frame(frame, direction)


def sample_process(pid):
    data = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith(("Rss:", "Pss:", "Swap:")):
                key, val = line.split(":", 1)
                data[key + "_kib"] = int(val.split()[0])
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        data["cpu_ticks"] = int(fields[11]) + int(fields[12])
    except (OSError, ValueError):
        data["unavailable"] = True
    return data


def diagnostic_command(node, destination):
    """Container capture goes to a file; --raw writes a stream to stdout.

    Never label a headerless stream .wav or wrap a WAV header as PCM.
    """
    if destination.suffix != ".wav":
        raise ValueError("diagnostic destination must be WAV")
    return [
        "pw-record",
        f"--target={node}",
        "--format=s16",
        "--rate=48000",
        "--channels=1",
        str(destination),
    ]


async def wait_diagnostics(taps, paths, timeout=5):
    async with asyncio.timeout(timeout):
        while True:
            if any(child.returncode is not None for child in taps):
                raise RuntimeError("diagnostic capture exited before readiness")
            if all(path.exists() and path.stat().st_size > 44 for path in paths):
                return
            await asyncio.sleep(0.05)


async def countdown(events, sleep=asyncio.sleep):
    print("[READY] Capture streams active. Remain silent. Trial begins after 3, 2, 1.", flush=True)
    for number in (3, 2, 1):
        print(f"[COUNTDOWN] {number}", flush=True)
        events.append({"event": "countdown", "number": number, "at": time.monotonic()})
        await sleep(1)
    print("[BEGIN] Trial clock starts NOW.", flush=True)
    events.append({"event": "trial_begin", "at": time.monotonic()})


async def cleanup_native(runner, job, inp, out, taps, monitor, prompts):
    """Attempt every owned cleanup even when an earlier shutdown step fails."""
    errors = []

    async def attempt(awaitable):
        try:
            await awaitable
        except BaseException as exc:
            errors.append(type(exc).__name__)

    if runner:
        await attempt(asyncio.wait_for(runner.cancel(), 10))
    if job:
        await attempt(asyncio.wait_for(job, 10))
    await attempt(inp.close_capture())
    await attempt(out.halt(disable=True))
    for child in taps:
        await attempt(reap(child))
    for task in (monitor, prompts):
        if task:
            task.cancel()
    await asyncio.gather(*[task for task in (monitor, prompts) if task], return_exceptions=True)
    return errors


async def run(args):
    print("[STARTUP] Initializing capture. Do not speak yet; wait for READY and BEGIN.", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / "pipewire.log").open("ab")
    taps = []
    runner = None
    job = None

    async def spawn(*a, **kw):
        kw["stderr"] = log
        return await asyncio.create_subprocess_exec(*a, **kw)

    inp = PipeWireInput(args.source, allow_capture=True, spawn=spawn)
    out = PipeWireOutput(args.sink, allow_playback=True, spawn=spawn)
    observer = Observe()
    samples = deque(maxlen=100)
    resource_log = (args.output / "resources.jsonl").open("x")
    sample_count = 0
    lag = []
    started = time.monotonic()
    audio_pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if entry.stat().st_uid == os.getuid() and (entry / "comm").read_text().strip() in {
                    "pipewire",
                    "wireplumber",
                }:
                    audio_pids.append(int(entry.name))
            except OSError:
                pass

    async def sampling():
        nonlocal sample_count
        while True:
            due = time.monotonic() + 0.1
            await asyncio.sleep(0.1)
            lag.append(max(0, time.monotonic() - due) * 1000)
            pids = [os.getpid(), *audio_pids] + [
                c.pid for c in [inp.child, out.child, *taps] if c is not None
            ]
            sample = {"at": time.monotonic(), "processes": [sample_process(pid) for pid in pids]}
            samples.append(sample)
            resource_log.write(json.dumps(sample) + "\n")
            sample_count += 1

    monitor = asyncio.create_task(sampling())
    prompts = None
    trial_error = None
    try:
        # Diagnostic taps are additional processes, not part of baseline topology.
        for name, node in [("raw", args.raw_mic), ("processed", args.source)]:
            taps.append(
                await spawn(
                    *diagnostic_command(node, args.output / (name + ".wav")),
                    stdout=asyncio.subprocess.DEVNULL,
                )
            )
        pair = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy(enable_interruptions=args.profile == "smoke")],
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(
                            user_speech_timeout=0.2, wait_for_transcript=False
                        )
                    ],
                )
            ),
        )
        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000, params=VADParams(confidence=0.7, start_secs=0.16, stop_secs=0.2)
            )
        )
        worker = PipelineWorker(
            Pipeline([inp, vad, pair.user(), observer, out, pair.assistant()]),
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        ready = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def on_ready(*_):
            ready.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        job = asyncio.create_task(runner.run())
        await asyncio.wait_for(ready.wait(), 15)
        await wait_diagnostics(taps, [args.output / "raw.wav", args.output / "processed.wav"])
        await countdown(observer.events)
        epoch = time.monotonic()
        observer.events.append({"event": "profile_start", "profile": args.profile, "at": epoch})
        if args.profile == "smoke":
            print(
                "24s: first 8s silent; next 8s speak over reference; last 8s speak alone. Reference stops on confirmed onset.",
                flush=True,
            )
        else:
            print(
                f"{args.seconds}s {args.profile} recording: "
                + {
                    "speech": "read the planned phrases; no playback",
                    "negative": "room sounds/non-directed speech; no playback",
                    "echo": "remain silent while reference plays",
                    "overlap": "speak the planned phrases while reference continues",
                }[args.profile]
                + ". Automatic interruption is DISABLED for corpus acquisition; this is not a barge-in test.",
                flush=True,
            )

        async def phases():
            observer.events.append({"event": "echo_only_prompt", "at": time.monotonic()})
            await asyncio.sleep(8)
            print("8 seconds: speak over reference; report any earlier false stop.", flush=True)
            observer.events.append({"event": "double_talk_prompt", "at": time.monotonic()})
            await asyncio.sleep(8)
            print("16 seconds: reference ends; speak alone.", flush=True)
            observer.events.append({"event": "user_only_prompt", "at": time.monotonic()})

        if args.profile == "smoke":
            prompts = asyncio.create_task(phases())
        playing = args.profile in {"smoke", "echo", "overlap"}
        wav = wave.open(str(args.reference), "rb") if playing else None
        try:
            if wav and (
                wav.getsampwidth() != 2
                or wav.getnchannels() != 1
                or wav.getframerate() not in (16000, 48000)
                or wav.getnframes() == 0
            ):
                raise ValueError("reference requires nonempty mono PCM16 WAV at 16k/48k")
            observer.playing = playing
            submitted = 0
            while time.monotonic() - epoch < args.seconds:
                if job.done():
                    job.result()
                    raise RuntimeError("native pipeline ended early")
                if any(child.returncode is not None for child in taps):
                    raise RuntimeError("diagnostic capture exited during trial")
                if inp.failure or out.failure:
                    raise RuntimeError(
                        "explicit degraded mode: audio unavailable; stop trial, use text/mock"
                    )
                should_stop = args.profile == "smoke" and (
                    observer.interrupted.is_set() or time.monotonic() - epoch >= 16
                )
                if playing and should_stop:
                    playing = observer.playing = False
                    await worker.queue_frame(InterruptionFrame())
                    observer.events.append(
                        {
                            "event": "reference_stop_requested",
                            "at": time.monotonic(),
                            "submitted_samples": submitted,
                        }
                    )
                if playing:
                    rate = wav.getframerate()
                    chunk = wav.readframes(rate // 50)
                    if not chunk:
                        wav.rewind()
                        continue
                    if submitted == 0:
                        observer.events.append(
                            {
                                "event": "first_reference_enqueue",
                                "at": time.monotonic(),
                                "rate": rate,
                            }
                        )
                    await worker.queue_frame(OutputAudioRawFrame(chunk, rate, 1))
                    submitted += len(chunk) // 2
                    await asyncio.sleep(len(chunk) / (rate * 2))
                else:
                    await asyncio.sleep(0.02)
            if playing:
                observer.playing = False
                await worker.queue_frame(InterruptionFrame())
            observer.events.append(
                {
                    "event": "acquisition_end",
                    "at": time.monotonic(),
                    "reference_samples_enqueued_not_rendered": submitted,
                }
            )
        finally:
            if wav:
                wav.close()
    except BaseException as exc:
        trial_error = type(exc).__name__
        raise
    finally:
        cleanup_errors = await cleanup_native(runner, job, inp, out, taps, monitor, prompts)
        log.close()
        resource_log.close()
        data = {
            "status": "PENDING_OPERATOR_VALIDATION",
            "trial_error": trial_error,
            "cleanup_errors": cleanup_errors,
            "mode": "physical operator acquisition; requires returned evidence and listening review",
            "events": observer.events,
            "samples": list(samples),
            "resource_samples_file": "resources.jsonl",
            "resource_sample_count": sample_count,
            "loop_lag_ms": lag,
            "started_monotonic": started,
            "input_failure": inp.failure,
            "input_failure_detail": inp.failure_detail,
            "diagnostic_returncodes": [child.returncode for child in taps],
            "output_failure": out.failure,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()
            if args.reference
            else None,
            "profile": args.profile,
            "planned_seconds": args.seconds,
            "automatic_interruptions_enabled": args.profile == "smoke",
            "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
            "target_source": args.source,
            "target_reference_sink": args.sink,
            "raw_microphone": args.raw_mic,
            "limitations": "Loop clock is not a calibrated acoustic clock. VAD onset/stop signals are software events; raw recording requires independent annotation. No inference requests.",
        }
        (args.output / "native-audio.json").write_text(json.dumps(data, indent=2) + "\n")
        # pw-record owns WAV container headers and finalization. Validate after
        # child cleanup; preserve malformed/incomplete files as failed evidence.
        diagnostics = {}
        for name in ("raw", "processed"):
            try:
                with wave.open(str(args.output / (name + ".wav")), "rb") as wav:
                    diagnostics[name] = {
                        "rate": wav.getframerate(),
                        "channels": wav.getnchannels(),
                        "frames": wav.getnframes(),
                        "seconds": wav.getnframes() / wav.getframerate(),
                    }
            except (OSError, wave.Error, EOFError) as exc:
                diagnostics[name] = {"error": type(exc).__name__}
        (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--sink", required=True)
    p.add_argument("--raw-mic", required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument(
        "--profile", choices=["smoke", "speech", "echo", "overlap", "negative"], default="smoke"
    )
    p.add_argument("--seconds", type=float)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capture", action="store_true")
    a = p.parse_args()
    if a.seconds is None:
        a.seconds = 24 if a.profile == "smoke" else 60
    if not 1 <= a.seconds <= 7200 or (a.profile == "smoke" and a.seconds != 24):
        p.error("duration must be 1..7200 seconds; smoke has fixed 24-second prompts")
    if a.profile in {"smoke", "echo", "overlap"} and a.reference is None:
        p.error("playback profiles require a local reference WAV")
    if not a.source.startswith("me3.phase2.aec.") or not a.sink.startswith("me3.phase2.aec."):
        p.error("select explicit reviewed Phase 2 AEC nodes; raw mode is not a duplex fallback")
    if not a.output.resolve().is_relative_to((Path.cwd() / "var").resolve()):
        p.error("output must be a fresh path inside project var/")
    if not a.capture:
        print("Capture disabled. Add --capture only for an attended authorized trial.")
        return
    if a.reference is not None:
        try:
            with wave.open(str(a.reference), "rb") as reference:
                if (
                    reference.getsampwidth() != 2
                    or reference.getnchannels() != 1
                    or reference.getframerate() not in (16000, 48000)
                    or reference.getnframes() == 0
                ):
                    p.error("reference requires nonempty mono PCM16 WAV at 16k/48k")
        except (OSError, wave.Error):
            p.error("cannot read reference WAV; no capture started")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
