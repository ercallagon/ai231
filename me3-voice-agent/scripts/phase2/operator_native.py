"""Attended native transport/VAD barge test; no ASR/LLM/TTS APIs.

Reference is an operator-supplied nonprivate PCM WAV. Logs are private under var/.
Does not certify AEC, speaker audibility or acoustic timing by itself.
"""

import argparse, asyncio, json, os, time, wave, hashlib
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


async def run(args):
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
    samples = []
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
        while True:
            due = time.monotonic() + 0.1
            await asyncio.sleep(0.1)
            lag.append(max(0, time.monotonic() - due) * 1000)
            pids = [os.getpid(), *audio_pids] + [
                c.pid for c in [inp.child, out.child, *taps] if c is not None
            ]
            samples.append(
                {"at": time.monotonic(), "processes": [sample_process(pid) for pid in pids]}
            )

    monitor = asyncio.create_task(sampling())
    prompts = None
    try:
        # Diagnostic taps are additional processes, not part of baseline topology.
        for name, node in [("raw", args.raw_mic), ("processed", args.source)]:
            taps.append(
                await spawn(
                    "pw-record",
                    f"--target={node}",
                    "--raw",
                    "--format=s16",
                    "--rate=48000",
                    "--channels=1",
                    str(args.output / (name + ".pcm")),
                )
            )
        pair = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy()],
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
        print(
            "24s attended trial: first 8s stay silent; next 8s speak over reference; last 8s speak alone. Reference stops on first confirmed onset. Keep volume/geometry fixed.",
            flush=True,
        )
        with wave.open(str(args.reference), "rb") as wav:
            if (
                wav.getsampwidth() != 2
                or wav.getnchannels() != 1
                or wav.getframerate() not in (16000, 48000)
                or wav.getnframes() == 0
            ):
                raise ValueError("reference requires nonempty mono PCM16 WAV at 16k/48k")
            observer.playing = True
            epoch = time.monotonic()
            reference_rate = wav.getframerate()

            async def phases():
                observer.events.append({"event": "echo_only_prompt", "at": time.monotonic()})
                await asyncio.sleep(8)
                print(
                    "8 seconds: speak over the reference now (if it already stopped, report false interruption).",
                    flush=True,
                )
                observer.events.append({"event": "double_talk_prompt", "at": time.monotonic()})
                await asyncio.sleep(8)
                print("16 seconds: reference ends; speak alone.", flush=True)
                observer.events.append({"event": "user_only_prompt", "at": time.monotonic()})

            prompts = asyncio.create_task(phases())
            while time.monotonic() - epoch < 16 and not observer.interrupted.is_set():
                chunk = wav.readframes(reference_rate // 50)
                if not chunk:
                    wav.rewind()
                    continue
                await worker.queue_frame(OutputAudioRawFrame(chunk, reference_rate, 1))
                await asyncio.sleep(len(chunk) / (reference_rate * 2))
            observer.playing = False
            # Explicit silence phase: flush reference even when no onset occurred.
            await worker.queue_frame(InterruptionFrame())
            print(
                "Reference stopped. Continue speaking when prompted; preserve recordings.",
                flush=True,
            )
            while time.monotonic() - epoch < 24:
                if job.done():
                    job.result()
                    raise RuntimeError("native pipeline ended early")
                if inp.failure or out.failure:
                    raise RuntimeError(
                        "explicit degraded mode: audio unavailable; stop trial, use text/mock"
                    )
                await asyncio.sleep(0.1)
    finally:
        try:
            if runner:
                await runner.cancel()
            if job:
                await asyncio.wait_for(job, 10)
        finally:
            await inp.close_capture()
            await out.halt(disable=True)
            for child in taps:
                await reap(child)
            monitor.cancel()
            if prompts:
                prompts.cancel()
            await asyncio.gather(monitor, *([prompts] if prompts else []), return_exceptions=True)
            log.close()
        data = {
            "status": "PENDING_OPERATOR_VALIDATION",
            "mode": "physical operator acquisition; requires returned evidence and listening review",
            "events": observer.events,
            "samples": samples,
            "loop_lag_ms": lag,
            "started_monotonic": started,
            "input_failure": inp.failure,
            "output_failure": out.failure,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
            "target_source": args.source,
            "target_reference_sink": args.sink,
            "raw_microphone": args.raw_mic,
            "limitations": "Loop clock is not a calibrated acoustic clock. VAD onset/stop signals are software events; raw recording requires independent annotation. No inference requests.",
        }
        (args.output / "native-audio.json").write_text(json.dumps(data, indent=2) + "\n")
        for name in ("raw", "processed"):
            pcm = args.output / (name + ".pcm")
            if pcm.exists():
                payload = pcm.read_bytes()
                payload = payload[: len(payload) // 2 * 2]
                with wave.open(str(args.output / (name + ".wav")), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(48000)
                    wav.writeframes(payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--sink", required=True)
    p.add_argument("--raw-mic", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capture", action="store_true")
    a = p.parse_args()
    if not a.source.startswith("me3.phase2.aec.") or not a.sink.startswith("me3.phase2.aec."):
        p.error("select explicit reviewed Phase 2 AEC nodes; raw mode is not a duplex fallback")
    if not a.output.resolve().is_relative_to((Path.cwd() / "var").resolve()):
        p.error("output must be a fresh path inside project var/")
    if not a.capture:
        print("Capture disabled. Add --capture only for an attended authorized trial.")
        return
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
