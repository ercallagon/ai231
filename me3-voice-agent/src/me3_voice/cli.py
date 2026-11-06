"""Offline entry points. All outputs go under var/, all providers are mock."""

import argparse
import asyncio
import importlib.metadata
import json
import platform
from contextlib import AsyncExitStack
from pathlib import Path

from .artifacts import output_path
from .configuration import ConfigurationError
from .providers import MockLLM
from .session import Session
from .speech import SpeechClient


async def mock_conversation(utterances, trace=None, transcript=False, backends=None):
    from .harness import run_utterances

    chunks = []

    async def sink(pcm):
        await asyncio.sleep(0)
        chunks.append(pcm)

    async with SpeechClient(backends=backends) as speech:
        protocols = dict(speech.protocols)
        async with AsyncExitStack() as stack:
            display = None
            if transcript:
                from .transcript import Transcript, TranscriptPublisher

                display = Transcript(mode="mock")
                await stack.enter_async_context(TranscriptPublisher(display))
            async with Session(
                speech,
                MockLLM(),
                sink,
                transcript=display,
                llm_timeout=backends["llm"].timeout_seconds if backends else 5,
            ) as session:
                session.activate()
                await run_utterances(session, utterances)
                history = list(session.history)
        result = {
            "mode": "mock",
            "python": platform.python_version(),
            "protocols": protocols,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("mcp", "pipecat-ai", "onnxruntime")
            },
            "history": history,
            "run_id": session.run_id,
            "session_id": session.session_id,
            "sink_chunks": len(chunks),
            "events": session.events,
            "limitations": "Mock ASR/LLM; TTS tones; memory sink; no acoustic timing",
        }
    if len(history) != 2 * len(utterances) or len(chunks) != 4 * len(utterances):
        raise RuntimeError("mock follow-up integration did not complete")
    if trace:
        from .telemetry import write_trace

        write_trace(trace, session.events)
    return result


async def demo(output, trace=None, transcript=False, backends=None):
    current = Path("var/results/ui-transcript.json").resolve()
    if transcript and (output.resolve() == current or trace and trace.resolve() == current):
        raise ValueError("transcript state must be separate from diagnostic/demo output")
    if trace and trace.resolve() == output.resolve():
        raise ValueError("trace and demo output must use distinct paths")
    if trace and trace.exists():
        raise FileExistsError("preserve existing trace; supply a new output path")
    result = await mock_conversation(
        [b"\x01\x00" * 1024, b"\x02\x00" * 1024],
        trace=trace,
        transcript=transcript,
        backends=backends,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Mock MCP/Pipecat follow-up passed; {result['sink_chunks']} memory-sink chunks; {output}"
    )


def replay(args):
    if not args.mock_conversation:
        if (
            args.trace
            or args.transcript
            or args.wake_frame is not None
            or args.backend_config
            or getattr(args, "secrets_file", None)
        ):
            raise ValueError(
                "trace/transcript, injected wake and backend configuration require --mock-conversation; use readiness replay for diagnostics only"
            )
        from .readiness import replay as diagnose

        result = diagnose(args)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CPU frontend diagnostics only; no speech MCP/providers started; {args.output}")
        return
    from .runtime import replay_application

    result = asyncio.run(replay_application(args))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Application replay: {result['utterances']} utterances; actual CPU frontend / {result['agent_mode']}; {args.output}"
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    mock = commands.add_parser("demo")
    mock.add_argument(
        "--scenarios",
        action="store_true",
        help="deterministic mock lifecycle/task suite through our MCP servers",
    )
    mock.add_argument(
        "--stress-turns",
        type=int,
        default=25,
        help="text-injected turns per concurrent mock session in the scenario reliability cycle",
    )
    mock.add_argument(
        "--stress-concurrency",
        type=int,
        default=4,
        help="concurrent post-ASR mock sessions in the scenario reliability cycle",
    )
    mock.add_argument(
        "--background-task",
        action="store_true",
        help="run the text-injected multi-turn background-task mock story",
    )
    mock.add_argument(
        "--background-task-seconds",
        type=int,
        default=120,
        help="mock worker wall-clock duration, bounded to 1..180 seconds",
    )
    mock.add_argument(
        "--background-task-output-dir",
        type=output_path,
        default=output_path("var/results/016-background-task-ui/run"),
        help="fresh experiment output directory under var/",
    )
    mock.add_argument(
        "--background-ack-comparison",
        action="store_true",
        help="matched synchronous-vs-background mock-worker acknowledgment timings",
    )
    mock.add_argument(
        "--ack-comparison-samples",
        type=int,
        default=5,
        help="paired controlled mock samples, 3..9",
    )
    mock.add_argument(
        "--ack-comparison-worker-delay",
        type=float,
        default=0.2,
        help="controlled mock worker wait in seconds, 0.05..1",
    )
    mock.add_argument(
        "--background-terminal-scenarios",
        action="store_true",
        help="save real Session transcript projections for mock failure and cancellation",
    )
    mock.add_argument(
        "--background-terminal-output-dir",
        type=output_path,
        default=output_path("var/results/016-background-task-ui/terminal-states"),
        help="fresh experiment output directory under var/",
    )
    mock.add_argument("--backend-config", type=Path)
    mock.add_argument("--output", type=output_path, default="var/results/mock-demo.json")
    mock.add_argument("--trace", type=output_path, help="exclusive JSONL diagnostic export")
    mock.add_argument("--transcript", action="store_true", help="opt in to current-session UI text")
    recorded = commands.add_parser("replay")
    recorded.add_argument("wav", type=Path)
    from .readiness import add_replay_options

    add_replay_options(recorded)
    recorded.set_defaults(output=output_path("var/results/application-replay.json"))
    recorded.add_argument("--wake-frame", type=int)
    recorded.add_argument("--backend-config", type=Path)
    recorded.add_argument(
        "--direct-clock",
        action="store_true",
        help="opt in exact local-clock requests to the typed task path; default off",
    )
    recorded.add_argument(
        "--secrets-file",
        type=Path,
        help="explicit private project var/cache/*.env; never passed to browser or MCP arguments",
    )
    recorded.add_argument(
        "--llm-secrets-file",
        type=Path,
        help="private reasoning credential file, kept separate from ASR/TTS keys",
    )
    recorded.add_argument(
        "--onit-python",
        type=Path,
        help="use the installed Onit environment as the primary general reasoning harness",
    )
    recorded.add_argument(
        "--onit-max-requests",
        type=int,
        help="optional per-run Onit API request cap (unset delegates quota to the provider)",
    )
    recorded.add_argument(
        "--turn-policy",
        choices=["silence", "smart-turn"],
        default="silence",
        help="optional bundled CPU Smart Turn with bounded silence fallback; Pi unverified",
    )
    recorded.add_argument(
        "--mock-transcripts", nargs="+", help="test-only ASR script, independent of WAV content"
    )
    recorded.add_argument(
        "--endpoint-ms",
        type=int,
        default=640,
        help="application silence endpoint; 32 ms steps, 160..2048; not semantic turn detection",
    )
    recorded.add_argument(
        "--pace",
        type=float,
        default=0,
        help="wall delay / audio duration; 0 is unpaced scheduling stress",
    )
    recorded.add_argument("--trace", type=output_path)
    recorded.add_argument(
        "--playback-wav",
        type=output_path,
        help="save acknowledged in-memory PCM as a 16 kHz WAV under var/; no acoustic claim",
    )
    recorded.add_argument("--transcript", action="store_true")
    recorded.add_argument(
        "--conversation",
        "--mock-conversation",
        dest="mock_conversation",
        action="store_true",
        help="connected application; providers default to mocks unless explicitly configured",
    )
    args = parser.parse_args()
    try:
        if args.trace and args.trace.resolve() == args.output.resolve():
            parser.error("report and trace must use distinct paths")
        if args.command == "demo":
            from .configuration import load_backends

            backends = load_backends(args.backend_config)
            if any(spec.backend != "mock" for spec in backends.values()):
                parser.error(
                    "demo is deterministic/mock-only; use replay --conversation --backend-config for external adapters"
                )
            if args.scenarios and args.backend_config:
                parser.error("scenario suite uses fixed controlled mocks; omit --backend-config")
            if args.scenarios and args.transcript:
                parser.error(
                    "scenario suite exports diagnostics; use demo/replay for opt-in transcript publication"
                )
            if args.background_task and args.background_ack_comparison:
                parser.error("choose either the long task scenario or acknowledgment comparison")
            selected_background_modes = sum(
                bool(value)
                for value in (
                    args.background_ack_comparison,
                    args.background_task,
                    args.background_terminal_scenarios,
                )
            )
            if selected_background_modes > 1:
                parser.error("choose only one background-task scenario mode")
            if args.background_terminal_scenarios:
                if args.scenarios or args.backend_config or args.transcript:
                    parser.error("terminal task scenarios are mock-only; omit other modes")
                if args.output.exists():
                    parser.error("preserve existing output; choose a fresh --output path")
                from .scenarios import run_background_terminal_scenarios

                result = asyncio.run(
                    run_background_terminal_scenarios(args.background_terminal_output_dir)
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(f"Mock task failure/cancellation snapshots passed; {args.output}")
            elif args.background_ack_comparison:
                if args.scenarios or args.backend_config or args.transcript:
                    parser.error("acknowledgment comparison is mock-only; omit other modes")
                if args.output.exists():
                    parser.error("preserve existing output; choose a fresh --output path")
                from .scenarios import run_background_ack_comparison

                result = asyncio.run(
                    run_background_ack_comparison(
                        Path("var/scratch/016-background-task-ui/ack-comparison"),
                        samples=args.ack_comparison_samples,
                        worker_delay_seconds=args.ack_comparison_worker_delay,
                    )
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                if not result["task_effects_once_each_trial"] or not result["stable_task_ids"]:
                    raise SystemExit(1)
                print(
                    "Mock acknowledgment comparison passed; median mock worker wait removed "
                    f"{result['median_mock_worker_wait_removed_ms']:.1f} ms; {args.output}"
                )
            elif args.background_task:
                if args.scenarios or args.backend_config or args.transcript:
                    parser.error(
                        "background task scenario is a fixed mock-only run; omit other modes"
                    )
                if args.output.exists():
                    parser.error("preserve existing output; choose a fresh --output path")
                from .scenarios import run_background_task_scenario

                result = asyncio.run(
                    run_background_task_scenario(
                        args.background_task_output_dir,
                        duration_seconds=args.background_task_seconds,
                    )
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                if result["worker_start_count"] != 1 or result["task_status"] != "completed":
                    raise SystemExit(1)
                print(
                    "Mock background task completed; "
                    f"{result['worker_elapsed_seconds']:.1f}s; {args.output}"
                )
            elif args.scenarios:
                from .scenarios import run_scenarios

                result = asyncio.run(
                    run_scenarios(
                        args.trace,
                        stress_turns=args.stress_turns,
                        stress_concurrency=args.stress_concurrency,
                    )
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                if not all(case["passed"] for case in result["scenarios"]):
                    raise SystemExit(1)
            else:
                asyncio.run(demo(args.output, args.trace, args.transcript, backends))
        else:
            if args.turn_policy != "silence" and not args.mock_conversation:
                parser.error("turn-policy requires --conversation")
            if args.onit_python and not args.mock_conversation:
                parser.error("--onit-python requires --conversation")
            if args.onit_max_requests is not None and not args.onit_python:
                parser.error("--onit-max-requests requires --onit-python")
            if args.onit_max_requests is not None and not 1 <= args.onit_max_requests <= 8:
                parser.error("--onit-max-requests must be 1..8 when explicitly set")
            if args.onit_python and (
                not args.onit_python.is_absolute() or not args.onit_python.is_file()
            ):
                parser.error("--onit-python must point to an existing absolute Python interpreter")
            if args.wake_frame is not None and args.wake_frame < 0:
                parser.error("wake frame cannot be negative")
            if args.endpoint_ms % 32 or not 160 <= args.endpoint_ms <= 2048:
                parser.error("endpoint-ms must be 160..2048 in 32 ms steps")
            if not args.mock_conversation and args.endpoint_ms != 640:
                parser.error(
                    "endpoint-ms applies to --mock-conversation; readiness diagnostics use their recorded policy"
                )
            if not 0 <= args.pace <= 2:
                parser.error("pace must be finite and between 0 and 2")
            replay(args)

    except ConfigurationError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
