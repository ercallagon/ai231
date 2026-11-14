"""Prepare a review manifest, or explicitly run budgeted live LLM evaluation.

Default: no inference/network. --run-authorized-live also requires the existing
live-enabled backend/research budgets. ASR and TTS must be mock; both real project
MCP services are used. Scores require independent human/source review.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
import platform
import time

from me3_voice.artifacts import output_path
from me3_voice.configuration import load_backends, load_runtime
from me3_voice.credentials import load_secrets
from me3_voice.research import ResearchClient
from me3_voice.speech import preflight_mock_speech


def terminal_status(case_status):
    """Classify the evaluator stop reason without conflating it with cap exhaustion."""
    if case_status == "RUNTIME_FAILURE":
        return "STOPPED_PROVIDER_OR_RUNTIME_ERROR"
    if case_status == "TERMINAL_FAILURE":
        return "STOPPED_TERMINAL_CASE_FAILURE"
    return "STOPPED_CUMULATIVE_GROQ_BUDGET"


async def evaluate(args, cases, output):
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.speech import SpeechClient

    specs = load_backends(args.backend_config)
    options = load_runtime(args.runtime_config, engine="native")
    if specs["asr"].backend != "mock" or specs["tts"].backend != "mock":
        raise ValueError("evaluation requires mock ASR/TTS; no paid speech or physical I/O")
    if (
        specs["llm"].backend != "external"
        or not specs["llm"].allow_live
        or not specs["llm"].request_limit
    ):
        raise ValueError("explicit external LLM allow_live and positive request_limit required")
    keys = (
        load_secrets(
            args.secrets_file, selected={specs["llm"].auth_env, "ME3_RESEARCH_TAVILY_API_KEY"}
        )
        if args.secrets_file
        else {}
    )
    output.write(
        json.dumps(
            {
                "model": specs["llm"].model,
                "provider_host": urlsplit(specs["llm"].endpoint).hostname,
                "model_request_limit": specs["llm"].request_limit,
                "research_request_limit": options.research.request_limit,
            }
        )
        + "\n"
    )
    output.flush()
    preflight = await preflight_mock_speech(specs)
    output.write(json.dumps({"status": "SPEECH_MCP_PREFLIGHT", "services": preflight}) + "\n")
    output.flush()
    if not all(item["ok"] for item in preflight):
        output.write(
            json.dumps({"status": "ABORTED_PREFLIGHT", "groq_requests": 0, "tavily_requests": 0})
            + "\n"
        )
        output.flush()
        return
    state = NativeState(
        specs,
        keys,
        research=ResearchClient(options.research, keys),
        speech_factory=lambda: SpeechClient(
            backends=specs, credentials=keys, transcript=[c["prompt"] for c in cases]
        ),
    )
    state.budget.requests = args.prior_groq_attempts
    state.research.requests = args.prior_research_http_operations
    state.research.search_requests = args.prior_tavily_search_attempts
    if (
        state.budget.requests > specs["llm"].request_limit
        or state.research.requests > options.research.request_limit
        or state.research.search_requests > options.research.request_limit
    ):
        await state.close()
        output.write(json.dumps({"status": "ABORTED_CUMULATIVE_BUDGET_EXCEEDED"}) + "\n")
        output.flush()
        return
    try:
        try:
            app_context = NativeApplication(
                state, MemorySink(capacity=10000), annotated=True, options=options
            )
            app = await app_context.__aenter__()
        except Exception as exc:
            output.write(
                json.dumps(
                    {
                        "status": "ABORTED_APPLICATION_STARTUP",
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                        "groq_requests": state.budget.requests,
                        "tavily_requests": getattr(state.research, "requests", 0),
                    }
                )
                + "\n"
            )
            output.flush()
            return
        try:
            for case in cases:
                start = time.monotonic()
                before = app.event_count
                groq_before = state.budget.requests
                research_before = state.research.requests
                tavily_before = state.research.search_requests
                case_error = None
                try:
                    await app.utterance(b"\1\0" * 1024, wait=False)
                    await asyncio.wait_for(app.settled.wait(), 120)
                    if app.error:
                        raise app.error
                    status = (
                        "TERMINAL_FAILURE"
                        if app.terminal_failure_code
                        else "PENDING_INDEPENDENT_REVIEW"
                    )
                    if app.terminal_failure_code:
                        case_error = app.terminal_failure_code
                except Exception as exc:
                    status = "RUNTIME_FAILURE"
                    case_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    await app.interrupt()
                latest = []
                for message in reversed(state.context.get_messages()):
                    if message.get("role") == "user":
                        break
                    if message.get("role") == "assistant" and isinstance(
                        message.get("content"), str
                    ):
                        latest.append(message["content"])
                output.write(
                    json.dumps(
                        {
                            **case,
                            "status": status,
                            "review_status": status,
                            "answer": " ".join(reversed(latest)),
                            "seconds": time.monotonic() - start,
                            "completed_utc": datetime.now(timezone.utc).isoformat(),
                            "events": [e for e in app.events if e["sequence"] > before],
                            "sources": list(state.research.sources.values()),
                            "model_requests_used": state.budget.requests,
                            "model_requests_this_case": state.budget.requests - groq_before,
                            "research_http_operations_used": state.research.requests,
                            "research_http_operations_this_case": state.research.requests
                            - research_before,
                            "tavily_search_attempts_used": state.research.search_requests,
                            "tavily_search_attempts_this_case": state.research.search_requests
                            - tavily_before,
                            "error": case_error,
                        }
                    )
                    + "\n"
                )
                output.flush()
                if (
                    status in {"RUNTIME_FAILURE", "TERMINAL_FAILURE"}
                    or state.budget.requests >= specs["llm"].request_limit
                ):
                    terminal = {
                        "record_type": "PILOT_TERMINAL",
                        "status": terminal_status(status),
                        "model_requests_used": state.budget.requests,
                        "research_http_operations_used": state.research.requests,
                        "tavily_search_attempts_used": state.research.search_requests,
                        "remaining_model_request_allowance": max(
                            0, (specs["llm"].request_limit or 0) - state.budget.requests
                        ),
                        "remaining_research_http_operation_allowance": max(
                            0, options.research.request_limit - state.research.requests
                        ),
                    }
                    output.write(json.dumps(terminal) + "\n")
                    output.flush()
                    break
            else:
                output.write(
                    json.dumps(
                        {
                            "record_type": "PILOT_TERMINAL",
                            "status": "COMPLETED_SELECTED_CASES",
                            "model_requests_used": state.budget.requests,
                            "research_http_operations_used": state.research.requests,
                            "tavily_search_attempts_used": state.research.search_requests,
                            "remaining_model_request_allowance": max(
                                0, specs["llm"].request_limit - state.budget.requests
                            ),
                            "remaining_research_http_operation_allowance": max(
                                0, options.research.request_limit - state.research.requests
                            ),
                        }
                    )
                    + "\n"
                )
                output.flush()
        finally:
            await app_context.__aexit__(None, None, None)
    finally:
        await state.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/phase4_questions.json"))
    parser.add_argument("--output", type=output_path, required=True)
    parser.add_argument("--run-authorized-live", action="store_true")
    parser.add_argument("--backend-config", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--secrets-file", type=Path)
    parser.add_argument("--prior-groq-attempts", type=int, default=0)
    parser.add_argument("--prior-research-http-operations", type=int, default=0)
    parser.add_argument("--prior-tavily-search-attempts", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing results; choose a fresh output")
    raw = args.cases.read_bytes()
    if len(raw) > 65536:
        parser.error("cases exceed64KiB")
    cases = json.loads(raw)["cases"]
    if not 1 <= len(cases) <= 40 or any(
        not isinstance(c.get("prompt"), str) or not 1 <= len(c["prompt"]) <= 500 for c in cases
    ):
        parser.error("1..40 bounded case prompts required")
    if args.run_authorized_live and args.backend_config is None:
        parser.error("--backend-config required for authorized live run")
    if (
        min(
            args.prior_groq_attempts,
            args.prior_research_http_operations,
            args.prior_tavily_search_attempts,
        )
        < 0
    ):
        parser.error("prior request counters must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "evidence": "LIVE_PILOT_ATTEMPT_PRECHECK_REQUIRED"
                    if args.run_authorized_live
                    else "PREPARED_NOT_EXECUTED",
                    "case_sha256": hashlib.sha256(raw).hexdigest(),
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "source_sha256": {
                        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [
                            Path("src/me3_voice") / name
                            for name in [
                                "native_app.py",
                                "native_llm.py",
                                "conversation_context.py",
                                "conversation_tools.py",
                                "research.py",
                                "configuration.py",
                            ]
                        ]
                    },
                    "platform": platform.platform(),
                    "scores": None,
                    "note": "No automatic correctness/grounding score; independent review required.",
                }
            )
            + "\n"
        )
        output.flush()
        if args.run_authorized_live:
            asyncio.run(evaluate(args, cases, output))
        else:
            for case in cases:
                output.write(json.dumps(case) + "\n")
    print("Evaluation result/manifest:", args.output)


if __name__ == "__main__":
    main()
