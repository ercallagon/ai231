"""Deterministic application-level evaluation through Pipecat and speech MCP."""

import asyncio
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

from langchain_core.messages import AIMessage
from me3_voice.configuration import load_backends, RuntimeOptions
from me3_voice.native_app import NativeApplication, NativeState
from me3_voice.playback import MemorySink
from me3_voice.providers import ToolCall
from me3_voice.research import ResearchClient, ResearchOptions
from me3_voice.speech import SpeechClient
from me3_voice.task_runtime import TaskRuntime


class TimedMemorySink(MemorySink):
    def __init__(self):
        super().__init__(capacity=10000)
        self.chunk_times = []

    async def __call__(self, pcm):
        await super().__call__(pcm)
        self.chunk_times.append(time.monotonic())


class ScenarioModel:
    """A transcript fixture that controls provider decisions, not a reasoner."""

    def __init__(self):
        self.steps = [
            "Four.",
            "Do you mean the most recent Senate election, a particular state, or a presidential election?",
            ToolCall(
                "senate-search",
                "web_search",
                {"query": "latest completed US Senate election official results", "max_results": 3},
            ),
            ToolCall("senate-fetch", "retrieve_source", {"url": "https://official.example/senate"}),
            "The fixture reports the latest completed result as described here [S1].",
            ToolCall(
                "ecb-search",
                "web_search",
                {"query": "latest ECB president official profile", "max_results": 3},
            ),
            ToolCall(
                "ecb-fetch",
                "retrieve_source",
                {
                    "url": "https://www.ecb.europa.eu/about/organisation/governance/html/index.en.html"
                },
            ),
            "The ECB president is Christine Lagarde.",
            ToolCall(
                "conflict-search",
                "web_search",
                {"query": "conflicting policy date reports", "max_results": 3},
            ),
            ToolCall(
                "conflict-fetch-a", "retrieve_source", {"url": "https://source-a.example/report"}
            ),
            ToolCall(
                "conflict-fetch-b", "retrieve_source", {"url": "https://source-b.example/report"}
            ),
            "The reports conflict: source A gives 12 May [S2], while source B gives 14 May [S3]. I can't resolve which is correct from this evidence.",
            "The 2024 US presidential election was held on November 5, 2024.",
            "Hello! What can I help with?",
        ]
        self.requests = []

    async def respond(self, history):
        self.requests.append(json.loads(json.dumps(history)))
        step = len(self.requests) - 1
        yield self.steps[step]


def _last_assistant(context):
    return next(
        message["content"]
        for message in reversed(context.get_messages())
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    )


async def test_native_application_intelligence_routes_and_efficiency(tmp_path):
    """Exercise real context/tool/MCP/output lifecycle with controlled model choices."""
    fixtures = Path(__file__).parent / "fixtures" / "research_html"
    # This scenario uses deterministic local research fixtures; expose the
    # same capabilities a live model receives when research is enabled.
    options = RuntimeOptions(engine="native", research=ResearchOptions(True, 20))
    specs = load_backends()
    model = ScenarioModel()
    sink = TimedMemorySink()
    fixed_now = datetime(2026, 11, 11, 4, 0, tzinfo=timezone.utc)
    research = ResearchClient(options.research)
    research_calls = {"search": 0, "retrieve": 0}

    async def search(query, max_results=5):
        research_calls["search"] += 1
        if "ECB" in query or "ecb" in query.lower():
            results = [
                {
                    "url": "https://www.ecb.europa.eu/about/organisation/governance/html/index.en.html",
                    "title": "ECB governance",
                }
            ]
        elif "conflicting" in query:
            results = [
                {"url": "https://source-a.example/report", "title": "Report A"},
                {"url": "https://source-b.example/report", "title": "Report B"},
            ]
        else:
            results = [
                {"url": "https://official.example/senate", "title": "Official election results"}
            ]
        return {"kind": "untrusted_search_leads", "query": query, "results": results[:max_results]}

    async def read(_method, url, **_kwargs):
        research_calls["retrieve"] += 1
        if "ecb.europa.eu" in url:
            body = (fixtures / "missing_article_text.html").read_bytes()
        elif "source-a" in url:
            body = b"<html><main><article><h1>Report A</h1><p>Report A states the policy date is 12 May, according to its published schedule.</p></article></main></html>"
        elif "source-b" in url:
            body = b"<html><main><article><h1>Report B</h1><p>Report B states the policy date is 14 May, according to its published schedule.</p></article></main></html>"
        else:
            body = b"<html><main><article><h1>Election result</h1><p>The frozen fixture records the latest completed Senate election result on November 3, 2026, with the final count certified on November 8, 2026.</p></article></main></html>"
        return {"type": "text/html", "data": body}

    research.search = search
    research.read = read

    def task_runtime_factory(spec, credentials, budget, research):
        return TaskRuntime(
            spec,
            credentials,
            budget=budget,
            research=research,
            database=tmp_path / "intelligence-eval.sqlite",
            artifact_root=tmp_path / "intelligence-eval-artifacts",
        )

    matrix_path = Path(__file__).parent / "fixtures" / "swift_agent_qualification.json"
    routing_matrix = json.loads(matrix_path.read_text())["routing_scenarios"]
    transcript = [case["utterance"] for case in routing_matrix]
    state = NativeState(
        specs,
        {},
        research=research,
        task_runtime_factory=task_runtime_factory,
        speech_factory=lambda: SpeechClient(backends=specs, transcript=transcript),
    )
    startup_started = time.monotonic()
    async with NativeApplication(
        state,
        sink,
        annotated=True,
        upstream=model,
        options=options,
        clock=lambda: fixed_now,
    ) as app:
        startup_seconds = time.monotonic() - startup_started
        questions = routing_matrix
        results = []
        for case in questions:
            name, expected_route = case["id"], case["expected_route"]
            event_start = app.event_count
            model_start = len(model.requests)
            sink_start = len(sink.chunk_times)
            started = time.monotonic()
            await app.utterance(b"\1\0" * 512)
            ended = time.monotonic()
            answer = _last_assistant(state.context)
            events = [event for event in app.events if event["sequence"] > event_start]
            first_audio = (
                sink.chunk_times[sink_start] - started
                if len(sink.chunk_times) > sink_start
                else None
            )
            prompt_chars = sum(
                len(json.dumps(item, ensure_ascii=False)) for item in model.requests[model_start:]
            )
            asr_final = next(
                (event.get("seconds") for event in events if event["event"] == "asr_final"), None
            )
            llm_first_token = next(
                (event.get("seconds") for event in events if event["event"] == "llm_first_token"),
                None,
            )
            tts_start = next(
                (event["at"] for event in events if event["event"] == "tts_start"), None
            )
            tts_pcm = next((event["at"] for event in events if event["event"] == "tts_pcm"), None)
            tool_starts = {
                event.get("tool_call_id"): event["at"]
                for event in events
                if event["event"] == "tool_started"
            }
            completed_tools = [
                event.get("tool_name") for event in events if event["event"] == "tool_completed"
            ]
            tool_latencies = [
                event["at"] - tool_starts[event["tool_call_id"]]
                for event in events
                if event["event"] == "tool_completed" and event.get("tool_call_id") in tool_starts
            ]
            if "web_search" in completed_tools and "retrieve_source" in completed_tools:
                if app.terminal_failure_code:
                    actual_route = "research_unavailable"
                elif completed_tools.count("retrieve_source") > 1:
                    actual_route = "multi_source_research"
                else:
                    actual_route = "research"
            elif answer.startswith("Do you mean"):
                actual_route = "clarify"
            else:
                actual_route = "direct_answer"
            results.append(
                {
                    "scenario": name,
                    "expected_route": expected_route,
                    "actual_route": actual_route,
                    "expected_action": case["expected_action"],
                    "actual_action": {
                        "completed_tools": completed_tools,
                        "answer": answer,
                        "terminal_failure": app.terminal_failure_code,
                    },
                    "status": "PASS" if actual_route == expected_route else "FAIL",
                    "elapsed_seconds": ended - started,
                    "first_memory_pcm_seconds": first_audio,
                    "asr_final_seconds": asr_final,
                    "llm_first_token_seconds": llm_first_token,
                    "tts_first_pcm_seconds": tts_pcm - tts_start
                    if tts_pcm is not None and tts_start is not None
                    else None,
                    "tool_latency_seconds": tool_latencies,
                    "model_requests": sum(event["event"] == "llm_request" for event in events),
                    "tool_calls": sum(event["event"] == "tool_completed" for event in events),
                    "search_operations": sum(
                        event.get("tool_name") == "web_search"
                        for event in events
                        if event["event"] == "tool_completed"
                    ),
                    "retrieval_operations": sum(
                        event.get("tool_name") == "retrieve_source"
                        for event in events
                        if event["event"] == "tool_completed"
                    ),
                    "context_chars_sent_to_mock": prompt_chars,
                    "approx_context_tokens_chars_div_4": (prompt_chars + 3) // 4,
                    "estimated_provider_cost_usd": 0,
                    "answer": answer,
                    "terminal_failure": app.terminal_failure_code,
                }
            )

        # Contract assertions: the fixed transcript selects routes; these prove
        # the application executes them faithfully, not that Groq would choose them.
        assert results[0]["tool_calls"] == 0 and results[0]["model_requests"] == 1
        assert "Do you mean" in results[1]["answer"] and results[1]["tool_calls"] == 0
        assert results[2]["search_operations"] == 1 and results[2]["retrieval_operations"] == 1
        assert "[S1]" in results[2]["answer"]
        assert "Who won the senatorial elections in the US?" in json.dumps(model.requests[2])
        assert results[3]["terminal_failure"] == "insufficient_verified_evidence"
        assert "won't guess" in results[3]["answer"]
        assert results[4]["retrieval_operations"] == 2 and "can't resolve" in results[4]["answer"]
        assert "[S2]" in results[4]["answer"] and "[S3]" in results[4]["answer"]
        assert "12 May" in research.sources["S2"]["text"]
        assert "14 May" in research.sources["S3"]["text"]
        assert results[5]["tool_calls"] == 0 and "2024" in results[5]["answer"]
        assert results[6]["tool_calls"] == 0 and "help" in results[6]["answer"].casefold()
        assert all(row["status"] == "PASS" for row in results)
        assert "Runtime clock:" in app.llm._settings.system_instruction
        assert "2026-11-11T04:00:00+00:00" in app.llm._settings.system_instruction
        assert "latest relevant completed event" in app.llm._settings.system_instruction
        assert "distinguish event" in app.llm._settings.system_instruction
        assert research_calls == {"search": 3, "retrieve": 4}
        assert not app.error

    await state.close()
    ordered_latency = sorted(row["elapsed_seconds"] for row in results)
    root = Path(__file__).resolve().parents[1]
    source_files = [
        root / "src/me3_voice/native_app.py",
        root / "src/me3_voice/native_llm.py",
        root / "src/me3_voice/native_speech.py",
        root / "src/me3_voice/research.py",
        Path(__file__).resolve(),
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    record = {
        "evidence_class": "offline deterministic mock; real Pipecat NativeApplication and project ASR/TTS MCP stdio servers",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pipecat_version": importlib.metadata.version("pipecat-ai"),
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "startup_seconds_separate": startup_seconds,
        "scenario_count": len(results),
        "scenarios": results,
        "routing_matrix": results,
        "totals": {
            "model_requests": sum(row["model_requests"] for row in results),
            "tool_calls": sum(row["tool_calls"] for row in results),
            "search_operations": research_calls["search"],
            "retrieval_operations": research_calls["retrieve"],
            "mock_cost_usd": 0,
            "live_cost_usd": None,
        },
        "response_latency_summary_seconds": {
            "n": len(results),
            "median": (
                ordered_latency[(len(results) - 1) // 2] + ordered_latency[len(results) // 2]
            )
            / 2,
            "p95_nearest_rank": ordered_latency[-1],
        },
        "limits": [
            "scripted provider output preselects direct/clarify/research behavior; no model reasoning quality claim",
            "actual_route is inferred from executed native events and terminal outcome; the fixture controls the upstream choice",
            "memory PCM is not audible playback or physical voice latency",
            "four retrieval attempts and three searches are deterministic mocked research operations",
            "context tokens are a chars/4 proxy, not provider tokenizer usage",
            "LangGraph voice delegation is separately covered by test_native_voice_mcp_remains_responsive_while_durable_task_runs",
        ],
    }
    target = Path("var/results/architecture_modernization/intelligence-offline.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


async def test_native_voice_routes_steers_and_verifies_persistent_website_task(tmp_path):
    """Five turns use one native context and one real task workspace/checkpointer."""
    from me3_voice.playback import MemorySink
    from me3_voice.task_runtime import TaskRuntime

    matrix = json.loads(
        (Path(__file__).parent / "fixtures" / "swift_agent_qualification.json").read_text()
    )
    scenario = matrix["scenario"]
    task_model_waiting = asyncio.Event()
    release_stale_tool = asyncio.Event()

    def tool(name, args, call_id):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])

    class WebsiteTaskModel:
        def __init__(self):
            self.calls = 0
            self.system_prompts = []

        async def __call__(self, messages, _tools):
            self.calls += 1
            self.system_prompts.append(messages[0].content)
            if self.calls == 1:
                return tool("create_directory", {"path": "portfolio/assets"}, "portfolio-dir")
            if self.calls == 2:
                task_model_waiting.set()
                await release_stale_tool.wait()
                # A pre-steering model call must be rejected by the revision fence.
                return tool(
                    "write_artifact",
                    {
                        "path": "portfolio/index.html",
                        "content": "<!doctype html><h1>Stale light theme</h1>",
                    },
                    "stale-light-index",
                )
            assert "requirement revision 2" in messages[0].content
            steps = [
                tool(
                    "write_artifact",
                    {
                        "path": "portfolio/index.html",
                        "content": """<!doctype html><html><head><title>Portfolio</title><link rel="stylesheet" href="assets/site.css"></head><body><nav><a href="about.html">About</a><a href="projects.html">Projects</a><a href="contact.html">Contact</a></nav><main><section id="about"><h1>About</h1><p>A short personal introduction.</p></section><section id="projects"><h2>Projects</h2><ul><li>Portfolio site</li><li>Community garden map</li></ul></section><section id="contact"><h2>Contact</h2><a href="mailto:hello@example.test">Email</a></section></main></body></html>""",
                    },
                    "write-index",
                ),
                tool(
                    "write_artifact",
                    {
                        "path": "portfolio/about.html",
                        "content": '<!doctype html><html><head><link rel="stylesheet" href="assets/site.css"></head><body><h1>About</h1><a href="index.html">Home</a></body></html>',
                    },
                    "write-about",
                ),
                tool(
                    "write_artifact",
                    {
                        "path": "portfolio/projects.html",
                        "content": '<!doctype html><html><head><link rel="stylesheet" href="assets/site.css"></head><body><h1>Projects</h1><ul><li>Portfolio site</li><li>Community garden map</li></ul><a href="index.html">Home</a></body></html>',
                    },
                    "write-projects",
                ),
                tool(
                    "write_artifact",
                    {
                        "path": "portfolio/contact.html",
                        "content": '<!doctype html><html><head><link rel="stylesheet" href="assets/site.css"></head><body><h1>Contact</h1><a href="mailto:hello@example.test">Email</a><a href="index.html">Home</a></body></html>',
                    },
                    "write-contact",
                ),
                tool(
                    "write_artifact",
                    {
                        "path": "portfolio/assets/site.css",
                        "content": "body { background: #111; color: #f4f4f4; } a { color: #8bd5ff; }",
                    },
                    "write-dark-css",
                ),
                tool("list_workspace", {"path": "portfolio"}, "list-portfolio"),
                tool("read_artifact", {"path": "portfolio/index.html"}, "read-index"),
                tool("read_artifact", {"path": "portfolio/projects.html"}, "read-projects"),
                tool("read_artifact", {"path": "portfolio/assets/site.css"}, "read-dark-css"),
                AIMessage(
                    content="Created and read back the five requested portfolio files. The existing project now uses a dark theme and includes the added project."
                ),
            ]
            return steps[self.calls - 3]

    class VoiceRouter:
        def __init__(self):
            self.calls_by_turn = {}
            self.task_id = None
            self.routes = []

        async def respond(self, history):
            turn = sum(message.get("role") == "user" for message in history)
            self.calls_by_turn[turn] = self.calls_by_turn.get(turn, 0) + 1
            call_index = self.calls_by_turn[turn]
            user_text = next(
                message["content"] for message in reversed(history) if message.get("role") == "user"
            )
            if turn == 1 and call_index == 1:
                self.routes.append("start_task")
                yield ToolCall("voice-start-portfolio", "start_task", {"goal": user_text})
            elif turn == 2 and call_index == 1:
                self.routes.append("steer_task")
                yield ToolCall(
                    "voice-steer-portfolio",
                    "steer_task",
                    {
                        "task_id": self.task_id,
                        "expected_revision": 1,
                        "delta": user_text,
                    },
                )
            elif turn in {3, 5} and call_index == 1:
                self.routes.append("task_status")
                yield ToolCall(f"voice-status-{turn}", "task_status", {"task_id": self.task_id})
            elif turn == 4 and call_index == 1:
                self.routes.append("direct_contextual_answer")
                yield "It has About, Projects, and Contact sections. The dark styling is in portfolio/assets/site.css."
            else:
                replies = {
                    1: "I started the portfolio task in the background.",
                    2: "I revised the active task; the dark theme and added project are being applied.",
                    3: "The portfolio task is complete and its files passed readback verification.",
                    5: "The verified dark theme is in portfolio/assets/site.css.",
                }
                yield replies[turn]

    task_model = WebsiteTaskModel()
    router = VoiceRouter()
    specs = load_backends()
    transcript = list(scenario["turns"])

    def task_runtime_factory(spec, credentials, budget, research):
        return TaskRuntime(
            spec,
            credentials,
            budget=budget,
            research=research,
            database=tmp_path / "qualification.sqlite",
            artifact_root=tmp_path / "task_workspace",
            model=task_model,
        )

    state = NativeState(
        specs,
        {},
        speech_factory=lambda: SpeechClient(backends=specs, transcript=transcript, timeout=5),
        task_runtime_factory=task_runtime_factory,
    )
    sink = MemorySink(capacity=10000)
    try:
        async with NativeApplication(
            state, sink, annotated=True, upstream=router, options=RuntimeOptions(engine="native")
        ) as app:
            for turn_index in range(5):
                await app.utterance(b"\1\0" * 512)
                if turn_index == 0:
                    task_id = next(iter(state.task_runtime.store.rows))
                    router.task_id = task_id
                    await asyncio.wait_for(task_model_waiting.wait(), 5)
                elif turn_index == 1:
                    release_stale_tool.set()
                    async with asyncio.timeout(10):
                        while (await state.task_runtime.status(router.task_id))["status"] not in {
                            "succeeded",
                            "completed_unverified",
                            "failed",
                        }:
                            await asyncio.sleep(0.01)

            final_status = await state.task_runtime.status(router.task_id)
            workspace = tmp_path / "task_workspace" / router.task_id / "portfolio"
            html_files = {p.name: p.read_text() for p in workspace.glob("*.html")}
            assert set(html_files) == {"index.html", "about.html", "projects.html", "contact.html"}
            for page, html in html_files.items():
                for target in re.findall(r'(?:href|src)="([^\"]+)"', html):
                    if target.startswith(("mailto:", "https://", "http://")):
                        continue
                    assert (workspace / page).parent.joinpath(target).resolve().is_file()
            index = html_files["index.html"]
            assert all(f'id="{name.lower()}"' in index for name in ("About", "Projects", "Contact"))
            assert "Community garden map" in html_files["projects.html"]
            css = (workspace / "assets/site.css").read_text()
            assert "background: #111" in css and "color: #f4f4f4" in css
            assert final_status["status"] == "succeeded"
            assert final_status["revision"] == 2
            assert final_status["verification"]["level"] == "artifact_readback"
            assert final_status["verification"]["artifacts_checked"] == 5
            assert not (
                tmp_path / "task_workspace" / router.task_id / "index.html"
            ).is_file()  # no unrelated root-level duplicate
            assert task_model.calls >= 12  # revised multi-tool plan plus readback
            assert "stale-light-index" not in str(final_status["progress"])
            assert router.routes == scenario["expected_routes"]
            user_turns = [
                message["content"]
                for message in state.context.get_messages()
                if message.get("role") == "user"
            ]
            assert user_turns == scenario["turns"]
            assert sum(event["event"] == "turn_completed" for event in app.events) == 5
            completed_tools = [
                event.get("tool_name")
                for event in app.events
                if event["event"] == "tool_completed" and event.get("ok")
            ]
            assert completed_tools.count("start_task") == 1
            assert completed_tools.count("steer_task") == 1
            assert completed_tools.count("task_status") == 2
            assert any(
                "About, Projects, and Contact" in message.get("content", "")
                for message in state.context.get_messages()
                if message.get("role") == "assistant"
            )
            assert sink.chunks
            assert state.speech.tools["asr"] >= {"asr_open", "asr_append", "asr_finish", "cancel"}
            assert state.speech.tools["tts"] >= {"tts_open", "tts_next", "cancel"}
    finally:
        release_stale_tool.set()
        await state.close()
