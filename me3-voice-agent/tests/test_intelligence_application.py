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
from uuid import uuid4

from langchain_core.messages import AIMessage
from me3_voice.configuration import load_backends, RuntimeOptions
from me3_voice.native_app import NativeApplication, NativeState
from me3_voice.playback import MemorySink
from me3_voice.providers import ToolCall
from me3_voice.research import ResearchClient, ResearchError, ResearchOptions
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
            ToolCall(
                "texas-search",
                "web_search",
                {"query": "Who won the Texas Senate election?", "max_results": 3},
            ),
            ToolCall("texas-fetch", "retrieve_source", {"url": "https://official.example/senate"}),
            "The retrieved official fixture confirms certification on November 8, 2026, but does not name the winner, so I can't verify who won [S1].",
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
        assert results[7]["actual_route"] == "research"
        assert results[7]["search_operations"] == 1 and results[7]["retrieval_operations"] == 1
        assert "November 8, 2026" in results[7]["answer"] and "[S1]" in results[7]["answer"]
        assert "does not name the winner" in results[7]["answer"]
        assert all(row["status"] == "PASS" for row in results)
        assert "Runtime clock:" in app.llm._settings.system_instruction
        assert "2026-11-11T04:00:00+00:00" in app.llm._settings.system_instruction
        assert "latest relevant completed event" in app.llm._settings.system_instruction
        assert "distinguish event" in app.llm._settings.system_instruction
        assert research_calls == {"search": 4, "retrieve": 4}
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


async def test_standalone_michigan_and_nvidia_news_questions_are_brief_and_expand_on_request(
    tmp_path,
):
    """Fresh contexts route two natural current-news questions through retrieval.

    Model choices and retrieval are deterministic fixtures. This verifies the
    native tool/evidence/concise-speech contract, not live Groq routing quality.
    """
    from me3_voice.native_speech import spoken_text

    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "swift_agent_qualification.json").read_text()
    )["standalone_current_research"]
    specs = load_backends()
    options = RuntimeOptions(engine="native", research=ResearchOptions(True, 8))

    class ScriptedResearchModel:
        def __init__(self, steps):
            self.steps = list(steps)
            self.requests = []

        async def respond(self, history):
            self.requests.append(json.loads(json.dumps(history)))
            yield self.steps[len(self.requests) - 1]

    async def run_fresh_case(case, index):
        utterances = [case["utterance"]]
        if case.get("follow_up"):
            utterances.append(case["follow_up"])

        if case["id"] == "michigan_us_senate_winner":
            steps = [
                ToolCall(
                    "mi-search",
                    "web_search",
                    {"query": "2026 Michigan U.S. Senate election winner AP", "max_results": 3},
                ),
                ToolCall(
                    "mi-result",
                    "retrieve_source",
                    {
                        "url": "https://www.michiganpublic.org/live-updates/michigan-election-2026-live-updates"
                    },
                ),
                "Abdul El-Sayed won Michigan’s 2026 U.S. Senate race, defeating Republican Mike Rogers, according to AP [S1].",
            ]
            sources = {
                "https://www.michiganpublic.org/live-updates/michigan-election-2026-live-updates": (
                    "Democrat Abdul El-Sayed wins election to U.S. Senate from Michigan, "
                    "defeating Republican challenger Mike Rogers. The Associated Press called the race."
                )
            }
        else:
            steps = [
                ToolCall(
                    "nv-search",
                    "web_search",
                    {
                        "query": "NVIDIA Jensen Huang CEO resignation current issue",
                        "max_results": 3,
                    },
                ),
                ToolCall(
                    "nv-company",
                    "retrieve_source",
                    {"url": "https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition"},
                ),
                ToolCall(
                    "nv-report",
                    "retrieve_source",
                    {"url": "https://apnews.com/article/4f8d5b5886bf4f097969a3249a45528e"},
                ),
                "NVIDIA confirmed Jensen Huang resigned on November 12; CFO Colette Kress is interim CEO. AP links it to allegations involving a woman accused of Chinese intelligence ties; those remain unproven [S1][S2].",
                "NVIDIA’s announcement confirms Huang’s immediate resignation and Kress’s interim appointment but gives no reason. AP reports the departure followed reports of an alleged affair with a woman accused of Chinese intelligence ties. Beijing denies she is a spy; I found no official confirmation of that allegation [S1][S2].",
            ]
            sources = {
                "https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition": (
                    "NVIDIA announced November 12, 2026 that Jensen Huang resigned as CEO "
                    "effective immediately. CFO Colette Kress will serve as interim CEO."
                ),
                "https://apnews.com/article/4f8d5b5886bf4f097969a3249a45528e": (
                    "The fallout follows Jensen Huang leaving as Nvidia CEO after reports of "
                    "an extramarital affair with a woman described as an alleged agent of the "
                    "Chinese government. Beijing denies that she is a spy."
                ),
            }

        model = ScriptedResearchModel(steps)
        research = ResearchClient(options.research)
        search_calls = []
        retrieval_calls = []

        async def search(query, max_results=5):
            search_calls.append(query)
            return {
                "kind": "untrusted_search_leads",
                "query": query,
                "results": [
                    {"url": url, "title": f"Current source {i + 1}"}
                    for i, url in enumerate(sources)
                ][:max_results],
            }

        async def read(_method, url, **_kwargs):
            retrieval_calls.append(url)
            body = sources[url]
            return {
                "type": "text/html",
                "data": f"<html><main><article><p>{body}</p></article></main></html>".encode(),
            }

        research.search = search
        research.read = read

        def task_runtime_factory(spec, credentials, budget, research):
            return TaskRuntime(
                spec,
                credentials,
                budget=budget,
                research=research,
                database=tmp_path / f"standalone-research-{index}.sqlite",
                artifact_root=tmp_path / f"standalone-research-artifacts-{index}",
            )

        state = NativeState(
            specs,
            {},
            research=research,
            task_runtime_factory=task_runtime_factory,
            speech_factory=lambda: SpeechClient(backends=specs, transcript=utterances),
        )
        sink = MemorySink(capacity=10000)
        async with NativeApplication(
            state,
            sink,
            annotated=True,
            upstream=model,
            options=options,
            clock=lambda: datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc),
        ) as app:
            turns = []
            for utterance in utterances:
                event_start = app.event_count
                await app.utterance(b"\1\0" * 512)
                turns.append(
                    {
                        "utterance": utterance,
                        "answer": _last_assistant(state.context),
                        "events": [e for e in app.events if e["sequence"] > event_start],
                    }
                )

            # The first model request must see this question in a new context,
            # with no preceding user turns that could reveal the expected route.
            first_users = [
                row.get("content")
                for row in model.requests[0]
                if isinstance(row, dict) and row.get("role") == "user"
            ]
            assert first_users == [case["utterance"]]
            assert len(search_calls) == 1
            assert len(retrieval_calls) == len(sources)
            assert "web_search" in [e.get("tool_name") for e in turns[0]["events"]]
            assert "retrieve_source" in [e.get("tool_name") for e in turns[0]["events"]]
            assert not app.error
            system_policy = app.llm._settings.system_instruction.casefold()
            assert "current salient news" in system_policy
            assert "do not turn an allegation into an established fact" in system_policy

            first_answer = turns[0]["answer"]
            normalized = spoken_text(first_answer)
            assert all(
                word.casefold() in first_answer.casefold() for word in case["expected_answer_facts"]
            )
            assert len(first_answer.split()) <= case["max_answer_words"]
            assert "[S" not in normalized and "http" not in normalized
            if case["id"] == "nvidia_ceo_issue":
                assert "allegations" in first_answer.casefold()
                assert "unproven" in first_answer.casefold()
                followup_answer = turns[1]["answer"]
                assert "announcement confirms" in followup_answer
                assert "Beijing denies" in followup_answer
                assert len(followup_answer.split()) > len(first_answer.split())
                assert "web_search" not in [e.get("tool_name") for e in turns[1]["events"]]
        await state.close()

    for index, case in enumerate(cases):
        await run_fresh_case(case, index)


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
            if turn == 1:
                self.routes.append("direct")
                yield "At sea level, water boils at 100 degrees Celsius."
                return
            if turn == 2:
                if call_index == 1:
                    self.routes.append("research")
                    yield ToolCall("search-current", "web_search", {"query": user_text})
                elif call_index == 2:
                    yield ToolCall(
                        "fetch-failed", "retrieve_source", {"url": "https://source.example/failed"}
                    )
                elif call_index == 3:
                    yield ToolCall("search-again", "web_search", {"query": "official dated result"})
                elif call_index == 4:
                    yield ToolCall(
                        "fetch-alternate",
                        "retrieve_source",
                        {"url": "https://source.example/alternate"},
                    )
                else:
                    yield "The dated official source supports the reported result [S1]."
                return
            if turn == 8:
                self.routes.append("interrupted_answer")
                yield "This explanation is intentionally long. " * 12
                return
            if turn == 9:
                self.routes.append("recovery")
                yield "I can continue with the verified portfolio and research context."
                return
            task_turn = turn - 2
            if task_turn == 1 and call_index == 1:
                self.routes.append("start_task")
                yield ToolCall("voice-start-portfolio", "start_task", {"goal": user_text})
            elif task_turn == 2 and call_index == 1:
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
            elif task_turn in {3, 5} and call_index == 1:
                self.routes.append("task_status")
                yield ToolCall(
                    f"voice-status-{task_turn}", "task_status", {"task_id": self.task_id}
                )
            elif task_turn == 4 and call_index == 1:
                self.routes.append("direct_contextual_answer")
                yield "It has About, Projects, and Contact sections. The dark styling is in portfolio/assets/site.css."
            else:
                replies = {
                    1: "I started the portfolio task in the background.",
                    2: "I revised the active task; the dark theme and added project are being applied.",
                    3: "The portfolio task is complete and its files passed readback verification.",
                    5: "The verified dark theme is in portfolio/assets/site.css.",
                }
                yield replies[task_turn]

    task_model = WebsiteTaskModel()
    router = VoiceRouter()
    specs = load_backends()
    transcript = [
        "What is the boiling point of water at sea level?",
        "Who won the recent Senate election?",
        *scenario["turns"],
        "Explain the portfolio in detail.",
        "Continue briefly after my interruption.",
    ]
    research = ResearchClient(ResearchOptions(True, 8))
    searches = []

    async def search(query, max_results=5):
        searches.append(query)
        if len(searches) == 1:
            await asyncio.sleep(0.7)
        result = {
            "kind": "untrusted_search_leads",
            "query": query,
            "results": [
                {"url": "https://source.example/failed", "title": "First source"},
                {"url": "https://source.example/alternate", "title": "Official source"},
            ],
        }
        research._remember_search_leads(result)
        return result

    async def retrieve(url):
        if url.endswith("failed"):
            research.failed_retrieval_urls.add(url)
            raise ResearchError(
                "research_http_failure", status=503, host="source.example", stage="http_status"
            )
        source = {
            "source_id": "S1",
            "url": url,
            "title": "Dated result",
            "text": "The certified Senate result is documented here with the election date and winner.",
            "text_sha256": "a" * 64,
            "evidence_status": "retrieved_unverified",
        }
        research.sources["S1"] = source
        return source

    research.search = search
    research.retrieve = retrieve

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
        research=research,
        speech_factory=lambda: SpeechClient(backends=specs, transcript=transcript, timeout=5),
        task_runtime_factory=task_runtime_factory,
    )
    sink = MemorySink(capacity=10000)
    try:
        cold_started = time.monotonic()
        async with NativeApplication(
            state, sink, annotated=True, upstream=router, options=RuntimeOptions(engine="native")
        ) as app:
            cold_ready_seconds = time.monotonic() - cold_started
            warm_started = time.monotonic()
            assert await state.start() is state.speech
            warm_reuse_seconds = time.monotonic() - warm_started
            print(
                json.dumps(
                    {
                        "kind": "offline_mock_native_startup",
                        "cold_ready_seconds": cold_ready_seconds,
                        "warm_reuse_seconds": warm_reuse_seconds,
                        "provider_evidence": False,
                    }
                )
            )
            for turn_index in range(7):
                await app.utterance(b"\1\0" * 512)
                if turn_index == 2:
                    task_id = next(iter(state.task_runtime.store.rows))
                    router.task_id = task_id
                    await asyncio.wait_for(task_model_waiting.wait(), 5)
                elif turn_index == 3:
                    release_stale_tool.set()
                    async with asyncio.timeout(10):
                        while (await state.task_runtime.status(router.task_id))["status"] not in {
                            "succeeded",
                            "completed_unverified",
                            "failed",
                        }:
                            await asyncio.sleep(0.01)

            await app.utterance(b"\1\0" * 512, wait=False)
            async with asyncio.timeout(5):
                while not any(
                    event["event"] == "tts_start" and event.get("turn_id") == 8
                    for event in app.events
                ):
                    await asyncio.sleep(0.001)
            await app.interrupt()
            await app.utterance(b"\1\0" * 512)

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
            assert router.routes == [
                "direct",
                "research",
                *scenario["expected_routes"],
                "interrupted_answer",
                "recovery",
            ]
            assert len(searches) == 1
            assert "S1" in research.sources
            assert (
                sum(event["event"] == "research_acknowledgment_queued" for event in app.events) == 1
            )
            events = list(app.events)
            progress_positions = [
                i for i, row in enumerate(events) if row["event"] == "tts_progress_started"
            ]
            answer_at = next(
                i
                for i, row in enumerate(events)
                if row["event"] == "tts_start" and row["turn_id"] == 2 and not row.get("owner_id")
            )
            assert all(position < answer_at for position in progress_positions)
            assert progress_positions or any(
                row["event"] == "tts_progress_suppressed" for row in events
            )
            user_turns = [
                message["content"]
                for message in state.context.get_messages()
                if message.get("role") == "user"
            ]
            assert user_turns == transcript
            assert sum(event["event"] == "turn_completed" for event in app.events) == 8
            assert {row["turn_id"] for row in app.events if row["event"] == "turn_completed"} == {
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                9,
            }
            assert any(event["event"] == "interrupt_requested" for event in app.events)
            assert any(
                row["response_state"] in {"interrupted", "superseded"}
                for row in app.response_records.values()
            )
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


async def test_search_keeps_retrieval_model_owned_and_oversized_source_falls_back(tmp_path):
    """Search leads stay bounded; one model-selected fetch retains verifiable evidence."""
    from me3_voice.conversation_context import POLICY

    policy = " ".join(POLICY.split())
    assert "cite each externally verifiable factual claim" in policy
    assert "A retrieved source without a citation does not support the spoken answer" in policy

    class SearchThenAnswer:
        def __init__(self):
            self.requests = []

        async def respond(self, history):
            self.requests.append(json.loads(json.dumps(history)))
            if len(self.requests) == 1:
                yield ToolCall(
                    "current-results-search",
                    "web_search",
                    {"query": "Michigan Senate election result", "max_results": 5},
                )
            elif len(self.requests) == 2:
                yield ToolCall(
                    "retrieve-oversized-result",
                    "retrieve_source",
                    {
                        "url": "https://www.nbcnews.com/politics/2026-elections/michigan-senate-results"
                    },
                )
            elif len(self.requests) == 3:
                yield ToolCall(
                    "retrieve-alternative-result",
                    "retrieve_source",
                    {"url": "https://www.michigan.gov/elections/results/senate"},
                )
            elif len(self.requests) == 4:
                yield ToolCall(
                    "verify-result-quote",
                    "verify_evidence",
                    {
                        "source_id": "S1",
                        "quote": "The certified Michigan Senate result names Candidate Green as the winner.",
                    },
                )
            else:
                yield "The certified Michigan result names Candidate Green as the winner [S1]."

    options = RuntimeOptions(engine="native", research=ResearchOptions(True, 5))
    specs = load_backends()
    model = SearchThenAnswer()
    research = ResearchClient(
        options.research, {"ME3_RESEARCH_TAVILY_API_KEY": "offline-test-token"}
    )
    # An older conversation source must not prevent retrieval for this new query.
    research.sources["S0"] = {
        "source_id": "S0",
        "url": "https://archive.example/old",
        "text": "An unrelated earlier source.",
    }
    operations = []

    async def search(query, max_results=5):
        research.authorize()
        operations.append(("search", query))
        result = {
            "kind": "untrusted_search_leads",
            "query": query,
            "results": [
                {
                    "url": "https://www.nbcnews.com/politics/2026-elections/michigan-senate-results",
                    "title": "Michigan Senate results",
                    "snippet": "Election results for the Michigan Senate race.",
                    "evidence_status": "discovery_only",
                },
                {
                    "url": "https://www.michigan.gov/elections/results/senate",
                    "title": "Michigan Secretary of State certified results",
                    "snippet": "Certified statewide Michigan Senate election results.",
                    "evidence_status": "discovery_only",
                },
            ][:max_results],
        }
        research._remember_search_leads(result)
        return result

    async def read(method, url, **_kwargs):
        research.authorize()
        operations.append((method, url))
        if method == "GET" and url.startswith("https://www.nbcnews.com/"):
            raise ResearchError(
                "source_too_large",
                host="www.nbcnews.com",
                stage="response_body",
                failure_class="body_limit",
            )
        if method == "POST":
            return {
                "data": json.dumps(
                    {
                        "results": [
                            {
                                "url": url,
                                "raw_content": "Content unavailable.",
                            }
                        ]
                    }
                ).encode(),
                "type": "application/json",
            }
        return {
            "type": "text/html",
            "data": (
                b"<html><main><article><h1>Michigan election results</h1><p>"
                b"The certified Michigan Senate result names Candidate Green as the winner. "
                b"This final statewide result was published by the Michigan Secretary of State. "
                b"The state election office certified the outcome after the final count."
                b"</p></article></main></html>"
            ),
        }

    research.search = search
    research.read = read

    def task_runtime_factory(spec, credentials, budget, research):
        return TaskRuntime(
            spec,
            credentials,
            budget=budget,
            research=research,
            database=tmp_path / "model-retrieval.sqlite",
            artifact_root=tmp_path / "model-retrieval-workspace",
        )

    state = NativeState(
        specs,
        {},
        research=research,
        task_runtime_factory=task_runtime_factory,
        speech_factory=lambda: SpeechClient(
            backends=specs, transcript=["Who won the Michigan Senate election?"]
        ),
    )
    sink = MemorySink(capacity=10000)
    try:
        async with NativeApplication(
            state,
            sink,
            annotated=True,
            upstream=model,
            options=options,
        ) as app:
            await app.utterance(b"\1\0" * 512)
            answer = _last_assistant(state.context)
            events = app.events
            search_event = next(
                event
                for event in events
                if event["event"] == "tool_completed" and event.get("tool_name") == "web_search"
            )
            search_result_json = json.dumps(model.requests[1])
            assert "discovery_only" in search_result_json
            assert "retrieved_source" not in search_result_json
            failed_fetch_json = json.dumps(model.requests[2])
            assert "source_too_large" in failed_fetch_json
            assert "alternative_leads" in failed_fetch_json
            assert "www.michigan.gov/elections/results/senate" in failed_fetch_json

            def tool_result(request, call_prefix):
                message = next(
                    item
                    for item in request
                    if item.get("role") == "tool"
                    and item.get("tool_call_id", "").startswith(call_prefix)
                )
                content = message["content"]
                return json.loads(content) if isinstance(content, str) else content

            retrieved = tool_result(model.requests[3], "retrieve-alternative-result")
            assert retrieved["url"] == "https://www.michigan.gov/elections/results/senate"
            assert retrieved["citation"] == "[S1]"
            assert retrieved["extraction_method"] == "local_html"
            assert retrieved["evidence_status"] == "retrieved_unverified"
            assert "Candidate Green" in retrieved["text"]
            assert len(retrieved["text"]) == 248
            verification = tool_result(model.requests[4], "verify-result-quote")
            assert verification["quote_found"] is True
            assert verification["claim_truth_verified"] is False
            assert operations == [
                ("search", "Michigan Senate election result"),
                ("GET", "https://www.nbcnews.com/politics/2026-elections/michigan-senate-results"),
                ("POST", "https://api.tavily.com/extract"),
                ("GET", "https://www.michigan.gov/elections/results/senate"),
            ]
            assert search_event["retained_source_count"] == 1
            retrieved_event = next(
                event
                for event in events
                if event["event"] == "tool_completed"
                and event.get("tool_name") == "retrieve_source"
                and event.get("ok")
            )
            evidence_diagnostics = retrieved_event["diagnostics"]
            assert evidence_diagnostics["source_id"] == "S1"
            assert evidence_diagnostics["evidence_characters"] == len(retrieved["text"])
            assert evidence_diagnostics["source_url_hash"]
            assert "text" not in json.dumps(evidence_diagnostics)
            assert research.requests == 4
            assert (
                research.sources["S1"]["url"] == "https://www.michigan.gov/elections/results/senate"
            )
            assert research.sources["S1"]["text_sha256"]
            assert len(research.sources) == 2  # Pre-existing context evidence is preserved.
            assert "[S1]" in answer and "Candidate Green" in answer
            assert any(
                event["event"] == "tool_completed" and event.get("tool_name") == "verify_evidence"
                for event in events
            )
            assert len(model.requests) == 5
            assert any(event["event"] == "llm_grounding_check" for event in events)
            citation_check = next(
                event for event in events if event["event"] == "llm_citation_check"
            )
            assert citation_check["citation_ids"] == ["S1"]
            assert citation_check["retained_source_ids"] == ["S0", "S1"]
            assert any(event["event"] == "turn_completed" for event in events)
            assert sink.chunks
            assert not app.error

            # Exercise the existing native JSONL journal, not just in-memory
            # application events or the projection helper in isolation.
            from me3_voice.native_bridge import NativeEventJournal

            journal_run_id = f"pytest-evidence-{uuid4().hex}"
            journal = NativeEventJournal("var/results", journal_run_id)
            journal_path = Path("var/results") / f"native-{journal_run_id}.jsonl"
            try:
                diagnostic_events = [
                    event
                    for event in events
                    if event["event"] == "llm_citation_check"
                    or (
                        event["event"] == "tool_completed"
                        and event.get("tool_name") == "retrieve_source"
                        and event.get("ok")
                    )
                ]
                journal.write_new(diagnostic_events)
                journal.close()
                persisted = [json.loads(line) for line in journal_path.read_text().splitlines()]
                assert [row["event"] for row in persisted] == [
                    "tool_completed",
                    "llm_citation_check",
                ]
                assert persisted[0]["sequence"] < persisted[1]["sequence"]
                persisted_source, persisted_citations = persisted
                assert persisted_source["research_source_id"] == "S1"
                assert (
                    persisted_source["research_source_url_hash"]
                    == evidence_diagnostics["source_url_hash"]
                )
                assert persisted_source["research_extraction_method"] == "local_html"
                assert persisted_source["research_evidence_characters"] == len(retrieved["text"])
                assert persisted_source["research_evidence_status"] == "retrieved_unverified"
                assert persisted_source["research_evidence_chunks"] == retrieved.get(
                    "evidence_chunks", 0
                )
                assert persisted_citations["citation_ids"] == ["S1"]
                assert persisted_citations["retained_source_ids"] == ["S0", "S1"]
                assert "Candidate Green" not in journal_path.read_text()
                assert "private" not in journal_path.read_text()
            finally:
                if not journal.stream.closed:
                    journal.close()
                journal_path.unlink(missing_ok=True)
    finally:
        await state.close()
