"""Connected Swift task ownership with the optional isolated Onit worker."""

import asyncio
import json
from pathlib import Path

import pytest

from me3_voice.configuration import BackendSpec, ConfigurationError
from me3_voice.onit_tasks import OnitPrimaryReasoning, OnitTaskExecutor, OnitTaskUnavailable
from me3_voice.onit_worker import (
    ArtifactTool,
    GitHubReadTool,
    ProjectDirectoryTool,
    TavilySearchTool,
)
from me3_voice.providers import MockLLM, ToolCall
from me3_voice.remote import ChatReasoning
from me3_voice.session import Session
from me3_voice.telemetry import summarize
from me3_voice.telemetry import validate as validate_trace_event
from me3_voice.transcript import Transcript


def spec():
    return BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        auth_env="ME3_LLM_API_KEY",
        model="nvidia/nemotron-3.5-lightning:free",
        allow_live=True,
        request_limit=3,
    )


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        yield b"mock-tone"


async def sink(pcm):
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_agent_task_flows_through_session_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var").mkdir()
    python = Path("/usr/bin/python3").resolve()
    if not python.is_file():
        pytest.skip("system Python absent")
    seen = []

    async def runner(task, directory, budget):
        seen.append((task, directory))
        await asyncio.sleep(0.04)
        (directory / "index.html").write_text("<main>Swift draft</main>")
        return {
            "mode": "local-artifact",
            "summary": "Created 1 draft file in the isolated task workspace; nothing was deployed.",
            "artifacts": ["index.html"],
            "requests": 2,
        }

    executor = OnitTaskExecutor(
        python, spec(), {"ME3_LLM_API_KEY": "offline-dummy"}, max_requests=3, runner=runner
    )
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, tasks=executor, transcript=display) as session:
        session.activate()
        await (await session.submit(b"agent task: create a tiny website"))
        assert len(seen) == 1
        receipt = display.snapshot()["tasks"][0]
        assert receipt["kind"] == "agent"
        assert receipt["mode"] == "reported"
        assert receipt["status"] == "reported"
        assert "nothing was deployed" in receipt["summary"]
        assert receipt["execution_target"] == "HPC task workspace"
        assert Path(receipt["absolute_path"]).name == receipt["artifact_id"]
        assert Path(receipt["absolute_path"]).parent == tmp_path / "var/scratch/onit-tasks"
        assert "not executed" in receipt["verification_level"]
        call = ToolCall("agent-task", "run_agent_task", {"task": "create a tiny website"})
        first = await executor.execute(
            type("K", (), {"session_id": session.session_id, "turn_id": 1})(), call
        )
        assert first["artifacts"] == ["index.html"] and first["task_id"] == seen[0][1].name
        assert (
            await executor.read_artifact(first["task_id"], "index.html")
            == b"<main>Swift draft</main>"
        )
        outside = tmp_path / "outside.html"
        outside.write_text("<script>outside</script>")
        artifact_path = seen[0][1] / "index.html"
        artifact_path.unlink()
        artifact_path.symlink_to(outside)
        with pytest.raises(FileNotFoundError):
            await executor.read_artifact(first["task_id"], "index.html")
        with pytest.raises(FileNotFoundError):
            await executor.read_artifact(first["task_id"], "../secret.md")
        assert len(seen) == 1
        assert any(e["event"] == "tool_completed" for e in session.events)
        stage = next(row for row in summarize(session.events)["turns"] if row["turn_id"] == 1)
        assert stage["durations_ms"]["tool_duration_ms"][0] >= 35
        assert stage["durations_ms"]["llm_request_total_ms"][0] < 35


@pytest.mark.asyncio
async def test_onit_task_cancellation_keeps_one_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var").mkdir()
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def runner(task, directory, budget):
        nonlocal count
        count += 1
        entered.set()
        await release.wait()
        return {"mode": "local-artifact", "summary": "Done.", "artifacts": [], "requests": 2}

    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline-dummy"},
        max_requests=3,
        runner=runner,
    )
    key = type("K", (), {"session_id": "session", "turn_id": 1})()
    call = ToolCall("task-1", "run_agent_task", {"task": "make a draft"})
    first = asyncio.create_task(executor.execute(key, call))
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(executor.execute(key, call))
    release.set()
    assert (await second)["summary"] == "Done."
    assert count == 1
    with pytest.raises(ValueError, match="different task"):
        await executor.execute(key, ToolCall("task-1", "run_agent_task", {"task": "different"}))
    await executor.close()


@pytest.mark.asyncio
async def test_busy_onit_task_is_reported_as_unavailable_without_false_quota_error(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var").mkdir()
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def runner(task, directory, budget):
        nonlocal count
        count += 1
        entered.set()
        await release.wait()
        return {
            "mode": "local-answer",
            "summary": "The first task completed.",
            "answer": "The first task completed.",
            "artifacts": [],
            "requests": 1,
        }

    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline-dummy"},
        max_requests=None,
        runner=runner,
    )
    key = type("K", (), {"session_id": "session", "turn_id": 1})()
    first_call = ToolCall("task-first", "run_agent_task", {"task": "first request"})
    first = asyncio.create_task(executor.execute(key, first_call))
    await entered.wait()

    with pytest.raises(OnitTaskUnavailable) as caught:
        await executor.execute(
            type("K", (), {"session_id": "session", "turn_id": 2})(),
            ToolCall("task-second", "run_agent_task", {"task": "second request"}),
        )
    assert caught.value.code == "task_busy"
    assert caught.value.task_status == "unavailable"
    assert "still running" in caught.value.user_summary
    assert "request allowance is exhausted" not in caught.value.user_summary
    assert count == 1
    release.set()
    await first
    await executor.close()


def test_onit_validates_optional_budget_and_project_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ME3_LLM_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="request limit"):
        OnitTaskExecutor(
            Path("/usr/bin/python3").resolve(),
            spec(),
            {"ME3_LLM_API_KEY": "dummy"},
            root=tmp_path / "var",
            max_requests=0,
        )
    with pytest.raises(ConfigurationError, match="credential"):
        OnitTaskExecutor(
            Path("/usr/bin/python3").resolve(), spec(), {}, root=tmp_path / "var", max_requests=1
        )


@pytest.mark.asyncio
async def test_onit_primary_routes_current_turn_with_context():
    provider = OnitPrimaryReasoning()
    items = [
        item
        async for item in provider.respond(
            [
                {"role": "user", "content": "What is the time in Manila?", "turn_id": 1},
                {"role": "assistant", "content": "It is noon.", "turn_id": 1},
                {"role": "user", "content": "And in Tokyo?", "turn_id": 2},
            ]
        )
    ]
    assert len(items) == 1
    assert items[0].name == "run_agent_task"
    task = items[0].arguments["task"]
    assert "What is the time in Manila?" in task
    assert "And in Tokyo?" in task
    assert items[0].call_id == "onit-turn-3"


@pytest.mark.asyncio
async def test_dgx_onit_contract_allows_provider_enforced_request_limit(tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    (tmp_path / "var").mkdir()
    dgx = replace(
        spec(),
        endpoint="http://202.92.159.240:8001/v1/chat/completions",
        model="Qwen/Qwen3.8-27B",
        request_limit=None,
        allow_private_http=True,
    )
    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        dgx,
        {"ME3_LLM_API_KEY": "offline-dummy"},
        runner=lambda *_: None,
    )
    assert executor.max_requests is None
    assert executor.remaining_requests is None
    await executor.close()


@pytest.mark.asyncio
async def test_onit_primary_uses_session_tool_owner_without_second_llm_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var").mkdir()
    calls = []

    async def runner(task, directory, budget):
        calls.append((task, budget))
        return {
            "mode": "local-answer",
            "summary": "Onit answered the request; no external action was reported.",
            "answer": "Manila is the capital of the Philippines.",
            "artifacts": [],
            "requests": 1,
        }

    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline-dummy"},
        max_requests=None,
        runner=runner,
    )
    async with Session(Speech(), OnitPrimaryReasoning(), sink, tasks=executor) as session:
        session.activate()
        await await_result(session.submit(b"What is the capital of the Philippines?"))
        assert len(calls) == 1
        assert calls[0][1] is None
        assert session.history[-1]["content"] == "Manila is the capital of the Philippines."
        assert not any(event["event"] == "llm_request_started" for event in session.events)
        assert any(event["event"] == "tool_completed" for event in session.events)


async def await_result(pending):
    generation = await pending
    await generation


def test_onit_answer_receipt_is_spoken_without_second_reasoning_call():
    call = ToolCall("onit-turn-1", "run_agent_task", {"task": "answer"})
    receipt = {"mode": "local-answer", "answer": "A concise answer."}
    assert Session.task_confirmation(call, receipt) == "A concise answer."


def test_onit_latency_event_is_safe_for_diagnostic_trace():
    event = {
        "event": "onit_model_completed",
        "monotonic": 1.0,
        "session_id": "session-id",
        "turn_id": 1,
        "mode": "HPC-live",
        "run_id": "run-id",
        "clock_id": "monotonic:run-id",
        "schema_version": 1,
        "state": "thinking",
        "onit_requests": 1,
        "onit_worker_ms": 1500.0,
        "onit_model_ms": 1200.0,
        "onit_ttft_ms": 400.0,
        "onit_completion_tokens": 36,
    }
    assert validate_trace_event(event) == event
    with pytest.raises(ValueError, match="diagnostic fields"):
        validate_trace_event({**event, "answer": "secret content"})


@pytest.mark.asyncio
async def test_agent_artifact_tool_stays_inside_workspace(tmp_path):
    completed = asyncio.Event()
    tool = ArtifactTool(tmp_path, completed=completed)
    assert "Rejected" in await tool(filename="../escape.md", content="bad")
    assert "Rejected" in await tool(filename="run.py", content="bad")
    assert "created" in await tool(filename="index.html", content="<h1>Safe</h1>")
    assert completed.is_set()
    assert "already exists" in await tool(filename="index.html", content="different")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.html"]
    assert (tmp_path / "index.html").read_text() == "<h1>Safe</h1>"


@pytest.mark.asyncio
async def test_project_directory_tool_creates_only_one_new_child(tmp_path):
    completed = asyncio.Event()
    tool = ProjectDirectoryTool(tmp_path, completed=completed)
    assert "Rejected" in await tool(name="../escape")
    assert "Rejected" in await tool(name=".hidden")
    assert "Created" in await tool(name="ME3-Test")
    assert completed.is_set()
    assert (tmp_path / "ME3-Test").is_dir()
    assert "Rejected" in await tool(name="AnotherProject")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ME3-Test"]
    other = ProjectDirectoryTool(tmp_path)
    assert "Created" in await other(name="AnotherProject")
    assert (tmp_path / "AnotherProject").is_dir()


@pytest.mark.asyncio
async def test_project_directory_tool_creates_bounded_static_site_without_overwrite(tmp_path):
    completed = asyncio.Event()
    tool = ProjectDirectoryTool(tmp_path, completed=completed)
    files = [
        {"filename": "index.html", "content": "<!doctype html><h1>Swift</h1>"},
        {"filename": "styles.css", "content": "body { color: #172b4d; }"},
        {"filename": "script.js", "content": "document.title = 'Swift';"},
    ]
    assert "Rejected" in await tool(
        name="UnsafeSite", files=[{"filename": "../escape.html", "content": "bad"}]
    )
    assert not (tmp_path / "UnsafeSite").exists()
    assert "Rejected" in await tool(name="MissingIndex", files=files[1:])
    assert not (tmp_path / "MissingIndex").exists()
    assert "static website project SwiftSite with 3 files" in await tool(
        name="SwiftSite", files=files
    )
    assert completed.is_set()
    site = tmp_path / "SwiftSite"
    assert sorted(path.name for path in site.iterdir()) == [
        "index.html",
        "script.js",
        "styles.css",
    ]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in site.iterdir())
    assert (site / "index.html").read_text() == files[0]["content"]
    assert tool.artifacts == ["index.html", "styles.css", "script.js"]
    assert "Rejected" in await tool(name="OtherSite", files=files)
    assert not (tmp_path / "OtherSite").exists()


def test_project_output_root_must_be_explicit_sandbox_parent(tmp_path, monkeypatch):
    project = tmp_path / "ME3-VoiceAgent"
    project.mkdir()
    (project / "var").mkdir()
    monkeypatch.chdir(project)
    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline-dummy"},
        project_output_root=tmp_path,
        max_requests=3,
        runner=lambda *_: None,
    )
    assert executor.project_output_root == tmp_path
    with pytest.raises(ConfigurationError, match="parent"):
        OnitTaskExecutor(
            Path("/usr/bin/python3").resolve(),
            spec(),
            {"ME3_LLM_API_KEY": "offline-dummy"},
            project_output_root=project,
            max_requests=3,
            runner=lambda *_: None,
        )


@pytest.mark.asyncio
async def test_tavily_search_uses_one_basic_credit_and_redacts_credential():
    import sys
    from types import SimpleNamespace

    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "answer": "Current conditions are reported here.",
                "usage": {"credits": 1},
                "results": [
                    {
                        "title": "Weather report",
                        "url": "https://example.test/weather",
                        "content": "A concise weather result.",
                        "published_date": "2026-11-01",
                    }
                ],
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, *, headers, json):
            calls.append((url, headers, json))
            assert url == "https://api.tavily.com/search"
            assert headers["Authorization"] == "Bearer secret-test-value"
            payload = json
            assert payload["search_depth"] == "basic"
            assert payload["include_raw_content"] is False
            return Response()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(
        sys.modules, "httpx", SimpleNamespace(AsyncClient=object, RequestError=OSError)
    )
    tool = TavilySearchTool("secret-test-value", 1, client_factory=lambda **_: Client())
    result = await tool(query="today weather in Manila")
    assert json.loads(result)["credits_used"] == 1
    assert "secret-test-value" not in result
    assert len(calls) == 1
    assert "allowance is exhausted" in await tool(query="another search")
    assert len(calls) == 1
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_github_read_tool_only_exposes_get_and_list():
    import sys
    from types import SimpleNamespace

    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "name": "swift-demo",
                "full_name": "example/swift-demo",
                "html_url": "https://github.com/example/swift-demo",
                "private": False,
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, *, headers, params):
            calls.append((url, headers, params))
            assert url == "https://api.github.com/repos/example/swift-demo"
            assert headers["Authorization"] == "Bearer github-test-value"
            return Response()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(
        sys.modules, "httpx", SimpleNamespace(AsyncClient=object, RequestError=OSError)
    )
    tool = GitHubReadTool("github-test-value", client_factory=lambda **_: Client())
    assert "Rejected" in await tool(action="delete", name="example/swift-demo")
    result = json.loads(await tool(action="get", name="example/swift-demo"))
    assert result["repository"]["full_name"] == "example/swift-demo"
    assert len(calls) == 1
    monkeypatch.undo()


def test_onit_optional_integrations_require_their_keys_and_explicit_tavily_budget(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ME3_LLM_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="Tavily search requires"):
        OnitTaskExecutor(
            Path("/usr/bin/python3").resolve(),
            spec(),
            {"ME3_LLM_API_KEY": "dummy"},
            max_requests=1,
            tavily_max_credits=1,
            root=tmp_path / "var",
            runner=lambda *_: None,
        )
    with pytest.raises(ConfigurationError, match="GitHub read access requires"):
        OnitTaskExecutor(
            Path("/usr/bin/python3").resolve(),
            spec(),
            {"ME3_LLM_API_KEY": "dummy"},
            max_requests=1,
            github_readonly=True,
            root=tmp_path / "var-github",
            runner=lambda *_: None,
        )


def test_onit_key_setup_is_masked_and_status_never_prints_values(tmp_path, monkeypatch, capsys):
    from me3_voice import credentials, onit_setup

    cache = tmp_path / "var" / "cache"
    monkeypatch.setattr(credentials, "CACHE_ROOT", cache)
    path = cache / "onit-secrets.env"
    supplied = iter(["gh-test-secret", "tvly-test-secret", "hf-test-secret"])
    onit_setup.configure(path, prompt=lambda _label: next(supplied))
    output = capsys.readouterr().out
    assert "Saved 3 key(s)" in output
    assert "gh-test-secret" not in output
    assert path.stat().st_mode & 0o777 == 0o600
    onit_setup.status(path)
    output = capsys.readouterr().out
    assert "GitHub token (read-only repository lookup): configured" in output
    assert "tvly-test-secret" not in output and "hf-test-secret" not in output
    with pytest.raises(ConfigurationError, match="never overwritten"):
        onit_setup.configure(path, prompt=lambda _label: "replacement-secret")


@pytest.mark.asyncio
async def test_project_directory_receipt_reaches_spoken_response(tmp_path, monkeypatch):
    project = tmp_path / "ME3-VoiceAgent"
    project.mkdir()
    (project / "var").mkdir()
    monkeypatch.chdir(project)
    calls = []

    async def runner(task, directory, budget):
        calls.append(task)
        (tmp_path / "ME3-Test").mkdir()
        return {
            "mode": "local-project",
            "summary": "Created empty project directory ME3-Test under the configured test-output root.",
            "artifacts": [],
            "projects": ["ME3-Test"],
            "requests": 2,
        }

    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline-dummy"},
        project_output_root=tmp_path,
        max_requests=3,
        runner=runner,
    )
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, tasks=executor, transcript=display) as session:
        session.activate()
        await (await session.submit(b"agent task: create ME3-Test project directory"))
        assert calls == ["create ME3-Test project directory"]
        assert (tmp_path / "ME3-Test").is_dir()
        receipt = display.snapshot()["tasks"][0]
        assert receipt["status"] == "reported"
        assert receipt["execution_target"] == "Configured sandbox"
        assert receipt["absolute_path"] == str(tmp_path / "ME3-Test")
        assert "New empty directory exists" in receipt["verification_level"]
        assert session.history[-1]["content"] == "I created the empty project directory ME3-Test."


@pytest.mark.asyncio
@pytest.mark.parametrize("var_target", [False, True])
async def test_static_site_receipt_and_files_reach_session_and_safe_reader(
    tmp_path, monkeypatch, var_target
):
    project = tmp_path / "ME3-VoiceAgent"
    project.mkdir()
    sandbox = tmp_path
    (project / "var").mkdir()
    if var_target:
        sandbox = project / "var/project-results"
        sandbox.mkdir()
    monkeypatch.chdir(project)
    calls = []

    async def runner(task, workspace, budget):
        calls.append((task, budget))
        site = sandbox / "SwiftSite"
        site.mkdir(mode=0o700)
        (site / "index.html").write_text("<!doctype html><h1>Swift</h1>")
        (site / "styles.css").write_text("body { color: #172b4d; }")
        return {
            "mode": "local-project-site",
            "summary": "Created static website project SwiftSite with 2 files; not run or deployed.",
            "artifacts": ["index.html", "styles.css"],
            "projects": ["SwiftSite"],
            "answer": "The requested action completed and its result was verified.",
            "requests": 1,
            "tavily_credits_used": 0,
            "github_reads_used": 0,
        }

    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline-dummy"},
        project_output_root=sandbox,
        runner=runner,
    )
    display = Transcript()
    async with Session(
        Speech(), OnitPrimaryReasoning(), sink, tasks=executor, transcript=display
    ) as session:
        session.activate()
        await await_result(session.submit(b"Create a small website called SwiftSite."))
        receipt = display.snapshot()["tasks"][0]
        assert len(calls) == 1
        assert receipt["status"] == "reported"
        assert receipt["artifacts"] == ["index.html", "styles.css"]
        assert receipt["execution_target"] == "Configured sandbox"
        assert receipt["absolute_path"] == str(sandbox / "SwiftSite")
        assert "not run" in receipt["verification_level"]
        assert session.history[-1]["content"] == (
            "The static website SwiftSite is ready with 2 files. It has not been run or deployed."
        )
        task_id = receipt["artifact_id"]
        assert (
            await executor.read_artifact(task_id, "index.html") == b"<!doctype html><h1>Swift</h1>"
        )
        outside = tmp_path / "outside.css"
        outside.write_text("unsafe")
        (sandbox / "SwiftSite" / "styles.css").unlink()
        (sandbox / "SwiftSite" / "styles.css").symlink_to(outside)
        with pytest.raises(FileNotFoundError):
            await executor.read_artifact(task_id, "styles.css")
    await executor.close()


@pytest.mark.asyncio
async def test_chat_reasoning_advertises_agent_task_only_when_enabled():
    class Transport:
        body = None

        async def events(self, body, *, total_timeout=True):
            self.body = body
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "draft-1",
                                    "function": {
                                        "name": "run_agent_task",
                                        "arguments": '{"task":"create a draft site"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

    transport = Transport()
    provider = ChatReasoning(spec(), transport, agent_enabled=True)
    result = [item async for item in provider.respond([{"role": "user", "content": "Make a site"}])]
    assert result == [ToolCall("draft-1", "run_agent_task", {"task": "create a draft site"})]
    assert "run_agent_task" in {item["function"]["name"] for item in transport.body["tools"]}
    assert transport.body["provider"]["max_price"] == {"prompt": 0, "completion": 0}

    directory_transport = Transport()
    directory_provider = ChatReasoning(
        spec(), directory_transport, agent_enabled=True, agent_directory_enabled=True
    )
    [
        item
        async for item in directory_provider.respond(
            [{"role": "user", "content": "Make a directory"}]
        )
    ]
    directory_tool = next(
        item["function"]
        for item in directory_transport.body["tools"]
        if item["function"]["name"] == "run_agent_task"
    )
    assert "test-project directory" in directory_tool["description"]

    ordinary = ChatReasoning(spec(), Transport())
    with pytest.raises(Exception, match="unadvertised reasoning tool name"):
        [item async for item in ordinary.respond([{"role": "user", "content": "Make a site"}])]


@pytest.mark.asyncio
async def test_late_onit_output_settles_original_receipt_without_new_speech_or_duplicate_file(
    tmp_path, monkeypatch
):
    from me3_voice.contracts import TurnKey

    monkeypatch.chdir(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []
    audio = []

    async def runner(task, directory, budget):
        entered.set()
        await release.wait()
        (directory / "index.html").write_text("<h1>Retained result</h1>")
        writes.append(directory / "index.html")
        return {
            "mode": "local-artifact",
            "summary": "One draft file created.",
            "artifacts": ["index.html"],
            "requests": 1,
        }

    async def capture(pcm):
        audio.append(pcm)

    executor = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline"},
        runner=runner,
        max_requests=3,
    )
    display = Transcript()
    async with Session(Speech(), MockLLM(), capture, tasks=executor, transcript=display) as session:
        session.activate()
        old = await session.submit(b"agent task: create one file")
        await entered.wait()
        original_id = session.session_id
        await session.interrupt()
        await asyncio.gather(old, return_exceptions=True)
        assert display.snapshot()["tasks"][0]["status"] == "unknown"
        await (await session.submit(b"Turn on the lights."))
        newer = display.snapshot()["tasks"][-1].copy()
        before = len(audio)
        release.set()
        await asyncio.gather(*(j for _, j in executor.operations.values()))
        await asyncio.sleep(0)
        rows = display.snapshot()["tasks"]
        assert rows[0]["status"] == "reported"
        assert rows[-1] == newer
        assert len(audio) == before and len(writes) == 1
        operation, (request, job) = next(iter(executor.operations.items()))
        again = await executor.execute(
            TurnKey(session_id=original_id, turn_id=1),
            ToolCall(operation.rsplit(":", 1)[1], "run_agent_task", {"task": request}),
        )
        assert len(writes) == 1 and len(executor.operations) == 1
        assert (
            await executor.read_artifact(again["task_id"], "index.html")
            == b"<h1>Retained result</h1>"
        )
        assert session.playback.queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_empty_worker_output_fails_truthfully_then_normal_turn_recovers(
    tmp_path, monkeypatch, legacy
):
    import sys

    monkeypatch.chdir(tmp_path)
    executor = OnitTaskExecutor(
        Path(sys.executable), spec(), {"ME3_LLM_API_KEY": "offline"}, max_requests=3
    )
    children = []

    async def start():
        result = (
            {
                "mode": "local-answer",
                "answer": "",
                "summary": "Onit answered the request.",
                "requests": 1,
            }
            if legacy
            else {"error": "NoOutput", "requests": 1}
        )
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-c",
            'import sys,json; print("READY",flush=True); result='
            + repr(result)
            + "; [(print(json.dumps(result),flush=True)) for line in sys.stdin]",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert await child.stdout.readline() == b"READY\n"
        executor.child = child
        children.append(child)
        return child

    executor._ensure_worker = start
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, tasks=executor, transcript=display) as s:
        s.activate()
        await (await s.submit(b"agent task: create a file"))
        row = display.snapshot()["tasks"][0]
        assert row["status"] == "failed" and "no answer or confirmed output" in row["summary"]
        assert any(e.get("error_code") == "onit_no_output" for e in s.events)
        await (await s.submit(b"Hello."))
        assert display.snapshot()["messages"][-1]["role"] == "assistant"
        assert s.playback.queue.empty() and len(executor.operations) == 1
    assert len(children) == 1 and children[0].returncode == 0


@pytest.mark.asyncio
async def test_end_engagement_preserves_independent_file_task_without_narration(
    tmp_path, monkeypatch
):
    from me3_voice.session import State

    monkeypatch.chdir(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    audio = []

    async def runner(task, directory, budget):
        entered.set()
        await release.wait()
        (directory / "index.html").write_text("<h1>Independent task</h1>")
        return {
            "mode": "local-artifact",
            "summary": "One draft file created.",
            "artifacts": ["index.html"],
            "requests": 1,
        }

    async def capture(pcm):
        audio.append(pcm)

    tasks = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec(),
        {"ME3_LLM_API_KEY": "offline"},
        max_requests=3,
        runner=runner,
    )
    display = Transcript()
    async with Session(Speech(), MockLLM(), capture, tasks=tasks, transcript=display) as s:
        s.activate()
        pending = await s.submit(b"agent task: create a file")
        await entered.wait()
        await s.exit()
        await asyncio.gather(pending, return_exceptions=True)
        assert not next(iter(tasks.operations.values()))[1].done()
        release.set()
        await asyncio.gather(*(j for _, j in tasks.operations.values()))
        await asyncio.sleep(0)
        assert s.state == State.IDLE and audio == []
        row = display.snapshot()["tasks"][0]
        assert row["status"] == "reported" and row["artifacts"] == ["index.html"]
        assert (
            await tasks.read_artifact(row["artifact_id"], "index.html")
            == b"<h1>Independent task</h1>"
        )
        s.activate()
        assert s.history == []
        assert len(tasks.operations) == 1


def test_project_output_can_be_explicit_isolated_var_root_without_source_or_symlinks(
    tmp_path, monkeypatch
):
    from me3_voice.artifacts import approved_project_root

    project = tmp_path / "ME3-VoiceAgent"
    destination = project / "var/scratch/projects"
    destination.mkdir(parents=True)
    monkeypatch.chdir(project)
    assert approved_project_root(destination) == destination
    for root in [project, project / "src", tmp_path / "other"]:
        root.mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            approved_project_root(root)
    link = project / "var/link"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError):
        approved_project_root(link)


def test_groq_tool_metadata_filter_preserves_authoritative_result_and_identity():
    from me3_voice.onit_worker import groq_tool_results

    original = [
        {"role": "assistant", "tool_calls": [{"id": "call1"}]},
        {
            "role": "tool",
            "tool_call_id": "call1",
            "content": "Created once",
            "parameters": {"name": "private"},
        },
    ]
    result = groq_tool_results(original)
    assert result[0] == original[0]
    assert result[1] == {"role": "tool", "tool_call_id": "call1", "content": "Created once"}
    assert "parameters" in original[1]
