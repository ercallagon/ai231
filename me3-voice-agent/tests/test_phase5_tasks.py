"""Phase 5 durable task graph, control, replay, and concurrent voice checks."""

import asyncio
import aiosqlite
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage
import pytest

from me3_voice.configuration import BackendSpec, load_backends
from me3_voice.remote import LiveTransport
from me3_voice.task_runtime import (
    MAX_MODEL_CALLS,
    MeasuredAsyncSqliteSaver,
    TaskError,
    TaskRuntime,
    TaskStore,
    validate_interactive_artifact,
)
from me3_voice.research import ResearchClient


TERMINAL = {"succeeded", "completed_unverified", "failed", "cancelled"}


def test_interactive_html_rejects_swallowed_script_and_accepts_active_code():
    with pytest.raises(TaskError, match="interactive_script_missing_or_disabled"):
        validate_interactive_artifact(
            "game.html", "<script>// Game state const board = [];</script>"
        )
    validate_interactive_artifact(
        "game.html", "<button onclick=\"play()\">Play</button><script>\n// Game state\nplay();\n</script>"
    )
    with pytest.raises(TaskError, match="interactive_html_attribute_quoting_invalid"):
        validate_interactive_artifact(
            "game.html",
            "<div id=&#34;board&#34;></div><script>document.getElementById('board');</script>",
        )
    with pytest.raises(TaskError, match="interactive_html_attribute_quoting_invalid"):
        validate_interactive_artifact(
            "game.html", "<button onclick=&#34;play()&#34;>Play</button>"
        )
    with pytest.raises(TaskError, match="interactive_script_missing_or_disabled"):
        validate_interactive_artifact("game.js", "// Game state function init() { start(); }")
    validate_interactive_artifact("game.js", "// Game state\nfunction init() { start(); }")
    with pytest.raises(TaskError, match="interactive_dom_reference_missing"):
        validate_interactive_artifact(
            "game.html",
            "<div id='message'></div><script>document.getElementById('board');</script>",
        )


async def test_unfinished_task_does_not_project_private_partial_artifact(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([]))
    try:
        task, created = await runtime.store.create(
            "Create a playable page",
            request_id="pending-artifact",
            turn_id=1,
            call_id="pending-artifact",
        )
        assert created
        await runtime.store.update(task["task_id"], status="running")
        await runtime.store.persist_artifact(
            task["task_id"], 1, "game.html", "<script>play();</script>", "partial-write"
        )
        await runtime.store.persist_artifact(
            task["task_id"], 1, "todo/index.html", "<h1>To-do</h1>", "partial-nested-write"
        )
        await runtime.store.update(task["task_id"], status="waiting_approval")
        projected = runtime.store._project(runtime.store.rows[task["task_id"]])
        assert projected["status"] == "awaiting_input"
        assert projected["artifacts"] == []
        assert "artifact_id" not in projected
        await runtime.store.update(task["task_id"], status="failed")
        failed_projection = runtime.store._project(runtime.store.rows[task["task_id"]])
        assert failed_projection["status"] == "failed"
        assert failed_projection["artifact_id"] == task["task_id"][5:]
        assert failed_projection["artifacts"] == ["game.html", "todo/index.html"]
        assert failed_projection["absolute_path"] == str(
            runtime.store.artifact_root / task["task_id"]
        )
    finally:
        await runtime.close()


async def test_interactive_task_corrects_disabled_script_after_write_rejection(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module

    monkeypatch.setattr(task_runtime_module, "_chromium_executable", lambda: None)
    model = ScriptedModel(
        [
            call(
                "write_artifact",
                {"path": "game.html", "content": "<script>// state const board = [];</script>"},
                "bad-write",
            ),
            call(
                "write_artifact",
                {"path": "game.html", "content": "<script>\nconst board = [];\n</script>"},
                "good-write",
            ),
            call("read_artifact", {"path": "game.html"}, "read-game"),
            AIMessage(content="I saved the interactive page after correcting its script."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a playable interactive page")
        result = await wait_for_status(runtime, task["task_id"])
        saved = await runtime.store.read_artifact(task["task_id"], "game.html")
        assert "const board = [];" in saved["content"]
        assert result["status"] == "completed_unverified"
        assert result["verification"]["interactive_runtime_check_required"] is True
        assert result["verification"]["interactive_runtime_check_verified"] is False
        assert result["model_calls"] == 4
        assert result["verification"]["interactive_repair_attempts"] == 1
    finally:
        await runtime.close()


async def test_task_write_tool_rejects_entity_quoted_dom_attributes(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module

    monkeypatch.setattr(task_runtime_module, "_chromium_executable", lambda: None)
    malformed = (
        "<div id=&#34;board&#34;></div>"
        "<script>document.getElementById('board').addEventListener('click', play);</script>"
    )
    model = ScriptedModel(
        [
            call("write_artifact", {"path": "game.html", "content": malformed}, "bad-entity"),
            call("write_artifact", {"path": "game.html", "content": TIC_TAC_TOE_HTML}, "good-html"),
            call("read_artifact", {"path": "game.html"}, "read-good-html"),
            AIMessage(content="I corrected the malformed HTML attributes."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a playable interactive page")
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "completed_unverified"
        assert result["verification"]["interactive_repair_attempts"] == 1
        saved = await runtime.store.read_artifact(task["task_id"], "game.html")
        assert "id=&#34;board&#34;" not in saved["content"]
        assert 'id="board"' in saved["content"]
        async with runtime.store.lock:
            async with runtime.store.conn.execute(
                "SELECT payload FROM me3_task_events WHERE task_id=? AND kind='progress'",
                (task["task_id"],),
            ) as cursor:
                events = [json.loads(row[0]) for row in await cursor.fetchall()]
        assert any(event.get("stage") == "browser_repair_reserved" for event in events)
    finally:
        await runtime.close()


TIC_TAC_TOE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tic-Tac-Toe</title></head>
<body><h1>Tic-Tac-Toe</h1><div id="board"></div><p id="message"></p>
<button id="reset">Reset Game</button><script>
const boardElement = document.getElementById('board');
const messageElement = document.getElementById('message');
const resetButton = document.getElementById('reset');
let board = Array(9).fill(null);
let currentPlayer = 'X';
let active = true;
const wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
function init() {
  board = Array(9).fill(null); currentPlayer = 'X'; active = true;
  boardElement.replaceChildren();
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('button'); cell.className = 'cell';
    cell.dataset.index = String(i); cell.addEventListener('click', () => play(i));
    boardElement.appendChild(cell);
  }
  messageElement.textContent = "Player X's turn";
}
function play(i) {
  if (!active || board[i] !== null) return;
  board[i] = currentPlayer; boardElement.children[i].textContent = currentPlayer;
  if (wins.some(([a,b,c]) => board[a] && board[a] === board[b] && board[b] === board[c])) {
    messageElement.textContent = `Player ${currentPlayer} wins!`; active = false; return;
  }
  if (board.every(value => value !== null)) { messageElement.textContent = 'Draw!'; active = false; return; }
  currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
  messageElement.textContent = `Player ${currentPlayer}'s turn`;
}
resetButton.addEventListener('click', init); init();
</script></body></html>"""


async def test_playable_artifact_is_chromium_verified_before_success(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module

    executable = task_runtime_module._chromium_executable()
    if not executable:
        pytest.skip("Chromium is unavailable")
    monkeypatch.setenv("ME3_CHROMIUM_EXECUTABLE_PATH", executable)
    model = ScriptedModel(
        [
            call("write_artifact", {"path": "game.html", "content": TIC_TAC_TOE_HTML}, "write-game"),
            call("read_artifact", {"path": "game.html"}, "read-game"),
            AIMessage(content="I created and browser-verified the playable game."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a playable Tic-Tac-Toe game")
        result = await wait_for_status(runtime, task["task_id"], timeout=25)
        assert result["status"] == "succeeded"
        assert result["verification"]["interactive_runtime_check_verified"] is True
        assert result["verification"]["interactive_repair_attempts"] == 0
        async with runtime.store.lock:
            async with runtime.store.conn.execute(
                "SELECT detail FROM me3_task_evidence WHERE task_id=? AND kind='browser_runtime_test' AND verified=1",
                (task["task_id"],),
            ) as cursor:
                evidence = json.loads((await cursor.fetchone())[0])
        checks = evidence["result"]["runtime_checks"]
        assert all(
            checks[name]
            for name in (
                "nine_cells",
                "alternating_moves",
                "occupied_cell_rejected",
                "win",
                "draw",
                "terminal_move_rejected",
                "reset_after_win",
                "reset_starts_x",
                "reset_after_draw",
            )
        )
        assert evidence["workspace_manifest"] == {"game.html": evidence["sha256"]}
        projected = runtime.store._project(runtime.store.rows[task["task_id"]])
        assert projected["status"] == "completed"
        assert projected["artifacts"] == ["game.html"]
        async with runtime.store.lock:
            await runtime.store.conn.execute(
                "UPDATE me3_artifacts SET sha256=? WHERE task_id=? AND path='game.html'",
                ("0" * 64, task["task_id"]),
            )
            await runtime.store.conn.commit()
        stale = await runtime.store.verify(task["task_id"])
        assert stale["verified"] is False
        assert stale["interactive_runtime_check_verified"] is False
    finally:
        await runtime.close()


async def test_chromium_failure_reports_runtime_dom_errors(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module

    executable = task_runtime_module._chromium_executable()
    if not executable:
        pytest.skip("Chromium is unavailable")
    monkeypatch.setenv("ME3_CHROMIUM_EXECUTABLE_PATH", executable)
    artifact = tmp_path / "dom-error.html"
    artifact.write_text(
        '<!doctype html><div id="board"></div><script>'
        'document.querySelector("#missing").addEventListener("click", () => {});'
        "</script>",
        encoding="utf-8",
    )
    result = await task_runtime_module.validate_in_chromium(artifact)
    assert result["available"] is True
    assert result["verified"] is False
    assert any("Cannot read properties of null" in error for error in result["errors"])


async def test_chromium_loads_registered_module_assets_and_blocks_external_requests(
    tmp_path, monkeypatch
):
    import me3_voice.task_runtime as task_runtime_module

    executable = task_runtime_module._chromium_executable()
    if not executable:
        pytest.skip("Chromium is unavailable")
    monkeypatch.setenv("ME3_CHROMIUM_EXECUTABLE_PATH", executable)
    html_path = tmp_path / "index.html"
    html = (
        '<!doctype html><p id="state"></p>'
        '<script type="module" src="app.mjs"></script>'
    )
    module = 'document.getElementById("state").textContent = "module loaded";'
    html_path.write_text(html, encoding="utf-8")
    result = await task_runtime_module.validate_in_chromium(
        html_path,
        artifact_path="index.html",
        artifact_files={"index.html": html, "app.mjs": module},
    )
    assert result["verified"] is True

    external_html = (
        '<!doctype html><p>offline page</p><script>'
        'fetch("https://example.invalid/private?token=not-logged").catch(() => {});'
        "</script>"
    )
    result = await task_runtime_module.validate_in_chromium(
        html_path,
        artifact_path="index.html",
        artifact_files={"index.html": external_html},
    )
    assert result["verified"] is False
    assert any(error == "external_resource_blocked:https://example.invalid" for error in result["errors"])


async def test_chromium_failure_allows_exactly_one_repair_pass(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module

    executable = task_runtime_module._chromium_executable()
    if not executable:
        pytest.skip("Chromium is unavailable")
    monkeypatch.setenv("ME3_CHROMIUM_EXECUTABLE_PATH", executable)
    broken = TIC_TAC_TOE_HTML.replace(
        "resetButton.addEventListener('click', init); init();",
        "resetButton.addEventListener('click', init); // init disabled on one line",
    )
    model = ScriptedModel(
        [
            call("write_artifact", {"path": "game.html", "content": broken}, "write-broken"),
            AIMessage(content="I created the game."),
            call("write_artifact", {"path": "game.html", "content": TIC_TAC_TOE_HTML}, "repair-game"),
            call("read_artifact", {"path": "game.html"}, "read-repaired-game"),
            AIMessage(content="I repaired and verified the playable game."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a playable Tic-Tac-Toe game")
        result = await wait_for_status(runtime, task["task_id"], timeout=25)
        assert result["status"] == "succeeded"
        assert result["model_calls"] == 5
        assert result["verification"]["interactive_repair_attempts"] == 1
        events = runtime.store.rows[task["task_id"]]["_recent_events"]
        assert any(event["kind"] == "browser_runtime_test" for event in events)
        assert any(
            event["kind"] == "progress"
            and json.loads(event["payload"]).get("stage") == "browser_repair_reserved"
            for event in events
        )
    finally:
        await runtime.close()


async def test_second_chromium_failure_stays_unverified_without_second_repair(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module

    executable = task_runtime_module._chromium_executable()
    if not executable:
        pytest.skip("Chromium is unavailable")
    monkeypatch.setenv("ME3_CHROMIUM_EXECUTABLE_PATH", executable)
    broken = TIC_TAC_TOE_HTML.replace(
        "resetButton.addEventListener('click', init); init();",
        "resetButton.addEventListener('click', init); // init disabled on one line",
    )
    model = ScriptedModel(
        [
            call("write_artifact", {"path": "game.html", "content": broken}, "write-broken"),
            AIMessage(content="I created the game."),
            AIMessage(content="I could not repair the game."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a playable Tic-Tac-Toe game")
        result = await wait_for_status(runtime, task["task_id"], timeout=25)
        assert result["status"] == "completed_unverified"
        assert result["model_calls"] == 3
        assert result["verification"]["interactive_runtime_check_verified"] is False
        assert result["verification"]["interactive_repair_attempts"] == 1
    finally:
        await runtime.close()


async def test_task_repairs_one_malformed_model_tool_call_within_task_budget(tmp_path):
    model = ScriptedModel(
        [
            TaskError("invalid_tool_json"),
            call(
                "write_artifact",
                {"path": "result.txt", "content": "recovered artifact"},
                "recovered-write",
            ),
            call("read_artifact", {"path": "result.txt"}, "recovered-read"),
            AIMessage(content="I created and read back the recovered artifact."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a small result file")
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "succeeded"
        assert result["model_calls"] == 4
        projected = runtime.store._project(runtime.store.rows[task["task_id"]])
        assert "Retrying malformed tool call" in projected["completed_steps"]
    finally:
        await runtime.close()


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def __call__(self, _messages, _tools):
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected model request")
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = await value()
        return value


class FakeResearch:
    async def search(self, query, _max_results):
        return {
            "query": query,
            "results": [{"title": "Fixture source", "url": "https://example.test/original"}],
        }

    async def retrieve(self, url):
        return {"source_id": "S1", "url": url, "text": "Fixture source statement."}

    def verify(self, source_id, quote):
        return {
            "source_id": source_id,
            "verified": quote == "Fixture source statement.",
            "claim_truth_verified": False,
        }


async def wait_for_status(runtime, task_id, statuses=TERMINAL, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = await runtime.status(task_id)
        if result["status"] in statuses:
            return result
        await asyncio.sleep(0.01)
    raise AssertionError(
        "task did not reach a requested state: " + str((await runtime.status(task_id))["status"])
    )


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_task_storage_paths_are_configurable_with_local_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("ME3_TASK_DATABASE", raising=False)
    monkeypatch.delenv("ME3_TASK_WORKSPACE", raising=False)
    default = TaskRuntime(load_backends()["llm"], model=ScriptedModel([]))
    assert default.database == Path("var/cache/task-runtime/tasks.sqlite")
    assert default.artifact_root == Path("var/scratch/task-runtime")

    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "pi-data" / "tasks.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "workspace"))
    configured = TaskRuntime(load_backends()["llm"], model=ScriptedModel([]))
    assert configured.database == tmp_path / "pi-data" / "tasks.sqlite"
    assert configured.artifact_root == tmp_path / "workspace"


async def make_runtime(tmp_path, model, **kwargs):
    runtime = TaskRuntime(
        load_backends()["llm"],
        database=kwargs.pop("database", tmp_path / "tasks.sqlite"),
        artifact_root=kwargs.pop("artifact_root", tmp_path / "artifacts"),
        model=model,
        **kwargs,
    )
    await runtime.start_runtime()
    return runtime


async def start_task(runtime, goal="an unfamiliar multi-step task", *, request_id="request-1"):
    return await runtime.start(goal, request_id=request_id, turn_id=1, call_id="call-" + request_id)


async def test_github_task_tool_pauses_for_approval_before_any_remote_publish(tmp_path, monkeypatch):
    import me3_voice.task_runtime as task_runtime_module
    from me3_voice.github_publish import GitHubPublisher

    class Writer:
        client_state = "INITIALIZED_UNVERIFIED"
        repository_name = "private-owner/demo-repo"
        base_branch = "main"
        branch_for = staticmethod(GitHubPublisher.branch_for)

        async def repository_visibility(self):
            return "public"

        async def existing_paths(self, paths, *, ref):
            assert ref == "swift-artifacts"
            return []

        async def pages_configuration(self):
            return {"configured": False, "branch": "gh-pages", "path": "/"}

        def __init__(self):
            self.calls = 0

        async def publish(self, *_args, **_kwargs):
            self.calls += 1
            return {"status": "published"}

    class ApprovalPause(Exception):
        pass

    def pause_before_approval(_request):
        raise ApprovalPause

    monkeypatch.setattr(task_runtime_module, "interrupt", pause_before_approval)
    browser_checks = []

    async def verify_pages_artifact(path, **kwargs):
        browser_checks.append((path, kwargs["artifact_path"], sorted(kwargs["artifact_files"])))
        return {"available": True, "verified": True, "browser": "chromium", "errors": []}

    monkeypatch.setattr(task_runtime_module, "validate_in_chromium", verify_pages_artifact)
    runtime = TaskRuntime(
        load_backends()["llm"],
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        model=ScriptedModel([]),
    )
    writer = Writer()
    runtime.github_writer = writer
    try:
        await runtime.start_runtime()
        task, _ = await runtime.store.create(
            "Prepare and publish a website artifact",
            "github-approval-test",
            1,
            "github-approval-test",
        )
        await runtime.store.update(task["task_id"], status="running")
        await runtime.store.persist_artifact(
            task["task_id"],
            1,
            "me3-voice-agent/artifacts/trials/nasa-artemis/index.html",
            "<h1>Private draft</h1>",
            "write-html",
        )
        await runtime.store.persist_artifact(
            task["task_id"],
            1,
            "me3-voice-agent/artifacts/trials/nasa-artemis/styles.css",
            "h1 { color: navy; }",
            "write-css",
        )
        tool = next(tool for tool in runtime.registered_tools() if tool.name == "publish_github_branch")
        state = {
            "task_id": task["task_id"],
            "revision": 1,
            "requirements": "Prepare and publish a website artifact",
        }
        token = runtime._tool_context.set((state, "publish-approval-01"))
        try:
            with pytest.raises(ApprovalPause):
                await tool.coroutine(
                    paths=[
                        "me3-voice-agent/artifacts/trials/nasa-artemis/index.html",
                        "me3-voice-agent/artifacts/trials/nasa-artemis/styles.css",
                    ],
                    commit_message="Publish website",
                    deploy_pages=True,
                )
        finally:
            runtime._tool_context.reset(token)
        assert writer.calls == 0
        assert browser_checks == [
            (
                Path("me3-voice-agent/artifacts/trials/nasa-artemis/index.html"),
                "me3-voice-agent/artifacts/trials/nasa-artemis/index.html",
                [
                    "me3-voice-agent/artifacts/trials/nasa-artemis/index.html",
                    "me3-voice-agent/artifacts/trials/nasa-artemis/styles.css",
                ],
            )
        ]
        row = await runtime.status(task["task_id"])
        assert row["approvals"]
        assert row["approvals"][0]["status"] == "pending"
        assert row["approvals"][0]["action"] == "github_publish"
        assert "does not configure GitHub Pages" in row["approvals"][0]["details"]
        assert "me3-voice-agent/artifacts/trials/nasa-artemis/" in row["approvals"][0]["details"]
        approval_events = [event for event in row["progress"] if event["kind"] == "approval_requested"]
        assert "index.html" in approval_events[-1]["payload"]["spoken_summary"]
        assert "styles.css" in approval_events[-1]["payload"]["spoken_summary"]
        assert "public" in approval_events[-1]["payload"]["spoken_summary"]
    finally:
        await runtime.close()


@pytest.mark.parametrize("pages_decision", ["approve", "reject"])
async def test_github_branch_and_pages_require_distinct_approvals(
    tmp_path, monkeypatch, pages_decision
):
    import me3_voice.task_runtime as task_runtime_module
    from me3_voice.github_publish import GitHubPublisher

    class Writer:
        repository_name = "public-owner/demo-repo"
        base_branch = "main"
        branch_for = staticmethod(GitHubPublisher.branch_for)

        def __init__(self):
            self.published = []
            self.pages = []

        async def repository_visibility(self):
            return "public"

        async def existing_paths(self, paths, *, ref):
            assert ref == "swift-artifacts"
            return []

        async def pages_configuration(self):
            return {"configured": False, "branch": "gh-pages", "path": "/"}

        async def publish(self, task_id, tool_call_id, artifacts, message, **kwargs):
            self.published.append((task_id, tool_call_id, tuple(artifacts), message, kwargs))
            return {
                "status": "published",
                "commit": "a" * 40,
                "url": "https://github.com/public-owner/demo-repo/tree/swift-artifacts",
            }

        async def enable_pages_for_published_branch(self, branch, commit, site_path, **kwargs):
            self.pages.append((branch, commit, site_path, kwargs))
            return {
                "status": "deployed",
                "deployment_url": self.expected_pages_url(site_path),
                "deployment_status": "verified",
            }

        def expected_pages_url(self, site_path=""):
            return "https://public-owner.github.io/demo-repo/" + (site_path + "/" if site_path else "")

    async def verify_pages_artifact(_path, **_kwargs):
        return {"available": True, "verified": True, "browser": "chromium", "errors": []}

    monkeypatch.setattr(task_runtime_module, "validate_in_chromium", verify_pages_artifact)
    tool_id = "two-step-publish"
    model = ScriptedModel(
        [
            call(
                "publish_github_branch",
                {
                    "paths": ["site/index.html", "site/styles.css"],
                    "commit_message": "Publish site",
                    "deploy_pages": True,
                },
                tool_id,
            ),
            AIMessage(content="The approved branch and Pages site are ready."),
        ]
    )
    runtime = TaskRuntime(
        load_backends()["llm"],
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        model=model,
    )
    writer = Writer()
    runtime.github_writer = writer
    try:
        await runtime.start_runtime()
        task, _ = await runtime.store.create("Publish a website", "two-step-approval", 1, "two-step-approval")
        await runtime.store.update(task["task_id"], status="running")
        await runtime.store.persist_artifact(task["task_id"], 1, "site/index.html", "<h1>Site</h1>", "html")
        await runtime.store.persist_artifact(task["task_id"], 1, "site/styles.css", "h1 { color: navy; }", "css")
        runtime._spawn(task["task_id"], initial=True)

        branch_wait = await wait_for_status(runtime, task["task_id"], {"waiting_approval"})
        branch_approval = branch_wait["approvals"][0]
        assert branch_approval["action"] == "github_publish"
        assert "site/index.html" in branch_approval["details"]
        assert "public" in branch_approval["details"]
        assert writer.published == []
        assert writer.pages == []

        await runtime.approve(
            task["task_id"], branch_approval["approval_id"], branch_approval["revision"], "approve"
        )
        pages_wait = await wait_for_status(runtime, task["task_id"], {"waiting_approval"})
        pages_approval = pages_wait["approvals"][0]
        assert pages_approval["action"] == "github_pages_deploy"
        assert "publicly accessible" in pages_approval["details"]
        assert "gh-pages:/" in pages_approval["details"]
        assert "stable" in pages_approval["details"]
        destination = "me3-voice-agent/artifacts/demos/" + task["task_id"][5:] + "/site"
        assert f"https://public-owner.github.io/demo-repo/{destination}/" in pages_approval["details"]
        assert set(writer.published[0][2]) == {
            f"{destination}/index.html",
            f"{destination}/styles.css",
        }
        assert len(writer.published) == 1
        assert writer.pages == []

        await runtime.approve(
            task["task_id"],
            pages_approval["approval_id"],
            pages_approval["revision"],
            pages_decision,
        )
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == (
            "succeeded" if pages_decision == "approve" else "completed_unverified"
        )
        assert len(writer.published) == 1
        assert len(writer.pages) == (1 if pages_decision == "approve" else 0)
        projection = runtime.store._project(runtime.store.rows[task["task_id"]])
        assert projection["publication"]["repository_url"].endswith("/tree/swift-artifacts")
        if pages_decision == "approve":
            assert projection["publication"]["pages_url"] == writer.expected_pages_url(destination)
        else:
            assert "pages_url" not in projection["publication"]
        assert result["verification"]["effects_checked"] == 2
    finally:
        await runtime.close()


async def test_github_publish_rejects_artifact_when_readback_no_longer_matches_record(
    tmp_path, monkeypatch
):
    class Writer:
        repository_name = "private-owner/demo-repo"
        base_branch = "main"

        async def publish(self, *_args, **_kwargs):
            raise AssertionError("unverified files must not reach the publisher")

    runtime = TaskRuntime(
        load_backends()["llm"],
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        model=ScriptedModel([]),
    )
    runtime.github_writer = Writer()
    try:
        await runtime.start_runtime()
        task, _ = await runtime.store.create(
            "Publish a task artifact", "github-readback-check", 1, "github-readback-check"
        )
        await runtime.store.update(task["task_id"], status="running")
        await runtime.store.persist_artifact(
            task["task_id"], 1, "notes.txt", "saved text", "write-notes"
        )

        async def mismatched_metadata(_task_id, _path):
            return {"sha256": "0" * 64, "size": len("saved text"), "revision": 1}

        monkeypatch.setattr(runtime.store, "artifact_metadata", mismatched_metadata)
        tool = next(tool for tool in runtime.registered_tools() if tool.name == "publish_github_branch")
        state = {"task_id": task["task_id"], "revision": 1, "requirements": "Publish a task artifact"}
        token = runtime._tool_context.set((state, "github-readback-mismatch"))
        try:
            with pytest.raises(TaskError, match="github_artifact_not_verified"):
                await tool.ainvoke(
                    {"paths": ["notes.txt"], "commit_message": "Publish notes"}
                )
        finally:
            runtime._tool_context.reset(token)
        row = await runtime.status(task["task_id"])
        assert row["approvals"] == []
    finally:
        await runtime.close()


async def test_native_task_snapshot_surfaces_real_progress_without_fake_percent(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([]))
    try:
        task, created = await runtime.store.create(
            "Create and verify a local artifact", "progress-ui", 1, "call-progress-ui"
        )
        assert created
        await runtime.store.event(
            task["task_id"], "progress", {"stage": "workspace_directory_created"}
        )
        await runtime.store.event(
            task["task_id"], "progress", {"stage": "artifact_written", "path": "index.html"}
        )

        row = runtime.store.snapshot()[0]
        assert row["status"] == "running"
        assert row["phase"] == "Artifact written"
        assert row["completed_steps"] == ["Workspace directory created", "Artifact written"]
        assert row["last_progress_utc"].endswith("+00:00")
        assert "progress_percent" not in row
    finally:
        await runtime.close()


async def test_graph_researches_writes_reads_and_independently_verifies_artifact(tmp_path):
    model = ScriptedModel(
        [
            call(
                "web_search",
                {"query": "a dynamically derived research query", "max_results": 2},
                "search-1",
            ),
            call(
                "write_artifact",
                {"path": "notes.md", "content": "Fixture source statement.\n"},
                "write-1",
            ),
            call("read_artifact", {"path": "notes.md"}, "read-1"),
            AIMessage(content="I saved the source notes and checked the file contents."),
        ]
    )
    runtime = await make_runtime(tmp_path, model, research=FakeResearch())
    try:
        task = await start_task(runtime, "Research a topic and save a concise sourced note")
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "succeeded"
        assert result["verification"]["level"] == "artifact_readback"
        assert result["verification"]["artifacts_checked"] == 1
        assert result["model_calls"] == 4
        assert result["tool_calls"] == 3
        projected = runtime.store._project(runtime.store.rows[task["task_id"]])
        assert projected["kind"] == "agent"
        assert projected["artifact_id"] == task["task_id"][5:]
        assert projected["artifacts"] == ["notes.md"]
        metadata = await runtime.store.artifact_metadata(task["task_id"], "notes.md")
        assert metadata["sha256"] == hashlib.sha256(b"Fixture source statement.").hexdigest()
        assert metadata["size"] == len(b"Fixture source statement.")
        assert (
            Path(tmp_path / "artifacts" / task["task_id"] / "notes.md").read_text()
            == "Fixture source statement."
        )
        assert (tmp_path / "artifacts" / task["task_id"]).stat().st_mode & 0o777 == 0o700
        assert (
            tmp_path / "artifacts" / task["task_id"] / "notes.md"
        ).stat().st_mode & 0o777 == 0o600
    finally:
        await runtime.close()


async def test_task_research_caps_are_durable_and_legacy_schema_migrates(tmp_path):
    assert MAX_MODEL_CALLS == 30
    conn = await aiosqlite.connect(":memory:", check_same_thread=False)
    try:
        saver = MeasuredAsyncSqliteSaver(conn)
        await saver.setup()
        store = TaskStore(saver, tmp_path)
        await store.setup()
        runtime = TaskRuntime(load_backends()["llm"], model=ScriptedModel([]))
        runtime.store = store
        task, _ = await store.create("Research deeply", "research-cap", 1, "call-cap")
        await store.update(task["task_id"], status="running")
        state = {"task_id": task["task_id"], "revision": 1}
        for _ in range(10):
            await runtime._tool_guard(state, research_kind="search")
        with pytest.raises(TaskError, match="task_search_budget_exhausted"):
            await runtime._tool_guard(state, research_kind="search")
        for _ in range(25):
            await runtime._tool_guard(state, research_kind="retrieval")
        with pytest.raises(TaskError, match="task_retrieval_budget_exhausted"):
            await runtime._tool_guard(state, research_kind="retrieval")
        row = await store.get(task["task_id"])
        assert (row["search_calls"], row["retrieval_calls"], row["tool_calls"]) == (10, 25, 35)
        await store.update(task["task_id"], model_calls=30)
        runtime.max_model_calls = 100  # Constructor overrides cannot bypass the hard ceiling.
        with pytest.raises(TaskError, match="model_budget_exhausted"):
            await runtime._agent(state)
        assert runtime.model.calls == 0

        # Reopening TaskStore on the same durable DB retains counters. Simulate
        # the previous schema and ensure startup migrates without deleting rows.
        reopened = TaskStore(saver, tmp_path)
        await reopened.setup()
        assert (await reopened.get(task["task_id"]))["search_calls"] == 10
        async with saver.lock:
            await saver.conn.execute("ALTER TABLE me3_tasks DROP COLUMN search_calls")
            await saver.conn.execute("ALTER TABLE me3_tasks DROP COLUMN retrieval_calls")
            await saver.conn.commit()
        migrated = TaskStore(saver, tmp_path)
        await migrated.setup()
        row = await migrated.get(task["task_id"])
        assert row["goal"] == "Research deeply"
        assert (row["search_calls"], row["retrieval_calls"]) == (0, 0)
    finally:
        await conn.close()


async def test_task_research_injects_relevant_sections_and_retains_full_source(tmp_path):
    research = ResearchClient()
    full_text = (
        "The old building opened in 1901. "
        "The official result named Candidate Blue as the winner of the Senate election. "
        "The archive moved to a new address in 1980. "
    ) * 10
    source = {
        "source_id": "S1",
        "url": "https://example.test/results",
        "title": "Official results",
        "text": full_text,
        "text_sha256": "fixture",
    }

    async def retrieve(_url):
        research.sources["S1"] = source
        return source

    research.retrieve = retrieve
    conn = await aiosqlite.connect(":memory:", check_same_thread=False)
    try:
        saver = MeasuredAsyncSqliteSaver(conn)
        await saver.setup()
        store = TaskStore(saver, tmp_path)
        await store.setup()
        runtime = TaskRuntime(
            load_backends()["llm"], model=ScriptedModel([]), research=research
        )
        runtime.store = store
        task, _ = await runtime.store.create(
            "Find who won the Senate election", "research-projection", 1, "call-projection"
        )
        await runtime.store.update(task["task_id"], status="running")
        state = {
            "task_id": task["task_id"],
            "revision": 1,
            "requirements": "Senate election winner",
        }
        token = runtime._tool_context.set((state, "retrieve-1"))
        try:
            retrieve_tool = next(tool for tool in runtime.registered_tools() if tool.name == "retrieve_source")
            projected = await retrieve_tool.ainvoke({"url": source["url"]})
        finally:
            runtime._tool_context.reset(token)
        assert "Candidate Blue" in projected["text"]
        assert "old building opened" not in projected["text"]
        assert research.verify("S1", "The old building opened in 1901.")["quote_found"]
        row = await runtime.store.get(task["task_id"])
        assert row["retrieval_calls"] == 1
    finally:
        await conn.close()


async def test_general_task_creates_lists_reads_and_updates_nested_project_files(tmp_path):
    model = ScriptedModel(
        [
            call("create_directory", {"path": "site/assets"}, "mkdir-site"),
            call(
                "write_artifact",
                {"path": "site/index.html", "content": "<h1>First draft</h1>"},
                "write-html-1",
            ),
            call(
                "write_artifact",
                {"path": "site/assets/site.css", "content": "body { color: navy; }"},
                "write-css",
            ),
            call("list_workspace", {"path": "site"}, "list-site"),
            call("read_artifact", {"path": "site/index.html"}, "read-html-1"),
            call(
                "write_artifact",
                {"path": "site/index.html", "content": "<!doctype html><h1>Revised project</h1>"},
                "write-html-2",
            ),
            call("read_artifact", {"path": "site/index.html"}, "read-html-2"),
            AIMessage(
                content="Created the local project files and read them back. No publication or browser execution was performed."
            ),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime, "Create a small local website with a heading and styling")
        result = await wait_for_status(runtime, task["task_id"])
        workspace = tmp_path / "artifacts" / task["task_id"]
        assert result["status"] == "succeeded"
        assert result["verification"]["artifacts_checked"] == 2
        assert (
            workspace / "site/index.html"
        ).read_text() == "<!doctype html><h1>Revised project</h1>"
        assert (workspace / "site/assets/site.css").read_text() == "body { color: navy; }"
        assert (workspace / "site").stat().st_mode & 0o777 == 0o700
        assert (workspace / "site/index.html").stat().st_mode & 0o777 == 0o600
        assert result["tool_calls"] == 7
        assert result["model_calls"] == 8
    finally:
        await runtime.close()


async def test_unfamiliar_multipage_site_artifact_has_valid_local_links_and_assets(tmp_path):
    model = ScriptedModel(
        [
            call("create_directory", {"path": "site/assets"}, "site-dir"),
            call(
                "write_artifact",
                {
                    "path": "site/index.html",
                    "content": '<!doctype html><html><head><title>Wetlands</title><link rel="stylesheet" href="assets/site.css"></head><body><h1>Wetlands guide</h1><a href="species.html">Species</a></body></html>',
                },
                "index",
            ),
            call(
                "write_artifact",
                {
                    "path": "site/species.html",
                    "content": '<!doctype html><html><head><title>Species</title><link rel="stylesheet" href="assets/site.css"></head><body><h1>Local species</h1><a href="index.html">Home</a><img src="assets/heron.svg" alt="Heron"></body></html>',
                },
                "species",
            ),
            call(
                "write_artifact",
                {"path": "site/assets/site.css", "content": "body { color: #123; }"},
                "css",
            ),
            call(
                "write_artifact",
                {
                    "path": "site/assets/heron.svg",
                    "content": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><title>Heron</title><circle cx="5" cy="5" r="4"/></svg>',
                },
                "svg",
            ),
            call("list_workspace", {"path": "site"}, "list-site"),
            AIMessage(
                content="Created the four local website files. Both page links and the stylesheet and image asset were independently checked from the workspace."
            ),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(
            runtime,
            "Create an unfamiliar two-page wetlands guide with shared styling and an illustration",
        )
        result = await wait_for_status(runtime, task["task_id"])
        workspace = tmp_path / "artifacts" / task["task_id"] / "site"
        assert result["status"] == "succeeded"
        assert result["verification"]["artifacts_checked"] == 4
        import re

        html = {path.name: path.read_text() for path in workspace.glob("*.html")}
        assert set(html) == {"index.html", "species.html"}
        refs = {name: re.findall(r'(?:href|src)="([^"]+)"', body) for name, body in html.items()}
        for page, links in refs.items():
            for link in links:
                assert (workspace / page).parent.joinpath(link).resolve().is_file()
        assert (workspace / "assets/site.css").read_text() == "body { color: #123; }"
        assert "<svg" in (workspace / "assets/heron.svg").read_text()
        assert result["verification"]["level"] == "artifact_readback"
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/tmp/escape",
        "site/../../escape",
        "site//index.html",
        "site/.secret",
        "site\\escape",
    ],
)
async def test_workspace_rejects_traversal_absolute_hidden_and_noncanonical_paths(tmp_path, path):
    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    try:
        with pytest.raises(TaskError, match="invalid_artifact_path"):
            await runtime.store.create_directory("task_" + "a" * 24, path)
    finally:
        await runtime.close()


async def test_workspace_never_follows_symlinked_directory_or_file(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    task_id = "task_" + "b" * 24
    root = tmp_path / "artifacts" / task_id
    root.mkdir(parents=True)
    (tmp_path / "artifacts").chmod(0o700)
    root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not for the agent")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    (root / "linked.txt").symlink_to(outside / "secret.txt")
    try:
        with pytest.raises(TaskError, match="invalid_artifact_path"):
            await runtime.store.read_artifact(task_id, "escape/secret.txt")
        with pytest.raises(TaskError):
            await runtime.store.read_artifact(task_id, "linked.txt")
        assert "escape" not in [
            item["name"] for item in (await runtime.store.list_workspace(task_id))["entries"]
        ]
    finally:
        await runtime.close()


async def test_workspace_read_rejects_hardlink_to_outside_file(tmp_path):
    import os

    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    task_id = "task_" + "d" * 24
    root = tmp_path / "artifacts" / task_id
    root.mkdir(parents=True, mode=0o700)
    (tmp_path / "artifacts").chmod(0o700)
    root.chmod(0o700)
    outside = tmp_path / "private.txt"
    outside.write_text("private information")
    outside.chmod(0o600)
    os.link(outside, root / "alias.txt")
    try:
        with pytest.raises(TaskError, match="invalid_artifact_file"):
            await runtime.store.read_artifact(task_id, "alias.txt")
    finally:
        await runtime.close()


async def test_parent_swap_during_atomic_replace_cannot_write_outside_workspace(
    tmp_path, monkeypatch
):
    import os

    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    task_id = "task_" + "c" * 24
    root = tmp_path / "artifacts" / task_id
    directory = root / "site"
    directory.mkdir(parents=True, mode=0o700)
    (tmp_path / "artifacts").chmod(0o700)
    root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    original = os.replace

    def swap_before_replace(source, target, **kwargs):
        directory.rename(root / "moved")
        directory.symlink_to(outside, target_is_directory=True)
        return original(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", swap_before_replace)
    try:
        result = await asyncio.to_thread(
            runtime.store._atomic_write_artifact, task_id, "site/index.html", b"private"
        )
        assert result == b"private"
        assert not (outside / "index.html").exists()
        assert (root / "moved/index.html").read_bytes() == b"private"
        with pytest.raises(TaskError):
            await runtime.store.read_artifact(task_id, "site/index.html")
    finally:
        await runtime.close()


async def test_workspace_write_replay_is_idempotent_and_identity_conflicts_fail_closed(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    try:
        row, _ = await runtime.store.create(
            "Write one local artifact", "local-write", 1, "local-write"
        )
        task_id = row["task_id"]
        await runtime.store.update(task_id, status="running")
        first = await runtime.store.persist_artifact(
            task_id, 1, "nested/file.txt", "stable", "write-once"
        )
        replay = await runtime.store.persist_artifact(
            task_id, 1, "nested/file.txt", "stable", "write-once"
        )
        assert first["readback_verified"] and not first["replayed"]
        assert replay["readback_verified"] and replay["replayed"]
        with pytest.raises(TaskError, match="operation_identity_conflict"):
            await runtime.store.persist_artifact(
                task_id, 1, "nested/file.txt", "different", "write-once"
            )
        assert (tmp_path / "artifacts" / task_id / "nested/file.txt").read_text() == "stable"
        revised = await runtime.store.steer(task_id, 1, "Use a newer requirement revision.")
        assert revised["revision"] == 2
        with pytest.raises(TaskError, match="operation_identity_conflict"):
            await runtime.store.persist_artifact(
                task_id, 2, "nested/file.txt", "stable", "write-once"
            )
    finally:
        await runtime.close()


async def test_workspace_fsync_work_does_not_block_the_asyncio_loop(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    row, _ = await runtime.store.create("Write a local artifact", "loop-write", 1, "loop-write")
    await runtime.store.update(row["task_id"], status="running")
    original = runtime.store._atomic_write_artifact

    def delayed_write(*args):
        time.sleep(0.15)
        return original(*args)

    runtime.store._atomic_write_artifact = delayed_write
    try:
        write = asyncio.create_task(
            runtime.store.persist_artifact(
                row["task_id"], 1, "page.html", "<h1>Nonblocking write</h1>", "loop-write-1"
            )
        )
        started = time.perf_counter()
        await asyncio.sleep(0.02)
        loop_delay_ms = (time.perf_counter() - started) * 1000
        result = await write
        assert result["readback_verified"]
        assert loop_delay_ms < 80
    finally:
        await runtime.close()


async def test_steering_during_workspace_io_fences_stale_artifact_completion(tmp_path):
    import threading

    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    row, _ = await runtime.store.create("Write a site", "steer-write", 1, "steer-write")
    task_id = row["task_id"]
    await runtime.store.update(task_id, status="running")
    entered, release = threading.Event(), threading.Event()
    original = runtime.store._atomic_write_artifact

    def delayed_write(*args):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test write gate was not released")
        return original(*args)

    runtime.store._atomic_write_artifact = delayed_write
    try:
        stale_write = asyncio.create_task(
            runtime.store.persist_artifact(
                task_id, 1, "site/index.html", "old requirement", "stale-site-write"
            )
        )
        assert await asyncio.to_thread(entered.wait, 3)
        revision = await runtime.store.steer(task_id, 1, "Use the corrected requirement instead.")
        assert revision["revision"] == 2
        release.set()
        with pytest.raises(TaskError, match="stale_or_inactive_task"):
            await stale_write
        assert not (await runtime.store.verify(task_id))["verified"]
        assert (
            tmp_path / "artifacts" / task_id / "site/index.html"
        ).read_text() == "old requirement"

        runtime.store._atomic_write_artifact = original
        await runtime.store.persist_artifact(
            task_id, 2, "site/index.html", "corrected requirement", "current-site-write"
        )
        assert (
            tmp_path / "artifacts" / task_id / "site/index.html"
        ).read_text() == "corrected requirement"
        assert (await runtime.store.verify(task_id))["verified"]
    finally:
        release.set()
        await runtime.close()


async def test_workspace_write_recovers_dispatching_intent_only_from_matching_readback(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    try:
        row, _ = await runtime.store.create(
            "Recover local artifact write", "recover-write", 1, "recover-write"
        )
        task_id, path, content, operation_id = (
            row["task_id"],
            "recover.txt",
            "durable",
            "recover-op",
        )
        await runtime.store.update(task_id, status="running")
        digest = hashlib.sha256(content.encode()).hexdigest()
        fingerprint = hashlib.sha256(
            json.dumps([1, path, digest], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        target = tmp_path / "artifacts" / task_id / path
        target.parent.mkdir(parents=True)
        (tmp_path / "artifacts").chmod(0o700)
        target.parent.chmod(0o700)
        target.write_text(content)
        target.chmod(0o600)
        async with runtime.store.lock:
            await runtime.store.conn.execute(
                "INSERT INTO me3_workspace_operations(task_id,operation_id,revision,fingerprint,path,sha256,size,status,created,updated) VALUES(?,?,?,?,?,?,?,'dispatching',?,?)",
                (
                    task_id,
                    operation_id,
                    1,
                    fingerprint,
                    path,
                    digest,
                    len(content),
                    time.time(),
                    time.time(),
                ),
            )
            await runtime.store.conn.commit()
        recovered = await runtime.store.persist_artifact(task_id, 1, path, content, operation_id)
        assert recovered["recovered"] and recovered["replayed"]
        assert (await runtime.store.verify(task_id))["verified"]

        path2, operation2 = "uncertain.txt", "uncertain-write"
        digest2 = hashlib.sha256(content.encode()).hexdigest()
        fingerprint2 = hashlib.sha256(
            json.dumps([1, path2, digest2], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        async with runtime.store.lock:
            await runtime.store.conn.execute(
                "INSERT INTO me3_workspace_operations(task_id,operation_id,revision,fingerprint,path,sha256,size,status,created,updated) VALUES(?,?,?,?,?,?,?,'dispatching',?,?)",
                (
                    task_id,
                    operation2,
                    1,
                    fingerprint2,
                    path2,
                    digest2,
                    len(content),
                    time.time(),
                    time.time(),
                ),
            )
            await runtime.store.conn.commit()
        with pytest.raises(TaskError, match="workspace_operation_outcome_unknown"):
            await runtime.store.persist_artifact(task_id, 1, path2, content, operation2)
        assert not (target.parent / path2).exists()
        assert not (await runtime.store.verify(task_id))["verified"]
    finally:
        await runtime.close()


async def test_steering_during_inflight_model_call_fences_stale_tool_dispatch(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    async def old_response():
        started.set()
        await release.wait()
        return call(
            "write_artifact", {"path": "stale.txt", "content": "old requirement"}, "old-write"
        )

    model = ScriptedModel(
        [
            old_response,
            call(
                "write_artifact", {"path": "revised.txt", "content": "new requirement"}, "new-write"
            ),
            AIMessage(content="Completed the corrected requirement."),
        ]
    )
    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime)
        await asyncio.wait_for(started.wait(), 2)
        updated = await runtime.steer(
            task["task_id"], 1, "Use the revised requirement and do not use stale output."
        )
        assert updated["revision"] == 2
        release.set()
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "succeeded"
        files = {p.name for p in (tmp_path / "artifacts" / task["task_id"]).iterdir()}
        assert files == {"revised.txt"}
        assert result["revision"] == 2
    finally:
        release.set()
        await runtime.close()


async def test_pause_and_resume_at_graph_checkpoint_keeps_voice_independent(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_final():
        started.set()
        await release.wait()
        return AIMessage(content="A final answer without an objective readback.")

    runtime = await make_runtime(tmp_path, ScriptedModel([delayed_final]))
    try:
        task = await start_task(runtime)
        await asyncio.wait_for(started.wait(), 2)
        paused = await runtime.pause(task["task_id"])
        assert paused["status"] == "pausing"
        release.set()
        await wait_for_status(runtime, task["task_id"], {"paused"})
        resumed = await runtime.resume(task["task_id"])
        assert resumed["status"] == "running"
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "completed_unverified", result["progress"]
        assert result["verification"]["verified"] is False
    finally:
        release.set()
        await runtime.close()


async def test_restart_recovers_interrupted_model_node_from_sqlite_checkpoint(tmp_path):
    started, never_release = asyncio.Event(), asyncio.Event()

    async def interrupted_model():
        started.set()
        await never_release.wait()
        return AIMessage(content="unreachable")

    first = await make_runtime(tmp_path, ScriptedModel([interrupted_model]))
    task = await start_task(first, request_id="restart-case")
    await asyncio.wait_for(started.wait(), 2)
    job = first._jobs[task["task_id"]]
    job.cancel()  # Simulated process loss during the model node, before its checkpoint.
    await asyncio.gather(job, return_exceptions=True)
    await first.close()

    restarted = await make_runtime(
        tmp_path, ScriptedModel([AIMessage(content="Recovered after restart.")])
    )
    try:
        recovered = await restarted.status(task["task_id"])
        assert recovered["status"] == "paused"
        assert any(e["kind"] == "recovered_after_restart" for e in recovered["progress"])
        await restarted.resume(task["task_id"])
        final = await wait_for_status(restarted, task["task_id"])
        assert final["status"] == "completed_unverified"
        assert final["model_calls"] == 2  # The interrupted request remains charged.
    finally:
        await restarted.close()


async def test_approval_is_scoped_expires_and_dispatches_mock_effect_once(tmp_path):
    dispatches = []

    async def dispatch(operation_id, action, details):
        dispatches.append((operation_id, action, details))
        return {"accepted": True, "mode": "mock"}

    model = ScriptedModel(
        [
            call(
                "request_external_action",
                {"action": "publish", "details": "publish approved fixture"},
                "publish-1",
            ),
            AIMessage(content="The approved mock action completed and passed readback."),
        ]
    )
    runtime = await make_runtime(tmp_path, model, effect_dispatcher=dispatch)
    try:
        task = await start_task(
            runtime, "Prepare an artifact and request approval before publication"
        )
        waiting = await wait_for_status(runtime, task["task_id"], {"waiting_approval"})
        approval = waiting["approvals"][0]
        assert dispatches == []
        with pytest.raises(ValueError, match="approval_revision_conflict"):
            await runtime.approve(task["task_id"], approval["approval_id"], 2, "approve")
        approved = await runtime.approve(task["task_id"], approval["approval_id"], 1, "approve")
        assert approved["status"] == "approved"
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "succeeded"
        assert result["verification"]["effects_checked"] == 1
        assert len(dispatches) == 1
        operation_id = dispatches[0][0]
        async with runtime.store.lock:
            async with runtime.store.conn.execute(
                "SELECT status,result FROM me3_effects WHERE task_id=? AND operation_id=?",
                (task["task_id"], operation_id),
            ) as cursor:
                status, persisted = await cursor.fetchone()
        assert status == "complete"
        prior = await runtime.store.finish_effect(task["task_id"], operation_id, {"wrong": True})
        assert prior == json.loads(persisted)
        assert len(dispatches) == 1
    finally:
        await runtime.close()


@pytest.mark.parametrize("decision", ["reject", "expire", "steer"])
async def test_rejected_expired_or_stale_approval_never_dispatches(tmp_path, decision):
    dispatches = []
    model = ScriptedModel(
        [
            call(
                "request_external_action",
                {"action": "write", "details": "a scoped mock operation"},
                "effect-1",
            ),
            AIMessage(content="I completed the task without performing the rejected action."),
        ]
    )
    runtime = await make_runtime(
        tmp_path,
        model,
        effect_dispatcher=lambda *args: _record_dispatch(dispatches, *args),
    )
    try:
        task = await start_task(runtime, request_id="control-" + decision)
        waiting = await wait_for_status(runtime, task["task_id"], {"waiting_approval"})
        approval = waiting["approvals"][0]
        if decision == "expire":
            async with runtime.store.lock:
                await runtime.store.conn.execute(
                    "UPDATE me3_approvals SET expires=? WHERE approval_id=?",
                    (time.time() - 1, approval["approval_id"]),
                )
                await runtime.store.conn.commit()
            await runtime.approve(task["task_id"], approval["approval_id"], 1, "approve")
        elif decision == "steer":
            await runtime.steer(task["task_id"], 1, "Do not perform the proposed operation.")
        else:
            await runtime.approve(task["task_id"], approval["approval_id"], 1, "reject")
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "completed_unverified", result["progress"]
        assert dispatches == []
    finally:
        await runtime.close()


async def test_unknown_dispatch_is_never_replayed_after_restart(tmp_path):
    dispatches = []
    dispatch_started = asyncio.Event()
    wait_for_crash = asyncio.Event()

    async def ambiguous_dispatch(operation_id, _action, _details):
        dispatches.append(operation_id)
        dispatch_started.set()
        await wait_for_crash.wait()
        return {"unreachable": True}

    first = await make_runtime(
        tmp_path,
        ScriptedModel(
            [call("request_external_action", {"action": "write", "details": "mock"}, "ambiguous-1")]
        ),
        effect_dispatcher=ambiguous_dispatch,
    )
    task = await start_task(first, request_id="ambiguous")
    waiting = await wait_for_status(first, task["task_id"], {"waiting_approval"})
    approval_wait = first._jobs.get(task["task_id"])
    if approval_wait:
        await approval_wait
    approval = waiting["approvals"][0]
    await first.approve(task["task_id"], approval["approval_id"], 1, "approve")
    await asyncio.wait_for(dispatch_started.wait(), 3)
    job = first._jobs[task["task_id"]]
    job.cancel()  # Process-loss simulation after durable dispatch claim, before receipt.
    await asyncio.gather(job, return_exceptions=True)
    await first.close()

    restarted = await make_runtime(
        tmp_path, ScriptedModel([AIMessage(content="Outcome remains unknown.")])
    )
    try:
        recovered = await restarted.status(task["task_id"])
        assert recovered["status"] == "paused"
        assert dispatches == [task["task_id"] + ":ambiguous-1"]
        await restarted.resume(task["task_id"])
        result = await wait_for_status(restarted, task["task_id"])
        assert result["status"] == "completed_unverified"
        assert result["verification"]["verified"] is False
        assert len(dispatches) == 1
        async with restarted.store.lock:
            async with restarted.store.conn.execute(
                "SELECT status FROM me3_effects WHERE task_id=?", (task["task_id"],)
            ) as cursor:
                assert (await cursor.fetchone())[0] == "unknown"
    finally:
        await restarted.close()


async def _record_dispatch(dispatches, operation_id, action, details):
    dispatches.append((operation_id, action, details))
    return {"accepted": True, "mode": "mock"}


async def test_cancel_revokes_pending_approval_without_resuming_effect(tmp_path):
    dispatched = []
    runtime = await make_runtime(
        tmp_path,
        ScriptedModel(
            [
                call(
                    "request_external_action",
                    {"action": "publish", "details": "pending"},
                    "pending-1",
                )
            ]
        ),
        effect_dispatcher=lambda *args: _record_dispatch(dispatched, *args),
    )
    try:
        task = await start_task(runtime, request_id="cancel-pending")
        await wait_for_status(runtime, task["task_id"], {"waiting_approval"})
        result = await runtime.cancel(task["task_id"])
        assert result["status"] == "cancelled"
        assert dispatched == []
    finally:
        await runtime.close()


async def test_cancel_missing_task_returns_actionable_not_found(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([]))
    try:
        with pytest.raises(TaskError, match="task_not_found"):
            await runtime.control("cancel", {"task_id": "task_" + "f" * 24})
    finally:
        await runtime.close()


async def test_model_tool_schemas_hide_runtime_context_and_forbid_unregistered_operations(tmp_path):
    runtime = await make_runtime(tmp_path, ScriptedModel([AIMessage(content="unused")]))
    try:
        names = {tool["function"]["name"] for tool in runtime._openai_tools}
        assert names == {
            "get_runtime_time",
            "web_search",
            "retrieve_source",
            "verify_evidence",
            "write_artifact",
            "read_artifact",
            "list_workspace",
            "create_directory",
            "request_external_action",
        }
        for tool in runtime._openai_tools:
            assert "runtime" not in tool["function"]["parameters"].get("properties", {})
    finally:
        await runtime.close()


async def test_live_provider_call_budget_is_closed_by_default(tmp_path):
    runtime = await make_runtime(tmp_path, None)
    try:
        with pytest.raises(ValueError, match="task_model_disabled"):
            await runtime._call_model([])
        assert runtime.budget.requests == 0
    finally:
        await runtime.close()


async def test_task_database_requires_private_directory_and_file_modes(tmp_path):
    private = await make_runtime(tmp_path, ScriptedModel([]))
    try:
        assert private.database.parent.stat().st_mode & 0o777 == 0o700
        assert private.database.stat().st_mode & 0o777 == 0o600
    finally:
        await private.close()

    insecure_dir = tmp_path / "insecure"
    insecure_dir.mkdir(mode=0o755)
    insecure_dir.chmod(0o755)
    insecure = TaskRuntime(
        load_backends()["llm"],
        database=insecure_dir / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts2",
        model=ScriptedModel([]),
    )
    with pytest.raises(TaskError, match="task_database_directory_permissions"):
        await insecure.start_runtime()


async def test_relative_artifact_root_is_normalized_before_readback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = await make_runtime(
        tmp_path,
        ScriptedModel(
            [
                call(
                    "write_artifact", {"path": "result.txt", "content": "relative fixture"}, "write"
                ),
                call("read_artifact", {"path": "result.txt"}, "read"),
                AIMessage(content="The relative artifact was read back."),
            ]
        ),
        artifact_root=Path("var/scratch/relative-artifacts"),
    )
    try:
        task = await start_task(runtime, request_id="relative-artifact")
        result = await wait_for_status(runtime, task["task_id"])
        assert result["status"] == "succeeded"
        assert (
            tmp_path / "var/scratch/relative-artifacts" / task["task_id"] / "result.txt"
        ).read_text() == "relative fixture"
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "endpoint,expect_groq_parameters,model",
    [
        ("https://api.groq.com/openai/v1/chat/completions", True, "openai/gpt-oss-120b"),
        ("https://openrouter.ai/api/v1/chat/completions", False, "openai/gpt-oss-120b"),
        ("https://openrouter.ai/api/v1/chat/completions", False, "anthropic/claude-haiku-5.5"),
    ],
)
async def test_openai_compatible_streamed_tool_fragments_are_reassembled_offline(
    tmp_path, monkeypatch, endpoint, expect_groq_parameters, model
):
    import me3_voice.task_runtime as module
    from langchain_core.messages import HumanMessage

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint=endpoint,
        auth_env="ME3_GROQ_TEST_KEY" if expect_groq_parameters else "ME3_OPENROUTER_TEST_KEY",
        model=model,
        allow_live=True,
        request_limit=1,
        max_input_tokens=32000,
        max_input_tokens_total=64000,
    )
    credentials = {spec.auth_env: "offline-fixture-not-a-real-key"}
    monkeypatch.setenv(
        "ME3_REQUEST_RUN_ID",
        "offline-groq-fragments-01"
        if expect_groq_parameters
        else ("offline-haiku-fragments-01" if "haiku" in model else "offline-openrouter-fragments-01"),
    )
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "provider-requests.sqlite"))
    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)

            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call-fragment",
                                        function=SimpleNamespace(
                                            name="web_search", arguments='{"query":"today'
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id=None,
                                        function=SimpleNamespace(
                                            name=None, arguments=' news","max_results":3}'
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )

            return chunks()

        async def close(self):
            pass

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    runtime = TaskRuntime(
        spec,
        credentials,
        budget=LiveTransport(spec, credentials),
        database=tmp_path / "stream.sqlite",
        artifact_root=tmp_path / "artifacts",
        max_completion_tokens=8192 if "haiku" in model else 1024,
    )
    # This contract test exercises the provider/request adapter only; leave
    # SQLite graph startup to the task-runtime integration tests.
    runtime._openai_tools = []
    try:
        answer = await runtime._call_model([HumanMessage(content="Research the current result")])
        assert answer.tool_calls == [
            {
                "name": "web_search",
                "args": {"query": "today news", "max_results": 3},
                "id": "call-fragment",
                "type": "tool_call",
            }
        ]
        assert calls[0]["stream"] is True
        assert calls[0]["parallel_tool_calls"] is False
        assert calls[0]["model"] == model
        if model == "anthropic/claude-haiku-5.5":
            assert "max_tokens" in calls[0] and calls[0]["max_tokens"] == 8192
            assert "max_completion_tokens" not in calls[0]
        if expect_groq_parameters:
            assert calls[0]["reasoning_effort"] == "low"
            assert calls[0]["extra_body"] == {"include_reasoning": False}
        else:
            assert "reasoning_effort" not in calls[0]
            assert "extra_body" not in calls[0]
        assert runtime.budget.requests == 1
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "content,expected_error",
    [
        (
            "<script>// Game state const board = [];</script>",
            "interactive_script_missing_or_disabled",
        ),
        (
            "<div id=&#34;board&#34;></div><script>document.getElementById('board');</script>",
            "interactive_html_attribute_quoting_invalid",
        ),
    ],
)
async def test_streamed_html_tool_json_preserves_source_then_write_guard_rejects(
    tmp_path, monkeypatch, content, expected_error
):
    import me3_voice.task_runtime as module
    from langchain_core.messages import HumanMessage

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        auth_env="ME3_OPENROUTER_OFFLINE_FIXTURE",
        model="openai/gpt-oss-120b",
        allow_live=True,
        request_limit=1,
        max_input_tokens=32000,
        max_input_tokens_total=64000,
    )
    credentials = {spec.auth_env: "offline-fixture-not-a-real-key"}
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "offline-html-tool-arguments")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "provider-requests.sqlite"))
    arguments = json.dumps(
        {"path": "game.html", "content": content}, ensure_ascii=False, separators=(",", ":")
    )
    pieces = [arguments[:17], arguments[17:41], arguments[41:]]

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            assert kwargs["stream"] is True

            async def chunks():
                for index, piece in enumerate(pieces):
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="tool_calls" if index == len(pieces) - 1 else None,
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="offline-html-write" if index == 0 else None,
                                            function=SimpleNamespace(
                                                name="write_artifact" if index == 0 else None,
                                                arguments=piece,
                                            ),
                                        )
                                    ],
                                ),
                            )
                        ],
                        usage=None,
                    )

            return chunks()

        async def close(self):
            pass

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    runtime = TaskRuntime(
        spec,
        credentials,
        budget=LiveTransport(spec, credentials),
        database=tmp_path / "streamed-html.sqlite",
        artifact_root=tmp_path / "artifacts",
    )
    runtime._openai_tools = []
    try:
        parsed = await runtime._call_model([HumanMessage(content="Create a playable page")])
        assert parsed.tool_calls[0]["name"] == "write_artifact"
        assert parsed.tool_calls[0]["args"]["content"] == content
        with pytest.raises(TaskError, match=expected_error):
            validate_interactive_artifact("game.html", parsed.tool_calls[0]["args"]["content"])
        assert runtime.budget.requests == 1  # SDK client is a local fixture; no HTTP call occurs.
    finally:
        await runtime.close()


async def test_task_request_usage_and_budgets_are_durable_and_observable(tmp_path, monkeypatch):
    import sqlite3
    import me3_voice.task_runtime as module
    from langchain_core.messages import HumanMessage

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        auth_env="ME3_TASK_USAGE_FIXTURE_KEY",
        model="openai/gpt-oss-120b",
        allow_live=True,
        request_limit=4,
        max_input_tokens=32000,
        max_input_tokens_total=64000,
    )
    ledger = tmp_path / "provider-requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "offline-task-usage-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **_kwargs):
            async def chunks():
                yield SimpleNamespace(
                    usage=SimpleNamespace(
                        model_dump=lambda: {
                            "prompt_tokens": 321,
                            "completion_tokens": 18,
                        }
                    ),
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(content="Task is complete.", tool_calls=[]),
                        )
                    ],
                )

            return chunks()

        async def close(self):
            return None

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    runtime = TaskRuntime(
        spec,
        {"ME3_TASK_USAGE_FIXTURE_KEY": "offline-fixture-not-a-real-key"},
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        max_model_calls=4,
    )
    runtime._openai_tools = []
    try:
        await runtime.start_runtime()
        task, created = await runtime.store.create(
            "Create a small page",
            request_id="usage-fixture-task",
            turn_id=1,
            call_id="usage-fixture-task",
        )
        assert created
        await runtime.store.update(task["task_id"], status="running", model_calls=1)
        answer = await runtime._call_model(
            [HumanMessage(content="Create a small page")],
            task_id=task["task_id"],
            model_call_number=1,
        )
        assert answer.content == "Task is complete."

        status = await runtime.status(task["task_id"])
        budgets = status["task_budgets"]
        assert budgets["model_calls"] == {"used": 1, "limit": 4}
        assert budgets["provider_requests_reserved"] == 1
        assert budgets["estimated_input_tokens_reserved"] > 321
        assert budgets["actual_input_tokens"] == 321
        assert budgets["actual_output_tokens"] == 18
        assert budgets["actual_usage_requests"] == 1
        usage_event = next(
            row for row in status["progress"] if row["kind"] == "model_request_usage"
        )
        assert usage_event["payload"]["serialized_request_bytes"] > 0
        assert usage_event["payload"]["actual_input_tokens"] == 321
        with sqlite3.connect(ledger) as db:
            assert db.execute(
                "SELECT status,input_tokens_reserved,input_tokens_actual,output_tokens_actual FROM provider_requests"
            ).fetchone() == (
                "success",
                budgets["estimated_input_tokens_reserved"],
                321,
                18,
            )
    finally:
        await runtime.close()


async def test_langgraph_input_budget_failure_is_reported_without_sdk_dispatch(
    tmp_path, monkeypatch
):
    import sqlite3
    from langchain_core.messages import HumanMessage
    import me3_voice.task_runtime as module

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        auth_env="ME3_TASK_BUDGET_TEST_KEY",
        model="openai/gpt-oss-120b",
        allow_live=True,
        request_limit=5,
        max_input_tokens=500,
        max_input_tokens_total=500,
    )
    credentials = {"ME3_TASK_BUDGET_TEST_KEY": "offline-fixture-not-a-real-key"}
    ledger = tmp_path / "provider-requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "task-budget-test-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setattr(
        module,
        "conservative_input_token_estimate",
        lambda _params: (100, 400),
    )
    runtime = TaskRuntime(
        spec,
        credentials,
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        model=None,
    )
    # Tool schemas would normally be constructed by the compiled graph. Keep
    # this direct pre-dispatch guard test below that boundary: no checkpoint
    # process/database or provider client is needed.
    runtime._openai_tools = []
    _, seeded = runtime.budget.authorize(
        input_tokens=450, input_token_limit=500, input_tokens_total_limit=500
    )
    runtime.budget.finish_request(seeded, "failure", "fixture_consumed")
    with pytest.raises(TaskError, match="input_token_budget_exhausted"):
        await runtime._call_model([HumanMessage(content="Continue the task.")])
    assert runtime._client is None
    with sqlite3.connect(ledger) as db:
        assert db.execute(
            "SELECT status,input_tokens_reserved FROM provider_requests"
        ).fetchall() == [("failure", 450)]


async def test_native_voice_mcp_remains_responsive_while_durable_task_runs(tmp_path):
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.providers import ToolCall
    from me3_voice.speech import SpeechClient

    model_started, release_model = asyncio.Event(), asyncio.Event()

    async def slow_task_model(_messages, _tools):
        model_started.set()
        await release_model.wait()
        return AIMessage(content="Task completed after the voice reply.")

    class StartTaskFromVoice:
        def __init__(self):
            self.calls = 0

        async def respond(self, _history):
            self.calls += 1
            if self.calls == 1:
                yield ToolCall(
                    "voice-start-1", "start_task", {"goal": "Complete an unfamiliar task"}
                )
            else:
                yield "I started the task in the background."

    def task_factory(spec, credentials, budget, research):
        return TaskRuntime(
            spec,
            credentials,
            budget=budget,
            research=research,
            database=tmp_path / "concurrent-tasks.sqlite",
            artifact_root=tmp_path / "concurrent-artifacts",
            model=slow_task_model,
        )

    state = NativeState(
        speech_factory=lambda: SpeechClient(transcript="hello while task runs", timeout=5),
        task_runtime_factory=task_factory,
    )
    try:
        async with NativeApplication(
            state, MemorySink(), annotated=True, upstream=StartTaskFromVoice()
        ) as app:
            # This uses both project-owned stdio MCP servers and the regular
            # native Pipecat tool call to start a separate LangGraph task.
            await app.utterance(b"\1\0" * 512)
            await asyncio.wait_for(model_started.wait(), 3)
            assert any(event["event"] == "turn_completed" for event in app.events)
            assert any(
                event["event"] == "tool_completed" and event.get("tool_name") == "start_task"
                for event in app.events
            )
            assert any(
                event["event"] == "task_update" and event.get("kind") == "started"
                for event in app.events
            )
            task_id = next(iter(state.task_runtime.store.rows))
            assert (await state.task_runtime.status(task_id))["status"] == "running"
            release_model.set()
            result = await wait_for_status(state.task_runtime, task_id)
            assert result["status"] == "completed_unverified"
            assert state.speech.tools["asr"] >= {"asr_open", "asr_append", "asr_finish", "cancel"}
            assert state.speech.tools["tts"] >= {"tts_open", "tts_next", "cancel"}
    finally:
        release_model.set()
        await state.close()


@pytest.mark.parametrize("mode", ["model", "effect"])
async def test_actual_sigkill_restart_retains_checkpoint_and_never_replays_unknown_effect(
    tmp_path, mode
):
    import os
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "tests/phase5_process_worker.py",
        str(tmp_path),
        mode,
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        for _ in range(2000):
            if (tmp_path / "ready").exists():
                break
            if process.returncode is not None:
                raise AssertionError((await process.stderr.read()).decode())
            await asyncio.sleep(0.01)
        assert (tmp_path / "ready").exists(), "worker never reached crash window"
        task_id = (tmp_path / "task_id").read_text()
        process.kill()
        await process.wait()
        assert process.returncode == -9
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    dispatched = []

    async def forbidden_dispatch(*args):
        dispatched.append(args)
        raise AssertionError("uncertain effect was replayed")

    runtime = await make_runtime(
        tmp_path,
        ScriptedModel([AIMessage(content="Recovered with limitations.")]),
        effect_dispatcher=forbidden_dispatch,
    )
    try:
        assert (await runtime.status(task_id))["status"] == "paused"
        async with runtime.store.conn.execute("PRAGMA integrity_check") as cursor:
            assert (await cursor.fetchone())[0] == "ok"
        await runtime.resume(task_id)
        result = await wait_for_status(runtime, task_id, timeout=15)
        assert result["status"] == "completed_unverified"
        assert not dispatched
        if mode == "effect":
            async with runtime.store.conn.execute("SELECT status FROM me3_effects") as cursor:
                assert (await cursor.fetchone())[0] == "unknown"
    finally:
        await runtime.close()


async def test_actual_sigkill_after_workspace_replace_recovers_by_readback(tmp_path):
    import os
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "tests/phase5_process_worker.py",
        str(tmp_path),
        "workspace",
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        for _ in range(2000):
            if (tmp_path / "ready").exists():
                break
            if process.returncode is not None:
                raise AssertionError((await process.stderr.read()).decode())
            await asyncio.sleep(0.01)
        assert (tmp_path / "ready").exists(), "worker did not reach the post-replace crash window"
        task_id = (tmp_path / "task_id").read_text()
        artifact = tmp_path / "artifacts" / task_id / "site" / "index.html"
        assert artifact.read_text() == "<!doctype html><h1>Crash safe</h1>"
        process.kill()
        await process.wait()
        assert process.returncode == -9
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()

    runtime = await make_runtime(
        tmp_path,
        ScriptedModel(
            [AIMessage(content="Recovered the file write after exact content readback.")]
        ),
    )
    try:
        assert (await runtime.status(task_id))["status"] == "paused"
        await runtime.resume(task_id)
        result = await wait_for_status(runtime, task_id, timeout=15)
        assert result["status"] == "succeeded"
        assert result["verification"]["artifacts_checked"] == 1
        assert artifact.read_text() == "<!doctype html><h1>Crash safe</h1>"
        async with runtime.store.conn.execute(
            "SELECT status FROM me3_workspace_operations WHERE task_id=?", (task_id,)
        ) as cursor:
            assert (await cursor.fetchone())[0] == "complete"
    finally:
        await runtime.close()


@pytest.mark.parametrize("control", ["steer", "cancel", "pause"])
async def test_control_during_verification_cannot_be_overwritten(tmp_path, control):
    runtime = await make_runtime(
        tmp_path, ScriptedModel([AIMessage(content="old"), AIMessage(content="new")])
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.store.verify

    async def delayed(task_id):
        entered.set()
        await release.wait()
        return await original(task_id)

    runtime.store.verify = delayed
    try:
        task = await start_task(runtime)
        task_id = task["task_id"]
        await asyncio.wait_for(entered.wait(), 5)
        if control == "steer":
            await runtime.steer(task_id, 1, "new requirement")
        else:
            await getattr(runtime, control)(task_id)
        release.set()
        result = await wait_for_status(runtime, task_id, statuses=TERMINAL | {"paused"})
        if control == "steer":
            assert result["revision"] == 2 and result["result"] == "new"
        else:
            assert result["status"] == ("paused" if control == "pause" else "cancelled")
            assert result["result"] == ""
    finally:
        release.set()
        await runtime.close()


async def test_concurrent_runtime_startup_uses_one_connection(tmp_path):
    runtime = TaskRuntime(
        load_backends()["llm"],
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        model=ScriptedModel([]),
    )
    try:
        await asyncio.gather(*(runtime.start_runtime() for _ in range(10)))
        assert runtime.graph is not None
        assert runtime.store.conn is runtime.checkpointer.conn
    finally:
        await runtime.close()


@pytest.mark.parametrize("control", ["cancel", "pause"])
async def test_human_control_wins_while_approval_checkpoint_returns(tmp_path, control):
    runtime = await make_runtime(
        tmp_path,
        ScriptedModel(
            [call("request_external_action", {"action": "publish", "details": "mock"}, "pending")]
        ),
    )
    entered, release = asyncio.Event(), asyncio.Event()
    invoke = runtime.graph.ainvoke

    async def delayed(*args, **kwargs):
        result = await invoke(*args, **kwargs)
        interrupts = result.get("__interrupt__", ())
        if interrupts and interrupts[0].value.get("kind") == "approval_request":
            entered.set()
            await release.wait()
        return result

    runtime.graph.ainvoke = delayed
    try:
        task = await start_task(runtime)
        await asyncio.wait_for(entered.wait(), 5)
        await getattr(runtime, control)(task["task_id"])
        job = runtime._jobs[task["task_id"]]
        release.set()
        await job
        assert (await runtime.status(task["task_id"]))["status"] == (
            "cancelled" if control == "cancel" else "paused"
        )
    finally:
        release.set()
        await runtime.close()


@pytest.mark.parametrize("control", ["cancel", "pause"])
async def test_inflight_model_failure_preserves_human_control(tmp_path, control):
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(*args):
        entered.set()
        await release.wait()
        raise RuntimeError("injected provider failure")

    runtime = await make_runtime(tmp_path, model)
    try:
        task = await start_task(runtime)
        await asyncio.wait_for(entered.wait(), 5)
        await getattr(runtime, control)(task["task_id"])
        release.set()
        result = await wait_for_status(runtime, task["task_id"], statuses=TERMINAL | {"paused"})
        assert result["status"] == ("cancelled" if control == "cancel" else "paused")
    finally:
        release.set()
        await runtime.close()


async def test_second_runtime_cannot_recover_a_live_owners_database(tmp_path):
    first = await make_runtime(tmp_path, ScriptedModel([]))
    second = TaskRuntime(
        load_backends()["llm"],
        database=tmp_path / "tasks.sqlite",
        artifact_root=tmp_path / "artifacts",
        model=ScriptedModel([]),
    )
    try:
        with pytest.raises(TaskError, match="task_database_already_owned"):
            await second.start_runtime()
        await first.close()
        await second.start_runtime()
        assert second.graph is not None
    finally:
        await first.close()
        await second.close()
