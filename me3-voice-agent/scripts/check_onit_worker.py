"""Installed-Onit task/tool loop against an in-memory SSE peer; no live API."""

import asyncio
import argparse
import json
import os
import tempfile
import sys
import time
from pathlib import Path

import httpx

from me3_voice import onit_worker


class Peer(httpx.AsyncBaseTransport):
    calls = []
    task_kind = "artifact"

    def __init__(self, **kwargs):
        pass

    async def handle_async_request(self, request):
        assert request.extensions.get("timeout", {}).get("read") == (
            15 if self.task_kind == "tool-free" else 10
        )
        body = json.loads(await request.aread())
        self.calls.append(body)
        if (
            self.task_kind == "timeout"
            or self.task_kind == "after-tool-timeout"
            and len(self.calls) == 2
        ):
            raise httpx.ReadTimeout("Offline controlled timeout", request=request)
        if self.task_kind == "dgx-contract":
            assert body["model"] == "Qwen/Qwen3.8-27B"
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            assert "provider" not in body
        else:
            assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0}
            assert body["model"].endswith(":free")
        expected = [] if self.task_kind == "tool-free" else ["write_artifact"]
        if self.task_kind in {"project", "project-site-var"}:
            expected.append("create_project_directory")
        if self.task_kind == "tavily":
            expected.append("tavily_search")
        if self.task_kind == "github":
            expected.append("github_repo_read")
        assert [tool["function"]["name"] for tool in body.get("tools", [])] == expected
        if self.task_kind == "tool-free":
            assert "tools" not in body and "tool_choice" not in body
            delta, finish = {"content": "OK"}, "stop"
        elif len(self.calls) == 1:
            name = {
                "project": "create_project_directory",
                "project-site-var": "create_project_directory",
                "tavily": "tavily_search",
                "github": "github_repo_read",
            }.get(self.task_kind, "write_artifact")
            arguments = {
                "project": {"name": "ME3-Test"},
                "project-site-var": {
                    "name": "OfflineVarProject",
                    "files": [
                        {
                            "filename": "index.html",
                            "content": "<h1>Offline root-boundary fixture</h1>",
                        }
                    ],
                },
                "tavily": {"query": "current weather in Manila"},
                "github": {"action": "get", "name": "example/swift-demo"},
            }.get(
                self.task_kind,
                {"filename": "index.html", "content": "<h1>Swift offline draft</h1>"},
            )
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_draft",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ]
            }
            finish = "tool_calls"
        else:
            assert any(m.get("role") == "tool" for m in body["messages"])
            delta, finish = {"content": "I completed the local task."}, "stop"
        event = {
            "id": "offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        done = {**event, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
        wire = f"data: {json.dumps(event)}\n\ndata: {json.dumps(done)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=wire.encode(),
            request=request,
        )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="var/results/onit-task-offline-benchmark.json")
    args = parser.parse_args()

    class OfflineTavilyTool:
        url = onit_worker.TavilySearchTool.url
        tool_item = onit_worker.TavilySearchTool.tool_item

        def __init__(self, key, credits):
            assert key == "offline-tavily-key" and credits == 1
            self.credits_reserved = 0

        def get_tool(self):
            return self.tool_item

        async def __call__(self, log_handler=None, **kwargs):
            self.credits_reserved += 1
            return json.dumps(
                {
                    "answer": "Mock weather result",
                    "results": [{"url": "https://example.test/weather"}],
                }
            )

    class OfflineGitHubReadTool:
        url = onit_worker.GitHubReadTool.url
        tool_item = onit_worker.GitHubReadTool.tool_item

        def __init__(self, token):
            assert token == "offline-github-token"
            self.calls = 0

        def get_tool(self):
            return self.tool_item

        async def __call__(self, log_handler=None, **kwargs):
            assert kwargs == {"action": "get", "name": "example/swift-demo"}
            self.calls += 1
            return json.dumps({"repository": {"full_name": "example/swift-demo"}})

    onit_worker.TavilySearchTool = OfflineTavilyTool
    onit_worker.GitHubReadTool = OfflineGitHubReadTool
    onit_worker._TAVILY_KEY = "offline-tavily-key"
    onit_worker._GITHUB_TOKEN = "offline-github-token"
    httpx.AsyncHTTPTransport = Peer
    Path("var/scratch").mkdir(parents=True, exist_ok=True)
    onit_worker.prewarm_sdk()
    with tempfile.TemporaryDirectory(prefix="onit-tool-", dir="var/scratch") as directory:
        os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
        result = await onit_worker.run(
            {
                "task": "Create a one-file website called index.html",
                "receipt_checkpoints": False,
                "record_final_reply": True,
                "workspace": str(Path(directory).resolve()),
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "timeout": 10,
                "max_requests": 3,
            }
        )
        assert result["artifacts"] == ["index.html"], result
        assert (Path(directory) / "index.html").read_text() == "<h1>Swift offline draft</h1>"
        assert result["requests"] == 2
        assert (
            json.loads((Path(directory) / "final-reply.json").read_text())["answer"]
            == result["answer"]
        )
        audit = [
            json.loads(line)
            for line in (Path(directory) / "request-attempts.jsonl").read_text().splitlines()
        ]
        assert [row["attempt"] for row in audit] == [1, 2]
        assert all(set(row) == {"attempt", "monotonic"} for row in audit)
        stages = json.loads((Path(directory) / "worker-stages.json").read_text())
        assert stages["last_stage"] == "worker_complete" and stages["attempts"] == 2
        assert {
            "response_headers",
            "response_body_started",
            "tool_execution_complete",
            "tool_result_submission",
        } <= {e["stage"] for e in stages["events"]}
    Peer.calls = []
    Peer.task_kind = "project"
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="onit-project-", dir="var/scratch") as directory:
        sandbox_root = Path(directory).resolve()
        project = sandbox_root / "ME3-VoiceAgent"
        workspace = project / "var" / "task"
        workspace.mkdir(parents=True)
        try:
            os.chdir(project)
            os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
            project_result = await onit_worker.run(
                {
                    "task": "Create a new project directory named ME3-Test",
                    "workspace": str(workspace),
                    "project_output_root": str(sandbox_root),
                    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                    "model": "nvidia/nemotron-3.5-lightning:free",
                    "timeout": 10,
                    "max_requests": 3,
                }
            )
        finally:
            os.chdir(original_cwd)
        assert project_result["mode"] == "local-project"
        assert project_result["projects"] == ["ME3-Test"]
        assert (sandbox_root / "ME3-Test").is_dir()
        assert project_result["requests"] == 2
    Peer.calls = []
    Peer.task_kind = "project-site-var"
    with tempfile.TemporaryDirectory(prefix="onit-var-project-", dir="var/scratch") as directory:
        base = Path(directory).resolve()
        workspace = base / "workspace"
        workspace.mkdir()
        projects = base / "projects"
        projects.mkdir()
        os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
        var_project_result = await onit_worker.run(
            {
                "task": "Create a static site fixture",
                "workspace": str(workspace),
                "project_output_root": str(projects),
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "timeout": 10,
                "max_requests": 3,
                "receipt_checkpoints": False,
            }
        )
        assert (
            var_project_result["mode"] == "local-project-site"
            and var_project_result["requests"] == 2
        )
        assert (
            projects / "OfflineVarProject/index.html"
        ).read_text() == "<h1>Offline root-boundary fixture</h1>"
    Peer.calls = []
    Peer.task_kind = "tavily"
    with tempfile.TemporaryDirectory(prefix="onit-search-", dir="var/scratch") as directory:
        os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
        search_result = await onit_worker.run(
            {
                "task": "Find the current weather in Manila",
                "workspace": str(Path(directory).resolve()),
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "timeout": 10,
                "max_requests": 3,
                "tavily_credit_budget": 1,
            }
        )
    assert search_result["tavily_credits_used"] == 1
    assert any(message.get("role") == "tool" for message in Peer.calls[1]["messages"])
    Peer.calls = []
    Peer.task_kind = "github"
    with tempfile.TemporaryDirectory(prefix="onit-github-", dir="var/scratch") as directory:
        os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
        github_result = await onit_worker.run(
            {
                "task": "Read details for example/swift-demo",
                "workspace": str(Path(directory).resolve()),
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "timeout": 10,
                "max_requests": 3,
                "github_read_enabled": True,
            }
        )
    assert github_result["github_reads_used"] == 1
    assert any(message.get("role") == "tool" for message in Peer.calls[1]["messages"])
    Peer.calls = []
    Peer.task_kind = "tool-free"
    with tempfile.TemporaryDirectory(prefix="onit-tool-free-", dir="var/scratch") as directory:
        os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
        tool_free_result = await onit_worker.run(
            {
                "task": "Reply exactly OK.",
                "workspace": str(Path(directory).resolve()),
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "timeout": 15,
                "max_requests": 1,
                "tool_free": True,
            }
        )
    assert len(Peer.calls) == tool_free_result["requests"] == 1
    assert tool_free_result["answer"] == "OK" and not tool_free_result["artifacts"]
    Peer.calls = []
    Peer.task_kind = "dgx-contract"
    with tempfile.TemporaryDirectory(prefix="onit-dgx-contract-", dir="var/scratch") as directory:
        os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
        dgx_result = await onit_worker.run(
            {
                "task": "Create one draft file",
                "workspace": str(Path(directory).resolve()),
                "endpoint": "http://202.92.159.240:8001/v1/chat/completions",
                "model": "Qwen/Qwen3.8-27B",
                "timeout": 10,
                "max_requests": 3,
                "receipt_checkpoints": False,
            }
        )
        assert dgx_result["requests"] == 2 and dgx_result["artifacts"] == ["index.html"]
    failure_checks = []
    for kind in ("timeout", "after-tool-timeout"):
        Peer.calls = []
        Peer.task_kind = kind
        with tempfile.TemporaryDirectory(prefix="onit-failure-", dir="var/scratch") as directory:
            os.environ["ME3_ONIT_KEY"] = "offline-dummy-key"
            failed_result = await onit_worker.run(
                {
                    "task": "Create one index.html draft",
                    "workspace": str(Path(directory).resolve()),
                    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                    "model": "nvidia/nemotron-3.5-lightning:free",
                    "timeout": 10,
                    "max_requests": 3,
                    "receipt_checkpoints": False,
                    "record_final_reply": True,
                }
            )
            observed = json.loads((Path(directory) / "worker-stages.json").read_text())
            assert any(
                e["stage"] == "transport_failed" and e["error_type"] == "ReadTimeout"
                for e in observed["events"]
            )
            if kind == "timeout":
                assert failed_result == {"error": "NoOutput", "requests": 1}
                assert not (Path(directory) / "index.html").exists()
            else:
                assert (
                    failed_result["mode"] == "local-artifact"
                    and failed_result["final_reply_status"] == "missing"
                )
                assert (
                    failed_result["requests"] == 2
                    and (Path(directory) / "index.html").read_text()
                    == "<h1>Swift offline draft</h1>"
                )
            failure_checks.append(
                {"scenario": kind, "requests": len(Peer.calls), "result": failed_result}
            )
    print(
        f"Offline Onit tool loop passed; task phase {result['worker_ms']:.1f} ms. Measuring cold worker…",
        flush=True,
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path("var/scratch").resolve()),
        "PYTHONPATH": str(Path("src").resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "ME3_ONIT_KEY": "offline-dummy-key",
    }
    start = time.monotonic()
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-m",
        "me3_voice.onit_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    try:
        async with asyncio.timeout(90):
            assert await child.stdout.readline() == b"READY\n"
        startup_ms = round((time.monotonic() - start) * 1000, 3)
    except TimeoutError:
        child.kill()
        await child.wait()
        startup_ms = None
    finally:
        if child.returncode is None:
            child.stdin.close()
            await child.wait()
    report = {
        "mode": "offline-wire-fixture",
        "failure_checks": failure_checks,
        "dgx_thinking_control_contract": {
            "requests": dgx_result["requests"],
            "enable_thinking": False,
        },
        "onit_cold_worker_ready_ms": startup_ms,
        "onit_warm_tool_task_ms": result["worker_ms"],
        "model_requests": result["requests"],
        "request_attempt_audit": audit,
        "worker_stages": stages,
        "onit_metrics": result.get("onit_metrics"),
        "verified_draft_files": 1,
        "verified_project_directories": 1,
        "verified_var_project": {
            "mode": var_project_result["mode"],
            "requests": var_project_result["requests"],
            "file_bytes_verified": True,
        },
        "verified_tavily_tool_calls": 1,
        "verified_github_read_tool_calls": 1,
        "tool_free_probe": {
            "requests": tool_free_result["requests"],
            "answer": tool_free_result["answer"],
            "no_advertised_tools": True,
        },
        "acoustic_latency_ms": None,
        "limitations": "Offline SSE fixture; no real model or Pi. Cold import is a startup measurement, not a user-turn latency if prewarmed.",
    }
    output = Path(args.output).resolve()
    if not output.is_relative_to((Path.cwd() / "var/results").resolve()):
        raise ValueError("offline benchmark output must be under var/results/")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Preserve previous benchmark: {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Installed Onit offline tool loop passed; cold worker ready {startup_ms if startup_ms is not None else '>90000'} ms, warm fixture task {result['worker_ms']:.1f} ms; {output}"
    )


if __name__ == "__main__":
    asyncio.run(main())
