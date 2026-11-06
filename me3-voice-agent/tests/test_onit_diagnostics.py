import asyncio
import json
import pytest
from me3_voice.onit_diagnostics import StageJournal, send_observed


class Response:
    def __init__(self, stream):
        self.stream = stream

    async def aread(self):
        return b"".join([chunk async for chunk in self.stream])

    async def aclose(self):
        await self.stream.aclose()


class CompleteBody:
    async def __aiter__(self):
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        pass


@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("headers", "request_submission"),
        ("body", "response_headers"),
        ("after-first-byte", "response_body_started"),
    ],
)
async def test_stalled_peer_retains_last_confirmed_stage_and_closes(boundary, expected, tmp_path):
    closed = []

    class Body:
        async def __aiter__(self):
            if boundary == "after-first-byte":
                yield b": keepalive\n\n"
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.append(True)

    class Peer:
        async def handle_async_request(self, request):
            if boundary == "headers":
                await asyncio.Event().wait()
            return Response(Body())

    journal = StageJournal(tmp_path)
    response = None
    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.03):
                response = await send_observed(
                    Peer(),
                    object(),
                    journal,
                    1,
                )
                await response.aread()
    finally:
        if response:
            await response.aclose()
    saved = json.loads((tmp_path / "worker-stages.json").read_text())
    assert saved["attempts"] == 1 and saved["last_stage"] == expected
    assert "keepalive" not in json.dumps(saved) and "example.test" not in json.dumps(saved)
    if response:
        assert closed


async def test_complete_stream_and_result_submission_stages_are_bounded(tmp_path):
    class Peer:
        async def handle_async_request(self, request):
            return Response(CompleteBody())

    journal = StageJournal(tmp_path, tail=4)
    for n in range(1, 6):
        response = await send_observed(Peer(), object(), journal, n, tool_result=n > 1)
        await response.aread()
        await response.aclose()
    saved = json.loads((tmp_path / "worker-stages.json").read_text())
    assert saved["attempts"] == 5 and saved["truncated"] and len(saved["events"]) == 4
    assert saved["last_stage"] == "response_body_complete"
    assert saved["events"][0]["stage"] == "tool_result_submission"
    assert (tmp_path / "worker-stages.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("cancel", [False, True])
async def test_parent_timeout_or_cancel_kills_owned_child_and_preserves_stage(
    tmp_path, monkeypatch, cancel
):
    import os
    import sys
    from pathlib import Path
    from me3_voice.onit_tasks import OnitTaskExecutor, OnitTaskUnavailable
    from me3_voice.configuration import BackendSpec

    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.chdir(tmp_path)
    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="nvidia/nemotron-3.5-lightning:free",
        auth_env="ME3_LLM_API_KEY",
        allow_live=True,
        request_limit=1,
    )
    executor = OnitTaskExecutor(
        Path(sys.executable),
        spec,
        {"ME3_LLM_API_KEY": "offline"},
        root=tmp_path / "var/tasks",
        max_requests=1,
    )
    directory = executor.root / "stage-case"
    directory.mkdir()
    children = []

    async def start():
        code = 'import sys,json,time; from me3_voice.onit_diagnostics import StageJournal; print("READY",flush=True); payload=json.loads(sys.stdin.readline()); StageJournal(payload["workspace"]).record("worker_received"); time.sleep(60)'
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-c",
            code,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert await child.stdout.readline() == b"READY\n"
        executor.child = child
        children.append(child)
        return child

    executor._ensure_worker = start
    executor._reply_deadline = lambda budget: 5 if cancel else 0.3
    executor.startup = asyncio.create_task(start())
    job = asyncio.create_task(executor._run_worker("No effects.", directory, 1))
    try:
        if cancel:
            async with asyncio.timeout(2):
                while (
                    not (directory / "worker-stages.json").exists()
                    or json.loads((directory / "worker-stages.json").read_text()).get("last_stage")
                    != "worker_received"
                ):
                    await asyncio.sleep(0.01)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job
        else:
            with pytest.raises(OnitTaskUnavailable) as error:
                await job
            assert (
                error.value.code == "onit_worker_timeout"
                and "worker_received" in error.value.user_summary
            )
        assert children[0].returncode is not None and executor.child is None
        assert executor.startup is None
        # Same executor can prepare a distinct worker after either failure.
        executor._reply_deadline = lambda budget: 0.3
        next_directory = executor.root / "next-stage-case"
        next_directory.mkdir()
        with pytest.raises(OnitTaskUnavailable):
            await executor._run_worker("New isolated request.", next_directory, 1)
        assert len(children) == 2 and all(c.returncode is not None for c in children)
        saved = json.loads((directory / "worker-stages.json").read_text())
        assert saved["last_stage"] == "worker_received" and saved["attempts"] == 0
    finally:
        if not job.done():
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
        await executor.close()
