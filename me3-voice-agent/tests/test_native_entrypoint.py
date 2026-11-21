"""Real application CLI and loopback WebSocket; speech MCP children are actual."""

import asyncio
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from uuid import uuid4

from aiohttp import ClientSession, WSMsgType
from aiohttp.test_utils import TestServer
import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from me3_voice.configuration import load_backends, load_runtime, ConfigurationError
from me3_voice.voice_bridge import build_app, BrowserSink


@pytest.fixture(autouse=True)
def isolated_native_task_storage(tmp_path, monkeypatch):
    """Tests must never contend with a running operator bridge's task store."""
    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "native-tasks.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "native-workspace"))


class Wake:
    frame_samples = 1280

    def reset(self):
        self.reset_count = getattr(self, "reset_count", 0) + 1

    def feed(self, _pcm):
        return {"triggered": True}

    def end_session(self):
        pass


class FixtureVAD(FrameProcessor):
    """Annotated detector boundary, never an acoustic reliability measurement."""

    def __init__(self):
        super().__init__()
        self.speaking = False
        self.quiet = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            voiced = any(frame.audio)
            if voiced and not self.speaking:
                self.speaking = True
                await self.broadcast_frame(VADUserStartedSpeakingFrame)
            self.quiet = 0 if voiced else self.quiet + 1
            if self.speaking and self.quiet >= 2:
                self.speaking = False
                await self.broadcast_frame(VADUserStoppedSpeakingFrame)
        await self.push_frame(frame, direction)


@pytest.fixture
def native_args():
    return Namespace(engine="native", runtime_config=None, followup_seconds=20, extended_seconds=20)


async def next_json(ws, kind, *, acknowledge=True, ack_fields=None, timeout=12):
    async with asyncio.timeout(timeout):
        while True:
            message = await ws.receive()
            if message.type == WSMsgType.BINARY:
                if acknowledge:
                    ack = {"type": "audio_ack", "id": struct.unpack("<I", message.data[:4])[0]}
                    ack.update(ack_fields or {})
                    await ws.send_json(ack)
            elif message.type == WSMsgType.TEXT:
                data = message.json()
                if data.get("type") == "error":
                    raise AssertionError(data)
                if data.get("type") == kind:
                    return data
            else:
                raise AssertionError("connection closed before " + kind)


async def send_turn(ws):
    for _ in range(5):
        await ws.send_bytes(b"\1\0" * 512)
    for _ in range(3):
        await ws.send_bytes(b"\0\0" * 512)


async def wait_answer(ws, count=1, *, ack_fields=None):
    async with asyncio.timeout(15):
        while True:
            data = await next_json(ws, "transcript", ack_fields=ack_fields)
            answers = [m for m in data["data"]["messages"] if m["role"] == "assistant"]
            if len(answers) >= count:
                return data["data"]


def test_runtime_selection_is_explicit_and_legacy_default(tmp_path):
    assert load_runtime().engine == "legacy"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"engine": "native", "silence_seconds": 0}))
    assert load_runtime(path).engine == "native"
    assert load_runtime(path, engine="legacy").engine == "legacy"
    path.write_text(json.dumps({"engine": "native", "task_model": "anthropic/claude-haiku-5.5"}))
    selected = load_runtime(path)
    assert selected.task_model == "anthropic/claude-haiku-5.5"
    assert selected.task_request_limit is None
    assert selected.task_input_tokens_total is None
    path.write_text(
        json.dumps(
            {
                "engine": "native",
                "task_model": "anthropic/claude-haiku-5.5",
                "task_max_output_tokens": 8192,
            }
        )
    )
    selected = load_runtime(path)
    assert selected.task_max_output_tokens == 8192
    path.write_text(
        json.dumps(
            {
                "engine": "native",
                "task_model": "anthropic/claude-haiku-5.5",
                "cost_alert_usd": 0,
            }
        )
    )
    with pytest.raises(ConfigurationError, match="invalid engine"):
        load_runtime(path)
    path.write_text('{"engine":"nativ","extra":true}')
    with pytest.raises(ConfigurationError):
        load_runtime(path)
    path.write_text('{"engine":"native","llm_max_output_tokens":8193}')
    with pytest.raises(ConfigurationError):
        load_runtime(path)
    with pytest.raises(ConfigurationError):
        load_runtime(tmp_path / "missing.json")


def test_laptop_demo_profile_uses_provider_managed_volume_and_context_caps():
    backends = load_backends(Path("configs/native-demo-backends.json"))
    runtime = load_runtime(Path("configs/native-demo-runtime.json"), engine="native")
    assert backends["llm"].model == "openai/gpt-oss-120b"
    assert backends["llm"].request_limit is None
    assert backends["llm"].max_input_tokens == 65536
    assert backends["llm"].max_input_tokens_total is None
    assert backends["asr"].request_limit is None
    assert backends["tts"].request_limit is None
    assert runtime.task_model == "anthropic/claude-haiku-5.5"
    assert runtime.task_request_limit is None
    assert runtime.task_input_tokens_total is None
    assert runtime.llm_max_output_tokens == 4096
    assert runtime.task_max_output_tokens == 8192
    assert runtime.cost_alert_usd == 0.5
    assert runtime.research.request_limit is None
    assert runtime.max_model_calls_per_turn == 7
    assert runtime.max_tool_calls_per_turn == 8
    assert runtime.max_repeated_tool_calls == 2


def test_haiku_task_model_has_separate_budget_and_preserves_gpt_oss_conversation():
    from dataclasses import replace
    from me3_voice.native_app import NativeState
    from me3_voice.configuration import RuntimeOptions, load_backends

    backends = load_backends()
    backends["llm"] = replace(
        backends["llm"],
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        auth_env="ME3_OPENROUTER_API_KEY",
        model="openai/gpt-oss-120b",
        allow_live=True,
        request_limit=None,
        max_input_tokens=65536,
        max_input_tokens_total=None,
    )
    state = NativeState(backends=backends, tasks=object())
    task_spec, task_budget = state.task_model_configuration(
        RuntimeOptions(engine="native", task_model="anthropic/claude-haiku-5.5")
    )
    assert task_spec.model == "anthropic/claude-haiku-5.5"
    assert task_spec.request_limit is None
    assert task_spec.max_input_tokens_total is None
    assert (
        RuntimeOptions(
            engine="native", task_model="anthropic/claude-haiku-5.5", task_max_output_tokens=8192
        ).task_max_output_tokens
        == 8192
    )
    assert task_budget.spec is task_spec
    assert task_budget is not state.budget
    assert state.budget.spec.model == "openai/gpt-oss-120b"
    assert backends["llm"].model == "openai/gpt-oss-120b"


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_production_dialogue_output_limit_preserves_request_accounting(
    tmp_path, monkeypatch, finish_reason
):
    import sqlite3

    from openai.types.chat import ChatCompletionChunk
    from pipecat.processors.aggregators.llm_context import LLMContext

    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import ProviderError

    runtime = load_runtime("configs/native-demo-runtime.json")
    spec = load_backends(Path("configs/native-demo-backends.json"))["llm"]
    ledger = tmp_path / "provider.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "production-output-fixture")
    llm = NativeLLM(
        spec,
        credentials={spec.auth_env: "offline-fixture"},
        max_output_tokens=runtime.llm_max_output_tokens,
    )

    async def create(**params):
        assert params["max_completion_tokens"] == 4096

        async def chunks():
            yield ChatCompletionChunk(
                id="fixture",
                object="chat.completion.chunk",
                created=0,
                model=spec.model,
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": "Fixture response."},
                        "finish_reason": finish_reason,
                    }
                ],
                usage={"prompt_tokens": 9000, "completion_tokens": 3072, "total_tokens": 12072},
            )

        return chunks()

    monkeypatch.setattr(llm._client.chat.completions, "create", create)
    context = LLMContext(messages=[{"role": "user", "content": "source evidence " * 2500}])
    try:
        stream = await llm.get_chat_completions(context)
        if finish_reason == "length":
            with pytest.raises(ProviderError) as error:
                async for _ in stream:
                    pass
            assert error.value.code == "response_budget_exhausted"
        else:
            assert [chunk.choices[0].delta.content async for chunk in stream] == [
                "Fixture response."
            ]
    finally:
        await llm._client.close()
    with sqlite3.connect(ledger) as db:
        status, reserved, actual_in, actual_out = db.execute(
            "SELECT status,input_tokens_reserved,input_tokens_actual,output_tokens_actual "
            "FROM provider_requests"
        ).fetchone()
    assert 32768 < reserved <= 65536
    if finish_reason == "stop":
        assert (status, actual_in, actual_out) == ("success", 9000, 3072)
    else:
        assert status != "success"


@pytest.mark.parametrize("engine", ["legacy", "native"])
def test_real_cli_entrypoint_with_mock_upstream_tool_events(tmp_path, engine):
    # Generated application artifacts must live under the repository's var/ boundary.
    output = Path.cwd() / "var" / "scratch" / f"pytest-{engine}-{uuid4().hex}.json"
    command = [
        sys.executable,
        "-B",
        "-m",
        "me3_voice.cli",
        "demo",
        "--engine",
        engine,
        "--mock-transcripts-json",
        json.dumps(["set mock light on", "mock light status"]),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    value = json.loads(output.read_text())
    assert value["engine"] == engine and set(value["protocols"]) == {"asr", "tts"}
    assert [m["content"] for m in value["history"] if m["role"] == "user"] == [
        "set mock light on",
        "mock light status",
    ]
    assert value["sink_chunks"] > 0
    if engine == "native":
        assert not value["legacy_session_imported"]
        assert len([e for e in value["events"] if e["event"] == "tool_completed"]) == 2


async def test_actual_native_browser_route_continuity_and_reconnect(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge
    import me3_voice.session as legacy

    wake = Wake()
    wake_builds = []
    monkeypatch.setattr(bridge, "build_wake", lambda _: (wake_builds.append(True), wake)[1])
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "browser-task-runtime.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "browser-task-workspace"))
    monkeypatch.setattr(
        legacy.Session,
        "__init__",
        lambda *_a, **_k: pytest.fail("Session instantiated in native path"),
    )
    credentials = {
        "ME3_DEEPGRAM_API_KEY": "offline-fixture",
        "ME3_OPENROUTER_API_KEY": "offline-fixture",
    }
    app = build_app(tmp_path, load_backends(), credentials, native_args)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.get(server.make_url("/api/health")) as response:
            startup = await response.json()
        assert startup["speech_services"] == "initialized"
        assert startup["voice_ready"] is True
        assert startup["task_runtime"] == "ready"
        assert startup["services"]["tasks"]["status"] == "READY"
        assert startup["services"]["me2_wake"]["client_state"] == "READY"
        async with client.post(server.make_url("/api/session/prepare")) as response:
            prepared = await response.json()
            assert response.status == 200
        assert prepared["voice_ready"] is True
        assert prepared["speech_services"] == "initialized"
        assert prepared["budget"]["gpt_oss_requests"]["used"] == 0
        async with client.get(server.make_url("/")) as response:
            page = await response.text()
            assert "Last source retrieval" in page
            assert "Check provider connections" not in page
        session_id = None
        for expected in (1, 2):
            async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
                await ws.send_json(
                    {
                        "type": "hello",
                        "rate": 16000,
                        "channels": 1,
                        "conversation_mode": "continuous",
                    }
                )
                ready = await next_json(ws, "ready")
                assert ready["engine"] == "native"
                if session_id:
                    assert ready["session_id"] == session_id
                session_id = ready["session_id"]
                await send_turn(ws)
                view = await wait_answer(ws, expected)
                assert len([m for m in view["messages"] if m["role"] == "user"]) == expected
                await ws.close()
            await asyncio.sleep(0.1)
        async with client.get(server.make_url("/api/health")) as response:
            assert (await response.json())["session_resume"]
        assert len(wake_builds) == 1
        assert wake.reset_count == 2


async def test_native_health_exposes_service_state_and_remaining_profile_quotas(
    tmp_path, monkeypatch, native_args
):
    monkeypatch.delenv("ME3_MUSIC_CONFIG", raising=False)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "health-profile-test-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "provider-requests.sqlite"))
    native_args.runtime_options = load_runtime(
        Path("configs/native-demo-runtime.json"), engine="native"
    )
    backends = load_backends(Path("configs/native-demo-backends.json"))
    credentials = {
        "ME3_DEEPGRAM_API_KEY": "offline-fixture",
        "ME3_OPENROUTER_API_KEY": "offline-fixture",
        "ME3_RESEARCH_TAVILY_API_KEY": "offline-fixture",
    }
    app = build_app(tmp_path, backends, credentials, native_args)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.get(server.make_url("/api/health")) as response:
            assert response.status == 200
            health = await response.json()
    services = health["services"]
    assert services["bridge"]["status"] == "READY"
    assert services["session"]["status"] == "READY"
    assert services["asr"]["status"] == "DEGRADED"
    assert services["gpt_oss"]["status"] == "DEGRADED"
    assert services["haiku"]["status"] == "DEGRADED"
    assert services["calls_messages"]["status"] == "UNCONFIGURED"
    assert services["github"]["status"] == "UNCONFIGURED"
    assert services["gpio"]["status"] == "DEGRADED"
    budget = health["budget"]
    assert budget["gpt_oss_requests"] == {
        "limit": None,
        "used": 0,
        "remaining": None,
        "limit_source": "provider/account",
    }
    assert budget["haiku_requests"] == {
        "limit": None,
        "used": 0,
        "remaining": None,
        "limit_source": "provider/account",
    }
    assert budget["asr_requests"]["limit_source"] == "provider/account"
    assert budget["tts_requests"]["limit_source"] == "provider/account"
    assert budget["tavily_requests"] == {
        "limit": None,
        "used": 0,
        "remaining": None,
        "limit_source": "provider/account",
    }
    assert budget["cost_alert"]["state"] == "NO_USAGE_DATA"
    assert budget["cost_alert"]["threshold_usd"] == 0.5
    assert services["tasks"]["status"] == "READY"
    assert services["gpt_oss"]["client_state"] == "READY"
    assert services["gpt_oss"]["provider_state"] == "NOT_PROBED"
    assert services["haiku"]["client_state"] == "INITIALIZED_UNVERIFIED"
    assert services["haiku"]["provider_state"] == "NOT_PROBED"
    assert services["asr"]["client_state"] == "READY"
    assert services["tts"]["client_state"] == "READY"
    assert services["tavily"]["client_state"] == "READY"
    assert budget["vendor_billed_cost_usd"] is None


async def test_managed_demo_stop_rejects_active_and_unowned_requests(tmp_path, monkeypatch, native_args):
    import os

    monkeypatch.setenv("ME3_DEMO_STOP_TOKEN", "private-launcher-fixture")
    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    app = build_app(tmp_path, load_backends(), {}, native_args)
    async with TestServer(app) as server, ClientSession() as client:
        stop = server.make_url("/api/demo/stop")
        ws_url = server.make_url("/api/voice/ws")
        headers = {"X-Swift-Demo-Token": "private-launcher-fixture"}
        async with client.post(stop) as response:
            assert response.status == 403
        async with client.ws_connect(ws_url):
            async with client.post(stop, headers=headers) as response:
                assert response.status == 409
        for _ in range(30):
            async with client.post(stop, headers=headers) as response:
                if response.status == 200:
                    assert (await response.json())["status"] == "stopping"
                    break
                assert response.status == 409
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Closed session never released the shutdown gate")
        async with client.get(ws_url) as response:
            assert response.status == 409
        await asyncio.sleep(0.15)
    assert len(signals) == 1


async def test_production_session_prepare_warms_local_pipeline_without_model_calls(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as native_app
    import me3_voice.native_bridge as bridge
    import me3_voice.task_runtime as task_runtime

    class Speech:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class TaskRuntime:
        provider_client_state = "INITIALIZED_UNVERIFIED"

        def __init__(self, *_args, **_kwargs):
            pass

        async def start_runtime(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(native_app, "SpeechClient", Speech)
    monkeypatch.setattr(task_runtime, "TaskRuntime", TaskRuntime)
    monkeypatch.setattr(bridge, "build_wake", lambda _args: None)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "session-prepare-offline-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "session-prepare-ledger.sqlite"))
    native_args.runtime_options = load_runtime(
        Path("configs/native-demo-runtime.json"), engine="native"
    )
    credentials = {
        "ME3_DEEPGRAM_API_KEY": "offline-fixture",
        "ME3_OPENROUTER_API_KEY": "offline-fixture",
    }
    app = build_app(tmp_path, load_backends(), credentials, native_args)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.get(server.make_url("/api/health")) as response:
            before = await response.json()
        async with client.post(server.make_url("/api/session/prepare")) as response:
            assert response.status == 200
            prepared = await response.json()
        async with client.get(server.make_url("/")) as response:
            page = await response.text()
    assert before["voice_ready"] is True
    assert prepared["voice_ready"] is True
    assert prepared["speech_services"] == "initialized"
    assert prepared["budget"]["gpt_oss_requests"]["used"] == 0
    assert "Check provider connections" not in page


def test_voice_readiness_separates_configuration_from_provider_reachability():
    from me3_voice.native_bridge import _voice_readiness

    services = {
        key: {"status": "DEGRADED", "client_state": "READY", "provider_state": "NOT_PROBED"}
        for key in ("asr", "gpt_oss", "tts")
    }
    assert _voice_readiness(True, services) == (True, True)
    services["gpt_oss"].update(status="DISCONNECTED", provider_state="AUTH_REJECTED")
    assert _voice_readiness(True, services) == (True, False)
    services["gpt_oss"].update(status="UNCONFIGURED", provider_state="NOT_CONFIGURED")
    assert _voice_readiness(True, services) == (False, False)


async def test_start_retries_a_failed_local_speech_owner_without_resetting_transport(tmp_path, monkeypatch):
    from me3_voice.native_app import NativeState

    class Tasks:
        async def close(self):
            return None

    class FailedSpeech:
        async def __aenter__(self):
            raise RuntimeError("fixture startup failure")

        async def __aexit__(self, *_args):
            return None

    class ReadySpeech:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    created = 0

    def speech_factory():
        nonlocal created
        created += 1
        return FailedSpeech() if created == 1 else ReadySpeech()

    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "recovery-tasks.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "recovery-workspace"))
    state = NativeState(speech_factory=speech_factory, tasks=Tasks())
    transport = state.budget
    try:
        with pytest.raises(RuntimeError, match="fixture startup failure"):
            await state.start()
        recovered = await state.start()
        assert isinstance(recovered, ReadySpeech)
        assert created == 2
        assert state.budget is transport
    finally:
        await state.close()


async def test_nonbillable_provider_check_updates_health_without_claiming_inference(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_bridge as bridge

    calls = []

    class Response:
        def __init__(self, status):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, *, headers, allow_redirects):
            assert allow_redirects is False
            calls.append((url, headers["Authorization"]))
            status = 200 if "openrouter.ai" in url else 403
            return Response(status)

    monkeypatch.setattr(bridge, "ClientSession", Client)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "health-probe-offline-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "probe-ledger.sqlite"))
    native_args.runtime_options = load_runtime(
        Path("configs/native-demo-runtime.json"), engine="native"
    )
    backends = load_backends(Path("configs/native-demo-backends.json"))
    credentials = {
        "ME3_DEEPGRAM_API_KEY": "deepgram-fixture-key",
        "ME3_OPENROUTER_API_KEY": "openrouter-fixture-key",
    }
    app = build_app(tmp_path, backends, credentials, native_args)
    import me3_voice.remote as remote

    async with TestServer(app) as server, ClientSession() as client:
        async with client.post(server.make_url("/api/health/provider-check")) as response:
            assert response.status == 200
            checked = await response.json()
        async with client.get(server.make_url("/api/health")) as response:
            health = await response.json()
        monkeypatch.setattr(
            remote,
            "request_ledger_summary",
            lambda *_args: {
                "run_id": "health-probe-offline-01",
                "requests": [
                    {
                        "provider": "openrouter.ai",
                        "model": backends["llm"].model,
                        "status": "success",
                        "updated_at": __import__("time").time() + 10,
                        "actual_input_tokens": 100,
                        "actual_output_tokens": 12,
                        "estimated_input_tokens_reserved": 120,
                        "error_code": None,
                    }
                ],
                "usage": {"actual_input_tokens": 100, "actual_output_tokens": 12},
            },
        )
        async with client.get(server.make_url("/api/health")) as response:
            inferred = await response.json()

    assert checked["checks"]["openrouter"]["provider_state"] == "AUTHENTICATED_REACHABLE"
    assert checked["checks"]["deepgram"]["provider_state"] == "AUTH_REJECTED"
    assert checked["inference"] == "NOT_PROBED"
    assert checked["tavily"] == "NOT_PROBED_COSTED_SEARCH_SKIPPED"
    services = health["services"]
    assert services["gpt_oss"]["provider_state"] == "AUTHENTICATED_REACHABLE"
    assert services["gpt_oss"]["inference_state"] == "NOT_PROBED"
    assert services["haiku"]["provider_state"] == "AUTHENTICATED_REACHABLE"
    assert services["asr"]["provider_state"] == "AUTH_REJECTED"
    assert services["asr"]["inference_state"] == "NOT_PROBED"
    assert services["tavily"]["provider_state"] == "NOT_PROBED"
    assert len(calls) == 2
    assert inferred["services"]["gpt_oss"]["provider_state"] == "INFERENCE_READY"
    assert inferred["services"]["gpt_oss"]["inference_state"] == "READY"


async def test_native_browser_writes_private_sanitized_session_events(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "browser-task-runtime.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "browser-task-workspace"))
    scratch = Path.cwd() / "var" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-journal-", dir=scratch) as folder:
        async with (
            TestServer(
                build_app(tmp_path, load_backends(), {}, native_args, trace_dir=Path(folder))
            ) as server,
            ClientSession() as client,
        ):
            async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
                await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
                ready = await next_json(ws, "ready")
                await send_turn(ws)
                await wait_answer(
                    ws,
                    ack_fields={
                        "schedule_gap_seconds": 0.012,
                        "queued_audio_seconds": 0.16,
                        "audio_context_state": "running",
                        "arrival_ms": 1234.5,
                    },
                )
                await ws.close()
            await asyncio.sleep(0.05)
        journal = Path(folder) / f"native-{ready['run_id']}.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        assert journal.stat().st_mode & 0o777 == 0o600
        assert rows and all(row["run_id"] == ready["run_id"] for row in rows)
        assert all(row["session_id"] == ready["session_id"] for row in rows)
        assert sum(row["event"] == "wake_detected" for row in rows) == 1
        assert any(row["event"] == "turn_completed" for row in rows)
        ingress = next(row for row in rows if row["event"] == "audio_frame_delivered")
        assert ingress["ingress_sequence"] == 0
        assert ingress["ingress_at"] <= ingress["at"]
        final = next(row for row in rows if row["event"] == "asr_final")
        assert final["has_recognized_text"] is True
        playback = next(row for row in rows if row["event"] == "playback_ack")
        assert playback["schedule_gap_seconds"] == 0.012
        assert playback["queued_audio_seconds"] == 0.16
        assert playback["audio_context_state"] == "running"
        assert playback["browser_arrival_ms"] == 1234.5
        assert all("text" not in row and "content" not in row for row in rows)


async def test_native_browser_manual_arm_accepts_one_turn_without_wake_detection(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    class NoWake(Wake):
        def feed(self, _pcm):
            return {"triggered": False}

    monkeypatch.setattr(bridge, "build_wake", lambda _: NoWake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "manual-arm-tasks.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "manual-arm-workspace"))
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            await next_json(ws, "ready")
            await ws.send_json({"type": "arm"})
            await send_turn(ws)
            result = await wait_answer(ws)
            assert any(row["role"] == "assistant" for row in result["messages"])


def test_native_event_projection_excludes_secrets_and_free_text():
    from me3_voice.native_bridge import _safe_event

    row = _safe_event(
        {
            "event": "asr_final",
            "at": 123.5,
            "turn_id": 2,
            "text": "private speech",
            "content": "private answer",
            "api_key": "secret",
            "auth_token": "secret",
            "error_code": "asr_timeout",
            "error_message": "private provider payload",
        }
    )
    assert row == {"event": "asr_final", "at": 123.5, "turn_id": 2, "error_code": "asr_timeout"}
    tts = _safe_event(
        {
            "event": "tts_segment_summary",
            "diagnostics": {
                "provider": {
                    "provider_audio_seconds": 2.4,
                    "synthesis_wall_seconds": 5.6,
                    "text": "private text",
                    "authorization": "secret",
                },
                "client_mcp": {
                    "tts_next_count": 8,
                    "tts_next_latency_seconds": {"p95_seconds": 0.02, "payload": "secret"},
                },
            },
        }
    )
    assert tts == {
        "event": "tts_segment_summary",
        "provider_provider_audio_seconds": 2.4,
        "provider_synthesis_wall_seconds": 5.6,
        "client_mcp_tts_next_count": 8,
        "client_mcp_next_p95_seconds": 0.02,
    }
    retrieved = _safe_event(
        {
            "event": "tool_completed",
            "tool_name": "retrieve_source",
            "diagnostics": {
                "source_id": "S12",
                "source_url_hash": "0123456789abcdef",
                "extraction_method": "local_html",
                "evidence_status": "retrieved_unverified",
                "evidence_characters": 248,
                "evidence_chunks": 2,
                "text": "private article text",
                "api_key": "private credential",
            },
        }
    )
    assert retrieved == {
        "event": "tool_completed",
        "tool_name": "retrieve_source",
        "research_source_id": "S12",
        "research_source_url_hash": "0123456789abcdef",
        "research_extraction_method": "local_html",
        "research_evidence_status": "retrieved_unverified",
        "research_evidence_characters": 248,
        "research_evidence_chunks": 2,
    }
    citation = _safe_event(
        {
            "event": "llm_citation_check",
            "citation_ids": ["S1", "S12"],
            "retained_source_ids": ["S0", "S1", "S12"],
            "evidence_valid": True,
            "citation_sources": [
                {
                    "source_id": "S1",
                    "title": "Search result title",
                    "url": "https://source.example/result",
                    "evidence_status": "search_result_snippet",
                    "source_type": "untrusted_search_result",
                    "fetched": False,
                    "truncated": True,
                },
                {
                    "source_id": "S12",
                    "title": "Missing URL",
                    "evidence_status": "retrieved_unverified",
                },
            ],
            "raw_response": "private answer",
        }
    )
    assert citation["citation_ids"] == ["S1", "S12"]
    assert citation["retained_source_ids"] == ["S0", "S1", "S12"]
    assert citation["citation_sources"] == [
        {
            "source_id": "S1",
            "url": "https://source.example/result",
            "title": "Search result title",
            "evidence_status": "search_result_snippet",
            "source_type": "untrusted_search_result",
            "fetched": False,
            "truncated": True,
        }
    ]
    assert citation["citation_source_mapping_complete"] is False
    assert "raw_response" not in citation
    rejection = _safe_event(
        {
            "event": "llm_citation_check",
            "error_code": "insufficient_verified_evidence",
            "rejection_rule": "claim_count_out_of_bounds",
            "draft_claim_count": 3,
            "draft_passage_ids": ["S2-P1", "S3-P1"],
            "retained_passage_ids": ["S2-P1"],
            "draft_source_ids": ["S2"],
            "raw_response": "private draft",
        }
    )
    assert rejection["draft_claim_count"] == 3
    assert rejection["draft_passage_ids"] == ["S2-P1", "S3-P1"]
    assert rejection["retained_passage_ids"] == ["S2-P1"]
    assert rejection["draft_source_ids"] == ["S2"]
    assert rejection["rejection_rule"] == "claim_count_out_of_bounds"
    assert "raw_response" not in rejection
    assert "draft_passage_ids" not in _safe_event(
        {**rejection, "draft_passage_ids": ["https://example.invalid/private"]}
    )
    rejected_mapping = _safe_event(
        {
            "event": "llm_citation_check",
            "citation_ids": ["S1"],
            "evidence_valid": True,
            "citation_sources": [
                {"source_id": "S9", "url": "https://source.example/not-cited"},
                {"source_id": "S1", "url": "https://source.example/?token=hidden"},
            ],
        }
    )
    assert rejected_mapping["citation_sources"] == []
    assert rejected_mapping["citation_source_mapping_complete"] is False
    malformed = _safe_event(
        {
            "event": "llm_citation_check",
            "citation_ids": ["S1", "private text"],
            "retained_source_ids": ["S0"],
        }
    )
    assert "citation_ids" not in malformed
    assert malformed["retained_source_ids"] == ["S0"]
    claim_diagnostic = _safe_event(
        {
            "event": "llm_claim_support_diagnostic",
            "source_id": "S1",
            "passage_id": "S1-P2",
            "contradiction_codes": ["polarity_conflict"],
            "claim": "private model text",
            "passage_text": "private source text",
        }
    )
    assert claim_diagnostic == {
        "event": "llm_claim_support_diagnostic",
        "source_id": "S1",
        "passage_id": "S1-P2",
        "contradiction_codes": ["polarity_conflict"],
    }
    malformed_codes = _safe_event(
        {
            "event": "llm_claim_support_diagnostic",
            "contradiction_codes": ["polarity_conflict", "private article text"],
        }
    )
    assert "contradiction_codes" not in malformed_codes


def test_native_interrupted_pending_answer_has_one_owned_terminal_outcome():
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_app import NativeApplication

    context = LLMContext([{"role": "user", "content": "Who is the prime minister of Japan?"}])
    app = NativeApplication(
        SimpleNamespace(session_id="interrupt-terminal", context=context), object()
    )
    app.turn_id = 1
    app.event("tts_response_started", response_id="japan-final-response")

    # A new turn arrives while the final model answer is still owned by turn 1.
    app.event("turn_started")
    interrupted = [event for event in app.events if event["event"] == "turn_interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["turn_id"] == 1
    assert interrupted[0]["response_id"] == "japan-final-response"
    assert app.response_records["japan-final-response"]["response_state"] == "interrupted"
    assert app.response_records["japan-final-response"]["superseded_by_turn_id"] == 2

    # A late provider end/failure cannot create a second terminal result.
    app.event("turn_failed", owner_turn_id=1, error_code="provider_incomplete_response")
    app.event("turn_completed", owner_turn_id=1)
    assert [event["event"] for event in app.events].count("turn_interrupted") == 1
    assert [
        event
        for event in app.events
        if event["event"] in {"turn_completed", "turn_failed", "turn_error", "turn_interrupted"}
    ] == interrupted


async def test_unclear_vad_does_not_start_or_interrupt_until_asr_text_exists():
    from me3_voice.native_pipeline import ASRConfirmedUserTurnStartStrategy
    from pipecat.turns.types import ProcessFrameResult

    strategy = ASRConfirmedUserTurnStartStrategy()
    starts = []

    async def record_start(**kwargs):
        starts.append(kwargs)

    strategy.trigger_user_turn_started = record_start
    assert (
        await strategy.process_frame(VADUserStartedSpeakingFrame()) is ProcessFrameResult.CONTINUE
    )
    assert (
        await strategy.process_frame(InterimTranscriptionFrame("  ", "user", ""))
        is ProcessFrameResult.CONTINUE
    )
    assert starts == []

    assert (
        await strategy.process_frame(InterimTranscriptionFrame("Stop", "user", ""))
        is ProcessFrameResult.STOP
    )
    assert starts == [{"enable_interruptions": False}]
    assert strategy.start_evidence == "asr_interim"

    # A provider that emits no interim but returns a final transcript must
    # retain that already-aggregated text instead of broadcasting a reset.
    starts.clear()
    assert (
        await strategy.process_frame(TranscriptionFrame("follow up", "user", "", finalized=True))
        is ProcessFrameResult.STOP
    )
    assert starts == [{"enable_interruptions": False}]


def test_native_pipeline_uses_asr_confirmed_turn_start():
    from me3_voice.native_pipeline import ASRConfirmedUserTurnStartStrategy, make_native_pipeline

    class Processor(FrameProcessor):
        pass

    parts = make_native_pipeline(Processor(), Processor(), Processor())
    start_strategies = parts.aggregators.user()._params.user_turn_strategies.start
    assert len(start_strategies) == 1
    assert isinstance(start_strategies[0], ASRConfirmedUserTurnStartStrategy)


async def test_final_only_asr_turn_stops_stale_audio_without_replaying_it():
    from types import SimpleNamespace
    from me3_voice.native_app import NativeOutput

    class Sink:
        def __init__(self):
            self.stopped = 0
            self.sent = []

        async def stop(self):
            self.stopped += 1
            return True

        async def send_audio(self, audio, *, response_id):
            self.sent.append((audio, response_id))
            return len(self.sent)

    sink = Sink()
    events = []
    output = NativeOutput(sink, lambda name, **fields: events.append((name, fields)))
    await output.stop_response("old-response")
    assert sink.stopped == 1
    assert not await output.write_audio_frame(
        SimpleNamespace(audio=b"stale", context_id="old-response")
    )
    assert await output.write_audio_frame(SimpleNamespace(audio=b"new", context_id="new-response"))
    assert sink.sent == [(b"new", "new-response")]
    assert events[:2] == [
        ("audio_stop_sent", {"response_id": "old-response"}),
        ("playback_stopped_browser", {"response_id": "old-response"}),
    ]


def test_tool_error_code_preserves_only_safe_symbolic_provider_codes():
    from me3_voice.native_app import _safe_tool_error_code

    class MusicFailure(Exception):
        code = "music_device_unavailable"

    class UnsafeFailure(Exception):
        code = "token leaked / https://private.example"

    assert _safe_tool_error_code(MusicFailure()) == "music_device_unavailable"
    assert _safe_tool_error_code(UnsafeFailure()) == "tool_service_unavailable"


def test_live_service_health_uses_plain_labels_and_live_timestamp():
    from me3_voice.dashboard import render

    html = render(
        {"schema_version": 1, "reports": [], "warnings": [], "generated_utc": ""},
        live=True,
    )
    assert "Bridge live ·" in html
    assert "updated ${stamp.toLocaleTimeString()}" in html
    assert "Configured · connects on use" in html
    assert "Set up needed" in html


def test_browser_playback_ack_reports_scheduling_gaps_and_queued_audio():
    from me3_voice.dashboard import render

    html = render(
        {"schema_version": 1, "reports": [], "warnings": [], "generated_utc": ""},
        live=True,
    )
    assert "schedule_gap_seconds:item.scheduleGapSeconds" in html
    assert "queued_audio_seconds:item.queuedAudioSeconds" in html
    assert "audio_context_state:item.audioContextState" in html
    assert "const previousEnd=runtime.nextPlay" in html


async def test_native_research_ack_runs_operation_immediately_and_queues_once(monkeypatch):
    from types import SimpleNamespace
    import me3_voice.native_app as native
    from me3_voice.providers import ToolCall

    monkeypatch.setattr(native, "RESEARCH_ACK_DELAY_SECONDS", 0.02)
    app = native.NativeApplication.__new__(native.NativeApplication)
    app.turn_id = 3
    app.research_ack_turns = set()
    app.error = None
    app.events = []

    class TTSQueue:
        def __init__(self):
            self.frames = []

        async def queue_frame(self, frame):
            self.frames.append(frame)

    tts = TTSQueue()
    app.parts = SimpleNamespace(tts=tts)
    app.event = lambda name, **fields: app.events.append((name, fields))
    operation_started = asyncio.Event()

    async def slow_search():
        operation_started.set()
        await asyncio.sleep(0.06)
        return {"ok": True}

    call = ToolCall("search-call-1", "web_search", {"query": "current event"})
    result = await app.research_operation_with_ack(call, slow_search())
    assert result == {"ok": True} and operation_started.is_set()
    assert len(tts.frames) == 1
    assert tts.frames[0].text == "I'll check that."
    assert tts.frames[0].append_to_context is False
    assert tts.frames[0].owner_id == call.call_id
    assert [name for name, _ in app.events] == ["research_acknowledgment_queued"]


async def test_native_research_ack_suppresses_fast_result_and_cancels_pending_search(
    monkeypatch,
):
    from types import SimpleNamespace
    import me3_voice.native_app as native
    from me3_voice.providers import ToolCall

    monkeypatch.setattr(native, "RESEARCH_ACK_DELAY_SECONDS", 0.02)
    app = native.NativeApplication.__new__(native.NativeApplication)
    app.turn_id = 1
    app.research_ack_turns = set()
    app.error = None
    app.parts = SimpleNamespace(tts=SimpleNamespace(queue_frame=lambda *_: None))
    app.event = lambda *_args, **_kwargs: None
    call = ToolCall("search-call-2", "web_search", {"query": "quick result"})

    async def immediate():
        return "ready"

    assert await app.research_operation_with_ack(call, immediate()) == "ready"
    assert not app.research_ack_turns

    cancelled = asyncio.Event()

    async def pending():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(app.research_operation_with_ack(call, pending()))
    await asyncio.sleep(0.04)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_progress_tts_context_does_not_replace_or_complete_answer(monkeypatch):
    from types import SimpleNamespace
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.tts_service import TTSService
    from me3_voice.native_app import NativeApplication
    from me3_voice.native_speech import McpTTSService, ProgressTTSSpeakFrame

    async def fake_tts_process(self, frame, direction):
        await self.on_turn_context_created("progress-context")
        self.response_has_audio = True

    monkeypatch.setattr(TTSService, "process_frame", fake_tts_process)
    speech = SimpleNamespace(timeout=2, backends={})
    service_events = []
    service = McpTTSService(speech, on_event=service_events.append)
    service.response_context_id = "answer-context"
    service.response_has_audio = True
    service.response_stop_emitted = False
    await service.process_frame(
        ProgressTTSSpeakFrame("I'll check that.", owner_id="search-call"),
        FrameDirection.DOWNSTREAM,
    )
    assert service.response_context_id == "answer-context"
    assert service.response_has_audio is True
    assert [event["event"] for event in service_events] == ["tts_progress_started"]

    app = NativeApplication(
        SimpleNamespace(session_id="ack-test", context=LLMContext()),
        object(),
    )
    app.event("tts_response_started", response_id="answer-context")
    app.event("tts_progress_started", response_id="progress-context", owner_id="search-call")
    app.event("tts_progress_stop_queued", owner_id="search-call")
    assert app.active_response_id == "answer-context"
    assert list(app.response_records) == ["answer-context"]


def test_spoken_text_removes_markdown_citations_and_urls_only_at_tts_boundary():
    from me3_voice.native_speech import spoken_text

    display = "The lamp is now **red** at 30 % brightness. [S1] https://example.test/source"
    spoken = spoken_text(display)
    assert spoken == "The lamp is now red at 30 percent brightness."
    assert "**red**" in display and "[S1]" in display and "https://" in display


def test_native_tts_event_journal_keeps_safe_server_timeout_diagnostics():
    from me3_voice.native_bridge import _safe_event

    row = _safe_event(
        {
            "event": "tts_segment_summary",
            "diagnostics": {
                "status": "failed",
                "provider": {
                    "status": "failed",
                    "connection_reused": True,
                    "connection_setup_seconds": 0.3,
                    "provider_receive_wait_seconds": 4.0,
                    "provider_flushed_monotonic": 0,
                    "error_code": "provider_timeout",
                    "failure_type": "TimeoutError",
                    "connection_trace": {
                        "dns_started": 10.0,
                        "connection_cancelled_monotonic": 14.5,
                        "authorization": "secret",
                        "dns_completed": "secret",
                    },
                    "text": "do not persist",
                },
                "mcp_server": {
                    "status": "failed",
                    "mcp": {
                        "tts_next_count": 7,
                        "pcm_bytes": 32000,
                        "audio_seconds": 1.0,
                        "tts_next_seconds": {"p95_seconds": 0.4, "payload": "secret"},
                    },
                },
            },
        }
    )
    assert row["provider_connection_reused"] is True
    assert row["provider_connection_setup_seconds"] == 0.3
    assert row["provider_provider_receive_wait_seconds"] == 4.0
    assert row["provider_error_code"] == "provider_timeout"
    assert row["tts_connection_dns_started"] == 10.0
    assert row["tts_connection_connection_cancelled_monotonic"] == 14.5
    assert "tts_connection_dns_completed" not in row
    assert row["mcp_server_tts_next_count"] == 7
    assert row["mcp_server_next_p95_seconds"] == 0.4
    assert "text" not in row and "secret" not in json.dumps(row)


async def test_native_bridge_serves_the_existing_local_aec_probe(tmp_path, native_args):
    app = build_app(tmp_path, load_backends(), {}, native_args)
    async with TestServer(app, host="127.0.0.1") as server, ClientSession() as client:
        async with client.get(
            server.make_url("/audio-probe"), headers={"Host": "example.com"}
        ) as response:
            assert response.status == 403
        async with client.get(server.make_url("/audio-probe")) as response:
            assert response.status == 200
            body = await response.text()
            policy = response.headers["Content-Security-Policy"]
            nonce = re.search(r"script-src 'nonce-([^']+)'", policy).group(1)
            assert f'<script nonce="{nonce}">' in body
            assert "connect-src 'none'" in policy
            assert "Nothing is uploaded" in body
            assert "echoCancellation:{ideal:requested}" in body
            assert response.headers["Permissions-Policy"] == "microphone=(self)"


async def test_native_bridge_binds_spotify_status_and_token_routes_without_live_auth(
    tmp_path, native_args
):
    app = build_app(tmp_path, load_backends(), {}, native_args)

    class Request:
        headers = {"Host": "127.0.0.1:8765"}
        content_type = "application/json"

        async def json(self):
            return {}

    def handler(path, method):
        routes = list(app.router.routes())
        return next(
            route.handler
            for route in routes
            if route.resource.canonical == path and route.method == method
        )

    try:
        status_response = await handler("/api/voice/music/status", "GET")(Request())
        assert status_response.status == 200
        assert json.loads(status_response.body) == {
            "configured": False,
            "status": "unavailable",
            "output_backend": "mock",
            "audible": None,
        }
        token_response = await handler("/api/voice/music/token", "GET")(Request())
        token = json.loads(token_response.body)
        assert token_response.status == 409
        assert token["available"] is False
        assert token["code"] == "music_streaming_scope_required"
    finally:
        await app.cleanup()


async def test_native_browser_opt_in_saves_bounded_received_audio(
    tmp_path, monkeypatch, native_args
):
    import wave

    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    native_args.diagnostic_audio = True
    native_args.diagnostic_max_seconds = 1
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            ready = await next_json(ws, "ready")
            assert ready["diagnostic_audio"] is True
            await ws.send_bytes(b"\1\0" * 512)
            await ws.send_json({"type": "stop"})
            while (await ws.receive()).type not in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                pass
    path = tmp_path / "var/recordings/diagnostics" / ready["run_id"]
    with wave.open(str(path / "browser-received.wav"), "rb") as captured:
        assert captured.getparams().nchannels == 1
        assert captured.readframes(512) == b"\1\0" * 512
    assert json.loads((path / "manifest.json").read_text())["raw"]["received_bytes"] == 1024


async def test_browser_stop_fences_a_writer_waiting_for_capacity():
    class Socket:
        closed = False

        def __init__(self):
            self.packets = []

        async def send_bytes(self, b):
            self.packets.append(b)

        async def send_json(self, _):
            pass

    socket = Socket()
    sink = BrowserSink(socket)
    for _ in range(16):
        await sink(b"\1\0")
    delayed = asyncio.create_task(sink(b"\2\0"))
    await asyncio.sleep(0)
    await sink.stop()
    assert await delayed is False
    assert len(socket.packets) == 16 and not sink.pending


def test_native_replay_event_ring_retains_turn_lifecycle_through_streamed_playback():
    from types import SimpleNamespace

    from me3_voice.native_app import NATIVE_REPLAY_EVENT_HISTORY_LIMIT, NativeApplication

    app = NativeApplication(SimpleNamespace(session_id="event-retention"), object(), annotated=True)
    app.event("asr_final")
    app.event("llm_usage", prompt_tokens=123, completion_tokens=45)
    app.event("tts_start")
    for _ in range(2890):
        app.event("playback_packet", bytes=1280)

    names = [event["event"] for event in app.events]
    assert names[:3] == ["asr_final", "llm_usage", "tts_start"]
    assert len(app.events) == 2893 < NATIVE_REPLAY_EVENT_HISTORY_LIMIT


def test_native_voice_latency_is_turn_correlated_and_separates_ack_audio():
    from collections import deque
    from types import SimpleNamespace

    from me3_voice.native_app import NativeApplication, _voice_latency_for_turn

    rows = [
        {"event": "vad_speech_stopped", "at": 1.0, "sequence": 1, "turn_id": 0, "utterance_id": 7},
        {"event": "asr_final", "at": 1.3, "sequence": 2, "turn_id": 0, "utterance_id": 7},
        {"event": "turn_started", "at": 1.31, "sequence": 3, "turn_id": 1},
        {"event": "tts_progress_started", "at": 1.5, "sequence": 4, "turn_id": 1, "response_id": "ack-1"},
        {"event": "tts_progress_start", "at": 1.51, "sequence": 5, "turn_id": 1},
        {"event": "tts_progress_pcm", "at": 1.6, "sequence": 6, "turn_id": 1},
        {"event": "playback_packet", "at": 1.62, "sequence": 7, "turn_id": 1, "response_id": "ack-1"},
        {"event": "llm_request", "at": 1.7, "sequence": 8, "turn_id": 1, "request_id": "llm-1"},
        {"event": "llm_provider_request_started", "at": 1.75, "sequence": 9, "turn_id": 1, "request_id": "provider-1"},
        {"event": "llm_first_token", "at": 2.0, "sequence": 10, "turn_id": 1, "request_id": "provider-1"},
        {"event": "tts_response_started", "at": 2.1, "sequence": 11, "turn_id": 1, "response_id": "answer-1"},
        {"event": "tts_start", "at": 2.2, "sequence": 12, "turn_id": 1, "response_id": "answer-1", "segment_id": 1},
        {"event": "tts_pcm", "at": 2.4, "sequence": 13, "turn_id": 1, "response_id": "answer-1", "segment_id": 1},
        {"event": "playback_packet", "at": 2.45, "sequence": 14, "turn_id": 1, "response_id": "answer-1"},
    ]
    metrics = _voice_latency_for_turn(
        rows + [{"event": "turn_completed", "at": 3.0, "sequence": 15, "turn_id": 1}],
        1,
    )
    assert metrics == {
        "turn_id": 1,
        "asr_utterance_id": 7,
        "response_kind": "final_answer",
        "vad_speech_end_to_asr_final_ms": 300.0,
        "llm_provider_request_to_first_token_ms": 250.0,
        "tts_start_to_first_pcm_ms": 200.0,
        "first_pcm_to_client_delivery_ms": 50.0,
        "acknowledgement_audio": [
            {
                "response_kind": "acknowledgement",
                "tts_start_to_first_pcm_ms": 90.0,
                "first_pcm_to_client_delivery_ms": 20.0,
            }
        ],
    }
    # The same summary is emitted by the real turn-completion event path.
    emitted = []
    state = SimpleNamespace(
        session_id="latency-test",
        context=SimpleNamespace(get_messages=lambda: []),
    )
    app = NativeApplication(state, object(), annotated=True, on_event=emitted.append)
    app.events = deque(rows, maxlen=64)
    app.event_count = rows[-1]["sequence"]
    app.turn_id = 1
    app.event("turn_completed", owner_turn_id=1)
    assert emitted[-1]["event"] == "voice_latency"
    assert emitted[-1]["response_kind"] == "final_answer"
    assert emitted[-1]["first_pcm_to_client_delivery_ms"] == 50.0
    app.event("tts_response_started", response_id="final-context")
    app.event("tts_start")
    app.event("tts_pcm")
    app.event("playback_packet", response_id=None)
    app.event("tts_progress_started", response_id="ack-context", owner_id="task-1")
    app.event("tts_progress_pcm", owner_id="task-1")
    app.event("playback_packet", response_id="ack-context")
    audio_events = [
        row
        for row in app.events
        if row.get("event") in {"tts_start", "tts_pcm", "tts_progress_pcm", "playback_packet"}
    ][-5:]
    assert all(row.get("response_id") == "final-context" for row in audio_events[:3])
    assert all(row.get("response_kind") == "final_answer" for row in audio_events[:3])
    assert audio_events[3]["response_id"] == "ack-context"
    assert audio_events[3]["response_kind"] == "acknowledgement"
    assert audio_events[4]["response_kind"] == "acknowledgement"


def test_native_voice_latency_leaves_missing_or_reversed_marks_unavailable():
    from me3_voice.native_app import _elapsed_ms, _voice_latency_for_turn

    assert _elapsed_ms({"at": 2.0}, {"at": 1.0}) is None
    assert _elapsed_ms({"at": 1.0}, {}) is None
    stale = [
        {"event": "asr_final", "at": 1.0, "sequence": 1, "turn_id": 0, "utterance_id": 3},
        {"event": "turn_started", "at": 1.1, "sequence": 2, "turn_id": 1},
        {"event": "turn_completed", "at": 1.2, "sequence": 3, "turn_id": 1},
        {"event": "turn_started", "at": 2.0, "sequence": 4, "turn_id": 2},
        {"event": "turn_completed", "at": 2.5, "sequence": 5, "turn_id": 2},
    ]
    metrics = _voice_latency_for_turn(stale, 2)
    assert metrics["asr_utterance_id"] is None
    assert metrics["vad_speech_end_to_asr_final_ms"] is None


def test_native_transcript_projection_shows_live_asr_partial_without_context_insertion():
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view
    from me3_voice.configuration import load_backends

    state = SimpleNamespace(
        context=LLMContext([{"role": "user", "content": "previous finalized question"}]),
        session_id="partial-session",
        backends=load_backends(),
        task_runtime=None,
    )
    application = SimpleNamespace(
        partial_transcript="current words so far",
        projection="listening",
        event_count=4,
    )
    view = context_view(state, application)
    assert view["messages"][-1] == {
        "id": "2:user",
        "turn_id": 2,
        "role": "user",
        "text": "current words so far",
        "confirmed_text": "",
        "status": "partial",
        "delivery": "asr_partial",
    }
    assert state.context.get_messages() == [
        {"role": "user", "content": "previous finalized question"}
    ]


def test_native_transcript_revision_advances_for_task_and_action_updates():
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view
    from me3_voice.configuration import load_backends

    owned = {"task_id": "current-task", "turn_id": 1}
    previous = {"task_id": "previous-task", "turn_id": 1}
    task_store = SimpleNamespace(revision=2, snapshot=lambda: [previous, owned])
    state = SimpleNamespace(
        context=LLMContext(),
        session_id="revision-session",
        backends=load_backends(),
        task_runtime=SimpleNamespace(store=task_store),
    )
    application = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=4,
        action_revision=0,
        fast_action_rows=[],
        _voice_owned_task=lambda task_id: owned if task_id == owned["task_id"] else None,
    )

    first_view = context_view(state, application)
    assert first_view["tasks"] == [owned]
    first = first_view["revision"]
    application.action_revision += 1
    second = context_view(state, application)["revision"]
    task_store.revision += 1
    third = context_view(state, application)["revision"]

    assert second > first
    assert third > second


async def test_native_transcript_view_archives_session_with_existing_publisher(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext
    import me3_voice.transcript as transcript_module
    from me3_voice.native_bridge import NativeTranscriptView
    from me3_voice.transcript import TranscriptPublisher
    from me3_voice.configuration import load_backends

    monkeypatch.setattr(transcript_module, "output_path", lambda value: Path(value))
    archive_dir = tmp_path / "voice-sessions"
    archive_dir.mkdir()
    state = SimpleNamespace(
        context=LLMContext(
            [
                {"role": "user", "content": "Turn the simulated lamp on"},
                {"role": "assistant", "content": "The simulated lamp is on."},
            ]
        ),
        session_id="native-session-archive",
        backends=load_backends(),
        task_runtime=None,
    )
    application = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=3,
        fast_action_rows=[],
    )
    publisher = TranscriptPublisher(
        NativeTranscriptView(state, application),
        path=tmp_path / "current.json",
        archive_dir=archive_dir,
    )
    await publisher.__aenter__()
    await publisher.__aexit__(None, None, None)

    archived = json.loads((archive_dir / "native-session-archive.json").read_text())
    assert [row["text"] for row in archived["messages"]] == [
        "Turn the simulated lamp on",
        "The simulated lamp is on.",
    ]
    assert archived["mode"] == "mock"


def test_native_transcript_response_projection_distinguishes_interrupted_rain_answer():
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view
    from me3_voice.configuration import load_backends

    response_id = "response-123"
    state = SimpleNamespace(
        context=LLMContext(
            [
                {"role": "user", "content": "Explain rain in three sentences."},
                {
                    "role": "assistant",
                    "content": "First sentence. Second sentence. Third sentence.",
                },
            ]
        ),
        session_id="rain-turn-session",
        backends=load_backends(),
        task_runtime=None,
    )
    application = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=8,
        fast_action_rows=[],
        response_by_conversation_turn={1: response_id},
        response_records={
            response_id: {
                "response_id": response_id,
                "response_state": "interrupted",
                "generated_state": "completed",
                "tts_state": "interrupted",
                "playback_state": "partially_acknowledged",
                "tts_segments": [
                    {
                        "segment_id": 1,
                        "characters": 48,
                        "text_sha256": "a" * 16,
                        "state": "cancelled",
                    }
                ],
                "playback_packets": 12,
                "playback_acks": 7,
                "superseded_by_turn_id": 2,
                "interruption_reason": "vad_barge_in",
            }
        },
    )
    row = context_view(state, application, archived=True)["messages"][1]
    assert row["status"] == "interrupted"
    assert row["delivery"] == "generated"
    assert row["generation_complete"] is True
    assert row["confirmed_text"] == ""
    assert row["response"]["state"] == "interrupted"
    assert row["response"]["archive"] == "included"
    assert row["response"]["superseded_by_turn_id"] == 2


def test_native_tts_submission_event_keeps_only_sanitized_identity_and_hash():
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="tts-event-session", context=LLMContext()), object()
    )
    app.event("tts_response_started", response_id="response-456")
    event = app.event(
        "tts_start",
        response_id="response-456",
        segment_id=1,
        characters=12,
        submitted_text="The lamp is red.",
    )
    assert "submitted_text" not in event
    assert "The lamp is red." not in repr(event)
    segment = app.response_records["response-456"]["tts_segments"][0]
    assert segment["characters"] == len("The lamp is red.")
    assert len(segment["text_sha256"]) == 16


def test_native_transcript_projects_generated_but_uncommitted_interrupted_response():
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.native_bridge import context_view

    response_id = "partial-response-1"
    state = SimpleNamespace(
        session_id="partial-session",
        context=LLMContext([{"role": "user", "content": "Explain rain."}]),
        task_runtime=None,
        backends=load_backends(),
    )
    app = SimpleNamespace(
        partial_transcript=None,
        projection="listening",
        event_count=2,
        fast_action_rows=[],
        response_order=[response_id],
        response_by_conversation_turn={1: response_id},
        response_records={
            response_id: {
                "response_id": response_id,
                "turn_id": 1,
                "generated_text": "Rain forms when water vapor cools.",
                "generated_state": "partial",
                "response_state": "interrupted",
                "tts_state": "interrupted",
                "playback_state": "partially_acknowledged",
                "playback_packets": 8,
                "playback_acks": 3,
                "tts_segments": [{"segment_id": 1, "state": "cancelled"}],
            }
        },
    )
    answer = next(
        row
        for row in context_view(state, app, archived=True)["messages"]
        if row["role"] == "assistant"
    )
    assert answer["text"] == "Rain forms when water vapor cools."
    assert answer["confirmed_text"] == ""
    assert answer["status"] == "interrupted"
    assert answer["delivery"] == "generated"
    assert answer["response"]["response_id"] == response_id


async def test_native_followup_window_starts_after_response_and_resets_on_followup():
    from types import SimpleNamespace

    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="followup-test"),
        object(),
        wake=SimpleNamespace(frame_samples=320),
        conversation_mode="followup",
        followup_seconds=0.06,
    )
    app.parts = SimpleNamespace(stt=SimpleNamespace(speaking=False))
    app.gate.active = True
    app.projection = "speaking"
    app.settled.clear()
    app._arm_followup_timer()
    await asyncio.sleep(0.08)
    assert app.gate.active  # An in-progress answer never expires.

    app.projection = "listening"
    app.settled.set()
    app._arm_followup_timer()
    await asyncio.sleep(0.03)
    app._cancel_followup_timer()  # The next valid user turn resets the window.
    app._arm_followup_timer()
    await asyncio.sleep(0.04)
    assert app.gate.active
    await asyncio.sleep(0.04)
    assert not app.gate.active
    assert any(row["event"] == "engagement_expired" for row in app.events)


async def test_native_continuous_conversation_survives_idle_without_wake_standby():
    from types import SimpleNamespace

    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="continuous-conversation-test"),
        object(),
        wake=SimpleNamespace(frame_samples=320),
        conversation_mode="continuous",
        followup_seconds=0.02,
    )
    app.parts = SimpleNamespace(stt=SimpleNamespace(speaking=False))
    app.gate.active = True
    app.projection = "listening"
    app.settled.set()
    app._arm_followup_timer()
    assert app.followup_timer is None
    await app._handle_user_idle()
    await asyncio.sleep(0.04)
    assert app.gate.active
    assert app.projection == "listening"


async def test_native_wake_without_speech_expires_but_processing_stays_active():
    from types import SimpleNamespace
    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="wake-lease-test"),
        object(),
        wake=SimpleNamespace(frame_samples=320),
        conversation_mode="followup",
        followup_seconds=0.04,
    )
    app.parts = SimpleNamespace(stt=SimpleNamespace(speaking=False))
    app.gate.active = True
    app._on_wake({"backend": "fixture"})
    assert app.projection == "listening"
    assert app.followup_timer is not None
    app.settled.clear()
    app.projection = "thinking"
    await asyncio.sleep(0.06)
    assert app.gate.active
    app.settled.set()
    app.projection = "listening"
    app._arm_followup_timer()
    await asyncio.sleep(0.06)
    assert not app.gate.active


async def test_native_followup_expiry_waits_for_browser_scheduled_playback():
    from types import SimpleNamespace
    from me3_voice.native_app import NativeApplication

    app = NativeApplication(
        SimpleNamespace(session_id="playback-lease-test"),
        object(),
        wake=SimpleNamespace(frame_samples=320),
        conversation_mode="followup",
        followup_seconds=0.04,
    )
    app.parts = SimpleNamespace(stt=SimpleNamespace(speaking=False))
    app.gate.active = True
    app.projection = "listening"
    app.active_response_id = "response-1"
    app.response_records["response-1"] = {
        "response_state": "completed",
        "playback_acks": 0,
        "playback_state": "queued",
    }
    app.event("playback_ack", response_id="response-1", queued_audio_seconds=0.08)
    app._arm_followup_timer()
    await asyncio.sleep(0.06)
    assert app.gate.active
    await asyncio.sleep(0.075)
    assert not app.gate.active


async def test_delayed_asr_interrupt_through_websocket_never_publishes_old_turn(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge
    from me3_voice.speech import SpeechClient

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setattr(
        module,
        "SpeechClient",
        lambda **kw: SpeechClient(**kw, transcript=["obsolete", "current"], delay=0.3),
    )
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            await next_json(ws, "ready")
            await send_turn(ws)
            async with asyncio.timeout(10):
                while (await next_json(ws, "runtime_event"))["data"][
                    "event"
                ] != "asr_finish_requested":
                    pass
            await ws.send_json({"type": "interrupt"})
            await next_json(ws, "audio_stop")
            await send_turn(ws)
            view = await wait_answer(ws)
            assert [m["text"] for m in view["messages"] if m["role"] == "user"] == ["current"]
            assert "obsolete" not in str(view["messages"])


async def test_barge_in_before_ack_rejects_old_text_and_late_ack(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge
    import me3_voice.native_llm as model_module

    class Stream:
        calls = 0

        async def respond(self, _):
            self.calls += 1
            if self.calls == 1:
                yield "First sentence. "
                await asyncio.sleep(0.5)
                yield "Obsolete tail."
            else:
                yield "Current reply."

    upstream = Stream()
    monkeypatch.setattr(model_module, "MockLLM", lambda: upstream)
    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            await next_json(ws, "ready")
            await send_turn(ws)
            async with asyncio.timeout(10):
                while True:
                    message = await ws.receive()
                    if message.type == WSMsgType.BINARY:
                        old_id = struct.unpack("<I", message.data[:4])[0]
                        break
            await ws.send_json({"type": "interrupt"})
            stop = await next_json(ws, "audio_stop", acknowledge=False)
            assert type(stop["id"]) is int
            await ws.send_json({"type": "audio_stop_ack", "id": stop["id"]})
            async with asyncio.timeout(5):
                while True:
                    mark = (await next_json(ws, "runtime_event", acknowledge=False))["data"]
                    if mark["event"] == "audio_queue_cleared_ack":
                        break
            assert mark["stop_id"] == stop["id"]
            assert mark["audio_stop_send_started_at"] <= mark["at"]
            await ws.send_json({"type": "audio_ack", "id": old_id})
            await send_turn(ws)
            view = await wait_answer(ws)
            assistants = [m["text"] for m in view["messages"] if m["role"] == "assistant"]
            assert assistants == ["Current reply."]
            assert upstream.calls == 2


async def test_same_connection_asr_failure_can_be_interrupted_and_recovered(
    tmp_path, monkeypatch, native_args
):
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.playback import MemorySink
    from me3_voice.speech import SpeechClient

    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "tasks.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "workspace"))
    state = NativeState(
        speech_factory=lambda: SpeechClient(
            transcript=["failed", "recovered"], delay=0.2, timeout=5
        )
    )
    try:
        async with NativeApplication(state, MemorySink(), annotated=True) as app:
            state.speech.timeout = 0.03
            with pytest.raises(RuntimeError):
                await app.utterance(b"\1\0" * 512)
            assert not [m for m in state.context.get_messages() if m.get("role") == "assistant"]
            await app.interrupt()
            state.speech.timeout = 5
            await app.utterance(b"\1\0" * 512)
            assert [m["content"] for m in state.context.get_messages() if m["role"] == "user"] == [
                "recovered"
            ]
    finally:
        await state.close()


async def test_native_feed_recovers_after_completed_turn_error_without_reconnecting(
    tmp_path, monkeypatch
):
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink

    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "tasks.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "workspace"))
    state = NativeState()
    try:
        async with NativeApplication(state, MemorySink(), annotated=True) as app:
            app.error = RuntimeError("simulated completed turn failure")
            await app.feed(b"\1\0" * 512, 0)
            await app.drain()
            assert app.error is None
            recovered = [event for event in app.events if event["event"] == "turn_recovered"]
            assert len(recovered) == 1
            assert recovered[0]["error_code"] == "pipeline_error"
    finally:
        await state.close()


async def test_native_known_device_command_uses_shared_fast_route_without_llm():
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    class NoLLM:
        async def respond(self, _history):
            raise AssertionError("known device commands must not call the LLM")
            yield "unreachable"

    state = NativeState()
    llm = None
    try:
        app = NativeApplication(state, object(), annotated=True)
        app.turn_id = 1
        context = LLMContext()
        context.add_message({"role": "user", "content": "Hey Swift, turn on the light."})
        llm = NativeLLM(state.backends["llm"], upstream=NoLLM(), fast_action=app.fast_action)
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        spoken = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert state.tasks.actions.light is True
        assert "lamp" in spoken
        assert llm.calls == 0
        assert any(e["event"] == "fast_action_completed" for e in app.events)
        route = next(e for e in app.events if e["event"] == "route_selected")
        assert route["route"] == "fast_command"
        assert isinstance(route["at"], (int, float))
        assert app.fast_action_rows[-1]["kind"] == "light_on"
        assert app.fast_action_rows[-1]["status"] == "simulated"
        from me3_voice.native_bridge import context_view

        projected = context_view(state, app)["tasks"]
        assert projected[-1]["kind"] == "light_on"
        assert projected[-1]["widget"]["type"] == "lamp"
        assert projected[-1]["summary"].startswith("Simulated")
        context.set_messages([])
        context.add_message({"role": "user", "content": "Pause the music"})
        app.turn_id = 2
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        assert (
            "pause" in "".join(chunk.choices[0].delta.content or "" for chunk in chunks).casefold()
        )
        assert llm.calls == 0
        assert len(state.tasks.music.operations) == 1
        context.set_messages([])
        context.add_message({"role": "user", "content": "Set the music volume to 30 percent"})
        app.turn_id = 3
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        spoken = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert "30%" in spoken
        assert state.tasks.music.mock_volume == 30
        assert llm.calls == 0
        assert len(state.tasks.music.operations) == 2

        context.set_messages([])
        context.add_message({"role": "user", "content": "time is it in Manila"})
        app.turn_id = 4
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        time_answer = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert time_answer.startswith("It's ") and time_answer.endswith(" local time.")
        assert len(time_answer) < len(app.fast_action_rows[-1]["summary"])
        assert app.fast_action_rows[-1]["summary"].startswith("System local time is")
        assert llm.calls == 0
        assert app.fast_action_rows[-1]["kind"] == "get_time"

        # A contextual percentage is only routed directly when the immediately
        # preceding turn actually completed a light action. This exercises the
        # same OpenAI-shaped tool history Pipecat supplies to the native LLM.
        context.set_messages(
            [
                {"role": "user", "content": "Turn on the lights"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "light-call",
                            "type": "function",
                            "function": {
                                "name": "run_device_action",
                                "arguments": '{"action":"light_on"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "light-call",
                    "content": '{"result_status":"verified","summary":"Lamp is on."}',
                },
                {"role": "assistant", "content": "The lamp is on."},
                {"role": "user", "content": "Dim it to thirty percent"},
            ]
        )
        app.turn_id = 2
        llm.new_turn()
        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        assert "30" in "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert state.tasks.actions.brightness == 30
        assert llm.calls == 0
    finally:
        if llm is not None:
            await llm._client.close()
        await state.close()


async def test_fast_music_receipt_maps_provider_mode_to_safe_ui_evidence():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.native_bridge import context_view
    from me3_voice.providers import ToolCall

    state = NativeState()
    try:
        app = NativeApplication(state, object(), annotated=True)
        call = ToolCall("music-ui-1", "control_music", {"action": "play", "query": "artist"})
        row = app._fast_action_row(call, 1)
        app._finish_fast_action(
            row,
            result={
                "mode": "spotify",
                "result_status": "reported",
                "summary": "Spotify reported playback started.",
                "display": "playing",
            },
        )

        receipt = context_view(state, app)["tasks"][-1]
        assert receipt["kind"] == "music_play"
        assert receipt["status"] == "reported"
        assert receipt["mode"] == "reported"
        assert receipt["summary"] == "Spotify reported playback started."
    finally:
        await state.close()


async def test_interrupted_fast_action_is_recorded_before_later_unrelated_turn():
    from me3_voice.native_app import NativeApplication, NativeState
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    try:
        app = NativeApplication(state, object(), annotated=True)
        app.turn_id = 1
        context = LLMContext()
        context.add_message({"role": "user", "content": "Turn on the lights"})

        answer = await app.fast_action(context)
        assert "lamp" in answer.casefold()
        assert state.tasks.actions.effects == 1

        messages = context.get_messages()
        assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
        call = messages[-2]["tool_calls"][0]
        result = json.loads(messages[-1]["content"])
        assert call["function"]["name"] == "run_device_action"
        assert result["ok"] is True
        assert result["action"] == "light_on"

        # The spoken confirmation may be interrupted before Pipecat commits its
        # assistant text. The next unrelated request must still see the tool
        # result and must not execute the previous command a second time.
        context.add_message({"role": "user", "content": "Explain how rain forms"})
        app.turn_id = 2
        assert await app.fast_action(context) is None
        assert state.tasks.actions.effects == 1
    finally:
        await state.close()


@pytest.mark.parametrize(
    ("utterance", "action", "decision"),
    [
        ("Yes, push it.", "github_publish", "approve"),
        ("Approve", "github_pages_deploy", "approve"),
        ("Approve", "github_publish_and_deploy", "approve"),
        ("Approved.", "github_publish_and_deploy", "approve"),
        ("Cancel", "github_publish", "reject"),
    ],
)
async def test_spoken_github_confirmation_targets_exact_current_session_approval(
    utterance, action, decision
):
    from me3_voice.native_app import NativeApplication, NativeState, _voice_task_request_id
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.session_id = "current-approval-session"
    task_id = "task_" + "a" * 24
    call_id = "website-task-call"
    request_id = _voice_task_request_id(state.session_id, call_id)
    approval = {
        "approval_id": "approval_" + "b" * 32,
        "action": action,
        "status": "pending",
        "expires": __import__("time").time() + 60,
        "revision": 7,
    }
    task = {
        "task_id": task_id,
        "call_id": call_id,
        "request_id": request_id,
        "status": "waiting_approval",
        "revision": 7,
    }
    controls = []

    class Runtime:
        store = type("Store", (), {"rows": {task_id: task}, "pending_approvals": {task_id: approval}})()
        github_writer = object()

        async def control(self, command, payload):
            controls.append((command, payload))
            return {"status": "approved" if payload["decision"] == "approve" else "rejected"}

        async def close(self):
            return None

    state.task_runtime = Runtime()
    app = NativeApplication(state, object(), annotated=True)
    context = LLMContext()
    context.add_message({"role": "user", "content": utterance})
    try:
        answer = await app.fast_action(context)
        assert "approval" in answer.casefold() or "cancelled" in answer.casefold()
        assert controls == [
            (
                "approve",
                {
                    "task_id": task_id,
                    "approval_id": approval["approval_id"],
                    "expected_revision": 7,
                    "decision": decision,
                },
            )
        ]
        event = next(event for event in app.events if event["event"] == "github_voice_approval_decided")
        assert event["task_id"] == task_id
        assert event["approval_id"] == approval["approval_id"]
        assert event["action"] == action
    finally:
        await state.close()


@pytest.mark.parametrize(
    ("utterance", "pages_request"),
    [
        ("create a new website on GitHub", False),
        ("edit the existing website on GitHub", False),
        ("push this file to GitHub", False),
        ("push this to GitHub", False),
        ("publish this on GitHub Pages", True),
        ("push this to GitHub.io", True),
        ("make this site live", True),
        ("approve that one", None),
        ("yes push it", None),
    ],
)
async def test_voice_github_request_prepares_but_never_approves(
    utterance, pages_request
):
    from me3_voice.native_app import NativeApplication, NativeState, _voice_task_request_id
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.session_id = "approval-session"
    task_id = "task_" + "c" * 24
    call_id = "active-website-task"
    task = {
        "task_id": task_id,
        "call_id": call_id,
        "request_id": _voice_task_request_id(state.session_id, call_id),
        "status": "running",
        "goal": "Build a website",
        "revision": 3,
    }
    controls = []
    steers = []

    class Runtime:
        store = type("Store", (), {"rows": {task_id: task}, "pending_approvals": {}})()
        github_writer = object()

        async def control(self, command, payload):
            controls.append((command, payload))

        async def steer(self, *args):
            steers.append(args)
            return {"status": "running"}

        async def close(self):
            return None

    state.task_runtime = Runtime()
    app = NativeApplication(state, object(), annotated=True)
    context = LLMContext()
    context.add_message({"role": "user", "content": utterance})
    try:
        answer = await app.fast_action(context)
        assert controls == []
        if pages_request is not None:
            assert len(steers) == 1
            expected_response = (
                "one approval" if pages_request else "before any repository change"
            )
            assert expected_response in answer.casefold()
            assert ("deploy_pages=true" in steers[0][2]) is pages_request
            assert "covers the branch push and public deployment" in steers[0][2]
        else:
            assert steers == []
            assert "no pending github approval" in answer.casefold()
    finally:
        await state.close()


@pytest.mark.parametrize(
    "utterance",
    [
        "Create a new website on GitHub Pages",
        "Edit the existing README file on GitHub",
    ],
)
async def test_first_github_request_starts_background_task_without_remote_write(utterance):
    from me3_voice.native_app import NativeApplication, NativeState
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.session_id = "new-github-session"
    state.task_runtime = type(
        "Runtime",
        (),
        {
            "store": type("Store", (), {"rows": {}, "pending_approvals": {}})(),
            "github_writer": object(),
            "close": lambda self: asyncio.sleep(0),
        },
    )()
    app = NativeApplication(state, object(), annotated=True)
    requests = []

    async def start_task(call, arguments):
        requests.append((call.name, arguments))
        return {"status": "queued", "task_id": "task_" + "f" * 24}

    app.dispatch_task_tool = start_task
    context = LLMContext()
    context.add_message({"role": "user", "content": utterance})
    try:
        response = await app.fast_action(context)
        assert "nothing will be pushed until you approve" in response
        assert len(requests) == 1
        assert requests[0][0] == "start_task"
        assert requests[0][1]["goal"].startswith(utterance)
        assert ("deploy_pages=true" in requests[0][1]["goal"]) is ("Pages" in utterance)
        assert not any(
            event.get("event") in {"github_voice_approval_decided", "github_write_dispatched"}
            for event in app.events
        )
    finally:
        await state.close()


async def test_make_this_site_live_reuses_the_saved_voice_owned_task():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from me3_voice.native_app import NativeApplication, NativeState, _voice_task_request_id

    state = NativeState()
    state.session_id = "saved-website-session"
    task_id = "task_" + "d" * 24
    row = {
        "task_id": task_id,
        "call_id": "saved-site",
        "request_id": _voice_task_request_id(state.session_id, "saved-site"),
        "status": "succeeded",
        "revision": 3,
        "_artifact_paths": ["index.html", "style.css"],
    }
    state.task_runtime = SimpleNamespace(
        store=SimpleNamespace(rows={task_id: row}, pending_approvals={}),
        github_writer=object(),
        publish_saved_site=AsyncMock(),
        close=AsyncMock(),
    )
    app = NativeApplication(state, object(), annotated=True)
    app.dispatch_task_tool = AsyncMock()
    try:
        answer = await app._handle_github_publish_request("Make this site live")
        assert "ask once" in answer
        state.task_runtime.publish_saved_site.assert_awaited_once_with(task_id, 3)
        app.dispatch_task_tool.assert_not_called()
    finally:
        await state.close()


async def test_native_saved_red_light_and_timer_use_controller_without_reasoning():
    from me3_voice.devices import DeviceActions, MockDeviceOutput
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.speech import SpeechClient
    from me3_voice.tools import MockTaskExecutor

    questions = ["Can you turn the lights on to red?", "Can you set a ten second timer?"]
    actions = DeviceActions(backend="pi", output=MockDeviceOutput())
    state = NativeState(
        tasks=MockTaskExecutor(actions=actions),
        speech_factory=lambda: SpeechClient(transcript=questions),
    )

    class NoLLM:
        async def respond(self, _history):
            raise AssertionError("these controller turns must not use the LLM")
            yield "unreachable"

    try:
        async with NativeApplication(state, MemorySink(), annotated=True, upstream=NoLLM()) as app:
            for _ in questions:
                await app.utterance(b"\1\0" * 1024)
            assert actions.effects == 2
            assert actions.light and actions.color == "red"
            assert actions.schedule["label"] == "10 seconds"
            assert actions.schedule["kind"] == "timer"
            assert app.llm.calls == 0
            assert [row["status"] for row in app.fast_action_rows] == ["accepted", "accepted"]
            assert "no sensor" in app.fast_action_rows[0]["summary"]
            assert "no buzzer" in app.fast_action_rows[1]["summary"]
            assert [
                row["content"]
                for row in state.context.get_messages()
                if row["role"] == "assistant" and row.get("content")
            ] == ["Light set to red.", "Timer set for 10 seconds on the display."]
            assert sum(event["event"] == "turn_completed" for event in app.events) == 2
    finally:
        await state.close()
        await actions.close()


async def test_native_github_task_status_replay_uses_session_task_without_reasoning():
    from me3_voice.native_app import NativeApplication, NativeState, _voice_task_request_id
    from me3_voice.playback import MemorySink
    from me3_voice.speech import SpeechClient

    questions = [
        "Can you create my personal website based on my available details and push to GitHub.io?",
        "But did you push the previous task already on GitHub IO?",
        "What's the status of the GitHub task?",
        "It was the task about personal website. I don't have the task ID.",
    ]
    task_id = "task_" + "a" * 24
    foreign_id = "task_" + "b" * 24
    website_id = "task_" + "c" * 24
    state = NativeState(speech_factory=lambda: SpeechClient(transcript=questions))
    state.session_id = "github-status-replay"
    state.owner_profile = {"display_name": "Fixture Owner"}

    class Runtime:
        github_writer = object()

        def __init__(self):
            from types import SimpleNamespace

            self.store = SimpleNamespace(rows={}, pending_approvals={})
            self.status_reads = []
            self.starts = []

        async def start(self, goal, *, request_id, turn_id, call_id):
            self.starts.append((goal, request_id, turn_id, call_id))
            self.store.rows[task_id] = {
                "task_id": task_id,
                "goal": goal,
                "request_id": request_id,
                "call_id": call_id,
                "status": "queued",
            }
            return {"task_id": task_id, "status": "queued"}

        async def status(self, requested_id):
            self.status_reads.append(requested_id)
            return {
                "task_id": requested_id,
                "status": self.store.rows[requested_id]["status"],
                "progress": [
                    {
                        "kind": "stage_update",
                        "payload": {
                            "stage": "build",
                            "message": "The site files are being prepared.",
                        },
                    }
                ],
            }

        async def close(self):
            pass

    class ModelSpy:
        calls = 0

        async def respond(self, _history):
            self.calls += 1
            yield "Could you tell me the task ID?"

    runtime = state.task_runtime = Runtime()
    model = ModelSpy()
    try:
        async with NativeApplication(state, MemorySink(), annotated=True, upstream=model) as app:
            await app.utterance(b"\1\0" * 1024)
            [(goal, request_id, _, call_id)] = runtime.starts
            assert goal.startswith(questions[0])
            assert "deploy_pages=true" in goal
            assert "Personal site identity: Fixture Owner" in goal
            assert request_id == _voice_task_request_id(state.session_id, call_id)
            runtime.store.rows[task_id]["status"] = "running"
            runtime.store.rows[foreign_id] = {
                "task_id": foreign_id,
                "goal": "Build and publish a GitHub website",
                "call_id": "voice-github-task-foreign",
                "request_id": _voice_task_request_id("other-session", "voice-github-task-foreign"),
                "status": "running",
            }
            runtime.store.rows[website_id] = {
                "task_id": website_id,
                "goal": "Build a local website",
                "call_id": "voice-fast-task-local",
                "request_id": _voice_task_request_id(state.session_id, "voice-fast-task-local"),
                "status": "running",
            }
            await app.utterance(b"\1\0" * 1024)
            await app.utterance(b"\1\0" * 1024)
            runtime.store.rows[task_id]["status"] = "completed_unverified"
            # Preserve the short intervening fragment from the Pi transcript.
            state.context.add_message({"role": "user", "content": "It was the"})
            await app.utterance(b"\1\0" * 1024)
            assert model.calls == 0
            assert runtime.status_reads == [task_id, task_id, task_id]
            assert [
                (row["task_id"], row["task_status"])
                for row in app.events
                if row["event"] == "task_progress_read"
            ] == [(task_id, "running"), (task_id, "running"), (task_id, "completed_unverified")]
            replies = [
                row["content"]
                for row in state.context.get_messages()
                if row["role"] == "assistant" and row.get("content")
            ]
            assert "nothing will be pushed until you approve" in replies[0]
            assert replies[1:] == [
                "The site files are being prepared.",
                "The site files are being prepared.",
                "The task finished, but its result needs review.",
            ]
            assert len(runtime.starts) == 1
            assert sum(row["event"] == "turn_completed" for row in app.events) == 4
    finally:
        await state.close()


def test_private_demo_owner_profile_persists_only_approved_facts(tmp_path):
    from me3_voice.native_bridge import _load_private_demo_owner_profile

    profile_path = tmp_path / "demo-owner.json"
    stored = {
        "display_name": "Edgardo Callagon Jr.",
        "aliases": ["Approved Alias"],
        "permissions": {"public_publish": False},
        "other_fact": "preserved",
    }
    profile_path.write_text(json.dumps(stored))
    profile_path.chmod(0o600)
    assert _load_private_demo_owner_profile(profile_path) == stored

    # A corrected fact is read from the same private file on the next load.
    stored["current_role"] = "Senior Data Platform Engineer"
    profile_path.write_text(json.dumps(stored))
    profile_path.chmod(0o600)
    assert _load_private_demo_owner_profile(profile_path) == stored
    assert profile_path.stat().st_mode & 0o777 == 0o600


async def test_owner_pronoun_questions_use_private_profile_without_model_call():
    from me3_voice.native_app import NativeApplication, NativeState
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.owner_profile = {
        "display_name": "Edgardo Callagon Jr.",
        "current_role": "Senior Data Platform Engineer",
    }
    state.owner_name = "Edgardo Callagon Jr."
    app = NativeApplication(state, object(), annotated=True)
    try:
        for utterance in ("What's my name?", "What is my current role?", "What do I do?"):
            context = LLMContext()
            context.add_message({"role": "user", "content": utterance})
            response = await app.fast_action(context)
            assert response is not None
            assert "Edgardo Callagon Jr." in response
            assert "Senior Data Platform Engineer" in response
        state.owner_profile = {"display_name": "Edgardo Callagon Jr."}
        context = LLMContext()
        context.add_message({"role": "user", "content": "What is my current role?"})
        assert "not confirmed" in await app.fast_action(context)
        assert all(
            event.get("route") == "local_demo_owner_identity"
            for event in app.events
            if event.get("event") == "route_selected"
        )
    finally:
        await state.close()


@pytest.mark.parametrize(
    ("utterance", "publish"),
    [
        ("Create my personal website", False),
        ("Create me a public profile", False),
        ("No, can you create me a personal website and push to GitHub.io?", True),
    ],
)
async def test_personal_site_task_gets_only_scoped_profile_context(utterance, publish):
    from me3_voice.native_app import NativeApplication, NativeState
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.owner_profile = {
        "display_name": "Edgardo Callagon Jr.",
        "current_role": "Senior Data Platform Engineer",
        "aliases": ["Fixture Alias"],
        "permissions": {"public_publish": False},
    }
    state.task_runtime = type(
        "Runtime",
        (),
        {
            "store": type("Store", (), {"rows": {}, "pending_approvals": {}})(),
            "github_writer": object(),
            "close": lambda self: asyncio.sleep(0),
        },
    )()
    app = NativeApplication(state, object(), annotated=True)
    requests = []

    async def start_task(call, arguments):
        requests.append(arguments["goal"])
        return {"status": "queued", "task_id": "task_" + "a" * 24}

    app.dispatch_task_tool = start_task
    try:
        for request in (utterance, "Create a website for my bakery"):
            context = LLMContext()
            context.add_message({"role": "user", "content": request})
            await app.fast_action(context)
        assert len(requests) == 2
        assert "Edgardo Callagon Jr." in requests[0]
        assert "Senior Data Platform Engineer" in requests[0]
        assert "Fixture Alias" not in requests[0]
        assert "public_publish" not in requests[0]
        assert "do not infer employer, contact details" in requests[0]
        assert "Public deployment requires explicit approval" in requests[0]
        assert "Search the exact owner name with GitHub" in requests[0]
        assert ("deploy_pages=true" in requests[0]) is publish
        assert "Edgardo Callagon Jr." not in requests[1]
        assert "Personal site identity:" not in requests[1]
    finally:
        await state.close()


async def test_tic_tac_toe_creation_and_status_use_the_same_background_task():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from me3_voice.native_app import NativeApplication, NativeState, _voice_task_request_id
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.session_id = "game-route-fixture"
    task_id = "task_" + "7" * 24
    runtime = state.task_runtime = SimpleNamespace(
        store=SimpleNamespace(rows={}, pending_approvals={}),
        status=AsyncMock(return_value={"status": "running", "progress": []}),
        close=AsyncMock(),
    )
    app = NativeApplication(state, object(), annotated=True)

    async def start(call, arguments):
        runtime.store.rows[task_id] = {
            "task_id": task_id,
            "call_id": call.call_id,
            "request_id": _voice_task_request_id(state.session_id, call.call_id),
            "status": "running",
            "goal": arguments["goal"],
        }
        return {"task_id": task_id, "status": "running"}

    app.dispatch_task_tool = AsyncMock(side_effect=start)
    context = LLMContext()
    try:
        context.add_message({"role": "user", "content": "Create a Tic-Tac-Toe game"})
        assert "game task in the background" in await app.fast_action(context)
        assert "index.html" in runtime.store.rows[task_id]["goal"]
        context.add_message({"role": "user", "content": "What's the status of the game?"})
        assert await app.fast_action(context) == "The task is in progress."
        game = runtime.store.rows[task_id]
        game["created"] = 2
        for number, session, created in [(8, state.session_id, 1), (9, "other-session", 3)]:
            other_id, call_id = f"task_{str(number) * 24}", f"game-task-{number}"
            runtime.store.rows[other_id] = {
                "task_id": other_id, "call_id": call_id,
                "request_id": _voice_task_request_id(session, call_id),
                "goal": "Create a Tic-Tac-Toe game", "status": "succeeded", "created": created,
            }
        context.add_message({"role": "user", "content": "What’s the status now?"})
        assert await app.fast_action(context) == "The task is in progress."
        game["status"] = "waiting_approval"
        runtime.status.return_value = {
            "status": "waiting_approval", "progress": [],
            "approvals": [{"action": "github_publish_and_deploy", "status": "pending"}],
        }
        context.add_message({"role": "user", "content": "Is it completed now?"})
        assert "one approval" in await app.fast_action(context)
        game["status"] = "running"
        runtime.status.return_value = {
            "status": "running", "progress": [],
            "approvals": [{"action": "github_publish_and_deploy", "status": "approved"}],
        }
        context.add_message({"role": "user", "content": "Check the status."})
        assert "waiting for the public link" in await app.fast_action(context)
        game["status"] = "succeeded"
        runtime.status.return_value = {
            "status": "succeeded", "progress": [],
            "publication": {"pages_url": "https://example.test/game/"},
        }
        context.add_message({"role": "user", "content": "I don't have it. Also check the latest one."})
        assert "live on GitHub Pages" in await app.fast_action(context)
        assert [entry.args for entry in runtime.status.await_args_list] == [(task_id,)] * 5
        assert app.dispatch_task_tool.await_count == 1
    finally:
        await state.close()


async def test_local_owner_reply_stays_local_and_repeat_keeps_profile_context():
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.conversation_context import LOCAL_OWNER_TURN, prepare_context
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.native_llm import NativeLLM
    from me3_voice.configuration import BackendSpec
    from me3_voice.request_budget import serialized_request

    state = NativeState()
    state.owner_profile = {"display_name": "Fixture Owner", "current_role": "Fixture Role"}
    app = NativeApplication(state, object(), annotated=True)
    llm = NativeLLM(BackendSpec())
    context = LLMContext()
    try:
        context.add_message({"role": "user", "content": "What is my current role?"})
        original = await app.fast_action(context)
        assert "Fixture Role" in original
        context.add_message({"role": "assistant", "content": original})
        context.add_message({"role": "user", "content": "Please say that again."})
        assert await app.fast_action(context) == original
        context.add_message({"role": "assistant", "content": original})
        context.add_message({"role": "user", "content": "Tell me about telescopes."})
        context.add_message({"role": "assistant", "content": "They collect light."})
        context.add_message({"role": "user", "content": "Repeat that."})
        assert await app.fast_action(context) is None
        _, policy, projected = prepare_context(context)
        await llm._update_settings(llm.Settings(system_instruction=policy))
        payload = serialized_request(await llm._prepare_provider_params(projected))
        assert b"Fixture Owner" not in payload and b"Fixture Role" not in payload
        assert LOCAL_OWNER_TURN.encode() not in payload
        assert b"They collect light." in payload
        assert [m["content"] for m in context.get_messages() if m["role"] == "assistant"] == [
            original,
            original,
            "They collect light.",
        ]
    finally:
        await llm._client.close()
        await state.close()


async def test_spoken_github_approval_is_session_scoped_and_action_mismatch_rejected():
    from me3_voice.native_app import NativeApplication, NativeState, _voice_task_request_id
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    state.session_id = "session-a"
    task_id = "task_" + "d" * 24
    call_id = "another-session-task"
    approval = {
        "approval_id": "approval_" + "e" * 32,
        "action": "github_pages_deploy",
        "status": "pending",
        "expires": __import__("time").time() + 60,
        "revision": 2,
    }
    task = {
        "task_id": task_id,
        "call_id": call_id,
        "request_id": _voice_task_request_id("session-b", call_id),
        "status": "waiting_approval",
        "revision": 2,
    }
    controls = []

    class Runtime:
        store = type("Store", (), {"rows": {task_id: task}, "pending_approvals": {task_id: approval}})()

        async def control(self, *args):
            controls.append(args)

        async def close(self):
            return None

    state.task_runtime = Runtime()
    app = NativeApplication(state, object(), annotated=True)
    context = LLMContext()
    context.add_message({"role": "user", "content": "Yes, push it"})
    try:
        answer = await app.fast_action(context)
        assert "no pending github approval" in answer.casefold()
        assert controls == []

        task["request_id"] = _voice_task_request_id(state.session_id, call_id)
        context.add_message({"role": "user", "content": "Yes, push it"})
        answer = await app.fast_action(context)
        assert "does not match" in answer.casefold()
        assert controls == []

        context.set_messages([])
        context.add_message({"role": "user", "content": "Approve that one"})
        answer = await app.fast_action(context)
        assert "did not clearly approve or cancel" in answer.casefold()
        assert controls == []

        approval["expires"] = __import__("time").time() - 1
        context.set_messages([])
        context.add_message({"role": "user", "content": "Approve"})
        answer = await app.fast_action(context)
        assert "no pending github approval" in answer.casefold()
        assert controls == []
    finally:
        await state.close()


async def test_native_websocket_test_text_routes_approval_without_provider_calls(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_bridge as bridge
    from me3_voice.native_app import NativeState, _voice_task_request_id

    monkeypatch.setenv("ME3_TEST_TEXT_INGRESS", "1")
    monkeypatch.setenv("ME3_TASK_DATABASE", str(tmp_path / "voice-approval.sqlite"))
    monkeypatch.setenv("ME3_TASK_WORKSPACE", str(tmp_path / "voice-approval-workspace"))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "voice-approval-offline")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "voice-approval-ledger.sqlite"))
    monkeypatch.setattr(bridge, "build_wake", lambda _args: Wake())
    captured = []

    class CapturedState(NativeState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(bridge, "NativeState", CapturedState)
    app = build_app(tmp_path, load_backends(), {}, native_args)
    async with TestServer(app) as server, ClientSession() as client:
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            ready = await next_json(ws, "ready")
            state = captured[-1]
            runtime = state.task_runtime
            assert runtime is not None
            task_call_id = "websocket-site-task"
            task, _ = await runtime.store.create(
                "Publish my verified website",
                _voice_task_request_id(ready["session_id"], task_call_id),
                1,
                task_call_id,
            )
            await runtime.store.update(task["task_id"], status="running")
            approval = await runtime.store.prepare_approval(
                task["task_id"],
                1,
                task["task_id"] + ":github-branch",
                "github_publish",
                "Push the approved artifact to owner/repo on the shown branch.",
                spoken_summary="GitHub approval: push index.html to owner/repo, branch swift/site. The repository is public.",
            )
            controls = []

            async def capture_control(command, payload):
                controls.append((command, payload))
                return {"task_id": payload["task_id"], "approval_id": payload["approval_id"], "status": "approved"}

            runtime.control = capture_control
            await ws.send_json({"type": "test_text", "text": "Yes, push it."})
            asr_bypassed = False
            async with asyncio.timeout(8):
                while True:
                    message = await ws.receive()
                    if message.type == WSMsgType.BINARY:
                        await ws.send_json(
                            {"type": "audio_ack", "id": struct.unpack("<I", message.data[:4])[0]}
                        )
                    elif message.type == WSMsgType.TEXT:
                        value = message.json()
                        event = value.get("data", {})
                        if (
                            value.get("type") == "runtime_event"
                            and event.get("event") == "test_text_injected"
                        ):
                            asr_bypassed = event.get("asr_bypassed") is True
                        if (
                            value.get("type") == "runtime_event"
                            and event.get("event") == "github_voice_approval_decided"
                        ):
                            break
            assert asr_bypassed is True
            assert controls == [
                (
                    "approve",
                    {
                        "task_id": task["task_id"],
                        "approval_id": approval["approval_id"],
                        "expected_revision": approval["revision"],
                        "decision": "approve",
                    },
                )
            ]
            assert runtime.budget.requests == 0
            assert state._task_budget is None or state._task_budget.requests == 0
            response_seen = False
            turn_completed = False
            async with asyncio.timeout(5):
                while not (response_seen and turn_completed):
                    message = await ws.receive()
                    if message.type == WSMsgType.BINARY:
                        await ws.send_json(
                            {"type": "audio_ack", "id": struct.unpack("<I", message.data[:4])[0]}
                        )
                    elif message.type == WSMsgType.TEXT:
                        value = message.json()
                        if value.get("type") == "transcript":
                            assistants = [
                                item
                                for item in value["data"]["messages"]
                                if item.get("role") == "assistant"
                            ]
                            response_seen = any(
                                "Approval recorded for that exact GitHub action" in str(
                                    item.get("text", item.get("content", ""))
                                )
                                for item in assistants
                            )
                        elif value.get("type") == "runtime_event":
                            turn_completed = (
                                value.get("data", {}).get("event") == "turn_completed"
                            )
            assert response_seen is True
            await ws.close()


async def test_outgoing_call_and_text_require_explicit_followup_confirmation():
    from me3_voice.communications import Communications
    from me3_voice.native_app import NativeApplication, NativeState
    from pipecat.processors.aggregators.llm_context import LLMContext

    state = NativeState()
    try:
        app = NativeApplication(state, object(), annotated=True)
        calls = []
        state.tasks.actions.communications = Communications(
            contacts=[
                {"name": "Alice Rivera", "aliases": ["Ally"], "phone_number": "+15551234567"},
                {"name": "Alex Home", "aliases": ["Alex"], "phone_number": "+15551234568"},
                {"name": "Alex Work", "aliases": ["Alex"], "phone_number": "+15551234569"},
            ]
        )

        async def count_execute(*args, **kwargs):
            call = args[1]
            calls.append(
                (
                    call.arguments.get("action"),
                    call.arguments.get("value"),
                    call.arguments.get("message"),
                )
            )
            return {"summary": "Offline confirmation intercepted; no phone action executed."}

        state.tasks.execute = count_execute
        context = LLMContext()
        context.add_message({"role": "user", "content": "Call the demo contact"})
        app.turn_id = 1
        prompt = await app.fast_action(context)
        assert "who would you like me to call" in prompt.casefold()
        assert calls == []

        context.add_message({"role": "user", "content": "Ally"})
        app.turn_id = 2
        prompt = await app.fast_action(context)
        assert "call alice rivera" in prompt.casefold()
        assert "repeat call to alice rivera" in prompt.casefold()
        assert calls == []

        context.add_message({"role": "user", "content": "repeat call to Alice Rivera"})
        app.turn_id = 3
        answer = await app.fast_action(context)
        assert calls == [("call_demo", "Alice Rivera", None)], answer
        assert any(event["event"] == "outgoing_action_confirmed" for event in app.events)

        context.set_messages([])
        original_body = "I'm running 10 minutes late."
        context.add_message(
            {"role": "user", "content": f"Draft a text to Alex saying {original_body}"}
        )
        app.turn_id = 4
        prompt = await app.fast_action(context)
        assert "more than one matching contact" in prompt.casefold()
        assert "Alex Home" in prompt and "Alex Work" in prompt
        assert calls == [("call_demo", "Alice Rivera", None)]

        context.add_message({"role": "user", "content": "Alex Work"})
        app.turn_id = 5
        prompt = await app.fast_action(context)
        assert "repeat message to alex work" in prompt.casefold()
        assert original_body in prompt
        assert calls == [("call_demo", "Alice Rivera", None)]

        edited_body = "I'll be there in fifteen minutes."
        context.add_message(
            {"role": "user", "content": f"Actually, change the message to {edited_body}"}
        )
        app.turn_id = 6
        prompt = await app.fast_action(context)
        assert edited_body in prompt and original_body not in prompt
        assert calls == [("call_demo", "Alice Rivera", None)]

        context.add_message({"role": "user", "content": "repeat message to Alex Work"})
        app.turn_id = 7
        answer = await app.fast_action(context)
        assert "offline confirmation intercepted" in answer.casefold()
        assert calls == [
            ("call_demo", "Alice Rivera", None),
            ("message_demo", "Alex Work", edited_body),
        ]

        context.set_messages([])
        context.add_message({"role": "user", "content": "Send a message to Unknown Person"})
        app.turn_id = 8
        answer = await app.fast_action(context)
        assert "couldn't find" in answer.casefold()
        assert len(calls) == 2

        context.set_messages([])
        context.add_message({"role": "user", "content": "Call Ally"})
        app.turn_id = 9
        prompt = await app.fast_action(context)
        assert "repeat call to alice rivera" in prompt.casefold()
        context.add_message({"role": "user", "content": "cancel"})
        app.turn_id = 10
        answer = await app.fast_action(context)
        assert "cancelled" in answer.casefold()
        assert len(calls) == 2
        assert any(
            event["event"] == "outgoing_action_confirmation_cancelled" for event in app.events
        )
    finally:
        await state.close()


def test_configurable_contact_lookup_handles_aliases_ambiguity_and_missing_names():
    from me3_voice.communications import Communications

    contacts = Communications(
        contacts=[
            {"name": "Alice Rivera", "aliases": ["Ally", "A. Rivera"], "phone_number": "+15551234567"},
            {"name": "Alex Home", "aliases": ["Alex"], "phone_number": "+15551234568"},
            {"name": "Alex Work", "aliases": ["Alex"], "phone_number": "+15551234569"},
        ]
    )
    assert contacts.match_contact("ALLY") == {"status": "matched", "name": "Alice Rivera"}
    assert contacts.match_contact("Alex") == {
        "status": "ambiguous",
        "matches": ["Alex Home", "Alex Work"],
    }
    assert contacts.match_contact("Unknown Person") == {"status": "missing", "matches": []}


async def test_native_fast_music_route_clarifies_missing_track_without_llm_or_tool_call():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    class NoLLM:
        async def respond(self, _history):
            raise AssertionError("incomplete music requests must clarify before calling tools")
            yield "unreachable"

    state = NativeState()
    llm = None
    try:
        app = NativeApplication(state, object(), annotated=True)
        app.turn_id = 1
        context = LLMContext()
        context.add_message({"role": "user", "content": "Can you turn on the music?"})
        llm = NativeLLM(state.backends["llm"], upstream=NoLLM(), fast_action=app.fast_action)
        llm.new_turn()

        chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
        answer = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert answer == "What song, artist, or playlist would you like me to play?"
        assert llm.calls == 0
        assert state.tasks.music.operations == {}
        assert any(event["event"] == "clarification_requested" for event in app.events)
    finally:
        if llm is not None:
            await llm._client.close()
        await state.close()


async def test_native_fast_route_does_not_intercept_research_or_delegated_work():
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    class CountingLLM:
        calls = 0

        async def respond(self, _history):
            self.calls += 1
            yield "Handled by the general conversation model."

    state = NativeState()
    upstream = CountingLLM()
    llm = None
    try:
        app = NativeApplication(state, object(), annotated=True)
        llm = NativeLLM(state.backends["llm"], upstream=upstream, fast_action=app.fast_action)
        for turn, text in enumerate(
            (
                "Research the latest Bank of England rate and summarize the source.",
                "Start a task to create a small website about tide pools.",
                "Could you explain how echo cancellation works?",
            ),
            start=1,
        ):
            app.turn_id = turn
            llm.new_turn()
            context = LLMContext()
            context.add_message({"role": "user", "content": text})
            chunks = [chunk async for chunk in await llm.get_chat_completions(context)]
            spoken = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
            assert "general conversation model" in spoken
            assert llm.calls == 1
        assert upstream.calls == 3
        assert not any(event["event"] == "fast_action_started" for event in app.events)
    finally:
        if llm is not None:
            await llm._client.close()
        await state.close()


def test_native_file_replay_uses_actual_cli_dispatch(tmp_path, monkeypatch):
    import wave
    import me3_voice.native_app as module
    from me3_voice.cli import main

    path = tmp_path / "input.wav"
    # Input fixtures may live in pytest's tmp_path; generated outputs may not.
    output = Path.cwd() / "var" / "scratch" / f"pytest-replay-{uuid4().hex}.json"
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"engine":"native","silence_seconds":0}')
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((b"\1\0" * 512) * 5 + (b"\0\0" * 512) * 4)
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "me3",
            "replay",
            str(path),
            "--conversation",
            "--runtime-config",
            str(runtime),
            "--wake-frame",
            "0",
            "--output",
            str(output),
        ],
    )
    main()
    value = json.loads(output.read_text())
    assert (
        value["engine"] == "native" and value["utterances"] == 1 and not value["unfinished_input"]
    )
    assert any(e["event"] == "turn_completed" for e in value["events"])


async def test_native_startup_failure_closes_partial_model_and_can_retry(monkeypatch):
    from me3_voice.native_app import NativeState, NativeApplication
    from me3_voice.playback import MemorySink

    original = NativeApplication.register_tools

    def broken(self):
        raise ValueError("fixture registration failure")

    state = NativeState()
    try:
        monkeypatch.setattr(NativeApplication, "register_tools", broken)
        app = NativeApplication(state, MemorySink(), annotated=True)
        with pytest.raises(ValueError):
            await app.__aenter__()
        assert app.llm._client.is_closed()
        monkeypatch.setattr(NativeApplication, "register_tools", original)
        async with NativeApplication(state, MemorySink(), annotated=True) as retry:
            await retry.utterance(b"\1\0" * 512)
            assert any(e["event"] == "turn_completed" for e in retry.events)
    finally:
        await state.close()


@pytest.mark.parametrize(
    (
        "answer_sentences",
        "inter_sentence_delay",
        "second_tts_startup_delay",
        "first_tts_startup_delay",
    ),
    [
        (("At sea level, water boils at 100 degrees Celsius.",), 0, 0, 0),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
            ),
            0,
            0,
            0,
        ),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
                "The boiling point changes when atmospheric pressure changes.",
            ),
            0,
            0,
            0,
        ),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
            ),
            3.1,
            0,
            0,
        ),
        (
            (
                "At sea level, water boils at 100 degrees Celsius.",
                "That is 212 degrees Fahrenheit.",
            ),
            0,
            3.2,
            0,
        ),
        (
            ("At sea level, water boils at 100 degrees Celsius.",),
            0,
            0,
            3.2,
        ),
    ],
    ids=(
        "one-sentence",
        "two-sentences",
        "three-sentences",
        "llm-idle-gap",
        "delayed-second-tts-pcm",
        "delayed-first-tts-pcm",
    ),
)
async def test_native_wav_replay_writes_sanitized_ledger_and_audio_summary(
    tmp_path,
    monkeypatch,
    answer_sentences,
    inter_sentence_delay,
    second_tts_startup_delay,
    first_tts_startup_delay,
):
    import sqlite3
    import wave
    from argparse import Namespace
    import me3_voice.native_app as native
    import me3_voice.readiness as readiness
    import me3_voice.task_runtime as task_runtime
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.research import ResearchOptions
    from me3_voice.speech import SpeechClient

    speech_timeouts = []
    settle_timeouts = []
    original_wait_for = native.asyncio.wait_for

    async def capture_wait_for(awaitable, timeout, *args, **kwargs):
        settle_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout, *args, **kwargs)

    class CapturingSpeechClient(SpeechClient):
        def __init__(self, *args, **kwargs):
            speech_timeouts.append(kwargs.get("timeout", 5))
            self.synthesis_count = 0
            super().__init__(*args, **kwargs)

        async def synthesize(self, key, text):
            self.synthesis_count += 1
            if self.synthesis_count == 1 and first_tts_startup_delay:
                await asyncio.sleep(first_tts_startup_delay)
            if self.synthesis_count == 2 and second_tts_startup_delay:
                await asyncio.sleep(second_tts_startup_delay)
            async for pcm in super().synthesize(key, text):
                yield pcm

    class ScriptedAnswer:
        def __init__(self, sentences, delay):
            self.sentences, self.delay = sentences, delay

        async def respond(self, _history):
            for index, sentence in enumerate(self.sentences):
                yield sentence
                if index + 1 < len(self.sentences) and self.delay:
                    await asyncio.sleep(self.delay)

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\1\0" * 5_120 + b"\0\0" * 12_800)
    monkeypatch.setattr(native, "native_vad", FixtureVAD)
    monkeypatch.setattr(readiness, "build_wake", lambda _args: None)
    monkeypatch.setattr(native, "SpeechClient", CapturingSpeechClient)
    monkeypatch.setattr(native.asyncio, "wait_for", capture_wait_for)
    original_task_runtime = task_runtime.TaskRuntime

    def isolated_task_runtime(*args, **kwargs):
        return original_task_runtime(
            *args,
            database=tmp_path / "task-runtime" / "tasks.sqlite",
            artifact_root=tmp_path / "task-artifacts",
            **kwargs,
        )

    monkeypatch.setattr(task_runtime, "TaskRuntime", isolated_task_runtime)
    ledger = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-replay-test-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    backend_config = tmp_path / "backends.json"
    backend_config.write_text(
        json.dumps(
            {
                "asr": {"timeout_seconds": 35},
                "llm": {"timeout_seconds": 45},
                "tts": {"timeout_seconds": 5 if first_tts_startup_delay else 60},
            }
        )
    )
    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / "result.json",
        trace=None,
        backend_config=backend_config,
        secrets_file=None,
        llm_secrets_file=None,
        wake_frame=0,
        transcript=True,
        mock_transcripts=["What is the boiling point of water at sea level?"],
        pace=0,
        playback_wav=tmp_path / "answer.wav",
    )
    result = await native.replay_native_application(
        args,
        RuntimeOptions(engine="native", research=ResearchOptions(False, 0)),
        upstream=ScriptedAnswer(answer_sentences, inter_sentence_delay),
    )
    assert native.NATIVE_REPLAY_SETTLE_TIMEOUT_SECONDS == 60
    assert 60 in settle_timeouts
    assert speech_timeouts == [45 if first_tts_startup_delay else 60]
    assert result["asr_transcription"] == "What is the boiling point of water at sea level?"
    assistant_text = " ".join(
        row["content"]
        for row in result["history"]
        if row.get("role") == "assistant" and isinstance(row.get("content"), str)
    )
    assert all(sentence in assistant_text for sentence in answer_sentences)
    assert "turn_completed" in result["completion_events"]
    assert sum(event["event"] == "tts_start" for event in result["events"]) == len(answer_sentences)
    assert sum(event["event"] == "tts_released" for event in result["events"]) == len(
        answer_sentences
    )
    summaries = [e for e in result["events"] if e["event"] == "tts_segment_summary"]
    assert len(summaries) == len(answer_sentences)
    assert all(e["status"] == "completed" for e in summaries)
    assert all(e["native_audio_seconds"] == pytest.approx(0.128) for e in summaries)
    assert all(
        e["diagnostics"]["status"] == "completed"
        and e["diagnostics"]["client_mcp"]["tts_next_count"] == 5
        for e in summaries
    )
    assert sum(e["event"] == "tts_response_started" for e in result["events"]) == 1
    assert sum(e["event"] == "tts_response_stop_queued" for e in result["events"]) == 1
    output_speaking_transitions = 2 if second_tts_startup_delay >= 3 else 1
    assert (
        sum(e["event"] == "output_speaking_started" for e in result["events"])
        == output_speaking_transitions
    )
    assert (
        sum(e["event"] == "output_speaking_stopped" for e in result["events"])
        == output_speaking_transitions
    )
    assert sum(e["event"] == "assistant_turn_started" for e in result["events"]) == 1
    assert sum(e["event"] == "assistant_turn_stopped" for e in result["events"]) == 1
    if second_tts_startup_delay >= 3:
        segment_two_start = next(
            e for e in result["events"] if e["event"] == "tts_start" and e.get("segment_id") == 2
        )
        segment_two_pcm = next(
            e
            for e in result["events"]
            if e["event"] == "tts_pcm" and e["sequence"] > segment_two_start["sequence"]
        )
        output_stops = [e for e in result["events"] if e["event"] == "output_speaking_stopped"]
        output_starts = [e for e in result["events"] if e["event"] == "output_speaking_started"]
        response_stop = next(
            e for e in result["events"] if e["event"] == "tts_response_stop_queued"
        )
        assert segment_two_start["sequence"] < output_stops[0]["sequence"]
        assert output_stops[0]["sequence"] < segment_two_pcm["sequence"]
        assert segment_two_pcm["sequence"] < output_starts[1]["sequence"]
        assert output_starts[1]["sequence"] < response_stop["sequence"]
    assert (
        next(e for e in result["events"] if e["event"] == "assistant_turn_stopped")["projection"]
        == "listening"
    )
    assert not any(
        e["event"] == "assistant_turn_stopped" and e["interrupted"] for e in result["events"]
    )
    event_names = [event["event"] for event in result["events"]]
    assert event_names.index("turn_completed") > max(
        index for index, name in enumerate(event_names) if name == "tts_pcm"
    )
    assert (
        event_names.index("tts_response_stop_queued")
        < max(index for index, name in enumerate(event_names) if name == "output_speaking_stopped")
        < event_names.index("turn_completed")
    )
    assert result["tts_audio"]["pcm16_valid"] and result["tts_audio"]["wav_valid"]
    assert result["request_ledger"] == {
        "run_id": "native-replay-test-01",
        "requests": [],
        "counts": {},
    }
    assert set(result["provider_budget"]) == {"asr", "llm", "tts"}
    assert all(not spec["allow_live"] for spec in result["provider_budget"].values())
    with wave.open(str(args.playback_wav), "rb") as rendered:
        assert rendered.getnchannels() == 1 and rendered.getsampwidth() == 2
        assert rendered.getframerate() == 16000 and rendered.getnframes() > 0
    assert not any(
        event["event"] == "tool_completed"
        and event.get("tool_name") in {"web_search", "retrieve_source"}
        for event in result["events"]
    )
    assert result["latency_seconds"]["turn_started_to_completed"] is not None
    assert "offline-test-secret" not in repr(result)
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


@pytest.mark.parametrize("credential_setup", ["valid", "missing", "misplaced"])
async def test_native_replay_tavily_credential_selection_reaches_only_mock_http(
    tmp_path, monkeypatch, credential_setup
):
    import wave
    from me3_voice import credentials as credential_module
    import me3_voice.native_app as native
    import me3_voice.readiness as readiness
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.providers import ToolCall
    from me3_voice.research import ResearchOptions, ResearchClient

    secret_root = tmp_path / "private-cache"
    monkeypatch.setattr(credential_module, "CACHE_ROOT", secret_root)
    provider_secrets = secret_root / "provider.env"
    llm_secrets = secret_root / "llm.env"
    if credential_setup == "valid":
        credential_module.write_secrets(
            provider_secrets, {"ME3_RESEARCH_TAVILY_API_KEY": "offline-test-tavily-key"}
        )
    elif credential_setup == "missing":
        credential_module.write_secrets(provider_secrets, {"ME3_DEEPGRAM_API_KEY": "unused-key"})
    else:
        credential_module.write_secrets(provider_secrets, {"ME3_DEEPGRAM_API_KEY": "unused-key"})
        credential_module.write_secrets(
            llm_secrets, {"ME3_RESEARCH_TAVILY_API_KEY": "misplaced-test-tavily-key"}
        )

    boundary_calls = []

    async def mock_research_http(self, method, url, **kwargs):
        boundary_calls.append((method, url, kwargs.get("headers", {}).get("Authorization")))
        if method == "POST" and url == "https://api.tavily.com/search":
            return {
                "data": json.dumps(
                    {
                        "results": [
                            {
                                "url": "https://fixture.example/japan-office",
                                "title": "Officeholder fixture",
                                "content": "A source lead for the current prime minister.",
                            }
                        ]
                    }
                ).encode(),
                "type": "application/json",
                "last_modified": None,
            }
        if method == "GET" and url == "https://fixture.example/japan-office":
            return {
                "data": (
                    b"As of the fixture date, Example Person holds the office of prime minister "
                    b"of Japan following a completed parliamentary selection."
                ),
                "type": "text/plain",
                "last_modified": None,
            }
        raise AssertionError("unexpected mocked research HTTP request")

    monkeypatch.setattr(ResearchClient, "read", mock_research_http)
    monkeypatch.setattr(native, "native_vad", FixtureVAD)
    monkeypatch.setattr(readiness, "build_wake", lambda _args: None)

    class ScriptedResearchModel:
        def __init__(self):
            self.calls = 0

        async def respond(self, _history):
            self.calls += 1
            if self.calls == 1:
                yield ToolCall(
                    "search-1", "web_search", {"query": "current prime minister of Japan"}
                )
            elif credential_setup == "valid" and self.calls == 2:
                yield ToolCall(
                    "retrieve-1",
                    "retrieve_source",
                    {"url": "https://fixture.example/japan-office"},
                )
            elif credential_setup == "valid":
                yield "The fixture source identifies the current officeholder [S1]."
            else:
                yield "I couldn't verify this from a retrieved source, so I won't guess."

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\1\0" * 5_120 + b"\0\0" * 12_800)
    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / f"{credential_setup}-result.json",
        trace=None,
        backend_config=None,
        secrets_file=provider_secrets,
        llm_secrets_file=llm_secrets if credential_setup == "misplaced" else None,
        wake_frame=0,
        transcript=True,
        mock_transcripts=["Who is the current prime minister of Japan?"],
        pace=0,
        playback_wav=None,
    )
    result = await native.replay_native_application(
        args,
        RuntimeOptions(engine="native", research=ResearchOptions(True, 4)),
        upstream=ScriptedResearchModel(),
    )
    search_events = [
        event
        for event in result["events"]
        if event.get("event") == "tool_completed" and event.get("tool_name") == "web_search"
    ]
    assert len(search_events) == 1
    if credential_setup == "valid":
        assert boundary_calls == [
            (
                "POST",
                "https://api.tavily.com/search",
                "Bearer offline-test-tavily-key",
            ),
            ("GET", "https://fixture.example/japan-office", None),
        ]
        assert search_events[0]["ok"]
        assert any(event["event"] == "turn_completed" for event in result["events"])
        assert "[S1]" in result["llm_response"]
    else:
        assert boundary_calls == []
        assert not search_events[0]["ok"]
        assert search_events[0]["error_code"] == "search_credential_missing"
        assert not any(event["event"] == "turn_completed" for event in result["events"])
    assert "offline-test-tavily-key" not in repr(result)
    assert "misplaced-test-tavily-key" not in repr(result)


@pytest.mark.parametrize(
    "delay_mode",
    [
        "baseline",
        "jitter",
        "append_ack_200",
        "append_ack_600",
        "append_ack_900",
        "append_ack_1200",
        "append_ack_1500",
    ],
)
async def test_labeled_nine_second_native_replay_with_delayed_mcp_appends(
    tmp_path, monkeypatch, delay_mode
):
    import wave
    import time
    import me3_voice.native_app as native
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.research import ResearchOptions
    from me3_voice.speech import SpeechClient

    source = Path("var/results/native-live-smoke/boiling-question-16k.wav")
    assert source.is_file()
    with wave.open(str(source), "rb") as wav:
        assert wav.getnframes() / wav.getframerate() == 9.0

    class JitterSpeechClient(SpeechClient):
        sequences = []

        async def call(self, kind, name, arguments):
            if kind == "asr" and name in {"asr_append", "asr_append_batch"}:
                sequence = arguments["sequence"]
                self.sequences.append(sequence)
                if delay_mode == "jitter":
                    delay = {7: 0.18, 31: 0.24, 68: 0.16}.get(
                        sequence, 0.012 + (sequence % 3) * 0.004
                    )
                    await asyncio.sleep(delay)
                elif delay_mode.startswith("append_ack_") and sequence == 0:
                    await asyncio.sleep(int(delay_mode.rsplit("_", 1)[1]) / 1000)
            return await super().call(kind, name, arguments)

    monkeypatch.setattr(native, "SpeechClient", JitterSpeechClient)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "offline-labeled-replay-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "requests.sqlite"))

    class Answer:
        async def respond(self, _history):
            yield "At sea level, water boils at 100 degrees Celsius."
            yield "That is 212 degrees Fahrenheit."

    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / f"{delay_mode}.json",
        trace=tmp_path / f"{delay_mode}.jsonl",
        backend_config=None,
        secrets_file=None,
        llm_secrets_file=None,
        wake_frame=0,
        transcript=True,
        mock_transcripts=["What is the boiling point of water at sea level?"],
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        pace=1,
        playback_wav=tmp_path / f"{delay_mode}.wav",
    )
    started = time.perf_counter()
    try:
        result = await native.replay_native_application(
            args,
            RuntimeOptions(engine="native", research=ResearchOptions(False, 0)),
            upstream=Answer(),
        )
    except RuntimeError:
        if not delay_mode.startswith(("append_ack_1200", "append_ack_1500")):
            raise
        trace_rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
        saturation = next(e for e in trace_rows if e["event"] == "asr_append_queue_saturated")
        assert saturation["queue_limit"] == 40
        assert saturation["depth"] == 40
        speech_start = next(e for e in trace_rows if e["event"] == "vad_speech_started")
        assert speech_start["onset_buffer_frames"] == 20
        assert sum(e["event"] == "vad_speech_started" for e in trace_rows) == 1
        speech_stop = next(e for e in trace_rows if e["event"] == "vad_speech_stopped")
        aborted = next(e for e in trace_rows if e["event"] == "asr_turn_aborted")
        failed = next(e for e in trace_rows if e["event"] == "replay_failed")
        assert speech_start["utterance_id"] == speech_stop["utterance_id"] == 1
        assert speech_stop["at"] > saturation["at"]
        assert aborted["error_code"] == failed["error_code"] == "queue_saturated"
        assert failed["error_stage"] == "asr_append"
        assert failed["failure_origin_event"] == "asr_turn_aborted"
        # Eight preserved onset frames leave about one second of post-VAD
        # headroom; a sustained 1.2s stall still fails explicitly and bounded.
        assert 0.95 <= saturation["at"] - speech_start["at"] <= 1.3
        return
    if delay_mode in {"append_ack_1200", "append_ack_1500"}:
        pytest.fail("append acknowledgement delay unexpectedly exceeded the bounded queue")
    elapsed = time.perf_counter() - started
    assert result["asr_transcription"] == "What is the boiling point of water at sea level?"
    assert "100 degrees Celsius" in result["llm_response"]
    assert "turn_completed" in result["completion_events"]
    assert result["tts_audio"]["wav_valid"] and result["tts_audio"]["pcm16_valid"]
    assert result["request_ledger"]["requests"] == []
    if delay_mode in {"jitter", "append_ack_200", "append_ack_600", "append_ack_900"}:
        assert JitterSpeechClient.sequences == list(range(len(JitterSpeechClient.sequences)))
        assert len(JitterSpeechClient.sequences) > 1
        assert not any(e.get("event") == "asr_append_queue_saturated" for e in result["events"])
        assert sum(e["event"] == "asr_opened" for e in result["events"]) == 1
        assert sum(e["event"] == "vad_speech_started" for e in result["events"]) == 1
        assert not any(
            e.get("idle_watchdog_triggered")
            for e in result["events"]
            if e["event"] == "vad_speech_stopped"
        )
        assert all(
            e.get("queue_depth", 0) <= e.get("queue_limit", 32)
            for e in result["events"]
            if e["event"] in {"asr_append_queued", "asr_append_queue_saturated"}
        )
        assert any(e["event"] == "asr_flush_completed" for e in result["events"])
        assert (
            max(
                e.get("queue_depth", 0)
                for e in result["events"]
                if e["event"] in {"asr_append_queued", "asr_append_started"}
            )
            <= 40
        )
    assert elapsed >= 9.0


async def test_failed_native_replay_persists_sanitized_partial_trace(tmp_path, monkeypatch):
    import wave
    import me3_voice.native_app as native
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.remote import ProviderError
    from me3_voice.research import ResearchOptions
    from me3_voice.speech import SpeechClient

    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        # Two quiet frames make the test exercise the same VAD-stop error
        # boundary used by a completed spoken turn.
        writer.writeframes(b"\1\0" * 2048 + b"\0\0" * 1024)

    class FailedAppend(SpeechClient):
        async def call(self, kind, name, arguments):
            if kind == "asr" and name in {"asr_append", "asr_append_batch"}:
                raise ProviderError(
                    "provider diagnostic contains private-value",
                    code="provider_timeout",
                    role="asr",
                    error_stage="asr_append",
                    error_boundary="deepgram_websocket",
                    error_detail="TimeoutError",
                )
            return await super().call(kind, name, arguments)

    monkeypatch.setattr(native, "SpeechClient", FailedAppend)
    monkeypatch.setattr(native, "native_vad", FixtureVAD)
    monkeypatch.setattr("me3_voice.readiness.build_wake", lambda _args: None)
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "partial-trace-test-01")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "ledger.sqlite"))
    trace = tmp_path / "failed.jsonl"
    args = Namespace(
        onit_python=None,
        direct_clock=False,
        turn_policy="silence",
        processed_wav=None,
        wav=source,
        output=tmp_path / "result.json",
        trace=trace,
        backend_config=None,
        secrets_file=None,
        llm_secrets_file=None,
        wake_frame=0,
        transcript=False,
        mock_transcripts=["offline question"],
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        pace=0,
        playback_wav=None,
    )
    with pytest.raises(RuntimeError):
        await native.replay_native_application(
            args,
            RuntimeOptions(engine="native", research=ResearchOptions(False, 0)),
        )
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(row["event"] == "asr_append_failed" for row in rows)
    failure = next(row for row in rows if row["event"] == "replay_failed")
    assert failure["error_code"] == "provider_timeout"
    assert failure["error_stage"] == "asr_append"
    assert failure["error_boundary"] == "deepgram_websocket"
    assert failure["provider_exception_type"] == "TimeoutError"
    assert failure["audio_frames_delivered"] >= 1
    assert "private-value" not in trace.read_text()


async def test_research_schemas_follow_live_permission_and_remaining_budget():
    from me3_voice.configuration import RuntimeOptions
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.research import ResearchClient, ResearchOptions

    research = ResearchClient(ResearchOptions(False, 0))
    state = NativeState(research=research)
    try:
        async with NativeApplication(
            state,
            MemorySink(),
            annotated=True,
            options=RuntimeOptions(engine="native"),
        ) as app:

            def names():
                return {schema.name for schema in state.context.tools.standard_tools}

            assert not names() & {"web_search", "retrieve_source", "verify_evidence"}

            research.options = ResearchOptions(True, 1)
            app.llm.refresh_tools(state.context)
            assert {"web_search", "retrieve_source"} <= names()
            assert "verify_evidence" not in names()  # No retained source yet.

            research.sources["S1"] = {"text": "fixture evidence"}
            research.requests = 1
            app.llm.refresh_tools(state.context)
            assert not names() & {"web_search", "retrieve_source"}
            assert "verify_evidence" in names()  # Local check remains usable.
    finally:
        await state.close()


async def test_native_provider_budget_survives_reconnect_without_sdk_retries(monkeypatch):
    import tempfile
    from pathlib import Path
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport, ProviderError
    from me3_voice.configuration import BackendSpec
    from pipecat.processors.aggregators.llm_context import LLMContext

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        model="openai/gpt-oss-120b",
        auth_env="ME3_TEST_ONLY",
        allow_live=True,
        request_limit=1,
        max_input_tokens=32000,
        max_input_tokens_total=64000,
    )
    credentials = {"ME3_TEST_ONLY": "offline-fixture-not-a-real-key"}
    ledger = Path(tempfile.mkdtemp()) / "provider.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-test-run-01")
    budget = LiveTransport(spec, credentials)
    calls = []

    from types import SimpleNamespace

    for index in (0, 1):
        llm = NativeLLM(spec, credentials=credentials, budget=budget)

        async def create(**_params):
            calls.append(1)
            from openai.types.chat import ChatCompletionChunk

            async def chunks():
                yield ChatCompletionChunk(
                    id="fixture",
                    object="chat.completion.chunk",
                    created=0,
                    model="fixture",
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": "fixture"},
                            "finish_reason": "stop",
                        }
                    ],
                )

            return chunks()

        async def close():
            return None

        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=close,
            max_retries=0,
        )
        try:
            assert llm._client.max_retries == 0 and not llm._retry_on_timeout
            assert llm._settings.max_completion_tokens == 2000
            if index:
                with pytest.raises(ProviderError):
                    await llm.get_chat_completions(LLMContext())
            else:
                stream = await llm.get_chat_completions(LLMContext())
                assert [c.choices[0].delta.content async for c in stream] == ["fixture"]
        finally:
            await llm._client.close()
    assert calls == [1] and budget.requests == 1
    import sqlite3

    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT status FROM provider_requests").fetchone() == ("success",)


async def test_native_stream_failure_and_cancellation_update_ledger_conservatively(
    tmp_path, monkeypatch
):
    import asyncio
    import sqlite3
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport, ProviderError
    from me3_voice.configuration import BackendSpec

    ledger = tmp_path / "provider.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "stream-test-run-01")
    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://llm.test/v1/chat/completions",
        model="test-model",
        auth_env="TEST_KEY",
        allow_live=True,
        request_limit=3,
        max_input_tokens=1000,
        max_input_tokens_total=3000,
    )
    budget = LiveTransport(spec, {"TEST_KEY": "offline-test-key"})

    class Tracker:
        safe_error = staticmethod(NativeLLM.safe_error)

    tracker = Tracker()
    tracker.budget = budget
    rows = []

    async def broken():
        yield "partial"
        raise ProviderError("offline failure", code="provider_timeout")

    _, request_id = budget.authorize(
        input_tokens=1, input_token_limit=1000, input_tokens_total_limit=3000
    )
    with pytest.raises(ProviderError):
        async for _ in NativeLLM._track_ledger_stream(tracker, broken(), request_id):
            pass
    with sqlite3.connect(ledger) as db:
        rows.append(
            db.execute(
                "SELECT status,error_code FROM provider_requests ORDER BY created"
            ).fetchall()
        )

    async def delayed():
        yield "partial"
        await asyncio.sleep(30)

    _, request_id = budget.authorize(
        input_tokens=1, input_token_limit=1000, input_tokens_total_limit=3000
    )

    async def consume():
        async for _ in NativeLLM._track_ledger_stream(tracker, delayed(), request_id):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sqlite3.connect(ledger) as db:
        rows.append(
            db.execute(
                "SELECT status,error_code FROM provider_requests ORDER BY created"
            ).fetchall()
        )
    assert rows[0] == [("failure", "provider_timeout")]
    assert rows[1] == [("failure", "provider_timeout"), ("dispatched_unknown", None)]


async def test_native_successful_stream_records_provider_usage_in_ledger(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport
    from me3_voice.configuration import BackendSpec

    ledger = tmp_path / "provider.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-actual-usage-01")
    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://llm.test/v1/chat/completions",
        model="test-model",
        auth_env="TEST_KEY",
        allow_live=True,
        request_limit=2,
        max_input_tokens=1000,
        max_input_tokens_total=2000,
    )
    budget = LiveTransport(spec, {"TEST_KEY": "offline-test-key"})
    _, request_id = budget.authorize(
        input_tokens=400, input_token_limit=1000, input_tokens_total_limit=2000
    )

    class Tracker:
        safe_error = staticmethod(NativeLLM.safe_error)

    tracker = Tracker()
    tracker.budget = budget
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 123, "completion_tokens": 17})

    async def completed():
        yield SimpleNamespace(usage=usage)

    rows = [
        chunk async for chunk in NativeLLM._track_ledger_stream(tracker, completed(), request_id)
    ]
    assert len(rows) == 1
    with sqlite3.connect(ledger) as db:
        assert db.execute(
            "SELECT status,input_tokens_reserved,input_tokens_actual,output_tokens_actual FROM provider_requests"
        ).fetchone() == ("success", 400, 123, 17)


async def test_native_llm_serialized_input_budget_rejects_before_reservation_or_sdk_dispatch(
    tmp_path, monkeypatch
):
    import sqlite3
    from types import SimpleNamespace
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.configuration import BackendSpec
    from me3_voice.native_llm import NativeLLM
    from me3_voice.remote import LiveTransport

    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-budget-test-01")
    ledger = tmp_path / "provider-requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(ledger))
    base = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://llm.example.test/v1/chat/completions",
        model="test-model",
        auth_env="ME3_TEST_BUDGET_KEY",
        allow_live=True,
        request_limit=5,
        max_input_tokens=129,
        max_input_tokens_total=512,
    )
    credentials = {"ME3_TEST_BUDGET_KEY": "offline-only-test-key"}
    budget = LiveTransport(base, credentials)
    llm = NativeLLM(base, credentials=credentials, budget=budget)
    calls = []

    async def forbidden_dispatch(**_params):
        calls.append("dispatched")
        raise AssertionError("a request over the input budget must not reach the SDK")

    async def close():
        return None

    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden_dispatch)),
        close=close,
        max_retries=0,
    )
    events = []
    llm.on_event = lambda name, **fields: events.append((name, fields))
    try:
        stream = await llm.get_chat_completions(LLMContext())
        # Consume the framework's explicit safe failure, not a provider stream.
        _ = [chunk async for chunk in stream]
    finally:
        await llm._client.close()
    assert calls == [] and budget.requests == 0
    assert any(name == "llm_input_budget_rejected" for name, _ in events)
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT COUNT(*) FROM provider_requests").fetchone()[0] == 0
        assert db.execute(
            "SELECT input_tokens_per_request_limit,input_tokens_limit FROM provider_campaign_caps"
        ).fetchone() == (129, 512)


async def test_native_llm_adaptively_compacts_context_before_enforcing_input_cap():
    import json
    from me3_voice.configuration import BackendSpec
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext
    from me3_voice.request_budget import conservative_input_token_estimate

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://llm.example.test/v1/chat/completions",
        model="offline-fixture",
        auth_env="ME3_TEST_CONTEXT_KEY",
        allow_live=False,
        max_input_tokens=12000,
    )
    llm = NativeLLM(spec, credentials={"ME3_TEST_CONTEXT_KEY": "offline-fixture"})
    context = LLMContext()
    leads = {
        "kind": "untrusted_search_leads",
        "query": "who won the Texas Senate election",
        "results": [
            {
                "url": f"https://elections.example.test/result-{index}",
                "title": f"Official Texas Senate election result {index}",
                "snippet": ("The reported election result and date are in this lead. " * 7)[:320],
                "evidence_status": "discovery_only",
                "published_date_reported": "2026-11-04",
                "ranked_for_query": True,
            }
            for index in range(5)
        ],
    }
    context.set_messages(
        [
            {"role": "user", "content": "Earlier unrelated question " * 45},
            {"role": "assistant", "content": "Earlier unrelated answer " * 45},
            {"role": "user", "content": "Another unrelated question " * 45},
            {"role": "assistant", "content": "Another unrelated answer " * 45},
            {"role": "user", "content": "Who won the Texas Senate election?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "search-call",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "search-call", "content": json.dumps(leads)},
        ]
    )

    async def serialized_params(current):
        return {
            "model": "offline-fixture",
            "messages": current.get_messages(),
            "tools": [{"description": "x" * 7000}],
        }

    llm._prepare_provider_params = serialized_params
    events = []
    llm.on_event = lambda name, **fields: events.append((name, fields))
    canonical_before = json.loads(json.dumps(context.get_messages()))
    try:
        params, estimated, _ = await llm._prepare_provider_params_within_budget(context)
        messages = params["messages"]
        result = json.loads(messages[-1]["content"])
        assert estimated <= spec.max_input_tokens
        assert any(
            row.get("role") == "user" and row.get("content") == "Who won the Texas Senate election?"
            for row in messages
        )
        assert not any(
            row.get("role") == "user" and "Earlier unrelated" in row.get("content", "")
            for row in messages
        )
        assert context.get_messages() == canonical_before
        assert 1 <= len(result["results"]) <= 3
        assert all(
            row["url"].startswith("https://elections.example.test/") for row in result["results"]
        )
        assert all(row["evidence_status"] == "discovery_only" for row in result["results"])
        compacted = next(fields for name, fields in events if name == "llm_context_compacted")
        assert compacted["estimated_tokens_before"] > spec.max_input_tokens
        assert compacted["estimated_tokens_after"] <= spec.max_input_tokens
        assert conservative_input_token_estimate(params)[0] == estimated
    finally:
        await llm._client.close()


def test_retrieved_evidence_compaction_preserves_citation_provenance():
    import json
    from me3_voice.native_llm import NativeLLM
    from pipecat.processors.aggregators.llm_context import LLMContext

    evidence = {
        "source_id": "S1",
        "url": "https://official.example.test/results",
        "title": "Official election results",
        "evidence_status": "retrieved_supporting_evidence",
        "text_sha256": "a" * 64,
        "text": "First supporting sentence. Second supporting sentence. " * 80,
        "evidence_chunks": 2,
    }
    context = LLMContext()
    context.set_messages(
        [{"role": "tool", "tool_call_id": "retrieve-call", "content": json.dumps(evidence)}]
    )
    assert NativeLLM._compact_retrieved_evidence(context, 600)
    compacted = json.loads(context.get_messages()[0]["content"])
    assert compacted["source_id"] == "S1"
    assert compacted["url"] == evidence["url"]
    assert compacted["evidence_status"] == evidence["evidence_status"]
    assert compacted["text_sha256"] == evidence["text_sha256"]
    assert 0 < len(compacted["text"]) <= 600
    assert compacted["text"].endswith(".")


async def test_native_wake_standby_does_not_submit_background_audio_to_asr():
    from me3_voice.native_app import NativeApplication, NativeState
    from me3_voice.playback import MemorySink
    from me3_voice.configuration import RuntimeOptions

    class NeverWake:
        frame_samples = 1280

        def feed(self, _pcm):
            return {"triggered": False}

        def end_session(self):
            pass

    state = NativeState()
    try:
        async with NativeApplication(
            state, MemorySink(), options=RuntimeOptions(engine="native"), wake=NeverWake()
        ) as app:
            # A bounded interval of audible but unrecognized background audio.
            for sequence in range(40):
                await app.feed(b"\x00\x20" * 512, sequence)
                await app.drain()
            assert not app.gate.active
            assert app.parts.stt.operation is None
            assert not app.parts.stt.onset_buffer
            assert not any(event["event"] == "asr_open_started" for event in app.events)
    finally:
        await state.close()


async def test_explicit_stop_ends_context_while_disconnect_preserves_it(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            first = (await next_json(ws, "ready"))["session_id"]
            await send_turn(ws)
            await wait_answer(ws)
            await ws.send_json({"type": "stop"})
            while (await ws.receive()).type not in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                pass
        await asyncio.sleep(0.05)
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            assert (await next_json(ws, "ready"))["session_id"] != first
            view = (await next_json(ws, "transcript"))["data"]
            assert not view["messages"]


async def test_dead_speech_owner_fails_closed_without_budget_reset():
    from me3_voice.native_app import NativeState

    state = NativeState()
    try:
        await state.start()
        state.owner.cancel()
        await asyncio.gather(state.owner, return_exceptions=True)
        with pytest.raises(RuntimeError, match="restart bridge"):
            await state.start()
    finally:
        await state.close()


async def test_native_dashboard_protocol_and_optional_telemetry(tmp_path, monkeypatch, native_args):
    import me3_voice.native_app as module
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    monkeypatch.setattr(module, "native_vad", FixtureVAD)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        response = await client.get(server.make_url("/api/voice/capabilities"))
        capabilities = await response.json()
        assert capabilities["available"] and capabilities["engine"] == "native"
        assert capabilities["devices"]["target"] == "mock"
        assert capabilities["devices"]["status"] == "simulated"
        assert not capabilities["music"]["configured"]
        assert (await client.get(server.make_url("/api/snapshot"))).status == 200
        async with client.ws_connect(server.make_url("/api/voice/ws")) as ws:
            await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})
            ready = await next_json(ws, "ready")
            status = await next_json(ws, "status")
            assert status["run_id"] == ready["run_id"]
            assert isinstance(status["turn_id"], int) and status["status_sequence"] > 0
            assert status["event_sequence"] >= 0
            await ws.send_json({"type": "music_ui_control", "action": "pause", "outcome": "failed"})
            await ws.send_json(
                {
                    "type": "music_playback_observed",
                    "status": "not_playing",
                    "source": "spotify_api",
                    "track_changed": False,
                }
            )
            event = await next_json(ws, "runtime_event")
            assert event["sequence"] == event["data"]["sequence"]
            assert event["data"]["session_id"] == ready["session_id"]
            action = {
                "type": "device_ui_action",
                "command_id": "ui-0123456789abcdef0123456789abcdef",
                "action": "light_on",
            }
            await ws.send_json(action)
            first = await next_json(ws, "device_action_ack")
            assert first["status"] == "accepted"
            await ws.send_json(action)
            assert (await next_json(ws, "device_action_ack"))["status"] == "duplicate"
            await send_turn(ws)
            assert (await wait_answer(ws))["messages"]


async def test_native_capability_reports_engine_and_shared_contract_fields(
    tmp_path, monkeypatch, native_args
):
    import me3_voice.native_bridge as bridge

    monkeypatch.setattr(bridge, "build_wake", lambda _: Wake())
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        response = await client.get(server.make_url("/api/voice/capabilities"))
        assert response.status == 200
        capability = await response.json()
        assert capability["engine"] == "native"
        assert capability["available"] is True
        assert capability["wake"] == "configured model"
        assert capability["capture"] == "browser-pcm16"
        assert capability["playback"] == "browser-acknowledged-pcm16"
        assert capability["aec"] == "browser-requested-unverified"
        assert "asr_partial" in capability
        assert capability["task_artifact_access"] is False
        assert capability["devices"]["target"] == "mock"
        assert capability["devices"]["status"] == "simulated"
        assert capability["music"]["configured"] is False
        assert capability["music"]["output_backend"] == "unavailable"
        report = capability["capability_report"]
        assert report["kind"] == "swift-me3-capability-report"
        assert report["profile"]["credential_values_included"] is False
        assert any(
            row["name"] == "fast_command_router" and row["registered"]
            for row in report["capabilities"]
        )
        assert any(
            row["name"] == "write_artifact" and row["surface"] == "langgraph_task"
            for row in report["capabilities"]
        )
        assert all(
            "secret_value" not in row and "credential_value" not in row
            for row in report["capabilities"]
        )


async def test_native_bridge_downloads_only_registered_task_artifacts(
    tmp_path, monkeypatch, native_args
):
    from types import SimpleNamespace

    import me3_voice.native_bridge as bridge
    from me3_voice.configuration import RuntimeOptions

    artifact_id = "a" * 24
    task_id = "task_" + artifact_id
    content = "<!doctype html><title>Game</title>"

    class ArtifactStore:
        async def artifact_metadata(self, requested_task, name):
            if requested_task == task_id and name in {"index.html", "todo/index.html"}:
                content_for_name = content if name == "index.html" else "<h1>Nested artifact</h1>"
                return {
                    "path": name,
                    "sha256": hashlib.sha256(content_for_name.encode()).hexdigest(),
                    "size": len(content_for_name),
                }
            return None

        async def read_artifact(self, requested_task, name):
            if requested_task != task_id or name not in {"index.html", "todo/index.html"}:
                raise FileNotFoundError
            content_for_name = content if name == "index.html" else "<h1>Nested artifact</h1>"
            return {
                "path": name,
                "content": content_for_name,
                "sha256": hashlib.sha256(content_for_name.encode()).hexdigest(),
            }

    class FakeState:
        def __init__(self):
            self.task_runtime = SimpleNamespace(store=ArtifactStore())
            self.tasks = SimpleNamespace(
                actions=SimpleNamespace(capabilities=lambda: {}),
                music=SimpleNamespace(capabilities=lambda: {}),
            )
            self.credentials = {}
            self.speech = None
            self.owner = None

        async def close(self):
            pass

    state = FakeState()
    monkeypatch.setattr(bridge, "NativeState", lambda *args, **kwargs: state)
    app = bridge.build_native_app(
        tmp_path,
        load_backends(),
        {},
        native_args,
        RuntimeOptions(engine="native"),
    )
    async with TestServer(app) as server, ClientSession() as client:
        url = server.make_url(f"/api/voice/tasks/{artifact_id}/artifacts/index.html")
        response = await client.get(url)
        assert response.status == 200
        assert await response.text() == content
        assert response.headers["Content-Disposition"] == 'attachment; filename="index.html"'
        assert response.headers["Content-Security-Policy"] == "default-src 'none'"
        nested = await client.get(
            server.make_url(f"/api/voice/tasks/{artifact_id}/artifacts/todo/index.html")
        )
        assert nested.status == 200
        assert await nested.text() == "<h1>Nested artifact</h1>"
        assert nested.headers["Content-Disposition"] == 'attachment; filename="index.html"'
        missing = await client.get(
            server.make_url(f"/api/voice/tasks/{artifact_id}/artifacts/missing.html")
        )
        assert missing.status == 404


def test_asr_browser_failure_trace_keeps_role_state_and_connection_stages():
    from me3_voice.native_bridge import _safe_event

    failure = _safe_event(
        {
            "event": "turn_error",
            "error_role": "asr",
            "error_code": "provider_timeout",
            "error_stage": "asr_open",
        }
    )
    assert failure["error_role"] == "asr"
    assert failure["error_stage"] == "asr_open"
    row = _safe_event(
        {
            "event": "asr_provider_transport",
            "connection_state": "closed",
            "failure_code": "provider_timeout",
            "sent_audio_bytes": 0,
            "connection_stages": {
                "dns_started": 1.2,
                "connection_cancelled_monotonic": 2.3,
                "connection_terminal_exception": "CancelledError",
                "Authorization": "private",
            },
        }
    )
    assert row["connection_state"] == "closed"
    assert row["dns_started"] == 1.2
    assert row["connection_cancelled_monotonic"] == 2.3
    assert row["connection_terminal_exception"] == "CancelledError"
    assert "private" not in str(row)


def test_interrupted_partial_generation_normalizes_for_archive():
    from me3_voice.transcript import normalize_transcript

    payload = {
        "schema_version": 1,
        "kind": "conversation-transcript",
        "display_authorized": True,
        "session_id": "partial-session",
        "revision": 1,
        "mode": "HPC-live",
        "state": "listening",
        "backend_modes": {},
        "updated_utc": "2026-11-14T00:00:00+00:00",
        "tasks": [],
        "messages": [
            {
                "id": "1:assistant",
                "turn_id": 1,
                "role": "assistant",
                "text": "Rain forms.",
                "confirmed_text": "",
                "status": "interrupted",
                "delivery": "generated",
                "generation_complete": False,
                "response": {
                    "response_id": "partial-response-1",
                    "state": "interrupted",
                    "generation": "interrupted",
                    "tts": "interrupted",
                    "playback": "partially_acknowledged",
                    "archive": "included",
                    "segments": [],
                    "packets": 8,
                    "acks": 3,
                    "superseded_by_turn_id": None,
                    "interruption_reason": "user",
                },
            }
        ],
    }
    assert normalize_transcript(payload)["messages"][0]["status"] == "interrupted"


def test_spoken_text_removes_fragmented_html_tags_and_entities():
    from me3_voice.native_speech import spoken_text

    assert (
        spoken_text("<section><h1>About</h1>&nbsp;I build sites.</section>")
        == "About I build sites."
    )


async def test_native_report_scan_keeps_event_loop_responsive(tmp_path, monkeypatch, native_args):
    import threading
    import me3_voice.native_bridge as bridge

    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def slow_snapshot(_root):
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(3), "report scan was not released"
        return {"schema_version": 1, "generated_utc": "", "reports": [], "warnings": []}

    monkeypatch.setattr(bridge, "snapshot", slow_snapshot)
    async with (
        TestServer(build_app(tmp_path, load_backends(), {}, native_args)) as server,
        ClientSession() as client,
    ):
        request = asyncio.create_task(client.get(server.make_url("/api/snapshot")))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            health = await asyncio.wait_for(client.get(server.make_url("/api/health")), 1)
            assert health.status == 200
        finally:
            release.set()
        response = await request
        assert response.status == 200
