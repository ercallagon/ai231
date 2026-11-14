"""Per-service MCP startup diagnostics; upstream inference is always mocked."""

import sys

import pytest

from me3_voice.configuration import BackendSpec, load_backends
import me3_voice.speech as speech
from me3_voice.speech import SpeechClient
from pathlib import Path


async def test_mock_speech_preflight_reports_each_protocol_and_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "scratch").mkdir(parents=True)
    backends = load_backends()
    result = await speech.preflight_mock_speech(backends)
    assert [item["service"] for item in result] == ["asr", "tts"]
    for service in result:
        assert service["ok"] is True
        assert service["exit_code"] == 0
        assert service["protocol_version"]
        assert service["startup_seconds"] > 0
        assert service["stderr"] == ""
    assert {"asr_open", "asr_finish"} <= set(result[0]["tools"])
    assert {"tts_open", "tts_next"} <= set(result[1]["tools"])


async def test_preflight_keeps_service_failures_separate_and_captures_stderr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "var" / "scratch").mkdir(parents=True)
    original = speech._server_params

    def fail_asr(kind, spec, transcript, delay=0, tts_delay=0, credentials=None):
        if kind == "asr":
            return speech.StdioServerParameters(
                command=sys.executable,
                args=[
                    "-c",
                    "import sys; print('forced ASR startup fault', file=sys.stderr); sys.exit(17)",
                ],
                env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
        return original(kind, spec, transcript, delay, tts_delay, credentials)

    monkeypatch.setattr(speech, "_server_params", fail_asr)
    result = await speech.preflight_mock_speech(load_backends())
    assert result[0]["ok"] is False
    assert result[0]["exit_code"] == 17
    assert "forced ASR startup fault" in result[0]["stderr"]
    assert result[1]["ok"] is True
    assert result[1]["exit_code"] == 0


async def test_preflight_rejects_non_mock_speech_before_spawning():
    backends = load_backends()
    backends["tts"] = BackendSpec(backend="external")
    with pytest.raises(ValueError, match="mocked inference"):
        await speech.preflight_mock_speech(backends)


async def test_native_provider_mcp_descriptors_handshake_without_inference():
    """Launch both owned MCP children with primary provider descriptors, no API calls."""
    backends = load_backends(Path("configs/native-speech.provisional.json"))
    assert backends["asr"].allow_live is False
    assert backends["tts"].allow_live is False
    async with SpeechClient(timeout=15, backends=backends, credentials={}) as client:
        assert client.protocols["asr"]
        assert client.protocols["tts"]
        assert client.capabilities["asr"]["contract_version"] >= 1
        assert client.capabilities["tts"]["contract_version"] >= 1
        assert {"asr_open", "asr_append", "asr_finish"} <= client.tools["asr"]
        assert {"tts_open", "tts_next"} <= client.tools["tts"]


def test_speech_mcp_subprocess_inherits_approved_ledger_identity(monkeypatch):
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "native-smoke-0001")
    monkeypatch.setenv("ME3_REQUEST_LEDGER", "/tmp/native-smoke.sqlite")
    spec = BackendSpec(
        backend="external",
        protocol="deepgram-listen-v1",
        endpoint="wss://api.deepgram.com/v1/listen",
        auth_env="ME3_DEEPGRAM_API_KEY",
        model="nova-3",
        allow_live=True,
        request_limit=1,
    )
    params = speech._server_params(
        "asr", spec, "unused", credentials={"ME3_DEEPGRAM_API_KEY": "offline-test-secret"}
    )
    assert params.env["ME3_REQUEST_RUN_ID"] == "native-smoke-0001"
    assert params.env["ME3_REQUEST_LEDGER"] == "/tmp/native-smoke.sqlite"
    assert params.env["ME3_DEEPGRAM_API_KEY"] == "offline-test-secret"
    assert "offline-test-secret" not in " ".join(params.args)
