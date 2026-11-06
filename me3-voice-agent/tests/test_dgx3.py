"""DGX3 wire/credential checks; local peers only, no class inference submitted."""

import asyncio
import base64
import io
import json
import os
import wave
from dataclasses import replace
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from me3_voice import credentials, dgx3
from me3_voice.configuration import ConfigurationError
from me3_voice.english import EnglishTextRejected, QWEN_TTS_STYLE_INSTRUCTIONS
from me3_voice.remote import (
    LiveTransport,
    ProviderError,
    QwenAudioASR,
    QwenTTS,
    VLLMQwen3ASR,
    make_provider,
    qwen_tts_instructions,
)


def test_dgx3_profile_is_explicit_final_asr_streaming_tts_and_narrow_http_exception():
    values = dgx3.profile()
    for role, spec in values.items():
        spec.validate(role, runnable=True)
    assert not values["asr"].streaming and values["tts"].streaming
    assert values["llm"].backend == "mock"
    assert (
        replace(values["tts"], instructions="Use a warm, natural delivery.")
        .validate("tts", runnable=True)
        .instructions
        == "Use a warm, natural delivery."
    )
    assert replace(values["tts"], seed=42).validate("tts", runnable=True).seed == 42
    with pytest.raises(ConfigurationError, match="seed is only supported"):
        replace(values["asr"], seed=42).validate("asr", runnable=True)
    with pytest.raises(ConfigurationError, match="only supported by the Qwen TTS adapter"):
        replace(values["asr"], instructions="Use a warm, natural delivery.").validate(
            "asr", runnable=True
        )
    assert all(not spec.allow_live for spec in values.values())
    assert isinstance(make_provider("asr", values["asr"]), QwenAudioASR)
    assert isinstance(make_provider("tts", values["tts"]), QwenTTS)
    for endpoint in [
        "http://10.0.0.1:8000/v1/chat/completions",
        "http://10.153.60.233:9000/v1/chat/completions",
        "http://10.153.60.233:8000/v1/completions",
    ]:
        with pytest.raises(ConfigurationError, match="restricted"):
            replace(values["asr"], endpoint=endpoint).validate("asr", runnable=True)
    with pytest.raises(ConfigurationError, match="TLS"):
        replace(values["asr"], allow_private_http=False).validate("asr", runnable=True)


def test_user_provided_qwen_reasoning_ports_are_allowed_but_other_routes_are_not():
    from me3_voice.configuration import BackendSpec

    for port in (8000, 8001):
        spec = BackendSpec(
            backend="external",
            protocol="chat-completions-v1",
            endpoint=f"http://202.92.159.240:{port}/v1/chat/completions",
            auth_env="ME3_LLM_API_KEY",
            model="Qwen/Qwen3.8-27B",
            allow_live=True,
            request_limit=1,
            allow_private_http=True,
        )
        assert spec.validate("llm", runnable=True).endpoint.endswith("/v1/chat/completions")

    configured = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="http://202.92.159.240:8000/v1/chat/completions",
        auth_env="ME3_LLM_API_KEY",
        model="Qwen/Qwen3.8-27B",
        allow_live=True,
        request_limit=1,
        allow_private_http=True,
    )
    for endpoint in (
        "http://202.92.159.240:9000/v1/chat/completions",
        "http://10.0.0.1:8000/v1/chat/completions",
        "http://202.92.159.240:8000/v1/completions",
    ):
        with pytest.raises(ConfigurationError, match="restricted"):
            replace(configured, endpoint=endpoint).validate("llm", runnable=True)


async def test_dedicated_qwen3_asr_contract_is_prepared_without_changing_live_profile():
    current = dgx3.profile()["asr"]
    dedicated = replace(
        current,
        protocol="vllm-qwen3-asr-v1",
        endpoint="http://10.153.60.233:8000/v1/audio/transcriptions",
        model="Qwen/Qwen3-ASR-1.7B",
        streaming=False,
    )
    dedicated.validate("asr", runnable=True)

    class OfflineTranscriptionTransport:
        def __init__(self):
            self.pcm = None

        async def transcribe(self, pcm):
            self.pcm = pcm
            return "What is the boiling point of water at sea level?"

        async def aclose(self):
            pass

    wire = OfflineTranscriptionTransport()
    provider = make_provider("asr", dedicated, transport=wire)
    assert isinstance(provider, VLLMQwen3ASR)
    result = await provider.transcribe(b"\1\0" * 32)
    assert result == "What is the boiling point of water at sea level?"
    assert wire.pcm == b"\1\0" * 32
    await provider.aclose()

    with pytest.raises(ConfigurationError, match="private HTTP exception"):
        replace(dedicated, endpoint="http://10.153.60.233:8000/v1/chat/completions").validate(
            "asr", runnable=True
        )
    with pytest.raises(ConfigurationError, match="file/Qwen-audio ASR is final-only"):
        replace(dedicated, streaming=True).validate("asr", runnable=True)


def test_qwen_tts_policy_follows_custom_instructions_as_the_final_guardrail():
    configured = "Speak warmly; a laugh may be welcome."
    instructions = qwen_tts_instructions(replace(dgx3.profile()["tts"], instructions=configured))
    assert instructions.startswith(configured)
    assert instructions == configured + "\n\nRequired Swift delivery policy: " + (
        QWEN_TTS_STYLE_INSTRUCTIONS
    )
    assert "Add no laughter, giggles, chuckles" in instructions


def test_legacy_default_tts_instruction_is_replaced_with_tuned_managed_policy():
    configured = "Speak in calm, matter-of-fact English. Do not laugh or add words."
    instructions = qwen_tts_instructions(replace(dgx3.profile()["tts"], instructions=configured))
    assert instructions == QWEN_TTS_STYLE_INSTRUCTIONS


def test_masked_setup_private_storage_non_tty_rejection_and_no_overwrite(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(credentials, "CACHE_ROOT", tmp_path)
    args = SimpleNamespace(
        secrets_file=tmp_path / "dgx3.env",
        profile=tmp_path / "dgx3.json",
        enable_inference=True,
        request_limit=2,
        voice="ryan",
        language="English",
    )
    monkeypatch.setattr(dgx3.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(ConfigurationError, match="interactive terminal"):
        dgx3.configure(args)
    assert not args.secrets_file.exists()
    monkeypatch.setattr(dgx3.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    entered = iter(["offline-asr-key", "offline-tts-key"])
    monkeypatch.setattr(dgx3.getpass, "getpass", lambda prompt: next(entered))
    dgx3.configure(args)
    assert os.stat(args.secrets_file).st_mode & 0o777 == 0o600
    assert credentials.load_secrets(args.secrets_file, {"ME3_ASR_API_KEY"}) == {
        "ME3_ASR_API_KEY": "offline-asr-key"
    }
    output = capsys.readouterr().out
    assert "offline-asr-key" not in output and "offline-tts-key" not in output
    saved = args.secrets_file.read_bytes()
    with pytest.raises(ConfigurationError, match="overwrite"):
        dgx3.configure(args)
    assert args.secrets_file.read_bytes() == saved
    os.chmod(args.secrets_file, 0o644)
    with pytest.raises(ConfigurationError, match="0600"):
        credentials.load_secrets(args.secrets_file)
    link = tmp_path / "linked.env"
    link.symlink_to(args.secrets_file)
    with pytest.raises(ConfigurationError, match="symlink"):
        credentials.load_secrets(link)
    fifo = tmp_path / "not-a-file.env"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ConfigurationError):
        credentials.load_secrets(fifo)
    args.secrets_file, args.profile = tmp_path / "fresh.env", tmp_path / "fresh.json"

    def unsupported_terminal(prompt):
        raise dgx3.getpass.GetPassWarning("no secure terminal")

    monkeypatch.setattr(dgx3.getpass, "getpass", unsupported_terminal)
    with pytest.raises(ConfigurationError, match="cannot mask"):
        dgx3.configure(args)
    assert not args.secrets_file.exists()


@pytest.mark.parametrize("format_metadata", [True, False])
async def test_qwen_audio_input_and_tts_sse_native_rate_conversion_on_local_peer(format_metadata):
    selected = dgx3.profile(enable=True, request_limit=2)
    requests = []
    native = b"\1\0" * 2400  # 100 ms at native 24 kHz.

    async def asr(request):
        assert request.headers["Authorization"] == "Bearer offline-asr-key"
        body = await request.json()
        requests.append(body)
        parts = body["messages"][0]["content"]
        assert [part["type"] for part in parts] == ["input_audio", "text"]
        assert parts[1]["text"] == "What does the person say?"
        audio = parts[0]["input_audio"]
        with wave.open(io.BytesIO(base64.b64decode(audio["data"])), "rb") as wav:
            assert wav.getparams()[:3] == (1, 2, 16000)
            assert wav.readframes(512) == b"\2\0" * 512
        return web.json_response(
            {"choices": [{"message": {"content": " recognized words "}, "finish_reason": "stop"}]}
        )

    async def tts(request):
        assert request.headers["Authorization"] == "Bearer offline-tts-key"
        body = await request.json()
        requests.append(body)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for pcm in [native[:1], native[1:]]:
            event = {
                "type": "speech.audio.delta",
                "audio": base64.b64encode(pcm).decode(),
                "response_format": "pcm",
                "sample_rate": 24000,
                "channels": 1,
            }
            if not format_metadata:
                del event["sample_rate"], event["channels"]
            await response.write(
                ("event: speech.audio.delta\ndata: " + json.dumps(event) + "\n\n").encode()
            )
        await response.write(b'event: speech.audio.done\ndata: {"type":"speech.audio.done"}\n\n')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/asr", asr)
    app.router.add_post("/tts", tts)
    async with TestServer(app, host="127.0.0.1") as server:
        # Lower transport boundary uses local HTTP; production configuration
        # limits the non-TLS exception to explicitly supplied campus endpoints.
        recognition = QwenAudioASR(
            replace(selected["asr"], endpoint=str(server.make_url("/asr"))),
            LiveTransport(
                replace(selected["asr"], endpoint=str(server.make_url("/asr"))),
                {"ME3_ASR_API_KEY": "offline-asr-key"},
            ),
        )
        tts_transport = LiveTransport(
            replace(selected["tts"], endpoint=str(server.make_url("/tts"))),
            {"ME3_TTS_API_KEY": "offline-tts-key"},
        )
        tts_events = []
        synthesis = QwenTTS(
            replace(selected["tts"], seed=42),
            tts_transport,
        )
        synthesis.on_event = tts_events.append
        try:
            stream = recognition.open_stream()
            assert await stream.append(b"\2\0" * 512) is None
            result = await stream.finish()
            assert result.final and result.text == "recognized words"
            await stream.aclose()
            chunks = [chunk async for chunk in synthesis.synthesize("Hello Swift.")]
            assert sum(map(len, chunks)) == 3200  # 100 ms at internal 16 kHz.
            assert all(len(chunk) % 2 == 0 and len(chunk) <= 1024 for chunk in chunks)
        finally:
            await recognition.aclose()
            await synthesis.aclose()
    assert requests[0]["stream"] is False
    assert requests[1] == {
        "model": selected["tts"].model,
        "input": "Hello Swift.",
        "voice": "ryan",
        "instructions": QWEN_TTS_STYLE_INSTRUCTIONS,
        "language": "English",
        "response_format": "pcm",
        "stream": True,
        "seed": 42,
    }
    assert tts_events == [
        "tts_request_submission_started",
        "tts_response_headers_received",
        "tts_first_sse_event",
    ]


async def test_local_qwen_asr_reasoning_tts_route_preserves_user_assistant_roles():
    from me3_voice.configuration import BackendSpec
    from me3_voice.remote import ChatReasoning
    from me3_voice.session import Session
    from me3_voice.transcript import Transcript

    expected_user = "What is the capital of the Philippines?"
    expected_reply = "The capital of the Philippines is Manila."
    native_pcm = bytes(24000)  # 500 ms mono PCM16 at Qwen TTS's native 24 kHz.
    observed_bodies = {}

    async def asr(request):
        body = await request.json()
        observed_bodies["asr"] = body
        return web.json_response(
            {"choices": [{"message": {"content": expected_user}, "finish_reason": "stop"}]}
        )

    async def llm(request):
        body = await request.json()
        observed_bodies["llm"] = body
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        payload = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": expected_reply},
                    "finish_reason": "stop",
                }
            ]
        }
        await response.write(("data: " + json.dumps(payload) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def tts(request):
        body = await request.json()
        observed_bodies["tts"] = body
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        delta = {
            "type": "speech.audio.delta",
            "audio": base64.b64encode(native_pcm).decode("ascii"),
            "response_format": "pcm",
            "sample_rate": 24000,
            "channels": 1,
        }
        await response.write(("data: " + json.dumps(delta) + "\n\n").encode())
        await response.write(b'data: {"type":"speech.audio.done"}\n\n')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/asr", asr)
    app.router.add_post("/llm", llm)
    app.router.add_post("/tts", tts)
    async with TestServer(app, host="127.0.0.1") as server:
        selected = dgx3.profile(enable=True, request_limit=1)
        asr_spec = replace(selected["asr"], endpoint=str(server.make_url("/asr")))
        tts_spec = replace(selected["tts"], endpoint=str(server.make_url("/tts")))
        llm_spec = BackendSpec(
            backend="external",
            protocol="chat-completions-sse-v1",
            endpoint=str(server.make_url("/llm")),
            auth_env="ME3_LLM_API_KEY",
            model="Qwen/Qwen3.8-27B",
            streaming=True,
            timeout_seconds=1,
            allow_live=True,
            request_limit=1,
        )
        asr = QwenAudioASR(asr_spec, LiveTransport(asr_spec, {"ME3_ASR_API_KEY": "offline-asr"}))
        llm = ChatReasoning(
            llm_spec,
            LiveTransport(llm_spec, {"ME3_LLM_API_KEY": "offline-llm"}),
        )
        tts_provider = QwenTTS(
            tts_spec,
            LiveTransport(tts_spec, {"ME3_TTS_API_KEY": "offline-tts"}),
        )
        sink_chunks = []

        class SpeechRoute:
            async def transcribe(self, _key, pcm):
                return await asr.transcribe(pcm)

            async def synthesize(self, _key, text):
                async for chunk in tts_provider.synthesize(text):
                    yield chunk

        transcript = Transcript()

        async def sink(pcm):
            sink_chunks.append(pcm)

        try:
            async with Session(
                SpeechRoute(), llm, sink, transcript=transcript, execution_mode="HPC-live"
            ) as session:
                llm.on_event = session.event
                tts_provider.on_event = session.event
                session.activate()
                await (await session.submit(bytes(16000)))
                snapshot = transcript.snapshot()
                assert [(row["role"], row["text"]) for row in snapshot["messages"]] == [
                    ("user", expected_user),
                    ("assistant", expected_reply),
                ]
                assert snapshot["messages"][1]["delivery"] == "mock_sink_confirmed"
                assert snapshot["messages"][1]["confirmed_text"] == expected_reply
                assert sink_chunks and sum(map(len, sink_chunks)) > 0
                assert [
                    event["event"]
                    for event in session.events
                    if event["event"] in {"asr_final", "llm_first_delta", "tts_first_chunk"}
                ] == ["asr_final", "llm_first_delta", "tts_first_chunk"]
                assert (
                    asr.transport.requests
                    == llm.transport.requests
                    == tts_provider.transport.requests
                    == 1
                )
        finally:
            await asr.aclose()
            await llm.aclose()
            await tts_provider.aclose()

    assert observed_bodies["asr"]["stream"] is False
    assert observed_bodies["llm"]["model"] == "Qwen/Qwen3.8-27B"
    assert observed_bodies["llm"]["stream"] is True
    assert observed_bodies["llm"]["messages"][-1] == {"role": "user", "content": expected_user}
    assert observed_bodies["tts"]["input"] == expected_reply
    assert observed_bodies["tts"]["stream"] is True


async def test_qwen_tts_http_rejection_is_reported_as_tts_without_provider_body():
    selected = dgx3.profile(enable=True, request_limit=1)["tts"]

    async def reject(request):
        assert request.headers["Authorization"] == "Bearer offline-tts-key"
        return web.json_response(
            {"error": {"message": "private provider payload must not leak"}}, status=400
        )

    app = web.Application()
    app.router.add_post("/tts", reject)
    async with TestServer(app, host="127.0.0.1") as server:
        spec = replace(selected, endpoint=str(server.make_url("/tts")))
        provider = QwenTTS(
            spec,
            LiveTransport(spec, {"ME3_TTS_API_KEY": "offline-tts-key"}),
        )
        try:
            with pytest.raises(ProviderError) as caught:
                _ = [chunk async for chunk in provider.synthesize("Hello Swift.")]
        finally:
            await provider.aclose()

    assert caught.value.code == "provider_rejected"
    assert caught.value.role == "tts"
    assert caught.value.error_stage == "tts"
    assert "private provider payload" not in str(caught.value)


@pytest.mark.parametrize(
    "transcript",
    [
        "你好。",
        "NO_CLEAR_ENGLISH_SPEECH",
        "No.Clear.English.Speech",
        "The spoken words are: 'Turn on the lights.'",
        "The original content of this audio is: turn on the lights.",
    ],
)
async def test_qwen_asr_rejects_foreign_script_or_uncertain_english(transcript):
    class JsonResult:
        def __init__(self):
            self.body = None

        async def json_request(self, body):
            self.body = body
            return {"choices": [{"message": {"content": transcript}, "finish_reason": "stop"}]}

        async def aclose(self):
            pass

    wire = JsonResult()
    with pytest.raises(EnglishTextRejected) as error:
        await QwenAudioASR(dgx3.profile()["asr"], wire).transcribe(b"\0\0" * 160)
    assert error.value.code == "english_only_asr"
    assert "你好" not in str(error.value)
    assert "original language" not in wire.body["messages"][0]["content"][1]["text"]


class SpeechEvents:
    def __init__(self, values):
        self.values, self.closed, self.requests = values, False, []

    async def events(self, body, **kwargs):
        self.requests.append(body)
        try:
            for value in self.values:
                yield value
        finally:
            self.closed = True

    async def aclose(self):
        self.closed = True


async def test_qwen_tts_blocks_non_latin_input_before_provider_request():
    wire = SpeechEvents([])
    with pytest.raises(ValueError) as error:
        _ = [chunk async for chunk in QwenTTS(dgx3.profile()["tts"], wire).synthesize("你好")]
    assert getattr(error.value, "code", None) == "english_only_tts"
    assert wire.requests == []


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"type": "speech.audio.done"}],
        [{"type": "speech.audio.error", "error": "private-body"}],
        [
            {
                "type": "speech.audio.delta",
                "audio": "AAAA",
                "sample_rate": 16000,
                "channels": 1,
                "response_format": "pcm",
            }
        ],
    ],
)
async def test_qwen_tts_never_confirms_empty_disconnected_or_wrong_format_audio(events):
    wire = SpeechEvents(events)
    with pytest.raises(ProviderError) as exc:
        _ = [chunk async for chunk in QwenTTS(dgx3.profile()["tts"], wire).synthesize("Hello")]
    assert wire.closed and "private-body" not in str(exc.value)


async def test_qwen_tts_cancel_closes_stream_without_waiting_for_more_audio():
    wire = SpeechEvents(
        [
            {
                "type": "speech.audio.delta",
                "audio": base64.b64encode(bytes(24000)).decode(),
                "sample_rate": 24000,
                "channels": 1,
                "response_format": "pcm",
            }
        ]
    )
    stream = QwenTTS(dgx3.profile()["tts"], wire).synthesize("Hello")
    assert len(await anext(stream)) == 1024
    await asyncio.wait_for(stream.aclose(), 1)
    assert wire.closed


async def test_only_role_key_enters_mcp_environment_never_arguments(monkeypatch):
    from me3_voice import speech

    captured = []

    class Client:
        def __init__(self, parameters, **kwargs):
            captured.append(parameters)
            self.session = SimpleNamespace(protocol_version="offline-test")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(name=name)
                    for name in (
                        "cancel",
                        "asr_open",
                        "asr_append",
                        "asr_finish",
                        "tts_open",
                        "tts_next",
                    )
                ]
            )

    monkeypatch.setattr(speech, "Client", Client)
    async with speech.SpeechClient(
        backends=dgx3.profile(),
        credentials={
            "ME3_ASR_API_KEY": "offline-asr-key",
            "ME3_TTS_API_KEY": "offline-tts-key",
            "ME3_LLM_API_KEY": "never-forward-this",
        },
    ):
        pass
    assert captured[0].env["ME3_ASR_API_KEY"] == "offline-asr-key"
    assert "ME3_TTS_API_KEY" not in captured[0].env
    assert captured[1].env["ME3_TTS_API_KEY"] == "offline-tts-key"
    assert "ME3_ASR_API_KEY" not in captured[1].env
    assert all("ME3_LLM_API_KEY" not in parameters.env for parameters in captured)
    assert all(
        "offline-asr-key" not in str(parameters.args)
        and "offline-tts-key" not in str(parameters.args)
        for parameters in captured
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "The person says: 'What is the boiling point of water at sea level.'",
            "What is the boiling point of water at sea level.",
        ),
        (
            'The person says: "What is the capital of the Philippines?"',
            "What is the capital of the Philippines?",
        ),
        ("The person says 'No, play Bruno Mars instead.'", "No, play Bruno Mars instead."),
        ("The person says, 'Tell me a joke.'", "Tell me a joke."),
        ("The person says: 'play music by maroon five'.", "play music by maroon five"),
        ("Turn on the lights.", "Turn on the lights."),
    ],
)
async def test_qwen_official_task_extracts_only_complete_transcript_wrapper(raw, expected):
    class Wire:
        async def json_request(self, body):
            assert body["messages"][0]["content"][1]["text"] == "What does the person say?"
            return {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}

    assert await QwenAudioASR(dgx3.profile()["asr"], Wire()).transcribe(b"\0\0" * 160) == expected


async def test_qwen_unquoted_transcript_commentary_is_not_published():
    class Wire:
        async def json_request(self, body):
            return {
                "choices": [
                    {
                        "message": {"content": "The person says: unclear audio"},
                        "finish_reason": "stop",
                    }
                ]
            }

    with pytest.raises(EnglishTextRejected):
        await QwenAudioASR(dgx3.profile()["asr"], Wire()).transcribe(b"\0\0" * 160)


@pytest.mark.parametrize("raw", ["Turn on the lights.'", "The person says: 'Turn on the lights.'"])
async def test_qwen_prefill_returns_transcript_and_uses_open_assistant_message(raw):
    spec = replace(dgx3.profile()["asr"], transcription_mode="prefill")
    spec.validate("asr", runnable=True)

    class Wire:
        async def json_request(self, body):
            assert body["messages"][-1] == {"role": "assistant", "content": "The person says: '"}
            assert body["messages"][1]["content"][1]["text"] == "What does the person say?"
            assert body["continue_final_message"] is True
            assert body["add_generation_prompt"] is False
            assert body["max_tokens"] == 256
            return {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}

    assert await QwenAudioASR(spec, Wire()).transcribe(b"\1\0" * 160) == "Turn on the lights."


@pytest.mark.parametrize(
    "raw,finish", [("Turn on the", "stop"), ("Turn on the lights.'", "length"), ("'", "stop")]
)
async def test_qwen_prefill_rejects_incomplete_transcript(raw, finish):
    class Wire:
        async def json_request(self, body):
            return {"choices": [{"message": {"content": raw}, "finish_reason": finish}]}

    with pytest.raises(ProviderError):
        await QwenAudioASR(
            replace(dgx3.profile()["asr"], transcription_mode="prefill"), Wire()
        ).transcribe(b"\1\0" * 160)


def test_transcription_prefill_is_restricted_to_qwen2_asr():
    with pytest.raises(ConfigurationError, match="prefill requires"):
        replace(dgx3.profile()["tts"], transcription_mode="prefill").validate("tts", runnable=True)
