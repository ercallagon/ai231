"""Offline contract peers, not live provider/quality or acoustic verification."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from me3_voice.configuration import BackendSpec, ConfigurationError
from me3_voice.remote import ChatReasoning, DeepgramASR, DeepgramTTS, LiveTransport, ProviderError
from me3_voice.providers import MockLLM
from me3_voice.playback import MemorySink
from me3_voice.session import Session
from me3_voice.servers import SpeechService
from me3_voice.contracts import TurnKey, encode_pcm


def spec(role):
    return BackendSpec(
        backend="external",
        protocol={
            "asr": "deepgram-listen-v1",
            "tts": "deepgram-speak-v1",
            "llm": "chat-completions-v1",
        }[role],
        endpoint="https://example.invalid/chat/completions"
        if role == "llm"
        else "wss://example.invalid/" + role,
        auth_env="ME3_TEST_ONLY_KEY",
        model="documented-test-model",
    )


class Peer:
    def __init__(self, messages=()):
        self.queue = asyncio.Queue()
        for message in messages:
            self.queue.put_nowait(message)
        self.sent = []
        self.received = asyncio.Event()
        self.closed = False

    async def send(self, value):
        self.sent.append(value)
        if value == {"type": "CloseStream"}:
            self.queue.put_nowait(
                {
                    "type": "Results",
                    "is_final": True,
                    "channel": {"alternatives": [{"transcript": "world"}]},
                }
            )
            self.queue.put_nowait({"type": "Metadata"})

    async def receive(self):
        value = await self.queue.get()
        self.received.set()
        if isinstance(value, Exception):
            raise value
        return value


class Wire:
    def __init__(self, peer=None, events=()):
        self.peer, self.values = peer, events
        self.params = self.body = None
        self.closed = False

    @asynccontextmanager
    async def socket(self, params):
        self.params = params
        try:
            yield self.peer
        finally:
            self.peer.closed = True

    async def events(self, body, *, total_timeout=True):
        self.body = body
        self.total_timeout = total_timeout
        try:
            for value in self.values:
                if isinstance(value, Exception):
                    raise value
                yield value
        finally:
            self.closed = True

    async def aclose(self):
        self.closed = True


def delta(content="", *, calls=None, finish=None):
    return {
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "tool_calls": calls or []},
                "finish_reason": finish,
            }
        ]
    }


async def test_deepgram_incremental_segments_close_boundary_and_mcp_cleanup():
    peer = Peer(
        [
            {
                "type": "Results",
                "is_final": True,
                "channel": {"alternatives": [{"transcript": "hello"}]},
            },
            {
                "type": "Results",
                "is_final": False,
                "channel": {"alternatives": [{"transcript": "wor"}]},
            },
        ]
    )
    wire = Wire(peer)
    provider = DeepgramASR(spec("asr"), wire)
    service = SpeechService(provider, DeepgramTTS(spec("tts"), Wire()))
    key = TurnKey(session_id="contract", turn_id=1)
    identifier = (await service.open(**key.model_dump()))["operation_id"]
    first = await service.append(identifier, 0, encode_pcm(b"\1\0" * 512))
    await peer.received.wait()
    result = await service.append(identifier, 1, encode_pcm(b"\1\0" * 512))
    assert (result["partial"] or first["partial"])["text"] == "hello wor"
    final = await service.finish(identifier)
    assert final["text"] == "hello world" and final["final"]
    assert peer.closed and not service.operations
    assert wire.params["endpointing"] == "false" and wire.params["sample_rate"] == 16000


async def test_asr_cancel_during_receive_and_disconnected_result_never_finalizes():
    peer = Peer()
    provider = DeepgramASR(spec("asr"), Wire(peer))
    stream = provider.open_stream()
    await stream.append(b"\1\0")
    await stream.aclose()
    assert stream.reader.done() and peer.closed
    peer = Peer([ProviderError("controlled disconnect")])
    stream = DeepgramASR(spec("asr"), Wire(peer)).open_stream()
    with pytest.raises(ProviderError):
        await stream.append(b"\1\0")
        await stream.finish()
    await stream.aclose()
    assert peer.closed


async def test_tts_wire_packet_boundaries_first_audio_flush_and_clear_on_cancel():
    peer = Peer([b"\1", b"\0\2\0", {"type": "Flushed", "sequence_id": 0}])
    provider = DeepgramTTS(spec("tts"), Wire(peer))
    chunks = [chunk async for chunk in provider.synthesize("Hello.")]
    assert b"".join(chunks) == b"\1\0\2\0" and peer.closed
    assert peer.sent == [{"type": "Speak", "text": "Hello."}, {"type": "Flush"}]
    peer = Peer([b"\1\0"])
    stream = DeepgramTTS(spec("tts"), Wire(peer)).synthesize("Hello.")
    assert await anext(stream) == b"\1\0"
    await stream.aclose()
    assert peer.sent[-1] == {"type": "Clear"} and peer.closed


@pytest.mark.parametrize(
    "messages",
    [
        [b"\1", {"type": "Flushed"}],
        [{"type": "Flushed"}],
        [{"type": "Error", "description": "private-body"}],
    ],
)
async def test_tts_rejects_incomplete_or_empty_output(messages):
    peer = Peer(messages)
    with pytest.raises(ProviderError) as error:
        _ = [chunk async for chunk in DeepgramTTS(spec("tts"), Wire(peer)).synthesize("Hello.")]
    assert peer.closed and "private-body" not in str(error.value)


async def test_chat_streams_coherent_text_and_reconstructs_tool_history():
    wire = Wire(
        events=[delta("Hello"), delta(" there. "), delta("Next"), delta(" sentence", finish="stop")]
    )
    model = ChatReasoning(spec("llm"), wire)
    events = []
    model.on_event = events.append
    assert [part async for part in model.respond([{"role": "user", "content": "hello"}])] == [
        "Hello there.",
        "Next sentence",
    ]
    assert (
        events
        == [
            "llm_first_delta",
            "llm_text_stream_completed",
            "llm_text_unpunctuated_finish",
        ]
        and wire.closed
    )
    assert wire.total_timeout is False
    assert (
        "All user-facing replies and spoken summaries must be in English"
        in wire.body["messages"][0]["content"]
    )
    assert "laughter, giggling" in wire.body["messages"][0]["content"]
    assert "complete grammatical sentences" in wire.body["messages"][0]["content"]
    history = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_request": {"name": "read_mock_light", "arguments": "{}"},
            "content": '{"enabled":false}',
        }
    ]
    mapped = model.messages(history)
    assert mapped[-1]["role"] == "assistant"
    assert "read_mock_light" in mapped[-1]["content"]
    assert all("tool_calls" not in message and message["role"] != "tool" for message in mapped)


async def test_empty_successful_sse_response_is_explicitly_diagnosed():
    model = ChatReasoning(spec("llm"), Wire(events=[delta(finish="stop")]))
    events = []
    model.on_event = events.append
    assert [
        part async for part in model.respond([{"role": "user", "content": "Where is Tokyo?"}])
    ] == []
    assert events == ["llm_empty_response"]


def test_completed_task_history_uses_spoken_receipt_without_tool_protocol():
    history = [
        {"role": "user", "content": "play Maroon Five", "turn_id": 1},
        {
            "role": "tool",
            "tool_call_id": "music-1",
            "tool_request": {
                "name": "control_music",
                "arguments": '{"action":"play","query":"Maroon Five"}',
            },
            "content": '{"result_status":"reported","summary":"Spotify acknowledged play: This Love. Playback is unverified.","display":"play"}',
            "turn_id": 1,
        },
        {
            "role": "assistant",
            "content": "Spotify acknowledged play: This Love. Playback is unverified.",
            "turn_id": 1,
        },
        {"role": "user", "content": "What is the capital of the Philippines?", "turn_id": 2},
    ]

    mapped = ChatReasoning.messages(history)
    assert [message["role"] for message in mapped] == ["system", "user", "assistant", "user"]
    assert mapped[-2]["content"].startswith("Spotify acknowledged play")
    assert all("tool_calls" not in message and message["role"] != "tool" for message in mapped)
    assert "letter-by-letter spelling, as authoritative" in mapped[0]["content"]
    assert "Do not invent a place identity or distance" in mapped[0]["content"]


async def test_agent_tool_description_matches_explicit_search_and_github_capabilities():
    wire = Wire(events=[delta("Ready.", finish="stop")])
    model = ChatReasoning(
        spec("llm"),
        wire,
        agent_enabled=True,
        agent_tavily_enabled=True,
        agent_github_readonly=True,
    )
    assert [
        part
        async for part in model.respond(
            [{"role": "user", "content": "Search the web for current weather reports."}]
        )
    ] == ["Ready."]
    agent = next(
        tool["function"]
        for tool in wire.body["tools"]
        if tool["function"]["name"] == "run_agent_task"
    )
    assert "bounded Tavily web searches" in agent["description"]
    assert "read GitHub repository metadata only" in agent["description"]
    assert "cannot write GitHub repositories" in agent["description"]


async def test_agent_tool_does_not_claim_unconfigured_external_integrations():
    wire = Wire(events=[delta("Ready.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, agent_enabled=True)
    _ = [
        part
        async for part in model.respond(
            [{"role": "user", "content": "Search the web for weather reports."}]
        )
    ]
    agent = next(
        tool["function"]
        for tool in wire.body["tools"]
        if tool["function"]["name"] == "run_agent_task"
    )
    assert (
        "Tavily" not in agent["description"]
        and "GitHub repository metadata" not in agent["description"]
    )


async def test_chat_incomplete_tool_args_and_provider_failure_cannot_execute_effects():
    calls = [
        {
            "index": 0,
            "id": "call_1",
            "function": {"name": "set_mock_light", "arguments": '{"enabled":'},
        }
    ]
    wire = Wire(events=[delta(calls=calls), ProviderError("controlled SSE disconnect")])
    model = ChatReasoning(spec("llm"), wire)
    with pytest.raises(ProviderError):
        _ = [part async for part in model.respond([{"role": "user", "content": "light on"}])]
    assert wire.closed
    wire = Wire(
        events=[
            delta(calls=calls),
            delta(calls=[{"index": 0, "function": {"arguments": "true}"}}], finish="tool_calls"),
        ]
    )
    model = ChatReasoning(spec("llm"), wire)
    output = [part async for part in model.respond([{"role": "user", "content": "light on"}])]
    assert len(output) == 1 and output[0].arguments == {"enabled": True}


async def test_malformed_tool_arguments_are_classified_as_response_parse_failures():
    calls = [
        {
            "index": 0,
            "id": "call_bad_json",
            "function": {"name": "set_mock_light", "arguments": '{"enabled":}'},
        }
    ]
    model = ChatReasoning(spec("llm"), Wire(events=[delta(calls=calls, finish="tool_calls")]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "light on"}])]
    assert error.value.code == "provider_invalid_response"
    assert error.value.error_stage == "response_parse"
    assert error.value.error_detail == "invalid_tool_arguments"


@pytest.mark.parametrize(
    ("bad_event", "expected_detail"),
    [
        ([], "event_not_object"),
        ({"error": {"message": "private provider text"}}, "provider_error_event"),
        ({"object": "chat.completion.chunk"}, "missing_choices"),
        ({"choices": "not-a-list"}, "choices_not_array"),
        ({"choices": [{"index": 0}, {"index": 1}]}, "unexpected_choice_count"),
        ({"choices": ["bad choice"]}, "choice_not_object"),
        ({"choices": [{"index": 1, "delta": {}}]}, "invalid_choice_index"),
        ({"choices": [{"delta": []}]}, "invalid_delta_shape"),
        (
            delta(calls=[{"index": 0, "function": None}], finish="tool_calls"),
            "invalid_tool_call_shape",
        ),
    ],
)
async def test_reasoning_stream_reports_safe_structural_failure_detail(bad_event, expected_detail):
    model = ChatReasoning(spec("llm"), Wire(events=[bad_event]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "hello"}])]
    assert error.value.code == "provider_invalid_response"
    assert error.value.error_stage == "response_parse"
    assert error.value.error_detail == expected_detail
    assert "hello" not in str(error.value)


async def test_reasoning_rejects_unfinished_provider_finish_reason_without_tool_effect():
    model = ChatReasoning(spec("llm"), Wire(events=[delta("A partial answer", finish="length")]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "hello"}])]
    assert error.value.code == "provider_invalid_response"
    assert error.value.error_detail == "invalid_finish_reason"


@pytest.mark.parametrize(
    ("call", "expected_detail"),
    [
        (
            {
                "index": 0,
                "id": "bad id",
                "function": {"name": "run_device_action", "arguments": "{}"},
            },
            "invalid_tool_identity",
        ),
        (
            {"index": 0, "id": "unknown_1", "function": {"name": "run_shell", "arguments": "{}"}},
            "unadvertised_tool_name",
        ),
    ],
)
async def test_reasoning_tool_identity_and_name_fail_closed(call, expected_detail):
    model = ChatReasoning(spec("llm"), Wire(events=[delta(calls=[call], finish="tool_calls")]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "create a file"}])]
    assert error.value.error_detail == expected_detail


async def test_chat_mock_action_contract_is_explicit_and_uses_one_validated_call():
    calls = [
        {
            "index": 0,
            "id": "demo-1",
            "function": {
                "name": "run_device_action",
                "arguments": '{"action":"set_timer","value":"10 seconds"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="tool_calls")])
    model = ChatReasoning(spec("llm"), wire)
    output = [
        part async for part in model.respond([{"role": "user", "content": "set a demo timer"}])
    ]
    assert len(output) == 1
    assert output[0].name == "run_device_action"
    assert output[0].arguments == {"action": "set_timer", "value": "10 seconds"}
    body = wire.body
    schema = next(
        tool["function"]
        for tool in body["tools"]
        if tool["function"]["name"] == "run_device_action"
    )
    assert "set_timer" in schema["parameters"]["properties"]["action"]["enum"]
    assert "set_temperature" in schema["parameters"]["properties"]["action"]["enum"]
    assert "cancel_schedule" in schema["parameters"]["properties"]["action"]["enum"]
    assert schema["parameters"]["additionalProperties"] is False
    names = {tool["function"]["name"] for tool in body["tools"]}
    assert not {"call", "message", "make_call", "send_message"} & names


async def test_pi_device_profile_does_not_advertise_in_memory_mock_light_tools():
    wire = Wire(events=[delta(content="The device profile is ready.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    _ = [part async for part in model.respond([{"role": "user", "content": "set a demo timer"}])]
    names = {tool["function"]["name"] for tool in wire.body["tools"]}
    assert "run_device_action" in names
    assert "set_mock_light" not in names and "read_mock_light" not in names


@pytest.mark.parametrize("agent_enabled", [False, True])
async def test_chat_accepts_advertised_control_music_tool(agent_enabled):
    calls = [
        {
            "index": 0,
            "id": "music-1",
            "function": {
                "name": "control_music",
                "arguments": '{"action":"play","query":"Bruno Mars"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="tool_calls")])
    model = ChatReasoning(spec("llm"), wire, agent_enabled=agent_enabled)
    output = [
        part
        async for part in model.respond([{"role": "user", "content": "play a Bruno Mars song"}])
    ]

    assert len(output) == 1
    assert output[0].call_id == "music-1"
    assert output[0].name == "control_music"
    assert output[0].arguments == {"action": "play", "query": "Bruno Mars"}
    music_tool = next(
        tool["function"]
        for tool in wire.body["tools"]
        if tool["function"]["name"] == "control_music"
    )
    assert music_tool["parameters"]["properties"]["action"]["enum"]
    assert "play_playlist" in music_tool["parameters"]["properties"]["action"]["enum"]
    assert "market" in music_tool["parameters"]["properties"]


async def test_chat_accepts_complete_qwen_tool_call_with_stop_finish_reason():
    calls = [
        {
            "index": 0,
            "id": "qwen-music-1",
            "function": {
                "name": "control_music",
                "arguments": '{"action":"pause"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="stop")])
    model = ChatReasoning(spec("llm"), wire)
    output = [
        part async for part in model.respond([{"role": "user", "content": "pause the music"}])
    ]

    assert len(output) == 1
    assert output[0].name == "control_music"
    assert output[0].arguments == {"action": "pause"}


async def test_chat_rejects_tool_call_marked_as_truncated():
    calls = [
        {
            "index": 0,
            "id": "truncated-1",
            "function": {
                "name": "control_music",
                "arguments": '{"action":"pause"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="length")])
    model = ChatReasoning(spec("llm"), wire)

    with pytest.raises(ProviderError, match="incomplete or truncated"):
        _ = [part async for part in model.respond([{"role": "user", "content": "pause music"}])]


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Play a Bruno Mars song", "control_music"),
        ("set a 10 second timer", "run_device_action"),
        ("create a new project folder called demo", "run_agent_task"),
        ("what is a Raspberry Pi", None),
        ("play a song and turn the light on", None),
    ],
)
async def test_live_tool_contract_is_scoped_to_one_clear_request(utterance, expected):
    wire = Wire(events=[delta("Done.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, agent_enabled=True, legacy_mock_tools=False)
    _ = [part async for part in model.respond([{"role": "user", "content": utterance}])]

    advertised = [tool["function"]["name"] for tool in wire.body.get("tools", [])]
    assert advertised == ([expected] if expected else [])
    if expected:
        assert wire.body["tool_choice"] == {
            "type": "function",
            "function": {"name": expected},
        }
        assert "must be in English" in wire.body["messages"][0]["content"]
    else:
        assert "tool_choice" not in wire.body


async def test_music_tool_prompt_drops_unrelated_and_cancelled_conversation_turns():
    wire = Wire(events=[delta("Done.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {"role": "user", "content": "Tell me a joke.", "turn_id": 1},
        {"role": "assistant", "content": "A short joke.", "turn_id": 1},
        {"role": "user", "content": "What is the capital of the Philippines?", "turn_id": 2},
        {"role": "assistant", "content": "Manila.", "turn_id": 2},
        {"role": "user", "content": "He's swift.", "turn_id": 3},
        {"role": "user", "content": "No, play Air Supply.", "turn_id": 4},
    ]

    _ = [part async for part in model.respond(history)]

    assert wire.body["messages"][0]["content"].startswith("You are Swift. Call")
    assert wire.body["messages"][1:] == [{"role": "user", "content": "No, play Air Supply."}]


async def test_music_follow_up_prompt_retains_last_completed_music_receipt_only():
    wire = Wire(events=[delta("Done.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {"role": "user", "content": "What time is it?", "turn_id": 1},
        {"role": "assistant", "content": "It is noon.", "turn_id": 1},
        {"role": "user", "content": "Play Maroon Five.", "turn_id": 2},
        {
            "role": "tool",
            "tool_call_id": "music-2",
            "tool_request": {
                "name": "control_music",
                "arguments": '{"action":"play","query":"Maroon Five"}',
            },
            "content": '{"result_status":"reported","summary":"Spotify acknowledged play: This Love. Playback is unverified."}',
            "turn_id": 2,
        },
        {
            "role": "assistant",
            "content": "Spotify acknowledged play: This Love. Playback is unverified.",
            "turn_id": 2,
        },
        {"role": "user", "content": "Pause it.", "turn_id": 3},
    ]

    _ = [part async for part in model.respond(history)]

    assert [message["role"] for message in wire.body["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert wire.body["messages"][1]["content"] == "Play Maroon Five."
    assert "This Love" in wire.body["messages"][2]["content"]
    assert wire.body["messages"][3]["content"] == "Pause it."


async def test_contextual_music_follow_up_keeps_only_music_tool_available():
    wire = Wire(events=[delta("I will lower it.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {
            "role": "tool",
            "tool_call_id": "music-previous",
            "tool_request": {"name": "control_music", "arguments": '{"action":"play"}'},
            "content": '{"result_status":"completed"}',
        },
        {"role": "user", "content": "turn it down"},
    ]
    _ = [part async for part in model.respond(history)]
    assert [tool["function"]["name"] for tool in wire.body["tools"]] == ["control_music"]


async def test_post_action_reasoning_cannot_repeat_completed_task_tool():
    wire = Wire(events=[delta("I paused playback.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {"role": "user", "content": "pause the music", "turn_id": 4},
        {
            "role": "tool",
            "tool_call_id": "music-4",
            "tool_request": {"name": "control_music", "arguments": '{"action":"pause"}'},
            "content": '{"result_status":"completed"}',
            "turn_id": 4,
        },
    ]
    _ = [part async for part in model.respond(history)]
    assert "tools" not in wire.body


def test_live_gate_precedes_credentials_and_network_and_budget_is_bounded(monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-test-not-a-project-credential")
    transport = LiveTransport(spec("llm"))
    with pytest.raises(ConfigurationError, match="live requests disabled"):
        transport.authorize()
    assert transport.client is None and transport.requests == 0
    transport = LiveTransport(replace(spec("llm"), allow_live=True, request_limit=1))
    assert transport.authorize() == "offline-test-not-a-project-credential"
    with pytest.raises(ProviderError, match="allowance exhausted"):
        transport.authorize()
    monkeypatch.delenv("ME3_TEST_ONLY_KEY")
    with pytest.raises(ConfigurationError, match="missing"):
        LiveTransport(replace(spec("llm"), allow_live=True, request_limit=1)).authorize()
    assert transport.client is None


def test_explicitly_unbounded_provider_counts_without_a_local_ceiling(monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-test-not-a-project-credential")
    transport = LiveTransport(replace(spec("llm"), allow_live=True, request_limit=None))
    for _ in range(25):
        assert transport.authorize() == "offline-test-not-a-project-credential"
    assert transport.requests == 25


async def test_pause_aware_gate_keeps_800ms_pause_and_bounds_incomplete_turn():
    from me3_voice.audio import ConversationInput, ReplayInput
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

    class VAD:
        def voiced(self, pcm):
            return any(pcm)

    class Speech:
        async def transcribe(self, key, pcm):
            return "hello"

        async def synthesize(self, key, text):
            yield b"\1\0"

    class Analyzer:
        def __init__(self):
            self.cleared = 0

        def append_audio(self, pcm, voiced):
            pass

        async def analyze_end_of_turn(self):
            return EndOfTurnState.INCOMPLETE, None

        def clear(self):
            self.cleared += 1

    analyzer = Analyzer()
    pcm, quiet = b"\1\0" * 512, bytes(1024)
    async with Session(Speech(), MockLLM(), MemorySink()) as session:
        source = ConversationInput(ReplayInput(VAD()), session, analyzer)
        source.activate()
        endpoints = 0
        for frame in [pcm] * 3 + [quiet] * 25 + [pcm] * 3 + [quiet] * 64:
            endpoints += bool(await source.feed(frame))
        await session.generation
        assert endpoints == 1 and analyzer.cleared == 1 and session.turn == 1
        assert any(e["event"] == "turn_silence_fallback" for e in session.events)


async def test_actual_http_sse_and_websocket_transport_against_local_contract_peer(monkeypatch):
    """Real HTTP/WS client framing, but an offline loopback peer and dummy key."""
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    seen = []

    async def chat(request):
        seen.append((request.headers.get("Authorization"), await request.json()))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in [
            b": keepalive\r\n\r\ndata:",
            b' {"choices":[{"index":0,"delta":{"content":"Hello. "},"finish_reason":"stop"}]}\r\n\r\n',
            b"data: [DONE]\n\n",
        ]:
            await response.write(chunk)
        await response.write_eof()
        return response

    async def rejected(request):
        return web.Response(status=429, text="private-provider-body")

    async def speak(request):
        seen.append((request.headers.get("Authorization"), dict(request.query)))
        peer = web.WebSocketResponse()
        await peer.prepare(request)
        assert (await peer.receive_json())["type"] == "Speak"
        assert (await peer.receive_json())["type"] == "Flush"
        await peer.send_bytes(b"\1\0\2\0")
        await peer.send_json({"type": "Flushed", "sequence_id": 0})
        await peer.close()
        return peer

    app = web.Application()
    app.router.add_post("/chat", chat)
    app.router.add_post("/reject", rejected)
    app.router.add_get("/speak", speak)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        # Lower transport boundary intentionally uses local HTTP for this test;
        # production factory validation requires HTTPS/WSS.
        chat_spec = replace(
            spec("llm"), endpoint=str(server.make_url("/chat")), allow_live=True, request_limit=2
        )
        model = ChatReasoning(chat_spec)
        try:
            assert [x async for x in model.respond([{"role": "user", "content": "Hi"}])] == [
                "Hello."
            ]
            model.transport.spec = replace(chat_spec, endpoint=str(server.make_url("/reject")))
            with pytest.raises(ProviderError) as error:
                _ = [x async for x in model.respond([{"role": "user", "content": "Hi"}])]
            assert "private-provider-body" not in str(error.value)
            assert error.value.code == "provider_rate_limited"
            assert model.transport.requests == 2
        finally:
            await model.aclose()
        tts = DeepgramTTS(
            replace(
                spec("tts"),
                endpoint=str(server.make_url("/speak")).replace("http:", "ws:"),
                allow_live=True,
                request_limit=1,
            )
        )
        try:
            assert b"".join([x async for x in tts.synthesize("Hello.")]) == b"\1\0\2\0"
        finally:
            await tts.aclose()
    assert (
        seen[0][0] == "Bearer offline-contract-key" and seen[1][0] == "Token offline-contract-key"
    )
    assert seen[1][1]["sample_rate"] == "16000"


async def test_live_chat_stream_waits_past_configured_deadline_for_provider_output(monkeypatch):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def delayed_chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"index":0,"delta":{"content":"Here."},"finish_reason":null}]}\n\n'
        )
        await asyncio.sleep(0.16)
        await response.write(
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        await asyncio.sleep(0.16)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/chat", delayed_chat)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.3,
        )
        transport = LiveTransport(selected)
        try:
            result = [
                event
                async for event in transport.events(
                    {"model": selected.model, "messages": [], "stream": True},
                    total_timeout=False,
                )
            ]
        finally:
            await transport.aclose()

    assert result[0]["choices"][0]["delta"]["content"] == "Here."


async def test_live_chat_stream_fails_after_sse_read_stalls_without_total_turn_deadline(
    monkeypatch,
):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def stalled_chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(0.15)
        return response

    app = web.Application()
    app.router.add_post("/chat", stalled_chat)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.05,
        )
        transport = LiveTransport(selected)
        reasoning = ChatReasoning(selected, transport)
        observed = []
        reasoning.on_event = observed.append
        try:
            with pytest.raises(
                ProviderError, match="timed out waiting for the first SSE event"
            ) as error:
                _ = [text async for text in reasoning.respond([{"role": "user", "content": "Hi."}])]
        finally:
            await transport.aclose()

    assert error.value.code == "provider_timeout"
    assert error.value.role == "llm" and error.value.error_stage == "reasoning"
    assert error.value.error_boundary == "first_sse_event"
    assert observed == ["llm_request_submission_started", "llm_response_headers_received"]


async def test_live_chat_timeout_before_response_headers_has_distinct_boundary(monkeypatch):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def delayed_headers(request):
        await asyncio.sleep(0.15)
        return web.Response(text="late")

    app = web.Application()
    app.router.add_post("/chat", delayed_headers)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.05,
        )
        transport = LiveTransport(selected)
        observed = []
        transport.on_event = observed.append
        try:
            with pytest.raises(
                ProviderError, match="timed out waiting for response headers"
            ) as error:
                _ = [
                    event
                    async for event in transport.events(
                        {"model": selected.model, "messages": [], "stream": True},
                        total_timeout=False,
                    )
                ]
        finally:
            await transport.aclose()

    assert error.value.code == "provider_timeout"
    assert error.value.error_boundary == "response_headers"
    assert observed == ["llm_request_submission_started"]


async def test_live_chat_timeout_before_first_text_delta_is_distinct_from_sse_start(monkeypatch):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def reasoning_only(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}\n\n'
        )
        await asyncio.sleep(0.15)
        return response

    app = web.Application()
    app.router.add_post("/chat", reasoning_only)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.05,
        )
        transport = LiveTransport(selected)
        reasoning = ChatReasoning(selected, transport)
        observed = []
        reasoning.on_event = observed.append
        try:
            with pytest.raises(
                ProviderError, match="timed out waiting for the next SSE event"
            ) as error:
                _ = [text async for text in reasoning.respond([{"role": "user", "content": "Hi."}])]
        finally:
            await transport.aclose()

    assert error.value.error_boundary == "first_text_delta"
    assert observed == [
        "llm_request_submission_started",
        "llm_response_headers_received",
        "llm_first_sse_event",
    ]


async def test_groq_file_asr_multipart_and_fragmented_response_has_no_fake_partials(monkeypatch):
    import io
    import wave
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from me3_voice.remote import GroqWhisperASR, make_provider

    captured = {}

    async def transcribe(request):
        assert request.headers["Authorization"] == "Bearer offline-contract-key"
        form = await request.multipart()
        async for part in form:
            captured[part.name] = await part.read()
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(b'{"text":')
        await asyncio.sleep(0)
        await response.write(b'"recognized offline fixture"}')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/transcribe", transcribe)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    selected = replace(
        spec("asr"),
        protocol="groq-whisper-v1",
        endpoint="https://example.invalid/transcribe",
        streaming=False,
    )
    assert isinstance(make_provider("asr", selected), GroqWhisperASR)
    with pytest.raises(ConfigurationError):
        replace(selected, streaming=True).validate("asr", runnable=True)
    async with TestServer(app, host="127.0.0.1") as server:
        model = GroqWhisperASR(
            replace(
                selected,
                endpoint=str(server.make_url("/transcribe")),
                allow_live=True,
                request_limit=1,
            )
        )
        try:
            stream = model.open_stream()
            assert await stream.append(b"\1\0" * 512) is None
            assert model.transport.requests == 0
            result = await stream.finish()
            assert result.final and result.text == "recognized offline fixture"
            await stream.aclose()
        finally:
            await model.aclose()
    with wave.open(io.BytesIO(captured["file"]), "rb") as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        assert wav.readframes(512) == b"\1\0" * 512
    assert captured["response_format"] == b"json"
    assert captured["language"] == b"en"


@pytest.mark.parametrize("model_id", ["google/gemma-4-31b-it:free", "openrouter/free"])
async def test_openrouter_free_reasoning_adds_zero_price_guard_and_rejects_paid_model(model_id):
    from dataclasses import replace
    from me3_voice.remote import ChatReasoning

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    free = replace(spec("llm"), endpoint=endpoint, model=model_id)
    free.validate("llm", runnable=True)
    with pytest.raises(ConfigurationError):
        replace(free, model="google/gemma-4-31b-it").validate("llm", runnable=True)

    class Peer:
        def __init__(self):
            self.body = None

        async def events(self, body, *, total_timeout=True):
            self.body = body
            self.total_timeout = total_timeout
            yield {
                "choices": [{"index": 0, "delta": {"content": "Hello."}, "finish_reason": "stop"}]
            }

    peer = Peer()
    model = ChatReasoning(free, peer)
    assert [part async for part in model.respond([{"role": "user", "content": "Hi"}])] == ["Hello."]
    assert peer.body["provider"] == {
        "max_price": {"prompt": 0, "completion": 0},
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert peer.body["model"] == model_id
