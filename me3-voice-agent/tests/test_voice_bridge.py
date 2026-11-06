import asyncio
import json
import re
import struct
import wave
from argparse import Namespace
from pathlib import Path

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import TestServer
from aiohttp import WSMsgType, WSServerHandshakeError

from me3_voice.configuration import load_backends
from me3_voice.voice_bridge import BrowserSink, VoiceRuntimeServices, build_app, serve


class FakeSocket:
    closed = False

    def __init__(self):
        self.audio = []
        self.control = []

    async def send_bytes(self, value):
        self.audio.append(value)

    async def send_json(self, value):
        self.control.append(value)


@pytest.mark.asyncio
async def test_runtime_error_publishes_provider_timeout_boundary_to_browser():
    from types import SimpleNamespace

    from me3_voice.voice_bridge import publish_state

    class Socket:
        closed = False

    class EventSink:
        def __init__(self, socket):
            self.socket = socket
            self.messages = []

        async def send(self, value):
            self.messages.append(value)
            if value["type"] == "runtime_event":
                self.socket.closed = True

    socket = Socket()
    event_sink = EventSink(socket)
    session = SimpleNamespace(
        events=[
            {
                "event": "turn_error",
                "run_id": "run-1",
                "session_id": "session-1",
                "turn_id": 1,
                "state": "thinking",
                "error_code": "provider_timeout",
                "error_stage": "reasoning",
                "error_boundary": "response_headers",
                "error_role": "llm",
            }
        ],
        state=SimpleNamespace(value="listening"),
        session_id="session-1",
        run_id="run-1",
        turn=1,
        input_open=False,
    )
    transcript = SimpleNamespace(revision=0, session_id="unstarted")
    await publish_state(socket, session, transcript, event_sink, None)

    assert event_sink.messages[0]["type"] == "runtime_event"
    assert event_sink.messages[0]["data"]["error_boundary"] == "response_headers"
    assert event_sink.messages[0]["data"]["error_role"] == "llm"


@pytest.mark.asyncio
async def test_runtime_services_keep_request_accounting_across_browser_reconnects(monkeypatch):
    from types import SimpleNamespace
    import me3_voice.voice_bridge as bridge

    class FakeSpeech:
        def __init__(self, **_kwargs):
            self.closed = False
            self.owner = None

        async def __aenter__(self):
            self.owner = asyncio.current_task()
            return self

        async def __aexit__(self, *_exc):
            assert asyncio.current_task() is self.owner
            self.closed = True

    class FakeReasoning:
        def __init__(self):
            self.transport = SimpleNamespace(
                spec=SimpleNamespace(request_limit=2),
                requests=0,
            )
            self.closed = False

        async def aclose(self):
            self.closed = True

    reasoning = FakeReasoning()
    monkeypatch.setattr(bridge, "SpeechClient", FakeSpeech)
    monkeypatch.setattr(bridge, "make_provider", lambda *_args, **_kwargs: reasoning)
    services = VoiceRuntimeServices(load_backends(None), {}, Namespace(onit_python=None))
    first = await services.open()
    reasoning.transport.requests = 1
    second = await services.open()
    assert first == second
    assert services.request_budget() == {
        "used": 1,
        "limit": 2,
        "remaining": 1,
        "scope": "bridge runtime",
    }
    await services.close()
    assert first[0].closed and reasoning.closed


@pytest.mark.asyncio
async def test_ui_device_command_dispatch_is_idempotent_for_bridge_lifetime():
    from argparse import Namespace

    services = VoiceRuntimeServices(load_backends(None), {}, Namespace(device_config=None))

    class Session:
        def __init__(self):
            self.calls = []

        async def submit_ui_device_action(self, call):
            self.calls.append((call.name, call.call_id, dict(call.arguments)))

    session = Session()
    command_id = "ui-" + "b" * 32
    first = await services.dispatch_ui_device_action(session, command_id, "light_on", None)
    second = await services.dispatch_ui_device_action(session, command_id, "light_on", None)
    assert first["status"] == "accepted"
    assert second["status"] == "duplicate"
    assert len(session.calls) == 1
    with pytest.raises(ValueError, match="already used"):
        await services.dispatch_ui_device_action(session, command_id, "light_off", None)


@pytest.mark.asyncio
async def test_browser_sink_bounds_audio_until_playback_ack_and_stops():
    socket = FakeSocket()
    sink = BrowserSink(socket)
    for _ in range(8):
        await sink(b"\0\0" * 512)
    ninth = asyncio.create_task(sink(b"\0\0" * 512))
    await asyncio.sleep(0)
    assert not ninth.done() and len(socket.audio) == 8
    sink.acknowledge(struct.unpack("<I", socket.audio[0][:4])[0])
    await asyncio.wait_for(ninth, 1)
    waiting = asyncio.create_task(sink.flush())
    await asyncio.sleep(0)
    assert not waiting.done()
    for packet in socket.audio[1:]:
        sink.acknowledge(struct.unpack("<I", packet[:4])[0])
    await asyncio.wait_for(waiting, 1)
    await sink.stop()
    assert socket.control[-1] == {"type": "audio_stop"}
    assert not sink.pending


@pytest.mark.asyncio
async def test_music_duck_coordinator_keeps_duck_on_barge_in_and_restores_after_turn_or_stop():
    from me3_voice.voice_bridge import MusicDuckingCoordinator

    class Music:
        ducking_enabled = True

        def __init__(self):
            self.duck_state = None
            self.begins = 0
            self.restores = 0

        async def begin_duck(self):
            self.begins += 1
            self.duck_state = {"device_id": "player"}
            return {"outcome": "confirmed"}

        async def restore_duck(self):
            self.restores += 1
            self.duck_state = None
            return {"outcome": "restored"}

    class Session:
        def __init__(self):
            self.events = []

        def event(self, name, **values):
            self.events.append({"event": name, **values})

    music, session = Music(), Session()
    ducker = MusicDuckingCoordinator(session, music)
    ducker.on_event({"event": "wake_detected_file"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert music.begins == 1 and music.restores == 0
    ducker.on_event({"event": "playback_stopped_browser", "input_following": True})
    await asyncio.sleep(0)
    assert music.restores == 0
    ducker.on_event({"event": "turn_completed"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert music.restores == 1
    ducker.on_event({"event": "vad_onset_file"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    ducker.on_event({"event": "playback_stopped_browser", "input_following": False})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert music.restores == 2
    assert all(event["event"] == "music_duck" for event in session.events)
    await ducker.close()


@pytest.mark.asyncio
async def test_music_started_during_active_turn_is_ducked_until_spoken_reply_finishes():
    from me3_voice.voice_bridge import MusicDuckingCoordinator

    class Music:
        ducking_enabled = True

        def __init__(self):
            self.playing = False
            self.duck_state = None
            self.begin_outcomes = []
            self.restores = 0

        async def begin_duck(self):
            self.begin_outcomes.append("playing" if self.playing else "not_playing")
            if not self.playing:
                return {"outcome": "not_playing"}
            self.duck_state = {"device_id": "browser-player"}
            return {"outcome": "confirmed"}

        async def restore_duck(self):
            self.restores += 1
            self.duck_state = None
            return {"outcome": "restored"}

    class Session:
        def __init__(self):
            self.events = []

        def event(self, name, **values):
            self.events.append({"event": name, **values})

    music, session = Music(), Session()
    ducker = MusicDuckingCoordinator(session, music)
    ducker.on_event({"event": "vad_onset_file"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert music.begin_outcomes == ["not_playing"]

    music.playing = True
    ducker.on_event({"event": "music_playback_observed", "status": "playing"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert music.begin_outcomes == ["not_playing", "playing"]
    assert music.duck_state == {"device_id": "browser-player"}
    assert music.restores == 0

    # Session emits turn_completed only after its playback queue drains.
    ducker.on_event({"event": "turn_completed"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert music.duck_state is None
    assert music.restores == 1
    await ducker.close()


@pytest.mark.asyncio
async def test_voice_bridge_is_loopback_only_and_rejects_bad_audio_hello():
    root = Path.cwd()
    args = Namespace(
        kws_assets=root / "var/cache/wake",
        keywords_file=root / "var/cache/wake/hey-swift-keywords.txt",
    )
    app = build_app(root, load_backends(None), {}, args)
    async with TestServer(app, host="127.0.0.1") as server:
        async with ClientSession() as client:
            url = str(server.make_url("/api/voice/capabilities"))
            async with client.get(url, headers={"Host": "example.com"}) as response:
                assert response.status == 403
            async with client.get(url) as response:
                assert response.status == 200
                assert (await response.json())["capture"] == "browser-pcm16"
            ws = await client.ws_connect(str(server.make_url("/api/voice/ws")))
            await ws.send_str(json.dumps({"type": "hello", "rate": 8000, "channels": 1}))
            reply = await ws.receive_json(timeout=2)
            assert reply["type"] == "error" and "audio format" in reply["reason"]
            await ws.close()


@pytest.mark.asyncio
async def test_health_identifies_bridge_and_active_slot_and_disconnect_allows_fresh_session():
    root = Path.cwd()
    args = Namespace(
        kws_assets=root / "var/cache/wake",
        keywords_file=root / "var/cache/wake/hey-swift-keywords.txt",
    )
    app = build_app(root, load_backends(None), {}, args)
    async with TestServer(app, host="127.0.0.1") as server:
        async with ClientSession() as client:
            url = str(server.make_url("/api/health"))
            async with client.get(url) as response:
                assert response.status == 200
                assert await response.json() == {
                    "bridge": "ready",
                    "voice_session": "idle",
                    "speech_services": "lazy",
                    "session_resume": False,
                }
            first = await client.ws_connect(str(server.make_url("/api/voice/ws")))
            async with client.get(url) as response:
                assert (await response.json())["voice_session"] == "active"
            with pytest.raises(WSServerHandshakeError) as conflict:
                await client.ws_connect(str(server.make_url("/api/voice/ws")))
            assert conflict.value.status == 409
            await first.close()
            for _ in range(50):
                async with client.get(url) as response:
                    state = await response.json()
                if state["voice_session"] == "idle":
                    break
                await asyncio.sleep(0.01)
            assert state["voice_session"] == "idle"
            replacement = await client.ws_connect(str(server.make_url("/api/voice/ws")))
            await replacement.close()


def test_bind_conflict_reports_safe_recovery_without_touching_existing_process():
    import socket

    from aiohttp import web

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(SystemExit, match="address already in use.*Inspect the listener"):
            serve(web.Application(), "127.0.0.1", port)
        # The conflicting process/listener remains open; Swift never kills it.
        assert listener.fileno() >= 0
    finally:
        listener.close()


def test_server_host_configuration_rejects_non_loopback_values():
    import me3_voice.voice_bridge as bridge

    assert bridge.loopback_host("127.0.0.1")
    assert bridge.loopback_host("localhost")
    assert not bridge.loopback_host("0.0.0.0")
    assert not bridge.loopback_host("10.0.0.2")


@pytest.mark.asyncio
async def test_laptop_aec_probe_is_served_same_origin_and_cannot_upload_audio():
    root = Path.cwd()
    args = Namespace(
        kws_assets=root / "var/cache/wake",
        keywords_file=root / "var/cache/wake/hey-swift-keywords.txt",
    )
    app = build_app(root, load_backends(None), {}, args)
    async with TestServer(app, host="127.0.0.1") as server:
        async with ClientSession() as client:
            url = str(server.make_url("/audio-probe"))
            async with client.get(url, headers={"Host": "example.com"}) as response:
                assert response.status == 403
            async with client.get(url) as response:
                assert response.status == 200
                body = await response.text()
                policy = response.headers["Content-Security-Policy"]
                nonce = re.search(r"script-src 'nonce-([^']+)'", policy).group(1)
                assert f'<style nonce="{nonce}">' in body
                assert f'<script nonce="{nonce}">' in body
                assert "connect-src 'none'" in policy
                assert "Nothing is uploaded" in body
                assert "echoCancellation:{ideal:requested}" in body
                assert response.headers["Permissions-Policy"] == "microphone=(self)"


@pytest.mark.asyncio
async def test_spotify_ui_routes_are_same_origin_and_return_only_playback_capabilities(monkeypatch):
    import me3_voice.voice_bridge as bridge

    class FakeMusic:
        def __init__(self):
            self.registered = None
            self.commands = []

        @classmethod
        def configured(cls):
            return music

        def capabilities(self):
            return {
                "configured": True,
                "output_backend": "web_playback_sdk",
                "browser_player": True,
                "browser_player_connected": bool(self.registered),
                "setup_required": False,
                "target_configured": False,
            }

        async def browser_access_token(self):
            return "short-lived-access-token"

        def register_browser_device(self, device_id):
            self.registered = device_id

        def unregister_browser_device(self, device_id=None):
            if device_id is None or self.registered == device_id:
                self.registered = None

        async def current_state(self, *, force=False):
            return {
                "configured": True,
                "status": "playing",
                "output_backend": "web_playback_sdk",
                "device_name": "Swift Pi Speaker",
                "track": {"name": "Track", "artists": "Artist", "artwork": None},
                "progress_ms": 1000,
                "duration_ms": 10000,
                "audible": None,
                "reported": True,
            }

        async def ui_control(self, operation, action, position_ms=None, volume_percent=None):
            self.commands.append((operation, action, position_ms, volume_percent))
            return {"mode": "spotify", "summary": "Spotify accepted the control."}

        async def close(self):
            pass

    music = FakeMusic()
    monkeypatch.setattr(bridge, "MusicController", FakeMusic)
    root = Path.cwd()
    args = Namespace(
        kws_assets=root / "var/cache/wake",
        keywords_file=root / "var/cache/wake/hey-swift-keywords.txt",
    )
    app = build_app(root, load_backends(None), {}, args)

    class Request:
        def __init__(self, headers=None, body=None):
            self.headers = {"Host": "127.0.0.1:8765", **(headers or {})}
            self.content_type = self.headers.get("Content-Type", "application/json")
            self.body = body
            self.match_info = {}

        async def json(self):
            return self.body

    def route(path, method):
        resource = next(item for item in app.router.resources() if item.canonical == path)
        return next(item.handler for item in resource if item.method == method)

    async def call(path, method, request):
        return await route(path, method)(request)

    try:
        token_response = await call("/api/voice/music/token", "GET", Request())
        token = json.loads(token_response.body)
        assert token_response.status == 200 and token["access_token"] == "short-lived-access-token"
        assert "no-store" in token_response.headers["Cache-Control"]
        assert token_response.headers["Referrer-Policy"] == "no-referrer"
        with pytest.raises(bridge.web.HTTPForbidden):
            await call(
                "/api/voice/music/token",
                "GET",
                Request({"Origin": "https://attacker.invalid"}),
            )
        status_response = await call("/api/voice/music/status", "GET", Request())
        assert json.loads(status_response.body)["audible"] is None
        headers = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}
        device = await call(
            "/api/voice/music/device",
            "POST",
            Request(headers, {"type": "ready", "device_id": "browser-123"}),
        )
        assert device.status == 200 and music.registered == "browser-123"
        command_id = "ui-" + "a" * 32
        controlled = await call(
            "/api/voice/music/control",
            "POST",
            Request(headers, {"command_id": command_id, "action": "seek", "position_ms": 4500}),
        )
        assert controlled.status == 200 and json.loads(controlled.body)["available"]
        assert music.commands == [(command_id, "seek", 4500, None)]
        await asyncio.sleep(0.21)
        volume_id = "ui-" + "b" * 32
        volume = await call(
            "/api/voice/music/control",
            "POST",
            Request(
                headers, {"command_id": volume_id, "action": "volume_set", "volume_percent": 23}
            ),
        )
        assert volume.status == 200 and json.loads(volume.body)["available"]
        assert music.commands[-1] == (volume_id, "volume_set", None, 23)
        with pytest.raises(bridge.web.HTTPBadRequest):
            await call(
                "/api/voice/music/control",
                "POST",
                Request(
                    headers,
                    {"command_id": "ui-" + "b" * 32, "action": "seek", "position_ms": True},
                ),
            )
    finally:
        await app.cleanup()


@pytest.mark.asyncio
async def test_browser_protocol_reaches_wake_session_mcp_and_acknowledged_playback(
    tmp_path, monkeypatch
):
    import me3_voice.diagnostics as diagnostics

    capture_type = diagnostics.DiagnosticCapture
    monkeypatch.setattr(
        diagnostics,
        "DiagnosticCapture",
        lambda root, run_id, rate, **options: capture_type(tmp_path, run_id, rate, **options),
    )
    root = Path.cwd()
    asset = root / "var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    recording = root / "var/recordings/wake-vad-rearm.wav"
    if not asset.is_dir() or not recording.is_file():
        pytest.skip("optional CPU wake asset and development fixture absent")
    args = Namespace(
        kws_assets=asset,
        keywords_file=root / "var/cache/wake/hey-swift-keywords.txt",
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        wake_threshold=0.5,
        kws_threshold=0.25,
        kws_boost=1.0,
        diagnostic_audio=True,
    )
    app = build_app(root, load_backends(None), {}, args)
    async with TestServer(app, host="127.0.0.1") as server:
        async with ClientSession() as client:
            ws = await client.ws_connect(str(server.make_url("/api/voice/ws")))
            await ws.send_json(
                {"type": "hello", "rate": 16000, "channels": 1, "aec_reported": False}
            )
            observations = {
                "audio": 0,
                "final_user": False,
                "final_assistant": False,
                "states": set(),
                "ready": None,
                "runtime_events": [],
            }

            async def receive():
                async for message in ws:
                    if message.type == WSMsgType.BINARY:
                        observations["audio"] += 1
                        identifier = struct.unpack("<I", message.data[:4])[0]
                        await asyncio.sleep((len(message.data) - 4) / 32000)
                        await ws.send_json({"type": "audio_ack", "id": identifier})
                    elif message.type == WSMsgType.TEXT:
                        value = json.loads(message.data)
                        if value.get("type") == "ready":
                            observations["ready"] = value
                        if value.get("type") == "runtime_event":
                            observations["runtime_events"].append(value)
                        if value.get("type") == "status":
                            assert value["run_id"] and isinstance(value["turn_id"], int)
                            observations["states"].add(value["state"])
                        if value.get("type") == "transcript":
                            for row in value["data"]["messages"]:
                                if row["role"] == "user" and row["status"] == "final":
                                    observations["final_user"] = True
                                if row["role"] == "assistant" and row["status"] == "final":
                                    observations["final_assistant"] = True
                    if observations["final_assistant"]:
                        break

            reader = asyncio.create_task(receive())
            with wave.open(str(recording), "rb") as source:
                assert source.getframerate() == 16000
                while pcm := source.readframes(320):
                    await ws.send_bytes(pcm)
                    await asyncio.sleep(0.02)
                    if observations["final_assistant"]:
                        break
            await asyncio.wait_for(reader, 15)

            ui_command_id = "ui-" + "d" * 32
            await ws.send_json(
                {
                    "type": "device_ui_action",
                    "command_id": ui_command_id,
                    "action": "light_on",
                }
            )
            ui_result = {"ack": None, "receipt": None, "assistant_final": False, "tool_events": []}
            async with asyncio.timeout(8):
                while not (ui_result["receipt"] and ui_result["assistant_final"]):
                    message = await ws.receive()
                    if message.type == WSMsgType.BINARY:
                        ui_result["audio"] = ui_result.get("audio", 0) + 1
                        identifier = struct.unpack("<I", message.data[:4])[0]
                        await ws.send_json({"type": "audio_ack", "id": identifier})
                    elif message.type == WSMsgType.TEXT:
                        value = json.loads(message.data)
                        if value.get("type") == "device_action_ack":
                            ui_result["ack"] = value
                        elif value.get("type") == "runtime_event":
                            data = value["data"]
                            if data.get("tool_call_id") == ui_command_id and data.get("event") in {
                                "tool_started",
                                "tool_completed",
                            }:
                                ui_result["tool_events"].append(data["event"])
                        elif value.get("type") == "transcript":
                            task = next(
                                (
                                    row
                                    for row in value["data"].get("tasks", [])
                                    if row.get("call_id") == ui_command_id
                                ),
                                None,
                            )
                            if task:
                                ui_result["receipt"] = task
                            ui_result["assistant_final"] = any(
                                row.get("turn_id") == 2
                                and row.get("role") == "assistant"
                                and row.get("status") == "final"
                                for row in value["data"].get("messages", [])
                            )
            assert ui_result["ack"]["status"] == "accepted"
            assert ui_result["receipt"]["status"] == "simulated"
            assert ui_result["receipt"]["mode"] == "mock"
            assert ui_result["receipt"]["widget"] == {
                "type": "lamp",
                "on": True,
                "color": "blue",
                "brightness": 60,
            }
            assert ui_result["assistant_final"] and ui_result["audio"] > 0
            assert ui_result["tool_events"] == ["tool_started", "tool_completed"]

            # Resending the same bridge command ID acknowledges the prior
            # action without entering Session or applying the lamp twice.
            await ws.send_json(
                {
                    "type": "device_ui_action",
                    "command_id": ui_command_id,
                    "action": "light_on",
                }
            )
            async with asyncio.timeout(2):
                while True:
                    value = await ws.receive_json()
                    if value.get("type") == "device_action_ack":
                        assert value["status"] == "duplicate"
                        break

            await ws.send_json({"type": "stop"})
            await ws.close()
            assert observations["final_user"] and observations["final_assistant"]
            assert observations["audio"] > 0
            assert observations["ready"]["run_id"]
            assert observations["runtime_events"]
            assert all(
                row["data"]["run_id"] == observations["ready"]["run_id"]
                for row in observations["runtime_events"]
            )
            assert [row["sequence"] for row in observations["runtime_events"]] == list(
                range(1, len(observations["runtime_events"]) + 1)
            )
            assert {"wake_listening", "listening", "speaking"} <= observations["states"]

    captures = list((tmp_path / "var/recordings/diagnostics").glob("*/manifest.json"))
    assert len(captures) == 1
    evidence = json.loads(captures[0].read_text())
    assert evidence["run_id"] == observations["ready"]["run_id"]
    assert observations["ready"]["diagnostic_audio"] is True
    assert evidence["raw"]["saved_bytes"] > 0
    assert evidence["turns"][0]["results"][-1]["final"] is True
    assert evidence["turns"][0]["submitted_bytes"] == evidence["turns"][0]["saved_bytes"]


def test_music_control_telemetry_cannot_include_payloads_or_execute_commands():
    from me3_voice.voice_bridge import music_playback_observed_fields, music_ui_control_fields

    assert music_ui_control_fields(
        {"type": "music_ui_control", "action": "next", "outcome": "accepted"}
    ) == {"action": "next", "outcome": "accepted", "audibility_verified": False}
    for value in (
        {"type": "music_ui_control", "action": "play", "outcome": "accepted"},
        {"type": "music_ui_control", "action": "next", "outcome": "accepted", "token": "private"},
    ):
        with pytest.raises(ValueError):
            music_ui_control_fields(value)

    assert music_playback_observed_fields(
        {
            "type": "music_playback_observed",
            "status": "playing",
            "source": "web_playback_sdk",
            "track_changed": True,
        }
    ) == {"status": "playing", "source": "web_playback_sdk", "track_changed": True}
    for value in (
        {
            "type": "music_playback_observed",
            "status": "playing",
            "source": "web_playback_sdk",
            "track_changed": True,
            "track_name": "private title",
        },
        {
            "type": "music_playback_observed",
            "status": "playing",
            "source": "unknown",
            "track_changed": False,
        },
    ):
        with pytest.raises(ValueError):
            music_playback_observed_fields(value)


@pytest.mark.parametrize("var_root", [False, True])
def test_onit_provider_cap_without_separate_cap_can_start(tmp_path, monkeypatch, var_root):
    """Launch validation must defer to a configured provider cap without None > int."""
    import sys
    import me3_voice.voice_bridge as bridge
    from me3_voice.configuration import BackendSpec

    monkeypatch.chdir(tmp_path)
    specs = load_backends(None)
    specs["llm"] = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="http://202.92.159.240:8001/v1/chat/completions",
        model="Qwen/Qwen3.8-27B",
        allow_private_http=True,
        allow_live=True,
        request_limit=2,
    )
    monkeypatch.setattr(bridge, "load_backends", lambda _: specs)
    monkeypatch.setattr(bridge.LiveTransport, "authorize", lambda _: None)
    built = {}

    def build_app(_root, _backends, _credentials, args, **_kwargs):
        built["args"] = args
        return "app"

    monkeypatch.setattr(bridge, "build_app", build_app)
    started = []
    monkeypatch.setattr(bridge, "serve", lambda *a: started.append(a))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bridge",
            "--onit-python",
            sys.executable,
            "--kws-assets",
            "unused",
            "--keywords-file",
            "unused",
        ],
    )
    if var_root:
        target = tmp_path / "var/project-output"
        target.mkdir(parents=True)
        sys.argv.extend(["--onit-output-root", str(target)])
    bridge.main()
    assert started == [("app", "127.0.0.1", 8765)]
    assert built["args"].onit_reasoning_effort is None
    assert built["args"].onit_max_output_tokens is None
    assert built["args"].onit_no_stream is False


def test_groq_onit_profile_is_explicit_and_reaches_executor(tmp_path, monkeypatch):
    import sys
    import me3_voice.voice_bridge as bridge
    from me3_voice.configuration import BackendSpec

    monkeypatch.chdir(tmp_path)
    specs = load_backends(None)
    specs["llm"] = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        model="openai/gpt-oss-120b",
        allow_live=True,
        request_limit=4,
    )
    monkeypatch.setattr(bridge, "load_backends", lambda _: specs)
    monkeypatch.setattr(bridge.LiveTransport, "authorize", lambda _: None)
    captured = {}

    def build(_root, _backends, _credentials, args, **_kwargs):
        captured["args"] = args
        return "app"

    monkeypatch.setattr(bridge, "build_app", build)
    monkeypatch.setattr(bridge, "serve", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bridge",
            "--onit-python",
            sys.executable,
            "--onit-primary",
            "--onit-max-requests",
            "4",
            "--onit-reasoning-effort",
            "low",
            "--onit-max-output-tokens",
            "2048",
            "--onit-no-stream",
            "--kws-assets",
            "unused",
            "--keywords-file",
            "unused",
        ],
    )
    bridge.main()
    args = captured["args"]
    assert args.onit_reasoning_effort == "low"
    assert args.onit_max_output_tokens == 2048
    assert args.onit_no_stream is True
    assert specs["llm"].endpoint == "https://api.groq.com/openai/v1/chat/completions"

    options = {}

    class CapturingExecutor:
        def __init__(self, *args, **kwargs):
            options.update(kwargs)
            self.mock = object()

        async def prepare(self):
            return None

    from me3_voice import onit_tasks

    monkeypatch.setattr(onit_tasks, "OnitTaskExecutor", CapturingExecutor)

    async def open_services():
        services = VoiceRuntimeServices(specs, {}, args)
        services.speech = object()
        services.devices = object()
        await services.open()

    asyncio.run(open_services())
    assert options["reasoning_effort"] == "low"
    assert options["max_output_tokens"] == 2048
    assert options["streaming"] is False
    assert options["max_requests"] == 4
