import asyncio

import pytest

from me3_voice.music import MusicController, MusicUnavailable


async def test_music_correction_deduplicates_completed_actions():
    music = MusicController()
    first = await music.execute("one", "play", "Taylor Swift")
    assert first["mode"] == "mock" and music.current == "Taylor Swift"
    music.cancel_pending()
    await music.execute("two", "play", "Bruno Mars")
    assert music.current == "Bruno Mars"
    assert await music.execute("one", "play", "Taylor Swift") == first
    assert music.current == "Bruno Mars"
    await music.execute("three", "pause")
    assert not music.playing
    await music.close()


async def test_spotify_interrupted_search_cannot_play_old_track():
    searching = asyncio.Event()
    release = asyncio.Event()
    mutations = []

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {
                "devices": [
                    {"id": "laptop", "name": "Laptop", "is_active": True, "is_restricted": False}
                ]
            }
        if path == "/search":
            if params["q"] == "Taylor Swift":
                searching.set()
                await release.wait()
            return {"tracks": {"items": [{"uri": "spotify:track:123", "name": params["q"]}]}}
        if path == "/me/player":
            return {}
        mutations.append(body)
        return {}

    music = MusicController(token="dummy", request=peer)
    old = asyncio.create_task(music.execute("old", "play", "Taylor Swift"))
    await searching.wait()
    music.cancel_pending()
    old.cancel()
    with pytest.raises(asyncio.CancelledError):
        await old
    await music.execute("new", "play", "Bruno Mars")
    release.set()
    await music.close()
    assert len(mutations) == 1
    assert music.current == "Bruno Mars"
    with pytest.raises(MusicUnavailable, match="superseded"):
        await music.execute("old", "play", "Taylor Swift")


async def test_spotify_acknowledgment_is_not_claimed_as_audible_playback():
    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/me/player/devices":
            return {
                "devices": [
                    {"id": "laptop", "name": "Laptop", "is_active": True, "is_restricted": False}
                ]
            }
        return (
            {"tracks": {"items": [{"uri": "spotify:track:123", "name": "Example"}]}}
            if method == "GET"
            else {}
        )

    music = MusicController(token="dummy", device_id="laptop", request=peer)
    receipt = await music.execute("play", "play", "Bruno Mars")
    assert receipt["mode"] == "spotify"
    assert "Spotify accepted play" in receipt["summary"]
    assert "playing" not in receipt["summary"].lower()
    mutation = next(call for call in calls if call[0] == "PUT")
    assert mutation[2]["params"]["device_id"] == "laptop"
    assert mutation[2]["body"] == {"uris": ["spotify:track:123"]}
    await music.execute("pause", "pause")
    assert any(path == "/me/player/pause" for _, path, _ in calls)
    await music.close()


async def test_spotify_playlist_discovery_uses_search_market_and_context_playback():
    calls = []
    playlist_uri = "spotify:playlist:chill123"

    async def peer(method, path, *, params=None, body=None):
        calls.append((method, path, params, body))
        if path == "/me/player/devices":
            return {"devices": [player("speaker", True)]}
        if path == "/search":
            assert params == {
                "q": "chill",
                "type": "playlist",
                "limit": "5",
                "market": "PH",
            }
            return {"playlists": {"items": [{"name": "Chill Hits", "uri": playlist_uri}]}}
        if path == "/me/player/play":
            return {}
        if path == "/me/player":
            return {
                "is_playing": True,
                "context": {"uri": playlist_uri, "type": "playlist"},
                "device": {"id": "speaker"},
                "item": {"name": "Quiet Track"},
            }
        raise AssertionError(f"unexpected Spotify request {method} {path}")

    music = MusicController(token="test-token", device_id="speaker", request=peer)
    receipt = await music.execute("playlist-search", "play_playlist", "chill", "PH")
    assert receipt["mode"] == "spotify"
    assert receipt["display"] == "playing"
    assert receipt["summary"] == (
        "Spotify reports the Chill Hits playlist is active; now playing Quiet Track."
    )
    write = next(row for row in calls if row[1] == "/me/player/play")
    assert write[3] == {"context_uri": playlist_uri}
    assert music.current == "Quiet Track"
    await music.close()


async def test_browser_player_is_the_only_web_sdk_target_and_requires_streaming_scope():
    from me3_voice.music import WEB_PLAYBACK_SCOPES

    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/me/player/devices":
            return {"devices": [player("swift-browser", True)]}
        if path == "/search":
            return {"tracks": {"items": [{"uri": "spotify:track:abc", "name": "Example"}]}}
        return {}

    music = MusicController(
        token="short-lived-test",
        request=peer,
        output_backend="web_playback_sdk",
        scopes=WEB_PLAYBACK_SCOPES | {"user-modify-playback-state", "user-read-playback-state"},
    )
    assert music.capabilities()["browser_player"]
    with pytest.raises(MusicUnavailable, match="Connect Spotify"):
        await music.execute("before-connect", "play", "Bruno Mars")
    assert [(method, path) for method, path, _ in calls] == [("GET", "/me/player/devices")]
    music.register_browser_device("swift-browser")
    receipt = await music.execute("play-after-connect", "play", "Bruno Mars")
    assert receipt["mode"] == "spotify"
    mutation = next(call for call in calls if call[1] == "/me/player/play")
    assert mutation[2]["params"] == {"device_id": "swift-browser"}
    assert mutation[2]["body"] == {"uris": ["spotify:track:abc"]}
    music.unregister_browser_device("wrong-device")
    assert music.capabilities()["browser_player_connected"]
    music.unregister_browser_device("swift-browser")
    assert not music.capabilities()["browser_player_connected"]
    await music.close()


async def test_browser_access_token_is_gated_by_granted_streaming_scope():
    from me3_voice.music import WEB_PLAYBACK_SCOPES

    async def peer(method, path, **kwargs):
        assert (method, path) == ("GET", "/me/player/devices")
        return {"devices": []}

    missing = MusicController(token="private-test-token", request=peer)
    with pytest.raises(MusicUnavailable) as error:
        await missing.browser_access_token()
    assert error.value.code == "music_streaming_scope_required"
    music = MusicController(
        token="private-test-token",
        request=peer,
        output_backend="web_playback_sdk",
        scopes=WEB_PLAYBACK_SCOPES,
    )
    assert await music.browser_access_token() == "private-test-token"
    await missing.close()
    await music.close()


async def test_spotify_state_is_normalized_without_claiming_audibility():
    async def peer(method, path, **kwargs):
        assert (method, path) == ("GET", "/me/player")
        return {
            "is_playing": True,
            "progress_ms": 12345,
            "device": {"name": "Swift Pi Speaker", "id": "pi-player"},
            "item": {
                "name": "Track",
                "artists": [{"name": "Artist"}],
                "duration_ms": 180000,
                "album": {"images": [{"url": "https://i.scdn.co/image/example"}]},
            },
        }

    music = MusicController(token="test-token", request=peer)
    state = await music.current_state(force=True)
    assert state["status"] == "playing"
    assert state["device_name"] == "Swift Pi Speaker"
    assert state["track"] == {
        "name": "Track",
        "artists": "Artist",
        "artwork": "https://i.scdn.co/image/example",
    }
    assert state["progress_ms"] == 12345 and state["duration_ms"] == 180000
    assert state["audible"] is None
    await music.close()


async def test_old_player_observation_does_not_replace_requested_track_receipt():
    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path))
        if path == "/me/player/devices":
            return {"devices": [player("speaker", True)]}
        if path == "/search":
            return {"tracks": {"items": [{"uri": "spotify:track:new", "name": "New song"}]}}
        if method == "GET" and path == "/me/player":
            return {
                "is_playing": True,
                "device": {"id": "speaker"},
                "item": {"uri": "spotify:track:old", "name": "Old song"},
            }
        return {}

    music = MusicController(token="mock", device_id="speaker", request=peer)
    try:
        receipt = await music.execute("play-new", "play", "Celine Dion")
        assert receipt["summary"] == "Spotify accepted play: New song."
        assert music.current == "New song"
        assert calls.count(("PUT", "/me/player/play")) == 1
    finally:
        await music.close()


async def test_spotify_ui_seek_is_bounded_and_targets_the_configured_receiver():
    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/me/player/devices":
            return {"devices": [player("pi-player")]}
        if path == "/me/player":
            return {}
        return {}

    music = MusicController(token="test-token", device_id="pi-player", request=peer)
    await music.ui_control("seek-1", "seek", 91234)
    request = next(call for call in calls if call[1] == "/me/player/seek")
    assert request[0] == "PUT"
    assert request[2]["params"] == {"device_id": "pi-player", "position_ms": "91234"}
    with pytest.raises(ValueError):
        await music.ui_control("seek-bad", "seek", -1)
    await music.close()


async def test_session_music_interruption_and_eight_explicit_requests_use_direct_route_once():
    from me3_voice.providers import ToolCall
    from me3_voice.session import Session
    from me3_voice.transcript import Transcript, normalize_transcript

    class Speech:
        async def transcribe(self, key, pcm):
            return pcm.decode()

        async def synthesize(self, key, text):
            yield b"\0\0" * 512

    class Reasoning:
        calls = 0

        async def respond(self, history):
            self.calls += 1
            query = next(row["content"] for row in reversed(history) if row["role"] == "user")
            yield ToolCall(
                f"music-{self.calls}", "control_music", {"action": "play", "query": query}
            )

    started, release = asyncio.Event(), asyncio.Event()

    async def sink(pcm):
        started.set()
        await release.wait()

    llm, book = Reasoning(), Transcript()
    async with Session(Speech(), llm, sink, transcript=book) as session:
        session.activate()
        first = await session.submit(b"Play Taylor Swift")
        await asyncio.wait_for(started.wait(), 2)
        await session.interrupt()
        assert first.done()
        release.set()
        await (await session.submit(b"Play Bruno Mars"))
        for _ in range(6):
            await (await session.submit(b"Play Bruno Mars"))
        assert llm.calls == 0
        assert len(session.tasks.music.operations) == 8
        assert session.tasks.music.current == "Bruno Mars"
        normalized = normalize_transcript(book.snapshot())
        assert len(normalized["tasks"]) == 8
        assert all(
            t["kind"] == "music_play" and t["status"] == "simulated" for t in normalized["tasks"]
        )
        assert not any(e["event"] == "turn_error" for e in session.events)


async def test_oauth_rejects_callback_state_before_token_exchange(monkeypatch):
    import me3_voice.music as module

    monkeypatch.setattr(
        module.getpass,
        "getpass",
        lambda prompt: "http://127.0.0.1:8876/callback?state=wrong&code=secret",
    )
    with pytest.raises(MusicUnavailable, match="callback rejected"):
        await module.authorize("example-public-client-id")


async def test_oauth_exchange_reports_safe_spotify_error_code_without_secrets(monkeypatch):
    import json
    import secrets
    import me3_voice.music as module

    values = iter(["test-verifier", "test-state"])
    monkeypatch.setattr(secrets, "token_urlsafe", lambda _: next(values))
    monkeypatch.setattr(
        module.getpass,
        "getpass",
        lambda prompt: "http://127.0.0.1:8876/callback?state=test-state&code=one-time-code",
    )

    class Reply:
        status = 400
        content = None

        def __init__(self):
            self.content = self

        async def read(self, limit):
            if getattr(self, "_body_consumed", False):
                return b""
            self._body_consumed = True
            return json.dumps(
                {"error": "invalid_grant", "error_description": "do-not-print-this-body"}
            ).encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return Reply()

    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda **kwargs: Client())
    with pytest.raises(MusicUnavailable, match="HTTP 400, invalid_grant") as error:
        await module.authorize("example-public-client-id")
    message = str(error.value)
    assert "fresh authorization" in message
    assert "one-time-code" not in message
    assert "test-verifier" not in message
    assert "do-not-print-this-body" not in message


def test_music_configuration_cannot_reference_an_external_source_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_MUSIC_CONFIG", str(tmp_path / "external.json"))
    with pytest.raises(ValueError, match="inside ME3"):
        MusicController.configured()


async def test_spotify_transport_reuses_connection_and_redacts_expired_token(monkeypatch):
    import me3_voice.music as module

    clients, calls = [], []
    statuses = [200, 204, 401]

    class Reply:
        def __init__(self):
            self.status = statuses.pop(0)
            self.content = self

        async def read(self, limit):
            if getattr(self, "_body_consumed", False):
                return b""
            self._body_consumed = True
            return b'{"tracks":{"items":[]}}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Client:
        closed = False

        def __init__(self, **kwargs):
            clients.append(self)

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return Reply()

        async def close(self):
            self.closed = True

    monkeypatch.setattr(module.aiohttp, "ClientSession", Client)
    music = MusicController(token="do-not-expose-this-token")
    await music.request("GET", "/search", params={"q": "Bruno Mars"})
    await music.request("PUT", "/me/player/pause")
    with pytest.raises(MusicUnavailable, match="token expired") as error:
        await music.request("GET", "/me/player")
    assert "do-not-expose-this-token" not in str(error.value)
    assert len(clients) == 1 and len(calls) == 3
    assert all(call[1].startswith("https://api.spotify.com/v1/") for call in calls)
    assert all(call[2]["allow_redirects"] is False for call in calls)
    await music.close()
    assert clients[0].closed


def player(identifier, active=False, restricted=False):
    return {
        "id": identifier,
        "name": "Test receiver",
        "is_active": active,
        "is_restricted": restricted,
    }


@pytest.mark.parametrize(
    "rows, configured, expected",
    [
        ([player("background")], None, "background"),
        ([player("one"), player("active", True)], None, "active"),
        ([player("chosen"), player("active", True)], "chosen", "chosen"),
    ],
)
async def test_available_receiver_is_targeted_without_manual_song_or_transfer(
    rows, configured, expected
):
    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/me/player/devices":
            return {"devices": rows}
        if path == "/search":
            return {"tracks": {"items": [{"uri": "spotify:track:123", "name": "Test track"}]}}
        return {}

    music = MusicController(token="dummy", device_id=configured, request=peer)
    await music.execute("bruno", "play", "Bruno Mars")
    writes = [call for call in calls if call[0] == "PUT"]
    assert len(writes) == 1 and writes[0][1] == "/me/player/play"
    assert writes[0][2]["params"] == {"device_id": expected}
    await music.execute("bruno", "play", "Bruno Mars")
    assert len([call for call in calls if call[0] == "PUT"]) == 1
    await music.close()


@pytest.mark.parametrize(
    "rows, configured, code",
    [
        ([], None, "music_device_unavailable"),
        ([player("restricted", restricted=True)], None, "music_device_unavailable"),
        ([player("one"), player("two")], None, "music_device_ambiguous"),
        ([player("other", True)], "missing", "music_device_unavailable"),
    ],
)
async def test_missing_ambiguous_or_wrong_receiver_never_searches_or_plays(rows, configured, code):
    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path))
        return {"devices": rows}

    music = MusicController(token="dummy", device_id=configured, request=peer)
    with pytest.raises(MusicUnavailable) as error:
        await music.execute("attempt", "play", "Bruno Mars")
    assert error.value.code == code
    assert calls == [("GET", "/me/player/devices")]
    await music.close()


async def test_correction_during_device_discovery_cannot_start_obsolete_music():
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def peer(method, path, **kwargs):
        calls.append((method, path))
        started.set()
        await release.wait()
        return {"devices": [player("laptop")]}

    music = MusicController(token="dummy", request=peer)
    task = asyncio.create_task(music.execute("old", "play", "Bruno Mars"))
    await started.wait()
    music.cancel_pending()
    release.set()
    with pytest.raises(MusicUnavailable, match="superseded"):
        await task
    assert calls == [("GET", "/me/player/devices")]
    await music.close()


@pytest.mark.parametrize("rotate", [False, True])
async def test_spotify_refresh_preserves_private_store_and_retries_only_401(
    monkeypatch, tmp_path, rotate
):
    import json
    from me3_voice import credentials

    monkeypatch.setattr(credentials, "CACHE_ROOT", tmp_path)
    secret = tmp_path / "spotify.env"
    credentials.write_secrets(
        secret,
        {
            "ME3_SPOTIFY_ACCESS_TOKEN": "expired-test",
            "ME3_SPOTIFY_REFRESH_TOKEN": "refresh-test",
            "ME3_LLM_API_KEY": "unrelated-test",
        },
    )
    calls = []

    class Reply:
        def __init__(self, status, data):
            self.status, self.data, self.content = status, data, self

        async def read(self, limit):
            if getattr(self, "_body_consumed", False):
                return b""
            self._body_consumed = True
            return json.dumps(self.data).encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Client:
        async def close(self):
            pass

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            if len(calls) == 1:
                return Reply(401, {})
            if method == "POST":
                data = {"access_token": "renewed-test", "token_type": "Bearer"}
                if rotate:
                    data["refresh_token"] = "rotated-test"
                return Reply(200, data)
            return Reply(200, {"devices": [player("laptop")]})

    music = MusicController(
        token="expired-test",
        client_id="test-client",
        refresh_token="refresh-test",
        secrets_path=secret,
    )
    music.client = Client()
    assert (await music.devices())[0]["id"] == "laptop"
    values = credentials.load_secrets(secret)
    assert values["ME3_SPOTIFY_ACCESS_TOKEN"] == "renewed-test"
    assert values["ME3_SPOTIFY_REFRESH_TOKEN"] == ("rotated-test" if rotate else "refresh-test")
    assert values["ME3_LLM_API_KEY"] == "unrelated-test"
    assert secret.stat().st_mode & 0o777 == 0o600
    assert len(calls) == 3 and "headers" not in calls[1][2]
    assert calls[1][1] == "https://accounts.spotify.com/api/token"
    assert calls[2][2]["headers"]["Authorization"] == "Bearer renewed-test"
    await music.close()


async def test_spotify_second_401_stops_without_looping_or_replaying_playback():
    import json

    calls = []

    class Reply:
        content = None

        def __init__(self, status):
            self.status, self.content = status, self

        async def read(self, limit):
            if getattr(self, "_body_consumed", False):
                return b""
            self._body_consumed = True
            return json.dumps({"access_token": "new-test", "token_type": "Bearer"}).encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Client:
        async def close(self):
            pass

        def request(self, method, url, **kwargs):
            calls.append((method, url))
            return Reply(200 if method == "POST" else 401)

    music = MusicController(token="old-test", refresh_token="refresh-test", client_id="test-client")
    music.client = Client()
    with pytest.raises(MusicUnavailable) as error:
        await music.devices()
    assert error.value.code == "music_auth_expired"
    assert len(calls) == 3 and not any(method == "PUT" for method, _ in calls)
    await music.close()


async def test_chat_decoder_to_spotify_then_direct_spoken_correction():
    import json
    from me3_voice.configuration import BackendSpec
    from me3_voice.remote import ChatReasoning
    from me3_voice.session import Session
    from me3_voice.tools import MockTaskExecutor
    from me3_voice.transcript import Transcript

    writes = []
    current = {}

    async def spotify(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [player("laptop")]}
        if path == "/search":
            current.update(uri="spotify:track:" + str(len(writes)), name=params["q"])
            return {"tracks": {"items": [current.copy()]}}
        if method == "PUT":
            writes.append((params["device_id"], body["uris"][0], current["name"]))
        return {"is_playing": True, "item": current.copy(), "device": {"id": "laptop"}}

    class Wire:
        calls = 0

        async def events(self, body, *, total_timeout=True):
            self.calls += 1
            query = next(
                row["content"] for row in reversed(body["messages"]) if row["role"] == "user"
            )
            if "maroon" in query.casefold():
                query = "Maroon 5"
            elif "bruno" in query.casefold():
                query = "Bruno Mars"
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": f"music-{self.calls}",
                                    "function": {
                                        "name": "control_music",
                                        "arguments": json.dumps({"action": "play", "query": query}),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

        async def aclose(self):
            pass

    class Speech:
        async def transcribe(self, key, pcm):
            return pcm.decode()

        async def synthesize(self, key, text):
            yield b"\0\0" * 512

    started, release = asyncio.Event(), asyncio.Event()

    async def sink(pcm):
        started.set()
        await release.wait()

    wire, book = Wire(), Transcript()
    llm = ChatReasoning(
        BackendSpec(
            backend="external",
            protocol="chat-completions-v1",
            endpoint="https://example.invalid/chat",
            auth_env="ME3_TEST_KEY",
            model="offline",
        ),
        wire,
    )
    tasks = MockTaskExecutor()
    tasks.music = MusicController(token="dummy", request=spotify)
    async with Session(Speech(), llm, sink, transcript=book, tasks=tasks) as session:
        session.activate()
        first = await session.submit(b"Play Bruno Mars")
        await asyncio.wait_for(started.wait(), 2)
        assert len(writes) == 1 and writes[0][2] == "Bruno Mars"
        await session.interrupt()
        assert first.done()
        release.set()
        await (await session.submit(b"No, play Maroon 5 instead"))
        assert [row[2] for row in writes] == ["Bruno Mars", "Maroon 5"]
        assert wire.calls == 0, "the explicit play and correction should bypass reasoning"
        assert not any(e["event"] == "turn_error" for e in session.events)
    assert all(row["status"] == "reported" for row in book.snapshot()["tasks"])


async def test_browser_token_refresh_does_not_depend_on_device_discovery():
    import time
    from me3_voice.music import WEB_PLAYBACK_SCOPES

    calls = []

    async def unavailable_devices(*args, **kwargs):
        raise AssertionError("SDK token must not depend on device discovery")

    music = MusicController(
        token="old-test",
        client_id="test-client",
        refresh_token="test-refresh",
        output_backend="web_playback_sdk",
        scopes=WEB_PLAYBACK_SCOPES,
        request=unavailable_devices,
    )

    class Client:
        async def close(self):
            pass

    music.client = Client()

    async def refresh(rejected):
        calls.append(rejected)
        music.token = "fresh-test"
        music.browser_token_until = time.monotonic() + 3300

    music.renew_token = refresh
    assert await music.browser_access_token() == "fresh-test"
    assert await music.browser_access_token() == "fresh-test"
    assert calls == ["old-test"]
    await music.close()


async def test_browser_state_identifies_only_registered_receiver():
    from me3_voice.music import WEB_PLAYBACK_SCOPES

    async def peer(*args, **kwargs):
        return {
            "device": {"id": "browser-one", "name": "Browser"},
            "is_playing": True,
            "item": {"name": "Track", "artists": []},
        }

    music = MusicController(
        token="test-token",
        request=peer,
        output_backend="web_playback_sdk",
        scopes=WEB_PLAYBACK_SCOPES,
    )
    assert (await music.current_state(force=True))["target_matched"] is False
    music.register_browser_device("browser-two")
    assert (await music.current_state(force=True))["target_matched"] is False
    music.register_browser_device("browser-one")
    assert (await music.current_state(force=True))["target_matched"] is True
    await music.close()


async def test_streaming_only_saved_grant_is_not_full_sdk_authorization():
    music = MusicController(
        token="test-token",
        output_backend="web_playback_sdk",
        scopes={"streaming", "user-read-playback-state", "user-modify-playback-state"},
    )
    assert not music.browser_sdk_authorized
    assert music.capabilities()["setup_required"]
    with pytest.raises(MusicUnavailable, match="profile permissions"):
        await music.browser_access_token()
    await music.close()


@pytest.mark.parametrize("valid_json", [True, False])
async def test_authorize_collects_fragmented_token_response(monkeypatch, valid_json):
    import json
    import secrets
    import me3_voice.music as module

    nonces = iter(["test-verifier", "test-state"])
    monkeypatch.setattr(secrets, "token_urlsafe", lambda _: next(nonces))
    monkeypatch.setattr(
        module.getpass,
        "getpass",
        lambda _: "http://127.0.0.1:8876/callback?state=test-state&code=private-code",
    )
    payload = (
        json.dumps(
            {
                "access_token": "private-test-access",
                "refresh_token": "private-test-refresh",
                "token_type": "Bearer",
                "scope": " ".join(module.BASE_SCOPES | module.WEB_PLAYBACK_SCOPES),
            }
        ).encode()
        if valid_json
        else b'{"access_token":"private-test-access",'
    )
    fragments = [payload[:9], payload[9:25], payload[25:], b""]

    class Reply:
        status = 200

        def __init__(self):
            self.content = self

        async def read(self, limit):
            return fragments.pop(0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            assert (
                module.WEB_PLAYBACK_SCOPES.issubset(set(kwargs["data"].get("scope", "").split()))
                is False
            )  # scopes are requested on authorize, not token exchange
            return Reply()

    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda **_: Client())
    if valid_json:
        result = await module.authorize("public-test-client", browser_player=True)
        assert result["access_token"] == "private-test-access"
        assert fragments == []
    else:
        with pytest.raises(MusicUnavailable, match="full response was collected") as error:
            await module.authorize("public-test-client", browser_player=True)
        assert "private-test-access" not in str(error.value)
        assert "private-code" not in str(error.value)


async def test_spotify_response_stream_enforces_size_limit():
    from me3_voice.music import read_spotify_body

    class Stream:
        async def read(self, limit):
            return b"x" * limit

    class Reply:
        content = Stream()

    with pytest.raises(MusicUnavailable, match="size limit"):
        await read_spotify_body(Reply(), 16)


async def test_artist_queue_supports_navigation_without_mixing_other_artists():
    writes = []

    async def peer(method, path, **kwargs):
        if path == "/me/player/devices":
            return {
                "devices": [
                    {"id": "swift", "name": "Swift", "is_active": True, "is_restricted": False}
                ]
            }
        if path == "/search":
            assert kwargs["params"]["limit"] == "5"
            return {
                "tracks": {
                    "items": [
                        {
                            "uri": "spotify:track:one",
                            "name": "One",
                            "artists": [{"id": "artist-a"}],
                        },
                        {"uri": "spotify:track:two", "artists": [{"id": "artist-a"}]},
                        {"uri": "spotify:track:cover", "artists": [{"id": "artist-b"}]},
                        {"uri": "spotify:track:one", "artists": [{"id": "artist-a"}]},
                    ]
                }
            }
        if path == "/me/player":
            return {}
        writes.append((method, path, kwargs))
        return {}

    music = MusicController(token="test-token", device_id="swift", request=peer)
    await music.execute("play-one", "play", "Artist A")
    assert writes[0][2]["body"] == {"uris": ["spotify:track:one", "spotify:track:two"]}
    await music.execute("play-one", "play", "Artist A")
    assert len(writes) == 1
    await music.ui_control("skip-one", "next")
    await music.ui_control("previous-one", "previous")
    assert [row[1] for row in writes[1:]] == ["/me/player/next", "/me/player/previous"]
    assert all(row[2]["params"] == {"device_id": "swift"} for row in writes)
    await music.close()


async def test_spoken_music_operations_cover_navigation_stop_and_volume_through_one_adapter():
    device = {
        "id": "laptop",
        "name": "Swift browser",
        "supports_volume": True,
        "volume_percent": 60,
    }
    state = {
        "device": device,
        "is_playing": True,
        "item": {"uri": "spotify:track:one", "name": "One"},
    }
    calls = []

    async def peer(method, path, *, params=None, body=None):
        calls.append((method, path, params, body))
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/search":
            return {"tracks": {"items": [{"uri": "spotify:track:one", "name": "One"}]}}
        if path == "/me/player":
            return state
        if path == "/me/player/volume":
            if method == "PUT":
                device["volume_percent"] = int(params["volume_percent"])
            return {}
        if path == "/me/player/pause":
            state["is_playing"] = False
        elif path in {"/me/player/play", "/me/player/next", "/me/player/previous"}:
            state["is_playing"] = True
        return {}

    music = MusicController(token="test-token", device_id="laptop", request=peer)
    for index, action in enumerate(("next", "previous", "pause", "resume", "stop")):
        await music.execute(f"voice-{index}", action)
    await music.execute("volume-up", "volume_up")
    assert device["volume_percent"] == 70
    await music.ui_control("volume-down", "volume_down")
    assert device["volume_percent"] == 60
    await music.ui_control("volume-absolute", "volume_set", volume_percent=23)
    assert device["volume_percent"] == 23
    stopped = await music.execute("stop", "stop")
    assert "does not expose a separate stop" in stopped["summary"]
    assert [
        row[1] for row in calls if row[0] in {"PUT", "POST"} and row[1].startswith("/me/player/")
    ] == [
        "/me/player/next",
        "/me/player/previous",
        "/me/player/pause",
        "/me/player/play",
        "/me/player/pause",
        "/me/player/volume",
        "/me/player/volume",
        "/me/player/volume",
        "/me/player/pause",
    ]
    await music.close()


async def test_music_ducking_restores_only_its_own_volume_without_resuming():
    device = {
        "id": "laptop",
        "name": "Swift browser",
        "supports_volume": True,
        "volume_percent": 80,
    }
    state = {"device": device, "is_playing": True}
    writes = []

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/me/player":
            return state
        if path == "/me/player/volume" and method == "PUT":
            writes.append((path, params["volume_percent"]))
            device["volume_percent"] = int(params["volume_percent"])
        return {}

    music = MusicController(
        token="test-token", device_id="laptop", request=peer, ducking_enabled=True
    )
    lowered = await music.begin_duck()
    assert lowered == {"outcome": "confirmed", "volume_percent": 28}
    assert device["volume_percent"] == 28 and state["is_playing"] is True
    restored = await music.restore_duck()
    assert restored == {"outcome": "restored", "volume_percent": 80}
    assert writes == [("/me/player/volume", "28"), ("/me/player/volume", "80")]
    assert state["is_playing"] is True
    await music.close()


async def test_duck_waits_for_explicit_volume_command_and_restores_its_new_level():
    device = {
        "id": "laptop",
        "name": "Swift browser",
        "supports_volume": True,
        "volume_percent": 80,
    }
    state = {"device": device, "is_playing": True}
    volume_write_started, release_volume_write = asyncio.Event(), asyncio.Event()
    writes = []

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/me/player":
            return state
        if path == "/me/player/volume" and method == "PUT":
            value = int(params["volume_percent"])
            writes.append(value)
            if value == 90:
                volume_write_started.set()
                await release_volume_write.wait()
            device["volume_percent"] = value
        return {}

    music = MusicController(
        token="test-token", device_id="laptop", request=peer, ducking_enabled=True
    )
    explicit = asyncio.create_task(music.execute("volume-up", "volume_up"))
    await asyncio.wait_for(volume_write_started.wait(), 1)
    duck = asyncio.create_task(music.begin_duck())
    await asyncio.sleep(0)
    assert not duck.done()
    release_volume_write.set()
    await explicit
    assert await duck == {"outcome": "confirmed", "volume_percent": 31}
    assert device["volume_percent"] == 31 and music.duck_state["original_volume"] == 90
    assert await music.restore_duck() == {"outcome": "restored", "volume_percent": 90}
    assert writes == [90, 31, 90]
    await music.close()


async def test_music_ducking_preserves_external_level_change_and_explicit_volume_intent():
    device = {
        "id": "laptop",
        "name": "Swift browser",
        "supports_volume": True,
        "volume_percent": 80,
    }
    state = {"device": device, "is_playing": True}
    writes = []

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/me/player":
            return state
        if path == "/me/player/volume" and method == "PUT":
            writes.append(int(params["volume_percent"]))
            device["volume_percent"] = int(params["volume_percent"])
        return {}

    music = MusicController(
        token="test-token", device_id="laptop", request=peer, ducking_enabled=True
    )
    await music.begin_duck()
    device["volume_percent"] = 42
    assert (await music.restore_duck())["outcome"] == "preserved_external_change"
    assert device["volume_percent"] == 42
    await music.begin_duck()
    assert device["volume_percent"] == 15
    await music.execute("explicit-up", "volume_up")
    assert device["volume_percent"] == 52
    assert (await music.restore_duck())["outcome"] == "not_ducked"
    state["is_playing"] = False
    await music.begin_duck()
    assert device["volume_percent"] == 52
    assert writes[-1] == 52
    await music.close()


async def test_spoken_volume_down_and_absolute_level_persist_after_ducking():
    device = {
        "id": "laptop",
        "name": "Swift browser",
        "supports_volume": True,
        "volume_percent": 80,
    }
    state = {"device": device, "is_playing": True}
    writes = []

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/me/player":
            return state
        if path == "/me/player/volume" and method == "PUT":
            value = int(params["volume_percent"])
            writes.append(value)
            device["volume_percent"] = value
        return {}

    music = MusicController(
        token="test-token", device_id="laptop", request=peer, ducking_enabled=True
    )
    try:
        assert (await music.begin_duck())["volume_percent"] == 28
        assert (await music.execute("voice-volume-down", "volume_down"))["display"] == "70%"
        assert await music.restore_duck() == {"outcome": "not_ducked"}
        assert device["volume_percent"] == 70

        assert (await music.begin_duck())["volume_percent"] == 24
        assert (await music.execute("voice-volume-set", "volume_set", "20"))["display"] == "20%"
        assert await music.restore_duck() == {"outcome": "not_ducked"}
        assert device["volume_percent"] == 20
        assert writes == [28, 70, 24, 20]
    finally:
        await music.close()


async def test_uncertain_volume_write_does_not_restore_over_a_new_user_level():
    from me3_voice.music import MusicUnavailable

    device = {
        "id": "laptop",
        "name": "Swift browser",
        "supports_volume": True,
        "volume_percent": 80,
    }
    state = {"device": device, "is_playing": True}

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/me/player":
            return state
        if path == "/me/player/volume" and method == "PUT":
            device["volume_percent"] = int(params["volume_percent"])
            if device["volume_percent"] == 20:
                raise MusicUnavailable(
                    "Spotify connection failed; playback state is uncertain",
                    code="music_transport_failed",
                )
        return {}

    music = MusicController(
        token="test-token", device_id="laptop", request=peer, ducking_enabled=True
    )
    try:
        await music.begin_duck()
        with pytest.raises(MusicUnavailable) as error:
            await music.execute("uncertain-volume", "volume_set", "20")
        assert error.value.outcome_uncertain is True
        assert await music.restore_duck() == {"outcome": "preserved_external_change"}
        assert device["volume_percent"] == 20
    finally:
        await music.close()


async def test_music_ducking_and_volume_fail_closed_when_player_cannot_control_volume():
    device = {
        "id": "laptop",
        "name": "Restricted volume",
        "supports_volume": False,
        "volume_percent": 50,
    }

    async def peer(method, path, **kwargs):
        if path == "/me/player/devices":
            return {"devices": [{**device, "is_active": True, "is_restricted": False}]}
        if path == "/me/player":
            return {"device": device, "is_playing": True}
        raise AssertionError("volume mutation must not be sent")

    music = MusicController(
        token="test-token", device_id="laptop", request=peer, ducking_enabled=True
    )
    assert (await music.begin_duck()) == {
        "outcome": "unsupported",
        "reason": "volume_control_unavailable",
    }
    with pytest.raises(MusicUnavailable) as error:
        await music.execute("set", "volume_set", "25")
    assert error.value.code == "music_volume_unsupported"
    await music.close()


async def test_maroon_file_correction_is_exact_and_keeps_raw_query_identity():
    searched = []

    async def peer(method, path, *, params=None, body=None):
        if path == "/me/player/devices":
            return {"devices": [player("laptop", True)]}
        if path == "/search":
            searched.append(params["q"])
            return {"tracks": {"items": [{"uri": "spotify:track:one", "name": "Example"}]}}
        if path == "/me/player":
            return {}
        return {}

    music = MusicController(token="test-token", device_id="laptop", request=peer)
    result = await music.execute("spoken-maroon", "play", "Maroon file")
    assert searched == ["Maroon 5"]
    assert result["query_normalization"] == "maroon_file_to_maroon_5"
    assert "Maroon file’ as ‘Maroon 5" in result["summary"]
    await music.execute("literal-file", "play", "Maroon file soundtrack")
    assert searched[-1] == "Maroon file soundtrack"
    await music.close()
