"""Project-local music control. Mock by default; Spotify calls require explicit config."""

import asyncio
import getpass
import json
import os
import re
import tempfile
import time
from pathlib import Path

import aiohttp

from .credentials import load_secrets, private_path, valid_key, write_secrets

ROOT = Path(__file__).resolve().parents[2]
BASE_SCOPES = {"user-modify-playback-state", "user-read-playback-state"}
WEB_PLAYBACK_SCOPE = "streaming"
WEB_PLAYBACK_SCOPES = {WEB_PLAYBACK_SCOPE, "user-read-email", "user-read-private"}
OUTPUT_BACKENDS = {"spotify_connect", "web_playback_sdk"}


def save_tokens(path, access, refresh=None, *, replace=False):
    """Keep Spotify tokens private; explicit renewal/rotation preserves other keys."""
    path = private_path(path)
    updates = {"ME3_SPOTIFY_ACCESS_TOKEN": access}
    if refresh is not None:
        updates["ME3_SPOTIFY_REFRESH_TOKEN"] = refresh
    if not all(valid_key(value) for value in updates.values()):
        raise ValueError("Invalid Spotify credentials; values excluded")
    if not replace:
        write_secrets(path, updates)
        return
    values = load_secrets(path)
    values.update(updates)
    descriptor, temporary = tempfile.mkstemp(prefix=".spotify-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for name, value in values.items():
                stream.write(name + "=" + value + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class MusicUnavailable(RuntimeError):
    def __init__(self, message, *, code="music_unavailable", outcome_uncertain=False):
        super().__init__(message)
        self.code = code
        self.outcome_uncertain = outcome_uncertain


async def read_spotify_body(response, limit):
    """Collect through EOF; StreamReader.read(n) may return only one network fragment."""
    body = bytearray()
    while len(body) <= limit:
        chunk = await response.content.read(min(8192, limit + 1 - len(body)))
        if not chunk:
            return bytes(body)
        body.extend(chunk)
    raise MusicUnavailable("Spotify response exceeded its size limit; values excluded")


class MusicController:
    def __init__(
        self,
        *,
        token=None,
        device_id=None,
        request=None,
        client_id=None,
        refresh_token=None,
        secrets_path=None,
        scopes=(),
        output_backend="spotify_connect",
        device_name=None,
        ducking_enabled=False,
        ducking_ratio=0.35,
    ):
        if device_id is not None and (
            not isinstance(device_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", device_id)
        ):
            raise ValueError("invalid Spotify device ID")
        if client_id is not None and (
            not isinstance(client_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", client_id)
        ):
            raise ValueError("invalid Spotify OAuth client ID")
        self.token, self.device_id = token, device_id
        if output_backend not in OUTPUT_BACKENDS:
            raise ValueError("invalid Spotify output backend")
        if device_name is not None and (
            not isinstance(device_name, str)
            or not 1 <= len(device_name) <= 80
            or not device_name.isprintable()
        ):
            raise ValueError("invalid Spotify device name")
        self.output_backend = output_backend
        self.device_name = device_name
        self.scopes = frozenset(s for s in scopes if isinstance(s, str))
        self.browser_device_id = None
        self.request_peer = request
        self.client = None
        self.revision = 0
        self.operations = {}
        self.lock = asyncio.Lock()
        self.current = ""
        self.playing = False
        self.client_id, self.refresh_token, self.secrets_path = (
            client_id,
            refresh_token,
            secrets_path,
        )
        self.refresh_lock = asyncio.Lock()
        self.state_cache = None
        self.state_cache_at = 0.0
        self.browser_token_until = 0.0
        if (
            type(ducking_enabled) is not bool
            or type(ducking_ratio) not in (int, float)
            or not 0.05 <= ducking_ratio <= 0.9
        ):
            raise ValueError("invalid Spotify ducking configuration")
        self.ducking_enabled = ducking_enabled
        self.ducking_ratio = float(ducking_ratio)
        self.duck_lock = asyncio.Lock()
        self.duck_state = None
        self.mock_volume = 50

    @classmethod
    def configured(cls, config_path=None):
        name = config_path or os.environ.get("ME3_MUSIC_CONFIG")
        if not name:
            return cls()
        path = Path(name).resolve()
        if not path.is_relative_to(ROOT / "var/cache"):
            raise ValueError("ME3_MUSIC_CONFIG must be inside ME3 var/cache")
        config = json.loads(path.read_text())
        if config.get("backend") != "spotify" or config.get("allow_live") is not True:
            raise ValueError("Spotify config requires backend=spotify and allow_live=true")
        ducking = config.get("ducking", {})
        if (
            not isinstance(ducking, dict)
            or set(ducking) - {"enabled", "ratio"}
            or type(ducking.get("enabled", False)) is not bool
            or type(ducking.get("ratio", 0.35)) not in (int, float)
            or not 0.05 <= ducking.get("ratio", 0.35) <= 0.9
        ):
            raise ValueError("invalid Spotify ducking settings")
        output_backend = config.get("output_backend", "spotify_connect")
        scopes = config.get("scopes", [])
        if (
            not isinstance(scopes, list)
            or len(scopes) > 8
            or not all(
                isinstance(scope, str) and scope in BASE_SCOPES | WEB_PLAYBACK_SCOPES
                for scope in scopes
            )
        ):
            scopes = []
        secret_path = ROOT / config["secrets_file"]
        secrets = load_secrets(
            secret_path, {"ME3_SPOTIFY_ACCESS_TOKEN", "ME3_SPOTIFY_REFRESH_TOKEN"}
        )
        token = secrets.get("ME3_SPOTIFY_ACCESS_TOKEN")
        if not token:
            raise MusicUnavailable("Spotify token missing; run python -m me3_voice.music configure")
        return cls(
            token=token,
            device_id=config.get("device_id"),
            client_id=config.get("client_id"),
            refresh_token=secrets.get("ME3_SPOTIFY_REFRESH_TOKEN"),
            secrets_path=secret_path,
            scopes=scopes,
            output_backend=output_backend,
            device_name=config.get("device_name"),
            ducking_enabled=ducking.get("enabled", False),
            ducking_ratio=ducking.get("ratio", 0.35),
        )

    @property
    def browser_sdk_authorized(self):
        return bool(
            self.token
            and self.output_backend == "web_playback_sdk"
            and WEB_PLAYBACK_SCOPES.issubset(self.scopes)
        )

    def register_browser_device(self, device_id):
        if not self.browser_sdk_authorized:
            raise MusicUnavailable(
                "Spotify browser playback permissions are incomplete; renew Spotify access with --browser-player",
                code="music_streaming_scope_required",
            )
        if not isinstance(device_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", device_id):
            raise ValueError("invalid Spotify browser player ID")
        self.browser_device_id = device_id

    def unregister_browser_device(self, device_id=None):
        if device_id is None or self.browser_device_id == device_id:
            self.browser_device_id = None

    async def browser_access_token(self):
        """Return a short-lived access token only to the same-origin live UI."""
        if not self.browser_sdk_authorized:
            raise MusicUnavailable(
                "Spotify browser playback needs streaming and profile permissions; run music authorize --renew --browser-player",
                code="music_streaming_scope_required",
            )
        # Refresh through OAuth, independently of transient Connect device discovery.
        # Reuse the validated token for 55 minutes; refresh credentials stay server-side.
        if self.refresh_token and self.client_id:
            if time.monotonic() >= self.browser_token_until:
                if self.client is None:
                    self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))
                await self.renew_token(self.token)
        else:
            await self.devices()
        return self.token

    def capabilities(self):
        configured = bool(self.token)
        authorized = self.browser_sdk_authorized
        return {
            "configured": configured,
            "output_backend": self.output_backend if configured else "unavailable",
            "browser_player": authorized,
            "browser_player_connected": bool(self.browser_device_id),
            "ducking_enabled": self.ducking_enabled,
            "setup_required": configured
            and self.output_backend == "web_playback_sdk"
            and not authorized,
            "device_name": self.device_name if configured else None,
            "target_configured": bool(configured and (self.device_name or self.device_id)),
        }

    async def current_state(self, *, force=False):
        """Return a normalized current Spotify state, never an audibility claim."""
        import time

        if not self.token:
            return {
                "configured": False,
                "status": "unavailable",
                "output_backend": "mock",
                "audible": None,
            }
        now = time.monotonic()
        if not force and self.state_cache is not None and now - self.state_cache_at < 1.5:
            return dict(self.state_cache)
        value = await self.request("GET", "/me/player")
        if not value:
            state = {
                "configured": True,
                "status": "not_playing",
                "output_backend": self.output_backend,
                "device_name": None,
                "track": None,
                "progress_ms": None,
                "duration_ms": None,
                "volume_percent": None,
                "supports_volume": None,
                "audible": None,
                "reported": True,
            }
        else:
            if not isinstance(value, dict):
                raise MusicUnavailable("Spotify returned an invalid playback state")
            item = value.get("item") if isinstance(value, dict) else None
            device = value.get("device") if isinstance(value, dict) else None
            artists = item.get("artists", []) if isinstance(item, dict) else []
            if not isinstance(artists, list):
                artists = []
            album = item.get("album", {}) if isinstance(item, dict) else {}
            images = album.get("images", []) if isinstance(album, dict) else []
            if not isinstance(images, list):
                images = []
            artwork = next(
                (
                    image.get("url")
                    for image in images[:5]
                    if isinstance(image, dict)
                    and isinstance(image.get("url"), str)
                    and image["url"].startswith("https://i.scdn.co/")
                    and len(image["url"]) <= 500
                ),
                None,
            )
            track = None
            if isinstance(item, dict):
                name = item.get("name")
                track = {
                    "name": name[:160] if isinstance(name, str) else "Unknown track",
                    "artists": ", ".join(
                        artist["name"][:80]
                        for artist in artists[:5]
                        if isinstance(artist, dict) and isinstance(artist.get("name"), str)
                    )[:240],
                    "artwork": artwork,
                }
            state = {
                "configured": True,
                "status": "playing"
                if value.get("is_playing") is True
                else "paused"
                if value.get("is_playing") is False and track
                else "not_playing",
                "output_backend": self.output_backend,
                "device_name": (
                    device.get("name")[:80]
                    if isinstance(device, dict) and isinstance(device.get("name"), str)
                    else None
                ),
                "track": track,
                "progress_ms": value.get("progress_ms")
                if type(value.get("progress_ms")) is int and 0 <= value["progress_ms"] <= 86_400_000
                else None,
                "duration_ms": item.get("duration_ms")
                if isinstance(item, dict)
                and type(item.get("duration_ms")) is int
                and 0 < item["duration_ms"] <= 86_400_000
                else None,
                "volume_percent": device.get("volume_percent")
                if isinstance(device, dict)
                and type(device.get("volume_percent")) is int
                and 0 <= device["volume_percent"] <= 100
                else None,
                "supports_volume": device.get("supports_volume")
                if isinstance(device, dict) and type(device.get("supports_volume")) is bool
                else None,
                "audible": None,
                "reported": True,
            }
        device = value.get("device") if isinstance(value, dict) else None
        identifier = device.get("id") if isinstance(device, dict) else None
        state["target_matched"] = bool(
            identifier
            and (
                identifier == self.browser_device_id
                if self.output_backend == "web_playback_sdk"
                else identifier == self.device_id
                if self.device_id
                else state.get("device_name") == self.device_name
                if self.device_name
                else False
            )
        )
        restrictions = (
            (value.get("actions") or {}).get("disallows") if isinstance(value, dict) else None
        )
        state["disallows"] = {
            key: restrictions[key]
            for key in ("skipping_next", "skipping_prev", "seeking")
            if isinstance(restrictions, dict) and type(restrictions.get(key)) is bool
        }
        self.state_cache, self.state_cache_at = state, now
        return dict(state)

    async def renew_token(self, rejected):
        async with self.refresh_lock:
            if rejected != self.token:
                return
            if not self.refresh_token or not self.client_id:
                raise MusicUnavailable(
                    "Spotify token expired; run music authorize --renew in the secured terminal",
                    code="music_auth_expired",
                )
            async with self.client.request(
                "POST",
                "https://accounts.spotify.com/api/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise MusicUnavailable(
                        "Spotify session renewal failed; authorize again in the secured terminal",
                        code="music_auth_expired",
                    )
                raw = await read_spotify_body(response, 16384)
                if len(raw) > 16384:
                    raise MusicUnavailable("Spotify token response exceeded its limit")
                value = json.loads(raw)
                access = value.get("access_token")
                refresh = value.get("refresh_token", self.refresh_token)
                if (
                    value.get("token_type", "").lower() != "bearer"
                    or not valid_key(access)
                    or not valid_key(refresh)
                ):
                    raise MusicUnavailable(
                        "Spotify returned invalid session credentials; values excluded"
                    )
                if self.secrets_path:
                    save_tokens(self.secrets_path, access, refresh, replace=True)
                self.token, self.refresh_token = access, refresh
                self.browser_token_until = time.monotonic() + 3300

    async def request(self, method, path, *, params=None, body=None, _renewed=False):
        if self.request_peer:
            try:
                return await self.request_peer(method, path, params=params, body=body)
            except (aiohttp.ClientError, TimeoutError):
                raise MusicUnavailable(
                    "Spotify connection failed; playback state is uncertain",
                    code="music_transport_failed",
                ) from None
        headers = {"Authorization": "Bearer " + self.token}
        try:
            if self.client is None:
                self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))
            async with self.client.request(
                method,
                "https://api.spotify.com/v1" + path,
                headers=headers,
                params=params,
                json=body,
                allow_redirects=False,
            ) as response:
                if response.status == 401:
                    if not _renewed:
                        await self.renew_token(headers["Authorization"][7:])
                        return await self.request(
                            method, path, params=params, body=body, _renewed=True
                        )
                    raise MusicUnavailable(
                        "Spotify authentication rejected after renewal; authorize again",
                        code="music_auth_expired",
                    )
                if response.status == 403:
                    raise MusicUnavailable(
                        "Spotify denied playback; check Premium and token scopes",
                        code="music_forbidden",
                    )
                if response.status == 404:
                    raise MusicUnavailable(
                        "Spotify player is unavailable; start the configured Spotify receiver"
                        if path
                        in {
                            "/me/player/play",
                            "/me/player/pause",
                            "/me/player/next",
                            "/me/player/previous",
                            "/me/player/seek",
                        }
                        else "Spotify resource is unavailable",
                        code="music_device_unavailable"
                        if path
                        in {
                            "/me/player/play",
                            "/me/player/pause",
                            "/me/player/next",
                            "/me/player/previous",
                            "/me/player/seek",
                        }
                        else "music_resource_unavailable",
                    )
                if response.status == 429:
                    raise MusicUnavailable(
                        "Spotify rate limit reached; no automatic retry was made",
                        code="music_rate_limited",
                    )
                if response.status >= 500:
                    raise MusicUnavailable(
                        "Spotify returned a server error; playback state is uncertain",
                        code="music_server_error",
                    )
                if response.status >= 300:
                    raise MusicUnavailable(
                        "Spotify rejected the request",
                        code="music_request_rejected",
                    )
                if response.status == 204:
                    return {}
                raw = await read_spotify_body(response, 65536)
                if len(raw) > 65536:
                    raise MusicUnavailable("Spotify response exceeded its limit")
                return json.loads(raw)
        except (aiohttp.ClientError, TimeoutError, ValueError):
            raise MusicUnavailable(
                "Spotify connection failed; playback state is uncertain",
                code="music_transport_failed",
            ) from None

    async def devices(self):
        """Read-only discovery. Does not launch a receiver or change playback."""
        if not self.token:
            raise MusicUnavailable("Spotify is not configured; music is simulated")
        response = await self.request("GET", "/me/player/devices")
        rows = response.get("devices") if isinstance(response, dict) else None
        if not isinstance(rows, list) or len(rows) > 50:
            raise MusicUnavailable("Spotify returned an invalid device list")
        result = []
        for row in rows:
            if not isinstance(row, dict):
                raise MusicUnavailable("Spotify returned an invalid device list")
            identifier = row.get("id")
            if identifier is None:
                continue
            if (
                not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier)
                or type(row.get("is_active")) is not bool
                or type(row.get("is_restricted")) is not bool
            ):
                raise MusicUnavailable("Spotify returned an invalid device list")
            name = row.get("name", "Spotify player")
            result.append(
                {
                    "id": identifier,
                    "name": name[:80]
                    if isinstance(name, str) and name.isprintable()
                    else "Spotify player",
                    "is_active": row["is_active"],
                    "is_restricted": row["is_restricted"],
                }
            )
        if len({row["id"] for row in result}) != len(result):
            raise MusicUnavailable("Spotify returned duplicate device IDs")
        return result

    async def playback_target(self):
        return self.choose_target(await self.devices())

    def choose_target(self, devices):
        rows = [row for row in devices if not row["is_restricted"]]
        if self.output_backend == "web_playback_sdk":
            if self.browser_device_id and any(row["id"] == self.browser_device_id for row in rows):
                return self.browser_device_id
            raise MusicUnavailable(
                "Connect Spotify in Swift before asking it to play; no other device was selected",
                code="music_browser_player_not_connected",
            )
        if self.device_name:
            named = [row for row in rows if row["name"] == self.device_name]
            if len(named) == 1:
                return named[0]["id"]
            raise MusicUnavailable(
                "Configured Spotify receiver is unavailable or ambiguous; check its device name",
                code="music_device_unavailable" if not named else "music_device_ambiguous",
            )
        if self.device_id:
            if any(row["id"] == self.device_id for row in rows):
                return self.device_id
            raise MusicUnavailable(
                "Configured Spotify player is unavailable; start that receiver",
                code="music_device_unavailable",
            )
        active = [row for row in rows if row["is_active"]]
        if len(active) == 1:
            return active[0]["id"]
        if len(rows) == 1:
            return rows[0]["id"]
        if not rows:
            raise MusicUnavailable(
                "No available Spotify player; start a Spotify receiver. App authorization does not create a player.",
                code="music_device_unavailable",
            )
        raise MusicUnavailable(
            "Multiple Spotify players available; configure a device ID to choose the intended speaker",
            code="music_device_ambiguous",
        )

    async def execute(self, operation, action, query="", market=None):
        if action not in {
            "play",
            "play_playlist",
            "pause",
            "resume",
            "stop",
            "next",
            "previous",
            "seek",
            "volume_up",
            "volume_down",
            "volume_set",
        } or not isinstance(query, str):
            raise ValueError("invalid music operation")
        query = query.strip()
        raw_query = query
        if market is not None and (
            action != "play_playlist"
            or not isinstance(market, str)
            or not re.fullmatch(r"[A-Z]{2}", market)
        ):
            raise ValueError(
                "music market must be an uppercase ISO country code for playlist search"
            )
        normalization = None
        if action == "play" and re.fullmatch(r"maroon\s+file", query, flags=re.IGNORECASE):
            query = "Maroon 5"
            normalization = "maroon_file_to_maroon_5"
        if action in {"play", "play_playlist"} and not 1 <= len(query) <= 120:
            raise ValueError("music search needs 1–120 characters")
        if action == "seek" and (not query.isdecimal() or int(query) > 7_200_000):
            raise ValueError("seek position must be between 0 and 7200000 ms")
        if action == "volume_set" and (not query.isdecimal() or not 0 <= int(query) <= 100):
            raise ValueError("volume must be an integer from 0 to 100")
        if action not in {"play", "play_playlist", "seek", "volume_set"} and query:
            raise ValueError("this music action does not accept a query")
        args = (action, raw_query, market)
        if operation in self.operations:
            previous, task = self.operations[operation]
            if previous != args:
                raise ValueError("music operation identity reused")
            return await asyncio.shield(task)
        if len(self.operations) >= 128:
            raise MusicUnavailable("Music operation ledger is full; restart the bridge")
        self.revision += 1
        revision = self.revision
        task = asyncio.create_task(
            self.apply(action, query, revision, normalization=normalization, market=market)
        )
        self.operations[operation] = args, task
        return await asyncio.shield(task)

    async def apply(self, action, query, revision, *, normalization=None, market=None):
        track = None
        playlist = None
        queue = []
        target = await self.playback_target() if self.token else None
        if revision != self.revision:
            raise MusicUnavailable("Music request superseded by a newer request")
        if self.token and action in {"play", "play_playlist"}:
            params = {
                "q": query,
                "type": "playlist" if action == "play_playlist" else "track",
                "limit": "5",
            }
            if market:
                params["market"] = market
            found = await self.request("GET", "/search", params=params)
            item_type = "playlists" if action == "play_playlist" else "tracks"
            items = found.get(item_type, {}).get("items", [])
            if not items:
                raise MusicUnavailable(
                    "No matching Spotify playlist was found; ask for a specific artist or track."
                    if action == "play_playlist"
                    else "No matching Spotify track was found"
                )
            if action == "play_playlist":
                playlist = items[0]
                if (
                    not isinstance(playlist, dict)
                    or not isinstance(playlist.get("uri"), str)
                    or not playlist["uri"].startswith("spotify:playlist:")
                    or not isinstance(playlist.get("name"), str)
                ):
                    raise MusicUnavailable("Spotify returned an invalid playlist result")
            else:
                track = items[0]
                if not isinstance(track.get("uri"), str) or not track["uri"].startswith(
                    "spotify:track:"
                ):
                    raise MusicUnavailable("Spotify returned an invalid track")
                # Keep navigation within the selected primary artist and bound the queue.
                artists = track.get("artists") or []
                primary = artists[0].get("id") if artists and isinstance(artists[0], dict) else None
                queue = [track["uri"]]
                for candidate in items[1:5]:
                    if not isinstance(candidate, dict):
                        continue
                    uri = candidate.get("uri")
                    candidate_artists = candidate.get("artists") or []
                    same_artist = (
                        primary
                        and candidate_artists
                        and isinstance(candidate_artists[0], dict)
                        and candidate_artists[0].get("id") == primary
                    )
                    if (
                        same_artist
                        and isinstance(uri, str)
                        and uri.startswith("spotify:track:")
                        and uri not in queue
                    ):
                        queue.append(uri)
        async with self.lock:
            if revision != self.revision:
                raise MusicUnavailable("Music request superseded by a newer request")
            if self.token:
                if action in {"volume_up", "volume_down", "volume_set"}:
                    return await self._apply_volume(action, query, target)
                try:
                    await self._send_playback_action(
                        "play" if action == "play_playlist" else action,
                        target,
                        track,
                        queue=queue,
                        context_uri=playlist.get("uri") if playlist else None,
                        position_ms=int(query) if action == "seek" else None,
                    )
                except MusicUnavailable as exc:
                    if exc.code in {"music_transport_failed", "music_server_error"}:
                        if action in {"pause", "stop"}:
                            observed = await self._confirm_paused_after_uncertain_control(
                                action, target
                            )
                            if observed is not None:
                                return observed
                        raise MusicUnavailable(
                            "Spotify could not confirm whether this control took effect. Check current playback before retrying.",
                            code="music_outcome_uncertain",
                            outcome_uncertain=True,
                        ) from None
                    raise
                # A successful PUT is an acknowledgment, not evidence of audible playback.
                self.current = str(track.get("name", query))[:120] if track else self.current
                summary = (
                    f"Spotify accepted playlist request: {playlist['name']}; playback is unverified."
                    if playlist
                    else f"Spotify accepted {action}: {self.current or 'current track'}."
                )
                display = "playlist" if playlist else action
                try:
                    async with asyncio.timeout(1):
                        status = await self.request("GET", "/me/player")
                    item = status.get("item") or {}
                    expected_context = status.get("context") or {}
                    matches = (
                        expected_context.get("uri") == playlist.get("uri")
                        if playlist
                        else not track or item.get("uri") == track["uri"]
                    ) and (status.get("device") or {}).get("id") == target
                    if matches and isinstance(item, dict) and isinstance(item.get("name"), str):
                        self.current = item["name"][:120]
                    if matches and status.get("is_playing") is True and action != "pause":
                        display = "playing"
                        summary = (
                            f"Spotify reports the {playlist['name']} playlist is active; now playing {self.current}."
                            if playlist
                            else f"Spotify reports playing {self.current or 'the current track'}."
                        )
                    elif (
                        matches
                        and status.get("is_playing") is False
                        and action in {"pause", "stop"}
                    ):
                        display = "paused"
                        summary = (
                            "Spotify reports playback paused."
                            if action == "pause"
                            else "Spotify playback is paused; Spotify does not expose a separate stop action here."
                        )
                except (MusicUnavailable, TimeoutError):
                    pass  # The mutation may have succeeded: retain its receipt, never retry it.
                self.state_cache = None
                if normalization:
                    summary = "Interpreted ‘Maroon file’ as ‘Maroon 5’. " + summary
                result = {"mode": "spotify", "summary": summary[:200], "display": display}
                if normalization:
                    result["query_normalization"] = normalization
                return result
            self.current = query if action in {"play", "play_playlist"} else self.current
            self.playing = (
                False
                if action in {"pause", "stop"}
                else True
                if action in {"play", "play_playlist", "resume"}
                else self.playing
            )
            if action in {"volume_up", "volume_down", "volume_set"}:
                current_volume = self.mock_volume
                desired = (
                    int(query)
                    if action == "volume_set"
                    else min(100, current_volume + 10)
                    if action == "volume_up"
                    else max(0, current_volume - 10)
                )
                self.mock_volume = desired
                return {
                    "mode": "mock",
                    "verified_in_memory": True,
                    "summary": f"Simulated music volume set to {desired}%; no audio is changed.",
                    "display": f"{desired}%",
                }
            return {
                "mode": "mock",
                "verified_in_memory": True,
                "summary": (
                    f"Simulated playlist search for {query}. No audio is played."
                    if action == "play_playlist"
                    else "Simulated music search for Maroon 5 (normalized from ‘Maroon file’). No audio is played."
                    if normalization
                    else f"Simulated music {action}: {self.current or 'current track'}. No audio is played."
                )[:200],
                "display": "playlist" if action == "play_playlist" else action,
                **({"query_normalization": normalization} if normalization else {}),
            }

    async def _confirm_paused_after_uncertain_control(self, action, target):
        """Use one bounded read to reconcile pause/stop without repeating the write."""
        self.state_cache = None
        try:
            async with asyncio.timeout(1):
                status = await self.request("GET", "/me/player")
        except (MusicUnavailable, TimeoutError):
            return None
        device = status.get("device") if isinstance(status, dict) else None
        if (
            not isinstance(device, dict)
            or device.get("id") != target
            or status.get("is_playing") is not False
        ):
            return None
        item = status.get("item")
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            self.current = item["name"][:120]
        summary = (
            "Spotify reports playback paused after the stop request; Spotify does not expose a separate stop action here."
            if action == "stop"
            else "Spotify reports playback paused after the pause request."
        )
        self.state_cache = None
        return {"mode": "spotify", "summary": summary[:200], "display": "paused"}

    async def _send_playback_action(
        self, action, target, track=None, position_ms=None, queue=None, context_uri=None
    ):
        endpoint = {
            "play": ("PUT", "/me/player/play"),
            "resume": ("PUT", "/me/player/play"),
            "pause": ("PUT", "/me/player/pause"),
            "stop": ("PUT", "/me/player/pause"),
            "next": ("POST", "/me/player/next"),
            "previous": ("POST", "/me/player/previous"),
            "seek": ("PUT", "/me/player/seek"),
        }[action]
        params = {"device_id": target} if target else {}
        if action == "seek":
            params["position_ms"] = str(position_ms)
        body = (
            {"context_uri": context_uri}
            if action == "play" and context_uri
            else {"uris": queue or [track["uri"]]}
            if action == "play" and track
            else None
        )
        await self.request(endpoint[0], endpoint[1], params=params, body=body)

    async def _apply_volume(self, action, query, target):
        state = await self.request("GET", "/me/player")
        device = state.get("device") if isinstance(state, dict) else None
        if not isinstance(device, dict) or device.get("id") != target:
            raise MusicUnavailable(
                "The selected Spotify player is not active; volume was not changed",
                code="music_device_unavailable",
            )
        if device.get("supports_volume") is not True:
            raise MusicUnavailable(
                "The active Spotify player does not report volume control support",
                code="music_volume_unsupported",
            )
        current = device.get("volume_percent")
        if type(current) is not int or not 0 <= current <= 100:
            raise MusicUnavailable(
                "Spotify did not report the active player's volume", code="music_volume_unavailable"
            )
        if action == "volume_set":
            desired = int(query)
        else:
            baseline = current
            if (
                self.duck_state
                and self.duck_state["device_id"] == target
                and current == self.duck_state["ducked_volume"]
            ):
                baseline = self.duck_state["original_volume"]
            desired = min(100, baseline + 10) if action == "volume_up" else max(0, baseline - 10)
        try:
            await self.request(
                "PUT",
                "/me/player/volume",
                params={"device_id": target, "volume_percent": str(desired)},
            )
        except MusicUnavailable as exc:
            # Preserve the duck snapshot until a volume update is accepted.
            # On a transport/server error the write may have succeeded: the later
            # restore read compares the current level before changing anything.
            if exc.code in {"music_transport_failed", "music_server_error"}:
                exc.outcome_uncertain = True
            raise
        self.duck_state = None  # The accepted explicit command owns the new level.
        self.state_cache = None
        observed = await self.request("GET", "/me/player")
        observed_device = observed.get("device") if isinstance(observed, dict) else None
        if (
            isinstance(observed_device, dict)
            and observed_device.get("id") == target
            and observed_device.get("volume_percent") == desired
        ):
            summary = f"Spotify reports volume at {desired}%."
        else:
            summary = f"Spotify accepted the volume request for {desired}%; the resulting level is unverified."
        return {"mode": "spotify", "summary": summary, "display": f"{desired}%"}

    async def begin_duck(self):
        """Lower only the owned, currently playing Spotify target, when enabled."""
        if not self.ducking_enabled or not self.token:
            return {"outcome": "disabled" if not self.ducking_enabled else "unavailable"}
        async with self.duck_lock:
            # Share the operation lock with playback and explicit volume
            # controls so a late duck/restore cannot overwrite a newer command.
            async with self.lock:
                target = await self.playback_target()
                state = await self.request("GET", "/me/player")
                device = state.get("device") if isinstance(state, dict) else None
                if not isinstance(device, dict) or device.get("id") != target:
                    return {"outcome": "unsupported", "reason": "target_not_active"}
                if state.get("is_playing") is not True:
                    return {"outcome": "not_playing"}
                if device.get("supports_volume") is not True:
                    return {"outcome": "unsupported", "reason": "volume_control_unavailable"}
                original = device.get("volume_percent")
                if type(original) is not int or not 0 <= original <= 100:
                    return {"outcome": "unsupported", "reason": "volume_unavailable"}
                if self.duck_state and self.duck_state["device_id"] == target:
                    if original == self.duck_state["ducked_volume"]:
                        return {"outcome": "already_ducked"}
                    # An external/user volume change supersedes this controller's restore.
                    self.duck_state = None
                ducked = max(0, min(original - 1, round(original * self.ducking_ratio)))
                if ducked == original:
                    return {"outcome": "unchanged"}
                self.duck_state = {
                    "device_id": target,
                    "original_volume": original,
                    "ducked_volume": ducked,
                }
                await self.request(
                    "PUT",
                    "/me/player/volume",
                    params={"device_id": target, "volume_percent": str(ducked)},
                )
                observed = await self.request("GET", "/me/player")
                observed_device = observed.get("device") if isinstance(observed, dict) else None
                if (
                    isinstance(observed_device, dict)
                    and observed_device.get("id") == target
                    and observed_device.get("volume_percent") == ducked
                ):
                    return {"outcome": "confirmed", "volume_percent": ducked}
                return {"outcome": "uncertain"}

    async def restore_duck(self):
        """Restore only if the same player still has the volume set by Swift."""
        async with self.duck_lock:
            async with self.lock:
                saved = self.duck_state
                if not saved:
                    return {"outcome": "not_ducked"}
                try:
                    state = await self.request("GET", "/me/player")
                except MusicUnavailable:
                    return {"outcome": "uncertain"}
                device = state.get("device") if isinstance(state, dict) else None
                if (
                    not isinstance(device, dict)
                    or device.get("id") != saved["device_id"]
                    or device.get("volume_percent") != saved["ducked_volume"]
                ):
                    self.duck_state = None
                    return {"outcome": "preserved_external_change"}
                try:
                    await self.request(
                        "PUT",
                        "/me/player/volume",
                        params={
                            "device_id": saved["device_id"],
                            "volume_percent": str(saved["original_volume"]),
                        },
                    )
                    confirmed = await self.request("GET", "/me/player")
                except MusicUnavailable:
                    return {"outcome": "uncertain"}
                actual = confirmed.get("device") if isinstance(confirmed, dict) else None
                if (
                    isinstance(actual, dict)
                    and actual.get("id") == saved["device_id"]
                    and actual.get("volume_percent") == saved["original_volume"]
                ):
                    self.duck_state = None
                    return {"outcome": "restored", "volume_percent": saved["original_volume"]}
                return {"outcome": "uncertain"}

    async def ui_control(self, operation, action, position_ms=None, volume_percent=None):
        if action not in {
            "pause",
            "resume",
            "stop",
            "next",
            "previous",
            "seek",
            "volume_up",
            "volume_down",
            "volume_set",
        }:
            raise ValueError("unsupported Spotify control")
        if action == "seek":
            if type(position_ms) is not int or not 0 <= position_ms <= 7_200_000:
                raise ValueError("invalid Spotify seek position")
            return await self.execute(operation, action, str(position_ms))
        if action == "volume_set":
            if type(volume_percent) is not int or not 0 <= volume_percent <= 100:
                raise ValueError("invalid Spotify volume")
            return await self.execute(operation, action, str(volume_percent))
        return await self.execute(operation, action)

    def cancel_pending(self):
        self.revision += 1

    async def close(self):
        # Drain bounded in-flight operations; a sent playback mutation must not be replayed.
        await asyncio.gather(*(t for _, t in self.operations.values()), return_exceptions=True)
        if self.client is not None:
            await self.client.close()


async def authorize(client_id, *, browser_player=False):
    """Interactive PKCE: paste the redirected address; no callback listener is opened."""
    import base64
    import hashlib
    import secrets
    from urllib.parse import parse_qs, urlencode, urlsplit

    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    state = secrets.token_urlsafe(24)
    redirect = "http://127.0.0.1:8876/callback"
    print("Open this URL in your laptop browser and approve Spotify access:")
    print(
        "https://accounts.spotify.com/authorize?"
        + urlencode(
            {
                "client_id": client_id,
                "response_type": "code",
                "redirect_uri": redirect,
                "scope": " ".join(
                    sorted(BASE_SCOPES | (WEB_PLAYBACK_SCOPES if browser_player else set()))
                ),
                "state": state,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
        )
    )
    print(
        "The callback may show an unreachable page. Copy its complete address from the address bar."
    )
    callback = getpass.getpass("Paste callback URL here (hidden): ").strip()
    if len(callback) >= 2 and callback[0] == callback[-1] and callback[0] in ("'", '"'):
        callback = callback[1:-1].strip()
    try:
        url = urlsplit(callback)
    except ValueError:
        raise MusicUnavailable(
            "Callback URL could not be parsed; copy the complete redirect address. No token request was sent and no credentials were saved"
        ) from None
    params = parse_qs(url.query)
    denied = params.get("error", [])
    if len(denied) == 1 and re.fullmatch(r"[a-z_]{1,40}", denied[0]):
        raise MusicUnavailable(
            f"Spotify authorization was declined ({denied[0]}); no credentials were saved"
        )
    if (
        (url.scheme, url.netloc, url.path) != ("http", "127.0.0.1:8876", "/callback")
        or params.get("state") != [state]
        or len(params.get("code", [])) != 1
    ):
        raise MusicUnavailable(
            "Authorization callback rejected; check state and registered redirect URI"
        )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as client:
        async with client.post(
            "https://accounts.spotify.com/api/token",
            data={
                "client_id": client_id,
                "grant_type": "authorization_code",
                "code": params["code"][0],
                "redirect_uri": redirect,
                "code_verifier": verifier,
            },
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raw = await read_spotify_body(response, 4096)
                code = None
                if len(raw) <= 4096:
                    try:
                        error = json.loads(raw).get("error")
                        if isinstance(error, str) and re.fullmatch(r"[a-z_]{1,40}", error):
                            code = error
                    except (json.JSONDecodeError, AttributeError):
                        pass
                detail = f", {code}" if code else ""
                hint = {
                    "invalid_grant": "the callback code may be expired or already used; start a fresh authorization and paste its complete callback URL",
                    "invalid_client": "check the Spotify app client ID and registered redirect URI",
                }.get(code, "check the app settings and exact registered redirect URI")
                raise MusicUnavailable(
                    f"Spotify rejected the authorization exchange (HTTP {response.status}{detail}); {hint}. No credentials were saved"
                )
            raw = await read_spotify_body(response, 16384)
            if len(raw) > 16384:
                raise MusicUnavailable("Authorization response exceeded its limit")
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                raise MusicUnavailable(
                    "Spotify token exchange returned HTTP 200 with unreadable JSON after the full response was collected; no credentials were saved. No response content was logged"
                ) from None
            if not isinstance(value, dict):
                raise MusicUnavailable(
                    "Spotify token exchange returned a non-object JSON response; no credentials were saved"
                )
            if value.get("token_type", "").lower() != "bearer":
                raise MusicUnavailable("Unsupported Spotify token type")
            if not valid_key(value.get("access_token")) or not valid_key(
                value.get("refresh_token")
            ):
                raise MusicUnavailable(
                    "Spotify returned invalid session credentials; values excluded"
                )
            return value


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["configure", "authorize", "inspect", "output"])
    parser.add_argument("--config", type=Path, default=ROOT / "var/cache/music.json")
    parser.add_argument("--device-id", default="")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--client-id", default="")
    parser.add_argument(
        "--browser-player",
        action="store_true",
        help="request Spotify's streaming scope and select the Swift browser as the output",
    )
    parser.add_argument(
        "--output-backend",
        choices=sorted(OUTPUT_BACKENDS),
        help="used with the output command to select the browser player or Spotify Connect receiver",
    )
    parser.add_argument(
        "--renew",
        action="store_true",
        help="explicitly replace existing Spotify credentials after fresh PKCE authorization",
    )
    args = parser.parse_args()
    config = args.config
    if args.command == "output":
        if (
            not config.is_file()
            or config.is_symlink()
            or not config.resolve().is_relative_to(ROOT / "var/cache")
        ):
            parser.error("--config must name an existing regular ME3 file under var/cache")
        if not args.output_backend:
            parser.error("output requires --output-backend")
        value = json.loads(config.read_text())
        if value.get("backend") != "spotify" or value.get("allow_live") is not True:
            parser.error("Spotify output selection requires an existing live Spotify config")
        value["output_backend"] = args.output_backend
        if args.output_backend == "web_playback_sdk":
            value.pop("device_id", None)
            value.pop("device_name", None)
        else:
            if args.device_name:
                if not args.device_name.isprintable() or not 1 <= len(args.device_name) <= 80:
                    parser.error("--device-name must contain 1..80 printable characters")
                value["device_name"] = args.device_name
                value.pop("device_id", None)
            elif args.device_id:
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.device_id):
                    parser.error("--device-id is invalid")
                value["device_id"] = args.device_id
                value.pop("device_name", None)
        config.write_text(json.dumps(value, indent=2) + "\n")
        print("Spotify output target saved in ME3 var/cache.")
        return
    if args.command == "inspect":

        async def inspect():
            controller = MusicController.configured(args.config)
            try:
                rows = await controller.devices()
                result = {"devices": rows, "playback_changed": False}
                try:
                    result.update(
                        ready_for_request=True, target_device_id=controller.choose_target(rows)
                    )
                except MusicUnavailable as exc:
                    result.update(ready_for_request=False, error_code=exc.code, reason=str(exc))
                print(json.dumps(result, indent=2))
                return result["ready_for_request"]
            finally:
                await controller.close()

        try:
            if not asyncio.run(inspect()):
                parser.exit(2)
        except MusicUnavailable as exc:
            parser.error(str(exc))
        except (ValueError, OSError):
            parser.error(
                "Spotify inspection failed; check configuration and connectivity. Credentials excluded."
            )
        return
    if not os.isatty(0):
        parser.error("Use an interactive terminal for masked token entry")
    secret = ROOT / "var/cache/spotify.env"
    config = ROOT / "var/cache/music.json"
    existing = config.exists() or secret.exists()
    if existing and not args.renew:
        parser.error("Existing Spotify config retained; use authorize --renew for secure renewal")
    if args.renew and (
        args.command != "authorize"
        or not config.is_file()
        or config.is_symlink()
        or not secret.is_file()
    ):
        parser.error(
            "--renew requires authorize and existing regular project Spotify config/secret files"
        )
    old_config = json.loads(config.read_text()) if args.renew else {}
    if args.renew:
        load_secrets(secret)  # Validate ownership/permissions before prompting or network.
    if args.renew and not args.client_id:
        args.client_id = old_config.get("client_id", "")
    if args.command == "authorize" and not args.client_id:
        parser.error(
            "--client-id is required; register http://127.0.0.1:8876/callback in the Spotify developer app"
        )
    try:
        result = (
            asyncio.run(authorize(args.client_id, browser_player=args.browser_player))
            if args.command == "authorize"
            else getpass.getpass("Spotify OAuth access token (hidden): ")
        )
    except MusicUnavailable as exc:
        parser.exit(2, f"error: {exc}\n")
    except (aiohttp.ClientError, TimeoutError):
        parser.exit(
            2,
            "error: Spotify authorization could not reach its token service; check HPC outbound connectivity. No credentials were saved.\n",
        )
    except (ValueError, KeyError):
        parser.exit(
            2,
            "error: Spotify returned an invalid authorization response; no credentials were saved. Restart authorization with a fresh URL.\n",
        )
    access = result["access_token"] if isinstance(result, dict) else result
    refresh = result.get("refresh_token") if isinstance(result, dict) else None
    granted_scopes = result.get("scope", "").split() if isinstance(result, dict) else []
    required_scopes = BASE_SCOPES | (WEB_PLAYBACK_SCOPES if args.browser_player else set())
    if args.command == "authorize" and not required_scopes.issubset(set(granted_scopes)):
        parser.exit(
            2,
            "error: Spotify did not grant the requested playback scopes; no credentials were saved. Start a fresh authorization.\n",
        )
    save_tokens(secret, access, refresh, replace=args.renew)
    old_backend = old_config.get("output_backend", "spotify_connect")
    output_backend = "web_playback_sdk" if args.browser_player else old_backend
    config.write_text(
        json.dumps(
            {
                "backend": "spotify",
                "allow_live": True,
                "device_id": None
                if args.browser_player
                else args.device_id or old_config.get("device_id"),
                "device_name": None
                if args.browser_player
                else args.device_name or old_config.get("device_name"),
                "output_backend": output_backend,
                "scopes": sorted(granted_scopes)
                if args.command == "authorize"
                else old_config.get("scopes", []),
                "client_id": args.client_id or old_config.get("client_id"),
                "secrets_file": "var/cache/spotify.env",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Configured under ME3 var/cache. Set ME3_MUSIC_CONFIG to var/cache/music.json before starting the bridge."
    )


if __name__ == "__main__":
    main()
