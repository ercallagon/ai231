"""Shared fixed music intents for mpv offline playback or local Spotify Connect."""

import argparse
import json
import os
from pathlib import Path
import re
import random
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

from hey_swift.actions.services.information import ActionResult
from hey_swift.integration.config import runtime_root

RESPONSES = {"PLAY_MUSIC": "Playing your music.", "PAUSE": "Your music is paused.",
             "STOP": "Your music has stopped.", "NEXT": "Playing the next track.",
             "VOLUME_UP": "Music volume increased.", "VOLUME_DOWN": "Music volume decreased."}
DEFAULT_SPOTIFY_URI = "spotify:playlist:37i9dQZEVXbNBz9cRCSFkY"


class MusicUnavailable(Exception):
    pass


class MusicStartUncertain(MusicUnavailable):
    """An online start may have succeeded and could not be paused safely."""


class MpvBackend:
    """Control an independently running mpv; state survives CLI invocations."""

    def __init__(self, tracks, endpoint, *, auto_start=False, max_volume_percent=None,
                 initial_volume_percent=None):
        self.tracks = [Path(p).resolve() for p in tracks]
        self.endpoint = endpoint
        self.auto_start = auto_start
        self.max_volume_percent = max_volume_percent
        self.initial_volume_percent = initial_volume_percent
        self._session_volume_initialized = False

    def is_running(self):
        try:
            self.command("get_property", "idle-active")
            return True
        except MusicUnavailable:
            return False

    def ensure_started(self):
        try:
            self.command("get_property", "idle-active")
            return
        except MusicUnavailable:
            if not self.auto_start or sys.platform == "win32":
                raise
        if not self.tracks or any(not p.is_file() for p in self.tracks):
            raise MusicUnavailable("Add a local music file before starting playback.")
        Path("logs").mkdir(exist_ok=True)
        try:
            with Path("logs/mpv.log").open("ab") as log:
                process = subprocess.Popen(["mpv", "--no-config", "--no-video", "--idle=yes",
                    "--ao=pulse", "--volume=" + str(self.initial_volume_percent if self.initial_volume_percent is not None else 60),
                    "--input-ipc-server=" + self.endpoint],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            for _ in range(30):
                if process.poll() is not None:
                    break
                time.sleep(0.1)
                try:
                    self.command("get_property", "idle-active")
                    return
                except MusicUnavailable:
                    pass
            if process.poll() is None:
                process.terminate()
        except OSError as exc:
            raise MusicUnavailable("Install mpv on the Pi before offline playback.") from exc
        raise MusicUnavailable("Unable to start mpv. Check logs/mpv.log.")

    def limit_volume(self):
        if self.max_volume_percent is not None:
            current = self.command("get_property", "volume", timeout=1)
            self.command("set_property", "volume", min(current, self.max_volume_percent), timeout=1)

    def reset_session_volume(self):
        volume = self.initial_volume_percent if self.initial_volume_percent is not None else self.max_volume_percent
        if volume is None or not self.is_running():
            return False
        self.command("set_property", "volume", volume, timeout=1)
        self._session_volume_initialized = True
        return True

    def command(self, *command, timeout=5):
        if sys.platform == "win32":
            raise MusicUnavailable("Use Pi/Linux for mpv IPC playback; Windows supports mock tests and Spotify API control.")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(timeout)
                connection.connect(self.endpoint)
                connection.sendall((json.dumps({"command": list(command), "request_id": 1}) + "\n").encode())
                deadline = time.monotonic() + timeout
                with connection.makefile("rb") as stream:
                    while time.monotonic() < deadline:
                        connection.settimeout(max(0.01, deadline - time.monotonic()))
                        line = stream.readline(65536)
                        if not line:
                            break
                        reply = json.loads(line)
                        if reply.get("request_id") == 1:
                            if reply.get("error") != "success":
                                raise MusicUnavailable("The music player could not perform that action.")
                            return reply.get("data")
        except (OSError, ValueError) as exc:
            raise MusicUnavailable("Offline music is unavailable. Check the mpv player and local file.") from exc
        raise MusicUnavailable("The music player did not respond.")

    def execute(self, intent):
        if intent == "PLAY_MUSIC":
            if self.auto_start:
                self.ensure_started()
            if self.initial_volume_percent is not None and not self._session_volume_initialized:
                self.command("set_property", "volume", self.initial_volume_percent)
                self._session_volume_initialized = True
            if self.max_volume_percent is not None:
                self.limit_volume()
            if self.command("get_property", "idle-active"):
                if not self.tracks or any(not p.is_file() for p in self.tracks):
                    raise MusicUnavailable("Add a local music file before starting playback.")
                playlist = list(self.tracks)
                random.shuffle(playlist)
                self.command("loadfile", str(playlist[0]), "replace")
                for track in playlist[1:]:
                    self.command("loadfile", str(track), "append")
            self.command("set_property", "pause", False)
        elif intent == "PAUSE":
            self.command("set_property", "pause", True)
        elif intent == "STOP":
            self.command("stop")
        elif intent == "NEXT":
            count = self.command("get_property", "playlist-count")
            if not count:
                raise MusicUnavailable("Start your music before skipping tracks.")
            position = self.command("get_property", "playlist-pos")
            self.command("set_property", "playlist-pos", (position + 1) % count)
            # With a one-track demo, next explicitly restarts the same track.
            if count == 1:
                self.command("seek", 0, "absolute")
            self.command("set_property", "pause", False)
        else:
            volume = self.command("get_property", "volume")
            ceiling = self.max_volume_percent if self.max_volume_percent is not None else 100
            limited = intent == "VOLUME_UP" and volume >= ceiling
            volume = max(0, min(ceiling, volume + (10 if intent == "VOLUME_UP" else -10)))
            self.command("set_property", "volume", volume)
            return {"backend": "offline", "volume_percent": volume,
                    "volume_limited": limited, "listening_limit_percent": ceiling}
        return {"backend": "offline"}


class SpotifyBackend:
    """go-librespot v0.10.3 local REST control, not Spotify's cloud Web API."""

    def __init__(self, uri="", *, base_url="http://127.0.0.1:3678", opener=urlopen):
        if uri and not re.fullmatch(r"spotify:(?:track|playlist|album):[A-Za-z0-9]+", uri):
            raise ValueError("Use a Spotify track, playlist, or album URI.")
        self.uri = uri
        self.base_url = base_url.rstrip("/")
        self.opener = opener

    def request(self, path, payload=None):
        request = Request(self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET")
        try:
            with self.opener(request, timeout=8) as response:
                if response.status == 204:
                    raise MusicUnavailable("Connect your Spotify account and select Swift Pi in Spotify first.")
                raw = response.read()
                return json.loads(raw) if raw else {}
        except (OSError, URLError, ValueError) as exc:
            raise MusicUnavailable("Spotify is unavailable. Check internet access and the Spotify player.") from exc

    def execute(self, intent):
        if intent == "PLAY_MUSIC":
            status = self.request("/status")
            if status.get("paused") and not status.get("stopped"):
                self.request("/player/resume", {})
            elif self.uri:
                self.request("/player/play", {"uri": self.uri})
            elif status.get("stopped"):
                raise MusicUnavailable("Choose music in Spotify or configure VCM_SPOTIFY_URI first.")
            else:
                self.request("/player/resume", {})
        elif intent in {"PAUSE", "STOP", "NEXT"}:
            path = {"PAUSE": "pause", "STOP": "stop", "NEXT": "next"}[intent]
            self.request("/player/" + path, {})
        else:
            current = self.request("/player/volume")
            maximum = current["max"]
            step = max(1, round(maximum * 0.1))
            value = max(0, min(maximum, current["value"] + (step if intent == "VOLUME_UP" else -step)))
            self.request("/player/volume", {"volume": value})
            return {"backend": "spotify", "volume_percent": round(value * 100 / maximum)}
        return {"backend": "spotify", "playback_confirmed": False}


class MusicService:
    def __init__(self, backend):
        self.backend = backend

    def execute(self, intent, slot_value=None):
        if intent not in RESPONSES:
            return ActionResult(intent, "unsupported", "Unsupported music intent.")
        if slot_value not in (None, ""):
            return ActionResult(intent, "invalid_slot", "This command does not accept a slot value.")
        try:
            data = self.backend.execute(intent)
        except MusicUnavailable as exc:
            return ActionResult(intent, "unavailable", str(exc))
        message = ("Music is at the listening volume limit."
                   if data.get("volume_limited") else RESPONSES[intent])
        if data.get("volume_limited") and "listening_limit_percent" in data:
            message = f"Music is at the {data['listening_limit_percent']} percent limit."
        if intent == "PLAY_MUSIC" and data.get("fallback"):
            message = "Spotify is unavailable. Playing your offline music."
        return ActionResult(intent, "ok", message, data)


class SoloistBackend:
    """Official Soloist ctl; account authorization stays in the daemon."""
    def __init__(self, uri="", *, binary="work/tools/soloist/soloist",
                 data_dir="work/soloist-runtime", runner=subprocess.run,
                 auto_start=False, key_file=None, launcher=subprocess.Popen,
                 max_volume_percent=None, operation_timeout=None, startup_timeout=8.,
                 control_timeout=6., readiness_attempts=20, initial_volume_percent=None):
        if uri and not re.fullmatch(r"spotify:(?:track|playlist|album):[A-Za-z0-9]+", uri):
            raise ValueError("Use a Spotify track, playlist, or album URI.")
        self.uri, self.binary, self.data_dir, self.runner = uri, binary, data_dir, runner
        self.stop_marker = Path(data_dir) / "swift-music-stopped"
        self.auto_start = auto_start
        self.max_volume_percent = max_volume_percent
        self.initial_volume_percent = initial_volume_percent
        self._session_volume_initialized = False
        self.key_file = Path(key_file or Path(data_dir) / "api-key")
        self.launcher = launcher
        self.operation_timeout = operation_timeout
        self.startup_timeout = startup_timeout
        self.control_timeout = control_timeout
        self.readiness_attempts = readiness_attempts
        self._operation_deadline = None
        self.phase = "idle"

    def is_running(self):
        # ctl discovery files are removed on normal shutdown. Probe the local
        # endpoint, rather than interpreting human-readable ctl status output.
        try:
            port = int((Path(self.data_dir) / "ws.port").read_text().strip())
            address = (Path(self.data_dir) / "ws.addr").read_text().strip()
            if address not in {"127.0.0.1", "localhost", "::1"}:
                return False
            with socket.create_connection((address, port), timeout=0.3):
                return True
        except (OSError, ValueError):
            return False

    def limit_volume(self):
        if self.max_volume_percent is not None:
            current = self.ctl("now", as_json=True, timeout=1).get("volume", 100)
            self.ctl("volume", str(min(int(current), self.max_volume_percent)), timeout=1)

    def reset_session_volume(self):
        volume = self.initial_volume_percent if self.initial_volume_percent is not None else self.max_volume_percent
        if volume is None or not self.is_running():
            return False
        self.ctl("volume", str(volume), timeout=1)
        self._session_volume_initialized = True
        return True

    def ensure_started(self):
        if self.is_running():
            return
        if sys.platform != "linux":
            raise MusicUnavailable("Automatic Soloist startup is supported on the Pi/Linux.")
        runtime = Path(self.data_dir)
        try:
            key = self.read_key()
            runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
            runtime.chmod(0o700)
            log_path = runtime / "soloist.log"
            # Keep startup logs private and bounded across restarts.
            if log_path.exists() and log_path.stat().st_size > 1024 * 1024:
                log_path.replace(runtime / "soloist.previous.log")
            descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.chmod(log_path, 0o600)
            with os.fdopen(descriptor, "ab") as log:
                process = self.launcher([str(Path(self.binary).resolve()), "-n", "Swift Pi", "-k", key,
                    "-D", str(runtime.resolve()), "-C", str((runtime / "cache").resolve()),
                    "-z", "100", "-i", str(self.initial_volume_percent if self.initial_volume_percent is not None else os.environ.get("VCM_SOLOIST_INITIAL_VOLUME", "60")),
                    "-w", os.environ.get("VCM_SOLOIST_WS", "127.0.0.1:3679")],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            deadline = time.monotonic() + self.startup_timeout
            if self._operation_deadline is not None:
                deadline = min(deadline, self._operation_deadline)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise MusicUnavailable("Soloist exited during startup. Check its private startup log.")
                if self.is_running():
                    return
                time.sleep(0.2)
            process.terminate()
        except OSError as exc:
            raise MusicUnavailable("Unable to start Soloist. Check the executable and private API key file.") from exc
        raise MusicUnavailable("Soloist startup timed out. Check its private startup log.")

    def read_key(self):
        session_key = os.environ.get("VCM_SOLOIST_API_KEY")
        if session_key:
            if any(c.isspace() for c in session_key):
                raise MusicUnavailable("Invalid session Soloist key")
            return session_key
        if self.key_file.stat().st_mode & 0o077:
            raise MusicUnavailable("Set the Soloist API key file permissions to 600.")
        key = self.key_file.read_text().strip()
        if not key or any(c.isspace() for c in key):
            raise MusicUnavailable("The Soloist API key file is empty or invalid.")
        return key

    def ctl(self, *args, as_json=False, timeout=6):
        timeout = min(timeout, self.control_timeout)
        # Reserve a bounded cleanup pause after a potentially submitted play.
        if self._operation_deadline is not None and args != ("pause",):
            timeout = min(timeout, self._operation_deadline - time.monotonic())
            if timeout <= 0:
                raise MusicUnavailable("Spotify response budget exceeded.")
        command = [self.binary, "ctl", *args, "-D", self.data_dir]
        if as_json:
            command.append("--json")
        try:
            result = self.runner(command, capture_output=True, text=True, check=True, timeout=timeout)
            return json.loads(result.stdout) if as_json else None
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise MusicUnavailable("Spotify is unavailable. Check Soloist and your account connection.") from exc

    def execute(self, intent):
        self._operation_deadline = (time.monotonic() + self.operation_timeout
                                    if self.operation_timeout is not None else None)
        try:
            return self._execute(intent)
        finally:
            self._operation_deadline = None

    def prepare(self, timeout=15.):
        """Warm this saved-login daemon without activating, playing or switching sources."""
        previous = self._operation_deadline
        self._operation_deadline = time.monotonic() + timeout
        try:
            self.phase = "startup"
            self.ensure_started()
            self.phase = "saved-login readiness"
            state = self._wait_state(lambda state: True, self.phase)
            return {"backend": "spotify", "control_ready": True,
                    "is_active": state.get("is_active") is True}
        finally:
            self._operation_deadline = previous

    def _execute(self, intent):
        if intent == "PLAY_MUSIC":
            self.phase = "startup"
            if self.auto_start:
                self.ensure_started()
            may_play = False
            try:
                self.phase = "saved-login readiness"
                state = self._wait_state(lambda state: True, "saved-login readiness")
                may_play = state.get("status") == "playing"
                if state.get("is_active") is not True:
                    self.phase = "device activation"
                    may_play = True
                    self.ctl("activate")
                    state = self._wait_state(lambda state: state.get("is_active") is True,
                                             "device activation")
                args = ["play"]
                fresh = state.get("status") != "paused" or self.stop_marker.exists()
                if self.uri and fresh:
                    args.append(self.uri)
                if self.initial_volume_percent is not None and not self._session_volume_initialized:
                    self.phase = "initial volume"
                    self.ctl("volume", str(self.initial_volume_percent))
                    self._session_volume_initialized = True
                if self.max_volume_percent is not None:
                    self.phase = "listening volume"
                    volume = self.ctl("now", as_json=True).get("volume", 100)
                    self.ctl("volume", str(min(int(volume), self.max_volume_percent)))
                if fresh and self.uri.startswith(("spotify:playlist:", "spotify:album:")):
                    self.phase = "shuffle"
                    self.ctl("shuffle", "on")
                may_play = True
                self.phase = "play dispatch"
                try:
                    self.ctl(*args)
                except MusicUnavailable as exc:
                    # A dispatched play with no acknowledgement may execute late.
                    # Pause is best-effort only; never start another player here.
                    try:
                        self.ctl("pause")
                    except MusicUnavailable:
                        pass
                    raise MusicStartUncertain("Spotify start acknowledgement is unknown. Check the player before switching music.") from exc
                self.phase = "playback confirmation"
                self._wait_state(lambda state: state.get("status") == "playing", "playback confirmation")
                self.stop_marker.unlink(missing_ok=True)
                return {"backend": "spotify", "player": "soloist", "playback_confirmed": True}
            except MusicStartUncertain:
                raise
            except MusicUnavailable as exc:
                if may_play:
                    try:
                        self._pause_before_fallback()
                    except MusicUnavailable:
                        raise MusicStartUncertain("Spotify playback is uncertain. Please check the player before switching music.") from exc
                raise
        elif intent == "STOP":
            self.ctl("pause")
            self.ctl("seek", "0")
            self.stop_marker.parent.mkdir(parents=True, exist_ok=True)
            self.stop_marker.touch()
        elif intent in {"PAUSE", "NEXT"}:
            self.ctl("pause" if intent == "PAUSE" else "next")
        else:
            state = self.ctl("now", as_json=True)
            ceiling = self.max_volume_percent if self.max_volume_percent is not None else 100
            limited = intent == "VOLUME_UP" and state["volume"] >= ceiling
            value = max(0, min(ceiling, state["volume"] + (10 if intent == "VOLUME_UP" else -10)))
            self.ctl("volume", str(value))
            return {"backend": "spotify", "player": "soloist", "volume_percent": value,
                    "volume_limited": limited, "listening_limit_percent": ceiling}
        return {"backend": "spotify", "player": "soloist"}

    def _wait_state(self, predicate, phase):
        for attempt in range(self.readiness_attempts):
            try:
                state = self.ctl("now", as_json=True)
                if isinstance(state, dict) and predicate(state):
                    return state
            except MusicUnavailable:
                pass
            if self._operation_deadline is not None and time.monotonic() >= self._operation_deadline:
                break
            if attempt + 1 < self.readiness_attempts:
                time.sleep(.25)
        raise MusicUnavailable(f"Spotify {phase} did not complete within its response budget.")

    def _pause_before_fallback(self):
        previous = self._operation_deadline
        self._operation_deadline = time.monotonic() + 3.
        try:
            self.ctl("pause")
            # A successful ctl response only acknowledges dispatch, not playback.
            # Observe quiescence twice before allowing a different player.
            self._wait_state(lambda state: state.get("status") in {"paused", "stopped"}, "pause confirmation")
            time.sleep(.25)
            self._wait_state(lambda state: state.get("status") in {"paused", "stopped"}, "pause confirmation")
        finally:
            self._operation_deadline = previous


def spotify_reachable():
    try:
        with urlopen("https://open.spotify.com", timeout=3):
            return True
    except (OSError, URLError):
        return False


class AutoMusicBackend:
    """Choose on PLAY only. Store selection so separate CLIs control the same player."""
    def __init__(self, online, offline, *, reachable=spotify_reachable,
                 state_path="work/music-routing.json"):
        self.online, self.offline, self.reachable = online, offline, reachable
        self.state_path = Path(state_path) if state_path else None
        self.active = None
        if self.state_path and self.state_path.exists():
            try:
                value = json.loads(self.state_path.read_text())["active"]
                self.active = value if value in {"spotify", "offline"} else None
            except (OSError, ValueError, KeyError):
                raise MusicUnavailable("Unable to read music routing state.")

    def save(self, value):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"active": value}))
            temporary.replace(self.state_path)
        self.active = value

    def limit_active_volume(self):
        if self.active is not None:
            player = self.online if self.active == "spotify" else self.offline
            player.limit_volume()

    def reset_active_session_volume(self):
        """Reset only the remembered player to the configured listening baseline."""
        if self.active is None:
            return False
        player = self.online if self.active == "spotify" else self.offline
        return player.reset_session_volume()

    def set_volume(self, percent):
        """Adjust only the selected player; never alter the OS/call sink gain."""
        if self.active is None:
            raise MusicUnavailable("Start your music first.")
        player = self.online if self.active == "spotify" else self.offline
        ceiling = getattr(player, "max_volume_percent", None)
        ceiling = ceiling if ceiling is not None else 100
        if type(percent) is not int or not 0 <= percent <= ceiling:
            raise MusicUnavailable(f"Choose a music volume from 0 to {ceiling} percent.")
        if self.active == "offline":
            player.command("set_property", "volume", percent, timeout=1)
        else:
            player.ctl("volume", str(percent), timeout=1)
        return {"backend": self.active, "volume_percent": percent}

    def execute(self, intent):
        if intent != "PLAY_MUSIC":
            if self.active is None:
                raise MusicUnavailable("Start your music first.")
            player = self.online if self.active == "spotify" else self.offline
            return player.execute(intent)
        # Stop the previously selected source before starting another; failure
        # blocks switching so two players cannot knowingly overlap.
        reachable = self.reachable()
        if self.active == "offline" or (self.active == "spotify" and not reachable):
            previous = self.online if self.active == "spotify" else self.offline
            try:
                if isinstance(previous, SoloistBackend):
                    previous._pause_before_fallback()
                else:
                    previous.execute("PAUSE")
            except MusicUnavailable:
                # A dead old process (e.g. after reboot) cannot overlap playback.
                if isinstance(previous, SoloistBackend) and not previous.is_running():
                    pass
                elif isinstance(previous, MpvBackend) and not previous.is_running():
                    pass
                else:
                    raise
        if reachable:
            try:
                data = self.online.execute("PLAY_MUSIC")
            except MusicStartUncertain:
                raise
            except MusicUnavailable:
                # Soloist either failed before start, or paused its attempted
                # start successfully before reporting failure.
                pass
            else:
                self.save("spotify")
                return data
        data = self.offline.execute("PLAY_MUSIC")
        self.save("offline")
        return {**data, "fallback": True}


def main():
    root = runtime_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", nargs="?", choices=list(RESPONSES))
    parser.add_argument("--menu", action="store_true")
    parser.add_argument("--backend", choices=["auto", "offline", "spotify", "soloist"], default="auto")
    parser.add_argument("--file", action="append", help="Local file; repeat for multiple tracks")
    parser.add_argument("--socket", default=os.environ.get("VCM_MPV_SOCKET", "/tmp/swift-mpv.sock"))
    parser.add_argument("--soloist-bin", default=str(root / "tools/soloist/soloist"))
    parser.add_argument("--soloist-key-file", default=str(root / "state/soloist/api-key"))
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice-model", default=str(root / "assets/voices/en_US-amy-medium.onnx"))
    args = parser.parse_args()
    if bool(args.intent) == args.menu:
        parser.error("Choose one intent or --menu")
    try:
        tracks = args.file or [str(p) for p in sorted((root / "assets/music").glob("*.mp3"))]
        offline = MpvBackend(tracks, args.socket, auto_start=True)
        spotify_uri = os.environ.get("VCM_SPOTIFY_URI", DEFAULT_SPOTIFY_URI)
        soloist = SoloistBackend(spotify_uri, binary=args.soloist_bin, auto_start=True,
                                key_file=args.soloist_key_file)
        backend = {"offline": lambda: offline, "soloist": lambda: soloist,
                   "spotify": lambda: SpotifyBackend(spotify_uri),
                   "auto": lambda: AutoMusicBackend(soloist, offline)}[args.backend]()
    except (ValueError, MusicUnavailable) as exc:
        parser.error(str(exc))
    service = MusicService(backend)
    speech = None
    if args.speak:
        from hey_swift.audio.speech import PiperSpeech, SpeechService
        speech = SpeechService(PiperSpeech(args.voice_model))

    def run(intent):
        result = service.execute(intent)
        payload = result.to_dict()
        if speech:
            from hey_swift.audio.speech import SpeechUnavailable
            try:
                speech.speak(result.message)
                payload["speech"] = {"status": "ok"}
            except SpeechUnavailable as exc:
                payload["speech"] = {"status": "unavailable", "message": str(exc)}
        print(json.dumps(payload, indent=2))
        return 2 if payload.get("speech", {}).get("status") == "unavailable" else (0 if result.status == "ok" else 1)

    if not args.menu:
        return run(args.intent)
    intents = list(RESPONSES)
    try:
        while True:
            for i, intent in enumerate(intents, 1):
                print(f"{i}. {intent}")
            choice = input("Select action (q to stop music and exit): ").strip()
            if choice.lower() == "q":
                return run("STOP")
            if choice.isdigit() and 1 <= int(choice) <= len(intents):
                run(intents[int(choice) - 1])
    except (KeyboardInterrupt, EOFError):
        return run("STOP")


if __name__ == "__main__":
    raise SystemExit(main())
