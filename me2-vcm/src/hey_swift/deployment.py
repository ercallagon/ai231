"""Explicit private setup and SSH commands; no account autodiscovery."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen

from .profiles import resolve_profile, PUBLIC, DISPLAY

ROOT = Path(__file__).resolve().parents[2]


class SetupError(ValueError):
    """Safe actionable setup message; never include secret values."""


SOLOIST_ARM64_URL = "https://soloist-builds.spotifycdn.com/soloist_release_arm64.tar.gz"
DEFAULT_SPOTIFY_URI = "spotify:playlist:37i9dQZEVXbNBz9cRCSFkY"
PIPER_AMY_REVISION = "664c651454f055ed34bd83f09e024ffbc0da09ac"
PIPER_AMY_FILES = (
    ("en_US-amy-medium.onnx", 63_201_294, "778d28aeb95fcdf8a882344d9df142fc"),
    ("en_US-amy-medium.onnx.json", 7_007, "eed7b28ea26b86395fd842942bdaea4c"),
)


def download_soloist(home):
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise SetupError("Automatic Soloist download is only supported on ARM64; use Spotify's official download page")
    downloads = home / "downloads"
    binary_dir = home / "bin"
    downloads.mkdir(parents=True, exist_ok=True, mode=0o700)
    binary_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = downloads / "soloist_release_arm64.tar.gz"
    binary = binary_dir / "soloist"
    if archive.exists() or binary.exists():
        raise SetupError("Soloist archive or executable already exists in this checkout; inspect it before replacing anything")

    archive_fd, archive_tmp = tempfile.mkstemp(prefix="soloist-", suffix=".download", dir=downloads)
    binary_tmp = None
    try:
        total = 0
        with os.fdopen(archive_fd, "wb") as output:
            with urlopen(SOLOIST_ARM64_URL, timeout=30) as response:
                if not response.geturl().startswith("https://"):
                    raise SetupError("Spotify download redirected away from HTTPS")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise SetupError("Soloist archive exceeded the 512 MiB download limit")
                    output.write(chunk)
        if total == 0:
            raise SetupError("Spotify returned an empty Soloist archive")
        os.link(archive_tmp, archive)
        print(f"Downloaded Spotify archive: {total / (1024 * 1024):.1f} MiB")

        with tarfile.open(archive, "r:gz") as bundle:
            members = [member for member in bundle.getmembers()
                       if member.isfile() and Path(member.name).name == "soloist"]
            if len(members) != 1:
                raise SetupError("Official archive did not contain exactly one regular soloist executable")
            if not 0 < members[0].size <= 512 * 1024 * 1024:
                raise SetupError("Soloist executable has an invalid or excessive size")
            source = bundle.extractfile(members[0])
            if source is None:
                raise SetupError("Could not read Soloist executable from the archive")
            binary_fd, binary_tmp = tempfile.mkstemp(prefix="soloist-", dir=binary_dir)
            with os.fdopen(binary_fd, "wb") as output, source:
                shutil.copyfileobj(source, output)
            os.chmod(binary_tmp, 0o700)
        os.link(binary_tmp, binary)
        print(f"Soloist executable installed at {binary}")
        return str(binary)
    except SetupError:
        raise
    except Exception as exc:
        raise SetupError(f"Soloist download or archive validation failed ({type(exc).__name__})") from None
    finally:
        for temporary in (archive_tmp, binary_tmp):
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


def download_default_voice(home):
    voice_dir = home / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    base_url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/"
        f"{PIPER_AMY_REVISION}/en/en_US/amy/medium"
    )
    temporary = []
    installed = []
    destinations = [voice_dir / name for name, _, _ in PIPER_AMY_FILES]
    if any(path.exists() for path in destinations):
        raise SetupError("A default voice file already exists; inspect it before replacing anything")
    try:
        for (name, expected_size, expected_md5), destination in zip(PIPER_AMY_FILES, destinations):
            fd, temp_path = tempfile.mkstemp(prefix="piper-voice-", dir=voice_dir)
            temporary.append(temp_path)
            digest = hashlib.md5(usedforsecurity=False)
            total = 0
            with os.fdopen(fd, "wb") as output:
                with urlopen(f"{base_url}/{name}", timeout=60) as response:
                    if not response.geturl().startswith("https://"):
                        raise SetupError("Voice download redirected away from HTTPS")
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > expected_size:
                            raise SetupError(f"Voice file {name} exceeded its published size")
                        digest.update(chunk)
                        output.write(chunk)
            if total != expected_size or digest.hexdigest() != expected_md5:
                raise SetupError(f"Downloaded voice file {name} failed its pinned size/checksum check")
        for temp_path, destination in zip(temporary, destinations):
            os.link(temp_path, destination)
            installed.append(destination)
        print(f"Downloaded default Piper voice: {PIPER_AMY_FILES[0][1] / (1024 * 1024):.1f} MiB model")
        return str(destinations[0])
    except SetupError:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for path in installed:
            path.unlink(missing_ok=True)
        raise SetupError(f"Default voice download failed ({type(exc).__name__})") from None
    finally:
        for temp_path in temporary:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


CONFIG_FIELDS = {
    "call-number": ("calling_number", r"\+[1-9][0-9]{7,14}"),
    "sms-number": ("messaging_number", r"\+[1-9][0-9]{7,14}"),
    "phone-mac": ("phone_mac", r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}"),
    "spotify-key": ("soloist_key", r"\S+"),
}


def update_config(home, key, value=None):
    cfg = load_config(home)
    if key in CONFIG_FIELDS:
        secret_key, pattern = CONFIG_FIELDS[key]
        if value is not None:
            raise SetupError(f"{key} is entered through a hidden prompt; omit the value from the command")
        cfg.setdefault("secrets", {})[secret_key] = hidden(f"New {key}", pattern)
    elif key == "spotify-uri":
        updated = value if value is not None else input("Spotify URI (Enter restores Top 50 - Philippines): ").strip()
        if updated and not re.fullmatch(r"spotify:(track|playlist|album):[A-Za-z0-9]+", updated):
            raise SetupError("Use a spotify:track, spotify:album or spotify:playlist URI")
        cfg["spotify_uri"] = updated or DEFAULT_SPOTIFY_URI
    elif key == "weather-location":
        updated = value if value is not None else input("Weather city label [Quezon City]: ").strip()
        cfg["city"] = updated or "Quezon City"
        cfg["latitude"] = float(input("Latitude [14.6488]: ").strip() or "14.6488")
        cfg["longitude"] = float(input("Longitude [121.0509]: ").strip() or "121.0509")
        if not -90 <= cfg["latitude"] <= 90 or not -180 <= cfg["longitude"] <= 180:
            raise SetupError("Invalid coordinates")
    else:
        raise SetupError("Editable fields: call-number, sms-number, phone-mac, spotify-key, spotify-uri, weather-location")
    cfg["updated_utc"] = datetime.now(timezone.utc).isoformat()
    private_json(home / "profile.json", cfg)
    print(f"Updated {key}; other settings were retained and the previous profile was backed up privately.")
    return 0



def yes(prompt, default=False):
    suffix = " [Y/n]: " if default else " [y/N]: "
    return input(prompt + suffix).strip().lower() in (("", "y", "yes") if default else ("y", "yes"))


def private_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.is_symlink():
        raise SetupError("Private file is a symlink")
    if path.exists():
        path.rename(path.with_name(path.stem + ".previous_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + path.suffix))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_config(home):
    path = home / "profile.json"
    if not path.is_file():
        raise SetupError("Run setup first")
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise SetupError("Private configuration must have mode 600")
    return json.loads(path.read_text())


def devices():
    import sounddevice as sd
    apis = sd.query_hostapis()
    return [dict(index=i, name=d["name"], hostapi=apis[d["hostapi"]]["name"],
                 inputs=d["max_input_channels"], outputs=d["max_output_channels"])
            for i, d in enumerate(sd.query_devices())]


def microphone(config):
    saved = config["microphone"]
    matches = [d for d in devices() if d["inputs"] and d["name"] == saved["name"] and d["hostapi"] == saved["hostapi"]]
    if len(matches) != 1:
        raise SetupError("Configured microphone absent or ambiguous; rerun setup")
    return str(matches[0]["index"])


def choose_microphone(listing):
    inputs = [device for device in listing if device["inputs"]]
    if not inputs:
        raise SetupError("No microphone input device found")
    if len(inputs) == 1:
        chosen = inputs[0]
        print(f"Using microphone {chosen['index']}: {chosen['name']} ({chosen['hostapi']}).")
        return chosen
    print("Select the microphone to use:")
    for device in inputs:
        print(f"{device['index']}: {device['name']} ({device['hostapi']})")
    while True:
        selected = input("Microphone index: ").strip()
        chosen = next((device for device in inputs if str(device["index"]) == selected), None)
        if chosen:
            return chosen
        print("Choose one of the listed input devices.")


def hidden(prompt, pattern=None):
    while True:
        value = getpass.getpass(prompt + ": ").strip()
        if value and (pattern is None or re.fullmatch(pattern, value)):
            return value
        print("Invalid or empty value; try again. Input is hidden.")


def existing_path(prompt, executable=False):
    while True:
        value = Path(input(prompt + ": ").strip()).expanduser().resolve()
        if value.is_file() and (not executable or os.access(value, os.X_OK)):
            return str(value)
        print("File is missing or not executable; obtain it first, then retry.")


def wizard(home, session=False):
    if not sys.stdin.isatty():
        raise SetupError("Setup needs an interactive SSH terminal (ssh -t)")
    print("Private setup for this checkout. Existing Spotify accounts/recipients are not imported.")
    print("OS audio routes, Bluetooth pairing and GPIO remain shared with other installations.")
    if not session and (home / "profile.json").exists() and yes("Explicitly reuse this checkout's saved configuration?"):
        return load_config(home)
    chosen = choose_microphone(devices())
    cfg = dict(schema=1, microphone={k: chosen[k] for k in ("name", "hostapi")},
               hardware="mock", enabled=["information", "reminders"], weather=False,
               spotify=False, speech=False, secrets={})
    print("Playback uses the OS default PipeWire sink; inspect it with wpctl status -n.")
    if yes("Enable RGB LED and TM1637 using the documented BCM17/27/22 and BCM23/24 wiring?"):
        cfg["enabled"].append("devices")
        cfg["hardware"] = "pi"
    if yes("Enable Open-Meteo weather? Uses internet and defaults to Quezon City."):
        cfg["enabled"] = sorted(set(cfg["enabled"] + ["information"]))
        cfg["weather"] = True
        cfg.update(city="Quezon City", latitude=14.6488, longitude=121.0509)
    if yes("Enable offline MP3 music?"):
        default_music_dir = home / "music"
        default_music_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cfg["music_dir"] = str(default_music_dir)
        cfg["enabled"].append("music")
        print(f"Add MP3s you are authorized to use under {default_music_dir}; no path entry is needed.")
        if not list(default_music_dir.glob("*.mp3")):
            print("No MP3 files found yet; setup will continue. Add files before using offline music.")
    if yes("Enable Spotify Soloist?"):
        print("Each user needs Spotify Premium and a personal Soloist key:")
        print("https://developer.spotify.com/documentation/soloist/concepts/authentication")
        print("Setup downloads Spotify's official ARM64 executable into this checkout when needed.")
        print("Use a phone browser/app on this LAN for authorization. No password is requested here.")
        default_soloist = home / "bin/soloist"
        if default_soloist.is_file() and os.access(default_soloist, os.X_OK):
            cfg["soloist_binary"] = str(default_soloist)
            print(f"Using this checkout's Soloist executable: {default_soloist}")
        elif not default_soloist.exists():
            cfg["soloist_binary"] = download_soloist(home)
        else:
            raise SetupError(f"Soloist path exists but is not executable; inspect it: {default_soloist}")
        cfg["secrets"]["soloist_key"] = hidden("Personal Soloist API key", r"\S+")
        cfg["spotify"] = True
        cfg["enabled"].append("music")
        cfg["spotify_uri"] = DEFAULT_SPOTIFY_URI
        print("Music default: Spotify Top 50 - Philippines. Change later with ./swift config set spotify-uri.")
    for group, label in (("calling", "phone calls"), ("messaging", "SMS")):
        if yes(f"Enable real {label} (accepted voice commands act automatically in live mode)?"):
            cfg["enabled"].append(group)
            cfg["secrets"][group + "_number"] = hidden("Recipient in international +country format", r"\+[1-9][0-9]{7,14}")
    if set(cfg["enabled"]) & {"calling", "messaging"}:
        cfg["secrets"]["phone_mac"] = hidden("Phone Bluetooth MAC", r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}")
        print("Pairing and phone permission prompts are completed using: swift.py bluetooth")
    if yes("Enable Piper spoken responses?"):
        print("Default: en_US-amy-medium (~60 MiB), synthesis only; not used for command detection.")
        print("Voice card: https://huggingface.co/rhasspy/piper-voices/tree/664c651454f055ed34bd83f09e024ffbc0da09ac/en/en_US/amy/medium")
        print("Its model card points to Mycroft Mimic3 voice terms: https://github.com/MycroftAI/mimic3-voices/blob/master/LICENSE")
        if yes("Enable Piper speech with the default Amy voice and accept its terms?"):
            cfg["voice_model"] = str(home / "voices/en_US-amy-medium.onnx")
            if not Path(cfg["voice_model"]).is_file() and not Path(cfg["voice_model"] + ".json").is_file():
                cfg["voice_model"] = download_default_voice(home)
            if not Path(cfg["voice_model"]).is_file() or not Path(cfg["voice_model"] + ".json").is_file():
                raise SetupError("Default voice model and matching JSON must both be present")
            cfg["speech"] = True
        else:
            print("Speech remains disabled; setup will continue without a voice model.")
    if cfg["speech"]:
        try:
            importlib.import_module("piper")
        except ImportError:
            if sys.prefix == sys.base_prefix:
                raise SetupError("Piper is required for spoken responses")
            print("Installing the pinned Piper dependencies into this checkout's virtual environment.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--require-hashes", "--only-binary=:all:",
                                   "-r", str(ROOT / "requirements-speech-py313-aarch64.lock")])
            importlib.invalidate_caches()
            importlib.import_module("piper")
        cfg["speech"] = True
    cfg["enabled"] = sorted(set(cfg["enabled"]))
    cfg["created_utc"] = datetime.now(timezone.utc).isoformat()
    if not session:
        private_json(home / "profile.json", cfg)
        print("Saved privately (directory 700, profile 600). Never share this directory.")
    return cfg


def isolated_env(cfg, state):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VCM_", "HEY_SWIFT_"))}
    for variable in ("VCM_SPEECH_CHANNEL_VOLUME", "VCM_SPEECH_SINK"):
        if variable in os.environ:
            env[variable] = os.environ[variable]
    env.update(HEY_SWIFT_RUNTIME_ROOT=str(state), HEY_SWIFT_ENABLED_INTEGRATIONS=",".join(cfg["enabled"]),
               HEY_SWIFT_CONFIRM_COMMUNICATIONS="0", VCM_SOLOIST_PROFILE="bundled",
               VCM_SPOTIFY_ENABLED="1" if cfg.get("spotify") else "0",
               VCM_SOLOIST_DATA_DIR=str(state / "state/soloist"),
               VCM_SOLOIST_KEY_FILE=str(state / "state/soloist/api-key"), VCM_SOLOIST_WS="127.0.0.1:0",
               VCM_SOLOIST_INITIAL_VOLUME="30",
               VCM_SOLOIST_BIN=cfg.get("soloist_binary", str(state / "not-installed")),
               VCM_WEATHER_ENABLED="1" if cfg.get("weather") else "0")
    for key, variable in (("calling_number", "VCM_CALL_NUMBER"), ("messaging_number", "VCM_MESSAGE_NUMBER"),
                          ("phone_mac", "VCM_PHONE_MAC"), ("soloist_key", "VCM_SOLOIST_API_KEY")):
        env[variable] = cfg.get("secrets", {}).get(key, "")
    for key, variable in (("voice_model", "VCM_VOICE_MODEL"), ("music_dir", "VCM_MUSIC_DIR"),
                          ("spotify_uri", "VCM_SPOTIFY_URI"), ("latitude", "VCM_WEATHER_LATITUDE"),
                          ("longitude", "VCM_WEATHER_LONGITUDE"), ("city", "VCM_WEATHER_CITY")):
        if cfg.get(key) not in (None, ""):
            env[variable] = str(cfg[key])
    return env


@contextmanager
def applied_env(env):
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def assistant(cfg, state, model="champion", live=False, latency=False,
              confirm_communications=False, speak_no_speech=False,
              music_start_volume=30, music_volume_limit=50):
    if not 0 <= music_start_volume <= music_volume_limit <= 100:
        raise SetupError("Choose 0 <= music start volume <= music limit <= 100")
    profile = resolve_profile(model)
    device = microphone(cfg)
    env = isolated_env(cfg, state)
    env["HEY_SWIFT_CONFIRM_COMMUNICATIONS"] = "1" if confirm_communications else "0"
    if latency:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logs = ROOT / "work/pi5_test_01" / (PUBLIC[profile] + "_live_" + stamp)
        command = [sys.executable, str(ROOT / "scripts/run_benchmark_assistant.py"),
                   "--profile", model, "--device", device, "--log-dir", str(logs)]
        try:
            subprocess.call(command, cwd=ROOT, env=env)
        except KeyboardInterrupt:
            pass
        subprocess.check_call([sys.executable, str(ROOT / "scripts/summarize_runtime_latency.py"),
            str(logs), "--output", str(logs / "latency.json")], cwd=ROOT)
        return 0
    command = [sys.executable, str(ROOT / "scripts/run_assistant.py"), "--profile", model,
               "--device", device, "--hardware", cfg["hardware"] if live else "mock",
               "--music-start-volume-percent", str(music_start_volume),
               "--music-listen-volume-percent", str(music_volume_limit)]
    if not cfg.get("spotify") and not cfg.get("weather"):
        command.append("--offline")
    if live:
        command.append("--live")
    if speak_no_speech:
        command.append("--speak-no-speech")
    if cfg.get("speech"):
        command.append("--speak")
    return subprocess.call(command, cwd=ROOT, env=env)


def spotify(cfg, state):
    if not cfg.get("spotify"):
        raise SetupError("Spotify not enabled")
    print("Starting this checkout's Swift Pi device and attempting saved-login activation.")
    print("A key may be visible to processes owned by this OS user; no key is printed or put in shell history.")
    with applied_env(isolated_env(cfg, state)):
        from .actions.services.music import SoloistBackend
        backend = SoloistBackend(binary=cfg["soloist_binary"], data_dir=str(state / "state/soloist"),
            auto_start=True, startup_timeout=15, control_timeout=3, max_volume_percent=50, initial_volume_percent=30,
            operation_timeout=25, readiness_attempts=20)
        try:
            backend.prepare()
            backend._operation_deadline = time.monotonic() + 25.
            backend.ctl("activate", timeout=3)
            status = backend._wait_state(lambda state: state.get("is_active") is True, "device activation")
            print("Spotify responding; active device:", bool(status.get("is_active")))
        except Exception:
            print("Authorization/activation pending. For first login select Swift Pi in the phone Spotify picker, then retry ./swift spotify. Check Premium/key and the private Soloist log.")
            return 1
        finally:
            backend._operation_deadline = None
    return 0


def bluetooth(cfg):
    mac = cfg.get("secrets", {}).get("phone_mac")
    if not mac:
        raise SetupError("No phone configured")
    print("Pair/trust/connect changes shared OS Bluetooth state. It is not isolated by this checkout.")
    if not yes("Proceed with pairing your configured phone?"):
        return 1
    print("Enable Bluetooth/discoverability on the phone. Approve matching passkeys and call/message access there.")
    subprocess.run(["bluetoothctl", "--timeout", "15", "scan", "on"], check=False)
    info = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, check=False)
    already_paired = re.search(r"^\s*Paired:\s+yes\s*$", info.stdout, re.MULTILINE) is not None
    if already_paired:
        print("Phone is already paired; continuing with trust/connect.")
    else:
        pair = subprocess.run(["bluetoothctl", "--agent", "KeyboardDisplay", "pair", mac],
                              capture_output=True, text=True, check=False)
        if pair.returncode and "org.bluez.Error.AlreadyExists" not in pair.stdout + pair.stderr:
            print("Bluetooth step failed; pairing is pending. No call or SMS was initiated.")
            return 1
    for command in (("trust",), ("connect",)):
        if subprocess.call(["bluetoothctl", *command, mac]):
            print("Bluetooth step failed; trust/connect is pending. No call or SMS was initiated.")
            return 1
    print("Bluetooth connected. Phone HFP/PipeWire telephony and MAP/OBEX support still require diagnostics.")
    return 0


def install(use_uv=False):
    if platform.machine() not in ("aarch64", "arm64") or sys.version_info[:2] != (3, 13):
        raise SetupError("The Pi lock targets ARM64 Python 3.13; use /usr/bin/python3 on Debian 13")
    venv = ROOT / ".venv"
    if venv.exists():
        raise SetupError("Environment already exists; use doctor or a separate checkout")
    lock = ROOT / "requirements-pi-py313-aarch64.lock"
    if use_uv:
        uv = shutil.which("uv")
        if not uv:
            raise SetupError("uv is not installed")
        subprocess.check_call([uv, "venv", "--python", sys.executable, "--system-site-packages", "--no-managed-python", "--seed", str(venv)])
        command = [uv, "pip", "install", "--python", str(venv / "bin/python"), "--require-hashes", "--only-binary", ":all:", "-r", str(lock)]
    else:
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
        command = [str(venv / "bin/python"), "-m", "pip", "install", "--require-hashes", "--only-binary=:all:", "-r", str(lock)]
    subprocess.check_call(command)
    subprocess.check_call([str(venv / "bin/python"), "-c", "import lgpio; print('OS-provided lgpio import OK:', lgpio.__file__)"])
    print("Installed. Next: .venv/bin/python scripts/swift.py setup")


def doctor(home):
    from .deployment_checks import diagnose
    cfg = load_config(home) if (home / "profile.json").exists() else None
    return diagnose(ROOT, home, cfg)


def gpio_self_test(home):
    cfg = load_config(home)
    if cfg.get("hardware") != "pi":
        raise SetupError("GPIO is not enabled in this checkout; review setup before running a physical test")
    print("This test briefly cycles the RGB LED and displays 1234 with the colon.")
    print("It directly drives only the configured LED/display pins; stop the assistant first.")
    if not yes("Run the physical GPIO/display test now?"):
        print("GPIO test cancelled; no result recorded.")
        return 1

    from .hardware.devices import PiHardware
    hardware = None
    error = None
    observed = {"rgb_sequence": False, "display_1234_colon": False}
    try:
        hardware = PiHardware()
        for color in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            hardware.led(color)
            time.sleep(.7)
        hardware.led((0, 0, 0))
        hardware.display("1234", True)
        observed["rgb_sequence"] = yes("Did the LED visibly show red, green, then blue?")
        observed["display_1234_colon"] = yes("Does the display currently show 1234 with the colon?")
    except Exception as exc:
        error = type(exc).__name__
    finally:
        if hardware is not None:
            try:
                hardware.led((0, 0, 0))
                hardware.display("    ")
            except Exception as exc:
                error = error or type(exc).__name__
            try:
                hardware.close()
            except Exception as exc:
                error = error or type(exc).__name__

    if error:
        print(f"GPIO write failed ({error}); visual confirmation was not requested.")
    passed = not error and all(observed.values())
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="PASS" if passed else "FAIL",
        scope="operator-confirmed physical GPIO/display check; not model-to-action validation",
        observations=observed,
        driver_error=error,
        wiring="RGB BCM17/27/22; TM1637 CLK/DIO BCM23/24",
    )
    private_json(home / "gpio_check.json", result)
    print(f"GPIO/display {result['status']}; result saved privately in this checkout")
    return 0 if passed else 1


def session_menu(home):
    if not Path("/dev/shm").is_dir():
        raise SetupError("Session-only mode requires /dev/shm; persistent private setup is available")
    print("Session-only: configuration and Soloist state use a temporary RAM filesystem; no saved profile is loaded.")
    print("OS Bluetooth state persists; RAM filesystem may be swappable depending on OS configuration.")
    with tempfile.TemporaryDirectory(prefix="swift-session-", dir="/dev/shm") as directory:
        state = Path(directory) / "runtime"
        cfg = wizard(home, session=True)
        try:
            while True:
                action = input("doctor / preview / live / spotify / bluetooth / quit: ").strip()
                if action == "quit":
                    break
                if action in ("preview", "live"):
                    try:
                        assistant(cfg, state, live=action == "live")
                    except KeyboardInterrupt:
                        print("\nReturning to session menu; type quit to end and clean session state.")
                elif action == "spotify":
                    spotify(cfg, state)
                elif action == "bluetooth":
                    bluetooth(cfg)
                elif action == "doctor":
                    from .deployment_checks import diagnose
                    diagnose(ROOT, Path(directory), cfg)
        finally:
            # Stop only the daemon started in this session's unique directory.
            pid = state / "state/soloist/soloist.pid"
            if pid.exists():
                import signal
                try:
                    value = int(pid.read_text().strip())
                    if value > 1 and str(state) in Path(f"/proc/{value}/cmdline").read_text():
                        os.kill(value, signal.SIGTERM)
                except (OSError, ValueError):
                    pass


def demo(cfg, state, model):
    print("Say Hey Swift, wait for capture readiness, then try the applicable commands:")
    print("What time is it? / Turn on the light / Set a timer for ten seconds /")
    print("Set the temperature to twenty-two degrees / Play music / Pause music /")
    print("Set the temperature to thirty degrees (unsupported: must not execute).")
    print("Only enabled integrations execute. Enabled calls/SMS act automatically on accepted voice commands.")
    print("Listen for prompt/response overlap; test music only at the configured listening cap. No AEC is claimed.")
    if not yes("Start the real voice demo with configured services?"):
        return 1
    try:
        assistant(cfg, state, model=model, live=True)
    except KeyboardInterrupt:
        print("\nVoice loop stopped; record your observations below.")
    print("Record only what you actually observed. Enter PASS, FAIL or NOT_RUN.")
    observations = {}
    checks = ["wake_and_command_capture", "unsupported_value_rejection", "offline_without_internet", "mic_recovery"]
    checks += ["integration_" + name for name in cfg["enabled"]]
    if cfg.get("speech"):
        checks.append("audible_TTS_and_mic_interference")
    for check in checks:
        value = input(check + " [NOT_RUN]: ").strip().upper() or "NOT_RUN"
        observations[check] = value if value in ("PASS", "FAIL", "NOT_RUN") else "NOT_RUN"
    private_json(ROOT / "work/pi5_test_01/DEMO.json", dict(
        created_utc=datetime.now(timezone.utc).isoformat(), model=resolve_profile(model),
        scope="operator-reported physical observations, not an automatically scored acoustic benchmark",
        observations=observations, shared_OS=True))
    print("Observations saved. Run swift.py report to refresh the consolidated report.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=ROOT / "work/user", help="Private configuration/state for this checkout")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("install"); p.add_argument("--uv", action="store_true")
    p = sub.add_parser("setup"); p.add_argument("--session", action="store_true")
    sub.add_parser("gpio", help="Run and record a physical RGB LED/TM1637 check")
    p = sub.add_parser("config", help="Update one value in this checkout's private setup")
    config_sub = p.add_subparsers(dest="config_action", required=True)
    p = config_sub.add_parser("set")
    p.add_argument("key", choices=(*CONFIG_FIELDS, "spotify-uri", "weather-location"))
    p.add_argument("value", nargs="?")
    sub.add_parser("doctor")
    for action in ("run", "live-latency", "demo"):
        p = sub.add_parser(action)
        p.add_argument("--model", type=resolve_profile, metavar="{champion,baseline}", default="champion",
                       help="Champion or baseline; historical aliases remain compatible")
        if action == "run":
            p.add_argument("--live", action="store_true")
            p.add_argument("--music-start-volume", type=int, default=30,
                           help="Music player start percentage (default: 30)")
            p.add_argument("--music-limit", type=int, default=50,
                           help="Music player ceiling (default: 50); lower if wake detection suffers")
            p.add_argument("--confirm-communications", action="store_true",
                           help="Optionally require typed approval for each real call/SMS")
            p.add_argument("--speak-no-speech", action="store_true",
                           help="Speak feedback on empty capture; default is console feedback only")
    p = sub.add_parser("evaluate"); p.add_argument("--reset", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--official", type=Path)
    sub.add_parser("spotify"); sub.add_parser("bluetooth")
    args = parser.parse_args()
    home = args.home.expanduser().resolve()
    if args.action == "install":
        install(args.uv)
        return 0
    if args.action == "setup":
        storage = "session" if args.session else input("Storage: private (mode 600) or session (RAM state) [private]: ").strip().lower() or "private"
        if storage not in ("private", "session"):
            raise SetupError("Choose private or session")
        session_menu(home) if storage == "session" else wizard(home)
        return 0
    if args.action == "doctor":
        return doctor(home)
    if args.action == "gpio":
        return gpio_self_test(home)
    if args.action == "config":
        return update_config(home, args.key, args.value)
    if args.action == "evaluate":
        print(f"Stage 1/3: frozen {DISPLAY['run043']} vs {DISPLAY['selected']} test/holdout scoring.", flush=True)
        code = subprocess.call([sys.executable, str(ROOT / "scripts/test_pi.py"), *( ["--reset"] if args.reset else [])], cwd=ROOT)
        if code:
            return code
        costs = {}
        print("Stage 2/3: model-only CPU timing (30 warmups, 200 iterations, one thread).", flush=True)
        for name, capacity in (("baseline", "2"), ("champion", "6")):
            profile = resolve_profile(name)
            path = ROOT / "work/pi5_test_01" / (name + "_cost.json")
            display_name = DISPLAY[profile]
            print(f"Benchmarking {display_name} ONNX model ({capacity}s nominal capacity)...", flush=True)
            code = subprocess.call([sys.executable, str(ROOT / "scripts/benchmark_onnx_pi.py"),
                "--model", str(ROOT / "models/vcm" / (profile + ".onnx")), "--audio-seconds", capacity,
                "--threads", "1", "--output", str(path)], cwd=ROOT)
            if code:
                return code
            costs[name] = json.loads(path.read_text())
            print(f"{display_name}: model-only p95 {costs[name]['p95_ms']:.2f} ms; nominal RTF p95 "
                  f"{costs[name]['nominal_model_only_rtf_p95']:.5f}.", flush=True)
        (ROOT / "work/pi5_test_01/model_costs.json").write_text(json.dumps(costs, indent=2) + "\n")
        print("Stage 3/3: refreshing diagnostics and consolidated report.", flush=True)
        doctor(home)
    if args.action in ("evaluate", "report"):
        from .submission_report import build_report
        build_report(ROOT, official=getattr(args, "official", None))
        return 0
    cfg = load_config(home)
    state = home / "runtime"
    if args.action == "spotify":
        return spotify(cfg, state)
    if args.action == "bluetooth":
        return bluetooth(cfg)
    if args.action == "demo":
        return demo(cfg, state, args.model)
    return assistant(cfg, state, model=args.model, live=getattr(args, "live", False), latency=args.action == "live-latency",
                     confirm_communications=getattr(args, "confirm_communications", False),
                     speak_no_speech=getattr(args, "speak_no_speech", False),
                     music_start_volume=getattr(args, "music_start_volume", 30),
                     music_volume_limit=getattr(args, "music_limit", 50))
