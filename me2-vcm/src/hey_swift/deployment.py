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
from urllib.request import urlopen

from .profiles import resolve_profile, PUBLIC

ROOT = Path(__file__).resolve().parents[2]


class SetupError(ValueError):
    """Safe actionable setup message; never include secret values."""


SOLOIST_ARM64_URL = "https://soloist-builds.spotifycdn.com/soloist_release_arm64.tar.gz"


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
    listing = devices()
    for d in listing:
        print(f"{d['index']}: {d['name']} ({d['hostapi']}) input={d['inputs']} output={d['outputs']}")
    while True:
        selected = input("Microphone index: ").strip()
        chosen = next((d for d in listing if str(d["index"]) == selected and d["inputs"]), None)
        if chosen:
            break
        print("Choose a listed input device.")
    cfg = dict(schema=1, microphone={k: chosen[k] for k in ("name", "hostapi")},
               hardware="mock", enabled=[], weather=False, spotify=False, speech=False, secrets={})
    print("Playback uses the OS default PipeWire sink; inspect it with wpctl status -n.")
    if yes("Enable local time and reminders?", True):
        cfg["enabled"] += ["information", "reminders"]
    if yes("Enable RGB LED and TM1637 using the documented BCM17/27/22 and BCM23/24 wiring?"):
        cfg["enabled"].append("devices")
        cfg["hardware"] = "pi"
    if yes("Enable internet weather (Open-Meteo; no API key)?"):
        cfg["enabled"] = sorted(set(cfg["enabled"] + ["information"]))
        cfg["weather"] = True
        cfg["city"] = input("Weather location label [Quezon City]: ").strip() or "Quezon City"
        cfg["latitude"] = float(input("Latitude [14.6488]: ").strip() or "14.6488")
        cfg["longitude"] = float(input("Longitude [121.0509]: ").strip() or "121.0509")
        if not -90 <= cfg["latitude"] <= 90 or not -180 <= cfg["longitude"] <= 180:
            raise SetupError("Invalid coordinates")
    if yes("Enable offline MP3 music?"):
        if not yes("Do you have permission to use these local audio files?"):
            raise SetupError("Authorized assets required")
        default_music_dir = home / "music"
        default_music_dir.mkdir(parents=True, exist_ok=True)
        entered_music_dir = input(
            f"Directory containing authorized MP3 files [{default_music_dir}]: "
        ).strip()
        directory = Path(entered_music_dir or default_music_dir).expanduser().resolve()
        if not list(directory.glob("*.mp3")):
            raise SetupError(f"No MP3 files in {directory}; copy authorized files there, then rerun setup")
        cfg["music_dir"] = str(directory)
        cfg["enabled"].append("music")
    if yes("Enable Spotify Soloist?"):
        print("Each user needs Spotify Premium and a personal Soloist key:")
        print("https://developer.spotify.com/documentation/soloist/concepts/authentication")
        print("Get the ARM64 binary from https://developer.spotify.com/documentation/soloist/reference/downloads-and-updates")
        print("Use a phone browser/app on this LAN for authorization. No password is requested here.")
        default_soloist = home / "bin/soloist"
        if default_soloist.is_file() and os.access(default_soloist, os.X_OK):
            if yes(f"Use this checkout's Soloist executable at {default_soloist}?", True):
                cfg["soloist_binary"] = str(default_soloist)
            else:
                cfg["soloist_binary"] = existing_path("Path to your officially downloaded Soloist executable", True)
        elif yes("Download the official ARM64 Soloist executable into this checkout now?", True):
            cfg["soloist_binary"] = download_soloist(home)
        else:
            cfg["soloist_binary"] = existing_path("Path to your officially downloaded Soloist executable", True)
        cfg["secrets"]["soloist_key"] = hidden("Personal Soloist API key", r"\S+")
        cfg["spotify"] = True
        cfg["enabled"].append("music")
        cfg["spotify_uri"] = input("Optional spotify:playlist/album/track URI (Enter for default): ").strip()
        if cfg["spotify_uri"] and not re.fullmatch(r"spotify:(track|playlist|album):[A-Za-z0-9]+", cfg["spotify_uri"]):
            raise SetupError("Invalid Spotify URI")
    for group, label in (("calling", "phone calls"), ("messaging", "SMS")):
        if yes(f"Enable real {label} (requires compatible phone and terminal confirmation for each action)?"):
            cfg["enabled"].append(group)
            cfg["secrets"][group + "_number"] = hidden("Recipient in international +country format", r"\+[1-9][0-9]{7,14}")
    if set(cfg["enabled"]) & {"calling", "messaging"}:
        cfg["secrets"]["phone_mac"] = hidden("Phone Bluetooth MAC", r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}")
        print("Pairing and phone permission prompts are completed using: swift.py bluetooth")
    if yes("Enable Piper spoken responses?"):
        print("Piper and a separately licensed .onnx voice plus matching .onnx.json are required.")
        if not yes("Have you checked and accepted the chosen voice's terms?"):
            raise SetupError("Authorized voice required")
        cfg["voice_model"] = existing_path("Voice .onnx path")
        if not Path(cfg["voice_model"] + ".json").is_file():
            raise SetupError("Missing voice configuration")
        try:
            importlib.import_module("piper")
        except ImportError:
            if sys.prefix == sys.base_prefix or not yes("Install locked Piper dependencies into this virtual environment?"):
                raise SetupError("Piper is required for spoken responses")
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
    env.update(HEY_SWIFT_RUNTIME_ROOT=str(state), HEY_SWIFT_ENABLED_INTEGRATIONS=",".join(cfg["enabled"]),
               HEY_SWIFT_CONFIRM_COMMUNICATIONS="1", VCM_SOLOIST_PROFILE="bundled",
               VCM_SOLOIST_DATA_DIR=str(state / "state/soloist"),
               VCM_SOLOIST_KEY_FILE=str(state / "state/soloist/api-key"), VCM_SOLOIST_WS="127.0.0.1:0",
               VCM_SOLOIST_INITIAL_VOLUME="25",
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


def assistant(cfg, state, model="champion", live=False, latency=False):
    profile = resolve_profile(model)
    device = microphone(cfg)
    env = isolated_env(cfg, state)
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
               "--device", device, "--hardware", cfg["hardware"] if live else "mock"]
    if not cfg.get("spotify") and not cfg.get("weather"):
        command.append("--offline")
    if live:
        command.append("--live")
    if cfg.get("speech"):
        command.append("--speak")
    return subprocess.call(command, cwd=ROOT, env=env)


def spotify(cfg, state):
    if not cfg.get("spotify"):
        raise SetupError("Spotify not enabled")
    print("Starting a separate Swift Setup device; use Spotify on your phone to select it.")
    print("A key may be visible to processes owned by this OS user; no key is printed or put in shell history.")
    with applied_env(isolated_env(cfg, state)):
        from .actions.services.music import SoloistBackend
        backend = SoloistBackend(binary=cfg["soloist_binary"], data_dir=str(state / "state/soloist"),
            auto_start=True, startup_timeout=15, control_timeout=3, max_volume_percent=25)
        backend.ensure_started()
        input("Select Swift Pi in your phone's Spotify device picker, then press Enter: ")
        try:
            backend.ctl("activate", timeout=3)
            status = backend.ctl("now", as_json=True, timeout=3)
            print("Spotify responding; active device:", bool(status.get("is_active")))
        except Exception:
            print("Authorization/activation pending. Check phone selection, Premium/key and the private Soloist log.")
            return 1
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
    for command in (("--agent", "KeyboardDisplay", "pair"), ("trust",), ("connect",)):
        if subprocess.call(["bluetoothctl", *command, mac]):
            print("Bluetooth step failed; pairing is pending. No call or SMS was initiated.")
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
    print("Only enabled integrations execute. Calls/SMS each require typed confirmation; they are optional.")
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
    sub.add_parser("doctor")
    for action in ("run", "live-latency", "demo"):
        p = sub.add_parser(action)
        p.add_argument("--model", choices=("champion", "baseline", "selected", "run043"), default="champion")
        if action == "run":
            p.add_argument("--live", action="store_true")
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
    if args.action == "evaluate":
        code = subprocess.call([sys.executable, str(ROOT / "scripts/test_pi.py"), *( ["--reset"] if args.reset else [])], cwd=ROOT)
        if code:
            return code
        costs = {}
        for name, capacity in (("baseline", "2"), ("champion", "6")):
            profile = resolve_profile(name)
            path = ROOT / "work/pi5_test_01" / (name + "_cost.json")
            code = subprocess.call([sys.executable, str(ROOT / "scripts/benchmark_onnx_pi.py"),
                "--model", str(ROOT / "models/vcm" / (profile + ".onnx")), "--audio-seconds", capacity,
                "--threads", "1", "--output", str(path)], cwd=ROOT)
            if code:
                return code
            costs[name] = json.loads(path.read_text())
        (ROOT / "work/pi5_test_01/model_costs.json").write_text(json.dumps(costs, indent=2) + "\n")
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
    return assistant(cfg, state, model=args.model, live=getattr(args, "live", False), latency=args.action == "live-latency")
