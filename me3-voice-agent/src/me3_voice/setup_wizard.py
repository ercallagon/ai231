"""One guided, offline setup for Swift's active production integrations."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from .communications import normalize_contacts, validate_bluetooth_address
from .configuration import ConfigurationError, load_backends
from .credentials import load_secrets, update_secrets

ROOT = Path(__file__).resolve().parents[2]
DEVICE_FIELDS = {"schema_version", "backend", "reminder_store", "weather", "communications"}


def _is_pi():
    try:
        return "raspberry pi" in Path("/proc/device-tree/model").read_text().casefold()
    except OSError:
        return False


def select_paths(profile="auto", root=ROOT):
    """Resolve the same backend/credential files consumed by the bridge."""
    pi_backend = root / "var/cache/pi_live-backends.json"
    if profile == "auto":
        profile = "pi" if _is_pi() or pi_backend.is_file() else "laptop"
    if profile not in {"pi", "laptop"}:
        raise ValueError("profile must be auto, pi, or laptop")
    backend = pi_backend if profile == "pi" else root / "configs/native-demo-backends.json"
    secrets = root / (
        "var/cache/demo-models.env"
        if profile == "pi"
        else "var/cache/selected-provider-credentials.env"
    )
    override = os.environ.get("ME3_SETUP_SECRETS_FILE")
    if override:
        secrets = Path(override)
        if not secrets.is_absolute():
            secrets = root / secrets
    device = root / (
        "var/cache/pi_live-devices.json" if profile == "pi" else "var/cache/devices.json"
    )
    return {"profile": profile, "backend": backend, "secrets": secrets, "device": device}


def _safe_secrets(path):
    if not path.exists():
        return {}
    return load_secrets(path)


def _status(paths, *, root=ROOT):
    specs = load_backends(paths["backend"], runnable=False)
    secrets = _safe_secrets(paths["secrets"])
    print(f"Swift setup profile: {paths['profile']}")
    print(
        f"Backend configuration: {paths['backend'].relative_to(root) if paths['backend'].is_relative_to(root) else paths['backend']}"
    )
    for role, spec in specs.items():
        host = urlsplit(spec.endpoint).hostname or "local/mock"
        configured = spec.backend == "mock" or bool(secrets.get(spec.auth_env))
        print(
            f"{role}: {spec.model or spec.backend} at {host}; credential {'configured' if configured else 'missing'}; access unverified"
        )
    runtime_path = root / "configs/native-demo-runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
    research_enabled = bool(runtime.get("research", {}).get("allow_live"))
    print(
        "native research: "
        + ("configured" if secrets.get("ME3_RESEARCH_TAVILY_API_KEY") else "missing key")
        + ("; runtime enabled" if research_enabled else "; runtime disabled")
    )
    for label, path in (
        ("GitHub repository", root / "var/cache/github-write.json"),
        ("GitHub write token", root / "var/cache/github-write-token.env"),
        ("Spotify OAuth", root / "var/cache/music.json"),
        ("Spotify token file", root / "var/cache/spotify.env"),
    ):
        print(f"{label}: {'configured' if path.is_file() else 'missing'}")
    soloist = (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "swift-me3/soloist-api-key"
    )
    print(f"Soloist key: {'configured' if soloist.is_file() else 'missing'}")
    device = _read_device(paths["device"], root=root)
    communications = device.get("communications", {})
    contacts = communications.get("contacts", [])
    print(
        f"device: {device.get('backend', 'mock')}; contacts {len(contacts) if isinstance(contacts, list) else 0}; "
        f"phone {'configured' if communications.get('phone_address') else 'missing'}; "
        f"calls/SMS {'enabled' if communications.get('enabled') else 'disabled'}; "
        f"weather {'enabled' if device.get('weather', {}).get('enabled') else 'disabled'}"
    )
    print("No connectivity checks or provider requests were made.")
    return {
        "backend_specs": specs,
        "secrets": secrets,
        "device": device,
        "research_enabled": research_enabled,
    }


def _read_device(path, *, root=ROOT):
    path = Path(path)
    if path.exists():
        _private_cache_path(path, root=root)
        if path.is_symlink():
            raise ValueError("device configuration must not be a symlink")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or set(value) - DEVICE_FIELDS
        ):
            raise ValueError("unsupported device configuration")
        return value
    example = root / "configs/devices.example.json"
    return json.loads(example.read_text(encoding="utf-8"))


def _private_cache_path(path, *, root=ROOT):
    path = Path(path).absolute()
    cache = (root / "var/cache").resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(cache):
        raise ValueError("configuration must be a regular file under project var/cache/")
    return path


def _atomic_json(path, value, *, root=ROOT):
    path = _private_cache_path(path, root=root)
    initial = path.stat(follow_symlinks=False) if path.exists() else None
    if initial and (not stat.S_ISREG(initial.st_mode) or initial.st_mode & 0o077):
        raise ValueError("configuration file must be owner-only")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if initial is None:
            if path.exists() or path.is_symlink():
                raise ValueError("configuration appeared during update; no changes made")
        else:
            current = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_uid != os.getuid()
                or current.st_mode & 0o077
                or (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino)
            ):
                raise ValueError("configuration changed during update; no changes made")
        os.replace(temporary, path)
        temporary = ""
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary:
            os.unlink(temporary)


def _prompt_action(label, *, present=False):
    default = "keep" if present else "skip"
    answer = (
        input(f"{label} [{default}; enter=keep/skip, a=add, u=update, s=skip]: ").strip().casefold()
    )
    if not answer:
        return "keep" if present else "skip"
    return {"a": "add", "u": "update", "s": "skip", "k": "keep"}.get(answer, "invalid")


def _configure_provider_keys(paths, status, *, secret_prompt=getpass.getpass):
    specs = status["backend_specs"]
    existing = status["secrets"]
    required = {}
    for spec in specs.values():
        if spec.backend == "external" and spec.allow_live and spec.auth_env:
            required[spec.auth_env] = "active ASR/LLM/TTS profile"
    if status["research_enabled"]:
        required["ME3_RESEARCH_TAVILY_API_KEY"] = "native Tavily research"
    updates = {}
    for name, label in required.items():
        current = name in existing
        action = _prompt_action(
            f"{label} credential {name} ({'present' if current else 'missing'})", present=current
        )
        if action == "invalid":
            print("Unrecognized choice; keeping current value.")
            continue
        if action in {"add", "update"} and (action == "update" or not current):
            value = secret_prompt(f"Enter {name} (hidden; blank keeps/skips): ").strip()
            if value:
                updates[name] = value
    if updates:
        update_secrets(paths["secrets"], updates)
        print(
            f"Updated {len(updates)} selected credential(s) in {paths['secrets'].relative_to(ROOT)}; values hidden."
        )
    else:
        print("Provider credentials unchanged.")


def _validate_device(config, path):
    if config.get("schema_version") != 1 or set(config) - DEVICE_FIELDS:
        raise ValueError("unsupported device configuration")
    if config.get("backend", "mock") not in {"mock", "pi"}:
        raise ValueError("device backend must be mock or pi")
    comm = config.get("communications", {})
    if not isinstance(comm, dict) or set(comm) - {
        "enabled",
        "phone_address",
        "contacts",
        "call_number",
        "message_number",
    }:
        raise ValueError("invalid phone configuration fields")
    contacts = normalize_contacts(comm.get("contacts", []))
    if comm.get("enabled") and (
        config.get("backend") != "pi" or not contacts or not comm.get("phone_address")
    ):
        raise ValueError(
            "calls/SMS require Pi target, paired-phone address and at least one contact"
        )
    if comm.get("enabled") or contacts or comm.get("phone_address"):
        if config.get("backend") != "pi":
            raise ValueError("phone configuration is allowed only in the Pi device profile")
        if Path(path).exists() and Path(path).stat().st_mode & 0o077:
            raise ValueError("phone configuration must remain owner-only")
        if comm.get("phone_address"):
            validate_bluetooth_address(comm["phone_address"])
    weather = config.get("weather")
    if weather is not None:
        if not isinstance(weather, dict) or set(weather) - {
            "provider",
            "enabled",
            "latitude",
            "longitude",
            "timezone",
            "location",
        }:
            raise ValueError("invalid weather fields")
        if weather.get("provider") != "open_meteo" or type(weather.get("enabled")) is not bool:
            raise ValueError("weather requires provider=open_meteo and an explicit enabled value")
        if weather["enabled"]:
            lat, lon = weather.get("latitude"), weather.get("longitude")
            if (
                type(lat) not in (float, int)
                or type(lon) not in (float, int)
                or not math.isfinite(lat)
                or not math.isfinite(lon)
                or not -90 <= lat <= 90
                or not -180 <= lon <= 180
            ):
                raise ValueError("weather coordinates are invalid")
            for field in ("timezone", "location"):
                if not isinstance(weather.get(field), str) or not weather[field].strip():
                    raise ValueError(f"weather {field} is required when enabled")
    return config


def _private_prompt(label, current=None, *, prompt=getpass.getpass):
    state = "configured; blank keeps" if current else "missing; blank skips"
    value = prompt(f"{label} ({state}; enter ! to clear): ").strip()
    if value == "!":
        return ""
    return current if not value else value


def _configure_contacts(config, *, secret_prompt=getpass.getpass):
    comm = dict(config.get("communications", {}))
    contacts = [dict(row) for row in comm.get("contacts", [])]
    action = (
        input(f"Contacts: {len(contacts)} configured; a=append, u=update, r=remove, Enter=keep: ")
        .strip()
        .casefold()
    )
    if action == "a":
        name = secret_prompt("Contact name (hidden): ").strip()
        aliases_text = secret_prompt("Aliases, comma-separated (hidden; optional): ").strip()
        number = secret_prompt("Phone number (hidden): ").strip()
        contacts.append(
            {
                "name": name,
                "aliases": [x.strip() for x in aliases_text.split(",") if x.strip()],
                "phone_number": number,
            }
        )
    elif action == "u" and contacts:
        index_text = input(f"Contact number to update (1-{len(contacts)}; names hidden): ").strip()
        if index_text.isdigit() and 1 <= int(index_text) <= len(contacts):
            row = dict(contacts[int(index_text) - 1])
            for field, label in (
                ("name", "Contact name"),
                ("aliases", "Aliases, comma-separated"),
                ("phone_number", "Phone number"),
            ):
                old = ", ".join(row.get(field, [])) if field == "aliases" else row.get(field, "")
                value = _private_prompt(label, old, prompt=secret_prompt)
                row[field] = (
                    [x.strip() for x in value.split(",") if x.strip()]
                    if field == "aliases"
                    else value
                )
            contacts[int(index_text) - 1] = row
    elif action == "r" and contacts:
        index_text = input(f"Contact number to remove (1-{len(contacts)}; names hidden): ").strip()
        if (
            index_text.isdigit()
            and 1 <= int(index_text) <= len(contacts)
            and input("Confirm removal? [y/N]: ").strip().casefold() == "y"
        ):
            contacts.pop(int(index_text) - 1)
    elif action not in {"", "k", "s"}:
        print("No contact change made.")
    if action in {"a", "u", "r"}:
        comm["contacts"] = normalize_contacts(contacts)
    if (
        "phone_address" in comm
        or input("Configure paired phone address? [y/N]: ").strip().casefold() == "y"
    ):
        value = _private_prompt(
            "Paired phone Bluetooth address", comm.get("phone_address"), prompt=secret_prompt
        )
        if value:
            comm["phone_address"] = value
        else:
            comm.pop("phone_address", None)
    if "enabled" in comm or comm.get("contacts"):
        current = "y" if comm.get("enabled") else "n"
        answer = (
            input(
                f"Enable calls/SMS after Bluetooth pairing? [y/n; current {current}; Enter=keep]: "
            )
            .strip()
            .casefold()
        )
        if answer in {"y", "n"}:
            comm["enabled"] = answer == "y"
    config["communications"] = comm
    return config


def _configure_devices(paths, *, root=ROOT):
    if (
        input("Configure contacts, weather, timers and Pi device target? [y/N]: ")
        .strip()
        .casefold()
        != "y"
    ):
        print("Device settings unchanged.")
        return
    config = _read_device(paths["device"], root=root)
    current_backend = config.get("backend", "mock")
    answer = (
        input(f"Device backend mock/pi [current {current_backend}; Enter=keep]: ")
        .strip()
        .casefold()
    )
    if answer in {"mock", "pi"} and answer != current_backend:
        if (
            answer != "pi"
            or input(
                "Type PI to confirm configuring real Pi actions (setup does not open GPIO): "
            ).strip()
            == "PI"
        ):
            config["backend"] = answer
    reminder = input(
        f"Reminder store path [{config.get('reminder_store', 'var/cache/device-state/reminders.json')}; Enter=keep]: "
    ).strip()
    if reminder:
        config["reminder_store"] = reminder
    weather = dict(config.get("weather", {"provider": "open_meteo", "enabled": False}))
    if (
        input(
            f"Configure weather/location fields? [y/N; current {'enabled' if weather.get('enabled') else 'disabled'}]: "
        )
        .strip()
        .casefold()
        == "y"
    ):
        enabled = (
            input(
                f"Enable weather lookup? [y/n; current {'y' if weather.get('enabled') else 'n'}; Enter=keep]: "
            )
            .strip()
            .casefold()
        )
        if enabled in {"y", "n"}:
            weather["enabled"] = enabled == "y"
        for field, label in (
            ("location", "Location"),
            ("latitude", "Latitude"),
            ("longitude", "Longitude"),
            ("timezone", "Timezone"),
        ):
            old = weather.get(field)
            prompt = getpass.getpass if field == "location" else input
            value = _private_prompt(label, str(old) if old is not None else None, prompt=prompt)
            if value == "":
                weather.pop(field, None)
            elif value is not None:
                if field in {"latitude", "longitude"}:
                    value = float(value)
                weather[field] = value
        weather["provider"] = "open_meteo"
        config["weather"] = weather
    config = _configure_contacts(config)
    _validate_device(config, paths["device"])
    _atomic_json(paths["device"], config, root=root)
    print(
        f"Private device profile updated at {paths['device'].relative_to(root)}; no hardware was opened."
    )


def _configure_soloist_key(*, secret_prompt=getpass.getpass):
    path = (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "swift-me3/soloist-api-key"
    )
    if path.is_symlink() or (
        path.exists()
        and (not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077)
    ):
        raise ValueError("Soloist key file must be a private regular file")
    present = path.is_file()
    action = _prompt_action("Soloist receiver API key", present=present)
    if action not in {"add", "update"} or (action == "add" and present):
        print("Soloist key unchanged.")
        return
    value = secret_prompt("Soloist API key from Spotify for Developers (hidden): ").strip()
    if not value:
        print("Soloist key unchanged.")
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".soloist-api-key.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            os.unlink(temporary)
    print(
        "Soloist key saved privately; value hidden. Start its receiver with bash scripts/run_soloist_receiver.sh."
    )


def _configure_integrations(*, root=ROOT):
    github = root / "var/cache/github-write-token.env"
    repo = root / "var/cache/github-write.json"
    if (
        input(
            f"Configure GitHub publishing? [y/N; repo {'configured' if repo.is_file() else 'missing'}, token {'configured' if github.is_file() else 'missing'}]: "
        )
        .strip()
        .casefold()
        == "y"
    ):
        script = root / "scripts/setup_github_write.py"
        if input("Update repository owner/name/base branch? [y/N]: ").strip().casefold() == "y":
            subprocess.run(
                [sys.executable, "-B", str(script), "--github-repository"],
                cwd=root,
                env=dict(os.environ, ME3_SETUP_FROM_WIZARD="1"),
                check=True,
            )
        if input("Add or replace GitHub write token? [y/N]: ").strip().casefold() == "y":
            subprocess.run(
                [sys.executable, "-B", str(script), "--github-token"],
                cwd=root,
                env=dict(os.environ, ME3_SETUP_FROM_WIZARD="1"),
                check=True,
            )
    if input("Run Spotify OAuth setup/renewal now? [y/N]: ").strip().casefold() == "y":
        current_name = ""
        music_path = root / "var/cache/music.json"
        if music_path.is_file():
            try:
                current_name = json.loads(music_path.read_text()).get("device_name", "") or ""
            except (OSError, ValueError, AttributeError):
                current_name = ""
        default_name = current_name or "Swift Soloist"
        name = input(f"Spotify Connect receiver name [{default_name}; Enter=keep]: ").strip()
        subprocess.run(
            ["bash", str(root / "scripts/setup_spotify_soloist.sh")],
            cwd=root,
            env=dict(os.environ, SPOTIFY_DEVICE_NAME=name or default_name),
            check=True,
        )
    _configure_soloist_key()


def interactive(paths, *, root=ROOT):
    state = _status(paths, root=root)
    print(
        "For each prompt: Enter keeps an existing value or skips a missing one; choose add/update to enter a secret hidden."
    )
    if (
        input("Configure active ASR/LLM/TTS credentials and native research? [y/N]: ")
        .strip()
        .casefold()
        == "y"
    ):
        _configure_provider_keys(paths, state)
    _configure_integrations(root=root)
    _configure_devices(paths, root=root)
    print(
        "Setup pass complete. Provider status is configuration-only; no provider checks were sent."
    )
    _status(paths, root=root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--interactive", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--profile", choices=("auto", "laptop", "pi"), default="auto")
    args = parser.parse_args(argv)
    try:
        paths = select_paths(args.profile)
        if not paths["backend"].is_file():
            raise ConfigurationError(
                f"active {paths['profile']} backend profile is missing; no files changed"
            )
        if args.status:
            _status(paths)
        else:
            interactive(paths)
    except (ConfigurationError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Setup stopped ({type(exc).__name__}); existing files were preserved.\n")


if __name__ == "__main__":
    main()
