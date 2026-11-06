"""Side-effect-free deployment and capability inventory; never opens audio or GPIO."""

import argparse
import importlib.util
import json
import platform
import shutil
from pathlib import Path

from .configuration import load_backends
from .credentials import load_secrets
from .devices import ACTION_TITLES, ACTION_VALUES


def inventory(args):
    specs = load_backends(args.backend_config)
    credentials = {}
    for path in (args.secrets_file, args.llm_secrets_file, args.onit_secrets_file):
        if path:
            credentials.update(load_secrets(path))
    device = json.loads(args.device_config.read_text()) if args.device_config else {}
    weather = device.get("weather", {})
    model_file = Path("/proc/device-tree/model")
    is_pi = model_file.exists() and "raspberry pi" in model_file.read_text().lower()
    target = device.get("backend", "mock")
    worker = bool(args.onit_python and args.onit_python.is_file())
    from .artifacts import approved_project_root

    try:
        project_ready = bool(args.onit_output_root and approved_project_root(args.onit_output_root))
    except (ValueError, OSError):
        project_ready = False
    roles = {}
    for role, spec in specs.items():
        configured = spec.backend == "external"
        key_present = bool(credentials.get(spec.auth_env))
        roles[role] = {
            "backend": spec.backend,
            "model": spec.model,
            "protocol": spec.protocol,
            "credential_name": spec.auth_env if configured else None,
            "credential_present": key_present if configured else None,
            "request_limit": spec.request_limit,
            "state": "configured but unverified"
            if configured and key_present
            else "missing credential"
            if configured
            else "mock only",
            "access_validated": False,
        }
    capabilities = []
    for action, title in ACTION_TITLES.items():
        state = "configured but unverified"
        required = []
        if action == "get_weather":
            state = (
                "configured but unverified" if weather.get("enabled") else "missing configuration"
            )
            required = (
                [] if weather.get("enabled") else ["weather.location/latitude/longitude/timezone"]
            )
        elif action in {"call_demo", "message_demo"}:
            state = (
                "configured but unverified"
                if is_pi and device.get("communications", {}).get("enabled")
                else "missing configuration"
            )
            required = [
                "communications.phone_address",
                "communications.call_number",
                "communications.message_number",
            ]
        elif target == "pi" and not is_pi:
            state = "temporarily unavailable"
        capabilities.append(
            {
                "capability": action,
                "title": title,
                "executor": "DeviceActions",
                "target": target,
                "parameters": sorted(ACTION_VALUES.get(action, [])),
                "conversation_route": "typed shared executor; final transcript routing required",
                "worker_exposed": False,
                "state": state,
                "required_inputs": required,
                "physical_verification": False,
            }
        )
    for name, exposed, key, enabled in (
        (
            "web_search",
            worker and args.tavily_credits > 0,
            "ME3_ONIT_TAVILY_API_KEY",
            args.tavily_credits > 0,
        ),
        (
            "github_read",
            worker and args.github_readonly,
            "ME3_ONIT_GITHUB_TOKEN",
            args.github_readonly,
        ),
    ):
        capabilities.append(
            {
                "capability": name,
                "executor": "Onit worker",
                "worker_exposed": exposed,
                "credential_name": key,
                "credential_present": bool(credentials.get(key)),
                "state": "disabled"
                if not enabled
                else "missing credential"
                if not credentials.get(key)
                else "configured but unverified",
                "access_validated": False,
            }
        )
    capabilities.extend(
        [
            {
                "capability": "website_generation",
                "executor": "Onit ProjectDirectoryTool",
                "worker_exposed": worker and project_ready,
                "state": "configured but unverified"
                if worker and project_ready
                else "missing configuration",
                "required_inputs": []
                if worker and project_ready
                else ["--onit-python", "--onit-output-root"],
                "preview": "bridge artifact inspection/download; interactive preview qualification required",
            },
            {
                "capability": "music",
                "executor": "MusicController",
                "state": "configured but unverified"
                if args.music_config and args.music_config.is_file()
                else "missing configuration",
                "required_inputs": []
                if args.music_config
                else ["ME3_MUSIC_CONFIG", "explicit Spotify receiver"],
            },
            {
                "capability": "calling_messaging",
                "executor": "ME3 Communications → PipeWire telephony / BlueZ MAP",
                "state": "configured but unverified"
                if is_pi and device.get("communications", {}).get("enabled")
                else "missing configuration",
                "original_me2": "selected calling/messaging service backends; stub handlers are not the full implementation",
                "required_inputs": [
                    "communications.phone_address",
                    "communications.call_number",
                    "communications.message_number",
                    "paired phone permissions; user D-Bus services",
                ],
            },
        ]
    )
    return {
        "schema_version": 1,
        "kind": "swift-capability-preflight",
        "host": {
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "raspberry_pi": is_pi,
            "audio_devices_present": Path("/dev/snd").exists(),
        },
        "utilities": {
            name: bool(shutil.which(name))
            for name in ("pw-record", "pw-play", "wpctl", "aplay", "arecord", "ssh")
        },
        "gpio_libraries": {
            name: importlib.util.find_spec(name) is not None for name in ("gpiozero", "lgpio")
        },
        "roles": roles,
        "capabilities": capabilities,
        "onit": {
            "interpreter_present": worker,
            "tavily_credit_limit": args.tavily_credits,
            "full_installed_harness_tools_enabled": False,
            "note": "Only explicit scoped tools are registered. Native shell/deployment tools are not exposed.",
        },
        "readiness": "degraded" if not is_pi else "unverified",
        "limitations": "Inventory only: zero network requests, inference, audio capture/playback or GPIO initialization. Credential presence is not access validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, help="omit for offline mock inventory")
    parser.add_argument("--secrets-file", type=Path)
    parser.add_argument("--llm-secrets-file", type=Path)
    parser.add_argument("--onit-secrets-file", type=Path)
    parser.add_argument("--device-config", type=Path)
    parser.add_argument("--music-config", type=Path)
    parser.add_argument("--onit-python", type=Path)
    parser.add_argument("--onit-output-root", type=Path)
    parser.add_argument("--tavily-credits", type=int, default=0)
    parser.add_argument("--github-readonly", action="store_true")
    args = parser.parse_args()
    try:
        report = inventory(args)
    except Exception as exc:
        parser.exit(
            2,
            f"Preflight failed ({type(exc).__name__}); inspect configuration paths and private-file permissions. Values excluded.\n",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
