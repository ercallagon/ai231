"""Prepare NEW explicit demo profiles; never change an existing configuration."""

import argparse
import json
import os
from pathlib import Path

from me3_voice.configuration import load_backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--asr", choices=("qwen", "groq"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pi", action="store_true")
    parser.add_argument(
        "--enable-inference",
        action="store_true",
        help="explicitly enable the generated profile; never sends inference during setup",
    )
    args = parser.parse_args()
    if not args.name.isidentifier():
        parser.error("name must be a simple identifier")
    load_backends(args.baseline)
    backends = json.loads(args.baseline.read_text())
    if args.enable_inference:
        for spec in backends.values():
            spec["allow_live"] = True
    llm = backends["llm"]
    if (
        llm["endpoint"] != "https://api.groq.com/openai/v1/chat/completions"
        or llm["model"] != "openai/gpt-oss-120b"
    ):
        parser.error("baseline must be the proven Groq GPT-OSS 120B profile")
    llm["request_limit"] = None  # Explicit demo profile; provider limits still apply.
    if args.asr == "groq":
        backends["asr"].update(
            protocol="groq-whisper-v1",
            model="whisper-large-v3-turbo",
            endpoint="https://api.groq.com/openai/v1/audio/transcriptions",
            auth_env=llm["auth_env"],
            transcription_mode="prompt",
            streaming=False,
            allow_private_http=False,
            request_limit=None,
        )
    cache = Path("var/cache")
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / f"{args.name}-backends.json", cache / f"{args.name}-devices.json"]
    if any(p.exists() for p in paths):
        parser.error("profile paths already exist; choose a fresh name")
    devices = {
        "schema_version": 1,
        "backend": "pi" if args.pi else "mock",
        "reminder_store": "var/cache/device-state/reminders.json",
        "weather": {
            "provider": "open_meteo",
            "enabled": True,
            "latitude": 14.6488,
            "longitude": 121.0509,
            "timezone": "Asia/Manila",
            "location": "Quezon City, Philippines",
        },
        "communications": {"enabled": False},
    }
    for path, content in zip(paths, (backends, devices)):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(content, stream, indent=2)
            stream.write("\n")
    load_backends(paths[0])
    print("Prepared:", *(str(p) for p in paths))
    print("Groq120B LOW/streaming/2048 must still be selected with the existing bridge flags.")
    print(
        "No service started, credentials copied, or inference sent. Pi/audio/access qualification remains pending."
    )


if __name__ == "__main__":
    main()
