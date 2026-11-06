"""Class DGX3 profile, read-only discovery and masked terminal credential setup."""

import argparse
import asyncio
import getpass
import json
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import aiohttp

from .artifacts import output_path
from .configuration import BackendSpec, ConfigurationError
from .credentials import load_secrets, private_path, valid_key, write_secrets

HOST = "10.153.60.233"


def profile(*, enable=False, request_limit=0, voice="ryan", language="English"):
    return {
        "asr": BackendSpec(
            backend="external",
            protocol="vllm-qwen-audio-v1",
            endpoint=f"http://{HOST}:8000/v1/chat/completions",
            auth_env="ME3_ASR_API_KEY",
            model="Qwen/Qwen2-Audio-7B-Instruct",
            streaming=False,
            timeout_seconds=60,
            allow_live=enable,
            request_limit=request_limit if enable else 0,
            allow_private_http=True,
        ),
        "tts": BackendSpec(
            backend="external",
            protocol="vllm-qwen-tts-v1",
            endpoint=f"http://{HOST}:8001/v1/audio/speech",
            auth_env="ME3_TTS_API_KEY",
            model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            timeout_seconds=60,
            allow_live=enable,
            request_limit=request_limit if enable else 0,
            allow_private_http=True,
            voice=voice,
            language=language,
        ),
        "llm": BackendSpec(),
    }


async def inspect(output, secrets_file=None):
    """GET only: no inference, metrics scrape, model load or service mutation."""
    keys = load_secrets(secrets_file) if secrets_file else {}
    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6), trust_env=False, cookie_jar=aiohttp.DummyCookieJar()
    ) as client:
        for port, role in ((8000, "asr"), (8001, "tts")):
            for route in ("/health", "/openapi.json", "/v1/models"):
                headers = {"Accept": "application/json"}
                if keys.get(f"ME3_{role.upper()}_API_KEY"):
                    headers["Authorization"] = "Bearer " + keys[f"ME3_{role.upper()}_API_KEY"]
                record = {"port": port, "path": route}
                try:
                    async with client.get(
                        f"http://{HOST}:{port}{route}", headers=headers, allow_redirects=False
                    ) as response:
                        record["status"] = response.status
                        if response.status == 200 and route != "/health":
                            raw = bytearray()
                            async for chunk in response.content.iter_chunked(8192):
                                raw.extend(chunk)
                                if len(raw) > 2000000:
                                    raise ValueError
                            value = json.loads(raw)
                            if route == "/openapi.json":
                                destination = Path(f"var/cache/dgx3-{port}-openapi.json")
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                destination.write_text(json.dumps(value, indent=2) + "\n")
                                record["routes"] = list(value.get("paths", {}))
                            else:
                                record["models"] = [
                                    row.get("id")
                                    for row in value.get("data", [])
                                    if isinstance(row, dict)
                                ]
                except Exception as exc:
                    record["error_type"] = type(exc).__name__
                results.append(record)
    output.write_text(
        json.dumps(
            {
                "kind": "dgx3-discovery",
                "mode": "read-only-service-inspection",
                "host": HOST,
                "checks": results,
                "inference_submitted": False,
            },
            indent=2,
        )
        + "\n"
    )
    for record in results:
        print(
            f"{record['port']} {record['path']}: {record.get('status', record.get('error_type'))}"
        )
    print(f"Read-only result: {output}; no inference submitted")


def configure(args):
    if not sys.stdin.isatty():
        raise ConfigurationError(
            "secure setup requires an interactive terminal; key entry is never echoed or accepted through chat/arguments"
        )
    secrets = private_path(args.secrets_file)
    destination = private_path(args.profile)
    if secrets.resolve() == destination.resolve():
        raise ConfigurationError("profile and secret file must have different paths")
    if secrets.exists() or destination.exists():
        raise ConfigurationError(
            "setup refuses to overwrite existing profile or keys; choose fresh paths explicitly"
        )
    values = profile(
        enable=args.enable_inference,
        request_limit=args.request_limit,
        voice=args.voice,
        language=args.language,
    )
    for role, spec in values.items():
        spec.validate(role, runnable=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            asr_key = getpass.getpass("DGX3 ASR API key (hidden): ")
            tts_key = (
                getpass.getpass("DGX3 TTS API key (hidden; Enter reuses ASR key): ") or asr_key
            )
    except getpass.GetPassWarning:
        raise ConfigurationError("this terminal cannot mask key entry; nothing saved") from None
    if not valid_key(asr_key) or not valid_key(tts_key):
        raise ConfigurationError("invalid key input; nothing saved")
    write_secrets(secrets, {"ME3_ASR_API_KEY": asr_key, "ME3_TTS_API_KEY": tts_key})
    try:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({role: asdict(spec) for role, spec in values.items()}, stream, indent=2)
            stream.write("\n")
    except OSError:
        raise ConfigurationError(
            "keys saved privately, but profile was not created; inspect destination without printing keys"
        ) from None
    print(f"Private keys saved to {secrets}; profile saved to {destination}")
    print("No inference submitted. Reasoning remains mock; configure its own backend separately.")
    print(
        "Shared inference enabled with per-role request bounds."
        if args.enable_inference
        else "Inference remains disabled; an explicit enable-inference setup is required before requests."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser(
        "inspect", help="read-only health/schema/model-list checks; no inference"
    )
    probe.add_argument("--output", type=output_path, default="var/results/dgx3-discovery.json")
    probe.add_argument("--secrets-file", type=Path)
    setup = commands.add_parser(
        "configure", help="masked terminal key entry; no inference or server changes"
    )
    setup.add_argument("--secrets-file", type=Path, default=Path("var/cache/dgx3.env"))
    setup.add_argument("--profile", type=Path, default=Path("var/cache/dgx3-backends.json"))
    setup.add_argument(
        "--enable-inference",
        action="store_true",
        help="explicitly authorize bounded calls to the supplied shared service; makes no calls during setup",
    )
    setup.add_argument("--request-limit", type=int, default=5)
    setup.add_argument("--voice", default="ryan")
    setup.add_argument("--language", default="English")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            asyncio.run(inspect(args.output, args.secrets_file))
        else:
            if not 1 <= args.request_limit <= 100:
                raise ConfigurationError("request-limit must be 1..100 per role/run")
            configure(args)
    except (ConfigurationError, KeyboardInterrupt) as exc:
        parser.error(str(exc) if isinstance(exc, ConfigurationError) else "setup cancelled")


if __name__ == "__main__":
    main()
