#!/usr/bin/env python3
"""Incrementally provision the existing approval-gated GitHub publisher."""

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from me3_voice.credentials import load_secrets, write_secrets  # noqa: E402


CACHE = ROOT / "var/cache"
PROVIDER_FILE = CACHE / "selected-provider-credentials.env"
TOKEN_FILE = CACHE / "github-write-token.env"
CONFIG_FILE = CACHE / "github-write.json"
PROVIDER_KEYS = {
    "OpenRouter": "ME3_OPENROUTER_API_KEY",
    "Deepgram": "ME3_DEEPGRAM_API_KEY",
    "Tavily": "ME3_RESEARCH_TAVILY_API_KEY",
}


def _read_private_json(path):
    if not path.exists():
        return None
    if path.is_symlink():
        raise SystemExit("GitHub configuration must not be a symlink.")
    try:
        status = path.stat()
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.getuid()
            or status.st_mode & 0o077
            or status.st_size > 8192
        ):
            raise ValueError
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise SystemExit(
            "GitHub configuration is unreadable or not private; no changes made."
        ) from None
    if not isinstance(value, dict):
        raise SystemExit("GitHub configuration is invalid; no changes made.")
    return value


def _atomic_private_write(path, payload):
    CACHE.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise SystemExit("Refusing to update a symlinked GitHub credential file.")
    temporary = CACHE / f".{path.name}.{os.getpid()}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            "Could not update the private GitHub configuration; existing files are preserved."
        ) from None


def _provider_status():
    try:
        values = load_secrets(PROVIDER_FILE, set(PROVIDER_KEYS.values()))
    except Exception:
        values = {}
    print("Existing provider credentials (values hidden):")
    for service, key in PROVIDER_KEYS.items():
        print(f"  {service}: {'configured' if values.get(key) else 'missing'}")


def _github_token():
    try:
        return load_secrets(TOKEN_FILE, {"ME3_GITHUB_WRITE_TOKEN"}).get("ME3_GITHUB_WRITE_TOKEN")
    except Exception:
        return None


def _repository_details(previous=None):
    owner = input(f"GitHub account or organization [{(previous or {}).get('owner', '')}]: ").strip()
    owner = owner or (previous or {}).get("owner", "")
    repository = input(f"Repository name [{(previous or {}).get('repository', '')}]: ").strip()
    repository = repository or (previous or {}).get("repository", "")
    base_branch = input(f"Base branch [{(previous or {}).get('base_branch', 'main')}]: ").strip()
    base_branch = base_branch or (previous or {}).get("base_branch", "main")
    if (
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", owner)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", repository)
        or not re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", base_branch)
        or ".." in base_branch
    ):
        raise SystemExit("Invalid GitHub repository details; no configuration written.")
    return {
        "schema_version": 1,
        "owner": owner,
        "repository": repository,
        "base_branch": base_branch,
    }


def _save_repository(config):
    previous = _read_private_json(CONFIG_FILE)
    if previous:
        # Keep optional existing publisher settings when changing only the repo.
        for key, value in previous.items():
            config.setdefault(key, value)
        _atomic_private_write(
            CONFIG_FILE,
            (json.dumps(config, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        )
    else:
        fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")


def _save_token(token, *, replace=False):
    values = {}
    if TOKEN_FILE.exists():
        values = load_secrets(TOKEN_FILE)
        if not replace:
            raise SystemExit(
                "A GitHub token is already configured; use --github-token to update only it."
            )
    values["ME3_GITHUB_WRITE_TOKEN"] = token
    if replace:
        body = "".join(f"{key}={value}\n" for key, value in sorted(values.items())).encode()
        _atomic_private_write(TOKEN_FILE, body)
    else:
        write_secrets(TOKEN_FILE, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    updates = parser.add_mutually_exclusive_group()
    updates.add_argument(
        "--github-token", action="store_true", help="add or update only the hidden token"
    )
    updates.add_argument(
        "--github-repository", action="store_true", help="add or update only repository settings"
    )
    args = parser.parse_args()

    if os.environ.get("ME3_SETUP_FROM_WIZARD") != "1":
        _provider_status()
    config = _read_private_json(CONFIG_FILE)
    token = _github_token()
    if args.github_token:
        value = getpass.getpass("Fine-grained GitHub token (hidden): ").strip()
        if not value:
            raise SystemExit("No token entered; existing configuration was not changed.")
        _save_token(value, replace=bool(token))
        print("GitHub token saved privately. Repository settings were not changed.")
    elif args.github_repository:
        _save_repository(_repository_details(config))
        print("Repository settings saved privately. GitHub token was not changed.")
    else:
        if config is None:
            _save_repository(_repository_details())
            config = _read_private_json(CONFIG_FILE)
        if not token:
            value = getpass.getpass("Fine-grained GitHub token (hidden): ").strip()
            if not value:
                raise SystemExit("No token entered; repository settings were preserved.")
            _save_token(value)
        else:
            print("GitHub write token: already configured (value hidden)")

    print(
        "GitHub configuration is repository-scoped. No remote read, write or deployment was performed."
    )
    print(
        "Grant Contents: write for branches; add Pages: write and Administration: write only if Pages is needed."
    )
    print(
        "The bridge loads these private files at startup. Branch publication and public Pages remain approval-gated."
    )


if __name__ == "__main__":
    main()
