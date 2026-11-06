"""Secure, project-local setup for optional OnIt task integrations."""

import argparse
import getpass
from pathlib import Path

from .configuration import ConfigurationError
from .credentials import load_secrets, private_path, write_secrets

DEFAULT_SECRETS = Path("var/cache/onit-secrets.env")
ONIT_KEYS = {
    "ME3_ONIT_GITHUB_TOKEN": "GitHub token (read-only repository lookup)",
    "ME3_ONIT_TAVILY_API_KEY": "Tavily API key (bounded web search)",
    "ME3_ONIT_HF_TOKEN": "Hugging Face token (reserved for a future approved download action)",
}


def configure(path=DEFAULT_SECRETS, *, prompt=getpass.getpass):
    path = private_path(path)
    if path.exists():
        raise ConfigurationError(
            f"{path} already exists; existing secret files are never overwritten"
        )
    values = {}
    for name, label in ONIT_KEYS.items():
        value = prompt(f"{label} (Enter to skip): ").strip()
        if value:
            values[name] = value
    if not values:
        raise ConfigurationError("no keys entered; no secret file was written")
    write_secrets(path, values)
    print(f"Saved {len(values)} key(s) to {path} with owner-only permissions.")
    print("Key values were not displayed. This stores credentials; it makes no API calls.")


def status(path=DEFAULT_SECRETS):
    path = private_path(path)
    values = load_secrets(path, set(ONIT_KEYS)) if path.exists() else {}
    for name, label in ONIT_KEYS.items():
        print(f"{label}: {'configured' if name in values else 'not configured'}")
    return {name for name in values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure", help="enter optional keys through masked prompts")
    setup.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS)
    inspect = commands.add_parser("status", help="show which keys are configured, never values")
    inspect.add_argument("--secrets-file", type=Path, default=DEFAULT_SECRETS)
    args = parser.parse_args()
    try:
        if args.command == "configure":
            configure(args.secrets_file)
        else:
            status(args.secrets_file)
    except ConfigurationError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
