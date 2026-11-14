"""Masked setup for optional Phase 5 task-model and Tavily credentials."""

import argparse
import getpass
from pathlib import Path
import sys
import warnings

from me3_voice.credentials import write_secrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("var/cache/phase5-secrets.env"))
    parser.add_argument(
        "--include-tavily",
        action="store_true",
        help="also prompt for read-only Phase 4/5 research; live runtime authorization remains separate",
    )
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("use an interactive terminal for masked key entry")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        values = {"ME3_GROQ_API_KEY": getpass.getpass("Groq API key (hidden): ")}
        if args.include_tavily:
            values["ME3_RESEARCH_TAVILY_API_KEY"] = getpass.getpass("Tavily API key (hidden): ")
    write_secrets(args.output, values)
    print("Private Phase 5 credentials saved; no provider request was sent.")


if __name__ == "__main__":
    main()
