"""Masked setup for the explicit Groq ASR/LLM + university TTS demo profile."""

import argparse
import getpass
import sys
import warnings
from pathlib import Path

from me3_voice.credentials import write_secrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("var/cache/demo-models.env"))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("use an interactive terminal for masked key entry")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        values = {
            "ME3_LLM_API_KEY": getpass.getpass("Groq API key for ASR and reasoning (hidden): "),
            "ME3_TTS_API_KEY": getpass.getpass("University TTS API key (hidden): "),
        }
    write_secrets(args.output, values)
    print("Private demo credentials saved; no inference sent. Existing files were not overwritten.")


if __name__ == "__main__":
    main()
