"""Masked setup for the explicit Groq Whisper/LLM + Qwen TTS demo profile."""

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
            "ME3_GROQ_API_KEY": getpass.getpass(
                "Groq API key for Whisper ASR and reasoning (hidden): "
            ),
            "ME3_QWEN_TTS_API_KEY": getpass.getpass("Qwen TTS service credential (hidden): "),
        }
    write_secrets(args.output, values)
    print("Private demo credentials saved; no inference sent. Existing files were not overwritten.")


if __name__ == "__main__":
    main()
