"""Create a private provider-qualified ME3 secrets file with masked prompts."""

import argparse
import getpass
from pathlib import Path
import sys
import warnings

from me3_voice.credentials import write_secrets

PROFILES = {
    "deepgram-groq": {
        "ME3_DEEPGRAM_API_KEY": "Deepgram API key for Nova-3 ASR and Aura-2 TTS",
        "ME3_GROQ_API_KEY": "Groq API key for openai/gpt-oss-120b",
    },
    "qwen": {
        "ME3_QWEN_ASR_API_KEY": "Qwen ASR service credential (confirm the configured model first)",
        "ME3_QWEN_LLM_API_KEY": "Qwen LLM service credential (confirm the configured model first)",
        "ME3_QWEN_TTS_API_KEY": "Qwen TTS service credential (confirm the configured model first)",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--output", type=Path, default=Path("var/cache/provider-keys.env"))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("use an interactive terminal for masked key entry")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        values = {
            name: getpass.getpass(f"{label} (hidden): ")
            for name, label in PROFILES[args.profile].items()
        }
    write_secrets(args.output, values)
    print(
        f"Private provider-qualified credentials saved to {args.output}; no API request was sent."
    )


if __name__ == "__main__":
    main()
