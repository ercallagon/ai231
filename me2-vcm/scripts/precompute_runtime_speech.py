#!/usr/bin/env python3
"""Render fixed demo responses into a new, voice-fingerprinted directory."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hey_swift.audio.speech import PiperSpeech
from hey_swift.integration.runtime import _common_speech_phrases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="work/integration/assets/voices/en_US-amy-medium.onnx")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    voice, output = Path(args.voice), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    speech = PiperSpeech(voice)
    phrases = {}
    try:
        speech.prepare()
        for text in _common_speech_phrases():
            path = speech._audio_path(text)
            dest = output / path.name
            shutil.copy2(path, dest)
            with wave.open(str(dest), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            phrases[text] = {"file": dest.name, "sha256": digest(dest), "audio_seconds": duration}
            print(f"{len(phrases)} {duration:.2f}s {text}", flush=True)
    finally:
        speech.close()
    manifest = {"voice_sha256": digest(voice),
                "voice_config_sha256": digest(Path(str(voice) + ".json")),
                "python": sys.version, "platform": platform.platform(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "phrases": phrases}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(phrases)} responses; longest {max(p['audio_seconds'] for p in phrases.values()):.2f}s")


if __name__ == "__main__":
    main()
