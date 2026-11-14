#!/usr/bin/env python3
"""Generate a single Swift AEC reference with Deepgram Aura-2 Thalia.

Uses exactly ONE live Deepgram Speak v1 request, with no retries.
The model's WAV is preserved; ffmpeg converts it to browser-friendly WebM.
Run from the ME3 project root; no Swift bridge or speech MCP required.
"""
import argparse
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

MODEL = "aura-2-thalia-en"
TEXT = (
    "Of course, I can help with that. Let's start with a quick overview of what "
    "you can do with Swift. You can ask a question, set a timer, check the time, "
    "or adjust the lights using a simple voice command. When I respond, I aim to "
    "sound clear, calm, and conversational, just like a helpful assistant at your desk. "
    "You should also be able to speak while I'm answering. If you want to change "
    "your request, or ask a follow-up question, you can interrupt at any point. "
    "In this audio test, we're checking whether your microphone hears you clearly "
    "without picking up too much of my voice from the nearby speaker."
)

def dotenv_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "DEEPGRAM_API_KEY":
            return value.strip().strip('"\'') or None
    return None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--secrets-file", type=Path, default=Path("var/cache/provider-keys.env"))
    p.add_argument("--output-dir", type=Path, default=Path("var/results/aec"))
    p.add_argument("--generate", action="store_true", help="Authorize exactly one live TTS call (uses Deepgram credits)")
    args = p.parse_args()
    outdir = args.output_dir
    wav_file = outdir / "swift_aec_aura2_thalia.wav"
    webm_file = outdir / "swift_aec_aura2_thalia.webm"
    if not args.generate:
        print(f"Model: {MODEL}; characters: {len(TEXT)}")
        print("No request sent. Add --generate to authorize one TTS call.")
        print(f"Expected outputs: {wav_file} and {webm_file}")
        return 0
    key = os.environ.get("DEEPGRAM_API_KEY") or dotenv_key(args.secrets_file)
    if not key:
        print("No DEEPGRAM_API_KEY found in environment or the supplied .env file.", file=sys.stderr)
        return 2
    if wav_file.exists() or webm_file.exists():
        print("Output file already exists; refusing to overwrite or send another provider request.", file=sys.stderr)
        return 2
    outdir.mkdir(parents=True, exist_ok=True)
    url = "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&container=wav&sample_rate=24000"
    data = json.dumps({"text": TEXT}).encode("utf-8")
    req = Request(url, data=data, method="POST", headers={
        "Authorization": f"Token {key}",
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    })
    print("Sending ONE Deepgram Aura-2 Thalia request (no retry)...", flush=True)
    try:
        with urlopen(req, timeout=45) as response:
            audio = response.read()
    except HTTPError as exc:
        print(f"Deepgram request failed: HTTP {exc.code}. No retry performed.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError) as exc:
        print(f"TTS request failed: {type(exc).__name__}. No retry performed; status may be unknown.", file=sys.stderr)
        return 1
    if len(audio) < 100 or not audio.startswith(b"RIFF"):
        print("Response was not a valid WAV header; no file written.", file=sys.stderr)
        return 1
    wav_file.write_bytes(audio)
    try:
        with wave.open(str(wav_file), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            print(f"WAV: {duration:.1f}s, {wf.getframerate()} Hz, {wf.getnchannels()} channel(s)")
    except (wave.Error, OSError) as exc:
        print(f"WAV validation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Saved {wav_file}")
    if duration < 17:
        print("Warning: voice clip under 17 seconds; verify probe playback timing.")
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found. WAV is ready; copy it to Windows and use that in the probe.")
        return 0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
           "-i", str(wav_file), "-vn", "-c:a", "libopus", "-b:a", "96k", str(webm_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        print("WebM conversion failed; WAV is still ready for use.", file=sys.stderr)
        return 0
    print(f"Saved {webm_file}")
    print("Transfer the WebM to Windows via VS Code Remote Explorer and select it in the 8766/audio-probe page.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
