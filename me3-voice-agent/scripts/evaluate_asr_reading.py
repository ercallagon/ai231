"""One bounded DGX3 ASR MCP request on a user-recorded reading passage."""

import argparse
import asyncio
import hashlib
import json
import re
import secrets
import subprocess
import time
from pathlib import Path

from me3_voice.artifacts import output_path
from me3_voice.configuration import load_backends
from me3_voice.contracts import MAX_AUDIO_BYTES, TurnKey
from me3_voice.credentials import load_secrets
from me3_voice.remote import LiveTransport
from me3_voice.speech import SpeechClient

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = (ROOT / "var/recordings").resolve()


def recording_path(value):
    path = Path(value)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(RECORDINGS):
        raise ValueError("recording must be a regular file under project var/recordings")
    if path.suffix.lower() not in {".webm", ".mp4", ".wav"} or path.stat().st_size > 15_000_000:
        raise ValueError("expected a WebM, MP4 or WAV recording under 15 MB")
    return path


def words(value):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value.lower())


def speech_content(value):
    match = re.fullmatch(
        r"\s*The speech,\s*transcribed verbatim,\s*is:\s*(['\"])(.*)\1\s*",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (
        (match.group(2), "quoted_speech_after_model_preamble") if match else (value, "raw_response")
    )


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, guess in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != guess)))
        previous = current
    return previous[-1]


async def run(pcm, backends, credentials):
    async with SpeechClient(
        backends=backends, credentials=credentials, timeout=backends["asr"].timeout_seconds
    ) as speech:
        return await speech.transcribe(
            TurnKey(session_id="asrreading" + secrets.token_hex(8), turn_id=1), pcm
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=recording_path)
    parser.add_argument(
        "--manifest", type=Path, help="matching JSON downloaded from the reading page"
    )
    parser.add_argument("--backend-config", type=Path, default=Path("var/cache/dgx3-backends.json"))
    parser.add_argument("--secrets-file", type=Path, default=Path("var/cache/dgx3.env"))
    parser.add_argument(
        "--output", type=output_path, default=output_path("var/results/asr-reading-evaluation.json")
    )
    args = parser.parse_args()
    manifest = args.manifest or args.recording.with_suffix(".json")
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or not manifest.resolve().is_relative_to(RECORDINGS)
        or manifest.stat().st_size > 20_000
    ):
        parser.error("reference JSON must be a regular file under project var/recordings")
    reference = json.loads(manifest.read_text())
    expected = reference.get("expected_text")
    if (
        reference.get("kind") != "swift-asr-reading"
        or reference.get("passage_id") != "swift-reading-v1"
        or not isinstance(expected, str)
        or not 20 <= len(expected) <= 500
    ):
        parser.error("matching Swift reading reference JSON is invalid")
    if args.output.exists():
        parser.error("output already exists; use a fresh var/results path")
    backends = load_backends(args.backend_config)
    if backends["asr"].backend != "external" or backends["asr"].request_limit < 1:
        parser.error("ASR live backend and a positive request limit are required")
    credentials = load_secrets(
        args.secrets_file,
        {spec.auth_env for spec in backends.values() if spec.backend == "external"},
    )
    LiveTransport(backends["asr"], credentials).authorize()
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(args.recording),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    pcm = decoded.stdout
    if not 3200 <= len(pcm) <= MAX_AUDIO_BYTES or len(pcm) % 2:
        parser.error("decoded audio must be 0.1 to 30 seconds of 16 kHz mono PCM16")
    started = time.monotonic()
    recognized = asyncio.run(run(pcm, backends, credentials))
    content, extraction = speech_content(recognized)
    reference_words = words(expected)
    raw_errors = edit_distance(reference_words, words(recognized))
    errors = edit_distance(reference_words, words(content))
    report = {
        "kind": "swift-asr-reading-evaluation",
        "mode": "HPC-live-recorded-audio",
        "recording_sha256": hashlib.sha256(args.recording.read_bytes()).hexdigest(),
        "passage_id": reference["passage_id"],
        "expected_text": expected,
        "recognized_text": recognized,
        "recognized_speech_text": content,
        "speech_extraction": extraction,
        "reference_words": len(reference_words),
        "raw_response_word_errors": raw_errors,
        "raw_response_word_error_rate": raw_errors / len(reference_words),
        "word_errors": errors,
        "word_error_rate": errors / len(reference_words),
        "asr_request_elapsed_seconds": round(time.monotonic() - started, 3),
        "audio_seconds": round(len(pcm) / 32000, 3),
        "asr_protocol": backends["asr"].protocol,
        "limitations": "One read passage; WER assumes verbatim reading. Browser mic and file replay, not Pi or live acoustic latency.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"ASR reading check: {errors}/{len(reference_words)} word edits; WER {report['word_error_rate']:.1%}; {args.output}"
    )


if __name__ == "__main__":
    main()
