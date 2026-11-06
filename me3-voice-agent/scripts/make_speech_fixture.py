"""Generate explicitly synthetic speech using existing espeak; no API calls."""

import io
import argparse
import subprocess
import wave

import numpy as np
import soxr

from me3_voice.artifacts import output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Hey Swift. What time is it?")
    parser.add_argument("--sample-rate", type=int, choices=[16000, 48000], default=16000)
    parser.add_argument("--output", type=output_path, default="var/recordings/synthetic-speech.wav")
    args = parser.parse_args()
    if not 1 <= len(args.text) <= 500:
        parser.error("fixture text must contain 1..500 characters")
    result = subprocess.run(
        ["espeak", "--stdout", "-s", "145", args.text],
        check=True,
        capture_output=True,
    )
    with wave.open(io.BytesIO(result.stdout)) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
        audio = soxr.resample(audio, wav.getframerate(), args.sample_rate)
    audio = np.concatenate([np.zeros(args.sample_rate // 2), audio, np.zeros(args.sample_rate)])
    path = args.output
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, args.sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(np.clip(np.rint(audio), -32768, 32767).astype("<i2").tobytes())
    print(f"Synthetic espeak fixture (not a real speaker recording): {path}")


if __name__ == "__main__":
    main()
