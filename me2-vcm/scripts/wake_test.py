#!/usr/bin/env python
"""Standalone wake-word tester -- "Hey Swift" only, no VCM.

Use this to validate the wake gate on the Pi 5 BEFORE the VCM is exported.
It drives the real WakeWordDetector (same ONNX + same NumPy feature path the
deployed pipeline uses) but skips VCMInference / ActionRouter, so it works
with zero VCM artifacts present.

Pi 5 audio handling (board mic accepts 48 kHz only; speaker rejects 16 kHz):
  * Capture at the device's supported rate (48 kHz on the Pi), then resample
    to 16 kHz CONTINUOUSLY (hey_swift.audio.capture.ContinuousResampler) so
    chunk boundaries do not introduce clicks.
  * On each wake, play the VOICY acknowledgement chime (audio/wake_chime.wav)
    at the supported speaker rate. The mic is paused for the whole playback and
    the detector is reset afterwards, so the cue cannot retrigger the gate.

The heavy lifting (capture + resample, playback) lives in the production
``hey_swift.audio`` modules; this script is a thin driver around them.

Usage (from the repo root, with the [audio] extra installed):
    python scripts/wake_test.py                     # live microphone
    python scripts/wake_test.py --wav a.wav         # replay a recording
    python scripts/wake_test.py --dry-run           # load model + self-test only
    python scripts/wake_test.py --threshold 0.25    # tune sensitivity
    python scripts/wake_test.py --device 1          # pick a non-default mic
    python scripts/wake_test.py --no-ack            # silence the acknowledgement

Stop with Ctrl+C. Each detected wake prints a line with the P(wake) prob.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def _repo_root() -> str:
    """Repo root = parent of scripts/ so relative config/model paths resolve."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _coerce_device(value):
    """Numeric --device args become integer ALSA indices; strings stay names."""
    if value is None:
        return None
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return value


def build_detector(root: str, threshold: float | None):
    from hey_swift.wakeword.inference import WakeWordDetector

    cfg_path = os.path.join(root, "configs", "wakeword.yaml")
    det = WakeWordDetector(cfg_path)
    if threshold is not None:
        # Override the fire threshold without touching the YAML.
        det.gate = type(det.gate)(
            threshold=threshold,
            sustain=det.gate.sustain,
            cooldown_seconds=det.gate.cooldown_seconds,
            rms_floor_db=det.gate.rms_floor_db,
            hop_seconds=det.gate.hop_seconds,
        )
    return det


def build_playback(root: str):
    """Real playback on the Pi; silent no-op fallback if audio is unavailable."""
    try:
        from hey_swift.audio.playback import make_playback
        return make_playback(os.path.join(root, "configs", "hardware.yaml"))
    except Exception:
        return None


def self_test(root: str, det) -> int:
    """Prove the ONNX + feature path work end-to-end (no mic needed)."""
    from hey_swift.wakeword.features import SAMPLE_RATE, mel_from_pcm

    # 1.2 s of 440 Hz sine -- real audio, should be scored without error.
    t = np.arange(int(1.2 * SAMPLE_RATE)) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    feat = mel_from_pcm(tone)
    assert feat.shape == (1, 80, 100), f"bad feature shape {feat.shape}"
    logits = det.sess.run(None, {det.input_name: feat})[0][0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    print(f"[self-test] feature shape OK: {feat.shape}")
    print(f"[self-test] model ran, P(wake) on 440 Hz tone = {float(p[0]):.4f}")
    print(f"[self-test] gate: threshold={det.gate.threshold} "
          f"sustain={det.gate.sustain} cooldown={det.gate.cooldown_seconds}s "
          f"rms_floor={det.gate.rms_floor_db}dB")
    print("[self-test] OK -- model + feature path load and run on this device.")
    return 0


def run_live(root: str, det, device, verbose: bool, ack: bool) -> int:
    import sounddevice as sd  # lazy: only needed for live capture
    from hey_swift.audio.capture import (ContinuousResampler,
                                         capture_supported_rate)
    from hey_swift.wakeword.features import SAMPLE_RATE

    device = _coerce_device(device)
    pb = build_playback(root) if ack else None
    chime = os.path.join(root, "audio", "wake_chime.wav")
    if not os.path.exists(chime):
        chime = None

    # Capture at the rate the device accepts (48 kHz on the Pi 5 board mic),
    # then resample continuously to the 16 kHz model rate.
    rate, hop = capture_supported_rate(device)
    rs = ContinuousResampler(rate, SAMPLE_RATE)
    t0 = time.time()
    n_fires = 0
    print(f"[live] streaming device={device!r} @ {rate} Hz -> {SAMPLE_RATE} Hz, "
          f"{hop / rate * 1000:.0f} ms hops (continuous resample).")
    print(f"[live] ack={'ON (' + os.path.basename(chime) + ')' if (pb and chime) else 'OFF'}")
    print("[live] say 'Hey Swift'. Ctrl+C to stop.")

    def play_ack_guarded():
        """Play the acknowledgement cue. ``sd.play``+``wait`` blocks the loop
        for the cue's duration, so the mic is not read while it sounds -- the
        cue cannot retrigger. Reset the detector + resampler afterwards so the
        next capture starts clean."""
        if pb is None or chime is None:
            return
        try:
            pb.play_ack(chime)
        except Exception:
            pass
        time.sleep(0.2)  # settle before resuming
        det.reset()
        rs.reset()

    with sd.RawInputStream(samplerate=rate, blocksize=hop, dtype="float32",
                           channels=1, device=device) as stream:
        while True:
            data, _overflow = stream.read(hop)
            chunk = np.frombuffer(data, dtype=np.float32).reshape(-1)  # decode
            chunk16 = rs.push(chunk)                                  # -> 16 kHz
            events, p_last = det.process(chunk16)
            if verbose:
                print(f"\r[p_wake={p_last:.3f}]", end="", flush=True)
            for ev in events:
                n_fires += 1
                print(f"\n>>> WAKE DETECTED  t={time.time()-t0:6.1f}s  "
                      f"P(wake)={ev['prob']:.3f}  (fire #{n_fires})")
                play_ack_guarded()
                print("    (no VCM attached -- command stage is disabled)")


def run_wav(root: str, det, path: str, ack: bool) -> int:
    events, n = det.detect_file(path)
    print(f"[wav] replayed {os.path.basename(path)} in {n} chunks")
    if not events:
        print("[wav] no wake detected.")
    pb = build_playback(root) if ack else None
    chime = os.path.join(root, "audio", "wake_chime.wav")
    for i, ev in enumerate(events, 1):
        print(f">>> WAKE DETECTED  t={ev['t']:6.2f}s  P(wake)={ev['prob']:.3f}  (#{i})")
        if pb is not None and os.path.exists(chime):
            try:
                pb.play_ack(chime)
            except Exception:
                pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", help="Replay a WAV file instead of the live mic.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load the model and run a self-test, then exit (no mic).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override P(wake) fire threshold (default from YAML = 0.30).")
    ap.add_argument("--device", default=None,
                    help="Microphone device: integer index or name (default: system default).")
    ap.add_argument("--no-ack", dest="ack", action="store_false",
                    help="Silence the acknowledgement chime on wake.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print the live P(wake) on every hop.")
    args = ap.parse_args()
    args.device = _coerce_device(args.device)

    root = _repo_root()
    det = build_detector(root, args.threshold)
    if args.dry_run:
        return self_test(root, det)
    if args.wav:
        return run_wav(root, det, args.wav, args.ack)
    try:
        return run_live(root, det, args.device, args.verbose, args.ack)
    except KeyboardInterrupt:
        print("\n[live] stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
