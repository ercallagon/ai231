#!/usr/bin/env python3
"""Record a clean candidate revision and verify active CPU contracts; no actions/audio capture."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root, expected=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    top = Path(git("rev-parse", "--show-toplevel"))
    if (top / "me2-vcm").resolve() != root.resolve():
        raise ValueError("Use the canonical ai231/me2-vcm Git checkout, not an exported/copied directory")
    head = git("rev-parse", "HEAD")
    if expected and head != expected:
        raise ValueError("Git revision does not match the expected tested commit")
    if git("status", "--porcelain", "--untracked-files=no", "--", "."):
        raise ValueError("Tracked project edits exist; preserve/review them before recording a frozen candidate")
    return head


def collect(root, endpoint_ms=800., expected=None, device="0"):
    if not math.isfinite(endpoint_ms) or endpoint_ms <= 0:
        raise ValueError("Endpoint silence must be finite and positive")
    head = revision(root, expected)
    artifacts = json.loads((root / "ARTIFACTS.json").read_text())["sha256"]
    registry = json.loads((root / "configs/assistant_profiles.json").read_text())["profiles"]
    wake = json.loads((root / "configs/wake_profiles.json").read_text())["profiles"]["exp15"]
    reference = json.loads((root / "data/model_reference.json").read_text())["models"]
    hashes = {}
    paths = {"configs/assistant_profiles.json", "configs/wake_profiles.json", "configs/hardware.yaml",
        "data/model_reference.json", "data/PROVENANCE.json", "audio/wake_chime.wav",
        "requirements-pi-py313-aarch64.lock", "requirements-speech-py313-aarch64.lock",
        "scripts/record_submission.py", "scripts/run_assistant.py", "scripts/run_benchmark_assistant.py",
        "src/hey_swift/integration/e2e_loop.py", "src/hey_swift/deployment.py"}
    paths.update(str(p.relative_to(root)) for p in (root / "data/manifests").glob("*.csv"))
    paths.update(wake.values())
    for profile in registry.values():
        paths.update(profile.values())
    for name in sorted(paths):
        digest = sha(root / name)
        if artifacts.get(name) != digest:
            raise ValueError("Frozen artifact checksum missing/mismatched: " + name)
        hashes[name] = digest
    import numpy as np
    from hey_swift.vcm.inference import VCMInference
    from hey_swift.wakeword.inference import WakeWordDetector
    import yaml
    models = {}
    for public, legacy in (("champion", "selected"), ("baseline", "run043")):
        p = registry[legacy]
        model = VCMInference(str(root/p["config"]), str(root/p["model"]), str(root/p["labels"]))
        if model.tau != reference[public]["threshold"] or hashes[p["model"]] != reference[public]["model_sha256"]:
            raise ValueError("Frozen model/policy mismatch: " + public)
        feed = {}
        for item in model.sess.get_inputs():
            if item.type != "tensor(float)":
                raise ValueError("Unsupported CPU contract input type: " + public)
            shape = [size if isinstance(size, int) else 1 for size in item.shape]
            feed[item.name] = (np.ones if item.name == "frame_mask" else np.zeros)(shape, dtype=np.float32)
        outputs = model.sess.run(None, feed)
        if not outputs or not all(np.isfinite(value).all() for value in outputs):
            raise ValueError("Nonfinite CPU contract probe: " + public)
        capture = yaml.safe_load((root/p["config"]).read_text())["capture"]
        models[public] = dict(legacy_profile=legacy, parameters=reference[public]["parameters"],
            threshold=model.tau, model_bytes=(root/p["model"]).stat().st_size, threads=model.threads,
            model_inputs={x.name: x.shape for x in model.sess.get_inputs()},
            profile_capture=capture, effective_capture={**capture, "end_silence_ms": endpoint_ms},
            contract="finite CPU execution; no accuracy or physical-action claim")
    detector = WakeWordDetector(str(root/wake["config"]), onnx_path=str(root/wake["model"]))
    probabilities = detector._score_window(np.zeros(detector.win, np.float32))[:2]
    if not all(math.isfinite(float(x)) for x in probabilities):
        raise ValueError("Nonfinite wake CPU contract probe")
    pi = Path("/proc/device-tree/model")
    hardware = pi.read_text().strip("\x00\n") if pi.exists() else platform.machine()
    versions = {}
    for name in ("numpy", "scipy", "onnxruntime", "soundfile", "sounddevice", "pyarrow", "PyYAML", "piper-tts", "gpiozero", "dbus-next"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), source_revision=head,
        tracked_project_clean=True, device=hardware, actual_pi5="Raspberry Pi 5" in hardware,
        python=platform.python_version(), platform=platform.platform(), packages=versions,
        artifact_sha256=hashes, models=models,
        wake=dict(profile="exp15", threshold=detector.gate.threshold,
                  sustain=detector.gate.sustain, cooldown_seconds=detector.gate.cooldown_seconds),
        benchmark_policy=dict(end_silence_ms=endpoint_ms, wake_gap_seconds=2., device_requested=device,
            actions="preview", speech=False, services="offline", frozen_model_rejection=True),
        scope="CPU contracts and provenance only; not accuracy, calibrated microphone, latency or successful Pi deployment",
        limitations=["Test/holdout were previously exposed", "Selected wake takes were development-screened",
            "800ms can end commands containing pauses longer than 800ms; 1200ms remains the fallback",
            "Record actual playback device, gain, placement and noise in the class benchmark configuration",
            "No private profile, recipient, account state or environment secret is read or copied"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--end-silence-ms", type=float, default=800.)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new result name")
    result = collect(ROOT, args.end_silence_ms, args.expected_commit, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Saved candidate provenance:", args.output)
    print("Revision:", result["source_revision"], "| actual Pi 5:", result["actual_pi5"])
    print("CPU contracts pass; microphone/acoustic/action checks remain separate.")


if __name__ == "__main__":
    main()
