#!/usr/bin/env python3
"""Run the frozen Baseline/Champion comparison in work/pi5_test_01 on Pi 5."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hey_swift.profiles import PUBLIC, DISPLAY
PROFILES = ("run043", "selected")
SPLITS = ("test", "holdout")


def prepare_results(root, reset=False):
    work = root / "work"
    target = work / "pi5_test_01"
    if work.is_symlink() or target.is_symlink():
        raise ValueError("Refusing to reset a symlinked work/results directory")
    work.mkdir(exist_ok=True)
    archive = None
    if target.exists():
        if not reset:
            raise ValueError("work/pi5_test_01 already exists. Use --reset to archive it and start clean.")
        archive = Path(tempfile.mkdtemp(prefix="pi5_test_01_previous_", dir=work)) / target.name
        target.rename(archive)
    target.mkdir()
    return target, archive


def run_logged(command, root, path):
    with path.open("x", buffering=1) as log:
        child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        from hey_swift.telemetry import ResourceSampler
        monitor = ResourceSampler(child.pid, path.parent / "resources.json")
        monitor.start()
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            monitor.finish()


def summarize(root, output):
    injection = output / "injection"
    completion = json.loads((injection / "COMPLETE.json").read_text())
    if set(completion["profiles"]) != set(PROFILES) or set(completion["splits"]) != set(SPLITS):
        raise ValueError("Evaluation did not complete both required profiles and splits")
    profiles = json.loads((root / "configs/assistant_profiles.json").read_text())["profiles"]
    summary = dict(scope="features_plus_ONNX_no_decode_resampling_capture_actions",
                   previously_exposed_evaluation=True, profiles={})
    for split in SPLITS:
        results = json.loads((injection / (split + "_RESULTS.json")).read_text())
        for name in PROFILES:
            result = results[name]
            digest = hashlib.sha256((root / profiles[name]["model"]).read_bytes()).hexdigest()
            if result["model_sha256"] != digest:
                raise ValueError("Result/model hash mismatch for " + name)
            timings = json.loads((injection / (split + "_" + name + "_timings.json")).read_text())
            counts = result["processed_subsets"]["all"]
            if len(timings) != counts["processed_n"]:
                raise ValueError("Timing/processed-count mismatch for " + name)
            summary["profiles"].setdefault(name, {})[split] = dict(
                model_sha256=digest, timed_n=len(timings), **counts,
                infer_p95_ms=result["infer_p95_ms"], per_clip_rtf_p95=result["per_clip_rtf_p95"],
                infer_scope=result["infer_scope"], threads=result["threads"],
                physical_pi_verified=result["physical_pi_verified"])
    with (output / "SUMMARY.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("\nFeature preprocessing + model inference (not speech-end latency):", flush=True)
    print("Profile   Split    Processed/attempted  p95 ms    RTF p95", flush=True)
    for split in SPLITS:
        for name in PROFILES:
            row = summary["profiles"][name][split]
            print(f"{PUBLIC[name]:9} {split:8} {row['processed_n']:5}/{row['manifest_n']:<8} "
                  f"{row['infer_p95_ms']:9.2f}  {row['per_clip_rtf_p95']:.5f}", flush=True)
    print("Full recognition/rejection metrics: injection/test_RESULTS.json and holdout_RESULTS.json", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true",
                        help="Archive the existing test folder and recreate it; no deletion")
    args = parser.parse_args()
    device_path = Path("/proc/device-tree/model")
    device = device_path.read_text().rstrip("\x00") if device_path.exists() else platform.machine()
    if "Raspberry Pi 5" not in device:
        parser.error("This command must run on the Raspberry Pi 5; no results were reset")
    work = ROOT / "work"
    if work.is_symlink():
        parser.error("Refusing a symlinked work directory")
    work.mkdir(exist_ok=True)
    with (work / ".pi5_test_01.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another test_pi.py run is active; do not reset its results")
        try:
            output, archive = prepare_results(ROOT, args.reset)
        except ValueError as exc:
            parser.error(str(exc))
        if archive:
            print("Previous results archived: " + str(archive), flush=True)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True)
        (output / "source_revision.txt").write_text(revision.stdout.strip() + "\n")
        versions = {}
        for name in ("numpy", "scipy", "onnxruntime", "soundfile", "pyarrow", "PyYAML"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "NOT INSTALLED"
        command = [sys.executable, "-u", str(ROOT / "scripts/reproduce.py"), "reproduce",
                   "--profiles", *PROFILES, "--splits", *SPLITS,
                   "--fetch", "--output", str(output / "injection")]
        environment = dict(started_utc=datetime.now(timezone.utc).isoformat(), device=device,
            python=sys.version, platform=platform.platform(), packages=versions,
            source_revision=revision.stdout.strip(), git_revision_error=revision.stderr.strip(),
            command=command, archived_results=str(archive) if archive else None,
            note="Stop other assistants/benchmarks; raw audio caches are reused. "
                 "No actions, training or threshold fitting. Physical acoustic tests are separate.")
        (output / "ENVIRONMENT.json").write_text(json.dumps(environment, indent=2) + "\n")
        print(f"Comparison: {DISPLAY['run043']} | {DISPLAY['selected']}", flush=True)
        print("Evaluating frozen test + holdout; results: " + str(output), flush=True)
        code = run_logged(command, ROOT, output / "injection.log")
        if code:
            print("Evaluation failed. Retain injection.log; --reset archives partial results before retrying.", file=sys.stderr)
            return code
        summarize(ROOT, output)
        print("\nDONE. Retrieve: " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped. Partial results preserved; use --reset for a clean retry.", file=sys.stderr)
        raise SystemExit(130)
