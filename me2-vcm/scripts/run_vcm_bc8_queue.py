#!/usr/bin/env python3
"""One persistent, GPU-gated BC8 trial after the existing corrected queue."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hey_swift.vcm.protocol import sha256
from vcm_bc8_trial import CONFIG, TAG

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
STAGE = Path("work/vcm_next_bc8")
STAGE.mkdir(exist_ok=True)


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def command(arguments, output=None, environment=None):
    subprocess.run([sys.executable, "-u", *arguments], env=environment,
                   stdout=output, stderr=subprocess.STDOUT if output else None, check=True)


def main():
    lock = (STAGE / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output_dir = Path("experiments/vcm") / TAG
    if output_dir.exists():
        raise FileExistsError(f"Fresh output required: {output_dir}")
    for result in Path("experiments/vcm").glob("*/result.json"):
        try:
            info = json.loads(result.read_text())
        except json.JSONDecodeError:
            log(f"unreadable historical result preserved: {result}")
            continue
        if info.get("arch") == "bc_resnet_broadcast8" and info.get("seed") == 42:
            raise RuntimeError(f"Equivalent candidate already completed: {result.parent}")
    prior_pid = int(Path("work/vcm_hf_alignment/hf_aligned_v1/queue.pid").read_text())
    while True:
        try:
            os.kill(prior_pid, 0)
            text = Path(f"/proc/{prior_pid}/cmdline").read_bytes()
            prior_active = b"run_vcm_aligned_trials.py" in text
        except (ProcessLookupError, FileNotFoundError):
            prior_active = False
        if not prior_active:
            break
        log("waiting for corrected A10/A6 queue; no concurrent GPU training")
        time.sleep(30)
    for name in ("metrics_test.json", "latency_cpu.json"):
        if not (Path("experiments/vcm/run_040_hf_aligned_v1_a6_s42") / name).exists():
            raise RuntimeError(f"Prior corrected queue did not finish: missing A6 {name}")
    command(["scripts/validate_vcm_protocol.py", "--config", CONFIG])
    proof = json.loads((STAGE / "preflight.json").read_text())
    if not (proof.get("onnx_shapes_and_parity_passed") and proof.get("finite_backward_passed")):
        raise RuntimeError("Candidate numerical/export preflight has not passed")
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=15, check=True)
        choices = []
        for line in result.stdout.splitlines():
            gpu, utilization, used, total = map(lambda x: int(x.strip()), line.split(","))
            if utilization < 40 and total - used >= 20000:
                choices.append((total - used, gpu))
        if choices:
            gpu = str(max(choices)[1])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            probe = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], env=environment, timeout=60)
            if probe.returncode == 0:
                break
        log("waiting for one GPU <40% utilization, >=20000 MiB free and working CUDA")
        time.sleep(30)
    receipt = {"tag": TAG, "gpu": gpu, "command": [sys.executable, "-u", "scripts/vcm_bc8_trial.py", "train", "--config", CONFIG], "config_sha256": sha256(CONFIG), "protocol_sha256": sha256("data/vcm/protocols/hf_aligned_v1/protocol.json"), "pid": os.getpid()}
    receipt["preflight_sha256"] = sha256(STAGE / "preflight.json")
    (STAGE / "launch.json").write_text(json.dumps(receipt, indent=2))
    log(f"start {TAG} GPU={gpu}")
    with (STAGE / f"{TAG}.log").open("x") as output:
        command(["scripts/vcm_bc8_trial.py", "train"], output, environment)
        command(["scripts/vcm_bc8_trial.py", "eval", "--split", "val"], output, environment)
        command(["scripts/vcm_bc8_trial.py", "calibrate"], output, environment)
        # All model/calibration choices are frozen before new test evaluation.
        for split in ("test_core", "test_negatives", "test"):
            command(["scripts/vcm_bc8_trial.py", "eval", "--split", split], output, environment)
        calibrated = output_dir / "calibrated"
        if calibrated.exists():
            for split in ("test_core", "test_negatives", "test"):
                command(["scripts/vcm_bc8_trial.py", "eval", "--split", split,
                         "--config", str(calibrated / "config.yaml"), "--exp-dir", str(calibrated)], output, environment)
        command(["scripts/vcm_bc8_trial.py", "latency"], output, environment)
        command(["scripts/vcm_bc8_trial.py", "export"], output, environment)
        # Apply the identical calibration procedure to frozen controls, so a
        # threshold gain is not misreported as an architecture gain.
        for baseline_tag, baseline_config in (
            ("run_039_hf_aligned_v1_a10_s42", "configs/vcm_hf_aligned_v1_a10.yaml"),
            ("run_040_hf_aligned_v1_a6_s42", "configs/vcm_hf_aligned_v1_a6.yaml"),
        ):
            command(["scripts/vcm_bc8_trial.py", "calibrate", "--tag", baseline_tag, "--config", baseline_config], output, environment)
            calibrated = Path("experiments/vcm") / baseline_tag / "calibrated"
            if calibrated.exists():
                command(["scripts/vcm_bc8_trial.py", "eval", "--tag", baseline_tag, "--split", "test", "--config", str(calibrated / "config.yaml"), "--exp-dir", str(calibrated)], output, environment)
    log("candidate complete; no model promoted and no further runs queued")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        with (STAGE / "queue.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        (STAGE / "queue.pid").write_text(str(child.pid) + "\n")
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError("Queue exited immediately; inspect work/vcm_next_bc8/queue.log")
        print(f"Persistent BC8 queue PID={child.pid}; logs={STAGE}/queue.log")
    else:
        main()
