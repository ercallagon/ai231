#!/usr/bin/env python3
"""Persistent single-GPU gate for one validated, fresh grouped BC8 trial."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hey_swift.vcm.protocol import sha256
from vcm_grouped_bc8_trial import CONFIG, TAG

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
STAGE = Path("work/vcm_grouped_v2")


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def gpu_choices(output):
    choices = []
    for line in output.splitlines():
        gpu, utilization, used, total = [int(x.strip()) for x in line.split(",")]
        if utilization < 40 and total - used >= 20000:
            choices.append((total - used, gpu))
    return choices


def main():
    lock = (STAGE / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    experiment = Path("experiments/vcm") / TAG
    if experiment.exists():
        raise FileExistsError(f"Fresh experiment required: {experiment}")
    for path in Path("experiments/vcm").glob("*/reproducibility.json"):
        record = json.loads(path.read_text())
        if record.get("protocol_sha256") == sha256("data/vcm/protocols/hf_grouped_v2b/protocol.json") and record.get("seed") == 42 and record.get("architecture") == "bc_resnet_broadcast8":
            raise RuntimeError(f"Equivalent experiment already exists: {path.parent}")
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42")
    subprocess.run([sys.executable, "-u", "scripts/validate_vcm_protocol.py", "--config", CONFIG], env=environment, check=True)
    ready = json.loads((STAGE / "readiness.json").read_text())
    if not ready["passed"] or ready["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Readiness checks missing or config changed")
    while True:
        query = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        choices = gpu_choices(subprocess.check_output(query, text=True, timeout=15))
        if choices:
            gpu = str(max(choices)[1])
            time.sleep(5)
            if int(gpu) in [i for _, i in gpu_choices(subprocess.check_output(query, text=True, timeout=15))]:
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                probe = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], env=environment, timeout=60)
                if probe.returncode == 0:
                    break
        log("waiting: one GPU <40% utilization, >=20000 MiB free, working CUDA")
        time.sleep(30)
    command = [sys.executable, "-u", "scripts/vcm_grouped_bc8_trial.py", "train", "--config", CONFIG]
    (STAGE / "launch.json").write_text(json.dumps({"tag": TAG, "command": command, "gpu": gpu,
        "config_sha256": sha256(CONFIG), "protocol_sha256": sha256("data/vcm/protocols/hf_grouped_v2b/protocol.json"),
        "readiness_sha256": sha256(STAGE / "readiness.json"), "pid": os.getpid()}, indent=2))
    log(f"start {TAG} GPU={gpu}")
    with (STAGE / f"{TAG}.log").open("x") as stream:
        def execute(mode, *arguments):
            subprocess.run([sys.executable, "-u", "scripts/vcm_grouped_bc8_trial.py", mode, *arguments],
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        execute("train")
        execute("eval", "--split", "val")
        execute("calibrate")
        # Choices are frozen before accessing the challenge or known HF test.
        for split in ("challenge", "test_core", "test_negatives", "test"):
            execute("eval", "--split", split)
        calibrated = experiment / "calibrated"
        if calibrated.exists():
            for split in ("challenge", "test"):
                execute("eval", "--split", split, "--config", str(calibrated / "config.yaml"), "--exp-dir", str(calibrated))
        execute("latency")
        execute("export")
    log("complete; no holdout evaluated, no Pi promotion, no further trials queued")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        with (STAGE / "queue.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (STAGE / "queue.pid").write_text(str(child.pid) + "\n")
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError("Queue failed immediately; inspect queue.log")
        print(f"Persistent grouped BC8 queue PID={child.pid}; log={STAGE}/queue.log")
    else:
        main()
