#!/usr/bin/env python3
"""One source-balanced fine-tune, gated behind complete run_042 and GPU safety."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hey_swift.vcm.protocol import sha256
from run_vcm_grouped_bc8_queue import gpu_choices
from vcm_bc8_source_finetune import CONFIG, TAG, PARENT

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
STAGE = Path("work/vcm_sourceft")


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def main():
    lock = (STAGE / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = Path("experiments/vcm") / TAG
    if output.exists():
        raise FileExistsError(f"Fresh output required: {output}")
    prior_pid = int(Path("work/vcm_grouped_v2/queue.pid").read_text())
    while True:
        try:
            active = b"run_vcm_grouped_bc8_queue.py" in Path(f"/proc/{prior_pid}/cmdline").read_bytes()
        except FileNotFoundError:
            active = False
        if not active:
            break
        log("waiting for run_042 training, evaluation and export to finish")
        time.sleep(30)
    for filename in ("result.json", "metrics_test.json", "metrics_challenge.json", "export_validation.json"):
        if not (PARENT / filename).exists():
            raise RuntimeError(f"Parent did not complete required stage: {filename}")
    if not json.loads((PARENT / "export_validation.json").read_text())["onnx_checker_and_batch_1_2_parity_passed"]:
        raise RuntimeError("Parent ONNX parity failed")
    parent_hash = sha256(PARENT / "best.pt")
    for path in Path("experiments/vcm").glob("*/finetune_provenance.json"):
        record = json.loads(path.read_text())
        if (record.get("parent_checkpoint_sha256") == parent_hash
                and record.get("target_real_intent_loss_mass") == 0.30):
            raise RuntimeError(f"Equivalent refinement already completed: {path.parent}")
    readiness = json.loads((STAGE / "readiness.json").read_text())
    if not readiness["passed"] or readiness["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Fine-tune readiness missing or stale")
    subprocess.run([sys.executable, "-u", "scripts/validate_vcm_protocol.py", "--config", CONFIG], check=True)
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42")
    query = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    while True:
        choices = gpu_choices(subprocess.check_output(query, text=True, timeout=15))
        if choices:
            gpu = str(max(choices)[1])
            time.sleep(5)
            if int(gpu) in [i for _, i in gpu_choices(subprocess.check_output(query, text=True, timeout=15))]:
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                probe = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], env=environment, timeout=60)
                if probe.returncode == 0:
                    break
        log("waiting for one GPU <40% utilization and >=20000 MiB free")
        time.sleep(30)
    (STAGE / "launch.json").write_text(json.dumps({"tag": TAG, "gpu": gpu,
        "command": [sys.executable, "-u", "scripts/vcm_bc8_source_finetune.py", "train", "--config", CONFIG],
        "config_sha256": sha256(CONFIG), "parent_checkpoint_sha256": sha256(PARENT / "best.pt"),
        "readiness_sha256": sha256(STAGE / "readiness.json")}, indent=2))
    log(f"start {TAG} GPU={gpu}")
    with (STAGE / f"{TAG}.log").open("x") as stream:
        def execute(mode, *args):
            subprocess.run([sys.executable, "-u", "scripts/vcm_bc8_source_finetune.py", mode, *args],
                env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        execute("train")
        execute("eval", "--split", "val")
        execute("calibrate")
        for split in ("challenge", "test_core", "test_negatives", "test"):
            execute("eval", "--split", split)
        calibrated = output / "calibrated"
        if calibrated.exists():
            for split in ("challenge", "test"):
                execute("eval", "--split", split, "--config", str(calibrated / "config.yaml"), "--exp-dir", str(calibrated))
        execute("latency")
        execute("export")
    log("complete; no further runs or Pi promotion; holdout untouched")


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
        print(f"Persistent fine-tune queue PID={child.pid}; log={STAGE}/queue.log")
    else:
        main()
