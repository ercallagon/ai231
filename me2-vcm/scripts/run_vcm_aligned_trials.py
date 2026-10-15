#!/usr/bin/env python3
"""Persistent, locked, single-GPU corrected A10 -> A6 queue. No SLURM."""
import fcntl
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hey_swift.vcm.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
STAGE = Path("work/vcm_hf_alignment/hf_aligned_v1")
STAGE.mkdir(parents=True, exist_ok=True)
TRIALS = [
    ("run_039_hf_aligned_v1_a10_s42", "configs/vcm_hf_aligned_v1_a10.yaml", "ds_resnet_se"),
    ("run_040_hf_aligned_v1_a6_s42", "configs/vcm_hf_aligned_v1_a6.yaml", "resnet2d_small"),
]


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def eligible_gpu():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    choices = []
    for line in result.stdout.splitlines():
        gpu, utilization, used, total = (int(x.strip()) for x in line.split(","))
        free = total - used
        if utilization < 40 and free >= 10000:
            choices.append((free, gpu))
    return str(max(choices)[1]) if choices else None


def main():
    lock = (STAGE / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_hash = sha256("data/vcm/protocols/hf_aligned_v1/protocol.json")
    for tag, config, arch in TRIALS:
        for prior in Path("experiments/vcm").glob("*/reproducibility.json"):
            record = json.loads(prior.read_text())
            result_path = prior.parent / "result.json"
            if record.get("protocol_sha256") == protocol_hash and record.get("seed") == 42 and result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("arch") == arch:
                    raise RuntimeError(f"Equivalent corrected trial already completed: {prior.parent}")
        if (Path("experiments/vcm") / tag).exists():
            raise FileExistsError(f"Fresh output required: {tag}")
        subprocess.run([sys.executable, "scripts/validate_vcm_protocol.py", "--config", config], check=True)
        while True:
            gpu = eligible_gpu()
            if gpu is not None:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                probe = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], env=env, timeout=60)
                if probe.returncode == 0 and eligible_gpu() == gpu:
                    break
            log(f"waiting for {tag}: one GPU <40% utilization and >=10000 MiB free")
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        command = [sys.executable, "-u", "-m", "hey_swift.cli", "train", "vcm", "--tag", tag, "--config", config, "--seed", "42"]
        receipt = {"tag": tag, "gpu": gpu, "command": command, "protocol_sha256": protocol_hash, "config_sha256": sha256(config), "pid": os.getpid(), "launch_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (STAGE / f"{tag}_launch.json").write_text(json.dumps(receipt, indent=2))
        log(f"start {tag} GPU={gpu}")
        with (STAGE / f"{tag}.log").open("x") as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
            # Threshold and checkpoint selection are frozen before test access.
            # Pi holdout and supplemental holdout remain unevaluated.
            for split in ("test_core", "test_negatives", "test"):
                subprocess.run([sys.executable, "-u", "-m", "hey_swift.vcm.evaluate", "--tag", tag, "--config", config, "--split", split, "--device", "cpu"], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, "scripts/benchmark_vcm_latency.py", "--tag", tag, "--config", config, "--device", "cpu", "--iters", "200", "--warmup", "30"], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        log(f"done {tag}")
    log("queue complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true", help="Launch a persistent host process with saved PID/log")
    args = parser.parse_args()
    if args.detach:
        with (STAGE / "queue.log").open("x") as output:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                                     stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=os.environ.copy())
        (STAGE / "queue.pid").write_text(str(child.pid) + "\n")
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError(f"Queue exited immediately: code={child.returncode}; see {STAGE}/queue.log")
        print(f"Persistent queue PID={child.pid}, log={STAGE}/queue.log", flush=True)
    else:
        main()
