#!/usr/bin/env python3
"""Plan or explicitly execute a fresh BC8 run; no test evaluation or warm-start."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def gpu_gate(index):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=15, check=True)
    for line in result.stdout.splitlines():
        values = [int(v.strip().split()[0]) for v in line.split(",")]
        if values[0] == index:
            if values[1] >= 40 or values[3] - values[2] < 20480:
                raise RuntimeError("Selected GPU fails <40% utilization and >=20GiB free safety gate")
            return {"physical_index": index, "utilization": values[1], "free_mib": values[3] - values[2]}
    raise ValueError("Selected GPU not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="New immutable experiment name")
    parser.add_argument("--execute", action="store_true", help="Opt in to actual training after preparation")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if not args.tag or Path(args.tag).name != args.tag or args.tag in {".", ".."}:
        parser.error("Tag must be a single safe directory name")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = Path("experiments/vcm") / args.tag
    if output.exists(): raise FileExistsError(output)
    plan = {"architecture": "bc_resnet_broadcast8", "initialization": "random_from_scratch",
        "config": "configs/vcm_hf_grouped_v2b_bc8.yaml", "output": str(output),
        "gpu_limit": 1, "test_holdout_evaluation": False,
        "requires": ["pinned raw HF snapshot", "published grouped-v2b protocol metadata",
                     "authenticated 2s grouped-v2b feature cache", "train/data extras", "approved available GPU"],
        "claim": "Retrains the BC8 baseline on the same protocol; not bit-identical run_043 source-finetune lineage"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute: return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(root / "src"))
    from validate_vcm_protocol import validate
    config = plan["config"]
    validate(config)
    gate = gpu_gate(args.gpu)
    import time
    time.sleep(10)
    gate = gpu_gate(args.gpu)
    from hey_swift.vcm.train_repro import run
    print(json.dumps({"gpu_gate": gate, "execute": True}), flush=True)
    # The existing audited trainer constructs random BC8 and refuses old outputs.
    run(args.tag, config)


if __name__ == "__main__": main()
