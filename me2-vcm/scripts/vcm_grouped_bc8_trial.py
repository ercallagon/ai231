#!/usr/bin/env python3
"""Fresh BC8 experiment on grouped v2 data, never overwrites v1 results."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import vcm_bc8_trial as reference
from hey_swift.vcm.protocol import sha256

TAG = "run_042_hf_grouped_v2b_bc8_s42"
CONFIG = "configs/vcm_hf_grouped_v2b_bc8.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval", "calibrate", "latency", "export"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--split", default="val", choices=["val", "challenge", "test_core", "test_negatives", "test"])
    parser.add_argument("--exp-dir")
    args = parser.parse_args()
    reference.check_candidate(args.config)
    output = Path("experiments/vcm") / TAG
    if args.mode == "train":
        from hey_swift.vcm.train_repro import run
        run(TAG, args.config)
    elif args.mode in ("eval", "calibrate"):
        args.tag = TAG
        reference.main(args)
    elif args.mode == "latency":
        with reference.candidate_factories():
            spec = importlib.util.spec_from_file_location("latency", "scripts/benchmark_vcm_latency.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.argv = [sys.argv[0], "--tag", TAG, "--config", args.config,
                        "--device", "cpu", "--iters", "200", "--warmup", "30"]
            module.main()
    elif args.mode == "export":
        from hey_swift.vcm import evaluate
        torch.set_num_threads(1)
        with reference.candidate_factories():
            model, taxonomy, _ = evaluate._load_model(args.config, output / "best.pt", torch.device("cpu"))
        session = reference.export_model(model, output / "model.onnx")
        timings = []
        import time
        import numpy as np
        sample = np.zeros((1, 1, 80, 198), dtype=np.float32)
        for _ in range(20):
            session.run(None, {"mel": sample})
        for _ in range(100):
            start = time.perf_counter()
            session.run(None, {"mel": sample})
            timings.append((time.perf_counter() - start) * 1000)
        if (output / "calibrated").exists():
            calibration = json.loads((output / "calibration.json").read_text())
            reference.export_model(model, output / "calibrated/model.onnx", calibration["temperature"])
        labels = {"intents": list(taxonomy.intents) + [taxonomy.oos_class],
                  "slots": {name: taxonomy.slot_classes(name) for name in taxonomy.slotted_intents}}
        (output / "labels.json").write_text(json.dumps(labels, indent=2))
        if (output / "calibrated").exists():
            (output / "calibrated/labels.json").write_text(json.dumps(labels, indent=2))
        (output / "export_validation.json").write_text(json.dumps({
            "onnx_sha256": sha256(output / "model.onnx"), "onnx_bytes": (output / "model.onnx").stat().st_size,
            "onnx_checker_and_batch_1_2_parity_passed": True,
            "host_onnx_p50_ms": float(np.median(timings)), "host_onnx_p95_ms": float(np.percentile(timings, 95)),
            "pi_verified": False}, indent=2))


if __name__ == "__main__":
    main()
