#!/usr/bin/env python3
"""Bounded, disposable memory measurement; no checkpoint or dataset fitting."""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml

from hey_swift.vcm.broadcast import build_broadcast_model
from hey_swift.vcm.protocol import sha256
from hey_swift.vcm.taxonomy import load_taxonomy
from hey_swift.vcm.train import _slot_loss_and_stats

MIB = 1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcm_hf_grouped_v2b_bc8.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    cfg = yaml.safe_load(Path(args.config).read_text())
    taxonomy = load_taxonomy(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    budget = min(free - 2048 * MIB, 12000 * MIB)
    if budget < 1024 * MIB:
        raise RuntimeError("Insufficient memory even for a bounded probe")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(cfg["training"]["seed"]))
    report = {"config_sha256": sha256(args.config), "probe_source_sha256": sha256(__file__),
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(),
              "batch_size": cfg["training"]["batch_size"], "input_shape": [cfg["training"]["batch_size"], 1, 80, 198],
              "free_before_mib": free / MIB, "allocator_budget_mib": budget / MIB,
              "deterministic_algorithms": True, "bf16_training": torch.cuda.is_bf16_supported(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "saved_model": False, "synthetic_inputs_only": True, "passed": False}
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        heads = {name: taxonomy.slot_classes(name) for name in taxonomy.slotted_intents}
        model = build_broadcast_model(taxonomy.n_way_head, heads).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                     weight_decay=cfg["training"]["weight_decay"])
        batch = int(cfg["training"]["batch_size"])
        x = torch.randn(batch, 1, 80, 198, device="cuda")
        y = torch.arange(batch, device="cuda") % taxonomy.n_way_head
        slots = torch.arange(batch, device="cuda") % 3
        weights = torch.ones(taxonomy.n_way_head, device="cuda")
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["training"]["label_smoothing"])
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=report["bf16_training"]):
                output = model(x)
                loss = loss_fn(output["intent"], y)
                slot_loss, _, _ = _slot_loss_and_stats(output, y, slots, taxonomy, torch.nn.CrossEntropyLoss())
                if slot_loss is not None:
                    loss = loss + cfg["training"]["slot_loss_weight"] * slot_loss
            loss.backward()
            optimizer.step()
            del output, loss, slot_loss
            torch.cuda.synchronize()
            print(f"probe step {step + 1}: reserved={torch.cuda.max_memory_reserved()/MIB:.0f} MiB", flush=True)
        # Match FP32 clean validation, including optimizer state still resident.
        model.eval()
        with torch.no_grad():
            output = model(x)
        del output
        torch.cuda.synchronize()
        report["passed"] = True
    except torch.cuda.OutOfMemoryError as error:
        report["error"] = "Bounded allocator OOM; full batch cannot be profiled within the safe current budget"
        report["detail"] = str(error)
    report.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / MIB,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / MIB,
                  elapsed_s=time.monotonic() - start)
    if report["passed"]:
        report["recommended_free_mib"] = int(math.ceil((1.30 * report["peak_reserved_mib"] + 2048) / 256) * 256)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
