#!/usr/bin/env python3
"""Isolated source-balanced intent fine-tune of the completed grouped BC8."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch
import yaml

import vcm_bc8_trial as reference
import vcm_grouped_bc8_trial as grouped
from hey_swift.vcm import train_repro
from hey_swift.vcm.protocol import sha256

TAG = "run_043_hf_grouped_v2b_bc8_sourceft_s42"
CONFIG = "configs/vcm_hf_grouped_v2b_bc8_sourceft.yaml"
PARENT = Path("experiments/vcm/run_042_hf_grouped_v2b_bc8_s42")


def source_weight(real_fraction, target_fraction):
    if not 0 < real_fraction < 1 or not 0 < target_fraction < 1:
        raise ValueError("Fractions must be strictly inside (0, 1)")
    return target_fraction * (1 - real_fraction) / (real_fraction * (1 - target_fraction))


def balanced_intent_loss(logits, labels, criterion, multipliers):
    values = torch.nn.functional.cross_entropy(logits, labels, weight=criterion.weight,
              label_smoothing=criterion.label_smoothing, reduction="none")
    denominator = criterion.weight[labels] if criterion.weight is not None else torch.ones_like(values)
    return (values * multipliers).sum() / (denominator * multipliers).sum()


class SourceDataset(train_repro.CachedDataset):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.augment:
            fraction = sum(r["is_synthetic"] == "0" for r in self.rows) / len(self.rows)
            self.real_multiplier = source_weight(fraction, config["experiment"]["source_balance_target"])

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        if self.augment:
            multiplier = self.real_multiplier if self.rows[i]["is_synthetic"] == "0" else 1.0
            return (*sample, multiplier)
        return sample


@contextlib.contextmanager
def refinement_engine(config, proof):
    original_dataset = train_repro.CachedDataset
    original_pass = train_repro.epoch_pass
    original_builder = train_repro.build_broadcast_model
    def balanced_pass(model, loader, device, taxonomy, criterion, optimizer=None, amp=False):
        if optimizer is None:
            return original_pass(model, loader, device, taxonomy, criterion)
        class WeightedCriterion(torch.nn.Module):
            def forward(self, logits, labels):
                return balanced_intent_loss(logits, labels, criterion, self.multipliers)
        weighted = WeightedCriterion()
        class InputBatches:
            def __iter__(self):
                for x, y, slots, multipliers in loader:
                    weighted.multipliers = multipliers.to(device, non_blocking=device.type == "cuda", dtype=torch.float32)
                    yield x, y, slots
        return original_pass(model, InputBatches(), device, taxonomy, weighted, optimizer, amp)
    def warm_builder(n_way, heads):
        model = original_builder(n_way, heads)
        checkpoint = torch.load(PARENT / "best.pt", map_location="cpu", weights_only=False)
        cfg = yaml.safe_load(Path(config).read_text())
        if checkpoint.get("protocol_sha256") != sha256(cfg["data"]["protocol"]):
            raise ValueError("Warm-start protocol mismatch")
        if checkpoint.get("arch") != "bc_resnet_broadcast8":
            raise ValueError("Warm-start architecture mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model
    train_repro.CachedDataset = SourceDataset
    train_repro.epoch_pass = balanced_pass
    train_repro.build_broadcast_model = warm_builder
    try:
        yield
    finally:
        train_repro.CachedDataset = original_dataset
        train_repro.epoch_pass = original_pass
        train_repro.build_broadcast_model = original_builder


def train(config):
    import csv
    cfg = yaml.safe_load(Path(config).read_text())
    if not (PARENT / "result.json").exists():
        raise RuntimeError("Parent training must finish before fine-tuning")
    with Path(cfg["data"]["manifests"], "train_19intent.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fraction = sum(r["is_synthetic"] == "0" for r in rows) / len(rows)
    target = cfg["experiment"]["source_balance_target"]
    proof = {"parent": str(PARENT), "parent_checkpoint_sha256": sha256(PARENT / "best.pt"),
             "training_real_fraction": fraction, "target_real_intent_loss_mass": target,
             "real_intent_multiplier": source_weight(fraction, target),
             "semantics": "Target is source mass before interacting with class weights; realized batch mass varies",
             "slot_loss_unchanged": True, "optimizer_reset": True,
             "architecture_unchanged": True, "new_audio": False,
             "test_holdout_or_challenge_used_for_choices": False}
    print(json.dumps(proof, indent=2), flush=True)
    with refinement_engine(config, proof):
        output = train_repro.run(TAG, config)
    (output / "finetune_provenance.json").write_text(json.dumps(proof, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval", "calibrate", "latency", "export"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--split", default="val", choices=["val", "challenge", "test_core", "test_negatives", "test"])
    parser.add_argument("--exp-dir")
    args = parser.parse_args()
    reference.check_candidate(args.config)
    if args.mode == "train":
        train(args.config)
    else:
        # Reuse unchanged grouped evaluation/calibration/export, routing only
        # this process's tag/config. Shared source files remain authenticated.
        grouped.TAG, grouped.CONFIG = TAG, args.config
        sys.argv = [sys.argv[0], args.mode, "--config", args.config]
        if args.mode == "eval":
            sys.argv += ["--split", args.split]
            if args.exp_dir:
                sys.argv += ["--exp-dir", args.exp_dir]
        grouped.main()


if __name__ == "__main__":
    main()
