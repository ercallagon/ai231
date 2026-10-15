#!/usr/bin/env python3
"""Controlled validation-only OOS discovery and isolated validity-head training."""
import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.special import logsumexp, softmax
from torch.utils.data import DataLoader

from hey_swift.vcm import train_repro as engine
from hey_swift.vcm.protocol import sha256, verify_protocol, load_feature_cache
from hey_swift.vcm.taxonomy import load_taxonomy
from hey_swift.vcm.train import _class_weights
from hey_swift.vcm.validity import NAME, ValidityModel, balanced_validity_loss
from vcm_bc8_source_finetune import SourceDataset, balanced_intent_loss

STAGE = Path("work/vcm_oos_discovery")
PARENT = "run_043_hf_grouped_v2b_bc8_sourceft_s42"


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def check(config):
    cfg = yaml.safe_load(Path(config).read_text())
    for path, digest in cfg["experiment"]["implementation_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Changed authenticated implementation: {path}")
    verify_protocol(cfg)
    return cfg


def model_for(config, checkpoint, device):
    taxonomy = load_taxonomy(config)
    heads = {k: taxonomy.slot_classes(k) for k in taxonomy.slotted_intents}
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(Path(config).read_text())
    if state["protocol_sha256"] != sha256(cfg["data"]["protocol"]):
        raise ValueError("Checkpoint protocol mismatch")
    if state["arch"] == NAME:
        model = ValidityModel(taxonomy.n_way_head, heads)
    elif state["arch"] == "bc_resnet_broadcast8":
        model = engine.build_broadcast_model(taxonomy.n_way_head, heads)
    else:
        raise ValueError("Unexpected architecture")
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), taxonomy


def collect(model, data, taxonomy, device, batch=256):
    logits, valid, slot_ok, slot_right, slot_count = [], [], [], 0, 0
    with torch.no_grad():
        for x, y, slots in DataLoader(data, batch_size=batch, shuffle=False):
            output = model(x.to(device))
            ok = slots < 0
            for intent, values in output["slots"].items():
                mask = (y == taxonomy.intent_to_idx[intent]) & (slots >= 0)
                correct = values.cpu().argmax(1)[mask] == slots[mask]
                ok[mask] = correct
                slot_right += int(correct.sum())
                slot_count += int(mask.sum())
            logits.extend(output["intent"].float().cpu().numpy())
            slot_ok.extend(ok.numpy())
            if "validity" in output:
                valid.extend(output["validity"].float().sigmoid().cpu().numpy())
    values = np.asarray(logits)
    y = data.labels
    pred = values.argmax(1)
    counts = np.bincount(y * taxonomy.n_way_head + pred, minlength=taxonomy.n_way_head**2).reshape(taxonomy.n_way_head, -1)
    raw = engine.summarize(counts)
    raw.update(e2e_accuracy=float(((pred == y) & slot_ok).mean()),
               slot_accuracy=slot_right / slot_count if slot_count else None, slot_n=slot_count,
               groups=engine.grouped_report(data, pred))
    result = {"logits": values, "labels": y, "slot_ok": np.asarray(slot_ok),
              "real": np.asarray([r["is_synthetic"] == "0" for r in data.rows])}
    if valid:
        result["validity"] = np.asarray(valid)
    return result, raw


def scores(pack, oos):
    logits = pack["logits"]
    probabilities = softmax(logits, axis=1)
    scope = np.delete(logits, oos, axis=1)
    result = {"confidence": probabilities.max(1), "scope_probability": 1 - probabilities[:, oos],
              "scope_energy": logsumexp(scope, axis=1)}
    if "validity" in pack:
        result["validity"] = pack["validity"]
    # Every policy retains the learned OOS veto; it can only reject more.
    veto = logits.argmax(1) == oos
    return {k: np.where(veto, -np.inf, v) for k, v in result.items()}


def decision_metrics(pack, score, threshold, oos):
    y = pack["labels"]
    predicted = pack["logits"].argmax(1)
    accept = score >= threshold
    prediction = np.where(accept, predicted, oos)
    known = y != oos
    confusion = np.bincount(y * pack["logits"].shape[1] + prediction,
                  minlength=pack["logits"].shape[1]**2).reshape(pack["logits"].shape[1], -1)
    result = engine.summarize(confusion)
    result.update(e2e_accuracy=float(((prediction == y) & pack["slot_ok"]).mean()),
        false_accepts=int((accept & ~known).sum()), oos_n=int((~known).sum()),
        false_rejects=int((~accept & known).sum()), in_scope_n=int(known.sum()),
        real_e2e_accuracy=float(((prediction == y) & pack["slot_ok"])[pack["real"]].mean()) if pack["real"].any() else None)
    return result


def select_policy(val, tune, oos):
    if not np.all(tune["labels"] == oos):
        raise ValueError("OOS tuning pool contains an in-scope label")
    val_scores, tune_scores = scores(val, oos), scores(tune, oos)
    policies = []
    for method, score in val_scores.items():
        finite = tune_scores[method][np.isfinite(tune_scores[method])]
        thresholds = np.unique(np.concatenate(([np.finfo(float).min, np.finfo(float).max],
            finite, np.nextafter(finite, np.inf))))
        best = None
        for threshold in thresholds:
            far = float((tune_scores[method] >= threshold).mean())
            if far > .05:
                continue
            metrics = decision_metrics(val, score, threshold, oos)
            item = {"method": method, "threshold": float(threshold), "tune_false_accept_rate": far,
                    "validation": metrics}
            if best is None or (metrics["e2e_accuracy"], -metrics["false_rejects"]) > (
                    best["validation"]["e2e_accuracy"], -best["validation"]["false_rejects"]):
                best = item
        policies.append(best)
    return max(policies, key=lambda p: (p["validation"]["e2e_accuracy"], -p["validation"]["false_rejects"])), policies


def datasets(config, device):
    cfg = check(config)
    cache, index = load_feature_cache(cfg)
    taxonomy = load_taxonomy(config)
    def make(split, augment=False):
        return SourceDataset(cfg, split, taxonomy, cache, index, int(cfg["training"]["seed"]), augment=augment)
    return cfg, taxonomy, make


def baseline():
    torch.set_num_threads(1)
    results = []
    for tag in ("run_042_hf_grouped_v2b_bc8_s42", PARENT):
        root = Path("experiments/vcm") / tag
        config = str(root / "config.yaml")
        _, tax, make = datasets(config, "cpu")
        model, _ = model_for(config, root / "best.pt", "cpu")
        val, raw = collect(model, make("val"), tax, "cpu", 16)
        tune, _ = collect(model, make("tune_oos"), tax, "cpu", 16)
        chosen, policies = select_policy(val, tune, tax.oos_idx)
        results.append({"tag": tag, "raw": raw, "chosen": chosen, "policies": policies,
                        "checkpoint_sha256": sha256(root / "best.pt")})
        print(tag, "baseline gated validation", chosen["validation"]["e2e_accuracy"], flush=True)
    write(STAGE / "baselines.json", results)


def train(config):
    cfg, tax, make = datasets(config, "cuda")
    tr, ex = cfg["training"], cfg["experiment"]
    output = Path("experiments/vcm") / ex["tag"]
    if output.exists():
        raise FileExistsError(output)
    seed = int(tr["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(tr["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ValidityModel(tax.n_way_head, {k: tax.slot_classes(k) for k in tax.slotted_intents})
    parent = Path("experiments/vcm") / PARENT / "best.pt"
    checkpoint = torch.load(parent, map_location="cpu", weights_only=False)
    if checkpoint["protocol_sha256"] != sha256(cfg["data"]["protocol"]):
        raise ValueError("Warm-start protocol mismatch")
    model.backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    training, validation, tuning = make("train", True), make("val"), make("tune_oos")
    workers = int(tr["num_workers"])
    loader = DataLoader(training, batch_size=tr["batch_size"], shuffle=True, num_workers=workers,
        generator=generator, worker_init_fn=engine.seed_worker, persistent_workers=workers > 0,
        pin_memory=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(training.labels, tax.n_way_head, device), label_smoothing=tr["label_smoothing"])
    class TrainingLoss(torch.nn.Module):
        def forward(self, logits, labels):
            return balanced_intent_loss(logits, labels, criterion, self.multipliers) + ex["validity_loss_weight"] * balanced_validity_loss(model.validity_logits, labels, tax.oos_idx)
    loss = TrainingLoss()
    class Batches:
        def __iter__(self):
            for x, y, slots, multipliers in loader:
                loss.multipliers = multipliers.to(device, dtype=torch.float32)
                yield x, y, slots
    def lr_factor(epoch):
        if epoch < tr["warmup_epochs"]:
            return (epoch + 1) / max(tr["warmup_epochs"], 1)
        return .5 * (1 + math.cos(math.pi * (epoch - tr["warmup_epochs"]) / max(tr["epochs"] - tr["warmup_epochs"], 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    output.mkdir()
    shutil.copy2(config, output / "config.yaml")
    for path in ex["implementation_sha256"]:
        target = output / "implementation_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(output / "reproducibility.json", {"protocol_sha256": sha256(cfg["data"]["protocol"]),
        "config_sha256": sha256(config), "parent_checkpoint_sha256": sha256(parent),
        "architecture": NAME, "seed": seed, "selection": "Best gated validation E2E at <=5% tune OOS false accepts",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": os.sys.version,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "deterministic": True, "optimizer_reset": True, "test_holdout_challenge_used_for_selection": False})
    (output / "environment_freeze.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
    best, bad, history = -1., 0, []
    start = time.monotonic()
    for epoch in range(tr["epochs"]):
        training.epoch.fill_(epoch)
        tm, _ = engine.epoch_pass(model, Batches(), device, tax, loss, optimizer,
            device.type == "cuda" and torch.cuda.is_bf16_supported())
        model.eval()
        val, raw = collect(model, validation, tax, device)
        tune, _ = collect(model, tuning, tax, device)
        policy, policies = select_policy(val, tune, tax.oos_idx)
        score = policy["validation"]["e2e_accuracy"]
        record = {"epoch": epoch, "augmented_train": tm, "validation": raw, "policy": policy,
                  "elapsed_s": time.monotonic() - start, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        scheduler.step()
        improved = score > best
        best, bad = (score, 0) if improved else (best, bad + 1)
        state = {"arch": NAME, "state_dict": model.state_dict(), "epoch": epoch,
            "protocol_sha256": sha256(cfg["data"]["protocol"]), "config_sha256": sha256(config),
            "seed": seed, "policy": policy, "raw_validation": raw, "policies": policies,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_score": best, "bad_epochs": bad}
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        write(output / "history.json", history)
        print(f'{ex["tag"]} epoch={epoch+1}/{tr["epochs"]} raw_e2e={raw["e2e_accuracy"]:.4f} gated_e2e={score:.4f} policy={policy["method"]} bad={bad}', flush=True)
        if bad >= tr["early_stop_patience"]:
            break
    winner = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    write(output / "result.json", {"tag": ex["tag"], "arch": NAME, "epochs_run": len(history),
        "best_epoch": winner["epoch"], "best_policy": winner["policy"],
        "raw_validation": winner["raw_validation"], "duration_s": time.monotonic() - start,
        "n_params": sum(p.numel() for p in model.parameters())})


def finish(config):
    cfg, tax, make = datasets(config, "cpu")
    torch.set_num_threads(1)
    output = Path("experiments/vcm") / cfg["experiment"]["tag"]
    model, _ = model_for(config, output / "best.pt", "cpu")
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    policy = checkpoint["policy"]
    for split in ("train", "challenge", "test_core", "test_negatives", "test"):
        pack, raw = collect(model, make(split), tax, "cpu", 16)
        gated = decision_metrics(pack, scores(pack, tax.oos_idx)[policy["method"]], policy["threshold"], tax.oos_idx)
        write(output / f"metrics_{split}.json", {"raw": raw, "gated": gated, "policy": policy,
            "selection_frozen_before_access": True, "manifest_sha256": sha256(Path(cfg["data"]["manifests"]) / f"{split}_19intent.csv")})
    class Export(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, x):
            result = self.model(x)
            return tuple([result["intent"]] + list(result["slots"].values()) + [result["validity"]])
    wrapper = Export(model).eval()
    names = ["intent_logits"] + ["slot_" + k for k in tax.slotted_intents] + ["validity_logit"]
    x = torch.randn(1, 1, 80, 198)
    torch.onnx.export(wrapper, x, str(output / "model.onnx"), input_names=["mel"], output_names=names,
        dynamic_axes={k: {0: "batch"} for k in ["mel"] + names}, opset_version=17)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(output / "model.onnx"))
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        sample = torch.randn(batch, 1, 80, 198)
        with torch.no_grad():
            expected = wrapper(sample)
        for a, b in zip(session.run(None, {"mel": sample.numpy()}), expected):
            np.testing.assert_allclose(a, b.numpy(), rtol=1e-4, atol=1e-4)
    durations = []
    sample = x.numpy()
    for _ in range(20):
        session.run(None, {"mel": sample})
    for _ in range(100):
        begin = time.perf_counter(); session.run(None, {"mel": sample})
        durations.append((time.perf_counter() - begin) * 1000)
    write(output / "export_validation.json", {"parity_passed": True, "pi_verified": False,
        "bytes": (output / "model.onnx").stat().st_size, "sha256": sha256(output / "model.onnx"),
        "host_p50_ms": float(np.median(durations)), "host_p95_ms": float(np.percentile(durations, 95)),
        "runtime_requires_new_validity_output_and_policy_support": True})
    write(output / "labels.json", {"intents": list(tax.intents) + [tax.oos_class],
        "slots": {k: tax.slot_classes(k) for k in tax.slotted_intents}})
    write(output / "deployment_policy.json", policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "train", "finish"])
    parser.add_argument("--config")
    args = parser.parse_args()
    {"baseline": baseline, "train": lambda: train(args.config), "finish": lambda: finish(args.config)}[args.mode]()
