"""Versioned BC8 trainer with explicit metrics and reproducibility controls.

The historical trainer remains untouched so its authenticated protocols work.
"""
import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .augmentation import AugConfig, augment_mel
from .broadcast import NAME, build_broadcast_model
from .protocol import load_feature_cache, sha256, verify_protocol
from .taxonomy import load_taxonomy
from .train import _class_weights, _slot_label, _slot_loss_and_stats


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class CachedDataset(Dataset):
    def __init__(self, config, split, taxonomy, cache, index, seed, augment=False):
        directory = Path(config["data"]["manifests"])
        with (directory / f"{split}_19intent.csv").open(newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        self.cache, self.index, self.seed, self.augment = cache, index, seed, augment
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.labels = np.array([taxonomy.oos_idx if r["intent"] == taxonomy.oos_class
                                else taxonomy.intent_to_idx[r["intent"]] for r in self.rows])
        self.slots = np.array([_slot_label(r["intent"], r["slot"] or None, taxonomy)
                               if r["slot"] else None for r in self.rows], dtype=object)
        self.slots = np.array([-1 if x is None else x for x in self.slots], dtype=np.int64)
        self.augmentation = AugConfig(enabled=True, noise_prob=0.0, speed_perturb=(1.0,))
        if any(r["audio_path"] not in index for r in self.rows):
            raise ValueError(f"Missing cached inputs in {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        mel = np.array(self.cache[self.index[self.rows[i]["audio_path"]]], copy=True)
        if self.augment:
            # Keyed to sample and epoch, not worker scheduling or loader count.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self.epoch), i]))
            mel = augment_mel(mel, self.augmentation, rng)
        return torch.from_numpy(mel), int(self.labels[i]), int(self.slots[i])


def summarize(confusion):
    diagonal = np.diag(confusion)
    supports = confusion.sum(1)
    predicted = confusion.sum(0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal, dtype=float), where=predicted > 0)
    recall = np.divide(diagonal, supports, out=np.zeros_like(diagonal, dtype=float), where=supports > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    return {"intent_accuracy": float(diagonal.sum() / max(confusion.sum(), 1)),
            "macro_f1": float(f1.mean()), "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(),
            "confusion_matrix": confusion.tolist()}


def epoch_pass(model, loader, device, taxonomy, intent_loss, optimizer=None, amp=False):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((taxonomy.n_way_head, taxonomy.n_way_head), dtype=np.int64)
    total, loss_sum, nll_sum, slot_correct, slot_n, e2e = 0, 0.0, 0.0, 0, 0, 0
    predictions = []
    with torch.set_grad_enabled(training):
        for x, y, slots in loader:
            x, y, slots = (a.to(device, non_blocking=device.type == "cuda") for a in (x, y, slots))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and training):
                output = model(x)
                logits = output["intent"]
                slot_loss, correct_slots, num_slots = _slot_loss_and_stats(
                    output, y, slots, taxonomy, nn.CrossEntropyLoss())
                loss = intent_loss(logits, y)
                if slot_loss is not None:
                    loss = loss + 0.5 * slot_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; training halted")
            if training:
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)
            intent_ok = pred == y
            slot_ok = slots < 0
            for intent, values in output["slots"].items():
                mask = (y == taxonomy.intent_to_idx[intent]) & (slots >= 0)
                slot_ok[mask] = values[mask].argmax(1) == slots[mask]
            labels, preds = y.cpu().numpy(), pred.detach().cpu().numpy()
            confusion += np.bincount(labels * taxonomy.n_way_head + preds,
                                    minlength=taxonomy.n_way_head**2).reshape(confusion.shape)
            predictions.extend(preds.tolist())
            count = len(y)
            total += count
            loss_sum += float(loss.detach()) * count
            nll_sum += float(nn.functional.cross_entropy(logits.detach().float(), y)) * count
            slot_correct += correct_slots
            slot_n += num_slots
            e2e += int((intent_ok & slot_ok).sum())
    result = summarize(confusion)
    result.update(n=total, loss=loss_sum / max(total, 1), intent_nll=nll_sum / max(total, 1),
                  slot_accuracy=slot_correct / slot_n if slot_n else None,
                  slot_n=slot_n, e2e_accuracy=e2e / max(total, 1))
    return result, predictions


def grouped_report(dataset, predictions):
    reports = {}
    for field in ("is_synthetic", "source", "speaker"):
        groups = {}
        for row, true, predicted in zip(dataset.rows, dataset.labels, predictions):
            name = row[field]
            if field == "speaker":
                identities = json.loads(row.get("speech_voices", json.dumps([row["speaker"]])))
                if len(identities) != 1:
                    continue
                name = identities[0]
            counts = groups.setdefault(name, [0, 0])
            counts[0] += 1
            counts[1] += int(true == predicted)
        reports[field] = {name: {"n": n, "intent_accuracy": correct / n}
                          for name, (n, correct) in sorted(groups.items())}
    speaker_scores = [x["intent_accuracy"] for x in reports["speaker"].values()]
    reports["speaker_macro_intent_accuracy"] = float(np.mean(speaker_scores))
    reports["speaker_metric_semantics"] = "Single declared speech identity only; exclude noise-only and multi-voice composites"
    return reports


def run(tag, config):
    cfg = yaml.safe_load(Path(config).read_text())
    if cfg["experiment"]["architecture"] != NAME:
        raise ValueError("This versioned trial trains genuine BC8 only")
    for path, digest in cfg["experiment"]["implementation_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Implementation changed: {path}")
    protocol = verify_protocol(cfg)
    cache, index = load_feature_cache(cfg)
    tr = cfg["training"]
    if (tr["slot_loss_weight"] != 0.5 or tr["augment_mode"] != "spec_aug"
            or tr["optimizer"].lower() != "adamw" or tr.get("intent_loss_weight", 1.0) != 1.0):
        raise ValueError("Controlled trial requires unchanged slot loss and augmentation")
    output = Path("experiments/vcm") / tag
    output.mkdir(exist_ok=False)
    shutil.copy2(config, output / "config.yaml")
    for path in cfg["experiment"]["implementation_sha256"]:
        target = output / "implementation_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    seed = int(tr["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if tr["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; never silently fall back to CPU")
    device = torch.device(tr["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    taxonomy = load_taxonomy(config)
    heads = {name: taxonomy.slot_classes(name) for name in taxonomy.slotted_intents}
    model = build_broadcast_model(taxonomy.n_way_head, heads).to(device)
    workers = int(tr["num_workers"])
    training = CachedDataset(cfg, "train", taxonomy, cache, index, seed, augment=True)
    validation = CachedDataset(cfg, "val", taxonomy, cache, index, seed)
    generator = torch.Generator().manual_seed(seed)
    def loader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=int(tr["batch_size"]), shuffle=shuffle,
                          num_workers=workers, pin_memory=device.type == "cuda",
                          persistent_workers=workers > 0, worker_init_fn=seed_worker,
                          generator=generator if shuffle else torch.Generator().manual_seed(seed + 1))
    train_loader, val_loader = loader(training, True), loader(validation)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(training.labels, taxonomy.n_way_head, device),
                                   label_smoothing=float(tr["label_smoothing"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    epochs, warmup = int(tr["epochs"]), int(tr["warmup_epochs"])
    def learning_rate(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    provenance = {"tag": tag, "seed": seed, "architecture": NAME,
                  "protocol_sha256": sha256(cfg["data"]["protocol"]), "protocol": protocol,
                  "config_sha256": sha256(config), "implementation_sha256": cfg["experiment"]["implementation_sha256"],
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                  "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                  "deterministic_algorithms": True, "cudnn_benchmark": False,
                  "precision": "BF16 training; FP32 clean evaluation" if amp else "FP32",
                  "selection": "best raw validation strict E2E; ties retain earlier epoch",
                  "test_or_holdout_used_for_training": False}
    (output / "reproducibility.json").write_text(json.dumps(provenance, indent=2))
    (output / "environment_freeze.txt").write_text(subprocess.check_output(
        [os.sys.executable, "-m", "pip", "freeze"], text=True))
    best_score, best_epoch, bad, history = -1.0, -1, 0, []
    start = time.monotonic()
    with (output / "train.log").open("x") as stream:
        def log(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()
        log(f"[{tag}] train={len(training)} val={len(validation)} deterministic=True amp={amp}")
        for epoch in range(epochs):
            training.epoch.fill_(epoch)
            lr = optimizer.param_groups[0]["lr"]
            train_metrics, _ = epoch_pass(model, train_loader, device, taxonomy, criterion, optimizer, amp)
            val_metrics, val_predictions = epoch_pass(model, val_loader, device, taxonomy, criterion)
            scheduler.step()
            record = {"epoch": epoch, "learning_rate": lr, "augmented_train": train_metrics,
                      "validation": val_metrics, "validation_groups": grouped_report(validation, val_predictions),
                      "elapsed_s": round(time.monotonic() - start, 2)}
            history.append(record)
            log(f"[{tag}] epoch {epoch:02d} augmented_train_acc={train_metrics['intent_accuracy']:.4f} "
                f"train_loss={train_metrics['loss']:.4f} val_acc={val_metrics['intent_accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} val_e2e={val_metrics['e2e_accuracy']:.4f} "
                f"val_slot={val_metrics['slot_accuracy']:.4f} ({record['elapsed_s']:.0f}s)")
            improved = val_metrics["e2e_accuracy"] > best_score
            if improved:
                best_score, best_epoch, bad = val_metrics["e2e_accuracy"], epoch, 0
            else:
                bad += 1
            checkpoint = {"state_dict": model.state_dict(), "epoch": epoch,
                          "val_acc": val_metrics["intent_accuracy"], "val_slot_acc": val_metrics["slot_accuracy"],
                          "arch": NAME, "seed": seed, "n_way": taxonomy.n_way_head,
                          "n_params": sum(p.numel() for p in model.parameters()),
                          "slot_heads": {k: len(v) for k, v in heads.items()},
                          "protocol_sha256": provenance["protocol_sha256"], "config_sha256": sha256(config),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                          "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                          "best_epoch": best_epoch, "best_val_score": best_score, "bad_epochs": bad}
            if improved:
                torch.save(checkpoint, output / "best.pt")
            torch.save(checkpoint, output / "last.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if bad >= int(tr["early_stop_patience"]):
                log(f"[{tag}] early stop after {bad} nonimproving epochs")
                break
        model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=False)["state_dict"])
        clean = CachedDataset(cfg, "train", taxonomy, cache, index, seed)
        clean_metrics, clean_predictions = epoch_pass(model, loader(clean), device, taxonomy, criterion)
        clean_metrics["groups"] = grouped_report(clean, clean_predictions)
        clean_metrics["semantics"] = "Best checkpoint, model.eval(), no augmentation, raw argmax, seen training examples"
        (output / "metrics_train_clean.json").write_text(json.dumps(clean_metrics, indent=2))
        result = {"tag": tag, "arch": NAME, "seed": seed, "n_params": checkpoint["n_params"],
                  "epochs_run": len(history), "best_epoch": best_epoch, "best_val_score": best_score,
                  "duration_s": round(time.monotonic() - start), "history": history}
        (output / "result.json").write_text(json.dumps(result, indent=2))
        log(f"[{tag}] DONE best_epoch={best_epoch} val_e2e={best_score:.4f} clean_train_acc={clean_metrics['intent_accuracy']:.4f}")
    return output
