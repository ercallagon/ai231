"""VCM evaluation: intent + slot + OOS-rejection metrics.

Loads a best.pt checkpoint and scores a named split (test/holdout/val).
Writes metrics.json + evaluation.log into the experiment dir.

Usage (module):
    python -m hey_swift.vcm.evaluate --tag a0_scratch --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import dataset as ds
from .features import FeatureConfig, MelFeatures
from .taxonomy import load_taxonomy


def _load_model(config: str, ckpt_path: Path, device):
    import torch
    from .models import ArchSpec, build_model
    tax = load_taxonomy(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Determine arch from checkpoint metadata
    arch_name = ckpt.get("arch", "ds_cnn")
    n_way = ckpt.get("n_way", tax.n_way_head)
    # Infer channels from n_params or use defaults
    n_params = ckpt.get("n_params", 0)
    if arch_name == "ds_cnn":
        # A0: [32,64,128,256] ~394K, A1: [64,128,256,256] ~1.2M
        if n_params > 800_000:
            spec = ArchSpec.a1(n_way=n_way)
        else:
            spec = ArchSpec.a0(n_way=n_way)
    elif arch_name == "crnn":
        spec = ArchSpec.a3(n_way=n_way)
    elif arch_name == "bc_resnet":
        spec = ArchSpec.a4(n_way=n_way)
    elif arch_name == "ds_cnn_sep":
        spec = ArchSpec.a5(n_way=n_way)
    elif arch_name == "resnet2d_small":
        spec = ArchSpec.a6(n_way=n_way)
    elif arch_name == "mobilenetv2_tiny":
        spec = ArchSpec.a7(n_way=n_way)
    elif arch_name == "crnn_attn":
        spec = ArchSpec.a8(n_way=n_way)
    elif arch_name == "conformer_tiny":
        spec = ArchSpec.a9(n_way=n_way)
    elif arch_name == "ds_resnet_se":
        spec = ArchSpec.a10(n_way=n_way)
    elif arch_name == "matchboxnet":
        spec = ArchSpec.a11(n_way=n_way)
    elif arch_name == "mobilenetv3_small":
        spec = ArchSpec.a12(n_way=n_way)
    elif arch_name == "tc_resnet":
        # A2 (w32, ~75K) or A2W (w64, ~298K) — distinguish by param count.
        # Gap between variants is wide (75K vs 298K), so a 150K midpoint
        # threshold is unambiguous.
        if n_params > 150_000:
            spec = ArchSpec("tc_resnet", [64, 64, 64, 64], width=64, n_way_head=n_way)
        else:
            spec = ArchSpec.a2(n_way=n_way)
    else:
        spec = ArchSpec.a2(n_way=n_way)
    # Only build slot heads if they exist in the checkpoint
    sd = ckpt["state_dict"]
    has_slots = any(k.startswith("slot_head_map.") for k in sd.keys())
    if has_slots:
        slot_heads = {intent: tax.slot_classes(intent)
                      for intent in tax.slotted_intents if tax.slot_classes(intent)}
    else:
        slot_heads = None
    model = build_model(spec, slot_heads=slot_heads)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, tax, ckpt


def evaluate(tag: str, split: str, config: str = "configs/vcm.yaml",
             exp_dir: Optional[Path] = None, device_str: str = "cpu") -> dict:
    import torch
    import torch.nn.functional as F

    exp_dir = Path(exp_dir or Path("experiments/vcm") / tag)
    ckpt_path = exp_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    model, tax, ckpt = _load_model(config, ckpt_path, device)
    from .protocol import load_feature_cache, sha256, verify_protocol
    cfg = yaml.safe_load(Path(config).read_text())
    protocol_info = verify_protocol(cfg)
    if protocol_info and ckpt.get("protocol_sha256") != sha256(cfg["data"]["protocol"]):
        raise ValueError("Checkpoint belongs to a different or legacy dataset protocol")
    feat_cfg = FeatureConfig.from_dict(cfg.get("audio", {}))
    feats = MelFeatures(feat_cfg)

    # Load MFCC cache if available (much faster than WAV decode)
    mfcc_cache, mfcc_index = load_feature_cache(cfg)
    if mfcc_cache is not None:
        print(f"[eval] MFCC cache loaded: {len(mfcc_index)} clips", flush=True)

    rows = ds.configured_rows(cfg, split)
    if protocol_info and any(row.audio_path not in mfcc_index for row in rows):
        raise ValueError("Corrected evaluation cache is incomplete")
    gated_confusion = {}
    gated_intent_correct = gated_e2e_correct = 0
    threshold = float(cfg.get("oos", {}).get("confidence_threshold", 0.5))
    intent_correct = 0
    slot_n = slot_correct = 0
    e2e_correct = 0
    oos_tp = oos_fp = oos_fn = 0
    confidences = []
    per_intent = {}
    true_counts = {}
    pred_counts = {}
    correct_counts = {}
    t0 = time.time()

    with torch.no_grad():
        for i, row in enumerate(rows):
            # Fast path: MFCC cache
            if mfcc_cache is not None and row.audio_path in mfcc_index:
                mel = torch.from_numpy(np.array(mfcc_cache[mfcc_index[row.audio_path]]))[None, :, :, :].to(device)
            else:
                wav = ds.load_row_wav(row)
                mel = torch.from_numpy(feats.transform(wav))[None, :, :, :].to(device)
            out = model(mel)
            logits = out["intent"] if isinstance(out, dict) else out
            probs = torch.softmax(logits, dim=-1)
            pred = int(logits.argmax().item())
            conf = float(probs.max().item())
            y = tax.oos_idx if row.intent == tax.oos_class else tax.intent_to_idx[row.intent]

            intent_correct += int(pred == y)
            intent_ok = pred == y
            y_name = tax.idx_to_intent.get(y, tax.oos_class)
            pred_name = tax.idx_to_intent.get(pred, tax.oos_class)
            key = (y_name, pred_name)
            per_intent[key] = per_intent.get(key, 0) + 1
            true_counts[y_name] = true_counts.get(y_name, 0) + 1
            pred_counts[pred_name] = pred_counts.get(pred_name, 0) + 1
            if pred_name == y_name:
                correct_counts[y_name] = correct_counts.get(y_name, 0) + 1

            if y == tax.oos_idx:
                if pred == tax.oos_idx:
                    oos_tp += 1
                else:
                    oos_fn += 1
            else:
                if pred == tax.oos_idx:
                    oos_fp += 1
            confidences.append(conf)

            slot_ok = True
            if row.intent in tax.slotted_intents and row.slot:
                slot_ok = False
                from .train import _slot_label
                target = _slot_label(row.intent, row.slot, tax)
                if target is None and protocol_info:
                    raise ValueError(f"Unsupported evaluation slot: {row.audio_path}")
                if target is not None:
                    slot_n += 1
                    slots = out.get("slots", {}) if isinstance(out, dict) else {}
                    if row.intent in slots:
                        slot_ok = int(slots[row.intent].argmax().item()) == target
                        slot_correct += int(slot_ok)
            e2e_correct += int(intent_ok and slot_ok)
            gated_pred = tax.oos_idx if conf < threshold else pred
            gated_name = tax.idx_to_intent.get(gated_pred, tax.oos_class)
            gated_intent_correct += int(gated_pred == y)
            gated_e2e_correct += int(gated_pred == y and slot_ok)
            gated_key = f"{y_name}->{gated_name}"
            gated_confusion[gated_key] = gated_confusion.get(gated_key, 0) + 1

    n = len(rows)
    oos_prec = oos_tp / max(oos_tp + oos_fp, 1)
    oos_rec = oos_tp / max(oos_tp + oos_fn, 1)
    class_names = [tax.idx_to_intent.get(i, tax.oos_class) for i in range(tax.n_way_head)]
    per_class = {}
    f1s = []
    for name in class_names:
        tp = correct_counts.get(name, 0)
        prec = tp / max(pred_counts.get(name, 0), 1)
        rec = tp / max(true_counts.get(name, 0), 1)
        f1 = (2 * prec * rec / max(prec + rec, 1e-12)) if tp else 0.0
        per_class[name] = {
            "support": true_counts.get(name, 0),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)
    metrics = {
        "tag": tag, "split": split, "n": n,
        "intent_accuracy": round(intent_correct / n, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "e2e_accuracy": round(e2e_correct / n, 4),
        "slot_accuracy": round(slot_correct / slot_n, 4) if slot_n else None,
        "slot_n": slot_n,
        "oos_precision": round(oos_prec, 4),
        "oos_recall": round(oos_rec, 4),
        "oos_tp": oos_tp, "oos_fp": oos_fp, "oos_fn": oos_fn,
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "per_class": per_class,
        "confusion": {f"{a}->{b}": c for (a, b), c in sorted(per_intent.items())},
        "ckpt": {"epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc"),
                 "seed": ckpt.get("seed"), "n_params": ckpt.get("n_params")},
        "duration_s": int(time.time() - t0),
        "manifest_sha256": sha256(ds.manifest_path(cfg, split)),
        "protocol_sha256": sha256(cfg["data"]["protocol"]) if protocol_info else None,
        "metric_semantics": "Headline metrics are raw argmax; slot accuracy uses the ground-truth intent head; strict E2E requires both intent and applicable slot.",
        "gated": {
            "threshold": threshold,
            "intent_accuracy": round(gated_intent_correct / n, 4),
            "e2e_accuracy": round(gated_e2e_correct / n, 4),
            "confusion": gated_confusion,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (exp_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
    with (exp_dir / f"evaluation_{split}.log").open("w") as fh:
        fh.write(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "test_core", "test_negatives", "holdout", "tune_oos"])
    ap.add_argument("--config", default="configs/vcm.yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m = evaluate(args.tag, args.split, args.config, device_str=args.device)
    print(json.dumps({k: v for k, v in m.items() if k != "confusion"}, indent=2))
