"""Wake-word evaluation (faithful exp15 sweep).

Loads a checkpoint (or the deployed ONNX) and runs the historical threshold
sweep on the frozen validation and test splits. Reports, per split, the best
operating point under the FA cap (max recall then F1 among thresholds with
FA <= fa_cap) plus the full sweep. This is the same procedure that selected the
historical exp15 checkpoint.

Requires the 'train' extra (torch) to load a ``.pt`` checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from . import dataset as ds
from .train import probs_of, sweep, temporal_metrics, _build_torch_dataset
from .model import WakeArchSpec, build_model

EXP_ROOT = Path("experiments/wakeword")


def _loaders(config, batch, device, workers):
    import torch
    from torch.utils.data import DataLoader
    cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
    seed = int((cfg.get("training", {}) or {}).get("seed", 20260926))
    loaders = {}
    for split in ("validation", "test"):
        items = ds.collect_split(split)
        d = _build_torch_dataset(items, False, None, 0.0, 0, 0.0, 0.0,
                                 (0, 1), 0.0, (0, 0), seed)
        loaders[split] = DataLoader(d, batch_size=batch, shuffle=False,
                                    num_workers=workers)
    return loaders


def run(tag: str, config: str = "configs/wakeword.yaml",
        threshold: Optional[float] = None, fa_cap: Optional[float] = None,
        ckpt: Optional[str] = None) -> dict:
    import torch

    cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
    tr = cfg.get("training", {}) or {}
    inf = cfg.get("inference", {}) or {}
    fa_cap = float(fa_cap if fa_cap is not None else tr.get("fa_cap", 0.05))
    batch = int(tr.get("batch", 32))
    workers = int(tr.get("workers", 0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(WakeArchSpec()).to(device)
    ckpt_path = Path(ckpt) if ckpt else (EXP_ROOT / tag / "best.pt")
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
    model.eval()

    loaders = _loaders(config, batch, device, workers)
    out = {"tag": tag, "checkpoint": str(ckpt_path), "fa_cap": fa_cap, "splits": {}}
    for name in ("val", "test"):
        split = "validation" if name == "val" else "test"
        p, y = probs_of(model, loaders[split], device)
        sv = sweep(p, y)
        cand = [m for m in sv if m["fa"] <= fa_cap] or sv
        best = max(cand, key=lambda m: (m["recall"], m["f1"]))
        tm = temporal_metrics(p, y, thr=best["thr"])
        out["splits"][name] = dict(best=best, temporal=tm, sweep=sv)
        print(f"[{name}] BEST thr={best['thr']:.2f} P={best['precision']:.3f} "
              f"R={best['recall']:.3f} F1={best['f1']:.3f} FA={best['fa'] * 100:.2f}% "
              f"FR={best['fr'] * 100:.1f}%  (tp{best['tp']} fp{best['fp']} "
              f"fn{best['fn']} tn{best['tn']})")

    dest = EXP_ROOT / tag / "eval_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    return out
