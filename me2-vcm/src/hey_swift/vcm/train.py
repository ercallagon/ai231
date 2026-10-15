"""Reproducible VCM training loop.

Contract (docs/decisions.md, "Hard rules"):
    * HF-only data (no local corpus).
    * From-scratch in phase 1 (no weight loading from prior runs).
    * Immutable artifacts: outputs go to experiments/vcm/<tag>/ and are written
      once; a re-run uses a NEW tag.
    * Frozen test + holdout are NEVER trained on.
    * Speaker/voice disjointness is verified and recorded.

Device policy (user constraint, Oct 2026): 1 GPU by default, max 2. We pin a
SINGLE GPU via CUDA_VISIBLE_DEVICES so a run never silently spreads across all
visible devices.

Requires the 'train' extra (torch).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from . import dataset as ds
from .augmentation import AugConfig, add_noise, apply_gain, augment_mel, perturb_speed
from .features import FeatureConfig, MelFeatures
from .taxonomy import load_taxonomy
from .protocol import load_feature_cache, sha256, verify_protocol

EXP_ROOT = Path("experiments/vcm")  # VCM experiment output root

# Raw slot_value strings (as stored in the HF dataset) -> canonical slot class
# names from configs/vcm.yaml slot_sets. Locked mapping; see docs/dataset.md.
SLOT_VALUE_MAP = {
    "TIMER": {"10 seconds": "10s", "30 seconds": "30s", "1 minute": "1m"},
    "ALARM": {"6:00 AM": "6AM", "8:00 AM": "8AM", "9:00 PM": "9PM"},
    "TEMPERATURE": {"18 degrees": "18", "22 degrees": "22", "26 degrees": "26"},
    "BRIGHTNESS": {"20 percent": "20", "60 percent": "60", "100 percent": "100"},
    "COLOR": {"red": "red", "blue": "blue", "green": "green"},
    "CREATE_REMINDER": {"drink water": "drink_water", "study": "study", "exercise": "exercise"},
}


def _seed_everything(seed: int) -> None:
    """Deterministic seeding across RNGs + cudnn (docs/decisions.md, seed 42)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _pick_device(cfg: dict):
    """Resolve the training device.

    Honors the 1-GPU-default / max-2 constraint: unless the config explicitly
    asks for CPU, pin a SINGLE GPU via CUDA_VISIBLE_DEVICES instead of letting
    torch spread across all visible devices.
    """
    import torch
    dev = str(cfg.get("training", {}).get("device", "cuda"))
    if dev == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_idx = int(cfg.get("training", {}).get("gpu_index", 0))
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    return torch.device("cuda:0")


def _resolve_config(config: str) -> dict:
    with open(config, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _class_weights(labels: list, n_classes: int, device) -> "object":
    """Inverse-frequency class weights (helps rare intents + OOS)."""
    import torch
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _slot_label(intent: str, raw_slot: Optional[str], tax) -> Optional[int]:
    """Map a raw slot_value string to a slot-class index, or None.

    None means: no slot supervision for this row (non-slotted intent, empty
    slot, or an unsupported value — the latter only occurs with slot_reject
    off, where unsupported values are dropped rather than REJECT-labeled).
    """
    if intent not in tax.slotted_intents or raw_slot is None:
        return None
    classes = tax.slot_classes(intent)
    canon = SLOT_VALUE_MAP.get(intent, {}).get(str(raw_slot).strip().lower()) \
        or SLOT_VALUE_MAP.get(intent, {}).get(str(raw_slot).strip())
    if canon is None:
        # try case-insensitive direct membership
        low = {c.lower(): c for c in classes}
        canon = low.get(str(raw_slot).strip().lower())
    if canon is None or canon not in classes:
        return None
    return classes.index(canon)


def _slot_loss_and_stats(out, yb, ys, tax, loss_fn) -> tuple:
    """Compute slot loss/accuracy on rows whose ground-truth intent is slotted."""
    import torch
    slots = out.get("slots", {}) if isinstance(out, dict) else {}
    losses = []
    slot_n = 0
    slot_correct = 0
    for intent_name, logits in slots.items():
        if intent_name not in tax.intent_to_idx:
            continue
        mask = (yb == tax.intent_to_idx[intent_name]) & (ys >= 0)
        if not torch.any(mask):
            continue
        losses.append(loss_fn(logits[mask], ys[mask]))
        slot_correct += (logits[mask].argmax(1) == ys[mask]).sum().item()
        slot_n += int(mask.sum().item())
    if losses:
        return torch.stack(losses).mean(), slot_correct, slot_n
    return None, slot_correct, slot_n


def _evaluate(model, dl, device, tax, slot_loss_fn=None, slot_loss_weight: float = 0.0) -> dict:
    """Return (mean_loss, intent_acc, slot_acc) for a dataloader, no grad."""
    import torch
    import torch.nn.functional as F
    model.eval()
    tot, n, correct = 0.0, 0, 0
    slot_n, slot_correct = 0, 0
    e2e_correct = 0
    with torch.no_grad():
        for xb, yb, ys in dl:
            xb, yb, ys = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), ys.to(device, non_blocking=True)
            out = model(xb)
            logits = out["intent"] if isinstance(out, dict) else out
            loss = out.get("loss") if isinstance(out, dict) else None
            if loss is None:
                loss = F.cross_entropy(logits, yb)
            pred = logits.argmax(1)
            intent_ok = pred == yb
            correct += intent_ok.sum().item()
            n += len(xb)
            slot_ok_by_row = torch.ones_like(intent_ok, dtype=torch.bool)
            if isinstance(out, dict) and out.get("slots") and slot_loss_fn is not None:
                slot_loss, sc, sn = _slot_loss_and_stats(out, yb, ys, tax, slot_loss_fn)
                if slot_loss is not None:
                    loss = loss + slot_loss_weight * slot_loss
                slot_correct += sc
                slot_n += sn
                slot_ok_by_row = ys < 0
                for intent_name, slogits in out.get("slots", {}).items():
                    if intent_name not in tax.intent_to_idx:
                        continue
                    mask = (yb == tax.intent_to_idx[intent_name]) & (ys >= 0)
                    if torch.any(mask):
                        slot_ok_by_row[mask] = slogits[mask].argmax(1) == ys[mask]
            e2e_correct += (intent_ok & slot_ok_by_row).sum().item()
            tot += loss.item() * len(xb)
    model.train()
    return {
        "loss": tot / max(n, 1),
        "intent_acc": correct / max(n, 1),
        "slot_acc": (slot_correct / slot_n) if slot_n else None,
        "slot_n": slot_n,
        "e2e_acc": e2e_correct / max(n, 1),
    }


def run(tag: str, config: str = "configs/vcm.yaml", seed: Optional[int] = None,
        epochs_override: Optional[int] = None, out_root: Optional[str] = None) -> Path:
    """Train one VCM run. Returns the experiment output dir."""
    try:
        import torch
        import torch.nn as nn
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Training requires torch: pip install 'hey-swift[train]'") from exc

    cfg = _resolve_config(config)
    protocol_info = verify_protocol(cfg)
    tax = load_taxonomy(config)
    tr = cfg.get("training", {})
    precomp_tag = str(tr.get("use_precomputed_aug", "") or "").strip()
    if seed is None:
        seed = int(tr.get("seed", 42))
    _seed_everything(seed)

    out_dir = Path(out_root or EXP_ROOT) / tag
    if protocol_info and out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite corrected experiment: {out_dir}")
    if protocol_info and (precomp_tag or tr.get("init_from")):
        raise ValueError("Corrected baseline requires from-scratch training without legacy augmented stores")
    if protocol_info and str(tr.get("device", "cuda")) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Corrected GPU run requires CUDA; refusing CPU fallback")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- reproducibility record ------------------------------------------------
    splits_info = protocol_info or ds.load_splits_json()
    disjoint = ({"validated_protocol": cfg["data"]["protocol"], "disjoint": True}
                if protocol_info else ds.verify_disjointness())
    env = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "reproducibility.json").write_text(json.dumps({
        "tag": tag, "seed": seed, "config": config,
        "precomputed_aug_store": precomp_tag or None,
        "hf_revision": splits_info.get("revision"),
        "splits": splits_info, "disjointness": disjoint,
        "environment": env,
        "config_sha256": sha256(config),
        "effective_manifest_sha256": {s: sha256(ds.manifest_path(cfg, s)) for s in ("train", "val", "test", "holdout")},
        "protocol_sha256": sha256(cfg["data"]["protocol"]) if protocol_info else None,
        "device_policy": "1-GPU-default, max-2 (CUDA_VISIBLE_DEVICES pinned)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    if protocol_info:
        (out_dir / "config.yaml").write_text(Path(config).read_text())

    # --- feature extractor -----------------------------------------------------
    feat_cfg = FeatureConfig.from_dict(cfg.get("audio", {}))
    feats = MelFeatures(feat_cfg)
    device = _pick_device(cfg)
    
    # --- MFCC cache (optional speedup) -----------------------------------------
    # Resolve relative to project root (parent of src/) so it works regardless
    # of the CWD the launcher uses.
    mfcc_cache_dir = Path(cfg.get("data", {}).get("feature_cache", "work/mfcc_cache"))
    mfcc_cache, mfcc_index = load_feature_cache(cfg)
    if mfcc_cache is not None:
        print(f"[{tag}] MFCC cache loaded: {len(mfcc_index)} clips, shape={mfcc_cache.shape} from {mfcc_cache_dir}", flush=True)
    else:
        mfcc_index = {}
        print(f"[{tag}] No MFCC cache at {mfcc_cache_dir}; computing features on-the-fly", flush=True)
    
    # --- Augmentation mode ------------------------------------------------------
    aug_mode = str(tr.get("augment_mode", "full")).lower()
    if protocol_info and aug_mode not in ("spec_aug", "none"):
        raise ValueError("Corrected clean-cache baseline supports spec_aug or none, not silent waveform-augmentation bypass")
    print(f"[{tag}] augmentation mode: {aug_mode}", flush=True)

    # --- dataset ----------------------------------------------------------------
    # Configure augmentation: spec_aug mode enables only spectral augmentation
    # (applied to cached mel, no WAV decode needed). "full" adds speed/noise/gain.
    if aug_mode == "spec_aug":
        aug_cfg = AugConfig(enabled=True, noise_prob=0.0, speed_perturb=(1.0,))
    elif aug_mode == "full":
        aug_cfg = AugConfig(enabled=True)
    else:
        aug_cfg = AugConfig(enabled=False)

    # Optional per-split manifest override (campaign E1/E2 controlled comparison):
    #   data.manifests_override.<split> = path to an alternate CSV with the same
    #   columns. Lets BASE and FULL-ENRICHED share one frozen config while
    #   differing only in the train population. val/test/holdout stay
    #   byte-identical across the comparison.
    def _rows_for(split: str) -> list:
        return ds.configured_rows(cfg, split)

    class VCMDataset(Dataset):
        def __init__(self, split, augment: bool):
            self.rows = _rows_for(split)
            self.augment = augment
            self.rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2, "holdout": 3}[split])
            if protocol_info and any(r.audio_path not in mfcc_index for r in self.rows):
                raise ValueError(f"Corrected cache missing {split} source identities")
            if protocol_info and any(r.intent in tax.slotted_intents and r.slot and _slot_label(r.intent, r.slot, tax) is None for r in self.rows):
                raise ValueError(f"Invalid slot supervision: {split}")
            # Fast path: precomputed store is only wired for the TRAIN split.
            self.use_store = (split == "train" and precomp_store is not None
                              and precomp_index is not None
                              and len(precomp_index) == len(self.rows))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            if protocol_info:
                worker = torch.utils.data.get_worker_info()
                worker_id = worker.id if worker else -1
                if getattr(self, "_rng_worker", None) != worker_id:
                    self.rng = np.random.default_rng(seed + (worker_id + 1) * 10007)
                    self._rng_worker = worker_id
            row = self.rows[i]

            # FAST PATH: precomputed augmented store (fixed tensor, no decode/aug)
            if self.use_store:
                mel = np.array(precomp_store[precomp_index[i]["i"]])
                y = tax.oos_idx if row.intent == tax.oos_class else tax.intent_to_idx[row.intent]
                ys = _slot_label(row.intent, row.slot, tax)
                return (torch.from_numpy(mel), torch.tensor(y, dtype=torch.long),
                        torch.tensor(ys, dtype=torch.long) if ys is not None else torch.tensor(-1, dtype=torch.long))

            # Try MFCC cache first (fast path)
            mel = None
            if mfcc_cache is not None and row.audio_path in mfcc_index:
                idx = mfcc_index[row.audio_path]
                mel = np.array(mfcc_cache[idx])
            
            # Fall back to WAV read + feature extraction (slow path)
            if mel is None:
                wav = ds.load_row_wav(row)
                if self.augment and aug_mode == "full":
                    if self.rng.random() < 0.3:
                        wav = perturb_speed(wav, feat_cfg.sample_rate,
                                            float(self.rng.choice(aug_cfg.speed_perturb)), self.rng)
                    wav = add_noise(wav, feat_cfg.sample_rate, aug_cfg, self.rng)
                    wav = apply_gain(wav, aug_cfg, self.rng)
                mel = feats.transform(wav)
            
            # Apply spec augmentation (all modes except "none")
            if self.augment and aug_mode != "none":
                mel = augment_mel(mel, aug_cfg, self.rng)
            
            y = tax.oos_idx if row.intent == tax.oos_class else tax.intent_to_idx[row.intent]
            ys = _slot_label(row.intent, row.slot, tax)
            return (torch.from_numpy(mel), torch.tensor(y, dtype=torch.long),
                    torch.tensor(ys, dtype=torch.long) if ys is not None else torch.tensor(-1, dtype=torch.long))

    def collate(batch):
        xb = torch.stack([b[0] for b in batch])
        yb = torch.stack([b[1] for b in batch])
        ys = torch.stack([b[2] for b in batch])
        return xb, yb, ys

    # --- precomputed augmented-feature store (fast path) ---------------------
    # When training.use_precomputed_aug is set, load the offline-baked store
    # (data/vcm/precomputed_aug/<tag>/mels.npy + index.json). Each row maps to a
    # fixed tensor; the dataloader indexes it directly (no WAV decode, no
    # on-the-fly augmentation). This replaced run_010's ~42 min/epoch on-the-fly
    # `full` augmentation with a ~5 min/epoch read.
    precomp_store = None
    precomp_index = None
    if precomp_tag:
        from pathlib import Path as _PP
        _store_dir = _PP("data/vcm/precomputed_aug") / precomp_tag
        if (_store_dir / "mels.npy").exists():
            precomp_store = np.load(_store_dir / "mels.npy", mmap_mode="r")
            _idx = json.loads((_store_dir / "index.json").read_text())
            # Build a positional list aligned to the train manifest rows.
            # index.json is in manifest order (one entry per emitted row); the
            # train manifest has the same rows, so row i -> index.json[i].
            precomp_index = _idx
            print(f"[{tag}] precomputed store '{precomp_tag}': "
                  f"{precomp_store.shape} ({precomp_store.nbytes/1e6:.0f} MB), "
                  f"{len(precomp_index)} rows", flush=True)
        else:
            print(f"[{tag}] WARNING: use_precomputed_aug='{precomp_tag}' but "
                  f"{_store_dir}/mels.npy not found; falling back to on-the-fly",
                  flush=True)

    batch_size = int(tr.get("batch_size", 64))
    num_workers = int(tr.get("num_workers", 0))
    train_dl = DataLoader(VCMDataset("train", augment=True), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=(num_workers > 0))
    val_dl = DataLoader(VCMDataset("val", augment=False), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=(num_workers > 0))

    # --- model -----------------------------------------------------------------
    from .models import ArchSpec, build_model
    arch_name = str(tr.get("arch", "a0")).lower()
    if arch_name == "a1":
        spec = ArchSpec.a1(n_way=tax.n_way_head)
    elif arch_name == "a2":
        spec = ArchSpec.a2(n_way=tax.n_way_head)
    elif arch_name == "a2w":
        # Wider TC-ResNet (width 64 vs 32) — more temporal capacity
        spec = ArchSpec("tc_resnet", [64, 64, 64, 64], width=64, n_way_head=tax.n_way_head)
    elif arch_name == "a3":
        spec = ArchSpec.a3(n_way=tax.n_way_head)
    elif arch_name == "a4":
        spec = ArchSpec.a4(n_way=tax.n_way_head)
    elif arch_name == "a5":
        spec = ArchSpec.a5(n_way=tax.n_way_head)
    elif arch_name == "a6":
        spec = ArchSpec.a6(n_way=tax.n_way_head)
    elif arch_name == "a7":
        spec = ArchSpec.a7(n_way=tax.n_way_head)
    elif arch_name == "a8":
        spec = ArchSpec.a8(n_way=tax.n_way_head)
    elif arch_name == "a9":
        spec = ArchSpec.a9(n_way=tax.n_way_head)
    elif arch_name == "a10":
        spec = ArchSpec.a10(n_way=tax.n_way_head)
    elif arch_name == "a11":
        spec = ArchSpec.a11(n_way=tax.n_way_head)
    elif arch_name == "a12":
        spec = ArchSpec.a12(n_way=tax.n_way_head)
    else:
        spec = ArchSpec.a0(n_way=tax.n_way_head)
    enable_slot_heads = bool(tr.get("enable_slot_heads", False))
    slot_heads = (
        {intent: tax.slot_classes(intent)
         for intent in tax.slotted_intents if tax.slot_classes(intent)}
        if enable_slot_heads else None
    )
    model = build_model(spec, slot_heads=slot_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] arch={spec.name} params={n_params} device={device}")

    init_from = str(tr.get("init_from", "") or "").strip()
    freeze_backbone = bool(tr.get("freeze_backbone", False))
    freeze_intent_head = bool(tr.get("freeze_intent_head", False))
    if init_from:
        ckpt_path = Path(init_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"training.init_from not found: {ckpt_path}")
        init_payload = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(init_payload["state_dict"], strict=False)
        print(f"[{tag}] initialized from {ckpt_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    if freeze_intent_head:
        for p in model.intent_head.parameters():
            p.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] trainable_params={trainable_params} "
          f"freeze_backbone={freeze_backbone} freeze_intent_head={freeze_intent_head}",
          flush=True)

    # --- performance: cudnn auto-tune + AMP bf16 ------------------------------
    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    print(f"[{tag}] amp={'bf16' if use_amp else 'off'} cudnn.benchmark=True", flush=True)

    # --- optimizer + scheduler + loss ------------------------------------------
    lr = float(tr.get("lr", 1e-3))
    opt_name = str(tr.get("optimizer", "adam")).lower()
    wd = float(tr.get("weight_decay", 0.0))
    if opt_name == "adamw":
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    epochs = int(epochs_override or tr.get("epochs", 30))
    patience = int(tr.get("early_stop_patience", 6))
    warmup_eps = int(tr.get("warmup_epochs", 0))
    
    # Warmup + cosine schedule
    if warmup_eps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        def warmup_cosine(epoch):
            if epoch < warmup_eps:
                return (epoch + 1) / warmup_eps
            progress = (epoch - warmup_eps) / max(epochs - warmup_eps, 1)
            return 0.5 * (1.0 + __import__('math').cos(3.14159265 * progress))
        sched = LambdaLR(opt, lr_lambda=warmup_cosine)
    else:
        sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))

    # class-weighted CE (rare intents + OOS get more weight)
    # Class weights from the manifest rows (vectorized; no full-dataset decode).
    # The prior line iterated train_dl (decoding every clip in the MAIN process
    # before the epoch loop) and starved the GPU - the cause of the E1 deaths.
    _train_ds = VCMDataset("train", augment=False)
    train_labels = [tax.oos_idx if r.intent == tax.oos_class else tax.intent_to_idx[r.intent]
                    for r in _train_ds.rows]
    cw = _class_weights(train_labels, tax.n_way_head, device)
    ls = float(tr.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)
    slot_loss_fn = nn.CrossEntropyLoss()
    slot_loss_weight = float(tr.get("slot_loss_weight", 0.5))
    intent_loss_weight = float(tr.get("intent_loss_weight", 1.0))

    # --- loop with validation, early-stop, best.pt checkpoint ------------------
    best_score = -1.0
    bad = 0
    history = []
    t0 = time.time()
    log_fh = (out_dir / "train.log").open("w", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[{tag}] training start: {epochs} epochs, batch={batch_size}, "
        f"workers={num_workers}, lr={lr}, opt={opt_name}(wd={wd}), "
        f"warmup={warmup_eps}ep, ls={ls}, arch={arch_name}, "
        f"slots={'yes' if enable_slot_heads else 'no'}, "
        f"intent_w={intent_loss_weight}, slot_w={slot_loss_weight}, "
        f"init_from={init_from or 'none'}, "
        f"freeze_backbone={freeze_backbone}, freeze_intent={freeze_intent_head}, "
        f"amp={'bf16' if use_amp else 'off'}, aug={aug_mode}, "
        f"mfcc_cache={'yes' if mfcc_cache is not None else 'no'}")
    log(f"[{tag}] train clips={len(train_dl.dataset)}, val clips={len(val_dl.dataset)}, "
        f"batches/epoch={len(train_dl)}")

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        n_batches = len(train_dl)
        for bi, (xb, yb, ys) in enumerate(train_dl):
            batch_t0 = time.time()
            xb, yb, ys = xb.to(device), yb.to(device), ys.to(device)
            if time.time() - batch_t0 > 120:
                raise RuntimeError(f"[{tag}] batch {bi+1}/{n_batches} took >120s - DataLoader worker deadlock suspected")
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(xb)
                # out is either a dict (multi-head) or a raw tensor (single-head)
                intent_logits = out["intent"] if isinstance(out, dict) else out
                loss = loss_fn(intent_logits, yb) * intent_loss_weight
                if enable_slot_heads and isinstance(out, dict):
                    slot_loss, _, _ = _slot_loss_and_stats(out, yb, ys, tax, slot_loss_fn)
                    if slot_loss is not None:
                        loss = loss + slot_loss_weight * slot_loss
                if not loss.requires_grad:
                    continue
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
            if (bi + 1) % 50 == 0:
                log(f"[{tag}] epoch {ep:02d} batch {bi+1}/{n_batches} loss={tot/max(n,1):.4f} ({int(time.time()-t0)}s)")
        sched.step()
        train_loss = tot / max(n, 1)

        val = _evaluate(model, val_dl, device, tax, slot_loss_fn, slot_loss_weight)
        score = val["e2e_acc"] if enable_slot_heads else val["intent_acc"]
        history.append({"epoch": ep, "train_loss": round(train_loss, 4),
                        "val_loss": round(val["loss"], 4),
                        "val_intent_acc": round(val["intent_acc"], 4),
                        "val_slot_acc": (round(val["slot_acc"], 4) if val["slot_acc"] is not None else None),
                        "val_slot_n": val["slot_n"],
                        "val_e2e_acc": round(val["e2e_acc"], 4)})
        log(f"[{tag}] epoch {ep:02d} train_loss={train_loss:.4f} "
            f"val_acc={val['intent_acc']:.4f} "
            f"val_slot={val['slot_acc'] if val['slot_acc'] is None else round(val['slot_acc'], 4)} "
            f"val_e2e={val['e2e_acc']:.4f} "
            f"({int(time.time() - t0)}s)")

        if score > best_score:
            best_score = score
            bad = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": ep, "val_acc": val["intent_acc"],
                "val_slot_acc": val["slot_acc"], "seed": seed,
                "arch": spec.name, "n_way": tax.n_way_head,
                "slot_heads": ({k: len(v) for k, v in (slot_heads or {}).items()}
                               if enable_slot_heads else {}),
                "slot_loss_weight": slot_loss_weight if enable_slot_heads else 0.0,
                "intent_loss_weight": intent_loss_weight,
                "init_from": init_from or None,
                "freeze_backbone": freeze_backbone,
                "freeze_intent_head": freeze_intent_head,
                "n_params": n_params,
                "protocol_sha256": sha256(cfg["data"]["protocol"]) if protocol_info else None,
                "config_sha256": sha256(config),
            }, out_dir / "best.pt")
        else:
            bad += 1
        if bad >= patience:
            log(f"[{tag}] early stop at epoch {ep} (patience {patience})")
            break

    # --- final record ------------------------------------------------------------
    result = {
        "tag": tag, "seed": seed, "arch": spec.name, "n_params": n_params,
        "epochs_run": len(history), "best_val_score": round(best_score, 4),
        "best_epoch": max(history, key=lambda h: (h["val_e2e_acc"] if enable_slot_heads else h["val_intent_acc"]))["epoch"],
        "slot_heads_enabled": enable_slot_heads,
        "trainable_params": trainable_params,
        "init_from": init_from or None,
        "freeze_backbone": freeze_backbone,
        "freeze_intent_head": freeze_intent_head,
        "intent_loss_weight": intent_loss_weight,
        "slot_loss_weight": slot_loss_weight if enable_slot_heads else 0.0,
        "duration_s": int(time.time() - t0),
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    log(f"[{tag}] DONE best_epoch={result['best_epoch']} "
        f"val_acc={result['history'][result['best_epoch']]['val_intent_acc']} "
        f"in {result['duration_s']}s")
    log_fh.close()
    return out_dir
