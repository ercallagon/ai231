#!/usr/bin/env python3
"""Sample the ~5k reproducible PILOT train manifest for run_011 (Oct 5, 2026).

PURPOSE
-------
Cheaply test whether targeted numeral/OOS injection improves OOS/rejection
performance before investing in 10k-15k or the full ~34k train set. This
manifest is the CONTROLLED BASELINE reference for any later scaling: same
sampling strategy, same seed, same per-row augmentation recipe.

RULES
-----
* Source of truth: data/vcm/manifests/oos_inject_4000/train_19intent.csv
  (34,487 rows = 29,363 standard + 5,124 OOS, of which 4,000 are injected
  numeral-OOS: SCv2 2,070 + MLEnd 1,930).
* val/test/holdout are NEVER touched (byte-identical to the frozen splits).
* Output (own directory, segregated):
    data/vcm/pilot/run_011_pilot_5k/train_19intent.csv
    data/vcm/pilot/run_011_pilot_5k/sampling_meta.json
    data/vcm/pilot/run_011_pilot_5k/README.md

SAMPLING (seed=42, deterministic)
---------------------------------
* OOS: 500 rows = 400 numeral-OOS (SCv2/MLEnd, stratified 50/50 by source,
  seeded) + 100 base OOS (seeded). Mirrors the full-set OOS mix ratio.
* Standard: 4,500 rows, stratified proportional to full-set intent counts
  (largest-remainder allocation, seeded within-intent sampling).
* Total: 5,000 rows.

Usage:
    python3 scripts/sample_pilot_5k.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC_MANIFEST = REPO / "data/vcm/manifests/oos_inject_4000/train_19intent.csv"
OUT_DIR = REPO / "data/vcm/pilot/run_011_pilot_5k"
SEED = 42
N_STD = 4500
N_NUM_OOS = 400
N_BASE_OOS = 100


def src_of(path: str) -> str:
    p = str(path)
    if "SpeechCommands" in p:
        return "SCv2"
    if "MLEnd" in p:
        return "MLEnd"
    return "OTHER"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def largest_remainder(counts: dict, total: int) -> dict:
    """Allocate `total` proportionally to `counts` (largest remainder)."""
    s = sum(counts.values())
    raw = {k: total * v / s for k, v in counts.items()}
    alloc = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(alloc.values())
    fracs = sorted(raw, key=lambda k: raw[k] - int(np.floor(raw[k])), reverse=True)
    for k in fracs[:rem]:
        alloc[k] += 1
    return alloc


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Historical pilot reproduction only")
    parser.add_argument("--legacy-output", type=Path, required=True)
    args = parser.parse_args()
    global OUT_DIR
    OUT_DIR = args.legacy_output
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite historical pilot: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)
    df = pd.read_csv(SRC_MANIFEST)
    assert len(df) == 34487, f"unexpected manifest size {len(df)}"
    assert df["audio_path"].duplicated().sum() == 0

    rng = np.random.default_rng(SEED)

    std = df[df.intent != "OUT_OF_SCOPE"].copy()
    oos = df[df.intent == "OUT_OF_SCOPE"].copy()
    oos["src"] = oos.audio_path.map(src_of)
    num_oos = oos[oos.src.isin(["SCv2", "MLEnd"])]
    base_oos = oos[~oos.src.isin(["SCv2", "MLEnd"])]
    assert len(num_oos) == 4021 and len(base_oos) == 1103, \
        f"unexpected OOS split: num={len(num_oos)} base={len(base_oos)}"

    # ---- OOS selection (seeded, stratified by source) ----------------------
    half = N_NUM_OOS // 2
    scv2 = num_oos[num_oos.src == "SCv2"].sample(n=half, random_state=SEED)
    mlen = num_oos[num_oos.src == "MLEnd"].sample(n=N_NUM_OOS - half, random_state=SEED)
    base_sel = base_oos.sample(n=N_BASE_OOS, random_state=SEED)
    oos_sel = pd.concat([scv2, mlen, base_sel])

    # ---- standard selection (stratified proportional, seeded) ---------------
    intent_counts = std.intent.value_counts().to_dict()
    alloc = largest_remainder(intent_counts, N_STD)
    parts = []
    for intent, n in alloc.items():
        sub = std[std.intent == intent]
        parts.append(sub.sample(n=n, random_state=SEED))
    std_sel = pd.concat(parts)

    # ---- assemble in source-manifest order (stable, reproducible) -----------
    sel = pd.concat([std_sel, oos_sel])
    order = {p: i for i, p in enumerate(df.audio_path)}
    sel = sel.assign(_o=sel.audio_path.map(order)).sort_values("_o").drop(columns="_o")
    sel = sel.reset_index(drop=True)

    assert len(sel) == 5000, f"expected 5000 rows, got {len(sel)}"
    assert sel.audio_path.duplicated().sum() == 0

    # ---- write pilot manifest (same columns as source) ----------------------
    cols = ["audio_path", "intent", "slot", "speaker", "transcript", "variation"]
    out_csv = OUT_DIR / "train_19intent.csv"
    sel[cols].to_csv(out_csv, index=False)

    # ---- provenance / reproducibility record --------------------------------
    meta = {
        "pilot_tag": "run_011_pilot_5k",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "source_manifest": str(SRC_MANIFEST.relative_to(REPO)),
        "source_manifest_sha256": sha256_of(SRC_MANIFEST),
        "pilot_manifest": str(out_csv.relative_to(REPO)),
        "pilot_manifest_sha256": sha256_of(out_csv),
        "rows_total": int(len(sel)),
        "rows_standard": int(len(std_sel)),
        "rows_oos": int(len(oos_sel)),
        "oos_composition": {
            "numeral_SCv2": int((oos_sel.src == "SCv2").sum()),
            "numeral_MLEnd": int((oos_sel.src == "MLEnd").sum()),
            "base_oos": int((oos_sel.src == "OTHER").sum()),
            "note": ("OTHER = CommonVoice-based OOS + 21 SCv2-named base rows; "
                     "numeral pool in full set = 4,021 (SCv2 2,091 + MLEnd 1,930)"),
        },
        "sampling_rule": {
            "standard": ("stratified proportional to full-set intent counts, "
                         "largest-remainder allocation, seeded per intent"),
            "oos": ("400 numeral-OOS stratified 50/50 SCv2/MLEnd + 100 base OOS, "
                    "seeded; mirrors full-set OOS mix"),
        },
        "intent_allocation": {k: int(v) for k, v in sorted(alloc.items())},
        "intent_counts_selected": sel.intent.value_counts().to_dict(),
        "label_mapping": {
            "note": "intent strings map 1:1 to taxonomy indices (19 intents + OUT_OF_SCOPE); see configs/vcm.yaml schema",
            "oos_class": "OUT_OF_SCOPE",
        },
        "preprocessing": {
            "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 80,
            "window_seconds": 2.0, "f_min": 20.0, "f_max": 8000.0,
            "center_crop_pad": True, "target_frames": 198,
            "feature_shape": "(1, 80, 198) float32",
        },
        "augmentation_recipe": {
            "note": ("deterministic per-row: rng = default_rng(seed*1000 + row_pos); "
                     "pipeline speed_perturb -> add_noise -> apply_gain -> spec_aug "
                     "(identical to scripts/build_precomputed_aug.py)"),
            "speed_perturb": [0.9, 1.0, 1.1],
            "noise_prob": 0.3, "snr_db": [10.0, 25.0], "gain_db": [-6.0, 6.0],
            "spec_aug": {"time_mask": 2, "time_width": 20, "freq_mask": 2, "freq_width": 24},
        },
        "splits_frozen": {
            "note": "val/test/holdout byte-identical to data/vcm/manifests/* (sha256 in splits.json)",
            "val": "data/vcm/manifests/val_19intent.csv",
            "test": "data/vcm/manifests/test_19intent.csv",
            "holdout": "data/vcm/manifests/holdout_19intent.csv",
        },
        "scaling_note": ("Larger runs (10k/15k/full-34k) MUST reuse this seed + "
                         "strategy (superset-compatible stratification) so this "
                         "pilot stays the controlled baseline."),
    }
    (OUT_DIR / "sampling_meta.json").write_text(json.dumps(meta, indent=2))

    readme = f"""# run_011 pilot (5,000-row train) — controlled baseline

Created {meta['created_utc']} · seed {SEED}

## Purpose
Cheap test of whether targeted numeral/OOS injection improves OOS/rejection
performance (vs run_006 baseline: OOS precision 0.5388, recall 0.7883,
false-accept 220/4367) before scaling to 10k-15k or full ~34k.

## Composition (5,000 rows)
- **Standard: {meta['rows_standard']}** — stratified proportional to the full-set
  intent distribution (largest-remainder), seeded per intent.
- **OOS: {meta['rows_oos']}** — 200 numeral SCv2 + 200 numeral MLEnd + 100 base OOS
  (mirrors the full-set OOS mix ratio).

Full-set reference: 34,487 rows = 29,363 standard + 5,124 OOS
(4,021 numeral-named: SCv2 2,091 / MLEnd 1,930; 1,103 base OOS).

## Reproducibility
- Source manifest sha256: `{meta['source_manifest_sha256']}`
- Pilot manifest sha256: `{meta['pilot_manifest_sha256']}`
- Sampling: `numpy.random.default_rng({SEED})` / pandas `sample(random_state={SEED})`
- Per-row augmentation: `rng = default_rng({SEED}*1000 + row_position)` —
  deterministic; the same row always gets the same augmented features.

## Frozen eval splits (UNTOUCHED)
val / test / holdout are byte-identical to `data/vcm/manifests/*`
(hashes in `data/vcm/manifests/splits.json`). This pilot changes the TRAIN
population only.

## Scaling contract
Any 10k/15k/full-34k run must reuse seed {SEED} + the same stratification
rules (superset-compatible) so this pilot remains the controlled baseline.
"""
    (OUT_DIR / "README.md").write_text(readme)

    print(f"[pilot] WROTE {out_csv} ({len(sel)} rows)")
    print(f"[pilot] std={len(std_sel)} oos={len(oos_sel)} "
          f"(SCv2={(oos_sel.src == 'SCv2').sum()} MLEnd={(oos_sel.src == 'MLEnd').sum()} "
          f"base={(oos_sel.src == 'OTHER').sum()})")
    print(f"[pilot] meta -> {OUT_DIR / 'sampling_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
