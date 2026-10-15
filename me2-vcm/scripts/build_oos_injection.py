#!/usr/bin/env python3
"""build_oos_injection.py — build a train manifest with injected numeral-OOS rows.

Motivation
----------
The 19-intent train set carries only ~270 OOS clips (0.9%). OOS *precision*
on the frozen test is weak (P~0.54-0.59) because the model sees very few
non-command utterances. The dataset ships a dedicated, deliberately-excluded
pool of 66,390 numeral clips (SpeechCommands_v2 + MLEnd), ALL of which are
natural OUT_OF_SCOPE for a smart-home command model.

This script rebuilds the standard train manifest (byte-identical logic to
scripts/02_build_manifests.py) and appends a stratified sample of numeral-OOS
rows, writing to a SEPARATE directory so the committed manifest is untouched:

    data/vcm/manifests/oos_inject_N/train_19intent.csv

The filename MUST be train_19intent.csv because train.py's
`manifests_override` loader uses Path(...).name.

Usage
-----
    python scripts/build_oos_injection.py --n-oos 4000 --out-tag oos_inject_4k

Deterministic: seeded sampling (seed=42) so the manifest is reproducible.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

# ---- mirror scripts/02_build_manifests.py constants -----------------------
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE",
    "LIST_REMINDERS", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN", "CALL",
    "MESSAGE", "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR",
    "CREATE_REMINDER",
]
OOS = "OUT_OF_SCOPE"
COL_AUDIO = "audio"
COL_INTENT = "command"
COL_SLOT = "slot_value"
COL_SPEAKER = "speaker_id"
COL_TRANSCRIPT = "transcript"
COL_VARIATION = "variation"
SEED = 42


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def raw_root() -> Path:
    return repo_root() / "data" / "vcm" / "raw"


def manifests_root() -> Path:
    return repo_root() / "data" / "vcm" / "manifests"


def load_parquet(subdir: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_root() / subdir / pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet matching {subdir}/{pattern} under {raw_root()}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def norm_speaker(df: pd.DataFrame) -> pd.Series:
    s = df[COL_SPEAKER].astype(str)
    return s.mask(s.str.match(r"^s\d+$"), s.str.replace(r"^s", "SYN:s", regex=True))


def frame_to_manifest(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "audio_path": [str(r["file"]) for _, r in df.iterrows()],
        "intent": df[COL_INTENT].astype(str),
        "slot": df[COL_SLOT].fillna("").astype(str),
        "speaker": df[COL_SPEAKER].astype(str),
        "transcript": df[COL_TRANSCRIPT].fillna("").astype(str),
        "variation": df[COL_VARIATION].fillna("").astype(str),
    })


def stratified_sample_numerals(numerals: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n numeral rows, balanced across source datasets.

    Source is inferred from the 'file' id substring (SpeechCommands_v2 vs
    MLEnd). Balance so neither source dominates the OOS pressure.
    """
    rng = pd.np.random.default_rng(seed) if hasattr(pd, "np") else __import__("numpy").random.default_rng(seed)

    def source_of(f: str) -> str:
        f = str(f)
        if "SpeechCommands" in f or "speech_command" in f.lower():
            return "SCv2"
        if "MLEnd" in f or "mlend" in f.lower():
            return "MLEnd"
        return "OTHER"

    numerals = numerals.copy()
    numerals["_src"] = numerals["file"].map(source_of)
    counts = numerals["_src"].value_counts().to_dict()
    total = sum(counts.values())
    if n >= total:
        return numerals.drop(columns=["_src"])

    # proportional allocation, floor to >=1 for any present source
    alloc = {src: max(1, int(round(n * c / total))) for src, c in counts.items()}
    # trim/adjust to hit exactly n
    picked = []
    for src, k in alloc.items():
        sub = numerals[numerals["_src"] == src]
        k = min(k, len(sub))
        picked.append(sub.sample(n=k, random_state=seed))
    sampled = pd.concat(picked, ignore_index=True)
    # fix count drift
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        leftover = numerals[~numerals["file"].isin(sampled["file"])]
        extra = leftover.sample(n=min(n - len(sampled), len(leftover)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.drop(columns=["_src"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-reproduce", action="store_true", help="Explicitly reproduce historical, non-HF-aligned data logic")
    ap.add_argument("--n-oos", type=int, default=4000, help="number of numeral-OOS rows to inject")
    ap.add_argument("--out-tag", type=str, default=None, help="output dir name (default oos_inject_<n>)")
    args = ap.parse_args()
    if not args.legacy_reproduce:
        ap.error("Legacy-only builder. Use scripts/prepare_vcm_protocol.py for corrected trials; numeral injection needs a separately audited protocol.")

    out_tag = args.out_tag or f"oos_inject_{args.n_oos}"
    out_dir = manifests_root() / out_tag
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite historical artifacts: {out_dir}")
    out_dir.mkdir(parents=True)

    # ---- load sources (identical to 02_build_manifests.py) -----------------
    train_core = load_parquet("data", "train-*.parquet")
    train_supp = load_parquet("supplemental_synth", "train-*.parquet")
    train_fil = load_parquet("supplemental_fil", "train-*.parquet")
    train_neg = load_parquet("synthetic_negatives", "train-*.parquet")
    test_core = load_parquet("data", "test-*.parquet")
    test_neg = load_parquet("synthetic_negatives", "test-*.parquet")
    numerals = load_parquet("data", "numerals-*.parquet")

    # sanity: taxonomy
    for name, df in [("train_core", train_core), ("train_supp", train_supp),
                     ("train_fil", train_fil), ("train_neg", train_neg),
                     ("numerals", numerals)]:
        bad = set(df[COL_INTENT].unique()) - set(INTENTS) - {OOS}
        if bad:
            raise ValueError(f"{name}: unexpected intents {bad}")

    # numerals should all be OOS; force-label defensively
    n_num_oos = int((numerals[COL_INTENT] == OOS).sum())
    print(f"[inject] numerals total={len(numerals)} already_OOS={n_num_oos}")

    # ---- val carve (identical rule) ----------------------------------------
    train_core["_spk"] = norm_speaker(train_core)
    groups = (train_core.groupby("_spk").size().rename("n").reset_index()
              .sort_values(["n", "_spk"], ascending=[False, True]).reset_index(drop=True))
    target_val = int(0.10 * len(train_core))
    cum = 0
    val_groups = set()
    for _, g in groups.iterrows():
        if cum >= target_val:
            break
        val_groups.add(g["_spk"])
        cum += g["n"]
    val_mask = train_core["_spk"].isin(val_groups)
    train_core_kept = train_core[~val_mask].reset_index(drop=True)

    # ---- assemble train (identical to 02_build_manifests.py) ---------------
    test_ids = set(pd.concat([test_core, test_neg], ignore_index=True)["file"])
    train_full = pd.concat([train_core_kept, train_supp, train_fil, train_neg], ignore_index=True)
    train_full = train_full[~train_full["file"].isin(test_ids)].reset_index(drop=True)

    # ---- inject numeral-OOS ------------------------------------------------
    num_sample = stratified_sample_numerals(numerals, args.n_oos, SEED)
    # force OOS label + empty slot (numerals are non-command utterances)
    num_sample = num_sample.copy()
    num_sample[COL_INTENT] = OOS
    num_sample[COL_SLOT] = ""
    # avoid colliding file ids already in train (shouldn't happen, but safe)
    train_files = set(train_full["file"])
    num_sample = num_sample[~num_sample["file"].isin(train_files)].reset_index(drop=True)

    train_with_oos = pd.concat([train_full, num_sample], ignore_index=True).reset_index(drop=True)

    # ---- write manifest (filename MUST be train_19intent.csv) --------------
    out_df = frame_to_manifest(train_with_oos)
    csv_path = out_dir / "train_19intent.csv"
    out_df.to_csv(csv_path, index=False)

    # ---- report ------------------------------------------------------------
    base_oos = int((out_df["intent"] == OOS).sum())
    injected = len(num_sample)
    meta = {
        "out_tag": out_tag,
        "csv": str(csv_path),
        "train_rows_total": int(len(out_df)),
        "train_rows_standard": int(len(train_full)),
        "oos_rows_after": int(base_oos),
        "oos_rows_injected": int(injected),
        "numeral_source_counts": num_sample["file"].map(
            lambda f: "SCv2" if "SpeechCommands" in str(f) else ("MLEnd" if "MLEnd" in str(f) else "OTHER")
        ).value_counts().to_dict(),
        "seed": SEED,
        "note": "standard train manifest + stratified numeral-OOS sample; val/test/holdout untouched",
    }
    (out_dir / "injection_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[inject] WROTE {csv_path}")
    print(f"[inject] train rows: {meta['train_rows_standard']} standard + {injected} numeral-OOS = {meta['train_rows_total']}")
    print(f"[inject] OOS rows after injection: {base_oos} (was ~270)")
    print(f"[inject] numeral source split: {meta['numeral_source_counts']}")
    print(f"[inject] config override -> data.manifests_override.train = {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
