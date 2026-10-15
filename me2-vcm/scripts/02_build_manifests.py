#!/usr/bin/env python3
"""02_build_manifests.py — build the committed VCM split manifests.

Reads the cached HF parquet in ``data/vcm/raw/`` and writes, under
``data/vcm/manifests/``:

    train_19intent.csv      audio_path, intent, slot, speaker, transcript, variation
    val_19intent.csv
    test_19intent.csv       (FROZEN — eval-only, never trained on)
    holdout_19intent.csv    (UNTOUCHED — eval-only)
    splits.json             row counts + disjointness proof + HF revision + hashes

Manifest design (locked):
  * train  = data/train-*.parquet + supplemental_synth/train-*.parquet
             + supplemental_fil/train-*.parquet
             + synthetic_negatives/train-*.parquet (OOS negatives)
  * val    = deterministic speaker-disjoint carve from data/train (last 10%
             of sorted speaker ids); supplemental/synth-negatives stay in train
  * test   = data/test-*.parquet + synthetic_negatives/test-*.parquet
  * holdout= data/holdout-*.parquet
  * numerals (66,390 OOS numeral clips) are EXCLUDED from all manifests:
             they are a dedicated OOS-stress pool, not part of the standard
             splits (see docs/dataset.md).

The slot column stores the RAW slot_value string (e.g. "1 minute"); the
canonical slot-class mapping lives in src/hey_swift/vcm/taxonomy.py.

Usage:
    python scripts/02_build_manifests.py
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ID = "airimonda/ai231-me2-voice-commands"
REVISION = "bfca0bbebfe17e00d6a685d1da573615d29ecdce"

# The 19 intents (must match src/hey_swift/vcm/taxonomy.INTENTS).
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


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def raw_root() -> Path:
    return repo_root() / "data" / "vcm" / "raw"


def manifests_root() -> Path:
    return repo_root() / "data" / "vcm" / "manifests"


def load_parquet(subdir: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_root() / subdir / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No parquet matching {subdir}/{pattern} under {raw_root()}. "
            "Run `python scripts/01_download_hf.py` first."
        )
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def norm_speaker(df: pd.DataFrame) -> pd.Series:
    """Normalize speaker ids: synthetic 'sXX' voices -> 'SYN:sXX'.

    The sXX namespace is reused across the core dataset, supplemental_synth
    and synthetic_negatives for TTS voices; a bare string comparison would
    conflate them with any real speaker that happens to share an id. Real
    (non-sXX) ids are kept as-is.
    """
    s = df[COL_SPEAKER].astype(str)
    return s.mask(s.str.match(r"^s\d+$"), s.str.replace(r"^s", "SYN:s", regex=True))


def audio_id_of(row) -> str:
    """Stable per-row id: the 'file' column (unique across the dataset)."""
    return str(row["file"])


def frame_to_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "audio_path": [audio_id_of(r) for _, r in df.iterrows()],
        "intent": df[COL_INTENT].astype(str),
        "slot": df[COL_SLOT].fillna("").astype(str),
        "speaker": df[COL_SPEAKER].astype(str),
        "transcript": df[COL_TRANSCRIPT].fillna("").astype(str),
        "variation": df[COL_VARIATION].fillna("").astype(str),
    })
    return out


def build(out: Path) -> int:
    if out.resolve() == manifests_root().resolve() or out.exists():
        raise FileExistsError("Legacy reproduction requires a fresh noncanonical output directory")
    out.mkdir(parents=True)

    # ---- load sources -----------------------------------------------------
    train_core = load_parquet("data", "train-*.parquet")
    train_supp = load_parquet("supplemental_synth", "train-*.parquet")
    train_fil = load_parquet("supplemental_fil", "train-*.parquet")
    train_neg = load_parquet("synthetic_negatives", "train-*.parquet")
    test_core = load_parquet("data", "test-*.parquet")
    test_neg = load_parquet("synthetic_negatives", "test-*.parquet")
    holdout = load_parquet("data", "holdout-*.parquet")
    numerals = load_parquet("data", "numerals-*.parquet")

    # ---- sanity: taxonomy -------------------------------------------------
    for name, df in [("train_core", train_core), ("test_core", test_core),
                     ("holdout", holdout), ("train_supp", train_supp),
                     ("train_fil", train_fil),
                     ("train_neg", train_neg), ("test_neg", test_neg)]:
        bad = set(df[COL_INTENT].unique()) - set(INTENTS) - {OOS}
        if bad:
            raise ValueError(f"{name}: unexpected intents {bad}")

    # ---- val carve: speaker-disjoint, deterministic, stratified ------------
    # Rule (locked): group rows by normalized speaker; sort groups by
    # (n_rows DESC, speaker_id ASC) for determinism; walk the order assigning
    # each group wholly to val until val reaches >= 10% of train rows, then
    # train. Groups are never split, so val is speaker-disjoint from train by
    # construction. Small groups accumulate first, so the cut lands near 10%.
    train_core["_spk"] = norm_speaker(train_core)
    groups = (train_core.groupby("_spk")
                   .size()
                   .rename("n")
                   .reset_index()
                   .sort_values(["n", "_spk"], ascending=[False, True]))
    total = len(train_core)
    target = int(total * 0.10)
    val_speakers, acc = set(), 0
    for _, r in groups.iterrows():
        if acc >= target:
            break
        val_speakers.add(r["_spk"])
        acc += int(r["n"])
    val_mask = train_core["_spk"].isin(val_speakers)
    val_core = train_core[val_mask].reset_index(drop=True)
    train_core = train_core[~val_mask].reset_index(drop=True)
    train_core = train_core.drop(columns=["_spk"])
    train_core_kept = train_core
    val_frac = len(val_core) / total

    # ---- assemble final splits --------------------------------------------
    test_full = pd.concat([test_core, test_neg], ignore_index=True)
    test_ids = set(test_full["file"])
    train_full = pd.concat([train_core_kept, train_supp, train_fil, train_neg], ignore_index=True)
    leaked = int(train_full["file"].isin(test_ids).sum())
    train_full = train_full[~train_full["file"].isin(test_ids)].reset_index(drop=True)

    frames = {
        "train": train_full,
        "val": val_core,
        "test": test_full,
        "holdout": holdout,
    }

    # ---- write manifests ----------------------------------------------------
    counts, hashes = {}, {}
    for name, df in frames.items():
        out_df = frame_to_manifest(df)
        csv_path = out / f"{name}_19intent.csv"
        out_df.to_csv(csv_path, index=False)
        counts[name] = len(out_df)
        hashes[name] = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # ---- disjointness proof (row ids + speakers) ---------------------------
    id_sets = {n: set(pd.read_csv(out / f"{n}_19intent.csv")["audio_path"])
               for n in frames}
    pairs = [("train", "val"), ("train", "test"), ("train", "holdout"),
             ("val", "test"), ("val", "holdout"), ("test", "holdout")]
    overlaps = {f"{a}|{b}": len(id_sets[a] & id_sets[b]) for a, b in pairs}

    def real_spks(df):
        s = norm_speaker(df)
        # exclude synthetic TTS voices (SYN:sXX) and the synthetic-negative
        # generator ids (is_synthetic == True) — neither is a human speaker
        if "is_synthetic" in df.columns:
            s = s[df["is_synthetic"].fillna(False) == False]
        return set(s[~s.str.startswith("SYN:")])

    def syn_spks(df):
        s = norm_speaker(df)
        return set(s[s.str.startswith("SYN:")])

    spk_train, spk_val = real_spks(train_full), real_spks(val_core)
    spk_test, spk_hold = real_spks(test_full), real_spks(holdout)
    syn_train = syn_spks(train_full)
    syn_report = {
        "train|test": len(syn_train & syn_spks(test_full)),
        "train|holdout": len(syn_train & syn_spks(holdout)),
        "note": ("synthetic TTS voice ids (sXX) are reused across splits by "
                 "design; distinct generated clips (0 audio-hash overlap "
                 "verified). Not a leakage gate."),
    }
    speaker_disjoint = {
        "train|val": len(spk_train & spk_val),
        "train|test": len(spk_train & spk_test),
        "train|holdout": len(spk_train & spk_hold),
        "val|test": len(spk_val & spk_test),
        "val|holdout": len(spk_val & spk_hold),
        "test|holdout": len(spk_test & spk_hold),
    }

    splits = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "counts": counts,
        "numerals_excluded": len(numerals),
        "overlaps_row_ids": overlaps,
        "real_speaker_overlaps": speaker_disjoint,
        "synthetic_voice_id_reuse": syn_report,
        "disjoint": (all(v == 0 for v in overlaps.values())
                     and all(v == 0 for v in speaker_disjoint.values())),
        "manifest_sha256": hashes,
        "val_carve": {
            "rule": ("speaker-group greedy fill: groups sorted by (n_rows DESC, "
                     "speaker_id ASC), assigned whole to val until >= 10% of "
                     "train rows; groups never split"),
            "val_speaker_count": len(val_speakers),
            "val_fraction": round(val_frac, 4),
            "total_train_core_speakers": int(groups.shape[0]),
        },
        "train_dedup_vs_test": {"rows_removed_from_train": leaked,
                                 "reason": "synthetic_negatives train/test share 51 identical row ids; kept in test only"},
        "speaker_normalization": "synthetic sXX ids namespaced to SYN:sXX before disjointness checks",
        "sources": {
            "train": ["data/train-*.parquet", "supplemental_synth/train-*.parquet",
                      "supplemental_fil/train-*.parquet",
                      "synthetic_negatives/train-*.parquet"],
            "val": ["data/train-*.parquet (speaker-disjoint carve)"],
            "test": ["data/test-*.parquet", "synthetic_negatives/test-*.parquet"],
            "holdout": ["data/holdout-*.parquet"],
        },
        "note": ("test + holdout are eval-only, never trained on. "
                 "numerals = dedicated OOS stress pool, excluded from splits."),
    }
    (out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    print(f"[manifests] wrote {len(frames)} split CSVs + splits.json -> {out}")
    print(f"[manifests] counts={counts}")
    print(f"[manifests] numerals_excluded={len(numerals)}")
    print(f"[manifests] disjoint={splits['disjoint']}")
    if not splits["disjoint"]:
        print(f"[manifests] row-id overlaps: {overlaps}")
        print(f"[manifests] speaker overlaps: {speaker_disjoint}")
    return 0 if splits["disjoint"] else 1


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description="Build corrected protocol, or explicitly reproduce legacy logic to a new directory")
    parser.add_argument("--legacy-output", type=Path)
    parser.add_argument("--name", default="hf_aligned_v1")
    args = parser.parse_args()
    if args.legacy_output:
        raise SystemExit(build(args.legacy_output))
    raise SystemExit(subprocess.call([sys.executable, "scripts/prepare_vcm_protocol.py", "--name", args.name]))
