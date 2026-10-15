"""Wake-word dataset loading (faithful exp15 layout).

The canonical snapshot under ``data/wakeword/raw/`` mirrors the historical
``data_v2_mined`` layout::

    raw/{train,validation,test}/{positive,negative}/*.wav

Positive examples are the "Hey Swift" phrase (label 0); negatives are mined /
confuser speech and non-speech (label 1). The committed manifests under
``data/wakeword/manifests/`` (train.csv / validation.csv / test.csv) are the
authoritative split membership and are byte-checked against the raw snapshot
(see ``dataset_fingerprint.json``).

``collect()`` returns ``[(abs_path, label)]`` in the exact sorted order the
historical ``train_v5.collect`` used, so split membership is reproducible.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

RAW_DIR = Path("data/wakeword/raw")
MANIFEST_DIR = Path("data/wakeword/manifests")
SPLITS = ("train", "validation", "test")
CLASS_TO_LABEL = {"positive": 0, "negative": 1}  # 0 = wake, 1 = not-wake


@dataclass(frozen=True)
class WakeRow:
    relpath: str      # e.g. "train/positive/pos_00038_....wav"
    abspath: str
    label: int        # 0 = wake ("Hey Swift"), 1 = not-wake
    split: str


def raw_root(root: str | None = None) -> Path:
    return Path(root) if root else RAW_DIR


def collect(split_dir: str | Path) -> List[Tuple[str, int]]:
    """``[(abs_pcm_path, label)]`` for one split dir, sorted, positive->0.

    Faithful to the historical ``train_v5.collect``: positives first (label 0),
    then negatives (label 1), each group sorted by filename.
    """
    split_dir = Path(split_dir)
    items: List[Tuple[str, int]] = []
    for f in sorted(glob.glob(str(split_dir / "positive" / "*.wav"))):
        items.append((f, 0))
    for f in sorted(glob.glob(str(split_dir / "negative" / "*.wav"))):
        items.append((f, 1))
    return items


def collect_split(split: str, root: str | None = None) -> List[Tuple[str, int]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return collect(raw_root(root) / split)


def iter_rows(split: str, root: str | None = None) -> Iterator[WakeRow]:
    for path, label in collect_split(split, root):
        rel = os.path.relpath(path, str(raw_root(root)))
        yield WakeRow(relpath=rel, abspath=path, label=label, split=split)


def load_manifest(split: str) -> List[Tuple[str, int]]:
    """Read the committed per-split manifest (relpath,label)."""
    csv_path = MANIFEST_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Wake-word manifest '{csv_path}' not found. Run the manifest builder "
            "(see data/README.md) to (re)generate it from data/wakeword/raw/."
        )
    rows: List[Tuple[str, int]] = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["relpath"].strip(), int(r["label"])))
    return rows


def verify_manifest(split: str, root: str | None = None) -> dict:
    """Check committed manifest membership == raw snapshot on disk."""
    man = {rel: lab for rel, lab in load_manifest(split)}
    disk = {os.path.relpath(p, str(raw_root(root))): lab
            for p, lab in collect_split(split, root)}
    missing = sorted(set(disk) - set(man))
    extra = sorted(set(man) - set(disk))
    mismatch = sorted(k for k in set(disk) & set(man) if disk[k] != man[k])
    return {
        "split": split,
        "manifest_rows": len(man),
        "disk_rows": len(disk),
        "missing_in_manifest": missing,
        "extra_in_manifest": extra,
        "label_mismatches": mismatch,
        "ok": not (missing or extra or mismatch),
    }
