#!/usr/bin/env python3
"""Check exact model-input overlap after center cropping/padding, not just WAVs."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    directory = Path("data/vcm/protocols/hf_aligned_v1")
    cache_directory = Path("data/vcm/processed/hf_aligned_v1")
    cache = np.load(cache_directory / "mels.npy", mmap_mode="r")
    index = json.loads((cache_directory / "index.json").read_text())
    fingerprints = {key: hashlib.sha256(np.ascontiguousarray(cache[offset]).tobytes()).hexdigest() for key, offset in index.items()}
    groups = {}
    for split in ("train", "val", "test", "holdout", "tune_oos"):
        frame = pd.read_csv(directory / "manifests" / f"{split}_19intent.csv", usecols=["audio_path"])
        groups[split] = {fingerprints[key] for key in frame.audio_path}
    overlaps = {a + "|" + b: sorted(groups[a] & groups[b]) for a, b in itertools.combinations(groups, 2)}
    report = {"cache_tensors": len(cache), "exact_model_input_overlaps": overlaps, "passed": not any(overlaps.values())}
    path = Path("work/vcm_hf_alignment/hf_aligned_v1/feature_overlap.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError("Cross-split model-input duplicates require a new corrected protocol")


if __name__ == "__main__":
    main()
