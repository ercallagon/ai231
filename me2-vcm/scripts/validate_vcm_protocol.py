#!/usr/bin/env python3
"""Validate effective corrected inputs before GPU work; never evaluate a model."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pyarrow.parquet as pq

from hey_swift.vcm.dataset import configured_rows, manifest_path
from hey_swift.vcm.protocol import load_feature_cache, sha256, verify_protocol
from hey_swift.vcm.taxonomy import default_taxonomy, load_taxonomy
from hey_swift.vcm.train import _slot_label


def validate(config):
    cfg = yaml.safe_load(Path(config).read_text())
    info = verify_protocol(cfg)
    if info is None:
        raise ValueError("An explicit corrected protocol is required")
    protocol_path = Path(cfg["data"]["protocol"])
    if sha256(protocol_path.parent / "audit.json") != info["audit_sha256"]:
        raise ValueError("Audit evidence changed")
    for name, digest in info["reserved_sha256"].items():
        if sha256(protocol_path.parent / "reserved" / f"{name}.csv") != digest:
            raise ValueError(f"Reserved subset changed: {name}")
    tax = load_taxonomy(config)
    reference_tax = default_taxonomy()
    if list(tax.intents) != list(reference_tax.intents) or tax.n_way_head != 20 or tax.oos_class != reference_tax.oos_class:
        raise ValueError("Corrected trial label ordering must match the locked 20-way schema")
    for intent in tax.slotted_intents:
        if not all(isinstance(value, str) for value in tax.slot_classes(intent)):
            raise ValueError(f"Slot classes must be strings: {intent}")
    frames = {s: pd.read_csv(manifest_path(cfg, s), keep_default_na=False) for s in ("train", "val", "test", "holdout", "tune_oos")}
    variations = pd.read_csv("data/vcm/raw/variations.csv", keep_default_na=False)
    variation_map = {r.phrase: (r.label, str(r.value).lower()) for r in variations.itertuples()}
    for split, frame in frames.items():
        for row in frame.itertuples():
            if row.variation and variation_map.get(row.variation) != (row.intent, str(row.slot).lower()):
                raise ValueError(f"Variation/label mismatch: {split}/{row.audio_path}")
    # Resolve every locator back to authoritative metadata, not flattened names.
    union = pd.concat(list(frames.values()), ignore_index=True)
    for parquet, group in union.groupby("parquet"):
        pf = pq.ParquetFile(parquet)
        columns = [c for c in ("file", "command", "slot_value", "speaker_id", "voice_split", "source_files") if c in pf.schema_arrow.names]
        for rg, sub in group.groupby("row_group"):
            records = pf.read_row_group(int(rg), columns=columns).to_pylist()
            for row in sub.itertuples():
                raw = records[int(row.row_offset)]
                if (raw["file"], raw["command"], raw.get("slot_value") or "", str(raw["speaker_id"])) != (row.file, row.intent, row.slot, str(row.speaker)):
                    raise ValueError(f"Manifest metadata disagrees with source: {row.audio_path}")
                voice_split = "" if pd.isna(row.voice_split) else row.voice_split
                source_files = "" if pd.isna(row.source_files) else row.source_files
                if (raw.get("voice_split") or "") != voice_split or (raw.get("source_files") or "") != source_files:
                    raise ValueError(f"Manifest routing/lineage disagrees with source: {row.audio_path}")
    for a, b in itertools.combinations(frames, 2):
        for column in ("audio_path", "audio_sha256", "pcm_sha256"):
            if set(frames[a][column]) & set(frames[b][column]):
                raise ValueError(f"{column} overlap: {a}/{b}")
        voices_a = set(itertools.chain.from_iterable(json.loads(x) for x in frames[a].speech_voices))
        voices_b = set(itertools.chain.from_iterable(json.loads(x) for x in frames[b].speech_voices))
        if voices_a & voices_b:
            raise ValueError(f"Speech identity overlap: {a}/{b}")
    cache, index = load_feature_cache(cfg)
    for split in frames:
        rows = configured_rows(cfg, split)
        for row in rows:
            if row.audio_path not in index or not row.parquet or not row.audio_sha256:
                raise ValueError(f"Missing source/cache contract: {row.audio_path}")
            if row.intent in tax.slotted_intents and row.slot and _slot_label(row.intent, row.slot, tax) is None:
                raise ValueError(f"Unsupported slot: {row.intent}/{row.slot}")
    for i in range(0, len(cache), 256):
        if not np.isfinite(cache[i:i + 256]).all():
            raise ValueError("Nonfinite cached features")
    train = frames["train"]
    if ((train.hf_config == "supplemental_synth") & (train.voice_split != "train")).any():
        raise ValueError("Wrong supplemental training routing")
    if ((train.hf_config == "synthetic_negatives") & (train.hf_split != "train")).any():
        raise ValueError("Wrong negative subset in training")
    test = frames["test"]
    holdout = frames["holdout"]
    if len(test) != 4693 or len(holdout) != 202:
        raise ValueError("Official evaluation membership changed")
    for split, frame in (("test", test[test.hf_config == "default"]), ("holdout", holdout)):
        official_files = set()
        for path in Path("data/vcm/raw/data").glob(split + "-*.parquet"):
            official_files.update(pq.read_table(path, columns=["file"])["file"].to_pylist())
        if set(frame.file) != official_files:
            raise ValueError(f"Official HF {split} membership mismatch")
    print(json.dumps({"passed": True, "protocol": info["name"], "protocol_sha256": sha256(protocol_path), "config_sha256": sha256(config), "counts": info["counts"], "cache_shape": list(cache.shape)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    validate(parser.parse_args().config)
