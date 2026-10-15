#!/usr/bin/env python3
"""Build an immutable HF-aligned VCM protocol, then its authenticated mel cache.

Run from the repository root with PYTHONPATH=src. Audio is streamed directly
from the pinned Parquet snapshot; legacy flattened WAVs/caches are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import soundfile as sf
import yaml

from hey_swift.vcm.features import FeatureConfig, MelFeatures
from hey_swift.vcm.protocol import sha256, verify_protocol
from hey_swift.vcm.taxonomy import default_taxonomy
from hey_swift.vcm.train import _slot_label

REVISION = "bfca0bbebfe17e00d6a685d1da573615d29ecdce"
RAW = Path("data/vcm/raw")
SPEC = {
    "default": ("data", ["train", "test", "holdout", "numerals"]),
    "supplemental_synth": ("supplemental_synth", ["train"]),
    "supplemental_fil": ("supplemental_fil", ["train"]),
    "synthetic_negatives": ("synthetic_negatives", ["train", "test", "noise", "tune_oos"]),
}
IMPLEMENTATION = [
    "scripts/prepare_vcm_protocol.py", "src/hey_swift/vcm/dataset.py",
    "src/hey_swift/vcm/protocol.py", "src/hey_swift/vcm/features.py",
    "src/hey_swift/vcm/taxonomy.py", "src/hey_swift/vcm/train.py",
    "src/hey_swift/vcm/evaluate.py",
]


def git(command):
    proc = subprocess.run(["git", *command], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def decode(raw):
    wav, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if rate != 16000 or len(wav) == 0 or not np.isfinite(wav).all():
        raise ValueError(f"Invalid audio: rate={rate}, frames={len(wav)}")
    return wav.mean(axis=1), rate, wav.shape[1]


def source_key(config, split, filename):
    encoded = json.dumps([REVISION, config, split, filename], separators=(",", ":"))
    return "hf:" + hashlib.sha256(encoded.encode()).hexdigest()


def inventory(stage):
    manifest = json.loads((RAW / "download_manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected HF revision")
    raw_hashes = {}
    for item in manifest["files"]:
        path = RAW / item["path"]
        digest = sha256(path)
        if (item["lfs_oid"] and digest != item["lfs_oid"]) or path.stat().st_size != item["size"]:
            raise ValueError(f"Raw snapshot verification failed: {path}")
        raw_hashes[str(path)] = digest
    fingerprint = hashlib.sha256(json.dumps(raw_hashes, sort_keys=True).encode()).hexdigest()
    cached = stage / "inventory.json"
    if cached.exists():
        info = json.loads(cached.read_text())
        if info["raw_fingerprint"] == fingerprint:
            return info["rows"], raw_hashes
        raise ValueError("Staging inventory belongs to a different snapshot")
    rows = []
    for config, (directory, splits) in SPEC.items():
        for split in splits:
            paths = sorted((RAW / directory).glob(split + "-*.parquet"))
            if not paths:
                raise FileNotFoundError(f"Missing {config}/{split}")
            for path in paths:
                pf = pq.ParquetFile(path)
                for group in range(pf.num_row_groups):
                    for offset, row in enumerate(pf.read_row_group(group).to_pylist()):
                        raw = row["audio"]["bytes"]
                        wav, rate, channels = decode(raw)
                        record = {k: v for k, v in row.items() if k != "audio"}
                        record.update({
                            "audio_path": source_key(config, split, row["file"]),
                            "hf_config": config, "hf_split": split,
                            "parquet": str(path), "row_group": group, "row_offset": offset,
                            "audio_sha256": hashlib.sha256(raw).hexdigest(),
                            "pcm_sha256": hashlib.sha256(wav.astype("<f4").tobytes()).hexdigest(),
                            "sample_rate": rate, "channels": channels,
                            "n_samples": len(wav), "duration_s": len(wav) / rate,
                            "peak": float(np.abs(wav).max()),
                        })
                        rows.append(record)
                print(f"[inventory] {path}: total={len(rows)}", flush=True)
    cached.write_text(json.dumps({"raw_fingerprint": fingerprint, "rows": rows}))
    return rows, raw_hashes


def attach_lineage(rows):
    lookup = {(r["hf_config"], r["hf_split"], r["file"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate HF source identity")
    noise = {Path(r["file"]).name: r for r in rows
             if r["hf_config"] == "synthetic_negatives" and r["hf_split"] == "noise"}
    for row in rows:
        voices, sources, noises = set(), [], []
        negative = row["hf_config"] == "synthetic_negatives" and row["hf_split"] in ("train", "test")
        if not negative and row.get("speaker_id"):
            voices.add(str(row["speaker_id"]))
        for reference in (row.get("source_files") or "").split(";"):
            if not reference:
                continue
            split, _, filename = reference.partition("/")
            if split in ("train", "test", "holdout"):
                dep = lookup.get(("default", split, filename))
                if dep is None:
                    raise ValueError(f"Unresolved speech dependency: {reference}")
                voices.add(str(dep["speaker_id"]))
                sources.append(dep["audio_path"])
            elif split == "noise":
                dep = noise.get(Path(filename).name)
                if dep is None:
                    raise ValueError(f"Unresolved noise dependency: {reference}")
                noises.append(dep["audio_path"])
            else:
                raise ValueError(f"Unknown lineage reference: {reference}")
        row["speech_voices"] = sorted(voices)
        row["speech_dependencies"] = sorted(sources)
        row["noise_dependencies"] = sorted(noises)


def overlap(a, b, column):
    def values(group):
        return set(itertools.chain.from_iterable(r[column] for r in group)) if column == "speech_voices" else {r[column] for r in group}
    return sorted(values(a) & values(b))


def write_csv(path, rows):
    records = []
    for row in rows:
        record = dict(row)
        record.update(intent=row["command"], slot=row.get("slot_value") or "",
                      speaker=str(row.get("speaker_id") or ""))
        for field in ("speech_voices", "speech_dependencies", "noise_dependencies"):
            record[field] = json.dumps(record[field])
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)


def historical_evidence():
    import torch
    evidence = []
    for directory in sorted(Path("experiments/vcm").glob("run_*")):
        record = {"tag": directory.name, "files": {}}
        for name in ("reproducibility.json", "result.json", "metrics_test.json", "best.pt", "train.log"):
            path = directory / name
            if path.exists():
                record["files"][name] = sha256(path)
        if (directory / "best.pt").exists():
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            record["checkpoint"] = {k: checkpoint.get(k) for k in ("arch", "epoch", "seed", "n_params", "n_way", "slot_heads", "init_from", "protocol_sha256")}
            record["checkpoint"]["slot_tensor_shapes"] = {k: list(v.shape) for k, v in checkpoint.get("state_dict", {}).items() if k.startswith("slot_head_map.")}
        for name in ("reproducibility.json", "result.json", "metrics_test.json"):
            path = directory / name
            if path.exists():
                content = json.loads(path.read_text())
                record[name] = {k: content.get(k) for k in (
                    "config", "seed", "arch", "n_params", "hf_revision", "splits",
                    "best_epoch", "best_val_score", "intent_accuracy", "e2e_accuracy", "macro_f1", "slot_accuracy")}
                config = content.get("config")
                if config and Path(config).exists():
                    record["files"][config] = sha256(config)
        evidence.append(record)
    return evidence


def build(args):
    output = Path("data/vcm/protocols") / args.name
    stage = Path("work/vcm_hf_alignment") / args.name
    stage.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol: {output}")
    rows, raw_hashes = inventory(stage)
    attach_lineage(rows)
    subset = lambda config, split: [r for r in rows if r["hf_config"] == config and r["hf_split"] == split]
    legacy_val = pd.read_csv("data/vcm/manifests/val_19intent.csv", dtype=str).fillna("")
    val_files = set(legacy_val.audio_path)
    core = subset("default", "train")
    val = [r for r in core if r["file"] in val_files]
    if {r["file"] for r in val} != val_files:
        raise ValueError("Cannot exactly recover historical validation membership")
    frames = {
        "val": val,
        "test_core": subset("default", "test"),
        "test_negatives": subset("synthetic_negatives", "test"),
        "holdout": subset("default", "holdout"),
        "tune_oos": subset("synthetic_negatives", "tune_oos"),
    }
    frames["test"] = frames["test_core"] + frames["test_negatives"]
    supp = subset("supplemental_synth", "train")
    if {r.get("voice_split") for r in supp} != {"train", "test", "holdout"}:
        raise ValueError("Unexpected supplemental voice_split")
    reserved = {
        "supplemental_test": [r for r in supp if r["voice_split"] == "test"],
        "supplemental_holdout": [r for r in supp if r["voice_split"] == "holdout"],
        "numerals": subset("default", "numerals"),
        "noise": subset("synthetic_negatives", "noise"),
    }
    boundaries = val + frames["test"] + frames["holdout"] + frames["tune_oos"] + reserved["supplemental_test"] + reserved["supplemental_holdout"]
    forbidden_voices = set(itertools.chain.from_iterable(r["speech_voices"] for r in boundaries))
    forbidden_ids = {r["audio_path"] for r in boundaries}
    forbidden_pcm = {r["pcm_sha256"] for r in boundaries}
    forbidden_bytes = {r["audio_sha256"] for r in boundaries}
    pool = [r for r in core if r["file"] not in val_files]
    pool += [r for r in supp if r["voice_split"] == "train"]
    pool += subset("supplemental_fil", "train") + subset("synthetic_negatives", "train")
    kept, excluded, seen = [], [], {}
    labels_by_pcm = {}
    for row in pool:
        labels_by_pcm.setdefault(row["pcm_sha256"], set()).add((row["command"], row.get("slot_value") or ""))
    for row in pool:
        reason = None
        if forbidden_voices.intersection(row["speech_voices"]):
            reason = "reserved_speech_identity"
        elif forbidden_ids.intersection(row["speech_dependencies"]):
            reason = "reserved_speech_dependency"
        elif row["pcm_sha256"] in forbidden_pcm or row["audio_sha256"] in forbidden_bytes:
            reason = "reserved_audio_duplicate"
        elif len(labels_by_pcm[row["pcm_sha256"]]) > 1:
            reason = "training_duplicate_label_conflict"
        elif row["pcm_sha256"] in seen:
            other = seen[row["pcm_sha256"]]
            reason = "training_audio_duplicate" if (row["command"], row.get("slot_value")) == (other["command"], other.get("slot_value")) else "training_duplicate_label_conflict"
        if reason:
            excluded.append(dict(row, exclusion_reason=reason))
        else:
            kept.append(row)
            seen[row["pcm_sha256"]] = row
    frames["train"] = kept
    tax = default_taxonomy()
    invalid, proxies = [], []
    for split, group in frames.items():
        for row in group:
            intent, slot = row["command"], row.get("slot_value") or ""
            if intent not in list(tax.intents) + [tax.oos_class]:
                invalid.append((split, row["audio_path"], "intent"))
            elif intent in tax.slotted_intents and _slot_label(intent, slot, tax) is None:
                if not slot and row.get("bucket") == intent + " (other slot value)" and row["hf_config"] == "default" and row["hf_split"] == "train":
                    row["slot_supervision"] = "intent_only_other_value_proxy"
                    proxies.append({"split": split, "audio_path": row["audio_path"], "intent": intent})
                else:
                    invalid.append((split, row["audio_path"], "missing_or_unsupported_slot", slot))
            elif intent not in tax.slotted_intents and slot:
                invalid.append((split, row["audio_path"], "unexpected_slot", slot))
    pairs = list(itertools.combinations(["train", "val", "test", "holdout", "tune_oos"], 2))
    overlaps = {a + "|" + b: {column: overlap(frames[a], frames[b], column)
                             for column in ("audio_path", "audio_sha256", "pcm_sha256", "speech_voices")}
                for a, b in pairs}
    # Official eval membership is immutable; any source overlap is reported,
    # never silently removed. Such a defect blocks the corrected trial.
    failures = [key for key, value in overlaps.items() if any(value.values())]
    audit = {
        "counts": {k: len(v) for k, v in frames.items()},
        "reserved_counts": {k: len(v) for k, v in reserved.items()},
        "excluded": dict(Counter(r["exclusion_reason"] for r in excluded)),
        "excluded_by_subset": dict(Counter(r["hf_config"] + "/" + r["hf_split"] for r in excluded)),
        "overlaps": overlaps, "invalid_labels": invalid, "intent_only_slot_proxies": proxies,
        "distributions": {s: {"intent": dict(Counter(r["command"] for r in group)),
                                "slots": dict(Counter(r["command"] + ":" + (r.get("slot_value") or "") for r in group)),
                                "sources": dict(Counter(str(r.get("source")) for r in group)),
                                "speaker_count": len(set(itertools.chain.from_iterable(r["speech_voices"] for r in group)))}
                          for s, group in frames.items()},
        "inventory_counts": dict(Counter(r["hf_config"] + "/" + r["hf_split"] for r in rows)),
        "legacy_manifest_sha256": {s: sha256(f"data/vcm/manifests/{s}_19intent.csv") for s in ("train", "val", "test", "holdout")},
        "historical_runs": historical_evidence(),
    }
    global_audio = {}
    for row in rows:
        global_audio.setdefault(row["pcm_sha256"], []).append(row)
    audit["global_pcm_duplicates"] = [
        [{k: r.get(k) for k in ("audio_path", "file", "hf_config", "hf_split", "command", "slot_value", "speaker_id")} for r in group]
        for group in global_audio.values() if len(group) > 1
    ]
    (stage / "audit.json").write_text(json.dumps(audit, indent=2))
    write_csv(stage / "excluded.csv", excluded)
    if invalid or failures:
        raise ValueError(f"Protocol validation failed: overlaps={failures}, invalid_labels={len(invalid)}; see {stage}/audit.json")
    required = set(tax.intents) | {tax.oos_class}
    for split in ("train", "val", "test"):
        if set(r["command"] for r in frames[split]) != required:
            raise ValueError(f"Missing intent coverage: {split}")
    output.mkdir(parents=True)
    (output / "manifests").mkdir()
    (output / "reserved").mkdir()
    for name, group in frames.items():
        write_csv(output / "manifests" / f"{name}_19intent.csv", group)
    for name, group in reserved.items():
        write_csv(output / "reserved" / f"{name}.csv", group)
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    info = {
        "name": args.name, "revision": REVISION, "seed": 42,
        "validation_passed": True, "counts": audit["counts"],
        "manifest_sha256": {s: sha256(output / "manifests" / f"{s}_19intent.csv") for s in frames},
        "reserved_sha256": {s: sha256(output / "reserved" / f"{s}.csv") for s in reserved},
        "raw_sha256": raw_hashes, "audit_sha256": sha256(output / "audit.json"),
        "implementation_sha256": {p: sha256(p) for p in IMPLEMENTATION},
        "split_logic": "Preserve legacy 1171-row core validation; exclude its voices/lineage globally. Honor supplemental voice_split. Keep HF test/holdout immutable. Remove training PCM duplicates and reserved-audio matches.",
        "augmentation": "Training-only online SpecAugment; deterministic clean mel cache. No numeral injection or noise augmentation in this controlled baseline.",
        "threshold": "Fixed 0.50 gate; tune_oos reserved, unused for fitting. No test-driven selection/calibration.",
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "soundfile": sf.__version__},
        "git": {"head": git(["rev-parse", "HEAD"]), "root": git(["rev-parse", "--show-toplevel"]), "status": git(["status", "--short"])},
        "limitations": ["Local validation retains six concentrated speaker groups; not an HF validation split.", "Clone identity equivalence for supplemental Filipino voices is not established by source metadata.", "Exact byte/decoded PCM and declared lineage checks do not prove absence of undeclared transformed/near-duplicate audio.", "HF training/test negatives can share environmental noise sources by design; speech lineage remains isolated.", "Historical test and holdout voices were exposed to some legacy models; only freshly trained checkpoints qualify for the corrected protocol.", "158 HF other-slot-value proxies have intent labels but no slot target. Preserve labels, use intent-only supervision, and report separately; 3-way heads cannot establish reliable rejection of unsupported slot values."],
    }
    (output / "protocol.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"protocol": str(output), "counts": info["counts"], "excluded": audit["excluded"]}, indent=2), flush=True)


def build_cache(args):
    output = Path("data/vcm/protocols") / args.name
    cfg = {"data": {"protocol": str(output / "protocol.json"), "manifests": str(output / "manifests")}}
    verify_protocol(cfg)
    directory = Path("data/vcm/processed") / args.name
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {directory}")
    directory.mkdir(parents=True)
    frames = [pd.read_csv(output / "manifests" / f"{s}_19intent.csv", keep_default_na=False)
              for s in ("train", "val", "test", "holdout", "tune_oos")]
    frame = pd.concat(frames, ignore_index=True)
    feat_cfg = FeatureConfig()
    feats = MelFeatures(feat_cfg)
    shape = (len(frame), 1, feat_cfg.n_mels, feats._expected_frames())
    store = np.lib.format.open_memmap(directory / "mels.npy", mode="w+", dtype="float32", shape=shape)
    index = {row.audio_path: i for i, row in enumerate(frame.itertuples())}
    if len(index) != len(frame):
        raise ValueError("Cache contains duplicate source identities")
    wanted = {row.audio_path: row for row in frame.itertuples()}
    def transform(item):
        key, raw = item
        if hashlib.sha256(raw).hexdigest() != wanted[key].audio_sha256:
            raise ValueError(f"Cache audio source mismatch: {key}")
        wav, _, _ = decode(raw)
        mel = feats.transform(wav)
        if mel.shape != shape[1:] or not np.isfinite(mel).all():
            raise ValueError(f"Invalid feature tensor: {key}")
        return key, mel
    done = set()
    parity = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for parquet, group in frame.groupby("parquet", sort=True):
            pf = pq.ParquetFile(parquet)
            for rg, sub in group.groupby("row_group", sort=True):
                table = pf.read_row_group(int(rg), columns=["audio"])["audio"]
                batch = [(r.audio_path, table[int(r.row_offset)].as_py()["bytes"]) for r in sub.itertuples()]
                for key, mel in pool.map(transform, batch):
                    store[index[key]] = mel
                    done.add(key)
                # Direct-vs-cache parity includes every used row group/source.
                key, raw = batch[0]
                np.testing.assert_allclose(store[index[key]], feats.transform(decode(raw)[0]), rtol=1e-6, atol=1e-6)
                parity.append(key)
            print(f"[cache] {parquet}: {len(done)}/{len(frame)}", flush=True)
    if done != set(index):
        raise ValueError("Incomplete feature cache")
    store.flush()
    (directory / "index.json").write_text(json.dumps(index, sort_keys=True))
    meta = {"protocol_sha256": sha256(output / "protocol.json"), "features": asdict(feat_cfg),
            "shape": list(shape), "dtype": "float32", "all_finite": True,
            "parity_checks": len(parity), "parity_source_keys": parity,
            "sha256": {name: sha256(directory / name) for name in ("mels.npy", "index.json")}}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cache] validated {shape}, parity={len(parity)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="hf_aligned_v1")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Protocol name must be alphanumeric/underscore")
    build_cache(args) if args.cache else build(args)
