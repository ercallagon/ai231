#!/usr/bin/env python3
"""Metadata-only grouped validation selection from authenticated v1 inputs."""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit

from hey_swift.vcm.protocol import load_feature_cache, sha256, verify_protocol

NAME = "hf_grouped_v2b"
BASE = Path("data/vcm/protocols/hf_aligned_v1")
OUT = Path("data/vcm/protocols") / NAME
STAGE = Path("work/vcm_grouped_v2")


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, fields):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def voices(row):
    return set(json.loads(row["speech_voices"]))


def profile(rows):
    return {
        "intent": Counter(r["intent"] for r in rows),
        "slot": Counter(r["intent"] + ":" + r["slot"] for r in rows if r["slot"]),
        "source": Counter(r["source"] for r in rows),
        "subset": Counter(r["hf_config"] for r in rows),
        "synthetic": Counter(r["is_synthetic"] for r in rows),
    }


def choose_validation(rows, seed=42, trials=4096):
    positive = [r for r in rows if r["hf_config"] != "synthetic_negatives" and voices(r)]
    if any(len(voices(r)) != 1 for r in positive):
        raise ValueError("Positive rows must declare exactly one speaker identity")
    groups = np.array([next(iter(voices(r))) for r in positive])
    totals = profile(positive)
    source_voices = {source: set().union(*(voices(r) for r in positive if r["source"] == source))
                     for source in totals["source"]}
    sampler = GroupShuffleSplit(n_splits=trials, test_size=0.10, random_state=seed)
    best = None
    for candidate, (_, indices) in enumerate(sampler.split(np.zeros(len(positive)), groups=groups)):
        selected = [positive[i] for i in indices]
        fraction = len(selected) / len(positive)
        counts = profile(selected)
        speaker_counts = Counter(groups[indices])
        if not 0.085 <= fraction <= 0.125 or max(speaker_counts.values()) / len(selected) > 0.35:
            continue
        if any(counts["intent"][k] < min(10, v) for k, v in totals["intent"].items()):
            continue
        if any(counts["slot"][k] < min(5, v) for k, v in totals["slot"].items()):
            continue
        if counts["synthetic"]["0"] < 30:
            continue
        if any((counts["source"][source] == 0 or counts["source"][source] == count)
               for source, count in totals["source"].items()
               if count >= 30 and len(source_voices[source]) >= 2):
            continue
        # Equal weight per profile dimension, not per row: rare sources/slots
        # must influence selection rather than disappearing behind large groups.
        distance = sum(np.mean([abs(counts[d][k] - 0.1 * v) / max(0.1 * v, 5)
                                for k, v in totals[d].items()]) for d in totals)
        score = float(distance + 5 * abs(fraction - 0.1))
        if best is None or score < best[0]:
            best = (score, candidate, set(groups[indices]), counts)
    if best is None:
        raise ValueError("No grouped split meets predeclared coverage/balance constraints")
    selected_voices = best[2]
    train, val, excluded = [], [], []
    for row in rows:
        identities = voices(row)
        if not identities:
            if row["hf_config"] != "synthetic_negatives" or row.get("neg_kind") not in ("noise_only", "near_silence"):
                excluded.append(dict(row, exclusion_reason="missing_speech_identity"))
                continue
            # Noise-only examples have no speech identity. Hash their source
            # keys, never pseudo-speaker IDs that would create giant groups.
            ticket = int(hashlib.sha256(f"{seed}:{row['audio_path']}".encode()).hexdigest()[:16], 16)
            (val if ticket / 2**64 < 0.1 else train).append(row)
        elif identities <= selected_voices:
            val.append(row)
        elif identities.isdisjoint(selected_voices):
            train.append(row)
        else:
            excluded.append(dict(row, exclusion_reason="cross_train_val_speech_composite"))
    proof = {"seed": seed, "candidate_trials": trials, "selected_candidate": best[1],
             "metadata_balance_score": best[0], "selected_voices": sorted(selected_voices),
             "selection_uses_audio_or_model_predictions": False,
             "algorithm": "seeded sklearn GroupShuffleSplit; metadata-only multi-profile selection",
             "positive_pool_profile": totals, "positive_val_profile": best[3]}
    return train, val, excluded, proof


def audit_boundaries(frames):
    overlaps = {}
    for a, b in itertools.combinations(frames, 2):
        result = {key: sorted({r[key] for r in frames[a]} & {r[key] for r in frames[b]})
                  for key in ("audio_path", "audio_sha256", "pcm_sha256")}
        result["speech_voices"] = sorted(set().union(*(voices(r) for r in frames[a])) &
                                          set().union(*(voices(r) for r in frames[b])))
        if any(result.values()):
            raise ValueError(f"Split overlap: {a}/{b}: {result}")
        ids_b = {r["audio_path"] for r in frames[b]}
        ids_a = {r["audio_path"] for r in frames[a]}
        if any(set(json.loads(r["speech_dependencies"])) & ids_b for r in frames[a]) or any(
                set(json.loads(r["speech_dependencies"])) & ids_a for r in frames[b]):
            raise ValueError(f"Speech dependency overlap: {a}/{b}")
        overlaps[a + "|" + b] = result
    return overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if OUT.exists() or Path("data/vcm/processed", NAME).exists():
        raise FileExistsError("Versioned protocol/cache already exists; validate instead of overwrite")
    cfg = yaml.safe_load(Path("configs/vcm_hf_aligned_v1_bc8.yaml").read_text())
    parent = verify_protocol(cfg)
    cache, index = load_feature_cache(cfg)
    train, val, excluded, selection = choose_validation(read_rows(BASE / "manifests/train_19intent.csv"))
    frames = {"train": train, "val": val, "challenge": read_rows(BASE / "manifests/val_19intent.csv")}
    for split in ("test", "holdout", "tune_oos"):
        frames[split] = read_rows(BASE / "manifests" / f"{split}_19intent.csv")
    overlaps = audit_boundaries(frames)
    tensor_sets = {s: set() for s in frames}
    tensor_fingerprints = {}
    for split, rows in frames.items():
        for row in rows:
            key = row["audio_path"]
            if key not in tensor_fingerprints:
                tensor_fingerprints[key] = hashlib.sha256(np.ascontiguousarray(cache[index[key]]).tobytes()).hexdigest()
            tensor_sets[split].add(tensor_fingerprints[key])
    if any(tensor_sets[a] & tensor_sets[b] for a, b in itertools.combinations(frames, 2)):
        raise ValueError("Exact model-input overlap")
    from hey_swift.vcm.taxonomy import default_taxonomy
    tax = default_taxonomy()
    from hey_swift.vcm.train import _slot_label
    for split in ("train", "val"):
        if set(r["intent"] for r in frames[split]) != set(tax.intents) | {tax.oos_class}:
            raise ValueError(f"Missing intent in {split}")
        for intent in tax.slotted_intents:
            supported = {_slot_label(intent, r["slot"] or None, tax)
                         for r in frames[split] if r["intent"] == intent}
            if not set(range(len(tax.slot_classes(intent)))) <= supported:
                raise ValueError(f"Missing supported slot in {split}/{intent}")
    STAGE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True)
    (OUT / "manifests").mkdir()
    for split, rows in frames.items():
        if split in ("train", "val"):
            write_rows(OUT / "manifests" / f"{split}_19intent.csv", rows, list(rows[0]))
        else:
            source = "val" if split == "challenge" else split
            shutil.copy2(BASE / "manifests" / f"{source}_19intent.csv", OUT / "manifests" / f"{split}_19intent.csv")
    for split in ("test_core", "test_negatives"):
        shutil.copy2(BASE / "manifests" / f"{split}_19intent.csv", OUT / "manifests" / f"{split}_19intent.csv")
    shutil.copytree(BASE / "reserved", OUT / "reserved")
    report = {"counts": {s: len(rows) for s, rows in frames.items()}, "overlaps": overlaps,
              "exact_feature_overlap_passed": True, "selection": selection,
              "distributions": {s: profile(rows) for s, rows in frames.items()},
              "excluded_total": len(excluded),
              "excluded_reasons": dict(Counter(r["exclusion_reason"] for r in excluded)),
              "excluded_by_kind": dict(Counter(r.get("neg_kind", "") for r in excluded))}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    if excluded:
        write_rows(STAGE / f"{NAME}_excluded.csv", excluded, list(excluded[0]))
    info = dict(parent)
    info.update(name=NAME, seed=42, counts=report["counts"], audit_sha256=sha256(OUT / "audit.json"),
                parent_protocol_sha256=sha256(BASE / "protocol.json"),
                split_logic="Seeded group shuffle selected by metadata-only intent/slot/source balance from v1 train; original v1 val retained as independent challenge; mixed speech-lineage composites excluded.",
                threshold="Validation and tune_oos only; official test and holdout never fit thresholds.")
    info["counts"].update(test_core=4443, test_negatives=250)
    info["manifest_sha256"] = {p.name.removesuffix("_19intent.csv"): sha256(p) for p in (OUT / "manifests").glob("*.csv")}
    info["implementation_sha256"] = dict(parent["implementation_sha256"])
    info["implementation_sha256"]["scripts/prepare_vcm_grouped_protocol.py"] = sha256(__file__)
    info["limitations"] = [x for x in parent["limitations"] if "Local validation retains" not in x]
    info["limitations"] += ["Grouped validation represents a synthetic-heavy eligible training pool; report real/synthetic/source metrics and the independent six-identity challenge separately.",
                            "Historical test already inspected; this is a known benchmark, not a pristine final generalization estimate.",
                            "Noise sources may be shared between training/validation by upstream design; speech identity and PCM/tensor overlap remain isolated."]
    (OUT / "protocol.json").write_text(json.dumps(info, indent=2))
    # Reuse byte-identical authenticated clean features. No raw audio decode or
    # duplicate 2.2-GB cache build is needed solely because membership changed.
    directory = Path("data/vcm/processed", NAME)
    directory.mkdir()
    for filename in ("mels.npy", "index.json"):
        (directory / filename).symlink_to(Path("../hf_aligned_v1") / filename)
    meta = json.loads(Path("data/vcm/processed/hf_aligned_v1/meta.json").read_text())
    meta["protocol_sha256"] = sha256(OUT / "protocol.json")
    meta["cache_reuse"] = "Authenticated unchanged features; unused excluded rows remain cached but never fitted"
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"protocol": NAME, "counts": info["counts"], "excluded_total": len(excluded),
                      "selected_voices": selection["selected_voices"], "cache_reused": True}, indent=2))


if __name__ == "__main__":
    main()
