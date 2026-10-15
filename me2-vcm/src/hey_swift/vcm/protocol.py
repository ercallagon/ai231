"""Fail-closed checks for versioned source-aware VCM protocols."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .dataset import manifest_path
from .features import FeatureConfig


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_protocol(cfg):
    path = cfg.get("data", {}).get("protocol")
    if not path:
        return None
    info = json.loads(Path(path).read_text())
    if not info.get("validation_passed"):
        raise ValueError("Dataset protocol has not passed validation")
    for split, digest in info["manifest_sha256"].items():
        if sha256(manifest_path(cfg, split)) != digest:
            raise ValueError(f"Effective manifest mismatch: {split}")
    for relative, digest in info["implementation_sha256"].items():
        if sha256(relative) != digest:
            raise ValueError(f"Protocol implementation changed: {relative}; rebuild validation")
    for relative, digest in info["raw_sha256"].items():
        if sha256(relative) != digest:
            raise ValueError(f"Raw snapshot mismatch: {relative}")
    return info


def load_feature_cache(cfg):
    corrected = bool(cfg.get("data", {}).get("protocol"))
    directory = Path(cfg.get("data", {}).get("feature_cache", "work/mfcc_cache"))
    if not (directory / "mels.npy").exists():
        if corrected:
            raise FileNotFoundError(f"Corrected feature cache required: {directory}")
        return None, {}
    index = json.loads((directory / "index.json").read_text())
    array = np.load(directory / "mels.npy", mmap_mode="r")
    if corrected:
        meta = json.loads((directory / "meta.json").read_text())
        protocol_hash = sha256(cfg["data"]["protocol"])
        if meta["protocol_sha256"] != protocol_hash:
            raise ValueError("Cache belongs to a different dataset protocol")
        if meta["features"] != asdict(FeatureConfig.from_dict(cfg.get("audio", {}))):
            raise ValueError("Cache preprocessing mismatch")
        for name in ("mels.npy", "index.json"):
            if sha256(directory / name) != meta["sha256"][name]:
                raise ValueError(f"Feature cache fingerprint mismatch: {name}")
        if list(array.shape) != meta["shape"] or array.dtype != np.float32:
            raise ValueError("Feature cache shape/dtype mismatch")
    return array, index
