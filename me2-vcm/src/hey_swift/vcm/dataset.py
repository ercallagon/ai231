"""VCM dataset loading from the HF parquet cache + committed manifests.

The heavy parquet files live in `data/vcm/raw/` (gitignored, re-downloadable
via scripts/01_download_hf.py). The committed reproducibility contract is the
set of CSV manifests + `splits.json` under `data/vcm/manifests/`.

This module is import-safe (no torch). It yields (audio_path, intent, slot)
rows; the training loop (train.py) turns them into tensors.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

MANIFESTS_DIR = Path("data/vcm/manifests")
RAW_ROOT = Path("data/vcm/raw")
PROCESSED_WAV = Path("data/vcm/processed/wav")


def _repo_root() -> Path:
    """Repo root = parents of src/hey_swift/vcm/."""
    return Path(__file__).resolve().parents[3]


def resolve_raw_root(base: Optional[str] = None) -> Path:
    """Locate data/vcm/raw, preferring CWD then the repo root."""
    cands = []
    if base:
        cands.append(Path(base))
    cands += [Path("data/vcm/raw"), _repo_root() / "data" / "vcm" / "raw"]
    for c in cands:
        if (c / "data").is_dir():
            return c
    raise FileNotFoundError(
        "data/vcm/raw not found (looked for data/ subdirs under: "
        + ", ".join(str(c) for c in cands)
        + "). Run `python scripts/01_download_hf.py` first."
    )


def resolve_processed_wav(base: Optional[str] = None) -> Optional[Path]:
    """Locate the extracted-WAV dir (data/vcm/processed/wav) if it exists.

    Returns None when the derivative has not been built yet, so callers can
    fall back to reading audio out of the raw parquet cache.
    """
    cands = []
    if base:
        cands.append(Path(base))
    cands += [Path("data/vcm/processed/wav"), _repo_root() / "data" / "vcm" / "processed" / "wav"]
    for c in cands:
        if c.is_dir():
            return c
    return None


def decode_audio(audio) -> bytes:
    """HF datasets 'audio' column -> raw WAV bytes.

    Accepts the dict form ({"bytes": ..., "path": ...}), a bytes payload, or a
    plain path string (returned as-is for on-disk files).
    """
    if isinstance(audio, dict):
        b = audio.get("bytes")
        if b is not None:
            return bytes(b)
        raise ValueError("audio dict has no 'bytes' payload")
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    return str(audio)


def _locate_parquet(raw_root: Path, file_id: str) -> Path:
    """Find the parquet containing a row id.

    Uses a prebuilt id->parquet map (work/tmp/file_to_parquet.json) when
    present; otherwise falls back to a full scan (cached per process).
    """
    import glob
    import json
    map_path = raw_root.parent.parent / "work" / "tmp" / "file_to_parquet.json"
    if not map_path.exists():
        map_path = Path("work/tmp/file_to_parquet.json")
    if map_path.exists():
        m = json.loads(map_path.read_text())
        hit = m.get(str(file_id))
        if hit:
            return Path(hit)
        raise FileNotFoundError(f"row id {file_id!r} not in id map")
    cache = getattr(_locate_parquet, "_cache", None)
    if cache is None:
        cache = {}
        _locate_parquet._cache = cache
    if file_id in cache:
        return cache[file_id]
    wanted = str(file_id)
    import pyarrow.parquet as pqmod
    for pq in sorted(glob.glob(str(raw_root / "**" / "*.parquet"), recursive=True)):
        pf = pqmod.ParquetFile(pq)
        for rg in range(pf.metadata.num_row_groups):
            names = pf.read_row_group(rg, columns=["file"]).to_pydict()["file"]
            if wanted in names:
                cache[file_id] = Path(pq)
                return Path(pq)
    raise FileNotFoundError(f"row id {wanted!r} not found under {raw_root}")


@dataclass(frozen=True)
class Row:
    audio_path: str
    intent: str
    slot: Optional[str]  # None for non-slotted intents
    split: str
    speaker: Optional[str] = None
    transcript: Optional[str] = None
    parquet: Optional[str] = None
    row_group: Optional[int] = None
    row_offset: Optional[int] = None
    audio_sha256: Optional[str] = None


def load_split_csv(name: str, manifests_dir: Path = MANIFESTS_DIR) -> pd.DataFrame:
    path = Path(manifests_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest '{path}' not found. Run `python scripts/02_build_manifests.py` "
            "after `python scripts/01_download_hf.py`."
        )
    return pd.read_csv(path)


def iter_rows(split: str, manifests_dir: Path = MANIFESTS_DIR,
              filename: Optional[str] = None) -> Iterator[Row]:
    """Yield Row objects for a named split (train/val/test/holdout)."""
    df = load_split_csv(filename or f"{split}_19intent.csv", manifests_dir)
    for _, r in df.iterrows():
        yield Row(
            audio_path=str(r["audio_path"]),
            intent=str(r["intent"]),
            slot=(str(r["slot"]) if "slot" in r and pd.notna(r["slot"]) and str(r["slot"]) != "" else None),
            split=split,
            speaker=(str(r["speaker"]) if "speaker" in r and pd.notna(r["speaker"]) else None),
            transcript=(str(r["transcript"]) if "transcript" in r and pd.notna(r["transcript"]) else None),
            parquet=str(r["parquet"]) if pd.notna(r.get("parquet")) else None,
            row_group=int(r["row_group"]) if pd.notna(r.get("row_group")) else None,
            row_offset=int(r["row_offset"]) if pd.notna(r.get("row_offset")) else None,
            audio_sha256=str(r["audio_sha256"]) if pd.notna(r.get("audio_sha256")) else None,
        )


def manifest_path(cfg: dict, split: str) -> Path:
    data = cfg.get("data", {})
    override = (data.get("manifests_override") or {}).get(split)
    if override:
        return Path(override)
    return Path(data.get("manifests", MANIFESTS_DIR)) / f"{split}_19intent.csv"


def configured_rows(cfg: dict, split: str) -> list[Row]:
    path = manifest_path(cfg, split)
    return list(iter_rows(split, path.parent, path.name))


def load_row_wav(row: Row) -> "np.ndarray":
    if row.parquet is None:
        if row.audio_path.startswith("hf:"):
            raise ValueError("Source-qualified row is missing its raw locator")
        return load_wav(row.split, row.audio_path)
    import hashlib
    import io
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    table = pq.ParquetFile(row.parquet).read_row_group(row.row_group, columns=["audio"])
    raw = decode_audio(table["audio"][row.row_offset].as_py())
    if hashlib.sha256(raw).hexdigest() != row.audio_sha256:
        raise ValueError(f"Audio fingerprint mismatch: {row.audio_path}")
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if sr != 16000 or not len(wav) or not np.isfinite(wav).all():
        raise ValueError(f"Invalid audio format: {row.audio_path}, sr={sr}")
    return wav.mean(axis=1)


def load_splits_json(manifests_dir: Path = MANIFESTS_DIR) -> dict:
    """Load splits.json (row counts, disjointness proof, HF revision)."""
    path = Path(manifests_dir) / "splits.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_disjointness(manifests_dir: Path = MANIFESTS_DIR) -> dict:
    """Prove train/val/test/holdout are audio-disjoint. Records the proof."""
    splits = {}
    for name in ("train", "val", "test", "holdout"):
        try:
            df = load_split_csv(f"{name}_19intent.csv", manifests_dir)
            splits[name] = set(df["audio_path"].astype(str))
        except FileNotFoundError:
            splits[name] = set()

    pairs = [("train", "val"), ("train", "test"), ("train", "holdout"),
             ("val", "test"), ("val", "holdout"), ("test", "holdout")]
    overlaps = {f"{a}|{b}": len(splits[a] & splits[b]) for a, b in pairs}
    return {"counts": {k: len(v) for k, v in splits.items()}, "overlaps": overlaps,
            "disjoint": all(v == 0 for v in overlaps.values())}


# --- Audio loading from the raw parquet cache ---------------------------------

_RG_INDEX: Optional[dict] = None

def _build_rg_index(raw_root: Path) -> dict:
    """Build file_id -> (parquet_path, row_group) index.

    Primary strategy: use the pre-built work/tmp/file_to_parquet.json map
    (file_id -> parquet path) to avoid scanning all 1038 row groups upfront.
    For each unique parquet file in the map, scan its row groups' 'file'
    column (one file at a time, ~1-2 s each). Total: ~30-60 s for 16 files.

    Fallback: if the JSON map is absent, fall back to a full recursive scan
    (slow, ~minutes, but correct).
    """
    import pyarrow.parquet as pqmod
    idx: dict = {}

    # Try the pre-built map first
    map_path = raw_root.parent.parent / "work" / "tmp" / "file_to_parquet.json"
    if not map_path.exists():
        map_path = Path("work/tmp/file_to_parquet.json")

    if map_path.exists():
        fmap = json.loads(map_path.read_text())
        # Group file_ids by parquet path.
        # The JSON stores paths relative to repo root (e.g. "data/vcm/raw/data/x.parquet").
        # raw_root is "data/vcm/raw" (relative) or absolute; repo root = raw_root.parent.parent.
        repo_root = _repo_root()
        by_pq: dict[str, list[str]] = {}
        for fid, pq_rel in fmap.items():
            pq_abs = str(repo_root / pq_rel)
            by_pq.setdefault(pq_abs, []).append(fid)
        # For each parquet file, scan its row groups
        for pq_path, fids in by_pq.items():
            pf = pqmod.ParquetFile(pq_path)
            for rg in range(pf.metadata.num_row_groups):
                names = pf.read_row_group(rg, columns=["file"]).to_pydict()["file"]
                for n in names:
                    idx.setdefault(str(n), (pq_path, rg))
    else:
        # Fallback: full recursive scan (slow)
        for pq in sorted(glob.glob(str(raw_root / "**" / "*.parquet"), recursive=True)):
            pf = pqmod.ParquetFile(pq)
            for rg in range(pf.metadata.num_row_groups):
                names = pf.read_row_group(rg, columns=["file"]).to_pydict()["file"]
                for n in names:
                    idx.setdefault(str(n), (pq, rg))
    return idx

def _rg_index(raw_root: Path) -> dict:
    global _RG_INDEX
    if _RG_INDEX is None:
        _RG_INDEX = _build_rg_index(raw_root)
    return _RG_INDEX


def load_audio_bytes(split: str, row_id: str, raw_root: Optional[str] = None) -> bytes:
    """Fetch the WAV bytes for a manifest row id.

    Fast path: if the extracted-WAV derivative (data/vcm/processed/wav) exists,
    read the clip straight from disk (one small file read). This is what the
    training loop uses once `python scripts/04_extract_wavs.py` has run, and it
    removes the per-access parquet row-group seek that starved the GPU.

    Slow path (fallback): read the audio out of the raw parquet cache, as
    before. Preserved so the loader still works from a fresh clone before the
    one-time extraction has been performed.
    """
    pw = resolve_processed_wav()
    if pw is not None:
        cand = pw / str(row_id)
        if cand.is_file():
            return cand.read_bytes()
    rr = resolve_raw_root(raw_root)
    import pyarrow.parquet as pqmod
    hit = _rg_index(rr).get(str(row_id))
    if hit is not None:
        pq_path, rg = hit
        tbl = pqmod.ParquetFile(str(pq_path)).read_row_group(rg, columns=["file", "audio"]).to_pydict()
        for i, fid in enumerate(tbl["file"]):
            if str(fid) == str(row_id):
                return decode_audio(tbl["audio"][i])
    # fallback: legacy linear scan (index miss should not happen)
    pq_path = _locate_parquet(rr, row_id)
    pf = pqmod.ParquetFile(str(pq_path))
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["file", "audio"]).to_pydict()
        for i, fid in enumerate(tbl["file"]):
            if str(fid) == str(row_id):
                return decode_audio(tbl["audio"][i])
    raise FileNotFoundError(f"row id {row_id!r} not in {pq_path}")


def load_wav(split: str, row_id: str, raw_root: Optional[str] = None) -> "np.ndarray":
    """Fetch + decode the WAV for a manifest row id -> float32 mono @16 kHz."""
    import io
    import wave
    import numpy as np
    raw = load_audio_bytes(split, row_id, raw_root)
    if raw[:4] == b"RIFF":
        w = wave.open(io.BytesIO(raw))
        n = w.getnframes()
        pcm = w.readframes(n)
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width == 2:
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width {width}")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return x
    import soundfile as sf
    x, _ = sf.read(io.BytesIO(raw), dtype="float32")
    return np.asarray(x, dtype=np.float32).reshape(-1)
