#!/usr/bin/env python3
"""04_extract_wavs.py - One-time extraction of WAV audio from HF parquet to disk.

WHY
   dataset.py's load_wav() reads WAV bytes out of the parquet per-access,
   per-epoch (random row-group seeks into ~300MB files) with num_workers=0,
   which starves the GPU. Extracting the WAVs once to
   data/vcm/processed/wav/<file> (a reproducible, gitignored derivative) lets
   the loader use its fast plain-path branch and lets us raise num_workers.

WHAT
   Reads every row from the split parquets (data/, supplemental_fil/,
   supplemental_synth/, synthetic_negatives/ -- the 13 files that feed
   train/val/test/holdout; numerals OOS pool is excluded) and writes each
   clip's audio.bytes verbatim to data/vcm/processed/wav/<file>.

   The `file` column value (e.g. audio/train_000004_....wav) becomes the
   output path relative to processed/wav/, so the manifest's audio_path column
   resolves directly to a real file.

GUARANTEES
   - Verbatim bytes (no resample/transcode): the loader already resamples to
     16k and crops/pads to window_seconds. Extraction must not alter audio.
   - Resumable: an existing output file with the expected byte length is
     skipped. A corrupt/short file is rewritten.
   - Parallel: one worker per parquet file (13 files -> 13 workers), each
     processing its row groups sequentially. Per-file progress logged to
     work/tmp/extract_wavs.log for stuck-detection.
   - Deterministic: same input parquet -> same output bytes.

OUTPUT
   data/vcm/processed/wav/<file>
   data/vcm/processed/extraction_manifest.json  (provenance + per-file counts)

Usage:
   python3 scripts/04_extract_wavs.py [--workers N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "vcm" / "raw"
OUT = REPO / "data" / "vcm" / "processed" / "wav"
LOG = REPO / "work" / "tmp" / "extract_wavs.log"
MANIFEST_OUT = REPO / "data" / "vcm" / "processed" / "extraction_manifest.json"

# The 13 parquet files that feed the splits (numerals OOS pool excluded).
SPLIT_PARQUETS = [
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
    "data/test-00000-of-00001.parquet",
    "data/holdout-00000-of-00001.parquet",
    "supplemental_fil/train-00000-of-00003.parquet",
    "supplemental_fil/train-00001-of-00003.parquet",
    "supplemental_fil/train-00002-of-00003.parquet",
    "supplemental_synth/train-00000-of-00001.parquet",
    "synthetic_negatives/train-00000-of-00001.parquet",
    "synthetic_negatives/test-00000-of-00001.parquet",
    "synthetic_negatives/noise-00000-of-00001.parquet",
    "synthetic_negatives/tune_oos-00000-of-00001.parquet",
]
# numerals-00000..00005-of-00006.parquet deliberately excluded (OOS stress pool,
# not part of train/val/test/holdout per splits.json).


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def extract_one(rel_parquet: str, dry_run: bool) -> dict:
    """Extract all WAVs from one parquet file. Returns a stats dict."""
    ppath = RAW / rel_parquet
    res = {
        "parquet": rel_parquet,
        "rows": 0,
        "written": 0,
        "skipped_existing": 0,
        "empty_bytes": 0,
        "bad_header": 0,
        "errors": 0,
        "bytes_out": 0,
        "elapsed_s": 0.0,
    }
    if not ppath.exists():
        res["errors"] = 1
        res["error_detail"] = f"MISSING parquet: {ppath}"
        log(f"ERROR {rel_parquet}: missing file")
        return res

    t0 = time.time()
    pf = pq.ParquetFile(ppath)
    n_rg = pf.num_row_groups
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=["file", "audio"])
        files = tbl.column("file").to_pylist()
        audios = tbl.column("audio").to_pylist()
        for fname, a in zip(files, audios):
            res["rows"] += 1
            if a is None:
                res["empty_bytes"] += 1
                continue
            raw_bytes = a.get("bytes") if isinstance(a, dict) else None
            if not raw_bytes:
                res["empty_bytes"] += 1
                continue
            outp = OUT / fname
            # Resumability: skip if already present with correct size.
            if outp.exists() and outp.stat().st_size == len(raw_bytes):
                res["skipped_existing"] += 1
                continue
            if dry_run:
                res["written"] += 1
                continue
            outp.parent.mkdir(parents=True, exist_ok=True)
            tmp = outp.with_suffix(outp.suffix + ".part")
            try:
                with tmp.open("wb") as f:
                    f.write(raw_bytes)
                tmp.replace(outp)
                res["written"] += 1
                res["bytes_out"] += len(raw_bytes)
            except Exception as e:  # noqa: BLE001
                res["errors"] += 1
                if res["errors"] <= 3:
                    log(f"ERROR {rel_parquet} rg{rg} {fname}: {e!r}")
        if (rg + 1) % 10 == 0 or rg == n_rg - 1:
            log(
                f"PROGRESS {rel_parquet}: rg {rg+1}/{n_rg} "
                f"rows={res['rows']} written={res['written']} "
                f"skip={res['skipped_existing']} err={res['errors']}"
            )
    res["elapsed_s"] = round(time.time() - t0, 1)
    log(f"DONE {rel_parquet}: {res}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-reproduce", action="store_true")
    ap.add_argument("--workers", type=int, default=len(SPLIT_PARQUETS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.legacy_reproduce:
        ap.error("Flattened WAV extraction has ambiguous filenames. Use prepare_vcm_protocol.py --cache; no WAV extraction is needed.")
    if not args.dry_run and OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing flattened audio: {OUT}")

    log(f"=== extract_wavs start workers={args.workers} dry_run={args.dry_run} ===")
    OUT.mkdir(parents=True, exist_ok=True)

    # Verify all expected parquets exist up front.
    missing = [p for p in SPLIT_PARQUETS if not (RAW / p).exists()]
    if missing:
        log(f"ERROR missing parquets: {missing}")
        return 2

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, p, args.dry_run): p for p in SPLIT_PARQUETS}
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["parquet"])
    tot_rows = sum(r["rows"] for r in results)
    tot_written = sum(r["written"] for r in results)
    tot_skip = sum(r["skipped_existing"] for r in results)
    tot_err = sum(r["errors"] for r in results)
    tot_empty = sum(r["empty_bytes"] for r in results)
    tot_bytes = sum(r["bytes_out"] for r in results)

    manifest = {
        "purpose": "One-time WAV extraction from HF parquet to processed/wav/ (reproducible derivative, gitignored).",
        "revision": "bfca0bbebfe17e00d6a685d1da573615d29ecdce",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": "data/vcm/processed/wav",
        "parquets": SPLIT_PARQUETS,
        "excluded": ["data/numerals-00000-of-00006.parquet", "(numerals OOS pool, not in splits)"],
        "totals": {
            "rows": tot_rows,
            "written": tot_written,
            "skipped_existing": tot_skip,
            "empty_bytes": tot_empty,
            "errors": tot_err,
            "bytes_out": tot_bytes,
        },
        "per_parquet": results,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"=== extract_wavs END rows={tot_rows} written={tot_written} skip={tot_skip} empty={tot_empty} err={tot_err} bytes={tot_bytes} ===")
    return 0 if tot_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
