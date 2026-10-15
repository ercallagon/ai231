#!/usr/bin/env python3
"""01_download_hf.py — fetch the airimonda/ai231-me2-voice-commands HF dataset.

Downloads the parquet splits into ``data/vcm/raw/`` (relative to the repo root).
Idempotent + resumable: files already present at the correct size are skipped.
Writes ``data/vcm/raw/download_manifest.json`` recording each file's path, size,
LFS oid, and dataset revision for reproducibility.

Usage:
    python scripts/01_download_hf.py            # full canonical snapshot (all 18 parquets)
    python scripts/01_download_hf.py --core     # core splits only (data/*)
    python scripts/01_download_hf.py --verify   # re-verify sizes/oids, no download

Pinned revision: bfca0bbe (2026-10-03T02:44:24Z) — the complete HF clone as of
the campaign freeze. All 18 parquets below are byte-identical to the remote LFS
objects at that revision (verified 2026-10-04). DO NOT bump the revision without
a new experiment tag (reproducibility contract — see data/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# --- Pinned dataset identity (DO NOT bump without a new experiment tag) ------
REPO_ID = "airimonda/ai231-me2-voice-commands"
REVISION = "bfca0bbebfe17e00d6a685d1da573615d29ecdce"

# (relative_path, expected_size_bytes, lfs_oid) — all 18 parquets at bfca0bbe.
CORE_FILES = [
    ("data/train-00000-of-00002.parquet", 295_319_324,
     "f4b0717e95dc1a7fa5566be6a20212c04b554c7f69de8815646b36606a30bd83"),
    ("data/train-00001-of-00002.parquet", 391_176_249,
     "5011a4ff41c99b32acf07b3e5ea23ba36b5c03bf286b0731a0f85b5b4d9b5669"),
    ("data/test-00000-of-00001.parquet", 272_526_000,
     "40e9b66b4e12bc5b012bfa2dc7367602740ac394467e48d4924364c46f912f32"),
    ("data/holdout-00000-of-00001.parquet", 15_363_388,
     "c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e"),
    ("data/numerals-00000-of-00006.parquet", 321_316_611,
     "cefd4884acf3aec7c33625dc46a970d7b5b921d0507e7b6e41bd430a3eb87504"),
    ("data/numerals-00001-of-00006.parquet", 326_077_666,
     "dcd8edade416bba1553d7d50db3bdfcacd8ac4aaad54b775316990741168a8a8"),
    ("data/numerals-00002-of-00006.parquet", 321_668_264,
     "768f4d9c426668831ddbece9d496c161ac3149ba1ab16bec722d1d0fd35592c8"),
    ("data/numerals-00003-of-00006.parquet", 347_705_349,
     "d9a58e4d10f625f170743f1623b7750f2933d8e6a2902ab99b634488a6041ec7"),
    ("data/numerals-00004-of-00006.parquet", 348_820_081,
     "b057c8930fa00f52481007be50e15382c4b68b9044a46cda1e923b295235f281"),
    ("data/numerals-00005-of-00006.parquet", 349_447_467,
     "aced4a5f0588d285df4155cb2da66ba327c1d35336079228a31f05a142bf315f"),
]

FULL_EXTRA_FILES = [
    ("supplemental_fil/train-00000-of-00003.parquet", 345_769_295,
     "cc633df2de7bc77bc0dc16bae2c75537eba1ed13f8b4f68e8672a782ff2ba139"),
    ("supplemental_fil/train-00001-of-00003.parquet", 333_143_732,
     "ac47a1e85122ba0488a0715558e566623464f7daceb67f2833828c467ec6aacd"),
    ("supplemental_fil/train-00002-of-00003.parquet", 314_234_258,
     "321bab5bd8cf85ee5311ae79435e9a8700589d2ca44cabd100dce1c4a138eabf"),
    ("supplemental_synth/train-00000-of-00001.parquet", 312_090_900,
     "a717d6113c9a5714e05f1daaca8e685e130c10ecc53bb9a493374eff85c9751e"),
    ("synthetic_negatives/train-00000-of-00001.parquet", 69_579_534,
     "82cd660ed5f37ed7a522f68a1fc7d6c3216659f1222ed4c68cf8f8114793452b"),
    ("synthetic_negatives/test-00000-of-00001.parquet", 17_411_607,
     "b861abd530bbf5e15e6a9f61caca8a029a2867e33c9397317ac5e8f01b74b9f4"),
    ("synthetic_negatives/noise-00000-of-00001.parquet", 239_610_849,
     "ca09df8a1fa50dea8eb2d83aaad184af7cbcb91e2380fc220c0f429319d35c22"),
    ("synthetic_negatives/tune_oos-00000-of-00001.parquet", 59_742_483,
     "6038de86e4efb18b380bb385494ca84054c4b89a46eb61a4f680da31a95f9876"),
]

MANIFEST_NAME = "download_manifest.json"


def repo_root() -> Path:
    """Return the me2-vcm/ repo root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def raw_root(root: Path) -> Path:
    return root / "data" / "vcm" / "raw"


def files_to_fetch(core_only: bool) -> list[tuple[str, int, str]]:
    files = list(CORE_FILES)
    if not core_only:
        files += FULL_EXTRA_FILES
    return files


def ensure_hf_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "[setup] Install the data extra first: python -m pip install -e '.[data]'"
        ) from exc


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_only(dest_root: Path, files: list) -> int:
    ok = True
    for rel, exp_size, oid in files:
        dest = dest_root / rel
        if not dest.exists():
            print(f"[MISSING] {rel}")
            ok = False
            continue
        sz = dest.stat().st_size
        if sz != exp_size:
            status = f"SIZE MISMATCH (got {sz}, want {exp_size})"
            file_ok = False
            ok = False
        else:
            actual_oid = sha256_file(dest)
            status = "SHA-256 OK" if actual_oid == oid else f"SHA-256 MISMATCH (got {actual_oid})"
            file_ok = actual_oid == oid
            ok = ok and file_ok
        print(f"[{'OK' if file_ok else 'BAD'}] {rel}  {human(sz)}  {status}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", action="store_true",
                    help="Only download the core data/* splits (skip supplemental/synth-negatives).")
    ap.add_argument("--verify", action="store_true",
                    help="Only verify existing files; do not download.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if file exists at correct size.")
    args = ap.parse_args()

    root = repo_root()
    dest_root = raw_root(root)
    dest_root.mkdir(parents=True, exist_ok=True)

    files = files_to_fetch(args.core)
    total_expected = sum(sz for _, sz, _ in files)
    print(f"[plan] repo_id={REPO_ID}")
    print(f"[plan] revision={REVISION}")
    print(f"[plan] files={len(files)}  total={human(total_expected)}")
    print(f"[plan] dest={dest_root}")

    if args.verify:
        return verify_only(dest_root, files)

    ensure_hf_hub()
    from huggingface_hub import hf_hub_download

    manifest: dict = {"repo_id": REPO_ID, "revision": REVISION, "files": []}
    ok = True
    for rel, exp_size, oid in files:
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size == exp_size and not args.force:
            actual_oid = sha256_file(dest)
            if actual_oid == oid:
                print(f"[skip] {rel} (size and SHA-256 verified, {human(exp_size)})")
                manifest["files"].append({"path": rel, "size": exp_size,
                                          "lfs_oid": oid, "sha256": actual_oid, "status": "ok"})
                continue
            print(f"[WARN] {rel}: SHA-256 mismatch; fetching the pinned object again")

        try:
            hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=rel,
                            revision=REVISION, local_dir=str(dest_root),
                            force_download=args.force)
            got = dest.stat().st_size
            actual_oid = sha256_file(dest)
            if got != exp_size or actual_oid != oid:
                print(f"[WARN] {rel}: expected {exp_size} bytes / {oid}; got {got} bytes / {actual_oid}")
                ok = False
                manifest["files"].append({"path": rel, "size": got, "lfs_oid": oid,
                                          "sha256": actual_oid,
                                          "status": "integrity_mismatch"})
            else:
                print(f"[done] {rel} ({human(got)}, SHA-256 verified)")
                manifest["files"].append({"path": rel, "size": got, "lfs_oid": oid,
                                          "sha256": actual_oid, "status": "ok"})
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {rel}: {e!r}")
            ok = False
            manifest["files"].append({"path": rel, "size": None, "lfs_oid": oid, "status": f"error: {e!r}"})

    manifest["total_size_bytes"] = sum(f["size"] or 0 for f in manifest["files"])
    manifest["total_files"] = len(files)
    (dest_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {dest_root / MANIFEST_NAME}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
