#!/usr/bin/env python3
"""Build a lean, preflighted Raspberry Pi E2E transfer tarball."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


DEFAULT_TAG = "run_021_arch_a6_resnet2d_slots"
DEFAULT_CONFIG = "configs/vcm_arch_screen_a6_resnet2d_slots.yaml"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ignore_runtime_clutter(_dir, names):
    ignored = []
    for name in names:
        if name == "__pycache__" or name.endswith((".pyc", ".pyo", ".tar.gz")):
            ignored.append(name)
        elif name in {".pytest_cache", ".mypy_cache", ".ruff_cache", "tests", "wakeword-test-bkp"}:
            ignored.append(name)
    return set(ignored)


def copy_file(root: Path, dest: Path, rel: str) -> None:
    src = root / rel
    if not src.is_file():
        raise FileNotFoundError(src)
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)


def copy_tree(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dest.exists():
        # Refuse an existing destination instead of deleting it: no silent
        # overwrite of a prior staging tree. The caller chooses a fresh path.
        raise FileExistsError(dest)
    shutil.copytree(src, dest, ignore=ignore_runtime_clutter)


def copy_integration_assets(root: Path, dest: Path) -> None:
    """Copy only runtime assets: Amy TTS voice and offline music."""
    assets = root / "work" / "integration" / "assets"
    voices_out = dest / "integration" / "assets" / "voices"
    music_out = dest / "integration" / "assets" / "music"
    voices_out.mkdir(parents=True, exist_ok=True)
    music_out.mkdir(parents=True, exist_ok=True)
    for name in ("en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"):
        shutil.copy2(assets / "voices" / name, voices_out / name)
    for src in sorted((assets / "music").glob("*.mp3")):
        shutil.copy2(src, music_out / src.name)


def write_sha256s(dest: Path) -> None:
    rows = []
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        rel = path.relative_to(dest).as_posix()
        if rel == "PATCH_SHA256SUMS":
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}")
    (dest / "PATCH_SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="work")
    args = ap.parse_args(argv)

    root = repo_root()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.name or f"rpi_transfer_{args.tag}_lean_{stamp}"
    out_dir = root / args.out_dir
    dest = out_dir / name
    tar_path = out_dir / f"{name}.tar.gz"

    if dest.exists():
        raise FileExistsError(dest)
    if tar_path.exists():
        raise FileExistsError(tar_path)

    dest.mkdir(parents=True)

    for rel in (
        "requirements.txt",
        "requirements-pi.txt",
        "pyproject.toml",
        "configs/wakeword.yaml",
        "configs/hardware.yaml",
        "configs/actions.yaml",
        args.config,
        "models/wakeword/wakeword_exp15_mined_lp.onnx",
        f"models/vcm/{args.tag}.onnx",
        f"models/vcm/{args.tag}.labels.json",
        "audio/wake_chime.wav",
        "scripts/run_e2e_rpi.py",
        "scripts/preflight_rpi_bundle.py",
        "scripts/test_local_pi_updates.py",
        "RUN_021_RPI_README.md",
        "LOCAL_PI_UPDATES.md",
    ):
        copy_file(root, dest, rel)

    copy_tree(root / "src" / "hey_swift", dest / "src" / "hey_swift")
    copy_integration_assets(root, dest)
    write_sha256s(dest)

    cmd = [
        sys.executable,
        "-B",
        str(root / "scripts" / "preflight_rpi_bundle.py"),
        "--root", str(dest),
        "--vcm-onnx", f"models/vcm/{args.tag}.onnx",
        "--vcm-labels", f"models/vcm/{args.tag}.labels.json",
        "--vcm-config", args.config,
    ]
    subprocess.run(cmd, check=True, cwd=root)

    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(dest, arcname=name, filter=lambda info: None if (
            "__pycache__" in info.name or info.name.endswith((".pyc", ".pyo"))
        ) else info)

    print(tar_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
