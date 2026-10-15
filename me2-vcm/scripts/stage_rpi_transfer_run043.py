#!/usr/bin/env python3
"""Stage a lean, preflighted Raspberry Pi E2E transfer bundle for run_043.

Versioned staging adapter. It does NOT modify the run_021 packager
(scripts/package_rpi_transfer.py) or any original artifact. It:

  * ships the run_043 CALIBRATED ONNX (temperature scaling already baked into
    the graph) with a runtime config whose oos.confidence_threshold matches the
    calibrated gate;
  * emits a runtime labels file in the packager format the runtime + preflight
    expect (n_way_head / slot_sets / slot_outputs), with STRING slot values
    (the run_043 protocol uses strings, not the legacy run_021 ints);
  * preserves the tested wakeword model, wiring, TTS voice, music and runtime
    fixes;
  * excludes training data, checkpoints, feature caches and backup trees;
  * writes SHA-256 sums and runs the existing preflight before tarring.

Usage:
    PYTHONPATH=src python -B scripts/stage_rpi_transfer_run043.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TAG = "run_043_hf_grouped_v2b_bc8_sourceft_s42"
EXP_DIR = Path("experiments/vcm") / TAG
CAL_DIR = EXP_DIR / "calibrated"
CAL_ONNX = CAL_DIR / "model.onnx"
CAL_CONFIG = CAL_DIR / "config.yaml"
CAL_LABELS = CAL_DIR / "labels.json"


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


def build_runtime_labels(root: Path, dest: Path, tag: str) -> None:
    """Write models/vcm/<tag>.labels.json in the runtime/preflight format.

    The export labels (calibrated/labels.json) carry only {intents, slots}.
    The runtime VCMInference and preflight need n_way_head, slot_sets and
    slot_outputs. Slot values are STRINGS (run_043 protocol), matching the
    runtime MODEL_ACTION_SLOTS and the action-services SLOTS.
    """
    import yaml

    cal = json.loads(CAL_LABELS.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CAL_CONFIG.read_text(encoding="utf-8"))
    schema = cfg.get("schema", {}) or {}
    n_way = int(schema.get("n_way_head", 20))
    slot_sets = schema.get("slot_sets", {}) or {}
    # Guarantee string values (the run_043 protocol stores strings).
    slot_sets = {k: [str(v) for v in vals] for k, vals in slot_sets.items()}
    slot_outputs = {intent: f"slot_{intent}" for intent in slot_sets}
    labels = {
        "intents": list(cal.get("intents", [])),
        "oos_class": schema.get("oos_class", "OUT_OF_SCOPE"),
        "n_way_head": n_way,
        "slot_reject": bool(schema.get("slot_reject", False)),
        "slot_sets": slot_sets,
        "slot_outputs": slot_outputs,
    }
    out = dest / "models" / "vcm" / f"{tag}.labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")


def build_runtime_config(root: Path, dest: Path) -> str:
    """Write a self-contained runtime VCM config (no dataset/protocol refs).

    Carries exactly what the runtime needs: audio feature params, the schema
    (n_way, OOS class, slotted intents + STRING slot sets) and the calibrated
    oos.confidence_threshold. Temperature scaling is NOT a runtime knob here --
    it is already inside the calibrated ONNX graph.
    """
    import yaml

    cfg = yaml.safe_load(CAL_CONFIG.read_text(encoding="utf-8"))
    audio = cfg.get("audio", {}) or {}
    schema = cfg.get("schema", {}) or {}
    slot_sets = {k: [str(v) for v in vals] for k, vals in (schema.get("slot_sets", {}) or {}).items()}
    runtime_cfg = {
        "audio": {
            "sample_rate": audio.get("sample_rate", 16000),
            "n_fft": audio.get("n_fft", 400),
            "hop_length": audio.get("hop_length", 160),
            "n_mels": audio.get("n_mels", 80),
            "window_seconds": audio.get("window_seconds", 2.0),
            "f_min": audio.get("f_min", 20.0),
            "f_max": audio.get("f_max", 8000.0),
        },
        "schema": {
            "n_way_head": schema.get("n_way_head", 20),
            "oos_class": schema.get("oos_class", "OUT_OF_SCOPE"),
            "slot_reject": bool(schema.get("slot_reject", False)),
            "slotted_intents": list(schema.get("slotted_intents", [])),
            "slot_sets": slot_sets,
        },
        "oos": {
            "strategy": (cfg.get("oos", {}) or {}).get("strategy", "learned_plus_confidence_gate"),
            "confidence_threshold": float((cfg.get("oos", {}) or {}).get("confidence_threshold", 0.875)),
        },
        "experiment": {
            "architecture": (cfg.get("experiment", {}) or {}).get("architecture", "bc_resnet_broadcast8"),
            "note": "run_043 calibrated export; temperature scaling is baked into the ONNX graph "
                    "(do not re-apply at runtime). confidence_threshold is the calibrated gate.",
            "calibration_temperature": float((cfg.get("experiment", {}) or {}).get("calibration_temperature", 1.0)),
        },
    }
    rel = "configs/vcm_run_043_calibrated.yaml"
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(runtime_cfg, sort_keys=False), encoding="utf-8")
    return rel


def write_sha256s(dest: Path) -> None:
    rows = []
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        rel = path.relative_to(dest).as_posix()
        if rel == "SHA256SUMS":
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}")
    (dest / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="work")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--speech-assets", help="Versioned precomputed speech directory to include.")
    args = ap.parse_args(argv)

    root = repo_root()
    if not CAL_ONNX.is_file():
        raise FileNotFoundError(CAL_ONNX)
    if not CAL_CONFIG.is_file():
        raise FileNotFoundError(CAL_CONFIG)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.name or f"rpi_transfer_{args.tag}_calibrated_lean_{stamp}"
    out_dir = root / args.out_dir
    dest = out_dir / name
    tar_path = out_dir / f"{name}.tar.gz"

    if dest.exists():
        raise FileExistsError(dest)
    if tar_path.exists():
        raise FileExistsError(tar_path)

    dest.mkdir(parents=True)

    vcm_config_rel = build_runtime_config(root, dest)
    build_runtime_labels(root, dest, args.tag)

    for rel in (
        "requirements.txt",
        "requirements-pi.txt",
        "pyproject.toml",
        "configs/wakeword.yaml",
        "configs/hardware.yaml",
        "configs/actions.yaml",
        "models/wakeword/wakeword_exp15_mined_lp.onnx",
        "audio/wake_chime.wav",
        "scripts/run_e2e_rpi.py",
        "scripts/preflight_rpi_bundle.py",
        "scripts/test_local_pi_updates.py",
        "scripts/test_prepi_runtime_v5.py",
        "scripts/validate_runtime_assets.py",
        "tests/test_rpi_slot_translation.py",
        "docs/rpi-e2e.md",
        "docs/rpi-runtime-v5.md",
        "docs/rpi-runtime-v6.md",
        "docs/command-cheatsheet.md",
        "scripts/test_pi_runtime_v6.py",
        "scripts/diagnose_pi_runtime.py",
        "LOCAL_PI_UPDATES.md",
    ):
        copy_file(root, dest, rel)

    # run_043 calibrated ONNX (temperature baked in) + runtime labels.
    onnx_out = dest / "models" / "vcm" / f"{args.tag}.onnx"
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CAL_ONNX, onnx_out)

    copy_tree(root / "src" / "hey_swift", dest / "src" / "hey_swift")
    copy_integration_assets(root, dest)
    if args.speech_assets:
        speech_assets = root / args.speech_assets
        manifest = json.loads((speech_assets / "manifest.json").read_text())
        voice = dest / "integration/assets/voices/en_US-amy-medium.onnx"
        if manifest["voice_sha256"] != hashlib.sha256(voice.read_bytes()).hexdigest():
            raise ValueError("Speech cache does not match packaged voice")
        if manifest["voice_config_sha256"] != hashlib.sha256(Path(str(voice) + ".json").read_bytes()).hexdigest():
            raise ValueError("Speech cache does not match packaged voice config")
        for entry in manifest["phrases"].values():
            path = speech_assets / entry["file"]
            if path.resolve().parent != speech_assets.resolve():
                raise ValueError("Speech cache path must be a direct child")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Invalid speech cache checksum: {path}")
        copy_tree(speech_assets, dest / "integration/assets/speech")
    shutil.copy2(root / "docs/rpi-runtime-v5.md", dest / "RUN_043_RPI_README.md")
    subprocess.run([sys.executable, "-B", "scripts/test_prepi_runtime_v5.py"], check=True, cwd=dest)
    subprocess.run([sys.executable, "-B", "scripts/test_local_pi_updates.py"], check=True, cwd=dest)
    subprocess.run([sys.executable, "-B", "tests/test_rpi_slot_translation.py"], check=True, cwd=dest)

    if not args.skip_preflight:
        cmd = [
            sys.executable, "-B",
            str(root / "scripts" / "preflight_rpi_bundle.py"),
            "--root", str(dest),
            "--vcm-onnx", f"models/vcm/{args.tag}.onnx",
            "--vcm-labels", f"models/vcm/{args.tag}.labels.json",
            "--vcm-config", vcm_config_rel,
        ]
        subprocess.run(cmd, check=True, cwd=root)

    if args.speech_assets:
        subprocess.run([sys.executable, "-B", "scripts/validate_runtime_assets.py", "--root", 
                        str(dest), "--output", str(dest / "HOST_ASSET_VALIDATION.json")], check=True, cwd=dest)

    write_sha256s(dest)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(dest, arcname=name, filter=lambda info: None if (
            "__pycache__" in info.name or info.name.endswith((".pyc", ".pyo"))
        ) else info)

    print(tar_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
