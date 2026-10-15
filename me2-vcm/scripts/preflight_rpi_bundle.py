#!/usr/bin/env python3
"""Validate a Raspberry Pi runtime bundle before packaging or field testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REQUIRED = [
    "configs/wakeword.yaml",
    "configs/hardware.yaml",
    "models/wakeword/wakeword_exp15_mined_lp.onnx",
    "audio/wake_chime.wav",
    "src/hey_swift/integration/e2e_loop.py",
    "src/hey_swift/integration/runtime.py",
    "src/hey_swift/audio/speech.py",
    "src/hey_swift/hardware/devices.py",
    "scripts/run_e2e_rpi.py",
    "requirements-pi.txt",
    "integration/assets/voices/en_US-amy-medium.onnx",
    "integration/assets/voices/en_US-amy-medium.onnx.json",
]


def fail(message: str) -> None:
    raise SystemExit(f"[preflight] ERROR: {message}")


def require(path: Path) -> None:
    if not path.exists():
        fail(f"missing required path: {path}")


def load_onnx(path: Path):
    try:
        import onnxruntime as ort
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        fail(f"ONNX model failed to load: {path}: {exc}")


def concrete_shape(shape) -> list:
    return [None if isinstance(v, str) else v for v in shape]


def check_wake_contract(sess, wake_cfg: dict) -> None:
    audio = wake_cfg.get("audio", {}) or {}
    expected = [
        None,
        int(audio.get("input_shape", [80, 100])[0]),
        int(audio.get("input_shape", [80, 100])[1]),
    ]
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    if len(inputs) != 1:
        fail(f"wake ONNX expected 1 input, found {len(inputs)}")
    if concrete_shape(inputs[0].shape) != expected:
        fail(f"wake ONNX input shape {inputs[0].shape} != expected {expected}")
    if len(outputs) != 1 or concrete_shape(outputs[0].shape) != [None, int(wake_cfg.get("model", {}).get("n_classes", 2))]:
        fail(f"wake ONNX output contract invalid: {[(o.name, o.shape) for o in outputs]}")


def check_vcm_contract(sess, vcm_cfg: dict, labels: dict) -> None:
    audio = vcm_cfg.get("audio", {}) or {}
    n_mels = int(audio.get("n_mels", 80))
    sample_rate = int(audio.get("sample_rate", 16000))
    n_fft = int(audio.get("n_fft", 400))
    hop = int(audio.get("hop_length", 160))
    window_seconds = float(audio.get("window_seconds", 2.0))
    frames = 1 + ((int(window_seconds * sample_rate) - n_fft) // hop)
    expected_input = [None, 1, n_mels, frames]
    n_way = int(labels.get("n_way_head") or vcm_cfg.get("schema", {}).get("n_way_head", 0))

    inputs = sess.get_inputs()
    if len(inputs) != 1:
        fail(f"VCM ONNX expected 1 input, found {len(inputs)}")
    if concrete_shape(inputs[0].shape) != expected_input:
        fail(f"VCM ONNX input shape {inputs[0].shape} != expected {expected_input}")

    outputs = {o.name: concrete_shape(o.shape) for o in sess.get_outputs()}
    if outputs.get("intent_logits") != [None, n_way]:
        fail(f"VCM intent output shape {outputs.get('intent_logits')} != [None, {n_way}]")
    slot_sets = labels.get("slot_sets") or {}
    slot_outputs = labels.get("slot_outputs") or {}
    for intent, out_name in slot_outputs.items():
        expected = [None, len(slot_sets.get(intent, []))]
        if outputs.get(out_name) != expected:
            fail(f"VCM slot output {out_name} shape {outputs.get(out_name)} != {expected}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="Bundle/staging root to validate.")
    ap.add_argument("--vcm-onnx", required=True, help="VCM ONNX path relative to root.")
    ap.add_argument("--vcm-labels", required=True, help="VCM labels JSON path relative to root.")
    ap.add_argument("--vcm-config", required=True, help="VCM config path relative to root.")
    ap.add_argument("--require-pi-imports", action="store_true",
                    help="Import all optional Pi inference, audio, speech, GPIO, and D-Bus dependencies.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        fail(f"root is not a directory: {root}")

    required = list(REQUIRED) + [args.vcm_onnx, args.vcm_labels, args.vcm_config]
    for rel in required:
        require(root / rel)

    vcm_cfg = yaml.safe_load((root / args.vcm_config).read_text(encoding="utf-8")) or {}
    wake_cfg = yaml.safe_load((root / "configs/wakeword.yaml").read_text(encoding="utf-8")) or {}
    hw_cfg = yaml.safe_load((root / "configs/hardware.yaml").read_text(encoding="utf-8")) or {}
    labels = json.loads((root / args.vcm_labels).read_text(encoding="utf-8"))

    n_way_cfg = int(vcm_cfg.get("schema", {}).get("n_way_head", 0) or 0)
    n_way_labels = int(labels.get("n_way_head", 0) or 0)
    if n_way_cfg and n_way_labels and n_way_cfg != n_way_labels:
        fail(f"VCM n_way mismatch: config={n_way_cfg}, labels={n_way_labels}")
    intents = labels.get("intents") or []
    if n_way_labels and len(intents) not in {n_way_labels, n_way_labels - 1}:
        fail(
            f"VCM labels intents length {len(intents)} is incompatible with "
            f"n_way_head={n_way_labels}"
        )

    wake_onnx = wake_cfg.get("inference", {}).get("onnx")
    if wake_onnx and not (root / wake_onnx).is_file():
        fail(f"wakeword config inference.onnx points to missing file: {wake_onnx}")

    led = hw_cfg.get("led", {}) or {}
    pins = [led.get("red_gpio"), led.get("green_gpio"), led.get("blue_gpio")]
    if led.get("type") != "rgb_gpio" or pins != [17, 27, 22]:
        fail(f"hardware.yaml LED must match tested RGB BCM pins 17/27/22, found {led}")
    tm = (hw_cfg.get("display", {}) or {}).get("tm1637", {}) or {}
    if tm.get("clk_gpio") != 23 or tm.get("dio_gpio") != 24:
        fail(f"hardware.yaml TM1637 must match tested BCM pins 23/24, found {tm}")

    vcm_sess = load_onnx(root / args.vcm_onnx)
    wake_sess = load_onnx(root / "models/wakeword/wakeword_exp15_mined_lp.onnx")
    check_vcm_contract(vcm_sess, vcm_cfg, labels)
    check_wake_contract(wake_sess, wake_cfg)

    sys.path.insert(0, str(root / "src"))
    for module in (
        "hey_swift.integration.e2e_loop",
        "hey_swift.integration.runtime",
        "hey_swift.vcm.inference",
        "hey_swift.wakeword.inference",
        "hey_swift.audio.speech",
        "hey_swift.hardware.devices",
    ):
        try:
            __import__(module)
        except Exception as exc:
            fail(f"runtime import failed for {module}: {exc}")

    if args.require_pi_imports:
        failed = []
        for module in ("sounddevice", "soundfile", "piper", "gpiozero", "lgpio",
                       "dbus_next", "numpy", "scipy", "onnxruntime"):
            try:
                __import__(module)
            except Exception as exc:
                failed.append(f"{module}: {exc}")
        if failed:
            fail("Pi runtime import failures: " + "; ".join(failed))

    print(f"[preflight] OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
