#!/usr/bin/env python3
"""05_stage_rpi.py — assemble the lean Raspberry Pi 5 deployment bundle.

Copies the exported ONNX models, labels, thresholds, and the systemd unit into
deploy/rpi/ so the folder can be scp'd straight to the Pi. CPU-only: no torch,
no training data.

Usage:
    python scripts/05_stage_rpi.py --tag a0_scratch
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="a0_scratch")
    ap.add_argument("--config", default="configs/vcm.yaml")
    ap.add_argument("--wakeword-onnx", default="models/wakeword/wakeword_exp15_mined_lp.onnx")
    args = ap.parse_args()

    root = repo_root()
    dest = root / "deploy" / "rpi"
    (dest / "models").mkdir(parents=True, exist_ok=True)

    # 1) VCM ONNX + labels
    vcm_onnx = root / "models" / "vcm" / f"{args.tag}.onnx"
    if not vcm_onnx.exists():
        print(f"[stage] VCM ONNX not found: {vcm_onnx}. Run 03_export_onnx.py first.")
        return 1
    shutil.copy2(vcm_onnx, dest / "models" / f"vcm_{args.tag}.onnx")

    labels_src = root / "models" / "vcm" / f"{args.tag}.labels.json"
    if labels_src.exists():
        shutil.copy2(labels_src, dest / "labels.json")
    else:
        print(f"[stage] WARNING: labels not found at {labels_src}")

    # 2) Wake-word ONNX (optional; warn if absent)
    ww_src = root / args.wakeword_onnx
    if ww_src.exists():
        shutil.copy2(ww_src, dest / "models" / "wakeword.onnx")
    else:
        print(f"[stage] WARNING: wake-word ONNX not found at {ww_src}")

    # 3) Thresholds (wake gates) from configs/wakeword.yaml
    ww_cfg = yaml.safe_load(open(root / "configs" / "wakeword.yaml", encoding="utf-8")) or {}
    inf = ww_cfg.get("inference", {}) or {}
    thresholds = {
        "threshold": inf.get("threshold", 0.30),
        "sustain": inf.get("sustain", 2),
        "cooldown_seconds": inf.get("cooldown_seconds", 2.0),
        "rms_floor_db": inf.get("rms_floor_db", -60.0),
        "hop_seconds": inf.get("hop_seconds", 0.1),
    }
    (dest / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n",
                                          encoding="utf-8")

    # 4) Thin launcher
    (dest / "run_live.py").write_text(
        "#!/usr/bin/env python3\n"
        "'''Hey Swift live launcher for the Pi.'''\n"
        "from hey_swift.app import main\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    print(f"[stage] bundle ready at {dest}")
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dest)}  ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
