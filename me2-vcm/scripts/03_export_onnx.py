#!/usr/bin/env python3
"""03_export_onnx.py — export a trained VCM to ONNX for CPU deployment.

Thin wrapper over hey_swift.vcm.inference.export_onnx so the pipeline can be
driven by scripts alone (no CLI install required).

Usage:
    python scripts/03_export_onnx.py --tag a0_scratch
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="a0_scratch")
    ap.add_argument("--config", default="configs/vcm.yaml")
    args = ap.parse_args()

    from hey_swift.vcm.inference import export_onnx
    path = export_onnx(tag=args.tag, config=args.config)
    print(f"[export] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
