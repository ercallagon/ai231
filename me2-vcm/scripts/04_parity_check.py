#!/usr/bin/env python3
"""04_parity_check.py — verify PyTorch <-> ONNX output parity.

Runs a fixed set of inputs through both the torch model and the exported ONNX
and asserts the logits match within tolerance. This is the gate that makes the
Pi (ONNX) deployment trustworthy against the training (torch) model.

Usage:
    python scripts/04_parity_check.py --tag a0_scratch --tol 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="a0_scratch")
    ap.add_argument("--config", default="configs/vcm.yaml")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--n", type=int, default=16, help="Number of random inputs.")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        print(f"[parity] missing dep: {exc}", file=__import__("sys").stderr)
        return 2

    from hey_swift.vcm.models import ArchSpec, build_model
    from hey_swift.vcm.taxonomy import load_taxonomy
    from hey_swift.vcm.features import FeatureConfig
    import yaml

    tax = load_taxonomy(args.config)
    spec = ArchSpec.a0(n_way=tax.n_way_head)
    model = build_model(spec)
    ckpt = Path("experiments/vcm") / args.tag / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    feat = FeatureConfig.from_dict(cfg.get("audio", {}))
    T = 197  # locked 2.0 s window frame count
    onnx_path = Path("models/vcm") / f"{args.tag}.onnx"
    if not onnx_path.exists():
        print(f"[parity] ONNX not found at {onnx_path}; run 03_export_onnx.py first.")
        return 1
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(args.n):
        x = rng.standard_normal((1, 1, feat.n_mels, T)).astype(np.float32)
        with torch.no_grad():
            y_torch = model(torch.from_numpy(x)).numpy()
        y_onnx = sess.run(None, {in_name: x})[0]
        diff = float(np.abs(y_torch - y_onnx).max())
        max_diff = max(max_diff, diff)

    ok = max_diff <= args.tol
    print(f"[parity] max |torch-onnx| = {max_diff:.3e}  (tol={args.tol:.1e})  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
