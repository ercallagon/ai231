#!/usr/bin/env python
"""Small VCM inference latency proxy.

This is not a Raspberry Pi benchmark. It is a controlled single-sample CPU/GPU
forward-pass proxy so architecture screens can compare relative deployment cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    import torch
    from hey_swift.vcm.evaluate import _load_model

    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments/vcm") / args.tag
    model, _, ckpt = _load_model(args.config, exp_dir / "best.pt", device)
    x = torch.zeros(1, 1, 80, 198, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    payload = {
        "tag": args.tag,
        "device": str(device),
        "iters": args.iters,
        "warmup": args.warmup,
        "mean_ms": round((elapsed / max(args.iters, 1)) * 1000.0, 4),
        "n_params": ckpt.get("n_params"),
        "checkpoint_bytes": (exp_dir / "best.pt").stat().st_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = exp_dir / f"latency_{device.type}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
