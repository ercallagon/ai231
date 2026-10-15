#!/usr/bin/env python3
"""Measure ONNX forward cost; no microphone, actions, datasets or model selection."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--audio-seconds", type=float, required=True,
                    help="Nominal input capacity used ONLY for model-only RTF")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--allow-host", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new result path")
    if args.audio_seconds <= 0 or args.threads < 1 or args.warmup < 1 or args.iterations < 20:
        ap.error("Positive capacity/threads/warmup and at least 20 iterations required")
    device_file = Path("/proc/device-tree/model")
    device = device_file.read_text().rstrip("\x00") if device_file.exists() else platform.machine()
    is_pi5 = "Raspberry Pi 5" in device
    if not is_pi5 and not args.allow_host:
        ap.error("Not a detected Raspberry Pi 5; --allow-host marks results as host-only")
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    feeds = {}
    for item in session.get_inputs():
        shape = [1 if i == 0 else d for i, d in enumerate(item.shape)]
        if item.type != "tensor(float)" or not all(isinstance(d, int) and d > 0 for d in shape):
            ap.error(f"Unsupported input contract: {item.name} {item.type} {item.shape}")
        feeds[item.name] = np.ones(shape, np.float32) if "mask" in item.name else np.zeros(shape, np.float32)
    samples = []
    for i in range(args.warmup + args.iterations):
        start = time.perf_counter_ns()
        outputs = session.run(None, feeds)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if not all(np.isfinite(x).all() for x in outputs):
            raise RuntimeError("Nonfinite ONNX outputs")
        if i >= args.warmup:
            samples.append(elapsed)
    p95 = float(np.percentile(samples, 95))
    result = dict(device=device, pi5_verified=is_pi5, platform=platform.platform(),
        python=platform.python_version(), numpy=importlib.metadata.version("numpy"),
        onnxruntime=ort.__version__, provider="CPUExecutionProvider", intra_threads=args.threads,
        inter_threads=1, execution="sequential", warmup=args.warmup, iterations=args.iterations,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), model_bytes=args.model.stat().st_size,
        inputs={k: list(v.shape) for k, v in feeds.items()},
        scope="model_forward_only_synthetic_feature_inputs_no_capture_preprocessing_or_actions",
        mean_ms=float(np.mean(samples)), p50_ms=float(np.median(samples)), p95_ms=p95,
        nominal_audio_seconds=args.audio_seconds, nominal_model_only_rtf_p95=p95/(1000*args.audio_seconds),
        accuracy_measured=False, samples_ms=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples_ms"}, indent=2))


if __name__ == "__main__":
    main()
