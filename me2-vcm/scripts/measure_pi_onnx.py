#!/usr/bin/env python3
"""CPU-only feature+ONNX timing; never captures audio or executes actions."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from hey_swift.vcm.features import FeatureConfig, MelFeatures
from hey_swift.vcm.taxonomy import load_taxonomy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_summary(samples_ms, duration_seconds):
    samples = np.asarray(samples_ms, dtype=float)
    if not len(samples) or not np.isfinite(duration_seconds) or duration_seconds <= 0 or not np.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Finite measurements and positive audio duration required")
    return {"n": len(samples), "mean_ms": float(samples.mean()),
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "rtf_p95": float(np.percentile(samples, 95) / (1000 * duration_seconds))}


def hardware_identity():
    path = Path("/proc/device-tree/model")
    model = path.read_text().rstrip("\x00\n") if path.exists() else "unknown"
    return {"board": model, "is_raspberry_pi": "Raspberry Pi" in model,
            "machine": platform.machine(), "platform": platform.platform(),
            "python": platform.python_version()}


def benchmark(model_path, config_path, *, audio=None, threads=1, iterations=100, warmup=20):
    if threads < 1 or iterations < 1 or warmup < 0:
        raise ValueError("Positive threads/iterations and nonnegative warmup required")
    import onnxruntime as ort
    config = yaml.safe_load(Path(config_path).read_text())
    cfg = FeatureConfig.from_dict(config.get("audio", {}))
    tax = load_taxonomy(config_path)
    if audio:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        wave, rate = sf.read(audio, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        if rate != cfg.sample_rate:
            common = gcd(rate, cfg.sample_rate)
            wave = resample_poly(wave, cfg.sample_rate // common, rate // common).astype(np.float32)
        duration = len(wave) / cfg.sample_rate
        if not 0 < duration <= cfg.window_seconds:
            raise ValueError("Use a complete development command within the model window; no silent truncation")
        input_record = {"kind": "development_audio", "sha256": sha256(audio)}
    else:
        wave = np.zeros(cfg.window_samples, dtype=np.float32)
        duration = cfg.window_seconds
        input_record = {"kind": "synthetic_zero_probe", "accuracy_evidence": False}
    if not np.isfinite(wave).all(): raise ValueError("Nonfinite audio")
    full_context = config.get("runtime", {}).get("frontend") == "whole6s_mask600_final_partial_v1"
    if full_context:
        from hey_swift.vcm.fullcontext import FullContextFeatures
        feature = FullContextFeatures()
        if not audio:
            wave = (.1 * np.sin(np.arange(cfg.window_samples) * 2 * np.pi * 440 / cfg.sample_rate)).astype(np.float32)
            input_record["kind"] = "synthetic_sine_probe"
    else:
        feature = MelFeatures(cfg)
    def prepare():
        if full_context:
            mel, mask, receipt = feature.transform(wave)
            if mel is None:
                raise ValueError("Input preprocessing failed: " + receipt["status"])
            return {"mel": mel[None], "frame_mask": mask[None]}
        return {"mel": feature.transform(wave)[None]}
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    expected_inputs = {"mel", "frame_mask"} if full_context else {"mel"}
    if {item.name for item in session.get_inputs()} != expected_inputs:
        raise ValueError("Unexpected model/frontend input contract")
    name = session.get_inputs()[0].name
    feed = prepare()
    probe = feed["mel"]
    declared = session.get_inputs()[0].shape
    if len(declared) != 4 or any(isinstance(d, int) and d != a for d, a in zip(declared, probe.shape)):
        raise ValueError("Feature/model input contract mismatch")
    outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, feed)))
    expected = {"intent_logits": tax.n_way_head}
    expected.update({"slot_" + slot: tax.slot_width(slot) for slot in tax.slotted_intents})
    for output, width in expected.items():
        if output not in outputs or outputs[output].shape != (1, width) or not np.isfinite(outputs[output]).all():
            raise ValueError("Config/model output contract mismatch: " + output)
    measurements = {"feature": [], "onnx": [], "feature_plus_onnx": []}
    for i in range(warmup + iterations):
        start = time.perf_counter_ns()
        feed = prepare()
        prepared = time.perf_counter_ns()
        session.run(None, feed)
        finished = time.perf_counter_ns()
        if i >= warmup:
            measurements["feature"].append((prepared - start) / 1e6)
            measurements["onnx"].append((finished - prepared) / 1e6)
            measurements["feature_plus_onnx"].append((finished - start) / 1e6)
    return {"hardware": hardware_identity(), "model_sha256": sha256(model_path),
        "config_sha256": sha256(config_path), "model_bytes": Path(model_path).stat().st_size,
        "input": input_record, "input_shape": list(probe.shape), "audio_seconds": duration,
        "model_window_seconds": cfg.window_seconds, "runtime": "onnxruntime",
        "runtime_version": ort.__version__, "providers": session.get_providers(),
        "intra_op_threads": threads, "inter_op_threads": 1, "warmup": warmup,
        "timings": {key: timing_summary(value, duration) for key, value in measurements.items()},
        "samples_ms": measurements, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Controlled feature+ONNX microbenchmark; not speech-end-to-action or live benchmark latency",
        "rtf_denominator": "actual resampled input duration, not endpoint wait or feature frame count",
        "session_policy": "Explicit benchmark threads; production runtime thread settings are unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", help="Complete development clip; never official holdout")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--require-pi", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists(): raise FileExistsError(target)
    if args.require_pi and not hardware_identity()["is_raspberry_pi"]:
        parser.error("Pi hardware required; do not label host timing as Pi timing")
    result = benchmark(args.model, args.config, audio=args.audio, threads=args.threads,
                       iterations=args.iterations, warmup=args.warmup)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps({"output": str(target), "hardware": result["hardware"], "timings": result["timings"]}, indent=2))


if __name__ == "__main__": main()
