#!/usr/bin/env python3
"""Isolated candidate adapter; leaves the frozen, actively used trainer intact.

The process-local a4 factory routing key is replaced with the genuine BC8 spec.
Checkpoints record bc_resnet_broadcast8, never the legacy bc_resnet name.
Training losses, data loading, early stopping and metrics reuse existing code.
"""
import argparse
import contextlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from hey_swift.vcm.broadcast import NAME, build_broadcast_model
from hey_swift.vcm.protocol import load_feature_cache, sha256, verify_protocol
from hey_swift.vcm.taxonomy import load_taxonomy

TAG = "run_041_hf_aligned_v1_bc8_s42"
CONFIG = "configs/vcm_hf_aligned_v1_bc8.yaml"


@contextlib.contextmanager
def candidate_factories():
    from hey_swift.vcm import models, evaluate
    import torch
    factory = models.build_model
    descriptor = models.ArchSpec.__dict__["a4"]
    loader = evaluate._load_model
    def build(spec, slot_heads=None):
        if spec.name == NAME:
            return build_broadcast_model(spec.n_way_head, slot_heads)
        return factory(spec, slot_heads)
    def load(config, path, device):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("arch") != NAME:
            return loader(config, path, device)
        taxonomy = load_taxonomy(config)
        heads = {name: taxonomy.slot_classes(name) for name in checkpoint["slot_heads"]}
        model = build_broadcast_model(taxonomy.n_way_head, heads)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.to(device).eval(), taxonomy, checkpoint
    models.ArchSpec.a4 = classmethod(lambda cls, n_way=20: cls(NAME, [64, 96, 128, 160], width=64, n_way_head=n_way))
    models.build_model = build
    evaluate._load_model = load
    try:
        yield
    finally:
        models.ArchSpec.a4 = descriptor
        models.build_model = factory
        evaluate._load_model = loader


def check_candidate(config):
    cfg = yaml.safe_load(Path(config).read_text())
    if cfg["experiment"]["architecture"] != NAME or cfg["training"]["arch"] != "a4":
        raise ValueError("Wrong isolated candidate config")
    for path, digest in cfg["experiment"]["implementation_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Candidate implementation changed: {path}")
    verify_protocol(cfg)
    return cfg


def export_model(model, path, temperature=1.0):
    import torch
    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
        def forward(self, x):
            result = self.model(x)
            return tuple([result["intent"] / temperature] + [result["slots"][name] for name in self.model.slot_head_map])
    wrapper = Export().eval()
    x = torch.randn(1, 1, 80, 198)
    names = ["intent_logits"] + ["slot_" + name for name in model.slot_head_map]
    torch.onnx.export(wrapper, x, str(path), input_names=["mel"], output_names=names,
                      dynamic_axes={name: {0: "batch"} for name in ["mel"] + names}, opset_version=17)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        sample = torch.randn(batch, 1, 80, 198)
        with torch.no_grad():
            expected = wrapper(sample)
        actual = session.run(None, {"mel": sample.numpy()})
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b.numpy(), rtol=1e-4, atol=1e-4)
    return session


def preflight():
    import torch
    torch.set_num_threads(1)
    tax = load_taxonomy(CONFIG)
    model = build_broadcast_model(20, {name: tax.slot_classes(name) for name in tax.slotted_intents}).eval()
    stage = Path("work/vcm_next_bc8")
    stage.mkdir(exist_ok=True)
    with torch.no_grad():
        sample = torch.randn(2, 1, 80, 198)
        result = model(sample)
        assert result["intent"].shape == (2, 20)
        assert all(value.shape == (2, 3) for value in result["slots"].values())
    model.train()
    output = model(sample)
    loss = output["intent"].square().mean() + sum(v.square().mean() for v in output["slots"].values())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    session = export_model(model, stage / "untrained_preflight.onnx")
    x = np.zeros((1, 1, 80, 198), dtype=np.float32)
    for _ in range(20):
        session.run(None, {"mel": x})
    samples = []
    for _ in range(100):
        begin = time.perf_counter()
        session.run(None, {"mel": x})
        samples.append((time.perf_counter() - begin) * 1000)
    proof = {"architecture": NAME, "n_params": sum(p.numel() for p in model.parameters()),
             "onnx_bytes": (stage / "untrained_preflight.onnx").stat().st_size,
             "onnx_shapes_and_parity_passed": True, "finite_backward_passed": True,
             "untrained_host_onnx_p50_ms": float(np.median(samples)), "untrained_host_onnx_p95_ms": float(np.percentile(samples, 95)),
             "not_pi_verified": True}
    (stage / "preflight.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


def calibrate(tag=TAG, config=CONFIG):
    import torch
    from scipy.optimize import minimize_scalar
    from hey_swift.vcm import dataset as ds, evaluate
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path(config).read_text())
    verify_protocol(cfg)
    model, taxonomy, _ = evaluate._load_model(config, Path("experiments/vcm") / tag / "best.pt", torch.device("cpu"))
    cache, index = load_feature_cache(cfg)
    logits, labels, groups = [], [], []
    with torch.no_grad():
        for split in ("val", "tune_oos"):
            rows = ds.configured_rows(cfg, split)
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                x = torch.from_numpy(np.stack([cache[index[r.audio_path]] for r in batch]))
                logits.extend(model(x)["intent"].numpy())
                labels.extend(taxonomy.oos_idx if r.intent == taxonomy.oos_class else taxonomy.intent_to_idx[r.intent] for r in batch)
                groups.extend([split] * len(batch))
    logits, labels, groups = np.asarray(logits), np.asarray(labels), np.asarray(groups)
    val = groups == "val"
    def nll(temperature):
        from scipy.special import logsumexp
        scaled = logits[val] / temperature
        return float((logsumexp(scaled, axis=1) - scaled[np.arange(val.sum()), labels[val]]).mean())
    optimum = minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded")
    from scipy.special import softmax
    probabilities = softmax(logits / optimum.x, axis=1)
    prediction, confidence = probabilities.argmax(1), probabilities.max(1)
    candidates = []
    for threshold in np.arange(0.1, 0.951, 0.025):
        gated = np.where(confidence < threshold, taxonomy.oos_idx, prediction)
        tune = groups == "tune_oos"
        in_scope = val & (labels != taxonomy.oos_idx)
        candidates.append({"threshold": float(threshold), "tune_oos_false_accept_rate": float((gated[tune] != taxonomy.oos_idx).mean()),
                           "val_in_scope_correct_action_intent_rate": float((gated[in_scope] == labels[in_scope]).mean()),
                           "val_in_scope_false_reject_rate": float((gated[in_scope] == taxonomy.oos_idx).mean())})
    feasible = [c for c in candidates if c["tune_oos_false_accept_rate"] <= 0.05]
    chosen = max(feasible, key=lambda c: (c["val_in_scope_correct_action_intent_rate"], -c["val_in_scope_false_reject_rate"], -c["threshold"])) if feasible else None
    output = Path("experiments/vcm") / tag
    if (output / "calibration.json").exists() or (output / "calibrated").exists():
        raise FileExistsError("Refusing to overwrite calibration artifacts")
    proof = {"temperature": float(optimum.x), "temperature_fit_success": bool(optimum.success), "val_nll_before": nll(1), "val_nll_after": nll(optimum.x),
             "chosen": chosen, "candidates": candidates, "protocol_sha256": sha256(cfg["data"]["protocol"]), "checkpoint_sha256": sha256(output / "best.pt"),
             "test_or_holdout_used": False, "provisional_not_pi_promoted": True}
    (output / "calibration.json").write_text(json.dumps(proof, indent=2))
    if chosen and optimum.success:
        calibrated = output / "calibrated"
        calibrated.mkdir(exist_ok=False)
        shutil.copy2(output / "best.pt", calibrated / "best.pt")
        cfg["oos"]["confidence_threshold"] = chosen["threshold"]
        cfg.setdefault("experiment", {})["calibration_temperature"] = float(optimum.x)
        (calibrated / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps({k: v for k, v in proof.items() if k != "candidates"}, indent=2))


def main(args):
    if args.mode == "preflight":
        preflight()
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("experiment", {}).get("architecture") == NAME:
        check_candidate(args.config)
    elif args.mode not in ("calibrate", "eval") or args.tag not in ("run_039_hf_aligned_v1_a10_s42", "run_040_hf_aligned_v1_a6_s42"):
        raise ValueError("Noncandidate configs are allowed only for the two frozen baseline calibration/evaluation references")
    else:
        verify_protocol(cfg)
    with candidate_factories():
        from hey_swift.vcm import train, evaluate
        if args.mode == "train":
            train.run(TAG, config=args.config, seed=42)
        elif args.mode == "calibrate":
            calibrate(args.tag, args.config)
        elif args.mode == "eval":
            cfg = yaml.safe_load(Path(args.config).read_text())
            temperature = cfg.get("experiment", {}).get("calibration_temperature", 1.0)
            if temperature != 1.0:
                old = evaluate._load_model
                def scaled_loader(*params):
                    model, taxonomy, checkpoint = old(*params)
                    import torch
                    class Scaled(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.model = model
                        def forward(self, x):
                            result = self.model(x)
                            return dict(result, intent=result["intent"] / temperature)
                    return Scaled().eval(), taxonomy, checkpoint
                evaluate._load_model = scaled_loader
            print(json.dumps(evaluate.evaluate(args.tag, args.split, args.config,
                                              exp_dir=Path(args.exp_dir) if args.exp_dir else None, device_str="cpu"), indent=2))
        elif args.mode == "latency":
            spec = importlib.util.spec_from_file_location("latency", "scripts/benchmark_vcm_latency.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.argv = [sys.argv[0], "--tag", TAG, "--config", args.config, "--device", "cpu", "--iters", "200", "--warmup", "30"]
            module.main()
        elif args.mode == "export":
            import torch
            output = Path("experiments/vcm") / TAG
            model, taxonomy, _ = evaluate._load_model(args.config, output / "best.pt", torch.device("cpu"))
            export_model(model, output / "model.onnx")
            if (output / "calibration.json").exists():
                calibration = json.loads((output / "calibration.json").read_text())
                if calibration["chosen"] and calibration["temperature_fit_success"]:
                    export_model(model, output / "calibrated/model.onnx", calibration["temperature"])
            (output / "labels.json").write_text(json.dumps({"intents": list(taxonomy.intents) + [taxonomy.oos_class], "slots": {n: taxonomy.slot_classes(n) for n in taxonomy.slotted_intents}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "train", "eval", "calibrate", "latency", "export"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--split", default="val", choices=["val", "tune_oos", "test_core", "test_negatives", "test"])
    parser.add_argument("--exp-dir")
    main(parser.parse_args())
