#!/usr/bin/env python3
"""Freeze plan, new implementation and numerical/export readiness."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import yaml
from hey_swift.vcm.protocol import sha256
from hey_swift.vcm.taxonomy import load_taxonomy
from hey_swift.vcm.validity import ValidityModel
from run_vcm_oos_discovery_queue import config_for
from vcm_oos_discovery import STAGE, check, write

SOURCES = ["src/hey_swift/vcm/validity.py", "scripts/vcm_oos_discovery.py",
           "scripts/run_vcm_oos_discovery_queue.py", "scripts/prepare_vcm_oos_discovery.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-readiness", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(json.loads((STAGE / "plan.json").read_text())["first_config"])
        return
    if args.seal_readiness:
        plan = json.loads((STAGE / "plan.json").read_text())
        check(plan["first_config"])
        from validate_vcm_protocol import validate
        validate(plan["first_config"])
        suites = list(ET.parse(STAGE / "regression.xml").getroot().iter("testsuite"))
        if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("errors", "failures")):
            raise ValueError("Regression suite failed")
        if sum(int(s.attrib["tests"]) for s in suites) < 80:
            raise ValueError("Incomplete regression suite")
        numerical = json.loads((STAGE / "preflight.json").read_text())
        if not numerical["finite_backward"] or not numerical["onnx_parity"]:
            raise ValueError("Model preflight failed")
        cfg = yaml.safe_load(Path(plan["first_config"]).read_text())
        sources = dict(cfg["experiment"]["implementation_sha256"])
        sources[plan["first_config"]] = sha256(plan["first_config"])
        sources[str(STAGE / "plan.json")] = sha256(STAGE / "plan.json")
        for path in list(cfg["experiment"]["implementation_sha256"]) + [plan["first_config"], "tests/test_vcm_validity.py", "docs/vcm-oos-discovery.md"]:
            target = STAGE / "source_snapshot_final" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        with (STAGE / "readiness.json").open("x") as stream:
            json.dump({"passed": True, "sources": sources, "preflight_sha256": sha256(STAGE / "preflight.json"),
                "regression_sha256": sha256(STAGE / "regression.xml"),
                "tests": sum(int(s.attrib["tests"]) for s in suites),
                "skipped": sum(int(s.attrib.get("skipped", 0)) for s in suites)}, stream, indent=2)
        print("Readiness sealed")
        return
    STAGE.mkdir(exist_ok=False)
    config = config_for("run_044_hf_grouped_v2b_bc8_validity_s42", .0003, .25, 42)
    write(STAGE / "plan.json", {"first_config": config, "maximum_new_training_runs": 3,
        "primary_change": "Shared BC8 embedding plus 257-parameter validity head, balanced BCE weight 0.25",
        "refinement_rules": "If gain qualifies: LR 1e-4; otherwise BCE 0.1 if raw E2E regressed >0.5pt, else BCE 0.5; same parent",
        "essential_gain": "Gated validation E2E >= best existing policy +0.5pt; raw E2E/slots loss <=0.5pt; real intent loss <=2pt",
        "selection_data": ["val", "tune_oos"], "test_for_finalists_only": True,
        "holdout_access": False, "pi_automatic_promotion": False})
    for path in SOURCES + [config, "tests/test_vcm_validity.py", "docs/vcm-oos-discovery.md"]:
        target = STAGE / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    historical = {}
    for path in Path("experiments/vcm").glob("*/*"):
        if path.is_file() and path.suffix in (".json", ".yaml", ".pt"):
            historical[str(path)] = sha256(path)
    write(STAGE / "provenance.json", {"historical_sha256": historical, "python": sys.version,
        "platform": platform.platform(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True)})
    (STAGE / "environment_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    preflight(config)
    print(f"Prepared {config}; new architecture/export preflight passed")


def preflight(config):
    torch.set_num_threads(1)
    tax = load_taxonomy(config)
    model = ValidityModel(tax.n_way_head, {k: tax.slot_classes(k) for k in tax.slotted_intents})
    sample = torch.randn(2, 1, 80, 198)
    result = model(sample)
    loss = result["intent"].square().mean() + result["validity"].square().mean() + sum(v.square().mean() for v in result["slots"].values())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    class Export(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, x):
            r = self.model(x)
            return tuple([r["intent"]] + list(r["slots"].values()) + [r["validity"]])
    wrapper = Export(model).eval()
    names = ["intent_logits"] + ["slot_" + k for k in tax.slotted_intents] + ["validity_logit"]
    path = STAGE / "preflight.onnx"
    torch.onnx.export(wrapper, sample[:1], str(path), input_names=["mel"], output_names=names,
        dynamic_axes={k: {0: "batch"} for k in ["mel"] + names}, opset_version=17)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        x = torch.randn(batch, 1, 80, 198)
        with torch.no_grad():
            expected = wrapper(x)
        for a, b in zip(session.run(None, {"mel": x.numpy()}), expected):
            np.testing.assert_allclose(a, b.numpy(), rtol=1e-4, atol=1e-4)
    write(STAGE / "preflight.json", {"finite_backward": True, "onnx_parity": True,
        "n_params": sum(p.numel() for p in model.parameters()), "pi_verified": False})
    print("New architecture/export preflight passed")


if __name__ == "__main__":
    main()
