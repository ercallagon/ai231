#!/usr/bin/env python3
"""Prepare fresh config/provenance or seal validated BC8 retraining readiness."""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from hey_swift.vcm.protocol import sha256

STAGE = Path("work/vcm_grouped_v2")
CONFIG = Path("configs/vcm_hf_grouped_v2b_bc8.yaml")
SOURCES = ["src/hey_swift/vcm/broadcast.py", "src/hey_swift/vcm/train_repro.py",
           "src/hey_swift/vcm/augmentation.py", "scripts/vcm_bc8_trial.py",
           "scripts/vcm_grouped_bc8_trial.py", "scripts/run_vcm_grouped_bc8_queue.py",
           "scripts/prepare_vcm_grouped_protocol.py", "scripts/prepare_vcm_grouped_trial.py",
           "src/hey_swift/vcm/THIRD_PARTY_BCRESNET_LICENSE.txt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-readiness", action="store_true")
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.seal_readiness:
        from validate_vcm_protocol import validate
        validate(str(CONFIG))
        from vcm_bc8_trial import check_candidate
        check_candidate(str(CONFIG))
        suites = list(ET.parse(STAGE / "regression.xml").getroot().iter("testsuite"))
        if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("errors", "failures")):
            raise ValueError("Regression suite did not pass")
        checks = json.loads(Path("work/vcm_next_bc8/preflight.json").read_text())
        previous = yaml.safe_load(Path("configs/vcm_hf_aligned_v1_bc8.yaml").read_text())
        if previous["experiment"]["implementation_sha256"]["src/hey_swift/vcm/broadcast.py"] != sha256("src/hey_swift/vcm/broadcast.py"):
            raise ValueError("Architecture changed; ONNX preflight must be repeated")
        if not checks["onnx_shapes_and_parity_passed"] or not checks["finite_backward_passed"]:
            raise ValueError("Numerical/export preflight not passed")
        info = {"passed": True, "config_sha256": sha256(CONFIG),
                "protocol_sha256": sha256("data/vcm/protocols/hf_grouped_v2b/protocol.json"),
                "regression_xml_sha256": sha256(STAGE / "regression.xml"),
                "tests": sum(int(s.attrib["tests"]) for s in suites),
                "skipped": sum(int(s.attrib.get("skipped", 0)) for s in suites),
                "architecture_export_preflight_reused": "Unchanged BC8 architecture; trained export checked again after retraining"}
        with (STAGE / "readiness.json").open("x") as stream:
            json.dump(info, stream, indent=2)
        print(json.dumps(info, indent=2))
        return
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    cfg = yaml.safe_load(Path("configs/vcm_hf_aligned_v1_bc8.yaml").read_text())
    cfg["data"].update(protocol="data/vcm/protocols/hf_grouped_v2b/protocol.json",
                       manifests="data/vcm/protocols/hf_grouped_v2b/manifests",
                       feature_cache="data/vcm/processed/hf_grouped_v2b")
    cfg["experiment"].update(entrypoint="scripts/vcm_grouped_bc8_trial.py",
        trainer="hey_swift.vcm.train_repro", implementation_sha256={p: sha256(p) for p in SOURCES},
        note="Fresh grouped validation; untouched old validation reserved as challenge. Use grouped trial entrypoint only.")
    CONFIG.write_text("# Versioned grouped BC8 retraining; use scripts/vcm_grouped_bc8_trial.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
    snapshot = STAGE / "source_snapshot_final"
    for path in SOURCES + [str(CONFIG), "tests/test_vcm_grouped.py", "docs/vcm-grouped-v2.md"]:
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    historical = {}
    for root in (Path("experiments/vcm"), Path("data/vcm/protocols/hf_aligned_v1")):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in (".pt", ".json", ".csv", ".yaml"):
                historical[str(path)] = sha256(path)
    record = {"source_sha256": cfg["experiment"]["implementation_sha256"],
              "config_sha256": sha256(CONFIG), "python": sys.version, "platform": platform.platform(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "historical_artifact_sha256": historical}
    (STAGE / "provenance_final.json").write_text(json.dumps(record, indent=2))
    (STAGE / "environment_freeze_final.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    print(f"Prepared {CONFIG}; preserved fingerprints of {len(historical)} historical artifacts")


if __name__ == "__main__":
    main()
