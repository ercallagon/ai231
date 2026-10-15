#!/usr/bin/env python3
"""Create or seal the isolated source-balanced BC8 experiment."""
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
from vcm_bc8_source_finetune import CONFIG

STAGE = Path("work/vcm_sourceft")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-readiness", action="store_true")
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.seal_readiness:
        from validate_vcm_protocol import validate
        from vcm_bc8_trial import check_candidate
        validate(CONFIG)
        check_candidate(CONFIG)
        suites = list(ET.parse(STAGE / "regression.xml").getroot().iter("testsuite"))
        if not suites or sum(int(s.attrib.get("tests", 0)) for s in suites) < 70:
            raise ValueError("Missing complete regression results")
        if any(int(s.attrib.get(k, 0)) for s in suites for k in ("errors", "failures")):
            raise ValueError("Regression failures")
        info = {"passed": True, "config_sha256": sha256(CONFIG),
                "regression_xml_sha256": sha256(STAGE / "regression.xml"),
                "tests": sum(int(s.attrib["tests"]) for s in suites),
                "skipped": sum(int(s.attrib.get("skipped", 0)) for s in suites),
                "architecture_unchanged": True, "pi_verified": False}
        with (STAGE / "readiness.json").open("x") as stream:
            json.dump(info, stream, indent=2)
        print(json.dumps(info, indent=2))
        return
    path = Path(CONFIG)
    if path.exists():
        raise FileExistsError(path)
    cfg = yaml.safe_load(Path("configs/vcm_hf_grouped_v2b_bc8.yaml").read_text())
    cfg["training"].update(lr=0.0003, epochs=12, warmup_epochs=1, early_stop_patience=5)
    cfg["experiment"].update(entrypoint="scripts/vcm_bc8_source_finetune.py",
        source_balance_target=0.30, parent="run_042_hf_grouped_v2b_bc8_s42",
        note="Warm-start BC8; reset optimizer; source-weighted intent CE only. No new audio or architecture changes.")
    sources = cfg["experiment"]["implementation_sha256"]
    for name in ("scripts/vcm_bc8_source_finetune.py", "scripts/run_vcm_sourceft_queue.py",
                 "scripts/prepare_vcm_sourceft_trial.py"):
        sources[name] = sha256(name)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name in list(sources) + [CONFIG, "tests/test_vcm_sourceft.py", "docs/vcm-source-finetune.md"]:
        target = STAGE / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    record = {"config_sha256": sha256(CONFIG), "source_sha256": sources,
              "python": sys.version, "platform": platform.platform(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "parent_config_sha256": sha256("configs/vcm_hf_grouped_v2b_bc8.yaml")}
    (STAGE / "provenance.json").write_text(json.dumps(record, indent=2))
    (STAGE / "environment_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    print(f"Prepared {CONFIG}")


if __name__ == "__main__":
    main()
