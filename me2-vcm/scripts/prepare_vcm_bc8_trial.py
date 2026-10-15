#!/usr/bin/env python3
"""Generate immutable BC8 config and inventory every available VCM run."""
import json
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from hey_swift.vcm.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    stage = Path("work/vcm_next_bc8")
    stage.mkdir(exist_ok=True)
    config = Path("configs/vcm_hf_aligned_v1_bc8.yaml")
    if config.exists() and not args.inventory_only:
        raise FileExistsError(config)
    baseline = yaml.safe_load(Path("configs/vcm_hf_aligned_v1_a6.yaml").read_text())
    baseline["training"]["arch"] = "a4"
    baseline["experiment"] = {
        "architecture": "bc_resnet_broadcast8", "factory_routing_key": "a4",
        "entrypoint": "scripts/vcm_bc8_trial.py",
        "note": "Isolated factory bridge; a4 is not the historical A4 architecture. Do not run this config through the generic CLI.",
        "implementation_sha256": {p: sha256(p) for p in ["src/hey_swift/vcm/broadcast.py", "scripts/vcm_bc8_trial.py", "scripts/run_vcm_bc8_queue.py", "src/hey_swift/vcm/THIRD_PARTY_BCRESNET_LICENSE.txt"]},
    }
    if not args.inventory_only:
        config.write_text("# Use scripts/vcm_bc8_trial.py; do not use the generic trainer CLI.\n" + yaml.safe_dump(baseline, sort_keys=False))
    inventory = []
    for directory in sorted(Path("experiments/vcm").glob("run_*")):
        record = {"tag": directory.name, "state": "completed" if (directory / "result.json").exists() else "incomplete", "fingerprints": {}}
        for path in directory.iterdir():
            if path.suffix in (".json", ".yaml", ".log", ".pt"):
                record["fingerprints"][path.name] = sha256(path)
            if path.suffix == ".json":
                try:
                    record[path.name] = json.loads(path.read_text())
                except ValueError:
                    record.setdefault("unreadable_json", []).append(path.name)
        config_path = record.get("reproducibility.json", {}).get("config")
        if config_path and Path(config_path).exists():
            record["effective_config_current"] = yaml.safe_load(Path(config_path).read_text())
            record["config_sha256_current"] = sha256(config_path)
        inventory.append(record)
    archives = []
    archive_root = Path("../archive-me2")
    patterns = ["me2-integrated/results2/*.json", "me2-integrated/models2/*/result.json", "me2-vcm-old/results/*.json", "me2-vcm-old/experiments/**/*.json"]
    for pattern in patterns:
        for path in sorted(archive_root.glob(pattern)):
            try:
                content = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                continue
            archives.append({"path": str(path), "sha256": sha256(path), "content": content})
    (stage / "all_run_inventory.json").write_text(json.dumps({"current_runs": inventory, "archived_artifacts": archives}, indent=2))
    lines = ["# All available VCM run inventory", "", "Snapshot; incomplete checkpoints/logs may continue changing.", "", "| Run | Arch | Params | Best val E2E/score | Test intent | Test E2E | Protocol |", "|---|---|---:|---:|---:|---:|---|"]
    for record in inventory:
        result = record.get("result.json", {})
        metric = record.get("metrics_test.json", {})
        repro = record.get("reproducibility.json", {})
        lines.append("| " + " | ".join(str(x) for x in [record["tag"], result.get("arch", "pending"), result.get("n_params", "-"), result.get("best_val_score", "-"), metric.get("intent_accuracy", "-"), metric.get("e2e_accuracy", "-"), "HF aligned" if repro.get("protocol_sha256") else "legacy / effective overrides require audit"]) + " |")
    lines += ["", f"Archived JSON artifacts reviewed: {len(archives)}.", "", "Legacy accuracy is not clean unseen-voice evidence. Intent-only runs and slot-enabled runs have different score semantics. Current config files are not necessarily historical snapshots."]
    (stage / "ALL_RUNS.md").write_text("\n".join(lines) + "\n")
    provenance = stage / "source_provenance.json"
    if not provenance.exists():
        paths = [str(config), "src/hey_swift/vcm/broadcast.py",
                 "src/hey_swift/vcm/THIRD_PARTY_BCRESNET_LICENSE.txt",
                 "scripts/vcm_bc8_trial.py", "scripts/run_vcm_bc8_queue.py",
                 "scripts/prepare_vcm_bc8_trial.py", "tests/test_vcm_broadcast.py",
                 "docs/vcm-bc8-trial.md"]
        for path in paths:
            destination = stage / "source_snapshot" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        def capture(command):
            return subprocess.check_output(command, text=True).strip()
        record = {"source_sha256": {p: sha256(p) for p in paths},
                  "protocol_sha256": sha256("data/vcm/protocols/hf_aligned_v1/protocol.json"),
                  "git_head": capture(["git", "rev-parse", "HEAD"]),
                  "git_status": capture(["git", "status", "--short"]),
                  "python": sys.version, "platform": platform.platform(),
                  "historical_inventory_sha256": sha256(stage / "all_run_inventory.json")}
        provenance.write_text(json.dumps(record, indent=2))
        (stage / "environment_freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]) + "\n")
    print(f"Prepared {config}; inventoried {len(inventory)} current runs and {len(archives)} archived JSON artifacts")


if __name__ == "__main__":
    main()
