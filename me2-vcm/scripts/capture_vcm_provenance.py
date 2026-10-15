#!/usr/bin/env python3
"""Preserve corrected source/config/test text alongside machine-readable proof."""
import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from hey_swift.vcm.protocol import sha256

STAGE = Path("work/vcm_hf_alignment/hf_aligned_v1")
TARGET = STAGE / "source_snapshot"


def main(suffix=""):
    target = STAGE / ("source_snapshot" + suffix)
    target.mkdir(exist_ok=False)
    files = list(Path("src/hey_swift/vcm").glob("*.py"))
    files += list(Path("scripts").glob("*.py"))
    files += list(Path("tests").glob("*.py"))
    files += list(Path("docs").glob("*.md"))
    files += [Path(".gitignore"), Path("pyproject.toml"), Path("README.md"), Path("Makefile"), Path("run.sh")]
    files += list(Path("configs").glob("*.yaml"))
    fingerprints = {}
    for source in files:
        destination = target / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        fingerprints[str(source)] = sha256(source)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    environment_path = STAGE / ("environment_freeze" + suffix + ".txt")
    environment_path.write_text(freeze.stdout)
    protocol = json.loads(Path("data/vcm/protocols/hf_aligned_v1/protocol.json").read_text())
    historical_mismatches = []
    for name, digest in json.loads(Path("data/vcm/protocols/hf_aligned_v1/audit.json").read_text())["legacy_manifest_sha256"].items():
        if sha256(f"data/vcm/manifests/{name}_19intent.csv") != digest:
            historical_mismatches.append(name)
    for record in json.loads(Path("data/vcm/protocols/hf_aligned_v1/audit.json").read_text())["historical_runs"]:
        for relative, digest in record["files"].items():
            path = Path(relative) if relative.startswith(("configs/", "work/")) else Path("experiments/vcm") / record["tag"] / relative
            if sha256(path) != digest:
                historical_mismatches.append(str(path))
    if historical_mismatches:
        raise ValueError(f"Historical artifact changes detected: {historical_mismatches}")
    proof = {"source_snapshot_sha256": fingerprints, "environment_sha256": sha256(environment_path), "historical_artifacts_unchanged": True, "legacy_manifest_sha256": json.loads(Path("data/vcm/protocols/hf_aligned_v1/audit.json").read_text())["legacy_manifest_sha256"], "protocol_sha256": sha256("data/vcm/protocols/hf_aligned_v1/protocol.json"), "git_provenance": protocol["git"]}
    (STAGE / ("preservation_proof" + suffix + ".json")).write_text(json.dumps(proof, indent=2))
    print(f"Captured {len(fingerprints)} source/config/test files; historical artifacts unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="")
    suffix = parser.parse_args().suffix
    if suffix and not suffix.replace("_", "").isalnum():
        parser.error("Suffix must be alphanumeric/underscore")
    main(suffix)
