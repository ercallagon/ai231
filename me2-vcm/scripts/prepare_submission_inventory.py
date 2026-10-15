#!/usr/bin/env python3
"""Create a draft publication inventory; no copies, Git staging or evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "experiments/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42"
PROTOCOL = ROOT / "data/vcm/protocols/hf_grouped_v2b"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def candidate(path):
    path = Path(path)
    if any(part.startswith(".") or part in ("__pycache__", "assets") for part in path.parts): return False
    return path.suffix in (".py", ".md", ".yaml", ".toml", ".txt", ".sh", ".typed", ".example") and not (
        path.parts[0] == "scripts" and path.name.startswith("_"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT / "work"): parser.error("Intermediate inventory must stay under work/")
    out.mkdir(parents=True, exist_ok=False)
    candidates = [ROOT / name for name in ("README.md", "LICENSE", ".gitignore", "pyproject.toml",
        "requirements.txt", "requirements-pi.txt", "run.sh", "Makefile")]
    for directory in ("src", "tests", "scripts", "configs", "docs"):
        candidates.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and candidate(p.relative_to(ROOT)))
    rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p),
             "status": "candidate_requires_review_not_staged"} for p in sorted(set(candidates))]
    artifacts = [CHAMPION / name for name in ("best.pt", "train.log", "history.json", "config.yaml",
        "calibration.json", "metrics_test.json", "calibrated/metrics_test.json", "calibrated/model.onnx",
        "finetune_provenance.json", "reproducibility.json")]
    artifacts += [ROOT / "models/wakeword/wakeword_exp15_mined_lp.onnx", PROTOCOL / "protocol.json"]
    hashes = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in artifacts}
    stats = {}
    for split in ("train", "val", "challenge", "test", "holdout", "tune_oos"):
        path = PROTOCOL / "manifests" / (split + "_19intent.csv")
        with path.open(newline="") as stream: records = list(csv.DictReader(stream))
        voices = set(v for row in records for v in json.loads(row["speech_voices"]))
        stats[split] = {"rows": len(records), "hours": sum(int(r["n_samples"]) / int(r["sample_rate"]) for r in records) / 3600,
                        "declared_speech_identities": len(voices), "manifest_sha256": digest(path)}
    raw = json.loads((CHAMPION / "metrics_test.json").read_text())
    calibrated = json.loads((CHAMPION / "calibrated/metrics_test.json").read_text())
    result = json.loads((CHAMPION / "result.json").read_text())
    summary = {"reference": CHAMPION.name, "selected_not_final": True, "n_params": result["n_params"],
        "best_epoch": result["best_epoch"], "finetune_duration_s": result["duration_s"],
        "raw_test": {k: raw[k] for k in ("n", "intent_accuracy", "macro_f1", "e2e_accuracy", "slot_accuracy", "slot_n")},
        "deployed_gated_test": {k: calibrated["gated"][k] for k in ("threshold", "intent_accuracy", "e2e_accuracy")},
        "dataset": stats, "dataset_doi": "10.57967/hf/10723", "test_inference_run": False,
        "pi_measurements": None, "scope": "Existing reports and manifest metadata only; no fresh test/holdout predictions",
        "speaker_caveat": "Declared voices include synthetic references; not independently verified human speakers"}
    blockers = ["Public source contains private contact/device defaults: src/hey_swift/integration/config.py",
        "Wake dataset fetch/consent/license and archive-free reproduction unresolved",
        "Speech/music fetch and licensing; packager currently reads ignored work/integration assets",
        "run.sh is historical A10, not current champion reconstruction",
        "Matching public runtime config/labels and trained-model retrieval not yet published",
        "Clean-checkout installation/reproduction and official Pi benchmark still pending",
        "Final checkpoint/model distribution terms and final candidate selection pending"]
    receipt = {"candidate_files": rows, "baseline_artifacts": hashes, "blockers": blockers,
        "git_root": subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "target_repo": "https://github.com/ercallagon/ai231", "target_subdirectory": "me2-vcm",
        "publication_ready": False, "staged_or_pushed": False,
        "excluded": ["work", "raw audio/parquet", "feature caches", "credentials/account state",
                     "historical experiments", "deployment backup trees", "unrelated parent files"],
        "note": "This is not a secret scanner or approval to publish; inspect every candidate and selected artifact"}
    for name, payload in (("inventory.json", receipt), ("deck_metrics.json", summary)):
        with (out / name).open("x") as stream: json.dump(payload, stream, indent=2); stream.write("\n")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "candidate_files": len(rows),
                      "blockers": blockers, "publication_ready": False}, indent=2))


if __name__ == "__main__": main()
