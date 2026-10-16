#!/usr/bin/env python3
"""Audit the curated tracked submission with --tracked; legacy archive mode is separate."""
import argparse
import csv
import hashlib
import json
import re
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


def tracked_inventory(root, names=None):
    root = Path(root)
    if names is None:
        names = subprocess.check_output(["git", "ls-files", "-z", "--", "."], cwd=root).decode().split("\0")
    patterns = {
        "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
        "private_key": re.compile(rb"(?m)^-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----\r?$"),
        "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    }
    files, issues, groups = [], [], {}
    for name in sorted(set(n for n in names if n)):
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            issues.append(dict(path=name, kind="outside_project")); continue
        if not path.is_file():
            issues.append(dict(path=name, kind="tracked_file_missing")); continue
        size = path.stat().st_size
        files.append(dict(path=name, bytes=size, sha256=digest(path)))
        group = groups.setdefault(Path(name).parts[0], dict(files=0, bytes=0))
        group["files"] += 1; group["bytes"] += size
        if any(p in {"work", "__pycache__", ".venv", ".ssh", ".aws", "raw", "backups"} for p in Path(name).parts):
            issues.append(dict(path=name, kind="private_or_generated_directory"))
        if Path(name).suffix.lower() in {".pyc", ".mp3", ".parquet", ".arrow", ".tar", ".gz", ".zip", ".env", ".pem", ".key"}:
            issues.append(dict(path=name, kind="private_or_generated_file_type"))
        if size > 50 * 1024 * 1024:
            issues.append(dict(path=name, kind="large_file_requires_review"))
        content = path.read_bytes()
        for kind, pattern in patterns.items():
            if pattern.search(content): issues.append(dict(path=name, kind=kind))
    return dict(scope="curated_tracked_ME2_files_only", file_count=len(files),
        total_bytes=sum(x["bytes"] for x in files), groups=groups, files=files,
        issues=issues, automated_checks_pass=not issues,
        limitations=["Pattern scan is not proof that every possible secret or unlicensed asset is absent",
            "Git history and ignored local archives are preserved; only current tracked project paths are inventoried",
            "Historical recipes and optional candidate evidence are not current deployment defaults"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracked", action="store_true", help="Current curated submission inventory, not research-archive preparation")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT / "work"): parser.error("Intermediate inventory must stay under work/")
    out.mkdir(parents=True, exist_ok=False)
    if args.tracked:
        result = tracked_inventory(ROOT)
        with (out / "inventory.json").open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
        print(json.dumps({k:v for k,v in result.items() if k != "files"}, indent=2))
        return
    if not CHAMPION.exists() or not PROTOCOL.exists():
        parser.error("Historical mode needs the original research archive; use --tracked for this curated checkout")
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
