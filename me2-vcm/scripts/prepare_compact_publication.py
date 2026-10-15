#!/usr/bin/env python3
"""Build a compact Git review payload and a blocked reference artifact catalog."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "run_043_hf_grouped_v2b_bc8_sourceft_s42"
SNAPSHOT = ROOT / "work/github_readiness_20261012_step2/public_source_v3"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def git_candidate(path):
    path = Path(path)
    if any(part.startswith(".") or part in {"__pycache__", "assets", "state", "logs", "backups"} for part in path.parts):
        return path.name == ".gitignore" and len(path.parts) == 1
    if path.name.startswith("auto_queue") or path.suffix not in {".py", ".md", ".txt", ".yaml", ".toml", ".sh", ".typed", ".example"}:
        return False
    if path.parts[0] in {"src", "tests", "scripts", "configs", "docs"}: return True
    return str(path) in {"README.md", "LICENSE", "pyproject.toml", "requirements.txt", "requirements-pi.txt",
                        "run.sh", "Makefile", "data/README.md", "models/README.md", "deploy/rpi/README.md"}


def metadata_files(root):
    files = []
    for directory in ("data/vcm/protocols/hf_grouped_v2b", "data/wakeword/manifests"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".md"})
    # The historical v1 preparation reads these exact base manifests, not OOS injections.
    files.extend((root / "data/vcm/manifests").glob("*_19intent.csv"))
    files.append(root / "data/vcm/manifests/splits.json")
    return sorted(set(files))


def metadata_archive(root, output, files):
    checksums = {}
    with Path(output).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(files):
                    relative = path.relative_to(root)
                    if path.suffix not in {".csv", ".json", ".md"} or ".." in relative.parts:
                        raise ValueError("Only safe dataset metadata allowed")
                    checksums[str(relative)] = digest(path)
                    info = tarfile.TarInfo(str(relative)); info.size = path.stat().st_size
                    info.mode = 0o644; info.mtime = 0
                    with path.open("rb") as stream: archive.addfile(info, stream)
                payload = (json.dumps(checksums, sort_keys=True, indent=2) + "\n").encode()
                info = tarfile.TarInfo("METADATA_SHA256.json"); info.size = len(payload); info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
    return checksums


def artifact(path, destination, role, *, terms, contract=None):
    return {"path": destination, "role": role, "source_path": str(path.relative_to(ROOT)),
            "sha256": digest(path), "bytes": path.stat().st_size, "url": None,
            "use_authorized": False, "terms_reference": terms,
            "download_ready": False, "blocker": "Release URL and distribution/use review pending",
            "contract": contract or {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", default=str(SNAPSHOT), help="Reviewed source snapshot with its matching receipt")
    args = parser.parse_args()
    snapshot = Path(args.snapshot).resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / "work"): parser.error("Intermediate publication work must remain under work/")
    if not snapshot.is_relative_to(ROOT / "work") or not snapshot.is_dir():
        raise ValueError("Snapshot must be an existing reviewed directory under work/")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source_tree/me2-vcm"
    source.mkdir(parents=True)
    receipt = json.loads((snapshot.parent / (snapshot.name + "_receipt.json")).read_text())
    metadata = metadata_files(snapshot)
    metadata_paths = {str(path.relative_to(snapshot)) for path in metadata}
    included, excluded = [], []
    for relative, proof in sorted(receipt["source_files"].items()):
        path = snapshot / relative
        if digest(path) != proof["snapshot_sha256"]: raise ValueError("Validated snapshot changed: " + relative)
        if git_candidate(relative) or relative in {"LICENSE", "Makefile"}:
            target = source / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if relative == "README.md":
                text = target.read_text()
                heading, rest = text.split("\n", 1)
                target.write_text(heading + "\n\n> Compact publication proposal: dataset metadata is a separate hash-pinned\n"
                    "> release artifact, not included in this Git tree. Restore it before retraining.\n"
                    "> See [compact publication](docs/compact-publication.md). Release URLs and\n"
                    "> complete one-command reproduction remain pending.\n" + rest)
            if relative == "docs/public-reproduction.md":
                text = target.read_text()
                target.write_text(text.replace("After the reviewed source and grouped-v2b metadata are published, from",
                    "For the compact Git layout, first restore the separately published, hash-verified\n"
                    "metadata archive as described in compact-publication.md. This source tree does\n"
                    "not contain those manifests. After source and metadata are available, from"))
            included.append({"path": "me2-vcm/" + relative, "sha256": digest(target), "bytes": target.stat().st_size})
        else:
            excluded.append({"path": relative, "bytes": path.stat().st_size,
                             "disposition": "metadata release archive" if relative in metadata_paths else "historical material omitted; preserved locally"})
    # Include this new preparation tool/tests without changing the validated baseline snapshot.
    for relative in ("scripts/prepare_compact_publication.py", "tests/test_compact_publication.py", "docs/compact-publication.md"):
        original = ROOT / relative; target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, target)
        included.append({"path": "me2-vcm/" + relative, "sha256": digest(target), "bytes": target.stat().st_size})
    archive_path = output / "dataset_metadata_hf_grouped_v2b_reference_v1.tar.gz"
    metadata_hashes = metadata_archive(snapshot, archive_path, metadata)
    exp = ROOT / "experiments/vcm" / REFERENCE
    bundle = ROOT / "work/rpi_transfer_run043_runtime_v5_20261010_04"
    import yaml
    runtime_config = bundle / "configs/vcm_run_043_calibrated.yaml"
    labels = bundle / "models/vcm" / (REFERENCE + ".labels.json")
    cfg, mapping = yaml.safe_load(runtime_config.read_text()), json.loads(labels.read_text())
    if cfg["schema"]["slot_sets"] != mapping["slot_sets"] or mapping["n_way_head"] != 20:
        raise ValueError("Baseline labels/config disagree")
    calibration = json.loads((exp / "calibration.json").read_text())
    if abs(cfg["oos"]["confidence_threshold"] - calibration["chosen"]["threshold"]) > 1e-12:
        raise ValueError("Baseline calibration/config threshold mismatch")
    contract = {"model": REFERENCE, "input_shape": ["batch", 1, 80, 198], "audio": cfg["audio"],
        "n_way_head": 20, "intents": mapping["intents"], "slot_sets": mapping["slot_sets"],
        "slot_outputs": mapping["slot_outputs"], "confidence_threshold": cfg["oos"]["confidence_threshold"],
        "temperature_baked_once": calibration["temperature"], "runtime_temperature_division": False}
    terms = "docs/dataset.md; mixed dataset terms; trained-model distribution review pending"
    assets = [artifact(exp / "calibrated/model.onnx", f"models/vcm/{REFERENCE}.onnx", "vcm_runtime", terms=terms, contract=contract),
        artifact(labels, f"models/vcm/{REFERENCE}.labels.json", "vcm_labels", terms="Reviewed project-generated label mapping"),
        artifact(runtime_config, "configs/vcm_run_043_calibrated.yaml", "vcm_runtime_config", terms="Project source/configuration"),
        artifact(ROOT / "models/wakeword/wakeword_exp15_mined_lp.onnx", "models/wakeword/wakeword_exp15_mined_lp.onnx",
                 "wake_runtime", terms="Wake recording provenance/consent and model distribution review pending",
                 contract={"input_shape": ["batch", 80, 100], "output_shape": ["batch", 2], "wake_output_index": 0,
                           "threshold": .30, "sustain": 2, "cooldown_seconds": 2, "window_seconds": 1}),
        artifact(ROOT / "configs/wakeword.yaml", "configs/wakeword.yaml", "wake_config", terms="Project source/configuration"),
        artifact(ROOT / "configs/hardware.yaml", "configs/hardware.yaml", "hardware_config", terms="Project source/configuration",
                 contract={"rgb_bcm": [17, 27, 22], "tm1637_bcm": [23, 24]}),
        artifact(exp / "best.pt", f"models/checkpoints/{REFERENCE}.best.pt", "final_checkpoint_commit_candidate", terms=terms)]
    for name in ("train.log", "history.json", "config.yaml", "calibration.json", "metrics_test.json", "metrics_val.json",
                 "calibrated/metrics_test.json", "finetune_provenance.json"):
        assets.append(artifact(exp / name, f"reports/reference_run043/{name}", "selected_training_evidence", terms="Review logs/provenance for private paths/identifiers before publication"))
    assets.append({"path": "release_metadata/" + archive_path.name, "role": "dataset_metadata",
        "sha256": digest(archive_path), "bytes": archive_path.stat().st_size, "url": None,
        "use_authorized": False, "terms_reference": "VCM DOI 10.57967/hf/10723 plus separate wake metadata rights review",
        "download_ready": False, "blocker": "Metadata privacy/terms review and immutable release URL pending"})
    manifest = {"schema_version": 1, "version": "run043_reference_v1", "role": "preserved_reference_not_final_release",
        "selected_final_champion": None, "publication_approved": False, "assets": assets,
        "required_inference_roles": ["vcm_runtime", "vcm_labels", "vcm_runtime_config", "wake_runtime", "wake_config", "hardware_config"],
        "unresolved_assets": ["Cue rights or reproducible replacement; preserve locally amplified Pi cue",
                              "Amy voice+JSON immutable retrieval and underlying terms review",
                              "User-owned/licensed offline MP3s; do not redistribute personal demo tracks",
                              "Wake raw audio fetch/generation and consent"],
        "invariants": ["No champion/model/threshold/wiring changes", "New candidate requires a new catalog version",
                       "No external pretrained classifier initialization", "Catalog is not permission to download or deploy"]}
    manifest_path = source / "models/releases/run043_reference_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as stream: json.dump(manifest, stream, indent=2); stream.write("\n")
    included.append({"path": "me2-vcm/models/releases/run043_reference_v1.json", "sha256": digest(manifest_path), "bytes": manifest_path.stat().st_size})
    proposal = {"publication_approved": False, "git_staged_or_pushed": False, "candidate_files": included,
        "excluded_from_git": excluded, "source_bytes": sum(row["bytes"] for row in included),
        "previous_snapshot_bytes": sum((snapshot / p).stat().st_size for p in receipt["source_files"]),
        "metadata_archive": {"path": archive_path.name, "sha256": digest(archive_path), "bytes": archive_path.stat().st_size,
                             "members": metadata_hashes}, "metadata_required_before_retraining": True,
        "reproduction_gate": "Fetch/extract hash-verified metadata before raw-data/cache validation; immutable URL and full clean-checkout execution still pending",
        "retained_checkpoint_policy": "Only a selected, reviewed final checkpoint is an explicit Git exception; none copied or committed now",
        "privacy_scope": "Snapshot has known contact defaults cleared, but complete privacy/license review is still pending"}
    with (output / "PUBLICATION_PROPOSAL.json").open("x") as stream: json.dump(proposal, stream, indent=2); stream.write("\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "git_candidate_files": len(included),
        "git_candidate_bytes": proposal["source_bytes"], "metadata_archive_bytes": archive_path.stat().st_size,
        "publication_approved": False, "final_champion": None}, indent=2))


if __name__ == "__main__": main()
