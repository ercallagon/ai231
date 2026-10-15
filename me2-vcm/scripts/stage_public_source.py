#!/usr/bin/env python3
"""Prepare an uncommitted source review copy; preserve the live source/defaults."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_NAMES = {"DEFAULT_CALL_NUMBER", "DEFAULT_MESSAGE_NUMBER", "DEFAULT_PHONE_MAC"}
SKIP = {"__pycache__", ".pytest_cache", ".git", ".aws", ".agents", ".codex", "assets"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_config(source):
    tree = ast.parse(source)
    constants = {}
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in PRIVATE_NAMES:
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str) or node.lineno != node.end_lineno:
                    raise ValueError("Unexpected private-default declaration; inspect rather than guessing")
                constants[name] = node.value.value
                lines[node.lineno - 1] = name + ' = ""  # Public install requires private environment configuration.\n'
    if set(constants) != PRIVATE_NAMES: raise ValueError("Missing private-default declarations")
    return "".join(lines), constants


def stage(destination):
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ROOT / "work"):
        raise ValueError("Review snapshot must remain under work/")
    destination.mkdir(parents=True, exist_ok=False)
    files = [ROOT / name for name in ("README.md", "LICENSE", "pyproject.toml", ".gitignore",
        "requirements.txt", "requirements-pi.txt", "run.sh", "Makefile")]
    suffixes = {".py", ".md", ".txt", ".yaml", ".toml", ".sh", ".typed", ".example"}
    for folder in ("src", "scripts", "tests", "configs", "docs"):
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file()
            and not any(part in SKIP or part.startswith(".") for part in path.relative_to(ROOT).parts)
            and path.suffix in suffixes
            and (folder != "scripts" or not path.name.startswith("_")))
    # Metadata is needed for the established protocol. No audio/features/weights.
    metadata = [ROOT / "data/README.md", ROOT / "models/README.md", ROOT / "deploy/rpi/README.md"]
    for folder in ("data/vcm/manifests", "data/wakeword/manifests", "data/vcm/protocols/hf_grouped_v2b"):
        metadata.extend(path for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in {".csv", ".json", ".md"})
    files.extend(metadata)
    private_path = ROOT / "src/hey_swift/integration/config.py"
    sanitized, private = public_config(private_path.read_text())
    record = {}; remaining = []
    for source in sorted(set(files)):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source == private_path:
            target.write_text(sanitized)
        else:
            shutil.copy2(source, target)
        text = target.read_text()
        if any(value and value in text for value in private.values()): remaining.append(str(relative))
        record[str(relative)] = {"source_sha256": digest(source), "snapshot_sha256": digest(target),
                                "transformed": source == private_path}
    receipt = {"source_files": record, "private_defaults_cleared": sorted(PRIVATE_NAMES),
        "remaining_private_value_paths": remaining, "no_credentials_or_accounts_copied": True,
        "no_model_or_audio_copied": True, "git_staged_or_pushed": False, "publication_approved": False,
        "privacy_scope": "Exact known contact/device defaults only; not a comprehensive secret/consent scan",
        "source_preserved": digest(private_path) == record[str(private_path.relative_to(ROOT))]["source_sha256"]}
    with (destination.parent / (destination.name + "_receipt.json")).open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
    if remaining: raise ValueError("Known private values remain in review snapshot: " + ", ".join(remaining))
    print(json.dumps({"snapshot": str(destination.relative_to(ROOT)), "files": len(record),
                      "private_defaults_cleared": sorted(PRIVATE_NAMES), "publication_approved": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    stage(parser.parse_args().output)
