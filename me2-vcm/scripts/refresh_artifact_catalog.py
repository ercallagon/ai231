#!/usr/bin/env python3
"""Check the release catalog; refresh reviewed source/docs without changing frozen artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ("models/", "configs/", "data/", "experiments/", "logs/", "audio/")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or any(
        part.startswith(".") and part not in (".gitignore", ".gitattributes")
        for part in relative.parts
    ) or any(part in {"work", "raw", "__pycache__", "backups"} for part in relative.parts):
        raise ValueError("Not a release path: " + name)
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or unsafe release file: " + name)
    return path


def refresh_catalog(root, *, refresh=False, include=(), pi5_status=None):
    root = Path(root).resolve()
    payload = json.loads((root / "ARTIFACTS.json").read_text())
    hashes = payload["sha256"]
    changed = []
    for name, expected in tuple(hashes.items()):
        actual = digest(release_path(root, name))
        if actual != expected:
            if name.startswith(PROTECTED) or (
                name.startswith("reports/") and Path(name).suffix in {".json", ".csv"}
            ):
                raise ValueError("Frozen artifact changed; separate review required: " + name)
            changed.append(name)
            if refresh:
                hashes[name] = actual
    if not refresh and (include or pi5_status is not None):
        raise ValueError("--include/--pi5-status require --refresh")
    if refresh:
        for name in include:
            if name == "ARTIFACTS.json" or name.startswith(PROTECTED):
                raise ValueError("Cannot add protected/self catalog entry: " + name)
            hashes[name] = digest(release_path(root, name))
        if pi5_status is not None:
            payload["pi5_verification"] = pi5_status
        payload["sha256"] = dict(sorted(hashes.items()))
    return payload, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Write reviewed mutable-file hashes after verifying every frozen artifact")
    parser.add_argument("--include", nargs="*", default=[],
                        help="Explicit additional source/test paths; no work/ or frozen artifacts")
    parser.add_argument("--pi5-status", help="Reviewed measurement-scope text; requires --refresh")
    args = parser.parse_args()
    payload, changed = refresh_catalog(ROOT, refresh=args.refresh, include=args.include,
                                      pi5_status=args.pi5_status)
    if args.refresh:
        with (ROOT / "ARTIFACTS.json").open("w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    print(json.dumps(dict(mode="refresh" if args.refresh else "check",
                         entries=len(payload["sha256"]), changed=changed,
                         included=args.include), indent=2))
    if changed and not args.refresh:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
