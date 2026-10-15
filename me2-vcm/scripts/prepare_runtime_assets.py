#!/usr/bin/env python3
"""Stage explicitly licensed, hash-pinned runtime assets without overwriting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from urllib.parse import urlparse


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def validate_entries(entries):
    seen = set()
    for entry in entries:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts or not path.parts or entry["path"] in seen:
            raise ValueError("Unsafe or duplicate asset path")
        seen.add(entry["path"])
        sha = entry.get("sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Every asset requires an exact SHA-256: " + entry["path"])
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise ValueError("Positive pinned asset size required")
        if not entry.get("terms_reference") or not entry.get("use_authorized"):
            raise ValueError("Asset use terms must be reviewed before preparation: " + entry["path"])
        if bool(entry.get("local_file")) == bool(entry.get("url")):
            raise ValueError("Specify exactly one local_file or URL per asset")
        if entry.get("url"):
            url = urlparse(entry["url"])
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise ValueError("Network assets require HTTPS without embedded credentials")


def prepare(manifest, destination, allow_network=False):
    entries = json.loads(Path(manifest).read_text())["assets"]
    validate_entries(entries)
    if not entries: raise ValueError("Empty asset manifest does not establish inference readiness")
    if any(e.get("url") for e in entries) and not allow_network:
        raise ValueError("Network fetch requires explicit --allow-network")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    for entry in entries:
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            if entry.get("local_file"):
                with Path(entry["local_file"]).expanduser().open("rb") as source:
                    shutil.copyfileobj(source, output)
            else:
                with urllib.request.urlopen(entry["url"], timeout=30) as response:
                    if urlparse(response.geturl()).scheme != "https": raise ValueError("Non-HTTPS redirect refused")
                    total = 0
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > entry["bytes"]: raise ValueError("Download exceeds pinned size")
                        output.write(block)
        if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
            raise ValueError("Asset fingerprint mismatch; incomplete stage retained: " + entry["path"])
        receipts.append({"path": entry["path"], "sha256": entry["sha256"], "bytes": entry["bytes"],
                         "terms_reference": entry["terms_reference"]})
    with (destination / "ASSET_RECEIPT.json").open("x") as stream:
        json.dump({"assets": receipts, "deployed": False, "manifest_sha256": digest(manifest)}, stream, indent=2)
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Fresh staging directory; never current Pi install")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output, args.allow_network), indent=2))
