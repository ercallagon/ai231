"""Fetch only the selected official openWakeWord ONNX inference assets.

Pretrained assets are CC BY-NC-SA 4.0 per upstream, unlike Apache-licensed code.
No training assets are downloaded. Runtime never downloads models implicitly.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", required=True, choices=["hey_jarvis", "alexa"])
    parser.add_argument(
        "--directory", type=Path, default=Path("var/cache/wake/openwakeword-v0.5.1")
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path("var").resolve()):
        parser.error("assets must remain under var/")
    directory.mkdir(parents=True, exist_ok=True)
    files = ["melspectrogram.onnx", "embedding_model.onnx", f"{args.phrase}_v0.1.onnx"]
    manifest_path = directory / f"{args.phrase}-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for name in files:
        path = directory / name
        url = f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{name}"
        if path.exists():
            if (
                name not in previous
                or hashlib.sha256(path.read_bytes()).hexdigest() != previous[name]["sha256"]
            ):
                raise ValueError(f"unverified existing asset; preserve and review it: {path}")
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("asset exceeds 32 MiB")
            path.write_bytes(data)
        data = path.read_bytes()
        manifest[name] = {
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "license": "CC-BY-NC-SA-4.0; see upstream README",
            "checksum_origin": "computed on first HTTPS download, not vendor-signed",
        }
        # Persist after each asset so a failed later download can be resumed safely.
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Selected inference assets: {manifest_path}")


if __name__ == "__main__":
    main()
