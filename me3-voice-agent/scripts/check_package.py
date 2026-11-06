"""Inspect built distributions without installing or executing their contents."""

import argparse
import json
import tarfile
import zipfile
from pathlib import Path


def check(directory):
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(sources[0]) as archive:
        source_names = {
            str(Path(name).relative_to(Path(name).parts[0])) for name in archive.getnames()
        }
    for names in (wheel_names, source_names):
        for name in names:
            parts = Path(name).parts
            if any(p in {"var", ".git", ".venv", "__pycache__"} for p in parts):
                raise ValueError("Runtime state leaked into distribution")
            if Path(name).name.startswith(".env") and Path(name).name != ".env.example":
                raise ValueError("Private environment file in distribution")
            if Path(name).suffix.lower() in {".wav", ".mp3", ".sqlite3", ".pem", ".key"}:
                raise ValueError("Private/runtime file type in distribution")
    for name in ("dashboard.html", "dashboard.css", "dashboard.js"):
        if f"me3_voice/ui/{name}" not in wheel_names:
            raise ValueError(f"Missing bundled UI: {name}")
    for name in (
        "uv.lock",
        "README.md",
        ".env.example",
        "scripts/setup.sh",
        "scripts/swift.sh",
        "deploy/rpi/README.md",
        "configs/demo-backends.example.json",
        "src/me3_voice/communications.py",
        "src/me3_voice/devices.py",
    ):
        if name not in source_names:
            raise ValueError(f"Missing installation source: {name}")
    return {"status": "pass", "wheel_files": len(wheel_names), "source_files": len(source_names)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory), indent=2))
