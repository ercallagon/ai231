"""Read-only inventory of saved project audio; never sends audio to a provider."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path

AUDIO = {".wav", ".webm", ".ogg", ".flac", ".mp3", ".m4a", ".pcm"}


def inspect_audio(path):
    row = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as wav:
                row["audio"] = {
                    "sample_rate": wav.getframerate(),
                    "channels": wav.getnchannels(),
                    "sample_width_bytes": wav.getsampwidth(),
                    "duration_seconds": wav.getnframes() / wav.getframerate(),
                }
        elif path.suffix.lower() != ".pcm" and shutil.which("ffprobe"):
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode:
                row["probe_error"] = "ffprobe rejected file"
            else:
                data = json.loads(result.stdout)
                row["audio"] = dict(data.get("streams", [{}])[0])
                duration = data.get("format", {}).get("duration")
                if duration is not None:
                    row["audio"]["duration_seconds"] = float(duration)
                else:
                    # MediaRecorder WebM often omits container duration.
                    packets = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "a:0",
                            "-show_packets",
                            "-show_entries",
                            "packet=pts_time,duration_time",
                            "-of",
                            "json",
                            str(path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if packets.returncode == 0:
                        timing = json.loads(packets.stdout).get("packets", [])
                        spans = [
                            (
                                float(t["pts_time"]),
                                float(t["pts_time"]) + float(t.get("duration_time", 0)),
                            )
                            for t in timing
                            if "pts_time" in t
                        ]
                        if spans:
                            row["audio"]["duration_seconds"] = max(t[1] for t in spans) - min(
                                t[0] for t in spans
                            )
                            row["audio"]["duration_source"] = (
                                "packet PTS extent; not decoded sample count"
                            )
        else:
            row["probe_unavailable"] = (
                "raw PCM requires explicit format"
                if path.suffix.lower() == ".pcm"
                else "ffprobe absent; no installation attempted"
            )
    except (OSError, ValueError, wave.Error, subprocess.TimeoutExpired) as exc:
        row["probe_error"] = type(exc).__name__
    return row


def inventory(roots):
    paths = sorted(
        {p for root in roots for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO}
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(inspect_audio, paths))
    by_path = {str(Path(row["path"]).resolve()): row for row in files}
    manifests = []
    for root in roots:
        for path in sorted(root.rglob("manifest.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or data.get("kind") != "swift-guided-recordings":
                continue
            cases = {c["id"]: c for c in data.get("cases", [])}
            item = {
                "path": str(path),
                "suite": data.get("suite"),
                "consent": data.get("consent"),
                "takes": [],
            }
            for take in data.get("takes", []):
                original = path.parent / take["file"]
                row = by_path.get(str(original.resolve()))
                case = cases.get(take["case_id"], {})
                check = {
                    "case_id": take["case_id"],
                    "file": str(original),
                    "present": row is not None,
                    "complete_flag": take.get("complete"),
                    "hash_matches": row["sha256"] == take.get("sha256") if row else False,
                    "expected_text_available": "expected_text" in case,
                    "expected_text_is_human_verified_transcript": False,
                }
                item["takes"].append(check)
                if row:
                    row["guided"] = {
                        "suite": item["suite"],
                        "case_id": take["case_id"],
                        "capture_settings": take.get("settings", data.get("capture_settings")),
                        "consent": item["consent"],
                        "manifest_integrity": check,
                    }
            manifests.append(item)
    groups = collections.defaultdict(list)
    for row in files:
        groups[row["sha256"]].append(row["path"])
    guided = [
        r for r in files if "guided" in r and Path(r["path"]).is_relative_to("var/recordings")
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "metadata and content hashes; no listening or inference",
        "roots": [str(p) for p in roots],
        "file_count": len(files),
        "unique_content_hashes": len(groups),
        "byte_identical_groups": [v for v in groups.values() if len(v) > 1],
        "guided_original_count": len(guided),
        "guided_original_seconds": sum(
            r.get("audio", {}).get("duration_seconds", 0) for r in guided
        ),
        "guided_original_suites": dict(collections.Counter(r["guided"]["suite"] for r in guided)),
        "manifest_count": len(manifests),
        "manifests": manifests,
        "files": files,
        "limitations": [
            "Expected prompts are not verified actual transcripts",
            "Metadata does not prove intelligibility, silence or physical Pi provenance",
            "Byte-distinct encodes may contain the same utterance; unique hashes are not unique human takes",
            "No provider upload permission is inferred",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("var").resolve()):
        parser.error("output must be under project var/")
    data = inventory([Path("var/recordings"), Path("var/experiments"), Path("var/results")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(data, handle, indent=2)
    print(
        json.dumps(
            {
                k: data[k]
                for k in [
                    "file_count",
                    "unique_content_hashes",
                    "guided_original_count",
                    "guided_original_seconds",
                    "guided_original_suites",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
