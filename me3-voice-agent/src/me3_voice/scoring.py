"""Score a completed diagnostic report against explicit exhaustive annotations."""

import hashlib
import json
import math
from pathlib import Path


def score(report_path: Path, labels_path: Path):
    raw = report_path.read_bytes()
    labels_raw = labels_path.read_bytes()
    if max(len(raw), len(labels_raw)) > 2 * 1024 * 1024:
        raise ValueError("report/annotations exceed scoring budget")
    report, labels = json.loads(raw), json.loads(labels_raw)
    if (
        labels.get("schema_version") != 1
        or labels.get("complete") is not True
        or labels.get("input_sha256") != report.get("input_sha256")
        or labels.get("audio_origin") not in ("synthetic", "recorded")
        or labels.get("audio_origin") != report.get("audio_origin")
        or labels.get("split") not in ("development", "qualification")
    ):
        raise ValueError("complete, matching origin/hash and explicit split required")
    if (
        report.get("finished") is not True
        or report.get("run_status") != "eof"
        or report.get("events_truncated")
        or report.get("vad_segments_truncated")
        or report.get("hash_scope") != "consumed PCM bytes, excluding WAV header"
    ):
        raise ValueError("complete untruncated EOF report with PCM hash required")
    duration = report["output_stats"]["samples"] / 16000

    def ranges(value):
        if not isinstance(value, list) or len(value) > 1000:
            raise ValueError("bounded annotation ranges required")
        previous = 0
        result = []
        for row in value:
            start, end = row["start"], row["end"]
            if (
                type(start) not in (float, int)
                or type(end) not in (float, int)
                or not math.isfinite(start)
                or not math.isfinite(end)
                or not previous <= start < end <= duration
            ):
                raise ValueError("ordered nonoverlapping ranges within input required")
            previous = end
            result.append((start, end))
        return result

    windows = ranges(labels["wake_windows"])
    speech = ranges(labels["speech_ranges"])
    predicted = ranges(report["vad_segments"])
    remaining = list(windows)
    detected, false_activations = 0, 0
    for event in report["wake_events"]:
        offset = event["audio_seconds"]
        match = next((w for w in remaining if w[0] <= offset < w[1]), None)
        if match is None:
            false_activations += 1
        else:
            remaining.remove(match)
            detected += 1
    # Midpoint frame classification explicitly excludes the unscored EOF tail.
    tp = fp = fn = tn = 0
    for frame in range(report["vad_frames"]):
        midpoint = (frame * 512 + 256) / 16000
        expected = any(start <= midpoint < end for start, end in speech)
        actual = any(start <= midpoint < end for start, end in predicted)
        if actual and expected:
            tp += 1
        elif actual:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
    negative_hours = duration / 3600 if not windows else None
    return {
        "kind": "frontend-score",
        "schema_version": 1,
        "mode": report["mode"],
        "audio_origin": labels["audio_origin"],
        "split": labels["split"],
        "replay_sha256": hashlib.sha256(raw).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_raw).hexdigest(),
        "input_sha256": report["input_sha256"],
        "wake_expected": len(windows),
        "wake_detected": detected,
        "wake_missed": len(remaining),
        "false_activations": false_activations,
        "negative_exposure_hours": negative_hours,
        "false_activations_per_hour": false_activations / negative_hours
        if negative_hours
        else None,
        "vad_tp": tp,
        "vad_fp": fp,
        "vad_fn": fn,
        "vad_tn": tn,
        "vad_precision": tp / (tp + fp) if tp + fp else None,
        "vad_recall": tp / (tp + fn) if tp + fn else None,
        "limitations": "Single exhaustively annotated clip; input-offset acceptance windows, "
        "not acoustic wake delay. VAD frame-midpoint scoring excludes unscored tail. "
        "Wake-once/session policy can suppress later phrases inside an active session. "
        "Use separate clips for independent wake attempts. Zero observed errors is not "
        "a zero population error rate. Synthetic qualification cannot certify real speakers.",
    }
