#!/usr/bin/env python3
"""Audit a saved class benchmark without changing its official scores or trials."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def summary(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    def percentile(p):
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": percentile(.5), "p95": percentile(.95)}


def review(folder):
    path = folder / "trials.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    wake = [r for r in rows if r["kind"] == "wake"]
    supported = [r for r in wake if r["true_intent"] not in ("OUT_OF_SCOPE", "REJECT")]
    def payload(row):
        try:
            value = json.loads(row.get("pred_raw", ""))
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}
    def cohort(group):
        return {"n": len(group), "accepted_correct": sum(r["correct_command"].lower() == "true" for r in group)}
    failures = Counter()
    for row in supported:
        if row["correct_command"].lower() == "true":
            continue
        if row["pred_intent"] == "NO_RESPONSE":
            failures["no_response"] += 1
        elif payload(row).get("rejection_reason") == "no_sustained_speech":
            failures["no_sustained_speech"] += 1
        elif row["pred_intent"] in ("OUT_OF_SCOPE", "REJECT"):
            failures["inferred_or_other_rejection"] += 1
        else:
            failures["wrong_supported_command"] += 1
    processed = [r for r in wake if (number(r.get("audio_ms")) or 0) > 0
                 and (number(r.get("infer_ms")) or 0) > 0]
    early = [r for r in wake if number(r.get("latency_s")) is not None and number(r["latency_s"]) < 0]
    multiple = [r for r in wake if (number(r.get("n_command_events")) or 0) > 1]
    no_wake = [r for r in rows if r["kind"] == "no_wake"]
    sources = {}
    for name in ("trials.csv", "metrics.json", "report.md", "pi_metrics.csv", "config.json", "plan.json"):
        source = folder / name
        if source.is_file():
            sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "scope": "supplementary_audit_not_replacement_official_scoring",
        "source_sha256": sources,
        "trials": len(rows), "wake_present": len(wake), "no_wake": len(no_wake),
        "supported": {"all": cohort(supported),
            "real": cohort([r for r in supported if r["is_synthetic"].lower() == "false"]),
            "synthetic": cohort([r for r in supported if r["is_synthetic"].lower() == "true"])},
        "supported_failure_outcomes": dict(failures),
        "early_decision_trials": {"n": len(early), "orders": [r["order"] for r in early]},
        "multiple_decision_trials": {"n": len(multiple), "orders": [r["order"] for r in multiple]},
        "zero_audio_decisions": sum(number(r.get("audio_ms")) == 0 for r in wake),
        "missing_timing_fields": sum(number(r.get("infer_ms")) is None or number(r.get("audio_ms")) is None
                                     for r in wake if (number(r.get("n_command_events")) or 0) > 0),
        "preprocessing_and_inference_ms_processed_only": summary([number(r["infer_ms"]) for r in processed]),
        "rtf_processed_audio_only": summary([number(r["infer_ms"]) / number(r["audio_ms"]) for r in processed]),
        "detector_speech_end_to_decision_ms": summary([number(payload(r).get("speech_end_to_decision_ms"))
            for r in wake if number(payload(r).get("speech_end_to_decision_ms")) is not None
            and number(payload(r).get("speech_end_to_decision_ms")) >= 0]),
        "no_wake_outcomes": {
            "n": len(no_wake),
            "trials_with_any_decision": sum((number(r.get("n_command_events")) or 0) > 0 for r in no_wake),
            "trials_with_explicit_wake_event": sum(r.get("wake_logged", "").lower() == "true" for r in no_wake),
            "selected_accepted_predictions": sum(payload(r).get("accepted") is True for r in no_wake)},
        "limitations": [
            "Early decisions are retained, not discarded or rescored. They can be premature detections or timing/association problems.",
            "Multiple decisions are not necessarily multiple accepted actions; full Pi diagnostics are needed.",
            "Inference scope includes input preparation/features/ONNX, not endpoint waiting, action or TTS.",
            "Detector speech end is not independently annotated acoustic ground truth.",
            "Selected predictions alone cannot reconstruct every decision or prove its acoustic cause."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
