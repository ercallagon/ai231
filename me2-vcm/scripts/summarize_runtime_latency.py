#!/usr/bin/env python3
"""Summarize detector-estimated speech-end to decision latency from live logs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def valid_ms(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def stats(values):
    if not values:
        return dict(n=0, p50_ms=None, p95_ms=None, mean_ms=None)
    return dict(n=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), mean_ms=float(np.mean(values)))


def summarize(paths):
    all_times, accepted, rejected, inference = [], [], [], []
    missing, invalid, failures, count = 0, 0, 0, 0
    reasons, threads, sources = Counter(), set(), []
    for path in paths:
        records = []
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                records.append(event)
        # Full console logs contain both events for the same decision.
        benchmarks = [e for e in records if e.get("event") == "benchmark_decision"]
        decisions = benchmarks or [e for e in records if e.get("event") == "command"]
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            decisions=len(decisions)))
        for event in decisions:
            count += 1
            if event.get("event") == "command":
                seconds = event.get("latency", {}).get("speech_end_to_decision_seconds")
                value = seconds * 1000 if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None
                is_accepted = event.get("prediction", {}).get("accepted") is True
                reason = event.get("vcm", {}).get("rejection_reason")
            else:
                value = event.get("speech_end_to_decision_ms")
                is_accepted = event.get("accepted") is True
                reason = event.get("rejection_reason")
                audio_ms, infer_ms = event.get("audio_ms"), event.get("infer_ms")
                if valid_ms(audio_ms) and audio_ms > 0 and valid_ms(infer_ms):
                    inference.append(infer_ms)
                if isinstance(event.get("threads"), int):
                    threads.add(event["threads"])
            if reason:
                reasons[str(reason)] += 1
                failures += 1
            if value is None:
                missing += 1
            elif not valid_ms(value):
                invalid += 1
            else:
                all_times.append(value)
                (accepted if is_accepted else rejected).append(value)
    return dict(scope="detector_estimated_speech_end_to_decision_before_services",
        endpoint_reference="last voiced frame read time; not annotated acoustic ground truth",
        sources=sources, total_decisions=count, missing_endpoint_or_legacy_timing=missing,
        invalid_latency=invalid, rejection_reason_counts=dict(reasons),
        decisions_with_rejection_reason=failures, intra_threads=sorted(threads),
        all_speech_decisions=stats(all_times), accepted_speech_decisions=stats(accepted),
        rejected_speech_decisions=stats(rejected),
        preprocessing_and_inference_processed_only=stats(inference),
        note="Latency does not establish recognition correctness; rejected decisions are included. "
             "No-speech/legacy missing timings are counted, never replaced with zero. "
             "Keep profiles and conditions separate. Small trial counts give uncertain p95.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Log files or one-profile log directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({p for entry in args.logs for p in
                    (entry.glob("*.log") if entry.is_dir() else [entry])})
    if not paths:
        parser.error("No log files found")
    if args.output.exists():
        parser.error("Output exists; choose a new result path")
    result = summarize(paths)
    if not result["total_decisions"]:
        parser.error("No JSON command/benchmark_decision records; use the benchmark launcher")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
