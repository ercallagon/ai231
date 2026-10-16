"""Consolidate existing evidence without fitting, hiding failures or inventing Pi results."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def count_rate(stat):
    if stat is None:
        return "MISSING"
    if not stat.get("n"):
        return "N/A (n=0)"
    n, k = stat["n"], stat["correct"]
    p, z = k / n, 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return f"{k}/{n} = {p:.2%} (Wilson 95% {center-half:.2%}..{center+half:.2%})"


def build_report(root, official=None):
    output = root / "work/pi5_test_01"
    output.mkdir(parents=True, exist_ok=True)
    reference = read(root / "data/model_reference.json") or {}
    provenance = read(root / "data/PROVENANCE.json") or {}
    document = dict(created_utc=datetime.now(timezone.utc).isoformat(), reference=reference,
        dataset=provenance, measured={}, missing=[], evidence={})
    lines = ["# ME2 Raspberry Pi report", "", "Generated from preserved evidence; no calibration/training performed.",
             "Previously exposed evaluation sets; isolated checkout on a shared OS, not a fresh-OS certification.",
             "", "## Models", "", "| Public name | Historical profile | Parameters | Bytes | Threshold | SHA-256 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for public, model in reference.get("models", {}).items():
        path = root / ("models/wakeword/wakeword_exp15_mined_lp.onnx" if public == "wakeword" else "models/vcm/" + model["legacy_profile"] + ".onnx")
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if digest != model["model_sha256"]:
            document["missing"].append(public + ": artifact hash mismatch or missing")
        lines.append(f"| {public} | {model['legacy_profile']} | {model['parameters']} | {path.stat().st_size if path.exists() else 'MISSING'} | {model['threshold']} | {digest} |")
        lines.extend(["", f"{public}: {model['architecture']}", ""])
    lines += ["## Dataset and training", "", f"Dataset: {provenance.get('dataset', 'MISSING')}; DOI {provenance.get('doi', 'MISSING')}; revision {provenance.get('revision', 'MISSING')}.",
              "", "| Split | Recordings | Original-waveform hours | Real recordings | Declared groups (not verified humans) |", "| --- | ---: | ---: | ---: | ---: |"]
    for split, row in provenance.get("manifests", {}).items():
        lines.append(f"| {split} | {row['rows']} | {row['hours']:.4f} | {row['real_rows']} | {row['declared_speaker_groups']} |")
    lines += ["", "Training accounting: " + json.dumps(reference.get("training", {})), "", "## Pi injected recognition", ""]
    keys = (("Raw intent, full manifest", "overall_intent"), ("Real intent", "real_intent"),
        ("Synthetic intent", "synthetic_intent"), ("Raw supported intent+slot", "supported_joint"),
        ("Accepted-correct supported intent+slot", "supported_accepted_joint"),
        ("Accepted-correct real supported", "real_supported_accepted"), ("Accepted-correct synthetic supported", "synthetic_supported_accepted"),
        ("Semantic success including rejection", "semantic"), ("OOS false accepts", "oos_false_accept"),
        ("Unsupported-value false accepts", "unsupported_false_accept"),
        ("Unsupported temperature false accepts", "unsupported_temperature_false_accept"),
        ("Temperature intent+value", "temperature_joint"), ("Temperature accepted-correct", "temperature_accepted"))
    for split in ("test", "holdout"):
        path = output / "injection" / (split + "_RESULTS.json")
        results = read(path)
        if results is None:
            document["missing"].append(split + ": no local injection results")
            continue
        document["measured"][split] = results
        document["evidence"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        lines += [f"### {split}", "", "| Metric | Champion | Baseline |", "| --- | --- | --- |"]
        for label, key in keys:
            values = [count_rate(results.get(name, {}).get(key)) for name in ("selected", "run043")]
            lines.append(f"| {label} | {values[0]} | {values[1]} |")
        for label, key in (("Macro-F1", "macro_f1"), ("Wrong supported accepts", "wrong_supported_accepts"),
                           ("Valid-command false rejects (includes processing failures)", "false_rejects"),
                           ("Processing failures", "failures"), ("Unscored labels", "unscored"),
                           ("Feature+ONNX p95 ms", "infer_p95_ms"), ("Per-clip RTF p95", "per_clip_rtf_p95")):
            values = [results.get(name, {}).get(key, "MISSING") for name in ("selected", "run043")]
            lines.append(f"| {label} | {values[0]} | {values[1]} |")
        for name in ("selected", "run043"):
            m = results.get(name)
            if not m:
                document["missing"].append(split + ": missing profile " + name)
                continue
            public_name = "champion" if name == "selected" else "baseline"
            expected = reference.get("models", {}).get(public_name, {}).get("model_sha256")
            if expected and m.get("model_sha256") != expected:
                document["missing"].append(split + "/" + name + ": result model hash differs from frozen profile; comparison requires review")
            lines += ["", f"{name} full/processed denominators: " + json.dumps(m.get("processed_subsets", {})),
                      "", "Duration/long-input cohorts: " + json.dumps(m.get("duration_cohorts", {})),
                      "", f"Timing scope: {m.get('infer_scope', 'MISSING')}; Pi-identified: {m.get('physical_pi_verified', 'MISSING')}; threads: {m.get('threads', 'MISSING')}.",
                      "", f"#### {name} per-family errors", "", "| Intent | Intent | Accepted intent+slot | P / R / F1 | False rejects | Processing failures |", "| --- | --- | --- | --- | ---: | ---: |"]
            for family, row in m.get("families", {}).items():
                lines.append(f"| {family} | {count_rate(row['intent'])} | {count_rate(row['accepted_joint'])} | {row['precision']:.3f} / {row['recall']:.3f} / {row['f1']:.3f} | {row['false_reject']} | {row['failures']} |")
            lines += ["", "Confusion matrix: see corresponding profile.confusion in " + str(path.relative_to(root)) + "; last column is processing failure."]
            lines += ["", "Slot accuracy given the true family (independent of intent prediction): " + json.dumps(m.get("slot_accuracy_given_true_family", "MISSING in older runs")), ""]
        lines += ["", "Paired accepted-correct changes: " + json.dumps(results.get("paired_supported", {})), ""]
    lines += ["## Live and physical evidence", ""]
    live = sorted(output.glob("*_live_*/latency.json"))
    document["measured"]["live_latency"] = [dict(source=str(p.relative_to(root)), data=read(p)) for p in live]
    for item in document["measured"]["live_latency"]:
        lines += [item["source"], "```json", json.dumps(item["data"], indent=2), "```", ""]
    if not live:
        document["missing"].append("Live detector-estimated speech-end to decision p50/p95 and no-speech failures")
    for filename in ("DIAGNOSTICS.json", "resources.json", "model_costs.json", "DEMO.json"):
        value = read(output / filename)
        document["measured"][filename] = value
        if value is None:
            document["missing"].append(filename + ": not collected")
        else:
            displayed = value
            if filename == "resources.json":
                displayed = {k: v for k, v in value.items() if k != "samples"}
            elif filename == "model_costs.json":
                displayed = {name: {k: v for k, v in row.items() if k != "samples_ms"} for name, row in value.items()}
            lines += ["### " + filename, "", "Raw samples remain in " + filename, "", "```json", json.dumps(displayed, indent=2), "```", ""]
    physical = document["measured"].get("DEMO.json")
    if physical:
        document["missing"] += ["Operator physical check: " + key + " = " + value
            for key, value in physical.get("observations", {}).items() if value != "PASS"]
    lines += ["## Wakeword reference (historical; not new Pi streaming)", "", "```json", json.dumps(reference.get("wake_clip_reference", {}), indent=2), "```", ""]
    if official:
        path = official / "metrics.json"
        value = read(path)
        document["measured"]["official_acoustic_benchmark"] = dict(source=str(path), data=value)
        if value:
            lines += ["## Supplied class acoustic benchmark", "", str(path), "```json", json.dumps(value, indent=2), "```", ""]
        else:
            document["missing"].append("Supplied official benchmark metrics.json missing")
    else:
        document["missing"].append("Official acoustic benchmark: no supplied result directory")
    document["missing"] += ["Fresh independent-speaker confirmation", "Fresh-OS installation verification",
        "Annotated acoustic speech-end reference", "Live wake misses/false triggers per hour/raw-AEC/music robustness unless separately benchmarked",
        "GPIO/display/timer, call/SMS, audible TTS/music fallback and API action effects: require explicit physical test evidence",
        "Action initiation/completion and TTS onset latency: not inferred from recognition latency",
        "Live assistant memory/CPU peaks, power/energy and independent repeated-device uncertainty: not measured by evaluation-worker telemetry"]
    lines += ["## Measurement scope", "", "Injected latency = features + ONNX, excluding download/decode/resampling/capture/endpoint wait/decision mapping/actions/TTS. RTF uses actual original waveform duration. First inference is included; one intra-op/inter-op thread. Failures remain in recognition denominators; timing only covers processed recordings.",
              "", "Live latency uses the detector's final voiced-frame read time and includes endpoint waiting; it is not annotated acoustic ground truth. Acceptance alone is not correct recognition.",
              "", "Wilson intervals are descriptive binomial intervals; correlated voices/recordings and prior exposure limit generalization.",
              "", "## Missing evidence / limits", ""]
    lines += ["- " + item for item in document["missing"]]
    lines += ["", reference.get("limits", ""), "", "Data/model licensing: data/DATASET_LICENSE.md, MODEL_TERMS.md, THIRD_PARTY_NOTICES.md. Owned code MIT. No accounts, recipients or private configuration included."]
    (output / "REPORT.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Consolidated report:", output / "REPORT.md")
    return document
