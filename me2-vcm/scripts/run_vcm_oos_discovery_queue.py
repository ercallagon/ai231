#!/usr/bin/env python3
"""Bounded autonomous OOS search: two screens and at most one confirmation."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from hey_swift.vcm.protocol import sha256
from run_vcm_grouped_bc8_queue import gpu_choices
from vcm_oos_discovery import STAGE, PARENT, check, write

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def config_for(tag, lr, alpha, seed):
    cfg = yaml.safe_load(Path("configs/vcm_hf_grouped_v2b_bc8_sourceft.yaml").read_text())
    cfg["training"].update(lr=lr, epochs=12, warmup_epochs=1, early_stop_patience=4, seed=seed)
    cfg["experiment"].update(architecture="bc8_command_validity", tag=tag,
        entrypoint="scripts/vcm_oos_discovery.py", parent=PARENT, validity_loss_weight=alpha,
        note="Train-only source-weighted intent CE plus class-balanced validity BCE; validation/tune-only selection")
    for name in ("src/hey_swift/vcm/validity.py", "scripts/vcm_oos_discovery.py",
                 "scripts/run_vcm_oos_discovery_queue.py", "scripts/prepare_vcm_oos_discovery.py"):
        cfg["experiment"]["implementation_sha256"][name] = sha256(name)
    path = Path("configs") / (tag + ".yaml")
    with path.open("x") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return str(path)


def qualifies(result, baseline):
    raw, reference = result["raw_validation"], baseline["raw"]
    real = raw["groups"]["is_synthetic"]["0"]["intent_accuracy"]
    parent_real = reference["groups"]["is_synthetic"]["0"]["intent_accuracy"]
    gain = result["best_policy"]["validation"]["e2e_accuracy"] - baseline["chosen"]["validation"]["e2e_accuracy"]
    return (gain >= .005 and raw["e2e_accuracy"] >= reference["e2e_accuracy"] - .005
            and raw["slot_accuracy"] >= reference["slot_accuracy"] - .005
            and real >= parent_real - .02), gain


def report():
    baselines = json.loads((STAGE / "baselines.json").read_text())
    screens = json.loads((STAGE / "screen_results.json").read_text())
    selection = STAGE / "selection.json"
    if selection.exists():
        tag = json.loads(selection.read_text())["confirmation"]
        screens.append(json.loads((Path("experiments/vcm") / tag / "result.json").read_text()))
    lines = ["# OOS discovery comparison", "", "Validation/tune-only selection; known test evaluated only after selection.",
             "", "| Run | Raw Val E2E | Gated Val E2E | Real Val Intent | Val Slots |",
             "|---|---:|---:|---:|---:|"]
    for result in baselines + screens:
        raw = result.get("raw", result.get("raw_validation"))
        policy = result.get("chosen", result.get("best_policy"))
        real = raw["groups"]["is_synthetic"]["0"]["intent_accuracy"]
        lines.append(f'| {result["tag"]} | {raw["e2e_accuracy"]:.4f} | {policy["validation"]["e2e_accuracy"]:.4f} | {real:.4f} | {raw["slot_accuracy"]:.4f} |')
    lines += ["", "## Selected Candidate Test and Efficiency", "",
              "| Run | Raw Test E2E | Gated Test E2E | OOS Accepted | Valid Rejected | ONNX Bytes | Host p95 ms |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for result in screens:
        root = Path("experiments/vcm") / result["tag"]
        if not (root / "export_validation.json").exists():
            continue
        test = json.loads((root / "metrics_test.json").read_text())
        export = json.loads((root / "export_validation.json").read_text())
        lines.append(f'| {result["tag"]} | {test["raw"]["e2e_accuracy"]:.4f} | {test["gated"]["e2e_accuracy"]:.4f} | {test["gated"]["false_accepts"]} | {test["gated"]["false_rejects"]} | {export["bytes"]} | {export["host_p95_ms"]:.2f} |')
        if export["host_p95_ms"] > 25 or export["bytes"] > 2000000:
            lines.append(f'\nDeployment cost flag: {result["tag"]} exceeds preferred 25ms host p95 or 2MB model size.\n')
    lines += ["", "## Limits", "", "Read summary.json/selection.json for stop and confirmation status.",
        "Per-class confusion, source groups and rejection counts are preserved in experiment metric files.",
        "Fine-tuning seeds share a pretrained parent. Small real/tuning pools and repeated validation reuse limit certainty.",
        "Pi latency, microphone/noise/music robustness and runtime validity-policy support remain unverified.",
        "No holdout use, automatic Pi promotion or test-feedback tuning."]
    (STAGE / "comparison.md").write_text("\n".join(lines) + "\n")


def gate(environment):
    query = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    while True:
        choices = gpu_choices(subprocess.check_output(query, text=True, timeout=15))
        if choices:
            gpu = str(max(choices)[1]); time.sleep(5)
            if int(gpu) in [i for _, i in gpu_choices(subprocess.check_output(query, text=True, timeout=15))]:
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                probe = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(not torch.cuda.is_available())"], env=environment, timeout=60)
                if probe.returncode == 0:
                    return gpu
        log("waiting for one GPU <40% utilization and >=20000 MiB free")
        time.sleep(30)


def execute(mode, config=None, environment=None):
    command = [sys.executable, "-u", "scripts/vcm_oos_discovery.py", mode]
    if config:
        command += ["--config", config]
    tag = yaml.safe_load(Path(config).read_text())["experiment"]["tag"] if config else "baseline"
    with (STAGE / (tag + ".log")).open("a") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    lock = (STAGE / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    readiness = json.loads((STAGE / "readiness.json").read_text())
    if not readiness["passed"]:
        raise RuntimeError("Readiness failed")
    for path, digest in readiness["sources"].items():
        if sha256(path) != digest:
            raise RuntimeError(f"Readiness stale: {path}")
    if not json.loads((Path("experiments/vcm") / PARENT / "export_validation.json").read_text())["onnx_checker_and_batch_1_2_parity_passed"]:
        raise RuntimeError("Parent pipeline incomplete")
    plan = json.loads((STAGE / "plan.json").read_text())
    for prefix in ("run_044_", "run_045_", "run_046_"):
        if list(Path("experiments/vcm").glob(prefix + "*")):
            raise RuntimeError("Planned run already exists; explicit recovery required")
    environment = dict(os.environ, PYTHONPATH="src", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42")
    if (STAGE / "baselines.json").exists():
        raise FileExistsError("Fresh baseline artifacts required")
    log("validation/tune-only rejection-score baseline; one CPU thread, no frozen test")
    execute("baseline", environment=environment)
    baselines = json.loads((STAGE / "baselines.json").read_text())
    baseline = next(r for r in baselines if r["tag"] == PARENT)
    baseline["chosen"] = max(baselines, key=lambda r: r["chosen"]["validation"]["e2e_accuracy"])["chosen"]
    results = []
    for index in range(2):
        if index == 0:
            config = plan["first_config"]
        else:
            first = results[0]
            improved, _ = qualifies(first, baseline)
            if improved:
                lr, alpha = .0001, .25
            elif first["raw_validation"]["e2e_accuracy"] < baseline["raw"]["e2e_accuracy"] - .005:
                lr, alpha = .0003, .10
            else:
                lr, alpha = .0003, .50
            config = config_for("run_045_hf_grouped_v2b_bc8_validity_refine_s42", lr, alpha, 42)
        check(config)
        tag = yaml.safe_load(Path(config).read_text())["experiment"]["tag"]
        gpu = gate(environment)
        command = [sys.executable, "-u", "scripts/vcm_oos_discovery.py", "train", "--config", config]
        write(STAGE / (tag + "_launch.json"), {"command": command, "gpu": gpu,
            "config_sha256": sha256(config), "parent_checkpoint_sha256": sha256(Path("experiments/vcm") / PARENT / "best.pt")})
        log(f"start {tag} GPU={gpu}; validation-only screen")
        execute("train", config, environment)
        result = json.loads((Path("experiments/vcm") / tag / "result.json").read_text())
        result["config"] = config
        results.append(result)
        eligible, gain = qualifies(result, baseline)
        write(STAGE / "screen_results.json", results)
        log(f"screen {tag}: gated validation gain={gain:.4f}; qualifies={eligible}")
    finalists = [r for r in results if qualifies(r, baseline)[0]]
    if not finalists:
        write(STAGE / "summary.json", {"status": "plateau", "reason": "Neither controlled screen achieved >=0.5-point validation-policy gain with recognition safeguards", "frozen_test_accessed": False, "results": results})
        report()
        log("STOP: no essential improvement; no test evaluation or more training")
        return
    winner = max(finalists, key=lambda r: r["best_policy"]["validation"]["e2e_accuracy"])
    cfg = yaml.safe_load(Path(winner["config"]).read_text())
    config = config_for("run_046_hf_grouped_v2b_bc8_validity_confirm_s7", cfg["training"]["lr"], cfg["experiment"]["validity_loss_weight"], 7)
    environment["PYTHONHASHSEED"] = "7"
    gpu = gate(environment)
    write(STAGE / "confirmation_launch.json", {"gpu": gpu, "config_sha256": sha256(config),
        "command": [sys.executable, "-u", "scripts/vcm_oos_discovery.py", "train", "--config", config]})
    log(f"confirmation seed 7 GPU={gpu}")
    execute("train", config, environment)
    tag = yaml.safe_load(Path(config).read_text())["experiment"]["tag"]
    confirmation = json.loads((Path("experiments/vcm") / tag / "result.json").read_text())
    stable = qualifies(confirmation, baseline)[0]
    write(STAGE / "selection.json", {"winner": winner["tag"], "confirmation": tag, "stable": stable,
        "test_used_for_selection": False, "holdout_used": False})
    # Selection is sealed before any final frozen-test access.
    for selected in (winner["config"], config):
        execute("finish", selected, dict(environment, CUDA_VISIBLE_DEVICES=""))
    write(STAGE / "summary.json", {"status": "complete", "stable_validation_gain": stable,
        "winner": winner["tag"], "confirmation": tag, "pi_promoted": False, "holdout_used": False,
        "next_step": "Review final per-class/rejection/latency reports and run live Pi tests; no automatic reliability claim"})
    report()
    log("complete: bounded search stopped; no Pi promotion, no holdout, no more queued runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        with (STAGE / "queue.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (STAGE / "queue.pid").write_text(str(child.pid) + "\n")
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError("Queue failed; inspect queue.log")
        print(f"Persistent OOS discovery queue PID={child.pid}")
    else:
        main()
