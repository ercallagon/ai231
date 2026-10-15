#!/usr/bin/env python3
"""Bounded, serialized matched waveform trial; no automatic Pi promotion."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hey_swift.vcm.protocol import sha256
from run_vcm_grouped_bc8_queue import gpu_choices
from vcm_wavevariants_trial import STAGE,PARENT,TAGS,verify

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def log(value): print(time.strftime("%Y-%m-%dT%H:%M:%S%z"),value,flush=True)


def save(path,value):
    with Path(path).open("x") as stream: json.dump(value,stream,indent=2)


def best_record(directory):
    result=json.loads((directory/"result.json").read_text())
    row=result["history"][result["best_epoch"]]
    return {"e2e":row["validation"]["e2e_accuracy"],"slots":row["validation"]["slot_accuracy"],
        "real_intent":row["validation_groups"]["is_synthetic"]["0"]["intent_accuracy"],
        "real_n":row["validation_groups"]["is_synthetic"]["0"]["n"],"best_epoch":result["best_epoch"],
        "epochs_run":result["epochs_run"],"duration_s":result["duration_s"]}


def main():
    lock=(STAGE/"queue.lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    regression=json.loads((STAGE/"regression_passed.json").read_text())
    if not regression["passed"] or sha256(STAGE/"regression.xml")!=regression["sha256"]: raise RuntimeError("Regression proof missing/stale")
    for tag in TAGS.values():
        if (Path("experiments/vcm")/tag).exists(): raise FileExistsError(tag)
    wake=Path("work/wakeword_targeted_v2")
    prior=int((wake/"queue.pid").read_text())
    save(STAGE/"dependency.json",{"wake_queue_pid":prior,"reason":"Serialize training to one GPU during contention"})
    while True:
        try: active=b"run_wakeword_targeted_queue.py" in Path(f"/proc/{prior}/cmdline").read_bytes()
        except FileNotFoundError: active=False
        if not active: break
        log("waiting for wakeword queue to complete; no competing training")
        time.sleep(10)
    wake_result=json.loads((wake/"summary.json").read_text())
    if wake_result["status"] not in ("plateau","complete"): raise RuntimeError("Wake dependency did not complete normally")
    environment=dict(os.environ,PYTHONPATH="src",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONHASHSEED="42",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    records={"parent":best_record(PARENT)}
    for arm,tag in TAGS.items():
        verify()
        query=["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"]
        while True:
            choices=gpu_choices(subprocess.check_output(query,text=True,timeout=15))
            if choices:
                gpu=str(max(choices)[1]); time.sleep(5)
                if int(gpu) in [i for _,i in gpu_choices(subprocess.check_output(query,text=True,timeout=15))]:
                    environment["CUDA_VISIBLE_DEVICES"]=gpu
                    probe=subprocess.run([sys.executable,"-c","import torch,sys;sys.exit(not torch.cuda.is_available())"],env=environment,timeout=60)
                    if probe.returncode==0: break
            log("waiting: GPU <40% utilization and >=20000 MiB free")
            time.sleep(10)
        command=[sys.executable,"-u","scripts/vcm_wavevariants_trial.py","train","--arm",arm]
        save(STAGE/(arm+"_launch.json"),{"command":command,"gpu":gpu,"readiness_sha256":sha256(STAGE/"readiness.json")})
        log(f"start {tag} GPU={gpu}")
        with (STAGE/(arm+".log")).open("x") as stream:
            subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        cpu=dict(environment,CUDA_VISIBLE_DEVICES="")
        with (STAGE/(arm+"_validation.log")).open("x") as stream:
            for mode in ("eval","calibrate"):
                subprocess.run([sys.executable,"-u","scripts/vcm_wavevariants_trial.py",mode,"--arm",arm],env=cpu,stdout=stream,stderr=subprocess.STDOUT,check=True)
        output=Path("experiments/vcm")/tag
        records[arm]=best_record(output)
        records[arm]["calibration"]=json.loads((output/"calibration.json").read_text())["chosen"]
        log(f"screen {arm}: {records[arm]}")
    a,b,p=records["wave"],records["control"],records["parent"]
    recognition=(a["e2e"]>=max(b["e2e"],p["e2e"])-.005 and a["slots"]>=max(b["slots"],p["slots"])-.005)
    improvement=(a["real_intent"]>=max(b["real_intent"],p["real_intent"])+.02 or
        (a["e2e"]>=max(b["e2e"],p["e2e"])+.005 and a["real_intent"]>=max(b["real_intent"],p["real_intent"])-1/a["real_n"]))
    parent_policy=json.loads((PARENT/"calibration.json").read_text())["chosen"]
    policies=[x for x in (b["calibration"],parent_policy) if x]
    rejection=(a["calibration"] is not None and a["calibration"]["tune_oos_false_accept_rate"]<=.05 and
        a["calibration"]["val_in_scope_correct_action_intent_rate"]>=max(x["val_in_scope_correct_action_intent_rate"] for x in policies)-.01)
    qualifies=bool(recognition and improvement and rejection)
    save(STAGE/"selection.json",{"qualified":qualifies,"records":records,"recognition_safeguard":recognition,
        "essential_improvement":improvement,"rejection_safeguard":rejection,"test_used_for_selection":False,
        "known_test_not_pristine":True,"single_seed_not_reproducibility_proof":True})
    if qualifies:
        log("Wave candidate qualified; selection sealed before test access")
        with (STAGE/"selected_evaluation.log").open("x") as stream:
            for arm in TAGS:
                output=Path("experiments/vcm")/TAGS[arm]
                for split in ("test","challenge"):
                    subprocess.run([sys.executable,"-u","scripts/vcm_wavevariants_trial.py","eval","--arm",arm,"--split",split],env=cpu,stdout=stream,stderr=subprocess.STDOUT,check=True)
                    if (output/"calibrated").exists():
                        subprocess.run([sys.executable,"-u","scripts/vcm_wavevariants_trial.py","eval","--arm",arm,"--split",split,
                            "--config",str(output/"calibrated/config.yaml"),"--exp-dir",str(output/"calibrated")],env=cpu,stdout=stream,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,"-u","scripts/vcm_wavevariants_trial.py","export","--arm",arm],env=cpu,stdout=stream,stderr=subprocess.STDOUT,check=True)
    verify()
    save(STAGE/"summary.json",{"status":"complete" if qualifies else "plateau","qualified":qualifies,
        "records":records,"test_accessed":qualifies,"holdout_accessed":False,"original_artifacts_unchanged":True,
        "pi_verified":False,"automatic_promotion":False})
    lines=["# Matched Waveform Trial", "", "| Model | Val E2E | Real Val Intent | Val Slots |", "|---|---:|---:|---:|"]
    lines.extend(f'| {arm} | {r["e2e"]:.4f} | {r["real_intent"]:.4f} | {r["slots"]:.4f} |' for arm,r in records.items())
    lines.extend(["",f"Wave variant qualified: {qualifies}. Test accessed: {qualifies}. Holdout untouched.",
        "", "See selection.json for fixed safeguards and experiment metric files for per-class errors.",
        "This is one seed and a small real validation pool, not proof of Pi noise robustness.",
        "No new human voices, actual recorded music or automatic deployment changes."])
    with (STAGE/"comparison.md").open("x") as stream: stream.write("\n".join(lines)+"\n")
    log("Finished bounded matched comparison; no more runs or deployment changes")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--detach",action="store_true"); args=parser.parse_args()
    if args.detach:
        with (STAGE/"queue.log").open("x") as stream:
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve())],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with (STAGE/"queue.pid").open("x") as stream: stream.write(str(child.pid)+"\n")
        time.sleep(2)
        if child.poll() is not None: raise RuntimeError("Queue failed; inspect queue.log")
        print(f"Persistent matched waveform queue PID={child.pid}")
    else: main()
