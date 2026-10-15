#!/usr/bin/env python3
"""Serialized wakeword optimization after the bounded VCM discovery queue."""
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
from wakeword_targeted_trial import STAGE, write

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def log(message): print(time.strftime("%Y-%m-%dT%H:%M:%S%z"),message,flush=True)


def verify():
    ready=json.loads((STAGE/"readiness.json").read_text())
    if not ready["passed"] or not ready["parent_deployed_onnx_parity"]: raise RuntimeError("Wake readiness failed")
    for path,digest in ready["sources"].items():
        if sha256(path)!=digest: raise RuntimeError(f"Wake source changed: {path}")
    regression=json.loads((STAGE/"regression_passed.json").read_text())
    if not regression["passed"]: raise RuntimeError("Wake regressions failed")
    if sha256(STAGE/"regression.xml")!=regression["sha256"]: raise RuntimeError("Wake regression evidence changed")
    audit=json.loads((STAGE/"audit.json").read_text())
    if sha256(STAGE/"audit.json")!=ready["audit_sha256"]: raise RuntimeError("Wake audit changed")
    for path,digest in audit["original_sha256"].items():
        if sha256(path)!=digest: raise RuntimeError(f"Original wake artifact changed: {path}")
    if sha256(STAGE/"baseline.json")!=ready["baseline_sha256"]: raise RuntimeError("Baseline changed")


def main():
    lock=(STAGE/"queue.lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    for tag in ("exp16_targeted_hardneg_s20261009","exp17_targeted_confirm_s7"):
        if (Path("experiments/wakeword")/tag).exists(): raise FileExistsError(tag)
    prior=int(Path("work/vcm_oos_discovery/queue.pid").read_text())
    write(STAGE/"dependency.json",{"vcm_controller_pid":prior,"reason":"One active training GPU maximum during contention"})
    while True:
        try: active=b"run_vcm_oos_discovery_queue" in Path(f"/proc/{prior}/cmdline").read_bytes()
        except FileNotFoundError: active=False
        if not active: break
        log("waiting for VCM discovery queue to finish; no competing GPU training")
        time.sleep(10)
    summary_path=Path("work/vcm_oos_discovery/summary.json")
    if not summary_path.exists(): raise RuntimeError("VCM controller stopped without completion; inspect before starting wake training")
    environment=dict(os.environ,PYTHONPATH="src",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    baseline=json.loads((STAGE/"baseline.json").read_text())
    results=[]
    for tag,strength,lr,seed in (("exp16_targeted_hardneg_s20261009",3.,.0001,20261009),
                                ("exp17_targeted_confirm_s7",3.,.0001,7)):
        if results:
            result=results[0]
            clean=result["validation"]; robust=result["perturbed_validation"]
            improved=(clean["recall"]>=baseline["selected"]["recall"]+.01 and robust["recall"]>=baseline["perturbed_at_selected"]["recall"]-.02)
            if not improved:
                write(STAGE/"summary.json",{"status":"plateau","reason":"No >=1pt recall improvement at <=1% validation clip FAR with perturbed-recall safeguard", "results":results,"test_accessed":False})
                log("STOP: no essential wake improvement; no test or further training")
                return
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
        environment["PYTHONHASHSEED"]=str(seed)
        command=[sys.executable,"-u","scripts/wakeword_targeted_trial.py","train","--tag",tag,"--strength",str(strength),"--lr",str(lr),"--seed",str(seed)]
        write(STAGE/(tag+"_launch.json"),{"command":command,"gpu":gpu,"readiness_sha256":sha256(STAGE/"readiness.json")})
        log(f"start {tag} GPU={gpu}")
        with (STAGE/(tag+".log")).open("x") as stream:
            subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        results.append(json.loads((Path("experiments/wakeword")/tag/"result.json").read_text()))
        write(STAGE/"screen_results.json",results)
    write(STAGE/"selection.json",{"candidates":[r["tag"] for r in results],"thresholds_frozen_from_validation":True,"test_used_for_selection":False})
    for result in results:
        tag=result["tag"]
        with (STAGE/(tag+"_evaluation.log")).open("x") as stream:
            subprocess.run([sys.executable,"-u","scripts/wakeword_targeted_trial.py","finish","--tag",tag],
                env=dict(environment,CUDA_VISIBLE_DEVICES=""),stdout=stream,stderr=subprocess.STDOUT,check=True)
    verify()
    write(STAGE/"summary.json",{"status":"complete","results":results,"original_artifacts_unchanged":True,
        "no_deployment_promotion":True,"limitations":"Clip FAR is not false activations/hour; real Pi music/far-field/long-stream testing remains required"})
    log("complete; originals/config/threshold unchanged; no more wake runs")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--detach",action="store_true"); args=parser.parse_args()
    if args.detach:
        with (STAGE/"queue.log").open("x") as stream:
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve())],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        (STAGE/"queue.pid").write_text(str(child.pid)+"\n"); time.sleep(2)
        if child.poll() is not None: raise RuntimeError("Wake queue failed; inspect log")
        print(f"Persistent wakeword queue PID={child.pid}")
    else: main()
