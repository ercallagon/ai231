#!/usr/bin/env python3
"""Log one decision per command for the class acoustic benchmark; mock actions."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    from hey_swift.profiles import resolve_profile
    ap.add_argument("--profile",type=resolve_profile,default="selected",help="champion or baseline; legacy IDs accepted")
    ap.add_argument("--device",default="0")
    ap.add_argument("--log-dir",type=Path,default=Path.home()/"vcm_benchmark")
    args=ap.parse_args()
    args.log_dir.mkdir(parents=True,exist_ok=True)
    path=args.log_dir/("swift_"+args.profile+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".log")
    command=[sys.executable,"-u",str(ROOT/"scripts/run_assistant.py"),"--profile",args.profile,
        "--device",args.device,"--hardware","mock","--offline","--log-format","json","--benchmark-log"]
    print("Benchmark decision log: "+str(path),flush=True)
    with path.open("x",buffering=1) as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                print(line,end="",flush=True)
                try: event=json.loads(line)
                except ValueError: continue
                if event.get("event") in ("benchmark_decision","wake"):
                    log.write(line);log.flush()
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill();child.wait()


if __name__=="__main__":
    raise SystemExit(main())
