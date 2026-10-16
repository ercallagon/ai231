#!/usr/bin/env python3
"""Log one decision per command for the class acoustic benchmark; mock actions."""
import argparse
from datetime import datetime
import json
import math
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
    ap.add_argument("--end-silence-ms",type=float,default=None,
        help="Override endpoint wait (e.g. 800); omit to retain the frozen profile's capture settings.")
    args=ap.parse_args()
    if args.end_silence_ms is not None and (not math.isfinite(args.end_silence_ms) or args.end_silence_ms <= 0):
        ap.error("--end-silence-ms must be finite and positive")
    args.log_dir.mkdir(parents=True,exist_ok=True)
    path=args.log_dir/("swift_"+args.profile+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
    diagnostics=path.with_suffix(".diagnostics")
    command=[sys.executable,"-u",str(ROOT/"scripts/run_assistant.py"),"--profile",args.profile,
        "--device",args.device,"--hardware","mock","--offline","--log-format","json","--benchmark-log"]
    if args.end_silence_ms is not None:
        command += ["--end-silence-ms",str(args.end_silence_ms)]
    print("Benchmark decision log: "+str(path),flush=True)
    print("Full runtime diagnostics: "+str(diagnostics),flush=True)
    with path.open("x",buffering=1) as log, diagnostics.open("x",buffering=1) as diagnostic:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                print(line,end="",flush=True)
                diagnostic.write(line)
                try: event=json.loads(line)
                except ValueError: continue
                if event.get("event") in ("benchmark_decision","wake"):
                    log.write(line);log.flush()
            return child.wait()
        except KeyboardInterrupt:
            return 130
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill();child.wait()


if __name__=="__main__":
    raise SystemExit(main())
