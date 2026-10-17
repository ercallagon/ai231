#!/usr/bin/env python3
"""Select a frozen profile; actions default to preview/mock."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    from hey_swift.profiles import resolve_profile
    parser.add_argument("--profile",type=resolve_profile,default="selected",help="champion or baseline; legacy selected/run043 accepted")
    parser.add_argument("--wake-profile",default="exp15")
    args,extra=parser.parse_known_args()
    os.chdir(ROOT)
    import yaml
    from hey_swift.integration.e2e_loop import main as run
    profile=json.loads((ROOT/"configs/assistant_profiles.json").read_text())["profiles"][args.profile]
    registry=ROOT/"configs/wake_profiles.json"
    wakes=json.loads(registry.read_text())["profiles"] if registry.exists() else {"exp15":{"config":"configs/wakeword.yaml","model":"models/wakeword/wakeword_exp15_mined_lp.onnx"}}
    if args.wake_profile not in wakes:
        parser.error("Unknown wake profile; inspect configs/wake_profiles.json")
    wake=wakes[args.wake_profile]
    config=yaml.safe_load((ROOT/profile["config"]).read_text())
    command=["--vcm-onnx",str(ROOT/profile["model"]),"--vcm-labels",str(ROOT/profile["labels"]),
        "--vcm-config",str(ROOT/profile["config"]),"--wake-config",str(ROOT/wake["config"]),
        "--wake-onnx",str(ROOT/wake["model"]),
        "--hardware-config",str(ROOT/"configs/hardware.yaml"),"--ack",str(ROOT/"audio/wake_chime.wav")]
    for key,value in config["capture"].items():
        if key in ("command_seconds","pre_roll_ms","speech_start_ms","speech_start_timeout"):
            command+=["--"+key.replace("_","-"),str(value)]
    # Keep historical capture settings in model YAML; explicit CLI arguments
    # below override the measured public runtime endpoint.
    command += ["--end-silence-ms", "800"]
    if "--threshold" in extra or any(x.startswith("--threshold=") for x in extra):
        parser.error("Frozen profiles do not permit confidence-threshold overrides")
    return run(command+extra)


if __name__=="__main__":
    raise SystemExit(main())
