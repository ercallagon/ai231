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
    parser.add_argument("--profile",choices=("selected","run043"),default="selected")
    args,extra=parser.parse_known_args()
    os.chdir(ROOT)
    import yaml
    from hey_swift.integration.e2e_loop import main as run
    profile=json.loads((ROOT/"configs/assistant_profiles.json").read_text())["profiles"][args.profile]
    config=yaml.safe_load((ROOT/profile["config"]).read_text())
    command=["--vcm-onnx",str(ROOT/profile["model"]),"--vcm-labels",str(ROOT/profile["labels"]),
        "--vcm-config",str(ROOT/profile["config"]),"--wake-config",str(ROOT/"configs/wakeword.yaml"),
        "--wake-onnx",str(ROOT/"models/wakeword/wakeword_exp15_mined_lp.onnx"),
        "--hardware-config",str(ROOT/"configs/hardware.yaml"),"--ack",str(ROOT/"audio/wake_chime.wav")]
    for key,value in config["capture"].items():
        if key in ("command_seconds","end_silence_ms","pre_roll_ms","speech_start_ms","speech_start_timeout"):
            command+=["--"+key.replace("_","-"),str(value)]
    if "--threshold" in extra or any(x.startswith("--threshold=") for x in extra):
        parser.error("Frozen profiles do not permit confidence-threshold overrides")
    return run(command+extra)


if __name__=="__main__":
    raise SystemExit(main())
