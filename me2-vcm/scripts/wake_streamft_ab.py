"""Frozen wake-only A/B: no actions, speech, GPIO or installation changes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/("runtime_src" if (ROOT/"runtime_src").exists() else "src")))
from hey_swift.wakeword.inference import WakeWordDetector
from hey_swift.audio.capture import ContinuousResampler,capture_supported_rate

PROFILES={"deployed":"configs/wakeword.yaml",
          "baseline":"configs/wake_exp15_fresh_control_v1.yaml",
          "candidate":"configs/wake_exp15_streamft_v1.yaml"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=PROFILES,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",type=Path);group.add_argument("--live",action="store_true");group.add_argument("--cost",action="store_true")
    p.add_argument("--device",default="0");p.add_argument("--output",type=Path)
    args=p.parse_args();config=ROOT/PROFILES[args.profile]
    cfg=yaml.safe_load(config.read_text());model=ROOT/cfg["inference"]["onnx"]
    d=WakeWordDetector(str(config),onnx_path=str(model))
    pi_model=Path("/sys/firmware/devicetree/base/model")
    report=dict(profile=args.profile,config_sha256=sha(config),model_sha256=sha(model),
        platform=platform.platform(),pi_model=pi_model.read_bytes().decode(errors="replace").strip("\x00") if pi_model.exists() else None,
        threads=dict(intra=1,inter=1),gate=dict(threshold=d.gate.threshold,sustain=d.sustain,window_samples=d.win,
        hop_samples=d.hop_samples,multiwin=d.gate.multiwin,cooldown_seconds=d.gate.cooldown_seconds),events=[])
    if args.cost:
        d.buf=np.random.default_rng(100).normal(0,.03,d.win+4*d.subhop_samples).astype(np.float32)
        durations=[]
        for i in range(120):
            before=time.perf_counter();d._score()
            if i>=20:durations.append((time.perf_counter()-before)*1000)
        report.update(full_scoring_hop_p50_ms=float(np.median(durations)),full_scoring_hop_p95_ms=float(np.percentile(durations,95)),
            hop_rtf_p95=float(np.percentile(durations,95))/(1000*d.gate.hop_seconds),
            scope="Features+all configured ONNX windows per scoring hop; excludes capture/resampling/actions; not speech-offset latency")
    elif args.input:
        import soundfile as sf
        wave,rate=sf.read(args.input,dtype="float32")
        if wave.ndim>1:wave=wave.mean(1)
        rs=ContinuousResampler(rate)
        for i in range(0,len(wave),max(1,rate//10)):
            report["events"].extend(d.process(rs.push(wave[i:i+max(1,rate//10)]))[0])
        report.update(input_sha256=sha(args.input),input_rate=rate,seconds=len(wave)/rate,
            scope="Digital injection/replay, not live acoustic benchmark; no utterance-level labels assumed")
    else:
        import sounddevice as sd
        device=int(args.device) if args.device.lstrip("-").isdigit() else args.device
        rate,hop=capture_supported_rate(device);rs=ContinuousResampler(rate);start=time.monotonic()
        report.update(input_rate=rate,device=str(device),device_name=sd.query_devices(device,"input")["name"],
                      scope="Live wake-only; no cue/command/actions; annotate utterance timestamps separately")
        print(json.dumps(dict(event="ready",**{k:v for k,v in report.items() if k!="events"})),flush=True)
        try:
            with sd.RawInputStream(samplerate=rate,blocksize=hop,dtype="float32",channels=1,device=device) as stream:
                while True:
                    raw,overflow=stream.read(hop)
                    if overflow:
                        d.reset();rs.reset();print(json.dumps(dict(event="capture_gap",reason="overflow; detector reset")),flush=True);continue
                    ev,_=d.process(rs.push(np.frombuffer(raw,dtype=np.float32)))
                    for e in ev:
                        e=dict(e,elapsed_monotonic=time.monotonic()-start);report["events"].append(e);print(json.dumps(dict(event="wake",**e)),flush=True)
        except KeyboardInterrupt:pass
        report["seconds"]=time.monotonic()-start
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as h:json.dump(report,h,indent=2);h.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
