"""Pi/host wake scoring cost and optional WAV replay, no action execution."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/"src").exists():ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))


def main():
    import numpy as np
    import yaml
    from hey_swift.wakeword.inference import WakeWordDetector
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/wakeword.yaml")
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=ROOT/args.config;cfg=yaml.safe_load(config.read_text())
    model=ROOT/cfg["inference"]["onnx"]
    detector=WakeWordDetector(str(config),onnx_path=str(model))
    timings=[];original=detector._score
    def measured():
        start=time.perf_counter();result=original();timings.append((time.perf_counter()-start)*1000);return result
    detector._score=measured
    if args.input:
        import soundfile as sf
        from hey_swift.audio.capture import ContinuousResampler
        raw,rate=sf.read(args.input,dtype="float32")
        if raw.ndim>1:raw=raw.mean(1)
        resampler=ContinuousResampler(rate)
        wave=np.concatenate([resampler.push(raw[i:i+rate//10]) for i in range(0,len(raw),rate//10)])
    else:
        wave=np.random.default_rng(20261015).normal(0,.03,16000*40).astype(np.float32)
    events=[]
    for i in range(0,len(wave),1600):events.extend(detector.process(wave[i:i+1600])[0])
    samples=timings[20:]
    if not samples:raise ValueError("Need at least20 warmup hops plus one measured hop")
    pi=Path("/proc/device-tree/model")
    report=dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        device=pi.read_text().strip("\x00") if pi.exists() else platform.platform(),
        physical_pi5=pi.exists() and "Raspberry Pi 5" in pi.read_text(),
        threads=dict(intra=detector.sess.get_session_options().intra_op_num_threads,inter=detector.sess.get_session_options().inter_op_num_threads),
        scope="entire scoring hop: feature extraction and ONNX including configured multiwindows; excludes capture/resampling/actions",
        p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),
        scoring_hop_rtf_p95=float(np.percentile(samples,95))/(detector.gate.hop_seconds*1000),
        seconds=len(wave)/16000,events=events,generated_noise_input=not bool(args.input),
        not_accuracy_or_acoustic_latency=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(report,f,indent=2);f.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
