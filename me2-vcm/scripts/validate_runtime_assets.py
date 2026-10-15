#!/usr/bin/env python3
"""CPU execution checks for shipped models and real precomputed Piper speech."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hey_swift.audio.speech import PiperSpeech
from hey_swift.vcm.inference import VCMInference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve(); report_path=Path(args.output)
    if report_path.exists(): raise FileExistsError(report_path)
    tag="run_043_hf_grouped_v2b_bc8_sourceft_s42"
    vcm=VCMInference(config=str(root/"configs/vcm_run_043_calibrated.yaml"),
        onnx_path=str(root/f"models/vcm/{tag}.onnx"),labels_path=str(root/f"models/vcm/{tag}.labels.json"))
    prediction=vcm.predict(np.zeros(32000,dtype=np.float32))
    features=vcm.feats.transform(np.zeros(32000,dtype=np.float32))[None]
    outputs=vcm.sess.run(None,{vcm.input_name:features})
    assert all(np.isfinite(output).all() for output in outputs)
    import onnxruntime as ort
    wake=ort.InferenceSession(str(root/"models/wakeword/wakeword_exp15_mined_lp.onnx"),providers=["CPUExecutionProvider"])
    wake_outputs=wake.run(None,{wake.get_inputs()[0].name:np.zeros((1,80,100),dtype=np.float32)})
    assert all(np.isfinite(output).all() for output in wake_outputs)
    speech_dir=root/"integration/assets/speech"
    manifest=json.loads((speech_dir/"manifest.json").read_text())
    played=[]
    speech=PiperSpeech(root/"integration/assets/voices/en_US-amy-medium.onnx",player=played.append,phrases_dir=speech_dir)
    try:
        speech.prepare()
        class NoSynthesis:
            def synthesize_wav(self,*args): raise AssertionError("Fixed response synthesized at runtime")
        speech._voice=NoSynthesis()
        timings={text:speech.speak(text) for text in manifest["phrases"]}
        assert len(played)==len(manifest["phrases"])
        assert all(timing["cached"] for timing in timings.values())
    finally:
        speech.close()
    files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
           [root/f"models/vcm/{tag}.onnx",root/"models/wakeword/wakeword_exp15_mined_lp.onnx"]}
    report={"scope":"Host CPU, no physical audio playback or Pi verification", "model_sha256":files,
            "vcm_output_shapes":[list(out.shape) for out in outputs],
            "wake_output_shapes":[list(out.shape) for out in wake_outputs],
            "silence_prediction":prediction,"silence_note":"Capture speech gate, not this prediction, blocks silence actions",
            "fixed_responses":len(timings),"longest_response_seconds":max(t["audio_seconds"] for t in timings.values()),
            "speech_timings":timings}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"PASS","fixed_responses":len(timings),"vcm":prediction,"receipt":str(report_path)}))


if __name__=="__main__": main()
