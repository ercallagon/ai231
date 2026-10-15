#!/usr/bin/env python3
"""Check committed checkpoint vs ONNX logits and decisions on owned probes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    import numpy as np
    import torch
    from scipy.special import softmax
    from hey_swift.vcm.inference import VCMInference
    from hey_swift.vcm.taxonomy import load_taxonomy
    from hey_swift.vcm.fullcontext_model import ContextBC8
    from hey_swift.vcm.broadcast import build_broadcast_model
    torch.set_num_threads(1)
    profiles=json.loads((ROOT/"configs/assistant_profiles.json").read_text())["profiles"]
    result={}
    for name,p in profiles.items():
        runtime=VCMInference(str(ROOT/p["config"]),str(ROOT/p["model"]),str(ROOT/p["labels"]))
        tax=load_taxonomy(ROOT/p["config"])
        if name=="selected":
            model=ContextBC8(tax)
            checkpoint=ROOT/"experiments/selected/checkpoint.pt"
            temperature=runtime.labels["temperature_baked"]
        else:
            model=build_broadcast_model(20,{n:tax.slot_classes(n) for n in tax.slotted_intents})
            checkpoint=ROOT/"experiments/baseline/checkpoint.pt"
            temperature=.6475641442117414
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model.load_state_dict(payload["state_dict"]);model.eval()
        checks=0
        for n in ((401,16000,96000) if name=="selected" else (401,16000,32000)):
            wave=(.1*np.sin(np.arange(n)*2*np.pi*440/16000)).astype(np.float32)
            if name=="selected":
                mel,mask,_=runtime.feats.transform(wave)
                feed={"mel":mel[None],"frame_mask":mask[None]}
                with torch.no_grad(): output=model(torch.from_numpy(feed["mel"]),torch.from_numpy(feed["frame_mask"]))
            else:
                feed={"mel":runtime.feats.transform(wave)[None]}
                with torch.no_grad(): output=model(torch.from_numpy(feed["mel"]))
            actual=dict(zip(runtime.output_names,runtime.sess.run(None,feed)))
            np.testing.assert_allclose(actual["intent_logits"],output["intent"].numpy()/temperature,rtol=2e-4,atol=2e-4)
            for slot,value in output["slots"].items():
                np.testing.assert_allclose(actual["slot_"+slot],value.numpy(),rtol=2e-4,atol=2e-4)
            prediction=runtime.predict(wave);prob=softmax(actual["intent_logits"][0])
            assert abs(prediction["confidence"]-float(prob.max()))<1e-6
            checks+=1
        result[name]=dict(probe_lengths_checked=checks,parameters=sum(p.numel() for p in model.parameters()),temperature_applied_once=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
