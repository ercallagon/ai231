#!/usr/bin/env python3
"""Frozen CPU evaluation: no threshold fitting, training or action execution."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("verify","reproduce"),nargs="?",default="verify")
    ap.add_argument("--splits",nargs="+",choices=("val","tune_oos","test","holdout"),default=["test","holdout"])
    ap.add_argument("--profiles",nargs="+",choices=("selected","run043"),default=["selected","run043"])
    ap.add_argument("--data-root",type=Path,default=ROOT/"data/vcm/raw")
    ap.add_argument("--fetch",action="store_true")
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    import numpy as np
    from hey_swift.vcm.inference import VCMInference
    from hey_swift.vcm import release_data as D
    from hey_swift.vcm.archived_metric_core import metrics
    from scipy.special import softmax
    profiles=json.loads((ROOT/"configs/assistant_profiles.json").read_text())["profiles"]
    models={name:VCMInference(str(ROOT/profiles[name]["config"]),str(ROOT/profiles[name]["model"]),str(ROOT/profiles[name]["labels"])) for name in args.profiles}
    if args.action=="verify":
        for path,digest in json.loads((ROOT/"ARTIFACTS.json").read_text())["sha256"].items():
            if D.sha(ROOT/path)!=digest:
                raise ValueError("Artifact integrity failure: "+path)
        for name,model in models.items():
            wave=.1*np.sin(np.arange(16000,dtype=np.float32)*2*np.pi*440/16000)
            result=model.predict(wave)
            assert np.isfinite(result["confidence"])
            print(json.dumps(dict(profile=name,contract="loaded",threads=model.threads)))
        wake_registry=ROOT/"configs/wake_profiles.json"
        if wake_registry.exists():
            from hey_swift.wakeword.inference import WakeWordDetector
            checked=set()
            for name,wake in json.loads(wake_registry.read_text())["profiles"].items():
                key=(wake["config"],wake["model"])
                if key in checked:continue
                checked.add(key)
                detector=WakeWordDetector(str(ROOT/wake["config"]),onnx_path=str(ROOT/wake["model"]))
                assert detector.sess.get_outputs()[0].shape[-1]==2
                wave=.1*np.sin(np.arange(detector.win,dtype=np.float32)*2*np.pi*440/16000)
                pw,pn,_=detector._score_window(wave)
                assert np.isfinite(pw) and np.isfinite(pn)
                print(json.dumps(dict(wake_profile=name,contract="loaded",frontend=detector.frontend,
                    threshold=detector.gate.threshold,physical_pi_verified=False)))
        print("Frozen model contracts verified; physical Pi checks pending")
        return
    if args.output is None:
        ap.error("--output is required")
    args.output.mkdir(parents=True,exist_ok=False)
    for split in args.splits:
        rows=D.manifest(split)
        D.ensure_files(rows,args.data_root,args.fetch)
        outputs,valid,statuses,times={},{},{},{}
        duration=np.zeros(len(rows))
        for name,model in models.items():
            outputs[name]={o.name:np.zeros((len(rows),int(o.shape[-1])),np.float32) for o in model.sess.get_outputs()}
            valid[name]=np.zeros(len(rows),bool)
            statuses[name]=["not_processed"]*len(rows)
            times[name]=[]
        parity={name:0 for name in models}
        for done,(i,wave) in enumerate(D.waves(rows,args.data_root),1):
            duration[i]=len(wave)/16000
            for name,model in models.items():
                start=time.perf_counter()
                if model.frontend=="legacy_center_crop":
                    mel=model.feats.transform(wave);feed={"mel":mel[None]};status="ok_legacy_center_crop"
                else:
                    mel,mask,receipt=model.feats.transform(wave)
                    status=receipt["status"]
                    if mel is None:
                        statuses[name][i]=status
                        continue
                    feed={"mel":mel[None],"frame_mask":mask[None]}
                feature_done=time.perf_counter()
                values=model.sess.run(None,feed)
                end=time.perf_counter()
                if not all(np.isfinite(x).all() for x in values):
                    statuses[name][i]="nonfinite_model_output"
                    continue
                valid[name][i]=True;statuses[name][i]=status
                for key,value in zip(outputs[name],values):
                    outputs[name][key][i]=value[0]
                times[name].append(dict(row=i,feature_ms=(feature_done-start)*1000,model_ms=(end-feature_done)*1000,
                    infer_ms=(end-start)*1000,audio_ms=duration[i]*1000))
                if parity[name]<12:
                    actual=model.predict(wave)
                    p=softmax(outputs[name]["intent_logits"][i]);idx=int(p.argmax())
                    expected=idx!=19 and p.max()>=model.tau;intent=model.intent_labels[idx]
                    if intent in model.slot_outputs:
                        sid=int(outputs[name][model.slot_outputs[intent]][i].argmax())
                        expected &= model.slot_sets[intent][sid]!="REJECT"
                    assert bool(expected)==actual["accepted"] and intent==actual["intent"]
                    parity[name]+=1
            if done%500==0:
                print(json.dumps(dict(split=split,completed=done,total=len(rows))),flush=True)
        summary={}
        y,ys,unsupported=D.labels(rows,split)
        known=np.ones(len(rows),bool)
        for intent in D.tax().slotted_intents:
            known[(y==D.tax().intent_to_idx[intent])&(ys<0)&~unsupported]=False
        supported=known&(y!=19)&~unsupported
        successes={}
        for name,model in models.items():
            out=outputs[name];slots={n:out["slot_"+n] for n in D.tax().slotted_intents}
            m=metrics(out["intent_logits"],slots,valid[name],rows,split,1.,model.tau)
            probability=softmax(out["intent_logits"],axis=1);pred=probability.argmax(1)
            accepted=valid[name]&(pred!=19)&(probability.max(1)>=model.tau)&~((pred==15)&(slots["TEMPERATURE"].argmax(1)==3))
            joint=valid[name]&(pred==y)
            for intent in D.tax().slotted_intents:
                ix=(y==D.tax().intent_to_idx[intent])&(ys>=0)
                joint[ix] &= slots[intent].argmax(1)[ix]==ys[ix]
            successes[name]=accepted&joint&supported
            subsets={}
            real=np.array([r["is_synthetic"]=="0" for r in rows])
            for cohort,ix in (("all",np.ones(len(rows),bool)),("real",real),("synthetic",~real)):
                processed=ix&valid[name];neg=processed&(y==19)
                subsets[cohort]=dict(manifest_n=int(ix.sum()),processed_n=int(processed.sum()),failures=int((ix&~valid[name]).sum()),
                    processed_intent_correct=int((processed&(pred==y)).sum()),
                    processed_intent_accuracy=float((processed&(pred==y)).sum()/processed.sum()) if processed.any() else None,
                    processed_supported_n=int((processed&supported).sum()),
                    processed_supported_accepted_correct=int((processed&successes[name]).sum()),
                    processed_oos_n=int(neg.sum()),processed_oos_false_accepts=int((neg&accepted).sum()),
                    processed_oos_far=float((neg&accepted).sum()/neg.sum()) if neg.any() else None)
            durations={}
            for label,ix in (("le2s",duration<=2),("gt2_le4s",(duration>2)&(duration<=4)),("gt4_le6s",(duration>4)&(duration<=6)),("gt6s",duration>6)):
                durations[label]=dict(n=int(ix.sum()),failures=int((ix&~valid[name]).sum()),
                    supported_n=int((ix&supported).sum()),accepted_correct=int((ix&successes[name]).sum()))
            pi=Path("/proc/device-tree/model")
            m.update(processed_subsets=subsets,duration_cohorts=durations,processing_status=dict(Counter(statuses[name])),
                runtime_parity_rows=parity[name],model_sha256=D.sha(ROOT/profiles[name]["model"]),threads=model.threads,
                temperature_already_baked_once=True,infer_scope="features_plus_ONNX_no_decode_resampling_capture_actions",
                infer_p95_ms=float(np.percentile([t["infer_ms"] for t in times[name]],95)),
                per_clip_rtf_p95=float(np.percentile([t["infer_ms"]/t["audio_ms"] for t in times[name]],95)),
                physical_pi_verified=pi.exists() and "Raspberry Pi 5" in pi.read_text(),
                complete_word_retention="UNVERIFIED",previously_exposed_evaluation=True)
            summary[name]=m
            np.savez_compressed(args.output/(split+"_"+name+"_predictions.npz"),**out,valid=valid[name],accepted=accepted,accepted_correct=successes[name])
            (args.output/(split+"_"+name+"_timings.json")).write_text(json.dumps(times[name],indent=2)+"\n")
            with (args.output/(split+"_"+name+"_errors.jsonl")).open("x") as f:
                for i,row in enumerate(rows):
                    f.write(json.dumps(dict(id=row["audio_path"],intent=row["intent"],slot=row.get("slot"),
                        is_synthetic=row["is_synthetic"],predicted_intent=model.intent_labels[pred[i]],
                        accepted=bool(accepted[i]),accepted_correct=bool(successes[name][i]),
                        supported=bool(supported[i]),status=statuses[name][i]))+"\n")
        if set(successes)=={"selected","run043"}:
            a,b=successes["run043"],successes["selected"]
            summary["paired_supported"]=dict(candidate_fixes=int((b&~a).sum()),candidate_regressions=int((a&~b).sum()))
        (args.output/(split+"_RESULTS.json")).write_text(json.dumps(summary,indent=2)+"\n")
    (args.output/"COMPLETE.json").write_text(json.dumps(dict(splits=args.splits,profiles=args.profiles,
        thresholds_frozen=True,thresholds={n:r.tau for n,r in models.items()},trained=False,actions_executed=False,
        previously_exposed_evaluation=True),indent=2)+"\n")
    print("Frozen evaluation complete: "+str(args.output))


if __name__=="__main__":
    main()
