"""Export, freeze validation calibration, then evaluate without refitting."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import trial as T
from hey_swift.wakeword.inference import WakeWordDetector


def scores(session,split):
    result={}
    for kind in ("positive","negative"):
        for cond in ("raw","hp80"):
            array=np.load(T.OLD/f"data/{split}_{kind}_{cond}_legacy.npy",mmap_mode="r");parts=[]
            for i in range(0,len(array),128):
                z=session.run(None,{session.get_inputs()[0].name:np.array(array[i:i+128])})[0]
                e=np.exp(z-z.max(1,keepdims=True));parts.extend((e/e.sum(1,keepdims=True))[:,0])
            result[kind,cond]=np.asarray(parts)
    np.savez_compressed(HERE/f"candidate_{split}_scores.npz",**{k+"_"+c:v for (k,c),v in result.items()})
    return result


def make_config(name,model,tau):
    cfg=yaml.safe_load((T.ROOT/"configs/wakeword.yaml").read_text())
    cfg["inference"].update(onnx=str(model.relative_to(T.ROOT)),frontend="legacy",window_seconds=1.,
        multiwin=False,n_subwin=1,hop_seconds=.25,sustain=2,cooldown_seconds=2.,gate_nonwake=.9,
        rms_floor_db=-60.,intra_op_threads=1,inter_op_threads=1)
    cfg["inference"]["threshold"]=tau
    cfg["experiment"]=dict(name="exp15_runtime_window_finetune_v1",default=False,pi_verified=False,
                           gate="single fresh1s score per250ms; two consecutive, integer2s cooldown")
    cfg.pop("training",None)
    path=HERE/(name+".yaml")
    with path.open("x") as h:yaml.safe_dump(cfg,h,sort_keys=False)
    return path


def host_checks(configs):
    result={};data=T.Data();examples=data.make_specs(101,8)
    for name,path in configs.items():
        d=WakeWordDetector(str(path));outputs=[]
        wave=np.random.default_rng(123).normal(0,.02,96000).astype(np.float32)
        for chunk in (1600,1733,96000):
            d.reset();ev=[]
            for i in range(0,len(wave),chunk):ev.extend(d.process(wave[i:i+chunk])[0])
            outputs.append(ev)
        assert outputs[0]==outputs[1]==outputs[2]
        d.reset();assert not d.process(np.zeros(96000,np.float32))[0]
        errors=[]
        for spec in examples:
            w,features,_=data.render(spec)
            p,_,_=d._score_window(w)
            z=d.sess.run(None,{d.input_name:features[None]})[0][0]
            e=np.exp(z-z.max());expected=float(e[0]/e.sum());errors.append(abs(p-expected))
        assert max(errors)<1e-7
        d.buf=wave[-(d.win+4*d.subhop_samples):];times=[]
        for i in range(120):
            before=time.perf_counter();d._score()
            if i>=20:times.append((time.perf_counter()-before)*1000)
        result[name]=dict(chunk_invariance=True,silence_rejection=True,training_wave_runtime_score_max_error=max(errors),
            full_scoring_hop_p95_ms=float(np.percentile(times,95)),hop_rtf_p95=float(np.percentile(times,95))/250,
            threads=dict(intra=1,inter=1),scope="host CPU NOT Pi capture/action latency")
    return result


def main():
    import torch
    from hey_swift.wakeword.model import build_model
    torch.set_num_threads(1)
    done=json.loads((HERE/"training/DONE.json").read_text())
    if done["best_finetuned_epoch"]==0:raise RuntimeError("No full fine-tune epoch completed")
    checkpoint=HERE/"training/best_finetuned.pt"
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=build_model();model.load_state_dict(state["state_dict"]);model.eval()
    onnx=HERE/"candidate.onnx"
    torch.onnx.export(model,torch.zeros(1,80,100),str(onnx),input_names=["features"],output_names=["logits"],
        dynamic_axes={"features":{0:"batch"},"logits":{0:"batch"}},opset_version=17,dynamo=False)
    runtime=T.ort_session(onnx);errors=[]
    for n in (1,2,17):
        x=np.random.default_rng(n).normal(size=(n,80,100)).astype(np.float32)
        with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
        actual=runtime.run(None,{"features":x})[0]
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5);errors.append(float(np.max(np.abs(actual-expected))))
    T.save(HERE/"EXPORT.json",dict(model_sha256=T.sha(onnx),checkpoint_sha256=T.sha(checkpoint),epoch=state["epoch"],
        parity_batches=[1,2,17],max_abs_errors=errors,passed=True))
    baseline=json.loads((HERE/"BASELINE_FREEZE.json").read_text());budget=baseline["primary_budget"]
    candidate_validation=scores(runtime,"validation");candidate_curve=T.curve(candidate_validation)
    T.save(HERE/"CANDIDATE_VALIDATION_CURVE.json",candidate_curve)
    budgets=sorted(set([5.,15.,30.,60.,120.,budget]))
    selected={name:{str(b):T.choose(rows,b) for b in budgets} for name,rows in
              (("baseline",baseline["curves"]),("candidate",candidate_curve))}
    base=selected["baseline"][str(budget)];cand=selected["candidate"][str(budget)]
    gain=T.rank(cand)[0]-T.rank(base)[0]
    nominated=gain>=.02 and T.maxrate(cand["metrics"])<=budget+1e-8
    configs=dict(baseline=make_config("baseline_fresh",T.MODEL,base["threshold"]),
                 candidate=make_config("candidate_fresh",onnx,cand["threshold"]))
    # All checkpoint/policy choices are immutable before exposed test scoring.
    T.save(HERE/"MODEL_POLICY_FREEZE.json",dict(time=time.time(),epoch=state["epoch"],checkpoint_sha256=T.sha(checkpoint),
        model_sha256=T.sha(onnx),baseline_sha256=T.sha(T.MODEL),policies=selected,primary_budget=budget,
        validation_worst_recall_gain=gain,validation_nomination=nominated,baseline_epoch0_winner=not done["improved_vs_initialization"],
        shared_policy=dict(window_samples=16000,hop_samples=4000,sustain=2,cooldown_samples=32000,multiwin=False),
        source_sha256=T.sha(__file__),test_accessed=False,test_previously_exposed=True,pi_verified=False,
        veto="No >2point test recall regression raw/HP; original false events<=1.25baseline (or1ifbaseline0); no extra known music-only dev events. No refit/selection from test."))
    test_scores=scores(runtime,"test");base_test=T.cached_scores("test")
    results={}
    for name,p in selected.items():
        current=test_scores if name=="candidate" else base_test
        results[name]={key:T.metrics(current,"test",row["threshold"],True) for key,row in p.items()}
    T.save(HERE/"EXPOSED_TEST_RESULTS.json",dict(primary_budget=budget,results=results,
        freeze_sha256=T.sha(HERE/"MODEL_POLICY_FREEZE.json"),official_positive_n=385,
        positive_type="synthetic only; previously exposed",processing_failure_n=sum(r["failure"] for r in T.metadata("test","positive","raw")["intervals"])))
    spec=importlib.util.spec_from_file_location("prior_frozen_evaluator",T.OLD/"finalize.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    development=module.replay(dict(configs,exp15_deployed=T.ROOT/"configs/wakeword.yaml"))
    T.save(HERE/"RAW_AEC_DEVELOPMENT.json",dict(recordings=development,
        limits="Previously exposed paired Pi recordings. No verified utterance timestamps: event counts are NOT recall/latency. No training or calibration on these recordings; raw/AEC correlated."))
    actual_base=results["baseline"][str(budget)];actual_candidate=results["candidate"][str(budget)]
    paired={}
    for cond in ("raw","hp80"):
        a=actual_base[cond];b=actual_candidate[cond];aa=np.asarray(a["matched"]);bb=np.asarray(b["matched"])
        delays=[(tb-ta)/16 for ta,tb in zip(a["times"],b["times"]) if ta is not None and tb is not None]
        paired[cond]=dict(fixes=int(np.sum(~aa&bb)),regressions=int(np.sum(aa&~bb)),both_detect=int(np.sum(aa&bb)),
            both_miss=int(np.sum(~aa&~bb)),paired_delay_median_ms=float(np.median(delays)) if delays else None,
            paired_delay_p95_ms=float(np.percentile(delays,95)) if delays else None)
    recall_ok=all(actual_candidate[c]["recall"]>=actual_base[c]["recall"]-.02 for c in ("raw","hp80"))
    fa_ok=all(actual_candidate[c]["original_events"]<=max(1,1.25*actual_base[c]["original_events"]) for c in ("raw","hp80"))
    music_ok=all(v["candidate"]["count"]<=v["baseline"]["count"] for row in development.values() if row["condition"]=="music_only" for v in row["paths"].values())
    checks=host_checks(dict(configs,exp15_deployed=T.ROOT/"configs/wakeword.yaml"));T.save(HERE/"HOST_CHECKS.json",checks)
    provenance=json.loads((HERE/"PROVENANCE.json").read_text())
    for path,digest in provenance["protected"].items():assert T.sha(T.ROOT/path)==digest,path
    recommendation="optional Pi confirmation candidate, not physically promoted" if nominated and recall_ok and fa_ok and music_ok else "retain exp15; close experimentation"
    T.save(HERE/"DECISION.json",dict(recommendation=recommendation,validation_nomination=nominated,
        validation_worst_recall_gain=gain,test_recall_veto_pass=recall_ok,test_false_event_veto_pass=fa_ok,
        development_music_veto_pass=music_ok,paired=paired,default="exp15",protected_unchanged=True,
        pi_verified=False,additional_training_queued=False))
    report(baseline,selected,results,development,checks,done,recommendation,paired)


def report(baseline,selected,results,development,checks,done,recommendation,paired):
    budget=baseline["primary_budget"];audit=json.loads((HERE/"DATA_AUDIT.json").read_text())
    lines=["# Final exp15 Runtime-Window Fine-Tune", "", "## Decision", "", recommendation+". Default remains exp15; physical Pi5 verification pending.",
        "", "One fine-tune, no scratch run or architecture change. Original best checkpoint verified against deployed export; epoch42 selected historically, despite erroneous archived config best_epoch60. Provenance/parity errors are saved in PROVENANCE.json.",
        f"Training completed{done['epochs']}epochs; best fine-tuned epoch{done['best_finetuned_epoch']}; initial-checkpoint improvement:{done['improved_vs_initialization']}. GPU wall{done['gpu_seconds']:.1f}s ({done['gpu_hours']:.4f}GPUhours), peak reserved{done['peak_reserved_bytes']/1e9:.3f}GB.",
        f"Training eligibility:{audit['positive_n']}complete <=1s positives/{audit['negative_n']}negatives;{len(audit['exclusions'])}positive exclusions, preserved in audit, never negative labels. All evaluation utterances retained regardless of training eligibility.",
        "Fresh epoch/index-seeded waveform positions and permitted non-wake TTS/confuser context; runtime1s crop BEFORE feature extraction. Half windows exercise actual48k->16k100ms-chunk resampling after deterministic upsampling. This is algorithmic parity, NOT a real microphone simulation. Partial overlaps are IGNORE; explicit neighbor labels logged; no fabricated word boundaries or partial-negative labels. Complete-word retention remains manually unverified.",
        "", "## Matched Validation", "",
        f"Primary budget{budget:.5f}events/hour is anchored BEFORE training to baseline under shared policy at tau.30. It is an adversarial-corpus comparison, not acceptable real-room FAR. Raw/HP and original/generated cohorts must satisfy budgets separately. HP80 is a filtering proxy, NOT AEC.",
        "One fresh1s scoring window every250ms; two consecutive fresh decisions; RMS/nonwake gates; integer2s cooldown. Stale-score test shows four-window max can reuse one spike as sustain. Both baseline/candidate disable that heuristic. GRU state resets each scoring call.",
        "", "| Budget | Model | Threshold | Raw validation recall | HP validation recall | Raw/HP original events |", "|---:|---|---:|---:|---:|---:|"]
    for key in selected["baseline"]:
        for name in selected:
            row=selected[name][key];m=row["metrics"]
            lines.append(f"|{float(key):.3f}|{name}|{row['threshold']:.4f}|{m['raw']['recall']:.2%}|{m['hp80']['recall']:.2%}|{m['raw']['original_events']}/{m['hp80']['original_events']}|")
    lines += ["", "## Frozen Exposed Test", "", "| Model | Raw detected | HP detected | Raw/HP original false events | Raw rate95%CI /hour |", "|---|---:|---:|---:|---:|"]
    for name in results:
        m=results[name][str(budget)];r=m["raw"];h=m["hp80"];ci=r["original_fa_h_ci95"]
        lines.append(f"|{name}|{r['detected']}/{r['n']} ({r['recall']:.2%})|{h['detected']}/{h['n']} ({h['recall']:.2%})|{r['original_events']}/{h['original_events']}|{r['original_fa_h']:.2f} ({ci[0]:.2f}..{ci[1]:.2f})|")
    lines += ["", "Full curves/policies froze before test scoring. All385official positives are synthetic and previously exposed; no fresh independent claim. Event counts cannot be mixed with clip TNs to fabricate accuracy/precision/F1. Original negative exposure~18.4min; generated17.1min reported separately. Existing automatic boundary-reference failures remain denominator misses.",
        "Signed endpoint latencies use automated energy reference, not manual human endpoints; paired common-detection delays:"]
    for cond,p in paired.items():lines.append(f"- {cond}: fixes{p['fixes']}, regressions{p['regressions']}, common{p['both_detect']}, median delay{p['paired_delay_median_ms']}ms, p95{p['paired_delay_p95_ms']}ms.")
    lines += ["", "## Actual Raw/AEC Development Replay", "", "No annotated utterance timestamps: these are event counts, NOT true recall or detection latency. Pairs correlated and previously exposed. Never used as gradient/calibration inputs.",
        "", "| Recording condition/path | Baseline fresh | Candidate fresh | Exp15 deployed |", "|---|---:|---:|---:|"]
    for key,row in development.items():
        for cond,v in row["paths"].items():lines.append(f"|{row['condition']}/{cond}/{key}|{v['baseline']['count']}|{v['candidate']['count']}|{v['exp15_deployed']['count']}|")
    lines += ["", "## Checks And Remaining Limits", ""]
    for name,c in checks.items():lines.append(f"- {name}: training-wave/runtime score parity maxerror{c['training_wave_runtime_score_max_error']}; silence/chunk checks PASS; host full-hop p95{c['full_scoring_hop_p95_ms']:.3f}ms, RTF{c['hop_rtf_p95']:.5f}, ORT intra1/inter1. NOT Pi latency.")
    lines += ["- Original exp15 initialization inherits historical split/exposure weaknesses; fine-tuning cannot erase them. Historical validation/test share17canonical Piper voice families. No real-speaker generalization claim.",
        "- No verified real-room/music training negatives with independent ancestry were available; context uses permitted TTS/confusers and generated stress. Actual AEC contains nonlinear effects not represented by HP80; proper AEC recall/latency needs phrase timestamps.",
        "- Source data/voice terms and access remain private-workspace limitations; no raw audio/credentials/state published. Public inference verification is reproducible; private wake fine-tuning is NOT advertised as publicly one-command reproducible.",
        "- Independent confirmation: consenting fresh5speakers x10wakes in quiet and5x10over low music, with timestamps; at least30min no-wake music/household/beep recordings. Freeze policies first; do not tune on this confirmation set. About3zero-event negative hours are needed for a95%upper rate near1/hour.",
        "- No Pi install, action, call/SMS, threshold replacement or historical-artifact overwrite occurred. No extra training is queued. See PI_TEST.md for wake-only comparisons and rollback."]
    with (HERE/"REPORT.md").open("x") as h:h.write("\n".join(lines)+"\n")


if __name__=="__main__":main()
