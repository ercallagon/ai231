"""Frozen CPU evaluation and opt-in curation after both scratch trials finish."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
import onnxruntime as ort
import yaml

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("sealed_final_study",OUT/"study.py")
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
from hey_swift.wakeword.inference import WakeWordDetector
from hey_swift.audio.capture import ContinuousResampler


def session(model):
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    return ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])


def probabilities(model,frontend,split):
    sess=session(model);result={}
    for kind in ("positive","negative"):
        for condition in ("raw","hp80"):
            x=np.load(OUT/f"data/{split}_{kind}_{condition}_{frontend}.npy",mmap_mode="r");parts=[]
            for i in range(0,len(x),128):
                z=sess.run(None,{sess.get_inputs()[0].name:np.array(x[i:i+128])})[0]
                e=np.exp(z-z.max(1,keepdims=True));parts.extend((e/e.sum(1,keepdims=True))[:,0])
            result[kind,condition]=np.array(parts)
    return result


def config(name,arm,model,threshold):
    base=yaml.safe_load((ROOT/"configs/wakeword.yaml").read_text())
    base["inference"].update(frontend=arm,window_seconds=1. if arm=="legacy" else 2.,
        multiwin=False,n_subwin=1,threshold=threshold,onnx=str(model.relative_to(ROOT)),intra_op_threads=1,inter_op_threads=1)
    base["audio"].update(target_frames=100 if arm=="legacy" else 200,input_shape=[80,100 if arm=="legacy" else 200],cmvn=arm!="pcen2s_v1")
    if arm!="legacy":
        base["training"]=json.loads((OUT/arm/"CONFIG.json").read_text())
        base["calibration"]=json.loads((OUT/arm/"POLICY_FREEZE.json").read_text())
    p=OUT/(name+".yaml")
    with p.open("x") as f:yaml.safe_dump(base,f,sort_keys=False)
    return p


def replay(configs):
    result={}
    for folder in sorted((ROOT/"work/hpc-pi-feedback-20261011/aec").glob("20*")):
        import soundfile as sf
        label=json.loads((folder/"report.json").read_text())["label"]
        row=dict(condition=label,paths={})
        for kind in ("raw","aec"):
            wave,sr=sf.read(folder/(kind+".wav"),dtype="float32")
            values={}
            for name,cfg in configs.items():
                detector=WakeWordDetector(str(cfg));resampler=ContinuousResampler(sr);events=[]
                for i in range(0,len(wave),sr//10):events.extend(detector.process(resampler.push(wave[i:i+sr//10]))[0])
                values[name]=dict(events=events,count=len(events))
            row["paths"][kind]=dict(seconds=len(wave)/sr,sha256=S.sha(folder/(kind+".wav")),**values)
        result[folder.name]=row
    return result


def checks(configs):
    result={}
    for name,cfg in configs.items():
        detector=WakeWordDetector(str(cfg));wave=np.random.default_rng(1).normal(0,.03,96000).astype(np.float32)
        outputs=[]
        for chunk in (1600,1733,96000):
            detector.reset();events=[]
            for i in range(0,len(wave),chunk):events.extend(detector.process(wave[i:i+chunk])[0])
            outputs.append(events)
        assert outputs[0]==outputs[1]==outputs[2]
        detector.reset();assert not detector.process(np.zeros(96000,dtype=np.float32))[0]
        elapsed=[]
        for i in range(120):
            start=time.perf_counter();detector._score_window(wave[:detector.win])
            if i>=20:elapsed.append((time.perf_counter()-start)*1000)
        # Cached scoring policy must match the actual process method exactly.
        meta=dict(samples=np.arange(detector.win,160000+1,4000).tolist(),rms=[.1]*((160000-detector.win)//4000+1))
        p=np.random.default_rng(7).uniform(0,1,len(meta["samples"]))
        expected=S.events(p,meta,detector.gate.threshold)
        scores=iter(p)
        detector.reset();detector._score=lambda:(lambda q:(q,1-q,True))(float(next(scores)))
        actual=[]
        for i in range(0,160000,1733):actual.extend(detector.process(np.zeros(min(1733,160000-i),dtype=np.float32))[0])
        assert [int(round(e["t"]*16000)) for e in actual]==expected
        result[name]=dict(chunk_invariance=True,silence_rejection=True,policy_vectorization_parity=True,
            feature_plus_onnx_p95_ms=float(np.percentile(elapsed,95)),window_rtf_p95=float(np.percentile(elapsed,95))/(detector.win/16),
            hop_rtf_p95=float(np.percentile(elapsed,95))/250,threads=dict(intra=1,inter=1),scope="host CPU NOT Pi")
    return result


def main():
    S.verify()
    for arm in S.ARMS:assert (OUT/arm/"EXPORT.json").exists()
    evaldir=OUT/"evaluation";evaldir.mkdir(exist_ok=False)
    policies={arm:json.loads((OUT/arm/"POLICY_FREEZE.json").read_text()) for arm in S.ARMS}
    baseline=json.loads((OUT/"BASELINE.json").read_text())
    nominated=max(S.ARMS,key=lambda a:tuple(policies[a]["policy"]["score"]))
    S.save(evaldir/"SELECTION_FREEZE.json",dict(nominated=nominated,policies=policies,baseline=baseline["selected"],
        selection="validation only; no test/development comparisons yet",test_previously_exposed=True,pi_verified=False,
        finalizer_source_sha256=S.sha(Path(__file__)),
        confirmation_veto="No new music false events; exposed-test recall nonregression within2percentage points on raw/HP80; empirical original/generated testFA <=5/hour. A veto retains exp15, never selects another model or refits."))
    audit=json.loads((ROOT/"work/wakeword_targeted_v2/audit.json").read_text())
    S.prepare_streams("test",audit["split_rows"]["test"])
    models={"exp15_calibrated":(ROOT/"models/wakeword/wakeword_exp15_mined_lp.onnx","legacy",baseline["selected"]["threshold"])}
    models.update({arm:(OUT/arm/"model.onnx",arm,policies[arm]["policy"]["threshold"]) for arm in S.ARMS})
    configs={name:config(name,arm,model,tau) for name,(model,arm,tau) in models.items()}
    configs["exp15_default"]=ROOT/"configs/wakeword.yaml"
    results={}
    for name,(model,frontend,tau) in models.items():
        for split in ("validation","test"):
            probs=probabilities(model,frontend,split)
            results.setdefault(name,{})[split]=S.stream_metric(probs,frontend,split,tau)
            if name=="exp15_calibrated":
                # Reference the shipped four-overlapping-window policy exactly,
                # including RMS from the same window that supplied max P(wake).
                maxima={};levels={}
                for (kind,cond),p in probs.items():
                    r=np.array(json.loads((OUT/f"data/{split}_{kind}_{cond}_legacy.json").read_text())["rms"])
                    pp=np.stack([np.r_[np.zeros(k),p][:len(p)] if k else p for k in (3,2,1,0)])
                    rr=np.stack([np.r_[np.zeros(k),r][:len(r)] if k else r for k in (3,2,1,0)])
                    ix=pp.argmax(0);maxima[kind,cond]=pp[ix,np.arange(len(p))];levels[kind,cond]=rr[ix,np.arange(len(p))]
                results.setdefault("exp15_default",{})[split]=S.stream_metric(maxima,"legacy",split,.3,level_overrides=levels)
                np.savez_compressed(evaldir/("exp15_default_"+split+".npz"),**{a+"_"+b:v for (a,b),v in maxima.items()},
                    **{a+"_"+b+"_rms":v for (a,b),v in levels.items()})
            np.savez_compressed(evaldir/(name+"_"+split+".npz"),**{a+"_"+b:v for (a,b),v in probs.items()})
    S.save(evaldir/"RESULTS.json",results)
    subprocess.run([sys.executable,"-B",str(OUT/"error_analysis.py")],cwd=ROOT,check=True)
    development=replay(configs);S.save(evaldir/"DEVELOPMENT.json",development)
    runtime=checks({k:v for k,v in configs.items() if k!="exp15_default"});S.save(evaldir/"RUNTIME.json",runtime)
    S.verify()
    lines=["# Final streaming wakeword trial", "", "Frozen validation selection; previously exposed synthetic-positive test; no physical Pi verification.",
        "Default remains exp15. Both candidate models are trained from random initialization; no external detector weights.",
        "Baseline calibrated BEFORE training. Its calibrated single-window policy is not the deployed four-window tau0.30 policy.",
        "Strict training ancestry filtering means the new arms are matched to each other, not an exact historical exp15 data match.","",
        "| Model | Split/condition | Wake detected | FA events / negative hours | FA/hour | Energy-end p95ms |",
        "|---|---|---:|---:|---:|---:|"]
    for name,splits in results.items():
        for split,conds in splits.items():
            for cond,m in conds.items():
                lines.append(f"| {name} | {split}/{cond} | {m['detected']}/{m['positive_n']} ({m['recall']:.2%}) | {m['negative_events']}/{m['negative_hours']:.4f} | {m['false_events_per_hour']:.2f} | {m['energy_endpoint_latency_p95_ms']} |")
    lines += ["", "Negative exposure combines original negative recordings and1024seconds of reproducible generated stress; raw/HP views are correlated. Reported zero-event upper bounds are in RESULTS.json.",
        "No negative silent padding inflates exposure. Calibration enforces <=5FA/hour separately on original and generated negative cohorts, in both raw/HP conditions; no dilution by easy generated hours. Per-cohort rates are in RESULTS.json. Events/hour on this assembled corpus is not measured real-room FA/hour.",
        "The5FA/hour target is empirical, NOT a statistical guarantee. Short negative exposure can yield a95% upper bound above5/hour even with zero observed events; separate cohort bounds are reported.",
        "Positive matching uses recorded phrase intervals and allows1s after waveform end; misses and automatic boundary failures remain in denominators.",
        "Energy-end latencies are automated estimates, not manually verified end-of-speech; signed latency can be negative.",
        "HP80 is a deterministic filtering stress, NOT a simulation or verification of actual AEC.","",
        "## Development Raw/AEC Replay", "", "Event counts are not true recall; phrase identities/timestamps have not been manually verified.",
        "| Condition/path/recording | exp15 default | exp15 calibrated | CMVN | PCEN |", "|---|---:|---:|---:|---:|"]
    music_regression=False
    for key,row in development.items():
        for kind,values in row["paths"].items():
            counts=[values[n]["count"] for n in ("exp15_default","exp15_calibrated",*S.ARMS)]
            lines.append(f"| {row['condition']}/{kind}/{key} | "+" | ".join(map(str,counts))+" |")
            if row["condition"]=="music_only" and values[nominated]["count"]>values["exp15_default"]["count"]:music_regression=True
    positive_gain=min(policies[nominated]["policy"]["metrics"][c]["recall"] for c in ("raw","hp80"))>=min(baseline["selected"]["metrics"][c]["recall"] for c in ("raw","hp80"))
    test_nonregression=all(results[nominated]["test"][c]["recall"]>=results["exp15_calibrated"]["test"][c]["recall"]-.02 for c in ("raw","hp80"))
    test_rate_guard=all(results[nominated]["test"][c][key]<=S.FA_LIMIT for c in ("raw","hp80") for key in ("original_false_events_per_hour","generated_false_events_per_hour"))
    recommendation="candidate for optional Pi A/B only" if positive_gain and not music_regression and test_nonregression and test_rate_guard else "not supported for deployment; optional diagnostic Pi A/B only"
    lines += ["", "## Recommendation And Limitations", "", f"Validation nominee: {nominated}. Recommendation: {recommendation}.",
        "exp15 remains default. No automatic deployment or claim of successful Pi verification.",
        "No complete human-source ancestry proof across distinct TTS engines; official positives are all synthetic and exposed test is not independent.",
        "Insufficient real call/ring/music diversity and real positive speakers remain limitations. Generated stress is not actual call audio.",
        "Both trials retain full original <=2s training waveforms with varied positions; over-capacity positives are excluded and enumerated, never silently cropped.",
        "Fixed PCEN is motivated by https://arxiv.org/abs/1607.05666 ; no published checkpoint or learned frontend is loaded.",
        "The integer sample clock fixes chunk-dependent cooldown; silence, chunk-size and vectorized/actual-gate parity checks passed.",
        "Host features+ONNX costs with intra1/inter1, NOT Pi or total action latency:"]
    for name,r in runtime.items():lines.append(f"- {name}: p95 {r['feature_plus_onnx_p95_ms']:.3f}ms, window RTF {r['window_rtf_p95']:.5f}, scoring-hop RTF {r['hop_rtf_p95']:.5f}.")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    S.save(OUT/"FINAL_RESULT.json",dict(nominated=nominated,recommendation=recommendation,default="exp15",music_regression=music_regression,
        positive_validation_nonregression=positive_gain,test_recall_nonregression=test_nonregression,test_rate_guard=test_rate_guard,
        pi_verified=False,protected_unchanged=True))
    curate(nominated,configs)


def curate(nominated,configs):
    public=ROOT/"work/finalize_submit/repository_20261014_v1/me2-vcm"
    original=json.loads((public/"ARTIFACTS.json").read_text())
    for p,h in original["sha256"].items():
        assert S.sha(public/p)==h,f"Unrelated frozen curated artifact changed: {p}"
    evidence=public/"experiments/wakeword/final_streaming_v1";evidence.mkdir(exist_ok=False)
    staged=[]
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);staged.append(dst)
    for name in ("AUDIT.json","PLAN.md","BASELINE.json","REPORT.md","FINAL_RESULT.json","READY.json"):
        copy(OUT/name,evidence/name)
    verified=json.loads((OUT/"EXP15_CHECKPOINT_VERIFIED.json").read_text())
    assert verified["passed"]
    archive=ROOT/"../archive-me2/me2-wakeword-swift/training/experiments/exp15_mined_lp"
    assert S.sha(archive/"best.pt")==verified["original_checkpoint_sha256"]
    for name in ("best.pt","config.json","history.json","final_eval.json","onnx_export.json","latency.json"):
        if (archive/name).exists():copy(archive/name,public/"experiments/wakeword/exp15_original"/name)
    copy(OUT/"EXP15_CHECKPOINT_VERIFIED.json",public/"experiments/wakeword/exp15_original/PARITY.json")
    provenance=public/"experiments/wakeword/ARTIFACT_PROVENANCE.json"
    previous=json.loads(provenance.read_text())
    previous["prior_checkpoint_status"]=previous["original_pytorch_checkpoint"]
    previous["original_pytorch_checkpoint"]="experiments/wakeword/exp15_original/best.pt"
    previous["checkpoint_status_correction"]="2026-10-15 archived original discovered and verified vs deployed ONNX on batches1/2/17"
    previous["original_checkpoint_sha256"]=verified["original_checkpoint_sha256"]
    provenance.write_text(json.dumps(previous,indent=2)+"\n");staged.append(provenance)
    for name in ("RESULTS.json","DEVELOPMENT.json","RUNTIME.json","SELECTION_FREEZE.json","ERROR_ANALYSIS.json"):
        copy(OUT/"evaluation"/name,evidence/name)
    for arm in S.ARMS:
        for name in ("CONFIG.json","history.json","TRAIN_DONE.json","POLICY_FREEZE.json","EXPORT.json","best.pt"):
            copy(OUT/arm/name,evidence/arm/name)
        copy(OUT/("train_"+arm+".log"),public/("logs/training/wake_final_"+arm+".log"))
        copy(OUT/arm/"model.onnx",public/("models/wakeword/final_"+arm+".onnx"))
    registry=dict(default="exp15",pi_verified=False,profiles={"exp15":dict(config="configs/wakeword.yaml",model="models/wakeword/wakeword_exp15_mined_lp.onnx")})
    for name,path in configs.items():
        if name=="exp15_default":continue
        cfg=yaml.safe_load(path.read_text());model="models/wakeword/wakeword_exp15_mined_lp.onnx" if name=="exp15_calibrated" else "models/wakeword/final_"+name+".onnx"
        cfg["inference"]["onnx"]=model
        destination=public/("configs/wake_"+name+".yaml")
        with destination.open("x") as f:yaml.safe_dump(cfg,f,sort_keys=False)
        staged.append(destination);registry["profiles"][name]=dict(config=str(destination.relative_to(public)),model=model,experimental=True)
    registry["profiles"]["final"]=registry["profiles"][nominated]
    S.save(public/"configs/wake_profiles.json",registry);staged.append(public/"configs/wake_profiles.json")
    for relative in ("src/hey_swift/wakeword/inference.py","src/hey_swift/wakeword/streaming_features.py","tests/test_wake_streaming_clock.py"):
        copy(ROOT/relative,public/relative)
    copy(OUT/"benchmark_wake.py",public/"scripts/benchmark_wake.py")
    copy(ROOT/"scripts/wake_test.py",public/"scripts/wake_test.py")
    # Reproducibility code is separate from runtime; raw datasets remain private.
    for name in ("study.py","finalize.py"):
        copy(OUT/name,evidence/"recipe"/name)
    guide="""# Experimental Wakeword Pi 5 A/B

exp15 remains default. Both candidates are opt-in, not verified Pi deployments.
VCM model/calibration, wiring, account state and existing installation are unchanged.
Install/update a separate verified checkout; do not overwrite ~/swift-e2e automatically.
Use the existing documented Pi venv and dependencies in docs/PI5.md.

Selective patch application (manual; stop the runtime first):

```bash
tar -xzf wake_final_streaming_pi_candidate.tar.gz
cd wake_final_streaming_20261015_v1
sha256sum -c SHA256SUMS
python apply_candidate.py --target /PATH/TO/ai231/me2-vcm
# Only after preflight passes and the target is the intended checkout:
python apply_candidate.py --target /PATH/TO/ai231/me2-vcm --apply
```

The installer refuses unrecognized local source edits, preserves VCM/models,
hardware config, wake chime, account state and venv, and backs up every existing
changed file under the target's work/. It prints exact source rollback commands.
Do not point it at ~/swift-e2e unless that directory really contains the verified
canonical initial ai231/me2-vcm checkout; older deployments are incompatible.

From the ai231/me2-vcm checkout, with its venv active:

```bash
python scripts/reproduce.py verify
# Preferred first A/B: microphone and wake only, NO VCM/actions/calls/SMS.
python scripts/wake_test.py --config configs/wakeword.yaml --dry-run
python scripts/wake_test.py --config configs/wake_FINAL_ARM.yaml --dry-run
python scripts/wake_test.py --config configs/wakeword.yaml --device 0 --no-ack
# Ctrl+C, then candidate on the same microphone and matched playback level:
python scripts/wake_test.py --config configs/wake_FINAL_ARM.yaml --device 0 --no-ack
# Optional full assistant only after wake-only tests; use harmless commands:
python scripts/run_assistant.py --wake-profile exp15 --device 0 --live --hardware pi --speak
# Stop the previous runtime before starting B; one mic owner at a time.
python scripts/run_assistant.py --wake-profile final --device 0 --live --hardware pi --speak
# Immediate model-policy rollback, no model file copying:
python scripts/run_assistant.py --wake-profile exp15 --device 0 --live --hardware pi --speak
```

Use harmless light/timer commands, not real calls/SMS. A/B repeat quiet, distance,
speech over music and music-only/beep conditions with matched speaker/sink levels.
Record missed wakes, false events with negative exposure seconds, and timestamps.
Do not interpret event counts as recall without manually labeled phrase intervals.
For CPU p95/RTF (no actions, no microphone required):

```bash
python scripts/benchmark_wake.py --config configs/wakeword.yaml --output work/pi_exp15_cost.json
python scripts/benchmark_wake.py --config configs/wake_FINAL_ARM.yaml --output work/pi_candidate_cost.json
# Optional replay of user-owned audio; logs events, not automatic accuracy:
python scripts/benchmark_wake.py --config configs/wake_FINAL_ARM.yaml --input YOUR_RECORDING.wav --output work/pi_candidate_replay.json
```

Above scoring-hop timing includes features and ONNX, not speech endpoint/action
latency. intra/inter threads=1; actual Pi model is recorded in JSON. Physical
latency, microphone, AEC, music interference and wiring effects remain unverified.
Never recalibrate from test/development or Pi A/B observations and call it an
independent final evaluation. Keep fresh consenting speakers for confirmation.

`--wake-profile exp15` restores exp15 weights/policy but retains the corrected
integer-clock source. To roll back source as well, use the preserved fbf01a6
checkout in a separate directory; do not overwrite the working installation.

Raw wake training audio, generator assets and private Pi recordings are not
published. Recipes require the documented permitted local datasets; there is no
claim that private wake training can be fetched/retrained publicly in one command.
VCM public one-command reproduction remains unchanged.

Results and limitations: experiments/wakeword/final_streaming_v1/REPORT.md.
""".replace("FINAL_ARM",nominated)
    (public/"docs/WAKE_FINAL_AB.md").write_text(guide);staged.append(public/"docs/WAKE_FINAL_AB.md")
    readme=public/"README.md"
    with readme.open("a") as f:f.write("\n## Experimental Wakeword A/B\n\nexp15 remains the default. Two matched scratch streaming trials and an integer-sample cooldown fix are documented in [WAKE_FINAL_AB.md](docs/WAKE_FINAL_AB.md). Candidates are opt-in; physical Pi 5 verification remains pending.\n")
    # Add evidence only; do not revise old numbers or overwrite old experiment rows.
    index=public/"experiments/RUN_INDEX.csv"
    with index.open(newline="") as f:reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
    for arm in S.ARMS:
        row={k:"" for k in fields}
        for key in ("run","run_id","name","path","directory"):
            if key in row:row[key]="wake_final_streaming_v1/"+arm
        if "status" in row:row["status"]="complete experimental; Pi pending"
        policy=json.loads((OUT/arm/"POLICY_FREEZE.json").read_text())
        details=dict(role="wakeword_streaming_scratch_candidate",architecture="WakeGRU",parameters="204738",seed=str(S.SEED),
            protocol="streaming_integer_clock_v1",data_revision="private_pinned_audit_plus_disjoint_generator_ids",
            preprocessing=arm,selection_metric="worst raw/HP recall at <=5FA/hour per negative cohort",selection_split="validation",
            best_epoch=str(policy["epoch"]),available_metrics="streaming val/test; raw/AEC development; CPU parity/cost",
            provenance_sha256=S.sha(OUT/"READY.json"),checkpoint_sha256=S.sha(OUT/arm/"best.pt"),checkpoint_bytes=str((OUT/arm/"best.pt").stat().st_size),
            checkpoint_retrieval="experiments/wakeword/final_streaming_v1/"+arm+"/best.pt",disposition="experimental opt-in; exp15 default preserved",
            comparability="new arms matched to each other; historical baseline data differs; exposed synthetic-positive test")
        row.update({k:v for k,v in details.items() if k in row})
        old.append(row)
    with index.open("w",newline="") as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(old)
    for p in staged:original["sha256"][str(p.relative_to(public))]=S.sha(p)
    original["wakeword_default"]="exp15";original["wakeword_experiments"]="final_streaming_v1; Pi pending"
    (public/"ARTIFACTS.json").write_text(json.dumps(original,indent=2)+"\n")
    patchdir=ROOT/"deploy/rpi/wake_final_streaming_20261015_v1";patchdir.mkdir(exist_ok=False)
    changed=[p for p in staged if str(p.relative_to(public)).startswith(("models/","configs/","src/","scripts/","docs/WAKE"))]
    changed.append(public/"scripts/run_assistant.py")
    changed.append(public/"models/wakeword/wakeword_exp15_mined_lp.onnx")
    changed.append(public/"configs/wakeword.yaml")
    for p in changed:
        dst=patchdir/p.relative_to(public);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    shutil.copy2(OUT/"apply_candidate.py",patchdir/"apply_candidate.py")
    payload=[]
    for p in changed:
        relative=str(p.relative_to(public));gitpath="me2-vcm/"+relative
        old=subprocess.run(["git","show","HEAD:"+gitpath],cwd=public.parent,capture_output=True)
        before=__import__("hashlib").sha256(old.stdout).hexdigest() if old.returncode==0 else None
        payload.append(dict(path=relative,sha256=S.sha(p),expected_before=before))
    protected={p:S.sha(public/p) for p in ("models/vcm/selected.onnx","models/vcm/run043.onnx","configs/hardware.yaml","models/wakeword/wakeword_exp15_mined_lp.onnx")}
    S.save(patchdir/"CHANGED_FILES.json",dict(files=payload,protected=protected,default="exp15",pi_verified=False))
    (patchdir/"README.md").write_text(guide+"\nSelective overlay for compatible ai231/me2-vcm checkout ONLY. Back up listed existing files before an explicit manual apply. No installer runs automatically. Preserve local audio, configs/hardware.yaml, VCM, venv and Spotify state.\n")
    files=sorted(p for p in patchdir.rglob("*") if p.is_file())
    (patchdir/"SHA256SUMS").write_text("".join(S.sha(p)+"  "+str(p.relative_to(patchdir))+"\n" for p in files))
    archive=OUT/"wake_final_streaming_pi_candidate.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:tar.add(patchdir,arcname=patchdir.name)
    S.save(OUT/"PACKAGE.json",dict(archive=str(archive.relative_to(ROOT)),sha256=S.sha(archive),bytes=archive.stat().st_size,
        expanded_files=str(patchdir.relative_to(ROOT)),nominated=nominated,default="exp15",pi_verified=False))
    with (OUT/"patch_rehearsal.log").open("x") as log:
        subprocess.run([sys.executable,"-u","-B",str(OUT/"rehearse_patch.py")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=="__main__":main()
