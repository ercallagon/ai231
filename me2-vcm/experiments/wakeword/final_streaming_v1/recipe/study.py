"""Matched scratch streaming experiments with validation-only calibration."""
import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.signal import butter, sosfilt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"work/wake_tts3x_scratch_20261013_v1")]
from hey_swift.wakeword import features as F
from hey_swift.wakeword.streaming_features import window_features
from acoustics import vary

SEED=20261015
ARMS=("cmvn2s_v1","pcen2s_v1")
GRID=np.r_[np.arange(.10,1.,.01),.995,.999,.9999,1.000001]
FA_LIMIT=5.


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,v):
    with Path(p).open("x") as f: json.dump(v,f,indent=2);f.write("\n")


def status(stage,**kw):
    value=dict(stage=stage,pid=os.getpid(),time=time.time(),**kw)
    tmp=OUT/"status.tmp";tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(OUT/"status.json")
    print(json.dumps(value),flush=True)


def canonical(s):
    s=re.sub(r"^piper[-_:]\d+[-_:]","piper-",s)
    return s.replace("MultilingualNeural","Neural").replace("ExpressiveNeural","Neural").replace("_","-")


def stress(seed,n=32000):
    rng=np.random.default_rng(seed);t=np.arange(n)/16000
    base=float(rng.uniform(55,2400));sweep=float(rng.uniform(-.5,.5))
    carrier=sum(np.sin(2*np.pi*(base*k*t+sweep*base*t*t/2)+rng.uniform(0,6.28))/k for k in (1,2,3))
    rate=float(rng.uniform(.5,8));duty=float(rng.uniform(.1,.7))
    envelope=np.maximum(0,np.sin(2*np.pi*rate*t))**2
    if rng.random()<.5: envelope=(np.mod(t*rate,1)<duty).astype(float)
    x=carrier*envelope + rng.normal(0,.03,n)
    x*=10**rng.uniform(-30,-7)/max(float(np.sqrt(np.mean(x*x))),1e-9)
    return (x/max(float(np.abs(x).max())/.95,1.)).astype(np.float32)


def read(row):
    p=ROOT/row["path"]
    if sha(p)!=row["audio_sha256"]:raise RuntimeError(f"Audio changed: {p}")
    return F.load_wav_mono_16k(p)


def prepare():
    folder=OUT/"data";folder.mkdir()
    audit=json.loads((ROOT/"work/wakeword_targeted_v2/audit.json").read_text())
    reserved=audit["split_rows"]["validation"]+audit["split_rows"]["test"]
    voices={canonical(r["voice_id"]) for r in reserved if r["label"]==0}
    pcm_reserved={r["pcm_sha256"] for r in reserved}
    originals=audit["split_rows"]["train"]
    by_name={Path(r["relpath"]).name:r for r in originals}
    candidates=[];exclusions=[]
    for row in originals:
        r=dict(row,path=str(Path("data/wakeword/raw")/row["relpath"]))
        voice=r["voice_id"]
        if r["source"]=="confuser":
            base=re.sub(r"_dsp_v\d+","",Path(r["relpath"]).name)
            voice=by_name.get(base,r)["voice_id"].split(":",1)[-1]
        reason=None
        if r["source"] not in ("tts","confuser"):
            reason="unverifiable speaker/source ancestry; do not assume filename is speaker"
        elif canonical(voice) in voices:reason="canonical voice family occurs in reserved metadata"
        elif not voice or voice.endswith(":"):reason="unresolved confuser parent voice"
        elif r["pcm_sha256"] in pcm_reserved:reason="reserved PCM overlap"
        if reason:exclusions.append(dict(path=r["path"],reason=reason))
        else:r["canonical_voice"]=canonical(voice);candidates.append(r)
    previous=ROOT/"work/wake_tts3x_scratch_20261013_v1"
    generated=json.loads((previous/"GENERATED_ROWS.json").read_text())
    added=[r for r in generated if r["split"]=="train_added"]
    added+=json.loads((ROOT/"work/wake_gender25_20261014_v1/GENERATED_ROWS.json").read_text())
    dev_ids=set(json.loads((previous/"DATA_VALIDATION.json").read_text())["dev_generator_ids"])
    for r in added:
        assert r["speaker_id"] not in dev_ids and r["pcm_sha256"] not in pcm_reserved
        candidates.append(dict(r,label=0,source="tts_added",canonical_voice=f"libritts-r:{r['speaker_id']}"))
    eligible=[];seen=set();spans=[]
    for r in candidates:
        wave=read(r);digest=hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest()
        if digest!=r["pcm_sha256"]:raise RuntimeError("PCM changed")
        reason=None
        if digest in seen:reason="within-training exact PCM duplicate"
        if r["label"]==0 and (len(wave)>32000 or np.max(np.abs(wave),initial=0)<.001):
            reason="positive original waveform exceeds2s or no reliable signal; never truncate"
        if reason:exclusions.append(dict(path=r["path"],reason=reason));continue
        seen.add(digest);eligible.append(r);spans.append(len(wave)/16000)
    train_voices={r["canonical_voice"] for r in eligible}
    assert not train_voices & voices
    save(OUT/"AUDIT.json",dict(train_input=len(originals)+len(added),train_retained=len(eligible),
        positive=sum(r["label"]==0 for r in eligible),negative=sum(r["label"]==1 for r in eligible),
        exclusions=exclusions,known_reserved_voice_overlap=[],known_reserved_pcm_overlap=[],
        source_terms="Existing private training only; no raw audio is published",
        limits="No complete human ancestry proof across independent TTS engines; historical exp15 saw overlapping voice families. Unknown negative ancestry excluded, not guessed.",
        input_duration_quantiles=np.quantile(spans,[.5,.9,.99,1]).tolist()))
    save(OUT/"TRAIN_ROWS.json",eligible)
    total=len(eligible)+2048
    caches={(arm,kind):np.lib.format.open_memmap(folder/f"train_{arm}_{kind}.npy",mode="w+",dtype=np.float32,shape=(total,80,200))
            for arm in ARMS for kind in ("clean","variant")}
    labels=[];records=[]
    for i in range(total):
        r=eligible[i] if i<len(eligible) else None
        y=int(r["label"]) if r else 1;labels.append(y)
        waves=[];settings=[]
        for view in range(2):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,i,view]))
            if r:
                raw=read(r)
                if len(raw)>32000:
                    a=int(rng.integers(0,len(raw)-32000+1));wave=raw[a:a+32000];left=0
                else:
                    a=0;left=int(rng.integers(0,32000-len(raw)+1));wave=np.pad(raw,(left,32000-len(raw)-left))
            else:
                a=left=0;wave=stress(SEED*100000+i)
            if view:
                wave,effects=vary(wave,rng);lead=effects["leading_pad_samples"];wave=wave[lead:lead+32000]
                if rng.random()<.5:
                    wave=sosfilt(butter(3,float(rng.uniform(50,180)),"highpass",fs=16000,output="sos"),wave).astype(np.float32)
                wave+=stress(SEED*100000+i+99123)*float(rng.uniform(.02,.15))
                wave/=max(float(np.abs(wave).max())/.98,1.)
            waves.append(wave.astype(np.float32));settings.append(dict(crop_start=a,padding_left=left))
        for arm in ARMS:
            for kind,wave in zip(("clean","variant"),waves):caches[arm,kind][i]=window_features(wave,arm)[0]
        records.append(dict(index=i,label=y,source=r["path"] if r else "generated_hard_negative",views=settings,
                            pcm_hashes=[hashlib.sha256(w.tobytes()).hexdigest() for w in waves]))
        if i%500==0:print("training windows",i,"/",total,flush=True)
    for cache in caches.values():cache.flush()
    np.save(folder/"train_labels.npy",np.array(labels,dtype=np.int64))
    save(OUT/"WINDOWS.json",records)
    for split in ("validation",):prepare_streams(split,audit["split_rows"][split])
    protected=[ROOT/"models/wakeword/wakeword_exp15_mined_lp.onnx",ROOT/"configs/wakeword.yaml",
        ROOT/"work/finalize_submit/repository_20261014_v1/me2-vcm/models/vcm/selected.onnx",
        ROOT/"work/finalize_submit/repository_20261014_v1/me2-vcm/models/vcm/run043.onnx"]
    files=[Path(__file__),OUT/"PLAN.md",ROOT/"src/hey_swift/wakeword/inference.py",ROOT/"src/hey_swift/wakeword/streaming_features.py",ROOT/"src/hey_swift/wakeword/model.py",OUT/"AUDIT.json",OUT/"TRAIN_ROWS.json",OUT/"WINDOWS.json"]+list(folder.glob("*.npy"))+list(folder.glob("*.json"))
    save(OUT/"READY.json",dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},
        protected={str(p.relative_to(ROOT)):sha(p) for p in protected},python=sys.executable,
        environment={n:importlib.metadata.version(n) for n in ("torch","numpy","scipy","onnxruntime")},
        curated_git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT/"work/finalize_submit/repository_20261014_v1",text=True).strip()))


def prepare_streams(split,rows):
    folder=OUT/"data";meta={};segments={"positive":[],"negative":[]};intervals=[];offset=32000
    segments["positive"].append(np.zeros(32000,dtype=np.float32))
    for r in rows:
        row=dict(r,path=str(Path("data/wakeword/raw")/r["relpath"]))
        wave=read(row)
        if r["label"]==0:
            power=wave[:len(wave)//160*160].reshape(-1,160).astype(np.float64)**2
            energy=power.mean(1);active=np.flatnonzero(energy>max(float(energy.max())*.01,1e-8))
            endpoint=offset+(int(active[-1])+1)*160 if len(active) else offset+len(wave)
            intervals.append(dict(start=offset,end=offset+len(wave),energy_end=endpoint,path=row["path"],failure=not bool(len(active))))
            segments["positive"].extend([wave,np.zeros(48000,dtype=np.float32)]);offset+=len(wave)+48000
        else:segments["negative"].append(wave)
    # Supplementary generated stress has disjoint deterministic seeds, no Pi data.
    segments["negative"].extend(stress(91000000+(0 if split=="validation" else 100000)+i,n=64000) for i in range(256))
    for kind,parts in segments.items():
        raw=np.concatenate(parts)
        for condition in ("raw","hp80"):
            wave=raw if condition=="raw" else sosfilt(butter(3,80,"highpass",fs=16000,output="sos"),raw).astype(np.float32)
            for frontend,win in (("legacy",16000),*( (a,32000) for a in ARMS)):
                ends=np.arange(win,len(wave)+1,4000)
                name=f"{split}_{kind}_{condition}_{frontend}"
                cache=np.lib.format.open_memmap(folder/(name+".npy"),mode="w+",dtype=np.float32,shape=(len(ends),80,100 if frontend=="legacy" else 200))
                levels=[]
                for i,end in enumerate(ends):
                    w=wave[end-win:end]
                    cache[i]=(F.mel_from_pcm(w) if frontend=="legacy" else window_features(w,frontend))[0]
                    levels.append(float(np.sqrt(np.mean(w*w))))
                cache.flush()
                save(folder/(name+".json"),dict(samples=ends.tolist(),rms=levels,total_samples=len(wave),
                     negative_hours=max(0,len(wave)-win)/16000/3600,intervals=intervals if kind=="positive" else [],
                     actual_recording_seconds=sum(len(w) for w in parts[:len([r for r in rows if r['label']==1])])/16000 if kind=="negative" else None,
                     synthetic_negative_seconds=1024 if kind=="negative" else 0,
                     latency_reference="waveform end and automated energy end; not manually verified speech offset"))
                print("stream cache",name,len(ends),flush=True)


def verify():
    ready=json.loads((OUT/"READY.json").read_text())
    for group in ("files","protected"):
        for p,h in ready[group].items():
            if sha(ROOT/p)!=h:raise RuntimeError(f"Frozen artifact changed: {p}")


def events(p,meta,tau):
    p=np.asarray(p,dtype=np.float64)
    q=(p>=tau)&(1-p<.90)&(20*np.log10(np.asarray(meta["rms"],dtype=np.float64)+1e-9)>-60.)
    candidates=np.flatnonzero(q & np.r_[False,q[:-1]])
    selected=[];last=-10**15
    for i in candidates:
        sample=meta["samples"][i]
        if sample-last>=32000:selected.append(sample);last=sample
    return selected


@lru_cache(maxsize=64)
def stream_metadata(path):
    return json.loads(Path(path).read_text())


def stream_metric(probabilities,frontend,split,tau,level_overrides=None):
    result={}
    for cond in ("raw","hp80"):
        metadata={k:dict(stream_metadata(str(OUT/f"data/{split}_{k}_{cond}_{frontend}.json"))) for k in ("positive","negative")}
        if level_overrides is not None:
            for kind in metadata:metadata[kind]["rms"]=level_overrides[kind,cond]
        neg=events(probabilities["negative",cond],metadata["negative"],tau)
        pos=events(probabilities["positive",cond],metadata["positive"],tau)
        pos_array=np.array(pos,dtype=np.int64)
        matches=[];energy_latency=[];wave_latency=[];used=set()
        for r in metadata["positive"]["intervals"]:
            j=int(np.searchsorted(pos_array,r["start"]))
            if j<len(pos) and pos[j]<=r["end"]+16000 and j not in used and not r["failure"]:
                e=pos[j];used.add(j);matches.append(True)
                energy_latency.append((e-r["energy_end"])/16)
                wave_latency.append((e-r["end"])/16)
            else:matches.append(False)
        hours=metadata["negative"]["negative_hours"]
        original_seconds=metadata["negative"]["actual_recording_seconds"]
        first_sample=metadata["negative"]["samples"][0]
        original_hours=max(0,original_seconds-first_sample/16000)/3600
        generated_hours=metadata["negative"]["synthetic_negative_seconds"]/3600
        original_events=sum(e<=round(original_seconds*16000) for e in neg)
        generated_events=len(neg)-original_events
        result[cond]=dict(recall=sum(matches)/max(len(matches),1),detected=sum(matches),positive_n=len(matches),
            misses=len(matches)-sum(matches),negative_events=len(neg),negative_hours=hours,
            false_events_per_hour=len(neg)/hours,zero_event_95_upper_per_hour=2.995732/hours if not neg else None,
            original_negative_events=original_events,original_negative_hours=original_hours,
            original_false_events_per_hour=original_events/original_hours,
            original_zero_event_95_upper_per_hour=2.995732/original_hours if not original_events else None,
            generated_negative_events=generated_events,generated_negative_hours=generated_hours,
            generated_false_events_per_hour=generated_events/generated_hours,
            generated_zero_event_95_upper_per_hour=2.995732/generated_hours if not generated_events else None,
            unmatched_positive_stream_events=len(pos)-len(used),
            energy_endpoint_latency_p95_ms=float(np.percentile(energy_latency,95)) if energy_latency else None,
            waveform_end_latency_p95_ms=float(np.percentile(wave_latency,95)) if wave_latency else None)
    return result


def choose(probabilities,frontend,split="validation"):
    best=None;table=[]
    for threshold in GRID:
        metrics=stream_metric(probabilities,frontend,split,float(threshold))
        fa=max(m[k] for m in metrics.values() for k in ("false_events_per_hour","original_false_events_per_hour","generated_false_events_per_hour"))
        score=(min(m["recall"] for m in metrics.values()),sum(m["recall"] for m in metrics.values())/2,
               -sum(m["unmatched_positive_stream_events"] for m in metrics.values()),-fa)
        row=dict(threshold=float(threshold),metrics=metrics,feasible=fa<=FA_LIMIT,score=score)
        table.append(row)
        if row["feasible"] and (best is None or score>tuple(best["score"])):best=row
    assert best is not None
    return best,table


def probabilities(model,frontend,device="cpu",split="validation"):
    import torch
    result={};model.eval()
    with torch.no_grad():
        for kind in ("positive","negative"):
            for cond in ("raw","hp80"):
                x=np.load(OUT/f"data/{split}_{kind}_{cond}_{frontend}.npy",mmap_mode="r");parts=[]
                for i in range(0,len(x),256):
                    batch=torch.from_numpy(np.array(x[i:i+256])).to(device)
                    parts.extend(model(batch).softmax(1)[:,0].cpu().numpy())
                result[kind,cond]=np.array(parts)
    return result


def baseline():
    import onnxruntime as ort
    verify();opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(ROOT/"models/wakeword/wakeword_exp15_mined_lp.onnx"),opts,providers=["CPUExecutionProvider"])
    probs={}
    for kind in ("positive","negative"):
        for cond in ("raw","hp80"):
            x=np.load(OUT/f"data/validation_{kind}_{cond}_legacy.npy",mmap_mode="r");parts=[]
            for i in range(0,len(x),128):
                logits=sess.run(None,{sess.get_inputs()[0].name:np.array(x[i:i+128])})[0];e=np.exp(logits-logits.max(1,keepdims=True));parts.extend((e/e.sum(1,keepdims=True))[:,0])
            probs[kind,cond]=np.array(parts)
    selected,table=choose(probs,"legacy")
    save(OUT/"BASELINE.json",dict(selected=selected,grid=table,fixed030=stream_metric(probs,"legacy","validation",.3),
        scope="single most recent window, integer clock; calibrated threshold is NOT deployed; historical model source overlap disclosed"))
    print("baseline selected",json.dumps(selected),flush=True)


def train(arm):
    import torch
    from torch.utils.data import DataLoader,Dataset
    from hey_swift.wakeword.model import build_model,count_params
    verify();assert (OUT/"BASELINE.json").exists()
    if not torch.cuda.is_available():raise RuntimeError("GPU unavailable; no CPU fallback")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.25)
    folder=OUT/arm;folder.mkdir()
    model=build_model().to("cuda");initial=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    if arm==ARMS[1]:assert initial==json.loads((OUT/ARMS[0]/"CONFIG.json").read_text())["initial_sha256"]
    save(folder/"CONFIG.json",dict(arm=arm,params=count_params(model),seed=SEED,initial_sha256=initial,initialization="random, no checkpoint loaded",
        epochs_max=80,minimum_epochs=40,patience=20,batch=128,optimizer="AdamW",lr=.0003,weight_decay=.0001,
        selection="validation streaming:<=5FA/h raw/HP80, max worst recall then mean recall then unmatched events",gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),baseline_sha256=sha(OUT/"BASELINE.json")))
    class Data(Dataset):
        def __init__(self):
            self.clean=np.load(OUT/f"data/train_{arm}_clean.npy",mmap_mode="r")
            self.variant=np.load(OUT/f"data/train_{arm}_variant.npy",mmap_mode="r")
            self.labels=np.load(OUT/"data/train_labels.npy");self.epoch=0
        def __len__(self):return len(self.labels)
        def __getitem__(self,i):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,self.epoch,i]))
            return torch.from_numpy(np.array((self.variant if rng.random()<.5 else self.clean)[i])),int(self.labels[i])
    data=Data();loader=DataLoader(data,batch_size=128,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80)
    history=[];best=None;bad=0
    for epoch in range(80):
        start=time.monotonic();data.epoch=epoch;model.train();total=right=0;loss_sum=0.
        for x,y in loader:
            x,y=x.to("cuda"),y.to("cuda");optimizer.zero_grad(set_to_none=True);logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=len(y);right+=int((logits.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
        scheduler.step();probs=probabilities(model,arm,"cuda");selected,_=choose(probs,arm)
        row=dict(epoch=epoch+1,train_loss=loss_sum/total,augmented_train_accuracy=right/total,selected=selected,
                 seconds=time.monotonic()-start,lr=scheduler.get_last_lr()[0],peak_reserved_bytes=torch.cuda.max_memory_reserved())
        history.append(row)
        checkpoint=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,
            seed=SEED,policy=selected,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=loader.generator.get_state())
        torch.save(checkpoint,folder/"last.pt")
        if best is None or tuple(selected["score"])>tuple(best):
            best=selected["score"];bad=0;torch.save(checkpoint,folder/"best.pt")
        else:bad+=1
        with (folder/"history.json").open("w") as f:json.dump(history,f,indent=2)
        print(json.dumps(row),flush=True)
        if epoch+1>=40 and bad>=20:break
    save(folder/"TRAIN_DONE.json",dict(epochs=len(history),best_checkpoint_sha256=sha(folder/"best.pt"),test_accessed=False))


def export(arm):
    import torch
    import onnxruntime as ort
    from hey_swift.wakeword.model import build_model
    torch.set_num_threads(1);folder=OUT/arm
    checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=False)
    model=build_model();model.load_state_dict(checkpoint["state_dict"]);model.eval()
    save(folder/"POLICY_FREEZE.json",dict(checkpoint_sha256=sha(folder/"best.pt"),frontend=arm,policy=checkpoint["policy"],epoch=checkpoint["epoch"],test_accessed=False))
    torch.onnx.export(model,torch.zeros(1,80,200),str(folder/"model.onnx"),input_names=["features"],output_names=["logits"],
        dynamic_axes={"features":{0:"batch"},"logits":{0:"batch"}},opset_version=17,dynamo=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    runtime=ort.InferenceSession(str(folder/"model.onnx"),opts,providers=["CPUExecutionProvider"])
    for batch in (1,2,17):
        x=np.random.default_rng(batch).normal(size=(batch,80,200)).astype(np.float32)
        actual=runtime.run(None,{"features":x})[0]
        with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
    save(folder/"EXPORT.json",dict(onnx_sha256=sha(folder/"model.onnx"),parity_batches=[1,2,17],pi_verified=False))


def controller():
    import fcntl
    with (OUT/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
        def run(mode,arm=None,gpu=""):
            command=[sys.executable,"-u","-B",str(Path(__file__)),mode]+(["--arm",arm] if arm else [])
            status(mode,arm=arm,gpu=gpu,command=command)
            with (OUT/(mode+("_"+arm if arm else "")+".log")).open("x") as log:
                subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            run("prepare");run("baseline")
            for arm in ARMS:
                def available():
                    text=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],text=True,timeout=20)
                    return [r[0] for r in [tuple(map(int,line.split(","))) for line in text.splitlines()] if r[1]<40 and r[3]-r[2]>=12288]
                deadline=time.monotonic()+48*3600
                while time.monotonic()<deadline:
                    ids=available()
                    if ids:
                        time.sleep(5)
                        if ids[0] in available():break
                    status("waiting_gpu",arm=arm);time.sleep(30)
                else:raise TimeoutError("No safe GPU within48h")
                run("train",arm,ids[0]);run("export",arm)
            status("training_complete",next="frozen CPU comparison/curation; no deployment")
        except Exception as error:
            status("failed",error=repr(error));raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["prepare","baseline","train","export","controller","detach"])
    parser.add_argument("--arm",choices=ARMS);args=parser.parse_args();os.chdir(ROOT)
    if args.mode=="detach":
        with (OUT/"queue.log").open("x") as log:
            child=subprocess.Popen([sys.executable,"-u","-B",str(Path(__file__)),"controller"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(OUT/"PID.json",dict(pid=child.pid));print(child.pid)
    elif args.mode in ("train","export"):globals()[args.mode](args.arm)
    else:globals()[args.mode]()
