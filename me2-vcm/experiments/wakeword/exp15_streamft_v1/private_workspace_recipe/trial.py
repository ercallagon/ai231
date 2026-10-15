"""One runtime-window fine-tune of original exp15; all outputs are versioned."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
from scipy.signal import butter,resample_poly,sosfilt
from scipy.stats import chi2

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/"work/wake_final_streaming_20261015_v1"
PUBLIC=ROOT/"work/finalize_submit/repository_20261014_v1/me2-vcm"
ARCHIVE=ROOT.parent/"archive-me2/me2-wakeword-swift"
CHECKPOINT=ARCHIVE/"training/experiments/exp15_mined_lp/best.pt"
MODEL=ROOT/"models/wakeword/wakeword_exp15_mined_lp.onnx"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"work/wake_tts3x_scratch_20261013_v1")]
from hey_swift.wakeword import features as F
from hey_swift.audio.capture import ContinuousResampler
SEED=20261017
GRID=np.r_[np.arange(.10,1.,.01),.995,.999,.9999,1.000001]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with Path(path).open("x") as h:json.dump(value,h,indent=2,allow_nan=False);h.write("\n")


def ort_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])


def events(p,meta,tau):
    q=(p>=tau)&(1-p<.9)&(20*np.log10(np.asarray(meta["rms"])+1e-9)>-60)
    streak=0;last=-10**15;result=[]
    for i in np.flatnonzero(q):
        streak=streak+1 if i and q[i-1] else 1
        if streak>=2 and meta["samples"][i]-last>=32000:
            last=meta["samples"][i];result.append(last);streak=0
    return result


@lru_cache(maxsize=16)
def metadata(split,kind,cond):
    return json.loads((OLD/f"data/{split}_{kind}_{cond}_legacy.json").read_text())


def metrics(scores,split,tau,detail=False):
    result={}
    for cond in ("raw","hp80"):
        pm=metadata(split,"positive",cond);nm=metadata(split,"negative",cond)
        pe=events(scores["positive",cond],pm,tau);ne=events(scores["negative",cond],nm,tau)
        matched=[];times=[];lat=[];used=set()
        for r in pm["intervals"]:
            j=int(np.searchsorted(pe,r["start"]))
            ok=j<len(pe) and pe[j]<=r["end"]+16000 and j not in used and not r["failure"]
            matched.append(bool(ok));times.append(int(pe[j]) if ok else None)
            if ok:used.add(j);lat.append((pe[j]-r["energy_end"])/16)
        cutoff=round(nm["actual_recording_seconds"]*16000)
        n=int(sum(e<=cutoff for e in ne));g=len(ne)-n
        hours=(cutoff-16000)/16000/3600;gh=nm["synthetic_negative_seconds"]/3600
        row=dict(detected=sum(matched),n=len(matched),recall=sum(matched)/len(matched),
            original_events=n,original_hours=hours,original_fa_h=n/hours,
            generated_events=g,generated_hours=gh,generated_fa_h=g/gh,
            unmatched_positive_events=len(pe)-len(used),
            latency_median_ms=float(np.median(lat)) if lat else None,
            latency_p95_ms=float(np.percentile(lat,95)) if lat else None)
        if detail:row.update(matched=matched,times=times,positive_events=pe,negative_events=ne,
            original_fa_h_ci95=[float(chi2.ppf(.025,2*n)/(2*hours)) if n else 0.,float(chi2.ppf(.975,2*(n+1))/(2*hours))])
        result[cond]=row
    return result


def maxrate(m):return max(v[k] for v in m.values() for k in ("original_fa_h","generated_fa_h"))


def curve(scores,split="validation"):
    return [dict(threshold=float(t),metrics=metrics(scores,split,float(t))) for t in GRID]


def rank(row):
    m=row["metrics"];lat=max(v["latency_p95_ms"] if v["latency_p95_ms"] is not None else 1e9 for v in m.values())
    return (min(v["recall"] for v in m.values()),sum(v["recall"] for v in m.values())/2,-maxrate(m),-lat)


def choose(rows,budget):return max([r for r in rows if maxrate(r["metrics"])<=budget+1e-8],key=rank)


def selection(rows,budget):
    primary=choose(rows,budget)
    secondary=np.mean([rank(choose(rows,b))[0] for b in (5.,15.,30.,60.,120.)])
    r=rank(primary)
    return primary,(r[0],r[1],float(secondary),r[2],r[3])


def cached_scores(split="validation"):
    data=np.load(OLD/f"evaluation/exp15_calibrated_{split}.npz")
    return {(k,c):data[k+"_"+c] for k in ("positive","negative") for c in ("raw","hp80")}


def prepare():
    import torch
    from hey_swift.wakeword.model import build_model,count_params
    torch.set_num_threads(1)
    assert sha(CHECKPOINT)=="d6d9069df46f01f1ed44a014d34cd37150b210e38a4428423f7e41b73c6254b0"
    model=build_model();model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=False));model.eval()
    assert count_params(model)==204738
    runtime=ort_session(MODEL);errors=[]
    for n in (1,2,17):
        x=np.random.default_rng(n).normal(size=(n,80,100)).astype(np.float32)
        with torch.no_grad():expected=model(torch.from_numpy(x)).numpy()
        actual=runtime.run(None,{runtime.get_inputs()[0].name:x})[0]
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5);errors.append(float(np.max(np.abs(actual-expected))))
    source=OLD/"TRAIN_ROWS.json";frozen=json.loads((OLD/"READY.json").read_text())
    assert sha(source)==frozen["files"][str(source.relative_to(ROOT))]
    audit=json.loads((OLD/"AUDIT.json").read_text())
    assert not audit["known_reserved_voice_overlap"] and not audit["known_reserved_pcm_overlap"]
    original=json.loads(source.read_text());rows=[];waves=[];excluded=[]
    for i,r in enumerate(original):
        p=ROOT/r["path"]
        if sha(p)!=r["audio_sha256"]:raise ValueError("Changed source: "+r["path"])
        wave=F.load_wav_mono_16k(p)
        assert hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest()==r["pcm_sha256"]
        if r["label"]==0 and (len(wave)>16000 or float(np.sqrt(np.mean(wave*wave)))<.002):
            excluded.append(dict(path=r["path"],samples=len(wave),reason="Complete original does not fit1s or weak signal; never labeled negative"));continue
        rows.append(dict(r,samples=len(wave),offset=sum(w.size for w in waves)));waves.append(wave)
        if i%1000==0:print("verified training rows",i,flush=True)
    pcm=np.lib.format.open_memmap(OUT/"training_pcm.npy",mode="w+",dtype=np.float32,shape=(sum(w.size for w in waves),))
    for row,wave in zip(rows,waves):pcm[row["offset"]:row["offset"]+len(wave)]=wave
    pcm.flush()
    positives=[i for i,r in enumerate(rows) if r["label"]==0];negatives=[i for i,r in enumerate(rows) if r["label"]==1]
    save(OUT/"TRAIN_ROWS.json",rows)
    save(OUT/"DATA_AUDIT.json",dict(positive_n=len(positives),negative_n=len(negatives),exclusions=excluded,
        training_source_sha256=sha(source),known_train_reserved_pcm_overlap=[],known_train_reserved_voice_overlap=[],
        limits="Inherited exp15 historical exposure remains. Unknown negative ancestry previously excluded. Official positives TTS;17canonical voice families shared by val/test. Full word retention not manually verified."))
    protected=[MODEL,ROOT/"configs/wakeword.yaml",PUBLIC/"models/vcm/selected.onnx",PUBLIC/"models/vcm/run043.onnx",
        PUBLIC/"configs/hardware.yaml",OLD/"cmvn2s_v1/best.pt",OLD/"pcen2s_v1/best.pt"]
    save(OUT/"PROVENANCE.json",dict(checkpoint_sha256=sha(CHECKPOINT),export_sha256=sha(MODEL),parity_batches=[1,2,17],
        max_abs_errors=errors,selected_epoch_from_log=42,incorrect_archived_config_best_epoch=60,
        source_sha256=sha(__file__),plan_sha256=sha(OUT/"PLAN.md"),
        protected={str(p.relative_to(ROOT)):sha(p) for p in protected},
        pcm_sha256=sha(OUT/"training_pcm.npy"),python=sys.executable,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PUBLIC,text=True).strip(),
        test_accessed=False,pi_verified=False))
    base=cached_scores();reference=metrics(base,"validation",.30)
    budget=maxrate(reference);rows_curve=curve(base)
    save(OUT/"BASELINE_FREEZE.json",dict(primary_budget=budget,shared_tau030=reference,
        selected_primary=choose(rows_curve,budget),curves=rows_curve,
        selection="latest1s/hop250ms/sustain2 fresh scores/cooldown2s; validation only",test_accessed=False))
    data=Data()
    scores=[]
    for j,i in enumerate(data.negative):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,991,i]));w=data.wave(i)
        if len(w)>16000:
            a=int(rng.integers(len(w)-16000+1));w=w[a:a+16000]
        else:
            left=int(rng.integers(16000-len(w)+1));w=np.pad(w,(left,16000-len(w)-left))
        z=runtime.run(None,{runtime.get_inputs()[0].name:F.mel_from_pcm(w)})[0][0]
        e=np.exp(z-z.max());scores.append(float(e[0]/e.sum()))
        if j%1000==0:print("training-only hard mining",j,flush=True)
    ix=np.argsort(scores)[-max(1,len(scores)//4):]
    save(OUT/"MINING.json",dict(hard_indices=[data.negative[int(j)] for j in ix],negative_indices=data.negative,
        scores=scores,source="training only; one deterministic source window per negative, not evaluation"))


def label_window(start,end,wake_start=None,wake_end=None):
    if wake_start is None:return 1
    if start<=wake_start and end>=wake_end:return 0
    if end<=wake_start or start>=wake_end:return 1
    return -100


class Data:
    def __init__(self):
        self.rows=json.loads((OUT/"TRAIN_ROWS.json").read_text())
        self.pcm=np.load(OUT/"training_pcm.npy",mmap_mode="r")
        self.positive=[i for i,r in enumerate(self.rows) if r["label"]==0]
        self.negative=[i for i,r in enumerate(self.rows) if r["label"]==1]
        self.hard=json.loads((OUT/"MINING.json").read_text())["hard_indices"] if (OUT/"MINING.json").exists() else self.negative
        self.specs=[]
    def wave(self,i):
        r=self.rows[i];return self.pcm[r["offset"]:r["offset"]+r["samples"]]
    def make_specs(self,epoch,n=8192):
        specs=[]
        for i in range(n):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,epoch,i]))
            label=i%2
            pool=self.positive if label==0 else (self.hard if rng.random()<.5 else self.negative)
            source=int(pool[int(rng.integers(len(pool)))])
            bg=[];size=0
            while size<80000:
                k=int(self.negative[int(rng.integers(len(self.negative)))]);bg.append(k);size+=self.rows[k]["samples"]
            end=int(rng.integers(4,13))*4000
            left=int(rng.integers(16000-min(16000,self.rows[source]["samples"])+1))
            specs.append(dict(index=i,epoch=epoch,label=label,source=source,background=bg,end=end,left=left,
                              capture_48k=bool(rng.random()<.5),seed=[SEED,epoch,i,777],
                              tonal_negative=label==1 and bool(rng.random()<.10)))
            if label==0:
                ws=end-16000+left;we=ws+self.rows[source]["samples"]
                specs[-1]["neighbor_labels"]={str(k):label_window(end-16000+k*4000,end+k*4000,ws,we) for k in (-2,-1,0,1,2)}
        return specs
    def render(self,spec):
        rng=np.random.default_rng(np.random.SeedSequence(spec["seed"]))
        bg=np.concatenate([self.wave(i) for i in spec["background"]]);a=int(rng.integers(len(bg)-64000+1));bg=bg[a:a+64000].copy()
        source=self.wave(spec["source"]);end=spec["end"];start=end-16000
        if len(source)>16000:
            a=int(rng.integers(len(source)-16000+1));source=source[a:a+16000]
        source=source*10**float(rng.uniform(-6,3)/20)
        source_rms=float(np.sqrt(np.mean(source*source)+1e-12))
        bg*=source_rms*10**(-float(rng.uniform(18,30))/20)/max(float(np.sqrt(np.mean(bg*bg))),1e-9)
        left=spec["left"];wake_start=start+left;wake_end=wake_start+len(source)
        bg[wake_start:wake_end]+=source
        if spec["tonal_negative"]:
            t=np.arange(64000)/16000;freq=float(rng.uniform(100,3000))
            tone=np.sin(2*np.pi*freq*t)*np.maximum(0,np.sin(2*np.pi*float(rng.uniform(1,7))*t))
            bg+=tone.astype(np.float32)*float(rng.uniform(.002,.03))
        if rng.random()<.2:bg=sosfilt(butter(2,float(rng.uniform(3000,5000)),fs=16000,output="sos"),bg).astype(np.float32)
        bg/=max(float(np.abs(bg).max())/.98,1.)
        if spec["capture_48k"]:
            native=resample_poly(bg,3,1).astype(np.float32);rs=ContinuousResampler(48000)
            bg=np.concatenate([rs.push(native[i:i+4800]) for i in range(0,len(native),4800)])
        assert len(bg)==64000
        if spec["label"]==0:assert label_window(start,end,wake_start,wake_end)==0
        wave=bg[start:end].astype(np.float32)
        return wave,F.mel_from_pcm(wave)[0],spec["label"]
    def __len__(self):return len(self.specs)
    def __getitem__(self,i):
        import torch
        _,x,y=self.render(self.specs[i]);return torch.from_numpy(x),y


def gpu_available():
    text=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],text=True,timeout=20)
    rows=[list(map(int,l.split(","))) for l in text.splitlines()]
    return rows,[r[0] for r in rows if r[1]<40 and r[3]-r[2]>=12288]


def train():
    folder=OUT/"training";folder.mkdir()
    deadline=time.monotonic()+1200;observations=[];gpu=None
    while time.monotonic()<deadline:
        first,ids=gpu_available();time.sleep(5);second,again=gpu_available()
        observations.append(dict(first=first,second=second,time=time.time()))
        safe=[i for i in ids if i in again]
        if safe:gpu=safe[0];break
        print("waiting for safe single GPU",flush=True);time.sleep(30)
    save(folder/"GPU_GATE.json",observations)
    if gpu is None:
        save(folder/"BLOCKED.json",dict(reason="No safe GPU within20min; no queue retained"));return
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    import torch
    from torch.utils.data import DataLoader
    from hey_swift.wakeword.model import build_model
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.15)
    model=build_model().cuda();model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=False))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    baseline=json.loads((OUT/"BASELINE_FREEZE.json").read_text());budget=baseline["primary_budget"]
    _,best_rank=selection(baseline["curves"],budget);best_epoch=0;bad=0;history=[]
    best_finetuned_rank=None;best_finetuned_epoch=0
    arrays={(k,c):np.load(OLD/f"data/validation_{k}_{c}_legacy.npy",mmap_mode="r") for k in ("positive","negative") for c in ("raw","hp80")}
    started=time.monotonic();deadline=started+1200
    save(folder/"CONFIG.json",dict(initial_checkpoint_sha256=sha(CHECKPOINT),architecture="unchanged WakeGRU204738/1s",
        lr=1e-5,optimizer="AdamW",weight_decay=1e-4,max_epochs=6,patience=2,maximum_gpu_seconds=1200,
        examples_per_epoch=8192,batch=128,workers=2,seed=SEED,bn_statistics="frozen; affine trainable",gpu=gpu,
        primary_budget=budget,source_sha256=sha(__file__),test_accessed=False))
    data=Data()
    for epoch in range(1,7):
        data.specs=data.make_specs(epoch)
        save(folder/f"epoch_{epoch:02d}_windows.json",data.specs)
        replay=[]
        for spec in data.specs[:8]:
            w,_,_=data.render(spec);replay.append(dict(index=spec["index"],pcm_sha256=hashlib.sha256(w.tobytes()).hexdigest()))
        save(folder/f"epoch_{epoch:02d}_replay.json",replay)
        loader=DataLoader(data,batch_size=128,shuffle=True,num_workers=2,
            generator=torch.Generator().manual_seed(SEED+epoch),pin_memory=True)
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        total=right=0;loss_sum=0.;t0=time.monotonic()
        for x,y in loader:
            if time.monotonic()>=deadline:break
            x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True);z=model(x)
            loss=torch.nn.functional.cross_entropy(z,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=len(y);right+=int((z.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
        if total!=len(data):break
        model.eval();scores={}
        with torch.no_grad():
            for key,array in arrays.items():
                pieces=[]
                for i in range(0,len(array),256):
                    x=torch.from_numpy(np.array(array[i:i+256])).cuda();pieces.extend(model(x).softmax(1)[:,0].cpu().numpy())
                scores[key]=np.array(pieces)
        rows_curve=curve(scores);selected,score=selection(rows_curve,budget)
        row=dict(epoch=epoch,loss=loss_sum/total,augmented_train_accuracy=right/total,selected=selected,
                 score=score,seconds=time.monotonic()-t0,elapsed_gpu_seconds=time.monotonic()-started,
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
        history.append(row);print(json.dumps(row),flush=True)
        snapshot=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,seed=SEED,
                      selection=selected,score=score,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(snapshot,folder/"last.pt")
        if best_finetuned_rank is None or tuple(score)>tuple(best_finetuned_rank):
            best_finetuned_rank=score;best_finetuned_epoch=epoch;torch.save(snapshot,folder/"best_finetuned.pt")
        if tuple(score)>tuple(best_rank):
            best_rank=score;best_epoch=epoch;bad=0;torch.save(snapshot,folder/"best.pt")
        else:bad+=1
        with (folder/"history.json").open("w") as h:json.dump(history,h,indent=2)
        if bad>=2 or time.monotonic()>=deadline:break
    elapsed=time.monotonic()-started
    save(folder/"DONE.json",dict(epochs=len(history),best_epoch=best_epoch,best_finetuned_epoch=best_finetuned_epoch,improved_vs_initialization=best_epoch>0,
        gpu_seconds=elapsed,gpu_hours=elapsed/3600,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        reason="early_stopping" if bad>=2 else "epoch_or_time_cap",test_accessed=False))


def controller():
    try:
        prepare();train()
        if (OUT/"training/BLOCKED.json").exists():return
        subprocess.run([sys.executable,"-u","-B",str(OUT/"complete.py")],cwd=ROOT,check=True)
    except Exception as e:
        save(OUT/"FAILED.json",dict(error=repr(e),time=time.time()));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","train","controller"])
    globals()[p.parse_args().mode]()
