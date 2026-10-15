#!/usr/bin/env python3
"""Immutable wakeword audit/cache, targeted fine-tunes and fixed-policy reports."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from hey_swift.vcm.protocol import sha256
from hey_swift.wakeword import dataset, features
from hey_swift.wakeword.model import build_model, count_params
from hey_swift.wakeword.train import lowpass_butter

STAGE = Path("work/wakeword_targeted_v2")
ARCHIVE = Path("../archive-me2/me2-wakeword-swift/training/experiments/exp15_mined_lp")
SOURCES = ["scripts/wakeword_targeted_trial.py", "scripts/run_wakeword_targeted_queue.py",
           "src/hey_swift/wakeword/model.py", "src/hey_swift/wakeword/features.py",
           "src/hey_swift/wakeword/dataset.py", "src/hey_swift/wakeword/train.py",
           "src/hey_swift/wakeword/inference.py"]


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def metrics(p, y, threshold):
    wake, fire = np.asarray(y) == 0, np.asarray(p) >= threshold
    tp, fp = int((fire & wake).sum()), int((fire & ~wake).sum())
    fn, tn = int((~fire & wake).sum()), int((~fire & ~wake).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "far": fp / max(fp + tn, 1), "frr": fn / max(fn + tp, 1)}


def choose(p, y, cap=.01):
    candidates = [metrics(p, y, threshold) for threshold in np.linspace(.01, .99, 99)]
    candidates += [metrics(p, y, 1.0)]
    feasible = [r for r in candidates if r["far"] <= cap]
    if not feasible:
        raise RuntimeError("No feasible validation policy")
    return max(feasible, key=lambda r: (r["recall"], r["f1"], -r["far"]))


def perturb(pcm, rng):
    pcm = lowpass_butter(pcm, float(rng.uniform(3000, 5000))).astype(np.float32)
    noise = rng.standard_normal(len(pcm)).astype(np.float32)
    rms = np.sqrt(np.mean(pcm**2) + 1e-12)
    pcm += noise * (rms / (10 ** (float(rng.uniform(10, 25)) / 20)))
    return pcm / max(float(np.abs(pcm).max()), 1.0)


def prepare():
    STAGE.mkdir(exist_ok=False)
    original = {str(p): sha256(p) for p in [Path("configs/wakeword.yaml"), Path("models/wakeword/wakeword_exp15_mined_lp.onnx")]
                + list(Path("work/training/exp15_repro").glob("*")) + list(ARCHIVE.glob("*")) if p.is_file()}
    for path in SOURCES:
        target = STAGE / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    shutil.copy2(ARCHIVE / "best.pt", STAGE / "parent_best.pt")
    meta = {}
    for row in csv.DictReader(Path("data/wakeword/raw/metadata.csv").open()):
        if row["filename"] in meta and row["label"] != meta[row["filename"]]["label"]:
            raise ValueError("Conflicting metadata labels")
        meta.setdefault(row["filename"], row)
    rows, by_split = [], {}
    for split in dataset.SPLITS:
        verification = dataset.verify_manifest(split)
        if not verification["ok"]:
            raise ValueError(verification)
        group = []
        for relpath, label in dataset.load_manifest(split):
            path = Path("data/wakeword/raw") / relpath
            row = meta[relpath]
            if int(row["label"]) != label or row["split"] != split:
                raise ValueError(f"Metadata mismatch: {path}")
            pcm = features.load_wav_mono_16k(path)
            record = {"relpath": relpath, "label": label, "split": split, "source": row["source"],
                "voice_id": row["voice_id"], "parent_id": row["parent_id"], "audio_sha256": sha256(path),
                "pcm_sha256": hashlib.sha256(np.asarray(pcm, dtype="<f4").tobytes()).hexdigest()}
            rows.append(record); group.append(record)
        by_split[split] = group
    original_counts = {s:len(r) for s,r in by_split.items()}
    overlap = {}
    for a,b in itertools.combinations(dataset.SPLITS,2):
        common = {r["pcm_sha256"] for r in by_split[a]} & {r["pcm_sha256"] for r in by_split[b]}
        overlap[f"{a}:{b}"] = {"shared_pcm_hashes":len(common),
            "rows":[r for r in rows if r["split"] in (a,b) and r["pcm_sha256"] in common]}
    # Keep historical parent training intact; do not evaluate on audio it saw.
    train_pcm = {r["pcm_sha256"] for r in by_split["train"]}
    train_voices = {r["voice_id"] for r in by_split["train"] if r["voice_id"] and r["source"] not in ("nonspeech","hard_mined")}
    original_test_pcm = {r["pcm_sha256"] for r in by_split["test"]}
    original_test_voices = {r["voice_id"] for r in by_split["test"] if r["voice_id"] and r["source"] not in ("nonspeech","hard_mined")}
    excluded = []
    for split in ("validation","test"):
        retained=[]; seen=set()
        for row in by_split[split]:
            voice = row["voice_id"] if row["source"] not in ("nonspeech","hard_mined") else ""
            reason = None
            if row["pcm_sha256"] in train_pcm or (voice and voice in train_voices): reason="historical_parent_training_overlap"
            elif split=="validation" and (row["pcm_sha256"] in original_test_pcm or (voice and voice in original_test_voices)): reason="test_overlap"
            elif row["pcm_sha256"] in seen: reason="within_split_duplicate"
            if reason: excluded.append(dict(row,exclusion_reason=reason))
            else: retained.append(row); seen.add(row["pcm_sha256"])
        by_split[split]=retained
        if {r["label"] for r in retained}!={0,1}: raise ValueError("Corrected evaluation split lost a class")
    for a,b in itertools.combinations(dataset.SPLITS,2):
        if {r["pcm_sha256"] for r in by_split[a]} & {r["pcm_sha256"] for r in by_split[b]}:
            raise ValueError("Corrected split still overlaps")
    write(STAGE / "audit.json", {"passed": True, "original_sha256": original,
        "manifest_sha256": {s: sha256(Path("data/wakeword/manifests") / f"{s}.csv") for s in dataset.SPLITS},
        "metadata_sha256": sha256("data/wakeword/raw/metadata.csv"), "counts": {s:len(r) for s,r in by_split.items()},
        "original_counts":original_counts,"historical_pcm_overlap":overlap,
        "excluded_evaluation_rows": excluded, "split_rows": by_split,
        "split_policy":"Versioned evaluation-only decontamination: preserve parent train; exclude eval audio/declared speakers seen in train, validation overlap with original test, and within-evaluation duplicates. Original manifests/test untouched; new scores not directly comparable to historical denominators.",
        "limits": "Declared voices are disjoint, but mined/noise pseudo-identities do not establish speaker independence; near-duplicate/semantic lineage and real long-stream coverage are unverified"})
    offset = 0
    for split, selected in by_split.items():
        clean = np.lib.format.open_memmap(STAGE / f"{split}_clean.npy", mode="w+", dtype=np.float32, shape=(len(selected),80,100))
        variant = np.lib.format.open_memmap(STAGE / f"{split}_perturbed.npy", mode="w+", dtype=np.float32, shape=clean.shape)
        for i, row in enumerate(selected):
            pcm = features.load_wav_mono_16k(Path("data/wakeword/raw") / row["relpath"])
            clean[i] = features.mel_from_pcm(pcm)[0]
            variant[i] = features.mel_from_pcm(perturb(pcm.copy(), np.random.default_rng(np.random.SeedSequence([20261009,offset+i]))))[0]
            if not np.isfinite(clean[i]).all() or not np.isfinite(variant[i]).all():
                raise ValueError("Nonfinite wake features")
            if i % 500 == 0:
                print(f"cache {split} {i}/{len(selected)}", flush=True)
        offset += len(selected)
        clean.flush(); variant.flush()
    parent = load_parent()
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession("models/wakeword/wakeword_exp15_mined_lp.onnx", sess_options=options, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if shape[-2:] != [80,100]:
        raise ValueError(f"Wake ONNX input mismatch: {shape}")
    for batch in (1,2):
        x = torch.randn(batch,80,100)
        with torch.no_grad(): expected = parent(x).numpy()
        actual = session.run(None,{session.get_inputs()[0].name:x.numpy()})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
    audit = json.loads((STAGE / "audit.json").read_text())
    baseline = {}
    train_p = None
    for split in ("train", "validation"):
        data = WakeCache(split)
        p, y = probabilities(parent, data, "cpu")
        if split == "train": train_p = p
        else:
            baseline["selected"] = choose(p,y)
            baseline["fixed_030"] = metrics(p,y,.30)
            p_aug,y_aug = probabilities(parent,WakeCache(split,condition="perturbed"),"cpu")
            baseline["perturbed_at_selected"] = metrics(p_aug,y_aug,baseline["selected"]["threshold"])
    np.save(STAGE / "training_parent_probabilities.npy",train_p)
    write(STAGE / "baseline.json",baseline)
    cache_hashes = {str(p):sha256(p) for p in STAGE.glob("*.npy")}
    write(STAGE / "readiness.json", {"passed": True, "sources": {p:sha256(p) for p in SOURCES},
        "cache_sha256": cache_hashes, "audit_sha256":sha256(STAGE / "audit.json"),
        "parent_sha256":sha256(STAGE / "parent_best.pt"),"baseline_sha256":sha256(STAGE / "baseline.json"),
        "parent_deployed_onnx_parity":True,"counts":audit["counts"]})
    (STAGE / "environment_freeze.txt").write_text(subprocess.check_output([os.sys.executable,"-m","pip","freeze"],text=True))
    write(STAGE / "git_provenance.json", {"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "status":subprocess.check_output(["git","status","--short"],text=True),"python":os.sys.version,"torch":torch.__version__})
    print("Wake audit/cache/deployed-parent parity and validation baseline ready",flush=True)


def load_parent():
    model = build_model()
    state = torch.load(STAGE / "parent_best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(state,strict=True)
    return model.eval()


class WakeCache(Dataset):
    def __init__(self,split,augment=False,seed=20261009,condition="clean"):
        audit = json.loads((STAGE / "audit.json").read_text())
        self.rows = audit["split_rows"][split]
        self.clean = np.load(STAGE / f"{split}_{condition}.npy",mmap_mode="r")
        self.variant = np.load(STAGE / f"{split}_perturbed.npy",mmap_mode="r") if augment else None
        self.augment,self.seed,self.epoch = augment,seed,0
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,i]))
        source = self.variant if self.augment and rng.random()<.5 else self.clean
        x = np.array(source[i],copy=True)
        if self.augment:
            x = np.roll(x,int(rng.integers(-8,9)),axis=1)
            if rng.random()<.3:
                start=int(rng.integers(0,90)); x[:,start:start+10]=0
        return torch.from_numpy(x), int(self.rows[i]["label"]), i


def probabilities(model,data,device):
    model.eval(); p,y=[],[]
    with torch.no_grad():
        for x,labels,_ in DataLoader(data,batch_size=128,shuffle=False):
            p.extend(model(x.to(device)).softmax(1)[:,0].cpu().numpy()); y.extend(labels.numpy())
    return np.asarray(p),np.asarray(y)


def train(tag, strength, lr, seed):
    ready=json.loads((STAGE / "readiness.json").read_text())
    for path,digest in ready["sources"].items():
        if sha256(path)!=digest: raise ValueError(f"Wake source changed: {path}")
    for path,digest in ready["cache_sha256"].items():
        if sha256(path)!=digest: raise ValueError(f"Wake cache changed: {path}")
    if sha256(STAGE / "parent_best.pt")!=ready["parent_sha256"]: raise ValueError("Parent changed")
    output=Path("experiments/wakeword")/tag
    output.mkdir(exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no fallback")
    device=torch.device("cuda"); model=load_parent().to(device)
    data=WakeCache("train",True,seed); val=WakeCache("validation"); val_aug=WakeCache("validation",condition="perturbed")
    parent_p=np.load(STAGE / "training_parent_probabilities.npy")
    labels=np.asarray([r["label"] for r in data.rows])
    weights=np.where(labels==1,1+strength*np.clip(parent_p/.30,0,1),1).astype(np.float32)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(data,batch_size=128,shuffle=True,num_workers=0,generator=generator,pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12)
    weights=torch.from_numpy(weights).to(device)
    recipe={"tag":tag,"parent":"deployed exp15","parent_sha256":ready["parent_sha256"],"strength":strength,"lr":lr,"seed":seed,
        "epochs":12,"patience":4,"batch":128,"data_readiness_sha256":sha256(STAGE / "readiness.json"),
        "augmentation":"One clean plus one seeded low-pass/noise view; stateless shift/time-mask; not the historical full online recipe",
        "selection":"max clean recall/F1 at val FAR <=1%; perturbed recall safeguard in finalist qualification",
        "test_used_for_selection":False,"sources":ready["sources"],"n_params":count_params(model)}
    write(output / "config.json",recipe)
    for path in SOURCES:
        target=output/"implementation_snapshot"/path; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    history=[]; best=(-1.,-1.); bad=0; start=time.monotonic()
    for epoch in range(12):
        data.epoch=epoch; model.train(); loss_sum,correct,n=0.,0,0
        for x,y,i in loader:
            x,y=x.to(device),y.to(device); w=weights[i.to(device)]
            optimizer.zero_grad(set_to_none=True); logits=model(x)
            losses=torch.nn.functional.cross_entropy(logits,y,reduction="none")
            loss=(losses*w).sum()/w.sum()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite wake loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            correct+=int((logits.argmax(1)==y).sum()); n+=len(y); loss_sum+=float(loss.detach())*len(y)
        scheduler.step(); p,y=probabilities(model,val,device); chosen=choose(p,y)
        pa,ya=probabilities(model,val_aug,device); robust=metrics(pa,ya,chosen["threshold"])
        record={"epoch":epoch+1,"augmented_train_accuracy":correct/n,"train_loss":loss_sum/n,
                "validation":chosen,"perturbed_validation":robust,"fixed_030":metrics(p,y,.30),"elapsed_s":time.monotonic()-start}
        history.append(record); score=(chosen["recall"],chosen["f1"]); improved=score>best
        best,bad=(score,0) if improved else (best,bad+1)
        state={"state_dict":model.state_dict(),"epoch":epoch+1,"policy":chosen,"validation_perturbed":robust,
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
            "loader_rng":generator.get_state(),"config":recipe}
        torch.save(state,output/"last.pt")
        if improved: torch.save(state,output/"best.pt")
        write(output/"history.json",history)
        print(f'{tag} epoch={epoch+1}/12 val_recall={chosen["recall"]:.4f} FAR={chosen["far"]:.4f} perturbed_recall={robust["recall"]:.4f} bad={bad}',flush=True)
        if bad>=4: break
    checkpoint=torch.load(output/"best.pt",map_location="cpu",weights_only=False)
    write(output/"result.json",{"tag":tag,"epochs_run":len(history),"best_epoch":checkpoint["epoch"],
        "validation":checkpoint["policy"],"perturbed_validation":checkpoint["validation_perturbed"],"duration_s":time.monotonic()-start})


def finish(tag):
    output=Path("experiments/wakeword")/tag
    state=torch.load(output/"best.pt",map_location="cpu",weights_only=False)
    model=build_model(); model.load_state_dict(state["state_dict"],strict=True); model.eval()
    threshold=state["policy"]["threshold"]
    report={"threshold_selected_on_validation_only":threshold,"candidate":{},"parent":{}}
    parent=load_parent(); baseline=json.loads((STAGE/"baseline.json").read_text())
    for name,current,th in (("candidate",model,threshold),("parent",parent,baseline["selected"]["threshold"])):
        for condition in ("clean","perturbed"):
            p,y=probabilities(current,WakeCache("test",condition=condition),"cpu")
            report[name][condition]=metrics(p,y,th)
            report[name][condition+"_fixed_030"]=metrics(p,y,.30)
    torch.onnx.export(model,torch.randn(1,80,100),str(output/"model.onnx"),input_names=["features"],output_names=["logits"],
        dynamic_axes={"features":{0:"batch"},"logits":{0:"batch"}},opset_version=17)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(output/"model.onnx"))
    options=ort.SessionOptions(); options.intra_op_num_threads=1
    session=ort.InferenceSession(str(output/"model.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    for batch in (1,2):
        x=torch.randn(batch,80,100)
        with torch.no_grad(): expected=model(x).numpy()
        np.testing.assert_allclose(session.run(None,{"features":x.numpy()})[0],expected,rtol=1e-4,atol=1e-4)
    latency=[]; sample=np.zeros((1,80,100),dtype=np.float32)
    for i in range(110):
        started=time.perf_counter(); session.run(None,{"features":sample})
        if i>=10: latency.append((time.perf_counter()-started)*1000)
    from hey_swift.wakeword.inference import WakeWordDetector
    # Replay individual clips independently, preserving actual sustain/multiwin/cooldown.
    for name,path in (("candidate",output/"model.onnx"),("parent",Path("models/wakeword/wakeword_exp15_mined_lp.onnx"))):
        detector=WakeWordDetector(onnx_path=str(path))
        detector.sess=session if name=="candidate" else ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
        data=WakeCache("test"); positives=hits=negative_clips=fires=0; negative_seconds=0.
        for row in data.rows:
            pcm=features.load_wav_mono_16k(Path("data/wakeword/raw")/row["relpath"])
            # Pad to allow two qualifying hops after a short utterance.
            detector.reset(); events,_=detector.process(np.concatenate((np.zeros(8000,dtype=np.float32),pcm,np.zeros(16000,dtype=np.float32))))
            if row["label"]==0: positives+=1; hits+=int(bool(events))
            else: negative_clips+=1; fires+=len(events); negative_seconds+=len(pcm)/16000
        report[name]["runtime_clip_replay_fixed_030"]={"positive_hit_rate":hits/max(positives,1),"positive_n":positives,
            "negative_fires":fires,"negative_clips":negative_clips,"negative_source_seconds":negative_seconds,
            "not_a_continuous_stream_or_false_activations_per_hour_estimate":True}
    write(output/"evaluation.json",report)
    write(output/"export_validation.json",{"onnx_parity_passed":True,"bytes":(output/"model.onnx").stat().st_size,
        "host_single_thread_p50_ms":float(np.median(latency)),"host_single_thread_p95_ms":float(np.percentile(latency,95)),
        "sha256":sha256(output/"model.onnx"),"pi_verified":False,"no_deployment_changes":True})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("mode",choices=["prepare","train","finish"])
    parser.add_argument("--tag"); parser.add_argument("--strength",type=float,default=3.); parser.add_argument("--lr",type=float,default=.0001)
    parser.add_argument("--seed",type=int,default=20261009); args=parser.parse_args(); torch.set_num_threads(1)
    {"prepare":prepare,"train":lambda:train(args.tag,args.strength,args.lr,args.seed),"finish":lambda:finish(args.tag)}[args.mode]()
