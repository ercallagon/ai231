#!/usr/bin/env python3
"""Build at most 6000 deterministic training-only views; preserve old artifacts."""
import argparse
import collections
import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import yaml
from scipy.signal import butter, fftconvolve, sosfilt

from hey_swift.vcm.dataset import decode_audio
from hey_swift.vcm.features import FeatureConfig,MelFeatures
from hey_swift.vcm.protocol import load_feature_cache,sha256
from vcm_wavevariants_trial import STAGE,PARENT,TAGS,VariantDataset
from validate_vcm_protocol import validate

SEED=20261009
LIMIT=6000
SOURCES=["scripts/prepare_vcm_wavevariants.py","scripts/vcm_wavevariants_trial.py",
         "scripts/run_vcm_wavevariants_queue.py","tests/test_vcm_wavevariants.py"]


def save(path,value):
    with Path(path).open("x") as stream: json.dump(value,stream,indent=2)


def stable_seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:16],16)


def select(rows,limit=LIMIT):
    ordered=sorted(rows,key=lambda r:(stable_seed(f'{SEED}:{r["audio_path"]}'),r["audio_path"]))
    real=[r for r in ordered if r["is_synthetic"]=="0"]
    chosen=real[:limit]; groups=collections.defaultdict(list)
    used={r["audio_path"] for r in chosen}
    for row in ordered:
        if row["audio_path"] not in used:
            groups[(row["intent"],row["slot"],row["source"])].append(row)
    groups={k:collections.deque(v) for k,v in sorted(groups.items())}
    while len(chosen)<min(limit,len(rows)):
        for group in groups.values():
            if group and len(chosen)<limit: chosen.append(group.popleft())
    return chosen


def transform(pcm,key):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,stable_seed(key),1]))
    profile=int(rng.integers(0,4)); wav=np.asarray(pcm,dtype=np.float32).copy()
    params={"seed":SEED,"key_seed":stable_seed(key),"recipe_version":1,"profile":profile}
    if profile in (1,2):
        rt=float(rng.uniform(.10,.25)); length=int(rt*16000)
        impulse=rng.normal(0,.006,length)*np.exp(-np.arange(length)/(16000*rt/6))
        impulse[0]=1.
        for _ in range(3): impulse[int(rng.integers(160,min(length,1000)))]+=float(rng.uniform(.03,.12))
        wav=fftconvolve(wav,impulse,mode="full").astype(np.float32)
        cutoff=float(rng.uniform(3500,7000)); wav=sosfilt(butter(2,cutoff,fs=16000,output="sos"),wav).astype(np.float32)
        params.update(rir_seconds=rt,lowpass_hz=cutoff)
    if profile in (0,2):
        noise=rng.normal(size=len(wav)).astype(np.float32)
        noise=sosfilt(butter(1,2500,fs=16000,output="sos"),noise).astype(np.float32)
        t=np.arange(len(wav),dtype=np.float32)/16000
        base=float(rng.uniform(100,300)); rhythm=float(rng.uniform(1,3))
        harmonic=sum(np.sin(2*np.pi*base*k*t)/k for k in (1,2,3))
        noise+=.3*harmonic*(.5+.5*np.sin(2*np.pi*rhythm*t))
        snr=float(rng.uniform(15,30)); rms=float(np.sqrt(np.mean(wav**2)+1e-12))
        noise/=max(float(np.sqrt(np.mean(noise**2))),1e-8)
        wav+=noise*rms/(10**(snr/20))
        params.update(snr_db=snr,harmonic_hz=base,rhythm_hz=rhythm,noise="seeded colored/harmonic non-speech proxy, not recorded music")
    gain=float(rng.uniform(.65,1.25)); wav*=gain
    if profile==3:
        shift=int(rng.integers(-640,641)); padding=(max(shift,0),max(-shift,0))
        wav=np.pad(wav,padding); params["padding_shift_samples"]=shift
    peak=max(float(np.abs(wav).max()),1e-8); scale=min(1.,.98/peak)
    wav*=scale; params.update(gain=gain,peak_safety_scale=scale)
    if not np.isfinite(wav).all(): raise ValueError("Nonfinite perturbation")
    return wav.astype(np.float32),params


def decode(raw,row):
    if hashlib.sha256(raw).hexdigest()!=row["audio_sha256"]: raise ValueError("Raw audio mismatch")
    pcm,sr=sf.read(io.BytesIO(raw),dtype="float32",always_2d=True)
    if sr!=16000 or not len(pcm) or not np.isfinite(pcm).all(): raise ValueError("Invalid PCM")
    pcm=pcm.mean(axis=1)
    if hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()!=row["pcm_sha256"]: raise ValueError("PCM mismatch")
    return pcm


def prepare():
    if STAGE.exists(): raise FileExistsError(STAGE)
    for tag in TAGS.values():
        if (Path("experiments/vcm")/tag).exists(): raise FileExistsError(tag)
    cfg=yaml.safe_load((PARENT/"config.yaml").read_text()); validate(str(PARENT/"config.yaml"))
    clean,index=load_feature_cache(cfg)
    rows=list(csv.DictReader((Path(cfg["data"]["manifests"])/"train_19intent.csv").open()))
    selected=select(rows); protected=set(); protected_voices=set(); manifests={}
    for path in Path(cfg["data"]["manifests"]).glob("*.csv"):
        manifests[str(path)]=sha256(path)
        if path.name=="train_19intent.csv": continue
        for row in csv.DictReader(path.open()):
            protected.add(row["pcm_sha256"]); protected_voices.update(json.loads(row["speech_voices"]))
    for row in selected:
        if row["pcm_sha256"] in protected or set(json.loads(row["speech_voices"])) & protected_voices:
            raise ValueError("Selected augmentation parent crosses protected boundary")
    STAGE.mkdir()
    old_hashes={str(p):sha256(p) for p in PARENT.rglob("*") if p.is_file()}
    for path in SOURCES:
        target=STAGE/"source_snapshot"/path; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    features=MelFeatures(FeatureConfig.from_dict(cfg["audio"]))
    array=np.lib.format.open_memmap(STAGE/"mels.npy",mode="w+",dtype=np.float32,shape=(len(selected),1,80,198))
    locators=collections.defaultdict(list)
    for i,row in enumerate(selected): locators[(row["parquet"],int(row["row_group"]))].append((i,row))
    recipes=[None]*len(selected); done=0; parity=0
    for (parquet,group),items in sorted(locators.items()):
        table=pq.ParquetFile(parquet).read_row_group(group,columns=["audio"])
        for i,row in items:
            raw=decode_audio(table["audio"][int(row["row_offset"])].as_py()); pcm=decode(raw,row)
            wav,params=transform(pcm,row["audio_path"])
            digest=hashlib.sha256(wav.astype("<f4").tobytes()).hexdigest()
            if digest in protected: raise ValueError("Augmented PCM matches protected audio")
            mel=features.transform(wav)
            if mel.shape!=(1,80,198) or not np.isfinite(mel).all(): raise ValueError("Invalid variant features")
            array[i]=mel
            if i%500==0:
                replay,_=transform(pcm,row["audio_path"])
                np.testing.assert_array_equal(features.transform(replay),array[i])
                np.testing.assert_allclose(features.transform(pcm),clean[index[row["audio_path"]]],rtol=1e-5,atol=1e-5)
                parity+=1
            recipes[i]={"variant_index":i,"parent":row,"parameters":params,"variant_pcm_sha256":digest}
            done+=1
        if done%500<len(items): print(f"wave cache {done}/{len(selected)}",flush=True)
    array.flush()
    save(STAGE/"recipes.json",recipes); save(STAGE/"index.json",{r["audio_path"]:i for i,r in enumerate(selected)})
    for arm in TAGS:
        recipe=json.loads(json.dumps(cfg))
        recipe["training"].update(epochs=12,early_stop_patience=4,lr=.0001,num_workers=2)
        recipe["experiment"].update(entrypoint="scripts/vcm_wavevariants_trial.py",parent=PARENT.name,
            waveform_views=arm=="wave",note="Matched continuation; only waveform-view substitution differs",
            waveform_cache=str(STAGE),waveform_view_probability=.5)
        recipe["experiment"]["implementation_sha256"].update({p:sha256(p) for p in SOURCES})
        with (STAGE/f"{arm}.yaml").open("x") as stream: yaml.safe_dump(recipe,stream,sort_keys=False)
    # Exercise the actual augmented dataset and all heads without altering the parent.
    import torch
    from hey_swift.vcm.broadcast import build_broadcast_model
    from hey_swift.vcm.taxonomy import load_taxonomy
    tax=load_taxonomy(STAGE/"wave.yaml")
    model=build_broadcast_model(tax.n_way_head,{k:tax.slot_classes(k) for k in tax.slotted_intents})
    checkpoint=torch.load(PARENT/"best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    data=VariantDataset(yaml.safe_load((STAGE/"wave.yaml").read_text()),"train",tax,clean,index,42,augment=True)
    batch=torch.stack([data[i][0] for i in (0,1)]); model.train(); output=model(batch)
    loss=output["intent"].square().mean()+sum(v.square().mean() for v in output["slots"].values()); loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError("Backward preflight failed")
    sources={**old_hashes,**manifests,**{p:sha256(p) for p in SOURCES},
             **{str(p):sha256(p) for p in STAGE.glob("*") if p.is_file()}}
    save(STAGE/"readiness.json",{"passed":True,"sha256":sources,"variant_count":len(selected),
        "variant_bytes":(STAGE/"mels.npy").stat().st_size,"seed":SEED,"parity_samples":parity,
        "selected_real":sum(r["is_synthetic"]=="0" for r in selected),
        "profiles":dict(collections.Counter(r["parameters"]["profile"] for r in recipes)),
        "train_parent_only":True,"evaluation_manifests_unchanged":True,"finite_backward_passed":True,
        "protocol_sha256":sha256(cfg["data"]["protocol"]),"original_artifacts_unchanged":True})
    save(STAGE/"environment.json",{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
         "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         "git_status":subprocess.check_output(["git","status","--short"],text=True)})
    (STAGE/"environment_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    print(f"Validated {len(selected)} training variants; {parity} direct/cache parity samples",flush=True)


def seal():
    root=ET.parse(STAGE/"regression.xml").getroot(); cases=list(root.iter("testcase"))
    if len(cases)<90 or list(root.iter("failure")) or list(root.iter("error")): raise RuntimeError("Regression suite incomplete/failing")
    if sum("vcm_wavevariants" in c.attrib.get("classname","") for c in cases)<5: raise RuntimeError("Variant tests missing")
    save(STAGE/"regression_passed.json",{"passed":True,"tests":len(cases),"sha256":sha256(STAGE/"regression.xml")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("mode",choices=["prepare","seal"])
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    {"prepare":prepare,"seal":seal}[args.mode]()
