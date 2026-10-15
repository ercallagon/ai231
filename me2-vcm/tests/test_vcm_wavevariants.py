import importlib
import sys
from pathlib import Path

import numpy as np


def module():
    sys.path.insert(0,str(Path("scripts").resolve()))
    return importlib.import_module("prepare_vcm_wavevariants")


def test_reproducible_waveform_and_recipe():
    pcm=np.random.default_rng(1).normal(0,.1,16000).astype(np.float32)
    a,p=module().transform(pcm,"train-only-key"); b,q=module().transform(pcm,"train-only-key")
    np.testing.assert_array_equal(a,b); assert p==q


def test_transform_preserves_source_and_peak_bound():
    pcm=np.random.default_rng(1).normal(0,.2,16000).astype(np.float32); before=pcm.copy()
    for key in ("a","b","c","d","e","f"):
        wave,_=module().transform(pcm,key)
        assert wave.dtype==np.float32 and np.isfinite(wave).all() and np.abs(wave).max()<=.981
    np.testing.assert_array_equal(pcm,before)


def test_silence_remains_finite():
    for key in ("a","b","c","d"):
        wave,_=module().transform(np.zeros(16000,dtype=np.float32),key)
        assert np.isfinite(wave).all()


def test_selection_bounded_and_real_prioritized():
    rows=[dict(audio_path=str(i),is_synthetic="0" if i<3 else "1",intent="I",slot="",source="S") for i in range(20)]
    chosen=module().select(rows,8)
    assert len(chosen)==8 and len({r["audio_path"] for r in chosen})==8
    assert {"0","1","2"}.issubset({r["audio_path"] for r in chosen})
    assert module().select(list(reversed(rows)),8)==chosen


def test_selection_handles_small_dataset():
    rows=[dict(audio_path=str(i),is_synthetic="1",intent="I",slot="",source="S") for i in range(3)]
    assert len(module().select(rows,10))==3


def test_all_profiles_supported():
    profiles={module().transform(np.ones(1600,dtype=np.float32)*.1,str(i))[1]["profile"] for i in range(40)}
    assert profiles=={0,1,2,3}


def dataset_fixture(monkeypatch,tmp_path,augment):
    import torch
    trial=importlib.import_module("vcm_wavevariants_trial")
    from hey_swift.vcm.augmentation import AugConfig
    def initialize(self,config,*args,**kwargs):
        self.rows=[dict(audio_path="train",is_synthetic="0")]
        self.cache=np.ones((1,1,80,198),dtype=np.float32)
        self.index={"train":0}; self.seed=42; self.epoch=torch.tensor(0)
        self.augment=kwargs["augment"]; self.labels=np.array([0]); self.slots=np.array([-1])
        self.real_multiplier=2.; self.augmentation=AugConfig(enabled=False)
    monkeypatch.setattr(trial.source.SourceDataset,"__init__",initialize)
    monkeypatch.setattr(trial,"STAGE",tmp_path)
    return trial.VariantDataset({"experiment":{"waveform_views":True}},augment=augment)


def test_evaluation_never_opens_variant_cache(monkeypatch,tmp_path):
    module()
    data=dataset_fixture(monkeypatch,tmp_path,False)
    assert data.views is None and not data.use_views and len(data[0])==3
    np.testing.assert_array_equal(data[0][0].numpy(),data.cache[0])


def test_variant_dataset_stateless_and_clean_cache_intact(monkeypatch,tmp_path):
    import json
    module()
    variants=np.full((1,1,80,198),2,dtype=np.float32)
    np.save(tmp_path/"mels.npy",variants)
    (tmp_path/"index.json").write_text(json.dumps({"train":0}))
    data=dataset_fixture(monkeypatch,tmp_path,True); seen=set()
    for epoch in range(20):
        data.epoch.fill_(epoch)
        a=data[0][0].numpy(); b=data[0][0].numpy()
        np.testing.assert_array_equal(a,b); seen.add(float(a.mean()))
    assert seen=={1.,2.} and len(data[0])==4
    np.testing.assert_array_equal(data.cache,np.ones_like(data.cache))
    np.testing.assert_array_equal(np.load(tmp_path/"mels.npy"),variants)
