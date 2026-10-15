import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def module():
    sys.path.insert(0,str(Path("scripts").resolve()))
    return importlib.import_module("wakeword_targeted_trial")


def test_wake_label_zero_and_false_accepts():
    result=module().metrics(np.array([.9,.2,.8,.1]),np.array([0,0,1,1]),.3)
    assert (result["tp"],result["fp"],result["fn"],result["tn"])==(1,1,1,1)
    assert result["far"]==.5 and result["recall"]==.5


def test_policy_selection_uses_false_accept_cap():
    result=module().choose(np.array([.9,.8,.5,.1]),np.array([0,0,1,1]),.01)
    assert result["far"]==0 and result["recall"]==1


def test_empty_positive_metrics_finite():
    result=module().metrics(np.array([.1,.2]),np.ones(2),.3)
    assert all(np.isfinite(v) for v in result.values())


def test_seeded_pcm_perturbation_finite_and_repeatable():
    pcm=np.random.default_rng(1).normal(0,.1,16000).astype(np.float32)
    a=module().perturb(pcm.copy(),np.random.default_rng(42))
    b=module().perturb(pcm.copy(),np.random.default_rng(42))
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all() and np.abs(a).max()<=1


def test_wake_cache_does_not_modify_shared_features(tmp_path,monkeypatch):
    import json
    m=module(); monkeypatch.setattr(m,"STAGE",tmp_path)
    rows=[{"label":0},{"label":1}]
    (tmp_path/"audit.json").write_text(json.dumps({"split_rows":{"train":rows}}))
    features=np.random.default_rng(1).normal(size=(2,80,100)).astype(np.float32)
    np.save(tmp_path/"train_clean.npy",features); np.save(tmp_path/"train_perturbed.npy",features*.8)
    data=m.WakeCache("train",True,42)
    np.testing.assert_array_equal(data[0][0].numpy(),data[0][0].numpy())
    np.testing.assert_array_equal(np.load(tmp_path/"train_clean.npy"),features)
