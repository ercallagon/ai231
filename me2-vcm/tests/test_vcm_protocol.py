"""Regression checks for source identity and corrected protocol isolation."""
import hashlib
import importlib.util
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hey_swift.vcm import dataset as ds
from hey_swift.vcm.features import FeatureConfig
from hey_swift.vcm.protocol import load_feature_cache, sha256, verify_protocol


@pytest.fixture
def preparation():
    spec = importlib.util.spec_from_file_location("prepare_vcm", Path("scripts/prepare_vcm_protocol.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_key_distinguishes_config_and_split(preparation):
    key = preparation.source_key
    assert key("synthetic_negatives", "train", "audio/a.wav") != key("synthetic_negatives", "test", "audio/a.wav")
    assert key("default", "train", "audio/a.wav") != key("supplemental_synth", "train", "audio/a.wav")
    assert key("default", "train", "audio/a.wav") == key("default", "train", "audio/a.wav")


def test_manifest_resolver_honors_directory_and_override(tmp_path):
    path = tmp_path / "test_19intent.csv"
    path.write_text("audio_path,intent,slot,speaker\nhf:one,TEMPERATURE,18 degrees,s1\n")
    cfg = {"data": {"manifests": str(tmp_path)}}
    assert ds.configured_rows(cfg, "test")[0].slot == "18 degrees"
    cfg["data"]["manifests_override"] = {"val": str(path)}
    assert ds.configured_rows(cfg, "val")[0].audio_path == "hf:one"
    with pytest.raises(FileNotFoundError):
        ds.configured_rows(cfg, "holdout")


def test_lineage_tracks_all_babble_contributors(preparation):
    rows = [
        {"hf_config": "default", "hf_split": "train", "file": "audio/a.wav", "speaker_id": "s1", "audio_path": "a"},
        {"hf_config": "default", "hf_split": "train", "file": "audio/b.wav", "speaker_id": "s2", "audio_path": "b"},
        {"hf_config": "synthetic_negatives", "hf_split": "train", "file": "audio/n.wav", "speaker_id": "neg_s1", "audio_path": "n", "source_files": "train/audio/a.wav;train/audio/b.wav"},
    ]
    preparation.attach_lineage(rows)
    assert rows[-1]["speech_voices"] == ["s1", "s2"]
    assert rows[-1]["speech_dependencies"] == ["a", "b"]
    rows[-1]["source_files"] = "train/audio/missing.wav"
    with pytest.raises(ValueError, match="Unresolved"):
        preparation.attach_lineage(rows)


def test_source_locator_bypasses_flattened_wav_cache(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    sf = pytest.importorskip("soundfile")
    rows = []
    for number in (1, 2):
        raw = io.BytesIO()
        sf.write(raw, np.full(800, number / 10, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
        payload = raw.getvalue()
        path = tmp_path / f"{number}.parquet"
        pq.write_table(pa.Table.from_pylist([{"file": "audio/same.wav", "audio": {"bytes": payload}}]), path)
        rows.append(ds.Row(f"hf:{number}", "TIME", None, "test", parquet=str(path), row_group=0, row_offset=0, audio_sha256=hashlib.sha256(payload).hexdigest()))
    assert not np.array_equal(ds.load_row_wav(rows[0]), ds.load_row_wav(rows[1]))
    bad = ds.Row("hf:bad", "TIME", None, "test", parquet=rows[0].parquet, row_group=0, row_offset=0, audio_sha256="bad")
    with pytest.raises(ValueError, match="fingerprint"):
        ds.load_row_wav(bad)


def test_protocol_fails_on_changed_manifest(tmp_path):
    path = tmp_path / "train_19intent.csv"
    path.write_text("original")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"validation_passed": True, "manifest_sha256": {"train": sha256(path)}, "raw_sha256": {}, "implementation_sha256": {}}))
    cfg = {"data": {"protocol": str(protocol), "manifests": str(tmp_path)}}
    verify_protocol(cfg)
    path.write_text("changed")
    with pytest.raises(ValueError, match="manifest"):
        verify_protocol(cfg)


def test_corrected_cache_requires_matching_protocol_and_features(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    np.save(tmp_path / "mels.npy", np.zeros((1, 1, 80, 198), dtype=np.float32))
    (tmp_path / "index.json").write_text('{"hf:one": 0}')
    meta = {"protocol_sha256": sha256(protocol), "features": asdict(FeatureConfig()), "shape": [1, 1, 80, 198], "sha256": {n: sha256(tmp_path / n) for n in ("mels.npy", "index.json")}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    cfg = {"data": {"protocol": str(protocol), "feature_cache": str(tmp_path)}}
    cache, index = load_feature_cache(cfg)
    assert cache.shape == (1, 1, 80, 198)
    assert index == {"hf:one": 0}
    cfg["audio"] = {"n_mels": 40}
    with pytest.raises(ValueError, match="preprocessing"):
        load_feature_cache(cfg)


def test_all_supported_numeric_slots_have_supervision():
    from hey_swift.vcm.taxonomy import default_taxonomy
    from hey_swift.vcm.train import _slot_label
    tax = default_taxonomy()
    for intent, values in {"TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"], "BRIGHTNESS": ["20 percent", "60 percent", "100 percent"]}.items():
        assert [_slot_label(intent, value, tax) for value in values] == [0, 1, 2]
