import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


def load_preparer():
    spec = importlib.util.spec_from_file_location("grouped_prepare", "scripts/prepare_vcm_grouped_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows():
    rows = []
    for group in range(40):
        voice = f"speaker_{group}"
        values = [("LIGHT_ON", "")] * 10 + [("OUT_OF_SCOPE", "")] * 10
        values += [("TIMER", value) for value in ("10s", "30s", "1m") for _ in range(6)]
        for i, (intent, slot) in enumerate(values):
            key = f"{voice}_{i}"
            rows.append({"intent": intent, "slot": slot, "speaker": voice,
                         "speech_voices": json.dumps([voice]), "speech_dependencies": "[]",
                         "audio_path": key, "audio_sha256": key, "pcm_sha256": key,
                         "hf_config": "default", "source": "real" if group % 2 else "synthetic",
                         "is_synthetic": str(int(group % 2 == 0))})
    return rows


def test_grouped_selection_reproducible_and_disjoint():
    module = load_preparer()
    rows = fixture_rows()
    a = module.choose_validation(rows, trials=32)
    b = module.choose_validation(rows, trials=32)
    assert a == b
    train, val, _, selection = a
    assert not selection["selection_uses_audio_or_model_predictions"]
    module.audit_boundaries({"train": train, "val": val})
    assert set(r["intent"] for r in val) == {"LIGHT_ON", "TIMER", "OUT_OF_SCOPE"}


def test_mixed_lineage_excluded_and_unknown_identity_not_randomly_split():
    module = load_preparer()
    rows = fixture_rows()
    _, _, _, selection = module.choose_validation(rows, trials=32)
    selected = selection["selected_voices"][0]
    other = next(r["speaker"] for r in rows if r["speaker"] not in selection["selected_voices"])
    composite = dict(rows[0], hf_config="synthetic_negatives", audio_path="composite",
                     speech_voices=json.dumps([selected, other]), neg_kind="babble")
    unknown = dict(rows[0], audio_path="unknown", speech_voices="[]")
    _, _, excluded, _ = module.choose_validation(rows + [composite, unknown], trials=32)
    assert {r["exclusion_reason"] for r in excluded} == {
        "cross_train_val_speech_composite", "missing_speech_identity"}


def test_cached_augmentation_keyed_to_sample_and_epoch():
    torch = pytest.importorskip("torch")
    from hey_swift.vcm.train_repro import CachedDataset
    data = CachedDataset.__new__(CachedDataset)
    from hey_swift.vcm.augmentation import AugConfig
    data.rows = [{"audio_path": "x"}]
    data.cache = np.random.default_rng(1).normal(size=(1, 1, 80, 198)).astype(np.float32)
    original = data.cache.copy()
    data.index = {"x": 0}
    data.seed, data.augment = 42, True
    data.epoch = torch.zeros((), dtype=torch.int64)
    data.labels, data.slots = np.array([0]), np.array([-1])
    data.augmentation = AugConfig()
    first = data[0][0]
    torch.testing.assert_close(first, data[0][0])
    data.epoch.fill_(1)
    assert not torch.equal(first, data[0][0])
    np.testing.assert_array_equal(data.cache, original)


def test_speaker_metrics_exclude_noise_pseudo_identities():
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from hey_swift.vcm.train_repro import grouped_report
    rows = [{"speaker": "real", "speech_voices": '["real"]', "source": "speech", "is_synthetic": "0"},
            {"speaker": "negative_pseudo", "speech_voices": "[]", "source": "noise", "is_synthetic": "1"}]
    report = grouped_report(SimpleNamespace(rows=rows, labels=[0, 1]), [0, 0])
    assert list(report["speaker"]) == ["real"]
    assert report["speaker_macro_intent_accuracy"] == 1.0


def test_versioned_trainer_cpu_smoke(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from hey_swift.vcm import train_repro as trainer
    config = yaml.safe_load(Path("configs/vcm_hf_aligned_v1_bc8.yaml").read_text())
    config["training"].update(device="cpu", epochs=1, batch_size=4, num_workers=0)
    config["experiment"]["implementation_sha256"] = {}
    manifest = tmp_path / "manifests"
    manifest.mkdir()
    config["data"]["manifests"] = str(manifest)
    rows = [{"audio_path": f"x{i}", "intent": "TIMER" if i % 2 else "LIGHT_ON",
             "slot": "10s" if i % 2 else "", "speaker": "voice", "source": "fixture",
             "is_synthetic": "0"} for i in range(12)]
    for split, selected in (("train", rows[:8]), ("val", rows[8:])):
        with (manifest / f"{split}_19intent.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    cache = np.random.default_rng(42).normal(size=(12, 1, 80, 198)).astype(np.float32)
    monkeypatch.setattr(trainer, "verify_protocol", lambda cfg: {})
    monkeypatch.setattr(trainer, "load_feature_cache", lambda cfg: (cache, {r["audio_path"]: i for i, r in enumerate(rows)}))
    monkeypatch.setattr(trainer, "sha256", lambda path: "fixture_hash")
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda *args, **kwargs: "fixture")
    monkeypatch.chdir(tmp_path)
    Path("experiments/vcm").mkdir(parents=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    torch.set_num_threads(1)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        output = trainer.run("fixture", str(path))
        result = json.loads((output / "result.json").read_text())
        assert result["epochs_run"] == 1
        assert "intent_accuracy" in result["history"][0]["augmented_train"]
        assert (output / "metrics_train_clean.json").exists()
        checkpoint = torch.load(output / "best.pt", weights_only=False)
        assert checkpoint["arch"] == "bc_resnet_broadcast8"
        assert "optimizer" in checkpoint and "loader_rng" in checkpoint
        repeat = trainer.run("fixture_repeat", str(path))
        repeated = torch.load(repeat / "best.pt", weights_only=False)
        for name, value in checkpoint["state_dict"].items():
            torch.testing.assert_close(value, repeated["state_dict"][name], rtol=0, atol=0)
        with pytest.raises(FileExistsError):
            trainer.run("fixture", str(path))
    finally:
        torch.use_deterministic_algorithms(previous)
