import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


def adapter():
    sys.path.insert(0, str(Path("scripts").resolve()))
    return importlib.import_module("vcm_bc8_source_finetune")


def test_source_weight_target():
    module = adapter()
    p, q = 1559 / 25366, 0.30
    weight = module.source_weight(p, q)
    assert p * weight / (p * weight + 1 - p) == pytest.approx(q)


@pytest.mark.parametrize("p,q", [(0, .3), (1, .3), (.1, 0), (.1, 1)])
def test_source_weight_invalid(p, q):
    with pytest.raises(ValueError):
        adapter().source_weight(p, q)


def test_weighted_loss_equivalence_and_gradients():
    import torch
    module = adapter()
    logits = torch.tensor([[2., 1., -1.], [.2, .4, .5]], requires_grad=True)
    labels = torch.tensor([0, 2])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([.5, 1., 2.]), label_smoothing=.05)
    actual = module.balanced_intent_loss(logits, labels, criterion, torch.ones(2))
    torch.testing.assert_close(actual, criterion(logits, labels))
    weights = torch.tensor([6., 1.])
    loss = module.balanced_intent_loss(logits, labels, criterion, weights)
    torch.testing.assert_close(loss, module.balanced_intent_loss(logits, labels, criterion, weights * 3))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_finetune_cpu_warmstart_and_restoration(tmp_path, monkeypatch):
    import torch
    module = adapter()
    trainer = module.train_repro
    from hey_swift.vcm.taxonomy import load_taxonomy
    config = yaml.safe_load(Path("configs/vcm_hf_grouped_v2b_bc8.yaml").read_text())
    config["training"].update(device="cpu", epochs=1, batch_size=4, num_workers=0)
    config["experiment"].update(implementation_sha256={}, source_balance_target=.3)
    manifest = tmp_path / "manifests"
    manifest.mkdir()
    config["data"]["manifests"] = str(manifest)
    rows = [{"audio_path": f"x{i}", "intent": "TIMER" if i % 2 else "LIGHT_ON",
             "slot": "10s" if i % 2 else "", "speaker": "voice", "source": "fixture",
             "is_synthetic": str(int(i % 3 != 0))} for i in range(12)]
    for split, chosen in (("train", rows[:8]), ("val", rows[8:])):
        with (manifest / f"{split}_19intent.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(chosen)
    cache = np.random.default_rng(42).normal(size=(12, 1, 80, 198)).astype(np.float32)
    monkeypatch.setattr(trainer, "verify_protocol", lambda cfg: {})
    monkeypatch.setattr(trainer, "load_feature_cache", lambda cfg: (cache, {r["audio_path"]: i for i, r in enumerate(rows)}))
    monkeypatch.setattr(trainer, "sha256", lambda path: "fixture_hash")
    monkeypatch.setattr(module, "sha256", lambda path: "fixture_hash")
    monkeypatch.setattr(trainer.subprocess, "check_output", lambda *args, **kwargs: "fixture")
    parent = tmp_path / "parent"
    parent.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    taxonomy = load_taxonomy(path)
    heads = {name: taxonomy.slot_classes(name) for name in taxonomy.slotted_intents}
    initial = trainer.build_broadcast_model(taxonomy.n_way_head, heads)
    torch.save({"state_dict": initial.state_dict(), "arch": "bc_resnet_broadcast8", "protocol_sha256": "fixture_hash"}, parent / "best.pt")
    (parent / "result.json").write_text("{}")
    monkeypatch.setattr(module, "PARENT", parent)
    monkeypatch.setattr(module, "TAG", "fixture_sourceft")
    originals = trainer.CachedDataset, trainer.epoch_pass, trainer.build_broadcast_model
    with module.refinement_engine(str(path), {"real_intent_multiplier": 6.0}):
        warm = trainer.build_broadcast_model(taxonomy.n_way_head, heads)
        for name, value in initial.state_dict().items():
            torch.testing.assert_close(value, warm.state_dict()[name], rtol=0, atol=0)
    assert originals == (trainer.CachedDataset, trainer.epoch_pass, trainer.build_broadcast_model)
    monkeypatch.chdir(tmp_path)
    Path("experiments/vcm").mkdir(parents=True)
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        module.train(str(path))
        output = Path("experiments/vcm/fixture_sourceft")
        result = json.loads((output / "result.json").read_text())
        assert result["epochs_run"] == 1
        proof = json.loads((output / "finetune_provenance.json").read_text())
        assert proof["training_real_fraction"] == 3 / 8
        assert proof["optimizer_reset"] and proof["slot_loss_unchanged"]
        assert (output / "metrics_train_clean.json").exists()
        assert originals == (trainer.CachedDataset, trainer.epoch_pass, trainer.build_broadcast_model)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
