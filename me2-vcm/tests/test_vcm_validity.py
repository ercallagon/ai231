import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def discovery():
    sys.path.insert(0, str(Path("scripts").resolve()))
    return importlib.import_module("vcm_oos_discovery")


def test_validity_shape_and_backbone_compatibility():
    import torch
    from hey_swift.vcm.broadcast import build_broadcast_model
    from hey_swift.vcm.validity import ValidityModel
    torch.set_num_threads(1)
    heads = {"TIMER": ["10s", "30s", "1m"]}
    original = build_broadcast_model(20, heads).eval()
    model = ValidityModel(20, heads).eval()
    model.backbone.load_state_dict(original.state_dict(), strict=True)
    for batch in (1, 2):
        x = torch.randn(batch, 1, 80, 198)
        with torch.no_grad():
            expected, actual = original(x), model(x)
        torch.testing.assert_close(expected["intent"], actual["intent"])
        torch.testing.assert_close(expected["slots"]["TIMER"], actual["slots"]["TIMER"])
        assert actual["validity"].shape == (batch,)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in original.parameters()) == 257


@pytest.mark.parametrize("labels", [[0, 1, 2], [2, 2], [0, 0]])
def test_balanced_validity_loss_finite(labels):
    import torch
    from hey_swift.vcm.validity import balanced_validity_loss
    logits = torch.zeros(len(labels), requires_grad=True)
    loss = balanced_validity_loss(logits, torch.tensor(labels), 2)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def pack(logits, labels):
    return {"logits": np.asarray(logits, dtype=float), "labels": np.asarray(labels),
            "slot_ok": np.ones(len(labels), dtype=bool), "real": np.ones(len(labels), dtype=bool)}


def test_policy_respects_oos_budget_and_veto():
    module = discovery()
    val = pack([[5, 0, -2], [0, 5, -2], [0, 0, 4]], [0, 1, 2])
    tune = pack([[.2, .1, 0], [0, 0, 5]], [2, 2])
    chosen, policies = module.select_policy(val, tune, 2)
    assert chosen["tune_false_accept_rate"] <= .05
    assert chosen["validation"]["e2e_accuracy"] == 1
    for score in module.scores(val, 2).values():
        assert score[-1] == -np.inf
    assert len(policies) == 3


def test_policy_metrics_false_accept_vs_false_reject():
    module = discovery()
    values = pack([[4, 0, 0], [0, 4, 0], [4, 0, 0]], [0, 1, 2])
    result = module.decision_metrics(values, np.array([.9, .1, .8]), .5, 2)
    assert result["false_accepts"] == 1
    assert result["false_rejects"] == 1
    assert result["e2e_accuracy"] == pytest.approx(1/3)


def test_policy_conservative_when_all_tune_scores_high():
    module = discovery()
    val = pack([[5, 0, -2]], [0])
    tune = pack([[5, 0, -2]] * 20, [2] * 20)
    chosen, _ = module.select_policy(val, tune, 2)
    assert chosen["tune_false_accept_rate"] == 0
    assert chosen["validation"]["false_rejects"] == 1


def test_qualification_preserves_recognition():
    sys.path.insert(0, str(Path("scripts").resolve()))
    queue = importlib.import_module("run_vcm_oos_discovery_queue")
    raw = {"e2e_accuracy": .94, "slot_accuracy": .96,
           "groups": {"is_synthetic": {"0": {"intent_accuracy": .74}}}}
    baseline = {"raw": raw, "chosen": {"validation": {"e2e_accuracy": .90}}}
    result = {"raw_validation": raw, "best_policy": {"validation": {"e2e_accuracy": .91}}}
    assert queue.qualifies(result, baseline)[0]
    result["raw_validation"] = dict(raw, slot_accuracy=.90)
    assert not queue.qualifies(result, baseline)[0]


def test_training_cpu_smoke(tmp_path, monkeypatch):
    import csv
    import json
    import torch
    import yaml
    from hey_swift.vcm.taxonomy import load_taxonomy
    from hey_swift.vcm.broadcast import build_broadcast_model
    module = discovery()
    cfg = yaml.safe_load(Path("configs/vcm_hf_grouped_v2b_bc8_sourceft.yaml").read_text())
    cfg["training"].update(device="cpu", epochs=1, batch_size=4, num_workers=0)
    cfg["experiment"].update(tag="fixture_validity", implementation_sha256={}, validity_loss_weight=.25)
    directory = tmp_path / "manifests"
    directory.mkdir()
    cfg["data"]["manifests"] = str(directory)
    rows = [{"audio_path": f"x{i}", "intent": ("LIGHT_ON", "TIMER", "OUT_OF_SCOPE")[i % 3],
             "slot": "10s" if i % 3 == 1 else "", "source": "fixture", "speaker": "voice",
             "is_synthetic": str(i % 2)} for i in range(14)]
    for split, selected in (("train", rows[:8]), ("val", rows[8:12]),
            ("tune_oos", [dict(r, intent="OUT_OF_SCOPE", slot="") for r in rows[12:]])):
        with (directory / f"{split}_19intent.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(selected)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    tax = load_taxonomy(config)
    model = build_broadcast_model(tax.n_way_head, {k: tax.slot_classes(k) for k in tax.slotted_intents})
    parent = tmp_path / "experiments/vcm/fixture_parent"
    parent.mkdir(parents=True)
    torch.save({"protocol_sha256": "fixture_hash", "state_dict": model.state_dict()}, parent / "best.pt")
    cache = np.random.default_rng(42).normal(size=(14, 1, 80, 198)).astype(np.float32)
    monkeypatch.setattr(module, "check", lambda config: cfg)
    monkeypatch.setattr(module, "load_feature_cache", lambda cfg: (cache, {r["audio_path"]: i for i, r in enumerate(rows)}))
    monkeypatch.setattr(module, "sha256", lambda path: "fixture_hash")
    monkeypatch.setattr(module, "PARENT", "fixture_parent")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "fixture")
    monkeypatch.chdir(tmp_path)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        module.train(str(config))
        output = Path("experiments/vcm/fixture_validity")
        result = json.loads((output / "result.json").read_text())
        assert result["epochs_run"] == 1
        assert result["n_params"] == 328007
        assert result["best_policy"]["tune_false_accept_rate"] <= .05
        checkpoint = torch.load(output / "best.pt", weights_only=False)
        assert checkpoint["arch"] == "bc8_command_validity"
        assert "optimizer" in checkpoint and "loader_rng" in checkpoint
        with pytest.raises(FileExistsError):
            module.train(str(config))
    finally:
        torch.use_deterministic_algorithms(previous)


def test_comparison_report(tmp_path, monkeypatch):
    import json
    sys.path.insert(0, str(Path("scripts").resolve()))
    queue = importlib.import_module("run_vcm_oos_discovery_queue")
    monkeypatch.setattr(queue, "STAGE", tmp_path)
    raw = {"e2e_accuracy": .94, "slot_accuracy": .96,
           "groups": {"is_synthetic": {"0": {"intent_accuracy": .74}}}}
    (tmp_path / "baselines.json").write_text(json.dumps([{"tag": "fixture_control", "raw": raw,
        "chosen": {"validation": {"e2e_accuracy": .90}}}]))
    (tmp_path / "screen_results.json").write_text("[]")
    queue.report()
    text = (tmp_path / "comparison.md").read_text()
    assert "fixture_control" in text and "0.9400" in text
    assert "No holdout use" in text
