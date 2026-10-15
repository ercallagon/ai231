import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_broadcast_shapes_and_gradients():
    from hey_swift.vcm.broadcast import build_broadcast_model
    from hey_swift.vcm.taxonomy import default_taxonomy
    taxonomy = default_taxonomy()
    torch.set_num_threads(1)
    model = build_broadcast_model(20, {name: taxonomy.slot_classes(name) for name in taxonomy.slotted_intents})
    output = model(torch.randn(2, 1, 80, 198))
    assert output["intent"].shape == (2, 20)
    assert len(output["slots"]) == 6
    assert all(value.shape == (2, 3) for value in output["slots"].values())
    loss = output["intent"].square().mean() + sum(value.square().mean() for value in output["slots"].values())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_candidate_bridge_preserves_legacy_factories():
    from hey_swift.vcm import models, evaluate
    spec = importlib.util.spec_from_file_location("bc_trial", Path("scripts/vcm_bc8_trial.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = models.ArchSpec.a4()
    builder = models.build_model
    loader = evaluate._load_model
    with module.candidate_factories():
        candidate = models.ArchSpec.a4()
        assert candidate.name == "bc_resnet_broadcast8"
        model = models.build_model(candidate, {"TIMER": ["10s", "30s", "1m"]})
        assert model(torch.zeros(1, 1, 80, 198))["intent"].shape == (1, 20)
    assert models.ArchSpec.a4() == original
    assert models.build_model is builder
    assert evaluate._load_model is loader


def test_candidate_checkpoint_roundtrip(tmp_path):
    from hey_swift.vcm import evaluate
    from hey_swift.vcm.broadcast import NAME, build_broadcast_model
    from hey_swift.vcm.taxonomy import load_taxonomy
    config = "configs/vcm_hf_aligned_v1_bc8.yaml"
    taxonomy = load_taxonomy(config)
    heads = {name: taxonomy.slot_classes(name) for name in taxonomy.slotted_intents}
    model = build_broadcast_model(20, heads).eval()
    path = tmp_path / "best.pt"
    torch.save({"arch": NAME, "state_dict": model.state_dict(), "slot_heads": list(heads)}, path)
    spec = importlib.util.spec_from_file_location("bc_trial", Path("scripts/vcm_bc8_trial.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.candidate_factories():
        loaded, _, checkpoint = evaluate._load_model(config, path, torch.device("cpu"))
    sample = torch.randn(1, 1, 80, 198)
    with torch.no_grad():
        expected, actual = model(sample), loaded(sample)
    assert checkpoint["arch"] == NAME
    torch.testing.assert_close(expected["intent"], actual["intent"])
    for name in heads:
        torch.testing.assert_close(expected["slots"][name], actual["slots"][name])
