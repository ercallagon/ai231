"""Tests for the ONNX export + PyTorch<->ONNX parity path.

These require the 'train' extra (torch) and an exported model, so they are
skipped gracefully when torch or the ONNX artifact is absent. The parity logic
itself is exercised in scripts/04_parity_check.py; here we guard the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch", reason="torch not installed (pip install 'hey-swift[train]')")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


@pytest.mark.skipif(
    not (Path("models/vcm/a0_scratch.onnx").exists()),
    reason="no exported ONNX at models/vcm/a0_scratch.onnx (run 03_export_onnx.py)",
)
def test_onnx_session_loads_and_runs():
    from hey_swift.vcm.taxonomy import load_taxonomy
    from hey_swift.vcm.features import FeatureConfig
    import yaml
    import numpy as np

    cfg = yaml.safe_load(open("configs/vcm.yaml", encoding="utf-8")) or {}
    tax = load_taxonomy("configs/vcm.yaml")
    feat = FeatureConfig.from_dict(cfg.get("audio", {}))

    sess = ort.InferenceSession("models/vcm/a0_scratch.onnx",
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal(
        (1, 1, feat.n_mels, 197)).astype(np.float32)
    logits = sess.run(None, {in_name: x})[0]
    assert logits.shape == (1, tax.n_way_head)


@pytest.mark.skipif(
    not (Path("models/vcm/a0_scratch.onnx").exists()),
    reason="no exported ONNX (run 03_export_onnx.py)",
)
def test_torch_onnx_parity_within_tol():
    import numpy as np
    from hey_swift.vcm.models import ArchSpec, build_model
    from hey_swift.vcm.taxonomy import load_taxonomy
    from hey_swift.vcm.features import FeatureConfig
    import yaml

    cfg = yaml.safe_load(open("configs/vcm.yaml", encoding="utf-8")) or {}
    tax = load_taxonomy("configs/vcm.yaml")
    spec = ArchSpec.a0(n_way=tax.n_way_head)
    model = build_model(spec)
    ckpt = Path("experiments/vcm/a0_scratch/best.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    feat = FeatureConfig.from_dict(cfg.get("audio", {}))
    sess = ort.InferenceSession("models/vcm/a0_scratch.onnx",
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    for _ in range(8):
        x = rng.standard_normal((1, 1, feat.n_mels, 197)).astype(np.float32)
        with torch.no_grad():
            y_t = model(torch.from_numpy(x)).numpy()
        y_o = sess.run(None, {in_name: x})[0]
        assert float(np.abs(y_t - y_o).max()) < 1e-3
