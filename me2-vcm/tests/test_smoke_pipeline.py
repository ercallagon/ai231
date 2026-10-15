"""End-to-end SMOKE test: prove the full VCM pipeline works WITHOUT the real dataset.

Generates a tiny synthetic fixture (short WAVs + manifests + splits.json) in a
tmp dir, then runs the real train() -> evaluate() path on it with a reduced
config (few epochs, CPU, small model). This is the de-risking gate: when the
real HF data lands, the only new variable is the data itself, not the harness.

Skipped automatically if torch is not installed (import-safe for CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

# Use a handful of intents so the fixture stays tiny but exercises the head.
SUBSET = ["PLAY_MUSIC", "STOP", "TIMER", "WEATHER", "OUT_OF_SCOPE"]
N_PER = 4  # clips per intent per split


def _make_fixture(root: Path) -> Path:
    """Write WAVs + manifests + splits.json under root. Returns the repo-like root."""
    (root / "data/vcm/manifests").mkdir(parents=True, exist_ok=True)
    (root / "data/vcm/raw").mkdir(parents=True, exist_ok=True)
    (root / "data/vcm/processed/wav").mkdir(parents=True, exist_ok=True)
    (root / "experiments/vcm").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    sr = 16000
    dur = 0.4  # short clips -> fast smoke
    n = int(sr * dur)

    # Give each intent a distinct tone so the model can actually fit it.
    freq_for = {i: 200 + 120 * k for k, i in enumerate(SUBSET)}
    paths = {}
    for split in ("train", "val", "test", "holdout"):
        rows = []
        for intent in SUBSET:
            for k in range(N_PER):
                fn = f"{split}_{intent}_{k}.wav"
                # Exercise the real extracted-WAV path, not a sandbox fallback.
                p = root / "data/vcm/processed/wav" / fn
                t = np.arange(n) / sr
                # distinct tone per intent + noise; OOS is broadband noise
                if intent == "OUT_OF_SCOPE":
                    wav = rng.standard_normal(n).astype("float32") * 0.1
                else:
                    f = freq_for[intent]
                    wav = (0.3 * np.sin(2 * np.pi * f * t)).astype("float32")
                    wav += rng.standard_normal(n).astype("float32") * 0.02
                sf.write(str(p), wav, sr)
                spk = f"s{k % 3}"
                rows.append(f"{p},{intent},,{spk}")
        (root / "data/vcm/manifests" / f"{split}_19intent.csv").write_text(
            "audio_path,intent,slot,speaker\n" + "\n".join(rows) + "\n")
        paths[split] = rows

    # splits.json (matches 02_build_manifests.py schema)
    import json
    (root / "data/vcm/manifests/splits.json").write_text(json.dumps({
        "repo_id": "synthetic-fixture", "revision": "smoke",
        "counts": {s: len(paths[s]) for s in paths},
        "overlaps": {}, "disjoint": True,
        "note": "synthetic smoke fixture",
    }))
    return root


def test_full_pipeline_smoke(tmp_path, monkeypatch):
    root = _make_fixture(tmp_path)
    monkeypatch.chdir(root)  # train()/evaluate() use relative data/ + experiments/

    # Minimal config: CPU, 2 epochs, small model, short window to keep it fast.
    cfg = tmp_path / "vcm_smoke.yaml"
    cfg.write_text(
        "audio:\n  sample_rate: 16000\n  n_fft: 400\n  hop_length: 160\n"
        "  n_mels: 40\n  window_seconds: 0.5\n  f_min: 20.0\n  f_max: 8000.0\n"
        "schema:\n  n_intents: 5\n  oos_class: OUT_OF_SCOPE\n  n_way_head: 5\n"
        "  intents: [PLAY_MUSIC, STOP, TIMER, WEATHER]\n"
        "  slot_reject: false\n  slotted_intents: []\n  slot_sets: {}\n"
        "training:\n  seed: 42\n  epochs: 2\n  batch_size: 8\n  lr: 0.01\n"
        "  early_stop_patience: 3\n  device: cpu\n  gpu_index: 0\n"
        "oos:\n  confidence_threshold: 0.5\n"
    )
    # Point the dataset module at the fixture manifests.
    from hey_swift.vcm import dataset as ds
    monkeypatch.setattr(ds, "MANIFESTS_DIR", root / "data/vcm/manifests")

    from hey_swift.vcm.train import run
    out = run("smoke_test", config=str(cfg))
    assert (out / "best.pt").exists(), "best.pt checkpoint missing"
    assert (out / "result.json").exists(), "result.json missing"
    assert (out / "reproducibility.json").exists(), "reproducibility.json missing"

    import json
    res = json.loads((out / "result.json").read_text())
    assert res["epochs_run"] >= 1
    assert res["best_val_score"] >= 0.0
    assert len(res["history"]) == res["epochs_run"]

    # This fixture validates execution/contracts, not convergence in two epochs.
    assert np.isfinite([h["train_loss"] for h in res["history"]]).all()
    from hey_swift.vcm.evaluate import evaluate
    metrics = evaluate("smoke_test", "test", config=str(cfg))
    assert metrics["n"] == len(SUBSET) * N_PER
    assert metrics["slot_n"] == 0
    assert 0 <= metrics["intent_accuracy"] <= 1
