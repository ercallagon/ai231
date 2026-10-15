"""Publication helpers do not change deployment or mislabel timing results."""
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_timing_percentiles_and_rtf():
    measure = load("measure_pi_onnx")
    result = measure.timing_summary([10, 20, 30, 40, 50], 2)
    assert result["p50_ms"] == 30
    assert result["p95_ms"] == 48
    assert result["rtf_p95"] == .024
    with pytest.raises(ValueError): measure.timing_summary([], 2)
    with pytest.raises(ValueError): measure.timing_summary([np.nan], 2)
    with pytest.raises(ValueError): measure.timing_summary([1], 0)
    with pytest.raises(ValueError): measure.timing_summary([-1], 1)


def test_inventory_filters_generated_and_binary_files():
    inventory = load("prepare_submission_inventory")
    assert inventory.candidate(Path("src/hey_swift/vcm/features.py"))
    assert inventory.candidate(Path("src/hey_swift/__init__.py"))
    for name in ("src/__pycache__/cached.py", "scripts/_patch_build.py", "src/.env",
                 "src/assets/voice.onnx", "configs/features.npy"):
        assert not inventory.candidate(Path(name))


def test_pi_requirement_fails_before_model_execution():
    measure = load("measure_pi_onnx")
    with tempfile.TemporaryDirectory() as directory:
        args = ["measure", "--model", "absent.onnx", "--config", "absent.yaml", "--require-pi",
                "--output", str(Path(directory) / "result.json")]
        with patch("sys.argv", args), patch.object(measure, "hardware_identity", return_value={"is_raspberry_pi": False}), \
             patch.object(measure, "benchmark", side_effect=AssertionError("Do not run host as Pi")):
            with pytest.raises(SystemExit) as error: measure.main()
        assert error.value.code == 2


def test_benchmark_rejects_invalid_budgets():
    measure = load("measure_pi_onnx")
    with pytest.raises(ValueError): measure.benchmark("absent", "absent", iterations=0)
    with pytest.raises(ValueError): measure.benchmark("absent", "absent", threads=0)


def test_existing_result_is_never_overwritten():
    measure = load("measure_pi_onnx")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"; path.write_text("preserved")
        with patch("sys.argv", ["measure", "--model", "absent", "--config", "absent", "--output", str(path)]):
            with pytest.raises(FileExistsError): measure.main()
        assert path.read_text() == "preserved"


def test_long_speech_rejected_before_onnx_session(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("onnxruntime")
    measure = load("measure_pi_onnx")
    audio = tmp_path / "too_long.wav"
    sf.write(audio, np.zeros(8 * 16000, dtype=np.float32), 16000)
    with patch("onnxruntime.InferenceSession", side_effect=AssertionError("No silent crop/model run")):
        with pytest.raises(ValueError, match="no silent truncation"):
            measure.benchmark("absent.onnx", ROOT / "configs/vcm.yaml", audio=audio)
