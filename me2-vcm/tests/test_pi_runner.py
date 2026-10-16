import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_pi as runner


def test_reset_archives_only_test_directory(tmp_path):
    target = tmp_path / "work/pi5_test_01"
    target.mkdir(parents=True)
    (target / "old.log").write_text("old evidence")
    cached = tmp_path / "data/vcm/raw/audio.parquet"
    cached.parent.mkdir(parents=True)
    cached.write_text("preserve dataset")
    model = tmp_path / "models/model.onnx"
    model.parent.mkdir()
    model.write_text("preserve model")
    fresh, archive = runner.prepare_results(tmp_path, reset=True)
    assert fresh == target and list(fresh.iterdir()) == []
    assert (archive / "old.log").read_text() == "old evidence"
    assert cached.read_text() == "preserve dataset"
    assert model.read_text() == "preserve model"


def test_existing_results_require_explicit_reset(tmp_path):
    target, _ = runner.prepare_results(tmp_path)
    (target / "marker").write_text("keep")
    with pytest.raises(ValueError, match="--reset"):
        runner.prepare_results(tmp_path)
    assert (target / "marker").read_text() == "keep"


def test_reset_refuses_symlink(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (work / "pi5_test_01").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner.prepare_results(tmp_path, reset=True)
    assert elsewhere.is_dir()


def result_fixture(root):
    (root / "configs").mkdir()
    profiles = {}
    for name in runner.PROFILES:
        path = root / (name + ".onnx")
        path.write_bytes(name.encode())
        profiles[name] = dict(model=path.name)
    (root / "configs/assistant_profiles.json").write_text(json.dumps(dict(profiles=profiles)))
    output, _ = runner.prepare_results(root)
    injection = output / "injection"
    injection.mkdir()
    (injection / "COMPLETE.json").write_text(json.dumps(dict(profiles=runner.PROFILES, splits=runner.SPLITS)))
    for split in runner.SPLITS:
        results = {}
        for name in runner.PROFILES:
            digest = hashlib.sha256((root / profiles[name]["model"]).read_bytes()).hexdigest()
            results[name] = dict(model_sha256=digest, infer_p95_ms=120, per_clip_rtf_p95=.1,
                infer_scope="features_plus_ONNX_no_decode_resampling_capture_actions", threads=1,
                physical_pi_verified=True, processed_subsets=dict(all=dict(manifest_n=2, processed_n=1, failures=1)))
            (injection / (split + "_" + name + "_timings.json")).write_text("[{}]")
        (injection / (split + "_RESULTS.json")).write_text(json.dumps(results))
    return output


def test_summary_uses_both_models_and_keeps_failure_counts(tmp_path):
    output = result_fixture(tmp_path)
    summary = runner.summarize(tmp_path, output)
    assert set(summary["profiles"]) == {"run043", "selected"}
    assert summary["profiles"]["selected"]["test"]["failures"] == 1
    assert summary["profiles"]["run043"]["holdout"]["timed_n"] == 1
    assert (output / "SUMMARY.json").is_file()


def test_summary_rejects_wrong_model_association(tmp_path):
    output = result_fixture(tmp_path)
    (tmp_path / "selected.onnx").write_bytes(b"different")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.summarize(tmp_path, output)
    assert not (output / "SUMMARY.json").exists()


def test_summary_requires_complete_run(tmp_path):
    output = result_fixture(tmp_path)
    (output / "injection/COMPLETE.json").write_text(json.dumps(dict(profiles=["selected"], splits=runner.SPLITS)))
    with pytest.raises(ValueError, match="both required"):
        runner.summarize(tmp_path, output)


def test_child_failure_is_logged_and_returned(tmp_path, capsys):
    log = tmp_path / "injection.log"
    code = runner.run_logged([sys.executable, "-c", "print('diagnostic'); raise SystemExit(3)"], tmp_path, log)
    assert code == 3
    assert log.read_text() == "diagnostic\n"
    assert "diagnostic" in capsys.readouterr().out
