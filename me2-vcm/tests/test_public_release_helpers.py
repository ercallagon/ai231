"""Review-copy privacy and opt-in asset/training contracts; no real effects."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def entry(tmp_path, data=b"licensed fixture"):
    path = tmp_path / "owned_fixture.bin"; path.write_bytes(data)
    return {"path": "integration/assets/fixture.bin", "local_file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "use_authorized": True, "terms_reference": "self-owned unit-test fixture"}


def test_public_copy_removes_only_known_private_defaults():
    module = load("stage_public_source")
    source = 'DEFAULT_CALL_NUMBER = "private-call"\nDEFAULT_MESSAGE_NUMBER = "private-message"\nDEFAULT_PHONE_MAC = "private-device"\nOTHER = "unchanged"\n'
    new, constants = module.public_config(source)
    assert 'OTHER = "unchanged"' in new
    for value in constants.values(): assert value not in new
    assert set(constants) == module.PRIVATE_NAMES
    assert "private-call" in source


def test_public_copy_refuses_unrecognized_declarations():
    with pytest.raises(ValueError): load("stage_public_source").public_config("OTHER = 1\n")


def test_source_snapshot_includes_package_initializers(tmp_path, monkeypatch):
    module = load("stage_public_source")
    root = tmp_path / "repo"
    (root / "src/hey_swift/integration").mkdir(parents=True)
    (root / "src/hey_swift/__init__.py").write_text('__version__ = "test"\n')
    (root / "src/hey_swift/integration/config.py").write_text(
        'DEFAULT_CALL_NUMBER = "PRIVATE_CALL_FIXTURE_483"\nDEFAULT_MESSAGE_NUMBER = "PRIVATE_MESSAGE_FIXTURE_483"\nDEFAULT_PHONE_MAC = "PRIVATE_DEVICE_FIXTURE_483"\n')
    for name in ("README.md", "LICENSE", "pyproject.toml", ".gitignore", "requirements.txt",
                 "requirements-pi.txt", "run.sh", "Makefile", "data/README.md", "models/README.md", "deploy/rpi/README.md"):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("")
    monkeypatch.setattr(module, "ROOT", root)
    destination = root / "work/public_snapshot"
    module.stage(destination)
    assert (destination / "src/hey_swift/__init__.py").read_text() == '__version__ = "test"\n'


def test_asset_local_copy_and_existing_destination(tmp_path):
    module = load("prepare_runtime_assets")
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"assets": [entry(tmp_path)]}))
    destination = tmp_path / "new-stage"
    receipt = module.prepare(manifest, destination)
    assert receipt[0]["sha256"] == module.digest(destination / "integration/assets/fixture.bin")
    with pytest.raises(FileExistsError): module.prepare(manifest, destination)


def test_asset_mismatch_keeps_failed_stage(tmp_path):
    module = load("prepare_runtime_assets")
    value = entry(tmp_path); value["sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"assets": [value]}))
    destination = tmp_path / "failed-stage"
    with pytest.raises(ValueError, match="fingerprint mismatch"): module.prepare(manifest, destination)
    assert (destination / value["path"]).exists()
    assert not (destination / "ASSET_RECEIPT.json").exists()


@pytest.mark.parametrize("change", [{"path": "../escape"}, {"use_authorized": False},
                                    {"sha256": "unknown"}, {"bytes": -1}])
def test_asset_manifest_fails_before_files_created(tmp_path, change):
    module = load("prepare_runtime_assets")
    value = entry(tmp_path); value.update(change)
    with pytest.raises(ValueError): module.validate_entries([value])


def test_network_fetch_is_explicit_and_credentials_refused(tmp_path):
    module = load("prepare_runtime_assets")
    value = entry(tmp_path); value.pop("local_file"); value["url"] = "https://example.com/fixture"
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"assets": [value]}))
    with patch.object(module.urllib.request, "urlopen", side_effect=AssertionError("No network")):
        with pytest.raises(ValueError, match="--allow-network"): module.prepare(manifest, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
    value["url"] = "https://user:password@example.com/fixture"
    with pytest.raises(ValueError): module.validate_entries([value])


def test_retraining_defaults_to_plan_without_cuda_or_files(monkeypatch):
    module = load("retrain_public_bc8")
    monkeypatch.chdir(ROOT)
    with patch("sys.argv", ["retrain", "--tag", "public_retrain_plan_only_unit_test"]), \
         patch.object(module, "gpu_gate", side_effect=AssertionError("No GPU access in plan")):
        module.main()
    assert not (ROOT / "experiments/vcm/public_retrain_plan_only_unit_test").exists()
