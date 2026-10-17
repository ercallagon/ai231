"""Final workflow controls; no audio devices, network or real services."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hey_swift import deployment


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("latency", [False, True])
@pytest.mark.parametrize("endpoint", [None, 800., 1200.])
def test_public_endpoint_is_explicit_not_a_changed_default(monkeypatch, tmp_path, latency, endpoint):
    commands = []
    monkeypatch.setattr(deployment, "microphone", lambda cfg: "0")
    monkeypatch.setattr(deployment, "isolated_env", lambda *args: {})
    monkeypatch.setattr(deployment.subprocess, "call", lambda cmd, **kw: commands.append(cmd) or 0)
    monkeypatch.setattr(deployment.subprocess, "check_call", lambda *a, **kw: 0)
    cfg = dict(hardware="mock")
    assert deployment.assistant(cfg, tmp_path, latency=latency, end_silence_ms=endpoint) == 0
    command = commands[0]
    if endpoint is None:
        assert "--end-silence-ms" not in command
    else:
        assert command[command.index("--end-silence-ms") + 1] == str(endpoint)
    assert "--live" not in command


def test_live_runtime_default_endpoint_is_pi_validated_800ms(monkeypatch):
    from hey_swift.integration import e2e_loop
    captured = {}
    def capture(args):
        captured["args"] = args
        return 0
    monkeypatch.setattr(e2e_loop, "run", capture)
    assert e2e_loop.main(["--vcm-onnx", "model.onnx"]) == 0
    assert captured["args"].end_silence_ms == 800.0


@pytest.mark.parametrize("profile", ["champion", "baseline"])
@pytest.mark.parametrize("endpoint", [None, 800., 1200.])
def test_profile_launcher_uses_runtime_endpoint_over_historical_yaml(monkeypatch, profile, endpoint):
    from hey_swift.integration import e2e_loop
    module = script("run_assistant")
    captured = {}
    monkeypatch.setattr(e2e_loop, "run", lambda args: captured.setdefault("args", args) and 0)
    argv = ["run_assistant.py", "--profile", profile, "--device", "0"]
    if endpoint is not None:
        argv += ["--end-silence-ms", str(endpoint)]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    assert captured["args"].end_silence_ms == (800. if endpoint is None else endpoint)
    assert captured["args"].pre_roll_ms == 200.


def test_catalog_refresh_updates_docs_but_preserves_frozen_hashes(tmp_path):
    module = script("refresh_artifact_catalog")
    (tmp_path / "models").mkdir()
    model = tmp_path / "models/frozen.onnx"
    model.write_bytes(b"fixed")
    doc = tmp_path / "README.md"
    doc.write_text("before")
    hashes = {"models/frozen.onnx": module.digest(model), "README.md": module.digest(doc)}
    (tmp_path / "ARTIFACTS.json").write_text(json.dumps({"sha256": hashes}))
    doc.write_text("after")
    payload, changed = module.refresh_catalog(tmp_path, refresh=True)
    assert changed == ["README.md"]
    assert payload["sha256"]["README.md"] == module.digest(doc)
    assert payload["sha256"]["models/frozen.onnx"] == hashes["models/frozen.onnx"]


@pytest.mark.parametrize("frozen", ["models/frozen.onnx", "reports/CHECKS.json"])
def test_catalog_refresh_refuses_changed_frozen_artifact(tmp_path, frozen):
    module = script("refresh_artifact_catalog")
    model = tmp_path / frozen
    model.parent.mkdir()
    model.write_bytes(b"fixed")
    catalog = tmp_path / "ARTIFACTS.json"
    catalog.write_text(json.dumps({"sha256": {frozen: module.digest(model)}}))
    before = catalog.read_bytes()
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        module.refresh_catalog(tmp_path, refresh=True)
    assert catalog.read_bytes() == before


def test_catalog_refresh_refuses_private_paths(tmp_path):
    module = script("refresh_artifact_catalog")
    (tmp_path / "ARTIFACTS.json").write_text('{"sha256": {}}')
    private = tmp_path / "work/user/profile.json"
    private.parent.mkdir(parents=True)
    private.write_text("{}")
    with pytest.raises(ValueError, match="Not a release path"):
        module.refresh_catalog(tmp_path, refresh=True, include=["work/user/profile.json"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1.])
def test_invalid_endpoint_fails_before_starting_any_process(value, tmp_path):
    with pytest.raises(ValueError, match="finite and positive"):
        deployment.assistant({}, tmp_path, end_silence_ms=value)


def test_revision_rejects_dirty_or_wrong_checkout(monkeypatch, tmp_path):
    module = script("record_submission")
    root = tmp_path / "me2-vcm"
    root.mkdir()
    def output(args, **kw):
        if "--show-toplevel" in args:
            return str(tmp_path)
        if "status" in args:
            return " M README.md"
        return "a" * 40
    monkeypatch.setattr(module.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="Tracked project edits"):
        module.revision(root)
    with pytest.raises(ValueError, match="expected tested commit"):
        module.revision(root, "b" * 40)
    with pytest.raises(ValueError, match="canonical"):
        module.revision(tmp_path / "wrong")


def test_recorder_refuses_existing_output_before_collect(monkeypatch, tmp_path):
    module = script("record_submission")
    output = tmp_path / "keep.json"
    output.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["record_submission.py", "--output", str(output)])
    monkeypatch.setattr(module, "collect", lambda *a: pytest.fail("Must not collect on overwrite"))
    with pytest.raises(SystemExit):
        module.main()
    assert output.read_text() == "preserve"


def test_inventory_identifies_sensitive_paths_without_printing_values(tmp_path):
    module = script("prepare_submission_inventory")
    (tmp_path / "README.md").write_text("public")
    private = tmp_path / "work/user/profile.json"
    private.parent.mkdir(parents=True)
    private.write_text(json.dumps({"key": "ghp_" + "A" * 36}))
    report = module.tracked_inventory(tmp_path, ["README.md", "work/user/profile.json"])
    assert report["file_count"] == 2
    assert not report["automated_checks_pass"]
    assert {x["kind"] for x in report["issues"]} == {"github_token", "private_or_generated_directory"}
    assert "A" * 36 not in json.dumps(report)


def test_report_does_not_import_stale_acoustic_audit(tmp_path):
    import hashlib
    from hey_swift.submission_report import build_report
    folder = tmp_path / "official"
    folder.mkdir()
    (folder / "metrics.json").write_text('{"accuracy": 0.8}')
    (folder / "trials.csv").write_text("trial\n1\n")
    audit = {"source_sha256": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ("metrics.json", "trials.csv")}, "early_decision_trials": {"n": 2}}
    (folder / "SWIFT_AUDIT.json").write_text(json.dumps(audit))
    assert build_report(tmp_path, folder)["measured"]["supplementary_acoustic_audit"] == audit
    (folder / "trials.csv").write_text("changed\n")
    result = build_report(tmp_path, folder)
    assert "supplementary_acoustic_audit" not in result["measured"]
    assert any("Matching SWIFT_AUDIT" in item for item in result["missing"])
