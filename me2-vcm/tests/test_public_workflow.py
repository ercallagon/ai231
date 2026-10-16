import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hey_swift import deployment as public
from hey_swift.profiles import resolve_profile, DISPLAY
from hey_swift.submission_report import build_report, count_rate


def configuration():
    return dict(enabled=["information"], hardware="mock", secrets={}, microphone=dict(name="USB", hostapi="ALSA"))


def test_aliases_preserve_profiles_hashes_and_thresholds():
    import yaml
    registry = json.loads((ROOT / "configs/assistant_profiles.json").read_text())
    refs = json.loads((ROOT / "data/model_reference.json").read_text())["models"]
    for name, legacy in (("champion", "selected"), ("baseline", "run043")):
        assert resolve_profile(name) == legacy and resolve_profile(legacy) == legacy
        profile = registry["profiles"][legacy]
        assert hashlib.sha256((ROOT / profile["model"]).read_bytes()).hexdigest() == refs[name]["model_sha256"]
        assert yaml.safe_load((ROOT / profile["config"]).read_text())["oos"]["confidence_threshold"] == refs[name]["threshold"]
    assert DISPLAY == {"selected": "Champion (6s BC8-TCN-ASP)", "run043": "Baseline (2s BC8)"}


def test_isolation_scrubs_existing_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("VCM_SOLOIST_PROFILE", "standalone")
    monkeypatch.setenv("VCM_CALL_NUMBER", "+639999999999")
    monkeypatch.setenv("VCM_SOLOIST_API_KEY", "old-key")
    monkeypatch.setenv("HEY_SWIFT_RUNTIME_ROOT", "/old/deployment")
    env = public.isolated_env(configuration(), tmp_path / "runtime")
    assert env["VCM_SOLOIST_PROFILE"] == "bundled"
    assert env["VCM_CALL_NUMBER"] == "" and env["VCM_SOLOIST_API_KEY"] == ""
    assert env["HEY_SWIFT_RUNTIME_ROOT"] == str(tmp_path / "runtime")
    assert env["VCM_SOLOIST_WS"] == "127.0.0.1:0"


def test_private_storage_and_backups(tmp_path):
    path = tmp_path / "private/profile.json"
    public.private_json(path, dict(secret="first"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    public.private_json(path, dict(secret="second"))
    assert json.loads(path.read_text())["secret"] == "second"
    backups = list(path.parent.glob("profile.previous_*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text())["secret"] == "first"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="600"):
        public.load_config(path.parent)


def test_microphone_identity_survives_reindex_but_not_ambiguity(monkeypatch):
    device = dict(index=7, name="USB", hostapi="ALSA", inputs=1)
    monkeypatch.setattr(public, "devices", lambda: [device])
    assert public.microphone(configuration()) == "7"
    monkeypatch.setattr(public, "devices", lambda: [device, {**device, "index": 8}])
    with pytest.raises(ValueError, match="ambiguous"):
        public.microphone(configuration())


def test_gpio_self_test_records_operator_confirmed_physical_result(monkeypatch, tmp_path):
    import hey_swift.hardware.devices as hardware_devices

    class FakePiHardware:
        def __init__(self):
            self.events = []
        def led(self, rgb):
            self.events.append(("led", rgb))
        def display(self, digits, colon=False):
            self.events.append(("display", digits, colon))
        def close(self):
            self.events.append(("close",))

    config_home = tmp_path / "user"
    public.private_json(config_home / "profile.json", {"hardware": "pi"})
    monkeypatch.setattr(public, "ROOT", tmp_path)
    monkeypatch.setattr(public.time, "sleep", lambda _: None)
    monkeypatch.setattr(hardware_devices, "PiHardware", FakePiHardware)
    answers = iter((True, True, True))
    monkeypatch.setattr(public, "yes", lambda _: next(answers))

    assert public.gpio_self_test(config_home) == 0
    result = json.loads((config_home / "gpio_check.json").read_text())
    assert result["status"] == "PASS"
    assert result["observations"] == {"rgb_sequence": True, "display_1234_colon": True}


def test_wizard_hides_private_numbers(monkeypatch, tmp_path, capsys):
    answers = iter(["0", "n", "n", "n", "n", "n", "y", "n", "n"])
    secrets = iter(["+639123456789", "AA:BB:CC:DD:EE:FF"])
    monkeypatch.setattr(public.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(public.getpass, "getpass", lambda prompt="": next(secrets))
    monkeypatch.setattr(public, "devices", lambda: [dict(index=0, name="USB", hostapi="ALSA", inputs=1, outputs=0)])
    cfg = public.wizard(tmp_path)
    assert cfg["enabled"] == ["calling"]
    assert cfg["secrets"]["calling_number"] == "+639123456789"
    output = capsys.readouterr().out
    assert "+639123456789" not in output and "AA:BB:CC:DD:EE:FF" not in output
    assert (tmp_path / "profile.json").stat().st_mode & 0o777 == 0o600


def test_session_wizard_does_not_write_profile(monkeypatch, tmp_path):
    answers = iter(["0", "y", "n", "n", "n", "n", "n", "n", "n"])
    monkeypatch.setattr(public.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(public, "devices", lambda: [dict(index=0, name="USB", hostapi="ALSA", inputs=1, outputs=0)])
    cfg = public.wizard(tmp_path, session=True)
    assert "information" in cfg["enabled"]
    assert not (tmp_path / "profile.json").exists()


def test_default_spotify_profile_does_not_reuse_saved_login(monkeypatch, tmp_path):
    from hey_swift.integration.config import spotify_paths
    monkeypatch.delenv("VCM_SOLOIST_PROFILE", raising=False)
    for key in ("VCM_SOLOIST_BIN", "VCM_SOLOIST_DATA_DIR", "VCM_SOLOIST_KEY_FILE"):
        monkeypatch.delenv(key, raising=False)
    old = tmp_path / "vcm-modular-test"
    (old / "work/tools/soloist").mkdir(parents=True)
    (old / "work/tools/soloist/soloist").touch()
    (old / "work/soloist-runtime").mkdir()
    (old / "work/soloist-runtime/api-key").write_text("old-key")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    chosen = spotify_paths(tmp_path / "new/integration")
    assert chosen["profile"] == "bundled"
    assert old not in chosen["data_dir"].parents
    monkeypatch.setenv("VCM_SOLOIST_PROFILE", "standalone")
    assert spotify_paths(tmp_path / "new/integration")["data_dir"] == old / "work/soloist-runtime"


def test_only_enabled_services_are_constructed(monkeypatch, tmp_path):
    import hey_swift.integration.runtime as runtime
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEY_SWIFT_ENABLED_INTEGRATIONS", "")
    monkeypatch.setattr(runtime, "ScheduledDevices", lambda *a, **k: pytest.fail("GPIO service should not be constructed"))
    monkeypatch.setattr(runtime, "IntegrationSoloist", lambda *a, **k: pytest.fail("Spotify should not be constructed"))
    router = runtime.build_router(live=True, hardware="pi")
    assert router.services == {}
    router.close()


def test_communication_requires_terminal_confirmation(monkeypatch):
    from hey_swift.integration.runtime import ConfirmedCommunication
    calls = []
    wrapped = ConfirmedCommunication(SimpleNamespace(execute=lambda *a, **k: calls.append(a) or "sent"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert wrapped.execute("CALL").status == "cancelled" and not calls
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda p: "no")
    assert wrapped.execute("MESSAGE").status == "cancelled" and not calls
    monkeypatch.setattr("builtins.input", lambda p: "CALL")
    assert wrapped.execute("CALL") == "sent" and len(calls) == 1


def test_report_marks_missing_results_and_never_reads_credentials(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/model_reference.json").write_text("{}")
    private = tmp_path / "work/user/profile.json"
    private.parent.mkdir(parents=True)
    private.write_text('{"secret":"must-not-leak"}')
    report = build_report(tmp_path)
    assert "test: no local injection results" in report["missing"]
    assert "must-not-leak" not in (tmp_path / "work/pi5_test_01/REPORT.json").read_text()
    assert count_rate(None) == "MISSING" and count_rate(dict(n=0)) == "N/A (n=0)"


def test_dataset_fetch_ignores_stale_fixed_download_and_keeps_it(monkeypatch, tmp_path):
    import hashlib
    import io
    from hey_swift.vcm import release_data as data

    payload = b"pinned parquet fixture"
    relative = "data/test.parquet"
    monkeypatch.setattr(data, "catalog", lambda: {relative: (len(payload), hashlib.sha256(payload).hexdigest())})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    stale = destination.with_suffix(".download")
    stale.write_bytes(b"partial from an earlier attempt")
    row = {"parquet": "data/vcm/raw/" + relative}

    data.ensure_files([row], tmp_path, fetch=True)

    assert destination.read_bytes() == payload
    assert stale.read_bytes() == b"partial from an earlier attempt"
    assert not list(destination.parent.glob("." + destination.name + ".*.download"))


def test_install_refuses_existing_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(public, "ROOT", tmp_path)
    monkeypatch.setattr(public.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(public.sys, "version_info", (3, 13, 5))
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(public.subprocess, "check_call", lambda *a, **k: pytest.fail("Must not modify environment"))
    with pytest.raises(ValueError, match="already exists"):
        public.install()


def test_injection_aliases_and_slot_metrics_with_synthetic_fixture(monkeypatch, tmp_path):
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    import reproduce
    from hey_swift.vcm import release_data as data, archived_metric_core as core
    rows = [dict(audio_path="fixture1", intent="LIGHT_ON", slot="", is_synthetic="1"),
            dict(audio_path="fixture2", intent="TEMPERATURE", slot="22", is_synthetic="0")]
    y = np.array([data.tax().intent_to_idx[r["intent"]] for r in rows])
    def labels(*args):
        return y, np.array([-1, 1]), np.array([False, False])
    monkeypatch.setattr(data, "manifest", lambda split: rows)
    monkeypatch.setattr(data, "ensure_files", lambda *args: None)
    monkeypatch.setattr(data, "labels", labels)
    monkeypatch.setattr(core, "labels", labels)
    wave = (.1*np.sin(np.arange(16000)*2*np.pi*440/16000)).astype(np.float32)
    monkeypatch.setattr(data, "waves", lambda *args: iter(enumerate([wave, wave])))
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "reproduce", "--profiles", "champion", "baseline",
                                    "--splits", "test", "--output", str(destination)])
    reproduce.main()
    result = json.loads((destination / "test_RESULTS.json").read_text())
    assert set(result) == {"selected", "run043", "paired_supported"}
    for name in ("selected", "run043"):
        assert result[name]["slot_accuracy_given_true_family"]["TEMPERATURE"]["n"] == 1
        assert result[name]["overall_intent"]["n"] == 2
