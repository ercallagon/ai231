import json

import pytest

from me3_voice.configuration import ConfigurationError
from me3_voice.credentials import load_secrets, update_secrets
from me3_voice.setup_wizard import (
    _atomic_json,
    _private_prompt,
    _validate_device,
    select_paths,
)


def test_update_secrets_appends_and_selectively_replaces(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    path = cache / "providers.env"
    update_secrets(path, {"ME3_LLM_API_KEY": "old-llm", "ME3_TTS_API_KEY": "tts"}, cache_root=cache)
    update_secrets(
        path,
        {"ME3_LLM_API_KEY": "new-llm", "ME3_RESEARCH_TAVILY_API_KEY": "tavily"},
        cache_root=cache,
    )

    assert load_secrets(path, cache_root=cache) == {
        "ME3_LLM_API_KEY": "new-llm",
        "ME3_TTS_API_KEY": "tts",
        "ME3_RESEARCH_TAVILY_API_KEY": "tavily",
    }
    assert path.stat().st_mode & 0o777 == 0o600


def test_update_secrets_refuses_invalid_and_symlink_paths(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "real.env"
    target.write_text("ME3_LLM_API_KEY=kept\n")
    target.chmod(0o600)
    link = cache / "linked.env"
    link.symlink_to(target)

    with pytest.raises(ConfigurationError):
        update_secrets(link, {"ME3_TTS_API_KEY": "new"}, cache_root=cache)
    with pytest.raises(ConfigurationError):
        update_secrets(target, {"ME3_TTS_API_KEY": ""}, cache_root=cache)
    assert load_secrets(target, cache_root=cache) == {"ME3_LLM_API_KEY": "kept"}


def test_select_paths_uses_active_profile_layout(tmp_path):
    pi = tmp_path / "var/cache/pi_live-backends.json"
    pi.parent.mkdir(parents=True)
    pi.write_text("{}")
    paths = select_paths("pi", root=tmp_path)
    assert paths["backend"] == pi
    assert paths["secrets"] == tmp_path / "var/cache/demo-models.env"
    assert paths["device"] == tmp_path / "var/cache/pi_live-devices.json"
    assert paths["runtime"] == tmp_path / "var/cache/pi-five-turn-runtime.json"

    laptop = select_paths("laptop", root=tmp_path)
    assert laptop["backend"] == tmp_path / "configs/native-demo-backends.json"
    assert laptop["secrets"] == tmp_path / "var/cache/selected-provider-credentials.env"

    primary = select_paths("pi-primary", root=tmp_path)
    assert primary["backend"] == laptop["backend"]
    assert primary["runtime"] == laptop["runtime"]
    assert primary["device"] == paths["device"]
    assert primary["secrets"] == paths["secrets"]


def test_device_json_update_is_atomic_private_and_preserves_fields(tmp_path):
    path = tmp_path / "var/cache/devices.json"
    path.parent.mkdir(parents=True)
    original = {
        "schema_version": 1,
        "backend": "mock",
        "weather": {"provider": "open_meteo", "enabled": False},
        "reminder_store": "var/cache/custom.json",
    }
    path.write_text(json.dumps(original))
    path.chmod(0o600)
    changed = dict(original, backend="pi")
    _atomic_json(path, changed, root=tmp_path)

    assert json.loads(path.read_text()) == changed
    assert path.stat().st_mode & 0o777 == 0o600


def test_contacts_require_private_pi_profile_without_opening_hardware(tmp_path):
    config = {
        "schema_version": 1,
        "backend": "pi",
        "communications": {
            "enabled": True,
            "phone_address": "AA:BB:CC:DD:EE:FF",
            "contacts": [{"name": "Sam", "aliases": ["friend"], "phone_number": "+15557654321"}],
        },
    }
    path = tmp_path / "devices.json"
    path.write_text("{}")
    path.chmod(0o600)
    assert _validate_device(config, path) == config

    with pytest.raises(ValueError, match="Pi target"):
        _validate_device(dict(config, backend="mock"), path)


def test_enabled_phone_config_accepts_paired_bluetooth_mac_without_opening_bus():
    from me3_voice.communications import Communications

    phone = Communications(
        enabled=True,
        phone_address="AA:BB:CC:DD:EE:FF",
        contacts=[{"name": "Sam", "aliases": [], "phone_number": "+15557654321"}],
    )
    assert phone.sms.address == "AA:BB:CC:DD:EE:FF"
    assert phone.match_contact("Sam")["status"] == "matched"
    with pytest.raises(ValueError, match="Bluetooth MAC"):
        Communications(
            enabled=True,
            phone_address="+15551234567",
            contacts=[{"name": "Sam", "aliases": [], "phone_number": "+15557654321"}],
        )


def test_private_prompt_blank_keeps_and_bang_clears():
    assert _private_prompt("field", "existing", prompt=lambda _: "") == "existing"
    assert _private_prompt("field", "existing", prompt=lambda _: "!") == ""
    assert _private_prompt("field", None, prompt=lambda _: "") is None
