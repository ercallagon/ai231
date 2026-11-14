"""Offline checks that model profiles bind to provider-qualified credentials."""

import json
from pathlib import Path

from me3_voice.credentials import load_secrets, write_secrets

ROOT = Path(__file__).resolve().parents[1]


def test_native_smoke_profile_binds_models_to_provider_specific_keys():
    config = json.loads((ROOT / "configs/native-speech.provisional.json").read_text())
    assert (config["asr"]["protocol"], config["asr"]["model"], config["asr"]["auth_env"]) == (
        "deepgram-listen-v1",
        "nova-3",
        "ME3_DEEPGRAM_API_KEY",
    )
    assert (config["llm"]["endpoint"], config["llm"]["model"], config["llm"]["auth_env"]) == (
        "https://api.groq.com/openai/v1/chat/completions",
        "openai/gpt-oss-120b",
        "ME3_GROQ_API_KEY",
    )
    assert (config["tts"]["protocol"], config["tts"]["model"], config["tts"]["auth_env"]) == (
        "deepgram-speak-v1",
        "aura-2-thalia-en",
        "ME3_DEEPGRAM_API_KEY",
    )
    assert all(
        not config[role]["allow_live"] and config[role]["request_limit"] == 0 for role in config
    )


def test_qwen_profile_keys_are_distinct_by_role():
    config = json.loads((ROOT / "var/cache/voice-backends-qwen3-asr.json").read_text())
    assert config["asr"]["auth_env"] == "ME3_QWEN_ASR_API_KEY"
    assert config["llm"]["auth_env"] == "ME3_QWEN_LLM_API_KEY"
    assert config["tts"]["auth_env"] == "ME3_QWEN_TTS_API_KEY"
    assert len({config[role]["auth_env"] for role in config}) == 3


def test_private_loader_accepts_provider_qualified_names_without_cross_mapping(
    tmp_path, monkeypatch
):
    import me3_voice.credentials as credentials

    monkeypatch.setattr(credentials, "CACHE_ROOT", tmp_path)
    path = tmp_path / "provider.env"
    write_secrets(
        path,
        {"ME3_DEEPGRAM_API_KEY": "dg-test", "ME3_GROQ_API_KEY": "groq-test"},
    )
    assert load_secrets(path, {"ME3_DEEPGRAM_API_KEY"}) == {"ME3_DEEPGRAM_API_KEY": "dg-test"}
    assert load_secrets(path, {"ME3_GROQ_API_KEY"}) == {"ME3_GROQ_API_KEY": "groq-test"}
