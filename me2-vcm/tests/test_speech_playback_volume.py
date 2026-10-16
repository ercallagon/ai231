import json
import sys
import types
import wave
from pathlib import Path

import pytest

from hey_swift.audio import speech


def test_tts_uses_per_stream_volume_and_validates_override(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(speech.sys, "platform", "linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/pw-play" if name == "pw-play" else None)
    monkeypatch.setattr(speech.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.delenv("VCM_SPEECH_CHANNEL_VOLUME", raising=False)
    speech.play_speech_wav(tmp_path / "phrase.wav")
    monkeypatch.setenv("VCM_SPEECH_CHANNEL_VOLUME", "0.72")
    monkeypatch.setenv("VCM_SPEECH_SINK", "alsa_output.usb-test")
    speech.play_speech_wav(tmp_path / "phrase.wav")
    assert [call[0][2] for call in calls] == ["0.5", "0.72"]
    assert all(call[0][0] == "pw-play" and call[0][1] == "--volume" for call in calls)
    assert calls[1][0][calls[1][0].index("--target") + 1] == "alsa_output.usb-test"
    for command, _ in calls:
        properties = json.loads(command[command.index("--properties") + 1])
        assert properties["state.restore-props"] is False
        assert properties["state.restore-target"] is False
        assert properties["media.role"] == "Notification"
        assert properties["application.id"] == "org.hey-swift.speech"

    for value in ("nan", "inf", "-0.1", "1.1", "loud"):
        with monkeypatch.context() as scoped:
            scoped.setenv("VCM_SPEECH_CHANNEL_VOLUME", value)
            with pytest.raises(speech.SpeechUnavailable, match="number from 0 to 1"):
                speech.play_speech_wav(tmp_path / "phrase.wav")


def test_cached_and_new_piper_phrases_use_same_tts_stream_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(speech.sys, "platform", "linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/pw-play" if name == "pw-play" else None)
    commands = []
    monkeypatch.setattr(speech.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setenv("VCM_SPEECH_CHANNEL_VOLUME", "0.5")

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"voice")
    config = Path(str(model) + ".json")
    config.write_text("{}")
    phrases = tmp_path / "phrases"
    phrases.mkdir()
    cached_wav = phrases / "cached.wav"
    with wave.open(str(cached_wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)
    import hashlib
    manifest = {
        "voice_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "voice_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "phrases": {"Cached phrase.": {"file": "cached.wav",
                                           "sha256": hashlib.sha256(cached_wav.read_bytes()).hexdigest()}},
    }
    (phrases / "manifest.json").write_text(json.dumps(manifest))

    def synthesize(text, output):
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)

    fake_voice = types.SimpleNamespace(synthesize_wav=synthesize)
    fake_piper = types.SimpleNamespace(PiperVoice=types.SimpleNamespace(
        load=lambda *_args, **_kwargs: fake_voice))
    monkeypatch.setitem(sys.modules, "piper", fake_piper)
    backend = speech.PiperSpeech(model, phrases_dir=phrases)
    backend.speak("Cached phrase.")
    backend.speak("Fresh phrase.")
    assert len(commands) == 2
    assert all(command[:3] == ["pw-play", "--volume", "0.5"] for command in commands)
    assert commands[0][-1] == str(cached_wav)
    assert commands[1][-1].endswith(".wav") and commands[1][-1] != str(cached_wav)
    backend.close()


def test_tts_refuses_uncontrollable_attenuation_without_global_gain(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(speech.sys, "platform", "linux")
    monkeypatch.setattr(speech.shutil, "which", lambda _name: None)
    monkeypatch.setattr(speech.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.delenv("VCM_SPEECH_CHANNEL_VOLUME", raising=False)
    with pytest.raises(speech.SpeechUnavailable, match="no global audio setting was changed"):
        speech.play_speech_wav(tmp_path / "phrase.wav")
    assert calls == []


def test_cue_has_own_identity_and_ignores_speech_gain_and_target(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(speech.sys, "platform", "linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/pw-play" if name == "pw-play" else None)
    monkeypatch.setattr(speech.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setenv("VCM_SPEECH_CHANNEL_VOLUME", "0.1")
    monkeypatch.setenv("VCM_SPEECH_SINK", "speech-only-target")

    speech.play_wav(tmp_path / "cue.wav")
    speech.play_wav(tmp_path / "cue.wav", sink="cue-only-target")
    speech.play_speech_wav(tmp_path / "speech.wav")

    assert [command[2] for command in commands] == ["1", "1", "0.1"]
    assert [command[command.index("--target") + 1] for command in commands] == [
        "auto", "cue-only-target", "speech-only-target"]
    properties = [json.loads(command[command.index("--properties") + 1]) for command in commands]
    assert [props["application.id"] for props in properties] == [
        "org.hey-swift.cue", "org.hey-swift.cue", "org.hey-swift.speech"]
    assert all(props["state.restore-props"] is False and props["state.restore-target"] is False
               for props in properties)


@pytest.mark.parametrize("sink", [None, "cue-only-target"])
def test_actual_ack_uses_state_isolated_cue_playback(monkeypatch, tmp_path, sink):
    from hey_swift.integration import e2e_loop

    path = tmp_path / "cue.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 160)
    commands = []
    events = []
    monkeypatch.setattr(speech.sys, "platform", "linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/pw-play" if name == "pw-play" else None)
    monkeypatch.setattr(speech.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(e2e_loop, "_emit", events.append)

    e2e_loop._play_ack(object(), path, sink=sink)

    assert len(commands) == 1
    command = commands[0]
    properties = json.loads(command[command.index("--properties") + 1])
    assert properties["application.id"] == "org.hey-swift.cue"
    assert properties["state.restore-props"] is False
    assert properties["state.restore-target"] is False
    assert command[command.index("--target") + 1] == (sink or "auto")
    assert any(event.get("event") == "ack" and event.get("status") == "ok" for event in events)
