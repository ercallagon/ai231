from __future__ import annotations

import sys
import types
import wave
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_workspace_initializes_missing_root_and_preserves_state(monkeypatch, tmp_path):
    import tempfile
    from hey_swift.integration import runtime

    root = tmp_path / "fresh_checkout" / "integration"
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", None)

    runtime.prepare_workspace()
    assert Path.cwd() == root
    assert tempfile.tempdir == str(root / "tmp")
    for name in ("state", "logs", "tmp"):
        assert (root / name).is_dir()
        assert (root / name).stat().st_mode & 0o777 == 0o700

    saved = root / "state" / "existing.json"
    saved.write_text("preserve me")
    runtime.prepare_workspace()
    assert saved.read_text() == "preserve me"


def test_piper_prewarms_and_reuses_file_backed_wav(monkeypatch, tmp_path):
    from hey_swift.audio.speech import PiperSpeech

    calls = []

    class FakeVoice:
        def synthesize_wav(self, text, output):
            calls.append(text)
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 160)

    fake_piper = types.SimpleNamespace(PiperVoice=types.SimpleNamespace(
        load=lambda *_args, **_kwargs: FakeVoice()))
    monkeypatch.setitem(sys.modules, "piper", fake_piper)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    Path(str(model) + ".json").write_text("{}")
    played = []
    speech = PiperSpeech(model, player=played.append)

    speech.prepare_phrases(["Lights on.", "Lights on."])
    speech.speak("Lights on.")
    speech.speak("The time is 9:42 AM.")
    speech.speak("The time is 9:42 AM.")

    assert calls == ["Lights on.", "The time is 9:42 AM."]
    assert len(played) == 3
    assert played[1] == played[2]
    assert all(path.is_file() for path in played)
    speech.close()


def test_music_volume_cap_is_applied_before_unpause():
    from hey_swift.actions.services.music import MpvBackend

    backend = MpvBackend([], "/unused", max_volume_percent=25)
    events = []

    def command(*args, **_kwargs):
        events.append(args)
        if args == ("get_property", "idle-active"):
            return False
        if args == ("get_property", "volume"):
            return 22
        return None

    backend.command = command
    backend.execute("PLAY_MUSIC")
    assert events[:3] == [
        ("set_property", "volume", 25),
        ("get_property", "idle-active"),
        ("set_property", "pause", False),
    ]

    events.clear()
    backend.execute("VOLUME_UP")
    assert events[-1] == ("set_property", "volume", 25)


def test_weather_prefetch_is_shared_with_first_request():
    from hey_swift.actions.services.weather import OpenMeteoWeatherProvider

    calls = []
    timestamp = int(datetime.now(timezone.utc).timestamp())
    payload = {
        "current_units": {"temperature_2m": "°C"},
        "current": {"temperature_2m": 27.0, "weather_code": 1, "time": timestamp},
    }

    def fetch(_url):
        calls.append(True)
        return payload

    provider = OpenMeteoWeatherProvider(fetch=fetch)
    provider.prefetch()
    result = provider.current_weather()
    assert result.temperature_c == 27.0
    assert len(calls) == 1
    assert provider.current_weather() is result


def test_dataset_hash_helper_matches_content(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts/01_download_hf.py"
    spec = importlib.util.spec_from_file_location("download_hf", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blob = tmp_path / "dataset.parquet"
    blob.write_bytes(b"reproducible dataset bytes")
    assert module.sha256_file(blob) == "e9286480b17967eb72840d2b5af24d0ccf0330b895b4d64b99beed7a3e298ec9"


def test_capture_timeout_does_not_consume_command_duration(monkeypatch):
    import numpy as np
    from hey_swift.integration.e2e_loop import _capture_command_vad

    class FakeStream:
        def __init__(self, samplerate, blocksize, **_kwargs):
            self.rate = samplerate
            self.frame = blocksize
            self.read_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _frame):
            time.sleep(self.frame / self.rate)
            post_calibration = self.read_count - 8
            amplitude = 0.1 if 15 <= post_calibration < 21 else 0.001
            self.read_count += 1
            return (np.full(self.frame, amplitude, dtype=np.float32).tobytes(), False)

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawInputStream=FakeStream))
    args = SimpleNamespace(
        vad_frame_ms=20, command_seconds=0.3, min_command_seconds=0.1,
        speech_start_ms=120, pre_roll_ms=200, end_silence_ms=160,
        speech_start_timeout=0.6, vad_noise_seconds=0.16,
        vad_margin_db=10, vad_min_speech_db=-45,
    )
    wav, meta = _capture_command_vad(None, args, rate=16000)
    assert meta["speech_started"] is True
    assert meta["speech_start_offset"] >= 0.25
    assert meta["capture_seconds"] >= 0.5
    assert wav.size > 0
