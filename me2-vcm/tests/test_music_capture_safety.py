"""A failed playback controller must never become a captured user command."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from hey_swift.audio import capture, speech
from hey_swift.audio.focus import AudioFocus
from hey_swift.integration import e2e_loop as loop, runtime
from hey_swift.integration.router import ActionRouter
from hey_swift.vcm import inference
from hey_swift.wakeword import inference as wake_inference


def broken_router(active="spotify"):
    player = SimpleNamespace(ctl=Mock(side_effect=OSError("controller unavailable")),
                             command=Mock(side_effect=OSError("player unavailable")))
    backend = SimpleNamespace(active=active, online=player, offline=player)
    return SimpleNamespace(services={"music": SimpleNamespace(backend=backend)})


@pytest.mark.parametrize("active", ["spotify", "offline"])
def test_known_music_failure_never_enters_capture(active):
    with patch.object(loop, "_emit"), pytest.raises(loop.MusicCaptureUnavailable):
        with loop._duck_music(broken_router(active)):
            pytest.fail("Capture must not proceed")


def test_no_active_music_remains_usable():
    router = SimpleNamespace(services={})
    with loop._duck_music(router) as restore:
        restore()


def test_failed_quieting_returns_to_listening_without_inference_or_action(monkeypatch):
    stub = broken_router()
    devices = SimpleNamespace(execute=Mock())
    router = ActionRouter({"music": stub.services["music"], "devices": devices})
    info = {"name": "USB microphone", "hostapi": 0, "max_input_channels": 1}
    api = SimpleNamespace(query_devices=lambda *args: info if args else [info],
                          query_hostapis=lambda *_: {"name": "ALSA"}, PortAudioError=OSError)
    monkeypatch.setitem(sys.modules, "sounddevice", api)
    detector = SimpleNamespace(gate=SimpleNamespace(threshold=.3), reset=Mock())
    model = SimpleNamespace(predict=Mock())
    monkeypatch.setattr(wake_inference, "WakeWordDetector", lambda *a, **kw: detector)
    monkeypatch.setattr(inference, "VCMInference", lambda *a, **kw: model)
    monkeypatch.setattr(runtime, "prepare_workspace", lambda: None)
    monkeypatch.setattr(runtime, "build_router", lambda **kw: router)
    monkeypatch.setattr(capture, "capture_supported_rate", lambda *_: (16000, 1600))
    listener = Mock(side_effect=[.01, KeyboardInterrupt])
    monkeypatch.setattr(loop, "_listen_for_wake", listener)
    command_capture = Mock()
    monkeypatch.setattr(loop, "_capture_command_vad", command_capture)
    events = []
    monkeypatch.setattr(loop, "_emit", events.append)
    args = SimpleNamespace(log_format="json", device="0", wake_config="w", vcm_config="v",
        hardware_config="h", wake_onnx="w.onnx", vcm_onnx="v.onnx", vcm_labels="l.json",
        ack=None, threshold=None, no_ack=True, live=False, hardware="mock", speak=False,
        music_duck_percent=0, once=False)
    with pytest.raises(KeyboardInterrupt):
        loop.run(args)
    assert listener.call_count == 2
    command_capture.assert_not_called()
    model.predict.assert_not_called()
    devices.execute.assert_not_called()
    assert any(event.get("event") == "command_aborted" for event in events)


def test_failed_speech_quieting_releases_focus_without_tts():
    router = broken_router()
    focus = AudioFocus(tail_seconds=0)
    backend = SimpleNamespace(speak=Mock())
    context = [None]
    def before():
        focus.before_playback()
        context[0] = loop._duck_music(router)
        context[0].__enter__()
    def after():
        try:
            if context[0] is not None:
                context[0].__exit__(None, None, None)
        finally:
            focus.after_playback()
    service = speech.SpeechService(backend, before_playback=before, after_playback=after)
    with patch.object(loop, "_emit"), pytest.raises(loop.MusicCaptureUnavailable):
        service.speak("Timer finished.")
    backend.speak.assert_not_called()
    assert not focus.suppress_input()
