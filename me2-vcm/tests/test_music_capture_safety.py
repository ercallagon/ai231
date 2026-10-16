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


def test_new_run_resets_only_persisted_music_route_to_configured_baseline(tmp_path):
    from hey_swift.actions.services.music import AutoMusicBackend

    state = tmp_path / "music-routing.json"
    state.write_text('{"active":"offline"}')
    online = SimpleNamespace(reset_session_volume=Mock(return_value=True))
    offline = SimpleNamespace(reset_session_volume=Mock(return_value=True))
    backend = AutoMusicBackend(online, offline, state_path=state)

    assert backend.reset_active_session_volume() is True
    online.reset_session_volume.assert_not_called()
    offline.reset_session_volume.assert_called_once_with()
    assert state.read_text() == '{"active":"offline"}'


def test_music_session_volume_reset_uses_player_baseline_only():
    from hey_swift.actions.services.music import MpvBackend, SoloistBackend

    offline = MpvBackend([], "/unused", max_volume_percent=25)
    offline.is_running = Mock(return_value=True)
    offline.command = Mock()
    assert offline.reset_session_volume() is True
    offline.command.assert_called_once_with("set_property", "volume", 25, timeout=1)

    online = SoloistBackend(max_volume_percent=25)
    online.is_running = Mock(return_value=True)
    online.ctl = Mock()
    assert online.reset_session_volume() is True
    online.ctl.assert_called_once_with("volume", "25", timeout=1)


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


def test_lights_after_music_speaks_while_muted_then_restores_volume(monkeypatch):
    import time
    import numpy as np
    from hey_swift.actions.services.music import MusicService
    from hey_swift.hardware.devices import MockHardware

    timeline = []
    volume = [20.]
    def command(*args, **kwargs):
        if args == ("get_property", "volume"):
            return volume[0]
        assert args[:2] == ("set_property", "volume")
        volume[0] = args[2]
        timeline.append(("volume", volume[0]))
    hardware = MockHardware()
    def speak(text):
        assert ("led", (1., 1., 1.)) in hardware.events
        timeline.append(("speech", text, volume[0]))
    tts = speech.SpeechService(SimpleNamespace(speak=speak))
    devices = runtime.ScheduledDevices(hardware, speech=tts)
    player = SimpleNamespace(active="offline", offline=SimpleNamespace(command=command))
    router = ActionRouter({"music": MusicService(player), "devices": devices}, speech=tts)
    info = {"name": "USB microphone", "hostapi": 0, "max_input_channels": 1}
    api = SimpleNamespace(query_devices=lambda *args: info if args else [info],
                          query_hostapis=lambda *_: {"name": "ALSA"}, PortAudioError=OSError)
    monkeypatch.setitem(sys.modules, "sounddevice", api)
    detector = SimpleNamespace(gate=SimpleNamespace(threshold=.3), reset=Mock())
    model = SimpleNamespace(feat_cfg=SimpleNamespace(window_seconds=6.), frontend="whole_recording",
        predict=lambda wave: dict(intent="LIGHT_ON", confidence=.99, accepted=True, slots={}))
    monkeypatch.setattr(wake_inference, "WakeWordDetector", lambda *args, **kwargs: detector)
    monkeypatch.setattr(inference, "VCMInference", lambda *args, **kwargs: model)
    monkeypatch.setattr(runtime, "prepare_workspace", lambda: None)
    monkeypatch.setattr(runtime, "build_router", lambda **kwargs: router)
    monkeypatch.setattr(capture, "capture_supported_rate", lambda *_: (16000, 1600))
    monkeypatch.setattr(loop, "_listen_for_wake", lambda *args: time.monotonic())
    def command_capture(*args, **kwargs):
        kwargs["before_ready"]()
        assert volume[0] == 0
        return np.zeros(16000, dtype=np.float32), dict(
            speech_started=True, speech_start_sample=0, speech_end_sample=16000,
            reason="end_silence", overflow=False, _speech_end_monotonic=time.monotonic())
    monkeypatch.setattr(loop, "_capture_command_vad", command_capture)
    events = []
    monkeypatch.setattr(loop, "_emit", events.append)
    args = SimpleNamespace(log_format="json", device="0", wake_config="w", vcm_config="v",
        hardware_config="h", wake_onnx="w.onnx", vcm_onnx="v.onnx", vcm_labels="l.json",
        ack=None, threshold=None, no_ack=True, live=True, offline=True, hardware="mock", speak=True,
        music_duck_percent=0, playback_tail_seconds=0, benchmark_log=False, once=True)
    assert loop.run(args) == 0
    assert [(item[1], item[2]) for item in timeline if item[0] == "speech"] == [("Lights on.", 0.)]
    assert timeline[-1] == ("volume", 20.)
    result = next(event for event in events if event.get("event") == "command")
    assert result["action"]["status"] == "ok"
    assert result["action"]["speech"]["status"] == "ok"
