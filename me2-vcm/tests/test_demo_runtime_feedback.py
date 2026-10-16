"""Pi feedback regressions using real routing and mocked hardware/services."""
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hey_swift import deployment
from hey_swift.actions.services.information import ActionResult
from hey_swift.actions.services.music import AutoMusicBackend, MpvBackend, SoloistBackend, MusicService, MusicUnavailable
from hey_swift.integration.call_guard import wait_for_call_end, CallGuardUnavailable
from hey_swift.integration.router import ActionRouter, Prediction
from hey_swift.integration.terminal import TerminalMusicControls


@pytest.mark.parametrize("confirmation", ["0", "1"])
def test_live_call_guard_uses_backend_through_optional_confirmation(monkeypatch, tmp_path, confirmation):
    from hey_swift.integration import runtime
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setenv("HEY_SWIFT_ENABLED_INTEGRATIONS", "calling")
    monkeypatch.setenv("HEY_SWIFT_CONFIRM_COMMUNICATIONS", confirmation)
    monkeypatch.setattr(runtime, "communication_values", lambda: {"call_number": "+12025550123"})
    backend = Mock()
    backend.dial.return_value = dict(intent="CALL", status="dialing", message="Call requested.", data={"gateway": "/gateway"})
    backend.has_active_calls.side_effect = [True, False, False, False]
    monkeypatch.setattr(runtime, "BluetoothCallingBackend", lambda **kwargs: backend)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "CALL" if confirmation == "1" else pytest.fail("Unexpected confirmation"))
    now = [0.]
    monkeypatch.setattr("hey_swift.integration.call_guard.time.monotonic", lambda: now[0])
    monkeypatch.setattr("hey_swift.integration.call_guard.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    with runtime.build_router(live=True) as router:
        result = router.dispatch(Prediction("CALL", accepted=True))
        assert result["status"] == "dialing"
        events = []
        wait_for_call_end(router.services["calling"], result, emit=events.append)
        assert events[-1]["status"] == "resuming"
        backend.dial.assert_called_once()


@pytest.mark.parametrize("intent, group", [("CALL", "calling"), ("MESSAGE", "messaging")])
def test_enabled_live_communications_execute_without_prompt(monkeypatch, tmp_path, intent, group):
    from hey_swift.integration import runtime
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setenv("HEY_SWIFT_ENABLED_INTEGRATIONS", group)
    monkeypatch.setenv("HEY_SWIFT_CONFIRM_COMMUNICATIONS", "0")
    monkeypatch.setattr(runtime, "communication_values", lambda: dict(call_number="+12025550123", phone_mac="00:11:22:33:44:55", message_number="+12025550123"))
    service = Mock()
    service.execute.return_value = ActionResult(intent, "ok", "Mock communication completed.")
    service.close = Mock()
    monkeypatch.setattr(runtime, "CallingService" if group == "calling" else "ConfiguredMessaging", lambda *a: service)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Unexpected confirmation"))
    with runtime.build_router(live=True) as router:
        assert router.dispatch(Prediction(intent, accepted=True))["status"] == "ok"
        service.execute.assert_called_once()


def test_no_speech_is_quiet_but_spoken_rejection_remains_available():
    speech = Mock()
    router = ActionRouter({}, speech=speech)
    result = router.dispatch(Prediction("OUT_OF_SCOPE"), rejection_reason="no_sustained_speech")
    assert result["status"] == "rejected"
    speech.speak.assert_not_called()
    router.dispatch(Prediction("OUT_OF_SCOPE"))
    speech.speak.assert_called_once()
    speech.reset_mock()
    ActionRouter({}, speech=speech, speak_no_speech=True).dispatch(Prediction("OUT_OF_SCOPE"), rejection_reason="no_sustained_speech")
    speech.speak.assert_called_once_with("I didn't hear a command.")


def test_call_state_transient_error_recovers_without_resuming_early(monkeypatch):
    now = [0.]
    states = iter([OSError("private diagnostic must not leak"), True, False, False])
    def active(gateway):
        state = next(states)
        if isinstance(state, Exception):
            raise state
        return state
    service = SimpleNamespace(backend=SimpleNamespace(has_active_calls=active))
    monkeypatch.setattr("hey_swift.integration.call_guard.time.monotonic", lambda: now[0])
    monkeypatch.setattr("hey_swift.integration.call_guard.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    events = []
    wait_for_call_end(service, {"data": {"gateway": "/gateway"}}, emit=events.append)
    assert [event["status"] for event in events] == ["paused", "state_retry", "state_recovered", "resuming"]
    assert "private diagnostic" not in str(events)


def test_persistent_call_state_error_keeps_capture_blocked(monkeypatch):
    now = [0.]
    backend = Mock()
    backend.has_active_calls.side_effect = OSError("private")
    monkeypatch.setattr("hey_swift.integration.call_guard.time.monotonic", lambda: now[0])
    monkeypatch.setattr("hey_swift.integration.call_guard.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    events = []
    with pytest.raises(CallGuardUnavailable):
        wait_for_call_end(SimpleNamespace(backend=backend), {"data": {"gateway": "/gateway"}}, emit=events.append, maximum=5.)
    assert not any(event["status"] == "resuming" for event in events)
    assert backend.has_active_calls.call_count == 3


@pytest.mark.parametrize("active", ["offline", "spotify"])
def test_terminal_volume_controls_selected_player_only(active):
    online = Mock(max_volume_percent=25)
    offline = Mock(max_volume_percent=25)
    backend = AutoMusicBackend(online, offline, state_path=None)
    backend.active = active
    events = []
    controls = TerminalMusicControls(SimpleNamespace(services={"music": MusicService(backend)}), emit=events.append)
    controls.handle(":volume 10")
    assert events[-1]["volume_percent"] == 10
    if active == "offline":
        offline.command.assert_called_once_with("set_property", "volume", 10, timeout=1)
        online.ctl.assert_not_called()
    else:
        online.ctl.assert_called_once_with("volume", "10", timeout=1)
        offline.command.assert_not_called()
    controls.handle(":volume 26")
    assert events[-1]["status"] == "unavailable"


def test_terminal_controls_never_route_call_or_message():
    service = Mock()
    service.execute.return_value = ActionResult("STOP", "ok", "Stopped.")
    router = SimpleNamespace(services={"music": service}, dispatch=Mock())
    events = []
    controls = TerminalMusicControls(router, emit=events.append)
    controls.handle(":quieter")
    controls.handle(":pause")
    controls.handle(":stop")
    assert [call.args[0] for call in service.execute.call_args_list] == ["VOLUME_DOWN", "PAUSE", "STOP"]
    controls.handle("Make a call")
    assert events[-1]["status"] == "unavailable"
    router.dispatch.assert_not_called()


def test_noninteractive_terminal_does_not_consume_input():
    stream = StringIO(":stop\n")
    controls = TerminalMusicControls(SimpleNamespace(services={}), emit=Mock(), stream=stream)
    assert not controls.poll() and stream.tell() == 0


def test_interactive_terminal_reads_one_control(monkeypatch):
    class Terminal(StringIO):
        def isatty(self):
            return True
    stream = Terminal(":pause\n")
    service = Mock()
    service.execute.return_value = ActionResult("PAUSE", "ok", "Paused.")
    monkeypatch.setattr("hey_swift.integration.terminal.select.select", lambda *args: ([stream], [], []))
    controls = TerminalMusicControls(SimpleNamespace(services={"music": service}), emit=Mock(), stream=stream)
    assert controls.poll()
    service.execute.assert_called_once_with("PAUSE")


def test_terminal_control_reopens_wake_stream_and_discards_stale_audio():
    import numpy as np
    from hey_swift.integration.e2e_loop import _listen_for_wake
    streams = []
    class Stream:
        def __init__(self, **kwargs):
            self.closed = False
            streams.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.closed = True
        def read(self, hop):
            return np.zeros(hop, dtype=np.float32).tobytes(), False
    sd = SimpleNamespace(RawInputStream=Stream, PortAudioError=RuntimeError)
    mic = SimpleNamespace(resolve=lambda: 0, healthy=lambda: None)
    detector = Mock()
    detector.process.return_value = ([{"prob": .9}], .9)
    resampler = Mock()
    resampler.push.side_effect = lambda samples: samples
    idle = Mock(side_effect=[True, False])
    args = SimpleNamespace(wake_log_interval=2., verbose=False)
    _listen_for_wake(sd, mic, 48000, 4800, detector, resampler,
                     SimpleNamespace(suppress_input=lambda: False), args, on_idle=idle)
    assert len(streams) == 2 and all(stream.closed for stream in streams)
    assert detector.process.call_count == 1
    detector.reset.assert_called_once()


def test_offline_replay_preserves_lowered_music_volume():
    backend = MpvBackend([], "unused", max_volume_percent=25)
    backend.command = Mock(side_effect=[10, None, False, None])
    backend.execute("PLAY_MUSIC")
    assert backend.command.call_args_list[1].args == ("set_property", "volume", 10)


def test_spotify_prepare_never_activates_or_plays():
    backend = SoloistBackend(operation_timeout=25)
    backend.ensure_started = Mock()
    backend.ctl = Mock(return_value={"status": "paused", "is_active": False})
    assert backend.prepare() == {"backend": "spotify", "control_ready": True, "is_active": False}
    assert [call.args for call in backend.ctl.call_args_list] == [("now",)]
    assert backend._operation_deadline is None


@pytest.mark.parametrize("fails,expected", [(False, 0), (True, 1)])
def test_public_spotify_activates_saved_login_without_phone_selection_prompt(monkeypatch, tmp_path, fails, expected):
    backend = Mock()
    backend._wait_state.return_value = {"is_active": True}
    if fails:
        backend.prepare.side_effect = MusicUnavailable("Unavailable")
    monkeypatch.setattr("hey_swift.actions.services.music.SoloistBackend", lambda **kwargs: backend)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Unexpected phone-selection prompt"))
    cfg = dict(spotify=True, soloist_binary="unused", enabled=["music"], secrets={})
    assert deployment.spotify(cfg, tmp_path) == expected
    if not fails:
        backend.ctl.assert_called_once_with("activate", timeout=3)
    assert backend._operation_deadline is None


def test_public_launch_passes_opt_in_flags_and_isolates_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(deployment, "microphone", lambda cfg: "0")
    run = Mock(return_value=0)
    monkeypatch.setattr(deployment.subprocess, "call", run)
    cfg = dict(enabled=["calling", "messaging"], hardware="pi", secrets={})
    deployment.assistant(cfg, tmp_path, live=True)
    assert run.call_args.kwargs["env"]["HEY_SWIFT_CONFIRM_COMMUNICATIONS"] == "0"
    assert run.call_args.kwargs["env"]["VCM_SPOTIFY_ENABLED"] == "0"
    deployment.assistant(cfg, tmp_path, live=True, confirm_communications=True, speak_no_speech=True)
    assert run.call_args.kwargs["env"]["HEY_SWIFT_CONFIRM_COMMUNICATIONS"] == "1"
    assert "--speak-no-speech" in run.call_args.args[0]
