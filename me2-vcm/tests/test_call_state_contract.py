"""Remote-ended state handling only; no actual Bluetooth calls or hangup RPCs."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from hey_swift.actions.services.calling import BluetoothCallingBackend, CallingUnavailable
from hey_swift.integration.call_guard import wait_for_call_end, CallGuardUnavailable


def backend_for(properties):
    runner = Mock(return_value=SimpleNamespace(stdout=json.dumps({"data": [properties]})))
    return BluetoothCallingBackend(runner=runner), runner


@pytest.mark.parametrize("state", ["active", "held", "dialing", "alerting", "incoming", "waiting"])
def test_live_state_never_treated_as_ended(state):
    backend, runner = backend_for({"/gateway/call0": {"State": {"type": "s", "data": state}}})
    assert backend.has_active_calls("/gateway")
    assert runner.call_args.args[0][7] == "GetCalls"


def test_disconnected_object_is_not_an_active_call():
    backend, _ = backend_for({"/gateway/call0": {"State": {"type": "s", "data": "disconnected"}}})
    assert not backend.has_active_calls("/gateway")


@pytest.mark.parametrize("state", [None, "ended", True, {"type": "i", "data": 0}])
def test_unknown_state_blocks_listening(state):
    backend, _ = backend_for({"/gateway/call0": {"State": state}})
    assert backend.has_active_calls("/gateway")


def test_mixed_live_and_disconnected_stays_blocked():
    backend, _ = backend_for({"/gateway/old": {"State": "disconnected"}, "/gateway/new": {"State": "active"}})
    assert backend.has_active_calls("/gateway")


def test_diagnostics_redact_recipient_and_name():
    backend, _ = backend_for({"/gateway/call0": {"State": "active", "LineIdentification": "+12025550123", "Name": "Private"}})
    assert backend.get_call_states("/gateway") == {"/gateway/call0": "active"}


def test_malformed_response_never_means_idle():
    backend, _ = backend_for({"/gateway/call0": False})
    with pytest.raises(CallingUnavailable):
        backend.has_active_calls("/gateway")


def test_grace_after_disconnected_never_sends_hangup():
    backend, runner = backend_for({"/gateway/call0": {"State": "disconnected"}})
    now = [0.]
    events = []
    with patch("hey_swift.integration.call_guard.time.monotonic", side_effect=lambda: now[0]), \
         patch("hey_swift.integration.call_guard.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
        wait_for_call_end(SimpleNamespace(backend=backend), {"data": {"gateway": "/gateway"}}, emit=events.append, tail=3.)
    assert now[0] == 6.
    assert events[-1]["status"] == "resuming"
    assert all(call.args[0][7] == "GetCalls" for call in runner.call_args_list)


def test_repeating_beeps_are_not_an_automatic_hangup_signal():
    backend, runner = backend_for({"/gateway/call0": {"State": "active"}})
    now = [0.]
    with patch("hey_swift.integration.call_guard.time.monotonic", side_effect=lambda: now[0]), \
         patch("hey_swift.integration.call_guard.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
        with pytest.raises(CallGuardUnavailable):
            wait_for_call_end(SimpleNamespace(backend=backend), {"data": {"gateway": "/gateway"}}, emit=lambda _: None, maximum=5.)
    assert all(call.args[0][7] == "GetCalls" for call in runner.call_args_list)
