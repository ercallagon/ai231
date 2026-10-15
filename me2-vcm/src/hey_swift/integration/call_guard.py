"""Pause assistant capture until an owned call ends; never dial or hang up."""
import time


class CallGuardUnavailable(RuntimeError):
    pass


def wait_for_call_end(service, result, *, emit, maximum=300., poll=1., tail=1.5):
    gateway = result.get("data", {}).get("gateway")
    if not gateway:
        raise CallGuardUnavailable("Call outcome is unknown and no gateway is available. Check the phone before restarting listening.")
    began = time.monotonic()
    empty = 0
    emit({"event": "call_audio_guard", "status": "paused", "message": "Wake/capture paused until Bluetooth confirms the call has ended."})
    while time.monotonic() - began < maximum:
        try:
            active = service.backend.has_active_calls(gateway)
        except Exception:
            raise CallGuardUnavailable("Cannot verify call state. Listening stays stopped; inspect Bluetooth before restarting.")
        empty = empty + 1 if not active else 0
        # Allow Dial to become visible; require two independent idle reads.
        if empty >= 2 and time.monotonic() - began >= 3.:
            time.sleep(tail)
            emit({"event": "call_audio_guard", "status": "resuming", "message": "Call ended; settling hangup audio before fresh microphone capture."})
            return
        time.sleep(poll)
    raise CallGuardUnavailable("Call guard timed out; runtime stopped without changing or hanging up the call.")
