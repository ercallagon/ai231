"""Pause assistant capture until an owned call ends; never dial or hang up."""
import time


class CallGuardUnavailable(RuntimeError):
    pass


def wait_for_call_end(service, result, *, emit, maximum=300., poll=1., tail=1.5,
                      on_idle=lambda: None):
    gateway = result.get("data", {}).get("gateway")
    if not gateway:
        raise CallGuardUnavailable("Call outcome is unknown and no gateway is available. Check the phone before restarting listening.")
    began = time.monotonic()
    empty = 0
    failures = 0
    emit({"event": "call_audio_guard", "status": "paused", "message": "Wake/capture paused until Bluetooth confirms the call has ended."})
    while time.monotonic() - began < maximum:
        on_idle()
        try:
            active = service.backend.has_active_calls(gateway)
        except Exception as exc:
            empty = 0
            failures += 1
            if failures == 1 or failures % 10 == 0:
                emit({"event": "call_audio_guard", "status": "state_retry",
                      "error_type": type(exc).__name__, "attempt": failures,
                      "message": f"Call state unavailable ({type(exc).__name__}, attempt {failures}); capture remains paused while Bluetooth is rechecked."})
            time.sleep(min(poll * min(failures, 5), max(0., maximum - (time.monotonic() - began))))
            continue
        if failures:
            emit({"event": "call_audio_guard", "status": "state_recovered",
                  "message": "Bluetooth call-state queries recovered."})
            failures = 0
        empty = empty + 1 if not active else 0
        # Allow Dial to become visible; require two independent idle reads.
        if empty >= 2 and time.monotonic() - began >= 3.:
            time.sleep(tail)
            emit({"event": "call_audio_guard", "status": "resuming", "message": "Call ended; settling hangup audio before fresh microphone capture."})
            return
        time.sleep(poll)
    raise CallGuardUnavailable("Call guard timed out; runtime stopped without changing or hanging up the call.")
