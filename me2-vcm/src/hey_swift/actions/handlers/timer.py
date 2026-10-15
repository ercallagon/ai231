"""Timer handlers: start a countdown."""

_SECONDS = {"10s": 10, "30s": 30, "1m": 60}


def start(slots, binding):
    val = slots.get("TIMER")
    secs = _SECONDS.get(val)
    return {"message": f"Timer set for {val}." if val else "Timer started."}
