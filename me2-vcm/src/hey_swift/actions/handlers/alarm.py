"""Alarm handlers: set an alarm at a fixed time."""


def set(slots, binding):
    val = slots.get("ALARM")
    return {"message": f"Alarm set for {val}." if val else "Alarm set."}
