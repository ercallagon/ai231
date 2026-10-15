"""Information handlers: weather / time (stub responses)."""

from datetime import datetime


def weather(slots, binding):
    return {"message": "Weather lookup is not wired to a provider yet."}


def time(slots, binding):
    now = datetime.now().strftime("%H:%M")
    return {"message": f"It's {now}."}
