"""Reminder handlers: create / list reminders."""

# In production these persist to a small store; the scaffold keeps an in-memory
# list so the router + handlers are testable without I/O.
_STORE: list = []


def create(slots, binding):
    val = slots.get("CREATE_REMINDER")
    _STORE.append(val)
    return {"message": f"Reminder added: {val}." if val else "Reminder added."}


def list(slots, binding):
    items = ", ".join(_STORE) if _STORE else "none"
    return {"message": f"Your reminders: {items}."}
