"""Communication handlers: call / message (stub responses)."""


def call(slots, binding):
    return {"message": "Call initiation is not wired to a dialer yet."}


def message(slots, binding):
    return {"message": "Messaging is not wired to a transport yet."}
