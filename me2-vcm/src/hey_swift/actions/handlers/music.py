"""Music transport handlers: play / pause / stop / next."""


def play(slots, binding):
    return {"message": "Playing music."}


def pause(slots, binding):
    return {"message": "Paused."}


def stop(slots, binding):
    return {"message": "Stopped."}


def next(slots, binding):
    return {"message": "Next track."}
