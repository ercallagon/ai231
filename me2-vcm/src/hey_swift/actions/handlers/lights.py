"""Light handlers: on / off / set brightness / set color."""


def on(slots, binding):
    return {"message": "Lights on."}


def off(slots, binding):
    return {"message": "Lights off."}


def set_brightness(slots, binding):
    val = slots.get("BRIGHTNESS")
    lo, hi = binding.get("range", [0, 100])
    if val is not None and not (lo <= int(val) <= hi):
        return {"message": f"Brightness must be between {lo} and {hi}."}
    return {"message": f"Brightness set to {val} percent." if val else "Brightness set."}


def set_color(slots, binding):
    color = slots.get("COLOR")
    return {"message": f"Color set to {color}." if color else "Color set."}
