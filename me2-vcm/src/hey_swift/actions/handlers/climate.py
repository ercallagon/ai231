"""Climate handlers: set temperature."""


def set_temperature(slots, binding):
    val = slots.get("TEMPERATURE")
    unit = binding.get("unit", "celsius")
    lo, hi = binding.get("range", [16, 30])
    if val is not None and not (lo <= int(val) <= hi):
        return {"message": f"Temperature must be between {lo} and {hi} {unit}."}
    return {"message": f"Temperature set to {val} degrees." if val else "Temperature set."}
