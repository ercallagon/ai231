"""Volume handlers: up / down by a configurable step."""


def up(slots, binding):
    step = binding.get("step", 10)
    return {"message": f"Volume up {step}."}


def down(slots, binding):
    step = binding.get("step", 10)
    return {"message": f"Volume down {step}."}
