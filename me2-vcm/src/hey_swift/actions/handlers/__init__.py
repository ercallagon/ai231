"""Action handlers — one module per action family.

Each handler is a pure function `handler(slots: dict, binding: dict) -> dict`
returning at least {"message": str}. Side effects (talking to a real light,
starting a real timer) are injected via a context object in production; in the
scaffold they return a descriptive message so the router is fully testable
without hardware.
"""
