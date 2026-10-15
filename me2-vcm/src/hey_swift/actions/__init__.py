"""hey_swift.actions — converts VCM predictions into device behavior.

router    routes (intent, slots) -> the appropriate handler
handlers/ one module per action family (music, lights, climate, timer, ...)

The router is config-driven (configs/actions.yaml): each intent maps to a
dotted handler path + slot declarations. Handlers are pure functions where
possible so they are trivially testable without hardware.
"""
