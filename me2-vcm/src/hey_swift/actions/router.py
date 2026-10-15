"""Action router: (intent, slots) -> typed action.

Config-driven via configs/actions.yaml. The router:
    1. Checks the OOS / rejection gate (confidence threshold).
    2. Looks up the intent's handler binding.
    3. Validates the predicted slots against the intent's declared slot sets.
    4. Dispatches to the handler and returns a structured result.

Handlers are resolved by dotted path into hey_swift.actions.handlers. Unknown
intents or invalid slots produce a structured 'rejected'/'error' result rather
than raising, so the live loop can always give feedback.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from ..vcm.taxonomy import load_taxonomy


def _slot_matches(value, valid_set) -> bool:
    """Type-coercive membership test for a predicted slot value.

    Predictions arrive as strings (e.g. "60"); the taxonomy slot sets may be
    ints (20/60/100) or strings ("10s", "red"). Compare in the natural domain:
    exact match first, then numeric match, then case-insensitive string match.
    """
    if value is None:
        return False
    if value in valid_set:
        return True
    sv = str(value).strip()
    for v in valid_set:
        if sv == str(v).strip():
            return True
        try:
            if float(sv) == float(v):
                return True
        except (TypeError, ValueError):
            pass
        if sv.lower() == str(v).strip().lower():
            return True
    return False


@dataclass
class ActionResult:
    ok: bool
    intent: str
    action: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    feedback: Dict[str, Any] = field(default_factory=dict)


class ActionRouter:
    def __init__(self, config: str = "configs/actions.yaml",
                 vcm_config: str = "configs/vcm.yaml"):
        self.cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
        self.tax = load_taxonomy(vcm_config)
        self.actions = self.cfg.get("actions", {}) or {}
        self.rejection = self.cfg.get("rejection", {}) or {}
        self.feedback_defaults = self.cfg.get("feedback", {}) or {}
        self.tau = float(self.rejection.get("oos_confidence_threshold", 0.5))

    def route(self, prediction: Dict[str, Any]) -> ActionResult:
        intent = prediction.get("intent")
        conf = float(prediction.get("confidence", 0.0))
        slots = prediction.get("slots", {}) or {}

        # 1) rejection gate
        if prediction.get("oos") or (intent != self.tax.oos_class and conf < self.tau):
            on_oos = self.rejection.get("on_oos", {})
            return ActionResult(
                ok=False, intent=intent or self.tax.oos_class,
                action="reject",
                message=on_oos.get("speak", "Sorry, I didn't catch that."),
                feedback={"led": on_oos.get("led", "error")},
            )

        # 2) lookup binding
        binding = self.actions.get(intent)
        if binding is None:
            return ActionResult(ok=False, intent=intent, action="unknown",
                                message="I don't know how to do that yet.")

        # 3) validate slots (type-coercive: predictions carry strings, the
        #    taxonomy slot sets may be ints/strings depending on the value).
        declared = binding.get("slots", []) or []
        for s in declared:
            valid = self.tax.slot_classes(intent)
            if valid:
                got = slots.get(s)
                if not _slot_matches(got, valid):
                    return ActionResult(
                        ok=False, intent=intent, action="invalid_slot",
                        slots=slots, message=f"I didn't understand the {s.lower()}.",
                    )

        # 4) dispatch
        handler_path = binding["handler"]
        handler = self._resolve(handler_path)
        try:
            payload = handler(slots, binding)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the loop
            return ActionResult(ok=False, intent=intent, action=handler_path,
                                message=f"Action failed: {exc}",
                                feedback={"led": "error"})

        return ActionResult(
            ok=True, intent=intent, action=handler_path, slots=slots,
            message=payload.get("message", self.feedback_defaults.get("speak_default", "Done.")),
            feedback={"led": self.feedback_defaults.get("led", "success")},
        )

    def _resolve(self, dotted: str):
        mod_name, _, fn_name = dotted.rpartition(".")
        mod = importlib.import_module(f"hey_swift.actions.handlers.{mod_name}")
        return getattr(mod, fn_name)
