"""Regression coverage for the model-value -> action-slot translation.

Drives the REAL runtime path -- ``e2e_loop._make_prediction`` followed by
``ActionRouter.dispatch`` -- rather than re-implementing the mapping, so a
regression in either the translation table or the router is caught.

Scope:
  * all 18 supported slot values, each in BOTH the canonical string form
    (run_043 and later) and the legacy integer form (run_021-era) for the
    numeric intents;
  * invalid/unsupported slot values (including near-miss numeric forms that
    must NOT be coerced);
  * rejected and OUT_OF_SCOPE predictions.

No external effects: services are stubs that record calls; nothing is executed.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hey_swift.integration import e2e_loop as loop
from hey_swift.integration.router import ActionRouter, INTENTS, Prediction


# (intent, model_value, expected_canonical_service_value)
# Numeric intents appear in BOTH the canonical string and legacy integer form.
SUPPORTED = [
    # TIMER
    ("TIMER", "10s", "10 seconds"),
    ("TIMER", "30s", "30 seconds"),
    ("TIMER", "1m", "1 minute"),
    # ALARM
    ("ALARM", "6AM", "6:00 AM"),
    ("ALARM", "8AM", "8:00 AM"),
    ("ALARM", "9PM", "9:00 PM"),
    # TEMPERATURE -- string (run_043) + legacy int (run_021)
    ("TEMPERATURE", "18", "18 degrees"),
    ("TEMPERATURE", "22", "22 degrees"),
    ("TEMPERATURE", "26", "26 degrees"),
    ("TEMPERATURE", 18, "18 degrees"),
    ("TEMPERATURE", 22, "22 degrees"),
    ("TEMPERATURE", 26, "26 degrees"),
    # BRIGHTNESS -- string (run_043) + legacy int (run_021)
    ("BRIGHTNESS", "20", "20 percent"),
    ("BRIGHTNESS", "60", "60 percent"),
    ("BRIGHTNESS", "100", "100 percent"),
    ("BRIGHTNESS", 20, "20 percent"),
    ("BRIGHTNESS", 60, "60 percent"),
    ("BRIGHTNESS", 100, "100 percent"),
    # COLOR
    ("COLOR", "red", "Red"),
    ("COLOR", "blue", "Blue"),
    ("COLOR", "green", "Green"),
    # CREATE_REMINDER
    ("CREATE_REMINDER", "drink_water", "Drink water"),
    ("CREATE_REMINDER", "study", "Study"),
    ("CREATE_REMINDER", "exercise", "Exercise"),
]


class _RecordingService:
    """Stub service that records every execute() call and reports ok."""
    def __init__(self, group):
        self.group = group
        self.calls = []
    def execute(self, intent, slot=None, **kwargs):
        from hey_swift.actions.services.information import ActionResult
        self.calls.append((self.group, intent, slot))
        return ActionResult(intent, "ok", "checked")
    def close(self):
        pass


class SlotTranslationTests(unittest.TestCase):
    def setUp(self):
        self.services = {group: _RecordingService(group) for group in set(INTENTS.values())}
        self.router = ActionRouter(self.services)
        self.router.__enter__()

    def tearDown(self):
        self.router.__exit__(None, None, None)

    def _dispatch(self, intent, slot, accepted=True):
        vcm = {"intent": intent, "accepted": accepted,
               "slots": {} if slot is None else {intent: slot}}
        prediction = loop._make_prediction(vcm)
        return prediction, self.router.dispatch(prediction)

    def test_all_supported_values_both_representations(self):
        # Exactly 18 distinct supported slot values; numeric ones appear twice.
        self.assertEqual(len(SUPPORTED), 24)
        distinct = {(intent, canon) for intent, _, canon in SUPPORTED}
        self.assertEqual(len(distinct), 18)
        for intent, model_value, canonical in SUPPORTED:
            prediction, out = self._dispatch(intent, model_value)
            # Translation must land on the canonical service value (a str).
            self.assertIsInstance(prediction.slot_value, str)
            self.assertEqual(prediction.slot_value, canonical)
            self.assertEqual(out["status"], "ok",
                             msg=f"{intent}/{model_value!r} did not route: {out}")
            # The real service received the canonical value.
            group = INTENTS[intent]
            self.assertEqual(self.services[group].calls[-1], (group, intent, canonical))

    def test_numeric_strings_are_distinct_from_ints_but_both_valid(self):
        # String "18" and int 18 both resolve, to the SAME canonical value.
        p_str, _ = self._dispatch("TEMPERATURE", "18")
        p_int, _ = self._dispatch("TEMPERATURE", 18)
        self.assertEqual(p_str.slot_value, "18 degrees")
        self.assertEqual(p_int.slot_value, "18 degrees")
        p_b_str, _ = self._dispatch("BRIGHTNESS", "100")
        p_b_int, _ = self._dispatch("BRIGHTNESS", 100)
        self.assertEqual(p_b_str.slot_value, "100 percent")
        self.assertEqual(p_b_int.slot_value, "100 percent")

    def test_invalid_slots_are_rejected_without_action(self):
        invalid = [
            ("TEMPERATURE", "30"),      # unsupported string
            ("TEMPERATURE", 30),        # unsupported int
            ("TEMPERATURE", "18 "),     # near-miss: trailing space, no coercion
            ("TEMPERATURE", "18.0"),    # near-miss: float-as-string, no coercion
            ("TEMPERATURE", 18.0),      # near-miss: float, no coercion
            ("TEMPERATURE", True),      # bool is not a valid slot
            ("TEMPERATURE", None),      # missing
            ("BRIGHTNESS", "150"),
            ("BRIGHTNESS", 150),
            ("BRIGHTNESS", "60%"),
            ("COLOR", "purple"),
            ("COLOR", None),
            ("TIMER", "5s"),
            ("ALARM", "10PM"),
            ("CREATE_REMINDER", "meditate"),
            ("CREATE_REMINDER", ""),
        ]
        before = sum(len(s.calls) for s in self.services.values())
        for intent, slot in invalid:
            prediction, out = self._dispatch(intent, slot)
            self.assertNotEqual(out["status"], "ok",
                                msg=f"invalid {intent}/{slot!r} was accepted: {out}")
            self.assertIn(out["status"], {"invalid_slot", "rejected", "unsupported"})
        after = sum(len(s.calls) for s in self.services.values())
        self.assertEqual(after, before, "an invalid slot fired a service action")

    def test_rejected_predictions_do_not_act(self):
        # Accepted=False must fail closed even for a perfectly valid slot.
        for intent, slot in (("TIMER", "1m"), ("TEMPERATURE", "18"),
                             ("BRIGHTNESS", 20), ("COLOR", "blue")):
            prediction, out = self._dispatch(intent, slot, accepted=False)
            self.assertFalse(prediction.accepted)
            self.assertNotEqual(out["status"], "ok")
            self.assertEqual(out["status"], "rejected")

    def test_out_of_scope_does_not_act(self):
        prediction, out = self._dispatch("OUT_OF_SCOPE", None, accepted=True)
        self.assertNotEqual(out["status"], "ok")
        self.assertEqual(out["status"], "rejected")

    def test_unsupported_intent_does_not_act(self):
        prediction, out = self._dispatch("FLY_TO_MOON", None, accepted=True)
        self.assertNotEqual(out["status"], "ok")
        self.assertEqual(out["status"], "unsupported")

    def test_router_rejects_nonstring_slot_directly(self):
        # Defense in depth: the router itself refuses a non-str slot_value.
        out = self.router.dispatch(Prediction("TEMPERATURE", 18, True))
        # 18 is a valid legacy int, but Prediction requires str; from_dict guards
        # that at construction. Constructed directly, dispatch must still not
        # treat an int slot as a valid canonical string.
        self.assertNotEqual(out["status"], "ok")


if __name__ == "__main__":
    unittest.main()
