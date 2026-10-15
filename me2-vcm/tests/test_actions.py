"""Tests for the action router and handlers (import-safe, no hardware).

The router depends only on the taxonomy + pure handler functions, so the whole
intent -> action mapping is testable headless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def router(tmp_path):
    from hey_swift.actions.router import ActionRouter
    actions = tmp_path / "actions.yaml"
    actions.write_text(
        "rejection:\n  oos_confidence_threshold: 0.5\n  on_oos:\n    led: error\n"
        "    speak: Sorry.\n"
        "actions:\n"
        "  PLAY_MUSIC:\n    handler: music.play\n"
        "  PAUSE:\n    handler: music.pause\n"
        "  VOLUME_UP:\n    handler: volume.up\n    step: 10\n"
        "  LIGHT_ON:\n    handler: lights.on\n"
        "  BRIGHTNESS:\n    handler: lights.set_brightness\n"
        "    slots: [BRIGHTNESS]\n    range: [0, 100]\n"
        "  COLOR:\n    handler: lights.set_color\n    slots: [COLOR]\n"
        "  TEMPERATURE:\n    handler: climate.set_temperature\n"
        "    slots: [TEMPERATURE]\n    unit: celsius\n    range: [16, 30]\n"
        "  TIMER:\n    handler: timer.start\n    slots: [TIMER]\n"
        "  ALARM:\n    handler: alarm.set\n    slots: [ALARM]\n"
        "  CREATE_REMINDER:\n    handler: reminder.create\n    slots: [CREATE_REMINDER]\n"
        "  TIME:\n    handler: info.time\n"
        "feedback:\n  led: success\n  speak_default: Done.\n"
    )
    vcm_cfg = tmp_path / "vcm.yaml"
    vcm_cfg.write_text(
        "schema:\n  n_intents: 19\n  oos_class: OUT_OF_SCOPE\n  n_way_head: 20\n"
        "  slot_reject: false\n"
        "  slotted_intents: [BRIGHTNESS, COLOR, TEMPERATURE, TIMER, ALARM, CREATE_REMINDER]\n"
        "  slot_sets:\n"
        "    BRIGHTNESS: [20, 60, 100, REJECT]\n"
        "    COLOR: [red, blue, green, REJECT]\n"
        "    TEMPERATURE: [18, 22, 26, REJECT]\n"
        "    TIMER: [10s, 30s, 1m, REJECT]\n"
        "    ALARM: [6AM, 8AM, 9PM, REJECT]\n"
        "    CREATE_REMINDER: [drink_water, study, exercise, REJECT]\n"
        "oos:\n  confidence_threshold: 0.5\n"
    )
    return ActionRouter(str(actions), str(vcm_cfg))


class TestRouting:
    def test_all_simple_intents_route(self, router):
        for intent in ("PLAY_MUSIC", "PAUSE", "VOLUME_UP", "LIGHT_ON", "TIME"):
            res = router.route({"intent": intent, "confidence": 0.9, "slots": {}})
            assert res.ok is True, f"{intent} -> {res}"

    def test_all_slotted_intents_route_with_valid_slot(self, router):
        cases = {
            "BRIGHTNESS": {"BRIGHTNESS": "60"},
            "COLOR": {"COLOR": "red"},
            "TEMPERATURE": {"TEMPERATURE": "22"},
            "TIMER": {"TIMER": "1m"},
            "ALARM": {"ALARM": "8AM"},
            "CREATE_REMINDER": {"CREATE_REMINDER": "study"},
        }
        for intent, slots in cases.items():
            res = router.route({"intent": intent, "confidence": 0.9, "slots": slots})
            assert res.ok is True, f"{intent} {slots} -> {res}"

    def test_unknown_intent_rejected(self, router):
        res = router.route({"intent": "TELEPORT", "confidence": 0.9, "slots": {}})
        assert res.ok is False
        assert res.action == "unknown"

    def test_oos_rejected(self, router):
        res = router.route({"intent": "OUT_OF_SCOPE", "confidence": 0.9,
                            "oos": True, "slots": {}})
        assert res.ok is False
        assert res.action == "reject"
        assert res.feedback.get("led") == "error"


class TestHandlers:
    def test_timer_message(self):
        from hey_swift.actions.handlers import timer
        assert "1m" in timer.start({"TIMER": "1m"}, {})["message"]

    def test_brightness_range_guard(self):
        from hey_swift.actions.handlers import lights
        # 150 out of [0,100] range
        msg = lights.set_brightness({"BRIGHTNESS": "150"}, {"range": [0, 100]})["message"]
        assert "between" in msg

    def test_temperature_unit(self):
        from hey_swift.actions.handlers import climate
        msg = climate.set_temperature({"TEMPERATURE": "22"},
                                      {"unit": "celsius", "range": [16, 30]})["message"]
        assert "22" in msg and "degrees" in msg

    def test_reminder_create_and_list(self):
        from hey_swift.actions.handlers import reminder
        reminder._STORE.clear()
        reminder.create({"CREATE_REMINDER": "study"}, {})
        msg = reminder.list({}, {})["message"]
        assert "study" in msg
