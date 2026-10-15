"""Validated prediction routing independent of any particular model backend."""
from dataclasses import dataclass
import threading
import time

from hey_swift.actions.services.information import ActionResult
from hey_swift.actions.services.device_actions import VALUES as DEVICE_VALUES, FIXED as DEVICE_FIXED
from hey_swift.actions.services.reminders import VALUES as REMINDER_VALUES
from hey_swift.actions.services.music import RESPONSES

SLOTS = {**{k: tuple(v) for k, v in DEVICE_VALUES.items()},
         "CREATE_REMINDER": tuple(REMINDER_VALUES)}
GROUPS = {
    "information": ("TIME", "WEATHER"),
    "devices": tuple(sorted(DEVICE_FIXED)) + tuple(DEVICE_VALUES),
    "music": tuple(RESPONSES),
    "reminders": ("CREATE_REMINDER", "LIST_REMINDERS"),
    "calling": ("CALL",), "messaging": ("MESSAGE",),
}
INTENTS = {intent: group for group, intents in GROUPS.items() for intent in intents}


@dataclass(frozen=True)
class Prediction:
    """Upstream must decide acceptance using the selected model's calibrated gate.

    Labels and slots are canonical strings, never model output indices. No
    confidence threshold is invented here. Missing acceptance fails closed.
    """
    intent: str
    slot_value: str | None = None
    accepted: bool = False

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) - {"intent", "slot_value", "accepted"}:
            raise ValueError("Expected intent, optional slot_value, and explicit accepted boolean.")
        prediction = cls(**payload)
        if (not isinstance(prediction.intent, str)
                or type(prediction.accepted) is not bool
                or (prediction.slot_value is not None and not isinstance(prediction.slot_value, str))):
            raise ValueError("Invalid prediction field types.")
        return prediction


class ActionRouter:
    """One long-lived router serializes actions and shares one speech service."""
    def __init__(self, services, *, speech=None):
        self.services, self.speech = dict(services), speech
        self._lock = threading.RLock()
        self._closed = False
        self.on_action = None

    def dispatch(self, prediction, *, rejection_reason=None):
        with self._lock:
            if not isinstance(prediction, Prediction):
                raise TypeError("Dispatch requires a Prediction.")
            dispatch_started = time.monotonic()
            intent, slot = prediction.intent, prediction.slot_value
            speech_status = None
            if self._closed:
                result = ActionResult(intent, "unavailable", "The action router is closed.")
            elif type(prediction.accepted) is not bool or not prediction.accepted or intent == "OUT_OF_SCOPE":
                message = {"command_too_long": "Please say a shorter command.",
                           "command_incomplete": "Please say a shorter command.",
                           "capture_overflow": "Please repeat your command.",
                           "no_sustained_speech": "I didn't hear a command."}.get(
                               rejection_reason, "Sorry, I didn't understand that command.")
                result = ActionResult(intent, "rejected", message)
            elif not isinstance(intent, str) or intent not in INTENTS:
                result = ActionResult(str(intent), "unsupported", "That command is not supported.")
            elif ((intent in SLOTS and slot not in SLOTS[intent])
                  or (intent not in SLOTS and slot not in (None, ""))):
                result = ActionResult(intent, "invalid_slot", "That value isn't supported in this demo.")
            elif INTENTS[intent] not in self.services:
                result = ActionResult(intent, "unavailable", "This action is not configured.")
            else:
                try:
                    service = self.services[INTENTS[intent]]
                    if intent == "CALL":
                        # Calling adapter invokes acknowledgement after readiness
                        # checks, before Dial, so it cannot mask a connected call.
                        def acknowledge():
                            nonlocal speech_status
                            speech_status = self._say("Placing your call.")
                        result = service.execute(intent, slot, before_dial=acknowledge)
                    else:
                        result = service.execute(intent, slot)
                except Exception:
                    # Do not retry effects or expose raw exceptions/recipients.
                    result = ActionResult(intent, "unavailable", "The action could not be completed. Please check its setup.")
            service_finished = time.monotonic()
            if self.on_action is not None:
                try:
                    self.on_action(result.to_dict(), service_finished - dispatch_started)
                except Exception:
                    pass  # Logging must never fail/retry an action.
            if not (intent == "CALL" and result.status == "dialing"):
                speech_status = self._say(result.message)
            output = result.to_dict()
            output["_service_finished_monotonic"] = service_finished
            output["timing"] = {
                "service_seconds": round(service_finished - dispatch_started, 3),
                "response_seconds": round(time.monotonic() - service_finished, 3),
            }
            if speech_status is not None:
                output["speech"] = speech_status
            return output

    def _say(self, message):
        if self.speech is None:
            return None
        try:
            timing = self.speech.speak(message)
            return {"status": "ok", **(timing if isinstance(timing, dict) else {})}
        except Exception:
            return {"status": "unavailable", "message": "Unable to synthesize or play speech."}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            failed = False
            for service in self.services.values():
                close = getattr(service, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        failed = True
            if self.speech is not None:
                try:
                    self.speech.close()
                except Exception:
                    failed = True
            if failed:
                raise RuntimeError("Some action resources could not be closed. Check the devices and playback.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def variations():
    return [(intent, slot) for intent in INTENTS
            for slot in SLOTS.get(intent, (None,))]
