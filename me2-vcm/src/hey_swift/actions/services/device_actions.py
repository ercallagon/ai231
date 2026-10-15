"""Validated LED/display actions with optional shared speech and background jobs."""
import argparse
from datetime import datetime, timedelta
from dataclasses import replace
import json
import math
import threading
import time

from hey_swift.actions.services.information import ActionResult

VALUES = {
    "TIMER": {"10 seconds": 10, "30 seconds": 30, "1 minute": 60},
    "ALARM": {"6:00 AM": 6, "8:00 AM": 8, "9:00 PM": 21},
    "TEMPERATURE": {"18 degrees": 18, "22 degrees": 22, "26 degrees": 26},
    "BRIGHTNESS": {"20 percent": .2, "60 percent": .6, "100 percent": 1.0},
    "COLOR": {"Red": (1, 0, 0), "Blue": (0, 0, 1), "Green": (0, 1, 0)},
}
FIXED = {"LIGHT_ON", "LIGHT_OFF"}

def supported_variations():
    return [(intent, None) for intent in sorted(FIXED)] + [
        (intent, value) for intent, values in VALUES.items() for value in values
    ]

def print_variations():
    for index, (intent, value) in enumerate(supported_variations(), 1):
        print(f"{index:2d}. {intent}" + (f" — {value}" if value else ""))

def run_menu(service):
    print_variations()
    print("m: show menu | c: cancel timer/alarm | q: quit")
    delivered = 0
    while True:
        for event in service.events[delivered:]:
            print(json.dumps(event.to_dict()), flush=True)
        delivered = len(service.events)
        try:
            choice = input("Select action: ").strip().lower()
        except EOFError:
            return
        if choice == "q":
            return
        if choice == "m":
            print_variations()
            continue
        if choice == "c":
            result = service.cancel()
            service._say(result.message)
        elif choice.isdigit() and 1 <= int(choice) <= len(supported_variations()):
            intent, value = supported_variations()[int(choice) - 1]
            result = service.execute(intent, value)
        else:
            print("Choose a listed number, m, c, or q.")
            continue
        print(json.dumps(result.to_dict()), flush=True)

class DeviceActions:
    """Reuse one instance. Call only after VCM confidence/OOS gating.

    New valid TIMER/ALARM/TEMPERATURE commands replace the active display action.
    Scheduling is in-memory; the process must remain running.
    """
    def __init__(self, hardware, *, speech=None):
        self.hardware, self.speech = hardware, speech
        self._lock = threading.RLock()
        self._commands = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self.color, self.brightness, self.on = (1, 1, 1), 1.0, False
        self.events = []
    def _led(self):
        self.hardware.led(tuple(x * self.brightness if self.on else 0 for x in self.color))
    def _say(self, message):
        if self.speech:
            self.speech.speak(message)
    def execute(self, intent, slot_value=None):
        with self._commands:
            return self._request(intent, slot_value)
    def _request(self, intent, slot_value):
        replaced = False
        if (not self._closed and intent in {"TIMER", "ALARM", "TEMPERATURE"}
                and slot_value in VALUES[intent]
                and self._thread and self._thread.is_alive()):
            # Join outside the display lock so the previous worker can exit.
            self.cancel()
            replaced = True
        with self._lock:
            if self._closed:
                return ActionResult(intent, "unavailable", "Device service is closed.")
            if intent not in VALUES and intent not in FIXED:
                return ActionResult(intent, "unsupported", "Unsupported device intent.")
            if (intent in FIXED and slot_value not in (None, "")) or (intent in VALUES and slot_value not in VALUES[intent]):
                result = ActionResult(intent, "invalid_slot", "That value isn't supported in this demo.")
            else:
                try:
                    result = self._execute(intent, slot_value)
                    if replaced:
                        result.data["replaced_previous_schedule"] = True
                        result = replace(result, message="Previous schedule cancelled. " + result.message)
                except Exception as exc:
                    return ActionResult(intent, "unavailable", "The hardware action could not be completed.", {"error": str(exc)})
            try:
                self._say(result.message)
            except Exception as exc:
                result.data["speech_error"] = str(exc)
            return result
    def _execute(self, intent, value):
        if intent in FIXED:
            self.on = intent == "LIGHT_ON"
            self._led()
            message = "Lights on." if self.on else "Lights off."
        elif intent == "COLOR":
            self.color, self.on = VALUES[intent][value], True
            self._led()
            message = f"Lights set to {value.lower()}."
        elif intent == "BRIGHTNESS":
            self.brightness, self.on = VALUES[intent][value], True
            self._led()
            message = f"Brightness set to {value}."
        elif intent == "TEMPERATURE":
            self.hardware.display(f"{VALUES[intent][value]:4d}")
            message = f"Temperature set to {value}."
        else:
            if intent == "TIMER":
                seconds = VALUES[intent][value]
                self.hardware.display(f"{seconds // 60:02d}{seconds % 60:02d}", True)
                target = None
                message = f"Timer set for {value}."
            else:
                now = datetime.now().astimezone()
                target = now.replace(hour=VALUES[intent][value], minute=0, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                seconds = None
                self.hardware.display(f"{target.hour:02d}{target.minute:02d}", True)
                message = f"Alarm set for {value}."
            self._stop.clear()
            self._thread = threading.Thread(target=self._job, args=(intent, seconds, target), daemon=True)
            self._thread.start()
        return ActionResult(intent, "ok", message, {"slot_value": value})
    def _job(self, intent, seconds, target):
        deadline = time.monotonic() + seconds if seconds is not None else None
        last_display = None
        try:
            while not self._stop.wait(.2):
                remaining = deadline - time.monotonic() if deadline else (target - datetime.now().astimezone()).total_seconds()
                with self._lock:
                    if self._stop.is_set():
                        return
                    if intent == "TIMER":
                        n = max(0, math.ceil(remaining))
                        digits = f"{n // 60:02d}{n % 60:02d}"
                        if digits != last_display:
                            self.hardware.display(digits, True)
                            last_display = digits
                if remaining <= 0:
                    message = "Your timer is finished." if intent == "TIMER" else "Your alarm is ringing."
                    self.events.append(ActionResult(intent, "completed", message))
                    self._say(message)
                    return
        except Exception as exc:
            self.events.append(ActionResult(intent, "unavailable", "Scheduled action failed.", {"error": str(exc)}))
    def cancel(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        with self._lock:
            self.hardware.display("    ")
        return ActionResult("CANCEL", "ok", "Scheduled action cancelled.")
    def wait(self):
        if self._thread:
            self._thread.join()
    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        with self._lock:
            self._closed = True
            self.hardware.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", nargs="?", choices=sorted(FIXED | VALUES.keys()))
    parser.add_argument("--list", action="store_true", help="List supported action/value combinations without hardware")
    parser.add_argument("--menu", action="store_true", help="Select action/value combinations by number")
    parser.add_argument("--value")
    parser.add_argument("--hardware", choices=["mock", "pi"], default="mock")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--hold", type=float, default=5, help="Hold LED/display output before cleanup (seconds)")
    args = parser.parse_args()
    if args.list:
        print_variations()
        return 0
    if args.menu and (args.intent or args.value):
        parser.error("--menu selects intent/value interactively")
    if not args.menu and not args.intent:
        parser.error("Choose an intent, --list, or --menu")
    if not math.isfinite(args.hold) or args.hold < 0:
        parser.error("--hold must be finite and nonnegative")
    # Validate BEFORE acquiring GPIO or installing any hardware dependencies.
    if args.intent in VALUES and args.value not in VALUES[args.intent]:
        parser.error("--value must be one of: " + ", ".join(VALUES[args.intent]))
    if args.intent in FIXED and args.value:
        parser.error("This intent has no slot")
    from hey_swift.hardware.devices import MockHardware, PiHardware
    from hey_swift.audio.speech import PiperSpeech, SpeechService
    service = None
    try:
        hardware = PiHardware() if args.hardware == "pi" else MockHardware()
        speech = SpeechService(PiperSpeech("audio/voices/en_US-amy-medium.onnx")) if args.speak else None
        service = DeviceActions(hardware, speech=speech)
        if args.menu:
            run_menu(service)
            return 0
        result = service.execute(args.intent, args.value)
        print(json.dumps(result.to_dict()), flush=True)
        if result.status == "ok":
            if args.intent in {"TIMER", "ALARM"}:
                service.wait()
                for event in service.events:
                    print(json.dumps(event.to_dict()), flush=True)
            else:
                time.sleep(args.hold)
        if args.hardware == "mock":
            print(json.dumps({"hardware_events": hardware.events}))
        return 0 if result.status == "ok" and not result.data.get("speech_error") and all(e.status == "completed" for e in service.events) else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"status": "unavailable", "message": str(exc)}))
        return 1
    finally:
        if service:
            service.close()

if __name__ == "__main__":
    raise SystemExit(main())
