"""Persistent offline CREATE_REMINDER/LIST_REMINDERS, independent of VCM/audio."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import threading

from hey_swift.actions.services.information import ActionResult

VALUES = {"Drink water": "drink water", "Study": "study", "Exercise": "exercise"}


class ReminderStorageUnavailable(Exception):
    pass


class JsonReminderStore:
    """Atomic JSON replacement. Use one shared instance/action worker for writes."""
    def __init__(self, path="data/runtime/reminders.json"):
        self.path = Path(path)
        self.lock = threading.RLock()

    def _read(self):
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            entries = payload["reminders"]
            if payload["version"] != 1 or not isinstance(entries, list):
                raise ValueError("Unsupported reminder storage")
            seen = set()
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("slot_value") not in VALUES:
                    raise ValueError("Invalid reminder")
                if entry["slot_value"] in seen:
                    raise ValueError("Duplicate reminder")
                seen.add(entry["slot_value"])
                timestamp = datetime.fromisoformat(entry["created_at"])
                if timestamp.utcoffset() is None:
                    raise ValueError("Missing timezone")
            return entries
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReminderStorageUnavailable("Unable to read saved reminders.") from exc

    def _write(self, entries):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                    dir=self.path.parent, prefix=".reminders-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"version": 1, "reminders": entries}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise ReminderStorageUnavailable("Unable to save reminders.") from exc
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def list(self):
        with self.lock:
            return self._read()

    def add(self, value, timestamp):
        with self.lock:
            entries = self._read()
            if any(entry["slot_value"] == value for entry in entries):
                return False, entries
            entries.append({"slot_value": value, "created_at": timestamp})
            self._write(entries)
            return True, entries

    def clear(self):
        with self.lock:
            self._read()  # Never silently overwrite corrupt/unsupported storage.
            self._write([])


class ReminderService:
    def __init__(self, store=None, *, clock=lambda: datetime.now().astimezone()):
        self.store = store if store is not None else JsonReminderStore()
        self.clock = clock

    def execute(self, intent, slot_value=None):
        if intent not in {"CREATE_REMINDER", "LIST_REMINDERS"}:
            return ActionResult(intent, "unsupported", "Unsupported reminder intent.")
        if intent == "CREATE_REMINDER" and slot_value not in VALUES:
            return ActionResult(intent, "invalid_slot", "That reminder isn't supported.")
        if intent == "LIST_REMINDERS" and slot_value not in (None, ""):
            return ActionResult(intent, "invalid_slot", "This command does not accept a slot value.")
        try:
            if intent == "CREATE_REMINDER":
                now = self.clock()
                if now.utcoffset() is None:
                    raise ValueError("Reminder clock must return a timezone-aware datetime.")
                added, entries = self.store.add(slot_value, now.isoformat())
                text = VALUES[slot_value]
                message = (f"Your reminder to {text} has been saved." if added else
                           f"You already have a reminder to {text}.")
                return ActionResult(intent, "ok", message,
                    {"slot_value": slot_value, "added": added, "count": len(entries)})
            entries = self.store.list()
            texts = [VALUES[entry["slot_value"]] for entry in entries]
            if not texts:
                message = "You have no reminders yet."
            elif len(texts) == 1:
                message = f"Your reminder is to {texts[0]}."
            else:
                listing = ", ".join(texts[:-1]) + " and " + texts[-1]
                message = f"Your reminders are to {listing}."
            return ActionResult(intent, "ok", message, {"count": len(entries), "reminders": entries})
        except ReminderStorageUnavailable:
            return ActionResult(intent, "unavailable", "Your reminders are unavailable. Please check the saved reminder file.")

    def clear_for_test(self):
        """Manual runtime control, not a classifier intent."""
        try:
            self.store.clear()
            return ActionResult("REMINDER_RESET", "ok", "Your reminders have been cleared.", {"count": 0})
        except ReminderStorageUnavailable:
            return ActionResult("REMINDER_RESET", "unavailable", "Unable to clear your reminders. Please check the saved reminder file.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", nargs="?", choices=["CREATE_REMINDER", "LIST_REMINDERS"])
    parser.add_argument("--value", choices=list(VALUES))
    parser.add_argument("--menu", action="store_true")
    parser.add_argument("--store", default="data/runtime/reminders.json")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice-model", default="audio/voices/en_US-amy-medium.onnx")
    args = parser.parse_args()
    if bool(args.intent) == args.menu:
        parser.error("Choose one intent or --menu")
    if args.menu and args.value:
        parser.error("--value is for a single CREATE_REMINDER command")
    service = ReminderService(JsonReminderStore(args.store))
    speech = None
    if args.speak:
        from hey_swift.audio.speech import PiperSpeech, SpeechService
        speech = SpeechService(PiperSpeech(args.voice_model))

    def output(result):
        payload = result.to_dict()
        if speech:
            from hey_swift.audio.speech import SpeechUnavailable
            try:
                speech.speak(result.message)
                payload["speech"] = {"status": "ok"}
            except SpeechUnavailable as exc:
                payload["speech"] = {"status": "unavailable", "message": str(exc)}
        print(json.dumps(payload, indent=2))
        return 2 if payload.get("speech", {}).get("status") == "unavailable" else (0 if result.status == "ok" else 1)

    if not args.menu:
        return output(service.execute(args.intent, args.value))
    values = list(VALUES)
    try:
        while True:
            print("1. Drink water\n2. Study\n3. Exercise\n4. List reminders\nc. Clear reminders\nq. Exit")
            choice = input("Select action: ").strip().lower()
            if choice == "q":
                return 0
            if choice in {"1", "2", "3"}:
                output(service.execute("CREATE_REMINDER", values[int(choice) - 1]))
            elif choice == "4":
                output(service.execute("LIST_REMINDERS"))
            elif choice == "c":
                if input("Clear all saved reminders? Type yes: ").strip().lower() == "yes":
                    output(service.clear_for_test())
    except (KeyboardInterrupt, EOFError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
