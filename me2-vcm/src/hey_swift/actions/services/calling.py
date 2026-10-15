"""Standalone Pi SIM dialing through PipeWire's native Bluetooth telephony API."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from hey_swift.integration.config import communication_values
from hey_swift.integration.config import runtime_root

from hey_swift.audio.speech import PiperSpeech, SpeechService, SpeechUnavailable


class CallingUnavailable(Exception):
    pass


class CallingUncertain(CallingUnavailable):
    """Dial may have reached the phone; never retry automatically."""


class BluetoothCallingBackend:
    def __init__(self, *, runner=subprocess.run, timeout=15):
        self._runner = runner
        self.timeout = timeout

    def _call(self, path, interface, method, *args):
        try:
            result = self._runner(
                ["busctl", "--user", "--json=short", "call", "org.pipewire.Telephony",
                 path, interface, method, *args],
                capture_output=True, text=True, check=True, timeout=self.timeout)
            return json.loads(result.stdout) if result.stdout.strip() else {}
        except subprocess.TimeoutExpired as exc:
            if method == "Dial":
                error = CallingUncertain("Check your phone before calling again.")
                error.gateway = path
                raise error from exc
            raise CallingUnavailable("Bluetooth calling response timed out.") from exc
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise CallingUnavailable("Bluetooth calling is unavailable. Check the tablet connection.") from exc

    def dial(self, number, *, before_dial=None):
        if not re.fullmatch(r"\+[1-9][0-9]{7,14}", number):
            raise ValueError("Configure the recipient in international format, e.g. +639...")
        modems = self._call("/org/pipewire/Telephony", "org.ofono.Manager", "GetModems")
        data = modems.get("data", [])
        # busctl JSON wraps method return arguments; GetModems returns one dict.
        gateways = data[0] if isinstance(data, list) and len(data) == 1 else data
        if not isinstance(gateways, dict) or len(gateways) != 1:
            raise CallingUnavailable("Connect exactly one calling tablet before dialing.")
        gateway = next(iter(gateways))
        if self.has_active_calls(gateway):
            raise CallingUnavailable("A call already exists. End it on the tablet first.")
        if before_dial:
            before_dial()
        response = self._call(gateway, "org.ofono.VoiceCallManager", "Dial", "s", number)
        returned = response.get("data", [])
        call_path = returned[0] if isinstance(returned, list) and len(returned) == 1 else returned
        if not isinstance(call_path, str) or not call_path.startswith(gateway + "/"):
            call_path = None
        return {"intent": "CALL", "status": "dialing", "message": "Call requested.",
                "data": {"gateway": gateway, "call_path": call_path}}

    def get_call_states(self, gateway):
        """Return only object identity/state; never expose recipient or caller data."""
        calls = self._call(gateway, "org.ofono.VoiceCallManager", "GetCalls")
        data = calls.get("data", [])
        active = data[0] if isinstance(data, list) and len(data) == 1 else data
        if active == []:
            return {}
        if not isinstance(active, dict):
            raise CallingUnavailable("Unexpected Bluetooth call-state response")
        states = {}
        known = {"active", "held", "dialing", "alerting", "incoming", "waiting", "disconnected"}
        for path, properties in active.items():
            if not isinstance(path, str) or not isinstance(properties, dict):
                raise CallingUnavailable("Unexpected Bluetooth call properties")
            state = properties.get("State")
            if isinstance(state, dict) and state.get("type") == "s":
                state = state.get("data")
            if isinstance(state, list) and len(state) == 1:
                state = state[0]
            states[path] = state if isinstance(state, str) and state in known else "unknown"
        return states

    def has_active_calls(self, gateway):
        # A disconnected object can briefly remain registered. Unknown state stays blocking.
        return any(state != "disconnected" for state in self.get_call_states(gateway).values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dial", action="store_true", help="Place a real SIM call to VCM_CALL_NUMBER")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice-model", default=str(runtime_root() / "assets/voices/en_US-amy-medium.onnx"))
    args = parser.parse_args()
    if not args.dial:
        parser.error("--dial is required to explicitly initiate a real call")
    number = communication_values()["call_number"]
    if not number:
        parser.error("Set VCM_CALL_NUMBER to your configured recipient")
    try:
        speech = SpeechService(PiperSpeech(args.voice_model)) if args.speak else None
        result = BluetoothCallingBackend().dial(
            number, before_dial=(lambda: speech.speak("Placing your call.")) if speech else None)
    except (CallingUnavailable, SpeechUnavailable, ValueError) as exc:
        print(json.dumps({"intent": "CALL", "status": "unavailable", "message": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
