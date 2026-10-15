"""Read-only, redacted PipeWire call-state snapshots; never dials or hangs up."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hey_swift.actions.services.calling import BluetoothCallingBackend, CallingUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", help="Known gateway object; otherwise require exactly one connected gateway.")
    parser.add_argument("--seconds", type=int, default=1, choices=range(1, 121), metavar="1..120")
    args = parser.parse_args()
    backend = BluetoothCallingBackend(timeout=3)
    try:
        gateway = args.gateway
        if not gateway:
            result = backend._call("/org/pipewire/Telephony", "org.ofono.Manager", "GetModems")
            data = result.get("data", [])
            modems = data[0] if isinstance(data, list) and len(data) == 1 else data
            if not isinstance(modems, dict) or len(modems) != 1:
                raise CallingUnavailable("Require exactly one tablet gateway or an explicit --gateway.")
            gateway = next(iter(modems))
        if not isinstance(gateway, str) or not gateway.startswith("/org/pipewire/Telephony/"):
            raise CallingUnavailable("Unexpected gateway path; refusing another D-Bus object.")
        deadline = time.monotonic() + args.seconds
        while True:
            print(json.dumps({"event": "call_state_diagnostic", "monotonic": round(time.monotonic(), 3),
                              "gateway": gateway, "states": backend.get_call_states(gateway)}), flush=True)
            if time.monotonic() + 1 >= deadline:
                return 0
            time.sleep(1)
    except CallingUnavailable as error:
        print(json.dumps({"event": "call_state_diagnostic", "status": "unavailable", "message": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
