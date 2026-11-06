"""Opt-in TCP routing checks for configured endpoints; no keys or inference sent.

Run this on the Pi itself. A laptop VPN does not supply the Pi's network route.
TCP reachability does not validate TLS, authentication, model health or inference.
"""

import argparse
import json
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

from me3_voice.configuration import load_backends


def inspect(backends, *, connect=socket.create_connection):
    rows = []
    checked = {}
    for role, spec in backends.items():
        if spec.backend != "external":
            rows.append({"role": role, "status": "not_required", "backend": "mock"})
            continue
        url = urlsplit(spec.endpoint)
        host = url.hostname
        port = url.port or (443 if url.scheme in ("https", "wss") else 80)
        target = (host, port)
        if target not in checked:
            start = time.monotonic()
            try:
                with connect(target, timeout=3):
                    status = "tcp_reachable"
            except OSError:
                status = "tcp_unreachable"
            checked[target] = {
                "status": status,
                "connect_ms": round((time.monotonic() - start) * 1000, 2),
            }
        rows.append({"role": role, "host": host, "port": port, **checked[target]})
    return {
        "checks": rows,
        "limitations": "TCP routing only; no HTTP, credentials, inference, model-health or audio test. Run on the actual Pi/network. Laptop VPN access is not inherited.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(load_backends(args.backend_config))
    print(json.dumps(report, indent=2))
    raise SystemExit(any(row["status"] == "tcp_unreachable" for row in report["checks"]))
