"""Manage the Pi demo through the existing native bridge entrypoint."""

import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import stat
import subprocess
import time
import urllib.error
import urllib.request

PROFILE = Path("var/cache/pi-demo-launch.json")
STATE = Path("var/cache/pi-demo")
BASE = "http://127.0.0.1:"
OPTIONS = {
    "--runtime-config", "--backend-config", "--device-config", "--secrets-file",
    "--llm-secrets-file", "--research-secrets-file", "--receipt-store", "--trace-dir",
}
ENV = {
    "ME3_REQUEST_RUN_ID", "ME3_REQUEST_LEDGER", "ME3_RESEARCH_LEDGER",
    "ME3_TASK_DATABASE", "ME3_TASK_WORKSPACE", "ME3_DEMO_OWNER_CONFIG",
    "ME3_MUSIC_CONFIG",
}


class LocalOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}), LocalOnly())


def private_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SystemExit(f"{path} must be a private file owned by this user.")
        return json.load(stream)


def request(port, path="/api/health", *, token=None):
    headers = {"X-Swift-Demo-Token": token} if token else {}
    req = urllib.request.Request(
        f"{BASE}{port}{path}", headers=headers, method="POST" if token else "GET"
    )
    with HTTP.open(req, timeout=2) as response:
        return json.load(response)


def health(port):
    try:
        return request(port)
    except (OSError, ValueError):
        return {}


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def owner(port, status):
    try:
        token = private_json(STATE / f"{port}.json")["stop_token"]
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if isinstance(token, str) and hashlib.sha256(token.encode()).hexdigest() == status.get(
        "demo_instance"
    ):
        return token
    return None


def show(port, status, managed):
    if not status:
        print("Swift demo: port occupied by an unavailable listener." if listening(port) else
              "Swift demo: stopped.")
    else:
        print(f"Swift bridge: {status.get('bridge')} / {status.get('engine')} / {status.get('voice_session')}"
              f" ({'managed' if managed else 'external'})")
        print(f"Runtime SHA: {status.get('source_revision') or 'unreported by external bridge'}")
        print(f"Voice ready: {status.get('voice_ready') is True}")
        print(f"Pi Chromium (Pi microphone and speaker): {BASE}{port}/")
        budget = status.get("budget", {})
        print(f"Ledger run: {budget.get('run_id') or 'none'}")
        for name, quota in budget.items():
            if name.endswith("_requests") and isinstance(quota, dict) and quota.get("limit") is not None:
                print(f"  {name}: {quota.get('remaining')} of {quota['limit']} remaining")
    fields = os.environ.get("SSH_CONNECTION", "").split()
    target = f"{getpass.getuser()}@{fields[2]}" if len(fields) == 4 else "<user@pi-address>"
    print(f"Laptop WSL tunnel: ./scripts/swift.sh pi-viewer --dashboard {shlex.quote(target)}")
    print("Then open the URL printed by the tunnel in your laptop browser (uses laptop audio).")
    print(f"Pi desktop tunnel (requires a compatible viewer): ./scripts/swift.sh pi-viewer {shlex.quote(target)}")
    print("Status: ./scripts/swift.sh demo status")
    print("Stop managed bridge: ./scripts/swift.sh demo stop")
    if status and not managed:
        print("This bridge is externally owned. End the conversation and stop it from its original launcher before starting a managed instance.")
    print("Recovery/relaunch: ./scripts/swift.sh demo")


def start(port, profile_path):
    profile = private_json(profile_path)
    if not isinstance(profile, dict):
        raise SystemExit("Invalid private Pi demo profile.")
    options, configured = profile.get("options", {}), profile.get("environment", {})
    if (
        not isinstance(options, dict) or set(options) - OPTIONS
        or not isinstance(configured, dict) or set(configured) - ENV
        or any(not isinstance(v, str) or not v for v in (*options.values(), *configured.values()))
    ):
        raise SystemExit("Invalid private Pi demo profile.")
    if any(key not in options for key in ("--backend-config", "--runtime-config")):
        raise SystemExit("Select explicit backend and runtime configurations in the private Pi profile.")
    if "--secrets-file" in options or "--llm-secrets-file" in options or "--research-secrets-file" in options:
        if not all(configured.get(key) for key in ("ME3_REQUEST_RUN_ID", "ME3_REQUEST_LEDGER")):
            raise SystemExit("Live Pi profiles require an explicitly configured run ID and ledger.")
        if not Path(configured["ME3_REQUEST_LEDGER"]).is_file():
            raise SystemExit("The configured ledger must already exist. No allowance was created.")
        if "--research-secrets-file" in options and not Path(
            configured.get("ME3_RESEARCH_LEDGER", "")
        ).is_file():
            raise SystemExit("Live research requires its configured existing ledger.")
    args = [part for item in options.items() for part in item]
    token = secrets.token_hex(32)
    env = os.environ.copy()
    for key in ENV:
        env.pop(key, None)
    env.update(configured, ME3_DEMO_STOP_TOKEN=token)
    record_path = STATE / f"{port}.json"
    log = STATE / f"{port}.log"
    temporary = STATE / f"{port}.new"
    temporary.write_text(json.dumps({"stop_token": token}))
    temporary.replace(record_path)
    with log.open("ab") as output:
        child = subprocess.Popen(
            ["bash", "scripts/swift.sh", "native-bridge", *args, "--port", str(port)],
            env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True,
        )
    deadline = time.monotonic() + 35
    while child.poll() is None and time.monotonic() < deadline:
        current = health(port)
        if owner(port, current) and current.get("bridge") == "ready" and current.get("voice_ready"):
            run_id = configured.get("ME3_REQUEST_RUN_ID")
            if run_id and current.get("budget", {}).get("run_id") != run_id:
                break
            show(port, current, managed=True)
            print(f"Log: {log}. Open the dashboard in Pi Chromium or use the viewer command above.")
            return
        time.sleep(0.3)
    if child.poll() is None:
        # This child was created by this invocation and never reached usable readiness.
        child.terminate()
    raise SystemExit(f"Swift did not become voice ready. Inspect {log}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"), nargs="?", default="start")
    parser.add_argument("--port", type=int, default=18770)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a loopback port between 1024 and 65535.")
    os.umask(0o077)
    STATE.mkdir(exist_ok=True, parents=True)
    info = STATE.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit(f"{STATE} must be private and owned by this user.")
    lock_fd = os.open(STATE / "launcher.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = health(args.port)
        token = owner(args.port, current) if current else None
        if args.action == "status" or args.action == "start" and current:
            show(args.port, current, bool(token))
            if args.action == "start" and (
                current.get("bridge") != "ready" or current.get("engine") != "native"
                or current.get("voice_ready") is not True
            ):
                raise SystemExit("The existing bridge is not voice ready.")
            return
        if args.action == "stop":
            if not token:
                show(args.port, current, False)
                if current or listening(args.port):
                    raise SystemExit("Refusing to stop an unowned listener.")
                return
            try:
                request(args.port, "/api/demo/stop", token=token)
            except urllib.error.HTTPError as exc:
                raise SystemExit(exc.read(300).decode(errors="replace")) from None
            for _ in range(60):
                if not listening(args.port):
                    print("Managed Swift bridge stopped. Relaunch: ./scripts/swift.sh demo")
                    return
                time.sleep(0.2)
            raise SystemExit("Graceful stop is still in progress. Check demo status.")
        if listening(args.port):
            raise SystemExit("Demo port is occupied; the existing listener was left running.")
        if not args.profile.is_file():
            raise SystemExit(f"Missing private Pi demo profile: {args.profile}")
        start(args.port, args.profile)


if __name__ == "__main__":
    main()
