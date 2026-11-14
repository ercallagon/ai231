"""Opt-in foreground VPN/bridge supervision. Never starts at import or boot."""

import argparse
import asyncio
import fcntl
import json
import os
import signal
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def private_file(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("VPN files must not be symlinks")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("VPN files must be private regular files owned by this user (mode 0600)")
    return path


def settings(path):
    config = private_file(path)
    data = json.loads(config.read_text())
    profile, auth = private_file(data["profile"]), private_file(data["auth_file"])
    rows = auth.read_text().splitlines()
    if len(rows) != 2 or not all(rows) or any("\x00" in row for row in rows):
        raise ValueError("VPN auth file must contain two nonempty lines; values excluded")
    # Managed ownership requires a foreground process and readable credentials
    # during later reauthentication. Do not silently alter a personal profile.
    directives = {
        line.split()[0]
        for line in profile.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";", "<"))
    }
    if directives & {"daemon", "user", "group", "chroot", "management"}:
        raise ValueError(
            "Profile has daemon/privilege-drop/management directives; use reviewed manual startup"
        )
    return profile, auth


class Supervisor:
    def __init__(
        self,
        profile,
        auth,
        backend,
        bridge_args,
        *,
        timeout=120,
        spawn=asyncio.create_subprocess_exec,
    ):
        self.profile, self.auth, self.backend = profile, auth, backend
        self.bridge_args, self.timeout, self.spawn = bridge_args, timeout, spawn
        self.vpn = self.bridge = self.gate = None
        self.stop = asyncio.Event()
        self.signal_code = 130

    async def authorize(self):
        process = await self.spawn("/usr/bin/sudo", "-v")
        if await process.wait():
            raise RuntimeError("OS sudo authorization failed; VPN was not started")

    async def ready(self):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("VPN initialization deadline expired")
            line = await asyncio.wait_for(self.vpn.stdout.readline(), remaining)
            if not line:
                raise RuntimeError(
                    "OpenVPN exited before initialization completed; check the profile/access"
                )
            # Consume output without recording connection details or printing
            # profile/key contents. Only safe stage/error categories are emitted.
            if b"AUTH_FAILED" in line:
                raise RuntimeError("VPN authentication failed; update the saved credentials")
            if b"Initialization Sequence Completed" in line:
                return

    async def drain(self):
        while await self.vpn.stdout.readline():
            pass

    async def terminate(self, process):
        if process is None or process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 8)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def terminate_vpn(self):
        if self.vpn is None or self.vpn.returncode is not None:
            return
        # The owned sudo/OpenVPN process group can contain root processes.
        # Never guess another PID, kill an existing tunnel, or cache root passwords.
        command = await self.spawn(
            "/usr/bin/sudo",
            "-n",
            "/bin/kill",
            "-TERM",
            "--",
            f"-{self.vpn.pid}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await command.wait():
            raise RuntimeError(
                "VPN cleanup not confirmed: OS sudo permission expired; disconnect the owned tunnel manually"
            )
        try:
            await asyncio.wait_for(self.vpn.wait(), 10)
        except TimeoutError:
            raise RuntimeError(
                "VPN did not exit after stop; inspect the owned tunnel before restarting"
            ) from None

    async def start(self, attribute, *args, **kwargs):
        # A cancellation while subprocess creation is in flight must still
        # retain the handle so the owner's finally block can reap that child.
        pending = asyncio.create_task(self.spawn(*args, **kwargs))
        try:
            process = await asyncio.shield(pending)
        except asyncio.CancelledError:
            process = await pending
            setattr(self, attribute, process)
            raise
        setattr(self, attribute, process)
        return process

    async def pipeline(self):
        self.vpn = await self.start(
            "vpn",
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/openvpn",
            "--cd",
            str(self.profile.parent),
            "--config",
            str(self.profile),
            "--auth-user-pass",
            str(self.auth),
            "--auth-nocache",
            "--auth-retry",
            "none",
            "--verb",
            "3",
            start_new_session=True,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        print("Waiting for VPN initialization…", flush=True)
        await self.ready()
        drain = asyncio.create_task(self.drain())
        try:
            print("VPN initialized. Checking configured TCP routes…", flush=True)
            self.gate = await self.start(
                "gate",
                sys.executable,
                "-B",
                str(ROOT / "scripts/check_network.py"),
                "--backend-config",
                str(self.backend),
                start_new_session=True,
            )
            if await asyncio.wait_for(self.gate.wait(), 15):
                raise RuntimeError("Network gate failed; bridge was not started")
            if self.vpn.returncode is not None:
                raise RuntimeError("VPN exited before bridge launch")
            self.bridge = await self.start(
                "bridge",
                sys.executable,
                "-B",
                "-m",
                "me3_voice.voice_bridge",
                "--host",
                "127.0.0.1",
                "--backend-config",
                str(self.backend),
                *self.bridge_args,
                start_new_session=True,
            )
            print("ME3 bridge started with managed VPN. Ctrl+C stops both.", flush=True)
            bridge_wait = asyncio.create_task(self.bridge.wait())
            vpn_wait = asyncio.create_task(self.vpn.wait())
            try:
                done, _ = await asyncio.wait(
                    (bridge_wait, vpn_wait), return_when=asyncio.FIRST_COMPLETED
                )
                if bridge_wait in done:
                    return bridge_wait.result()
                raise RuntimeError("VPN disconnected; stopping the owned bridge")
            finally:
                for task in (bridge_wait, vpn_wait):
                    task.cancel()
                await asyncio.gather(bridge_wait, vpn_wait, return_exceptions=True)
        finally:
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    async def run(self):
        await self.authorize()
        work = asyncio.create_task(self.pipeline())
        stop_wait = asyncio.create_task(self.stop.wait())
        try:
            done, _ = await asyncio.wait((work, stop_wait), return_when=asyncio.FIRST_COMPLETED)
            if stop_wait in done:
                return self.signal_code
            return await work
        finally:
            work.cancel()
            stop_wait.cancel()
            await asyncio.gather(work, stop_wait, return_exceptions=True)
            try:
                await self.terminate(self.gate)
                await self.terminate(self.bridge)
            finally:
                await self.terminate_vpn()


async def launch(args):
    profile, auth = settings(args.vpn_config)
    if not Path("/usr/sbin/openvpn").is_file():
        raise ValueError(
            "Install approved OpenVPN first; no system package is installed automatically"
        )
    backend = Path(args.backend_config).resolve()
    from .configuration import load_backends

    load_backends(backend)  # Fail before changing routes if the profile is malformed.
    lockpath = ROOT / "var/cache/vpn-launch.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lockpath, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another managed VPN launch is already running") from None
        existing = await asyncio.create_subprocess_exec(
            "/usr/bin/pgrep", "-x", "openvpn", stdout=asyncio.subprocess.DEVNULL
        )
        if await existing.wait() == 0:
            raise ValueError(
                "An OpenVPN process is already running; inspect/stop it before managed launch"
            )
        tail = args.bridge_args[1:] if args.bridge_args[:1] == ["--"] else args.bridge_args
        if any(
            value in {"--host", "--backend-config"}
            or value.startswith(("--host=", "--backend-config="))
            for value in tail
        ):
            raise ValueError(
                "The VPN launcher controls bridge host/backend; omit duplicate flags after --"
            )
        supervisor = Supervisor(profile, auth, backend, tail, timeout=args.wait_seconds)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):

            def stop(sig=sig):
                supervisor.signal_code = 128 + sig
                supervisor.stop.set()

            loop.add_signal_handler(sig, stop)
        try:
            return await supervisor.run()
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vpn-config", type=Path, default=Path("var/cache/vpn-launch.json"))
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("bridge_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 300:
        parser.error("--wait-seconds must be 1..300")
    try:
        raise SystemExit(asyncio.run(launch(args)))
    except RuntimeError as error:
        parser.exit(1, str(error) + "\n")
    except (OSError, ValueError, TimeoutError):
        # Provider/credential values are never interpolated into error output.
        parser.exit(
            1,
            "VPN/bridge startup or cleanup failed; check profile permissions, saved credentials, network and OS sudo permission. No secrets were printed.\n",
        )


if __name__ == "__main__":
    main()
