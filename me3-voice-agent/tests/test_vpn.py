"""VPN supervision with owned unprivileged subprocess peers; no VPN/network/root."""

import asyncio
import importlib.util
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from me3_voice.vpn import Supervisor, settings

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("configure_vpn", ROOT / "scripts/configure_vpn.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_private_masked_setup_and_no_overwrite(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "private-home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(setup.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter(("fixture-user", "fixture-password"))
    monkeypatch.setattr(setup.getpass, "getpass", lambda prompt: next(answers))
    profile = home / "vpn.ovpn"
    profile.write_text("# approved client fixture\nclient\nremote example.invalid\n")
    profile.chmod(0o600)
    config = tmp_path / "var/cache/vpn-launch.json"
    setup.configure(profile, config)
    auth = home / ".config/openvpn/me3-auth.txt"
    assert auth.stat().st_mode & 0o777 == 0o600
    assert config.stat().st_mode & 0o777 == 0o600
    assert auth.parent.stat().st_mode & 0o777 == 0o700
    assert settings(config) == (profile, auth)
    output = capsys.readouterr().out
    assert "fixture-user" not in output and "fixture-password" not in output
    with pytest.raises(FileExistsError):
        setup.configure(profile, config)
    auth.chmod(0o644)
    with pytest.raises(ValueError, match="private regular"):
        settings(config)


class Peers:
    """Map external command boundaries to real harmless owned Python children."""

    def __init__(self, *, ready=True, gate_ok=True, bridge_sleep=0.05, kill_ok=True):
        self.ready, self.gate_ok = ready, gate_ok
        self.bridge_sleep, self.kill_ok = bridge_sleep, kill_ok
        self.calls, self.children = [], []
        self.bridge_started = asyncio.Event()

    async def spawn(self, *args, **kwargs):
        self.calls.append(args)
        if args[:2] == ("/usr/bin/sudo", "-v"):
            code = "pass"
        elif "/usr/sbin/openvpn" in args:
            code = (
                "print('Initialization Sequence Completed', flush=True);" if self.ready else ""
            ) + "import time;time.sleep(30)"
        elif "/bin/kill" in args:
            pid = abs(int(args[-1]))
            assert pid in {p.pid for p in self.children}
            if self.kill_ok:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            code = "raise SystemExit(" + ("0" if self.kill_ok else "1") + ")"
        elif any(str(a).endswith("check_network.py") for a in args):
            code = "raise SystemExit(" + ("0" if self.gate_ok else "1") + ")"
        elif "me3_voice.voice_bridge" in args:
            code = f"import time;time.sleep({self.bridge_sleep})"
            self.bridge_started.set()
        else:
            raise AssertionError("Unexpected subprocess boundary")
        kwargs.pop("start_new_session", None)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", code, start_new_session=True, **kwargs
        )
        self.children.append(process)
        return process

    async def close(self):
        for child in self.children:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await child.wait()


def supervisor(peers, *, timeout=2):
    return Supervisor(
        Path("/private/profile.ovpn"),
        Path("/private/auth.txt"),
        Path("backend.json"),
        ["--secrets-file", "private.env"],
        timeout=timeout,
        spawn=peers.spawn,
    )


@pytest.mark.asyncio
async def test_initialize_gate_bridge_exit_cleanup_order():
    peers = Peers()
    try:
        assert await supervisor(peers).run() == 0
        vpn = next(c for c in peers.calls if "/usr/sbin/openvpn" in c)
        assert vpn.index("--config") < vpn.index("--auth-user-pass")
        assert "--auth-nocache" in vpn
        assert not any("fixture-password" in str(c) for c in peers.calls)
        assert peers.bridge_started.is_set()
        assert all(p.returncode is not None for p in peers.children)
    finally:
        await peers.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ready,gate_ok", [(False, True), (True, False)])
async def test_failed_initialization_or_gate_never_starts_bridge(ready, gate_ok):
    peers = Peers(ready=ready, gate_ok=gate_ok)
    try:
        with pytest.raises((TimeoutError, RuntimeError)):
            await supervisor(peers, timeout=0.1).run()
        assert not peers.bridge_started.is_set()
        assert all(p.returncode is not None for p in peers.children)
    finally:
        await peers.close()


@pytest.mark.asyncio
async def test_stop_event_reaps_bridge_and_tunnel():
    peers = Peers(bridge_sleep=30)
    owner = supervisor(peers)
    try:
        run = asyncio.create_task(owner.run())
        await asyncio.wait_for(peers.bridge_started.wait(), 3)
        owner.stop.set()
        assert await asyncio.wait_for(run, 3) == 130
        assert all(p.returncode is not None for p in peers.children)
    finally:
        await peers.close()


@pytest.mark.asyncio
async def test_failed_privileged_cleanup_is_explicit():
    peers = Peers(kill_ok=False)
    try:
        with pytest.raises(RuntimeError, match="cleanup not confirmed"):
            await supervisor(peers).run()
    finally:
        await peers.close()  # Test owns harmless children; never a real tunnel.


@pytest.mark.asyncio
async def test_lost_vpn_stops_owned_bridge():
    peers = Peers(bridge_sleep=30)
    owner = supervisor(peers)
    try:
        run = asyncio.create_task(owner.run())
        await asyncio.wait_for(peers.bridge_started.wait(), 3)
        while owner.bridge is None:
            await asyncio.sleep(0)
        os.killpg(owner.vpn.pid, signal.SIGTERM)
        with pytest.raises(RuntimeError, match="VPN disconnected"):
            await asyncio.wait_for(run, 3)
        assert owner.bridge.returncode is not None
    finally:
        await peers.close()


def test_cli_forwards_existing_bridge_options_after_separator(monkeypatch):
    import me3_voice.vpn as vpn

    captured = []

    async def launch(args):
        captured.append(args)
        return 0

    monkeypatch.setattr(vpn, "launch", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vpn",
            "--backend-config",
            "backend.json",
            "--",
            "--secrets-file",
            "models.env",
            "--device-config",
            "mock.json",
        ],
    )
    with pytest.raises(SystemExit) as result:
        vpn.main()
    assert result.value.code == 0
    assert captured[0].bridge_args == [
        "--",
        "--secrets-file",
        "models.env",
        "--device-config",
        "mock.json",
    ]
