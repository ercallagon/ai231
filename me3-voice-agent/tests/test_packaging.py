"""Portable operator entry points without services, credentials or devices."""

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mock_preflight_without_private_configuration():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "me3_voice.preflight"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    report = json.loads(result.stdout)
    assert all(role["backend"] == "mock" for role in report["roles"].values())
    assert not any(role["credential_present"] for role in report["roles"].values())


def test_launch_help_from_outside_checkout(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/swift.sh"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert "bridge" in result.stdout and "audio" in result.stdout
    assert not list(tmp_path.iterdir())


def test_setup_rejects_unknown_option_before_installation(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/setup.sh"), "--unknown-option"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "Unknown option" in result.stderr
    assert not list(tmp_path.iterdir())


def test_network_check_deduplicates_hosts_and_reports_blocked_route():
    import importlib.util

    from me3_voice.configuration import BackendSpec

    module_spec = importlib.util.spec_from_file_location(
        "check_network", ROOT / "scripts/check_network.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    calls = []

    def connect(target, *, timeout):
        calls.append(target)
        if target[0] == "10.153.60.233":
            raise TimeoutError("no private network route")
        return nullcontext()

    result = module.inspect(
        {
            "asr": BackendSpec(backend="external", endpoint="https://api.groq.com/a"),
            "llm": BackendSpec(backend="external", endpoint="https://api.groq.com/b"),
            "tts": BackendSpec(backend="external", endpoint="http://10.153.60.233:8001/a"),
        },
        connect=connect,
    )
    assert calls == [("api.groq.com", 443), ("10.153.60.233", 8001)]
    assert [row["status"] for row in result["checks"]] == [
        "tcp_reachable",
        "tcp_reachable",
        "tcp_unreachable",
    ]
