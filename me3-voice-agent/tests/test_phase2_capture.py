import asyncio
from pathlib import Path
import subprocess
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase2"))
from operator_native import diagnostic_command, wait_diagnostics, countdown


async def test_diagnostic_uses_wav_file_not_raw_stdout(tmp_path):
    output = tmp_path / "raw.wav"
    command = diagnostic_command("explicit.source", output)
    # Executable boundary fixture models the reported --raw behavior. No audio
    # utility/device is called; a real child writes/finalizes a WAV container.
    code = """
import sys,wave
if '--raw' in sys.argv:
    sys.stdout.buffer.write(b'\\0\\0'*32)
else:
    with wave.open(sys.argv[-1], 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(b'\\1\\0'*32)
"""
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-B", "-c", code, *command[1:], stdout=subprocess.DEVNULL
    )
    assert await child.wait() == 0
    with wave.open(str(output), "rb") as wav:
        assert wav.getnframes() == 32 and wav.getframerate() == 48000
    assert "--raw" not in command and "--target=explicit.source" in command


async def test_diagnostic_readiness_fails_on_early_exit_and_timeout(tmp_path):
    class Child:
        returncode = 1

    with pytest.raises(RuntimeError, match="exited"):
        await wait_diagnostics([Child()], [tmp_path / "missing.wav"], timeout=0.01)
    Child.returncode = None
    with pytest.raises(TimeoutError):
        await wait_diagnostics([Child()], [tmp_path / "missing.wav"], timeout=0.01)


async def test_countdown_is_visible_and_separate_from_trial_start(capsys):
    events, waits = [], []

    async def sleep(seconds):
        waits.append(seconds)

    await countdown(events, sleep)
    text = capsys.readouterr().out
    assert text.index("[READY]") < text.index("[COUNTDOWN] 3") < text.index("[BEGIN]")
    assert waits == [1, 1, 1]
    assert [e["event"] for e in events] == ["countdown"] * 3 + ["trial_begin"]


def test_operator_entrypoint_refuses_unarmed_run():
    script = Path(__file__).resolve().parents[1] / "scripts/phase2/qualify_audio.sh"
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "No devices open" in result.stdout


async def test_cleanup_continues_after_error_and_stops_owned_tasks():
    from operator_native import cleanup_native

    calls = []

    class Input:
        async def close_capture(self):
            calls.append("input")
            raise OSError("fixture failure")

    class Output:
        async def halt(self, *, disable):
            assert disable
            calls.append("output")

    class Child:
        returncode = None

        def terminate(self):
            calls.append("tap")
            self.returncode = -15

        async def wait(self):
            return self.returncode

    monitor = asyncio.create_task(asyncio.sleep(10))
    errors = await cleanup_native(None, None, Input(), Output(), [Child()], monitor, None)
    assert errors == ["OSError"] and calls == ["input", "output", "tap"] and monitor.cancelled()
