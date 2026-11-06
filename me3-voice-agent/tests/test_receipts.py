import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from me3_voice.contracts import TurnKey
from me3_voice.providers import ToolCall, MockLLM
from me3_voice.tools import MockTaskExecutor
from me3_voice.music import MusicController
from me3_voice.receipts import ReceiptStore
from me3_voice.session import Session
from me3_voice.transcript import Transcript


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        yield b"\0\0"


async def test_late_native_completion_settles_same_unknown_receipt_without_new_action():
    tasks = MockTaskExecutor(music=MusicController())
    start = asyncio.Event()
    release = asyncio.Event()
    original = tasks.actions._run

    async def pending(action, *args):
        if action == "light_on":
            start.set()
            await release.wait()
        return await original(action, *args)

    tasks.actions._run = pending
    display = Transcript()

    async def sink(pcm):
        pass

    async with Session(
        Speech(), MockLLM(), sink, tasks=tasks, transcript=display, direct_clock_enabled=True
    ) as s:
        s.activate()
        old = await s.submit(b"Turn on the lights.")
        await start.wait()
        await s.interrupt()
        await asyncio.gather(old, return_exceptions=True)
        row = display.snapshot()["tasks"][0]
        identity = row["id"]
        assert row["status"] == "unknown"
        await (await s.submit(b"What time is it?"))
        newer = dict(display.snapshot()["tasks"][1])
        release.set()
        async with asyncio.timeout(1):
            while display.snapshot()["tasks"][0]["status"] == "unknown":
                await asyncio.sleep(0)
        rows = display.snapshot()["tasks"]
        assert (
            len(rows) == 2
            and rows[1] == newer
            and rows[0]["id"] == identity
            and rows[0]["status"] == "simulated"
        )
        assert tasks.actions.effects == 1 and tasks.actions.light
        assert s.playback.queue.empty()


async def test_durable_completed_and_pending_are_not_reexecuted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "var/cache/receipts.sqlite3"
    key = TurnKey(session_id="old", turn_id=1)
    call = ToolCall("ui-" + "a" * 32, "run_device_action", {"action": "light_on"})
    first = MockTaskExecutor(music=MusicController())
    first.receipt_store = ReceiptStore(path)
    result = await first.execute(key, call)
    assert result["result_status"] == "simulated" and first.actions.effects == 1
    await first.close()
    second = MockTaskExecutor(music=MusicController())
    second.receipt_store = ReceiptStore(path)
    recovered = await second.execute(TurnKey(session_id="new", turn_id=9), call)
    assert (
        second.actions.effects == 0
        and "Recorded mock" in recovered["summary"]
        and "widget" not in recovered
    )
    with pytest.raises(ValueError):
        await second.execute(key, ToolCall(call.call_id, call.name, {"action": "light_off"}))
    pending = ToolCall("ui-" + "b" * 32, "run_device_action", {"action": "light_on"})
    await second.receipt_store.claim(pending.call_id, pending, second.receipt_context(pending))
    unknown = await second.execute(key, pending)
    assert unknown["result_status"] == "unknown" and second.actions.effects == 0
    await second.close()


# Real disposable child crashes, with an external mock effect file independent of SQLite.
CHILD = r"""
import asyncio,json,os,sys
from pathlib import Path
from me3_voice.receipts import ReceiptStore
from me3_voice.providers import ToolCall
from me3_voice.contracts import TurnKey
from me3_voice.tools import MockTaskExecutor
from me3_voice.music import MusicController
async def main():
 store=ReceiptStore(Path('var/cache/crash.sqlite3'));executor=MockTaskExecutor(music=MusicController());executor.receipt_store=store
 timer=sys.argv[1].startswith('timer')
 call=ToolCall('ui-'+'c'*32,'run_device_action',{'action':'set_timer','value':'10 seconds'} if timer else {'action':'light_on'})
 original=executor._execute
 async def effect(key,call):
  if sys.argv[1]=='before-effect':os._exit(17)
  result=await original(key,call)
  with open('effect.log','a') as f:f.write('applied\n');f.flush();os.fsync(f.fileno())
  if sys.argv[1]=='after-effect':os._exit(18)
  return result
 executor._execute=effect
 result=await executor.execute(TurnKey(session_id='same',turn_id=1),call)
 if sys.argv[1] in {'after-receipt','timer-create'}:os._exit(19)
 Path('readback.json').write_text(json.dumps({'state':'pending' if result.get('result_status')=='unknown' else 'complete','result':result,'schedule':executor.actions.schedule}))
 await executor.close()
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "window,expected_effects,state",
    [
        ("before-effect", 0, "pending"),
        ("after-effect", 1, "pending"),
        ("after-receipt", 1, "complete"),
    ],
)
def test_real_process_crash_never_replays_pending_effect(tmp_path, window, expected_effects, state):
    source = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": source, "PYTHONDONTWRITEBYTECODE": "1"}
    script = tmp_path / "crash.py"
    script.write_text(CHILD)
    first = subprocess.run(
        [sys.executable, "-B", str(script), window],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert first.returncode in {17, 18, 19}, first.stderr.decode()
    second = subprocess.run(
        [sys.executable, "-B", str(script), "reconnect"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert second.returncode == 0, second.stderr.decode()
    assert json.loads((tmp_path / "readback.json").read_text())["state"] == state
    effects = (
        (tmp_path / "effect.log").read_text().count("applied")
        if (tmp_path / "effect.log").exists()
        else 0
    )
    assert effects == expected_effects


def test_real_timer_process_restart_recovers_receipt_without_rearming_timer(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = tmp_path / "timer_crash.py"
    script.write_text(CHILD)
    a = subprocess.run(
        [sys.executable, "-B", str(script), "timer-create"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert a.returncode == 19, a.stderr.decode()
    b = subprocess.run(
        [sys.executable, "-B", str(script), "timer-reconnect"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert b.returncode == 0, b.stderr.decode()
    result = json.loads((tmp_path / "readback.json").read_text())
    assert result["schedule"] is None and "widget" not in result["result"]
    assert "not restored" in result["result"]["summary"]
    assert (tmp_path / "effect.log").read_text().count("applied") == 1


async def test_timeout_before_execution_stays_unknown_without_retry(tmp_path, monkeypatch):
    from me3_voice.receipts import DispatchUncertain

    monkeypatch.chdir(tmp_path)
    executor = MockTaskExecutor(music=MusicController())
    executor.receipt_store = ReceiptStore(Path("var/cache/timeout.db"))
    calls = []

    async def unavailable(key, call):
        calls.append(call.call_id)
        raise TimeoutError("no execution")

    executor._execute = unavailable
    key = TurnKey(session_id="s", turn_id=1)
    call = ToolCall("ui-" + "d" * 32, "run_device_action", {"action": "light_on"})
    try:
        with pytest.raises(DispatchUncertain):
            await executor.execute(key, call)
        # Same process does not retry even a failed awaited operation.
        with pytest.raises(DispatchUncertain):
            await executor.execute(key, call)
        assert len(calls) == 1 and executor.actions.effects == 0
    finally:
        await executor.close()
    recovered = MockTaskExecutor(music=MusicController())
    recovered.receipt_store = ReceiptStore(Path("var/cache/timeout.db"))
    try:
        value = await recovered.execute(key, call)
        assert value["result_status"] == "unknown" and recovered.actions.effects == 0
    finally:
        await recovered.close()


async def test_bridge_services_attach_opt_in_store(tmp_path, monkeypatch):
    from argparse import Namespace
    from me3_voice.configuration import load_backends
    from me3_voice.voice_bridge import VoiceRuntimeServices

    monkeypatch.chdir(tmp_path)
    services = VoiceRuntimeServices(
        load_backends(None),
        {},
        Namespace(onit_python=None, receipt_store=Path("var/cache/runtime.db")),
    )
    services.speech = object()  # No MCP/provider startup required for this binding check.
    try:
        _, _, tasks = await services.open()
        assert isinstance(tasks.receipt_store, ReceiptStore)
        result = await tasks.execute(
            TurnKey(session_id="s", turn_id=1),
            ToolCall("ui-" + "e" * 32, "run_device_action", {"action": "light_on"}),
        )
        assert result["result_status"] == "simulated" and tasks.actions.effects == 1
    finally:
        await services.close()


@pytest.mark.parametrize("applied", [True, False])
async def test_lost_music_ack_uses_fresh_state_without_repeating_pause(applied):
    from me3_voice.music import MusicUnavailable

    writes = []
    reads = []
    device = {"id": "owned", "is_active": True, "is_restricted": False, "name": "Mock player"}
    state = {
        "device": device,
        "is_playing": True,
        "item": {"uri": "spotify:track:one", "name": "One"},
    }

    async def peer(method, path, **kwargs):
        if path == "/me/player/devices":
            return {"devices": [device]}
        if path == "/me/player/pause":
            writes.append(path)
            if applied:
                state["is_playing"] = False
            raise MusicUnavailable("ack lost", code="music_transport_failed")
        if path == "/me/player":
            reads.append(path)
            return state
        raise AssertionError((method, path))

    music = MusicController(token="offline-token", device_id="owned", request=peer)
    try:
        if applied:
            receipt = await music.execute("same-operation", "pause")
            assert receipt["display"] == "paused" and "reports" in receipt["summary"]
            assert await music.execute("same-operation", "pause") == receipt
        else:
            with pytest.raises(MusicUnavailable) as error:
                await music.execute("same-operation", "pause")
            assert error.value.outcome_uncertain
            with pytest.raises(MusicUnavailable):
                await music.execute("same-operation", "pause")
        assert len(writes) == 1 and reads and state["is_playing"] is (not applied)
    finally:
        await music.close()


async def test_invalid_durable_config_never_silently_falls_back(tmp_path, monkeypatch):
    from argparse import Namespace
    from me3_voice.configuration import load_backends
    from me3_voice.voice_bridge import VoiceRuntimeServices

    monkeypatch.chdir(tmp_path)
    services = VoiceRuntimeServices(
        load_backends(None), {}, Namespace(onit_python=None, receipt_store=tmp_path / "outside.db")
    )
    services.speech = object()
    try:
        for _ in range(2):
            with pytest.raises(ValueError, match="under project"):
                await services.open()
    finally:
        await services.close()


async def test_two_executors_share_intent_without_duplicate_dispatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    one = MockTaskExecutor(music=MusicController())
    two = MockTaskExecutor(music=MusicController())
    one.receipt_store = ReceiptStore(Path("var/cache/shared.db"))
    two.receipt_store = ReceiptStore(Path("var/cache/shared.db"))
    started = asyncio.Event()
    release = asyncio.Event()
    original = one._execute

    async def delayed(key, call):
        started.set()
        await release.wait()
        return await original(key, call)

    one._execute = delayed
    call = ToolCall("ui-" + "f" * 32, "run_device_action", {"action": "light_on"})
    key = TurnKey(session_id="s", turn_id=1)
    pending = asyncio.create_task(one.execute(key, call))
    try:
        await started.wait()
        blocked = await two.execute(key, call)
        assert blocked["result_status"] == "unknown" and two.actions.effects == 0
        release.set()
        await pending
        assert one.actions.effects + two.actions.effects == 1
    finally:
        release.set()
        await pending
        await one.close()
        await two.close()


async def test_changed_target_cannot_recover_mock_as_a_pi_action(tmp_path, monkeypatch):
    from me3_voice.devices import DeviceActions, UnavailablePiDeviceOutput

    monkeypatch.chdir(tmp_path)
    path = Path("var/cache/targets.db")
    key = TurnKey(session_id="s", turn_id=1)
    call = ToolCall("ui-" + "1" * 32, "run_device_action", {"action": "light_on"})
    first = MockTaskExecutor(music=MusicController())
    first.receipt_store = ReceiptStore(path)
    await first.execute(key, call)
    await first.close()
    devices = DeviceActions(output=UnavailablePiDeviceOutput("not connected"), backend="pi")
    second = MockTaskExecutor(music=MusicController(), devices=devices)
    second.receipt_store = ReceiptStore(path)
    try:
        with pytest.raises(ValueError, match="different arguments"):
            await second.execute(key, call)
        assert devices.effects == 0
    finally:
        await second.close()
        await devices.close()


async def test_recovery_inspection_is_read_only(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.chdir(tmp_path)
    store = ReceiptStore(Path("var/cache/inspect.db"))
    call = ToolCall("ui-" + "2" * 32, "run_device_action", {"action": "light_on"})
    await store.claim(call.call_id, call)
    await store.close()
    path = Path("var/cache/inspect.db")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    check = subprocess.run(
        [sys.executable, "-B", "-m", "me3_voice.receipts", "inspect", "--store", str(path)],
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert check.returncode == 0, check.stderr.decode()
    row = json.loads(check.stdout)[0]
    assert row["operation_id"] == call.call_id and row["state"] == "unknown"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
