import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase2"))
import pytest
from pipecat.frames.frames import OutputAudioRawFrame, StartFrame
from pipewire_transport import PipeWireInput, PipeWireOutput, open_owned


class Child:
    def __init__(self):
        self.returncode = None
        self.terminated = 0
        self.frames = []
        self.stdin = self

    def write(self, pcm):
        self.frames.append(pcm)

    async def drain(self):
        await asyncio.sleep(0)

    def terminate(self):
        self.terminated += 1
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


async def test_explicit_device_gate():
    async def forbidden(*a, **k):
        raise AssertionError("device was opened")

    for obj in (
        PipeWireInput("reviewed.node", spawn=forbidden),
        PipeWireOutput("reviewed.sink", spawn=forbidden),
    ):
        with pytest.raises(RuntimeError, match="disabled"):
            await obj.start(StartFrame())


async def test_output_reaps_stops_and_reopens_only_explicit_target():
    children = []
    commands = []

    async def spawn(*args, **kwargs):
        commands.append(args)
        child = Child()
        children.append(child)
        return child

    out = PipeWireOutput("me3.phase2.aec.sink", allow_playback=True, spawn=spawn)
    out.enabled = True
    frame = OutputAudioRawFrame(b"\1\0" * 320, 16000, 1)
    assert await out.write_audio_frame(frame)
    assert "--target=me3.phase2.aec.sink" in commands[0]
    await out.halt()
    assert children[0].terminated == 1
    assert await out.write_audio_frame(frame)
    assert len(children) == 2
    await out.halt(disable=True)
    assert not await out.write_audio_frame(frame)
    assert len(children) == 2


async def test_cancelled_open_is_owned_and_reaped():
    release = asyncio.Event()
    child = Child()

    async def spawn(*a, **k):
        await release.wait()
        return child

    job = asyncio.create_task(open_owned(spawn, "harmless-substitute"))
    await asyncio.sleep(0)
    job.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert child.terminated == 1


async def test_output_failure_never_selects_another_sink():
    child = Child()
    n = 0

    async def fail():
        raise BrokenPipeError()

    child.drain = fail

    async def spawn(*a, **k):
        nonlocal n
        n += 1
        return child

    out = PipeWireOutput("only.this.sink", allow_playback=True, spawn=spawn)
    out.enabled = True
    assert not await out.write_audio_frame(OutputAudioRawFrame(b"\1\0" * 160, 16000, 1))
    assert out.failure == "BrokenPipeError" and not out.enabled
    assert not await out.write_audio_frame(OutputAudioRawFrame(b"\1\0" * 160, 16000, 1))
    assert n == 1


async def test_interrupt_during_open_never_writes_stale_audio():
    entered = asyncio.Event()
    release = asyncio.Event()
    child = Child()

    async def spawn(*a, **k):
        entered.set()
        await release.wait()
        return child

    out = PipeWireOutput("only.this.sink", allow_playback=True, spawn=spawn)
    out.enabled = True
    writer = asyncio.create_task(
        out.write_audio_frame(OutputAudioRawFrame(b"\1\0" * 160, 16000, 1))
    )
    await entered.wait()
    stop = asyncio.create_task(out.halt())
    await asyncio.sleep(0)
    release.set()
    assert not await writer
    await stop
    assert child.terminated == 1 and not child.frames


async def test_capture_rate_boundary_failure_and_no_fallback():
    inp = PipeWireInput("only.this.source", allow_capture=True)
    inp._audio_in_queue = asyncio.Queue()

    class Source(Child):
        count = 0

        async def readexactly(self, n):
            assert n == 1920
            self.count += 1
            if self.count > 100:
                raise asyncio.IncompleteReadError(b"", n)
            return b"\1\0" * 960

    child = Source()
    child.stdout = child
    inp.child = child
    frames = []
    errors = []

    async def push(frame):
        frames.append(frame)

    async def error(**kwargs):
        errors.append(kwargs)

    inp.push_audio_frame = push
    inp.push_error = error
    await inp.capture()
    assert frames and all(f.sample_rate == 16000 and f.num_channels == 1 for f in frames)
    assert 30000 <= sum(len(f.audio) // 2 for f in frames) <= 32000
    assert inp.failure == "IncompleteReadError" and len(errors) == 1 and child.terminated == 1


async def test_sidecar_and_native_same_pcm_target_and_owned_stop(monkeypatch):
    from me3_voice.pi_audio import PipeWirePlayback

    children = []
    acks = []
    commands = []

    async def spawn(*args, **kwargs):
        child = Child()
        children.append(child)
        commands.append(args)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def ack(seq):
        acks.append(seq)

    old = PipeWirePlayback("same.explicit.sink", ack, None)
    payload = b"\1\0" * 160
    await old.write(1, payload)
    await asyncio.wait_for(old.queue.join(), 2)
    await old.stop()
    native = PipeWireOutput("same.explicit.sink", allow_playback=True, spawn=spawn)
    native.enabled = True
    assert await native.write_audio_frame(OutputAudioRawFrame(payload, 16000, 1))
    await native.halt(disable=True)
    assert acks == [1]
    assert all(c.frames == [payload] and c.returncode is not None for c in children)
    assert all("--target=same.explicit.sink" in cmd for cmd in commands)


def test_operator_without_capture_never_opens_devices(tmp_path):
    import os, subprocess, sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "phase2" / "operator_native.py"
    destination = Path.cwd() / "var/scratch/phase2-device-gate-must-not-exist"
    assert not destination.exists()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--source",
            "me3.phase2.aec.source",
            "--sink",
            "me3.phase2.aec.sink",
            "--raw-mic",
            "explicit.physical.mic",
            "--reference",
            str(tmp_path / "no-reference.wav"),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "Capture disabled" in result.stdout and not destination.exists()
