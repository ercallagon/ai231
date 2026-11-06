"""Opt-in loopback benchmark transport smoke: synthetic input, canonical MCP mocks.

No microphone, speakers, inference, GPIO or paid judge. Uses the pinned external
benchmark checkout's real WebSocket transport, not a substitute transport.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def check(args):
    bench = args.benchmark_root.resolve()
    out = args.output_dir.resolve()
    if not bench.is_relative_to(ROOT / "var") or not out.is_relative_to(ROOT / "var"):
        raise ValueError("Reference and output must be under ignored project var/")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(bench / "adapters"))
    import fdbench_marks
    from pipecat_fdbench import make_benchmark_transport
    from websockets.asyncio.server import serve

    spec = importlib.util.spec_from_file_location(
        "me3_benchmark_entry", ROOT / "scripts/fdbench_entry.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    os.environ["ME3_FDBENCH_MOCK"] = "1"
    os.environ["ME3_FDBENCH_EXPORT_TEXT"] = "1"
    marks = []
    fdbench_marks.set_sink(marks.append)
    with wave.open(str(args.synthetic_wav)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Synthetic fixture must be PCM16 16 kHz mono")
        pcm = wav.readframes(wav.getnframes())
    results = []
    try:
        for activation in ("active", "inactive"):
            os.environ["ME3_FDBENCH_ACTIVATION"] = activation
            connected = asyncio.get_running_loop().create_future()
            audio, cleared = asyncio.Event(), asyncio.Event()
            received = bytearray()

            async def handler(ws):
                connected.set_result(ws)
                async for message in ws:
                    if isinstance(message, bytes):
                        received.extend(message)
                        audio.set()
                    elif json.loads(message).get("event") == "clear":
                        cleared.set()

            async with serve(handler, "127.0.0.1", 0) as server:
                transport = make_benchmark_transport(
                    f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                )
                run, cancel = await entry.build(transport, out / activation)
                task = asyncio.create_task(run)
                try:
                    ws = await asyncio.wait_for(connected, 5)
                    await asyncio.sleep(
                        0.4
                    )  # Allow the actual transport's audio queue to initialize.
                    fixture = bytes(16000) + pcm + bytes(40000)
                    for offset in range(0, len(fixture), 640):
                        await ws.send(fixture[offset : offset + 640])
                        await asyncio.sleep(0.02)
                    if activation == "active":
                        await asyncio.wait_for(audio.wait(), 5)
                        sink = entry.BenchmarkSink(transport.output())
                        await asyncio.wait_for(sink.stop(), 3)
                        await asyncio.wait_for(cleared.wait(), 3)
                    else:
                        await asyncio.sleep(1)
                        assert not received, "Inactive fresh session emitted unsolicited audio"
                finally:
                    await asyncio.wait_for(cancel(), 8)
                    await asyncio.wait_for(task, 8)
            summary = json.loads((out / activation / "me3-benchmark-summary.json").read_text())
            assert summary["playback_queue_size"] == 0
            results.append(
                {
                    "activation": activation,
                    "input_frames": summary["input_frames"],
                    "output_bytes": len(received),
                    "clear_received": cleared.is_set(),
                    "cleanup": summary["stage"],
                }
            )
        assert any(m["stage"] == "tts_first_audio" for m in marks)
        assert any(m["stage"] == "asr_final" and "text" in m for m in marks)
        report = {
            "evidence": "synthetic loopback, real benchmark transport, actual ME3 Session/MCP mocks; not live inference or acoustics",
            "cases": results,
            "stage_counts": {
                name: sum(m["stage"] == name for m in marks) for name in {m["stage"] for m in marks}
            },
            "live_requests": 0,
        }
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        fdbench_marks.set_sink(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root", type=Path, default=ROOT / "var/cache/fd-benchmark-reference"
    )
    parser.add_argument("--synthetic-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(check(parser.parse_args()))
