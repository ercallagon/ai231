"""Matched actual CLI-dispatch workload; mock inference, actual speech MCP.

Run each engine in a fresh process. Both use explicit complete-utterance EOF
without an extra silence timer. Not a physical or commercial-model benchmark.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import wave
from datetime import datetime, timezone

from me3_voice.cli import mock_conversation
from me3_voice.configuration import RuntimeOptions
from me3_voice.speech import SpeechClient


def distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "p50": values[len(values) // 2] if values else None,
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))] if values else None,
        "max": max(values) if values else None,
    }


def resource_sample():
    pids = [os.getpid()]
    path = Path(f"/proc/self/task/{os.getpid()}/children")
    if path.exists():
        pids.extend(map(int, path.read_text().split()))
    rows = []
    for pid in pids:
        row = {"pid": pid}
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith(("Rss:", "Pss:", "Swap:")):
                    name, value = line.split(":", 1)
                    row[name + "_kib"] = int(value.split()[0])
            row["threads"] = int(
                next(
                    v.split()[1]
                    for v in Path(f"/proc/{pid}/status").read_text().splitlines()
                    if v.startswith("Threads:")
                )
            )
        except (OSError, StopIteration):
            row["unavailable"] = True
        rows.append(row)
    return rows


async def run(args):
    paths = [
        Path("src/me3_voice") / name
        for name in [
            "cli.py",
            "configuration.py",
            "native_app.py",
            "native_bridge.py",
            "native_llm.py",
            "conversation_context.py",
            "conversation_tools.py",
            "research.py",
            "credentials.py",
            "native_pipeline.py",
            "native_speech.py",
            "session.py",
            "harness.py",
            "speech.py",
            "servers.py",
            "remote.py",
        ]
    ] + [Path(__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    pcm = b"\1\0" * 1024
    if args.wav:
        with wave.open(str(args.wav), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("mono PCM16/16k WAV required")
            pcm = wav.readframes(1024)
    calls = []
    lag = []
    resources = []
    stop = False
    source = None
    queue_peaks = {}
    if args.engine == "native":
        from me3_voice.native_app import NativeApplication as Controller
    else:
        from me3_voice.session import Session as Controller
    original_enter = Controller.__aenter__

    async def enter(self):
        nonlocal source
        result = await original_enter(self)
        source = self
        return result

    Controller.__aenter__ = enter
    original = SpeechClient.call

    async def call(self, kind, name, arguments):
        start = time.monotonic()
        result = await original(self, kind, name, arguments)
        calls.append(
            {
                "kind": kind,
                "name": name,
                "start": start,
                "end": time.monotonic(),
                "done": result.get("done"),
            }
        )
        return result

    SpeechClient.call = call

    async def monitor():
        count = 0
        while not stop:
            expected = time.monotonic() + 0.01
            await asyncio.sleep(0.01)
            lag.append((time.monotonic(), max(0, time.monotonic() - expected) * 1000))
            if source is not None:
                if args.engine == "native":
                    depths = {"native_admission": len(source.pending)}
                    if source.parts:
                        depths["native_stt_pending_bytes"] = len(source.parts.stt.pending)
                        for processor in source.parts.pipeline.processors:
                            for name, value in vars(processor).items():
                                if isinstance(value, asyncio.Queue):
                                    depths[type(processor).__name__ + "." + name] = value.qsize()
                else:
                    depths = {
                        "legacy_input": source.input_queue.qsize() if source.input_queue else 0,
                        "legacy_playback": source.playback.queue.qsize(),
                    }
                for name, depth in depths.items():
                    queue_peaks[name] = max(queue_peaks.get(name, 0), depth)
            if count % 10 == 0:
                resources.append({"at": time.monotonic(), "processes": resource_sample()})
            count += 1

    heartbeat = asyncio.create_task(monitor())
    await asyncio.sleep(0)
    started = time.monotonic()
    cpu = time.process_time()
    try:
        value = await mock_conversation(
            [pcm] * (args.samples + 1), runtime=RuntimeOptions(engine=args.engine)
        )
    finally:
        stop = True
        await heartbeat
        SpeechClient.call = original
        Controller.__aenter__ = original_enter
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()), (
        "source changed during benchmark"
    )
    events = value["events"]

    def stamp(e):
        return e.get("at", e.get("monotonic"))

    endpoints = [stamp(e) for e in events if e["event"] == "endpoint_submitted"]
    finals = [stamp(e) for e in events if e["event"] == "asr_final"]
    writes = [
        stamp(e)
        for e in events
        if e["event"] == "playback_packet" or e["event"].startswith("sink_write_started_")
    ]
    opens = [c["start"] for c in calls if c["name"] == "asr_open"]
    assert len(endpoints) == len(finals) == len(opens) == args.samples + 1
    rows = []
    for index, endpoint in enumerate(endpoints):
        end = opens[index + 1] if index + 1 < len(opens) else float("inf")
        active = [c for c in calls if opens[index] <= c["start"] < end]
        first_write = next(t for t in writes if endpoint <= t < end)
        tts_open = next(c["start"] for c in active if c["name"] == "tts_open")
        first_pcm = next(c["end"] for c in active if c["name"] == "tts_next" and c["done"] is False)
        if index:
            row = {
                "endpoint_to_final_ms": (finals[index] - endpoint) * 1000,
                "endpoint_to_first_memory_write_ms": (first_write - endpoint) * 1000,
                "tts_open_to_first_pcm_ms": (first_pcm - tts_open) * 1000,
                "mcp_calls": len(active),
            }
            if args.engine == "native":
                row["llm_request_to_first_token_ms"] = next(
                    e["seconds"] * 1000
                    for e in events
                    if e["event"] == "llm_first_token" and opens[index] <= stamp(e) < end
                )
            rows.append(row)
    ready = value["application_ready_at"]
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "samples": args.samples,
        "evidence": "mock upstream; real application CLI dispatch and MCP; immediate memory delivery",
        "turn_semantics": value["turn_completion"],
        "warmup_turns_excluded": 1,
        "model": "existing MockLLM",
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "pcm_samples": len(pcm) // 2,
        "environment": {"platform": platform.platform(), "python": sys.version},
        "source_sha256": hashes,
        "protocols": value["protocols"],
        "startup_seconds": ready - started,
        "main_cpu_seconds": time.process_time() - cpu,
        "stages": {key: distribution([r[key] for r in rows]) for key in rows[0]},
        "append_rpc_ms": distribution(
            [(c["end"] - c["start"]) * 1000 for c in calls if c["name"] == "asr_append"]
        ),
        "startup_lag_ms": distribution([v for at, v in lag if at < ready]),
        "steady_lag_ms": distribution([v for at, v in lag if at >= ready]),
        "resource_samples": resources,
        "queue_peaks_sampled_10ms": queue_peaks,
        "raw_trials": rows,
        "limitations": [
            "Legacy marker is immediately before memory sink invocation; native marker immediately after it",
            "Native framework output uses40ms packets versus legacy32ms and may pad final silence",
            "No physical, provider latency/accuracy or durable-task result",
            "10ms heartbeat/queue and100ms resource snapshots are sampled, not hard maxima",
            "Startup excludes initial Python/framework imports and benchmark-hook setup",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(data, file, indent=2)
    print(
        json.dumps(
            {
                "engine": args.engine,
                "stages": data["stages"],
                "steady_lag_ms": data["steady_lag_ms"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["legacy", "native"], required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 75 or not args.output.resolve().is_relative_to(
        Path("var").resolve()
    ):
        parser.error("samples1..75 and fresh output under var/ required")
    if args.output.exists():
        parser.error("preserve existing output")
    asyncio.run(run(args))
