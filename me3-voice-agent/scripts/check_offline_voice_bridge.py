"""Saved-audio smoke check of the loopback voice bridge; no device or live API."""

import asyncio
import json
import struct
import wave
from argparse import Namespace
from pathlib import Path

from aiohttp import ClientSession, WSMsgType
from aiohttp.test_utils import TestServer

from me3_voice.configuration import load_backends
from me3_voice.voice_bridge import build_app


async def check():
    root = Path.cwd()
    fixture = root / "var/recordings/wake-vad-rearm.wav"
    assets = root / "var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    keyword = root / "var/cache/wake/hey-swift-keywords.txt"
    if not all(path.is_file() for path in (fixture, keyword, assets / "tokens.txt")):
        raise FileNotFoundError("saved wake fixture or Sherpa assets missing; see README setup")
    wake = Namespace(
        kws_assets=assets,
        keywords_file=keyword,
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        wake_threshold=0.5,
        kws_threshold=0.25,
        kws_boost=1.0,
    )
    backends = load_backends(None)  # Explicitly all mock; never read private config.
    seen = {"run_id": None, "pcm_packets": 0, "final_user_seen": False, "sent_samples": 0}
    async with TestServer(
        build_app(root, backends, {}, wake, trace_dir=root / "var/results"),
        host="127.0.0.1",
    ) as server:
        async with ClientSession() as client:
            async with client.ws_connect(str(server.make_url("/api/voice/ws"))) as ws:
                await ws.send_json({"type": "hello", "rate": 16000, "channels": 1})

                async def receive():
                    async for message in ws:
                        if message.type == WSMsgType.BINARY:
                            seen["pcm_packets"] += 1
                            await ws.send_json(
                                {
                                    "type": "audio_ack",
                                    "id": struct.unpack("<I", message.data[:4])[0],
                                }
                            )
                        elif message.type == WSMsgType.TEXT:
                            value = json.loads(message.data)
                            if value.get("type") == "ready":
                                seen["run_id"] = value["run_id"]
                            elif value.get("type") == "transcript":
                                seen["final_user_seen"] |= any(
                                    row["role"] == "user" and row["status"] == "final"
                                    for row in value["data"]["messages"]
                                )
                            elif value.get("type") == "error":
                                raise RuntimeError(
                                    f"voice bridge: {value.get('reason', 'unknown error')}"
                                )

                receiver = asyncio.create_task(receive())
                try:
                    with wave.open(str(fixture), "rb") as wav:
                        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
                            16000,
                            1,
                            2,
                        ):
                            raise ValueError("wake fixture must be mono PCM16 16 kHz")
                        while pcm := wav.readframes(320):
                            await ws.send_bytes(pcm)
                            seen["sent_samples"] += len(pcm) // 2
                            await asyncio.sleep(0.02)
                            if receiver.done():
                                await receiver
                    await asyncio.sleep(1)
                    await ws.send_json({"type": "stop"})
                    await asyncio.wait_for(receiver, 10)
                finally:
                    if not receiver.done():
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
    run_id = seen["run_id"]
    if not run_id:
        raise AssertionError("voice bridge did not become ready")
    report_path = root / "var/results" / f"voice-{run_id}.json"
    report = json.loads(report_path.read_text())
    turns = [row for row in report["turns"] if row["turn_id"] > 0]
    summary = {
        "mode": "synthetic file / mock ASR, reasoning and TTS / simulated browser ACK",
        "report": str(report_path.relative_to(root)),
        "wake_activations": report["runtime_io"]["frontend"]["activation_count"],
        "completed_turns": sum(row["outcome"] == "completed" for row in turns),
        "interrupted_turns": sum(row["outcome"] == "interrupted" for row in turns),
        "pcm_packets": seen["pcm_packets"],
        "sent_audio_seconds": round(seen["sent_samples"] / 16000, 2),
        "final_user_seen": seen["final_user_seen"],
    }
    print(json.dumps(summary, indent=2))
    if not all(
        (
            summary["wake_activations"] >= 1,
            summary["completed_turns"] >= 1,
            summary["pcm_packets"] >= 1,
            summary["final_user_seen"],
        )
    ):
        raise AssertionError("offline bridge smoke did not complete a wake-to-playback turn")


if __name__ == "__main__":
    asyncio.run(check())
