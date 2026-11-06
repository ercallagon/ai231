"""Opt-in bounded PCM evidence; never opens a capture device or calls a provider."""

import hashlib
import json
import os
import time
import wave
from pathlib import Path


class DiagnosticCapture:
    def __init__(self, root, run_id, rate, *, max_seconds=60, clock=time.monotonic):
        if rate not in (16000, 44100, 48000) or not 1 <= max_seconds <= 180:
            raise ValueError("invalid diagnostic capture bounds")
        if not run_id.isalnum() or len(run_id) > 80:
            raise ValueError("invalid capture identity")
        self.path = Path(root) / "var/recordings/diagnostics" / run_id
        base = Path(root).resolve() / "var/recordings/diagnostics"
        if self.path.resolve() != base / run_id or self.path.exists():
            raise ValueError("diagnostic capture needs a fresh project-local path")
        self.rate, self.max_seconds, self.clock = rate, max_seconds, clock
        self.origin = clock()
        self.raw, self.turns = bytearray(), []
        self.received_bytes = self.asr_bytes = 0
        self.closed = False
        self.run_id = run_id
        self.omitted_turns = 0
        self.last_received_monotonic = None
        self.raw_packets = []
        self.omitted_packets = 0

    def received(self, pcm):
        self.last_received_monotonic = self.clock()
        self.received_bytes += len(pcm)
        remaining = max(0, self.rate * 2 * self.max_seconds - len(self.raw))
        if remaining and len(self.raw_packets) < 4096:
            self.raw_packets.append(
                {
                    "offset_bytes": len(self.raw),
                    "saved_bytes": min(len(pcm), remaining),
                    "received_monotonic": self.last_received_monotonic,
                }
            )
        elif remaining:
            self.omitted_packets += 1
        self.raw.extend(pcm[:remaining])

    def wrap(self, speech):
        capture = self

        class CapturedSpeech:
            def synthesize(self, *args, **kwargs):
                return speech.synthesize(*args, **kwargs)

            async def transcribe_stream(self, key, chunks):
                if len(capture.turns) >= 128:
                    capture.omitted_turns += 1
                    stream = speech.transcribe_stream(key, chunks)
                    try:
                        async for result in stream:
                            yield result
                    finally:
                        await stream.aclose()
                    return
                row = {
                    **key.model_dump(),
                    "started_monotonic": capture.clock(),
                    "pcm": bytearray(),
                    "submitted_bytes": 0,
                    "results": [],
                    "outcome": "cancelled_or_abandoned",
                    "received_bytes_at_open": capture.received_bytes,
                }
                capture.turns.append(row)

                async def measured():
                    async for pcm in chunks:
                        row["submitted_bytes"] += len(pcm)
                        remaining = max(0, 32000 * capture.max_seconds - capture.asr_bytes)
                        saved = pcm[:remaining]
                        row["pcm"].extend(saved)
                        capture.asr_bytes += len(saved)
                        yield pcm

                stream = speech.transcribe_stream(key, measured())
                try:
                    async for result in stream:
                        if len(row["results"]) < 100:
                            row["results"].append(
                                {
                                    "text": result.text,
                                    "provider_text": getattr(result, "provider_text", None),
                                    "final": result.final,
                                    "monotonic": capture.clock(),
                                }
                            )
                        yield result
                    row["outcome"] = "completed"
                except Exception as exc:
                    row["outcome"] = "error"
                    row["error_type"] = type(exc).__name__
                    raise
                finally:
                    await stream.aclose()
                    row["finished_monotonic"] = capture.clock()

        return CapturedSpeech()

    def save(self):
        if self.closed:
            return
        self.closed = True
        self.path.mkdir(parents=True, mode=0o700, exist_ok=False)

        def audio(name, pcm, rate):
            destination = self.path / name
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as file:
                with wave.open(file, "wb") as writer:
                    writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                    writer.writeframes(pcm)
            return {
                "file": name,
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "saved_bytes": len(pcm),
            }

        raw = audio("browser-received.wav", self.raw, self.rate)
        turns = []
        for index, row in enumerate(self.turns):
            pcm = row["pcm"]
            turns.append(
                {
                    **{k: v for k, v in row.items() if k != "pcm"},
                    **audio(f"asr-{index + 1}.wav", pcm, 16000),
                    "truncated": len(pcm) != row["submitted_bytes"],
                }
            )
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "origin_monotonic": self.origin,
            "last_received_monotonic": self.last_received_monotonic,
            "omitted_turns": self.omitted_turns,
            "raw_packets": self.raw_packets,
            "omitted_packet_timestamps": self.omitted_packets,
            "max_seconds": self.max_seconds,
            "raw": {
                **raw,
                "received_bytes": self.received_bytes,
                "truncated": len(self.raw) != self.received_bytes,
            },
            "turns": turns,
            "limitations": "Browser-received PCM, not pre-browser/DSP microphone audio. Server receive/ASR clocks are not acoustic sample alignment. No playback reference; no AEC or audible-latency qualification. Capture stops retaining bytes at its bounds; truncated files are not complete reproductions.",
        }
        fd = os.open(self.path / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(manifest, file, allow_nan=False)
            file.write("\n")
        self.raw.clear()
        for row in self.turns:
            row["pcm"].clear()
