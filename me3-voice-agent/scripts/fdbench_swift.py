"""Run the pinned AI231 evaluator against Swift's native production WebSocket.

The official fdbench Runner, TrialScript, dataset, labels and scorer remain the
authority. This adapter only bridges its 16 kHz sample-clock rig to
``/api/voice/ws``. Playback is a simulated PCM playout clock, never physical
audio evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_BENCHMARK = "57eaddbc188ce9a19b8fd002b575f703426956f4"
PINNED_STAGE_BENCHMARK = PINNED_BENCHMARK
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "var/cache/fd-benchmark-reference"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "var/results/fdbench-swift"
STAGE_RUNS_ROOT = PROJECT_ROOT / "var/results/fdbench-swift-stage"
BENCHMARK_ROOT = Path(os.environ.get("SWIFT_FDBENCH_ROOT", DEFAULT_BENCHMARK_ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from fdbench import SR  # noqa: E402
from fdbench.rig.base import Recording, RigBase, TrialScript  # noqa: E402


_TRACE_EVENTS = {
    "wake_manual_armed",
    "vad_speech_started",
    "vad_speech_stopped",
    "asr_finish_requested",
    "asr_final",
    "llm_request",
    "llm_provider_request_started",
    "llm_first_token",
    "llm_usage",
    "tts_response_started",
    "tts_start",
    "tts_pcm",
    "tts_progress_started",
    "tts_progress_start",
    "tts_progress_pcm",
    "playback_packet",
    "playback_ack",
    "turn_interrupted",
    "voice_latency",
    "turn_completed",
    "turn_failed",
    "turn_error",
    "asr_error",
    "llm_provider_error",
}
_TRACE_FIELDS = {
    "event",
    "at",
    "sequence",
    "turn_id",
    "utterance_id",
    "request_id",
    "response_id",
    "response_kind",
    "segment_id",
    "bytes",
    "error_code",
    "error_role",
    "error_stage",
    "error_boundary",
    "actual_input_tokens",
    "actual_output_tokens",
    "estimated_input_tokens",
    "prompt_tokens",
    "completion_tokens",
    "vad_speech_end_to_asr_final_ms",
    "llm_provider_request_to_first_token_ms",
    "tts_start_to_first_pcm_ms",
    "first_pcm_to_client_delivery_ms",
    "acknowledgement_audio",
}
_USAGE_ROLES = (
    "gpt_oss_requests",
    "haiku_requests",
    "asr_requests",
    "tts_requests",
    "tavily_requests",
)
_TOKEN_FIELDS = (
    "estimated_input_tokens_reserved",
    "actual_input_tokens",
    "actual_output_tokens",
    "requests_with_actual_usage",
)

_STAGE_NAMES = {
    "vad_speech_stopped": "vad_end",
    "asr_final": "asr_final",
    "llm_provider_request_started": "llm_request",
    "llm_first_token": "llm_first_token",
    "llm_usage": "llm_usage",
    "tts_progress_start": "tts_request",
    "tts_progress_pcm": "tts_first_audio",
    "tts_start": "tts_request",
    "tts_pcm": "tts_first_audio",
    "playback_packet": "audio_out",
}


def _publish_stage_marks(store, events, transcripts, *, src):
    """Use Pi timestamps, with WSL receive times only for audio-clock alignment."""
    finals = sum(row["event"] == "asr_final" for row, _ in events)
    # The bridge emits final timing and committed text on separate messages.
    # A single final and single committed user row in a fresh session can
    # be aligned without guessing. An empty ASR mark would score as a deletion.
    text = next(iter(transcripts.values())) if finals == len(transcripts) == 1 else None
    requests = set()
    for row, received in events:
        name = row["event"]
        mark = {"stage": _STAGE_NAMES[name], "t": row["at"]}
        if name == "asr_final":
            if text is None:
                continue
            mark["text"] = text
        elif name == "llm_provider_request_started":
            if row.get("request_id"):
                requests.add(row["request_id"])
        elif name == "llm_first_token" and (text is None or row.get("request_id") not in requests):
            # The scorer otherwise falls back to request-to-token for missing ASR.
            # Local fast routes emit a synthetic token without a provider request.
            continue
        elif name == "llm_usage" and type(row.get("prompt_tokens")) is int:
            mark["prompt_tokens"] = row["prompt_tokens"]
        store.add(mark, recv=received, src=src)


def encode_pcm16_mono(normalized_pcm: np.ndarray) -> bytes:
    """Convert fdbench float32 [-1, 1] samples to the bridge's signed PCM16 wire format."""
    samples = np.asarray(normalized_pcm, dtype=np.float32)
    return (
        np.clip(np.rint(np.clip(samples, -1.0, 1.0) * 32768.0), -32768, 32767)
        .astype("<i2")
        .tobytes()
    )


async def _get_health(base_url: str) -> dict:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=4, connect=2)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async with client.get(base_url.rstrip("/") + "/api/health") as response:
            response.raise_for_status()
            value = await response.json()
    if not isinstance(value, dict):
        raise RuntimeError("Swift /api/health returned an invalid response")
    return value


def _health(base_url: str) -> dict:
    return asyncio.run(_get_health(base_url))


def _usage_delta(before: dict, after: dict) -> dict:
    left = before.get("budget", {})
    right = after.get("budget", {})
    requests = {}
    for role in _USAGE_ROLES:
        a, b = left.get(role, {}).get("used"), right.get(role, {}).get("used")
        requests[role] = b - a if type(a) is int and type(b) is int and b >= a else None
    left_tokens, right_tokens = left.get("token_usage", {}), right.get("token_usage", {})
    tokens = {}
    for key in _TOKEN_FIELDS:
        a, b = left_tokens.get(key, 0), right_tokens.get(key, 0)
        tokens[key] = b - a if type(a) is int and type(b) is int and b >= a else None
    a = left.get("listed_price_estimate_from_actual_model_tokens_usd")
    b = right.get("listed_price_estimate_from_actual_model_tokens_usd")
    if type(a) in (int, float) and type(b) in (int, float) and b >= a:
        cost = round(b - a, 10)
    elif not any(requests.get(key) for key in ("gpt_oss_requests", "haiku_requests")) and not any(
        (left_tokens.get(key, 0) for key in _TOKEN_FIELDS)
    ):
        cost = b if type(b) in (int, float) else None
    else:
        cost = None
    left_vendor = left.get("vendor_billed_cost_usd")
    right_vendor = right.get("vendor_billed_cost_usd")
    vendor_cost = (
        round(right_vendor - left_vendor, 10)
        if type(left_vendor) in (int, float)
        and type(right_vendor) in (int, float)
        and right_vendor >= left_vendor
        else None
    )
    return {
        "requests": requests,
        "tokens": tokens,
        "openrouter_list_price_estimate_usd": cost,
        "vendor_billed_cost_usd": vendor_cost,
    }


class SwiftWebSocketRig(RigBase):
    """Official fdbench sample-clock rig over the native bridge's client WS API."""

    name = "swift-native-websocket"

    def __init__(
        self, base_url: str, *, chunk_ms: int = 32, connect_timeout: float = 12, marks=None
    ):
        super().__init__(chunk_ms)
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("bridge URL must be an HTTP(S) origin, without a path")
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self._connect_timeout = connect_timeout
        self._worker: threading.Thread | None = None
        self._worker_done = threading.Event()
        self._worker_ready = threading.Event()
        self._worker_error: str | None = None
        self._sess_lock = threading.Lock()
        self._loop = None
        self._commands = None
        self._output_packets: queue.Queue = queue.Queue()
        self._pcm = deque()
        self._native_events: list[dict] = []
        self.marks = marks
        self._stage_events: list[tuple[dict, float]] = []
        self._stage_transcripts: dict[str, str] = {}
        self._transcripts: list[str] = []
        self._assistant_text: list[str] = []
        self._event_lock = threading.Lock()
        self._terminal = threading.Event()
        self._latency = threading.Event()
        self._packet_acknowledgements = 0
        self._output_bytes = 0
        self._input_frames = 0
        self._input_frames_sent = 0
        self._input_bytes_sent = 0
        self._input_samples_sent = 0
        self._input_peak = 0
        self._input_square_sum = 0
        self._input_frames_accepted = 0
        self._input_bytes_accepted = 0
        self._vad_starts = 0
        self._vad_stops = 0
        self._trial_started = 0.0
        self._ready_message: dict = {}

    def open(self) -> None:
        return None

    def close(self) -> None:
        if self._worker and self._worker.is_alive():
            self._send_control({"type": "stop"})
            self._worker.join(timeout=8)

    def link_info(self) -> dict:
        return {
            "rig": self.name,
            "bridge": self.base_url,
            "audio_delivery": "synthetic_pcm_playout_clock",
        }

    def _send(self, value) -> None:
        loop, commands = self._loop, self._commands
        if loop is None or commands is None or self._worker_done.is_set():
            raise RuntimeError("Swift benchmark WebSocket is not connected")
        loop.call_soon_threadsafe(commands.put_nowait, value)

    def _send_control(self, value: dict) -> None:
        self._send(value)

    async def _client(self, active: bool, reset_only: bool = False) -> None:
        import aiohttp

        self._loop = asyncio.get_running_loop()
        self._commands = asyncio.Queue()
        timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_read=None)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as client:
                endpoint = self.base_url.replace("https://", "wss://", 1).replace(
                    "http://", "ws://", 1
                )
                async with client.ws_connect(
                    endpoint + "/api/voice/ws", heartbeat=10, max_msg_size=16384
                ) as ws:
                    await ws.send_json(
                        {
                            "type": "hello",
                            "rate": SR,
                            "channels": 1,
                            "conversation_mode": "continuous",
                            "capture_source": "browser",
                        }
                    )
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        value = json.loads(message.data)
                        if value.get("type") == "ready":
                            self._ready_message = value
                            if reset_only:
                                await ws.send_json({"type": "stop"})
                                return
                            if active:
                                await ws.send_json({"type": "arm"})
                            self._worker_ready.set()
                            break
                        if value.get("type") == "error":
                            raise RuntimeError("Swift rejected the benchmark voice session")
                    else:
                        raise RuntimeError("Swift closed before the voice session became ready")

                    async def sender():
                        while True:
                            item = await self._commands.get()
                            if item is None:
                                return
                            if isinstance(item, bytes):
                                await ws.send_bytes(item)
                                self._record_input_sent(item)
                            else:
                                await ws.send_json(item)
                                if item == {"type": "stop"}:
                                    return

                    async def receiver():
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.BINARY:
                                payload = bytes(message.data)
                                if len(payload) < 6 or (len(payload) - 4) % 2:
                                    self._worker_error = "invalid server PCM packet"
                                    return
                                packet_id = struct.unpack_from("<I", payload)[0]
                                pcm = (
                                    np.frombuffer(payload, dtype="<i2", offset=4).astype(np.float32)
                                    / 32768.0
                                )
                                self._output_packets.put((packet_id, pcm))
                                self._output_bytes += len(payload) - 4
                            elif message.type == aiohttp.WSMsgType.TEXT:
                                self._receive_json(json.loads(message.data))
                            elif message.type in {
                                aiohttp.WSMsgType.ERROR,
                                aiohttp.WSMsgType.CLOSED,
                            }:
                                return

                    sender_task = asyncio.create_task(sender())
                    receiver_task = asyncio.create_task(receiver())
                    done, pending = await asyncio.wait(
                        {sender_task, receiver_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        error = task.exception()
                        if error:
                            raise error
        except Exception as exc:
            self._worker_error = type(exc).__name__
        finally:
            self._worker_done.set()
            self._worker_ready.set()

    def _receive_json(self, value: dict) -> None:
        if value.get("type") == "transcript" and isinstance(value.get("data"), dict):
            for row in value["data"].get("messages", []):
                if not isinstance(row, dict) or row.get("status") != "final":
                    continue
                text = row.get("text")
                if (
                    self.marks is not None
                    and row.get("role") == "user"
                    and row.get("delivery") == "asr_final"
                    and isinstance(row.get("id"), str)
                    and isinstance(text, str)
                    and text.strip()
                ):
                    self._stage_transcripts[row["id"]] = text
                if isinstance(text, str) and text and len(text) <= 1000:
                    target = (
                        self._transcripts if row.get("role") == "user" else self._assistant_text
                    )
                    if text not in target:
                        target.append(text)
        if value.get("type") != "runtime_event" or not isinstance(value.get("data"), dict):
            return
        row = value["data"]
        name = row.get("event")
        if name == "audio_frame_delivered":
            self._input_frames_accepted += 1
            size = row.get("bytes")
            if type(size) is int and size >= 0:
                self._input_bytes_accepted += size
            return
        if name == "vad_speech_started":
            self._vad_starts += 1
        elif name == "vad_speech_stopped":
            self._vad_stops += 1
        if name not in _TRACE_EVENTS:
            return
        with self._event_lock:
            count_same = sum(event.get("event") == name for event in self._native_events)
            # Preserve stage boundaries without logging every streaming PCM or packet.
            if (
                name in {"tts_pcm", "tts_progress_pcm", "playback_packet", "playback_ack"}
                and count_same
            ):
                return
            projected = {key: row[key] for key in _TRACE_FIELDS if key in row}
            self._native_events.append(projected)
            if (
                self.marks is not None
                and name in _STAGE_NAMES
                and type(projected.get("at")) in (int, float)
            ):
                self._stage_events.append((projected, time.monotonic()))
        if name == "voice_latency":
            self._latency.set()
        elif name in {"turn_completed", "turn_failed", "turn_error"}:
            self._terminal.set()

    def _record_input_sent(self, payload: bytes) -> None:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.int32)
        if not samples.size:
            return
        self._input_frames_sent += 1
        self._input_bytes_sent += len(payload)
        self._input_samples_sent += int(samples.size)
        self._input_peak = max(self._input_peak, int(np.abs(samples).max()))
        self._input_square_sum += int(np.square(samples, dtype=np.int64).sum())

    def _start_connection(self, active: bool, *, reset_only: bool = False) -> None:
        self._worker_done.clear()
        self._worker_ready.clear()
        self._worker_error = None
        self._ready_message = {}
        self._native_events = []
        self._stage_events = []
        self._stage_transcripts = {}
        self._transcripts = []
        self._assistant_text = []
        self._output_packets = queue.Queue()
        self._pcm = deque()
        self._packet_acknowledgements = 0
        self._output_bytes = self._input_frames = 0
        self._input_frames_sent = self._input_bytes_sent = self._input_samples_sent = 0
        self._input_peak = self._input_square_sum = 0
        self._input_frames_accepted = self._input_bytes_accepted = 0
        self._vad_starts = self._vad_stops = 0
        self._terminal.clear()
        self._latency.clear()
        self._worker = threading.Thread(
            target=lambda: asyncio.run(self._client(active, reset_only)), daemon=True
        )
        self._worker.start()
        if not self._worker_ready.wait(timeout=self.connect_timeout):
            raise TimeoutError("Swift native voice WebSocket readiness timed out")
        if self._worker_error:
            raise RuntimeError("Swift native voice WebSocket failed: " + self._worker_error)
        self._trial_started = time.monotonic()

    def _reset_session(self) -> str:
        """Use the native stop control to clear retained conversation context before a trial."""
        self._start_connection(active=False, reset_only=True)
        assert self._worker is not None
        self._worker.join(timeout=8)
        if self._worker.is_alive():
            raise TimeoutError("Swift did not finish the benchmark session reset")
        if self._worker_error:
            raise RuntimeError("Swift native session reset failed: " + self._worker_error)
        reset_from = self._ready_message.get("session_id")
        if not isinstance(reset_from, str) or not reset_from:
            raise RuntimeError("Swift native session reset did not identify its session")
        self._wait_for_idle()
        return reset_from

    def _wait_for_idle(self, timeout: float = 5.0) -> dict:
        """Wait briefly for the native stop control's cleanup before reading run usage."""
        deadline = time.monotonic() + timeout
        while True:
            health = _health(self.base_url)
            self._require_health(health, idle=False)
            if health.get("voice_session") == "idle":
                return health
            if time.monotonic() >= deadline:
                raise TimeoutError("Swift did not return to idle after the benchmark session stop")
            time.sleep(0.05)

    def _append_packets(self) -> None:
        while True:
            try:
                packet_id, pcm = self._output_packets.get_nowait()
            except queue.Empty:
                return
            self._pcm.append([packet_id, pcm, 0])

    def _next_output(self, samples: int) -> np.ndarray:
        self._append_packets()
        output = np.zeros(samples, dtype=np.float32)
        offset = 0
        while offset < samples and self._pcm:
            packet_id, pcm, consumed = self._pcm[0]
            take = min(samples - offset, len(pcm) - consumed)
            output[offset : offset + take] = pcm[consumed : consumed + take]
            consumed += take
            offset += take
            if consumed == len(pcm):
                self._pcm.popleft()
                self._packet_acknowledgements += 1
                self._send_control(
                    {
                        "type": "audio_ack",
                        "id": packet_id,
                        "schedule_gap_seconds": 0,
                        "queued_audio_seconds": 0,
                        "audio_context_state": "running",
                        "arrival_ms": (time.monotonic() - self._trial_started) * 1000,
                    }
                )
            else:
                self._pcm[0][2] = consumed
        return output

    def _drain_until(self, event: threading.Event, timeout: float) -> None:
        """Keep the playback ACK clock running after the scored track freezes."""
        deadline = time.monotonic() + timeout
        period = self.chunk / SR
        while not event.is_set() and not self._worker_done.is_set():
            remaining = deadline - time.monotonic()
            if remaining < period:
                if remaining > 0:
                    event.wait(remaining)
                return
            # Consume at the usual playout rate without adding scored audio
            # or sending additional microphone input to the native bridge.
            if not event.wait(period) and not self._worker_done.is_set():
                self._next_output(self.chunk)

    def _require_health(self, health: dict, *, idle: bool) -> None:
        if health.get("bridge") != "ready" or health.get("engine") != "native":
            raise RuntimeError("Swift benchmark target is not a ready native bridge")
        if health.get("voice_ready") is not True or health.get("voice_configured") is not True:
            raise RuntimeError("Swift benchmark target is not locally voice-ready")
        if idle and health.get("voice_session") != "idle":
            raise RuntimeError("Swift bridge already has an active voice session")

    def run(self, script: TrialScript) -> Recording:
        health_before = _health(self.base_url)
        self._require_health(health_before, idle=True)
        reset_from_session = self._reset_session()
        health_before = _health(self.base_url)
        self._require_health(health_before, idle=True)
        active = script.tag != "foreign"
        session = self._Session(self, script)
        self._start_connection(active)
        next_tick = time.monotonic()
        period = self.chunk / SR
        deadline = next_tick + script.max_duration + self._connect_timeout
        with self._sess_lock:
            self._sess = session
        try:
            while not session.done.is_set() and time.monotonic() < deadline:
                pcm = session.next_out(self.chunk)
                self._send(encode_pcm16_mono(pcm))
                session.feed_in(self._next_output(self.chunk))
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:
                    next_tick = time.monotonic()
            if not session.done.is_set():
                raise TimeoutError("official benchmark trial exceeded its declared max_duration")
            scored_stage_events = list(self._stage_events)
            capture_acks = self._packet_acknowledgements
            if active and session.bot_starts:
                self._drain_until(self._terminal, timeout=3)
                self._drain_until(self._latency, timeout=2)
        finally:
            with self._sess_lock:
                self._sess = None
            if self._worker and self._worker.is_alive():
                self._send_control({"type": "stop"})
                self._worker.join(timeout=8)

        if self._worker and self._worker.is_alive():
            raise TimeoutError("Swift did not close the per-trial native session")
        if self._worker_error:
            raise RuntimeError("Swift native voice WebSocket failed: " + self._worker_error)
        rec = session.finish()
        try:
            health_after = self._wait_for_idle()
            before_run = health_before.get("budget", {}).get("run_id")
            after_run = health_after.get("budget", {}).get("run_id")
            usage = (
                _usage_delta(health_before, health_after)
                if before_run == after_run
                else {"error": "bridge_run_changed_during_trial"}
            )
        except Exception as exc:
            health_after = {}
            usage = {"error": type(exc).__name__}
        current_session = self._ready_message.get("session_id")
        reset_verified = isinstance(current_session, str) and current_session != reset_from_session
        if not reset_verified:
            raise RuntimeError("Swift did not create a fresh session after the benchmark reset")
        if self.marks is not None:
            _publish_stage_marks(
                self.marks, scored_stage_events, self._stage_transcripts, src=self.base_url
            )
        rec.notes.update(
            {
                "rig": self.name,
                "bridge_run_id": health_before.get("budget", {}).get("run_id"),
                "native_session_id": current_session,
                "session_reset_verified": reset_verified,
                "reset_from_session_id": reset_from_session,
                "activation": "manual_arm" if active else "wake_standby",
                "transcripts": list(self._transcripts),
                "assistant_confirmed_text": list(self._assistant_text),
                "terminal": next(
                    (
                        row.get("event")
                        for row in reversed(self._native_events)
                        if row.get("event") in {"turn_completed", "turn_failed", "turn_error"}
                    ),
                    "no_terminal_turn_event",
                ),
                "latency": next(
                    (
                        row
                        for row in reversed(self._native_events)
                        if row.get("event") == "voice_latency"
                    ),
                    None,
                ),
                "latency_by_turn": [
                    row for row in self._native_events if row.get("event") == "voice_latency"
                ],
                "provider_usage_delta": usage,
                "provider_usage_scope": "bridge ledger delta across this isolated trial; unrelated concurrent background requests may contribute",
                "output_pcm_bytes": self._output_bytes,
                "output_packets_acknowledged": self._packet_acknowledgements,
                "post_capture_packets_acknowledged": self._packet_acknowledgements - capture_acks,
                "input_ingress": {
                    "frames_sent": self._input_frames_sent,
                    "bytes_sent": self._input_bytes_sent,
                    "frames_accepted_by_native_bridge": self._input_frames_accepted,
                    "bytes_accepted_by_native_bridge": self._input_bytes_accepted,
                    "normalized_peak": round(self._input_peak / 32768.0, 6),
                    "normalized_rms": round(
                        (self._input_square_sum / self._input_samples_sent) ** 0.5 / 32768.0,
                        6,
                    )
                    if self._input_samples_sent
                    else 0.0,
                    "vad_speech_starts": self._vad_starts,
                    "vad_speech_stops": self._vad_stops,
                },
                "audio_delivery": "official rig sample-clock PCM playout and protocol ACK; no physical playback",
                "physical_audio_verified": False,
                "native_trace_events": list(self._native_events),
            }
        )
        return rec


def verify_benchmark_checkout(root: Path, *, stage_only: bool = False) -> None:
    root = root.resolve()
    if not root.is_relative_to(PROJECT_ROOT / "var"):
        raise ValueError("Pinned AI231 checkout must remain under ignored project var/")
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = PINNED_STAGE_BENCHMARK if stage_only else PINNED_BENCHMARK
    if actual != expected:
        raise RuntimeError(f"Expected pinned AI231 benchmark {expected}; found {actual}")
    if not (root / "datasets/manifest.json").is_file():
        raise FileNotFoundError("Pinned AI231 dataset manifest is missing")


def _validate_resume_run(resume_dir: Path, output_root: Path) -> Path:
    run = resume_dir.resolve()
    root = output_root.resolve()
    if (
        not run.is_relative_to(root)
        or not (run / "session.json").is_file()
        or not (run / "queue.json").is_file()
    ):
        raise ValueError("resume is limited to a saved run under this ignored output root")
    return run


def _install_official_runner(
    root: Path, bridge_url: str, output_root: Path, *, stage_only: bool = False
) -> None:
    verify_benchmark_checkout(root, stage_only=stage_only)
    if not output_root.resolve().is_relative_to(PROJECT_ROOT / "var"):
        raise ValueError("Benchmark runs and recordings must stay under ignored project var/")
    import fdbench.cli as cli
    import fdbench.util as util
    from fdbench.runner import Runner
    from fdbench.preflight import run_preflight

    util.RUNS_DIR = output_root.resolve()
    cli.RUNS_DIR = output_root.resolve()
    cli.load_dotenv = lambda: (
        None
    )  # provider credentials belong to the ME3 bridge, not fdbench/.env
    if stage_only:
        from fdbench.marks import MarksStore

        cli._direct_rig = lambda _args, _run_dir=None: SwiftWebSocketRig(
            bridge_url, marks=MarksStore()
        )
    else:
        cli._direct_rig = lambda _args, _run_dir=None: SwiftWebSocketRig(bridge_url)
    original_run_trial = Runner._run_trial

    def run_trial_once(self, *args, **kwargs):
        result = original_run_trial(self, *args, **kwargs)
        if result.get("error"):
            raise RuntimeError("stopping official round after the first failed Swift trial")
        return result

    Runner._run_trial = run_trial_once

    def production_preflight(args, judge=None):
        pf = run_preflight(Path(args.datasets), sim=True, mode="sim", judge=judge)
        for check in pf.checks:
            if check.text.startswith("rig: simulated assistant"):
                check.text = (
                    "rig: Swift native /api/voice/ws; sample-clock PCM only; no physical audio"
                )
        health = _health(bridge_url)
        SwiftWebSocketRig(bridge_url)._require_health(health, idle=True)
        from fdbench.preflight import Check

        pf.checks.append(
            Check(True, "Swift native production bridge ready; provider quotas not probed")
        )
        return pf

    cli.cmd_check = production_preflight


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18770")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument(
        "--stage-only", action="store_true", help="run the official 10 × 3 stage test"
    )
    options, official_args = parser.parse_known_args(argv)
    if official_args and official_args[0] == "run":
        official_args = official_args[1:]
    owned_flags = (
        "--sim",
        "--connect",
        "--where",
        "--assistant-cmd",
        "--project",
        "--student",
        "--datasets",
        "--recordings",
        "--port",
        "--reset-cmd",
    )
    if any(
        flag == item or flag.startswith(item + "=")
        for flag in owned_flags
        for item in official_args
    ):
        parser.error("Swift integration owns the transport, dataset, and per-trial session reset")
    root = options.benchmark_root.resolve()
    if root != BENCHMARK_ROOT.resolve():
        parser.error("Swift integration uses only the already-pinned benchmark checkout")
    for index, value in enumerate(official_args):
        if (
            value == "--rounds"
            and index + 1 < len(official_args)
            and official_args[index + 1] != "1"
        ):
            parser.error("one bounded official batch is limited to the harness's first round")
        if value.startswith("--rounds=") and value.split("=", 1)[1] != "1":
            parser.error("one bounded official batch is limited to the harness's first round")
    if options.stage_only:
        if official_args:
            parser.error("Stage-only mode fixes the official plan at 10 phrases × 3 runs, seed 1")
        official_args = [
            "--rounds",
            "0",
            "--phrases",
            "10",
            "--runs",
            "3",
            "--seed",
            "1",
            "--judge-provider",
            "none",
            "--marks-port",
            "0",
        ]
    output_root = options.runs_root or (
        STAGE_RUNS_ROOT if options.stage_only else DEFAULT_RUNS_ROOT
    )
    for index, value in enumerate(official_args):
        if value == "--resume" and index + 1 < len(official_args):
            try:
                _validate_resume_run(Path(official_args[index + 1]), output_root)
            except ValueError:
                parser.error("resume is limited to a saved run under this ignored output root")
        elif value.startswith("--resume="):
            try:
                _validate_resume_run(Path(value.split("=", 1)[1]), output_root)
            except ValueError:
                parser.error("resume is limited to a saved run under this ignored output root")
    _install_official_runner(root, options.bridge_url, output_root, stage_only=options.stage_only)
    sys.argv = [
        "fdbench",
        "run",
        "--connect",
        "direct",
        "--where",
        "local",
        "--datasets",
        str(root / "datasets"),
        "--student",
        "SwiftME3",
        "--yes",
        "--recordings",
        "r",
        *official_args,
    ]
    import fdbench.cli as cli

    return int(cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
