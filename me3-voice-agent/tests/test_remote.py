"""Offline contract peers, not live provider/quality or acoustic verification."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sqlite3
import time

import pytest

from me3_voice.configuration import BackendSpec, ConfigurationError
from me3_voice.remote import (
    ChatReasoning,
    DeepgramASR,
    DeepgramASRStream,
    DeepgramTTS,
    LiveTransport,
    ProviderError,
    request_ledger_summary,
)
from me3_voice.providers import MockLLM, MockTTS
from me3_voice.playback import MemorySink
from me3_voice.session import Session
from me3_voice.servers import SpeechService
from me3_voice.contracts import TurnKey, encode_pcm


async def test_asr_pending_connection_flush_timeout_preserves_failure_stage():
    from types import SimpleNamespace

    stream = DeepgramASRStream(
        SimpleNamespace(spec=SimpleNamespace(timeout_seconds=0.01, max_audio_seconds=30))
    )

    async def started():
        return None

    stream.start = started
    stream.outbound.put_nowait(b"\1\0")
    stream.outbound_bytes = 2
    with pytest.raises(ProviderError) as caught:
        await stream._flush_outbound()
    assert (caught.value.code, caught.value.error_stage) == ("provider_timeout", "asr_open")
    stream.connection_ready_at = time.monotonic()
    with pytest.raises(ProviderError) as caught:
        await stream._flush_outbound()
    assert (caught.value.code, caught.value.error_stage) == ("provider_timeout", "asr_flush")


@pytest.fixture(autouse=True)
def isolated_request_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(tmp_path / "provider-requests.sqlite"))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "pytest-run-0001")


def spec(role):
    return BackendSpec(
        backend="external",
        protocol={
            "asr": "deepgram-listen-v1",
            "tts": "deepgram-speak-v1",
            "llm": "chat-completions-v1",
        }[role],
        endpoint="https://example.invalid/chat/completions"
        if role == "llm"
        else "wss://example.invalid/" + role,
        auth_env="ME3_TEST_ONLY_KEY",
        model="documented-test-model",
        # Transport tests exercise timeouts/tool lifecycle, not campaign
        # exhaustion. Dedicated ledger tests below use small explicit caps.
        max_input_tokens=32000 if role == "llm" else 0,
        max_input_tokens_total=64000 if role == "llm" else 0,
    )


def _ledger_transport(path, limit=1):
    import os

    os.environ["ME3_REQUEST_LEDGER"] = str(path)
    return LiveTransport(
        replace(spec("llm"), allow_live=True, request_limit=limit),
        {"ME3_TEST_ONLY_KEY": "offline-key"},
    )


def authorize_chat(transport, *, estimate=1, per_request=100, total=500):
    return transport.authorize(
        input_tokens=estimate,
        input_token_limit=per_request,
        input_tokens_total_limit=total,
    )


def test_durable_request_reservation_and_outcome_survive_restart(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transport = _ledger_transport(path)
    _, request_id = authorize_chat(transport)
    transport.finish_request(request_id, "success")
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT provider, model, status, error_code FROM provider_requests"
        ).fetchone()
    assert row == ("example.invalid", "documented-test-model", "success", None)
    restarted = _ledger_transport(path)
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        authorize_chat(restarted)


def test_concurrent_durable_reservations_enforce_single_slot(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transports = [_ledger_transport(path) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda t: _reserve_result(t), transports))
    assert sorted(results) == ["rejected", "reserved"]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM provider_requests").fetchone()[0] == 1


def test_concurrent_input_token_reservations_are_atomic(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transports = [_ledger_transport(path, limit=5) for _ in range(2)]

    def reserve(transport):
        try:
            transport.authorize(
                input_tokens=400,
                input_token_limit=500,
                input_tokens_total_limit=500,
            )
            return "reserved"
        except ProviderError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, transports))
    assert sorted(results) == ["input_token_budget_exhausted", "reserved"]
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT COUNT(*),SUM(input_tokens_reserved) FROM provider_requests"
        ).fetchone() == (1, 400)


def test_unknown_outcome_keeps_aggregate_input_tokens_consumed_after_restart(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transport = _ledger_transport(path, limit=5)
    _, request_id = authorize_chat(transport, estimate=300, per_request=400, total=500)
    # Leave the dispatched row unknown, as if the process died before a
    # terminal update. A restart must retain the input-token reservation.
    restarted = _ledger_transport(path, limit=5)
    with pytest.raises(ProviderError) as exhausted:
        authorize_chat(restarted, estimate=201, per_request=400, total=500)
    assert exhausted.value.code == "input_token_budget_exhausted"
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT status,input_tokens_reserved FROM provider_requests WHERE id=?", (request_id,)
        ).fetchone() == ("dispatched_unknown", 300)


def _reserve_result(transport):
    try:
        authorize_chat(transport)
        return "reserved"
    except ProviderError:
        return "rejected"


def test_dispatched_unknown_crash_row_remains_consumed(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transport = _ledger_transport(path)
    _, request_id = authorize_chat(transport)
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT status FROM provider_requests WHERE id=?", (request_id,)
        ).fetchone() == ("dispatched_unknown",)
    with pytest.raises(ProviderError):
        authorize_chat(_ledger_transport(path))


@pytest.mark.parametrize(
    "legacy_columns",
    [
        None,
        "id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, "
        "protocol TEXT NOT NULL, status TEXT NOT NULL, error_code TEXT, "
        "created REAL NOT NULL, updated REAL NOT NULL",
        "id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, "
        "status TEXT NOT NULL, created REAL NOT NULL",
    ],
    ids=["fresh", "phase7-run-id-appended", "older-legacy"],
)
def test_authorize_uses_named_columns_for_fresh_and_migrated_ledgers(
    tmp_path, monkeypatch, legacy_columns
):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    if legacy_columns:
        with sqlite3.connect(path) as db:
            db.execute(f"CREATE TABLE provider_requests ({legacy_columns})")
            if "protocol" in legacy_columns:
                db.execute(
                    "INSERT INTO provider_requests "
                    "(id,provider,model,protocol,status,error_code,created,updated) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        "historical",
                        "old.example",
                        "old-model",
                        "old-protocol",
                        "success",
                        None,
                        1.0,
                        1.0,
                    ),
                )
            else:
                db.execute(
                    "INSERT INTO provider_requests (id,provider,model,status,created) "
                    "VALUES(?,?,?,?,?)",
                    ("historical", "old.example", "old-model", "success", 1.0),
                )

    transport = _ledger_transport(path)
    _, request_id = authorize_chat(transport)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(provider_requests)")}
        assert {
            "id",
            "run_id",
            "provider",
            "model",
            "protocol",
            "status",
            "error_code",
            "created",
            "updated",
        } <= columns
        reserved = db.execute(
            "SELECT run_id,provider,model,protocol,status,error_code,created,updated "
            "FROM provider_requests WHERE id=?",
            (request_id,),
        ).fetchone()
        assert reserved[:6] == (
            "pytest-run-0001",
            "example.invalid",
            "documented-test-model",
            "chat-completions-v1",
            "dispatched_unknown",
            None,
        )
        assert reserved[6] > 0 and reserved[7] == reserved[6]
        expected_rows = 2 if legacy_columns else 1
        assert db.execute("SELECT COUNT(*) FROM provider_requests").fetchone()[0] == expected_rows
        if legacy_columns:
            assert db.execute(
                "SELECT status FROM provider_requests WHERE id='historical'"
            ).fetchone() == ("success",)

    # A reservation consumes the run/provider/model/protocol allowance even
    # before completion, and another transport cannot authorize a duplicate.
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        authorize_chat(_ledger_transport(path))


def test_terminal_and_unknown_reservations_remain_consumed_on_migrated_ledger(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE provider_requests (id TEXT PRIMARY KEY, provider TEXT NOT NULL, "
            "model TEXT NOT NULL, protocol TEXT NOT NULL, status TEXT NOT NULL, "
            "error_code TEXT, created REAL NOT NULL, updated REAL NOT NULL)"
        )
        db.execute(
            "INSERT INTO provider_requests "
            "(id,provider,model,protocol,status,error_code,created,updated) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                "legacy-success",
                "example.invalid",
                "documented-test-model",
                "chat-completions-v1",
                "success",
                None,
                1.0,
                1.0,
            ),
        )
    transport = _ledger_transport(path, limit=3)
    _, failed_id = authorize_chat(transport)
    transport.finish_request(failed_id, "failure", "provider_timeout")
    _, unknown_id = authorize_chat(transport)
    # A crash after dispatch leaves the reservation consumed as unknown.
    restarted = _ledger_transport(path, limit=3)
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT status,error_code FROM provider_requests WHERE id=?", (failed_id,)
        ).fetchone() == ("failure", "provider_timeout")
        assert db.execute(
            "SELECT status,error_code FROM provider_requests WHERE id=?", (unknown_id,)
        ).fetchone() == ("dispatched_unknown", None)
    _, success_id = authorize_chat(restarted)
    restarted.finish_request(success_id, "success")
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT status FROM provider_requests WHERE id=?", (success_id,)
        ).fetchone() == ("success",)
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        authorize_chat(_ledger_transport(path, limit=3))


async def test_asr_append_uses_real_ledger_authorization_with_mocked_provider_socket(
    tmp_path, monkeypatch
):
    """Exercise SpeechService ASR append through LiveTransport's real SQLite SQL."""
    from me3_voice.remote import DeepgramASR
    from me3_voice.servers import SpeechService
    from me3_voice.providers import MockTTS
    from aiohttp import WSMsgType
    from aiohttp.http_websocket import WSMessage

    path = tmp_path / "migrated.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "asr-mcp-test-01")
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE provider_requests (id TEXT PRIMARY KEY, provider TEXT NOT NULL, "
            "model TEXT NOT NULL, protocol TEXT NOT NULL, status TEXT NOT NULL, "
            "error_code TEXT, created REAL NOT NULL, updated REAL NOT NULL)"
        )

    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})

    class MockPeer:
        def __init__(self):
            self.messages = asyncio.Queue()

        async def send_bytes(self, _pcm):
            pass

        async def send_json(self, message):
            if message.get("type") == "CloseStream":
                await self.messages.put(
                    WSMessage(
                        WSMsgType.TEXT,
                        json.dumps(
                            {
                                "type": "Results",
                                "is_final": True,
                                "channel": {
                                    "alternatives": [{"transcript": "mocked provider transcript"}]
                                },
                            }
                        ),
                        None,
                    )
                )
                await self.messages.put(
                    WSMessage(
                        WSMsgType.TEXT,
                        json.dumps({"type": "Metadata"}),
                        None,
                    )
                )

        async def receive(self):
            return await self.messages.get()

    peer = MockPeer()

    class SocketContext:
        async def __aenter__(self):
            return peer

        async def __aexit__(self, *_exc):
            return False

    class MockSession:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: MockSession())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    try:
        opened = await service.open("session-ledger-01", 1)
        appended = await service.append(
            opened["operation_id"], 0, base64.b64encode(b"\x01\x00" * 512).decode("ascii")
        )
        assert appended["next_sequence"] == 1
        final = await service.finish(opened["operation_id"])
        assert final["text"] == "mocked provider transcript"
    finally:
        await service.close()

    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT run_id,provider,model,protocol,status,error_code,created,updated "
            "FROM provider_requests"
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row[:6] == (
        "asr-mcp-test-01",
        "example.invalid",
        "documented-test-model",
        "deepgram-listen-v1",
        "success",
        None,
    )
    assert row[6] > 0 and row[7] >= row[6]
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        transport.authorize()


async def test_labeled_nine_second_wav_uses_production_deepgram_asr_service_with_delayed_socket(
    tmp_path, monkeypatch
):
    import wave
    from aiohttp import WSMsgType
    from aiohttp.http_websocket import WSMessage

    from me3_voice.contracts import FRAME_BYTES

    recording = Path("var/results/native-live-smoke/boiling-question-16k.wav")
    assert recording.is_file()
    with wave.open(str(recording), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        assert wav.getsampwidth() == 2 and len(pcm) == 9 * 16000 * 2

    path = tmp_path / "deepgram-service.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "labeled-asr-service-01")
    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1, timeout_seconds=20)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})

    class Peer:
        def __init__(self):
            self.messages = asyncio.Queue()
            self.writes = []

        async def send_bytes(self, value):
            index = len(self.writes)
            await asyncio.sleep({7: 0.18, 31: 0.24, 68: 0.16}.get(index, 0.001))
            self.writes.append(value)

        async def send_json(self, value):
            if value.get("type") == "CloseStream":
                await asyncio.sleep(0.025)  # Delayed provider finalization response.
                await self.messages.put(
                    WSMessage(
                        WSMsgType.TEXT,
                        json.dumps(
                            {
                                "type": "Results",
                                "is_final": True,
                                "channel": {
                                    "alternatives": [
                                        {"transcript": "What is the boiling point of water?"}
                                    ]
                                },
                            }
                        ),
                        None,
                    )
                )
                await self.messages.put(
                    WSMessage(WSMsgType.TEXT, json.dumps({"type": "Metadata"}), None)
                )

        async def receive(self):
            return await self.messages.get()

    peer = Peer()

    class SocketContext:
        async def __aenter__(self):
            return peer

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    opened = await service.open("labeled-asr-service", 1)
    sequence = 0
    for offset in range(0, len(pcm), FRAME_BYTES):
        frame = pcm[offset : offset + FRAME_BYTES]
        if len(frame) % 2:
            frame = frame[:-1]
        await service.append(opened["operation_id"], sequence, encode_pcm(frame))
        sequence += 1
        # Preserve live replay cadence. Sending nine seconds of audio as a
        # burst would correctly exceed the two-second outbound bound.
        await asyncio.sleep(0.032)
    final = await service.finish(opened["operation_id"])
    assert final["final"] and final["text"] == "What is the boiling point of water?"
    assert b"".join(peer.writes) == pcm
    assert len(peer.writes) == sequence
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT status,error_code FROM provider_requests").fetchall()
    assert rows == [("success", None)]
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        transport.authorize()
    await service.close()


@pytest.mark.parametrize("connect_delay", [0.5, 0.9, 1.2, 1.5, 2.5, 4.5])
async def test_deepgram_connection_starts_on_open_and_buffers_audio_once_in_order(
    tmp_path, monkeypatch, connect_delay
):
    from aiohttp import WSMsgType
    from aiohttp.http_websocket import WSMessage

    path = tmp_path / f"connect-{int(connect_delay * 1000)}.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", f"connect-{int(connect_delay * 1000)}")
    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})

    class Peer:
        def __init__(self):
            self.messages = asyncio.Queue()
            self.writes = []

        async def send_bytes(self, pcm):
            self.writes.append(pcm)

        async def send_json(self, value):
            self.writes.append(value)
            if value.get("type") == "CloseStream":
                await self.messages.put(
                    WSMessage(
                        WSMsgType.TEXT,
                        json.dumps(
                            {
                                "type": "Results",
                                "is_final": True,
                                "channel": {"alternatives": [{"transcript": "recognized"}]},
                            }
                        ),
                        None,
                    )
                )
                await self.messages.put(
                    WSMessage(WSMsgType.TEXT, json.dumps({"type": "Metadata"}), None)
                )

        async def receive(self):
            return await self.messages.get()

    peer = Peer()

    class SocketContext:
        async def __aenter__(self):
            await asyncio.sleep(connect_delay)
            return peer

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    opened_at = asyncio.get_running_loop().time()
    opened = await service.open("startup", 1)
    assert asyncio.get_running_loop().time() - opened_at < 0.05
    stream = service.operations[opened["operation_id"]].asr_stream
    await asyncio.sleep(0)
    assert stream.connect_task is not None and not stream.connect_task.done()
    assert stream.connection_started_at is not None

    # Pace through readiness even for the 2.5 s delay, then check final flush.
    frame_count = max(50, int(connect_delay / 0.032) + 6)
    frames = [bytes([i % 255 + 1, 0]) * 512 for i in range(frame_count)]
    first_append_seconds = None
    first_diagnostics = None
    for sequence, frame in enumerate(frames):
        append_started = asyncio.get_running_loop().time()
        row = await service.append(opened["operation_id"], sequence, encode_pcm(frame))
        if sequence == 0:
            first_append_seconds = asyncio.get_running_loop().time() - append_started
            first_diagnostics = row["provider_diagnostics"]
        assert row["next_sequence"] == sequence + 1
        if sequence < len(frames) - 2:
            await asyncio.sleep(0.032)
    assert first_append_seconds < 0.05
    assert first_diagnostics["connection_state"] == "connecting"
    assert first_diagnostics["first_append_monotonic"] is not None
    assert first_diagnostics["accepted_audio_bytes"] == len(frames[0])
    assert stream.peak_outbound_bytes <= stream.OUTBOUND_BUFFER_BYTES
    assert stream.peak_outbound_bytes > 0
    assert stream.connection_ready_at is not None
    ready_delay = stream.connection_ready_at - stream.connection_started_at
    # Includes local process scheduling and SQLite reservation time around the
    # controlled handshake; the fake socket delay itself is exact.
    assert connect_delay <= ready_delay <= connect_delay + 0.3
    assert stream.first_send_started_at >= stream.connection_ready_at
    (tmp_path / "connection-timing.json").write_text(
        json.dumps(
            {
                "connection_started_monotonic": stream.connection_started_at,
                "connection_ready_monotonic": stream.connection_ready_at,
                "connection_ready_delay_ms": round(ready_delay * 1000, 3),
                "first_append_wait_ms": round(first_append_seconds * 1000, 3),
                "accepted_bytes": stream.accepted_bytes,
                "sent_bytes": stream.sent_bytes,
                "peak_queued_bytes": stream.peak_outbound_bytes,
                "queue_limit_bytes": stream.OUTBOUND_BUFFER_BYTES,
            },
            sort_keys=True,
        )
        + "\n"
    )
    final = await service.finish(opened["operation_id"])
    assert final["text"] == "recognized"
    assert b"".join(value for value in peer.writes if isinstance(value, bytes)) == b"".join(frames)
    assert stream.accepted_bytes == stream.sent_bytes == sum(map(len, frames))
    assert stream.outbound_bytes == 0
    final_diag = final["provider_diagnostics"]
    assert final_diag["connection_state"] == "closed"
    assert (
        final_diag["connection_attempt_monotonic"]
        <= final_diag["websocket_handshake_monotonic"]
        <= final_diag["connection_ready_monotonic"]
        <= final_diag["first_send_completed_monotonic"]
        <= final_diag["connection_closed_monotonic"]
    )
    assert final_diag["accepted_audio_bytes"] == final_diag["sent_audio_bytes"]
    assert final_diag["outbound_flush_completed_monotonic"] is not None
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,error_code FROM provider_requests").fetchall() == [
            ("success", None)
        ]
    await service.close()


async def test_deepgram_connection_timeout_is_explicit_and_consumes_one_slot(tmp_path, monkeypatch):
    path = tmp_path / "connect-timeout.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "connect-timeout")
    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})

    class SocketContext:
        async def __aenter__(self):
            await asyncio.sleep(0.01)
            raise ProviderError(
                "speech connection timed out; payload excluded",
                code="provider_timeout",
                role="asr",
                error_stage="asr_open",
                error_boundary="deepgram_websocket",
                error_detail="TimeoutError",
            )

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    opened = await service.open("startup-timeout", 1)
    stream = service.operations[opened["operation_id"]].asr_stream
    appended = await service.append(opened["operation_id"], 0, encode_pcm(b"\x01\x00" * 512))
    assert appended["next_sequence"] == 1
    with pytest.raises(ProviderError) as caught:
        await service.finish(opened["operation_id"])
    assert caught.value.code == "provider_timeout"
    assert caught.value.error_stage == "asr_open"
    assert caught.value.error_boundary == "deepgram_websocket"
    assert stream.connection_started_at is not None and stream.connection_failed_at is not None
    (tmp_path / "connection-failure-timing.json").write_text(
        json.dumps(
            {
                "connection_started_monotonic": stream.connection_started_at,
                "connection_failed_monotonic": stream.connection_failed_at,
                "failure_elapsed_ms": round(
                    (stream.connection_failed_at - stream.connection_started_at) * 1000, 3
                ),
                "error_code": caught.value.code,
                "error_stage": caught.value.error_stage,
            },
            sort_keys=True,
        )
        + "\n"
    )
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,error_code FROM provider_requests").fetchall() == [
            ("failure", "provider_timeout")
        ]
    await service.close()


async def test_deepgram_startup_buffer_is_bounded_and_rejects_without_silent_loss(
    tmp_path, monkeypatch
):
    from me3_voice.servers import speech_tool

    path = tmp_path / "connection-buffer.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "connection-buffer")
    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})

    class SocketContext:
        async def __aenter__(self):
            await asyncio.sleep(0.5)
            return Peer()

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    opened = await service.open("connection-buffer", 1)
    identifier = opened["operation_id"]
    stream = service.operations[identifier].asr_stream
    # Shrink only this isolated test's buffer to hit the same fail-closed path
    # without allocating a long speech fixture.
    stream.OUTBOUND_BUFFER_BYTES = 2048
    first = await service.append(identifier, 0, encode_pcm(b"\1\0" * 512))
    second = await service.append(identifier, 1, encode_pcm(b"\2\0" * 512))
    assert first["next_sequence"] == 1 and second["next_sequence"] == 2
    failed = await speech_tool(service.append, "asr")(identifier, 2, encode_pcm(b"\3\0" * 512))
    assert failed["speech_error"]["code"] == "asr_buffer_full"
    assert failed["speech_error"]["stage"] == "asr_append"
    assert stream.accepted_bytes == 2048
    assert stream.peak_outbound_bytes == 2048
    assert (
        stream.accepted_bytes == stream.sent_bytes + stream.discarded_bytes + stream.outbound_bytes
    )
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,error_code FROM provider_requests").fetchall() == [
            ("unknown", None)
        ]
    await service.close()


async def test_deepgram_stream_honors_configured_audio_duration_limit():
    spec_asr = replace(spec("asr"), max_audio_seconds=1)
    spec_asr.validate("asr", runnable=True)
    stream = DeepgramASR(spec_asr, transport=None).open_stream()
    stream.total = 16000 * 2 - 2
    with pytest.raises(ProviderError, match="invalid ASR stream input"):
        await stream.append(b"\0\0\0\0")
    assert stream.total == 16000 * 2 - 2
    assert stream.accepted_bytes == 0
    assert stream.diagnostics()["audio_limit_bytes"] == 32000


async def test_deepgram_default_startup_buffer_fails_closed_at_five_seconds(tmp_path, monkeypatch):
    path = tmp_path / "five-second-buffer.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "five-second-buffer")
    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})
    entered = asyncio.Event()

    class SocketContext:
        async def __aenter__(self):
            entered.set()
            await asyncio.Event().wait()

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    stream = DeepgramASR(spec_asr, transport).open_stream()
    frame = b"\1\0" * 512
    for _ in range(156):
        await stream.append(frame)
    await asyncio.sleep(0)
    await entered.wait()
    await stream.append(b"\1\0" * 128)
    with pytest.raises(ProviderError) as caught:
        await stream.append(b"\0\0")
    assert caught.value.code == "asr_buffer_full"
    assert stream.accepted_bytes == stream.peak_outbound_bytes == 160000
    assert stream.sent_bytes == stream.discarded_bytes == 0
    assert stream.outbound.qsize() <= stream.OUTBOUND_QUEUE_ITEMS
    await stream.aclose()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM provider_requests").fetchone() == ("unknown",)


async def test_deepgram_pending_connection_cancellation_is_joined_and_remains_unknown(
    tmp_path, monkeypatch
):
    path = tmp_path / "connect-cancel.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", "connect-cancel")
    spec_asr = replace(spec("asr"), allow_live=True, request_limit=1)
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})
    entered = asyncio.Event()

    class SocketContext:
        async def __aenter__(self):
            entered.set()
            await asyncio.Event().wait()

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    opened = await service.open("startup-cancel", 1)
    stream = service.operations[opened["operation_id"]].asr_stream
    first_task = stream.start_background()
    assert stream.start_background() is first_task
    await entered.wait()
    result = await service.cancel(opened["operation_id"])
    assert result["cancelled"] is True
    assert result["provider_diagnostics"]["connection_state"] == "closed"
    assert result["provider_diagnostics"]["connection_closed_monotonic"] is not None
    assert (
        result["provider_diagnostics"]["connection_stages"]["connection_cancelled_monotonic"]
        is not None
    )
    assert result["provider_diagnostics"]["connection_stages"]["request_outcome"] == "unknown"
    assert first_task.done() and first_task.cancelled()
    assert stream.connection_failed_at is None
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status,error_code FROM provider_requests").fetchall() == [
            ("unknown", None)
        ]
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        transport.authorize()
    await service.close()


async def test_deepgram_aiohttp_trace_records_safe_connection_stages():
    transport = LiveTransport(spec("asr"))
    session = transport.session()
    config = session._trace_configs[0]
    assert len(config.on_dns_resolvehost_start) == 1
    assert len(config.on_dns_resolvehost_end) == 1
    assert len(config.on_connection_create_start) == 1
    assert len(config.on_connection_create_end) == 1
    assert len(config.on_request_headers_sent) == 1
    assert len(config.on_request_end) == 1
    assert not config.on_request_headers_sent[0].__closure__ or all(
        "Authorization" not in str(cell.cell_contents)
        for cell in config.on_request_headers_sent[0].__closure__
    )
    await session.close()


@pytest.mark.parametrize(
    "failure_mode", ["incomplete_close", "receive_timeout", "provider_disconnect"]
)
async def test_deepgram_asr_terminal_ledger_state_and_mcp_error_diagnostics(
    tmp_path, monkeypatch, failure_mode
):
    from types import SimpleNamespace
    from me3_voice.servers import speech_tool

    path = tmp_path / f"{failure_mode}.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    monkeypatch.setenv("ME3_REQUEST_RUN_ID", f"asr-{failure_mode}-01")
    spec_asr = replace(
        spec("asr"),
        allow_live=True,
        request_limit=1,
        timeout_seconds=0.03 if failure_mode == "receive_timeout" else 2,
    )
    transport = LiveTransport(spec_asr, {"ME3_TEST_ONLY_KEY": "offline-key"})

    class Peer:
        def __init__(self):
            self.sent = []
            self.receiving = asyncio.Event()
            self.messages = asyncio.Queue()

        async def send_bytes(self, pcm):
            await asyncio.sleep(0.01)  # Lowest-boundary write delay.
            self.sent.append(pcm)

        async def send_json(self, value):
            self.sent.append(value)

        async def receive(self):
            self.receiving.set()
            if failure_mode == "receive_timeout":
                await asyncio.Event().wait()
            if failure_mode == "provider_disconnect":
                from aiohttp import WSMsgType

                await asyncio.sleep(0.02)
                return SimpleNamespace(type=WSMsgType.CLOSED, data=None)
            return await self.messages.get()

    peer = Peer()

    class SocketContext:
        async def __aenter__(self):
            return peer

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def ws_connect(self, *_args, **_kwargs):
            return SocketContext()

        async def close(self):
            pass

    monkeypatch.setattr(transport, "session", lambda: Session())
    service = SpeechService(DeepgramASR(spec_asr, transport), MockTTS())
    opened = await service.open("asr-diagnostic", 1)
    identifier = opened["operation_id"]
    first = await service.append(identifier, 0, encode_pcm(b"\1\0" * 512))
    assert first["next_sequence"] == 1
    await peer.receiving.wait()

    if failure_mode == "incomplete_close":
        await service.cancel(identifier)
        expected = ("unknown", "request_interrupted")
    else:
        await asyncio.sleep(0.05)
        error_result = await speech_tool(service.append, "asr")(
            identifier, 1, encode_pcm(b"\2\0" * 512)
        )
        expected_code = (
            "provider_timeout" if failure_mode == "receive_timeout" else "provider_disconnected"
        )
        expected_type = "TimeoutError" if failure_mode == "receive_timeout" else "WebSocketClosed"
        assert error_result["speech_error"]["code"] == expected_code
        assert error_result["speech_error"]["stage"] == "asr_receive"
        assert error_result["speech_error"]["boundary"] == "deepgram_websocket"
        assert error_result["speech_error"]["exception_type"] == expected_type
        from me3_voice.speech import SpeechClient

        class ToolResult:
            async def call_tool(self, *_args):
                return SimpleNamespace(is_error=False, structured_content=error_result)

        client = SpeechClient(timeout=1, backends={"asr": spec_asr})
        client.clients = {"asr": ToolResult()}
        with pytest.raises(ProviderError) as caught:
            await client.call("asr", "asr_append", {})
        assert caught.value.code == expected_code
        assert caught.value.error_stage == "asr_receive"
        assert caught.value.error_boundary == "deepgram_websocket"
        assert caught.value.error_detail == expected_type
        expected = ("failure", expected_code)

    with sqlite3.connect(path) as db:
        status = db.execute(
            "SELECT status,error_code FROM provider_requests WHERE run_id=?",
            (f"asr-{failure_mode}-01",),
        ).fetchone()
    assert status == expected
    with pytest.raises(ProviderError, match="durable provider request allowance exhausted"):
        transport.authorize()
    await service.close()


async def test_speech_client_transport_error_preserves_only_exception_class_chain():
    from me3_voice.speech import SpeechClient

    class BrokenTool:
        async def call_tool(self, *_args):
            try:
                raise ValueError("private payload should not be retained")
            except ValueError as cause:
                raise RuntimeError("private transport message should not be retained") from cause

    client = SpeechClient(timeout=1, backends={"asr": spec("asr")})
    client.clients = {"asr": BrokenTool()}
    with pytest.raises(ProviderError) as caught:
        await client.call("asr", "asr_append", {})
    assert caught.value.code == "mcp_transport_error"
    assert caught.value.error_stage == "asr_append"
    assert caught.value.error_detail == "RuntimeError"
    assert caught.value.cause_type == "ValueError"
    assert "private" not in str(caught.value)


def test_request_ledger_summary_is_run_scoped_and_sanitized(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transport = _ledger_transport(path, limit=2)
    _, request_id = authorize_chat(transport)
    transport.finish_request(request_id, "failure", "provider_timeout")
    _, other_id = authorize_chat(transport)
    transport.finish_request(other_id, "success")
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO provider_requests "
            "(id,run_id,provider,model,protocol,status,error_code,created,updated) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "x" * 32,
                "another-run-0001",
                "example.invalid",
                "other-model",
                "other-protocol",
                "success",
                None,
                1.0,
                1.0,
            ),
        )
    summary = request_ledger_summary("pytest-run-0001", path)
    assert len(summary["requests"]) == 2
    assert summary["counts"] == {
        "example.invalid|documented-test-model|chat-completions-v1": {"failure": 1, "success": 1}
    }
    assert [request["request_id"] for request in summary["requests"]] == [request_id, other_id]
    assert {request["status"] for request in summary["requests"]} == {"failure", "success"}
    assert request_ledger_summary("another-run-0001", path)["requests"][0]["model"] == "other-model"
    invalid = request_ledger_summary("api_key=do-not-echo", path)
    assert invalid["run_id"] is None and "do-not-echo" not in repr(invalid)


class Peer:
    def __init__(self, messages=()):
        self.queue = asyncio.Queue()
        for message in messages:
            self.queue.put_nowait(message)
        self.sent = []
        self.received = asyncio.Event()
        self.closed = False

    async def send(self, value):
        self.sent.append(value)
        if value == {"type": "CloseStream"}:
            self.queue.put_nowait(
                {
                    "type": "Results",
                    "is_final": True,
                    "channel": {"alternatives": [{"transcript": "world"}]},
                }
            )
            self.queue.put_nowait({"type": "Metadata"})

    async def send_json(self, value):
        await self.send(value)

    async def send_bytes(self, value):
        await self.send(value)

    async def receive(self):
        value = await self.queue.get()
        self.received.set()
        if isinstance(value, Exception):
            raise value
        return value


class Wire:
    def __init__(self, peer=None, events=()):
        self.peer, self.values = peer, events
        self.params = self.body = None
        self.closed = False
        self.connection_count = 0
        self.requests = []
        self.connect_delay = 0.0
        self.last_websocket_timing = {}

    def authorize(self):
        request_id = f"test-request-{len(self.requests) + 1}"
        self.requests.append({"id": request_id, "status": "dispatched_unknown"})
        return "offline-test-key", request_id

    def finish_request(self, request_id, outcome, error_code=None):
        for request in self.requests:
            if request["id"] == request_id:
                request.update(status=outcome, error_code=error_code)
                return
        raise AssertionError("unknown offline request identity")

    async def open_socket(self, params, _key, *, role="tts"):
        from me3_voice.remote import Socket
        from aiohttp import WSMsgType
        from aiohttp.http_websocket import WSMessage

        self.params = params
        self.connection_count += 1
        self.last_websocket_timing = {"websocket_handshake_monotonic": time.monotonic()}
        if self.connect_delay:
            await asyncio.sleep(self.connect_delay)
        self.last_websocket_timing["connection_ready_monotonic"] = time.monotonic()
        actual_peer = self.peer

        class AiohttpPeerAdapter:
            async def send_json(self, value):
                await actual_peer.send(value)

            async def send_bytes(self, value):
                await actual_peer.send(value)

            async def receive(self):
                value = await actual_peer.receive()
                if isinstance(value, bytes):
                    return WSMessage(WSMsgType.BINARY, value, None)
                return WSMessage(WSMsgType.TEXT, json.dumps(value), None)

        class Connection:
            def __init__(self, peer):
                self.socket = Socket(peer, 1, role)
                self.closed = False

            async def aclose(self):
                self.closed = True
                actual_peer.closed = True

        return Connection(AiohttpPeerAdapter())

    @asynccontextmanager
    async def socket(self, params):
        self.params = params
        try:
            yield self.peer
        finally:
            self.peer.closed = True

    async def events(self, body, *, total_timeout=True):
        self.body = body
        self.total_timeout = total_timeout
        try:
            for value in self.values:
                if isinstance(value, Exception):
                    raise value
                yield value
        finally:
            self.closed = True

    async def aclose(self):
        self.closed = True


def delta(content="", *, calls=None, finish=None):
    return {
        "choices": [
            {
                "index": 0,
                "delta": {"content": content, "tool_calls": calls or []},
                "finish_reason": finish,
            }
        ]
    }


async def test_deepgram_incremental_segments_close_boundary_and_mcp_cleanup():
    peer = Peer(
        [
            {
                "type": "Results",
                "is_final": True,
                "channel": {"alternatives": [{"transcript": "hello"}]},
            },
            {
                "type": "Results",
                "is_final": False,
                "channel": {"alternatives": [{"transcript": "wor"}]},
            },
        ]
    )
    wire = Wire(peer)
    provider = DeepgramASR(spec("asr"), wire)
    service = SpeechService(provider, DeepgramTTS(spec("tts"), Wire()))
    key = TurnKey(session_id="contract", turn_id=1)
    identifier = (await service.open(**key.model_dump()))["operation_id"]
    first = await service.append(identifier, 0, encode_pcm(b"\1\0" * 512))
    await peer.received.wait()
    result = await service.append(identifier, 1, encode_pcm(b"\1\0" * 512))
    assert (result["partial"] or first["partial"])["text"] == "hello wor"
    final = await service.finish(identifier)
    assert final["text"] == "hello world" and final["final"]
    assert peer.closed and not service.operations
    assert wire.params["endpointing"] == "false" and wire.params["sample_rate"] == 16000


async def test_asr_cancel_during_receive_and_disconnected_result_never_finalizes():
    peer = Peer()
    provider = DeepgramASR(spec("asr"), Wire(peer))
    stream = provider.open_stream()
    await stream.append(b"\1\0")
    # Append now admits PCM while the one connection task starts in the
    # background; explicitly wait for readiness to test cancellation during
    # the provider receive lifecycle, rather than assuming eager connect.
    await stream.start()
    await stream.aclose()
    assert stream.reader.done() and peer.closed
    peer = Peer([ProviderError("controlled disconnect")])
    stream = DeepgramASR(spec("asr"), Wire(peer)).open_stream()
    with pytest.raises(ProviderError):
        await stream.append(b"\1\0")
        await stream.finish()
    await stream.aclose()
    assert peer.closed


async def test_tts_wire_packet_boundaries_first_audio_flush_and_clear_on_cancel():
    peer = Peer([b"\1", b"\0\2\0", {"type": "Flushed", "sequence_id": 0}])
    wire = Wire(peer)
    provider = DeepgramTTS(spec("tts"), wire)
    chunks = [chunk async for chunk in provider.synthesize("Hello.")]
    assert b"".join(chunks) == b"\1\0\2\0" and not peer.closed
    assert wire.connection_count == 1 and wire.requests[0]["status"] == "success"
    assert peer.sent == [{"type": "Speak", "text": "Hello."}, {"type": "Flush"}]
    diagnostics = provider.diagnostics()
    assert diagnostics["status"] == "completed"
    assert diagnostics["connection_reused"] is False
    assert diagnostics["connection_attempt_monotonic"] <= diagnostics["connection_ready_monotonic"]
    assert diagnostics["connection_trace"]
    assert diagnostics["provider_pcm_packets"] == 2
    assert diagnostics["provider_pcm_bytes"] == 4
    assert diagnostics["provider_audio_seconds"] == pytest.approx(4 / 32000)
    assert diagnostics["first_pcm_monotonic"] <= diagnostics["last_pcm_monotonic"]
    assert diagnostics["last_pcm_monotonic"] <= diagnostics["provider_flushed_monotonic"]
    assert diagnostics["synthesis_wall_seconds"] >= 0
    await provider.aclose()
    assert peer.closed

    peer = Peer([b"\1\0"])
    cancelled_provider = DeepgramTTS(spec("tts"), Wire(peer))
    stream = cancelled_provider.synthesize("Hello.")
    assert await anext(stream) == b"\1\0"
    await stream.aclose()
    assert peer.sent[-1] == {"type": "Clear"} and peer.closed
    assert cancelled_provider.transport.requests[0]["status"] == "unknown"
    assert cancelled_provider.diagnostics()["status"] == "cancelled"


async def test_tts_provider_and_mcp_pull_timing_are_separately_reported():
    peer = Peer([b"\1", b"\0\2\0", {"type": "Flushed", "sequence_id": 0}])
    provider = DeepgramTTS(spec("tts"), Wire(peer))
    service = SpeechService(None, provider)
    try:
        opened = await service.open("tts-timing-session", 1, "Hello.")
        chunk = await service.next(opened["operation_id"])
        terminal = await service.next(opened["operation_id"])
        assert not chunk["done"] and terminal["done"]
        diagnostics = terminal["tts_diagnostics"]
        assert diagnostics["status"] == "completed"
        assert diagnostics["mcp"]["tts_next_count"] == 2
        assert diagnostics["mcp"]["pcm_chunks"] == 1
        assert diagnostics["mcp"]["pcm_bytes"] == 4
        assert diagnostics["mcp"]["audio_seconds"] == pytest.approx(4 / 32000)
        assert diagnostics["provider"]["status"] == "completed"
        assert diagnostics["provider"]["provider_flushed_monotonic"] is not None
        assert diagnostics["provider"]["provider_pcm_packets"] == 2
        assert diagnostics["provider"]["provider_pcm_bytes"] == 4
        assert diagnostics["mcp"]["tts_next_seconds"]["sample_count"] == 2
    finally:
        await service.close()


async def test_tts_reuses_one_deepgram_connection_across_mcp_segments_and_reduces_setup_wall():
    class SegmentedPeer(Peer):
        def __init__(self, responses):
            super().__init__()
            self.responses = iter(responses)

        async def send(self, value):
            await super().send(value)
            if value == {"type": "Flush"}:
                for message in next(self.responses):
                    self.queue.put_nowait(message)

    texts = (
        "Short answer.",
        "A medium answer uses another sentence.",
        "A longer explanation uses a third segment.",
    )
    segment_audio = [b"\1\0" * 80 for _ in texts]

    # Baseline emulates the prior per-segment socket lifecycle with the same
    # fixed 40 ms handshake cost and same output for each segment.
    cold_started = time.perf_counter()
    cold_connections = 0
    for text, pcm in zip(texts, segment_audio, strict=True):
        wire = Wire(Peer([pcm, {"type": "Flushed", "sequence_id": 0}]))
        wire.connect_delay = 0.04
        provider = DeepgramTTS(spec("tts"), wire)
        _ = [chunk async for chunk in provider.synthesize(text)]
        cold_connections += wire.connection_count
        await provider.aclose()
    cold_wall = time.perf_counter() - cold_started

    warm_peer = SegmentedPeer(
        [
            [pcm, {"type": "Flushed", "sequence_id": index}]
            for index, pcm in enumerate(segment_audio)
        ]
    )
    wire = Wire(warm_peer)
    wire.connect_delay = 0.04
    provider = DeepgramTTS(spec("tts"), wire)
    service = SpeechService(None, provider)
    warm_started = time.perf_counter()
    outputs = []
    try:
        for index, text in enumerate(texts, start=1):
            opened = await service.open("tts-reuse-session", index, text)
            chunks = []
            while True:
                result = await service.next(opened["operation_id"])
                if result["done"]:
                    assert result["tts_diagnostics"]["status"] == "completed"
                    break
                chunks.append(base64.b64decode(result["pcm_b64"]))
            outputs.append(b"".join(chunks))
        warm_wall = time.perf_counter() - warm_started
        assert outputs == segment_audio
        assert wire.connection_count == 1
        assert [request["status"] for request in wire.requests] == ["success"] * len(texts)
        assert warm_peer.sent == [
            item for text in texts for item in ({"type": "Speak", "text": text}, {"type": "Flush"})
        ]
        assert warm_wall < cold_wall
        assert cold_connections == len(texts)
    finally:
        await service.close()
    assert warm_peer.closed
    print(
        json.dumps(
            {
                "kind": "offline_fake_websocket_setup_benchmark",
                "segments": len(texts),
                "fixed_fake_handshake_seconds": 0.04,
                "cold_connections": cold_connections,
                "reused_connections": wire.connection_count,
                "cold_wall_seconds": cold_wall,
                "reused_wall_seconds": warm_wall,
                "wall_reduction_seconds": cold_wall - warm_wall,
                "provider_evidence": False,
            },
            sort_keys=True,
        )
    )


def test_live_chat_input_budget_is_frozen_aggregate_and_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "requests.sqlite"
    monkeypatch.setenv("ME3_REQUEST_LEDGER", str(path))
    transport = _ledger_transport(path, limit=5)

    # An over-cap first request freezes the campaign profile but creates no
    # billable request reservation or network client.
    with pytest.raises(ProviderError) as too_large:
        transport.authorize(input_tokens=101, input_token_limit=100, input_tokens_total_limit=500)
    assert too_large.value.code == "input_token_request_limit_reached"
    assert transport.client is None and transport.requests == 0
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM provider_requests").fetchone()[0] == 0
        assert db.execute(
            "SELECT request_limit,input_tokens_per_request_limit,input_tokens_limit FROM provider_campaign_caps"
        ).fetchone() == (5, 100, 500)
    # A later process cannot widen the frozen profile after this rejection.
    with pytest.raises(ProviderError) as widened:
        transport.authorize(input_tokens=101, input_token_limit=101, input_tokens_total_limit=500)
    assert widened.value.code == "campaign_cap_increase_rejected"

    aggregate_path = tmp_path / "aggregate-requests.sqlite"
    aggregate_transport = _ledger_transport(aggregate_path, limit=5)
    _, first = aggregate_transport.authorize(
        input_tokens=300, input_token_limit=400, input_tokens_total_limit=500
    )
    aggregate_transport.finish_request(first, "failure", "provider_timeout")
    # Failed/unknown provider outcomes still consume their reserved input size.
    with pytest.raises(ProviderError) as exhausted:
        aggregate_transport.authorize(
            input_tokens=201, input_token_limit=400, input_tokens_total_limit=500
        )
    assert exhausted.value.code == "input_token_budget_exhausted"
    with sqlite3.connect(aggregate_path) as db:
        assert db.execute(
            "SELECT status,input_tokens_reserved FROM provider_requests WHERE id=?", (first,)
        ).fetchone() == ("failure", 300)
        assert db.execute("SELECT COUNT(*) FROM provider_requests").fetchone()[0] == 1

    # The same run identity cannot enlarge any frozen campaign ceiling.
    with pytest.raises(ProviderError) as cap_increase:
        _ledger_transport(aggregate_path, limit=6).authorize(
            input_tokens=1, input_token_limit=400, input_tokens_total_limit=500
        )
    assert cap_increase.value.code == "campaign_cap_increase_rejected"


def test_conservative_estimator_covers_actual_serialized_request():
    from me3_voice.request_budget import conservative_input_token_estimate, serialized_request

    params = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "What is in the 🐢 image?"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
    }
    encoded = serialized_request(params)
    estimate, serialized_bytes = conservative_input_token_estimate(params)
    assert serialized_bytes == len(encoded)
    assert estimate == len(encoded) + max(128, (len(encoded) + 3) // 4)
    assert encoded.startswith(b'{"messages"')


def test_conservative_estimator_fails_closed_for_unserializable_or_oversize_params():
    from me3_voice.request_budget import RequestBudgetError, serialized_request

    with pytest.raises(RequestBudgetError, match="safely serialized"):
        serialized_request({"opaque": object()})
    with pytest.raises(RequestBudgetError, match="estimation limits"):
        serialized_request({"content": "x" * 4_000_001})


@pytest.mark.parametrize(
    "messages",
    [
        [b"\1", {"type": "Flushed"}],
        [{"type": "Flushed"}],
        [{"type": "Error", "description": "private-body"}],
    ],
)
async def test_tts_rejects_incomplete_or_empty_output(messages):
    peer = Peer(messages)
    with pytest.raises(ProviderError) as error:
        _ = [chunk async for chunk in DeepgramTTS(spec("tts"), Wire(peer)).synthesize("Hello.")]
    assert peer.closed and "private-body" not in str(error.value)


async def test_chat_streams_coherent_text_and_reconstructs_tool_history():
    wire = Wire(
        events=[delta("Hello"), delta(" there. "), delta("Next"), delta(" sentence", finish="stop")]
    )
    model = ChatReasoning(spec("llm"), wire)
    events = []
    model.on_event = events.append
    assert [part async for part in model.respond([{"role": "user", "content": "hello"}])] == [
        "Hello there.",
        "Next sentence",
    ]
    assert (
        events
        == [
            "llm_first_delta",
            "llm_text_stream_completed",
            "llm_text_unpunctuated_finish",
        ]
        and wire.closed
    )
    assert wire.total_timeout is False
    assert (
        "All user-facing replies and spoken summaries must be in English"
        in wire.body["messages"][0]["content"]
    )
    assert "laughter, giggling" in wire.body["messages"][0]["content"]
    assert "complete grammatical sentences" in wire.body["messages"][0]["content"]
    history = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_request": {"name": "read_mock_light", "arguments": "{}"},
            "content": '{"enabled":false}',
        }
    ]
    mapped = model.messages(history)
    assert mapped[-1]["role"] == "assistant"
    assert "read_mock_light" in mapped[-1]["content"]
    assert all("tool_calls" not in message and message["role"] != "tool" for message in mapped)


async def test_empty_successful_sse_response_is_explicitly_diagnosed():
    model = ChatReasoning(spec("llm"), Wire(events=[delta(finish="stop")]))
    events = []
    model.on_event = events.append
    assert [
        part async for part in model.respond([{"role": "user", "content": "Where is Tokyo?"}])
    ] == []
    assert events == ["llm_empty_response"]


def test_completed_task_history_uses_spoken_receipt_without_tool_protocol():
    history = [
        {"role": "user", "content": "play Maroon Five", "turn_id": 1},
        {
            "role": "tool",
            "tool_call_id": "music-1",
            "tool_request": {
                "name": "control_music",
                "arguments": '{"action":"play","query":"Maroon Five"}',
            },
            "content": '{"result_status":"reported","summary":"Spotify acknowledged play: This Love. Playback is unverified.","display":"play"}',
            "turn_id": 1,
        },
        {
            "role": "assistant",
            "content": "Spotify acknowledged play: This Love. Playback is unverified.",
            "turn_id": 1,
        },
        {"role": "user", "content": "What is the capital of the Philippines?", "turn_id": 2},
    ]

    mapped = ChatReasoning.messages(history)
    assert [message["role"] for message in mapped] == ["system", "user", "assistant", "user"]
    assert mapped[-2]["content"].startswith("Spotify acknowledged play")
    assert all("tool_calls" not in message and message["role"] != "tool" for message in mapped)
    assert "letter-by-letter spelling, as authoritative" in mapped[0]["content"]
    assert "Do not invent a place identity or distance" in mapped[0]["content"]


async def test_agent_tool_description_matches_explicit_search_and_github_capabilities():
    wire = Wire(events=[delta("Ready.", finish="stop")])
    model = ChatReasoning(
        spec("llm"),
        wire,
        agent_enabled=True,
        agent_tavily_enabled=True,
        agent_github_readonly=True,
    )
    assert [
        part
        async for part in model.respond(
            [{"role": "user", "content": "Search the web for current weather reports."}]
        )
    ] == ["Ready."]
    agent = next(
        tool["function"]
        for tool in wire.body["tools"]
        if tool["function"]["name"] == "run_agent_task"
    )
    assert "bounded Tavily web searches" in agent["description"]
    assert "read GitHub repository metadata only" in agent["description"]
    assert "cannot write GitHub repositories" in agent["description"]


async def test_agent_tool_does_not_claim_unconfigured_external_integrations():
    wire = Wire(events=[delta("Ready.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, agent_enabled=True)
    _ = [
        part
        async for part in model.respond(
            [{"role": "user", "content": "Search the web for weather reports."}]
        )
    ]
    agent = next(
        tool["function"]
        for tool in wire.body["tools"]
        if tool["function"]["name"] == "run_agent_task"
    )
    assert (
        "Tavily" not in agent["description"]
        and "GitHub repository metadata" not in agent["description"]
    )


async def test_chat_incomplete_tool_args_and_provider_failure_cannot_execute_effects():
    calls = [
        {
            "index": 0,
            "id": "call_1",
            "function": {"name": "set_mock_light", "arguments": '{"enabled":'},
        }
    ]
    wire = Wire(events=[delta(calls=calls), ProviderError("controlled SSE disconnect")])
    model = ChatReasoning(spec("llm"), wire)
    with pytest.raises(ProviderError):
        _ = [part async for part in model.respond([{"role": "user", "content": "light on"}])]
    assert wire.closed
    wire = Wire(
        events=[
            delta(calls=calls),
            delta(calls=[{"index": 0, "function": {"arguments": "true}"}}], finish="tool_calls"),
        ]
    )
    model = ChatReasoning(spec("llm"), wire)
    output = [part async for part in model.respond([{"role": "user", "content": "light on"}])]
    assert len(output) == 1 and output[0].arguments == {"enabled": True}


async def test_malformed_tool_arguments_are_classified_as_response_parse_failures():
    calls = [
        {
            "index": 0,
            "id": "call_bad_json",
            "function": {"name": "set_mock_light", "arguments": '{"enabled":}'},
        }
    ]
    model = ChatReasoning(spec("llm"), Wire(events=[delta(calls=calls, finish="tool_calls")]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "light on"}])]
    assert error.value.code == "provider_invalid_response"
    assert error.value.error_stage == "response_parse"
    assert error.value.error_detail == "invalid_tool_arguments"


@pytest.mark.parametrize(
    ("bad_event", "expected_detail"),
    [
        ([], "event_not_object"),
        ({"error": {"message": "private provider text"}}, "provider_error_event"),
        ({"object": "chat.completion.chunk"}, "missing_choices"),
        ({"choices": "not-a-list"}, "choices_not_array"),
        ({"choices": [{"index": 0}, {"index": 1}]}, "unexpected_choice_count"),
        ({"choices": ["bad choice"]}, "choice_not_object"),
        ({"choices": [{"index": 1, "delta": {}}]}, "invalid_choice_index"),
        ({"choices": [{"delta": []}]}, "invalid_delta_shape"),
        (
            delta(calls=[{"index": 0, "function": None}], finish="tool_calls"),
            "invalid_tool_call_shape",
        ),
    ],
)
async def test_reasoning_stream_reports_safe_structural_failure_detail(bad_event, expected_detail):
    model = ChatReasoning(spec("llm"), Wire(events=[bad_event]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "hello"}])]
    assert error.value.code == "provider_invalid_response"
    assert error.value.error_stage == "response_parse"
    assert error.value.error_detail == expected_detail
    assert "hello" not in str(error.value)


async def test_reasoning_rejects_unfinished_provider_finish_reason_without_tool_effect():
    model = ChatReasoning(spec("llm"), Wire(events=[delta("A partial answer", finish="length")]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "hello"}])]
    assert error.value.code == "provider_invalid_response"
    assert error.value.error_detail == "invalid_finish_reason"


@pytest.mark.parametrize(
    ("call", "expected_detail"),
    [
        (
            {
                "index": 0,
                "id": "bad id",
                "function": {"name": "run_device_action", "arguments": "{}"},
            },
            "invalid_tool_identity",
        ),
        (
            {"index": 0, "id": "unknown_1", "function": {"name": "run_shell", "arguments": "{}"}},
            "unadvertised_tool_name",
        ),
    ],
)
async def test_reasoning_tool_identity_and_name_fail_closed(call, expected_detail):
    model = ChatReasoning(spec("llm"), Wire(events=[delta(calls=[call], finish="tool_calls")]))
    with pytest.raises(ProviderError) as error:
        _ = [part async for part in model.respond([{"role": "user", "content": "create a file"}])]
    assert error.value.error_detail == expected_detail


async def test_chat_mock_action_contract_is_explicit_and_uses_one_validated_call():
    calls = [
        {
            "index": 0,
            "id": "demo-1",
            "function": {
                "name": "run_device_action",
                "arguments": '{"action":"set_timer","value":"10 seconds"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="tool_calls")])
    model = ChatReasoning(spec("llm"), wire)
    output = [
        part async for part in model.respond([{"role": "user", "content": "set a demo timer"}])
    ]
    assert len(output) == 1
    assert output[0].name == "run_device_action"
    assert output[0].arguments == {"action": "set_timer", "value": "10 seconds"}
    body = wire.body
    schema = next(
        tool["function"]
        for tool in body["tools"]
        if tool["function"]["name"] == "run_device_action"
    )
    assert "set_timer" in schema["parameters"]["properties"]["action"]["enum"]
    assert "set_temperature" in schema["parameters"]["properties"]["action"]["enum"]
    assert "cancel_schedule" in schema["parameters"]["properties"]["action"]["enum"]
    assert schema["parameters"]["additionalProperties"] is False
    names = {tool["function"]["name"] for tool in body["tools"]}
    assert not {"call", "message", "make_call", "send_message"} & names


async def test_pi_device_profile_does_not_advertise_in_memory_mock_light_tools():
    wire = Wire(events=[delta(content="The device profile is ready.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    _ = [part async for part in model.respond([{"role": "user", "content": "set a demo timer"}])]
    names = {tool["function"]["name"] for tool in wire.body["tools"]}
    assert "run_device_action" in names
    assert "set_mock_light" not in names and "read_mock_light" not in names


@pytest.mark.parametrize("agent_enabled", [False, True])
async def test_chat_accepts_advertised_control_music_tool(agent_enabled):
    calls = [
        {
            "index": 0,
            "id": "music-1",
            "function": {
                "name": "control_music",
                "arguments": '{"action":"play","query":"Bruno Mars"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="tool_calls")])
    model = ChatReasoning(spec("llm"), wire, agent_enabled=agent_enabled)
    output = [
        part
        async for part in model.respond([{"role": "user", "content": "play a Bruno Mars song"}])
    ]

    assert len(output) == 1
    assert output[0].call_id == "music-1"
    assert output[0].name == "control_music"
    assert output[0].arguments == {"action": "play", "query": "Bruno Mars"}
    music_tool = next(
        tool["function"]
        for tool in wire.body["tools"]
        if tool["function"]["name"] == "control_music"
    )
    assert music_tool["parameters"]["properties"]["action"]["enum"]
    assert "play_playlist" in music_tool["parameters"]["properties"]["action"]["enum"]
    assert "market" in music_tool["parameters"]["properties"]


async def test_chat_accepts_complete_qwen_tool_call_with_stop_finish_reason():
    calls = [
        {
            "index": 0,
            "id": "qwen-music-1",
            "function": {
                "name": "control_music",
                "arguments": '{"action":"pause"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="stop")])
    model = ChatReasoning(spec("llm"), wire)
    output = [
        part async for part in model.respond([{"role": "user", "content": "pause the music"}])
    ]

    assert len(output) == 1
    assert output[0].name == "control_music"
    assert output[0].arguments == {"action": "pause"}


async def test_chat_rejects_tool_call_marked_as_truncated():
    calls = [
        {
            "index": 0,
            "id": "truncated-1",
            "function": {
                "name": "control_music",
                "arguments": '{"action":"pause"}',
            },
        }
    ]
    wire = Wire(events=[delta(calls=calls, finish="length")])
    model = ChatReasoning(spec("llm"), wire)

    with pytest.raises(ProviderError, match="incomplete or truncated"):
        _ = [part async for part in model.respond([{"role": "user", "content": "pause music"}])]


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Play a Bruno Mars song", "control_music"),
        ("set a 10 second timer", "run_device_action"),
        ("create a new project folder called demo", "run_agent_task"),
        ("what is a Raspberry Pi", None),
        ("play a song and turn the light on", None),
    ],
)
async def test_live_tool_contract_is_scoped_to_one_clear_request(utterance, expected):
    wire = Wire(events=[delta("Done.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, agent_enabled=True, legacy_mock_tools=False)
    _ = [part async for part in model.respond([{"role": "user", "content": utterance}])]

    advertised = [tool["function"]["name"] for tool in wire.body.get("tools", [])]
    assert advertised == ([expected] if expected else [])
    if expected:
        assert wire.body["tool_choice"] == {
            "type": "function",
            "function": {"name": expected},
        }
        assert "must be in English" in wire.body["messages"][0]["content"]
    else:
        assert "tool_choice" not in wire.body


async def test_music_tool_prompt_drops_unrelated_and_cancelled_conversation_turns():
    wire = Wire(events=[delta("Done.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {"role": "user", "content": "Tell me a joke.", "turn_id": 1},
        {"role": "assistant", "content": "A short joke.", "turn_id": 1},
        {"role": "user", "content": "What is the capital of the Philippines?", "turn_id": 2},
        {"role": "assistant", "content": "Manila.", "turn_id": 2},
        {"role": "user", "content": "He's swift.", "turn_id": 3},
        {"role": "user", "content": "No, play Air Supply.", "turn_id": 4},
    ]

    _ = [part async for part in model.respond(history)]

    assert wire.body["messages"][0]["content"].startswith("You are Swift. Call")
    assert wire.body["messages"][1:] == [{"role": "user", "content": "No, play Air Supply."}]


async def test_music_follow_up_prompt_retains_last_completed_music_receipt_only():
    wire = Wire(events=[delta("Done.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {"role": "user", "content": "What time is it?", "turn_id": 1},
        {"role": "assistant", "content": "It is noon.", "turn_id": 1},
        {"role": "user", "content": "Play Maroon Five.", "turn_id": 2},
        {
            "role": "tool",
            "tool_call_id": "music-2",
            "tool_request": {
                "name": "control_music",
                "arguments": '{"action":"play","query":"Maroon Five"}',
            },
            "content": '{"result_status":"reported","summary":"Spotify acknowledged play: This Love. Playback is unverified."}',
            "turn_id": 2,
        },
        {
            "role": "assistant",
            "content": "Spotify acknowledged play: This Love. Playback is unverified.",
            "turn_id": 2,
        },
        {"role": "user", "content": "Pause it.", "turn_id": 3},
    ]

    _ = [part async for part in model.respond(history)]

    assert [message["role"] for message in wire.body["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert wire.body["messages"][1]["content"] == "Play Maroon Five."
    assert "This Love" in wire.body["messages"][2]["content"]
    assert wire.body["messages"][3]["content"] == "Pause it."


async def test_contextual_music_follow_up_keeps_only_music_tool_available():
    wire = Wire(events=[delta("I will lower it.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {
            "role": "tool",
            "tool_call_id": "music-previous",
            "tool_request": {"name": "control_music", "arguments": '{"action":"play"}'},
            "content": '{"result_status":"completed"}',
        },
        {"role": "user", "content": "turn it down"},
    ]
    _ = [part async for part in model.respond(history)]
    assert [tool["function"]["name"] for tool in wire.body["tools"]] == ["control_music"]


async def test_post_action_reasoning_cannot_repeat_completed_task_tool():
    wire = Wire(events=[delta("I paused playback.", finish="stop")])
    model = ChatReasoning(spec("llm"), wire, legacy_mock_tools=False)
    history = [
        {"role": "user", "content": "pause the music", "turn_id": 4},
        {
            "role": "tool",
            "tool_call_id": "music-4",
            "tool_request": {"name": "control_music", "arguments": '{"action":"pause"}'},
            "content": '{"result_status":"completed"}',
            "turn_id": 4,
        },
    ]
    _ = [part async for part in model.respond(history)]
    assert "tools" not in wire.body


def test_live_gate_precedes_credentials_and_network_and_budget_is_bounded(monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-test-not-a-project-credential")
    transport = LiveTransport(spec("llm"))
    with pytest.raises(ConfigurationError, match="live requests disabled"):
        transport.authorize()
    assert transport.client is None and transport.requests == 0
    transport = LiveTransport(replace(spec("llm"), allow_live=True, request_limit=1))
    assert authorize_chat(transport)[0] == "offline-test-not-a-project-credential"
    with pytest.raises(ProviderError, match="allowance exhausted"):
        authorize_chat(transport)
    monkeypatch.delenv("ME3_TEST_ONLY_KEY")
    with pytest.raises(ConfigurationError, match="missing"):
        authorize_chat(LiveTransport(replace(spec("llm"), allow_live=True, request_limit=1)))
    assert transport.client is None


def test_explicitly_unbounded_provider_counts_without_a_local_ceiling(monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-test-not-a-project-credential")
    transport = LiveTransport(replace(spec("llm"), allow_live=True, request_limit=None))
    for _ in range(25):
        assert authorize_chat(transport)[0] == "offline-test-not-a-project-credential"
    assert transport.requests == 25


def test_runnable_live_chat_requires_both_input_token_ceilings():
    from dataclasses import replace

    unbounded = replace(
        spec("llm"),
        allow_live=True,
        request_limit=1,
        max_input_tokens=0,
        max_input_tokens_total=0,
    )
    with pytest.raises(ConfigurationError, match="live chat requires"):
        unbounded.validate("llm", runnable=True)


def test_phase7_campaign_example_has_finite_caps_but_cannot_dispatch():
    from me3_voice.configuration import load_backends

    profile = Path(__file__).resolve().parents[1] / "configs/phase7-live-campaign.example.json"
    values = load_backends(profile, runnable=True)
    assert all(not backend.allow_live for backend in values.values())
    assert values["asr"].request_limit == 6 and values["asr"].max_audio_seconds == 20
    assert values["llm"].request_limit == 8
    assert (values["llm"].max_input_tokens, values["llm"].max_input_tokens_total) == (
        32000,
        64000,
    )
    assert values["tts"].request_limit == 16


async def test_pause_aware_gate_keeps_800ms_pause_and_bounds_incomplete_turn():
    from me3_voice.audio import ConversationInput, ReplayInput
    from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

    class VAD:
        def voiced(self, pcm):
            return any(pcm)

    class Speech:
        async def transcribe(self, key, pcm):
            return "hello"

        async def synthesize(self, key, text):
            yield b"\1\0"

    class Analyzer:
        def __init__(self):
            self.cleared = 0

        def append_audio(self, pcm, voiced):
            pass

        async def analyze_end_of_turn(self):
            return EndOfTurnState.INCOMPLETE, None

        def clear(self):
            self.cleared += 1

    analyzer = Analyzer()
    pcm, quiet = b"\1\0" * 512, bytes(1024)
    async with Session(Speech(), MockLLM(), MemorySink()) as session:
        source = ConversationInput(ReplayInput(VAD()), session, analyzer)
        source.activate()
        endpoints = 0
        for frame in [pcm] * 3 + [quiet] * 25 + [pcm] * 3 + [quiet] * 64:
            endpoints += bool(await source.feed(frame))
        await session.generation
        assert endpoints == 1 and analyzer.cleared == 1 and session.turn == 1
        assert any(e["event"] == "turn_silence_fallback" for e in session.events)


async def test_actual_http_sse_and_websocket_transport_against_local_contract_peer(monkeypatch):
    """Real HTTP/WS client framing, but an offline loopback peer and dummy key."""
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    seen = []

    async def chat(request):
        seen.append((request.headers.get("Authorization"), await request.json()))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in [
            b": keepalive\r\n\r\ndata:",
            b' {"choices":[{"index":0,"delta":{"content":"Hello. "},"finish_reason":"stop"}]}\r\n\r\n',
            b"data: [DONE]\n\n",
        ]:
            await response.write(chunk)
        await response.write_eof()
        return response

    async def rejected(request):
        return web.Response(status=429, text="private-provider-body")

    async def speak(request):
        seen.append((request.headers.get("Authorization"), dict(request.query)))
        peer = web.WebSocketResponse()
        await peer.prepare(request)
        assert (await peer.receive_json())["type"] == "Speak"
        assert (await peer.receive_json())["type"] == "Flush"
        await peer.send_bytes(b"\1\0\2\0")
        await peer.send_json({"type": "Flushed", "sequence_id": 0})
        await peer.close()
        return peer

    app = web.Application()
    app.router.add_post("/chat", chat)
    app.router.add_post("/reject", rejected)
    app.router.add_get("/speak", speak)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        # Lower transport boundary intentionally uses local HTTP for this test;
        # production factory validation requires HTTPS/WSS.
        chat_spec = replace(
            spec("llm"), endpoint=str(server.make_url("/chat")), allow_live=True, request_limit=2
        )
        model = ChatReasoning(chat_spec)
        try:
            assert [x async for x in model.respond([{"role": "user", "content": "Hi"}])] == [
                "Hello."
            ]
            model.transport.spec = replace(chat_spec, endpoint=str(server.make_url("/reject")))
            with pytest.raises(ProviderError) as error:
                _ = [x async for x in model.respond([{"role": "user", "content": "Hi"}])]
            assert "private-provider-body" not in str(error.value)
            assert error.value.code == "provider_rate_limited"
            assert model.transport.requests == 2
        finally:
            await model.aclose()
        import os
        import sqlite3

        with sqlite3.connect(os.environ["ME3_REQUEST_LEDGER"]) as db:
            llm_outcomes = db.execute(
                "SELECT status, error_code FROM provider_requests WHERE protocol='chat-completions-v1' ORDER BY created"
            ).fetchall()
        assert llm_outcomes == [("success", None), ("failure", "provider_rate_limited")]
        tts = DeepgramTTS(
            replace(
                spec("tts"),
                endpoint=str(server.make_url("/speak")).replace("http:", "ws:"),
                allow_live=True,
                request_limit=1,
            )
        )
        try:
            assert b"".join([x async for x in tts.synthesize("Hello.")]) == b"\1\0\2\0"
        finally:
            await tts.aclose()
        with sqlite3.connect(os.environ["ME3_REQUEST_LEDGER"]) as db:
            assert db.execute(
                "SELECT status FROM provider_requests WHERE protocol='deepgram-speak-v1'"
            ).fetchone() == ("success",)
    assert (
        seen[0][0] == "Bearer offline-contract-key" and seen[1][0] == "Token offline-contract-key"
    )
    assert seen[1][1]["sample_rate"] == "16000"


async def test_native_aura_cycles_reuse_one_real_local_websocket_without_mixing_audio(
    monkeypatch,
):
    """The local peer exercises aiohttp + multiple Speak/Flush cycles, no provider."""
    from dataclasses import replace
    from aiohttp import WSMsgType, web
    from aiohttp.test_utils import TestServer

    connections = []
    spoken_texts = []

    async def speak(request):
        peer = web.WebSocketResponse()
        await peer.prepare(request)
        connections.append(peer)
        sequence = 0
        try:
            while True:
                message = await peer.receive()
                if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break
                if message.type != WSMsgType.TEXT:
                    continue
                value = json.loads(message.data)
                if value.get("type") == "Speak":
                    spoken_texts.append(value["text"])
                elif value.get("type") == "Flush":
                    await peer.send_bytes((b"\1\0" * 80) + bytes([sequence, 0]))
                    await peer.send_json({"type": "Flushed", "sequence_id": sequence})
                    sequence += 1
        finally:
            if not peer.closed:
                await peer.close()
        return peer

    app = web.Application()
    app.router.add_get("/speak", speak)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        tts_spec = replace(
            spec("tts"),
            endpoint=str(server.make_url("/speak")).replace("http:", "ws:"),
            allow_live=True,
            request_limit=2,
        )
        provider = DeepgramTTS(tts_spec)
        try:
            first = b"".join([chunk async for chunk in provider.synthesize("First answer.")])
            first_diagnostics = provider.diagnostics()
            second = b"".join([chunk async for chunk in provider.synthesize("Follow-up answer.")])
            second_diagnostics = provider.diagnostics()
            assert first == (b"\1\0" * 80) + b"\0\0"
            assert second == (b"\1\0" * 80) + b"\1\0"
            assert len(connections) == 1
            assert spoken_texts == ["First answer.", "Follow-up answer."]
            assert first_diagnostics["connection_reused"] is False
            assert first_diagnostics["connection_setup_seconds"] >= 0
            trace = first_diagnostics["connection_trace"]
            assert trace["websocket_handshake_monotonic"] <= trace["connection_ready_monotonic"]
            assert second_diagnostics["connection_reused"] is True
            assert second_diagnostics["connection_reuse_monotonic"] > 0
            assert "offline-contract-key" not in json.dumps(first_diagnostics)
        finally:
            await provider.aclose()
        import os

        with sqlite3.connect(os.environ["ME3_REQUEST_LEDGER"]) as db:
            assert db.execute(
                "SELECT status FROM provider_requests WHERE protocol='deepgram-speak-v1' "
                "ORDER BY created"
            ).fetchall() == [("success",), ("success",)]
    assert connections[0].closed


async def test_aura_releases_idle_socket_before_receive_deadline(monkeypatch):
    """A follow-up after the provider read deadline opens a fresh socket."""
    from dataclasses import replace
    from aiohttp import WSMsgType, web
    from aiohttp.test_utils import TestServer

    connections = []

    async def speak(request):
        peer = web.WebSocketResponse()
        await peer.prepare(request)
        connections.append(peer)
        async for message in peer:
            if message.type != WSMsgType.TEXT:
                continue
            if json.loads(message.data).get("type") == "Flush":
                await peer.send_bytes(b"\1\0" * 80)
                await peer.send_json({"type": "Flushed"})
        return peer

    app = web.Application()
    app.router.add_get("/speak", speak)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        provider = DeepgramTTS(
            replace(
                spec("tts"),
                endpoint=str(server.make_url("/speak")).replace("http:", "ws:"),
                timeout_seconds=0.3,
                allow_live=True,
                request_limit=2,
            )
        )
        try:
            assert b"".join([c async for c in provider.synthesize("First.")]) == b"\1\0" * 80
            await asyncio.sleep(0.27)
            assert b"".join([c async for c in provider.synthesize("Second.")]) == b"\1\0" * 80
            assert len(connections) == 2
            assert provider.diagnostics()["connection_reused"] is False
        finally:
            await provider.aclose()


async def test_live_chat_stream_waits_past_configured_deadline_for_provider_output(monkeypatch):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def delayed_chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"index":0,"delta":{"content":"Here."},"finish_reason":null}]}\n\n'
        )
        await asyncio.sleep(0.16)
        await response.write(
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        await asyncio.sleep(0.16)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/chat", delayed_chat)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.3,
        )
        transport = LiveTransport(selected)
        try:
            result = [
                event
                async for event in transport.events(
                    {"model": selected.model, "messages": [], "stream": True},
                    total_timeout=False,
                )
            ]
        finally:
            await transport.aclose()

    assert result[0]["choices"][0]["delta"]["content"] == "Here."


async def test_live_chat_stream_fails_after_sse_read_stalls_without_total_turn_deadline(
    monkeypatch,
):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def stalled_chat(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(0.15)
        return response

    app = web.Application()
    app.router.add_post("/chat", stalled_chat)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.05,
        )
        transport = LiveTransport(selected)
        reasoning = ChatReasoning(selected, transport)
        observed = []
        reasoning.on_event = observed.append
        try:
            with pytest.raises(
                ProviderError, match="timed out waiting for the first SSE event"
            ) as error:
                _ = [text async for text in reasoning.respond([{"role": "user", "content": "Hi."}])]
        finally:
            await transport.aclose()

    assert error.value.code == "provider_timeout"
    assert error.value.role == "llm" and error.value.error_stage == "reasoning"
    assert error.value.error_boundary == "first_sse_event"
    assert observed == ["llm_request_submission_started", "llm_response_headers_received"]


async def test_live_chat_timeout_before_response_headers_has_distinct_boundary(monkeypatch):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def delayed_headers(request):
        await asyncio.sleep(0.15)
        return web.Response(text="late")

    app = web.Application()
    app.router.add_post("/chat", delayed_headers)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.05,
        )
        transport = LiveTransport(selected)
        observed = []
        transport.on_event = observed.append
        try:
            with pytest.raises(
                ProviderError, match="timed out waiting for response headers"
            ) as error:
                _ = [
                    event
                    async for event in transport.events(
                        {"model": selected.model, "messages": [], "stream": True},
                        total_timeout=False,
                    )
                ]
        finally:
            await transport.aclose()

    assert error.value.code == "provider_timeout"
    assert error.value.error_boundary == "response_headers"
    assert observed == ["llm_request_submission_started"]


async def test_live_chat_timeout_before_first_text_delta_is_distinct_from_sse_start(monkeypatch):
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def reasoning_only(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}\n\n'
        )
        await asyncio.sleep(0.15)
        return response

    app = web.Application()
    app.router.add_post("/chat", reasoning_only)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    async with TestServer(app, host="127.0.0.1") as server:
        selected = replace(
            spec("llm"),
            endpoint=str(server.make_url("/chat")),
            allow_live=True,
            request_limit=None,
            timeout_seconds=0.05,
        )
        transport = LiveTransport(selected)
        reasoning = ChatReasoning(selected, transport)
        observed = []
        reasoning.on_event = observed.append
        try:
            with pytest.raises(
                ProviderError, match="timed out waiting for the next SSE event"
            ) as error:
                _ = [text async for text in reasoning.respond([{"role": "user", "content": "Hi."}])]
        finally:
            await transport.aclose()

    assert error.value.error_boundary == "first_text_delta"
    assert observed == [
        "llm_request_submission_started",
        "llm_response_headers_received",
        "llm_first_sse_event",
    ]


async def test_groq_file_asr_multipart_and_fragmented_response_has_no_fake_partials(monkeypatch):
    import io
    import wave
    from dataclasses import replace
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from me3_voice.remote import GroqWhisperASR, make_provider

    captured = {}

    async def transcribe(request):
        assert request.headers["Authorization"] == "Bearer offline-contract-key"
        form = await request.multipart()
        async for part in form:
            captured[part.name] = await part.read()
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(b'{"text":')
        await asyncio.sleep(0)
        await response.write(b'"recognized offline fixture"}')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/transcribe", transcribe)
    monkeypatch.setenv("ME3_TEST_ONLY_KEY", "offline-contract-key")
    selected = replace(
        spec("asr"),
        protocol="groq-whisper-v1",
        endpoint="https://example.invalid/transcribe",
        streaming=False,
    )
    assert isinstance(make_provider("asr", selected), GroqWhisperASR)
    with pytest.raises(ConfigurationError):
        replace(selected, streaming=True).validate("asr", runnable=True)
    async with TestServer(app, host="127.0.0.1") as server:
        model = GroqWhisperASR(
            replace(
                selected,
                endpoint=str(server.make_url("/transcribe")),
                allow_live=True,
                request_limit=1,
            )
        )
        try:
            stream = model.open_stream()
            assert await stream.append(b"\1\0" * 512) is None
            assert model.transport.requests == 0
            result = await stream.finish()
            assert result.final and result.text == "recognized offline fixture"
            await stream.aclose()
        finally:
            await model.aclose()
    with wave.open(io.BytesIO(captured["file"]), "rb") as wav:
        assert wav.getparams()[:3] == (1, 2, 16000)
        assert wav.readframes(512) == b"\1\0" * 512
    assert captured["response_format"] == b"json"
    assert captured["language"] == b"en"


@pytest.mark.parametrize("model_id", ["google/gemma-4-31b-it:free", "openrouter/free"])
async def test_openrouter_free_reasoning_adds_zero_price_guard_and_rejects_paid_model(model_id):
    from dataclasses import replace
    from me3_voice.remote import ChatReasoning

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    free = replace(spec("llm"), endpoint=endpoint, model=model_id)
    free.validate("llm", runnable=True)
    with pytest.raises(ConfigurationError):
        replace(free, model="google/gemma-4-31b-it").validate("llm", runnable=True)

    class Peer:
        def __init__(self):
            self.body = None

        async def events(self, body, *, total_timeout=True):
            self.body = body
            self.total_timeout = total_timeout
            yield {
                "choices": [{"index": 0, "delta": {"content": "Hello."}, "finish_reason": "stop"}]
            }

    peer = Peer()
    model = ChatReasoning(free, peer)
    assert [part async for part in model.respond([{"role": "user", "content": "Hi"}])] == ["Hello."]
    assert peer.body["provider"] == {
        "max_price": {"prompt": 0, "completion": 0},
        "require_parameters": True,
        "allow_fallbacks": True,
    }
    assert peer.body["model"] == model_id


async def test_deepgram_partial_arrives_between_audio_appends_via_mcp_event_read():
    peer = Peer()
    service = SpeechService(DeepgramASR(spec("asr"), Wire(peer)), DeepgramTTS(spec("tts"), Wire()))
    identifier = (await service.open("late-partial", 1))["operation_id"]
    await service.append(identifier, 0, encode_pcm(b"\1\0" * 512))
    peer.queue.put_nowait(
        {
            "type": "Results",
            "is_final": False,
            "channel": {"alternatives": [{"transcript": "arrived while paused"}]},
        }
    )
    await peer.received.wait()
    row = await service.events(identifier)
    assert row["partial"]["text"] == "arrived while paused" and not row["partial"]["final"]
    assert (await service.events(identifier))["partial"] is None
    # No second audio append was needed to observe the new hypothesis.
    assert len([x for x in peer.sent if isinstance(x, bytes)]) == 1
    await service.cancel(identifier)
    assert peer.closed
