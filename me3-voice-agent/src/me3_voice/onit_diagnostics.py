"""Payload-free bounded worker stages, independent of result delivery."""

import json
import os
import time
from pathlib import Path

STAGES = {
    "worker_dispatch_started",
    "worker_received",
    "request_submission",
    "tool_result_submission",
    "response_headers",
    "connection_established",
    "http_request_headers_sent",
    "http_request_body_sent",
    "response_body_started",
    "response_body_complete",
    "tool_invoked",
    "tool_execution_started",
    "tool_execution_complete",
    "worker_complete",
    "worker_no_output",
    "transport_failed",
}


class StageJournal:
    def __init__(self, directory, tail=32):
        if not 4 <= tail <= 128:
            raise ValueError("invalid stage retention")
        self.path = Path(directory) / "worker-stages.json"
        self.tail = tail
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"schema_version": 1, "attempts": 0, "events": [], "event_count": 0}
        )

    def record(self, stage, attempt=None, *, tool_name=None, error_type=None):
        if stage not in STAGES:
            raise ValueError("invalid worker stage")
        event = {"stage": stage, "monotonic": time.monotonic()}
        if error_type is not None:
            if error_type not in {
                "ReadTimeout",
                "ConnectTimeout",
                "WriteTimeout",
                "PoolTimeout",
                "ConnectError",
                "ReadError",
                "RemoteProtocolError",
                "TransportError",
            }:
                raise ValueError("invalid diagnostic error type")
            event["error_type"] = error_type
        if tool_name is not None:
            if tool_name not in {"write_artifact", "create_project_directory"}:
                raise ValueError("invalid diagnostic tool name")
            event["tool_name"] = tool_name
        if attempt is not None:
            self.data["attempts"] = max(self.data["attempts"], attempt)
            event["attempt"] = attempt
        self.data["event_count"] += 1
        self.data["events"] = (self.data["events"] + [event])[-self.tail :]
        self.data["last_stage"] = stage
        self.data["truncated"] = self.data["event_count"] > len(self.data["events"])
        temp = self.path.with_suffix(".tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.path)


class ObservedBody:
    def __init__(self, stream, journal, attempt):
        self.stream, self.journal, self.attempt = stream, journal, attempt

    async def __aiter__(self):
        first = True
        async for chunk in self.stream:
            if first and chunk:
                self.journal.record("response_body_started", self.attempt)
                first = False
            yield chunk
        self.journal.record("response_body_complete", self.attempt)

    async def aclose(self):
        await self.stream.aclose()


async def send_observed(
    transport, request, journal, attempt, *, tool_result=False, wrapper=ObservedBody
):
    journal.record("tool_result_submission" if tool_result else "request_submission", attempt)

    async def trace(event, info):
        # Observe completion names only; info may contain credentials/payloads.
        stages = {
            "connection.connect_tcp.complete": "connection_established",
            "http11.send_request_headers.complete": "http_request_headers_sent",
            "http11.send_request_body.complete": "http_request_body_sent",
            "http2.send_request_headers.complete": "http_request_headers_sent",
            "http2.send_request_body.complete": "http_request_body_sent",
        }
        if event in stages:
            journal.record(stages[event], attempt)

    if hasattr(request, "extensions"):
        request.extensions["trace"] = trace
    try:
        response = await transport.handle_async_request(request)
    except Exception as exc:
        kind = type(exc).__name__
        if kind not in {
            "ReadTimeout",
            "ConnectTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "ConnectError",
            "ReadError",
            "RemoteProtocolError",
        }:
            kind = "TransportError"
        journal.record("transport_failed", attempt, error_type=kind)
        raise
    journal.record("response_headers", attempt)
    if getattr(response, "is_stream_consumed", False):
        # Deterministic transports may return an already-buffered HTTPX response.
        if response.content:
            journal.record("response_body_started", attempt)
        journal.record("response_body_complete", attempt)
    else:
        response.stream = wrapper(response.stream, journal, attempt)
    return response
