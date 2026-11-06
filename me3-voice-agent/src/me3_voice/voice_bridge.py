"""Loopback browser audio bridge for the existing Swift Session path.

The browser owns microphone capture, echo-cancellation request and audio output.
This bridge owns wake/VAD, MCP speech services, reasoning, transcript and tasks.
Browser playback acknowledgments are software callbacks, not acoustic evidence.
"""

import argparse
import asyncio
import errno
import fcntl
import json
import logging
import time
import os
import re
import struct
from contextlib import AsyncExitStack, suppress
from pathlib import Path

from aiohttp import WSMsgType, web

from .audio import SileroVAD
from .artifacts import output_path
from .configuration import load_backends
from .credentials import load_secrets
from .dashboard import render, render_audio_probe, snapshot
from .harness import PipecatAudioInput
from .music import MusicController, MusicUnavailable
from .readiness import build_wake
from .recording_lab import add_recording_routes
from .remote import LiveTransport, make_provider
from .runtime import AudioApplication
from .session import Session
from .speech import SpeechClient
from .telemetry import TraceJournal, summarize
from .transcript import Transcript, TranscriptPublisher
from .devices import load_device_service
from .tools import MockTaskExecutor


def music_ui_control_fields(value):
    if (
        set(value) != {"type", "action", "outcome"}
        or value.get("action")
        not in {
            "pause",
            "resume",
            "stop",
            "next",
            "previous",
            "seek",
            "volume_up",
            "volume_down",
            "volume_set",
        }
        or value.get("outcome") not in {"accepted", "failed"}
    ):
        raise ValueError("invalid music control telemetry")
    return {"action": value["action"], "outcome": value["outcome"], "audibility_verified": False}


def music_playback_observed_fields(value):
    if (
        set(value) != {"type", "status", "source", "track_changed"}
        or value.get("status") not in {"playing", "paused", "not_playing", "buffering"}
        or value.get("source") not in {"web_playback_sdk", "spotify_api"}
        or type(value.get("track_changed")) is not bool
    ):
        raise ValueError("invalid music playback observation")
    return {
        "status": value["status"],
        "source": value["source"],
        "track_changed": value["track_changed"],
    }


class MusicDuckingCoordinator:
    """Serialize opt-in duck/restore decisions against the live voice lifecycle."""

    def __init__(self, session, music):
        self.session, self.music = session, music
        self.tasks = set()
        self.active_turn = False
        self.closed = False

    def on_event(self, event):
        if not self.music.ducking_enabled or self.closed:
            return
        name = event.get("event")
        if name in {"wake_detected_file", "vad_onset_file"}:
            self.active_turn = True
            self._spawn(self._duck())
        elif (
            name == "music_playback_observed"
            and event.get("status") == "playing"
            and self.active_turn
        ):
            # A play command can start music after the speech-onset duck check
            # has already returned "not playing". Recheck on the confirmed
            # player event so music also ducks under this turn's spoken receipt.
            self._spawn(self._duck())
        elif name in {"turn_completed", "turn_error", "session_exit", "explicit_exit"}:
            self.active_turn = False
            self._spawn(self._restore())
        elif name in {"playback_stopped_browser", "playback_stopped_mock"} and not event.get(
            "input_following"
        ):
            self.active_turn = False
            self._spawn(self._restore())

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def _report(self, result):
        outcome = result.get("outcome", "failed")
        reason = result.get("reason")
        allowed_reasons = {"target_not_active", "volume_control_unavailable", "volume_unavailable"}
        self.session.event(
            "music_duck",
            outcome=outcome
            if outcome
            in {
                "confirmed",
                "restored",
                "uncertain",
                "unsupported",
                "not_playing",
                "already_ducked",
                "unchanged",
                "preserved_external_change",
                "failed",
            }
            else "failed",
            reason=reason if reason in allowed_reasons else None,
        )

    async def _duck(self):
        if not self.active_turn:
            return
        try:
            result = await self.music.begin_duck()
        except Exception:
            result = {"outcome": "failed"}
        if self.active_turn and result.get("outcome") not in {"disabled", "unavailable"}:
            self._report(result)
        elif not self.active_turn:
            await self._restore()

    async def _restore(self):
        if self.active_turn or not self.music.duck_state:
            return
        try:
            result = await self.music.restore_duck()
        except Exception:
            result = {"outcome": "uncertain"}
        self._report(result)

    async def close(self):
        self.closed = True
        self.active_turn = False
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        await self._restore()


def local_request(request):
    """Reject cross-origin websocket and non-loopback forwarded authorities."""
    host = request.headers.get("Host", "")
    if not re.fullmatch(r"(?:localhost|127\.0\.0\.1):[0-9]{1,5}", host):
        raise web.HTTPForbidden()
    if not 1 <= int(host.rsplit(":", 1)[1]) <= 65535:
        raise web.HTTPForbidden()
    origin = request.headers.get("Origin")
    if origin and origin != "http://" + host:
        raise web.HTTPForbidden()


class BrowserSink:
    """At most eight browser-scheduled chunks; confirm a segment after onended ACKs."""

    def __init__(self, ws):
        self.delivery = "sink_confirmed"
        self.ws = ws
        self.send_lock = asyncio.Lock()
        self.pending = set()
        self.changed = asyncio.Event()
        self.sequence = 0
        self.closed = False

    async def send(self, value):
        async with self.send_lock:
            await self.ws.send_json(value)

    async def __call__(self, pcm):
        if self.closed:
            raise RuntimeError("browser audio closed")
        async with asyncio.timeout(10):
            while len(self.pending) >= 8:
                self.changed.clear()
                await self.changed.wait()
        self.sequence += 1
        number = self.sequence
        self.pending.add(number)
        try:
            async with self.send_lock:
                await self.ws.send_bytes(struct.pack("<I", number) + pcm)
        except BaseException:
            self.pending.discard(number)
            self.changed.set()
            raise

    def acknowledge(self, number):
        if type(number) is int and number in self.pending:
            self.pending.remove(number)
            self.changed.set()

    async def flush(self):
        async with asyncio.timeout(30):
            while self.pending:
                self.changed.clear()
                await self.changed.wait()

    async def stop(self):
        self.pending.clear()
        self.changed.set()
        if not self.ws.closed:
            with suppress(ConnectionError, RuntimeError):
                await self.send({"type": "audio_stop"})

    async def close(self):
        self.closed = True
        await self.stop()


class VoiceRuntimeServices:
    """Lazily opened services shared across browser reconnects in one bridge run.

    Provider request limits, MCP subprocesses, model connections and task state
    therefore remain attached to the bridge lifetime rather than resetting on
    each WebSocket connection.
    """

    def __init__(self, backends, credentials, wake_args):
        self.backends, self.credentials, self.wake_args = backends, credentials, wake_args
        self.speech = self.reasoning = self.tasks = None
        self.devices = None
        self.ui_device_commands = {}
        self.music = MusicController.configured()
        self.speech_task = self.speech_ready = self.speech_close = None
        self.open_lock = asyncio.Lock()

    async def _own_speech_lifecycle(self):
        """Enter and exit MCP stdio scopes on their owning asyncio task."""
        speech = SpeechClient(
            backends=self.backends,
            credentials=self.credentials,
            timeout=max(spec.timeout_seconds for spec in self.backends.values()),
        )
        try:
            await speech.__aenter__()
        except BaseException as exc:
            if self.speech_ready and not self.speech_ready.done():
                self.speech_ready.set_exception(exc)
            raise
        self.speech = speech
        if self.speech_ready and not self.speech_ready.done():
            self.speech_ready.set_result(speech)
        await self.speech_close.wait()
        await speech.__aexit__(None, None, None)

    async def open(self):
        async with self.open_lock:
            if self.speech is None:
                if self.speech_task is None:
                    self.speech_ready = asyncio.get_running_loop().create_future()
                    self.speech_close = asyncio.Event()
                    self.speech_task = asyncio.create_task(self._own_speech_lifecycle())
                await self.speech_ready
            if self.tasks is None:
                if self.devices is None:
                    self.devices = load_device_service(
                        getattr(self.wake_args, "device_config", None)
                    )
                onit_python = getattr(self.wake_args, "onit_python", None)
                onit_primary = getattr(self.wake_args, "onit_primary", False)
                if onit_python:
                    from .onit_tasks import OnitPrimaryReasoning, OnitTaskExecutor

                    self.tasks = OnitTaskExecutor(
                        onit_python,
                        self.backends["llm"],
                        self.credentials,
                        max_requests=self.wake_args.onit_max_requests,
                        tavily_max_credits=getattr(self.wake_args, "onit_tavily_credits", 0),
                        github_readonly=getattr(self.wake_args, "onit_github_readonly", False),
                        project_output_root=getattr(self.wake_args, "onit_output_root", None),
                        reasoning_effort=getattr(self.wake_args, "onit_reasoning_effort", None),
                        streaming=not getattr(self.wake_args, "onit_no_stream", False),
                        max_output_tokens=getattr(self.wake_args, "onit_max_output_tokens", None)
                        or 512,
                        music=self.music,
                        devices=self.devices,
                    )
                    await self.tasks.prepare()
                    if onit_primary:
                        self.reasoning = OnitPrimaryReasoning()
                else:
                    self.tasks = MockTaskExecutor(music=self.music, devices=self.devices)
            receipt_path = getattr(self.wake_args, "receipt_store", None)
            receipt_target = getattr(self.tasks, "mock", self.tasks)
            if receipt_path and getattr(receipt_target, "receipt_store", None) is None:
                from .receipts import ReceiptStore

                # A failed configuration remains failed on later Start attempts.
                receipt_target.receipt_store = ReceiptStore(receipt_path)
            if self.reasoning is None:
                self.reasoning = make_provider(
                    "llm",
                    self.backends["llm"],
                    credentials=self.credentials,
                    agent_enabled=getattr(self.wake_args, "onit_python", None) is not None,
                    agent_directory_enabled=getattr(self.wake_args, "onit_output_root", None)
                    is not None,
                    agent_tavily_enabled=getattr(self.wake_args, "onit_tavily_credits", 0) > 0,
                    agent_github_readonly=getattr(self.wake_args, "onit_github_readonly", False),
                    # Live providers use the typed shared task executor; the
                    # older direct mock-light functions are offline-only.
                    legacy_mock_tools=False,
                )
            return self.speech, self.reasoning, self.tasks

    def configured_device_target(self):
        path = getattr(self.wake_args, "device_config", None)
        if path is None:
            return "mock"
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "unavailable"
        return value.get("backend", "mock") if isinstance(value, dict) else "unavailable"

    async def dispatch_ui_device_action(self, session, command_id, action, value):
        from .providers import ToolCall

        args = (action, value)
        previous = self.ui_device_commands.get(command_id)
        if previous is not None:
            if previous != args:
                raise ValueError("UI command ID was already used for another action")
            return {"status": "duplicate", "command_id": command_id}
        if len(self.ui_device_commands) >= 128:
            raise ValueError("UI device command ledger is full for this bridge run")
        arguments = {"action": action}
        if value is not None:
            arguments["value"] = value
        call = ToolCall(command_id, "run_device_action", arguments)
        await session.submit_ui_device_action(call)
        self.ui_device_commands[command_id] = args
        return {"status": "accepted", "command_id": command_id}

    def request_budget(self):
        transport = getattr(self.reasoning, "transport", None)
        spec = getattr(transport, "spec", None)
        requests = getattr(transport, "requests", None)
        if spec is None or type(requests) is not int:
            return None
        return {
            "used": requests,
            "limit": spec.request_limit,
            "remaining": max(0, spec.request_limit - requests)
            if spec.request_limit is not None
            else None,
            "scope": "bridge runtime" if spec.request_limit is not None else "provider enforced",
        }

    async def read_artifact(self, task_id, name):
        if not self.tasks or not hasattr(self.tasks, "read_artifact"):
            raise FileNotFoundError
        return await self.tasks.read_artifact(task_id, name)

    async def close(self):
        failure = None
        closers = []
        if self.tasks:
            closers.append(self.tasks.close)
        if self.devices:
            closers.append(self.devices.close)
        closers.append(self.music.close)
        if self.reasoning and hasattr(self.reasoning, "aclose"):
            closers.append(self.reasoning.aclose)
        if self.speech_close:
            self.speech_close.set()
        if self.speech_task:
            closers.append(lambda: asyncio.gather(self.speech_task, return_exceptions=False))
        for close in closers:
            try:
                await close()
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure:
            raise failure


async def publish_state(ws, session, transcript, sink, request_budget, journal=None):
    revision, state, event_index, status_index = -1, None, 0, 0
    last_checkpoint = 0
    while not ws.closed:
        if journal and time.monotonic() - last_checkpoint >= 1:
            try:
                await asyncio.to_thread(journal.flush, tuple(session.events))
            except (OSError, ValueError) as exc:
                logging.error("Swift diagnostic checkpoint failed (%s)", type(exc).__name__)
                journal = None
            last_checkpoint = time.monotonic()
        while event_index < len(session.events):
            event = session.events[event_index]
            event_index += 1
            data = {
                key: event[key]
                for key in (
                    "event",
                    "run_id",
                    "session_id",
                    "turn_id",
                    "tool_call_id",
                    "state",
                    "error_code",
                    "error_role",
                    "error_stage",
                    "error_boundary",
                    "outcome",
                    "reason",
                    "input_following",
                    "llm_requests_used",
                    "llm_request_limit",
                )
                if key in event
            }
            if request_budget and event["event"] in {"llm_request_finished", "turn_error"}:
                data["reasoning_budget"] = request_budget()
            await sink.send({"type": "runtime_event", "sequence": event_index, "data": data})
        if ws.closed:
            break
        await session.tick()
        next_state = session.state.value
        identity = (
            next_state,
            session.session_id,
            session.turn,
            session.input_open,
            int(session.engagement_snapshot()["remaining_seconds"] or 0),
        )
        if identity != state:
            state = identity
            status_index += 1
            await sink.send(
                {
                    "type": "status",
                    "state": "wake_listening" if next_state == "idle" else next_state,
                    "run_id": session.run_id,
                    "session_id": session.session_id,
                    "turn_id": session.turn,
                    "input_open": session.input_open,
                    "capture_active": True,
                    "engagement": session.engagement_snapshot(),
                    "event_sequence": event_index,
                    "status_sequence": status_index,
                }
            )
        if transcript.revision != revision and transcript.session_id != "unstarted":
            revision = transcript.revision
            await sink.send({"type": "transcript", "data": transcript.snapshot()})
        await asyncio.sleep(0.05)


def build_app(root, backends, credentials, wake_args, *, trace_dir=None):
    app = web.Application(client_max_size=16384)
    add_recording_routes(app, root)
    busy = False
    services = VoiceRuntimeServices(backends, credentials, wake_args)
    last_music_command = 0.0

    async def close_services(_app):
        await services.close()

    app.on_cleanup.append(close_services)

    async def page(request):
        local_request(request)
        body = render(snapshot(root), live=True)
        return web.Response(
            text=body,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )

    async def audio_probe(request):
        local_request(request)
        body, policy = render_audio_probe()
        return web.Response(
            text=body,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": policy,
                "Permissions-Policy": "microphone=(self)",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
            },
        )

    async def reports(request):
        local_request(request)
        return web.json_response(snapshot(root), headers={"Cache-Control": "no-store"})

    async def health(request):
        """Report bridge process reachability without implying audio readiness."""
        local_request(request)
        return web.json_response(
            {
                "bridge": "ready",
                "voice_session": "active" if busy else "idle",
                "speech_services": "initialized" if services.speech is not None else "lazy",
                "session_resume": False,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def task_artifact(request):
        local_request(request)
        task_id, name = request.match_info["task_id"], request.match_info["name"]
        try:
            content = await services.read_artifact(task_id, name)
        except (FileNotFoundError, OSError, ValueError):
            raise web.HTTPNotFound(text="Task artifact unavailable") from None
        return web.Response(
            body=content,
            content_type="text/plain",
            charset="utf-8",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="{name}"',
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    async def capability(request):
        local_request(request)
        return web.json_response(
            {
                "available": True,
                "capture": "browser-pcm16",
                "playback": "browser-acknowledged-pcm16",
                "wake": "Hey Swift" if wake_args.kws_assets else "configured model",
                "asr_partial": backends["asr"].streaming,
                "aec": "browser-requested-unverified",
                "task_artifact_access": bool(
                    services.tasks and hasattr(services.tasks, "read_artifact")
                ),
                "devices": {
                    "target": services.configured_device_target(),
                    "status": "not_initialized",
                    "available": None,
                    "physical_readback": False,
                },
                "music": services.music.capabilities(),
            },
            headers={"Cache-Control": "no-store"},
        )

    async def music_token(request):
        local_request(request)
        origin = request.headers.get("Origin")
        expected_origin = "http://" + request.headers.get("Host", "")
        if origin and origin != expected_origin:
            raise web.HTTPForbidden()
        if request.headers.get("Sec-Fetch-Site") not in {None, "same-origin"}:
            raise web.HTTPForbidden()
        try:
            token = await services.music.browser_access_token()
        except MusicUnavailable as exc:
            return web.json_response(
                {"available": False, "code": exc.code, "message": str(exc)},
                status=409 if exc.code == "music_streaming_scope_required" else 503,
                headers={"Cache-Control": "no-store, private", "Pragma": "no-cache"},
            )
        return web.json_response(
            {"available": True, "access_token": token},
            headers={
                "Cache-Control": "no-store, private",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cross-Origin-Resource-Policy": "same-origin",
            },
        )

    async def music_device(request):
        local_request(request)
        if request.headers.get("Origin") != "http://" + request.headers.get("Host", ""):
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            value = await request.json()
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="Invalid player registration") from None
        if (
            not isinstance(value, dict)
            or set(value) != {"type", "device_id"}
            or value.get("type") not in {"ready", "not_ready"}
            or not isinstance(value.get("device_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value["device_id"])
        ):
            raise web.HTTPBadRequest(text="Invalid player registration")
        if value["type"] == "ready":
            try:
                services.music.register_browser_device(value["device_id"])
            except (ValueError, MusicUnavailable) as exc:
                raise web.HTTPConflict(text=str(exc)[:160]) from None
        else:
            services.music.unregister_browser_device(value["device_id"])
        return web.json_response(
            services.music.capabilities(),
            headers={"Cache-Control": "no-store, private"},
        )

    async def music_status(request):
        local_request(request)
        try:
            state = await services.music.current_state()
        except MusicUnavailable as exc:
            return web.json_response(
                {
                    "configured": True,
                    "status": "unavailable",
                    "code": exc.code,
                    "message": str(exc)[:160],
                    "audible": None,
                },
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(state, headers={"Cache-Control": "no-store"})

    async def music_control(request):
        nonlocal last_music_command
        import time

        local_request(request)
        if request.headers.get("Origin") != "http://" + request.headers.get("Host", ""):
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            value = await request.json()
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="Invalid Spotify command") from None
        action = value.get("action") if isinstance(value, dict) else None
        valid_fields = (
            {"command_id", "action", "position_ms"}
            if action == "seek"
            else {"command_id", "action", "volume_percent"}
            if action == "volume_set"
            else {"command_id", "action"}
        )
        if (
            not isinstance(value, dict)
            or set(value) != valid_fields
            or not isinstance(value.get("command_id"), str)
            or not re.fullmatch(r"ui-[a-f0-9]{32}", value["command_id"])
            or value.get("action")
            not in {
                "pause",
                "resume",
                "stop",
                "next",
                "previous",
                "seek",
                "volume_up",
                "volume_down",
                "volume_set",
            }
            or value.get("action") == "seek"
            and (
                type(value.get("position_ms")) is not int
                or not 0 <= value["position_ms"] <= 7_200_000
            )
            or value.get("action") == "volume_set"
            and (
                type(value.get("volume_percent")) is not int
                or not 0 <= value["volume_percent"] <= 100
            )
        ):
            raise web.HTTPBadRequest(text="Invalid Spotify command")
        now = time.monotonic()
        if now - last_music_command < 0.2:
            raise web.HTTPTooManyRequests(text="Wait for the current Spotify command")
        last_music_command = now
        try:
            receipt = await services.music.ui_control(
                value["command_id"],
                value["action"],
                value.get("position_ms"),
                value.get("volume_percent"),
            )
            state = await services.music.current_state(force=True)
        except (MusicUnavailable, ValueError) as exc:
            return web.json_response(
                {
                    "available": False,
                    "code": getattr(exc, "code", "music_invalid_command"),
                    "message": str(exc)[:160],
                },
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {"available": True, "receipt": receipt, "state": state},
            headers={"Cache-Control": "no-store"},
        )

    async def voice(request):
        nonlocal busy
        local_request(request)
        if busy:
            raise web.HTTPConflict(text="voice session already active")
        busy = True
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
        journal = None
        session = None
        diagnostic = None
        try:
            await ws.prepare(request)
            sink = BrowserSink(ws)
            message = await asyncio.wait_for(ws.receive(), 10)
            if message.type != WSMsgType.TEXT:
                raise ValueError("expected audio format first")
            hello = json.loads(message.data)
            if (
                not isinstance(hello, dict)
                or hello.get("type") != "hello"
                or hello.get("rate") not in (16000, 44100, 48000)
                or type(hello.get("rate")) is not int
                or hello.get("channels") != 1
                or not {"type", "rate", "channels"} <= set(hello)
                or set(hello)
                - {
                    "type",
                    "rate",
                    "channels",
                    "aec_reported",
                    "conversation_mode",
                    "capture_source",
                }
                or hello.get("capture_source", "browser") not in {"browser", "pipewire"}
                or hello.get("aec_reported") is not None
                and type(hello["aec_reported"]) is not bool
            ):
                raise ValueError("unsupported audio format")
            if hello.get("conversation_mode", "followup") not in {"single", "followup", "extended"}:
                raise ValueError("invalid conversation mode")
            wake = build_wake(wake_args)
            if wake is None:
                raise ValueError("a real wake model is required for live capture")
            transcript = Transcript(
                mode="HPC-live"
                if any(s.backend == "external" for s in backends.values())
                else "mock",
                backend_modes={role: spec.backend for role, spec in backends.items()},
                playback="browser",
            )
            speech, reasoning, tasks = await services.open()
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(
                    TranscriptPublisher(transcript, archive_dir=Path("var/results/voice-sessions"))
                )
                session = Session(
                    speech,
                    reasoning,
                    sink,
                    transcript=transcript,
                    # While a voice turn is thinking/speaking, keep it active
                    # until a provider result, transport failure, or barge-in.
                    # Idle expiry only applies while listening without a turn.
                    idle_seconds=60,
                    conversation_mode=hello.get("conversation_mode", "followup"),
                    followup_seconds=getattr(wake_args, "followup_seconds", 7),
                    answer_seconds=getattr(wake_args, "answer_seconds", 18),
                    extended_seconds=getattr(wake_args, "extended_seconds", 20),
                    turn_timeout=None,
                    llm_timeout=None,
                    tasks=tasks,
                    close_tasks=False,
                    execution_mode="HPC-live"
                    if any(s.backend == "external" for s in backends.values())
                    else "mock",
                    direct_clock_enabled=getattr(wake_args, "direct_clock", False),
                )
                if getattr(wake_args, "diagnostic_audio", False):
                    from .diagnostics import DiagnosticCapture

                    diagnostic = DiagnosticCapture(
                        root,
                        session.run_id,
                        hello["rate"],
                        max_seconds=getattr(wake_args, "diagnostic_max_seconds", 60),
                    )
                    session.speech = diagnostic.wrap(speech)
                ducker = MusicDuckingCoordinator(session, services.music)
                if trace_dir:
                    journal = TraceJournal(trace_dir / f"voice-{session.run_id}.jsonl")
                session.on_event = ducker.on_event
                if hasattr(tasks, "on_event"):
                    tasks.on_event = lambda name, **fields: session.event(name, **fields)
                stack.push_async_callback(ducker.close)
                await stack.enter_async_context(session)
                if hasattr(reasoning, "on_event"):
                    reasoning.on_event = session.event
                controller = AudioApplication(
                    session,
                    wake,
                    SileroVAD(0.7),
                    rate=hello["rate"],
                    channels=1,
                    silence_frames=32,
                    # Require sustained evidence before cutting browser playback.
                    # 8/12 VAD frames is about 256 ms at this 32 ms frame size;
                    # it reduces short echo/noise bursts but is not AEC.
                    barge_min_voiced_frames=8,
                    barge_window_frames=12,
                )
                async with PipecatAudioInput(controller) as ingress:
                    await sink.send(
                        {
                            "type": "ready",
                            "run_id": session.run_id,
                            "session_id": session.session_id,
                            "mode": transcript.mode,
                            "reasoning_budget": services.request_budget(),
                            "provider_roles": {
                                role: (
                                    "Mock"
                                    if spec.backend == "mock"
                                    else "Groq"
                                    if spec.endpoint.startswith("https://api.groq.com/")
                                    else "University service"
                                    if spec.allow_private_http
                                    else "Configured API"
                                )
                                + " · "
                                + spec.model
                                for role, spec in backends.items()
                            },
                            "task_artifact_access": hasattr(tasks, "read_artifact"),
                            "devices": services.devices.capabilities(),
                            "diagnostic_audio": bool(diagnostic),
                        }
                    )
                    publisher = asyncio.create_task(
                        publish_state(
                            ws, session, transcript, sink, services.request_budget, journal
                        )
                    )
                    sequence = 0
                    try:
                        async for message in ws:
                            if message.type == WSMsgType.BINARY:
                                pcm = message.data
                                if not pcm or len(pcm) > 8192 or len(pcm) % 2:
                                    raise ValueError("invalid PCM frame")
                                if diagnostic:
                                    diagnostic.received(pcm)
                                await ingress.feed(pcm, sequence)
                                sequence += 1
                            elif message.type == WSMsgType.TEXT:
                                value = json.loads(message.data)
                                if not isinstance(value, dict):
                                    raise ValueError("invalid control message")
                                if value.get("type") == "audio_ack" and set(value) == {
                                    "type",
                                    "id",
                                }:
                                    sink.acknowledge(value["id"])
                                elif value.get("type") == "music_ui_control":
                                    session.event(
                                        "music_ui_control", **music_ui_control_fields(value)
                                    )
                                elif value.get("type") == "music_playback_observed":
                                    fields = music_playback_observed_fields(value)
                                    session.observe_music_playback(fields.pop("status"), **fields)
                                elif value.get("type") == "device_ui_action":
                                    if (
                                        not isinstance(value, dict)
                                        or set(value)
                                        not in (
                                            {"type", "command_id", "action"},
                                            {"type", "command_id", "action", "value"},
                                        )
                                        or not isinstance(value.get("command_id"), str)
                                        or not re.fullmatch(r"ui-[a-f0-9]{32}", value["command_id"])
                                        or value.get("action")
                                        not in {
                                            "light_on",
                                            "light_off",
                                            "set_color",
                                            "set_brightness",
                                        }
                                        or "value" in value
                                        and not isinstance(value["value"], str)
                                    ):
                                        raise ValueError("invalid UI device action")
                                    try:
                                        receipt = await services.dispatch_ui_device_action(
                                            session,
                                            value["command_id"],
                                            value["action"],
                                            value.get("value"),
                                        )
                                    except ValueError as exc:
                                        receipt = {
                                            "status": "unavailable",
                                            "command_id": value["command_id"],
                                            "message": str(exc)[:120],
                                        }
                                    await sink.send({"type": "device_action_ack", **receipt})
                                elif value == {"type": "interrupt"}:
                                    await session.interrupt()
                                elif value == {"type": "stop"}:
                                    break
                                else:
                                    raise ValueError("invalid control message")
                            elif message.type in {
                                WSMsgType.ERROR,
                                WSMsgType.CLOSE,
                                WSMsgType.CLOSED,
                            }:
                                break
                    finally:
                        publisher.cancel()
                        await asyncio.gather(publisher, return_exceptions=True)
            if trace_dir:
                report = output_path(str(trace_dir / f"voice-{session.run_id}.json"))
                await asyncio.to_thread(journal.flush, tuple(session.events))
                with report.open("x") as stream:
                    data = summarize(session.events)
                    data["runtime_io"] = {
                        "browser_aec_reported": hello.get("aec_reported"),
                        "input_stats": controller.pre.input_stats.report(),
                        "normalized_stats": controller.pre.output_stats.report(),
                        "frontend": controller.monitor.report(),
                        "endpoint_events": list(controller.source.events)[-100:],
                        "endpoint_events_truncated": controller.source.event_count > 100,
                        "input_frames": ingress.processor.frames,
                        "playback_packets": sink.sequence,
                        "capture_source": hello.get("capture_source", "browser"),
                        "limitations": "Browser transport counters; capture/render devices and AEC are not verified by this report.",
                    }
                    json.dump(data, stream, allow_nan=False)
                    stream.write("\n")
            return ws
        except (ValueError, json.JSONDecodeError, TimeoutError) as exc:
            if not ws.closed:
                await ws.send_json({"type": "error", "reason": str(exc)[:120]})
            return ws
        finally:
            busy = False
            if diagnostic:
                try:
                    await asyncio.to_thread(diagnostic.save)
                except (OSError, ValueError) as exc:
                    logging.error("Swift audio evidence save failed (%s)", type(exc).__name__)
            if journal and session:
                try:
                    await asyncio.to_thread(journal.flush, tuple(session.events))
                except (OSError, ValueError) as exc:
                    logging.error("Swift diagnostic checkpoint failed (%s)", type(exc).__name__)
            if not ws.closed:
                await ws.close()

    app.router.add_get("/", page)
    app.router.add_get("/audio-probe", audio_probe)
    app.router.add_get("/api/snapshot", reports)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/voice/tasks/{task_id}/artifacts/{name}", task_artifact)
    app.router.add_get("/api/voice/capabilities", capability)
    app.router.add_get("/api/voice/music/token", music_token)
    app.router.add_get("/api/voice/music/status", music_status)
    app.router.add_post("/api/voice/music/device", music_device)
    app.router.add_post("/api/voice/music/control", music_control)
    app.router.add_get("/api/voice/ws", voice)
    return app


def serve(app, host, port):
    """Start the local bridge and turn common bind failures into safe guidance."""
    try:
        web.run_app(app, host=host, port=port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Swift could not bind {host}:{port}: address already in use. "
                "Inspect the listener and stop only a Swift process you own, or choose another --port."
            ) from None
        if isinstance(exc, PermissionError) or exc.errno in {errno.EACCES, errno.EPERM}:
            raise SystemExit(
                f"Swift could not bind {host}:{port}: permission denied. "
                "Use an unprivileged loopback port above 1024."
            ) from None
        raise


def loopback_host(host):
    """Accept only stable IPv4 loopback names supported by request validation."""
    return host in {"127.0.0.1", "localhost"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, help="omit for deterministic mock providers")
    parser.add_argument(
        "--device-config",
        type=Path,
        help="optional private device config under var/cache/; absent config is mock-only",
    )
    parser.add_argument("--secrets-file", type=Path)
    parser.add_argument(
        "--llm-secrets-file",
        type=Path,
        help="private LLM credential file, overrides only the LLM key",
    )
    parser.add_argument(
        "--onit-secrets-file",
        type=Path,
        help="optional private project Onit integrations file; used only for explicitly enabled tools",
    )
    parser.add_argument(
        "--onit-python",
        type=Path,
        help="opt-in existing Onit Python; task tools are scoped and optional integrations default off",
    )
    parser.add_argument(
        "--onit-max-requests",
        type=int,
        help="optional per-run Onit request cap (unset delegates quota to the provider)",
    )
    parser.add_argument(
        "--onit-reasoning-effort",
        choices=("none", "low", "medium"),
        help="explicit reasoning effort for the allowlisted Groq Onit profile; omitted keeps the existing provider default",
    )
    parser.add_argument(
        "--onit-max-output-tokens",
        type=int,
        help="explicit Onit completion-token ceiling (128..4096); omitted keeps the existing 512-token ceiling",
    )
    parser.add_argument(
        "--onit-no-stream",
        action="store_true",
        help="disable provider streaming for Onit; omitted keeps the existing streaming behavior",
    )
    parser.add_argument(
        "--onit-primary",
        action="store_true",
        help="route general turns to Onit as reasoning owner; existing direct music intents stay local",
    )
    parser.add_argument(
        "--onit-output-root",
        type=Path,
        help="explicit existing project root under var/ or legacy sandbox parent; creates new child projects only",
    )
    parser.add_argument(
        "--onit-tavily-credits",
        type=int,
        default=0,
        help="maximum Tavily basic-search credits for this bridge run (0 disables web search; one basic search costs one credit)",
    )
    parser.add_argument(
        "--onit-github-readonly",
        action="store_true",
        help="enable bounded GitHub repository get/list only; no repository writes",
    )
    parser.add_argument(
        "--diagnostic-audio",
        action="store_true",
        help="opt in private bounded audio evidence after Start; 60 seconds raw/ASR retained, under var/recordings/diagnostics/",
    )
    parser.add_argument(
        "--receipt-store",
        type=Path,
        help="opt-in private SQLite device/music dispatch ledger under var/cache; pending records are never replayed",
    )
    parser.add_argument("--followup-seconds", type=float, default=7)
    parser.add_argument("--answer-seconds", type=float, default=18)
    parser.add_argument("--extended-seconds", type=float, default=20)
    parser.add_argument("--kws-assets", type=Path, required=True)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--kws-threshold", type=float, default=0.25)
    parser.add_argument(
        "--direct-clock",
        action="store_true",
        help="opt in exact local-clock requests to the existing typed device task; default off",
    )
    parser.add_argument("--kws-boost", type=float, default=1.0)
    parser.add_argument(
        "--host",
        default=os.environ.get("ME3_VOICE_HOST", "127.0.0.1"),
        help="loopback bind only (127.0.0.1 or localhost; default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--trace-dir", type=Path, default=Path("var/results"), help="ignored var/ output directory"
    )
    parser.add_argument(
        "--diagnostic-max-seconds",
        type=int,
        default=60,
        help="1..180 seconds retained per newly opted-in audio diagnostic stream",
    )
    args = parser.parse_args()
    if not 1 <= args.diagnostic_max_seconds <= 180:
        parser.error("--diagnostic-max-seconds must be 1..180")
    if args.device_config:
        candidate = (
            args.device_config
            if args.device_config.is_absolute()
            else Path.cwd() / args.device_config
        )
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(
            (Path.cwd() / "var" / "cache").resolve()
        ):
            parser.error("--device-config must be a regular file under project var/cache/")
    if not loopback_host(args.host):
        parser.error("--host must be 127.0.0.1 or localhost; public binds are disabled")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    args.wake_model = args.mel_model = args.embedding_model = None
    args.wake_threshold = 0.5
    output_path(str(args.trace_dir / "validation.json"))
    backends = load_backends(args.backend_config)
    if args.onit_max_requests is not None and not 1 <= args.onit_max_requests <= 8:
        parser.error("--onit-max-requests must be 1..8 when explicitly set")
    if args.onit_max_requests is not None and not args.onit_python:
        parser.error("--onit-max-requests requires --onit-python")
    if args.onit_reasoning_effort is not None and not args.onit_python:
        parser.error("--onit-reasoning-effort requires --onit-python")
    if args.onit_max_output_tokens is not None and not args.onit_python:
        parser.error("--onit-max-output-tokens requires --onit-python")
    if args.onit_max_output_tokens is not None and not 128 <= args.onit_max_output_tokens <= 4096:
        parser.error("--onit-max-output-tokens must be 128..4096")
    if args.onit_no_stream and not args.onit_python:
        parser.error("--onit-no-stream requires --onit-python")
    if args.onit_primary and not args.onit_python:
        parser.error("--onit-primary requires --onit-python")
    if args.onit_output_root and not args.onit_python:
        parser.error("--onit-output-root requires --onit-python")
    if not 0 <= args.onit_tavily_credits <= 20:
        parser.error("--onit-tavily-credits must be 0..20")
    if args.onit_tavily_credits and not args.onit_python:
        parser.error("--onit-tavily-credits requires --onit-python")
    if args.onit_github_readonly and not args.onit_python:
        parser.error("--onit-github-readonly requires --onit-python")
    if args.onit_secrets_file and not args.onit_python:
        parser.error("--onit-secrets-file requires --onit-python")
    if args.onit_output_root:
        from .artifacts import approved_project_root

        try:
            args.onit_output_root = approved_project_root(args.onit_output_root)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if args.onit_python:
        selected = backends["llm"]
        groq_models = {
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        }
        groq_onit = (
            selected.endpoint == "https://api.groq.com/openai/v1/chat/completions"
            and selected.model in groq_models
        )
        if not args.onit_python.is_absolute() or not args.onit_python.is_file():
            parser.error("--onit-python must point to an existing absolute Python interpreter")
        if args.onit_reasoning_effort is not None and not groq_onit:
            parser.error(
                "--onit-reasoning-effort is only supported for the allowlisted Groq Onit models"
            )
        if (
            selected.backend != "external"
            or selected.protocol != "chat-completions-v1"
            or not (
                (
                    selected.endpoint == "https://openrouter.ai/api/v1/chat/completions"
                    and (selected.model.endswith(":free") or selected.model == "openrouter/free")
                )
                or (
                    selected.endpoint == "http://202.92.159.240:8001/v1/chat/completions"
                    and selected.model == "Qwen/Qwen3.8-27B"
                    and selected.allow_private_http
                )
                or (groq_onit and selected.allow_live)
            )
            or not selected.allow_live
            or (
                selected.request_limit is not None
                and args.onit_max_requests is not None
                and args.onit_max_requests > selected.request_limit
            )
        ):
            parser.error(
                "Onit requires the documented DGX Qwen contract, exact free-only OpenRouter route, or allowlisted Groq model"
            )
    selected_keys = {spec.auth_env for spec in backends.values() if spec.backend == "external"}
    if args.onit_python:
        if args.onit_tavily_credits:
            selected_keys.add("ME3_ONIT_TAVILY_API_KEY")
        if args.onit_github_readonly:
            selected_keys.add("ME3_ONIT_GITHUB_TOKEN")
    credentials = (
        load_secrets(
            args.secrets_file,
            selected_keys,
        )
        if args.secrets_file
        else {}
    )
    if args.llm_secrets_file:
        credentials.update(load_secrets(args.llm_secrets_file, {backends["llm"].auth_env}))
    if args.onit_tavily_credits or args.onit_github_readonly:
        if not args.onit_secrets_file:
            parser.error("enabled Onit integrations require --onit-secrets-file")
        selected_onit_keys = set()
        if args.onit_tavily_credits:
            selected_onit_keys.add("ME3_ONIT_TAVILY_API_KEY")
        if args.onit_github_readonly:
            selected_onit_keys.add("ME3_ONIT_GITHUB_TOKEN")
        onit_credentials = load_secrets(args.onit_secrets_file, selected_onit_keys)
        if args.onit_tavily_credits and "ME3_ONIT_TAVILY_API_KEY" not in onit_credentials:
            parser.error("Tavily is enabled but its key is absent from --onit-secrets-file")
        if args.onit_github_readonly and "ME3_ONIT_GITHUB_TOKEN" not in onit_credentials:
            parser.error(
                "GitHub read-only access is enabled but its token is absent from --onit-secrets-file"
            )
        credentials.update(onit_credentials)
    for spec in backends.values():
        if spec.backend == "external":
            LiveTransport(spec, credentials).authorize()
    lock_path = Path("var/cache/voice-runtime.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another Swift voice runtime owns the current transcript")
        print(f"Swift voice bridge: http://{args.host}:{args.port} (loopback only)", flush=True)
        if args.diagnostic_audio:
            print(
                "Audio diagnostics enabled: up to 60 seconds retained after browser Start, private var/recordings/diagnostics/",
                flush=True,
            )
        print(
            "Provider modes: "
            + ", ".join(
                f"{role.upper()}={backends[role].backend}" for role in ("asr", "llm", "tts")
            ),
            flush=True,
        )
        serve(
            build_app(Path.cwd(), backends, credentials, args, trace_dir=args.trace_dir),
            args.host,
            args.port,
        )
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
