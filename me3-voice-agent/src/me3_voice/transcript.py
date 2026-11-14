"""Opt-in UI transcript projection, independent of reasoning history and metrics.

Partial ASR replaces a hypothesis. Assistant draft text is distinct from sink
acknowledgment; interruption removes unconfirmed text. No audio/word alignment
or actual acoustic delivery is inferred. The runtime remains the sole owner.
"""

import asyncio
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import output_path
from .async_utils import await_thread_future
from .contracts import TurnKey
from .devices import normalize_rgb_color

MAX_TEXT = 8000
MAX_CHARACTERS = 100000  # Also bounds ASCII-escaped non-BMP JSON below the report file limit.
MODES = {"mock", "HPC-live", "Pi-verified"}
STATES = {"idle", "listening", "thinking", "speaking", "interrupted", "ended"}
STATUSES = {"partial", "final", "interrupted", "error"}
DELIVERIES = {
    "asr_partial",
    "asr_final",
    "generated",
    "mock_sink_confirmed",
    "sink_confirmed",
    "benchmark_transport_acknowledged",
}


def identity_with_session(key, call_id):
    return f"{key.session_id}:{key.turn_id}:{call_id}"


TASK_STATES = {
    "running",
    "simulated",
    "accepted",
    "awaiting_input",
    "completed",
    "reported",
    "unavailable",
    "replaced",
    "cancelled",
    "failed",
    "unknown",
}
TASK_MODES = {"mock", "reported", "Pi-verified"}
TASK_KINDS = {
    "music_play",
    "music_playlist",
    "music_pause",
    "music_resume",
    "music_stop",
    "music_next",
    "music_previous",
    "music_volume_up",
    "music_volume_down",
    "music_volume_set",
    "light_on",
    "light_off",
    "set_color",
    "set_brightness",
    "set_timer",
    "set_alarm",
    "cancel_schedule",
    "set_temperature",
    "create_reminder",
    "list_reminders",
    "get_time",
    "get_weather",
    "call_demo",
    "message_demo",
    "agent",
}


def normalize_transcript(data):
    """Validate/allowlist one intentionally published transcript, never raw history."""
    if not isinstance(data, dict) or data.get("kind") != "conversation-transcript":
        raise ValueError("expected a typed conversation transcript")
    if data.get("schema_version") != 1 or data.get("display_authorized") is not True:
        raise ValueError("transcript display must be explicitly enabled")
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", session_id):
        raise ValueError("invalid transcript session")
    if data.get("mode") not in MODES or data.get("state") not in STATES:
        raise ValueError("invalid transcript mode/state")
    if type(data.get("revision")) is not int or data["revision"] < 0:
        raise ValueError("invalid transcript revision")
    if not isinstance(data.get("updated_utc"), str) or len(data["updated_utc"]) > 60:
        raise ValueError("invalid transcript timestamp")
    rows = data.get("messages")
    if not isinstance(rows, list) or len(rows) > 200:
        raise ValueError("transcript exceeds message bounds")
    messages, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get("role") not in {"user", "assistant"}:
            raise ValueError("transcript accepts user/assistant text only")
        turn = row.get("turn_id")
        if type(turn) is not int or turn < 1:
            raise ValueError("invalid transcript turn")
        identity = f"{turn}:{row['role']}"
        if row.get("id") != identity or identity in seen:
            raise ValueError("duplicate/invalid transcript message ID")
        seen.add(identity)
        text, confirmed = row.get("text"), row.get("confirmed_text")
        if any(not isinstance(t, str) or len(t) > MAX_TEXT for t in (text, confirmed)):
            raise ValueError("transcript text exceeds bounds")
        if row.get("status") not in STATUSES or row.get("delivery") not in DELIVERIES:
            raise ValueError("invalid transcript status/delivery")
        if row["role"] == "user":
            if (
                row["delivery"] not in {"asr_partial", "asr_final"}
                or row["status"] == "partial"
                and row["delivery"] != "asr_partial"
                or row["status"] == "final"
                and row["delivery"] != "asr_final"
            ):
                raise ValueError("invalid user hypothesis state")
            if confirmed:
                raise ValueError("ASR text is not playback acknowledgment")
        else:
            if "generation_complete" in row and type(row["generation_complete"]) is not bool:
                raise ValueError("invalid assistant generation state")
            if row["delivery"] in {"asr_partial", "asr_final"} or not text.startswith(confirmed):
                raise ValueError("invalid assistant delivery prefix")
            if row["status"] == "final" and text != confirmed:
                raise ValueError("completed assistant text must be playback-confirmed")
            if row["status"] in {"interrupted", "error"} and not text.startswith(confirmed):
                raise ValueError("confirmed assistant text must be a prefix of generated text")
            if row["status"] == "final" and row["delivery"] == "generated":
                raise ValueError("generated text is not confirmed delivery")
        normalized = {
            "id": identity,
            "turn_id": turn,
            "role": row["role"],
            "text": text,
            "confirmed_text": confirmed,
            "status": row["status"],
            "delivery": row["delivery"],
        }
        if row["role"] == "assistant":
            normalized["generation_complete"] = row.get(
                "generation_complete", row["status"] == "final"
            )
            response = row.get("response")
            if response is not None:
                if (
                    not isinstance(response, dict)
                    or not isinstance(response.get("response_id"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", response["response_id"])
                    or response.get("state")
                    not in {"pending", "completed", "interrupted", "superseded", "failed"}
                    or response.get("generation")
                    not in {"streaming", "completed", "interrupted", "failed", "unknown"}
                    or response.get("tts")
                    not in {"pending", "submitted", "completed", "interrupted", "failed", "unknown"}
                    or response.get("playback")
                    not in {
                        "pending",
                        "queued",
                        "browser_acknowledged",
                        "partially_acknowledged",
                        "not_acknowledged",
                        "unknown",
                    }
                    or response.get("archive") not in {"pending", "included"}
                    or type(response.get("packets")) is not int
                    or type(response.get("acks")) is not int
                    or not 0 <= response["acks"] <= response["packets"]
                    or response.get("superseded_by_turn_id") is not None
                    and (
                        type(response["superseded_by_turn_id"]) is not int
                        or response["superseded_by_turn_id"] < 1
                    )
                    or not isinstance(response.get("segments"), list)
                    or len(response["segments"]) > 50
                ):
                    raise ValueError("invalid assistant response lifecycle")
                for segment in response["segments"]:
                    if (
                        not isinstance(segment, dict)
                        or type(segment.get("segment_id")) is not int
                        or type(segment.get("characters")) is not int
                        or segment.get("state")
                        not in {"submitted", "completed", "cancelled", "failed", "unknown"}
                        or segment.get("text_sha256") is not None
                        and not re.fullmatch(r"[a-f0-9]{16}", segment["text_sha256"])
                    ):
                        raise ValueError("invalid assistant response segment")
                if response.get("interruption_reason") is not None and not isinstance(
                    response["interruption_reason"], str
                ):
                    raise ValueError("invalid assistant interruption reason")
                normalized["response"] = response
        messages.append(normalized)
    if sum(len(m["text"]) + len(m["confirmed_text"]) for m in messages) > MAX_CHARACTERS:
        raise ValueError("transcript exceeds total text bounds")
    task_rows = data.get("tasks", [])
    if not isinstance(task_rows, list) or len(task_rows) > 50:
        raise ValueError("transcript exceeds task bounds")
    tasks, task_ids = [], set()
    for row in task_rows:
        if not isinstance(row, dict):
            raise ValueError("invalid task receipt")
        turn, call_id = row.get("turn_id"), row.get("call_id")
        if (
            type(turn) is not int
            or turn < 1
            or not isinstance(call_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", call_id)
        ):
            raise ValueError("invalid task identity")
        identity = f"{turn}:{call_id}"
        if row.get("id") != identity or identity in task_ids:
            raise ValueError("duplicate/invalid task receipt")
        task_ids.add(identity)
        if (
            row.get("kind") not in TASK_KINDS
            or row.get("status") not in TASK_STATES
            or row.get("mode") not in TASK_MODES
        ):
            raise ValueError("invalid task state")
        for field, bound in (("title", 80), ("summary", 200), ("display", 32)):
            if not isinstance(row.get(field), str) or len(row[field]) > bound:
                raise ValueError("invalid task display")
        if (
            not row["title"]
            or row["status"] == "simulated"
            and row["mode"] != "mock"
            or row["mode"] == "Pi-verified"
            and data["mode"] != "Pi-verified"
        ):
            raise ValueError("invalid task evidence")
        artifacts = row.get("artifacts", [])
        artifact_id = row.get("artifact_id")
        if (
            not isinstance(artifacts, list)
            or len(artifacts) > 8
            or any(
                not isinstance(name, str)
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)", name
                )
                for name in artifacts
            )
            or len(set(artifacts)) != len(artifacts)
            or artifacts
            and (
                row["kind"] != "agent"
                or row["status"] != "reported"
                or not isinstance(artifact_id, str)
                or not re.fullmatch(r"[a-f0-9]{24}", artifact_id)
            )
            or artifact_id is not None
            and not artifacts
        ):
            raise ValueError("invalid task artifact receipt")
        normalized_task = {
            key: row[key]
            for key in (
                "id",
                "turn_id",
                "call_id",
                "kind",
                "status",
                "mode",
                "title",
                "summary",
                "display",
            )
        }
        task_id = row.get("task_id")
        if "answer_only" in row:
            if type(row["answer_only"]) is not bool:
                raise ValueError("invalid answer-only task marker")
            if row["answer_only"] and (
                row["kind"] != "agent"
                or row["status"] != "reported"
                or artifacts
                or "widget" in row
                or "absolute_path" in row
            ):
                raise ValueError("answer-only marker cannot conceal action evidence")
            normalized_task["answer_only"] = row["answer_only"]
        if task_id is not None:
            if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", task_id):
                raise ValueError("invalid task ID")
            normalized_task["task_id"] = task_id
        for field in (
            "started_utc",
            "updated_utc",
            "last_progress_utc",
            "heartbeat_utc",
            "finished_utc",
        ):
            stamp = row.get(field)
            if stamp is not None:
                if not isinstance(stamp, str) or len(stamp) > 60:
                    raise ValueError("invalid task timestamp")
                try:
                    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("invalid task timestamp") from exc
                if parsed.tzinfo is None:
                    raise ValueError("task timestamps must include a timezone")
                normalized_task[field] = stamp
        phase = row.get("phase")
        if phase is not None:
            if (
                not isinstance(phase, str)
                or len(phase) > 100
                or any(ord(char) < 32 for char in phase)
            ):
                raise ValueError("invalid task phase")
            normalized_task["phase"] = phase
        completed_steps = row.get("completed_steps", [])
        if (
            not isinstance(completed_steps, list)
            or len(completed_steps) > 12
            or any(
                not isinstance(step, str)
                or not 1 <= len(step) <= 80
                or any(ord(char) < 32 for char in step)
                for step in completed_steps
            )
        ):
            raise ValueError("invalid completed task steps")
        if completed_steps:
            normalized_task["completed_steps"] = completed_steps
        if artifacts:
            normalized_task["artifact_id"] = artifact_id
            normalized_task["artifacts"] = artifacts
        if "widget" in row:
            normalized_task["widget"] = normalize_task_widget(row["widget"])
        location_fields = ("execution_target", "absolute_path", "verification_level")
        has_location = any(field in row for field in location_fields)
        if has_location:
            target, absolute_path, verification = (row.get(field) for field in location_fields)
            if (
                not isinstance(target, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,47}", target)
                or not isinstance(absolute_path, str)
                or not Path(absolute_path).is_absolute()
                or len(absolute_path) > 512
                or any(ord(char) < 32 for char in absolute_path)
                or not isinstance(verification, str)
                or not verification
                or len(verification) > 120
                or any(ord(char) < 32 for char in verification)
            ):
                raise ValueError("invalid task location evidence")
            normalized_task.update(
                execution_target=target,
                absolute_path=absolute_path,
                verification_level=verification,
            )
        tasks.append(normalized_task)
    return {
        "schema_version": 1,
        "kind": "conversation-transcript",
        "display_authorized": True,
        "session_id": session_id,
        "revision": data["revision"],
        "mode": data["mode"],
        "state": data["state"],
        "backend_modes": {
            role: mode
            for role, mode in data.get("backend_modes", {}).items()
            if role in {"asr", "llm", "tts"} and mode in {"mock", "external"}
        }
        if isinstance(data.get("backend_modes"), dict)
        else {},
        "updated_utc": data["updated_utc"],
        "messages": messages,
        "tasks": tasks,
    }


def normalize_task_widget(widget):
    """Strictly project the small set of structured device result widgets."""
    if not isinstance(widget, dict):
        raise ValueError("invalid task widget")
    kind = widget.get("type")
    if kind == "lamp":
        if (
            set(widget) != {"type", "on", "color", "brightness"}
            or type(widget.get("on")) is not bool
            or normalize_rgb_color(widget.get("color")) is None
            or type(widget.get("brightness")) is not int
            or not 0 <= widget["brightness"] <= 100
        ):
            raise ValueError("invalid lamp widget")
    elif kind == "timer":
        if (
            set(widget)
            != {"type", "schedule_kind", "active", "deadline_utc", "target_local", "display_value"}
            or widget.get("schedule_kind") not in {"timer", "alarm"}
            or type(widget.get("active")) is not bool
            or not isinstance(widget.get("deadline_utc"), str)
            or len(widget["deadline_utc"]) > 60
            or not isinstance(widget.get("target_local"), str)
            or len(widget["target_local"]) > 60
            or not isinstance(widget.get("display_value"), str)
            or len(widget["display_value"]) > 32
        ):
            raise ValueError("invalid timer widget")
        for value in (widget["deadline_utc"], widget["target_local"]):
            if value:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("invalid timer widget timestamp") from exc
                if parsed.utcoffset() is None:
                    raise ValueError("timer widget timestamps must include a timezone")
    elif kind == "temperature":
        if (
            set(widget) != {"type", "setpoint_c", "measurement"}
            or type(widget.get("setpoint_c")) is not int
            or not 16 <= widget["setpoint_c"] <= 30
            or widget.get("measurement") is not False
        ):
            raise ValueError("invalid temperature widget")
    elif kind == "reminders":
        items = widget.get("items")
        if (
            set(widget) != {"type", "items", "persistent"}
            or not isinstance(items, list)
            or len(items) > 3
            or any(item not in {"drink water", "study", "exercise"} for item in items)
            or type(widget.get("persistent")) is not bool
        ):
            raise ValueError("invalid reminders widget")
    elif kind == "time":
        if (
            set(widget) != {"type", "local_time", "timezone"}
            or not isinstance(widget.get("local_time"), str)
            or len(widget["local_time"]) > 20
            or not isinstance(widget.get("timezone"), str)
            or len(widget["timezone"]) > 80
        ):
            raise ValueError("invalid time widget")
    elif kind == "weather":
        if (
            set(widget)
            != {"type", "location", "condition", "temperature_c", "observed_utc", "provider"}
            or not isinstance(widget.get("location"), str)
            or len(widget["location"]) > 80
            or not isinstance(widget.get("condition"), str)
            or len(widget["condition"]) > 50
            or type(widget.get("temperature_c")) not in (int, float)
            or not -100 <= widget["temperature_c"] <= 70
            or not isinstance(widget.get("observed_utc"), str)
            or len(widget["observed_utc"]) > 60
            or widget.get("provider") != "Open-Meteo"
        ):
            raise ValueError("invalid weather widget")
        try:
            observed = datetime.fromisoformat(widget["observed_utc"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid weather observation timestamp") from exc
        if observed.utcoffset() is None:
            raise ValueError("weather observation timestamp must include a timezone")
    else:
        raise ValueError("unsupported task widget")
    return dict(widget)


class Transcript:
    def __init__(self, *, mode="mock", max_messages=200, backend_modes=None, playback="memory"):
        if mode not in MODES or not 2 <= max_messages <= 200:
            raise ValueError("invalid transcript mode/bounds")
        if playback not in {"memory", "browser", "benchmark"}:
            raise ValueError("invalid playback evidence")
        self.mode, self.max_messages = mode, max_messages
        self.playback = playback
        self.backend_modes = backend_modes or {}
        self.session_id, self.state, self.revision = "unstarted", "idle", 0
        self.messages, self.segments, self.tasks = [], [], []
        self.pending_archives = []
        self.active = None
        self.last_turn = 0
        self.updated_utc = datetime.now(timezone.utc).isoformat()

    def changed(self):
        self.revision += 1
        self.updated_utc = datetime.now(timezone.utc).isoformat()

    def start_session(self, session_id):
        TurnKey(session_id=session_id, turn_id=1)
        if self.session_id != "unstarted" and self.messages:
            self.pending_archives.append(self.snapshot())
        self.session_id, self.state = session_id, "listening"
        self.messages, self.segments, self.tasks = [], [], []
        self.active, self.last_turn = None, 0
        self.changed()

    def set_state(self, state):
        if state not in STATES:
            raise ValueError("invalid transcript state")
        if self.state != state:
            self.state = state
            self.changed()

    def begin_turn(self, key):
        if (
            self.state == "ended"
            or key.session_id != self.session_id
            or key.turn_id <= self.last_turn
        ):
            return False
        if self.active:
            self.interrupt(self.active)
        self.active, self.last_turn, self.segments = key, key.turn_id, []
        self.set_state("thinking")
        return True

    def _message(self, key, role):
        identity = f"{key.turn_id}:{role}"
        for row in self.messages:
            if row["id"] == identity:
                return row
        row = {
            "id": identity,
            "turn_id": key.turn_id,
            "role": role,
            "text": "",
            "confirmed_text": "",
            "status": "partial",
            "delivery": "asr_partial" if role == "user" else "generated",
        }
        if role == "assistant":
            row["generation_complete"] = False
        self.messages.append(row)
        self._trim()
        return row

    def _trim(self):
        while (
            len(self.messages) > self.max_messages
            or sum(len(m["text"]) + len(m["confirmed_text"]) for m in self.messages)
            > MAX_CHARACTERS
        ):
            oldest = self.messages[0]["turn_id"]
            self.messages = [m for m in self.messages if m["turn_id"] != oldest]

    def task_started(self, key, call_id, title, kind="agent", mode="mock"):
        if (
            self.active != key
            or not isinstance(call_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", call_id)
        ):
            return False
        if (
            kind not in TASK_KINDS
            or mode not in TASK_MODES
            or not isinstance(title, str)
            or not 0 < len(title) <= 80
        ):
            raise ValueError("invalid task presentation")
        identity = f"{key.turn_id}:{call_id}"
        if any(row["id"] == identity for row in self.tasks):
            return False
        now = datetime.now(timezone.utc).isoformat()
        self.tasks.append(
            {
                "id": identity,
                "task_id": hashlib.sha256(identity_with_session(key, call_id).encode()).hexdigest()[
                    :24
                ],
                "turn_id": key.turn_id,
                "call_id": call_id,
                "kind": kind,
                "status": "running",
                "mode": mode,
                "title": title,
                "summary": "",
                "display": "",
                "started_utc": now,
                "updated_utc": now,
            }
        )
        self.tasks = self.tasks[-50:]
        self.changed()
        return True

    def task_progress(
        self,
        key,
        call_id,
        *,
        status=None,
        phase=None,
        completed_steps=None,
        heartbeat=False,
        summary=None,
        task_id=None,
    ):
        """Update one task independently of its originating conversational turn.

        A heartbeat records liveness only; it never alters the real-progress timestamp.
        """
        if key.session_id != self.session_id or type(heartbeat) is not bool:
            return False
        if status is not None and status not in {"accepted", "running", "awaiting_input"}:
            raise ValueError("invalid active task status")
        if phase is not None and (
            not isinstance(phase, str)
            or not 1 <= len(phase) <= 100
            or any(ord(c) < 32 for c in phase)
        ):
            raise ValueError("invalid task phase")
        if summary is not None and (
            not isinstance(summary, str) or len(summary) > 200 or any(ord(c) < 32 for c in summary)
        ):
            raise ValueError("invalid task progress summary")
        if completed_steps is not None and (
            not isinstance(completed_steps, list)
            or len(completed_steps) > 12
            or any(
                not isinstance(step, str)
                or not 1 <= len(step) <= 80
                or any(ord(char) < 32 for char in step)
                for step in completed_steps
            )
        ):
            raise ValueError("invalid completed task steps")
        if task_id is not None and (
            not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", task_id)
        ):
            raise ValueError("invalid task ID")
        identity = f"{key.turn_id}:{call_id}"
        row = next((item for item in self.tasks if item["id"] == identity), None)
        if row is None or row["status"] not in {"accepted", "running", "awaiting_input"}:
            return False
        now = datetime.now(timezone.utc).isoformat()
        changed = False
        if task_id is not None and row.get("task_id") != task_id:
            row["task_id"] = task_id
            changed = True
        if heartbeat:
            row["heartbeat_utc"] = now
            changed = True
        if status is not None and row["status"] != status:
            row["status"] = status
            changed = True
        phase_changed = phase is not None and row.get("phase") != phase
        if phase is not None and phase_changed:
            row["phase"] = phase
            changed = True
        steps_changed = (
            completed_steps is not None and row.get("completed_steps", []) != completed_steps
        )
        if completed_steps is not None and steps_changed:
            row["completed_steps"] = list(completed_steps)
            changed = True
        if summary is not None and row.get("summary") != summary:
            row["summary"] = summary
            changed = True
        if changed:
            row["updated_utc"] = now
            if phase_changed or steps_changed or (status is not None and not heartbeat):
                row["last_progress_utc"] = now
            self.changed()
        return changed

    def task_finished(
        self,
        key,
        call_id,
        status,
        summary="",
        display="",
        *,
        artifact_id=None,
        artifacts=None,
        execution_target=None,
        absolute_path=None,
        verification_level=None,
        mode=None,
        widget=None,
        task_id=None,
        phase=None,
        completed_steps=None,
        answer_only=False,
    ):
        if key.session_id != self.session_id or status not in TASK_STATES - {
            "running",
            "accepted",
            "awaiting_input",
        }:
            return False
        if (
            not isinstance(summary, str)
            or len(summary) > 200
            or not isinstance(display, str)
            or len(display) > 32
        ):
            raise ValueError("invalid task result display")
        if mode is not None and mode not in TASK_MODES:
            raise ValueError("invalid task result mode")
        if phase is not None and (
            not isinstance(phase, str) or len(phase) > 100 or any(ord(char) < 32 for char in phase)
        ):
            raise ValueError("invalid task phase")
        if completed_steps is not None and (
            not isinstance(completed_steps, list)
            or len(completed_steps) > 12
            or any(
                not isinstance(step, str)
                or not 1 <= len(step) <= 80
                or any(ord(char) < 32 for char in step)
                for step in completed_steps
            )
        ):
            raise ValueError("invalid completed task steps")
        if task_id is not None and (
            not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", task_id)
        ):
            raise ValueError("invalid task ID")
        normalized_widget = normalize_task_widget(widget) if widget is not None else None
        if (
            type(answer_only) is not bool
            or answer_only
            and (status != "reported" or artifacts or widget is not None or absolute_path)
        ):
            raise ValueError("answer-only marker cannot conceal an action output")
        if artifacts is None:
            artifacts = []
        if (
            not isinstance(artifacts, list)
            or len(artifacts) > 8
            or any(
                not isinstance(name, str)
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)", name
                )
                for name in artifacts
            )
            or artifacts
            and (
                not isinstance(artifact_id, str)
                or not re.fullmatch(r"[a-f0-9]{24}", artifact_id)
                or status != "reported"
            )
            or artifact_id is not None
            and not artifacts
        ):
            raise ValueError("invalid task artifact display")
        location = (execution_target, absolute_path, verification_level)
        if any(value is not None for value in location) and (
            not isinstance(execution_target, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,47}", execution_target)
            or not isinstance(absolute_path, str)
            or not Path(absolute_path).is_absolute()
            or len(absolute_path) > 512
            or any(ord(char) < 32 for char in absolute_path)
            or not isinstance(verification_level, str)
            or not verification_level
            or len(verification_level) > 120
            or any(ord(char) < 32 for char in verification_level)
        ):
            raise ValueError("invalid task location evidence")
        identity = f"{key.turn_id}:{call_id}"
        row = next((row for row in self.tasks if row["id"] == identity), None)
        if row is None or row["status"] not in {"accepted", "running", "awaiting_input", "unknown"}:
            return False
        if answer_only and row["kind"] != "agent":
            raise ValueError("answer-only marker requires an agent answer")
        row.update(status=status, summary=summary, display=display)
        if answer_only:
            row["answer_only"] = True
        row["updated_utc"] = datetime.now(timezone.utc).isoformat()
        row["finished_utc"] = row["updated_utc"]
        if task_id is not None:
            row["task_id"] = task_id
        if phase is not None:
            row["phase"] = phase
        if completed_steps is not None:
            row["completed_steps"] = list(completed_steps)
        if phase is not None or completed_steps is not None:
            row["last_progress_utc"] = row["updated_utc"]
        if mode is not None:
            row["mode"] = mode
        if normalized_widget is not None:
            row["widget"] = normalized_widget
        if any(value is not None for value in location):
            row.update(
                execution_target=execution_target,
                absolute_path=absolute_path,
                verification_level=verification_level,
            )
        if artifacts:
            row.update(artifact_id=artifact_id, artifacts=list(dict.fromkeys(artifacts)))
        self.changed()
        return True

    def task_transition(self, key, turn_id, call_id, status, summary):
        """Apply a later replacement or cancellation to the original schedule receipt."""
        if (
            key.session_id != self.session_id
            or type(turn_id) is not int
            or not isinstance(call_id, str)
            or status not in {"replaced", "cancelled", "completed", "unavailable"}
            or not isinstance(summary, str)
            or len(summary) > 200
        ):
            return False
        identity = f"{turn_id}:{call_id}"
        row = next((task for task in self.tasks if task["id"] == identity), None)
        if row is None or row["kind"] not in {"set_timer", "set_alarm"}:
            return False
        if row["status"] not in {"running", "unknown", "accepted", "simulated"}:
            return False
        row["status"], row["summary"] = status, summary
        widget = row.get("widget")
        if widget and widget.get("type") == "timer":
            row["widget"] = dict(widget, active=False)
        self.changed()
        return True

    def task_observation(self, key, call_id, summary):
        """Attach later player-state evidence without upgrading an uncertain action."""
        if (
            key.session_id != self.session_id
            or not isinstance(call_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", call_id)
            or not isinstance(summary, str)
            or len(summary) > 200
            or any(ord(char) < 32 for char in summary)
        ):
            return False
        identity = f"{key.turn_id}:{call_id}"
        row = next((task for task in self.tasks if task["id"] == identity), None)
        if (
            row is None
            or row["kind"] not in {"music_pause", "music_stop"}
            or row["status"] != "unknown"
            or row["summary"] == summary
        ):
            return False
        row["summary"] = summary
        row["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.changed()
        return True

    def user_text(self, key, text, *, final=False):
        if self.active != key:
            return False
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            raise ValueError("invalid user transcript text")
        row = self._message(key, "user")
        if row["status"] == "final":
            return False
        row.update(
            text=text,
            status="final" if final else "partial",
            delivery="asr_final" if final else "asr_partial",
        )
        self._trim()
        self.changed()
        return True

    def assistant_segment(self, key, segment_id, text):
        if self.active != key:
            return False
        if any(
            row["turn_id"] == key.turn_id
            and row["role"] == "assistant"
            and row.get("generation_complete")
            for row in self.messages
        ):
            return False
        if type(segment_id) is not int or segment_id != len(self.segments) + 1:
            raise ValueError("assistant segments must be contiguous")
        if not isinstance(text, str) or not text or len(self.segments) >= 128:
            raise ValueError("invalid assistant segment")
        if len(" ".join([s["text"] for s in self.segments] + [text])) > MAX_TEXT:
            raise ValueError("assistant transcript exceeds bounds")
        self.segments.append({"text": text, "confirmed": False})
        self._refresh_assistant(key)
        return True

    def finalize_text(self, key):
        """Mark generation done independently of browser playback acknowledgment."""
        if self.active != key:
            return False
        row = next(
            (
                row
                for row in self.messages
                if row["turn_id"] == key.turn_id and row["role"] == "assistant"
            ),
            None,
        )
        if row is None or row["generation_complete"]:
            return False
        row["generation_complete"] = True
        self.changed()
        return True

    def _refresh_assistant(self, key):
        row = self._message(key, "assistant")
        row["text"] = " ".join(s["text"] for s in self.segments)
        row["confirmed_text"] = " ".join(s["text"] for s in self.segments if s["confirmed"])
        self._trim()
        self.changed()

    def confirm_segment(self, key, segment_id):
        if self.active != key:
            return False
        if (
            type(segment_id) is not int
            or not 1 <= segment_id <= len(self.segments)
            or any(not s["confirmed"] for s in self.segments[: segment_id - 1])
        ):
            return False
        if self.segments[segment_id - 1]["confirmed"]:
            return False
        self.segments[segment_id - 1]["confirmed"] = True
        self._refresh_assistant(key)
        return True

    def _finish(self, key, status):
        if self.active != key:
            return False
        for row in self.messages:
            if row["turn_id"] != key.turn_id:
                continue
            if row["role"] == "assistant":
                row["text"] = row["confirmed_text"]
                row["status"] = status
                if status == "final":
                    row["generation_complete"] = True
                row["delivery"] = (
                    "generated"
                    if not row["text"]
                    else "sink_confirmed"
                    if self.playback == "browser"
                    else "benchmark_transport_acknowledged"
                    if self.playback == "benchmark"
                    else "mock_sink_confirmed"
                )
            elif row["status"] == "partial":
                row["status"] = status
        # Task execution has its own lifecycle. Ending speech must not cancel or
        # downgrade work that the executor still owns.
        self.active, self.segments = None, []
        self.state = "interrupted" if status in {"interrupted", "error"} else "listening"
        self.changed()
        return True

    def complete(self, key):
        if (
            self.active != key
            or any(not s["confirmed"] for s in self.segments)
            or any(
                m["turn_id"] == key.turn_id and m["role"] == "user" and m["status"] == "partial"
                for m in self.messages
            )
        ):
            return False
        return self._finish(key, "final")

    def interrupt(self, key, *, error=False):
        return self._finish(key, "error" if error else "interrupted")

    def end_session(self):
        if self.active:
            self.interrupt(self.active)
        self.set_state("ended")

    def snapshot(self):
        return normalize_transcript(
            {
                "schema_version": 1,
                "kind": "conversation-transcript",
                "display_authorized": True,
                "session_id": self.session_id,
                "revision": self.revision,
                "mode": self.mode,
                "backend_modes": self.backend_modes,
                "state": self.state,
                "updated_utc": self.updated_utc,
                "messages": [dict(row) for row in self.messages],
                "tasks": [dict(row) for row in self.tasks],
            }
        )


class TranscriptPublisher:
    """Runtime-owned current state, with optional per-session diagnostic snapshots.

    Changes stay in memory in callbacks. Publication is coalesced and file I/O
    runs off the event loop. The final snapshot is flushed on clean shutdown.
    """

    def __init__(
        self,
        transcript,
        path=Path("var/results/ui-transcript.json"),
        *,
        interval=0.1,
        archive_dir=None,
    ):
        self.transcript = transcript
        self.path = output_path(str(path))
        self.archive_dir = output_path(str(archive_dir / ".probe")).parent if archive_dir else None
        self.interval, self.last_revision = interval, -1
        if not 0 < interval <= 5:
            raise ValueError("invalid transcript publication interval")
        self.task = None
        self.stopped = asyncio.Event()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="me3-transcript")

    def _write_one(self, path, payload):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".transcript-",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    def _write(self, payload, archives=()):
        if self.archive_dir:
            for earlier in archives:
                self._write_one(self.archive_dir / f"{earlier['session_id']}.json", earlier)
            if payload["session_id"] != "unstarted" and payload["messages"]:
                self._write_one(self.archive_dir / f"{payload['session_id']}.json", payload)
        self._write_one(self.path, payload)

    async def flush(self):
        if self.transcript.revision != self.last_revision or self.transcript.pending_archives:
            payload = self.transcript.snapshot()
            archives = tuple(self.transcript.pending_archives)
            await await_thread_future(self.writer.submit(self._write, payload, archives))
            del self.transcript.pending_archives[: len(archives)]
            self.last_revision = payload["revision"]

    async def run(self):
        while not self.stopped.is_set():
            try:
                await asyncio.wait_for(self.stopped.wait(), self.interval)
            except TimeoutError:
                await self.flush()

    async def __aenter__(self):
        try:
            await self.flush()
        except BaseException:
            self.writer.shutdown(wait=True)
            raise
        self.task = asyncio.create_task(self.run())
        return self

    async def __aexit__(self, *exc):
        self.stopped.set()
        try:
            if self.task:
                await self.task  # Let any in-flight atomic write finish before final publication.
            await self.flush()
        finally:
            self.writer.shutdown(wait=True)
