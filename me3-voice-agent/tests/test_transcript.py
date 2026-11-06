import asyncio
import json
import threading

import pytest

from me3_voice.contracts import TurnKey
from me3_voice.dashboard import snapshot
from me3_voice.transcript import Transcript, TranscriptPublisher, normalize_transcript


def started(**kwargs):
    book = Transcript(**kwargs)
    book.start_session("session")
    key = TurnKey(session_id="session", turn_id=1)
    assert book.begin_turn(key)
    return book, key


def test_asr_hypotheses_replace_same_message_and_final_is_not_overwritten():
    book, key = started()
    book.user_text(key, "Find a", final=False)
    book.user_text(key, "Find a paper", final=False)
    assert len(book.snapshot()["messages"]) == 1
    book.user_text(key, "Find a paper on audio", final=True)
    revision = book.revision
    assert not book.user_text(key, "stale hypothesis", final=False)
    assert book.revision == revision
    row = book.snapshot()["messages"][0]
    assert row["text"] == "Find a paper on audio" and row["delivery"] == "asr_final"


def test_generation_delivery_and_completion_are_separate():
    book, key = started()
    book.user_text(key, "hello", final=True)
    book.assistant_segment(key, 1, "First sentence.")
    book.assistant_segment(key, 2, "Second sentence.")
    assert book.state == "thinking"  # Generated text alone is not playback activity.
    assert book.finalize_text(key)
    assert not book.finalize_text(key)
    assert not book.assistant_segment(key, 3, "Late text")
    preplay = book.snapshot()["messages"][-1]
    assert preplay["generation_complete"] is True
    assert preplay["status"] == "partial" and preplay["confirmed_text"] == ""
    assert not book.complete(key)
    assert not book.confirm_segment(key, 2)
    assert book.confirm_segment(key, 1)
    row = book.snapshot()["messages"][-1]
    assert row["text"] == "First sentence. Second sentence." and row["status"] == "partial"
    assert row["confirmed_text"] == "First sentence."
    assert book.confirm_segment(key, 2) and book.complete(key)
    row = book.snapshot()["messages"][-1]
    assert row["status"] == "final" and row["delivery"] == "mock_sink_confirmed"
    assert not book.confirm_segment(key, 2)


def test_benchmark_output_has_transport_ack_not_mock_or_acoustic_claim():
    book, key = started(playback="benchmark")
    book.user_text(key, "test request", final=True)
    book.assistant_segment(key, 1, "A short response.")
    assert book.confirm_segment(key, 1)
    assert book.complete(key)
    row = book.snapshot()["messages"][-1]
    assert row["delivery"] == "benchmark_transport_acknowledged"


def test_later_music_player_observation_updates_only_matching_uncertain_task():
    book, key = started()
    call_id = "local-stop-1"
    assert book.task_started(key, call_id, "Stop music", kind="music_stop", mode="reported")
    assert book.task_finished(
        key,
        call_id,
        "unknown",
        "Spotify could not confirm whether this control took effect.",
        mode="reported",
    )
    summary = (
        "Spotify could not confirm the stop request; the browser player now reports paused. "
        "Whether the request caused this state remains unverified."
    )
    stale_key = TurnKey(session_id=key.session_id, turn_id=key.turn_id + 1)
    assert not book.task_observation(stale_key, call_id, summary)
    assert book.task_observation(key, call_id, summary)
    task = book.snapshot()["tasks"][-1]
    assert task["status"] == "unknown"
    assert task["summary"] == summary


def test_interruption_removes_unconfirmed_tail_and_fences_late_updates():
    book, key = started()
    book.user_text(key, "hello", final=True)
    book.assistant_segment(key, 1, "Confirmed sentence.")
    book.confirm_segment(key, 1)
    book.assistant_segment(key, 2, "Unplayed tail.")
    book.interrupt(key)
    row = book.snapshot()["messages"][-1]
    assert row["text"] == "Confirmed sentence." and row["status"] == "interrupted"
    assert "Unplayed" not in json.dumps(book.snapshot())
    newer = TurnKey(session_id="session", turn_id=2)
    book.begin_turn(newer)
    book.user_text(newer, "correction", final=True)
    assert not book.assistant_segment(key, 3, "late text")
    assert not book.confirm_segment(key, 2)
    assert [m["turn_id"] for m in book.snapshot()["messages"]] == [1, 1, 2]


def test_unplayed_interruption_and_session_reset_do_not_invent_delivery():
    book, key = started()
    book.user_text(key, "hel", final=False)
    assert not book.complete(key)
    book.assistant_segment(key, 1, "Never played")
    book.interrupt(key, error=True)
    rows = book.snapshot()["messages"]
    assert rows[0]["delivery"] == "asr_partial" and rows[0]["status"] == "error"
    assert rows[-1]["text"] == "" and rows[-1]["delivery"] == "generated"
    book.end_session()
    assert not book.begin_turn(TurnKey(session_id="session", turn_id=2))
    book.start_session("new-session")
    assert book.snapshot()["messages"] == []
    assert not book.user_text(key, "late old ASR", final=True)


def test_task_receipts_are_sanitized_fenced_and_keep_completed_actions():
    book, key = started()
    assert book.task_started(key, "call-1", "Set a timer", "set_timer")
    assert not book.task_started(key, "call-1", "Set a timer", "set_timer")
    running = book.snapshot()["tasks"][0]
    assert running["status"] == "running" and "arguments" not in running
    assert book.task_finished(
        key, "call-1", "simulated", "Demo timer saved; no alert will ring.", "10 seconds"
    )
    completed = book.snapshot()["tasks"][0]
    assert completed["status"] == "simulated"
    book.interrupt(key)
    assert book.snapshot()["tasks"][0] == completed
    assert not book.task_finished(key, "call-1", "simulated", "late duplicate")
    book.start_session("new-session")
    assert book.snapshot()["tasks"] == []
    assert not book.task_finished(key, "call-1", "simulated", "late old session")


def test_timer_widget_updates_for_replacement_and_rejects_untrusted_fields():
    book, key = started()
    assert book.task_started(key, "timer-1", "Set a timer", "set_timer")
    widget = {
        "type": "timer",
        "schedule_kind": "timer",
        "active": True,
        "deadline_utc": "2026-11-01T12:00:10+00:00",
        "target_local": "",
        "display_value": "10 seconds",
    }
    assert book.task_finished(
        key, "timer-1", "simulated", "Simulated timer.", "10 seconds", widget=widget
    )
    assert book.task_started(key, "alarm-2", "Set an alarm", "set_alarm")
    assert book.task_finished(key, "alarm-2", "simulated", "Alarm set.", "8:00 AM")
    assert book.task_transition(key, 1, "timer-1", "replaced", "Replaced by a newer timer/alarm.")
    assert book.snapshot()["tasks"][0]["widget"]["active"] is False
    assert book.snapshot()["tasks"][0]["status"] == "replaced"

    payload = book.snapshot()
    payload["tasks"][0]["widget"]["gpio"] = 17
    with pytest.raises(ValueError, match="timer widget"):
        normalize_transcript(payload)


def test_rgb_lamp_widget_allows_bounded_hex_color_without_other_fields():
    book, key = started()
    assert book.task_started(key, "rgb-1", "Set RGB color", "set_color")
    assert book.task_finished(
        key,
        "rgb-1",
        "simulated",
        "Simulated RGB LED #12AB34 at 37%.",
        "#12AB34",
        widget={"type": "lamp", "on": True, "color": "#12AB34", "brightness": 37},
    )
    assert normalize_transcript(book.snapshot())["tasks"][0]["widget"]["color"] == "#12AB34"
    payload = book.snapshot()
    payload["tasks"][0]["widget"]["color"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="lamp widget"):
        normalize_transcript(payload)


def test_running_task_keeps_its_lifecycle_when_speech_is_interrupted():
    book, key = started()
    book.task_started(key, "slow", "Agent task", "agent")
    book.interrupt(key)
    assert book.snapshot()["tasks"][0]["status"] == "running"
    assert book.task_finished(key, "slow", "reported", "Task service reported completion.")
    assert book.snapshot()["tasks"][0]["status"] == "reported"
    payload = book.snapshot()
    payload["tasks"][0]["summary"] = "x" * 201
    with pytest.raises(ValueError, match="invalid task display"):
        normalize_transcript(payload)
    payload = book.snapshot()
    payload["tasks"][0]["mode"] = "Pi-verified"
    with pytest.raises(ValueError, match="invalid task evidence"):
        normalize_transcript(payload)


def test_background_progress_and_heartbeat_are_separate_and_survive_new_turn():
    book, key = started()
    assert book.task_started(key, "long", "Create project", "agent", mode="mock")
    task_id = book.snapshot()["tasks"][0]["task_id"]
    assert book.task_progress(
        key,
        "long",
        status="running",
        phase="Drafting project files",
        completed_steps=["Request accepted"],
        task_id=task_id,
    )
    before = book.snapshot()["tasks"][0]
    assert book.task_progress(key, "long", heartbeat=True, task_id=task_id)
    alive = book.snapshot()["tasks"][0]
    assert alive["heartbeat_utc"] >= before["updated_utc"]
    assert alive["last_progress_utc"] == before["last_progress_utc"]
    assert alive["phase"] == "Drafting project files"

    book.interrupt(key)
    newer = TurnKey(session_id="session", turn_id=2)
    assert book.begin_turn(newer)
    assert book.task_progress(
        key,
        "long",
        phase="Checking output",
        completed_steps=["Request accepted", "Draft created"],
    )
    assert book.task_finished(
        key,
        "long",
        "completed",
        "Mock output completed.",
        "project.md",
        mode="mock",
        phase="Complete",
        completed_steps=["Request accepted", "Draft created", "Checked"],
        task_id=task_id,
    )
    final = book.snapshot()["tasks"][0]
    assert final["status"] == "completed"
    assert final["phase"] == "Complete"
    assert len(final["completed_steps"]) == 3
    assert "finished_utc" in final


def test_background_task_schema_rejects_fake_percent_or_invalid_progress():
    book, key = started()
    book.task_started(key, "long", "Create project", "agent", mode="mock")
    payload = book.snapshot()
    payload["tasks"][0]["phase"] = "x" * 101
    with pytest.raises(ValueError, match="task phase"):
        normalize_transcript(payload)
    payload = book.snapshot()
    payload["tasks"][0]["completed_steps"] = ["step"] * 13
    with pytest.raises(ValueError, match="completed task steps"):
        normalize_transcript(payload)
    payload = book.snapshot()
    payload["tasks"][0]["progress_percent"] = 83
    assert "progress_percent" not in normalize_transcript(payload)["tasks"][0]


def test_agent_task_artifact_receipt_is_bounded_and_displayable():
    book, key = started()
    assert book.task_started(key, "agent-call", "Create a project", "agent", mode="reported")
    assert book.task_finished(
        key,
        "agent-call",
        "reported",
        "Created a draft file.",
        "index.html",
        artifact_id="a" * 24,
        artifacts=["index.html", "README.md"],
    )
    task = book.snapshot()["tasks"][0]
    assert task["artifact_id"] == "a" * 24
    assert task["artifacts"] == ["index.html", "README.md"]
    payload = book.snapshot()
    payload["tasks"][0]["artifact_id"] = "../outside"
    with pytest.raises(ValueError, match="artifact"):
        normalize_transcript(payload)


def test_filesystem_task_location_is_explicit_and_allowlisted():
    book, key = started()
    assert book.task_started(key, "project-call", "Create a project", "agent", mode="reported")
    assert book.task_finished(
        key,
        "project-call",
        "reported",
        "Created an empty project directory.",
        execution_target="HPC workspace",
        absolute_path="/sandbox/ME3-Test",
        verification_level="Directory exists and is empty; checked by the executor.",
    )
    task = normalize_transcript(book.snapshot())["tasks"][0]
    assert task["execution_target"] == "HPC workspace"
    assert task["absolute_path"] == "/sandbox/ME3-Test"
    assert "executor" in task["verification_level"]
    assert task["status"] == "reported"

    with pytest.raises(ValueError, match="location"):
        book.task_finished(
            key,
            "project-call",
            "reported",
            "bad path",
            execution_target="HPC workspace",
            absolute_path="../ME3-Test",
            verification_level="Exists",
        )


def test_bounded_projection_keeps_whole_recent_turns():
    book = Transcript(max_messages=2)
    book.start_session("session")
    for turn in range(1, 4):
        key = TurnKey(session_id="session", turn_id=turn)
        book.begin_turn(key)
        book.user_text(key, f"question {turn}", final=True)
        book.assistant_segment(key, 1, f"answer {turn}")
        book.confirm_segment(key, 1)
        book.complete(key)
    assert [(m["turn_id"], m["role"]) for m in book.snapshot()["messages"]] == [
        (3, "user"),
        (3, "assistant"),
    ]


def test_unicode_conversations_stay_within_dashboard_file_budget():
    from me3_voice.dashboard import MAX_FILE

    book = Transcript()
    book.start_session("unicode-session")
    for turn in range(1, 21):
        key = TurnKey(session_id="unicode-session", turn_id=turn)
        book.begin_turn(key)
        book.user_text(key, "🎤" * 4000, final=True)
        book.assistant_segment(key, 1, "🎤" * 4000)
        book.confirm_segment(key, 1)
        book.complete(key)
    payload = book.snapshot()
    assert len(json.dumps(payload, ensure_ascii=True).encode()) < MAX_FILE
    assert payload["messages"][0]["turn_id"] > 1
    assert payload["messages"][-1]["turn_id"] == 20
    assert all(
        sum(m["turn_id"] == row["turn_id"] for m in payload["messages"]) == 2
        for row in payload["messages"]
    )


def test_dashboard_only_accepts_explicit_canonical_transcript(tmp_path):
    book, key = started()
    book.user_text(key, "visible user text", final=True)
    payload = book.snapshot()
    payload["credentials"] = "never include this"
    payload["messages"][0]["reasoning"] = "never include this"
    results = tmp_path / "var/results"
    results.mkdir(parents=True)
    (results / "old-transcript.json").write_text(json.dumps(payload))
    assert not snapshot(tmp_path)["reports"]
    path = results / "ui-transcript.json"
    path.write_text(json.dumps(payload))
    report = snapshot(tmp_path)["reports"][0]
    assert report["kind"] == "transcript" and report["evidence"] == "mock"
    assert report["data"]["messages"][0]["text"] == "visible user text"
    assert "never include this" not in json.dumps(report)
    payload["display_authorized"] = False
    path.write_text(json.dumps(payload))
    assert not snapshot(tmp_path)["reports"]
    assert snapshot(tmp_path)["warnings"]


def test_schema_rejects_reasoning_roles_duplicates_and_unconfirmed_final():
    book, key = started()
    book.user_text(key, "hello", final=True)
    payload = book.snapshot()
    payload["messages"][0]["role"] = "system"
    with pytest.raises(ValueError):
        normalize_transcript(payload)
    payload = book.snapshot()
    payload["messages"].append(dict(payload["messages"][0]))
    with pytest.raises(ValueError):
        normalize_transcript(payload)
    book.assistant_segment(key, 1, "draft")
    payload = book.snapshot()
    payload["messages"][-1]["status"] = "final"
    with pytest.raises(ValueError):
        normalize_transcript(payload)


async def test_atomic_publisher_shutdown_waits_for_inflight_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    book, key = started()
    publisher = TranscriptPublisher(book, interval=0.01)
    async with publisher:
        real_write = publisher._write
        entered, release = threading.Event(), threading.Event()

        def delayed(payload, archives=()):
            entered.set()
            assert release.wait(3)
            real_write(payload, archives)

        publisher._write = delayed
        book.user_text(key, "partial", final=False)
        deadline = asyncio.get_running_loop().time() + 2
        while not entered.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        assert entered.is_set()
        book.user_text(key, "final text", final=True)
        # Finish the in-flight old write; context exit must subsequently flush the new revision.
        release.set()
    payload = json.loads(publisher.path.read_text())
    assert payload["revision"] == book.revision
    assert payload["messages"][0]["text"] == "final text"
    assert not list(publisher.path.parent.glob(".transcript-*"))


async def test_voice_archive_retains_each_session_through_fast_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    book, key = started(mode="HPC-live")
    async with TranscriptPublisher(
        book, interval=0.5, archive_dir=tmp_path / "var/results/voice-sessions"
    ) as publisher:
        book.user_text(key, "turn on the lights", final=True)
        book.end_session()
        book.start_session("next-session")
        newer = TurnKey(session_id="next-session", turn_id=1)
        book.begin_turn(newer)
        book.user_text(newer, "explain Raspberry Pi", final=True)
    first = json.loads((tmp_path / "var/results/voice-sessions/session.json").read_text())
    second = json.loads((tmp_path / "var/results/voice-sessions/next-session.json").read_text())
    current = json.loads(publisher.path.read_text())
    assert first["state"] == "ended"
    assert first["messages"][0]["text"] == "turn on the lights"
    assert second["messages"][0]["text"] == "explain Raspberry Pi"
    assert current == second
    assert not list(publisher.path.parent.glob(".transcript-*"))
