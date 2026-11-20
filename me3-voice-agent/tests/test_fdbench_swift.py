from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fdbench_swift  # noqa: E402
from fdbench import SR  # noqa: E402
from fdbench.rig.base import TrialScript  # noqa: E402


def test_full_duplex_and_stage_adapters_use_professor_revision_57eaddbc():
    expected = "57eaddbc188ce9a19b8fd002b575f703426956f4"
    assert fdbench_swift.PINNED_BENCHMARK == expected
    assert fdbench_swift.PINNED_STAGE_BENCHMARK == expected


def test_resume_is_limited_to_complete_saved_run_under_output_root(tmp_path):
    import pytest

    output = tmp_path / "runs"
    run = output / "SwiftME3" / "saved-round"
    run.mkdir(parents=True)
    (run / "session.json").write_text("{}")
    (run / "queue.json").write_text("{}")
    assert fdbench_swift._validate_resume_run(run, output) == run.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "session.json").write_text("{}")
    (outside / "queue.json").write_text("{}")
    with pytest.raises(ValueError, match="ignored output root"):
        fdbench_swift._validate_resume_run(outside, output)

    incomplete = output / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="ignored output root"):
        fdbench_swift._validate_resume_run(incomplete, output)


def _health(run_id="fixture-run", *, usage=0, session="idle"):
    return {
        "bridge": "ready",
        "engine": "native",
        "voice_ready": True,
        "voice_configured": True,
        "voice_session": session,
        "budget": {
            "run_id": run_id,
            "gpt_oss_requests": {"used": usage},
            "haiku_requests": {"used": 0},
            "asr_requests": {"used": 0},
            "tts_requests": {"used": 0},
            "tavily_requests": {"used": 0},
            "token_usage": {
                "estimated_input_tokens_reserved": usage * 1000,
                "actual_input_tokens": usage * 200,
                "actual_output_tokens": usage * 20,
                "requests_with_actual_usage": usage,
            },
            "listed_price_estimate_from_actual_model_tokens_usd": usage * 0.01,
            "vendor_billed_cost_usd": None,
        },
    }


def test_native_rig_projects_trial_trace_pcm_ack_and_usage(monkeypatch):
    state = {"session": 1, "usage": 0, "activation": [], "commands": [], "foreign": []}
    monkeypatch.setattr(fdbench_swift, "_health", lambda _url: _health(usage=state["usage"]))
    rig = fdbench_swift.SwiftWebSocketRig("http://fixture.invalid")

    def reset_session():
        previous = f"fixture-session-{state['session']}"
        state["session"] += 1  # mirrors native stop: clear retained context, rotate session ID
        return previous

    def start_connection(active, *, reset_only=False):
        assert not reset_only
        rig._worker_done.clear()
        rig._worker_error = None
        rig._ready_message = {"session_id": f"fixture-session-{state['session']}"}
        rig._trial_started = 1.0
        rig._native_events = []
        rig._transcripts = []
        rig._assistant_text = []
        rig._terminal.clear()
        rig._latency.clear()
        rig._output_bytes = 0
        rig._packet_acknowledgements = 0
        rig._input_frames_sent = 0
        rig._input_bytes_sent = 0
        rig._input_samples_sent = 0
        rig._input_peak = 0
        rig._input_square_sum = 0
        rig._input_frames_accepted = 0
        rig._input_bytes_accepted = 0
        rig._vad_starts = 0
        rig._vad_stops = 0
        state["activation"].append(active)
        if active:
            state["usage"] += 1
            rig._record_input_sent(fdbench_swift.encode_pcm16_mono(np.full(512, 0.5)))
            rig._receive_json(
                {
                    "type": "runtime_event",
                    "data": {"event": "audio_frame_delivered", "bytes": 1024},
                }
            )
            rig._receive_json({"type": "runtime_event", "data": {"event": "vad_speech_started"}})
            rig._receive_json({"type": "runtime_event", "data": {"event": "vad_speech_stopped"}})
            rig._receive_json(
                {
                    "type": "transcript",
                    "data": {
                        "messages": [
                            {"role": "user", "status": "final", "text": "fixture prompt"},
                            {"role": "assistant", "status": "final", "text": "fixture response"},
                        ]
                    },
                }
            )
            rig._receive_json(
                {
                    "type": "runtime_event",
                    "data": {
                        "event": "voice_latency",
                        "at": 1.0,
                        "turn_id": "fixture-turn",
                        "vad_speech_end_to_asr_final_ms": 120.0,
                        "llm_provider_request_to_first_token_ms": 230.0,
                        "tts_start_to_first_pcm_ms": 340.0,
                        "first_pcm_to_client_delivery_ms": 12.0,
                    },
                }
            )
            rig._receive_json(
                {"type": "runtime_event", "data": {"event": "turn_completed", "at": 1.1}}
            )
        else:
            state["foreign"].append(True)

    def capture_send(value):
        state["commands"].append(value)

    original_append_packets = rig._append_packets
    output_pending = [True]

    def append_fixture_packet():
        if output_pending[0]:
            rig._output_packets.put((7, np.full(320, 0.2, dtype=np.float32)))
            rig._output_bytes = 640
            output_pending[0] = False
        original_append_packets()

    monkeypatch.setattr(rig, "_reset_session", reset_session)
    monkeypatch.setattr(rig, "_start_connection", start_connection)
    monkeypatch.setattr(rig, "_send", capture_send)
    monkeypatch.setattr(rig, "_send_control", capture_send)
    monkeypatch.setattr(rig, "_append_packets", append_fixture_packet)

    script = TrialScript(
        play_first=np.zeros(int(SR * 0.08), dtype=np.float32),
        record_after=0.05,
        max_duration=0.3,
        tag="turn",
    )
    active = rig.run(script)
    foreign = rig.run(
        TrialScript(
            play_first=np.zeros(int(SR * 0.04), dtype=np.float32),
            record_after=0.03,
            max_duration=0.2,
            tag="foreign",
        )
    )

    assert active.notes["rig"] == "swift-native-websocket"
    assert active.notes["session_reset_verified"] is True
    assert active.notes["native_session_id"] != active.notes["reset_from_session_id"]
    assert active.notes["transcripts"] == ["fixture prompt"]
    assert active.notes["assistant_confirmed_text"] == ["fixture response"]
    assert active.notes["terminal"] == "turn_completed"
    assert active.notes["latency"]["vad_speech_end_to_asr_final_ms"] == 120.0
    assert active.notes["provider_usage_delta"]["requests"]["gpt_oss_requests"] == 1
    assert active.notes["provider_usage_delta"]["tokens"]["actual_input_tokens"] == 200
    assert active.notes["provider_usage_delta"]["openrouter_list_price_estimate_usd"] == 0.01
    assert active.notes["output_pcm_bytes"] == 640
    assert active.notes["output_packets_acknowledged"] == 1
    assert active.notes["input_ingress"] == {
        "frames_sent": 1,
        "bytes_sent": 1024,
        "frames_accepted_by_native_bridge": 1,
        "bytes_accepted_by_native_bridge": 1024,
        "normalized_peak": 0.5,
        "normalized_rms": 0.5,
        "vad_speech_starts": 1,
        "vad_speech_stops": 1,
    }
    assert active.notes["physical_audio_verified"] is False
    assert "no physical playback" in active.notes["audio_delivery"]
    assert foreign.notes["activation"] == "wake_standby"
    assert foreign.notes["terminal"] == "no_terminal_turn_event"
    assert state["activation"] == [True, False]
    assert state["foreign"] == [True]
    assert len([item for item in state["commands"] if isinstance(item, bytes)]) > 0
    assert any(
        isinstance(item, dict) and item.get("type") == "audio_ack" for item in state["commands"]
    )


def test_usage_delta_subtracts_vendor_cost_only_when_both_values_exist():
    before = {"budget": {"vendor_billed_cost_usd": 0.12}}
    after = {"budget": {"vendor_billed_cost_usd": 0.19}}
    assert fdbench_swift._usage_delta(before, after)["vendor_billed_cost_usd"] == 0.07
    assert fdbench_swift._usage_delta(before, {"budget": {}})["vendor_billed_cost_usd"] is None


def test_native_stop_reset_waits_for_idle_health(monkeypatch):
    active = _health(session="active")
    idle = _health(session="idle")
    responses = iter((active, idle))
    monkeypatch.setattr(fdbench_swift, "_health", lambda _url: next(responses))
    rig = fdbench_swift.SwiftWebSocketRig("http://fixture.invalid")

    class CompletedWorker:
        def join(self, timeout):
            assert timeout == 8

        def is_alive(self):
            return False

    def reset_handshake(active, *, reset_only=False):
        assert active is False
        assert reset_only is True
        rig._worker = CompletedWorker()
        rig._worker_error = None
        rig._ready_message = {"session_id": "previous-session"}

    monkeypatch.setattr(rig, "_start_connection", reset_handshake)
    assert rig._reset_session() == "previous-session"


def test_pcm16_encoding_preserves_float_audio_level():
    samples = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    encoded = np.frombuffer(fdbench_swift.encode_pcm16_mono(samples), dtype="<i2")
    assert encoded.tolist() == [-32768, -16384, 16384, 32767]


def test_stage_cli_builds_official_thirty_item_plan_without_running(monkeypatch):
    import pytest

    pytest.importorskip("fdbench.marks")
    from fdbench import cli
    from fdbench.datasets import Dataset
    from fdbench.queue import build_plan

    def inspect_launch(args):
        assert (args.rounds, args.phrases, args.runs, args.seed) == (0, 10, 3, 1)
        assert args.judge_provider == "none" and args.marks_port == 0
        ds = Dataset(Path(args.datasets))
        plan = build_plan(
            ds, rounds=args.rounds, seed=args.seed, phrases=args.phrases, runs=args.runs
        )
        assert len(plan.items) == 30
        assert {item.group for item in plan.items} == {"stage_latency"}
        clips = [{item.clip_id for item in plan.items if item.round == r} for r in (1, 2, 3)]
        assert len(clips[0]) == 10 and clips[0] == clips[1] == clips[2]
        assert all(
            clip.meta.get("text") for clip in ds.clips("stage_latency") if clip.id in clips[0]
        )
        assert cli._direct_rig(args).marks is not None
        assert cli.RUNS_DIR == fdbench_swift.STAGE_RUNS_ROOT
        return 0

    # Capture the real upstream parser's launch plan without I/O to a bridge.
    monkeypatch.setattr(cli, "cmd_run", inspect_launch)
    monkeypatch.setattr(sys, "argv", ["fdbench_swift"])
    monkeypatch.setattr(cli, "_direct_rig", cli._direct_rig)
    monkeypatch.setattr(cli, "cmd_check", cli.cmd_check)
    monkeypatch.setattr(cli, "load_dotenv", cli.load_dotenv)
    monkeypatch.setattr(cli, "RUNS_DIR", cli.RUNS_DIR)
    import fdbench.util as util
    from fdbench.runner import Runner

    monkeypatch.setattr(util, "RUNS_DIR", util.RUNS_DIR)
    monkeypatch.setattr(Runner, "_run_trial", Runner._run_trial)
    assert fdbench_swift.main(["--stage-only"]) == 0


def test_stage_marks_align_transcript_and_emit_official_artifacts(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import pytest

    marks = pytest.importorskip("fdbench.marks")
    from fdbench.report.files import write_results
    from fdbench.runner import Runner
    from fdbench.score import stages

    store = marks.MarksStore()
    rig = fdbench_swift.SwiftWebSocketRig("http://fixture.invalid", marks=store)
    events = [
        ("vad_speech_stopped", 1.0, {}),
        ("asr_final", 1.2, {}),
        ("llm_provider_request_started", 1.3, {"request_id": "answered"}),
        ("llm_first_token", 1.4, {}),  # local synthetic token
        ("llm_first_token", 1.8, {"request_id": "answered"}),
        ("tts_start", 2.0, {}),
        ("tts_pcm", 2.4, {}),
        ("playback_packet", 2.5, {}),
        ("llm_usage", 2.6, {"prompt_tokens": 101}),
    ]
    for event, ts, fields in events:
        monkeypatch.setattr(fdbench_swift.time, "monotonic", lambda ts=ts: 100 + ts)
        rig._receive_json({"type": "runtime_event", "data": {"event": event, "at": ts, **fields}})

    transcript = {
        "id": "1:user",
        "role": "user",
        "status": "final",
        "delivery": "asr_final",
        "text": "Open the red door.",
    }
    snapshot = {"type": "transcript", "data": {"messages": [transcript]}}
    # Neither arrival order nor repeat snapshots change the Pi ASR timestamp.
    rig._receive_json(snapshot)
    rig._receive_json(snapshot)
    fdbench_swift._publish_stage_marks(
        store, rig._stage_events, rig._stage_transcripts, src=rig.base_url
    )
    assert (
        Runner._trial_marks(
            SimpleNamespace(marks=store), tmp_path, SimpleNamespace(mono0=100.0, duration=4.0)
        )
        == 8
    )
    rows = stages.read_marks(tmp_path)
    result = stages.turn_stages(rows, 0.9, 2.5, "Open the blue door.")
    assert result["transcript"] == transcript["text"]
    assert result["wer"]["wer"] == pytest.approx(0.25)
    assert result["asr_final"] == 0.2
    assert result["llm_ttft"] == 0.6
    assert result["tts_ttfb"] == 0.4
    assert result["e2e"] == 1.5
    assert result["context_tokens"] == 101
    trial = {
        "trial": "r01_stage_latency_fixture",
        "group": "stage_latency",
        "clip": "fixture",
        "clip_meta": {"text": "Open the blue door."},
        "final": {"stages": result},
    }
    metrics = stages.aggregate([trial], tmp_path)
    write_results(tmp_path, {"metrics": metrics, "trials": 1}, {}, None, None)
    for name in ("stage_latency.json", "results.json", "summary.md", "latency_chart.svg"):
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "results.json").read_text())["metrics"]["course.wer"]["n"] == 1

    ambiguous = marks.MarksStore()
    fdbench_swift._publish_stage_marks(
        ambiguous, rig._stage_events, {"1:user": "first", "2:user": "second"}, src=rig.base_url
    )
    assert "asr_final" not in ambiguous.stages_seen
    assert "llm_first_token" not in ambiguous.stages_seen


def test_post_capture_drains_acks_for_completion_and_failure_recovery(monkeypatch):
    rig = fdbench_swift.SwiftWebSocketRig("http://fixture.invalid")
    script = TrialScript(play_first=np.zeros(rig.chunk, dtype=np.float32))
    score = rig._Session(rig, script)
    score.next_out(rig.chunk)
    score.feed_in(np.zeros(rig.chunk, dtype=np.float32))
    before = score.finish()
    acks = []
    next_terminal = "turn_failed"

    def bridge_after_ack(message):
        assert message["type"] == "audio_ack"
        acks.append(message["id"])
        # The bridge can emit its terminal outcome only after receiving the
        # final ACK, including when a response fails after partial playback.
        if message["id"] in (2, 3):
            rig._receive_json({"type": "runtime_event", "data": {"event": next_terminal}})

    monkeypatch.setattr(rig, "_send_control", bridge_after_ack)
    rig._output_packets.put((1, np.ones(rig.chunk, dtype=np.float32)))
    rig._output_packets.put((2, np.ones(rig.chunk, dtype=np.float32)))
    assert not rig._terminal.is_set()
    rig._drain_until(rig._terminal, timeout=0.5)
    assert rig._terminal.is_set()
    assert acks == [1, 2]
    assert rig._native_events[-1]["event"] == "turn_failed"

    # The next observation can complete, and the previously scored track
    # still contains none of the PCM drained after its cutoff.
    rig._terminal.clear()
    next_terminal = "turn_completed"
    rig._output_packets.put((3, np.ones(rig.chunk, dtype=np.float32)))
    rig._drain_until(rig._terminal, timeout=0.5)
    assert rig._terminal.is_set()
    assert acks == [1, 2, 3]
    assert rig._native_events[-1]["event"] == "turn_completed"
    after = score.finish()
    assert before.duration == after.duration
    np.testing.assert_array_equal(before.played, after.played)
    np.testing.assert_array_equal(before.heard, after.heard)


def test_post_capture_drain_is_bounded_when_bridge_never_settles():
    rig = fdbench_swift.SwiftWebSocketRig("http://fixture.invalid")
    rig._drain_until(rig._terminal, timeout=0)
    assert not rig._terminal.is_set()
    assert rig._packet_acknowledgements == 0
