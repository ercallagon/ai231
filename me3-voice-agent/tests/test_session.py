import asyncio

import pytest

from me3_voice.english import EnglishTextRejected
from me3_voice.providers import ASRResult, MockLLM, ToolCall
from me3_voice.session import Session, State, direct_clock_query, direct_light_command
from me3_voice.tools import MOCK_ACTION_TITLES, MockActions
from me3_voice.transcript import Transcript


def test_benchmark_transport_events_are_not_mislabeled_as_mock():
    benchmark_sink = sink
    benchmark_sink.delivery = "benchmark_transport_acknowledged"
    session = Session(Speech(), MockLLM(), benchmark_sink)
    session.activate()
    session.playback_started(session.key)
    names = [row["event"] for row in session.events]
    assert "capture_ready_benchmark" in names
    assert "sink_write_started_benchmark" in names
    assert "capture_ready_mock" not in names
    assert "sink_write_started_mock" not in names
    del benchmark_sink.delivery


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        for _ in range(16):
            await asyncio.sleep(0)
            yield b"tone"


async def sink(pcm):
    await asyncio.sleep(0)


async def test_exact_clock_route_skips_delayed_reasoning_but_keeps_typed_receipt():
    from me3_voice.telemetry import summarize

    class DelayedToolModel:
        def __init__(self):
            self.calls = 0

        async def respond(self, _history):
            self.calls += 1
            await asyncio.sleep(0.2)
            yield ToolCall("model-time", "run_device_action", {"action": "get_time"})

    samples = {False: [], True: []}
    for pair in range(5):
        for enabled in (False, True) if pair % 2 == 0 else (True, False):
            model = DelayedToolModel()
            display = Transcript()
            async with Session(
                Speech(), model, sink, transcript=display, direct_clock_enabled=enabled
            ) as session:
                session.activate()
                await (await session.submit(b"what time is it"))
                task = display.snapshot()["tasks"]
                assert len(task) == 1
                assert task[0]["kind"] == "get_time"
                assert task[0]["status"] == "completed"
                assert session.history[-1]["role"] == "assistant"
                assert session.playback.queue.empty()
                assert model.calls == (0 if enabled else 1)
                if enabled:
                    assert not any(
                        event["event"] == "llm_first_segment" for event in session.events
                    )
                row = next(
                    item for item in summarize(session.events)["turns"] if item["turn_id"] == 1
                )
                samples[enabled].extend(row["durations_ms"]["endpoint_to_sink_write_mock_ms"])
    assert len(samples[False]) == len(samples[True]) == 5
    assert sorted(samples[False])[2] - sorted(samples[True])[2] >= 150


def test_clock_route_rejects_location_corrections_and_incomplete_requests():
    assert direct_clock_query("Please, what time is it?", 1) is None
    assert direct_clock_query("what time is it", 1).arguments == {"action": "get_time"}
    assert direct_clock_query("What time is it in the Philippines?", 2).arguments == {
        "action": "get_time",
        "value": "Asia/Manila",
    }
    assert direct_clock_query("What time is it in Japan?", 3).arguments["value"] == "Asia/Tokyo"
    assert direct_clock_query("time is it in Manila", 4).arguments == {
        "action": "get_time",
        "value": "Asia/Manila",
    }
    assert direct_clock_query("What time is it in London?", 4) is None
    for text in (
        "what time is it and set a timer",
        "what time is it—actually in Manila",
        "what time is it and...",
        "what time",
    ):
        assert direct_clock_query(text, 2) is None


async def test_direct_clock_interruption_keeps_receipt_without_stale_speech():
    from me3_voice.playback import MemorySink

    playback = MemorySink(delay=30)
    display = Transcript()
    async with Session(
        Speech(), MockLLM(), playback, transcript=display, direct_clock_enabled=True
    ) as session:
        session.activate()
        old = await session.submit(b"what time is it")
        await asyncio.wait_for(playback.entered.wait(), 2)
        assert display.snapshot()["tasks"][0]["status"] == "completed"
        await session.interrupt()
        assert old.done() and old.cancelled()
        assert not any(
            row["role"] == "assistant" and row["turn_id"] == 1 for row in session.history
        )
        playback.delay = 0
        await (await session.submit(b"follow up"))
        assert session.history[-1]["turn_id"] == 2
        assert len(display.snapshot()["tasks"]) == 1
        assert session.playback.queue.empty()


async def test_direct_clock_and_volume_followups_execute_once_without_model():
    from me3_voice.music import MusicController
    from me3_voice.tools import MockTaskExecutor

    class NoReasoning:
        async def respond(self, _history):
            raise AssertionError("direct route must not call the model")
            yield

    music = MusicController()
    tasks = MockTaskExecutor(music=music)
    display = Transcript()
    try:
        async with Session(
            Speech(),
            NoReasoning(),
            sink,
            tasks=tasks,
            transcript=display,
            direct_clock_enabled=True,
        ) as session:
            session.activate()
            await (await session.submit(b"What time is it in the Philippines?"))
            await (await session.submit(b"Volume 50%."))
            await (await session.submit(b"Can you set the volume to 20%?"))
            await (await session.submit(b"Volume down."))
            rows = display.snapshot()["tasks"]
            assert [row["kind"] for row in rows] == [
                "get_time",
                "music_volume_set",
                "music_volume_set",
                "music_volume_down",
            ]
            assert rows[0]["widget"]["timezone"] == "Asia/Manila"
            assert rows[1]["status"] == "simulated"
            assert rows[2]["display"] == "20%"
            assert rows[3]["display"] == "10%"
            assert len(music.operations) == 3
            assert not any(event["event"] == "llm_request_started" for event in session.events)
    finally:
        await music.close()


async def test_light_percentage_followup_is_contextual_and_preserves_prior_action():
    class LightModel:
        def __init__(self):
            self.calls = 0

        async def respond(self, _history):
            self.calls += 1
            yield ToolCall("light-on", "run_device_action", {"action": "light_on"})

    model = LightModel()
    display = Transcript()
    async with Session(Speech(), model, sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"Turn on the lights."))
        await (await session.submit(b"Can you make it 80%?"))
        tasks = display.snapshot()["tasks"]
        assert [task["kind"] for task in tasks] == ["light_on", "set_brightness"]
        assert tasks[0]["widget"]["brightness"] == 60
        assert tasks[1]["widget"]["brightness"] == 80
        assert model.calls == 0
        assert session.tasks.actions.effects == 2


def test_direct_light_commands_are_exact_and_reject_ambiguous_phrases():
    expected = {
        "Turn on the lights.": {"action": "light_on"},
        "Can you turn on the light?": {"action": "light_on"},
        "Switch the lights off.": {"action": "light_off"},
        "Set the lights to blue.": {"action": "set_color", "value": "blue"},
        "Change light color to red.": {"action": "set_color", "value": "red"},
        "Set the lights to orange.": {"action": "set_color", "value": "orange"},
        "Set the lights to #12AB34.": {"action": "set_color", "value": "#12ab34"},
        "Set brightness to 60 percent.": {"action": "set_brightness", "value": "60"},
        "Make the lights 20%.": {"action": "set_brightness", "value": "20"},
    }
    for text, arguments in expected.items():
        call = direct_light_command(text, 4)
        assert call is not None
        assert call.name == "run_device_action"
        assert call.call_id == "local-light-4"
        assert call.arguments == arguments

    for text in (
        "Turn it on.",
        "Turn on the lights and set a timer.",
        "Set volume to 60 percent.",
        "Make the lights brighter.",
        "Set brightness to 150 percent.",
    ):
        assert direct_light_command(text, 4) is None


def test_light_followup_uses_conversation_turn_receipt_and_supports_dim_and_color_corrections():
    from me3_voice.fast_commands import direct_light_followup

    history = [
        {"role": "user", "content": "Can you turn on the light?", "turn_id": 1},
        {
            "role": "tool",
            "turn_id": 1,
            "tool_request": {
                "name": "run_device_action",
                "arguments": '{"action": "light_on"}',
            },
            "content": '{"ok": true, "result_status": "simulated"}',
        },
    ]
    dim = direct_light_followup("dim it to thirty percent", 12, history, history_turn=2)
    assert dim is not None
    assert dim.arguments == {"action": "set_brightness", "value": "30"}
    color = direct_light_followup("sorry you can set it to red", 13, history, history_turn=2)
    assert color is not None
    assert color.arguments == {"action": "set_color", "value": "red"}
    history[-1]["content"] = '{"ok": false, "result_status": "failed"}'
    assert direct_light_followup("dim it to thirty percent", 14, history, history_turn=2) is None


async def test_exact_light_commands_use_shared_device_executor_without_reasoning():
    class NoReasoning:
        calls = 0

        async def respond(self, _history):
            self.calls += 1
            raise AssertionError("exact final light command should bypass reasoning")
            yield

    model = NoReasoning()
    display = Transcript()
    async with Session(Speech(), model, sink, transcript=display) as session:
        session.activate()
        for phrase in (
            "Turn on the lights.",
            "Set the lights to blue.",
            "Set brightness to 60 percent.",
            "Turn off the lights.",
        ):
            await (await session.submit(phrase.encode()))

        tasks = display.snapshot()["tasks"]
        assert [task["kind"] for task in tasks] == [
            "light_on",
            "set_color",
            "set_brightness",
            "light_off",
        ]
        assert all(task["status"] == "simulated" for task in tasks)
        assert model.calls == 0
        assert session.tasks.actions.effects == 4
        assert [event["event"] for event in session.events].count("local_light_command_routed") == 4
        assert display.snapshot()["tasks"][-1]["widget"]["on"] is False


async def test_empty_reasoning_produces_honest_reply_without_task():
    class EmptyReasoning:
        async def respond(self, _history):
            if False:
                yield "unreachable"

    display = Transcript()
    async with Session(Speech(), EmptyReasoning(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"Explain this"))
        assert display.snapshot()["tasks"] == []
        assert session.history[-1]["content"] == "I couldn't produce a reply. Please try again."


async def test_incomplete_reminder_asks_for_content_without_creating_action():
    class NoReasoning:
        async def respond(self, _history):
            raise AssertionError("exact clarification should not call the model")
            yield

    display = Transcript()
    async with Session(Speech(), NoReasoning(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"Set my reminders."))
        assert not display.snapshot()["tasks"]
        assert session.history[-1]["content"] == "What would you like me to remind you about?"
        await (await session.submit(b"Make a call"))
        assert not display.snapshot()["tasks"]
        assert (
            session.history[-1]["content"]
            == "Calling and messaging aren't configured in this session."
        )


def test_light_followup_rejects_ambiguous_and_unrelated_context():
    from me3_voice.session import direct_light_followup

    other = [
        {
            "role": "tool",
            "turn_id": 1,
            "tool_request": {"name": "control_music", "arguments": "{}"},
            "content": "{}",
        }
    ]
    assert direct_light_followup("make it 80%", 2, other) is None
    assert direct_light_followup("make it 101%", 2, other) is None
    assert direct_light_followup("make it brighter", 2, other) is None


def test_direct_music_requests_preserve_user_words_and_need_context_for_corrections():
    from me3_voice.session import direct_music_control, direct_music_request

    user = {"role": "user", "turn_id": 1, "content": "Play a Bruno Mars song."}
    music = {"role": "tool", "turn_id": 2, "tool_request": {"name": "control_music"}}
    assert direct_music_request(user["content"], 1, []).arguments == {
        "action": "play",
        "query": "Bruno Mars",
    }
    assert (
        direct_music_request("No, play Taylor Swift.", 2, [user]).arguments["query"]
        == "Taylor Swift"
    )
    assert (
        direct_music_request("No, can you make it Celine Dion?", 3, [music]).arguments["query"]
        == "Celine Dion"
    )
    assert (
        direct_music_request("make it air supplies now", 3, [music]).arguments["query"]
        == "air supplies"
    )
    assert direct_music_request("No, can you make it Celine Dion?", 3, []) is None
    assert direct_music_request("No, play Adele and stop the music", 2, [user]) is None
    assert direct_music_request("play music", 2, [user]) is None
    assert direct_music_control("Can you stop the music?", 4).arguments == {"action": "stop"}


def test_direct_music_request_routes_explicit_track_title_without_model_rewriting():
    from me3_voice.session import direct_music_request

    request = direct_music_request("Can you play What a Beautiful Name It Is?", 1, [])
    assert request.arguments == {
        "action": "play",
        "query": "What a Beautiful Name It Is",
    }
    assert direct_music_request("Play me a song", 1, []) is None
    assert direct_music_request("Play next", 1, []) is None
    assert direct_music_request("Play Maroon Five instead", 1, []) is None


def test_music_discovery_requests_select_playlist_search_instead_of_track_literals():
    from me3_voice.session import direct_music_request

    popular = direct_music_request("Play a popular song in PH", 1, [])
    assert popular.arguments == {
        "action": "play_playlist",
        "query": "Top 50 Philippines",
        "market": "PH",
    }
    assert direct_music_request("Play popular songs in the Philippines", 5, []).arguments == {
        "action": "play_playlist",
        "query": "Top 50 Philippines",
        "market": "PH",
    }
    chill = direct_music_request("Play a chill song", 2, [])
    assert chill.arguments == {"action": "play_playlist", "query": "chill"}
    explicit = direct_music_request("Play my Focus playlist in the Philippines", 3, [])
    assert explicit.arguments == {
        "action": "play_playlist",
        "query": "my Focus",
        "market": "PH",
    }
    exact = direct_music_request("Play Bruno Mars", 4, [])
    assert exact.arguments == {"action": "play", "query": "Bruno Mars"}


async def test_interrupted_music_receipt_then_correction_executes_each_once():
    from me3_voice.music import MusicController
    from me3_voice.playback import MemorySink
    from me3_voice.tools import MockTaskExecutor

    class NoReasoning:
        async def respond(self, _history):
            raise AssertionError("explicit play requests should use the direct route")
            yield

    music = MusicController()
    tasks = MockTaskExecutor(music=music)
    playback = MemorySink(delay=30)
    display = Transcript()
    try:
        async with Session(
            Speech(), NoReasoning(), playback, tasks=tasks, transcript=display
        ) as session:
            session.activate()
            first = await session.submit(b"Play a Bruno Mars song.")
            await asyncio.wait_for(playback.entered.wait(), 2)
            assert display.snapshot()["tasks"][0]["status"] == "simulated"
            await session.interrupt()
            assert first.cancelled()
            # Barge-in stops the reply audio only. No replacement command has
            # arrived yet, so the already completed music action stays active.
            assert music.current == "Bruno Mars"
            assert music.playing is True
            assert len(music.operations) == 1
            playback.delay = 0
            await (await session.submit(b"No, play Taylor Swift."))
            assert len(music.operations) == 2
            assert music.current == "Taylor Swift"
            assert [item["kind"] for item in display.snapshot()["tasks"]] == [
                "music_play",
                "music_play",
            ]
            assert session.playback.queue.empty()
    finally:
        await music.close()


async def test_explicit_music_title_reaches_existing_action_without_reasoner():
    from me3_voice.music import MusicController
    from me3_voice.tools import MockTaskExecutor

    class NoReasoning:
        async def respond(self, _history):
            raise AssertionError("an explicit track title should use the direct action route")
            yield

    music = MusicController()
    display = Transcript()
    try:
        async with Session(
            Speech(), NoReasoning(), sink, tasks=MockTaskExecutor(music=music), transcript=display
        ) as session:
            session.activate()
            await (await session.submit(b"Can you play What a Beautiful Name It Is?"))
        assert music.current == "What a Beautiful Name It Is"
        assert [task["kind"] for task in display.snapshot()["tasks"]] == ["music_play"]
    finally:
        await music.close()


async def test_mood_and_region_music_requests_use_playlist_route_without_model():
    from me3_voice.music import MusicController
    from me3_voice.tools import MockTaskExecutor

    class NoReasoning:
        async def respond(self, _history):
            raise AssertionError("known playlist discovery intents use the direct action route")
            yield

    music = MusicController()
    display = Transcript()
    try:
        async with Session(
            Speech(), NoReasoning(), sink, tasks=MockTaskExecutor(music=music), transcript=display
        ) as session:
            session.activate()
            await (await session.submit(b"Play a popular song in PH"))
            assert music.current == "Top 50 Philippines"
            await (await session.submit(b"Play a chill song"))
            assert music.current == "chill"
        tasks = display.snapshot()["tasks"]
        assert [task["kind"] for task in tasks] == ["music_playlist", "music_playlist"]
        assert all(task["status"] == "simulated" for task in tasks)
        assert "No audio is played" in tasks[0]["summary"]
    finally:
        await music.close()


async def test_followup_explicit_exit_and_inactivity():
    now = [0.0]
    async with Session(Speech(), MockLLM(), sink, clock=lambda: now[0]) as session:
        with pytest.raises(ValueError, match="activation"):
            await session.submit(b"hello")
        session.activate()
        identifier = session.session_id
        for text in [b"hello", b"follow up"]:
            await (await session.submit(text))
        assert session.session_id == identifier
        assert session.state == State.LISTENING
        assert session.history[-1]["content"].startswith("Mock reply to turn 2")
        await (await session.submit(b"goodbye"))
        assert session.state == State.IDLE
        session.activate()
        assert session.session_id != identifier and not session.history
        now[0] = 31
        await session.tick()
        assert session.state == State.IDLE


def test_direct_music_stop_requires_an_explicit_music_target():
    from me3_voice.session import direct_music_control

    assert direct_music_control("stop the current song", 3).arguments == {"action": "stop"}
    assert direct_music_control("Please stop the music please.", 4).arguments == {"action": "stop"}
    assert direct_music_control("Stoppit.", 5) is None
    assert direct_music_control("stop speaking", 6) is None
    assert direct_music_control("stop it", 7) is None
    assert direct_music_control("Volume fifty percent.", 8).arguments == {
        "action": "volume_set",
        "query": "50",
    }
    assert direct_music_control("set music volume to 25%", 9).arguments == {
        "action": "volume_set",
        "query": "25",
    }
    assert direct_music_control("volume 101 percent", 10) is None
    assert direct_music_control("volume fifty", 11) is None


def test_direct_music_volume_phrases_route_to_the_music_volume_action():
    from me3_voice.session import direct_music_control

    assert direct_music_control("Can you set the volume to 20%?", 1).arguments == {
        "action": "volume_set",
        "query": "20",
    }
    assert direct_music_control("Volume down.", 2).arguments == {"action": "volume_down"}
    assert direct_music_control("Turn the music down", 3).arguments == {"action": "volume_down"}
    assert direct_music_control("Lower the music", 6).arguments == {"action": "volume_down"}
    assert direct_music_control("Could you raise the volume please?", 4).arguments == {
        "action": "volume_up"
    }
    assert direct_music_control("Turn it down", 5) is None


async def test_explicit_spelling_correction_is_echoed_exactly_without_reasoning():
    from me3_voice.session import direct_spelling_correction

    class NoReasoning:
        async def respond(self, _history):
            raise AssertionError("letter-by-letter corrections use the exact local acknowledgment")
            yield

    phrase = "Marilau is spelled as M-A-R-I-L-A-O."
    assert direct_spelling_correction(phrase) == "Got it. I’ll use the spelling M-A-R-I-L-A-O."
    transcript = Transcript()
    async with Session(Speech(), NoReasoning(), sink, transcript=transcript) as session:
        session.activate()
        await (await session.submit(phrase.encode()))
        messages = transcript.snapshot()["messages"]
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["text"] == "Got it. I’ll use the spelling M-A-R-I-L-A-O."
        assert not any(event["event"] == "llm_request_started" for event in session.events)


async def test_invalid_music_volume_tool_failure_is_not_reported_as_a_simulated_action():
    from me3_voice.music import MusicController
    from me3_voice.tools import MockTaskExecutor

    class MalformedVolumeCall:
        async def respond(self, _history):
            yield ToolCall("invalid-volume", "control_music", {"action": "volume_set"})

    music = MusicController()
    transcript = Transcript()
    try:
        async with Session(
            Speech(),
            MalformedVolumeCall(),
            sink,
            tasks=MockTaskExecutor(music=music),
            transcript=transcript,
        ) as session:
            session.activate()
            await (await session.submit(b"Adjust the music volume now"))
            task = transcript.snapshot()["tasks"][0]
            assert task["kind"] == "music_volume_set"
            assert task["status"] == "failed"
            assert task["summary"] == (
                "Spotify rejected the volume request. Use a percentage from 0 to 100."
            )
    finally:
        await music.close()


@pytest.mark.parametrize(
    "uncertain,observed_playing,expected_status",
    [
        (False, False, "reported"),
        (True, False, "reported"),
        (True, True, "unknown"),
        (True, None, "unknown"),
    ],
)
async def test_direct_music_stop_avoids_reasoner_and_preserves_player_outcome(
    uncertain, observed_playing, expected_status
):
    from me3_voice.music import MusicController
    from me3_voice.tools import MockTaskExecutor

    requests = []

    async def peer(method, path, *, params=None, body=None):
        requests.append((method, path))
        if path == "/me/player/devices":
            return {
                "devices": [
                    {
                        "id": "speaker",
                        "name": "Swift speaker",
                        "is_active": True,
                        "is_restricted": False,
                    }
                ]
            }
        if path == "/me/player/pause":
            if uncertain:
                raise TimeoutError
            return {}
        if path == "/me/player":
            if observed_playing is None:
                raise TimeoutError
            return {
                "is_playing": observed_playing,
                "device": {"id": "speaker", "name": "Swift speaker"},
                "item": {"name": "Test track", "uri": "spotify:track:test"},
            }
        raise AssertionError(f"unexpected Spotify request {method} {path}")

    class NoReasoning:
        calls = 0

        async def respond(self, _history):
            self.calls += 1
            raise AssertionError("explicit music stop should not need a reasoning request")
            yield "unreachable"

    reasoner = NoReasoning()
    music = MusicController(token="test-token", device_id="speaker", request=peer)
    executor = MockTaskExecutor(music=music)
    transcript = Transcript()
    async with Session(Speech(), reasoner, sink, tasks=executor, transcript=transcript) as session:
        session.activate()
        await (await session.submit(b"stop the current song"))
        if expected_status == "unknown" and observed_playing is None:
            session.observe_music_playback("paused", source="web_playback_sdk", track_changed=False)

    assert reasoner.calls == 0
    assert requests.count(("PUT", "/me/player/pause")) == 1
    assert not any(event["event"] == "llm_request_started" for event in session.events)
    assert sum(event["event"] == "music_command_routed" for event in session.events) == 1
    task = transcript.snapshot()["tasks"][-1]
    assert task["status"] == expected_status
    if expected_status == "unknown":
        if observed_playing is None:
            assert "now reports paused" in task["summary"]
            assert "remains unverified" in task["summary"]
        else:
            assert "before retrying" in task["summary"]
        failure = next(event for event in session.events if event["event"] == "turn_error")
        assert failure["error_code"] == "music_outcome_uncertain"
        assert failure["error_stage"] == "tool"
    else:
        assert "does not expose a separate stop" in task["summary"]
        if uncertain:
            assert "after the stop request" in task["summary"]
            assert not any(event["event"] == "turn_error" for event in session.events)
        else:
            assert task["summary"] == (
                "Spotify playback is paused; Spotify does not expose a separate stop action here."
            )


async def test_session_trace_records_stage_boundaries_without_transcript(tmp_path, monkeypatch):
    from me3_voice.telemetry import read_trace, summarize, write_trace

    monkeypatch.chdir(tmp_path)
    async with Session(Speech(), MockLLM(), sink) as session:
        session.activate()
        await (await session.submit(b"private utterance"))
    trace = tmp_path / "var/results/trace.jsonl"
    write_trace(trace, session.events)
    rows = summarize(read_trace(trace))["turns"]
    turn = next(row for row in rows if row["turn_id"] == 1)
    assert turn["outcome"] == "completed"
    assert "endpoint_to_sink_write_mock_ms" in turn["durations_ms"]
    assert "asr_request_ms" in turn["durations_ms"]
    assert "private utterance" not in trace.read_text()


async def test_reasoning_limit_failure_is_classified_and_traceable():
    from types import SimpleNamespace

    from me3_voice.remote import ProviderError
    from me3_voice.telemetry import summarize

    class CappedReasoning:
        transport = SimpleNamespace(
            spec=SimpleNamespace(request_limit=2),
            requests=2,
        )

        async def respond(self, _history):
            raise ProviderError(
                "configured provider request allowance exhausted; no request was sent",
                code="request_limit_reached",
            )
            yield "unreachable"

    async with Session(Speech(), CappedReasoning(), sink) as session:
        session.activate()
        await (await session.submit(b"a follow-up"))
        failure = next(event for event in session.events if event["event"] == "turn_error")
        assert failure["error_code"] == "request_limit_reached"
        assert failure["llm_requests_used"] == failure["llm_request_limit"] == 2
        result = summarize(session.events)["turns"]
        failed_turn = next(row for row in result if row["turn_id"] == 1)
        assert failed_turn["error_code"] == "request_limit_reached"
        assert failed_turn["reasoning_request_budget"] == {
            "used": 2,
            "limit": 2,
            "remaining": 0,
        }


async def test_reasoning_timeout_boundary_is_reported_without_a_fake_assistant_reply():
    from me3_voice.remote import ProviderError

    class StalledReasoning:
        async def respond(self, _history):
            raise ProviderError(
                "reasoning timed out waiting for the first text delta",
                code="provider_timeout",
                role="llm",
                error_stage="reasoning",
                error_boundary="first_text_delta",
            )
            yield "unreachable"

    display = Transcript()
    async with Session(Speech(), StalledReasoning(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"what is the capital of the Philippines?"))

    messages = display.snapshot()["messages"]
    assert [(message["role"], message["text"]) for message in messages] == [
        ("user", "what is the capital of the Philippines?")
    ]
    failure = next(event for event in session.events if event["event"] == "turn_error")
    assert failure["error_role"] == "llm"
    assert failure["error_stage"] == "reasoning"
    assert failure["error_boundary"] == "first_text_delta"


async def test_reasoning_parse_detail_survives_session_event_without_response_payload():
    from me3_voice.remote import ProviderError

    class MalformedReasoning:
        async def respond(self, _history):
            raise ProviderError(
                "invalid reasoning tool call",
                code="provider_invalid_response",
                role="llm",
                error_stage="response_parse",
                error_detail="invalid_tool_arguments",
            )
            yield "unreachable"

    display = Transcript()
    async with Session(Speech(), MalformedReasoning(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"what is the capital of Japan"))

    failure = next(event for event in session.events if event["event"] == "turn_error")
    assert failure["error_code"] == "provider_invalid_response"
    assert failure["error_detail"] == "invalid_tool_arguments"
    assert [message["role"] for message in display.snapshot()["messages"]] == ["user"]


async def test_next_reasoning_segment_arrives_while_first_tts_is_running():
    first_tts_started = asyncio.Event()
    second_segment_emitted = asyncio.Event()
    release_first_tts = asyncio.Event()

    class SegmentedLLM:
        async def respond(self, _history):
            yield "First sentence."
            await first_tts_started.wait()
            second_segment_emitted.set()
            yield "Second sentence."

    class SlowFirstTTS(Speech):
        async def synthesize(self, key, text):
            if text == "First sentence.":
                first_tts_started.set()
                await release_first_tts.wait()
            yield b"tone"

    async with Session(SlowFirstTTS(), SegmentedLLM(), sink) as session:
        session.activate()
        turn = await session.submit(b"test")
        await asyncio.wait_for(second_segment_emitted.wait(), 1)
        assert not release_first_tts.is_set()
        release_first_tts.set()
        await asyncio.wait_for(turn, 1)
        assert [
            event["segment_id"] for event in session.events if event["event"] == "tts_started"
        ] == [1, 2]
        assert session.state == State.LISTENING


async def test_barge_in_stops_inflight_and_discards_queue():
    entered, cancelled = asyncio.Event(), asyncio.Event()
    writes = []

    async def blocked_sink(pcm):
        entered.set()
        try:
            await asyncio.Event().wait()
            writes.append(pcm)
        finally:
            cancelled.set()

    async with Session(Speech(), MockLLM(), blocked_sink) as session:
        session.activate()
        first = await session.submit(b"old input")
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0.01)
        assert 0 < session.playback.queue.qsize() <= 8
        await asyncio.wait_for(session.interrupt(), 1)
        assert cancelled.is_set() and not writes
        assert first.done() and session.playback.queue.empty()
        assert not any(message["role"] == "assistant" for message in session.history)
        session.playback.sink = sink
        await (await session.submit(b"correction"))
        assert session.history[-1]["turn_id"] == 2


async def test_late_asr_cannot_update_next_turn():
    started = asyncio.Event()

    class LateSpeech(Speech):
        async def transcribe(self, key, pcm):
            if pcm == b"old":
                started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    return "late old result"
            return "new result"

    display = Transcript()
    async with Session(LateSpeech(), MockLLM(), sink, transcript=display) as session:
        session.activate()
        await session.submit(b"old")
        await started.wait()
        await (await session.submit(b"new"))
        assert [m["content"] for m in session.history if m["role"] == "user"] == ["new result"]
        assert [m["text"] for m in display.snapshot()["messages"] if m["role"] == "user"] == [
            "new result"
        ]


async def test_turn_timeout_cleans_playback():
    async def slow_sink(pcm):
        await asyncio.sleep(60)

    async with Session(Speech(), MockLLM(), slow_sink, turn_timeout=0.05) as session:
        session.activate()
        await (await session.submit(b"test"))
        assert session.state == State.LISTENING
        assert session.playback.queue.empty()
        assert any(event["event"] == "turn_error" for event in session.events)


async def test_live_style_active_turn_waits_for_result_or_user_interruption():
    started, cancelled, never = asyncio.Event(), asyncio.Event(), asyncio.Event()
    now = [0.0]

    class PendingLLM:
        async def respond(self, history):
            if history[-1]["content"] == "pending":
                started.set()
                try:
                    await never.wait()
                finally:
                    cancelled.set()
                if False:
                    yield "obsolete"
            else:
                yield "Recovered."

    async with Session(
        Speech(),
        PendingLLM(),
        sink,
        idle_seconds=60,
        turn_timeout=None,
        llm_timeout=None,
        clock=lambda: now[0],
    ) as session:
        session.activate()
        pending = await session.submit(b"pending")
        await started.wait()
        now[0] = 120
        await session.tick()
        assert session.state == State.THINKING and not pending.done()

        recovered = await session.submit(b"next")
        await recovered
        assert cancelled.is_set()
        assert session.state == State.LISTENING
        assert [row["content"] for row in session.history if row["role"] == "assistant"] == [
            "Recovered."
        ]


async def test_failed_turn_starts_a_fresh_inactivity_window():
    now = [5.0]

    class FailingLLM:
        async def respond(self, history):
            if False:
                yield "unreachable"
            raise RuntimeError("temporary provider failure")

    async with Session(
        Speech(),
        FailingLLM(),
        sink,
        idle_seconds=60,
        turn_timeout=None,
        llm_timeout=None,
        clock=lambda: now[0],
    ) as session:
        session.activate()
        await (await session.submit(b"question"))
        assert session.state == State.LISTENING
        assert session.last_activity == 5.0
        now[0] = 64.9
        await session.tick()
        assert session.state == State.LISTENING
        now[0] = 65.0
        await session.tick()
        assert session.state == State.IDLE


async def test_mock_task_effect_survives_speech_cancellation_without_repeat():
    actions = MockActions()
    caller = asyncio.create_task(actions.set_light("stable-operation", True))
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    result = await actions.set_light("stable-operation", True)
    assert result["verified_in_memory"] and actions.light and actions.effects == 1
    with pytest.raises(ValueError, match="different arguments"):
        await actions.set_light("stable-operation", False)
    await actions.close()


async def test_eleven_me2_style_actions_are_bounded_mock_effects_with_receipts():
    from me3_voice.providers import ToolCall
    from me3_voice.tools import MockTaskExecutor

    executor = MockTaskExecutor()
    key = type("Key", (), {"session_id": "s", "turn_id": 1})()
    values = {
        "set_color": "blue",
        "set_brightness": "20",
        "set_timer": "10 seconds",
        "set_alarm": "6:00 AM",
        "set_temperature": "22 degrees",
        "create_reminder": "study",
    }
    for index, action in enumerate(MOCK_ACTION_TITLES):
        args = {"action": action}
        if action in values:
            args["value"] = values[action]
        call = ToolCall(f"action-{index}", "run_mock_action", args)
        result = await executor.execute(key, call)
        if action == "get_time":
            assert result["mode"] == "reported" and result["result_status"] == "completed"
        elif action == "get_weather":
            assert result["mode"] == "mock" and result["result_status"] == "unavailable"
        else:
            assert result["mode"] == "mock" and result["verified_in_memory"] is True
        assert result["action"] == action and result["summary"]
        assert await executor.execute(key, call) == result
    assert executor.actions.effects == 8
    assert executor.actions.reminders == ["study"]
    assert executor.actions.timers == ["10 seconds"]  # A bounded monotonic timer is active.
    assert executor.actions.alarms == ["6:00 AM"]
    assert executor.actions.temperature == 22
    with pytest.raises(ValueError, match="different arguments"):
        await executor.execute(
            key, ToolCall("action-2", "run_mock_action", {"action": "set_color", "value": "red"})
        )
    with pytest.raises(ValueError, match="unsupported device action value"):
        await executor.execute(
            key,
            ToolCall(
                "invalid",
                "run_mock_action",
                {"action": "set_color", "value": "ultraviolet infrared"},
            ),
        )
    await executor.close()


async def test_temperature_mock_action_reaches_transcript_as_labeled_simulation():
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"simulate temperature 22 degrees"))
        receipt = display.snapshot()["tasks"][-1]
        assert receipt["kind"] == "set_temperature"
        assert receipt["status"] == "simulated"
        assert receipt["display"] == "22°C"
        assert "not a measured temperature" in receipt["summary"].lower()
        assert session.tasks.actions.temperature == 22


async def test_mock_task_receipt_survives_spoken_reply_interruption():
    display = Transcript()
    entered = asyncio.Event()

    async def blocked_sink(_pcm):
        entered.set()
        await asyncio.Event().wait()

    async with Session(Speech(), MockLLM(), blocked_sink, transcript=display) as session:
        session.activate()
        turn = await session.submit(b"simulate timer 10 seconds")
        await asyncio.wait_for(entered.wait(), 1)
        receipts = display.snapshot()["tasks"]
        assert len(receipts) == 1
        assert receipts[0]["kind"] == "set_timer"
        assert receipts[0]["status"] == "simulated"
        assert "no buzzer or alert" in receipts[0]["summary"]
        assert session.tasks.actions.timers == ["10 seconds"]
        await session.interrupt()
        assert turn.done()
        assert display.snapshot()["tasks"] == receipts
        assert session.tasks.actions.effects == 1


async def test_followup_schedule_replaces_prior_timer_once_and_receipt_is_updated():
    display = Transcript()
    async with Session(Speech(), MockLLM(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"simulate timer 10 seconds"))
        assert display.snapshot()["tasks"][-1]["status"] == "simulated"
        await (await session.submit(b"simulate alarm 8:00 AM"))
        rows = display.snapshot()["tasks"]
        assert [row["kind"] for row in rows] == ["set_timer", "set_alarm"]
        assert rows[0]["status"] == "replaced" and rows[0]["widget"]["active"] is False
        assert rows[1]["status"] == "simulated" and rows[1]["widget"]["active"] is True
        assert session.tasks.actions.effects == 2


async def test_task_receipt_acknowledgment_saves_model_request_for_followup_and_correction():
    from me3_voice.providers import ToolCall

    class TwoTaskTurns:
        def __init__(self):
            self.calls = []

        async def respond(self, history):
            self.calls.append(tuple(dict(row) for row in history))
            if self.calls[-1][-1]["role"] != "user":
                raise AssertionError("task reply should not require a post-tool model request")
            yield ToolCall("correct-light", "set_mock_light", {"enabled": False})

    llm = TwoTaskTurns()
    display = Transcript()
    async with Session(Speech(), llm, sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"turn on the light"))
        assert llm.calls == []
        await (await session.submit(b"wait, turn it off again"))
        assert len(llm.calls) == 1
        assert sum(event["event"] == "llm_request_started" for event in session.events) == 1
        assert [row["content"] for row in session.history if row["role"] == "assistant"] == [
            "Simulated lamp blue, 60% brightness, on.",
            "The simulated light is off.",
        ]
        assert session.tasks.actions.light is False
        assert session.tasks.actions.effects == 2
        assert [row["status"] for row in display.snapshot()["tasks"]] == [
            "simulated",
            "simulated",
        ]


async def test_playback_failure_does_not_acknowledge_unspoken_text():
    async def broken_sink(pcm):
        raise OSError("device unavailable")

    async with Session(Speech(), MockLLM(), broken_sink, turn_timeout=1) as session:
        session.activate()
        await asyncio.wait_for(await session.submit(b"test"), 1)
        assert not any(m["role"] == "assistant" for m in session.history)
        assert any(e["event"] == "turn_error" for e in session.events)


async def test_interrupted_history_retains_only_confirmed_segment():
    second_segment = asyncio.Event()
    release = asyncio.Event()

    class Segments:
        async def respond(self, history):
            yield "first sentence"
            second_segment.set()
            await release.wait()
            yield "unspoken second sentence"

    display = Transcript()
    async with Session(Speech(), Segments(), sink, transcript=display) as session:
        session.activate()
        await session.submit(b"hello")
        await second_segment.wait()

        async def first_segment_confirmed():
            while not any(e["event"] == "segment_sink_confirmed" for e in session.events):
                await asyncio.sleep(0.001)

        await asyncio.wait_for(first_segment_confirmed(), 1)
        await session.interrupt()
        assert [m["content"] for m in session.history if m["role"] == "assistant"] == [
            "first sentence"
        ]
        assistant = display.snapshot()["messages"][-1]
        assert assistant["text"] == "first sentence" and assistant["status"] == "interrupted"


async def test_text_finalization_precedes_playback_acknowledgment():
    entered, release = asyncio.Event(), asyncio.Event()

    class OneChunkSpeech(Speech):
        async def synthesize(self, key, text):
            yield b"tone"

    async def blocked_sink(pcm):
        entered.set()
        await release.wait()

    display = Transcript()
    async with Session(OneChunkSpeech(), MockLLM(), blocked_sink, transcript=display) as session:
        session.activate()
        turn = await session.submit(b"hello")
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(100):
            if any(e["event"] == "llm_text_finalized" for e in session.events):
                break
            await asyncio.sleep(0.01)
        assistant = display.snapshot()["messages"][-1]
        assert assistant["generation_complete"] is True
        assert assistant["status"] == "partial" and assistant["confirmed_text"] == ""
        assert session.state == State.SPEAKING and not turn.done()
        release.set()
        await asyncio.wait_for(turn, 1)
        assistant = display.snapshot()["messages"][-1]
        assert assistant["status"] == "final" and assistant["confirmed_text"] == assistant["text"]


async def test_reasoning_failure_before_model_selected_action_does_not_execute_and_followup_recovers():
    from me3_voice.remote import ProviderError

    class FailsOnce:
        def __init__(self):
            self.calls = 0

        async def respond(self, history):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("reasoning unavailable", code="provider_timeout")
            yield "I recovered and can answer the next request."

    reasoning = FailsOnce()
    actions = MockActions()
    async with Session(Speech(), reasoning, sink) as session:
        session.tasks.actions = actions
        session.activate()
        failed = await session.submit(b"make the room more inviting")
        await failed
        assert actions.effects == 0 and actions.light is False
        error = next(event for event in session.events if event["event"] == "turn_error")
        assert error["error_code"] == "provider_timeout"
        assert error["error_stage"] == "reasoning"
        await (await session.submit(b"hello again"))
        assert actions.effects == 0 and reasoning.calls == 2
        assert session.state == State.LISTENING


async def test_completed_task_receipt_survives_tts_failure_and_followup_does_not_repeat_it():
    from me3_voice.remote import ProviderError

    class TTSFailsOnce(Speech):
        def __init__(self):
            self.calls = 0

        async def synthesize(self, key, text):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("speech render timed out", code="provider_timeout")
            yield b"tone"

    display = Transcript()
    speech = TTSFailsOnce()
    async with Session(speech, MockLLM(), sink, transcript=display) as session:
        session.activate()
        failed = await session.submit(b"simulate timer 10 seconds")
        await failed
        receipts = display.snapshot()["tasks"]
        assert len(receipts) == 1
        assert receipts[0]["status"] == "simulated"
        assert session.tasks.actions.timers == ["10 seconds"]
        failure = next(event for event in session.events if event["event"] == "turn_error")
        assert failure["error_code"] == "provider_timeout"
        assert failure["error_stage"] == "tts"
        await (await session.submit(b"simulate light on"))
        assert session.tasks.actions.timers == ["10 seconds"]
        assert session.tasks.actions.light is True
        assert len(display.snapshot()["tasks"]) == 2
        assert session.state == State.LISTENING


@pytest.mark.parametrize(
    "transcript_text",
    [
        "你好。",
        "NO_CLEAR_ENGLISH_SPEECH",
        "No.Clear.English.Speech",
        "The spoken words are: 'Turn on the lights.'",
    ],
)
async def test_english_asr_rejections_are_blocked_before_transcript_or_reasoning(transcript_text):
    class ForeignSpeech(Speech):
        async def transcribe(self, _key, _pcm):
            return transcript_text

    class CountingModel:
        def __init__(self):
            self.calls = 0

        async def respond(self, _history):
            self.calls += 1
            yield "This must not run."

    display, model = Transcript(), CountingModel()
    async with Session(ForeignSpeech(), model, sink, transcript=display) as session:
        session.activate()
        with pytest.raises(EnglishTextRejected):
            await session.submit(b"captured audio")
        assert session.state == State.LISTENING
        assert model.calls == 0
        assert display.snapshot()["messages"] == []
        error = next(event for event in session.events if event["event"] == "turn_error")
        assert error["error_code"] == "english_only_asr"
        assert error["error_role"] == "asr"


async def test_foreign_asr_partial_is_hidden_if_final_result_is_english():
    class CorrectedSpeech(Speech):
        async def transcribe_stream(self, _key, chunks):
            async for _ in chunks:
                pass
            yield ASRResult("你好。", final=False)
            yield ASRResult("Hello Swift.", final=True)

    display = Transcript()
    async with Session(CorrectedSpeech(), MockLLM(), sink, transcript=display) as session:
        session.activate()
        pending = await session.submit(b"captured audio")
        await pending
        user_messages = [
            message for message in display.snapshot()["messages"] if message["role"] == "user"
        ]
        assert len(user_messages) == 1
        assert user_messages[0]["text"] == "Hello Swift."


async def test_non_latin_assistant_output_is_blocked_before_transcript_and_playback():
    class ForeignModel:
        async def respond(self, _history):
            yield "你好。"

    class CountSpeech(Speech):
        def __init__(self):
            self.synthesized = []

        async def synthesize(self, key, text):
            self.synthesized.append(text)
            async for chunk in super().synthesize(key, text):
                yield chunk

    speech, display = CountSpeech(), Transcript()
    async with Session(speech, ForeignModel(), sink, transcript=display) as session:
        session.activate()
        pending = await session.submit(b"What is this?")
        await pending
        assert speech.synthesized == []
        messages = display.snapshot()["messages"]
        assert [message["role"] for message in messages] == ["user"]
        error = next(event for event in session.events if event["event"] == "turn_error")
        assert error["error_code"] == "english_only_output"
        assert error["error_role"] == "llm"
        assert session.state == State.LISTENING


async def test_ui_lamp_action_uses_session_task_receipt_and_spoken_ack_without_reasoning():
    from me3_voice.tools import MockTaskExecutor

    class NoReasoning:
        calls = 0

        async def respond(self, _history):
            self.calls += 1
            raise AssertionError("a fixed lamp control must use the shared task path")
            yield

    reasoner = NoReasoning()
    executor = MockTaskExecutor()
    display = Transcript()
    call = ToolCall("ui-" + "a" * 32, "run_device_action", {"action": "light_on"})
    try:
        async with Session(Speech(), reasoner, sink, tasks=executor, transcript=display) as session:
            session.activate()
            key = await session.submit_ui_device_action(call)
            assert key.turn_id == 1
            await session.generation
            receipt = display.snapshot()["tasks"][-1]
            assert receipt["kind"] == "light_on"
            assert receipt["status"] == "simulated"
            assert receipt["mode"] == "mock"
            assert receipt["widget"]["on"] is True
            assert any(message["role"] == "assistant" for message in display.snapshot()["messages"])
            assert session.history[-1]["role"] == "assistant"
            assert reasoner.calls == 0
            assert executor.actions.effects == 1
            assert any(event["event"] == "device_ui_action_routed" for event in session.events)
    finally:
        await executor.close()


async def test_interrupt_discards_ui_device_action_that_has_not_reached_reasoning():
    from types import SimpleNamespace

    from me3_voice.tools import MockTaskExecutor

    executor = MockTaskExecutor()
    call = ToolCall("ui-" + "c" * 32, "run_device_action", {"action": "light_on"})
    try:
        async with Session(Speech(), MockLLM(), sink, tasks=executor) as session:
            session.activate()
            session.key = SimpleNamespace(turn_id=1)
            session.turn = 1
            session.ui_action_calls[1] = call
            session.ui_action_turns.add(1)
            await session.interrupt()

            assert 1 not in session.ui_action_calls
            assert 1 not in session.ui_action_turns
            assert executor.actions.effects == 0
            assert executor.actions.light is False
    finally:
        await executor.close()


async def test_unavailable_pi_light_keeps_conversation_usable_and_never_simulates():
    from me3_voice.devices import DeviceActions, UnavailablePiDeviceOutput
    from me3_voice.tools import MockTaskExecutor

    class LightThenAnswer:
        async def respond(self, history):
            if history[-1]["content"] == "turn on the light":
                yield ToolCall("pi-light", "run_device_action", {"action": "light_on"})
            else:
                yield "I can still answer while the Raspberry Pi is unavailable."

    devices = DeviceActions(output=UnavailablePiDeviceOutput("not_a_pi"), backend="pi")
    executor = MockTaskExecutor(devices=devices)
    display = Transcript()
    try:
        async with Session(
            Speech(), LightThenAnswer(), sink, tasks=executor, transcript=display
        ) as session:
            session.activate()
            await (await session.submit(b"turn on the light"))
            receipt = display.snapshot()["tasks"][-1]
            assert receipt["status"] == "unavailable" and receipt["mode"] == "reported"
            assert receipt["summary"] == "Raspberry Pi is not connected. No device action was sent."
            assert session.history[-1]["role"] == "assistant"
            assert devices.effects == 0 and devices.output.__class__.__name__ != "MockDeviceOutput"

            await (await session.submit(b"what is 2 plus 2"))
            assert session.history[-1]["content"] == (
                "I can still answer while the Raspberry Pi is unavailable."
            )
            assert len(display.snapshot()["tasks"]) == 1
            assert devices.effects == 0
    finally:
        await executor.close()


@pytest.mark.parametrize("text", ["music", "the music", "playlist"])
async def test_music_fragment_clarifies_without_reasoning_or_effect(text):
    class Model:
        async def respond(self, history):
            raise AssertionError("fragment must not reach model tool selection")
            yield

    display = Transcript()
    async with Session(Speech(), Model(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(text.encode()))
        assert display.snapshot()["tasks"] == []
        assert "play, pause, or stop" in display.snapshot()["messages"][-1]["text"]


@pytest.mark.parametrize(
    "text", ["When I was young.", "I said stop.", "Pause the music.", "Don't play music."]
)
async def test_unrequested_model_playlist_is_rejected_before_task_execution(text):
    class Model:
        async def respond(self, history):
            yield ToolCall(
                "unwanted-selection",
                "control_music",
                {"action": "play_playlist", "query": "Random Mood"},
            )

    display = Transcript()
    async with Session(Speech(), Model(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(text.encode()))
        assert display.snapshot()["tasks"] == []
        assert session.playback.queue.empty()


def test_explicit_stop_correction_and_volume_level_have_bounded_routes():
    from me3_voice.session import direct_music_control, direct_music_request

    history = [
        {"role": "tool", "tool_request": {"name": "control_music", "arguments": {"action": "play"}}}
    ]
    assert direct_music_control("I said stop.", 4, history).arguments == {"action": "stop"}
    assert direct_music_control("Stop speaking.", 4, history) is None
    assert direct_music_control("I said stop.", 4) is None
    assert direct_music_control("Volume level twenty.", 5).arguments == {
        "action": "volume_set",
        "query": "20",
    }
    assert direct_music_request("Play music Bruno Mars.", 6, []).arguments["query"] == "Bruno Mars"


async def test_play_fragment_stop_correction_and_followup_share_one_action_path():
    from me3_voice.music import MusicController
    from me3_voice.tools import MockTaskExecutor

    class NoModel:
        async def respond(self, history):
            raise AssertionError("explicit play/stop/volume and clarification should route locally")
            yield

    music = MusicController()
    display = Transcript()
    async with Session(
        Speech(), NoModel(), sink, transcript=display, tasks=MockTaskExecutor(music=music)
    ) as session:
        session.activate()
        for text in ["Play music Bruno Mars.", "music", "I said stop.", "Volume level twenty."]:
            await (await session.submit(text.encode()))
        assert not music.playing
        tasks = display.snapshot()["tasks"]
        assert [task["kind"] for task in tasks] == ["music_play", "music_stop", "music_volume_set"]
        assert len(music.operations) == 3
        await session.interrupt()
        assert len(music.operations) == 3


async def test_initials_only_transcript_is_preserved_and_requests_repeat_without_tools():
    class NoModel:
        async def respond(self, history):
            raise AssertionError("initials fragment needs clarification")
            yield

    display = Transcript()
    async with Session(Speech(), NoModel(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(b"R A."))
        messages = display.snapshot()["messages"]
        assert messages[0]["text"] == "R A."
        assert "repeat the full request" in messages[-1]["text"]
        assert not display.snapshot()["tasks"]


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("thirty seven", 37),
        ("thirty-seven", 37),
        ("zero", 0),
        ("one hundred", 100),
        ("nineteen", 19),
        ("ninety nine", 99),
        ("37", 37),
    ],
)
async def test_spoken_brightness_uses_exact_shared_task_and_preserves_transcript(spoken, expected):
    from me3_voice.transcript import Transcript

    class NoModel:
        async def respond(self, history):
            raise AssertionError("explicit brightness must not wait for a model")
            yield ""

    display = Transcript()
    text = f"Set brightness to {spoken} percent."
    async with Session(Speech(), NoModel(), sink, transcript=display) as session:
        session.activate()
        await (await session.submit(text.encode()))
        snap = display.snapshot()
        assert snap["messages"][0]["text"] == text
        assert len(snap["tasks"]) == 1
        assert snap["tasks"][0]["kind"] == "set_brightness"
        assert snap["tasks"][0]["widget"]["brightness"] == expected
        await session.interrupt()
        assert len(display.snapshot()["tasks"]) == 1


def test_brightness_does_not_guess_uncertain_or_out_of_range_values():
    for text in [
        "Set brightness to twenty ten percent.",
        "Set brightness to 101 percent.",
        "Set brightness to thirty seven or forty percent.",
        "Set brightness to half percent.",
    ]:
        assert direct_light_command(text, 1) is None
