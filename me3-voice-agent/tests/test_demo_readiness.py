import asyncio
from datetime import datetime, timezone

import pytest

from me3_voice.contracts import TurnKey
from me3_voice.devices import DeviceActions, OpenMeteoWeather, WeatherLocationError
from me3_voice.session import Session, direct_me2_command
from me3_voice.tools import MockTaskExecutor
from me3_voice.transcript import Transcript


async def test_required_device_commands_reach_shared_executor_over_eight_turns():
    commands = iter(
        [
            "set a timer for thirty seconds",
            "cancel the timer",
            "set temperature to twenty two degrees",
            "set an alarm for six am",
            "cancel the alarm",
            "remind me to drink water",
            "list my reminders",
            "what is the weather",
            "turn on the lights",
        ]
    )

    class Speech:
        async def transcribe(self, key, pcm):
            return next(commands)

        async def synthesize(self, key, text):
            yield b"\0\0" * 160

    class NoModel:
        async def respond(self, history):
            raise AssertionError("Exact supported command must reach shared tools")
            yield "unreachable"

    async def sink(pcm):
        await asyncio.sleep(0)

    tasks = MockTaskExecutor()
    book = Transcript()
    session = Session(Speech(), NoModel(), sink, transcript=book, tasks=tasks)
    await session.__aenter__()
    session.activate()
    try:
        for _ in range(9):
            await (await session.submit(b"\0\0" * 160))
        rows = book.snapshot()["tasks"]
        assert len(rows) == 9
        assert rows[-1]["widget"]["on"] is True
        assert tasks.actions.reminders == ["drink water"]
        assert rows[-2]["status"] == "unavailable"  # No invented weather.
        assert not any(e["event"] == "turn_error" for e in session.events)
        assert not tasks.actions.schedule
    finally:
        await session.__aexit__(None, None, None)


def test_exact_route_rejects_negation_and_compound_tasks():
    for text in (
        "don't set a timer",
        "set a timer for one minute and call mom",
        "remind me to exercise and study",
        "cancel that",
    ):
        assert direct_me2_command(text, 1) is None


async def test_dynamic_weather_uses_named_source_and_clarifies_ambiguity():
    requests = []

    async def fetch(url):
        requests.append(url)
        if "geocoding" in url:
            return {
                "results": [
                    {
                        "name": "Tokyo",
                        "country": "Japan",
                        "latitude": 35.7,
                        "longitude": 139.7,
                        "timezone": "Asia/Tokyo",
                    }
                ]
            }
        return {
            "current_units": {"temperature_2m": "°C"},
            "current": {
                "temperature_2m": 21,
                "weather_code": 0,
                "time": datetime.now(timezone.utc).timestamp(),
            },
        }

    weather = OpenMeteoWeather(
        latitude=14.6488,
        longitude=121.0509,
        timezone_name="Asia/Manila",
        location="Quezon City, Philippines",
        fetch=fetch,
    )
    result = await weather.current_at("Tokyo, Japan")
    assert result["location"] == "Tokyo, Japan"
    assert "name=Tokyo%2C+Japan" in requests[0]
    assert "latitude=35.7" in requests[1]
    devices = DeviceActions(weather=weather)
    clock = await devices.run("default-clock", "get_time", None)
    assert clock["widget"]["timezone"] == "Asia/Manila"
    await devices.close()

    async def ambiguous(url):
        return {"results": [{}, {}]}

    weather.fetch = ambiguous
    with pytest.raises(WeatherLocationError):
        await weather.current_at("Springfield")


def test_answer_only_metadata_preserves_receipts_and_cannot_hide_artifacts():
    book = Transcript()
    book.start_session("test")
    key = TurnKey(session_id="test", turn_id=1)
    book.begin_turn(key)
    book.task_started(key, "answer", "Assistant response", mode="reported")
    book.task_finished(key, "answer", "reported", "Answer returned", answer_only=True)
    assert book.snapshot()["tasks"][0]["answer_only"] is True
    book.task_started(key, "file", "Project", mode="reported")
    with pytest.raises(ValueError, match="cannot conceal"):
        book.task_finished(key, "file", "reported", artifacts=["index.html"], answer_only=True)


async def test_phone_effects_use_device_identity_and_unknown_is_not_retried():
    from me3_voice.communications import Communications, MessagingUncertain

    phone = Communications()
    phone.enabled = phone.on_pi = True
    calls = []

    class Caller:
        def dial(self, number):
            calls.append("dial")
            return {"status": "dialing"}

    class SMS:
        async def _operate(self, number):
            calls.append("sms")
            raise MessagingUncertain("submission uncertain")

    phone.caller, phone.sms = Caller(), SMS()
    devices = DeviceActions(communications=phone)
    result = await devices.run("one-call", "call_demo", None)
    assert "not yet confirmed" in result["summary"]
    assert await devices.run("one-call", "call_demo", None) == result
    uncertain = await devices.run("one-sms", "message_demo", None)
    assert uncertain["result_status"] == "unknown"
    assert await devices.run("one-sms", "message_demo", None) == uncertain
    assert calls == ["dial", "sms"]
    await devices.close()


async def test_pi_playback_stop_clears_queue_and_reaps_only_its_process(monkeypatch):
    from me3_voice.pi_audio import PipeWirePlayback

    writes, acks, processes = [], [], []

    class Process:
        returncode = None

        def __init__(self):
            self.stdin = self

        def write(self, pcm):
            writes.append(pcm)

        async def drain(self):
            await asyncio.sleep(0)

        def terminate(self):
            self.returncode = -15

        async def wait(self):
            return self.returncode

    async def spawn(*args, **kwargs):
        process = Process()
        processes.append(process)
        return process

    async def ack(number):
        acks.append(number)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    output = PipeWirePlayback("explicit-usb-sink", ack, None)
    await output.write(1, b"\0\0" * 8000)
    await output.write(2, b"\0\0" * 8000)
    for _ in range(20):
        if writes:
            break
        await asyncio.sleep(0)
    await asyncio.wait_for(output.stop(), 2)
    assert len(processes) == 1 and processes[0].returncode == -15
    assert output.queue.empty() and output.worker is None and not acks


async def test_raw_transcription_boundary_survives_normalization_without_ui_rewrite():
    from me3_voice.providers import BufferedASRStream

    class Provider:
        last_raw_text = None

        async def transcribe(self, pcm):
            self.last_raw_text = "The person says: 'What is two plus two?'"
            return "What is two plus two?"

    stream = BufferedASRStream(Provider())
    await stream.append(b"\0\0" * 160)
    result = await stream.finish()
    assert result.text == "What is two plus two?"
    assert result.provider_text.startswith("The person says:")
    await stream.aclose()


async def test_demo_onit_profile_accepts_six_turns_without_four_request_test_cap(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from me3_voice.configuration import BackendSpec
    from me3_voice.onit_tasks import OnitPrimaryReasoning, OnitTaskExecutor

    monkeypatch.chdir(tmp_path)
    (tmp_path / "var").mkdir()
    requests = []

    async def runner(task, directory, budget):
        requests.append(budget)
        return {
            "mode": "local-answer",
            "summary": "Answer returned.",
            "answer": "A concise answer.",
            "artifacts": [],
            "requests": 1,
        }

    class Speech:
        async def transcribe(self, key, pcm):
            return "Tell me an interesting fact."

        async def synthesize(self, key, text):
            yield b"\0\0" * 160

    async def sink(pcm):
        pass

    spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        model="openai/gpt-oss-120b",
        auth_env="ME3_LLM_API_KEY",
        allow_live=True,
        request_limit=None,
    )
    tasks = OnitTaskExecutor(
        Path("/usr/bin/python3").resolve(),
        spec,
        {"ME3_LLM_API_KEY": "offline-only"},
        max_requests=None,
        runner=runner,
    )
    book = Transcript()
    async with Session(
        Speech(), OnitPrimaryReasoning(), sink, tasks=tasks, transcript=book
    ) as session:
        session.activate()
        for _ in range(6):
            await (await session.submit(b"\0\0" * 160))
        assert requests == [None] * 6
        assert len(book.snapshot()["tasks"]) == 6
        assert all(t["answer_only"] for t in book.snapshot()["tasks"])
        assert not any(e["event"] == "turn_error" for e in session.events)
