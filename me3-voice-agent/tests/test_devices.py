import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from me3_voice.devices import (
    DeviceActions,
    MockDeviceOutput,
    OpenMeteoWeather,
    clock_timezone_label,
    load_device_service,
)


async def act(service, index, action, value=None):
    return await service.run(f"session:{index}:call-{index}", action, value)


async def test_lamp_state_is_serialized_and_idempotent():
    output = MockDeviceOutput()
    service = DeviceActions(output=output)
    on = await act(service, 1, "light_on")
    green = await act(service, 2, "set_color", "green")
    dim = await act(service, 3, "set_brightness", "20")
    off = await act(service, 4, "light_off")

    assert on["widget"] == {"type": "lamp", "on": True, "color": "blue", "brightness": 60}
    assert green["widget"]["on"] is True and green["widget"]["color"] == "green"
    assert dim["widget"]["brightness"] == 20 and dim["widget"]["on"] is True
    assert off["widget"]["on"] is False and off["widget"]["color"] == "green"
    before = len(output.events)
    assert await act(service, 4, "light_off") == off
    assert len(output.events) == before
    assert service.effects == 4
    await service.close()


async def test_rgb_led_accepts_full_hex_gamut_and_continuous_brightness():
    output = MockDeviceOutput()
    service = DeviceActions(output=output)
    color = await act(service, 1, "set_color", "#12ab34")
    assert color["widget"] == {
        "type": "lamp",
        "on": True,
        "color": "#12AB34",
        "brightness": 60,
    }
    assert output.led_state == pytest.approx((18 / 255 * 0.6, 171 / 255 * 0.6, 52 / 255 * 0.6))
    level = await act(service, 2, "set_brightness", "37")
    assert level["widget"]["brightness"] == 37
    assert output.led_state == pytest.approx((18 / 255 * 0.37, 171 / 255 * 0.37, 52 / 255 * 0.37))
    with pytest.raises(ValueError, match="unsupported device action value"):
        await act(service, 3, "set_color", "#GG2233")
    await service.close()


async def test_pi_mode_uses_accepted_receipts_and_disables_mock_sensor_claims():
    from types import SimpleNamespace

    from me3_voice.providers import ToolCall
    from me3_voice.tools import MockTaskExecutor

    service = DeviceActions(output=MockDeviceOutput(), backend="pi")
    executor = MockTaskExecutor(devices=service)
    key = SimpleNamespace(session_id="pi-session", turn_id=1)
    result = await executor.execute(
        key, ToolCall("lamp-1", "run_device_action", {"action": "light_on"})
    )
    assert result["mode"] == "reported" and result["result_status"] == "accepted"
    assert "no sensor" in result["summary"]
    with pytest.raises(ValueError, match="sensor readback"):
        await executor.execute(key, ToolCall("read-1", "read_mock_light", {}))
    with pytest.raises(ValueError, match="legacy mock-light"):
        await executor.execute(key, ToolCall("legacy-1", "set_mock_light", {"enabled": False}))
    await service.close()


async def test_timer_schedule_survives_temperature_overlay_and_is_replaced_or_cancelled():
    output = MockDeviceOutput()
    service = DeviceActions(output=output)
    first = await act(service, 1, "set_timer", "10 seconds")
    timer_id = service.schedule["id"]
    assert first["result_status"] == "simulated"
    assert first["widget"]["active"] is True
    assert output.display_state[0] == "0010"

    temperature = await act(service, 2, "set_temperature", "22 degrees")
    assert temperature["widget"]["measurement"] is False
    assert output.display_state[0] == "  22"
    assert service.schedule["id"] == timer_id

    service._overlay["until"] = time.monotonic() - 1
    async with service._lock:
        await service._render_schedule_locked()
    assert service.schedule["active"] is True
    assert output.display_state[0].isdigit() and output.display_state[1] is True

    alarm = await act(service, 3, "set_alarm", "8:00 AM")
    assert alarm["replaced_task"] == {"session_id": "session", "turn_id": 1, "call_id": "call-1"}
    assert service.schedule["kind"] == "alarm"
    cancelled = await act(service, 4, "cancel_schedule")
    assert cancelled["replaced_task"] == {
        "session_id": "session",
        "turn_id": 3,
        "call_id": "call-3",
    }
    assert service.schedule is None
    assert output.display_state[0] == "    "
    await service.close()


async def test_timer_uses_elapsed_time_and_marks_display_without_claiming_an_alert():
    output = MockDeviceOutput()
    service = DeviceActions(output=output)
    result = await act(service, 1, "set_timer", "10 seconds")
    service.schedule["deadline_mono"] = time.monotonic() - 0.1
    await asyncio.sleep(0.65)
    assert service.schedule["active"] is False
    assert output.display_state == ("0000", True)
    assert "no buzzer" in result["summary"]
    await service.close()


async def test_temperature_overlay_without_schedule_clears_after_deadline():
    output = MockDeviceOutput()
    service = DeviceActions(output=output)
    await act(service, 1, "set_temperature", "18 degrees")
    assert output.display_state[0] == "  18"
    service._overlay["until"] = time.monotonic() - 1
    async with service._lock:
        await service._render_schedule_locked()
    assert output.display_state[0] == "    "
    await service.close()


async def test_reminders_are_atomic_persistent_and_duplicate_safe(tmp_path):
    store = tmp_path / "reminders.json"
    service = DeviceActions(reminder_store=store)
    created = await act(service, 1, "create_reminder", "study")
    duplicate = await act(service, 2, "create_reminder", "study")
    assert created["result_status"] == "completed"
    assert duplicate["widget"]["items"] == ["study"]
    assert len(json.loads(store.read_text())["reminders"]) == 1
    service2 = DeviceActions(reminder_store=store)
    listed = await act(service2, 3, "list_reminders")
    assert listed["widget"] == {"type": "reminders", "items": ["study"], "persistent": True}
    assert store.stat().st_mode & 0o077 == 0
    await service.close()
    await service2.close()


async def test_weather_is_unavailable_by_default_and_open_meteo_adapter_is_mockable():
    service = DeviceActions()
    unavailable = await act(service, 1, "get_weather")
    assert unavailable["result_status"] == "unavailable"
    assert "not configured" in unavailable["summary"]
    await service.close()

    now = datetime.now(timezone.utc).replace(microsecond=0)
    called = []

    def fetch(url):
        called.append(url)
        return {
            "current_units": {"temperature_2m": "°C"},
            "current": {
                "temperature_2m": 29.2,
                "weather_code": 2,
                "time": now.isoformat(),
            },
        }

    weather = OpenMeteoWeather(
        latitude=14.6488,
        longitude=121.0509,
        timezone_name="Asia/Manila",
        location="Quezon City, Philippines",
        fetch=fetch,
    )
    service = DeviceActions(weather=weather)
    result = await act(service, 2, "get_weather")
    assert result["mode"] == "reported" and result["result_status"] == "completed"
    assert result["widget"]["temperature_c"] == 29.2
    assert len(called) == 1 and "api.open-meteo.com" in called[0]
    await act(service, 3, "get_weather")
    assert len(called) == 1
    await service.close()


async def test_no_config_selects_mock_and_pi_config_must_be_private_cache_file(tmp_path):
    service = load_device_service()
    assert service.backend == "mock"

    with pytest.raises(ValueError, match="var/cache"):
        load_device_service(tmp_path / "devices.json")


async def test_pi_adapter_initialization_failure_stays_unavailable_and_never_becomes_mock(
    tmp_path, monkeypatch
):
    import me3_voice.devices as devices

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "var/cache/devices.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"schema_version":1,"backend":"pi"}', encoding="utf-8")

    def missing_gpio():
        raise ImportError("gpio support is not installed")

    monkeypatch.setattr(devices, "PiDeviceOutput", missing_gpio)
    service = load_device_service(config)
    try:
        assert service.backend == "pi"
        assert service.capabilities()["status"] == "unavailable"
        assert service.capabilities()["available"] is False
        light = await service.run("session:1:pi-light", "light_on", None)
        assert light["result_status"] == "unavailable"
        assert light["mode"] == "reported"
        assert light["summary"] == "Raspberry Pi is not connected. No device action was sent."
        assert service.effects == 0
        assert service.output.__class__.__name__ != "MockDeviceOutput"
        clock = await service.run("session:1:clock", "get_time", None)
        assert clock["result_status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_time_accepts_valid_timezone_and_rejects_unknown():
    from me3_voice.devices import DeviceActions, MockDeviceOutput

    device = DeviceActions(output=MockDeviceOutput(), backend="mock")
    try:
        result = await device.run("timezone-manila", "get_time", "Asia/Manila")
        assert result["widget"]["timezone"] == "Asia/Manila"
        alias = await device.run("timezone-philippines", "get_time", "Philippines")
        assert alias["widget"]["timezone"] == "Asia/Manila"
        with pytest.raises(ValueError, match="unknown timezone"):
            await device.run("timezone-invalid", "get_time", "Invalid/Place")
    finally:
        await device.close()


def test_local_clock_label_uses_numeric_offset_not_a_misleading_host_abbreviation():
    host_now = datetime(2026, 11, 4, 13, 45, tzinfo=timezone(timedelta(hours=8), name="PST"))
    assert clock_timezone_label(host_now) == "UTC+08:00"
    assert clock_timezone_label(host_now, "Asia/Manila") == "Asia/Manila"


async def test_timer_survives_voice_disconnect_and_expires_without_reexecution():
    from me3_voice.providers import MockLLM
    from me3_voice.session import Session
    from me3_voice.tools import MockTaskExecutor

    class Speech:
        async def transcribe(self, key, pcm):
            return pcm.decode()

        async def synthesize(self, key, text):
            yield b"\0\0"

    async def sink(pcm):
        pass

    output = MockDeviceOutput()
    devices = DeviceActions(output=output)
    tasks = MockTaskExecutor(devices=devices)
    try:
        async with Session(Speech(), MockLLM(), sink, tasks=tasks, close_tasks=False) as first:
            first.activate()
            await (await first.submit(b"simulate timer 10 seconds"))
            schedule_id = devices.schedule["id"]
        assert devices.schedule["id"] == schedule_id and devices.schedule["active"]
        async with Session(Speech(), MockLLM(), sink, tasks=tasks, close_tasks=False) as second:
            second.activate()
            await (await second.submit(b"hello"))
            assert devices.schedule["id"] == schedule_id
            assert devices.effects == 1
            devices.schedule["deadline_mono"] = time.monotonic() - 1
            async with asyncio.timeout(1.5):
                while devices.schedule["active"]:
                    await asyncio.sleep(0.01)
            assert not devices.schedule["active"]
            assert output.display_state == ("0000", True)
            assert devices.effects == 1
    finally:
        await tasks.close()
        await devices.close()
