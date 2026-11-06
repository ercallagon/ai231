"""One serialized owner for Swift's shared LED, TM1637, timer and reminder state.

Mock output is the default. Pi GPIO support is lazy and only selected by an
explicit, private config under var/cache/. This module never opens GPIO merely
because it is imported.
"""

import asyncio
import inspect
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ACTION_TITLES = {
    "light_on": "Turn on the light",
    "light_off": "Turn off the light",
    "set_color": "Change light color",
    "set_brightness": "Adjust brightness",
    "set_timer": "Set a timer",
    "set_alarm": "Set an alarm",
    "cancel_schedule": "Cancel timer or alarm",
    "set_temperature": "Show temperature setpoint",
    "create_reminder": "Save a reminder",
    "list_reminders": "Check reminders",
    "get_time": "Check the time",
    "get_weather": "Check the weather",
    "call_demo": "Call demo contact",
    "message_demo": "Send demo message",
}
COLOR_RGB = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 1.0),
    "pink": (1.0, 0.41, 0.71),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "warm white": (1.0, 0.94, 0.82),
    "teal": (0.0, 0.65, 0.62),
    "amber": (1.0, 0.67, 0.0),
    "violet": (0.56, 0.0, 1.0),
    "lime": (0.5, 1.0, 0.0),
    "navy": (0.0, 0.0, 0.5),
    "coral": (1.0, 0.5, 0.31),
}


def normalize_rgb_color(value):
    """Return a supported color name or canonical #RRGGBB value, else None."""
    if not isinstance(value, str):
        return None
    name = value.strip().casefold()
    if name in COLOR_RGB:
        return name
    if re.fullmatch(r"#?[0-9a-f]{6}", name):
        return "#" + name.removeprefix("#").upper()
    return None


def rgb_channels(value):
    """Convert a validated color name or hex value into gpiozero RGB channels."""
    normalized = normalize_rgb_color(value)
    if normalized is None:
        raise ValueError("unsupported RGB color")
    if normalized in COLOR_RGB:
        return COLOR_RGB[normalized]
    digits = normalized[1:]
    return tuple(int(digits[index : index + 2], 16) / 255 for index in (0, 2, 4))


def clock_timezone_label(now, requested=None):
    """Use an IANA name when requested; otherwise avoid misleading host abbreviations."""
    if requested:
        return requested
    offset = now.utcoffset()
    if offset is None:
        return "local timezone"
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    hours, minutes = divmod(abs(minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def valid_action_value(action, value):
    if action == "set_color":
        return normalize_rgb_color(value) is not None
    return action in ACTION_VALUES and value in ACTION_VALUES[action]


ACTION_VALUES = {
    "set_color": set(COLOR_RGB),
    "set_brightness": {str(value) for value in range(101)},
    "set_timer": {"10 seconds", "30 seconds", "1 minute"},
    "set_alarm": {"6:00 AM", "8:00 AM", "9:00 PM"},
    "set_temperature": {"18 degrees", "22 degrees", "26 degrees"},
    "create_reminder": {"drink water", "study", "exercise"},
}
SEGMENTS = (0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F)


class MockDeviceOutput:
    """Event-ledger output for tests and the laptop/HPC demo."""

    target = "mock"
    available = True

    def __init__(self):
        self.events = []
        self.led_state = (0.0, 0.0, 0.0)
        self.display_state = ("    ", False)
        self.closed = False

    def led(self, rgb):
        if self.closed:
            raise RuntimeError("device output is closed")
        self.led_state = tuple(rgb)
        self.events.append(("led", self.led_state))

    def display(self, digits, colon=False):
        if self.closed:
            raise RuntimeError("device output is closed")
        if len(digits) != 4 or any(char not in "0123456789 " for char in digits):
            raise ValueError("TM1637 display requires four digits or spaces")
        self.display_state = digits, bool(colon)
        self.events.append(("display", digits, bool(colon)))

    def close(self):
        if not self.closed:
            self.led_state = (0.0, 0.0, 0.0)
            self.display_state = ("    ", False)
            self.events.extend([("led", self.led_state), ("display", "    ", False)])
            self.closed = True


class PiDeviceOutput:
    """Lazy Pi 5 RGBLED/TM1637 adapter preserving the documented ME2 wiring.

    BCM 17/27/22 are the RGB LED channels; BCM 23/24 are TM1637 CLK/DIO.
    Instantiating this class opens the pins, so config selection is explicit.
    """

    def __init__(self):
        model_path = Path("/proc/device-tree/model")
        try:
            model = model_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
        except OSError:
            model = ""
        if "raspberry pi" not in model.casefold():
            raise OSError("Pi device target is not present")
        from gpiozero import RGBLED
        from gpiozero.pins.lgpio import LGPIOFactory

        self.factory = LGPIOFactory()
        self.rgb = None
        self.clk = self.dio = None
        self.closed = False
        self.available = True
        self.target = "pi"
        try:
            self.rgb = RGBLED(
                17, 27, 22, active_high=True, initial_value=(0, 0, 0), pin_factory=self.factory
            )
            self.clk = self.factory.pin(23)
            self.dio = self.factory.pin(24)
            self.clk.function = "output"
            self.clk.state = 1
            self.dio.function = "input"
            self.dio.pull = "up"
            self.display("    ")
        except BaseException:
            self.close()
            raise

    def led(self, rgb):
        if self.closed:
            raise RuntimeError("Pi device output is closed")
        self.rgb.value = rgb

    def _clock(self, value):
        self.clk.state = value
        time.sleep(0.0001)

    def _data(self, value):
        if value:
            self.dio.function = "input"
            self.dio.pull = "up"
        else:
            self.dio.function = "output"
            self.dio.state = 0
        time.sleep(0.0001)

    def _start(self):
        self._data(1)
        self._clock(1)
        self._data(0)
        self._clock(0)

    def _stop(self):
        self._clock(0)
        self._data(0)
        self._clock(1)
        self._data(1)

    def _write(self, value):
        for bit in range(8):
            self._clock(0)
            self._data((value >> bit) & 1)
            self._clock(1)
        self._clock(0)
        self._data(1)
        self._clock(1)
        acknowledged = not self.dio.state
        self._clock(0)
        if not acknowledged:
            raise OSError("TM1637 did not acknowledge; check the existing BCM 23/24 wiring")

    def _packet(self, values):
        self._start()
        try:
            for value in values:
                self._write(value)
        finally:
            self._stop()

    def display(self, digits, colon=False):
        if self.closed:
            raise RuntimeError("Pi device output is closed")
        if len(digits) != 4 or any(char not in "0123456789 " for char in digits):
            raise ValueError("TM1637 display requires four digits or spaces")
        segments = [0 if char == " " else SEGMENTS[int(char)] for char in digits]
        if colon:
            segments[1] |= 0x80
        self._packet([0x40])
        self._packet([0xC0, *segments])
        self._packet([0x8B])

    def close(self):
        if self.closed:
            return
        try:
            if self.rgb:
                self.rgb.value = (0, 0, 0)
            if self.clk and self.dio:
                try:
                    self.display("    ")
                except Exception:
                    pass
        finally:
            self.closed = True
            if self.rgb:
                self.rgb.close()
            if self.clk:
                self.clk.close()
            if self.dio:
                self.dio.close()
            self.factory.close()


class UnavailablePiDeviceOutput:
    """Explicit Pi target that failed preflight; it never simulates output."""

    target = "pi"
    available = False

    def __init__(self, reason):
        self.reason = reason

    def led(self, _rgb):
        raise RuntimeError("Raspberry Pi GPIO output is unavailable")

    def display(self, _digits, _colon=False):
        raise RuntimeError("Raspberry Pi display output is unavailable")

    def close(self):
        return None


class WeatherLocationError(ValueError):
    """A named location needs clarification; never guess among geocoder matches."""


class OpenMeteoWeather:
    """Optional, on-demand current-conditions adapter; no request at startup."""

    def __init__(self, *, latitude, longitude, timezone_name, location, fetch=None):
        if type(latitude) not in (int, float) or not -90 <= latitude <= 90:
            raise ValueError("weather latitude must be within -90..90")
        if type(longitude) not in (int, float) or not -180 <= longitude <= 180:
            raise ValueError("weather longitude must be within -180..180")
        if not isinstance(location, str) or not 1 <= len(location) <= 80:
            raise ValueError("weather location must be 1..80 characters")
        try:
            ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError("weather timezone must be a valid IANA zone") from exc
        self.latitude, self.longitude = float(latitude), float(longitude)
        self.timezone_name, self.location = timezone_name, location
        self.fetch = fetch or self._fetch
        self.cached = None
        self.cached_at = None

    @property
    def url(self):
        query = urlencode(
            {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current": "temperature_2m,weather_code",
                "temperature_unit": "celsius",
                "timezone": self.timezone_name,
            }
        )
        return "https://api.open-meteo.com/v1/forecast?" + query

    async def current_at(self, location=None):
        aliases = {self.location.casefold(), self.location.split(",")[0].strip().casefold()}
        if "philippines" in self.location.casefold():
            aliases.update({"philippines", "ph"})
        if not location or location.casefold() in aliases:
            return await self.current()
        query = urlencode({"name": location, "count": 5, "language": "en", "format": "json"})
        pending = self.fetch("https://geocoding-api.open-meteo.com/v1/search?" + query)
        payload = await asyncio.wait_for(pending, 3.5) if inspect.isawaitable(pending) else pending
        matches = payload.get("results", [])
        if not isinstance(matches, list) or len(matches) != 1:
            raise WeatherLocationError(
                "Which city and country should I check? Please include both in your weather request."
            )
        place = matches[0]
        selected = OpenMeteoWeather(
            latitude=place.get("latitude"),
            longitude=place.get("longitude"),
            timezone_name=place.get("timezone"),
            location=f"{place.get('name', '')}, {place.get('country', '')}",
            fetch=self.fetch,
        )
        return await selected.current()

    @staticmethod
    async def _fetch(url):
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.get(url) as response:
                response.raise_for_status()
                return await response.json(content_type=None)

    async def current(self):
        now = datetime.now(timezone.utc)
        if self.cached and now - self.cached_at < timedelta(minutes=5):
            return self.cached
        pending = self.fetch(self.url)
        payload = (
            await asyncio.wait_for(pending, timeout=3.5)
            if inspect.isawaitable(pending)
            else pending
        )
        current = payload.get("current", {})
        units = payload.get("current_units", {})
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        stamp = current.get("time")
        if (
            units.get("temperature_2m") != "°C"
            or type(temp) not in (int, float)
            or not -100 <= temp <= 70
            or type(code) is not int
            or isinstance(stamp, bool)
        ):
            raise ValueError("weather response did not match the current-conditions contract")
        try:
            observed = (
                datetime.fromtimestamp(stamp, timezone.utc)
                if type(stamp) in (int, float)
                else datetime.fromisoformat(stamp)
            )
            if observed.utcoffset() is None:
                observed = observed.replace(tzinfo=ZoneInfo(self.timezone_name))
            observed = observed.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weather observation time is invalid") from exc
        if abs((now - observed).total_seconds()) > 3 * 60 * 60:
            raise ValueError("weather observation is stale")
        conditions = {
            0: "clear",
            1: "mostly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "foggy",
            51: "drizzle",
            53: "drizzle",
            55: "drizzle",
            61: "rain",
            63: "rain",
            65: "heavy rain",
            71: "snow",
            73: "snow",
            75: "heavy snow",
            80: "rain showers",
            81: "rain showers",
            82: "heavy rain showers",
            95: "thunderstorms",
            96: "thunderstorms",
            99: "thunderstorms",
        }
        result = {
            "temperature_c": round(float(temp), 1),
            "condition": conditions.get(code, "conditions unavailable"),
            "observed_utc": observed.isoformat(),
            "location": self.location,
        }
        self.cached, self.cached_at = result, now
        return result


def _valid_state_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    cache_root = (Path.cwd() / "var" / "cache").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(cache_root) or path.is_symlink():
        raise ValueError("device state files must stay under project var/cache/")
    return resolved


def load_device_service(config_path=None):
    """Load explicit device config; absent config always constructs mock output."""
    config = {"schema_version": 1, "backend": "mock"}
    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        cache_root = (Path.cwd() / "var" / "cache").resolve()
        if path.is_symlink() or not path.resolve().is_relative_to(cache_root):
            raise ValueError("device config must be a regular file under project var/cache/")
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValueError("unsupported device config schema")
    backend = config.get("backend", "mock")
    if backend not in {"mock", "pi"}:
        raise ValueError("device backend must be mock or pi")
    allowed = {"schema_version", "backend", "reminder_store", "weather", "communications"}
    if set(config) - allowed:
        raise ValueError("device config contains unknown fields")
    reminder_path = None
    weather = None
    from .communications import Communications

    communication_config = config.get("communications", {})
    if not isinstance(communication_config, dict) or set(communication_config) - {
        "enabled",
        "phone_address",
        "call_number",
        "message_number",
    }:
        raise ValueError("invalid communication configuration")
    if type(communication_config.get("enabled", False)) is not bool:
        raise ValueError("communication enabled must be boolean")
    if communication_config.get("enabled") and (backend != "pi" or path.stat().st_mode & 0o077):
        raise ValueError("phone actions require explicit Pi mode and an owner-only configuration")
    communications = Communications(**communication_config)
    weather_config = config.get("weather")
    if weather_config is not None:
        if not isinstance(weather_config, dict) or set(weather_config) - {
            "provider",
            "enabled",
            "latitude",
            "longitude",
            "timezone",
            "location",
        }:
            raise ValueError("invalid weather config fields")
        if (
            weather_config.get("provider") != "open_meteo"
            or type(weather_config.get("enabled")) is not bool
        ):
            raise ValueError("weather provider must be open_meteo with an explicit enabled flag")
        if weather_config["enabled"]:
            weather = OpenMeteoWeather(
                latitude=weather_config.get("latitude"),
                longitude=weather_config.get("longitude"),
                timezone_name=weather_config.get("timezone"),
                location=weather_config.get("location"),
            )
    if backend == "pi":
        reminder_path = _valid_state_path(
            config.get("reminder_store", "var/cache/device-state/reminders.json")
        )
        try:
            output = PiDeviceOutput()
        except Exception as exc:
            # Keep voice, software tasks and diagnostics alive. Crucially this
            # is an unavailable Pi target, never a mock fallback.
            output = UnavailablePiDeviceOutput(type(exc).__name__)
    else:
        output = MockDeviceOutput()
        if "reminder_store" in config:
            reminder_path = _valid_state_path(config["reminder_store"])
    try:
        return DeviceActions(
            output=output,
            backend=backend,
            reminder_store=reminder_path,
            weather=weather,
            communications=communications,
        )
    except BaseException:
        output.close()
        raise


class DeviceActions:
    """Idempotent action service and sole owner of the shared device output."""

    def __init__(
        self, *, output=None, backend="mock", reminder_store=None, weather=None, communications=None
    ):
        if backend not in {"mock", "pi"}:
            raise ValueError("unsupported device backend")
        self.output = output or MockDeviceOutput()
        self.backend = backend
        self.mode = "mock" if backend == "mock" else "reported"
        self.reminder_store = Path(reminder_store) if reminder_store else None
        self.weather = weather
        self.communications = communications
        self.operations = {}
        self.capacity = 128
        self.light = False
        self.color = "blue"
        self.brightness = 60
        self.timers = []
        self.alarms = []
        self.temperature = None
        self.reminders = []
        self.effects = 0
        self.schedule = None
        self._overlay = None
        self._runner = None
        self._overlay_task = None
        self._last_display = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._load_reminders()

    def capabilities(self):
        """Report the configured adapter without implying physical readback."""
        available = getattr(self.output, "available", self.backend == "mock") is True
        return {
            "target": self.backend,
            "status": "simulated"
            if self.backend == "mock"
            else "ready_unverified"
            if available
            else "unavailable",
            "available": available,
            "physical_readback": False,
            "light_actions": ["light_on", "light_off", "set_color", "set_brightness"],
            "display_actions": ["set_timer", "set_alarm", "cancel_schedule", "set_temperature"],
            "message": "Mock device adapter · no GPIO output"
            if self.backend == "mock"
            else "Pi GPIO adapter ready · physical output is not sensor-verified"
            if available
            else "Raspberry Pi is not connected",
        }

    def _base_result(
        self, *, result_status, summary, display="", action=None, widget=None, **extra
    ):
        result = {
            "mode": self.mode,
            "verified_in_memory": self.mode == "mock",
            "result_status": result_status,
            "summary": summary,
            "display": display,
        }
        if action:
            result["action"] = action
        if widget is not None:
            result["widget"] = widget
        result.update(extra)
        return result

    async def set_light(self, operation_id, enabled):
        if not operation_id or type(enabled) is not bool:
            raise ValueError("expected operation id and boolean light state")
        return await self.run(operation_id, "light_on" if enabled else "light_off", None)

    async def run(self, operation_id, action, value):
        if not operation_id or action not in ACTION_TITLES:
            raise ValueError("unsupported device action")
        if action == "get_time" and isinstance(value, str):
            value = {"philippines": "Asia/Manila", "japan": "Asia/Tokyo"}.get(
                value.strip().casefold(), value
            )
        if action == "set_color":
            value = normalize_rgb_color(value)
            if value is None:
                raise ValueError("unsupported device action value")
        elif action in ACTION_VALUES:
            if value not in ACTION_VALUES[action]:
                raise ValueError("unsupported device action value")
        elif action == "get_time" and value is not None:
            if not isinstance(value, str) or len(value) > 80:
                raise ValueError("invalid timezone")
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError):
                raise ValueError("unknown timezone; use an IANA timezone") from None
        elif action == "get_weather" and value is not None:
            if (
                not isinstance(value, str)
                or not 2 <= len(value) <= 80
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("invalid weather location")
        elif value is not None:
            raise ValueError("this device action takes no value")
        arguments = (action, value)
        if operation_id in self.operations:
            previous, task = self.operations[operation_id]
            if previous != arguments:
                raise ValueError("idempotency key reused with different arguments")
        else:
            if len(self.operations) >= self.capacity:
                raise ValueError("device operation ledger full")
            task = asyncio.create_task(self._run(action, value, operation_id))
            self.operations[operation_id] = arguments, task
        return await asyncio.shield(task)

    def _current_led(self):
        scale = self.brightness / 100 if self.light else 0.0
        self.output.led(tuple(channel * scale for channel in rgb_channels(self.color)))

    def _read_reminders(self):
        if not self.reminder_store or not self.reminder_store.exists():
            return [{"slot_value": value, "created_at": None} for value in self.reminders]
        try:
            payload = json.loads(self.reminder_store.read_text(encoding="utf-8"))
            entries = payload["reminders"]
            if payload.get("version") != 1 or not isinstance(entries, list):
                raise ValueError
            values, seen = [], set()
            for entry in entries:
                value = entry.get("slot_value") if isinstance(entry, dict) else None
                stamp = entry.get("created_at") if isinstance(entry, dict) else None
                if value not in ACTION_VALUES["create_reminder"] or value in seen:
                    raise ValueError
                datetime.fromisoformat(stamp)
                values.append(entry)
                seen.add(value)
            return values
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError("Saved reminders are invalid or unavailable") from exc

    def _load_reminders(self):
        entries = self._read_reminders()
        self.reminders = [entry["slot_value"] for entry in entries]

    def _save_reminders(self):
        if not self.reminder_store:
            return
        path = self.reminder_store
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        entries = [
            {"slot_value": value, "created_at": created}
            for value, created in self._reminder_entries
        ]
        fd, temporary = tempfile.mkstemp(prefix=".reminders-", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "reminders": entries}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @property
    def _reminder_entries(self):
        if not self.reminder_store:
            return [(value, datetime.now().astimezone().isoformat()) for value in self.reminders]
        current = self._read_reminders()
        by_value = {entry["slot_value"]: entry["created_at"] for entry in current}
        return [
            (value, by_value.get(value) or datetime.now().astimezone().isoformat())
            for value in self.reminders
        ]

    def _write_display(self, digits, colon=False):
        state = (digits, bool(colon))
        if state != self._last_display:
            self.output.display(digits, colon)
            self._last_display = state

    def _replace_schedule(self, schedule):
        previous = self.schedule if self.schedule and self.schedule.get("active") else None
        replaced_task = self._task_owner(previous["operation_id"]) if previous else None
        if self._runner and not self._runner.done():
            self._runner.cancel()
        self.schedule = schedule
        self._overlay = None
        if self._overlay_task and not self._overlay_task.done():
            self._overlay_task.cancel()
        self._runner = asyncio.create_task(self._run_schedule(schedule["id"]))
        return replaced_task

    @staticmethod
    def _task_owner(operation_id):
        pieces = operation_id.rsplit(":", 2)
        if len(pieces) != 3 or not pieces[1].isdigit():
            return None
        return {"session_id": pieces[0], "turn_id": int(pieces[1]), "call_id": pieces[2]}

    async def _render_schedule_locked(self):
        schedule = self.schedule
        if self._overlay and time.monotonic() < self._overlay["until"]:
            self._write_display(self._overlay["digits"], False)
            return
        self._overlay = None
        if not schedule:
            self._write_display("    ")
        elif schedule["kind"] == "timer":
            remaining = max(0, math.ceil(schedule["deadline_mono"] - time.monotonic()))
            self._write_display(f"{remaining // 60:02d}{remaining % 60:02d}", True)
        else:
            self._write_display(schedule["target"].strftime("%H%M"), True)

    async def _run_schedule(self, schedule_id):
        try:
            while True:
                await asyncio.sleep(0.5)
                async with self._lock:
                    schedule = self.schedule
                    if not schedule or schedule["id"] != schedule_id:
                        return
                    await self._render_schedule_locked()
                    remaining = (
                        schedule["deadline_mono"] - time.monotonic()
                        if schedule["kind"] == "timer"
                        else (schedule["target"] - datetime.now().astimezone()).total_seconds()
                    )
                    if remaining <= 0:
                        schedule["active"] = False
                        if schedule["kind"] == "timer":
                            self._write_display("0000", True)
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            # The initial write already returned acceptance; a later display
            # transport failure is not silently reclassified as success here.
            return

    async def _expire_overlay(self):
        try:
            await asyncio.sleep(3.0)
            async with self._lock:
                await self._render_schedule_locked()
        except asyncio.CancelledError:
            raise

    async def _run(self, action, value, operation_id):
        await asyncio.sleep(0)
        if action in {"call_demo", "message_demo"}:
            if self.communications is None:
                return self._base_result(
                    result_status="unavailable",
                    summary="Phone actions are not configured.",
                    display="Phone unavailable",
                    action=action,
                )
            return await self.communications.execute(action)
        physical_actions = {
            "light_on",
            "light_off",
            "set_color",
            "set_brightness",
            "set_timer",
            "set_alarm",
            "cancel_schedule",
            "set_temperature",
        }
        if (
            action in physical_actions
            and self.backend == "pi"
            and getattr(self.output, "available", True) is not True
        ):
            return self._base_result(
                result_status="unavailable",
                summary="Raspberry Pi is not connected. No device action was sent.",
                display="Unavailable",
                action=action,
                adapter_reason=getattr(self.output, "reason", "unavailable"),
            )
        if action == "get_weather":
            if self.weather is None:
                return self._base_result(
                    result_status="unavailable",
                    summary="Weather is not configured. Enable the optional Open-Meteo provider with an explicit location.",
                    display="Unavailable",
                    action=action,
                )
            try:
                weather = await self.weather.current_at(value)
            except WeatherLocationError as exc:
                return self._base_result(
                    result_status="unavailable",
                    summary=str(exc),
                    display="Location needed",
                    action=action,
                )
            except Exception:
                return self._base_result(
                    result_status="unavailable",
                    summary="Current weather is unavailable; try again later.",
                    display="Unavailable",
                    action=action,
                )
            return self._base_result(
                result_status="completed",
                summary=f"{weather['location']}: {weather['condition']}, {weather['temperature_c']}°C. Current conditions from Open-Meteo; not a local sensor reading.",
                display=f"{weather['temperature_c']}°C",
                action=action,
                mode="reported",
                verified_in_memory=False,
                widget={
                    "type": "weather",
                    "location": weather["location"],
                    "condition": weather["condition"],
                    "temperature_c": weather["temperature_c"],
                    "observed_utc": weather["observed_utc"],
                    "provider": "Open-Meteo",
                },
            )
        async with self._lock:
            if self._closed:
                return self._base_result(
                    result_status="unavailable",
                    summary="The device service is closed.",
                    action=action,
                )
            if action in {"light_on", "light_off", "set_color", "set_brightness"}:
                previous_state = (self.light, self.color, self.brightness)
                if action == "light_on":
                    self.light = True
                elif action == "light_off":
                    self.light = False
                elif action == "set_color":
                    self.color, self.light = value, True
                else:
                    self.brightness, self.light = int(value), True
                try:
                    self._current_led()
                except Exception:
                    self.light, self.color, self.brightness = previous_state
                    return self._base_result(
                        result_status="unknown",
                        summary="The Pi did not confirm the light command. Check the LED before retrying.",
                        action=action,
                    )
                self.effects += 1
                status = "simulated" if self.mode == "mock" else "accepted"
                summary = (
                    f"Simulated lamp {self.color}, {self.brightness}% brightness, {'on' if self.light else 'off'}."
                    if self.mode == "mock"
                    else f"Pi LED command accepted: {self.color}, {self.brightness}% brightness, {'on' if self.light else 'off'}; no sensor confirms the output."
                )
                return self._base_result(
                    result_status=status,
                    summary=summary,
                    display="on" if self.light else "off",
                    action=action,
                    widget={
                        "type": "lamp",
                        "on": self.light,
                        "color": self.color,
                        "brightness": self.brightness,
                    },
                )
            if action in {"set_timer", "set_alarm"}:
                replaced = None
                now_utc = datetime.now(timezone.utc)
                if action == "set_timer":
                    seconds = {"10 seconds": 10, "30 seconds": 30, "1 minute": 60}[value]
                    deadline = now_utc + timedelta(seconds=seconds)
                    schedule = {
                        "id": operation_id,
                        "operation_id": operation_id,
                        "kind": "timer",
                        "label": value,
                        "deadline_utc": deadline.isoformat(),
                        "deadline_mono": time.monotonic() + seconds,
                        "active": True,
                    }
                    self.timers.append(value)
                    summary = f"Timer accepted for {value}; the display counts down and no buzzer or alert is configured."
                    display = value
                    widget = {
                        "type": "timer",
                        "schedule_kind": "timer",
                        "active": True,
                        "deadline_utc": deadline.isoformat(),
                        "target_local": "",
                        "display_value": value,
                    }
                else:
                    hour = {"6:00 AM": 6, "8:00 AM": 8, "9:00 PM": 21}[value]
                    now_local = datetime.now().astimezone()
                    target = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
                    if target <= now_local:
                        target += timedelta(days=1)
                    schedule = {
                        "id": operation_id,
                        "operation_id": operation_id,
                        "kind": "alarm",
                        "label": value,
                        "target": target,
                        "deadline_utc": target.astimezone(timezone.utc).isoformat(),
                        "active": True,
                    }
                    self.alarms.append(value)
                    summary = f"Alarm accepted for {value}; the display shows its time, but no buzzer or alert is configured."
                    display = value
                    widget = {
                        "type": "timer",
                        "schedule_kind": "alarm",
                        "active": True,
                        "deadline_utc": target.astimezone(timezone.utc).isoformat(),
                        "target_local": target.isoformat(),
                        "display_value": value,
                    }
                initial_display = (
                    f"{int((deadline - now_utc).total_seconds()) // 60:02d}{int((deadline - now_utc).total_seconds()) % 60:02d}"
                    if action == "set_timer"
                    else schedule["target"].strftime("%H%M")
                )
                try:
                    self._write_display(initial_display, True)
                except Exception:
                    return self._base_result(
                        result_status="unknown",
                        summary="The Pi display did not confirm the timer/alarm command. Check the display before retrying.",
                        display="Unavailable",
                        action=action,
                    )
                replaced = self._replace_schedule(schedule)
                self.effects += 1
                if replaced:
                    summary = "A new timer/alarm replaced the previous schedule. " + summary
                return self._base_result(
                    result_status="simulated" if self.mode == "mock" else "accepted",
                    summary=("Simulated schedule; " + summary if self.mode == "mock" else summary),
                    display=display,
                    action=action,
                    widget=widget,
                    replaced_task=replaced,
                )
            if action == "cancel_schedule":
                previous = self.schedule if self.schedule and self.schedule.get("active") else None
                cancelled = self._task_owner(previous["operation_id"]) if previous else None
                if self._runner and not self._runner.done():
                    self._runner.cancel()
                self.schedule = None
                self._overlay = None
                try:
                    self._write_display("    ")
                except Exception:
                    return self._base_result(
                        result_status="unknown",
                        summary="The schedule was cancelled in software, but the Pi display did not confirm it was cleared.",
                        display="Uncertain display",
                        action=action,
                        replaced_task=cancelled,
                    )
                return self._base_result(
                    result_status="simulated" if self.mode == "mock" else "completed",
                    summary="Timer/alarm cancelled."
                    if previous
                    else "There is no active timer or alarm.",
                    display="Cancelled" if previous else "No active schedule",
                    action=action,
                    widget={
                        "type": "timer",
                        "schedule_kind": "timer",
                        "active": False,
                        "deadline_utc": "",
                        "target_local": "",
                        "display_value": "Cancelled",
                    },
                    replaced_task=cancelled,
                )
            if action == "set_temperature":
                degrees = int(value.split()[0])
                self.temperature = degrees
                self._overlay = {"digits": f"{degrees:4d}", "until": time.monotonic() + 3.0}
                try:
                    self._write_display(self._overlay["digits"])
                except Exception:
                    self._overlay = None
                    return self._base_result(
                        result_status="unknown",
                        summary="The Pi display did not confirm the setpoint. Check the display before retrying.",
                        action=action,
                    )
                if self._overlay_task and not self._overlay_task.done():
                    self._overlay_task.cancel()
                self._overlay_task = asyncio.create_task(self._expire_overlay())
                self.effects += 1
                status = "simulated" if self.mode == "mock" else "accepted"
                return self._base_result(
                    result_status=status,
                    summary=(
                        f"Displayed {degrees}°C setpoint for three seconds; this is not a measured temperature or HVAC control."
                        if self.mode == "mock"
                        else f"TM1637 accepted the {degrees}°C demo setpoint; this is not a measurement or HVAC control."
                    ),
                    display=f"{degrees}°C",
                    action=action,
                    widget={"type": "temperature", "setpoint_c": degrees, "measurement": False},
                )
            if action == "create_reminder":
                if value not in self.reminders:
                    self.reminders.append(value)
                    self.effects += 1
                    try:
                        self._save_reminders()
                    except Exception:
                        self.reminders.remove(value)
                        return self._base_result(
                            result_status="unavailable",
                            summary="Reminder storage could not save this item.",
                            action=action,
                        )
                summary = (
                    f"Reminder saved: {value}."
                    if self.reminder_store
                    else f"Demo reminder saved in this session: {value}."
                )
                return self._base_result(
                    result_status="completed" if self.reminder_store else "simulated",
                    summary=summary,
                    display=value,
                    action=action,
                    mode="reported" if self.reminder_store else self.mode,
                    verified_in_memory=not bool(self.reminder_store) and self.mode == "mock",
                    widget={
                        "type": "reminders",
                        "items": list(self.reminders),
                        "persistent": bool(self.reminder_store),
                    },
                )
            if action == "list_reminders":
                items = list(self.reminders)
                return self._base_result(
                    result_status="completed" if self.reminder_store else "simulated",
                    summary=("Saved reminders: " + ", ".join(items) + ".")
                    if items
                    else "No reminders saved.",
                    display=str(len(items)),
                    action=action,
                    mode="reported" if self.reminder_store else self.mode,
                    verified_in_memory=not bool(self.reminder_store) and self.mode == "mock",
                    widget={
                        "type": "reminders",
                        "items": items,
                        "persistent": bool(self.reminder_store),
                    },
                )
            if action == "get_time":
                requested_zone = value or getattr(self.weather, "timezone_name", None)
                now = (
                    datetime.now(ZoneInfo(requested_zone))
                    if requested_zone
                    else datetime.now().astimezone()
                )
                stamp = now.strftime("%I:%M %p").lstrip("0")
                tz = clock_timezone_label(now, requested_zone)
                return self._base_result(
                    result_status="completed",
                    summary=f"System local time is {stamp} ({tz}); the Pi clock has not been verified on this host.",
                    display=f"{now:%H:%M}",
                    action=action,
                    mode="reported",
                    verified_in_memory=False,
                    widget={"type": "time", "local_time": stamp, "timezone": tz},
                )
        raise ValueError("unsupported device action")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        tasks = [task for task in (self._runner, self._overlay_task) if task and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            self.schedule = None
            self._overlay = None
            self.output.close()
        pending = [task for _, task in self.operations.values() if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
