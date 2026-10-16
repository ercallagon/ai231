"""Keyless Open-Meteo adapter pinned to Quezon City, Philippines."""

from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from hey_swift.actions.services.weather_types import WeatherReading, WeatherUnavailable


CONDITIONS = {
    0: "clear skies", 1: "mainly clear skies", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle",
    53: "moderate drizzle", 55: "dense drizzle", 56: "light freezing drizzle",
    57: "dense freezing drizzle", 61: "slight rain", 63: "moderate rain",
    65: "heavy rain", 66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers", 95: "thunderstorms",
    96: "thunderstorms with slight hail", 99: "thunderstorms with heavy hail",
}


class OpenMeteoWeatherProvider:
    """Model-based current conditions; no key, geolocation, or GPU required.

    Reuse one instance for a five-minute in-memory cache. Expired readings are
    never served following a failed refresh. Network wait is bounded to 1 second on the action thread, with a
    30-second failure backoff and at most one pending fetch per instance.
    """

    def __init__(self, *, fetch=None, clock=None, offline=False, wait_seconds=1.0):
        self._fetch = fetch or self._fetch_json
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._cached = None
        self._cached_at = None
        self._pending = None
        self._lock = threading.Lock()
        self._retry_after = 0.0
        self.offline = offline
        self.wait_seconds = wait_seconds
        self.url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
            "latitude": float(os.environ.get("VCM_WEATHER_LATITUDE", "14.6488")),
            "longitude": float(os.environ.get("VCM_WEATHER_LONGITUDE", "121.0509")),
            "current": "temperature_2m,weather_code", "temperature_unit": "celsius",
            "timezone": "Asia/Manila", "timeformat": "unixtime", "forecast_days": 1,
        })

    @staticmethod
    def _fetch_json(url):
        with urlopen(url, timeout=5) as response:
            return json.load(response)

    def _start_fetch_locked(self):
        done = threading.Event()
        result = {}

        def fetch():
            try:
                result["payload"] = self._fetch(self.url)
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=fetch, name="weather-fetch", daemon=True)
        self._pending = (thread, done, result)
        thread.start()

    def _bounded_fetch(self):
        with self._lock:
            if self.offline:
                raise WeatherUnavailable("Offline mode: live weather requires internet")
            if self._pending is not None and not self._pending[1].is_set() and time.monotonic() < self._retry_after:
                raise WeatherUnavailable("Weather refresh still pending")
            if self._pending is None:
                if time.monotonic() < self._retry_after:
                    raise WeatherUnavailable("Weather temporarily unavailable; retry shortly")
                self._start_fetch_locked()
            _, done, result = self._pending
            if not done.wait(self.wait_seconds):
                self._retry_after = time.monotonic() + 30
                raise WeatherUnavailable("Weather request exceeded its response budget")
            self._pending = None
            if "error" in result:
                self._retry_after = time.monotonic() + 30
                raise WeatherUnavailable("Weather request failed") from result["error"]
            return result["payload"]

    def prefetch(self):
        """Warm the weather cache without delaying runtime startup or wake capture."""
        if self.offline:
            return
        with self._lock:
            now = self._clock()
            if (self._pending is None
                    and (self._cached_at is None
                         or now - self._cached_at >= timedelta(minutes=5))
                    and time.monotonic() >= self._retry_after):
                self._start_fetch_locked()

    def current_weather(self):
        now = self._clock()
        if self._cached_at is not None and timedelta(0) <= now - self._cached_at < timedelta(minutes=5):
            return self._cached
        try:
            payload = self._bounded_fetch()
            current = payload["current"]
            if payload["current_units"]["temperature_2m"] != "°C":
                raise ValueError("Unexpected temperature unit")
            temperature = current["temperature_2m"]
            code = current["weather_code"]
            timestamp = current["time"]
            if any(isinstance(value, bool) for value in (temperature, code, timestamp)):
                raise ValueError("Invalid numeric weather fields")
            description = CONDITIONS[code]
            observed = datetime.fromtimestamp(timestamp, timezone(timedelta(hours=8)))
            if not -timedelta(minutes=15) <= now - observed <= timedelta(hours=2):
                raise ValueError("Weather timestamp is not current")
            reading = WeatherReading(os.environ.get("VCM_WEATHER_CITY", "Quezon City, Philippines"), float(temperature), description, observed)
        except (URLError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            self._retry_after = time.monotonic() + 30
            raise WeatherUnavailable("Unable to obtain current Quezon City weather") from exc
        self._cached, self._cached_at = reading, now
        return reading
