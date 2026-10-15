"""Shared weather contracts; safe for imports and module CLI execution."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Protocol


@dataclass(frozen=True)
class WeatherReading:
    location: str
    temperature_c: float
    description: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.location.strip() or not self.description.strip():
            raise ValueError("Weather location and description must be nonempty.")
        if not math.isfinite(self.temperature_c):
            raise ValueError("Weather temperature must be finite.")
        if self.observed_at.utcoffset() is None:
            raise ValueError("Weather observation must have a timezone.")


class WeatherProvider(Protocol):
    def current_weather(self) -> WeatherReading:
        """Return a reading or raise WeatherUnavailable for expected failures."""
        ...


class WeatherUnavailable(Exception):
    """Expected offline, configuration, or provider availability failure."""


