"""TIME and WEATHER actions, independent of models, GPIO, and audio output."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from typing import Callable


from hey_swift.actions.services.weather_types import WeatherReading, WeatherProvider, WeatherUnavailable


@dataclass(frozen=True)
class ActionResult:
    intent: str
    status: str
    message: str
    data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def local_now() -> datetime:
    return datetime.now().astimezone()


class InformationService:
    """Execute validated fixed intents; rejection gating belongs to the router.

    The clock controls timezone. A weather adapter controls location, network
    timeout, freshness, and conversion of expected failures to WeatherUnavailable.
    No network call occurs unless a provider is explicitly supplied.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = local_now,
        weather_provider: WeatherProvider | None = None,
    ) -> None:
        self._clock = clock
        self._weather = weather_provider

    def execute(self, intent: str, slot_value: str | None = None) -> ActionResult:
        if intent not in {"TIME", "WEATHER"}:
            return ActionResult(intent, "unsupported", "Unsupported information intent.")
        if slot_value not in (None, ""):
            return ActionResult(intent, "invalid_slot", "This command does not accept a slot value.")
        if intent == "TIME":
            now = self._clock()
            if now.utcoffset() is None:
                raise ValueError("The clock must return a timezone-aware datetime.")
            hour = now.hour % 12 or 12
            period = "AM" if now.hour < 12 else "PM"
            return ActionResult(
                intent, "ok", f"The time is {hour}:{now.minute:02d} {period}.",
                {"timestamp": now.isoformat(), "hour": now.hour,
                 "minute": now.minute, "timezone": now.tzname()},
            )
        if self._weather is None:
            return ActionResult(intent, "unavailable", "Weather is unavailable. No weather provider is configured.")
        try:
            reading = self._weather.current_weather()
        except WeatherUnavailable:
            return ActionResult(intent, "unavailable", "Live weather is unavailable. Please check the internet.")
        return ActionResult(
            intent, "ok",
            f"In {reading.location}, it is {reading.temperature_c:g} degrees Celsius, {reading.description}.",
            {"location": reading.location, "temperature_c": reading.temperature_c,
             "description": reading.description, "observed_at": reading.observed_at.isoformat()},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intent", choices=["TIME", "WEATHER"])
    parser.add_argument("--speak", action="store_true", help="Speak the action response using Piper")
    parser.add_argument("--voice-model", default="audio/voices/en_US-amy-medium.onnx",
                        help="Path to the pretrained Piper voice (default: Amy medium)")
    args = parser.parse_args()
    provider = None
    if args.intent == "WEATHER":
        from hey_swift.actions.services.weather import OpenMeteoWeatherProvider
        provider = OpenMeteoWeatherProvider()
    result = InformationService(weather_provider=provider).execute(args.intent)
    payload = result.to_dict()
    if args.speak:
        from hey_swift.audio.speech import PiperSpeech, SpeechService, SpeechUnavailable
        try:
            SpeechService(PiperSpeech(args.voice_model)).speak(result.message)
            payload["speech"] = {"status": "ok"}
        except SpeechUnavailable as exc:
            payload["speech"] = {"status": "unavailable", "message": str(exc)}
    print(json.dumps(payload, indent=2))
    if args.speak and payload["speech"]["status"] != "ok":
        return 2
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
