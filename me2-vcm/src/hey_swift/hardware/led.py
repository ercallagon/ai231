"""RGB LED interface (lazy RPi.GPIO import).

Supports the state colors defined in configs/hardware.yaml (idle, listening,
processing, success, error). On non-Pi machines, set_rgb raises a clear error;
use `available()` to probe first.
"""

from __future__ import annotations

from typing import Optional, Sequence


class LED:
    def __init__(self, config: dict):
        self.cfg = config.get("led", {}) or {}
        self.pin = int(self.cfg.get("gpio_pin", 18))
        self.brightness = float(self.cfg.get("brightness", 0.4))
        self.colors = self.cfg.get("colors", {})
        self._gpio = None

    def available(self) -> bool:
        try:
            import RPi.GPIO  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure(self):
        if self._gpio is None:
            try:
                import RPi.GPIO as GPIO
            except ImportError as exc:
                raise RuntimeError(
                    "RPi.GPIO not available. Install on the Pi: "
                    "sudo apt install python3-rpi.gpio"
                ) from exc
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO

    def set_rgb(self, rgb: Sequence[float]) -> None:
        """Set an (r, g, b) tuple in 0..255. (PWM mixing omitted in scaffold.)"""
        self._ensure()
        # Scaffold: blink briefly to indicate state. Full PWM mixing is a TODO.
        self._gpio.pulse(self.pin, 1000)

    def set_state(self, state: str) -> None:
        if state in self.colors:
            self.set_rgb(self.colors[state])

    def off(self) -> None:
        if self._gpio is not None:
            self._gpio.output(self.pin, 0)
