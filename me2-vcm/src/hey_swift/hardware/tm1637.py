"""TM1637 4-digit 7-segment display interface (lazy import).

The TM1637 is driven via bit-banged GPIO (clk + dio), so it needs RPi.GPIO.
Lazy import keeps the module import-safe off-Pi.
"""

from __future__ import annotations


class TM1637Display:
    def __init__(self, config: dict):
        disp = config.get("display", {}).get("tm1637", {}) or {}
        self.enabled = bool(disp.get("enabled", False))
        self.clk = int(disp.get("clk_gpio", 23))
        self.dio = int(disp.get("dio_gpio", 24))
        self.brightness = int(disp.get("brightness", 8))
        self._gpio = None

    def available(self) -> bool:
        if not self.enabled:
            return False
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
                    "RPi.GPIO not available for TM1637. Install on the Pi."
                ) from exc
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.clk, GPIO.OUT)
            GPIO.setup(self.dio, GPIO.OUT)
            self._gpio = GPIO

    def show(self, text: str) -> None:
        """Render up to 4 characters. Bit-bang protocol is a scaffold TODO."""
        if not self.available():
            return
        self._ensure()
        # Scaffold: no-op render; full TM1637 protocol is a TODO.
