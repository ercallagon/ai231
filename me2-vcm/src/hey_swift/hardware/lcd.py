"""Character LCD (I2C) interface (lazy import).

Driven over I2C (typically PCF8574 backpack). Needs the I2C bus enabled on the
Pi. Lazy import keeps the module import-safe off-Pi.
"""

from __future__ import annotations


class LCD:
    def __init__(self, config: dict):
        disp = config.get("display", {}).get("lcd", {}) or {}
        self.enabled = bool(disp.get("enabled", False))
        self.addr = int(disp.get("i2c_addr", 0x27))
        self.cols = int(disp.get("cols", 16))
        self.rows = int(disp.get("rows", 2))
        self._bus = None

    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import smbus  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure(self):
        if self._bus is None:
            try:
                import smbus
            except ImportError as exc:
                raise RuntimeError(
                    "smbus not available. Enable I2C on the Pi: sudo raspi-config"
                ) from exc
            self._bus = smbus.SMBus(1)

    def write(self, line: int, text: str) -> None:
        if not self.available():
            return
        self._ensure()
        # Scaffold: no-op write; full HD44780-over-PCF8574 protocol is a TODO.
