"""Pi 5 adapters: KY-016 RGB and TM1637, on preserved wiring."""
import time

SEGMENTS = (0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F)

class MockHardware:
    def __init__(self):
        self.events = []
    def led(self, rgb):
        self.events.append(("led", rgb))
    def display(self, digits, colon=False):
        self.events.append(("display", digits, colon))
    def close(self):
        pass

class PiHardware:
    def __init__(self):
        from gpiozero import RGBLED
        from gpiozero.pins.lgpio import LGPIOFactory
        self.factory = LGPIOFactory()
        self.rgb = None
        self.clk = self.dio = None
        try:
            self.rgb = RGBLED(17, 27, 22, active_high=True, pin_factory=self.factory)
            self.clk = self.factory.pin(23)
            self.dio = self.factory.pin(24)
            self.clk.function = "output"
            self.clk.state = 1
            self.dio.function = "input"
            self.dio.pull = "up"
        except Exception:
            self.close()
            raise
    def led(self, rgb):
        self.rgb.value = rgb
    def _clock(self, value):
        self.clk.state = value
        time.sleep(0.0001)
    def _data(self, value):
        # Release high, drive low, including releasing DIO for the ACK bit.
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
            raise OSError("TM1637 did not acknowledge; check the existing wiring")
    def _packet(self, values):
        self._start()
        try:
            for value in values:
                self._write(value)
        finally:
            self._stop()
    def display(self, digits, colon=False):
        if len(digits) != 4 or any(c not in "0123456789 " for c in digits):
            raise ValueError("Display requires four digits/spaces")
        segments = [0 if c == " " else SEGMENTS[int(c)] for c in digits]
        if colon:
            segments[1] |= 0x80
        self._packet([0x40])
        self._packet([0xC0, *segments])
        self._packet([0x8B])  # display on, brightness 3 of 7
    def close(self):
        if self.rgb:
            self.rgb.close()
        if self.clk:
            self.clk.close()
        if self.dio:
            self.dio.close()
        self.factory.close()
