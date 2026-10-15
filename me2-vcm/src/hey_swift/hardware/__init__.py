"""hey_swift.hardware — Raspberry Pi peripheral interfaces.

led       RGB LED interface
tm1637    4-digit 7-segment display
lcd       character LCD (I2C)

CRITICAL DESIGN RULE: these modules import RPi.GPIO / spidev LAZILY (inside
methods), never at module import time. That guarantees `import hey_swift` works
on a laptop with no Pi libraries, and a clear error is raised only when a
Pi-specific action is actually invoked. Tests run headless.
"""
