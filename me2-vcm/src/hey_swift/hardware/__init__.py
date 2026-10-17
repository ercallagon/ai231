"""Pi hardware adapters.

The supported runtime uses devices.PiHardware (RGB BCM17/27/22 and TM1637
BCM23/24), or devices.MockHardware for host tests. GPIO imports are lazy.

led, lcd and tm1637 are unsupported historical scaffolds retained for source
compatibility. Their methods do not implement working RGB/LCD/TM1637 drivers;
they must not be used to validate hardware or substituted for devices.py.
"""
