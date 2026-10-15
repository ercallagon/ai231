# Hardware

Raspberry Pi 5 peripheral wiring and configuration. All interfaces are
**lazy-imported** (D-011): `import hey_swift` works off-Pi; a clear error is
raised only when a peripheral is actually driven.

## Overview

| Peripheral | Driver | Bus | Config key |
|---|---|---|---|
| RGB LED | `hardware/devices.py` / `gpiozero.RGBLED` | GPIO via lgpio | `led` |
| 4-digit display | `hardware/devices.py` | GPIO bit-bang via lgpio | `display.tm1637` |
| Character LCD | `hardware/lcd.py` | I2C (PCF8574) | `display.lcd` |
| Microphone | `audio/capture.py` | ALSA / USB | `audio_io.capture_device` |
| Speaker | `audio/playback.py` | ALSA / USB | `audio_io.playback_device` |

## LED (RGB)

- **Pins:** BCM 17 red, BCM 27 green, BCM 22 blue — `led.red_gpio`,
  `led.green_gpio`, `led.blue_gpio`.
- **States** (`led.colors`): idle (dim blue), listening (green), processing
  (amber), success (bright green), error (red).
- **Enable:** `sudo apt install python3-lgpio` and `pip install gpiozero`.
  Use a venv created with `--system-site-packages` so apt's `lgpio` is visible.

## TM1637 4-digit display

- **Pins:** CLK = BCM 23, DIO = BCM 24 — `display.tm1637`.
- **Driver:** bit-banged via the tested `PiHardware` adapter.
- **Enable:** set `display.tm1637.enabled: true`.

## Character LCD (I2C)

- **Addr:** 0x27 (PCF8574 backpack) — `display.lcd.i2c_addr`.
- **Enable I2C:** `sudo raspi-config` → Interface Options → I2C → Enable.
- **Driver:** `smbus` (`sudo apt install python3-smbus`).
- **Enable:** set `display.lcd.enabled: true`.

## Audio

- **Capture:** default mic, or set `audio_io.capture_device` to an index/name.
- **Playback:** default speaker, or set `audio_io.playback_device`.
- **Portaudio:** `sudo apt install libportaudio2` (for
  `sounddevice`).
- **Command window:** `audio_io.capture_block_seconds` (3.0 s) + `pre_roll_ms`
  (200 ms) — the audio captured after a wake.

## Wiring diagram (GPIO)

```
Raspberry Pi 5
┌─────────────────────────────┐
│  BCM 17 ──────────▶ RGB LED red
│  BCM 27 ──────────▶ RGB LED green
│  BCM 22 ──────────▶ RGB LED blue
│  BCM 23 ──────────▶ TM1637 CLK
│  BCM 24 ──────────▶ TM1637 DIO
│  I2C (SDA/SCL) ───▶ LCD (PCF8574, 0x27)
│  3.3V / GND ──────▶ peripherals
└─────────────────────────────┘
Mic (USB/ALSA) ──▶ audio_io.capture_device
Speaker (USB/ALSA) ─▶ audio_io.playback_device
```

> Exact resistor values / LED common-anode-vs-cathode depend on your specific
> module — verify against your parts before powering. This doc documents the
> *logical* mapping; the physical schematic is a TODO for the build phase.
