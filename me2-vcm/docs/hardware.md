# Raspberry Pi 5 Hardware

The supported live adapter is [devices.py](../src/hey_swift/hardware/devices.py):
`PiHardware` uses gpiozero with the OS-provided lgpio backend; `MockHardware`
records actions for host tests. GPIO imports occur only when the Pi adapter is
constructed. The public entry point is `./swift run --model champion --live`.

## Tested Wiring

The RGB LED and TM1637 remain connected simultaneously.

| Peripheral | Connection | BCM | Physical pin |
| --- | --- | ---: | ---: |
| KY-016 RGB | Red | 17 | 11 |
| KY-016 RGB | Green | 27 | 13 |
| KY-016 RGB | Blue | 22 | 15 |
| KY-016 RGB | Ground | - | 9 |
| TM1637 | CLK | 23 | 16 |
| TM1637 | DIO | 24 | 18 |
| TM1637 | VCC (3.3 V) | - | 17 |
| TM1637 | Ground | - | 6 |

The adapter uses active-high RGB and fixed tested pins. The matching entries in
`configs/hardware.yaml` document that contract; editing the YAML does not
rewire `PiHardware`. The TM1637 adapter uses brightness level 3 of 7, independent
of the legacy YAML brightness value. No I2C LCD is used.

Install `python3-lgpio` through the OS package manager and use the documented
`--system-site-packages` virtual environment so that its ABI matches OS Python.
`./swift install` handles the Python dependencies; [the quickstart](QUICKSTART_PI.md)
lists the required system packages.

## Checks and Display Ownership

With other GPIO users stopped, run:

```bash
./swift gpio
./swift doctor
```

The check cycles red/green/blue and displays `1234`. The operator's visual
confirmation is recorded separately from import checks and voice recognition.
The operator confirmed these physical effects on the tested Pi. Timer/alarm
updates run independently of the wake-listening loop; a new display action
supersedes earlier ownership. [Commands](command-cheatsheet.md) lists supported
timer and alarm values.

## Audio and Capture

The tested mono USB microphone supports 48 kHz capture. Setup stores its device
name and host API; runtime resolves the index and resamples continuously to
16 kHz. Output uses the selected/default PipeWire sink, with independent
speech/cue/music controls described in [AUDIO.md](AUDIO.md).

The current capture path is `integration/e2e_loop.py`: sustained onset,
200 ms pre-roll, dynamic trailing-silence endpointing (800 ms by default),
a three-second speech-start timeout, and explicit retained-input capacity
checks. Champion capacity is six seconds; the safety deadline is 7.4 seconds.
These are not the legacy `audio_io.capture_block_seconds: 3.0` fixed-window
settings in `hardware.yaml`. Capture policy comes from the profile launcher
and CLI; the YAML's historical audio fields are not the current device selector.

## Legacy Interfaces

`hardware/led.py`, `hardware/tm1637.py` and `hardware/lcd.py` are unsupported
historical scaffolds retained for import compatibility. The LED scaffold lacks
RGB mixing and uses an obsolete pin; the display scaffolds contain no-op writes.
Their `available()` methods test dependency imports, not functional hardware.
The current runtime, CLI and tests use `hardware/devices.py`; no character LCD
driver or schematic beyond the tested pin table is provided.
