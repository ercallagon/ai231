# Raspberry Pi E2E Runtime

> Historical research guide. Current frozen defaults, installation, rollback and
> reproduction commands are in README.md, docs/PI5.md and docs/BENCHMARK.md.
> Old artifact tags, package paths and readiness snapshots below are not current
> release instructions. Physical Pi 5 verification of the six-second profile is pending.

Current field-feedback fixes and selective installation commands:
[Pi Runtime v6](rpi-runtime-v6.md). [Command cheatsheet](command-cheatsheet.md).
The v5 notes below remain historical; v6 adds explicit shared PipeWire capture,
bounded recovery, call guard, cue-only gain and confirmed Spotify activation.

For the current run_043 runtime-only safety/offline update, follow
[Pi Runtime v5](rpi-runtime-v5.md). That guide includes installation, the complete
offline run command, reproducible fixed-response generation, latency semantics,
and the distinction between sandbox tests and pending physical Pi checks.

Canonical runtime code now lives under `src/hey_swift/`, not `work/`.

Promoted action/runtime modules:

- `hey_swift.actions.services.*` — full action services from the standalone bundle
- `hey_swift.audio.speech` — Piper speech wrapper
- `hey_swift.hardware.devices` — mock and Pi hardware adapters
- `hey_swift.integration.router` — validated prediction router
- `hey_swift.integration.runtime` — action-service construction
- `hey_swift.integration.e2e_loop` — wake -> VCM -> action -> idle loop

Runtime assets and writable state stay outside `src/`. The runtime root is:

1. `HEY_SWIFT_RUNTIME_ROOT`, if set;
2. `<project>/integration`, if present;
3. `<project>/work/integration`, in this sandbox;
4. `<project>/integration`, created on first use.

The runtime root contains assets such as voices/music and writable folders
`state/`, `logs/`, and `tmp/`.

## Export Champion

After selecting a slot-enabled champion:

```sh
PYTHONPATH=src python -B scripts/03_export_onnx.py \
  --tag run_018_e2_a2_wide_slots \
  --config configs/vcm_e2_a2_wide_slots.yaml
```

## Package Transfer Bundle

Use the preflighted packager; it stages only runtime files and refuses to create
the tar if required assets or model/config contracts are missing:

```sh
PYTHONPATH=src python -B scripts/package_rpi_transfer.py \
  --tag run_021_arch_a6_resnet2d_slots \
  --config configs/vcm_arch_screen_a6_resnet2d_slots.yaml
```

The generated tar includes root-level wakeword, VCM, chime, Amy Piper voice,
offline music assets, configs, and runtime source. It excludes `__pycache__`,
`.pyc`, tarballs, and the old wakeword backup tree. It includes
`scripts/test_local_pi_updates.py`, `LOCAL_PI_UPDATES.md`,
`RUN_021_RPI_README.md`, and `PATCH_SHA256SUMS`.

Before packaging, run the local regression checks:

```sh
PYTHONPATH=src python -B scripts/test_local_pi_updates.py
```

These are offline regression tests for routing, synthetic command capture,
music duck/restore, speech caching, formatting, lifecycle cleanup, and bounded
weather fetch behavior. They do not verify physical Pi ALSA/PipeWire mixing,
GPIO effects, saved Spotify login, live microphone recognition, or actual Pi
latency.

## Install On Pi

Install system packages first:

```sh
sudo apt update
sudo apt install -y python3-venv python3-pip python3-lgpio libportaudio2 libsndfile1 \
  alsa-utils pipewire-audio-client-libraries mpv
```

Use `--system-site-packages` so the virtualenv can import apt's `lgpio` module:

```sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-pi.txt
```

Run the Pi-side preflight before live testing:

```sh
sha256sum -c PATCH_SHA256SUMS
PYTHONPATH=src python -B -c 'from hey_swift.integration.e2e_loop import PATCH_VERSION; print(PATCH_VERSION)'
python -B scripts/test_local_pi_updates.py
PYTHONPATH=src python -B scripts/preflight_rpi_bundle.py \
  --root . \
  --vcm-onnx models/vcm/run_021_arch_a6_resnet2d_slots.onnx \
  --vcm-labels models/vcm/run_021_arch_a6_resnet2d_slots.labels.json \
  --vcm-config configs/vcm_arch_screen_a6_resnet2d_slots.yaml \
  --require-pi-imports
```

## Tested GPIO Wiring

`configs/hardware.yaml` is aligned with the existing tested Pi adapter:

- RGB LED: BCM 17 red, BCM 27 green, BCM 22 blue
- TM1637 display: CLK BCM 23, DIO BCM 24

Preserve those adapter pins unless the hardware adapter is changed too.

## Dry E2E Test

```sh
PYTHONPATH=src python -B scripts/run_e2e_rpi.py \
  --vcm-onnx models/vcm/run_018_e2_a2_wide_slots.onnx \
  --vcm-labels models/vcm/run_018_e2_a2_wide_slots.labels.json \
  --vcm-config configs/vcm_e2_a2_wide_slots.yaml \
  --device 0 \
  --hardware mock \
  --once
```

## Real Pi Test

```sh
PYTHONPATH=src python -B scripts/run_e2e_rpi.py \
  --vcm-onnx models/vcm/run_021_arch_a6_resnet2d_slots.onnx \
  --vcm-labels models/vcm/run_021_arch_a6_resnet2d_slots.labels.json \
  --vcm-config configs/vcm_arch_screen_a6_resnet2d_slots.yaml \
  --hardware pi --device 0 --live --speak --verbose \
  --log-format text --wake-log-interval 5 \
  --command-seconds 5 --speech-start-timeout 3 \
  --speech-start-ms 120 --min-command-seconds 0.5 \
  --end-silence-ms 550 --music-duck-percent 0 \
  --music-listen-volume-percent 25
```

The loop dispatches only accepted model predictions. OOS, low-confidence,
unknown intents, and invalid slots execute nothing.

## Class Live Benchmark

Use the selected slot-correct model export and enable the benchmark decision
record. It prints one flushed JSON line per command after the action result is
available and before spoken feedback. The record includes `intent`, `slot`,
`infer_ms`, and `audio_ms`, which the class benchmark uses for scoring and
latency. JSON runtime events also include wake detections.

Save stdout in the benchmark's expected directory on the Pi:

```sh
mkdir -p "$HOME/vcm_benchmark"
PYTHONPATH=src python -B scripts/run_e2e_rpi.py \
  --vcm-onnx models/vcm/<selected-model>.onnx \
  --vcm-labels models/vcm/<selected-model>.labels.json \
  --vcm-config configs/<selected-model>.yaml \
  --hardware pi --device 0 --live --speak \
  --log-format json --benchmark-log \
  >> "$HOME/vcm_benchmark/$(date +%Y%m%d-%H%M%S).log" 2>&1
```

The benchmark's command gap must allow the runtime to detect wake, play its
chime, and become ready for command audio. Check this in its sound check before
starting the full run; the benchmark's short default gap may be too short for
this wake/chime handoff.

The startup banner must show `pi-flow-v4-low-latency-20261007`. Old JSON-default output with
no acknowledgement status indicates the wrong process or file is running.

Current Pi evidence before this merge:

- Wake detection, light on/off, and speech were previously Pi-verified.
- Weather diagnostic succeeded on Pi.
- COLOR=blue was recognized but needed slot translation; that is covered by
  local regression and still needs Pi retest.
- Physical ALSA/PipeWire chime mixing, GPIO side effects for this patch,
  saved Spotify login, live microphone false accepts, and actual latency remain
  Pi-verification items.

## Command Capture Timing

The loop does not wait for the full `--command-seconds` unless it has to. The
runtime closes the wake stream before command capture, reuses the startup-probed
mic rate, mutes the owned music player during command capture, calibrates
ambient audio before the chime, and prints the prompt only when the command
stream is ready. While music is playing, the runtime caps its volume at 25% by
default to reduce speaker bleed into the microphone. This is a demo workaround,
not acoustic echo cancellation: place the speaker away from and off-axis to the
mic, and lower the cap if wake recall degrades. After the wake chime it uses
adaptive RMS end-of-speech detection:

1. estimate ambient noise briefly;
2. wait up to `--speech-start-timeout` for speech to start;
3. include a short pre-roll;
4. stop when silence lasts long enough;
5. run VCM immediately.

Recommended live defaults:

```text
--command-seconds 5.0          # maximum capture length after speech starts
--min-command-seconds 0.5      # avoid cutting tiny bursts too early
--speech-start-timeout 3.0     # no speech after wake -> reject/OOS path
--speech-start-ms 120          # require sustained above-threshold sound
--end-silence-ms 550           # main latency/robustness tradeoff
--pre-roll-ms 200
--vad-frame-ms 40
--vad-noise-seconds 0.16       # ambient calibration before cue
--vad-margin-db 10
--vad-min-speech-db -45
--music-listen-volume-percent 25  # cap playback so wake detection can hear
```

Finite common action phrases are synthesized once before listening and reused
from a bounded temporary WAV cache. Variable time/weather text is synthesized
on demand, then cached if it repeats. TTS playback stays synchronous so it
cannot overlap the next microphone capture.

Each command event prints:

```json
{
  "capture": {
    "reason": "end_silence",
    "speech_end_offset": 1.24
  },
  "latency": {
    "speech_end_to_action_seconds": 0.73,
    "vcm_seconds": 0.08,
    "action_seconds": 0.01
  }
}
```

The command-duration clock starts once sustained speech is detected, so waiting
briefly after the beep does not reduce the available command time. For lower
latency in quiet rooms, reduce `--end-silence-ms` to `450-500`. For noisy
rooms or clipped commands, raise it to `750-900` or increase `--vad-margin-db`.

## run_043 calibrated bundle (current deployment-testing reference)

The run_043 calibrated BC8 export is staged as a separate lean bundle (e.g.
`work/rpi_transfer_run_043_hf_grouped_v2b_bc8_sourceft_s42_calibrated_lean_<stamp>.tar.gz`),
distinct from the run_021 packager defaults. It ships the **calibrated** ONNX
(temperature scaling baked into the graph — do not re-apply at runtime), the
matching `configs/vcm_run_043_calibrated.yaml` (calibrated gate
`oos.confidence_threshold = 0.875`, string slot values), and runtime-format
labels. See the bundle's `RUN_043_RPI_README.md` for install/run commands and
`PI_CHECKLIST_RUN043.md` for the field checklist.

### Slot translation (TEMPERATURE / BRIGHTNESS)
`MODEL_ACTION_SLOTS` in `src/hey_swift/integration/e2e_loop.py` accepts **both**
the canonical string values emitted by string-labeled exports (run_043: `"18"`,
`"20"`, ...) and the legacy integer values from run_021 (`18`, `20`, ...). Each
key is an exact literal match — no numeric conversion — so near-miss forms
(`"18 "`, `"18.0"`, `18.0`, booleans) stay invalid and fail closed. run_021
compatibility is preserved by keeping the integer keys. All 18 supported slot
values route in both representations; unsupported values and rejected/OOS
predictions are blocked from executing any action. Covered by
`tests/test_rpi_slot_translation.py` (real `_make_prediction` → `ActionRouter`
path) plus the existing `scripts/test_local_pi_updates.py`.

Sandbox-verified: bundle structure, preflight contract, CPU ONNX load/shapes,
gate fail-closed, router rejection safety, and the slot-translation regression
suite. **Pi-only (pending):** live microphone capture, speech/music
interference, end-to-end command latency, ALSA/PipeWire mixing, GPIO side
effects, and physical TEMPERATURE/BRIGHTNESS actuation. Host timings are not Pi
latency.
# Sandbox Safety Update (2026-10-12)

The sandbox runtime banner is now `pi-flow-v6-quieting-guard-20261012`.
If known Spotify/offline music cannot be quieted, the runtime emits
`command_aborted` and resumes listening without command capture, model inference,
or action. Failed speech quieting suppresses TTS and releases audio focus.
This retains existing volume/restore behavior: it does not add echo cancellation
or change shared Bluetooth/tablet/sink gain. Music before the wake can still
mask the wakeword. Repair the unavailable player/controller before retrying.

Shared PipeWire capture now uses the tested `--channel-map MONO --format f32 --raw`
contract, with the explicit configured source and no silent microphone fallback.
Models, calibrated thresholds, hardware pins, chime assets and accounts are
unchanged. This source update was not transferred to the Pi; compare installed
source hashes and run physical audio verification before a selective rollout.
Backups and regression receipts: `work/experiments/night_20261012_172034/`.

Call-end follow-up: terminal disconnected Bluetooth objects no longer keep the
guard paused simply because the object still exists. Unknown or active states
stay blocking. The live loop uses a3ssettling tail after verified idle reads
(`--call-end-tail-seconds 3`), then reopens wake listening; it does not start a
command without a fresh wake. This is not a2-3beep detector and never sends an
automatic hangup. The reported tablet-active/post-remote-hangup beep condition
still needs actual state/stream diagnostics before any safe automatic cleanup.
After the source helper is selectively installed on Pi:
`.venv/bin/python -B scripts/call_state_diagnostics.py --seconds 60` prints only
gateway/call-object states, never recipient numbers. Do not assume that helper
already exists in the current Pi bundle. No call/SMS was executed by HPC tests.

