# run_043 Pi Runtime v5

Field-feedback updates are documented in [Pi Runtime v6](rpi-runtime-v6.md).
Apply the selective update to an existing v5 installation; do not re-extract the
full v5 tar over a locally adjusted chime or account state.

This is a runtime-only update. The calibrated run_043 model, temperature scaling
inside ONNX, confidence threshold 0.875, wake exp15 model, and tested wiring are
unchanged. Prior bundles and experiment artifacts are preserved.

## Installation

Extract the tar and enter its top-level directory. Install dependencies before
disconnecting the demo Pi from the internet:

```sh
sudo apt update
sudo apt install -y python3-venv python3-pip python3-lgpio libportaudio2 libsndfile1 alsa-utils pipewire-audio-client-libraries mpv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-pi.txt
sha256sum -c SHA256SUMS
python -B scripts/test_prepi_runtime_v5.py
python -B scripts/test_local_pi_updates.py
python -B tests/test_rpi_slot_translation.py
PYTHONPATH=src python -B scripts/preflight_rpi_bundle.py --root . --vcm-onnx models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.onnx --vcm-labels models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.labels.json --vcm-config configs/vcm_run_043_calibrated.yaml --require-pi-imports
```

## Offline Demo

```sh
PYTHONPATH=src python -B scripts/run_e2e_rpi.py --vcm-onnx models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.onnx --vcm-labels models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.labels.json --vcm-config configs/vcm_run_043_calibrated.yaml --live --hardware pi --speak --offline --benchmark-log --log-format json
```

`--offline` bypasses new Spotify playback and weather HTTP entirely. A previously
selected Spotify player may still receive local pause/control requests to prevent
overlapping playback; inability to confirm stopping blocks switching safely.
Wake, VCM, Piper, bundled
MP3s, local time, LED/display controls, timer/alarm scheduling, and reminder
storage do not need internet. System time must already be correct for time/alarm.
Weather reports unavailability rather than inventing a result. Calls/messages
still require the configured paired phone, Bluetooth permissions, and its
service; offline mode never simulates their success. GPIO effects are real only
for attached hardware; temperature is the existing display demo, not HVAC.

Without `--offline`, weather waits at most one second on the action thread,
uses fresh cache when available, and backs off failed/pending requests. Spotify
attempts are bounded before safe local-music fallback. If an online player may
already be playing and stopping cannot be confirmed, fallback remains fail-closed
to prevent two players. Internet loss must not cause false weather/call success.

## Interaction and Latency

Say Hey Swift, wait for the complete chime, then speak. Speech may start
immediately after the chime or within the existing three-second start allowance.
Energy endpointing remains the default: 120 ms sustained onset, 550 ms ending
silence, five-second capture ceiling. Thresholds and these timings are unchanged.
The classifier was trained on two-second inputs: detected speech that cannot fit
is rejected with a short retry response instead of silently dropping words.
Capture overflow and incomplete recordings also fail closed.

Main responses and queued timer/alarm announcements share half-duplex audio
ownership. Scheduled announcements wait until the command/action cycle completes;
timer threads never wait for microphone ownership. Wake detection ignores TTS
playback and its 160 ms tail. This is not echo cancellation or full-duplex barge-in.
Music stays limited during listening and ducks during command capture/TTS;
physical speaker/microphone placement still matters.

Fixed responses are included as voice-fingerprinted WAV files. They avoid live
synthesis; only changing content is synthesized at runtime. Response JSON includes
cache, preparation, playback, and audio-duration fields. Action-result/benchmark
records are emitted before response playback. Use `speech_end_to_service_seconds`
for action completion; `speech_end_to_action_seconds` includes speech playback.
The speech-end value is an estimated last-voiced-frame read time, not a ground-truth
acoustic annotation. Do not call host timings Pi latency.

Optional speech-aware endpointing is experimental, not required for the demo:

```sh
python -m pip install --only-binary=:all: webrtcvad-wheels==2.0.14.post1
```

Add `--endpoint-mode webrtc` only after comparing quiet/soft speech and noise on
the Pi. Missing optional dependencies fail explicitly. Native silence/framing
is not evidence of improved live recognition. API reference:
https://github.com/daanzu/py-webrtcvad-wheels

Calling uses four-second D-Bus subprocess limits per readiness/dial request.
Messaging has a fifteen-second total operation budget, five-second per-request
limit, and at most two seconds for session cleanup. A timeout after dialing or
attempting message submission reports an unknown outcome and never retries the
effect. Verify the phone before repeating it. Real Bluetooth timing remains a
Pi-side check; these limits do not promise fast cellular delivery.

## Pi Verification Still Required

Sandbox tests use synthetic capture, mock GPIO, injected playback/music, actual
service construction, and CPU ONNX loading. They do not prove live recognition,
ALSA/PipeWire mixing, speaker interference, Bluetooth delivery, or Pi latency.

On Pi check immediate and delayed speech after chime, soft speech, within-command
pauses, each numeric slot, OOS rejection, commands while music plays, and a timer
announcement arriving during capture. Repeat with Wi-Fi disconnected. Confirm
real LEDs RGB BCM17/27/22 and TM1637 BCM23/24. Collect raw JSON and failed phrases;
do not change model thresholds based only on this synthetic regression suite.

## Reproduce Packaging

From the sandbox repository, use a fresh output path for each build:

```sh
PYTHONPATH=src python -B scripts/precompute_runtime_speech.py --output work/runtime_v5_audit/speech_NEW
PYTHONPATH=src python -B scripts/stage_rpi_transfer_run043.py --speech-assets work/runtime_v5_audit/speech_NEW --name rpi_transfer_run043_runtime_v5_NEW
```

Packaging checks phrase/model contracts and runs bundled regression tests before
writing checksums and creating the tar. Existing destinations are refused.
