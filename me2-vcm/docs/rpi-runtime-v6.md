# run_043 Selective Pi Runtime v6

> Historical research guide. Current frozen defaults, installation, rollback and
> reproduction commands are in README.md, docs/PI5.md and docs/BENCHMARK.md.
> Old artifact tags, package paths and readiness snapshots below are not current
> release instructions. Physical Pi 5 verification of the six-second profile is pending.

## Scope and Evidence

Authoritative sandbox: `PROJECT_ROOT`.
Base: `rpi_transfer_run043_runtime_v5_20261010_04.tar.gz`, runtime
`pi-flow-v5-safe-capture-offline-20261010`. All 62 bundled source files matched
the sandbox before edits. New marker: `pi-flow-v6-selective-recovery-20261010`.
This is code-only: no model, confidence threshold, calibration, GPIO, account,
action-default, virtualenv or music-asset replacement. No training or GPU job.

Confirmed: the Pi's old 26/8/7 suites passed and CALL dialing succeeded. The crash
was on later microphone reopen, not a failed call. During and after hangup, the
user's PipeWire graph showed Bluetooth linked to the same USB input/output, and
ALSA capture was occupied. USB card and source/sink identities did not disappear
in these snapshots. Raw ALSA contention is strongly suggested, not proven until
the venv device list and device-owner results are available.

### Temperature Duration

The model input remains `(batch,1,80,198)`, two seconds at 16 kHz. Total capture
time includes delayed onset and endpoint waiting; it is not the retained model
audio duration. v6 logs retained audio, detected speech span, model window and
actual inference input separately. It never discards detected words to fit.

Training-manifest evidence (train only, hf_grouped_v2b):

| Phrase Family | Rows for 18 / 22 / 26 | Median Whole-Clip Seconds |
| --- | --- | --- |
| Temperature N degrees | 259 / 262 / 252 | 2.32 / 2.36 / 2.44 |
| Set the temperature to N degrees | 249 / 241 / 292 | 2.86 / 2.88 / 3.02 |
| Change the temperature to N degrees | 280 / 265 / 263 | 3.02 / 3.12 / 3.20 |

Whole-clip length includes silence. Training center-crops mel features to two
seconds, unlike v5's safe live rejection for detected speech exceeding the
window. Therefore offline accuracy does not prove full natural-phrase coverage.
Use the covered shorter phrases "Temperature eighteen degrees", "Temperature
twenty-two degrees", "Temperature twenty-six degrees" at a clear natural pace.
These are suggestions, not guaranteed wins. If they still exceed the window or
fail, collect the complete rejected recording; do not race the phrase, coerce
numbers, lower the threshold, enable implicit truncation, or infer temperature
from a partial command. Longer-input modeling needs separate validation and is
outside this runtime-only patch.

### Timers and Display

10s/30s/1m translation and threaded scheduling are correct in source. Host tests
now verify `0010 -> 0009 ...`, `0030 -> 0029 ...`, `0100 -> 0059 ... -> 0000`
while the wake loop is otherwise idle. TM1637 writes occur only when digits
change. Background completion/failure events are printed while listening, so
GPIO ACK errors no longer stay hidden in a service event list. TIMER/ALARM/
TEMPERATURE still override previous display schedules; unsupported slots do not.
The Pi's physical one-minute failure remains unproven until isolated hardware
tests show either changing digits or the actual ACK/error event.

Wiring is unchanged, with both modules connected:
TM1637 VCC physical17, GND6, CLK BCM23/physical16, DIO BCM24/physical18;
KY-016 GND physical9, R BCM17/physical11, G BCM27/physical13, B BCM22/physical15.

### Microphone and Calls

Default sounddevice mode pins selected input name plus host API, re-resolves
indexes on each open, never falls back to another microphone, and retries
unavailability up to five times with 1/2/4/4/4-second sleeps. A command interrupted
by capture failure is abandoned; effects or calls are never replayed. Recovery
exhaustion stops cleanly with device diagnostics. PortAudio's cached device list
may require process restart after genuine USB removal/re-enumeration.

For this Pi's persistent Bluetooth/USB links, the explicit shared PipeWire option
below avoids competing for raw ALSA. It requires `pw-record` and `pw-dump`, normally
provided by the existing PipeWire installation. It verifies the exact USB source
node, refuses fallback/reconnection to another source, keeps draining/discarding
cue audio, and reports queued-buffer overflow as unsafe input. It is an optional
backend, host-tested only until the Pi test passes. It neither restarts PipeWire
nor changes a Bluetooth profile or the global default microphone.

After an owned CALL returns dialing/unknown, continuous mode closes assistant
capture, suppresses wake and scheduled speech, polls only GetCalls, requires two
idle reads after at least three seconds, waits another 1.5 seconds for hangup
audio, then opens fresh input. If state is unavailable or the five-minute guard
expires, the runtime stops without hanging up or pretending the call failed.
This guard covers assistant-initiated calls, not independently initiated tablet
calls; pause the assistant for external calls. `--once` ends the assistant after
one command, so it does not run a continuous call guard.

### Cue, Music and Call Volume

The locally amplified +3 dB VOICY chime and `wake_chime_original.wav` are excluded
from the patch. They are not overwritten. `--cue-gain-db 3` applies additional
cue-only gain to whichever WAV is already installed, limited to peak 0.98 without
clipping; logs include requested/applied gain, peak/RMS and actual asset hash.
Default gain is zero, preserving the local change. `--cue-sink NODE_NAME` targets
only the cue through pw-play; otherwise it follows the existing default sink.
No speech/music/call/global volume is changed. Peak limiting cannot create
unlimited loudness; compare cue RMS and playback routing rather than increasing
global gain blindly. [PipeWire pw-cat reference](https://docs.pipewire.org/page_man_pw-cat_1.html).

Music's listening cap remains 25%. The response now explains the actual limit
and its purpose. Owned music is muted during capture/TTS and restored before
volume commands so effects are not overwritten. Loud playback can still mask
Hey Swift before ducking begins. This is half-duplex/volume control, NOT acoustic
echo cancellation. Do not raise the cap to hide a wake problem.

Tablet call-volume, Bluetooth profile/hardware volume, call-stream software gain,
and USB speaker sink gain are separate controls that can interact. The posted
USB sink stays at 1.00; the live Bluetooth capture/playback loopbacks are confirmed.
The exact volume synchronization mechanism is not proven by wpctl topology alone.
Collect read-only node and stream volume evidence before changing policy. A
verified call-stream-only attenuation/limiter would be preferable to changing the
shared USB sink, since the latter also changes chime/TTS/music. No automatic
global gain, forced HFP/A2DP transition, or unverified volume restoration is added.
Keep tablet call volume conservative for the demo; loud hangup audio remains a
Pi-side issue requiring per-stream routing/gain verification.
[WirePlumber Bluetooth controls](https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/bluetooth.html).

### Spotify

Saved-login discovery still prefers `~/vcm-modular-test/work/tools/soloist/soloist`
and `~/vcm-modular-test/work/soloist-runtime`; no credential copying or HPC paths.
The integration allows a 15-second total operation budget, five-second owned
startup limit, two-second controls and up to twelve state observations. It
activates once when inactive, waits for confirmed active state before play, then
waits for actual playing state. Failures identify the phase without exposing keys.
No manual activation should be needed on the normal tested path, but Pi retest
must confirm the saved-session timing. The old short budget is a hypothesis,
not a proven sole cause.

ctl success acknowledges dispatch, not completed playback. Unknown play-command
acknowledgement never falls back, because playback could arrive late. Other
post-activation/play failures require an acknowledged pause plus two observed
paused/stopped states before local fallback. A 3-second cleanup budget is reserved
outside the normal operation budget. Uncertain state fails closed rather than
starting overlapping MP3s. [Official Soloist control reference](https://developer.spotify.com/documentation/soloist/reference/soloist-ctl).
The same observed-pause rule applies when switching an already active Soloist
session to offline music after connectivity is lost.

### Unrelated Buzzes, Rings and Beeps

The wake model, threshold and sustain are unchanged. Known assistant TTS/cue and
owned-call tones are suppressed. No unvalidated narrow-band beep veto is enabled:
it could reject real wake phrases. Opt-in `--wake-audio-log-dir` saves up to two
seconds preceding each wake; label false events for targeted analysis. Optional
WebRTC command endpointing is not a wakeword detector fix. Never describe these
safeguards as eliminating third-party beep false wakes.

## Transfer: HPC -> Windows -> Existing Pi Installation

The project/export command on HPC is:

```sh
cd PROJECT_ROOT
PYTHONPATH=src python -B scripts/package_rpi_patch_v6.py --name swift-runtime-v6-selective_20261010_02
```

The final unzipped patch is under `deploy/rpi/swift-runtime-v6-selective_20261010_02/`.
Intermediate staging and transfer archives remain under `work/`.
Download the supplied `work/swift-runtime-v6-selective_20261010_02.tar.gz`
link directly to `C:\Users\EJ\Documents\ChatGPT\ME2-VCM 2`. If using SCP instead
of the download link, substitute your actual reachable HPC SSH login hostname
(the sandbox compute-node/private address is not assumed reachable):

```powershell
$Project = 'C:\Users\EJ\Documents\ChatGPT\ME2-VCM 2'
$Patch = 'swift-runtime-v6-selective_20261010_02.tar.gz'
$HpcHost = 'YOUR_HPC_SSH_LOGIN_HOST'
scp "HPC_USER@${HpcHost}:PROJECT_ROOT/work/$Patch" "$Project\$Patch"
Get-FileHash "$Project\$Patch" -Algorithm SHA256
ssh ej@192.168.100.103 'mkdir -p /home/ej/swift-e2e/work'
scp "$Project\$Patch" 'ej@192.168.100.103:/home/ej/swift-e2e/work/'
ssh ej@192.168.100.103
```

In the Pi terminal, stop the E2E process with Ctrl-C in its running terminal
before applying. Do not stop Bluetooth or Soloist, uninstall dependencies or
create another installation. Then:

```sh
cd /home/ej/swift-e2e
mkdir -p deploy/rpi/updates
tar -xzf work/swift-runtime-v6-selective_20261010_02.tar.gz -C deploy/rpi/updates
cd deploy/rpi/updates/swift-runtime-v6-selective_20261010_02
sha256sum -c SHA256SUMS
/home/ej/swift-e2e/.venv/bin/python -B apply.py --root /home/ej/swift-e2e --check
/home/ej/swift-e2e/.venv/bin/python -B apply.py --root /home/ej/swift-e2e --apply
cd /home/ej/swift-e2e
sha256sum -c V6_PATCH_SHA256SUMS
```

The updater refuses source conflicts, checks immutable model/config hashes, makes
fresh backups under `deploy/rpi/backups/`, applies only the listed payload, and
runs the self-contained tests plus CPU model preflight. It records chime hashes
without requiring them to match HPC. Original SHA256SUMS is left as historical
v5 provenance; it is not the checksum list for patched code.

## Pi Commands and Hardware Checks

```sh
cd /home/ej/swift-e2e
source .venv/bin/activate
python -B scripts/diagnose_pi_runtime.py > integration/logs/pi_audio_idle.json
python -m sounddevice
```

Read-only diagnostics do not need pactl and never dial, send, play, activate, stop
services or change profiles. Repeat with new filenames during/after a call only
if you choose to perform that real call. Do not post private audio/recipients/keys.

Recommended shared-capture continuous run for the USB source in the supplied logs:

```sh
PYTHONPATH=src python -B scripts/run_e2e_rpi.py \
  --vcm-onnx models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.onnx \
  --vcm-labels models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.labels.json \
  --vcm-config configs/vcm_run_043_calibrated.yaml \
  --pipewire-source alsa_input.usb-Jieli_Technology_UACDemoV1.0_1120030506060402-00.mono-fallback \
  --live --hardware pi --speak --log-format json --benchmark-log
```

Do not combine `--device` with `--pipewire-source`. To retain direct sounddevice
mode, replace the latter with `--device 0` after verifying the venv device list.
For cue comparison add `--cue-gain-db 3`, optionally with the confirmed speaker
`--cue-sink alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035473535330D-00.analog-stereo`.
For diagnosis only, add `--wake-audio-log-dir /home/ej/swift-e2e/integration/logs/wake_samples`
and `--rejected-audio-log-dir /home/ej/swift-e2e/integration/logs/rejected_samples`.
These save personal audio; logging is disabled by default. Add `--offline` to
skip new Spotify playback/weather HTTP while retaining local demo functions.

With the E2E process stopped, isolate TM1637 without relying on VCM:

```sh
PYTHONPATH=src python -B -m hey_swift.actions.services.device_actions TIMER --value '10 seconds' --hardware pi
PYTHONPATH=src python -B -m hey_swift.actions.services.device_actions TIMER --value '30 seconds' --hardware pi
PYTHONPATH=src python -B -m hey_swift.actions.services.device_actions TIMER --value '1 minute' --hardware pi
```

Then test timers in continuous listening and override a running timer with a
temperature or another timer command. Log completion/ACK failures. Test immediate
and delayed command onset with the PipeWire backend, quiet speech, internal pauses,
OOS, all supported slots, Spotify cold inactive startup, music -> wake -> pause,
and repeated call/hangup -> wake cycles only when you authorize real calls.
Keep ringing volume low while characterizing the Bluetooth stream.

## Rollback

The updater prints the exact fresh backup path. Stop E2E first and substitute it:

```sh
cd /home/ej/swift-e2e/deploy/rpi/updates/swift-runtime-v6-selective_20261010_02
/home/ej/swift-e2e/.venv/bin/python -B apply.py --root /home/ej/swift-e2e --rollback /home/ej/swift-e2e/deploy/rpi/backups/PRINTED_BACKUP_NAME
```

Rollback verifies backups and refuses to overwrite subsequent local edits. It
restores original overwritten files; newly added files stay unused rather than
being deleted. Models, config, credentials, venv and local chime are never touched.
Reapplying after rollback requires reconciliation of the retained new files and
update receipt; the updater refuses to overwrite them automatically.
Host tests and CPU preflight are NOT successful Pi deployment or acoustic tests.
# Upstream Shared-Capture Alignment (2026-10-12)

The sandbox shared PipeWire backend now uses the Pi-verified command contract:
`pw-record --target <explicit-source> --channels 1 --channel-map MONO --format f32 --raw`.
The former `--channelmap mono` option was invalid for the tested PipeWire version;
explicit raw output prevents WAV/container headers entering float32 capture.
This does not enable AEC, change capture defaults, switch microphones or modify
models, rejection gates, cue assets, accounts or GPIO wiring. No Pi deployment
was performed. Existing Pi installations may already contain this correction:
compare the module hash before applying any future selective update.

The sandbox also refuses command capture when known music quieting fails;
it logs `command_aborted` and returns to listening. The new banner is
`pi-flow-v6-quieting-guard-20261012`. Failed TTS quieting releases audio focus.
Existing gain/restore ownership is otherwise unchanged; this is not AEC.

Call-state handling now distinguishes a terminal `disconnected` object from a
live call; unknown/malformed states remain blocking. The returned owned call path
is retained when Dial supplies one. After two idle confirmations, the live loop
settles for `--call-end-tail-seconds 3` before reopening wake listening. This is
a time delay, not an acoustic count of2-3rings. It never calls Hangup/HangupAll.
If the tablet still reports `active` while beeping, automatic cleanup is **not
solved**: a verified remote-ended signal or labeled tone evidence is required
before safely ending a live call. No active-call timeout was shortened.

On a Pi with these source updates installed, this diagnostic is read-only and
redacts numbers/names: `.venv/bin/python -B scripts/call_state_diagnostics.py --seconds 60`.
Run only during a call you choose to test; no call is placed by the diagnostic.
Record connected/post-remote-hangup/post-local-hangup states and stream routing.
No Pi transfer or actual call test has been performed from HPC.

API reference: [PipeWire1.4.2 Bluetooth telephony](https://github.com/PipeWire/pipewire/blob/1.4.2/spa/plugins/bluez5/README-Telephony.md).

Source regressions: `PYTHONPATH=src python -B -m pytest -q tests/test_pipewire_capture_contract.py tests/test_music_capture_safety.py`.
Before/after provenance and host regression logs are retained under
`work/experiments/night_20261012_172034/`; physical sample flow and call recovery
remain Pi verification. Model candidates in that directory are not deployable
drop-in replacements and do not supersede run_043/exp15 automatically.

