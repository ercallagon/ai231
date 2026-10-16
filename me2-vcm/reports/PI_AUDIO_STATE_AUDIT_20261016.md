# Pi Audio and Repeat-Run State Audit

## Evidence Boundary

The Pi was not reachable from the sandbox. No fresh Pi sink, stream, mute,
environment or audibility measurements were made. The current published source
was inspected alongside the retained `pi_runtime_reference` speech patch.
Physical audibility and Pi stream routing remain user verification items.

## TTS Playback

The published source sent Piper WAVs through plain `pw-play`; it did not set a
TTS stream volume. The launcher also removed inherited `VCM_*` values, so a
`VCM_SPEECH_CHANNEL_VOLUME` shell override could not reach the assistant. The
retained Pi reference used `pw-play --volume` with a 0.5 default. Its older
channel-volume property path is not used for speech here.

The current source now sends both cached/precomputed and freshly synthesized
speech through `pw-play --volume` (default 0.5), or `paplay --volume` when
PipeWire's CLI player is absent. `VCM_SPEECH_CHANNEL_VOLUME` accepts finite
values from 0 through 1. Optional `VCM_SPEECH_SINK` pins only the speech stream;
both overrides are passed by the public launcher. If neither per-stream player
exists, speech fails with a clear error; the code does not alter the global
sink. Wake chime still uses `play_wav`, independent of TTS gain and target.
Music player controls and Bluetooth call routing are unchanged.

## Start State and Persistent Data

| State | Behavior across `./swift run` | Demo implication |
| --- | --- | --- |
| TTS output gain | Defaults to 0.5 each process; one-run override is inherited only for `VCM_SPEECH_CHANNEL_VOLUME`. | A shell override does not mutate the private profile or sink. |
| Output route | Speech follows the OS default output, or the optional `VCM_SPEECH_SINK`; chime follows its own cue setting/default. The code does not change the sink or Bluetooth profile. | Inspect the actual Pi PipeWire graph; no HPC claim about physical routing. |
| Music route | `state/music-routing.json` remembers which managed player owns playback. | This avoids sending pause/stop/volume to the wrong player. |
| Music volume | At startup, the remembered running player is reset to its configured 25% baseline. If no player is running, first playback starts at the configured cap. `:volume` changes apply for that assistant run. | Repeated runs start consistently without changing sink or call volume. |
| Reminders | `state/reminders.json` intentionally persists. Only three distinct values are supported; duplicate creates are ignored. | It cannot grow without bound. `LIST_REMINDERS` shows previous saved items. Do not clear it automatically at process exit. |
| Timer/alarm jobs | In-memory only. Runtime shutdown cancels the job and clears the display. | Recreate timer/alarm after starting a new assistant process. |
| Private profile and Spotify state | Persist under this checkout's private `work/user/` tree. | These include setup and authorization state; do not reset for repeat trials. |
| Demo observations | `private_json` archives an existing `DEMO.json` before writing a new one; report reads the newest `DEMO.json`. | Earlier observations remain in timestamped `.previous_*.json` files. |
| Injected evaluation | `./swift evaluate --reset` archives the prior `work/pi5_test_01`; live latency logs use timestamped directories. | Preserve these reports when comparing trials. |

## Reproduction Checks

Host regression tests cover per-stream TTS gain, cached and generated WAVs,
launcher override forwarding, and resetting only the remembered music player.
They do not establish that a person can hear the output on the Pi. Do not use a
phone call during the sequence below; call-transition testing needs explicit
user approval.

On the Pi, first capture the read-only baseline before pulling the source update:

```bash
cd ~/swift-clean/me2-vcm
git rev-parse --short HEAD
git status --short
rg -n 'VCM_SPEECH_CHANNEL_VOLUME|pw-play|def play_speech_wav|def play_wav' src/hey_swift/audio/speech.py src/hey_swift/deployment.py
if [ "${VCM_SPEECH_CHANNEL_VOLUME+x}" = x ]; then printf 'TTS volume override is set: %s\n' "$VCM_SPEECH_CHANNEL_VOLUME"; else echo 'TTS volume override is unset'; fi
if [ "${VCM_SPEECH_SINK+x}" = x ]; then printf 'TTS sink override is set: %s\n' "$VCM_SPEECH_SINK"; else echo 'TTS sink override is unset'; fi
wpctl status -n
wpctl get-volume @DEFAULT_AUDIO_SINK@
wpctl inspect @DEFAULT_AUDIO_SINK@ | rg -i 'node.name|mute|volume|channelVolumes'
```

During playback, use a second SSH terminal to run `wpctl status -n` and inspect
the temporary playback node with `wpctl inspect NODE_ID`. Compare sink volume
before/after; it should not change. Report whether the TTS stream uses the same
USB output sink and whether it is muted. Keep private profile, Soloist logs,
phone identifiers and credentials out of shared output.

After updating from GitHub, make one WAV through the configured local Piper
voice. This writes a timestamped non-secret phrase into the existing Pi test
folder and refuses to overwrite a file:

```bash
wav="work/pi5_test_01/tts_probe_$(date +%Y%m%d_%H%M%S).wav"
WAV="$wav" PYTHONPATH=src .venv/bin/python - <<'PY'
import json, os, shutil
from pathlib import Path
from hey_swift.audio.speech import PiperSpeech
root = Path.cwd()
profile = json.loads((root / "work/user/profile.json").read_text())
model = Path(profile["voice_model"])
if not model.is_absolute():
    model = root / model
target = root / os.environ["WAV"]
if target.exists():
    raise SystemExit(f"Refusing to overwrite {target}")
voice = PiperSpeech(model)
try:
    voice.prepare()
    shutil.copy2(voice._audio_path("Lights on."), target)
finally:
    voice.close()
print(target)
PY
pw-play --volume 1.0 "$wav"
pw-play --volume 0.5 "$wav"
```

This compares the identical speech waveform at two per-stream gains. Then test
the public assistant with a TTS-only 0.65 override:

```bash
VCM_SPEECH_CHANNEL_VOLUME=0.65 ./swift run --model champion --live
```

If the OS default output changes between checks, pin speech to the exact USB
speaker node name shown by `wpctl status -n` with
`VCM_SPEECH_SINK=EXACT_NODE_NAME` on that command. Do not guess a node name.

Within one run, try `What time is it?`, `Play music`, `What time is it?`,
`Stop music`, `What time is it?`. Confirm TTS remains audible over each state,
music stops, `Music_startup` reports the selected route/baseline, and
`wpctl get-volume @DEFAULT_AUDIO_SINK@` stays unchanged. Stop and start a new
assistant run; the remembered backend should be retained and reset to 25%.
Please report whether each phrase is clearly audible from the physical speaker;
SSH output cannot verify loudness.

## Remaining Pi Checks

- Confirm the baseline checkout and shell override before update.
- Compare direct WAV playback and assistant TTS on the same default sink.
- Confirm TTS stays audible through TIME / PLAY_MUSIC / TIME / STOP / TIME.
- Confirm the music volume reset and route log after a fresh process restart.
- Confirm no sink mute/volume change and report perceived loudness.
- Repeat around a real call only after explicit authorization; no call was made
  or tested by this audit.
