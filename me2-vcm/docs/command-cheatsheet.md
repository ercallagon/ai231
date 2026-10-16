# run_043 Pi Command Cheatsheet

Say "Hey Swift", wait for the complete cue, then speak clearly. These variants
exist in the corrected training manifest; recognition is not guaranteed. The
two-second model window is unchanged. Do not rush or silently shorten a command
after rejection; use duration diagnostics in [v6 guide](rpi-runtime-v6.md).

| Intent | Suggested Covered Phrase | Supported Values / Effect |
| --- | --- | --- |
| TEMPERATURE | Temperature eighteen degrees | 18 / 22 / 26; TM1637 display demo, not HVAC |
| TIMER | Timer ten seconds | 10 seconds / 30 seconds / one minute |
| ALARM | Alarm six AM | six AM / eight AM / nine PM |
| BRIGHTNESS | Brightness level twenty percent | 20 / 60 / 100 percent |
| COLOR | Set color to blue | red / blue / green |
| LIGHT_ON | Lights on | RGB LED |
| LIGHT_OFF | Lights out | RGB LED |
| TIME | Time | Pi clock; must be correct offline |
| WEATHER | Weather | Fresh online data or honest unavailable response |
| CREATE_REMINDER | Remind me to study | drink water / study / exercise |
| LIST_REMINDERS | Reminders | Local saved reminders |
| PLAY_MUSIC | Play music | Auto Spotify activation, safe offline fallback |
| PAUSE | Pause | Selected music player |
| STOP | Stop | Music, not a universal timer/call cancel command |
| NEXT | Next song | Selected music player |
| VOLUME_UP | Volume up | Music only; capped at 25% for wake audibility |
| VOLUME_DOWN | Volume down | Music only; no system/call volume change |
| CALL | Make a call | REAL configured recipient; wake paused during owned call |
| MESSAGE | Send a message | REAL configured message; unknown outcomes must not be retried blindly |

Run flags:

- `--pipewire-source EXACT_USB_SOURCE_NODE_NAME`: shared capture; do not also use --device.
- `--device 0`: direct ALSA/sounddevice; confirm the venv device list first.
- `--cue-gain-db 3`: additional cue-only gain, peak-limited; local WAV unchanged.
- `--cue-sink EXACT_USB_SPEAKER_NODE_NAME`: cue routing only, no default sink changes.
- `--offline`: local demo; no fresh weather/Spotify requests, no fake phone success.
- `--log-format json --benchmark-log`: capture/model/service/speech timing details.
- `--wake-audio-log-dir PATH`: private false-wake recordings, opt-in.
- `--rejected-audio-log-dir PATH`: full failed-command audio/duration evidence, opt-in.
- `--mic-recovery-attempts 5 --mic-recovery-delay 1`: bounded same-input retries.
- `--call-guard-seconds 300`: max wait before stopping safely, never hangs up.

TTS/cue/owned-call suppression is not acoustic echo cancellation. Unrelated
buzz/ring/beep false wakes still need recordings; model/thresholds are preserved.
Real call/SMS tests require your deliberate authorization. No automatic test
script places calls, sends messages or changes live audio profiles.

## Private Setup Updates

From the repository root, update a single saved setting without rerunning setup:

```bash
./swift config set call-number
./swift config set sms-number
./swift config set phone-mac
./swift config set spotify-key
./swift config set spotify-uri spotify:playlist:PLAYLIST_ID
./swift config set weather-location
```

Phone numbers, MAC address and Spotify key are requested through hidden input;
never append them as command arguments. Updates retain every other field and
create a private timestamped profile backup. See [the Pi quickstart](QUICKSTART_PI.md)
for setup defaults, asset locations and update commands after installation.

## GPIO Physical Check

With the assistant stopped and the documented wiring connected, run
`./swift gpio`, visually confirm the RGB sequence and `1234` display, then run
`./swift doctor`. The private confirmation persists across evaluation resets and
is included in diagnostics; it does not verify voice-to-action behavior.
# Capture Safety Note (2026-10-12)

If the runtime reports `Music could not be quieted`, the command was skipped;
no inference/action occurred. Restore the playback controller and wake again.
Do not compensate by raising music volume, lowering wake/OOS thresholds, or
changing the supported spoken slot values. The2s champion still requires a
complete short utterance; new3s experimental models are not installed.
This is half-duplex playback control, not acoustic echo cancellation. The
sandbox fix has host tests; the currently installed Pi version may differ.

Calls: after Bluetooth confirms the call is finished, the updated sandbox loop
waits3seconds for end tones to settle, then returns to **wake listening**. It
does not count audible beeps or automatically end a call still reported active.
If the tablet remains in a live-call state after the remote party leaves, end
it manually for now and collect the redacted call-state diagnostics. No calling
recipient/default or Bluetooth volume was changed.

