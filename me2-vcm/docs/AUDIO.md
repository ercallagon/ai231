# Audio Settings

All commands below run inside the existing Pi project directory. Keep your
saved setup; no new installation, voice download or account authorization is
needed when changing volume.

| Audio | Default | Control | Affects |
| --- | --- | --- | --- |
| Piper speech | 0.5 stream gain | VCM_SPEECH_CHANNEL_VOLUME, range 0..1 | Speech only, cached and new responses |
| Speech output | Current OS default | Optional VCM_SPEECH_SINK node name | Speech route only |
| Wake chime | Unity stream gain, existing cue waveform/gain | Advanced runner --cue-gain-db and --cue-sink | Cue only |
| Music start | 30% each assistant run | Public run --music-start-volume | Selected music player |
| Music ceiling | 50% | Public run --music-limit | Player volume commands and terminal controls |
| Music during capture/TTS | Muted, then restored | Existing half-duplex focus hooks | Selected music player |
| Global device/master gain | Existing OS/hardware setting | OS, hardware, or an external controller | Every sound on that output |

## Speech Volume and Route

Lower TTS for one run (confirmed by the Pi operator):

```bash
VCM_SPEECH_CHANNEL_VOLUME=0.1 ./swift run --model champion --live
```

Use any finite value from 0 through 1. Zero mutes speech; 0.5 is the default;
1 requests unity stream gain. These values are gain controls, not measured
perceived loudness. Changing gain does not require regenerating cached phrases.

To pin only speech to a specific output, inspect `wpctl status -n` and use the
exact node name rather than a temporary numeric ID:

```bash
VCM_SPEECH_SINK='EXACT_OUTPUT_NODE_NAME' VCM_SPEECH_CHANNEL_VOLUME=0.5 ./swift run --model champion --live
```

A command-prefix override applies only to that invocation and is not saved in
the setup profile. Reuse the same command to get the same requested settings.
An exported shell variable instead lasts in that shell until unset. To return
to defaults explicitly:

```bash
unset VCM_SPEECH_CHANNEL_VOLUME VCM_SPEECH_SINK
./swift run --model champion --live
```

## Music Headroom

```bash
./swift run --model champion --live --music-start-volume 30 --music-limit 50
```

These are also the defaults. PLAY starts at 30, the first two VOLUME_UP commands
reach 40 and 50, and another VOLUME_UP at the ceiling reports the 50% limit.
PAUSE/PLAY preserves a lower volume chosen during that run. A new run restores
its configured start level. Existing custom low limits on the advanced runner
remain supported. Start must be between zero and the limit, and limit <= 100.

During a running assistant, `:volume 10`, `:quieter`, `:pause` and `:stop` control
music without requiring Hey Swift. Tablet Spotify controls can adjust the same
music session independently. Higher music can mask wake speech; 50% is a demo
ceiling with headroom, not a validated acoustic guarantee. For quieter use:

```bash
./swift run --model champion --live --music-start-volume 20 --music-limit 40
```

The assistant changes player volume, not the global speaker sink. A tablet that
changes the shared master or hardware amplifier can still affect all audio.

## Quiet Empty Captures

A false wake followed by `speech_start_timeout` / `no_sustained_speech` returns
quietly to listening. The console retains "I didn't hear a command", but no
spoken rejection plays by default. Inference is skipped: confidence 0 and model
time 0 are placeholders, and missing speech-end timings remain null. Add
`--speak-no-speech` only if spoken timeout feedback is wanted. Noise that passes
the speech gate can still reach the model and receive a spoken rejection.

## Verification and Troubleshooting

Speech and cues use distinct Notification streams and disable saved stream
volume/mute/channel-map and target restoration. Default output follows the OS;
no developer's USB device name or system-wide volume reset is embedded.
`pw-play` is the Pi-tested player. Other playback fallbacks do not establish
equivalent WirePlumber behavior. Piper with an authorized voice pair supplies
synthesis; sounddevice handles capture, not Linux speech playback.

After source revision 1100e76, the Pi operator reported audible TTS, successful
music stop, unchanged perceived TTS loudness at tablet Spotify volume zero,
repeat-run consistency, and lower TTS at gain 0.1. These are qualitative
operator reports. The exact Pi HEAD was not independently supplied with them.
The louder initial probe changed identity, role, restoration and target together;
it did not isolate a single original cause. See
[the evidence report](../reports/PI_STREAM_STATE_FIX_20261016.md).

If speech is unexpectedly quiet, check for an exported speech override or
incorrect speech target first. `./swift doctor` loads the voice but does not
prove audible output. Keep source WAV level, stream volume/mute, route and
global hardware gain distinct when collecting diagnostics. Do not delete
WirePlumber state or raise the global sink as an automatic workaround.
Call transitions and standardized acoustic benchmarking remain separate checks.
