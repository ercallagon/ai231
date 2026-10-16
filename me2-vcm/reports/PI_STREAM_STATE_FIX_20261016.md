# Speech and Cue Stream State Follow-up

## Evidence

Operator reports from the fresh ~/swift-clean/me2-vcm Pi installation:

- Piper generated a 3.94-second WAV with RMS -16.7 dBFS. Both direct playback
  and the assistant playback function were audible but very quiet at unity.
- Plain pw-play returned 0; direct chime playback was also very quiet.
- Default USB sink was 0.90, ALSA card 3 PCM enabled on both channels at 90%
  (-3.74 dB). Observed PipeWire links led to that USB speaker.
- The operator confirmed a louder cue using --volume 1.0, an explicit verified
  USB node target, and these properties:
  application.name=SwiftAudioProbe, media.role=Notification,
  state.restore-props=false, state.restore-target=false.

This is an audible combined-probe result. It does not establish whether the
identity, restoration, media role, target, or their interaction was decisive.
No missing synthesis/playback dependency has been demonstrated. Successful
imports and player exit codes alone never establish audible output.

## Source Changes

The speech and cue PipeWire commands share one command builder. Each sets its
own application.name/application.id, Notification role, explicit requested
volume, and disabled saved volume/mute/channel-map/target restoration. Default
target is auto; an existing explicit speech or cue target remains respected.
The default cue path and --cue-sink path both use this behavior.

Speech retains default 0.5 and VCM_SPEECH_CHANNEL_VOLUME / VCM_SPEECH_SINK.
Cue uses unity stream volume after its existing peak-limited waveform gain.
No model, threshold, audio asset, hardware wiring, music cap, credential, global
sink volume, Bluetooth profile, or previous Pi deployment is changed.
The tested USB node name is deliberately not baked into portable source.

Native pw-play is the operator-tested path. paplay/aplay fallbacks are not
claimed to provide the same WirePlumber stream-state behavior. Existing fallback
behavior remains subject to target/volume capability checks.

## Isolation Boundary

Music commands control the selected mpv/Soloist player. Speech and cue have
their own stream gains. Tablet changes limited to Spotify/Bluetooth stream
volume can therefore remain independent. A shared device/master/amplifier gain
change necessarily affects all audio routed through that device. The runtime
does not repeatedly force a global volume to counter external adjustments.

## Host Checks and Pi Procedure

Focused regressions check the real acknowledgement path for both target modes,
speech/cue identity separation, disabled restoration flags, separate gains and
targets, cached/generated TTS, and the existing music/loop safety tests.
Recorded test receipts are kept under ignored work/audio_stream_fix_20261016_v1/.
Host tests cannot reproduce WirePlumber hardware policy or certify loudness.
All 67 focused checks passed, and the frozen artifact-integrity / CPU model
contract verification passed. Source-checksum metadata was refreshed; model
weights, mappings, gates and hardware configuration remain unchanged.

Update the existing Pi checkout after stopping the assistant:

```bash
cd ~/swift-clean
git pull --ff-only origin main
cd me2-vcm
unset VCM_SPEECH_CHANNEL_VOLUME VCM_SPEECH_SINK
./swift run --model champion --live
```

No setup/install rerun is needed. Test TIME, PLAY_MUSIC, TIME, STOP, TIME;
listen for both cue and speech. Start with the retained 0.5 speech default.
If needed, compare a single run with VCM_SPEECH_CHANNEL_VOLUME=1.0 without
changing sink gain. Test a second assistant launch for repeatability. Observe
tablet Spotify-volume changes separately from global sink/mixer changes.
Call transition testing requires deliberate user approval; none was initiated.

The pre-update source backup is work/audio_stream_fix_20261016_v1/source_before.tar.gz
(SHA-256 3075d9b1e5e8b0d3d5935f017bea77a3f8b87a7a3478d4f54fc1407a69b64c3e).
The previous published commit is e2db2a8739ffd276c92822dd0a981a5d3f41ab86.
To compare the earlier revision without overwriting private state, stop the
assistant and use a detached checkout of that commit, then git switch main
to return. An older source revision needs its own matching ARTIFACTS.json.

At initial publication, only the standalone probe was operator-verified.
Subsequent operator reports confirm audible assistant TTS, music stop,
consistent TTS after restarting, and unchanged perceived TTS loudness when
tablet Swift Pi Spotify volume is lowered to zero. The operator also confirmed
that VCM_SPEECH_CHANNEL_VOLUME=0.1 lowers TTS. These are qualitative observations,
not measured SPL; no independent Pi HEAD receipt accompanied the reports.
The final chime level was not explicitly reconfirmed in the latest report.
Arbitrary global/master controls, call transitions and acoustic benchmark
results remain unverified. Later music settings now separate a 30% start from
a 50% ceiling; see docs/AUDIO.md. The earlier 25% value above describes 1100e76.

References:
https://pipewire.pages.freedesktop.org/wireplumber/daemon/configuration/stream.html
https://docs.pipewire.org/page_man_pw-cat_1.html
