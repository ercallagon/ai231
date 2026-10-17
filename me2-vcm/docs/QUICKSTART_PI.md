# Hey Swift on Raspberry Pi 5

Already installed? Use [the final update/benchmark handoff](FINAL_PI_VALIDATION.md)
instead of reinstalling or rerunning private setup. Optional `--end-silence-ms 800`
works with `./swift run`, `live-latency` and `demo`; the public runtime default is 800 ms.

Use these exact commands over SSH. The public names are **Champion (6s BC8-TCN-ASP)**
(threshold 0.954) and **Baseline (2s BC8)** (threshold 0.875); select them with
`--model champion` and `--model baseline`. Historical profile/run identifiers remain
in provenance. No model bytes or gates change.
The separate exp15 wakeword stays at its existing gate. Six seconds is command
capacity, not mandatory listening time.

## 1. Separate clean-install dry run

From a laptop terminal, connect using `ssh -t USER@PI_ADDRESS`. All following
commands run inside that Pi SSH session. Keep your working ~/swift-e2e and
~/ai231 installations intact. Stop their listeners/evaluations using Ctrl+C
before starting tests; do not uninstall them or change their environments.

This dry run has its own checkout, virtual environment, private configuration
and state. It shares the same OS packages, PipeWire routes, Bluetooth pairings,
network, OS clock and physical GPIO. It is not fresh-SD/fresh-OS verification.
Pairing/connecting a phone or selecting a Spotify device changes shared state.

Required OS packages on Debian 13 are git, python3-venv, python3-lgpio,
libportaudio2, libsndfile1, alsa-utils and mpv. Bluetooth integrations also need
bluez, bluez-obexd and the user's PipeWire/WirePlumber telephony support.
The current tested platform is ARM64, /usr/bin/python3 3.13. If these packages
are missing, install the missing packages explicitly with apt; the Python
installer does not change services, OS audio settings or Bluetooth state.
For a new system, the basic package command is:

```bash
sudo apt update
sudo apt install git python3-venv python3-lgpio libportaudio2 libsndfile1 alsa-utils mpv bluez bluez-obexd
```

On your existing Pi, first inspect rather than reinstall/upgrade the working
audio stack. Do not run the apt step merely to repeat this dry run.

```bash
/usr/bin/python3 --version
/usr/bin/python3 -c 'import lgpio; print(lgpio.__file__)'
git clone --filter=blob:none --sparse https://github.com/ercallagon/ai231.git ~/swift-clean
git -C ~/swift-clean sparse-checkout set me2-vcm/src me2-vcm/scripts me2-vcm/configs me2-vcm/models me2-vcm/audio me2-vcm/data/manifests me2-vcm/docs me2-vcm/reports
cd ~/swift-clean/me2-vcm
./swift install
```

If ~/swift-clean exists, stop and inspect it; do not clone over it. The installer
refuses an existing .venv. It creates an environment with --system-site-packages
so apt's lgpio is visible, installs a hash-locked ARM64 wheel set, then actually
imports lgpio. A missing/incompatible lgpio import is a blocker, not a reason
to compile random GPIO packages or replace the OS Python. See DEPENDENCIES.md.

## 2. First-run wizard and diagnostics

```bash
./swift setup
./swift doctor
```

If you enabled the RGB LED/TM1637, run the focused physical check with the
assistant stopped, confirm what you see, then refresh the combined diagnostics:

```bash
./swift gpio
./swift doctor
```

The check cycles red/green/blue and displays `1234`; it records your visual
confirmation separately from software-only imports. It does not test voice
recognition or model-to-action behavior.

Choose private local storage (profile mode 600, directory 700) or session-only
RAM state. Private files stay under this checkout's ignored work/user; never
upload or commit that directory. Session mode runs its own terminal menu and
loads no stored profile. /dev/shm is required; RAM can be swapped by the OS.
An abrupt kill may leave RAM files/its own daemon; normal exit cleans them.
Existing Bluetooth pairing remains OS state in either mode.

The wizard selects the only microphone automatically, or asks you to choose if
there are several. It stores microphone name + host API and resolves the device
index again at each run. Playback uses the OS default sink; inspect it with
`wpctl status -n`. Changing that default affects other installations too.

Choose only the optional integrations you use. Required recipients, MAC
addresses and API keys use hidden input. Paths and service defaults are filled
in automatically:

| Feature | Required setup |
| --- | --- |
| Time/reminders | Correct OS time/timezone; reminders stored in this checkout |
| LED/display | gpiozero/lgpio, permission to /dev/gpiochip*, tested wiring below |
| Weather | Internet and selected coordinates; no API key; unavailable honestly offline |
| Offline music | Put authorized MP3s in the checkout's `work/user/music/`; mpv |
| Spotify | Internet for setup, personal Soloist key/Premium, phone Spotify Connect authorization |
| Calls | Compatible paired phone/tablet, SIM/call permissions, native PipeWire telephony/HFP gateway, private international-format recipient |
| SMS | Phone supports MAP, message-access permission, BlueZ OBEX user service, private recipient |
| Speech | Optional pinned Piper dependencies and a voice you accept the terms for |

Pretrained Piper voices are synthesis-only; neither wake nor VCM detector loads
pretrained weights. Voice/song terms are separate from the project's MIT code.
After you enable spoken responses and accept the voice terms, setup downloads
the pinned voice and installs its hash-locked dependencies into this venv.
It does not copy existing account profiles, contacts, login state or assets.

If you enable spoken responses, setup offers the pinned `en_US-amy-medium`
Piper voice (about 60 MiB), shows its model card and source terms, and asks for
acceptance. On acceptance it downloads the `.onnx` and matching `.onnx.json`
into ignored `work/user/voices/`, verifies published sizes and checksums, and
installs the pinned Piper dependencies in this checkout's venv automatically.
The voice is used only for synthesis, not by either detector. No voice is
bundled or downloaded unless speech is enabled and its terms are accepted.

The Amy model card identifies Mycroft Mimic3 Voices as its source and refers
to that source for license terms. Review the [voice model card](https://huggingface.co/rhasspy/piper-voices/tree/664c651454f055ed34bd83f09e024ffbc0da09ac/en/en_US/amy/medium)
and [source voice license](https://github.com/MycroftAI/mimic3-voices/blob/master/LICENSE)
before accepting. Voice-model rights are distinct from this project's code
license.

For offline music, setup creates
`~/swift-clean/me2-vcm/work/user/music` and uses it automatically. It continues
if the folder is empty; copy authorized MP3 files there before trying music.
For example, from the laptop terminal, upload files you own or are licensed to use:

```bash
scp /path/to/authorized/*.mp3 USER@PI_ADDRESS:~/swift-clean/me2-vcm/work/user/music/
```

No path entry is needed. Music stays under the new checkout after copying.

For Spotify, obtain your key through a phone/browser at the official page:
https://developer.spotify.com/documentation/soloist/concepts/authentication
When you enable Spotify, setup automatically downloads the official ARM64
archive into ignored `work/user/downloads/` and installs the executable under
`work/user/bin/` if it is not already there. It prints the actual archive size.
This requires internet on the Pi and runs only after you enable Spotify. You can
also install from Spotify's [official downloads page](https://developer.spotify.com/documentation/soloist/reference/downloads-and-updates).
The project does not redistribute Soloist. Spotify builds expire after 90 days.
The setup uses Spotify Top 50 - Philippines by default. To use another playlist,
set its Spotify URI after setup with `./swift config set spotify-uri URI`.
Then authorize it:

```bash
./swift spotify
```

This command tries saved-login activation automatically and reports whether it
succeeded. For first authorization only, select **Swift Pi** in the phone
Spotify device picker on the same LAN and rerun `./swift spotify`. No password
login is automated on the Pi. The new
data directory and dynamic loopback control port avoid importing/colliding
with the working deployment. A user-owned process may inspect Soloist's key
argument; it is never echoed, logged by the wrapper, or entered in shell history.
Normal runtime reuses only this explicitly configured installation's login.
Its default initial playback level is 30 percent with a 50 percent music limit;
two voice volume-up commands increase it to 40 then 50 percent. This is
half-duplex volume control, not echo cancellation or a global call-volume fix.
Soloist builds expire; an expired executable needs a fresh official download.

For phone integrations, explicitly pair/connect and approve phone permissions:

```bash
./swift bluetooth
./swift doctor
```

The pairing command asks before changing shared Bluetooth state. If the phone is
already paired, it skips pairing and proceeds to trust/connect. Passkeys and
permissions need physical phone interaction. Diagnostics do not place calls or
send messages. Compatible gateway/MAP detection is not successful call/SMS proof.
Unsupported phones remain a clearly reported integration blocker.

### Change One Setting Later

Run setup from the checkout to review service enablement. To replace one private
value without repeating setup, use `config set`; the profile is backed up
privately and all other settings are retained. Sensitive values are requested
with hidden input, so do not put phone numbers or keys in shell arguments:

```bash
./swift config set call-number
./swift config set sms-number
./swift config set phone-mac
./swift config set spotify-key
./swift config set spotify-uri spotify:playlist:PLAYLIST_ID
./swift config set weather-location
```

The first four commands prompt privately. Press Enter at the Spotify URI prompt
to restore the Top 50 - Philippines default. `weather-location` prompts for a
city and coordinates. These commands change one saved field and preserve the
rest of the private profile.

### Update After Installation

If `./swift install` already succeeded and setup was cancelled, update source
without reinstalling dependencies:

```bash
cd ~/swift-clean
git pull --ff-only origin main
cd me2-vcm
./swift setup
./swift doctor
```

The setup files and downloaded assets under `work/user/` remain in place. If a
profile was already saved, setup asks whether to reuse it; use `./swift config
set ...` to update one value without rebuilding the profile.

## 3. Preview and real demo

```bash
./swift run --model champion
```

Say Hey Swift, wait for the cue and **capture-ready message**, then speak.
Preview validates commands without actions. Empty captures are logged and return
quietly to listening. Add `--speak-no-speech` if you want spoken empty-capture
feedback. Over-limit input rejects safely. The default silence endpoint is
800 ms; capture setup adds a separate delay after the cue. An explicit
`--end-silence-ms 1200` allows longer pauses at the cost of response time.
The frozen model YAML retains its historical 1200 ms capture entry; the current
launcher supplies the 800 ms runtime default. Music can mask wake detection;
the runtime uses playback suppression and ducking, not acoustic echo cancellation.

For enabled real actions:

```bash
./swift run --model champion --live
```

Microphone selection comes from setup; the public `swift run` command does not
take `--device`. `champion` and `baseline` are the two models. `selected` and
`run043` remain compatibility aliases for the same models; they add no models.
Accepted calls and SMS route automatically to the saved recipient. To request
typed approval instead, add `--confirm-communications`. Recognition rejection
still prevents actions. Empty captures stay quiet unless `--speak-no-speech`
is specified. None of these options changes a model or rejection threshold.

While music is playing, type a control into this same SSH terminal and press
Enter; no wake phrase is required:

```text
:volume 10
:quieter
:pause
:stop
:help
```

Controls run between capture operations and affect only the selected music
player. `:volume` accepts whole percentages up to the configured music limit
cap; `:quieter` decreases by ten points. The selected backend is retained so
pause/stop/volume commands continue to target the player that owns playback.
At each new assistant launch, that player's volume is restored to the configured
30% start level by default; terminal changes apply for the current run. This affects only
the selected music player, not the speaker sink or call volume.
This provides a demo recovery path when playback masks Hey Swift; acoustic
echo cancellation is not implemented.

Piper speech uses a separate PipeWire playback stream at 50% by default. This
does not change the chime, music, call stream or global sink. To adjust TTS for
one run, use a value from 0 to 1. To pin only speech to a specific PipeWire
output node, set `VCM_SPEECH_SINK` to the exact node name shown by `wpctl status -n`:

```bash
VCM_SPEECH_CHANNEL_VOLUME=0.65 VCM_SPEECH_SINK=alsa_output.usb-DEVICE.analog-stereo ./swift run --model champion --live
```

The launcher passes only these speech overrides while continuing to isolate
other inherited `VCM_*` variables. With no volume override, every new run starts at 0.5.
`pw-play` (or `paplay`) is required to apply speech-only gain; the runtime will
not raise global output volume as a fallback.

Speech and wake cues have distinct `Hey Swift Speech` / `Hey Swift Cue` stream
identities with the Notification role. Their PipeWire properties disable saved
volume/mute/channel-map and target restoration; otherwise an old `pw-play`
setting can override the requested behavior. They follow the current OS default
output unless explicitly targeted. Cue playback starts at unity stream volume;
its separate peak-limited cue gain still applies. No speaker node is hard-coded.
These settings isolate application streams, not the physical amplifier: tablet
changes to Spotify volume affect music, but changes to the shared speaker's
master/hardware volume still affect speech and cues too.

The Pi operator confirmed TTS remains audible after restarts and when the
tablet's Swift Pi Spotify volume is zero. `VCM_SPEECH_CHANNEL_VOLUME=0.1` was
also confirmed to lower TTS independently. These are operator observations,
not measured sound-pressure levels. See [audio settings](AUDIO.md) for defaults,
one-run overrides, reset instructions and troubleshooting.

To use quieter music settings for one run while retaining your saved setup:

```bash
./swift run --model champion --live --music-start-volume 20 --music-limit 40
```

The defaults are start 30 / limit 50. Higher music may mask Hey Swift; this
ceiling is configurable, not a verified wake-detection guarantee. Pause/resume
preserves a lower user-selected volume within the run. A new run restores its
configured start level. None of these music options changes TTS or cue gain.

With Spotify enabled, live startup warms its saved-login daemon for up to
15 seconds before listening, without requesting activation or playback.
Play music then performs automatic activation and confirms playback within
a 25-second operation budget (plus bounded cleanup where required).
Offline fallback still waits for safe online quiescence. A first-attempt
failure remains diagnostic evidence, not proof that startup timing caused it.
Keep the `Spotify:` line including its phase when reporting a fallback.

Calls pause wake capture while Bluetooth reports an active or unknown call.
The optional confirmation wrapper preserves call-state access. Transient
query failures retry within the guard's five-minute limit; listening resumes
after confirmed idle and the existing three-second end-tone tail. The runtime
does not hang up calls. A persistent unknown state or guard timeout still
stops capture safely; see [diagnostics and rollback](PI_RUNTIME_FEEDBACK.md).

Stop preview with Ctrl+C, then collect the automated comparison below. The
real demo comes afterward so reset does not archive your new observations.

## 4. Automated baseline/champion report metrics

Stop the live listener. For a clean comparison:

```bash
./swift evaluate --reset
```

This reuses test_pi.py/reproduce.py and benchmark_onnx_pi.py. Old work/pi5_test_01
is archived, not deleted; model/data/private-state directories remain intact.
Pinned Parquet downloads use unique temporary files, so an interrupted fetch does
not block retry; no manual removal of cached or partial data is needed.
It evaluates exactly baseline + champion on pinned test (4,693) and holdout
(202) recordings, reusing cached downloads; the first fetch is about 305 MB.
It collects intent/slot/joint/rejection/family/cohort/coverage results, paired
errors, feature/model timings, model-only timings, CPU/RSS/temperature sampling,
and source/environment/model hashes. No calibration, training or real action
occurs. It prints three stage markers and about ten progress updates per split,
then reports each model-only timing. The report is work/pi5_test_01/REPORT.md
with REPORT.json and raw evidence.
Run this before live/demo evidence collection: --reset archives old demo/live
logs too. If you already performed a demo, its archived file is retained and
may be reviewed separately; repeat the guided checks for the active report.

For the slide's primary latency use **feature preprocessing + inference p95**
and per-clip RTF from the test results, with processed/attempted/failure counts.
It excludes decoding/resampling/capture/endpoint wait/actions/TTS. No live speech
is required. Full test/holdout sets are previously exposed; do not tune from them.
All recognition failures remain in denominators; timing covers processed clips.

## 5. Real demo, optional live latency and consolidated report

Stop preview with Ctrl+C. Start the guided real demo:

```bash
./swift demo --model champion
```

This asks before starting real configured actions and provides example phrases.
Every CALL/MESSAGE additionally needs the exact typed CALL/MESSAGE approval;
noninteractive execution fails closed. You can leave both disabled. After
Ctrl+C, record PASS/FAIL/NOT_RUN only for effects you actually observed. These
are operator observations, separate from automatically scored recognition.
To use the baseline, replace champion with baseline. `run --live` is available
for direct configured operation; demo is preferable for a first hardware check.

Wiring is unchanged: KY-016 RGB BCM17/27/22 (GND physical9); TM1637 CLK BCM23,
DIO BCM24, VCC physical17 and GND physical6. Keep both attached. Real appliance
control beyond these adapters is not claimed. New display actions retain the
existing timer/alarm override policy.

For the separate live speech-end-to-decision measurement:

```bash
./swift live-latency --model champion
```

Speak trials across families; Ctrl+C stops the session and automatically
summarizes its unique log directory. Use 30 trials for a preliminary check,
preferably 100+ for p95. This metric starts at detector-estimated final speech
frame read time, includes endpoint waiting and ends at prediction readiness.
It excludes actions/TTS and is not independently annotated acoustic timing.
No-speech trials are counted separately, never assigned zero latency.

After live/demo testing, refresh the combined report:

```bash
./swift report
```

REPORT.md includes available counts, failures, uncertainty and scopes; missing
live/acoustic/physical results stay marked missing. Metrics and acquisition
methods are listed in METRICS.md. Exp15's historical clip-test metrics and
training facts are references, not newly measured Pi streaming performance.
The completed Champion laptop-speaker -> Pi-microphone benchmark is summarized
in [the Pi acoustic report](../reports/PI5_ACOUSTIC_CHAMPION_20261017.md).
Baseline and Champion were compared on frozen injected test data; no acoustic
Baseline comparison was performed. For a new acoustic run, follow
[BENCHMARK.md](BENCHMARK.md). Import a locally available result directory with
`./swift report --official PATH`. Injection and acoustic playback have different
measurement scopes.

## 6. Retrieve and rollback

From Windows PowerShell (archive an older local pi5_test_01 folder first):

```powershell
scp -r USER@PI_ADDRESS:~/swift-clean/me2-vcm/work/pi5_test_01 .
```

Only transfer results, never work/user or runtime/account directories. SSH/SCP
works over your LAN/hotspot without public internet once dependencies/data are
cached. Spotify/weather still need their own internet connectivity.
For an offline demo, keep the SSH LAN/hotspot up while disabling its internet
uplink (for example hotspot mobile data). Do not turn off the Pi's only SSH
interface just to simulate unavailable internet.

Rollback: Ctrl+C the new assistant, then run the preserved command using the
existing ~/swift-e2e/.venv. No file restoration is required. If pairing or output
selection changed shared OS state, restore the intended phone/output in the
phone and PipeWire controls. Stop the new Soloist device before restoring music;
never kill another deployment's daemon by a broad process-name command.
