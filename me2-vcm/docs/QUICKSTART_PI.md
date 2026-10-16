# Hey Swift on Raspberry Pi 5

Use these exact commands over SSH. Public model names are **champion** (six-second
BC8/TCN/pooling, threshold 0.954) and **baseline** (run_043 BC8, threshold 0.875).
They alias the existing selected/run043 artifacts; no model bytes or gates change.
The separate exp15 wakeword stays at its existing gate. Six seconds is command
capacity, not mandatory listening time.

## 1. Separate clean-install dry run

From a laptop terminal, connect using `ssh -t USER@PI_ADDRESS` (for the current
project Pi: `ssh -t ej@192.168.100.103`). All following
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

Choose private local storage (profile mode 600, directory 700) or session-only
RAM state. Private files stay under this checkout's ignored work/user; never
upload or commit that directory. Session mode runs its own terminal menu and
loads no stored profile. /dev/shm is required; RAM can be swapped by the OS.
An abrupt kill may leave RAM files/its own daemon; normal exit cleans them.
Existing Bluetooth pairing remains OS state in either mode.

The wizard lists audio devices and stores microphone name + host API, resolving
the index again at each run. Missing/ambiguous devices fail explicitly. Playback
uses the OS default sink; inspect it with `wpctl status -n` and listen at a safe
level. Choosing/changing a default sink affects other installations too.

Enable only integrations you actually have. Private recipients, MAC addresses
and API keys use hidden input. Each feature's prerequisites are explicit:

| Feature | Required setup |
| --- | --- |
| Time/reminders | Correct OS time/timezone; reminders stored in this checkout |
| LED/display | gpiozero/lgpio, permission to /dev/gpiochip*, tested wiring below |
| Weather | Internet and selected coordinates; no API key; unavailable honestly offline |
| Offline music | Your authorized MP3 directory; mpv; no songs are redistributed |
| Spotify | Internet for setup, personal Soloist key/Premium, phone Spotify Connect authorization |
| Calls | Compatible paired phone/tablet, SIM/call permissions, native PipeWire telephony/HFP gateway, private international-format recipient |
| SMS | Phone supports MAP, message-access permission, BlueZ OBEX user service, private recipient |
| Speech | Optional pinned Piper dependencies and a voice you accept the terms for |

Pretrained Piper voices are synthesis-only; neither wake nor VCM detector loads
pretrained weights. Voice/song terms are separate from the project's MIT code.
The wizard can install the optional speech lock into this venv after approval.
It does not copy existing account profiles, contacts, login state or assets.

For spoken responses, setup offers a pinned `en_US-amy-medium` Piper voice
(about 60 MiB) and downloads its `.onnx` plus matching `.onnx.json` into this
checkout's ignored `work/user/voices/`. It displays the model card and voice
terms before asking for acceptance. The default is synthesis-only; it is not
used by either detector. Users may instead provide another voice they have
reviewed. No voice is bundled or downloaded unless speech is enabled and the
user accepts its terms. The default files are pinned to a Piper Voices
revision and checked against the published sizes and MD5 values.

The Amy model card identifies Mycroft Mimic3 Voices as its source and refers
to that source for license terms. Review the [voice model card](https://huggingface.co/rhasspy/piper-voices/tree/664c651454f055ed34bd83f09e024ffbc0da09ac/en/en_US/amy/medium)
and [source voice license](https://github.com/MycroftAI/mimic3-voices/blob/master/LICENSE)
before accepting. Voice-model rights are distinct from this project's code
license.

For offline music, the wizard defaults to
`~/swift-clean/me2-vcm/work/user/music` for this checkout. Copy your authorized
MP3 files into that directory before enabling music, or enter another directory
inside this checkout. The setup rejects a directory without `.mp3` files.
For example, from a second Pi SSH terminal, after confirming the source folder
contains only files you are authorized to use:

```bash
mkdir -p ~/swift-clean/me2-vcm/work/user/music
cp ~/ai231/me2-vcm/integration/assets/*.mp3 ~/swift-clean/me2-vcm/work/user/music/
```

Then enter the checkout-local destination when the setup wizard asks. This
keeps runtime music files out of the code checkout's tracked files while
avoiding a dependency on the older installation after copying.

For Spotify, obtain your key through a phone/browser at the official page:
https://developer.spotify.com/documentation/soloist/concepts/authentication
When you enable Spotify, setup can download the official ARM64 archive directly
from Spotify into this checkout's ignored `work/user/downloads/` and install the
executable under `work/user/bin/`. It prints the actual archive size. This
requires internet on the Pi and does not run during general dependency
installation. You can instead provide an executable you downloaded from
Spotify's [official downloads page](https://developer.spotify.com/documentation/soloist/reference/downloads-and-updates).
The project does not redistribute Soloist. Spotify builds expire after 90 days.
Then run:

```bash
./swift spotify
```

Select **Swift Pi** in the phone Spotify device picker on the same LAN, then
press Enter in SSH. No browser/password login is automated on the Pi. The new
data directory and dynamic loopback control port avoid importing/colliding
with the working deployment. A user-owned process may inspect Soloist's key
argument; it is never echoed, logged by the wrapper, or entered in shell history.
Normal runtime reuses only this explicitly configured installation's login.
Its initial playback level and listening cap are both 25 percent; this is
half-duplex volume control, not echo cancellation or a global call-volume fix.
Soloist builds expire; an expired executable needs a fresh official download.

For phone integrations, explicitly pair/connect and approve phone permissions:

```bash
./swift bluetooth
./swift doctor
```

The pairing command asks before changing shared Bluetooth state. Passkeys and
permissions need physical phone interaction. Diagnostics do not place calls or
send messages. Compatible gateway/MAP detection is not successful call/SMS proof.
Unsupported phones remain a clearly reported integration blocker.

## 3. Preview and real demo

```bash
./swift run --model champion
```

Say Hey Swift, wait for the cue and **capture-ready message**, then speak.
Preview validates commands without actions. No-speech or over-limit input
rejects safely. The current conservative endpoint hangover is 1.2 seconds;
there can also be capture-setup delay after the cue, still requiring Pi review.
No acoustic echo cancellation is claimed. Pause tolerance and music wake
audibility need physical testing.

Stop preview with Ctrl+C, then collect the automated comparison below. The
real demo comes afterward so reset does not archive your new observations.

## 4. Automated baseline/champion report metrics

Stop the live listener. For a clean comparison:

```bash
./swift evaluate --reset
```

This reuses test_pi.py/reproduce.py and benchmark_onnx_pi.py. Old work/pi5_test_01
is archived, not deleted; model/data/private-state directories remain intact.
It evaluates exactly baseline + champion on pinned test (4,693) and holdout
(202) recordings, reusing cached downloads; the first fetch is about 305 MB.
It collects intent/slot/joint/rejection/family/cohort/coverage results, paired
errors, feature/model timings, model-only timings, CPU/RSS/temperature sampling,
and source/environment/model hashes. No calibration, training or real action
occurs. The report is work/pi5_test_01/REPORT.md with REPORT.json and raw evidence.
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
The official class acoustic benchmark still needs laptop-speaker -> Pi-mic
playback; injection or SSH alone does not replace it. See BENCHMARK.md. Import
its result directory when available with `swift.py report --official PATH`.

## 6. Retrieve and rollback

From Windows PowerShell (archive an older local pi5_test_01 folder first):

```powershell
scp -r ej@192.168.100.103:~/swift-clean/me2-vcm/work/pi5_test_01 .
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
