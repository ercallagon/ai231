# Pi test and transition workflow

Pi software installation and bounded mock checks are now verified on the deployment below; **live inference, audio I/O and physical effects remain unverified**. This is the canonical Pi runbook. Mock acceptance needs no model API or account; real speech requires configured providers and Pi-side network access. Application source remains `src/me3_voice/`; do not maintain another implementation here.

## Verified Pi software deployment — 2026-11-06

Operator-reported checks used published revision `057321f150dd0c3df72de086e29b572000f2190b` on Raspberry Pi 5, Raspberry Pi OS/Debian 13.7 aarch64, Python 3.13.5, approximately 4 GB RAM. These findings qualify this software smoke path, not every supported OS/Python combination.

| Check | Observed result and limit |
|---|---|
| Locked installation and `swift.sh check` | Passed on the Pi. |
| Deterministic demo and scenarios | First demo had an MCP initialize timeout; retry passed. Its cause is unproven. All 10 mock scenarios passed. |
| Wake/VAD preparation | Official Sherpa assets prepared; one-second synthetic-silence Sherpa/Silero replay passed. No real-voice/wake-accuracy conclusion. |
| Mock bridge | Loopback health check passed; not model readiness or audio verification. |
| Onit | 0.1.5a0 at source pin `3d014cfae043130d956ac042de7142516146e96b` installed separately with uv; `pip check`, actual ME3 worker `READY` handshake with dummy key/empty stdin, and scoped tool imports passed. Transitive packages were frozen, not fully lock-pinned. No model task or website was run. |
| Audio inventory | USB microphone and USB speaker detected. Keep actual node identifiers in private deployment evidence, not this reusable guide. Neither device was opened. |
| VPN/routing | One authorized foreground OpenVPN session connected; TTS route used `tun0`. TCP checks passed for university TTS port 8001 (43 ms) and Groq port 443 (136 ms). User later pressed Ctrl+C: **that tunnel is now stopped**. These are single TCP observations, not inference latency or current reachability. |
| Credentials | Masked setup created a mode-0600 ignored file; preflight found expected Groq/TTS credential names. Values are not published; provider authentication/inference remain unverified. |

ME2 tracked files remained unchanged. GPIO, calls/messages, microphone capture, playback, real inference and website generation stayed inactive. No acoustic, AEC, audible-interruption or physical-effect claim is supported. Earlier HPC-only limitations remain historical; use this dated status for the current Pi software evidence.

## Fresh checkout installation and operation

Canonical checkout location: `~/ai231/me3-voice-agent/`. The reviewed runnable source is published at [ercallagon/ai231 — me3-voice-agent](https://github.com/ercallagon/ai231/tree/main/me3-voice-agent). Clone `https://github.com/ercallagon/ai231.git` into `~/ai231`, then enter its `me3-voice-agent/` subdirectory. Do not clone the entire coursework repository inside the application directory. Do not nest a second application under the existing source. Commands below run from the checkout root and use a project-local editable install: security boundaries and assets are relative to this checkout. A wheel installed globally is not the supported operator workflow.

### Prerequisites and scope

- Target: Raspberry Pi 5 / 4 GB, Raspberry Pi OS **64-bit**, Python 3.11–3.13, `python3-venv`, Git and network access for package/model downloads. The deployment above verifies the locked software smoke path on one aarch64 system; record OS, native wheels, RAM, thermals and audio for each new installation. Physical audio remains unverified.
- Existing PipeWire with `pw-record`, `pw-play`, `wpctl`; do not install/replace an audio service to follow this guide. Browser microphone mode is the supported laptop alternative and requires a trusted localhost origin.
- RGB LED: documented ME2 configuration BCM R/G/B **17/27/22**, active high; TM1637 CLK/DIO **23/24**. Confirm actual module pin labels, common anode/cathode, resistor requirements, safe 3.3 V connections and existing wiring before selecting real hardware. Do not rewire from a generic module diagram. See the shared-device section below.
- GPIO support uses approved OS `python3-gpiozero`/`python3-lgpio` packages. If installed outside the environment, create the project venv with `python3 -m venv --system-site-packages var/cache/venv` **before** setup, then verify imports. No Pi libraries are needed for mock startup.
- No sibling ME2 files or HPC home directory are required. The selected phone/device adapters are in `src/me3_voice/`. Onit is an explicitly installed external dependency in its own environment, not an implicit pre-existing checkout.

### Install, test, and download explicit assets

```bash
bash scripts/setup.sh --pi
bash scripts/swift.sh check
bash scripts/swift.sh demo
# Explicit pretrained asset download and Hey Swift tokenization; no training.
bash scripts/swift.sh assets
```

The asset fetcher uses the official Sherpa GitHub release recorded in `scripts/fetch_kws_assets.py`, selectively extracts inference files, and writes hashes/provenance under `var/cache/wake/`. Preserve its README/license notices. It refuses to overwrite an existing keyword file. Runtime never silently downloads wake weights. Silero is supplied by the pinned Pipecat stack. No ASR/LLM/TTS model weights need to be downloaded for remote inference. Existing model archives, keys and HPC environments are not distributable prerequisites.

For developer regression checks install `bash scripts/setup.sh --pi --dev`, then:

```bash
var/cache/venv/bin/python -B -m pytest -q tests/test_demo_readiness.py tests/test_devices.py tests/test_transcript.py --basetemp=var/scratch/pytest-acceptance
```

Those tests use mocks and software timing; they do not activate hardware or microphone. First run the three-command mock acceptance above before enabling any integration. Package installation and asset downloads need internet; mock execution after setup does not need university access.

### Configuration and real launch

Versioned examples are `configs/backends.example.json` (mock/provider contract), `configs/demo-backends.example.json` (explicit Groq/university demo, live disabled), `configs/devices.example.json` and `configs/music.example.json`. `.env.example` documents empty names; the app does **not** automatically load `.env`. Use the masked setup below to write owner-only files under `var/cache/`, or provide the configured environment variables to the process. Never put keys in CLI values, screenshots or Git.

Follow steps 2–4 of the current demo path immediately below for pinned Onit installation, profile creation, masked key entry, inventory and the full bridge/native-audio commands. They work relative to this checkout. Do not add `--diagnostic-audio` unless deliberately collecting the bounded opt-in recording with the speaker/user present; ordinary operation should omit it.

Use `bash scripts/swift.sh bridge` in place of `var/cache/venv/bin/python -B -m me3_voice.voice_bridge` with the same documented flags. Use `bash scripts/swift.sh audio` in place of the native peer module command. They change to this checkout, set cache/CPU environment, and execute foreground processes; Ctrl+C is preserved. Ports are explicit (`bridge --port 8765`, `audio --port 8765`). No automatic service or public endpoint is created. With the bridge running in terminal A:

```bash
# Terminal B, health only; opens no audio device.
bash scripts/swift.sh health 8765
```

For browser capture, open the forwarded `http://127.0.0.1:8765/` and press Start once after health is ready. For native capture, use only the explicit device peer, and keep the browser as observer. Never run two capture owners. The larger current demo path documents private SSH forwarding; e.g. on the viewing laptop use `ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 YOUR_PI_SSH_ALIAS`. VS Code Ports is equivalent; use its actual local address.

### Accounts and configurable alternatives

| Capability | Required setup | Already-supported alternative / absence |
|---|---|---|
| Demo ASR and reasoning | Own Groq key; free-plan availability/provider quotas apply | University Qwen ASR profile is implemented but answer substitution remains open. Default mocks allow offline startup. |
| Demo TTS | Authorized university endpoint/key and VPN/DILNET route | Other documented adapters require their own credentials/authorization; see ARCHITECTURE provider contracts. Mock tones are not real speech. No automatic paid fallback. |
| Onit project tools | Separate pinned installation below plus configured compatible model/key | Omit Onit flags for mock/basic mode. This loses general project generation, not basic startup. |
| Weather | Network and explicit Open-Meteo location, QC default in prepared profiles | Disabled/unavailable honestly when not configured. Dynamic city lookup supported, ambiguity requires clarification. Service attribution/terms apply. |
| Music | Own Spotify authorization, Premium, appropriate developer permissions, real browser/Connect receiver | Omit music configuration. A token alone is not a speaker/receiver. Pi receiver setup is external and not automatically installed. |
| Calls/SMS | Paired compatible phone, telephony/MAP services/permissions and bounded demo recipients | Disabled by default; setup never pairs/dials/sends. Real answered/delivered state requires separate observation. |
| GPIO/display | Verified wiring, libraries, explicit Pi adapter | Explicit mock mode for software-only testing. A requested real target never silently falls back. |
| Search/GitHub | Optional own Tavily/GitHub tokens plus explicit capability flags | Omit flags; no mandatory account for basic conversation/project tools. HF token alone enables no tool. |

### Network access on the Pi

The laptop's OpenVPN Connect session routes **laptop traffic only**. A Pi on the same Wi-Fi/hotspot does not inherit it. A working SSH/UI port forward does not establish Pi-to-model connectivity. This matters even when all keys are correct.

| Destination | Route needed from the machine running ME3 |
|---|---|
| Groq ASR/LLM, GitHub/Onit/model downloads, Spotify, Open-Meteo | Working DNS/public internet and appropriate HTTPS access |
| University TTS `10.153.60.233:8001` (and optional ASR `:8000`) | University DILNET or approved VPN/routed network from the **Pi** |
| Loopback dashboard `127.0.0.1:8765` | Local Pi browser or explicit private SSH forwarding from the viewing laptop |

For a campus-connected Pi, use authorized DILNET and run the network gate below. Otherwise use an approved, device-appropriate VPN profile; do not assume the laptop's OpenVPN client supplies it.

#### One-time VPN credential setup and managed launch

The verified deployment used OpenVPN 2.6.14 on Debian after user authorization. ME3 does not install OpenVPN. Obtain your own approved client profile and network permission; never copy another user's profile. Protect a profile containing inline certificates/keys in the Pi user's `~/.config/openvpn/` directory (0700 directory, 0600 file). Never commit the profile, its hash, contents, credentials or connection logs.

1. **Run once on the Pi** to save VPN credentials using masked prompts. Enter the path to your private profile; username and password are never command arguments:

   ```bash
   bash scripts/swift.sh vpn-setup --profile /path/to/private-profile.ovpn
   ```

   On a fresh installation, the same optional one-time step can be requested as part of dependency setup:

   ```bash
   bash scripts/setup.sh --pi --vpn-profile /path/to/private-profile.ovpn
   ```

   Omit `--vpn-profile` on later dependency updates: setup refuses to overwrite an existing auth/config file. No VPN connection starts during setup.

   Credentials are saved to `~/.config/openvpn/me3-auth.txt` with mode0600; the launcher settings under `var/cache/` are also mode0600. **The auth file is plaintext protected by Unix permissions, not encrypted at rest.** Anyone with this Pi account or root access can read it. Use this only on a Pi with appropriate disk/user security. To revoke it, stop ME3 and remove the auth file and launcher config; rotate the VPN password if needed. Setup makes no connection.

2. **Start ME3 with its managed VPN command** from the Pi project root, supplying the prepared model/device config and existing bridge arguments:

   ```bash
   bash scripts/swift.sh vpn \
     --backend-config var/cache/pi_live-backends.json -- \
     --secrets-file var/cache/demo-models.env \
     --device-config var/cache/pi_demo-devices.json \
     --onit-python "$PWD/var/cache/onit-venv/bin/python" --onit-primary \
     --onit-reasoning-effort low --onit-max-output-tokens 2048 \
     --onit-output-root "$PWD/var/scratch/demo-projects" --direct-clock \
     --receipt-store var/cache/pi-demo-receipts.sqlite3 \
     --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
     --keywords-file var/cache/wake/hey-swift-keywords.txt
   ```

   The wrapper waits up to 120 seconds for OpenVPN’s initialization message, then runs the existing configured TCP gate with a bounded timeout. It starts the loopback bridge only after that gate passes. The gate sends no key, HTTP or inference request; TCP reachability does not confirm TLS, provider authentication, model health or speech. Use a mock device profile for software-only trials; physical outputs need their separate authorization.

The operator may still need to enter the OS sudo password at launch; saved VPN credentials do not grant root permission. No sudoers policy or root-password persistence is added. The managed launcher rejects an existing OpenVPN process and concurrent launches rather than modifying an unowned tunnel. It stops on authentication failure; there is no repeated-password loop. Profiles that daemonize, drop privilege, chroot or use a management interface require manual review instead of managed startup.

3. Keep the launcher terminal open. Ctrl+C/SIGTERM/SIGHUP or bridge exit initiates cleanup of the owned bridge and VPN. If VPN disconnects during operation, the launcher stops its bridge. If routes fail or initialization expires, the bridge is not started. If OS permission expires or OpenVPN will not stop, cleanup failure is reported and the operator must inspect/disconnect that owned tunnel manually. A forced kill or lost power cannot provide graceful cleanup guarantees; a subsequent launch refuses an existing tunnel.

This is opt-in for each ME3 run, not boot-time autostart. Reboot requires running the launcher again. OpenVPN uses `--auth-nocache`; ME3 retains the protected auth file for future starts. The earlier manually started tunnel is stopped. Do not start a second VPN tunnel alongside this command. Managed startup has offline unprivileged-process tests, **not yet a real Pi/OpenVPN qualification**. Until the attended Pi startup/stop check passes, retain the verified manual command as fallback:

```bash
sudo /usr/sbin/openvpn --auth-nocache --config /path/to/private-profile.ovpn
```

The saved file is for VPN credentials only; service API keys remain in the existing separate private model-key file. [OpenVPN2.6 documents](https://openvpn.net/community-docs/community-articles/openvpn-2-6-manual/) the two-line auth-user-pass file and auth-nocache behavior.

If neither is available, stop before claiming a network-ready Pi. An explicit private relay/tunnel through an authorized host is a separate setup decision, requires the Pi to reach that host, adds availability/latency dependencies, and is **not implemented or validated here**. The application deliberately rejects arbitrary plaintext/local forwarding endpoints; do not rewrite `10.153.60.233` to localhost or widen that protection silently. An alternative already-supported TTS provider is possible only with its own credentials, documented contract and authorization; no paid fallback is selected automatically. Do not expose university APIs publicly or share a VPN connection by enabling IP forwarding/NAT without network-owner permission.

After routing is resolved, perform one bounded real speech test separately. Record the Pi network/route and per-role result; laptop/HPC reachability does not qualify the Pi.

### Stop, diagnose, update and rollback

Stop the native peer with Ctrl+C first, then the bridge/preview server you started. Do not kill listeners blindly or touch ME2/system services. Spoken End ends engagement, not the process or independent tasks. Check provider/device status with `swift.sh preflight` and the same explicit config paths, inspect `/api/health`, Diagnostics, and `var/results/`/`var/logs/`. The existing receipt ledger is opt-in; it restores historical receipts, not chat memory or active timers.

- **Page unavailable:** check the bridge foreground error and `swift.sh health`; dashboard-only mode has no voice health. Check chosen local/remote ports. A busy port requires identifying its owner or choosing another port, never killing an unrelated process. Successful TCP forwarding alone does not prove HTTP health.
- **Missing wake paths:** run the explicit asset setup once and verify all three ONNX files, tokens and keyword file. No copied HPC cache is assumed.
- **Microphone unavailable:** distinguish browser permission/device errors from bridge/network health; native mode requires actual `wpctl status` node identities. Never select a speaker monitor as microphone.
- **Thinking/provider failures:** check model/role labels and provider reachability/limits independently; keys are not printed. Preserve receipts and reconcile Unknown before repeating an external action.
- **Pi action unavailable:** inspect OS imports, permissions and verified wiring. Keep real target unavailable rather than concealing it as mock. Do not initialize GPIO solely for preflight.

Before an update, record both parent `HEAD` and the deployed ME3 source revision, preserve private config/ledger paths and local source edits, and stop only ME3-owned processes. For a shared or sparse coursework checkout, **do not run a root-level `git pull`**: use the ME3-only synchronization procedure below, then rerun applicable software acceptance before separately qualifying physical devices. For rollback select the previously recorded revision only with local work preserved, rerun its locked setup, and explicitly select the previous private profile. No update/rollback script resets or overwrites user changes. Do not regenerate an existing profile over itself, delete uncertain receipts, replay external effects, or copy environments between architectures. The independent Onit pin and installed package list must also be preserved. The dated Pi milestone qualifies the initial locked installation/mock smoke path; this update/rollback procedure and real audio/hardware still require their own verification.

### Synchronize only ME3 in a shared or sparse checkout

The reported Pi parent `HEAD` remained `b7d5c14a25ec55caf4aa0fbcae2f4d22d0a38313` while cached `origin/main` and the deployed ME3 baseline were `057321f150dd0c3df72de086e29b572000f2190b`. ME3 was deliberately an untracked overlay relative to that older parent to preserve ME2. Therefore parent `HEAD` and ordinary `git diff` do not identify the deployed ME3 source reliably.

Run the following **from the parent repository root on the Pi**, after reviewing/preserving any unpublished ME3 source edits. Do not copy the Pi checkout wholesale or restore the entire parent tree. Obtain the exact published commit from the documentation update's report; replace the placeholder before running the restore:

```bash
set -euo pipefail
# Baseline the existing tracked ME2 file contents; stop if this check fails.
mkdir -p me3-voice-agent/var/results
git ls-files -z -- me2-vcm | xargs -0 -r sha256sum > me3-voice-agent/var/results/me2-before-me3-sync.sha256
test -s me3-voice-agent/var/results/me2-before-me3-sync.sha256

git fetch origin main
ME3_PUBLISHED_COMMIT='REPLACE_WITH_REVIEWED_COMMIT_SHA'
git cat-file -e "${ME3_PUBLISHED_COMMIT}^{commit}"
printf 'ME3 source revision: %s\n' "$ME3_PUBLISHED_COMMIT"
# Worktree only: preserve the parent index/HEAD and every other coursework path.
git restore --source="$ME3_PUBLISHED_COMMIT" --worktree -- me3-voice-agent
sha256sum --check --quiet me3-voice-agent/var/results/me2-before-me3-sync.sha256

python3 - "$ME3_PUBLISHED_COMMIT" <<'PYTHON'
from pathlib import Path
import subprocess
import sys

revision = sys.argv[1]
for name in ("me3-voice-agent/deploy/rpi/README.md", "me3-voice-agent/docs/PLAN.md"):
    published = subprocess.check_output(["git", "show", f"{revision}:{name}"])
    if Path(name).read_bytes() != published:
        raise SystemExit(f"Published-file mismatch: {name}")
print("Both documentation files match the exact published revision")
PYTHON
```

Run sequentially and stop on any failure; if no ME2 files are tracked in that checkout, the empty hash list does not establish ME2 preservation—use the existing read-only ME2 inventory instead. Preserve ignored `var/`, credentials and the private VPN directory; never use `git clean`, root reset/pull or delete those files to synchronize. The restore updates only published ME3 paths and does not rearm tasks or start services. Record the exact ME3 source commit separately from unchanged parent `HEAD`; byte comparisons and rechecked ME2 hashes, not parent Git status alone, establish this synchronization.

## Current one-day demo path — 2026-11-06

User will transfer/setup the Pi separately; no Pi SSH address or device access was supplied to this HPC run. The following was the initial operator-run preparation path; the dated verification table above supersedes its software-installation status, while physical/audio steps remain unverified. The new PipeWire peer is software-tested with mocked subprocesses. Keep the final four hours before submission for clean starts, acoustic checks and rollback. Stop feature work before that reserve; exact submission time is still unspecified.

Topology: Pi runs the bridge/Session/Pipecat, CPU Sherpa/Silero, local ASR/TTS MCP subprocesses, configured GPIO/phone adapters, installed Onit worker and native PipeWire peer. Groq provides ASR/LLM inference; university DGX provides TTS. A Pi Spotify Connect receiver, configured separately, renders music. A laptop browser is an observer/inspection UI in native mode. Do not press its Start button while the native peer owns the voice connection. The UI report feed and device controls are not a second microphone connection.

1. Transfer the one canonical ME3 source checkout and explicit inference assets using the existing private SSH workflow. Do not copy ME2, the HPC virtual environment, user recordings, cache trees, or credentials into an archive. Preserve ME2's deployment/services. Before installation run `uname -m`, `getconf GNU_LIBC_VERSION`, `python3 --version`, `wpctl status`, `pw-record --help`, `pw-play --help`, and the existing inventory command below. Confirm GPIO wiring and the actual node names. No hostname is hardcoded.
2. Create the project environment as below and sync `--extra kws --extra pi`; the new `pi` extra only adds dbus-next0.2.3. GPIO libraries remain explicitly installed Pi OS prerequisites, not laptop dependencies. Do not change PipeWire/BlueZ services to force readiness. Use the already selected Sherpa assets. Onit must have its own compatible environment; the tested installation is onit0.1.5a0, clean source commit `3d014cfae043130d956ac042de7142516146e96b`. The dated Pi milestone now verifies installation, scoped imports and worker readiness on one aarch64 target; model execution remains unverified there.

```bash
# ON THE PI, from the transferred ME3 root; no boot services are installed.
python3 -m venv var/cache/venv
var/cache/venv/bin/python -m pip install --cache-dir var/cache/pip uv==0.13.4
UV_PROJECT_ENVIRONMENT=var/cache/venv UV_PYTHON_DOWNLOADS=never var/cache/venv/bin/uv sync --locked --extra kws --extra pi
python3 -m venv var/cache/onit-venv
var/cache/venv/bin/uv pip install --python var/cache/onit-venv/bin/python 'onit @ git+https://github.com/sibyl-oracles/onit@3d014cfae043130d956ac042de7142516146e96b'
mkdir -p var/results var/logs var/scratch/demo-projects
var/cache/onit-venv/bin/python -m pip freeze > var/results/onit-pi-packages.txt
PYTHONPATH=src var/cache/venv/bin/python -B scripts/prepare_demo_profile.py --baseline configs/demo-backends.example.json --asr groq --name pi_demo --pi --enable-inference
PYTHONPATH=src var/cache/venv/bin/python -B scripts/configure_demo_keys.py --output var/cache/demo-models.env
```

Locked application installation and the uv-based separate Onit installation passed on the dated aarch64 target; that does not qualify every subsequent live launch or optional integration. Key entry is masked and owner-only; `ME3_LLM_API_KEY` supplies Groq ASR and LLM, and `ME3_TTS_API_KEY` supplies university TTS. The template has live requests disabled; the setup command explicitly enables only its new private profile. It preserves120B LOW/streaming/2048 through the bridge flags below and removes the old four-request experiment cap only in that new profile. Provider limits and existing tool/operation/context bounds remain. The Qwen ASR baseline is preserved for rollback, not selected silently.

```bash
# Inventory only: no network calls, inference, GPIO or audio opens.
PYTHONPATH=src var/cache/venv/bin/python -B -m me3_voice.preflight --backend-config var/cache/pi_demo-backends.json --secrets-file var/cache/demo-models.env --device-config var/cache/pi_demo-devices.json --onit-python "$PWD/var/cache/onit-venv/bin/python" --onit-output-root "$PWD/var/scratch/demo-projects" > var/results/pi-demo-preflight.json
# Optional phone setup: hidden prompts, no call/SMS or pairing performed.
PYTHONPATH=src var/cache/venv/bin/python -B -m me3_voice.communications --config var/cache/pi_demo-devices.json
```

Phone setup needs the paired tablet MAC and configured authorized demo call/SMS recipient numbers, stored only in `communications.phone_address/call_number/message_number`. Require exactly one PipeWire telephony gateway, no active call, user D-Bus access, and a BlueZ MAP-capable phone with message permission. Read-only checks: `busctl --user list` and `bluetoothctl devices Connected`. No recipient or Bluetooth address belongs in chat/logs. The code never retries Dial/PushMessage; unknown outcome requires checking the phone. Calling reports requested, SMS reports transfer complete; neither asserts answered/delivered. Current transport adapters are mock-tested, not Pi/carrier-tested.

3. Start the Pi bridge in terminal A **only after the separate Pi VPN/network gate passes**. Use the actual deployed profile paths consistently; the earlier preparation example below uses `pi_demo`, while the verified network gate uses the local `pi_live` profile. Omit ME3_MUSIC_CONFIG for the first quiet test. No device opens until a voice peer starts; Pi device configuration explicitly selects real GPIO. If GPIO is unavailable, voice should remain available and actions report unavailable.

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
var/cache/venv/bin/python -B -m me3_voice.voice_bridge \
  --backend-config var/cache/pi_demo-backends.json --secrets-file var/cache/demo-models.env \
  --device-config var/cache/pi_demo-devices.json \
  --onit-python "$PWD/var/cache/onit-venv/bin/python" --onit-primary \
  --onit-reasoning-effort low --onit-max-output-tokens 2048 \
  --onit-output-root "$PWD/var/scratch/demo-projects" --direct-clock \
  --receipt-store var/cache/pi-demo-receipts.sqlite3 --diagnostic-audio \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt
```

No `--onit-max-requests 4`: that was a test-only cap. `/api/health` returning bridge ready/speech lazy is expected before the native peer starts; it is not inference qualification. Original capture/submitted-ASR PCM and outward ASR provider text are retained only for the explicit diagnostic run, bounded to60s. End normally within that window; there is no unattended recording or synchronized speaker reference. Keys are excluded.

4. Terminal B: replace both placeholders with **verified** PipeWire node names, then explicitly open devices. This is the real microphone/speaker command, not a typed inference bypass.

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.pi_audio \
  --capture-node 'VERIFIED_USB_MIC_NODE_NAME' --playback-node 'VERIFIED_USB_SPEAKER_NODE_NAME' --capture
```

The peer captures48k mono PCM16, reuses the bridge resampler/wake/VAD/ASR/Session, and outputs16k mono to PipeWire (which negotiates the physical sink rate). It has one owned capture process, a bounded output queue, and kills its owned playback process on stop before cleanup. Ctrl+C stops/reaps only its owned processes. ACKs are duration-based software estimates—not hardware playback callbacks, audible completion, or AEC evidence. No reference-based AEC is added. Record actual negotiated device formats and acoustic behavior before claiming full duplex. Native mute/start/end UI controls are not implemented; the dashboard observes reports, native peer is controlled in its terminal, and spoken End returns to wake standby.

5. Forward Pi loopback8765 privately through your current SSH alias (or use Pi-local browser). Observe transcript/receipts in the served dashboard; never expose a public endpoint. Exported `me3-dashboard.html` is a snapshot, not native runtime control. Website task example: “Hey Swift, build a website called SwiftDemo with title Swift validation, heading ME3 output check, and a labelled increment counter. Use embedded CSS and JavaScript, responsive layout, and no external requests.” Confirm actual files and the real receipt path. The current Onit worker remains synchronous for general task execution; the mock multi-minute worker's progress/cancel contract is not enabled for real Onit. Do not promise arbitrary concurrent conversation during a long Onit task. Direct supported music/device commands remain local; model-requiring turns can report busy. This is a remaining demo limitation.

For an interactive static preview after inspecting generated HTML/JS, run a disposable loopback server from that **specific new output directory**, then forward8877 privately. This is an operator preview, not deployment or Onit running commands:

```bash
# Replace only with the actual fresh directory recorded in the task receipt.
cd var/scratch/demo-projects/SwiftDemo
python3 -m http.server 8877 --bind 127.0.0.1
```

Click the generated counter, check console/network, and verify dashboard text preview/download bytes. Stop that server with Ctrl+C. A file's existence, an empty directory, or model narration does not pass website acceptance.

6. For music, preserve the laptop browser profile. Configure a separate Pi profile to the **already installed/paired** Spotify Connect receiver; do not start or replace ME2's audio services. Use the existing `music output --config <private-Pi-config> --output-backend spotify_connect --device-name <verified-receiver>` command and pass its file with ME3_MUSIC_CONFIG only for the controlled music phase. Spotify authorization/API keys alone do not create an audio receiver. Stop maps to pause. Verify fresh device state and volume before claiming an effect; no actual playback was changed during this sprint.

### Submission acceptance and rollback

Freeze at least4h before submission. Prepare counts before testing:5 quiet wake-plus-command attempts,5 at a fixed stated moderate music volume,2 playback-only/no-wake exposures of60s, the selected recorded ASR pack, every action in PLAN's canonical ME2 matrix, one fresh voice-triggered website, and3 clean-start multi-turn sequences. Each sequence includes more than4 accepted turns, an800ms deliberate pause, one correction, intentional interruption, one controlled unavailable-service case and successful recovery. Zero wrong/duplicate completed actions and no stale accepted text/audio are required. Record every miss/error; these are small smoke checks, not an accuracy rate/SLA. Do not shorten the endpoint to pass a timing score.

Physically inspect RGB commanded color/on-off/brightness and TM1637 values/countdowns; temperature is a demo setpoint. Timer/alarm expiry currently updates display/state without buzzer/completion speech, and schedules are process-local. Verify actual phone call state and SMS receipt on intended devices; carrier fees require operator approval. Readback/API state is separate from audibility and delivery. Pi startup, physical I/O, acoustic stop/AEC, intended-device wake and all carrier effects remain pending this operator qualification.

Keep a source SHA256 manifest and `uv.lock`; keep original private configurations and regenerate new names for candidates. No archives are generated automatically. Rollback means stop only the ME3 processes you started, select the prior explicit backend/device/music files and known Onit interpreter, then restart manually. Do not replay uncertain actions, reset counters to evade provider limits, restore process-local timers from historical receipts, change wiring, or enable boot services. Existing recordings and ME2 files remain untouched.

## 1. Inventory before installation or device access

On the Pi, from an approved ME3 source checkout (inspect the destination first; never overwrite ~/swift-e2e):

```bash
PYTHONPATH=src python3 -B -m me3_voice.readiness inventory --output var/results/pi-audio-inventory.json
```

This reads system/device listings without recording or changing services. Check aarch64, Python 3.11–3.13, available memory/storage, physical source/sink identities, current audio stack and ownership. The output does not certify supported native rates, power/cooling, stable simultaneous I/O or AEC. Inspect the actual USB adapters and connections. Preserve the ME2 deployment, routing and services. Resolve existing audio ownership before any capture.

## 2. Install only the current wake candidate

These are HPC-verified setup commands; target wheel/dependency compatibility must still be checked on the Pi. Use the same pyproject.toml/uv.lock, without dev/browser or openWakeWord comparison extras:

```bash
python3 -m venv var/cache/venv
var/cache/venv/bin/python -m pip install --cache-dir var/cache/pip uv==0.13.4
UV_PROJECT_ENVIRONMENT=var/cache/venv UV_PYTHON_DOWNLOADS=never var/cache/venv/bin/uv sync --locked --extra kws
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B scripts/fetch_kws_assets.py
```

An offline cross-platform resolver check was run on HPC with `uv sync --locked --dry-run --offline --python-platform aarch64-unknown-linux-gnu --extra kws` using CPython 3.13.9: the lock resolved 99 packages and uv selected an install plan for aarch64. This checks target markers and wheel metadata, not native installation, ABI loading, model inference, Pi memory/CPU, or the target libc. The pinned native audio/ONNX stack includes `sherpa-onnx`, `onnxruntime`, `sentencepiece`, and `soxr`; PyPI publishes ARM64 Linux artifacts for the selected releases, but the target Pi's Python minor version and glibc floor still matter. For example, selected `soxr` and `sentencepiece` wheels require glibc 2.26/2.27 or newer, while `sherpa-onnx` has a manylinux/glibc 2.17 wheel. Check the Pi itself before install with `uname -m; getconf GNU_LIBC_VERSION; python3 --version`, then run the locked install and import/smoke checks below. [sherpa-onnx 2.0.1 ARM64 wheel](https://pypi.org/project/sherpa-onnx/2.0.1/), [soxr 1.0.0 ARM64 wheels](https://pypi.org/project/soxr/1.0.0/), [SentencePiece 0.2.2 ARM64 wheels](https://pypi.org/project/sentencepiece/0.2.2/), [ONNX Runtime 1.24.4 ARM64 wheels](https://pypi.org/project/onnxruntime/1.24.4/).

Explicit setup downloads the selected pretrained inference assets into var/cache; runtime never downloads. Reuse an existing verified keyword file; the fetcher refuses to overwrite it. The default is HEY SWIFT tokenization with unchanged weights. It is a candidate with known synthetic misses, not a qualified reliable wake engine. Do not substitute a phrase or train a model silently.

## 3. Recorded-file test first

Supply a short uncompressed PCM16 WAV under var/recordings/, at its original rate/channels (supported rates are listed by --help; mono/stereo). Each file must be at most 60 seconds. Use separate clips for independent wake attempts: one wake starts a diagnostic session, which allows follow-up speech without repeating the wake word. Silence endpoints default to 640 ms, session inactivity to 10.016 s, cooldown to 1 s. These are provisional policies.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m me3_voice.readiness replay var/recordings/trial.wav --audio-origin recorded --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt --output var/results/wake-vad-trial.json
```

Choose a fresh result path for each retained trial. An explicitly selected output path is atomically replaced, not archived. Reports contain model/keyword hashes, thresholds, native PCM clipping, voiced ranges, wake offsets, endpoint/timeout events, unscored tails and per-call CPU timing. Diagnostic pre-roll is a reported range; this monitor does **not** export utterance PCM to an agent. The existing conversation demo remains separate and mock-only. Pi reports say Pi-local-inference and remain unverified in the UI; a host label is not an acoustic qualification.

Annotate recordings using the schema/method in [BENCHMARK.md](../../docs/BENCHMARK.md#wakevad-report-scoring). Do not infer ground truth from this model's output. Score with:

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.readiness score var/results/wake-vad-trial.json --labels var/recordings/trial-labels.json --output var/results/wake-vad-trial-score.json
```

## 4. PCM stream diagnostic, after device approval

The maintained monitor reads little-endian PCM16 from stdin, never opens a device itself, and publishes a current report about once per second **while input arrives**. Max duration is an input-audio bound (1–600 seconds), not a wall-clock watchdog; a stalled producer can block until Ctrl+C. Upstream capture timestamps, overruns and device identity are not carried by raw PCM. This is not a production capture/health/recovery driver.

After current inventory, explicit real-capture authorization and confirmation that the existing PipeWire version supports these options, a bounded user-only microphone check can use its existing capture utility. Set ME3_CAPTURE_NODE to the **verified physical microphone node.name**, never auto/default or a speaker monitor. For the historically reported 48 kHz device path:

```bash
ME3_CAPTURE_NODE='replace-with-verified-microphone-node.name'
pw-record --target="$ME3_CAPTURE_NODE" --raw --format=s16 --rate=48000 --channels=1 --sample-count=1440000 - | PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m me3_voice.readiness monitor --rate 48000 --channels 1 --max-seconds 30 --audio-origin recorded --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt --output var/results/wake-vad-current.json
```

This example records 30 seconds; it has **not** been run on hardware. Do not run it with the placeholder. Verify the installed utility's --help first, including --sample-count; if unsupported, use an explicitly bounded capture method after review. Use a currently supported native rate/channel configuration. Retain capture stderr under var/logs/ when collecting evidence. Do not add audio services or fall back to another source silently. Nonzero producer exit/overruns must be checked independently; a clean PCM EOF alone cannot establish capture success. The monitor does not persist microphone audio unless --processed-wav is deliberately supplied (model-rate mono only).

The raw/target/rate/channel/sample-count switches follow the current [official PipeWire pw-cat/pw-record manual](https://docs.pipewire.org/page_man_pw-cat_1.html). Target-version validation remains pending. This test performs no speaker playback or AEC; later assistant-only and double-talk tests need a reviewed reference/render path and separate authorization.

## 5. UI on the test host

Run the existing dashboard server locally on the Pi, or use the terminal-file watcher with the approved Pi SSH alias and actual source directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\watch_dashboard.ps1" -SshHost YOUR_PI_ALIAS -RemoteProject /actual/path/to/ME3-VoiceAgent
```

The approved source must already exist; this does not deploy it. Restart the watcher to load updated UI code, then Connect feed and select its printed results folder. Diagnostics → audio report shows the wake/VAD session, event offsets, annotated scores when available, origin and CPU timing. The Assistant view explicitly marks voice/task controls unavailable; capture is started in the test terminal. Report polling (about two seconds) is not audio latency or a waveform. The watcher must read reports on the host running the test: an HPC watcher will not automatically see Pi files.

## Exit evidence

Record the actual Pi environment, model hashes/thresholds, room/distance/gain, CPU/RSS and duration. Test independent Hey Swift attempts, confusable/no-wake audio, immediate commands, follow-ups, pauses, inactivity/re-wake and recovery. Only after stable capture, test actual speaker-only and double-talk with reference-based AEC. Keep raw audio/results under var/ and retain failures. Follow the qualification protocol in BENCHMARK.md. No physical M1 exit or full-duplex claim follows from the replay/stream/controller checks alone.

## 6. Connected application and backend gate

After source/environment checks, the HPC-verified provider-independent application can be exercised on saved audio, before approving devices. The same source and lock are used; no service/unit is installed:

```bash
var/cache/venv/bin/uv lock --check --offline
var/cache/venv/bin/uv pip check --python var/cache/venv/bin/python
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -c 'import aiohttp, mcp, numpy, onnxruntime, pipecat, sentencepiece, sherpa_onnx, soxr; print("runtime imports OK")'
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.cli demo --scenarios --output var/results/pi-scenarios.json --trace var/results/pi-scenarios.jsonl
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m me3_voice.cli replay var/recordings/trial.wav --mock-conversation --audio-origin recorded --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt --pace 1 --output var/results/pi-application.json --trace var/results/pi-application.jsonl
```

The install, imports, scenario runner and replay commands above are verified on HPC/x86_64, **not aarch64**. The resolver dry run is preliminary cross-platform evidence only. Keep --extra kws only for this path; adding the optional openWakeWord comparison can introduce the documented unused-TFLite metadata exception. Report any target wheel/ABI/dependency failure rather than changing the package workflow or claiming compatibility. Record actual Python/architecture, installed versions, cold startup and CPU/RSS; compare available memory/thermal throttling on the Pi 5/4 GB. If `/usr/bin/time` is already present, its verbose resource report can wrap a finite command and stderr can be retained under var/logs/. No package/system/audio-service installation is authorized by this runbook itself beyond the user's agreed project environment scope.

A diagnostic replay remains separate from an application replay. The latter is now connected through our local MCP processes and Session, using mocked ASR/reasoning/TTS/tasks and memory playback. Mock transcription text does not come from the WAV. No Pi speaker or microphone is opened by these commands, and success does not certify physical full duplex. File-clock inactivity is a replay policy; a future continuous device source must drive the existing monotonic Session.tick hook and capture-health/overrun handling.

Use configs/backends.example.json only as an empty schema reference; actual local files belong under var/cache/. Backend roles default to mocks without a config file. Before selecting a commercial or shared HPC service obtain the exact documented endpoint/protocol/authentication/model/audio/streaming/cancellation/concurrency/timeout/quota contract listed in ARCHITECTURE.md. Explicit access and spending authorization are still required. Documented adapters are implemented with offline validation; the class DGX3 adapters now follow inspected service schemas, but authenticated inference remains untested. Our ASR/TTS MCP services remain on the Pi under the written requirement; any change or auxiliary audio transport retains the professor-interpretation gate. Credentials remain server-side and never enter the dashboard snapshot.

## Shared LED and TM1637 action adapter

ME3 now has one bridge-lifetime `DeviceActions` owner for the RGB LED and four-digit display. Its default remains explicit mock output. The optional Pi adapter preserves the read-only ME2 wiring: RGB red/green/blue BCM 17/27/22 and TM1637 CLK/DIO BCM 23/24. The adapter requires the Raspberry Pi OS packages `python3-gpiozero` and `python3-lgpio`; it imports them only after a private device config explicitly selects `backend: "pi"`. Pi model and GPIO initialization are checked before actions. If the Pi or packages are unavailable, the voice runtime remains usable and physical actions report unavailable; it never falls back to a simulated output. This adapter has **not** been installed or run on a Pi during this work.

Before selecting `backend: "pi"`, perform these read-only checks on the intended Pi:

```bash
uname -m
getconf GNU_LIBC_VERSION
python3 --version
tr -d '\000' </proc/device-tree/model
python3 -c 'import importlib.util; print({name: importlib.util.find_spec(name) is not None for name in ("gpiozero", "lgpio")})'
```

Expected architecture is `aarch64`, model is Raspberry Pi 5, and both modules must be visible to the environment that runs ME3. These checks open no GPIO. Do not copy the ME2 environment or deployment. If the approved Pi environment needs Debian's GPIO packages, the documented setup is `sudo apt install python3-gpiozero python3-lgpio` followed by a separate ME3 venv with system package visibility; do not install or run this on the HPC/laptop as a substitute Pi check:

```bash
python3 -m venv --system-site-packages var/cache/pi-venv
var/cache/pi-venv/bin/python -m pip install --cache-dir var/cache/pip uv==0.13.4
UV_PROJECT_ENVIRONMENT=var/cache/pi-venv UV_PYTHON_DOWNLOADS=never var/cache/pi-venv/bin/uv sync --locked --extra kws
var/cache/pi-venv/bin/python -c 'import gpiozero, lgpio; print("Pi GPIO imports OK")'
```

If either import fails, leave the target on mock for software-only use or retain explicit `pi` and expect clear unavailable action receipts. Never change it to mock to conceal a failed physical request.

After read-only Pi inventory, inspect the existing Python environment and confirm the two packages are importable from the environment that runs ME3. If an environment cannot see the OS packages, resolve that through the documented Pi environment workflow before starting ME3; do not silently install an unpinned pip GPIO dependency into the locked application environment. Hardware setup and the following bridge command require the separate, authorized Pi action test:

```bash
sudo apt install python3-gpiozero python3-lgpio
cp configs/devices.example.json var/cache/devices.json
var/cache/venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("var/cache/devices.json")
config = json.loads(path.read_text())
config["backend"] = "pi"
path.write_text(json.dumps(config, indent=2) + "\n")
PY
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B -m me3_voice.voice_bridge \
  --device-config var/cache/devices.json \
  --backend-config var/cache/voice-backends.json \
  --secrets-file var/cache/voice-backends.env \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt
```

Before using that command, use the Pi-compatible environment in place of `var/cache/venv` if that is where the locked dependencies were installed, and check that GPIO pins 17/27/22/23/24 match the existing safe wiring. Config selection is explicit; no pin is guessed. The adapter starts the LED off and display blank, and closes both on bridge shutdown. Stop the bridge with Ctrl+C after the authorized test; an end-of-conversation action does not close the shared service. No service/boot unit is installed.

Laptop/HPC software-only checks (no GPIO import/action) use the default mock target:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m pytest -q tests/test_devices.py tests/test_session.py --basetemp=var/scratch/pytest-device-actions
```

The live Assistant RGB LED module widget provides On/Off, a native color picker and a 0–100% brightness slider with 1% steps only while a voice session is connected. ME2's read-only software config describes a KY-016/common RGB LED via `gpiozero.RGBLED` + `lgpio`, with configured BCM red/green/blue 17/27/22 and `active_high: true`; it is not proof of the physical unit's exact variant or wiring. The manufacturer's [Joy-IT KY-016 product page](https://joy-it.net/en/products/COM-KY016RGBLED) and [Pi manual](https://www.joy-it.net/files/files/Produkte/COM-KY016-RGBLED/COM-KY016RGBLED_Manual_2024-05-02.pdf) describe one common-cathode product but show BCM red/green/blue 25/24/23, a different map. Before Pi mode, inspect the user's board labels and existing leads and resolve this mismatch; do not change wiring based only on either config or a generic KY-016 name. ME2's action mapping only used red/green/blue and 20/60/100%; ME3 supports common named colors, six-digit `#RRGGBB`, and 0–100% PWM duty. The UI percent is not photometric calibration. Mock mode labels each result as simulation; Pi-unavailable mode disables the controls and says the Pi is not connected. These controls submit through the active Session and task executor, not through browser GPIO or a separate UI state store. A repeated UI command ID is deduplicated for the bridge lifetime.

After separate approval for a physical output trial, use this bounded sequence with the Pi attended and visible: start ME3 with the explicit Pi config; confirm Diagnostics reports the Pi adapter ready but not physically verified; voice-command red, then an arbitrary named color (for example orange) and 37% brightness; use the widget color picker/slider to select a different color/percentage; turn the LED off; compare each software receipt with direct visual observation; ask for a 10-second timer and verify only the display countdown; set the 22°C demo display and confirm the countdown returns; cancel the schedule; end the session and stop the bridge. Record requested versus visually observed color/brightness, not as sensor telemetry. Report any mismatch as unknown and inspect the wiring/device before retrying. Do not test calls/SMS, unapproved live services, or claim sensor-confirmed state. If Pi initialization fails, confirm voice still works and a physical action says the Pi is not connected without a simulation result.

Display ownership and receipts:

- One timer or alarm owns the display schedule. A newly accepted timer/alarm explicitly replaces the prior schedule; `cancel timer` / `cancel alarm` maps to `cancel_schedule` and clears it.
- A temperature setpoint appears briefly as four digits for three seconds, then the active timer/alarm display is restored. It never replaces/cancels the schedule. Time/weather/reminder requests do not take display ownership.
- Timers use a monotonic software deadline and update the display. At zero, the display shows `0000`; there is no buzzer or audible alert. Alarms display their next local occurrence and report “time reached”; there is no buzzer.
- A device-service process restart cancels its in-memory timer/alarm. Reminders can persist in the configured JSON store at `var/cache/device-state/reminders.json`.
- The dashboard’s countdown is a visualization of the receipt deadline. `Accepted` means the software output call returned; it is not sensor verification. The LED has no readback sensor, and no task is labelled Pi-verified by this adapter.

For future weather use, an explicit config may set `weather.enabled` and provide latitude/longitude/IANA timezone. It makes on-demand GET requests to Open-Meteo's documented current-weather endpoint; this work used only a mocked response. See [Open-Meteo API parameters](https://open-meteo.com/en/docs) and [service terms](https://open-meteo.com/en/terms): the free API is limited to non-commercial use, its weather data is CC BY 4.0 and needs attribution. Weather remains disabled in the tracked config. ME2's optional runtime includes Bluetooth telephony and a fixed-message OBEX MAP SMS service. ME3 keeps both unavailable until a project-specific demo recipient and paired phone are configured and real-call/message testing is separately authorized; do not copy ME2's concrete recipient/device defaults. Dialing and Bluetooth transfer receipts are not proof of a connected call or carrier delivery.

## Pi acceptance checklist

Natural-interaction checks must include wake immediately followed by a command, short and long mid-sentence pauses, follow-ups, correction, speech interruption, slow/disconnected services and recovery. The mock suite exposes an 800 ms pause splitting at the provisional 640 ms silence threshold; --endpoint-ms 1024 preserves that scripted pause at the cost of 384 ms extra endpoint wait. Neither profile is Pi-qualified. Check local device stop/flush before remote cancellation completes, and collect physical speech-end-to-first-audible-response, interruption-to-audio-stop and verified task-completion measurements. File-frame/memory-sink proxies cannot satisfy those measurements.

- Read-only current inventory, connection topology, owners/routing, power/cooling and native capabilities recorded; ME2 unchanged. No unsupported source fallback.
- Same lock installs on actual aarch64/Python and imports/inference succeed within measured RAM/CPU/thermal limits. Stdio discovery and cancellation/reuse work on target.
- Saved-file connected scenarios pass with honest mock/CPU labels; actual microphone capture then separately approved and checked for clipping, continuity, loss/overruns and recovery.
- Independent real-speaker Hey Swift positives, negatives/confusables, immediate commands, follow-ups, silence timeout/re-wake and held-out annotations collected. Keep the existing synthetic misses visible.
- Simultaneous real capture/playback, reference-based AEC, assistant-only and double-talk verified before claiming natural interruptible acoustics. AEC configuration is still review-only, not installed or tested here.
- Real playback stop/flush, preserved incoming speech, stale-result rejection, confirmed-history semantics and no duplicate completed actions checked on target. MemorySink cannot satisfy this gate.
- Dashboard viewing tested on the actual demo laptop/display and chosen Pi report transport. A report connection is never a microphone/runtime/Pi-health connection.

Connected --mock-conversation replay now uses PipecatAudioInput for continuous native PCM ingestion. It does not require or select a Pipecat LocalAudioTransport and opens no device. Keep the same recorded-file acceptance commands; check input_pipeline=pipecat-streaming-pcm and acknowledged packet counts in the report. Live capture and playback still need one approved audio owner, explicit format/routing, bounded device buffers/overrun reporting and verified stop/flush. Do not enable a second Pipecat VAD/session policy beside the existing controller or bypass our speech MCP servers with provider services.

## Updated service and pause-policy gate

remote.py now has offline-validated Deepgram listen/speak v1, Groq file-ASR (streaming=false) and chat-completions SSE adapters. The class DGX3 speech profile has completed authenticated HPC trials; no separate reasoning backend or Pi deployment is configured. --conversation is the preferred replay flag; --mock-conversation remains a compatibility alias. Default providers remain mocks. Keep our MCP servers on Pi; only explicitly selected authentication variables belong in their environment. Read ARCHITECTURE.md before enabling allow_live/request_limit, and obtain a separate actual credit/monetary ceiling. Counters are not billing controls. Do not print keys or put them in command arguments/UI files.

--turn-policy smart-turn optionally uses the bundled CPU ONNX model with one inference thread; it awaits analysis after candidate silence and has a bounded fallback. This was tested on HPC x86_64, not aarch64. Compare capture continuity/overruns, endpoint errors, CPU/RSS/temperature and response delay on Pi before enabling it. No new model training/download is needed for the pinned wheel, and PIPECAT_SMART_TURN_LOG_DATA must stay disabled. AEC remains review-only and no simultaneous capture/playback reliability is established.

The dashboard theme selector and reported-state icon are presentation only. A live report folder is not a live microphone or action connection; no session start/stop backend exists. Verify actual laptop theme/contrast, file permissions and projector readability separately from Pi audio acceptance.

### Class speech-service transition

Use README's canonical dgx3 configure / inspect / replay commands from the target checkout only after Pi access and setup approval. Re-enter keys through its masked terminal; do not copy secrets into deployment archives or UI files. Confirm VPN/DILNET routing to 10.153.60.233, authorized model/voice access and limits. No fresh Pi network or aarch64 test was performed. The owner-only secret-file checks are Linux-specific and require correct permissions on that host. Private HTTP is an explicit class-endpoint exception, not encrypted transport. Verify the 24-to-16 kHz TTS conversion on aarch64 and actual device-rate conversion separately. ASR is final-only; reasoning is mock by default. These service adapters do not qualify microphone/speaker routing, AEC, double-talk or audible cancellation.

Pipecat ingress startup now advertises the native source rate; a 48 kHz stereo CPU regression verifies one conversion to 16 kHz mono. This is not a fresh Pi device-rate inventory or an aarch64 result.

## Architecture audit transition gate — 2026-11-04

The current tested I/O path is a laptop browser connected to an HPC bridge. Native Pi capture/playback and reference-based AEC are not implemented. Keep “Install AI agent harness … with MCP support on RPi” and “you build and run your MCP servers (on RPi)” explicit; ask whether the verbal “mostly VAD” guidance permits a remote harness before changing that boundary. Two custom speech MCP subprocesses are implemented; tasks currently use typed in-process adapters, not a separate task MCP server.

Before any approved Pi install: verify Python/glibc/aarch64, Onit's separate Python compatibility, existing audio service ownership, USB source/sink and GPIO wiring against the canonical PLAN compatibility matrix. Keep mock mode selected for software acceptance; do not add GPIO packages to the laptop environment. Check shutdown and display restoration with mock devices before controlled physical LED/display tests. Timer/alarm completion speech is still absent; schedules are not restart-persistent.

The first later acoustic sequence is one variable at a time: same-session transcription with speakers silent; factual/task/positive-content TTS listening; intentional interruption with acoustic recording; music-only baseline then command-over-music with ducking disabled/enabled; finally nearby conversation and engagement expiry. Fixed placement and volume are required for comparisons. Opt-in bridge `--diagnostic-audio` captures received/ASR PCM and result IDs for 60 seconds, after browser Start. It has no synchronized speaker reference and cannot establish Pi AEC. Never enable it unattended or as a boot service.

### Optional receipt recovery before any physical trial

After approved Pi installation, a user-started ME3 bridge may opt in with `--receipt-store var/cache/task-receipts.sqlite3`. Use an ME3-owned private local regular file; do not point at ME2 data or share one SQLite file across hosts. First run the disposable-process mock tests in tests/test_receipts.py. Never test recovery by intentionally crashing a physical action here without a separately controlled hardware plan.

The ledger protects exact request IDs, not newly spoken requests. Pending intents remain Unknown and are never replayed automatically. Completed records are historical receipts; they do not prove current GPIO/player state. Timers/alarms are process-local and **not restored** after process death. Inspect with `python -B -m me3_voice.receipts inspect --store var/cache/task-receipts.sqlite3`; reconcile actual target state before a new effect. Do not delete/archive the ledger merely to bypass uncertain outcomes or a capacity limit. SQLite/fsync behavior, power-loss resilience and Pi device state require separate validation.
