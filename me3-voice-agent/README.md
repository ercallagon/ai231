# ME3 — Swift: Full-Duplex Voice AI Agent

Swift is a Raspberry Pi 5 (4 GB) voice assistant with local wake/VAD, our own ASR/TTS MCP servers, configurable remote inference, an Onit task worker, and a browser interface. This is the actual runnable application, not a demo shell. **Full-duplex acoustic reliability and fresh Pi installation remain pending physical qualification.**

## Install and run from a fresh checkout

Use `ai231/me3-voice-agent/` as the checkout directory. The application is a subdirectory of the `ercallagon/ai231` coursework repository:

```bash
git clone --filter=blob:none --sparse https://github.com/ercallagon/ai231.git ~/ai231
git -C ~/ai231 sparse-checkout set me3-voice-agent
cd ~/ai231/me3-voice-agent
bash scripts/setup.sh
bash scripts/swift.sh check
bash scripts/swift.sh demo
```

Linux, Python 3.11–3.13 and `python3-venv` are required. HPC/x86_64 Python 3.13 is tested; Raspberry Pi OS 64-bit on Pi 5 is the target, not yet an installation claim. Setup installs the editable application with `uv==0.13.4` and `uv.lock` into `var/cache/venv`. It downloads dependencies but opens no devices and uses no model API. Add `--dev` for tests or `--pi` for optional Bluetooth support. Do not copy an HPC virtual environment to the Pi. Windows users can view the browser UI; native Windows installation is not qualified.

The `demo` command uses deterministic mock providers, memory playback and mock actions through our real Session/MCP path. It requires no account, university network, wake assets or GPIO. It does not use your microphone. To view the actual dashboard separately:

```bash
bash scripts/swift.sh dashboard --port 8765
```

Open `http://127.0.0.1:8765/`. This dashboard-only command does not start a voice runtime. Stop it with Ctrl+C before starting the bridge on the same port.

**For real voice, model assets, credentials, hardware and Onit:** follow the [fresh-install and launch guide](deploy/rpi/README.md#fresh-checkout-installation-and-operation), then the current demo procedure below it. **University TTS requires a route from the Pi itself (university VPN/DILNET) plus an authorized key; your laptop’s OpenVPN connection is not inherited by the Pi.** See the [network checklist](deploy/rpi/README.md#network-access-on-the-pi). Groq ASR/LLM requires your own account/key and is subject to its provider limits. Spotify requires its own authorization, Premium and an actual playback receiver. Phone actions require explicitly configured recipients, paired compatible devices and permissions. Weather uses Open-Meteo without a key; its service terms apply. Optional integrations are not required for mock startup.

[Architecture](docs/ARCHITECTURE.md) · [Actual results and open issues](docs/PLAN.md) · [Hardware, acceptance, troubleshooting, update and rollback](deploy/rpi/README.md)

The Python package/module remains `me3_voice`; moving the checkout does not rename it. Run commands from the project root. All credentials, models, logs, recordings and outputs stay in ignored `var/`; source code never implicitly reads ME2 or another coursework checkout. Preserve the original recordings on the development machine rather than publishing them.

## Intended experience

Say a wake phrase once to start a session. Speak naturally, ask follow-ups, execute real tasks, and interrupt spoken answers. Active reasoning and playback remain interruptible without a wall-clock turn cutoff. The browser bridge defaults to Follow-up conversation: after confirmed playback finishes, it accepts follow-ups for 7 seconds, or 18 seconds for an explicit supported clarification. Select Single request to return to wake standby after one response, or Extended conversation for a 20-second idle window. End explicitly with “That’s all,” “Goodbye,” or “End conversation,” or use End in the UI. The microphone transport remains active for wake detection in standby; ordinary speech is no longer submitted to ASR. These initial windows need laptop/Pi validation. CLI replay retains its existing inactivity policy unless adaptive mode is explicitly selected in Session.

## Requirements

- Run an agent harness with MCP support on our Raspberry Pi 5 with 4 GB RAM.
- Build and run our own ASR and TTS MCP servers on the Pi; commercial model APIs are allowed.
- Run local VAD on the Pi.
- Demonstrate interruptible conversation and real task execution.
- Provide a reproducible latency benchmark.

User clarification (2026-10-27): integrate pretrained/open-weight models or services; no training from scratch. Demonstrate open-ended agent tasks through tools, beyond a fixed light/timer command vocabulary. The voice/task-harness split and model candidates are recorded in ARCHITECTURE.md. The opt-in Onit worker can produce isolated draft artifacts and, when its sandbox output root is explicitly enabled, create a new static website project with bounded files. It cannot run, test or deploy that project; broader task reliability and Pi deployment remain pending.

Execution priorities: (1) qualify existing hardware early, (2) establish reliable wake-word, audio I/O and echo cancellation, (3) integrate onit as the preferred task harness and choose mostly-free components that meet correctness/reliability requirements, then optimize latency and model quality. Hermes is a fallback. ElevenLabs models remain excluded. Previously proposed paid providers are comparison options, not the default stack. Use only purposeful MCP integrations; server count is not a project objective. See PLAN.md for readiness gates and ARCHITECTURE.md for component decisions.

Milestones now follow that order: **M1 = hardware/audio front end (AEC, preprocessing, wake word and VAD); M2 = onit harness and speech MCP integration**. Existing HPC MCP mock tests are preliminary M2 evidence, not a reason to skip M1 hardware qualification.

The supplied specification contains the incomplete sentence “Note that this is .”; its missing condition remains unknown. Wake-word session entry is our agreed design, not a requirement quoted from the original specification.

## Read in this order

1. [Project rules](AGENTS.md)
2. [Architecture and decisions](docs/ARCHITECTURE.md)
3. [Milestones and current status](docs/PLAN.md)
4. [Benchmark contract](docs/BENCHMARK.md)
5. [SSH agent handoff](docs/SSH_HANDOFF.md)

## Directory contract

```text
src/me3_voice/       Python application, audio, session, MCP servers, providers, tools
src/me3_voice/ui/    Bundled dashboard HTML, CSS and JavaScript; no separate source tree
tests/              Focused automated tests
configs/            Versioned configuration examples; no secrets
scripts/            Maintained setup, run, verification, benchmark entry points
deploy/rpi/         Pi deployment instructions and service templates
docs/               Canonical design, plan, benchmark and handoff documents
docs/references/    Supplied reference material; interpretation stays in canonical docs
var/                Ignored experiments, logs, recordings, results, caches and scratch work
```

Create implementation directories only when they contain useful files. Do not add placeholder trees, duplicate source bundles, or timestamped project copies. The Pi wake/VAD test runbook is prepared in deploy/rpi/README.md; production hardware transports and service templates remain pending Pi inventory.

Intermediate files belong in var/scratch/. Exploratory trials belong in var/experiments/<number>-<topic>/ with a short README.md describing the question, configuration, commands and findings. Other output uses var/logs/, var/recordings/, var/results/, var/cache/ and var/exports/. Promote reusable code to src/me3_voice/ or scripts/, regression checks to tests/, and selected sanitized reports to docs/evidence/. See AGENTS.md for cleanup and retention rules.

## Development and Git

Use this repository as the source of truth. Local and HPC work prepare reviewable changes; separate Pi deployment validates the implementation on hardware. Use focused `codex/<topic>` branches. Never sync by copying an entire working directory over another checkout.

The user's Codex SSH agent works in the HPC sandbox, which also has an onit agent. HPC is the development environment; Raspberry Pi 5 with 4 GB RAM is the deployment target. Pi access is separate. HPC tests do not establish Pi compatibility or physical audio behavior.

Commits are deferred at the user's request until more structure is established. Continue setup without staging or committing; an initial commit is not a prerequisite for the Pi inventory.

Stage explicit paths, inspect the staged diff, and scan for secrets, recordings, generated output and oversized files before committing. Push and external publication require user authorization. No GitHub remote has been configured by this planning baseline.

## Verified HPC development workflow

Run from this directory, using Python 3.11–3.13. Tested here with CPython 3.13.9 on Linux x86_64. All dependencies, caches and generated results stay under var/. Package downloads require network permission; subprocess MCP tests require working local IPC. This Codex sandbox blocks AnyIO thread IPC, so those checks were run with command escalation, without opening ports.

```bash
python3 -m venv var/cache/venv
var/cache/venv/bin/python -m pip install --cache-dir var/cache/pip uv==0.13.4
UV_PROJECT_ENVIRONMENT=var/cache/venv UV_PYTHON_DOWNLOADS=never var/cache/venv/bin/uv sync --locked --extra dev
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m pytest --basetemp=var/scratch/pytest
var/cache/venv/bin/ruff check src tests scripts
var/cache/venv/bin/ruff format --check src tests scripts
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m me3_voice.cli demo
```

`demo` starts our ASR and TTS MCP servers as local stdio subprocesses through a Pipecat bridge. ASR returns a configured mock transcript, the mock LLM retains follow-ups, and mock TTS emits tones into a memory sink. It never opens audio devices or makes model API calls. `uv.lock` pins the full dependency graph; it remains provisional for Pi/aarch64.

Replay your own short uncompressed mono PCM16 16 kHz WAV, or generate an explicitly synthetic fixture with the HPC host's existing espeak:

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B scripts/make_speech_fixture.py
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m me3_voice.cli replay var/recordings/synthetic-speech.wav --mock-conversation
```

Without explicit wake assets this example uses a clearly labelled injected wake at frame 0; configured assets enable actual detection. Replay runs CPU Silero, preserves pre-roll and endpoints on sustained silence. `--mock-conversation` now feeds continuous native PCM through Pipecat into the existing frontend, Session and our MCP servers. Packet acknowledgments wait for ingestion, not answer completion, so subsequent input can interrupt output. The ordered Pipecat compatibility bridge remains used by demo. Neither path opens audio hardware; Pipecat device/output transport and AEC remain separate work. No real-speaker recordings were supplied in this run. Generated speech does not validate wake accuracy, real ASR, echo handling or acoustic interruption. See [current evidence and remaining gates](docs/PLAN.md).

## Connected provider-independent checks

No ASR/TTS host, account or LLM provider is required. The existing MCP servers participate in the application path; all recognition/reasoning/task/audio-output behavior remains explicitly mock. No hardware is opened.

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.cli demo --scenarios --stress-turns 25 --stress-concurrency 4 --output var/results/scenarios.json --trace var/results/scenarios.jsonl
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m me3_voice.cli replay var/recordings/synthetic-speech.wav --mock-conversation --audio-origin synthetic --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt --mock-transcripts "set mock light on" --pace 1 --transcript --output var/results/application-trial.json --trace var/results/application-trial.jsonl
```

Use fresh trace paths. Scenario results cover follow-up/exit, reasoning/synthesis/playback interruption, completed mock-action idempotency/correction and controlled provider failure/recovery. They also run 25 post-ASR scripted turns in each of four concurrent sessions (100 total), checking drained bounded playback queues, task effect counts, cleanup and per-stage monotonic software timings. The text-injected stress path uses placeholder PCM plus deterministic ASR/LLM/TTS mocks; it does not measure speech recognition, provider/network latency or acoustic response. The recorded path uses the same preprocessing/wake/VAD components, starts incremental ASR through our MCP server at speech onset, and commits after endpointing. When ASR is mock, its scripted text is independent of the WAV's words and does not measure recognition accuracy. External adapters have not been tested against live services. Omitting --mock-conversation runs frontend diagnostics only. Pace=0 is unpaced scheduling stress; pace=1 schedules against a monotonic sample timeline and records lateness, not acoustic timing.

Interfaces and the external-backend checklist are in [ARCHITECTURE.md](docs/ARCHITECTURE.md#backend-configuration-and-external-service-gate). Optional --backend-config reads asr/llm/tts JSON descriptors. The tracked [empty example](configs/backends.example.json) must be filled deliberately; store actual local configuration under var/cache/. Documented Deepgram listen/speak v1, Groq file-ASR and chat-completions SSE adapters are implemented and validated offline. Live use requires an explicit documented contract, authorization and credentials; `request_limit: null` means no ME3 request-count ceiling for an explicitly authorized service. Operational timeouts, cancellation and concurrency still apply. Unknown contracts fail without network access. Use --conversation (the older --mock-conversation alias remains supported). A shared HPC service is not assumed OpenAI-compatible. Do not put credentials in endpoints, reports, UI files or command-line arguments.

ME2 non-music actions use the existing typed task executor and one shared device-action service: RGB light on/off/color/brightness; a display-only temperature setpoint; timer/alarm scheduling and cancellation; system-local time; reminders; and optional current weather. ME2's checked-in software configuration labels its device KY-016/common RGB and maps red/green/blue to BCM 17/27/22 using `gpiozero.RGBLED` and `lgpio`; this is not a physical inventory of the user's module. KY-016 vendor variants can have different pin mappings, so verify the board markings and wiring before selecting the Pi adapter. ME3 supports named colors plus any six-digit RGB hex and 0–100% PWM duty in 1% UI steps; the percentage is not photometrically calibrated. Voice tool calls and the Assistant's RGB color/brightness controls use the same Session/task executor and adapter; task receipts drive both spoken acknowledgments and UI state. The default is an explicitly labelled mock. Select a Pi target only through the private device config. If Pi GPIO initialization fails, the bridge remains usable, reports “Raspberry Pi is not connected” for physical actions, and does not substitute simulation. A successful GPIO write is still not sensor verification. Timers use monotonic time, a new timer/alarm replaces the current schedule, and the temperature display overlays it for three seconds before restoration. There is no buzzer, measured-temperature sensor, HVAC control, or LED readback sensor. Weather is unavailable unless an explicit location/provider config enables on-demand Open-Meteo. Persistent reminders require an explicit project-local reminder path. The Pi adapter preserves ME2's configured BCM mapping but has not been run on a Pi. ME2's optional Bluetooth call and SMS services remain unavailable in ME3 until separately configured and authorized. See the compatibility matrix in PLAN.md for parameters, dependencies, wiring, coverage and verification.

To select device settings explicitly, copy `configs/devices.example.json` to `var/cache/devices.json` and pass `--device-config var/cache/devices.json` to `me3_voice.voice_bridge`. The example remains mock-only with weather disabled. Do not change its backend to `pi` until the documented Pi wiring and acceptance checks are approved and verified. In particular, a Pi task receipt can say a GPIO/display command was accepted, never that an LED's physical output was sensor-confirmed.

Onit can be selected as the primary general-reasoning harness in replay or the loopback bridge using the exact configured DGX Qwen chat contract. It runs in the separate Onit Python environment and is prewarmed before voice readiness. Existing direct music intents still route through ME3's `MusicController`. The confined worker supports draft artifacts, an explicitly configured new project directory, and small static websites written only inside that new directory. Project output is opt-in; existing paths are never overwritten. Website files are reported as created but are not run, browser-tested or deployed. Optional Tavily search and GitHub read-only tools are separately gated. Shell, repository writes, broad filesystem access, voice approvals and hardware actions remain unavailable. The Onit task tools are not the ASR/TTS MCP servers. See PLAN.md for exact checks and remaining browser/Pi gates. The Codex assistant developing this repository is separate from the application.

For a CPU-only preview through the existing application, replay the synthetic fixture with `--mock-transcripts "simulate timer 10 seconds" --transcript` and fresh `--output`/`--trace` paths under `var/results/`. This exercises the real frontend and our speech MCP pair with mock reasoning/task/TTS. The saved `ui-transcript.json` carries the task receipt into the dashboard; a later voice session replaces that current view. Other exact mock-provider phrases include `simulate color blue`, `simulate brightness 20`, `simulate temperature 22 degrees`, `simulate alarm 6:00 am`, `simulate reminder study`, `simulate reminders`, `simulate time`, and `simulate weather`. The live Assistant RGB widget also offers a 24-bit color picker and 0–100% slider in 1% increments. Voice direct routing includes explicit color names/hex and numeric brightness; the widget posts through the same task path. An external tool-calling LLM can request the same typed mock actions; live requests are subject to provider/model behavior, and the authorized DGX profile has no ME3 request-count ceiling. These are not physical or general-task completions.

## Guided recordings for repeatable debugging

Use **Diagnostics → Audio → Guided recordings** in the live bridge, or launch the lightweight recorder alone:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.recording_lab serve --port 8765
```

Open `http://127.0.0.1:8765/recording-lab` through the existing VS Code port forward. Start with the six-clip **Core transcription** set. The page provides countdowns and phrases, automatically records/uploads each take, supports review/retakes and resume, and saves directly under `var/recordings/guided/`; no manual transfer is needed. Keep speakers silent and end any live voice conversation first. No model or task executes during capture. See [the recording and bounded replay workflow](docs/BENCHMARK.md#guided-reusable-recording-corpus) for developer evaluation and its limits.

## Laptop AEC recording check

The loopback voice bridge and local dashboard serve the existing **Diagnostics → Audio → Open laptop AEC check** page at `/audio-probe`. It requests one browser microphone stream only after Start is clicked, plays a user-selected local reference file, and downloads a WebM recording plus JSON settings/phase timestamps; it does not upload either file. Keep the live voice conversation ended while using the probe. Run the same source at fixed speaker level and placement with requested AEC off and on. Check `reportedAEC` in each JSON: the browser may not honor an ideal constraint. The guided phases are playback with silence, speech over playback, then speech with playback stopped. The timestamps are browser events, not sample-aligned acoustic loopback, so analysis must compare intelligibility and false turns rather than claim ERLE. Use a song file for the music-leak test; a separate live Swift conversation with Spotify off is needed to observe possible self-capture during actual TTS. This qualifies only the laptop/browser path.

### Bridge startup and recovery

The voice bridge is a foreground process bound to loopback. Its host may be `127.0.0.1` or `localhost` only; `--port` selects the remote port (default 8765). It never binds to a public interface. Use the same host/port for the browser tunnel and the bridge terminal:

```bash
ME3_MUSIC_CONFIG=var/cache/music.json \
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
var/cache/venv/bin/python -B -m me3_voice.voice_bridge \
  --host 127.0.0.1 --port 8765 \
  --backend-config var/cache/voice-backends.json \
  --secrets-file var/cache/voice-backends.env \
  --llm-secrets-file var/cache/dgx3-llm.env \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt
```

For opt-in durable device/music dispatch receipts, append `--receipt-store var/cache/task-receipts.sqlite3` to that launch. Pending records are never automatically replayed. Recovered receipts are historical evidence; they do not restore chat, current device state or timers. Inspect without dispatch using `var/cache/venv/bin/python -B -m me3_voice.receipts inspect --store var/cache/task-receipts.sqlite3`. Keep this ME3-only store private and separate per deployment.

Run that in an **HPC terminal** from the ME3 root. Readiness appears as `Swift voice bridge: http://127.0.0.1:8765 (loopback only)`. Check the same host on HPC with `curl -fsS http://127.0.0.1:8765/api/health`; expect `bridge: ready`, `voice_session: idle`, and initially `speech_services: lazy`. In VS Code Remote-SSH, forward remote port 8765 once. Open the exact local address shown in the Ports panel (for example `http://127.0.0.1:8765/`; VS Code may choose another local port). On the **local browser machine**, request `<that-local-address>/api/health`. A timeout/connection refusal points to the bridge or tunnel; a JSON 404/non-ME3 response points to the wrong forwarded service. Only after health returns the Swift JSON should you click **Start conversation**; that check happens before browser microphone permission is requested.

The browser reports permission denial, missing input device, device busy/driver errors and bridge/WebSocket errors separately. If health is ready but microphone setup fails, check the site microphone permission and selected input in the browser, then close other apps using the device. If health works on HPC but not at the local address, check that the VS Code Ports row forwards remote 8765 to local loopback; do not remove/re-add it routinely. If the bridge reports an active voice session, use that tab or end it there; the service deliberately permits only one active session. If a WebSocket drops, the page checks bridge health with bounded backoff and reports when it is reachable. It does **not** resume the old session or replay microphone audio/tasks; click Start to create a fresh session. Stop only the bridge you started with Ctrl+C. On address-in-use errors, inspect the listener and stop only a Swift process you own, or select another `--port`; ME3 never kills a process automatically.

## M1 readiness commands

Read-only inventory needs only the source tree and Python; it opens no device stream:

```bash
PYTHONPATH=src python3 -B -m me3_voice.readiness inventory --output var/results/audio-inventory.json
```

Optional wake backend setup uses the same project/environment:

```bash
UV_PROJECT_ENVIRONMENT=var/cache/venv UV_PYTHON_DOWNLOADS=never var/cache/venv/bin/uv sync --locked --extra dev --extra wake
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B scripts/fetch_wake_assets.py --phrase hey_jarvis
```

Hey Jarvis is a compatibility-test phrase, not the selected user phrase. Pretrained assets carry upstream CC BY-NC-SA 4.0 terms. The released openWakeWord package declares TFLite even for ONNX; the versioned uv override omits unused TFLite on Python 3.12+. Actual ONNX inference is locally tested, but a generic pip compatibility check still flags that upstream dependency. Review this exception on the target Pi; do not silently use another dependency workflow or claim full compatibility.

`python -B -m me3_voice.readiness replay --help` documents native-rate PCM16 WAV conversion, clipping diagnostics, Silero scoring and optional explicit wake-model paths. `aec-config --help` produces a review artifact for explicit physical PipeWire node names; it never installs or applies it. A generated AEC config and synthetic replay are not evidence of real echo cancellation. PLAN.md records the actual commands, findings and remaining Pi gates.

The exact **Hey Swift** phrase has a pretrained configurable Sherpa-ONNX candidate and a maintained readiness replay adapter. Optional dependencies use `--extra kws`; retain `--extra wake` only for the openWakeWord comparison. This now connects to both diagnostics and the recorded-audio Session/MCP path, while real devices and real-speaker/Pi qualification remain pending. The earlier experiment's misses remain evidence. See [architecture](docs/ARCHITECTURE.md#hey-swift-without-training-candidate-decision-2026-10-27) and [verification](docs/PLAN.md).

```bash
UV_PROJECT_ENVIRONMENT=var/cache/venv UV_PYTHON_DOWNLOADS=never var/cache/venv/bin/uv sync --locked --extra dev --extra kws
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B scripts/fetch_kws_assets.py
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m me3_voice.readiness replay var/recordings/synthetic-speech.wav --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt --output var/results/sherpa-replay.json
```

The explicit setup script fetches inference assets and creates a phrase token file, without changing weights. If that keyword file already exists, reuse it; the script refuses to overwrite it. Replay itself performs no network access.

## Wake and VAD testing now

Wake/VAD testing is independent of DGX3 speech-service authentication and the undecided reasoning backend. The selected testing candidate remains pretrained Sherpa-ONNX **Hey Swift** plus local **Silero VAD**. No retraining or live API is needed. Use the [Pi test runbook](deploy/rpi/README.md) for inventory, target environment setup, saved recordings, optional approved PipeWire capture, UI connection and remaining physical gates. This workflow is prepared; Pi compatibility, microphone routing and acoustic reliability are not verified.

The readiness replay command above now reports voiced ranges, wake/utterance/endpoint/timeout events, incomplete speech, pre-roll range, thresholds and CPU timing. Add `--audio-origin synthetic` for the generated fixture, or `--audio-origin recorded` for actual recordings. `readiness monitor --help` describes the bounded raw PCM stdin path, which shares the same controller and atomically publishes reports while input arrives. It opens no microphone itself. Annotated reports can be scored with `readiness score`; see [scoring schema](docs/BENCHMARK.md#wakevad-report-scoring). No annotated real recordings have been supplied.

Restart the laptop watcher to load updated UI code, reconnect its feed folder and choose **Diagnostics → audio report**. Wake/VAD state, input origin, event offsets, CPU p95 and matched annotation scores appear there. The Assistant view explicitly marks voice/task controls unavailable; capture starts in the test terminal after the Pi checks. Report connectivity is separate from a microphone connection. The watcher must target the host producing reports; HPC does not automatically receive Pi results.

## Terminal diagnostics

M1 needs inspectable audio/wake results and operational feedback; a graphical UI is not an M1 exit requirement. The user subsequently requested the dashboard below as a supporting tool. The [supplied 12-page reference](docs/references/building-ai-voice-agents-summary.pdf) and its applicability/corrections are covered in ARCHITECTURE.md. Presentation stays separate from microphone capture, task state and cancellation.

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.cli demo --output var/results/diagnostic-demo.json --trace var/results/diagnostic-trace.jsonl
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.telemetry var/results/diagnostic-trace.jsonl --output var/results/diagnostic-summary.json
```

Use fresh trace/summary paths: those commands preserve existing evidence. Trace export excludes transcript/audio payloads. Demo providers and playback remain mocks; its trace measures local stage calls and memory-sink writes, not audible latency or exact words heard. The older demo JSON still includes mock conversation history. No browser, web service, device access or account is needed for these diagnostics.

## Assistant UI

The same bundled UI runs locally, on HPC or eventually alongside the Pi application. It has two views: **Assistant** for the conversation surface and recorded activity, and **Diagnostics** for audio results, timings and expandable report details. The static export and report server are read-only; they do not expose voice controls. The separate loopback voice bridge below enables browser microphone capture, playback and session controls. Its microphone indicator and small waveform use browser audio, while saved reports remain clearly labelled as recorded or mock evidence.

### Laptop updates from a terminal, without port forwarding

Use this path while Windows local-port binding remains unresolved. It requires the working SSH alias and Chrome or Edge; no Python or web server is needed on the laptop. Run in **local PowerShell**, with VPN connected:

```powershell
scp YOUR_SSH_ALIAS:/absolute/path/to/ai231/me3-voice-agent/scripts/watch_dashboard.ps1 "$env:USERPROFILE\Downloads\watch_dashboard.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\watch_dashboard.ps1"
```

The execution-policy option applies to that process only. The helper authenticates one SSH connection, receives the current UI plus sanitized report snapshots, and opens the generated HTML. In **Chrome or Edge**, open **Diagnostics → Connect report feed → Choose feed folder**, then select:

```text
C:\Users\EJ\Downloads\ME3\var\results
```

The helper prints the exact folder for your account. Leave the terminal open. New report files/results from HPC terminal commands appear through the two-second report feed; the browser reads only the selected folder's me3-dashboard.json. Ctrl+C stops the watcher; frozen/interrupted updates become stale while the last good data remains visible. The browser can disconnect the feed explicitly. This is a one-way report feed, not terminal control, microphone streaming or a live onit session. Inputs still require the documented recognized report formats.

The helper's parameters are `-SshHost`, `-RemoteProject`, `-Workspace`, `-Interval` and `-NoOpen`. When deploying the application to Pi, change the host/project parameters rather than copying another source tree. Windows PowerShell/file replacement and end-to-end laptop SSH delivery are not verified from this Linux host; HPC stream framing and the browser file adapter are tested. The Chromium browser check uses real filesystem handles in browser-owned storage, substitutes the folder-picker prompt, and separately checks folder API availability on a file URL. A real local-folder permission grant remains a user-side check. Unsupported browsers can use **Open report file** for a manual JSON import instead of continuous updates.

### Local report server or static export

From the ME3 root, the existing server remains available:

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.dashboard serve
```

It binds only 127.0.0.1:8765. Open that address on the same machine or use an explicitly authorized private forward; it is not a public/multi-user service. Refresh/polling reads reports, not audio/runtime state. Windows access through ports 8765/18765 remains unresolved; the terminal-file workflow above avoids that dependency without changing system networking.

### Live browser voice bridge (HPC prototype)

The same Swift Assistant page can connect to the existing Session/MCP path when served by the loopback voice bridge. It requests browser microphone capture and echo cancellation, sends bounded PCM over a same-origin WebSocket, and plays server PCM with per-packet browser completion acknowledgments. **Start conversation** arms the real Hey Swift detector; **Mute microphone** toggles that capture track; **End conversation** closes the browser session. **Stop speaking** stops browser audio immediately and requests cancellation of the current response; the UI follows the confirmed server event. After the runtime confirms a completed reply, the UI says **Reply complete** and returns to **Listening** for a follow-up. The static export and read-only report server above still have no voice controls.

For a controlled trial, `--direct-clock` opts exact final-transcript local-time questions into the existing typed task path without a model request. It is off by default; location-specific or multi-part time requests remain with the agent. Its mock speed improvement and synthetic replay are documented in `docs/PLAN.md`; real ASR accuracy and audible speed remain unverified.

#### Bridge readiness and recovery

The bridge binds only to `127.0.0.1` by default. An alternate port is selected explicitly with `--port 8766`; `--host` accepts only `127.0.0.1` or `localhost`, and public binds are rejected. Before requesting microphone permission, the live page checks `GET /api/health` (expected: `bridge: ready`, `voice_session: idle`, `session_resume: false`). This confirms only that the Swift bridge answered; it does not prove wake assets, speech services, a microphone, a Pi, or audio playback are ready. A second browser tab cannot create another session while one is active.

For an HPC browser test, keep the bridge terminal running, use VS Code Remote-SSH to `YOUR_SSH_ALIAS`, and forward remote port `8765` from the Ports view. Open the exact local address displayed in that row (VS Code may choose a local port other than 8765), then check `<that-address>/api/health` in the same browser. The UI checks this same endpoint before opening the microphone. If that check fails, verify the bridge terminal and the existing port-forward row; do not routinely remove/re-add the forward. If health succeeds but microphone setup fails, the UI distinguishes permission denial, missing input device, device-in-use, unsupported format, WebSocket failure, and runtime startup timeout. Allow microphone permission for the localhost origin or select/enable the intended browser input, as appropriate.

If the WebSocket drops, the browser releases its local capture/playback resources and performs bounded health-only retries (0.5, 1, 2, 4 and 8 seconds). It never reconnects the session or replays commands automatically: the current runtime has no session-resume contract. When health returns, click **Start conversation** for a fresh session. If the bridge reports an occupied port, it prints a safe diagnostic; inspect the listener and stop only a Swift process you own, or choose another unprivileged port. Swift never kills an arbitrary listener. These recovery paths have been checked with local browser/socket mocks, not by injecting an HPC network failure or testing a real microphone.

From the ME3 root, with the already configured owner-only `var/cache/voice-backends.json` and `.env` files:

```bash
ME3_MUSIC_CONFIG=var/cache/music.json env PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m me3_voice.voice_bridge --backend-config var/cache/voice-backends.json --secrets-file var/cache/voice-backends.env --llm-secrets-file var/cache/dgx3-llm.env --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt
```

Omit `--backend-config` and `--secrets-file` for an explicitly labelled mock-provider check. The bridge binds only `127.0.0.1:8765`, accepts one browser session, and keeps credentials in server-side processes. For laptop testing, connect VS Code Remote-SSH to `YOUR_SSH_ALIAS`, open its Ports view, forward remote port `8765`, and open the **local HTTP** address shown there in Chrome or Edge. VS Code may choose a different local port. The private VS Code forwarding path worked in the 2026-10-29 laptop trial; if forwarding fails later, the report watcher above remains the read-only UI path. Do not treat that watcher or a downloaded HTML file as microphone connectivity. The browser must use a trusted localhost origin for microphone permission. Stop the bridge with Ctrl+C.

Opt-in `--diagnostic-audio --diagnostic-max-seconds 60` retains bounded original/submitted PCM evidence under `var/recordings/diagnostics/` (duration configurable from 1–180 seconds; default 60). It never starts capture by itself: the user must explicitly start the browser voice session. The cap limits retained evidence, not provider input or conversation duration. Existing evidence is never automatically deleted. Use only for the targeted diagnostic trial; omit the flag for normal operation.

DGX3 Qwen2-Audio is final-only, so the user transcript appears after the turn endpoint; other configured ASR adapters may publish partials. `var/results/voice-<run>.jsonl` and `.json` contain payload-free software diagnostics from each browser session. Browser playback callbacks are stronger than memory writes but still do not prove audible delivery, AEC, Pi audio, or acoustic full duplex. The present browser capture uses a 1,024-frame ScriptProcessor callback and a provisional 1,024 ms silence endpoint; these need laptop/Pi timing and double-talk qualification before a demo claim. The canonical HTML/CSS/JS remain under `src/me3_voice/ui/`.

After the 2026-10-29 laptop trial, the browser path requires at least five VAD-positive 32 ms frames within an eight-frame window before a microphone candidate interrupts active playback. A shorter detection is logged as `barge_candidate_rejected`; its audio is not sent to ASR. The input onset remains buffered for a confirmed interruption. This is a software false-trigger guard, not echo cancellation or proof of reliable barge-in. A real speaker leak can last longer than this window, so repeat the wired-device trial before claiming acoustic behavior.

#### Automated check without speaking

From the ME3 root, the following focused check feeds the maintained synthetic Hey Swift WAV through the **actual loopback voice bridge** with real CPU wake/VAD and Pipecat ingress, but mock ASR/reasoning/TTS. A Python client stands in for browser PCM delivery and playback acknowledgments; it opens no microphone or speaker and reads no private backend configuration or credentials. It prints wake, turn and output-packet counts and writes the usual unique `var/results/voice-<run>.json[.jsonl]` timing reports. It needs the installed project environment and downloaded Sherpa assets above; expect model startup to dominate wall time.

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B scripts/check_offline_voice_bridge.py
```

For broader cancellation/pause/failure coverage, use `me3_voice.cli demo --scenarios` as documented above; for browser appearance and event projection, use `scripts/check_voice_ui.py`. The mock ASR text is scripted and unrelated to the WAV's words. These checks can catch software regressions without user speech, but they cannot qualify microphone capture, AEC, recognition accuracy, acoustic interruption, or Pi behavior. The old Chrome fake-microphone comparison in experiment 008 is inconsistent on wake activation and is not the preferred quick gate.

#### One laptop speaking trial and diagnostic handoff

No project files need to be transferred to the laptop for live voice: the browser opens the latest UI through the private VS Code port forward and sends its selected USB microphone audio over that connection. Set the USB microphone and speaker as the laptop's input/output devices before opening the page. Use the live bridge command above with the existing backend and secrets files, then follow these steps:

1. Keep the HPC terminal running and forward remote port 8765 in the VS Code Ports view. Open its local HTTP URL in Chrome or Edge. A downloaded `me3-dashboard.html`, the PowerShell report watcher and `me3_voice.dashboard serve` are read-only and cannot run this speaking trial.
2. Allow browser microphone access. Click **Start conversation** and wait for **Ready to listen** and **Microphone on**. If neither appears, stop before speaking and check the terminal/port forward.
3. Say one short, known utterance beginning with **Hey Swift**, for example “Hey Swift, explain a Raspberry Pi in one sentence.” Stop speaking and allow roughly a second of silence for endpointing. Observe Listening, final user text, Thinking, Speaking, browser audio, then **Reply complete** and Listening. The UI does not promise partial words from the configured final-only DGX3 ASR.
4. If that succeeds, ask one short follow-up while it says Listening. The current DGX LLM profile has no ME3 request-count ceiling (`request_limit: null`); provider/model limits still apply. Counters are diagnostic only. A straightforward completed task is confirmed from its receipt without a second reasoning call; ordinary answers still need a model response, and long speech can use multiple TTS segments. The optional Onit worker has a separate explicit request bound and is not compatible with the DGX profile yet, so omit Onit flags for this music/device trial. Optionally test **Stop speaking** during audible playback. End with **End conversation**, then allow the terminal to finish writing the trace.
5. Tell the assistant that the trial is done, the approximate local time, the phrase you actually said, whether the USB speaker played, and whether wake, follow-up, interruption or echo failed. The assistant can read the files in this shared ME3 workspace. Do not send the `.env` file, configuration credentials or browser permission data.

After **End conversation**, the bridge writes a unique `var/results/voice-<run-id>.jsonl` event trace and `voice-<run-id>.json` software timing summary. `var/results/ui-transcript.json` contains only the latest displayed session. The bridge now also saves each session's last text snapshot as `var/results/voice-sessions/<session-id>.json`; the trace maps session IDs to turns. These snapshots contain spoken/assistant text, are local owner-only files, and remain ignored under `var/`. They are available only for trials run after restarting the bridge with this change. To see the latest summary path in the HPC terminal:

```bash
ls -t var/results/voice-*.json | head -1
```

If a separate laptop upload is preferred, copy the exact run ID from that filename and use **local PowerShell** with VPN connected:

```powershell
$run = '<run-id>'
$remoteResults = '/absolute/path/to/ai231/me3-voice-agent/var/results'
$destination = Join-Path $env:USERPROFILE 'Downloads\ME3\voice-trial'
New-Item -ItemType Directory -Force $destination | Out-Null
scp "YOUR_SSH_ALIAS:${remoteResults}/voice-${run}.json" $destination
scp "YOUR_SSH_ALIAS:${remoteResults}/voice-${run}.jsonl" $destination
scp "YOUR_SSH_ALIAS:${remoteResults}/ui-transcript.json" $destination
```

Upload those three files only if needed. The JSON/JSONL files do not contain audio or keys, while the transcript intentionally contains spoken and assistant text. The summary can locate ASR, reasoning, TTS and server-send bottlenecks. It cannot determine when sound first became audible, whether the wired speaker actually played, or whether browser echo cancellation worked; report those observations separately or provide an optional synchronized external recording. Failed wake attempts before activation may leave only control/frontend events, so retain the approximate speaking time and the exact phrase even when no transcript appears.

The session-details dialog and payload-free runtime report distinguish **browser-reported** echo cancellation from a measured echo reduction. A small Chrome fake-microphone comparison under `var/experiments/008-browser-fake-aec/` missed the synthetic wake inconsistently across trials, so it does not establish wake reliability or an AEC effect. Use the intended wired microphone and speaker for that decision.

For a single self-contained snapshot:

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.dashboard export --output var/results/me3-dashboard.html
```

Open/download var/results/me3-dashboard.html. The default export filename remains var/results/dashboard.html for compatibility; both are generated from the same source, never maintained separately. Without a connected folder it remains a snapshot. Export data downloads sanitized JSON; Open report file imports that format. No external assets, CDN, microphone access, account or frontend build is needed for the HTML itself. Unknown report bodies, credentials, raw recordings and arbitrary conversation histories are not served. An explicitly published current-session transcript is the intentional text-display exception described below. Source/modified time/hash details and mock/HPC-live labels remain in Diagnostics; Pi acoustics stay unverified.

### Display and status semantics

The Swift identity and Assistant/Diagnostics views use centralized neutral/teal design tokens and a 16 px base (18 px at wide presentation widths), readable secondary labels, stronger contrast, clear focus and responsive wrapping. Assistant prioritizes the transcript and last reported session state/time. Its unavailable-controls notice replaces the nonfunctional composer. Session text, including partial/draft/interrupted/confirmed prefixes, remains separate from actual voice connectivity. One last-reported session label is shown. Jump to latest appears when reading earlier messages; new content does not force scrolling. Each activity card names its source file and available run/session identity, so separate tests are not presented as one interaction.

**Reports · Local feed / Folder feed** means report delivery only. **Snapshot / Imported file** means saved data without ongoing updates. **Stale** retains the last readable reports. None connects or establishes microphone, conversation runtime or Pi state. Component evidence follows the selected report; unsupported status is Unknown. Pi inventory/reported execution cannot establish Pi verification. The internal HPC-live evidence category is displayed as **HPC execution**, with file/PCM and synthetic/recorded origin where supplied. ASR/TTS provider and deployment location remain undecided; this viewer does not infer them or publish credentials/endpoints.

Missing measurements are Unavailable; actual measured zero stays zero. Software timings and per-call CPU percentiles are distinct from acoustic latency. Earlier synthetic wake cases are collapsed and explicitly separate from real-speaker accuracy. Typed reports and the existing evidence inspection/import/export/SSH feed remain the source of display data. Restart the watcher after source/UI changes to load the current template. See PLAN.md for browser checks and platform limits.

### Verification

```bash
PLAYWRIGHT_BROWSERS_PATH=var/cache/playwright var/cache/venv/bin/python -B -m playwright install chromium
PYTHONDONTWRITEBYTECODE=1 PLAYWRIGHT_BROWSERS_PATH=var/cache/playwright var/cache/venv/bin/python -B scripts/check_dashboard_browser.py --output-dir var/results/ui-assistant-check
```

Development-only Chromium checks cover desktop/mobile layout, source inspection, file import, continuous report updates, stale/error recovery and source payload safety. Generated files stay under var/. See PLAN.md for exact results and limitations. Frontend source remains src/me3_voice/ui/; the report adapters/stream live in dashboard.py. No second runtime/session owner or graphical framework was added.


### Conversation transcripts

The Assistant view supports incremental user hypotheses when an ASR adapter supplies them, finalized user text, assistant drafts, playback acknowledgments and interrupted responses. This is presentation state, separate from model reasoning/history and the payload-free latency trace. Recorded DGX3 speech and the loopback browser bridge now feed the same Transcript API; Qwen2-Audio remains final-only.

To publish a clearly labelled mock conversation, run **in the HPC terminal from the ME3 root**:

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.cli demo --transcript --output var/results/transcript-demo.json
```

The explicit flag publishes the current view to var/results/ui-transcript.json. The existing dashboard/terminal feed includes it; no separate client transport is needed. To get updated UI/code after this change, stop/restart the laptop watcher and reconnect the feed folder. The watcher bootstrap reopens the current HTML. A mock preview has already been generated here.

Only that canonical file with the typed conversation-transcript schema and display_authorized flag is read as conversation text. Ordinary demo history files are not imported. Without --transcript (or deliberate runtime attachment), no transcript publisher is created. When text is enabled, HTML/JSON snapshots and the laptop presentation files include visible conversation text; diagnostic JSONL still excludes it. Text is local generated state, not versioned evidence. A new session replaces the current session's view; this is not a permanent transcript archive. The in-memory view retains up to 200 messages and 100,000 text characters by removing whole older turns.

Draft assistant output is shown as a draft. On interruption/error, unconfirmed text is removed and only confirmed segments remain; a response interrupted before playback is shown without its generated words. Mock playback is labelled. Device/sink acknowledgment does not establish exactly which words a person heard, and word-level alignment remains pending. User text revisions replace one bubble rather than append duplicates. Old-session/turn results are fenced; the transcript observer cannot execute or repeat a task.

## Service adapters, turn analysis and themes

The existing recorded application accepts --conversation --backend-config <local-json>. Supported role/protocol pairs include asr/deepgram-listen-v1, asr/groq-whisper-v1, asr/vllm-qwen-audio-v1 (final-only), tts/deepgram-speak-v1, tts/vllm-qwen-tts-v1 and llm/chat-completions-v1. These are documented wire contracts, not a final model/provider selection. Shared HPC inference works only if its documented API matches a selected contract; another protocol needs its own adapter. The exact endpoint/auth/model/format and authorization requirements are in [ARCHITECTURE.md](docs/ARCHITECTURE.md#backend-configuration-and-external-service-gate). Our ASR/TTS MCP servers remain in the path. Bounded DGX3 and free-only OpenRouter calls have been made; their timings and failures are in PLAN.md.

Optional --turn-policy smart-turn uses Pipecat's bundled pretrained CPU turn model with local Silero and a bounded silence fallback. The default stays silence: the synthetic CPU smoke check does not establish natural-pause accuracy or Pi latency. No training or model download is needed with the pinned Pipecat wheel. See PLAN.md for the measured inference overhead and qualification limits.

The Color theme selector offers System/Light/Dark. System explicitly displays its resolved theme, such as System (Light); it correctly looks identical to Light when your device prefers light mode. The central Swift graphic opens session details. The static/exported viewer follows reported events and never activates a microphone. When the loopback voice bridge is available, separate Start voice, Stop voice and Interrupt controls appear and its actual connection state takes precedence over report status. Speaking begins at the sink write boundary, not at text generation; only the browser callback confirms scheduled playback completion. User bubbles distinguish partial and final transcription. Restart the laptop watcher to load updated source for report-only views.

## Class DGX3 speech integration

The announced Qwen2-Audio ASR and Qwen3-TTS CustomVoice services now have configurable adapters through **our existing ASR/TTS MCP servers**. Authenticated HPC checks confirmed both model catalogs, followed by one successful recorded-audio ASR/TTS trial through our MCP pair. Reasoning and playback remain mock; no Pi/acoustic verification is implied. See PLAN.md for the preserved initial TTS mismatch, fix and software timings. See [contracts and limits](docs/ARCHITECTURE.md#class-dgx3-service-contracts).

When you have the project keys and are ready to authorize bounded class-service calls, run this from the **HPC terminal in ME3**, not PowerShell or chat:

```bash
var/cache/venv/bin/python -B -m me3_voice.dgx3 configure --enable-inference --request-limit 5
```

The prompt masks both keys; Enter at the TTS prompt explicitly reuses the ASR key. It stores keys in ignored var/cache/dgx3.env with owner-only 0600 permissions and configuration in var/cache/dgx3-backends.json. Setup makes no requests and refuses overwrites. These files are not encrypted. The supplied service URLs are private HTTP, reachable through VPN/DILNET, not end-to-end HTTPS. No key enters the dashboard, report, command arguments or source tree. Omitting --enable-inference saves a disabled profile; use explicit fresh paths for a subsequent setup rather than expecting overwrite.

After setup, authenticate the read-only model check before a bounded recorded trial:

```bash
var/cache/venv/bin/python -B -m me3_voice.dgx3 inspect --secrets-file var/cache/dgx3.env --output var/results/dgx3-authenticated-discovery.json
# Only after checking the model catalog and approved class-service access:
var/cache/venv/bin/python -B -m me3_voice.cli replay var/recordings/synthetic-speech.wav --conversation --backend-config var/cache/dgx3-backends.json --secrets-file var/cache/dgx3.env --pace 1 --transcript --output var/results/dgx3-application.json --trace var/results/dgx3-application.jsonl
```

Use fresh trace paths on subsequent runs. This example uses an injected wake without explicit wake assets, mock reasoning and memory playback; it does not play audible speech or verify the Pi. ASR is final-only; TTS uses streamed PCM. Default Ryan/English voice selection needs authenticated validation and can be changed through configure --voice/--language. Request counters are retained by the live bridge across browser reconnects and reset when that bridge process ends; the current DGX profile has no ME3 request-count ceiling, and service/model limits still apply. Counters are not billing controls.

The Assistant keeps the active conversation compact beside task results. Opening or refreshing the page shows a clean welcome screen. Use the conversation-history dropdown in the header to view a saved session; choose New session to return to welcome. Report-feed updates never open an old conversation automatically. Starting voice displays only that new runtime session; ending it retains its history on the current page. The dropdown includes sessions loaded from reports or observed during this page visit, not every archive on disk. Mock light and timer results have event-derived widgets; generated files show preview/download actions and an execution receipt. Spotify has a live in-page browser player for laptop trials and a Spotify Connect target for the final Pi output path. The Assistant shows the track/state Spotify reports and offers real play/pause, previous/next, and seek controls; it never claims that reported playback proves audible sound. See [Spotify playback on laptop and Pi](docs/ARCHITECTURE.md#spotify-playback-on-laptop-and-pi). One fresh authorization with `--browser-player` is required to grant the streaming scope; refresh credentials remain server-side and the short-lived access token is sent only to the live loopback page. For Pi, Soloist outputs through the Pi audio device while the local ME3 bridge reads and controls the same Spotify Connect session. No Spotify audio passes through the UI or HPC bridge. Report import and feed controls are in **Diagnostics**, separate from voice-runtime status; the Swift voice mark opens read-only session details. Diagnostics includes sanitized DGX3 discovery status; “Key required” is an inspection result, not proof of inference. Restart your laptop watcher to receive regenerated HTML; refreshing an old downloaded file does not fetch the new template.

Latest validation: the existing wake/rearm fixture now completes two actual Sherpa/Silero sessions through Pipecat and DGX3 ASR/TTS (mock reasoning, memory output). Diagnostics shows correlated historical speech-trial results, software timings and Inspect buttons separately from catalog authentication. Conversation bubbles distinguish partial/draft/interrupted text and the welcome layout is compact. Restart the laptop watcher to fetch the latest canonical UI. Exact evidence and remaining reasoning/Pi gates are in PLAN.md.

## Existing Onit / free-only OpenRouter trial

Reuse the existing sandbox Onit installation for a bounded reasoning smoke test; do not reinstall or activate it over ME3. First run this from the ME3 directory in an interactive HPC terminal:

```bash
var/cache/venv/bin/python -B scripts/onit_trial.py configure
```

Enter your project OpenRouter key at the hidden prompt. This creates ignored owner-only var/cache/openrouter.env containing ME3_LLM_API_KEY; it refuses overwrites and makes no request. It does not edit or read global Onit credential configuration. Existing DGX3 keys are separate.

The prepared live command, to be run after key setup, is:

```bash
var/cache/onit-venv/bin/python -B scripts/onit_trial.py run
```

The initial candidate nvidia/nemotron-3.5-lightning:free appeared in the public catalog with zero prompt/completion price and tool support when checked. It is a candidate, not a validated final choice. Requests are restricted to the exact OpenRouter chat endpoint and model, with zero-price provider limits, no provider fallbacks, no SDK/Onit retries, at most two outgoing attempts and a 60-second response deadline. If unavailable, the trial fails rather than switching to a paid model. Default task tools, verification calls, code execution and prior sessions are disabled. No project source or credentials enter the fixed arithmetic prompt. A result is saved exclusively under var/results/onit-reasoning-smoke.json; it must be reviewed before repeating.

The initial arithmetic trial invokes installed Onit's chat loop, not its default CLI/server stack. A later opt-in browser task route now connects that loop through Swift's Session task executor and a persistent, confined worker. Its single artifact tool is in process; Onit's own MCP discovery/tool invocation remains to be qualified separately. Offline verification with a dummy HTTP/SSE peer uses scripts/check_onit_trial.py and scripts/check_onit_worker.py. See PLAN.md for results and remaining gates.

Live trial update: project OpenRouter authentication and installed Onit inference worked after removing an unsupported sampling parameter. The initial free candidate returned an incorrect answer on the fixed arithmetic check, so it is **not accepted as Swift's reasoning backend**. It later created one draft website file in a bounded task trial, with a 17.07 s task span; this is one task outcome, not a quality or latency distribution. See PLAN.md and var/results/onit-reasoning-evaluation.json for the connection/correctness distinction.

### Direct Onit + DGX Qwen qualification

The installed Onit runtime can be smoke-tested directly against the active DGX Qwen profile without changing the voice bridge or loading Onit's global config. Use the dedicated LLM key already stored in `var/cache/dgx3-llm.env` (not the ASR/TTS key file):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
var/cache/onit-venv/bin/python -B scripts/onit_trial.py run-dgx \
  --output var/results/onit-dgx-qwen-smoke.json
```

This sends one fixed no-tools prompt to the exact `Qwen/Qwen3.8-27B` DGX chat-completions route, refuses other models/endpoints, and saves a sanitized receipt under `var/results/`. It does not change the active voice runtime. `run-dgx-tools` is an optional forced inert-tool-loop diagnostic with a maximum of three requests; its latest result remains unqualified because the tool probe was not invoked. This is not an MCP integration test. The current `OnitTaskExecutor` accepts the documented DGX Qwen profile or the exact free-only OpenRouter route. On 2026-11-01, an Onit-primary synthetic-file replay completed through the ME3 ASR/Onit/TTS path; it does not verify live microphone, browser speaker, or Pi behavior.

### Optional Onit task route

Configure optional Onit credentials with masked prompts; this creates only the ignored project file `var/cache/onit-secrets.env`, never prints keys, refuses overwrites and makes no API calls:

```bash
var/cache/venv/bin/python -B -m me3_voice.onit_setup configure
var/cache/venv/bin/python -B -m me3_voice.onit_setup status
```

The speech/ASR/TTS keys remain in `var/cache/voice-backends.env`; use `--llm-secrets-file var/cache/dgx3-llm.env` to select the separate DGX reasoning key. Add `--onit-secrets-file var/cache/onit-secrets.env` only when enabling optional Onit tools. `--onit-python` enables the restricted worker; `--onit-primary` routes general questions to Onit as the sole reasoning owner, while supported direct music intents remain on ME3's music path. Without `--onit-primary`, the configured ME3 reasoning provider decides whether to delegate a task to Onit. `--onit-max-requests N` is an optional ME3-side cap from 1 to 8; omit it to delegate request limits to the configured provider. Each bridge run starts a fresh worker and the provider may impose its own quota. General Onit-primary questions invoke the worker; Tavily/GitHub integrations remain disabled unless explicitly enabled.

For a short Onit-primary laptop voice smoke using the existing DGX profile, run this from the ME3 root in an HPC terminal. It does not enable Tavily, GitHub or writes outside ME3:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
var/cache/venv/bin/python -B -m me3_voice.voice_bridge \
  --host 127.0.0.1 --port 8765 \
  --backend-config var/cache/voice-backends.json \
  --secrets-file var/cache/voice-backends.env \
  --llm-secrets-file var/cache/dgx3-llm.env \
  --onit-python var/cache/onit-venv/bin/python \
  --onit-primary \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt
```

This follows the current provider-enforced request setting; no ME3 request cap or Tavily allowance is added. Check `/api/health` through the existing VS Code port forward before opening the UI. Use one short question and one follow-up, then End and inspect the `voice-*.jsonl`, summary and `var/results/voice-sessions/<session-id>.json` files. A full Onit response is not guaranteed by the mock tests; stop if the worker/provider reports an error. Do not combine this first smoke with a project-directory write or the multi-minute mock background-task scenario.

For an isolated project output target, explicitly choose an existing absolute canonical directory under this project's `var/` and pass it with `--onit-output-root`. The model must create a new child directory; source, unrelated and symlink roots are rejected. The legacy parent-sandbox scope remains opt-in: to create a new sibling under the HPC sandbox, add `--onit-output-root /absolute/path/to/approved-sandbox` to that bridge command. This is an explicit write scope; without the flag, website files remain isolated drafts under `var/scratch/onit-tasks/`. The worker creates only a new directory and flat static files, never overwrites an existing path, and does not run or deploy the site. Because speech recognition can mishear names, review the reported directory name and absolute path before using the result.

Without `--onit-output-root`, Onit can create up to eight small `.html`, `.css`, `.js`, `.md`, `.txt` or `.json` draft files inside `var/scratch/onit-tasks/`; the UI offers safe text preview and download. For a website project in the sandbox, pass `--onit-output-root /absolute/path/to/sandbox`. An explicit existing canonical root inside this project's `var/` is also supported for isolation. This enables one new child directory per task and up to eight flat files including `index.html` (16 KiB per file, 64 KiB total). Files are created exclusively and existing paths are never overwritten. The UI previews files as text; the worker does not execute HTML/JavaScript, run a build, test the page in a browser, or deploy it. With this opt-in output root, the model response cap is 2,048 tokens; otherwise it remains 512. Optional external tools are separately opt-in:

- `--onit-tavily-credits 1` enables at most one basic Tavily search credit for the entire bridge lifetime. Each basic search costs one credit; Tavily is general web search and can find current weather pages, but it is not OnIt's structured `get_weather` forecast tool. Structured weather needs an OpenWeatherMap key, which has not been configured. The OnIt build installed here uses `OLLAMA_API_KEY` for its own search route and does not natively consume Tavily, so Swift registers this bounded Tavily tool explicitly. The user authorized a one-credit ceiling for the first Tavily test. Tavily's current docs list 1,000 monthly free credits and $0.008 per pay-as-you-go credit if free credits are exhausted; account overage settings remain unknown, so inspect the Tavily account before use. [Tavily credits and pricing](https://docs.tavily.com/documentation/api-credits), [OnIt tools](https://github.com/sibyl-oracles/onit/blob/main/docs/TOOLS.md).
- `--onit-github-readonly` exposes repository `get`/`list` only, with at most three reads per task. It cannot create, fork, push or delete repositories. The upstream OnIt `github_repo` tool can mutate repositories, so Swift does not enable it until the requested spoken approval flow exists.
- `ME3_ONIT_HF_TOKEN` is stored if entered, but is not forwarded to this worker: no model/dataset download action is enabled. Hugging Face is a model-hub credential, not a general reasoning key.

With `--onit-primary`, Onit owns general reasoning and Session remains the sole voice/task owner with a per-operation idempotency key; without it, the configured ME3 model decides whether to invoke Onit. Task results are reported accurately; they are not Pi-verified. Drafts can be previewed/downloaded only while the bounded loopback route is running. A task interrupted during execution continues only inside the bounded worker; the UI marks its outcome unknown, and manual reconciliation is still needed. A general spoken readback/approval exchange remains pending before broader write, shell, repository, deployment or device actions can be considered.

Onit's upstream chat loop supports an `ask_approval` callback for gated tools, but this Swift worker does not yet carry that callback through the voice session. Gated operations must remain unavailable until Swift can speak a specific pending action, receive an explicit spoken approval or denial tied to its ID, and return the decision to Onit without treating it as a new task. The directory tool above requires no new approval in this session because the user explicitly requested that example and it only creates an empty test-output directory.

The Onit worker now accepts the configured DGX profile, but creating a sibling project directory is an external filesystem effect. Keep `--onit-output-root` disabled for the first laptop voice smoke. If a later project-directory trial is authorized, choose and verify its exact output path first; never reuse the old `ME3-Test` example or create a path based only on a possibly misrecognized spoken name. Then pass `--onit-output-root` with that path's existing parent and verify the task receipt and filesystem state independently. A task bubble alone is not proof of the effect. The current route has no spoken confirmation gate for inferred directory names, so do not attempt a high-impact or valuable target.

Offline checks from the ME3 root make no live model request:

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m pytest -q tests/test_onit_tasks.py tests/test_session.py --basetemp=var/scratch/pytest-onit-task
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 var/cache/onit-venv/bin/python -B scripts/check_onit_worker.py
```

The second check needs local subprocess IPC and preserves its first result at `var/results/onit-task-offline-benchmark.json`; review it before rerunning. It measures cold worker readiness separately from a warm offline tool turn. Neither result measures live model latency or audible response.

### Adaptive conversation settings

Choose the conversation mode before Start; the preference is saved in this browser and sent to the runtime when connecting. The runtime owns expiry. Configure windows with `--followup-seconds 7 --answer-seconds 18 --extended-seconds 20` on the existing bridge command. End-of-utterance silence remains unchanged. A pending-answer window is used only for the runtime's explicit clarification path; punctuation in model text does not extend it. Wake standby still captures/processes audio locally for wake detection and retains bounded transient pre-roll, but does not stream ordinary standby speech to ASR or store microphone recordings. Existing intentionally published transcripts/traces remain retained. This does not identify the speaker or distinguish addressed speech from side conversation within an open window.
