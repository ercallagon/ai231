# Phase 6 — actions and ME2 capability preservation

**Checkpoint:** repository-local implementation, laptop/HPC mocks only. **Assessment: CONDITIONAL.** Phase 2 remains `PENDING_OPERATOR_VALIDATION`; no live-provider, Spotify, phone, external-write, publication, Raspberry Pi or Phase 7 validation was performed.

## Existing capabilities recovered

Evidence inspected in current ME3 source/tests and the archived ME2 files `action_router.py`, `slot_parser.py`, and `hardware.py`. No ME2 runtime, model, dataset, checkpoint, or asset was copied into ME3.

| Capability | Current status | Evidence and limitations |
|---|---|---|
| Wake and voice command handling | WORKING in software; physical qualification pending | Current runtime uses Sherpa-ONNX wake gating, Silero VAD, Pipecat native context/turns, and the project ASR/TTS MCP adapters. `tests/test_native_entrypoint.py` and speech-service tests cover mock/replay paths. ME2's trained VCM classifier is not restored. |
| RGB LED on/off/color/brightness | PARTIAL | `devices.py` validates actions and drives mock output by default; explicit Pi config selects gpiozero RGB on BCM 17/27/22. `tests/test_devices.py` covers mock state and failure. No physical output/readback was tested. |
| TM1637 display | PARTIAL | Current adapter uses BCM 23/24 and has tested mock output; physical display remains unqualified. |
| Timers and alarms | PARTIAL | One software schedule can be created/replaced/cancelled; timer countdown/alarm time can be displayed. Only enumerated values (10s, 30s, 1m; 6:00 AM, 8:00 AM, 9:00 PM) route deterministically. No audible buzzer/alarm is implemented in current ME3; schedules are process-local and not restart durable. |
| Reminders | PARTIAL | Current allowlisted demo reminders can be listed and optionally persisted under `var/cache`; arbitrary due dates and notification delivery are not implemented. |
| Time and weather | PARTIAL | Runtime time is local; weather is an optional explicitly configured Open-Meteo adapter. Weather uses network and was not called in this work. |
| Temperature/thermostat | PARTIAL | ME3 displays a demo setpoint and explicitly reports that it is neither a measurement nor HVAC control. No thermostat interface exists. |
| Music and Spotify | PARTIAL | `MusicController` implements mock playback and Spotify Connect/Web Playback API actions, authentication refresh and uncertain-outcome handling. It now reaches native voice tools and the native fast path. `ME3_MUSIC_CONFIG`, Spotify token environment variables, and the configured secret file were absent during the environment check, so the native default is mock. No live Spotify call or playback was performed. |
| Local MP3 playback | OBSOLETE in current ME3 | Archived ME2 `action_router.py` selected local files and launched a background audio thread; only a placeholder tune asset was present. Current ME3 has no local file-player adapter. No audio subsystem was copied or introduced. |
| Phone call / SMS | PARTIAL | The old ME2 handler only acknowledged a call/text intent. Current ME3 has an opt-in Bluetooth/BlueZ/telephony integration with bounded/uncertain outcomes and fixed demo recipients. It requires explicit Pi configuration and has no hardware validation. |
| General research/tasks | PARTIAL | Phase 4 ResearchClient and Phase 5 LangGraph provide generic research and task-private workspace create/update/read/list operations, SQLite checkpoints, steering, write receipts and readback verification. Workspace paths are confined to a per-task directory. No arbitrary host filesystem, browser execution, shell, GitHub, or real external write tool is connected. |

## Implementation

- Moved the deterministic, side-effect-free ME2/early-ME3 command parsers from `session.py` into [fast_commands.py](../../src/me3_voice/fast_commands.py). Legacy Session imports/re-exports them, preserving rollback behavior; the native path imports the pure functions without importing or constructing Session.
- `NativeState` now binds `load_device_service(device_config)`, configured `MusicController`, and optional `ReceiptStore` to the existing shared `MockTaskExecutor`. Mock remains the default. Device/music adapters are closed with the native bridge.
- Native tool schemas now expose validated `control_music` actions and describe device result semantics accurately. Native tool calls use the actual Pipecat user-turn ID rather than a fixed turn number.
- Added an exact-command Pipecat LLM fast path for known immediate time/light/timer/alarm/reminder/music requests. It executes through the same idempotent action executor and returns the adapter summary through the normal Pipecat/TTS/context lifecycle without a provider request. The route is fenced to one attempt per Pipecat turn, including tool-continuation turns. Additional explicit pause/resume/next/previous patterns are native-only so the legacy Session behavior remains unchanged.
- The native fast path reconstructs only the minimal prior user/tool-call/tool-result history needed by existing contextual parsers. A regression exercises “Turn on the lights” followed by “Can you make it 80%?” through the actual Pipecat `NativeLLM` boundary; it updates the mock light and makes zero provider requests. The direct correction still requires a successful immediately preceding light-action receipt.
- The native browser device-control message now dispatches through the same typed action executor, enforces the same four-action lamp allowlist as the rollback UI, and deduplicates command IDs in-process. Capabilities report the actual configured mock/Pi/music adapters. No physical result is represented as sensor-verified.
- Extended LangGraph's existing task-private artifact tools to nested normalized relative paths, directory creation, bounded directory listing, UTF-8 reads, and create/update file writes. Workspace directories are owner-only (`0700`), files are owner-only (`0600`), file bodies are bounded at 64 KiB, paths reject traversal/absolute/hidden/backslash forms, and symlinks/special files are not followed. No general host/project filesystem tool is exposed.
- Each file write uses the LangGraph tool-call ID plus task revision and content/path digest as its durable operation identity. A same-identity replay is accepted only if readback still matches; conflicting reuse, superseded writes, and unresolved dispatches fail closed. A SIGKILL test kills the worker after atomic file replacement but before receipt commit; restart replays the tool, reconciles the exact file hash, repairs the receipt, and verifies the artifact. Ambiguous state is not automatically overwritten.
- Atomic writes, file reads/listing and directory creation run through `asyncio.to_thread` so disk I/O and `fsync` do not block Pipecat's shared event loop. This reuses asyncio's default executor; it adds no dedicated thread, service or dependency. Local workspace changes are scoped to the user-started task's private workspace. External or destructive operations remain disabled.
- Offline graph tests create a nested multi-file static-site artifact, list/read/update files, and independently verify content hashes. Additional tests cover traversal and symlink denial, operation-ID conflicts, task-revision fencing, crash/retry recovery, and prove that unrelated research, conversational, and delegated requests bypass the deterministic fast-command route. File generation/readback does not prove that a site executes or renders correctly.
- Preserved the legacy default and rollback. No package, process, worker, framework or always-running service was added.

### Files changed

- Runtime: `src/me3_voice/fast_commands.py` (new shared pure parser module), `session.py`, `native_app.py`, `native_bridge.py`, `native_llm.py`, `conversation_tools.py`, and `task_runtime.py`.
- Regression: `tests/test_native_entrypoint.py`, `tests/test_phase5_tasks.py`, and `tests/phase5_process_worker.py`.
- Modernization records: `MASTER_PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, `BENCHMARKS.md`, `docs/PLAN.md`, and this report.
- Reproducible results: ignored `var/results/architecture_modernization/phase6-offline-20261111.json` (fast-command checkpoint), `phase6-general-actions-20261111.json` (workspace tests/source hashes), and `phase6-workspace-benchmark-20261111.json` (30 local write/readback samples).

The parser implementation moved from `session.py` rather than being copied into a second router. This relocates roughly 480 lines of parser code; the native binding and schema add a smaller amount of new logic. The runtime process topology and dependency manifest are unchanged.

## Ownership and boundaries

Pipecat remains the only live conversation/turn/context/playback owner. Its NativeLLM checks the small exact-command parser once, then either emits a local action completion through its normal streaming output or continues with provider reasoning and native tool calls. `MockTaskExecutor` remains the single immediate device/music dispatcher and its existing `DeviceActions`/`MusicController` instances own those adapter states. LangGraph remains the only delegated multi-step task runtime and owns durable task checkpoints/control through the existing SQLite task store. Simple device commands do not create LangGraph tasks. Speech interruption does not call task cancellation.

External writes remain unconnected. The existing task approval fixture dispatches only a mock effect; exactly-once external execution is not claimed. No unrestricted shell was added: the architecture decision requires confinement of filesystem, network and credentials before exposing one. Task artifacts remain under the existing private per-task scratch root with readback/hash verification.

## Validation and measurement

Environment for the final run: Python3.13.9, Pipecat1.13.0, LangGraph1.2.20, LangGraph SQLite checkpoint3.1.3, MCP SDK2.3.0, aiohttp3.14.5. No dependencies were installed or changed.

- Focused Phase 6 subset: **57 passed** (device and music suite plus fast route and parser regressions; mock output, no paid/live calls).
- Changed-path spot checks: **7 passed** for native device capability projection, browser UI action/duplicate replay, native local fast route, and legacy music routing; one existing Pipecat deprecation warning.
- Workspace security/recovery/routing focused checks: **13 passed** (earlier complete focused set); final steering/I/O-offload/crash core: **3 passed**. Final full regression: **631 passed, 0 failed**, one existing `AudioContextTTSService` deprecation warning, 303.88s. Command: `env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m pytest -q --basetemp=var/scratch/phase6-general-tools-final2-20261111`. Local SQLite/stdio subprocess tests required loopback/subprocess permissions; no external endpoints were contacted.
- No full voice/audio latency, CPU, RSS, Pi resource, or acoustic improvement is claimed. Existing historical Phase 4/5 mock measurements are not re-attributed to these changes.
- No live Groq, Tavily, Spotify, phone, Open-Meteo, external action, or Pi access occurred.

### Task-workspace file operation measurement

Thirty local writes with atomic replace, durable SQLite operation receipts, and independent readback/hash verification measured p50 **79.323 ms**, nearest-rank p95 **96.280 ms**, max **105.865 ms** on HPC. Model inference and full LangGraph model/tool cycles were excluded. CPU time p50/p95 was 2.073/2.183 ms. Peak process-lifetime RSS was 745,848 KiB for the loaded Python process and dependencies; this is not incremental tool memory and cannot establish Pi capacity. A delayed-write regression injects a 150 ms filesystem stall and confirms an asyncio 20 ms tick proceeds within its bound after offloading. These are local mock/filesystem results, not voice latency or a Pi result. One prior synchronous exploratory sample contained a 2.1 s maximum; sequential runs were not a controlled A/B, so no comparative latency improvement is claimed. Details are in the machine-readable result.

### Fast-action measurement

An isolated benchmark fed 50 exact brightness commands to the actual `NativeLLM.get_chat_completions()` fast path, with the existing mock `DeviceActions` output. Median 0.113 ms, nearest-rank p95 0.176 ms, maximum 12.425 ms; 0 provider requests and 0 model tool calls. This measures local routing, mock action, and text-chunk generation only; it excludes microphone, ASR MCP transport, TTS inference, playback, and audio acknowledgments. It is not end-to-end voice latency or a Pi performance result. Full environment and source SHA256 values are in `var/results/architecture_modernization/phase6-offline-20261111.json`.

## Remaining gaps and acceptance

Phase 6 is **CONDITIONAL**, not qualified. Task-private workspace file operations are implemented and offline-tested, but arbitrary host filesystem access, safe isolated code execution, browser execution, real external integrations/approval receipts, and physical device outcomes remain unavailable or unverified. Static-site files were not executed or browser-tested. Local MP3 playback and audible timer/alarm notification remain missing. Approval enforcement for real external writes cannot be tested while those tools are intentionally not connected.

Next milestone under the master plan is completion of this repository-local Phase 6 gap inventory, then explicit Phase 7 end-to-end evaluation authorization. Phase 7 must include unseen tasks, failures, concurrency, recovery and external-write safety only after safe generic tool boundaries are designed. Phase 2 physical audio/AEC qualification remains an independent prerequisite for final deployment; no hardware gate is marked passed here.

Rollback: select the legacy runtime engine. The old `Session` path still imports the same parser module and keeps its previous control semantics. Remove native selection/configuration to return to the previous native behavior; no schema migration or database reset is needed.

## Independent security and reliability review — 2026-11-11

**Confirmed defect and correction.** The initial workspace implementation checked path components for symlinks, then reopened the parent path for atomic replacement. A concurrent same-user process could exchange that parent for a symlink between validation and `os.replace`, redirecting a write outside the task workspace. The corrected store pins each workspace directory with `O_DIRECTORY | O_NOFOLLOW` and performs creation, replacement, readback, and listing relative to those file descriptors. It also verifies owner-only directory permissions. A deterministic adversarial test swaps the directory immediately before replacement: the write remains in the original pinned directory and does not create the outside file. Directory creation now validates the child relative to its pinned parent, too.

**Related read boundary.** A hard link to an outside private file could bypass symlink checks. Workspace reads now reject files with multiple links; an adversarial hard-link test covers this case. These checks protect the task tools from path redirection and accidental access to linked files. They are not an operating-system sandbox against an adversarial process running as the same account: the configured artifact-root parent and account credentials remain trusted.

**Replay and concurrency.** The existing SIGKILL-after-replace test was repaired to match the new read helper signature and passed after an actual subprocess restart. Existing operation fingerprints, SQLite dispatch/complete receipts, revision checks, readback, and final verification remain in place. The task runtime still allows one active task, bounds each file at 64 KiB, and caps tool calls at 60; it has no independent aggregate disk quota. `asyncio.to_thread` still keeps filesystem `fsync` off the conversation loop. No process, dependency, service, or tool permission was added. No Pi responsiveness, acoustic result, or external-write exactly-once guarantee follows from these offline checks.

**Assessment:** Phase 6 repository-local security review is CONDITIONAL. The tested path-swap, symlink, hard-link, replay, and restart boundaries pass offline; real external effects, browser execution of artifacts, live services, and physical hardware remain outside this checkpoint. This review does not start Phase 7.

**Final verification:** focused workspace/artifact/restart selection 18 passed after the crash-harness signature correction; targeted hard-link test 1 passed after correcting its fixture ID. Final exact-tree full regression **633 passed, 0 failed**, one existing Pipecat deprecation warning, 314.51 s. Command: `env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m pytest -q --basetemp=var/scratch/phase6-review-final-full tests`. Ruff check/format and `git diff --check` passed. Machine record with source hashes: ignored `var/results/architecture_modernization/phase6-independent-review-20261111.json`. The prior 631-test result remains the pre-review baseline. No new latency, memory, CPU or physical measurements were made.

The subsequent Phase6 qualification checkpoint adds an offline LangGraph multi-page website task test; current exact-tree regression is **634 passed, 0 failed**, one existing warning, 308.67s. See `PHASE7_PREPARATION.md` for artifact checks, laptop voice readiness and operator gates. Phase6 assessment remains CONDITIONAL: filesystem task outputs are structurally verified, but no browser rendering, real external effects, live provider or Pi device qualification occurred.
