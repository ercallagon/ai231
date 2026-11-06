# Development and SSH handoff

Read AGENTS.md, README.md, docs/PLAN.md, docs/ARCHITECTURE.md, docs/BENCHMARK.md and deploy/rpi/README.md before changes. This is the canonical operational handoff; older results and superseded configurations remain in PLAN.md, not a second source tree.

## Current application and ownership

The application module is `me3_voice`. Browser or explicit PipeWire peer → loopback bridge → Pipecat PCM ingress → preprocessing/Sherpa wake/Silero VAD → canonical Session → our ASR MCP → configured provider. Session routes supported typed actions to the shared device/music adapter or uses the installed Onit worker for model/tool requests. Our TTS MCP produces bounded queued output. Both custom speech servers participate; Codex is development tooling, not the runtime harness.

Prepared demo profile: Groq Whisper ASR, Groq GPT-OSS120B reasoning/Onit, university Qwen3-TTS. Original Qwen ASR is retained explicitly, with answer substitution/silence issues open. The scoped Onit static-project tools have historical independently verified live output; this does not establish arbitrary shell/build/deploy capabilities. Real Onit remains foreground, unlike the optional deterministic background worker. No unattended microphone capture or physical testing is implied.

The Pi target is Raspberry Pi 5/4 GB with local harness/MCP/wake/VAD and configured hardware adapters; inference is remote. A laptop VPN does not route the Pi. University speech needs direct authorized DILNET/VPN reachability from the Pi; see runbook network checks. Native PipeWire I/O is software-tested only and has no reference-based AEC. All Pi installation/audio/GPIO/phone acceptance remains pending the operator's fresh installation.

## Source, startup and recovery

GitHub project path is `ercallagon/ai231` → `me3-voice-agent/`. Never depend on sibling coursework folders. Use a reviewed source manifest for publication/patch transfer; private var/, model caches, recordings and credentials are excluded. User explicitly authorized this initial ME3-only commit/push; future publishing still requires the applicable authorization. Preserve unrelated branch/index/worktree state.

From the application root use `bash scripts/setup.sh`, `bash scripts/swift.sh check`, and `bash scripts/swift.sh demo` for credential-free software acceptance. The canonical Pi runbook supplies assets, separate pinned Onit install, masked key setup, explicit bridge/device flags, health checks and rollback. No boot unit or audio service is installed by ME3. Missing Pi adapters must report unavailable without silently changing to mock.

Dashboard source is `src/me3_voice/ui/`; exports are generated. It supports live browser controls, transcript/task receipts, report import/feed, diagnostics, themes and artifact access. Native audio mode uses it as an observer; browser Start must not create a second capture owner. Feed connectivity never proves microphone, model or Pi readiness. Reconnect/receipt recovery does not restore chat or rearm process-local timers. Preserve unknown outcomes and action idempotency; stopping speech must not repeat completed actions.

Keep generated evidence under var/. Do not modify ME2, shared services, wiring or deployments, or perform physical effects without specific authorization. Inspect git status before edits. Run focused changed-boundary checks and update PLAN.md with actual commands/results, evidence type and limitations. ASR fidelity, acoustic interruption/echo/ducking, native I/O, TTS listening quality, intermittent reconnect behavior and Pi qualification remain separate open gates; mock passes do not close them.
