# Phase 0 — baseline and evidence inventory

Audit date: 2026-11-09. This is a read-only source and artifact audit. No application, tests, models, microphone, speaker, Pi device, provider, or external integration was invoked.

## Repository and provenance

The ME3 directory is currently an untracked directory inside the parent sandbox Git repository. Git resolves to `/mnt/jfs_hpc/home/edgardo.callagon.jr/sandbox`; its current HEAD is `9338bc205420747988f8b2254cae37701c13488a`, dated 2026-10-16, with subject “ME2: Voice Controlled Smart Device - models and Pi 5 readiness”. Parent branch reports `main...origin/main [ahead 1, behind 32]`, with numerous unrelated untracked paths. That parent commit is **not an ME3 source revision**. Existing ME3 logs/results are therefore not reliably bound to a reproducible ME3 commit. No branch, index, or history operation was performed.

Canonical current project docs are README, `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `docs/BENCHMARK.md`, and `deploy/rpi/README.md`. `docs/PLAN.md` has an old “Updated” header but detailed entries through 2026-11-06; consult the dated tail for later status. Supporting cumulative evidence is in `docs/evidence/cumulative-assessment.md`; raw trials are under ignored `var/experiments/` and `var/results/`.

## Implemented runtime path

```text
Browser getUserMedia + Web Audio ScriptProcessor ─┐
                                                  ├─ loopback WebSocket
Pi `pi_audio` → pw-record ────────────────────────┘
       → aiohttp `voice_bridge`
       → Pipecat PipelineWorker (one custom ingress processor)
       → PCMPreprocessor (NumPy downmix, persistent Soxr resample to 16 kHz)
       → FrontendMonitor (Sherpa-ONNX/openWakeWord and Silero VAD)
       → ReplayInput/ConversationInput (pre-roll, endpoint, barge candidate)
       → Session (turn/history/cancellation/tool orchestration/playback queue)
          ├─ SpeechClient → ASR MCP stdio child → configured ASR provider
          ├─ configured LLM OR optional Onit subprocess → task tools/executor
          └─ SpeechClient → TTS MCP stdio child → configured TTS provider
       → BrowserSink/WebSocket → browser speaker
         OR Pi peer → pw-play → PipeWire sink
```

`PipecatAudioInput` wraps one `PipelineWorker` and a single `StreamingInputProcessor`; Pipecat turn tracking and RTVI are disabled. `voice_bridge.voice()` serially awaits each input frame before consuming the next WebSocket message. The bridge creates the audio/session controller and `SileroVAD(0.7)`. `PCMPreprocessor` does not perform AEC, denoising, AGC, or normalization. Browser capture requests echo cancellation, but that setting is not acoustic proof. Native Pi audio currently has no reference-based AEC.

`FrontendMonitor` calculates VAD and wake decisions; its session policy is disabled in the live bridge. `ReplayInput` separately owns pre-roll and the active endpoint buffer; live bridge silence endpoint is 32 × 32 ms (1.024 s), with a 30-second cap. `ConversationInput` retains and confirms input during playback/reasoning, then invokes Session. Session performs a turn invalidation before cancellation and clears playback. `Session.respond()` waits for final recognition, streams reasoning/tool rounds, and runs response synthesis concurrently with reasoning through a bounded segment queue.

The application owns one asyncio loop in the bridge process. Pipecat runs its worker as async tasks, not a distinct ME3 process. The speech MCP pair are Python subprocesses with individual `asyncio.run()` loops, connected over stdio. Optional Onit is a separate Python subprocess/environment with newline JSON over stdin/stdout and its own asyncio loop per worker request. Browser and Pi sidecar have their own loops. Pi sidecar additionally owns `pw-record` and `pw-play` processes. `asyncio.to_thread()` is used for trace/diagnostic file writes; Onit uses a one-thread executor for artifact reads. Native library worker threads are not established by this source audit.

### Responsibility boundaries and overlap

- Session is the authority for turn/session IDs, conversation history, interruption fencing, task orchestration, and playback queue.
- Audio frontend owns model-level wake/VAD and capture endpoint decisions. Session owns semantic state transitions and accepted turns. This is an intentional boundary with duplicated endpoint/session diagnostics in FrontendMonitor and ReplayInput.
- Speech MCP servers own speech-operation state and provider calls. SpeechClient owns MCP connection, client sequence checks, operation identity, and cancellation requests.
- Session invokes one `run_agent_task` tool call; Onit owns a nested model/tool loop inside its worker. The outer task is shielded and can continue after speech interruption; current Onit `cancel_pending()` only cancels mock tasks.
- Deterministic Session routes cover selected device/music/time commands while LLM tool routing also exists, creating overlapping routing paths.

## Processes, connections, queueing, and external calls

- **Bridge process:** one `aiohttp` loopback HTTP server and one main asyncio event loop. Browser voice WebSocket carries input PCM, controls, transcripts/status, output PCM, and ACKs. A separate 50 ms publisher task emits state.
- **Pi peer (optional):** HTTP health request, one loopback WebSocket, concurrent input/output tasks, subprocess pipes to `pw-record`/`pw-play`. Input is read in 1,920-byte blocks at 48 kHz mono (10 ms); output ACK is estimated by PCM duration and sleep, not physical playback completion.
- **Speech MCP:** two stdio subprocesses and two client connections. ASR uses `asr_open`, ordered `asr_append`, `asr_finish`; TTS uses `tts_open`, serial pull `tts_next`; `cancel` is available in both. Audio is PCM16 mono 16 kHz encoded as base64 in bounded 32 ms chunks. Servers are single-operation/single-client.
- **Session buffers:** 32-item ASR queue (32 ms frames), eight-item playback queue, two-item sentence queue. Browser output caps eight pending packets; Pi playback caps eight queued packets. Capture is backpressured end-to-end by awaited ingress. Session event/history arrays can grow during a long connection.
- **Provider network calls:** selected ASR/TTS adapters issue HTTP or provider WebSocket requests from their MCP subprocess. Configured reasoning adapters make HTTP/SSE chat-completions requests from the bridge. Calls are async, but transport operations and MCP chunk requests are sequential within each stream. There is no automatic retry for paid/effectful requests.
- **Onit network calls (optional):** Onit worker sends model requests over its allowlisted route using its HTTP client; its own tools may call Tavily and GitHub when separately enabled. `io_lock` serializes worker requests.
- **Other optional external services:** Spotify Web API/Connect; Open-Meteo; communications use local PipeWire/BlueZ/D-Bus interfaces. These are not part of the minimum voice route.
- **Optional VPN:** separate managed launcher path, not an implicit bridge dependency. The Pi-reported VPN smoke in the runbook was later stopped.

## Providers, packages, and active defaults

`pyproject.toml` supports Python 3.11–3.13 and pins Hatchling 1.30.1, aiohttp 3.14.5, MCP 2.3.0, Pipecat 1.13.0, Soxr 1.0.0; NumPy has a bounded range. `uv.lock` is canonical. Extras pin pytest 9.0.2, pytest-asyncio 1.3.0, Ruff 0.15.7, Playwright 1.64.1, pypdf 6.21.2; `wake` pins openWakeWord 0.6.0; `kws` pins Sherpa-ONNX 2.0.1 and sentencepiece 0.2.2; `pi` pins dbus-next 0.2.3.

| Role | Implemented choices in source | Default/qualification |
|---|---|---|
| ASR | Mock; Groq Whisper `whisper-large-v3-turbo`; Qwen2-Audio-7B-Instruct; Qwen3-ASR adapter; Deepgram Listen v1 | Mock by default. Historical live/Qwen trials are mixed; ASR fidelity and Pi access are not qualified. Qwen2 adapter is final-only. |
| LLM/reasoning | Mock; chat-completions adapter; Groq allowlist includes `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`; DGX `Qwen/Qwen3.8-27B`; exact free-only OpenRouter route for Onit | Mock by default. Live routes require `allow_live`, key and profile. No automatic provider fallback. DGX and Groq results are route/task-specific. |
| TTS | Mock tones; Qwen3-TTS 1.7B CustomVoice; Deepgram Speak v1 | Mock by default. Historical Qwen TTS and Deepgram adapter checks do not certify current Pi speech output. |
| Wake | Sherpa-ONNX Zipformer GigaSpeech 3.3M KWS with explicit “Hey Swift” keyword assets; openWakeWord ONNX adapter | Voice bridge requires explicit model assets. Historical Sherpa synthetic results include misses; no held-out room/Pi qualification. |
| VAD/endpoint | Pipecat Silero ONNX; fixed `ReplayInput` silence policy; optional Smart Turn local analyzer in replay | Silero is active in bridge. Smart Turn is not wired into live bridge. |
| Task harness | Onit worker subprocess with scoped artifact/project tools; typed ME3 task executor | Onit is opt-in via `--onit-python`; current worker is serialized, and speech cancellation does not cancel its task. |
| Search/repository | Tavily Basic; GitHub read-only get/list | Disabled by default; bounded flags and separate credentials required. No GitHub write/publish path. |

## Existing tests and measurement evidence

There are 31 test modules under `tests/`, covering audio, frontend, Session, providers, speech MCP, interruption, background mock tasks, memory, receipts, devices, music, Onit, bridge, packaging, diagnostics, benchmark adapter, UI, and VPN. This inventory does not assert they all pass at the current source snapshot. Existing logs report recent focused passes, including benchmark/packaging (8 tests) and VPN/packaging (12 tests); logs are not all bound to an ME3 commit. No tests were run for this Phase 0 audit.

Prior evidence is extensive but mixed. `docs/evidence/cumulative-assessment.md` inventories experiments 001–021 and records source-boundary failures alongside successes. Examples:

- CPU synthetic wake/VAD replay and preprocessing checks; Sherpa wake experiment documented 7/12 synthetic positives and 0/30 negative triggers in one setup. Not a real-room estimate.
- Laptop AEC-requested A/B sample with conditional attenuation, lacking calibrated playback reference/geometry; not ERLE, AEC qualification, or Pi evidence.
- Browser protocol, mock/session, and Pipecat transport tests; synthetic and mock output ACKs are not audible output.
- Qwen2 ASR prompt/format trials with substitutions and failures; limited controlled later passes do not establish general accuracy.
- DGX Qwen route probes: small text/tool requests sometimes pass, task-shaped Onit requests repeatedly timed out in experiment 020.
- Groq GPT-OSS 120B Onit scoped project runs succeeded in experiment 021 with exact artifact checks for selected cases; they do not establish broad task reliability or a qualified voice path.
- Pi operator report: locked install/check, mock demo/scenarios, health, synthetic wake/VAD preparation, USB device enumeration, and one VPN route smoke. No mic/speaker opened; no live model inference or physical action. VPN was subsequently stopped.
- The benchmark contract defines correlated stage timestamps and p50/p95, but full acoustic benchmark scores remain unavailable. Missing include real speech-end-to-ASR-final/first audible timings, acoustic interrupt stop, Pi event-loop lag, Pi CPU/PSS/RSS/swap, reproducible task grounding/verification rates, and measured token/API cost on a frozen profile.

## Permissions and high-risk gaps

This audit was authorized as read-only. Project rules keep mock hardware as default, preserve ME2, keep credentials private, and prohibit public endpoints, deployment overwrite, service changes, external publication, or paid model calls without authorization. No credential values were read. No Pi connection, audio open, provider request, VPN, GPIO, GitHub write, or public deployment occurred.

High-risk boundaries for Phase 1 decisions:

1. **Source provenance:** ME3 has no clean tracked revision/manifest in this checkout, so historical artifacts cannot be reliably paired with the exact inspected code.
2. **Physical duplex:** Pi capture/render identity, rate, routing ownership, physical geometry/volume, simultaneous I/O, reference-based AEC and acoustic stop latency lack current evidence. Browser echo cancellation does not transfer to Pi.
3. **Input backpressure:** receive handler awaits each Pipecat input frame; synchronous wake/VAD work runs on the bridge loop; ASR MCP calls are serial per 32 ms chunk.
4. **Speech path overhead:** MCP base64 JSON per 32 ms ASR append and pull-per-chunk TTS may add serialization/CPU cost. End-to-end Pi cost is unmeasured.
5. **Cancellation semantics:** Session cancels speech generation and playback; Onit task is shielded and continues. There is no complete steering/pause/cancel/reconcile API for live Onit work.
6. **Unbounded state:** Session event/history retention grows over long connections. Existing optional receipt DB covers selected device/music effects, not general Onit tasks or session resume.
7. **External action safety:** website output is bounded draft creation; GitHub is read-only; no public publish/approval workflow. Research tool/source verification remains optional and needs task-level evaluation.
8. **Provider fallback/cost:** a single selected profile is used; no health-based failover or independently verified fallback behavior. Provider limits, latency and Pi network availability vary.
9. **Service lifecycle:** Pi runbook has no approved systemd production qualification; managed VPN launcher has only mocked process-boundary tests and no attended Pi lifecycle qualification.

## Phase 1 decision agenda

Phase 1 should compare complete architectures and freeze interfaces before Phase 2–8 implementation:

1. Pi-local deployment boundary: what remains local, which cloud services are allowed, and how optional HPC VPN is excluded from basic startup.
2. Audio chain: actual device ownership/routing, placement and AEC reference; WebRTC AEC/PipeWire integration versus available hardware AEC; callback/process topology and restart ownership.
3. Turn policy: one authority for wake, VAD, endpoint, barge-in, interruption and engagement timeout; retain or replace fixed silence policy with measured Smart Turn comparison.
4. Speech MCP transport: provider shortlist and whether 32 ms MCP calls should be batched or use an explicitly permitted streaming data channel while preserving the two required MCP control servers.
5. Conversation/runtime: keep Pipecat plus Session or simplify; define single ownership for history, tools, cancellation, task lifecycle and fallback.
6. Harness and persistence: Onit fit versus LangGraph/Deep Agents only if evidence warrants; task process isolation, durable task IDs/state, user steering, and separate speech interrupt/task cancel semantics.
7. Tool security: typed capabilities, least privilege, approval policy, source provenance/verification, bounded filesystem/GitHub permissions, independent result checks, unknown-outcome recovery.
8. Providers: exact streaming ASR/TTS/LLM candidates, region/network path, hard request/spend caps, fallback trigger and no-duplicate semantics. Do not make HPC mandatory.
9. Resource and service management: aarch64/Python compatibility, CPU/RSS/swap budgets, systemd scope and shutdown/recovery, rollback and deployment provenance.
10. Qualification protocol: paired repeated measurements with p50/p95, event-loop lag, Pi resource telemetry, per-stage boundaries, acoustic trials, unseen task set, human steering/approval, and independent verification.

No Phase 1 technology choice is made by this baseline. Phase 1 requires a separate go-ahead after review of the decision agenda.
