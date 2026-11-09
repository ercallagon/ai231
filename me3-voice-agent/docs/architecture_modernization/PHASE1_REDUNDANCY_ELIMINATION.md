# Final architecture correction — eliminate redundant orchestration

Decision: 2026-11-09. **Select Pipecat-native conversation and one LangGraph task runtime. Retire the custom Session conversation engine, narrow ingress harness, Onit integration and proposed custom durable step executor after contract parity.** Preserve ME3 hardware, functional speech MCP services, authorization and independent verification. This supersedes the contrary controller/runtime choices in Phase 1 and the simplification review. Migration effort did not decide this selection.

This is a source-based final target, not a benchmarked claim of lower Pi memory or latency. No runtime edits, installs, inference, hardware trials or production migration occurred. New dependency combinations still require an isolated qualification before installation into the application environment.

## 1. Target architecture and ownership

```mermaid
flowchart TD
  mic[Pi USB mic] --> aec[PipeWire reference AEC]
  aec --> input[Thin PipeWire BaseInputTransport]
  input --> wake[ME3 Sherpa engagement gate / native VADProcessor]
  wake --> stt[Thin Pipecat STTService adapter]
  stt <--> asrmcp[Pi project ASR MCP service]
  asrmcp <--> asrcloud[Cloud Nova-3]
  stt --> user[Pipecat user aggregator / one turn policy]
  user --> llm[Pipecat GroqLLMService / native function lifecycle]
  llm --> tts[Thin Pipecat TTSService adapter]
  tts <--> ttsmcp[Pi project TTS MCP service]
  ttsmcp <--> ttscloud[Cloud Aura-2]
  tts --> output[BaseOutputTransport / PipeWire reference sink]
  output --> speaker[Pi USB speaker]
  output --> assistant[Pipecat assistant context aggregator]
  speaker -. actual render reference .-> aec
  llm <--> ctx[One Pipecat LLMContext]
  user --> ctx
  assistant --> ctx
  llm --> facade[Thin task-control tool / local IPC]
  facade <--> graph[One LangGraph worker / native checkpoints and interrupts]
  graph --> model[Cloud same Groq model]
  graph <--> state[SQLite framework checkpoint state]
  graph --> tools[Typed broker / source retrieval / files / browser / MCP / APIs]
  tools --> ledger[Effect facts and idempotency receipts]
  graph --> delivery[Deduplicated event projection / pause delivery]
  delivery --> user
```

Pipecat owns conversational history, turn aggregation, streaming inference, conversational function-call lifecycle, interruption propagation, TTS aggregation and output sequencing. **There is no parallel ME3 Session state machine.** One user-turn strategy accepts turn boundaries; native VADProcessor supplies evidence and Sherpa gates engagement, not a second turn controller. Start/End/wake/engagement timeout are a small ME3 policy adapter around the pipeline. Put installed `processors/audio/vad_processor.py:VADProcessor` before the MCP STT adapter and leave `LLMUserAggregatorParams.vad_analyzer=None`, so Silero executes once and local onset detection is not downstream of a slow ASR append. The aggregator consumes native VAD events and is the sole turn/interruption authority. Preserve native system-frame priority across STT. Background task cancellation is a separate explicit command.

LangGraph owns task requirements/revisions, model/tool observations, pending approval interrupts, safe checkpoints and terminal task state. A thin ME3 API facade maps start/steer/status/pause/resume/cancel/approve to runtime operations; it must not implement a second checkpoint manager or mirrored durable task state machine. Read models for UI/status are projections. Effect receipts are immutable external-effect facts, not a competing task orchestrator.

Both speech MCP processes execute actual ASR/TTS provider operations on Pi. The voice process has no direct speech-provider bypass. Cloud choices stay provisional: Groq `openai/gpt-oss-120b` low effort for voice/task, Deepgram `nova-3` and `aura-2-thalia-en`. Optional `claude-sonnet-5-5` backup stays disabled until API/budget parity tests. No separate coding model or classifier LLM. University Qwen/VPN is optional, absent from basic readiness. Pi hosts all framework control, wake/VAD, task state, tools and device routing; cloud hosts inference/search. Existing browser UI remains optional.

## 2. Source-level redundancy inventory

Evidence: current ME3 Python source and installed Pipecat **1.13.0** under `var/cache/venv/lib/python3.13/site-packages/pipecat/`. Read source without importing model runtimes. Counts below are physical source lines including comments/docstrings, excluding tests/generated files; they are not measured executable size or deletion guarantees.

| Custom responsibility / exact code | Classification | Native replacement or retained reason |
|---|---|---|
| `session.Session`, `State`; `activate`, `begin_input`, `commit_input`, `tick`, `exit` | **REPLACE WITH FRAMEWORK** | `PipelineWorker`, `LLMUserAggregator` / `UserTurnStrategies` own conversational execution. Retain only explicit wake/engagement policy, no THINKING/SPEAKING clone. |
| `Session.history`, `acknowledge`, `recognize_stream`, `respond` history writes | **REPLACE WITH FRAMEWORK** | One `LLMContext` and `LLMContextAggregatorPair`; assistant aggregator handles interruption resets and function results. Preserve the delivered-text invariant through output timing integration, not another history list. |
| `Session.reasoning` eight-round loop and `remote.ChatReasoning.respond` SSE/tool parsing | **REPLACE WITH FRAMEWORK** | Installed `GroqLLMService` → `OpenAILLMService`/base implementation plus `LLMService.register_function`. Delete custom conversational request/retry/token stream loops. Framework tool parsing is not authorization or guaranteed schema correctness. |
| `Session.direct_music_*`, `direct_me2_command`, `direct_light_*`, `direct_clock_query`, task confirmation branches | **REMOVE AS REDUNDANT** in production orchestration | Register generic typed tools with native function calling. Keep normalization, device validation and effect logic in tool implementations. Immediate explicit stop/control grammar is ME3 policy; unfamiliar goals must not require new command routes. |
| `Session.record_tool_outcome`, `reconcile_task_result`, `background_task_update` | **KEEP AS THIN ADAPTER** | Project framework events/receipts into UI and dialogue context. Durable task/effect truth is outside the speech turn; no second tool-result history owner. |
| `Session.Playback` queue/worker/inflight task; `respond` two-segment queue and `synthesize_segments` | **REPLACE WITH FRAMEWORK** | `TTSService` aggregation/audio contexts and `BaseOutputTransport` ordering/interrupt handling. Do not retain old queue behind framework output. |
| `Session.interrupt`, `valid` turn fencing, ASR/generation tasks and input queue | **REPLACE WITH FRAMEWORK** plus **THIN ADAPTER** | Native `InterruptionFrame` cancels/reset frame work. Speech MCP adapter retains operation/generation IDs to reject late cross-process output; PipeWire sink must physically flush/stop. Framework cancellation does not prove external provider abort. |
| All six `harness.py` classes: `CaptureFrame`, `InputEndFrame`, `StreamingInputProcessor`, `PipecatAudioInput`, `UtteranceFrame`, `SessionProcessor`; `run_utterances` | **REMOVE AS REDUNDANT** | Direct framework pipeline and transport input; replay tests feed native frames. Remove nested completion futures/runner wrapper and its per-frame scheduling. |
| `runtime.AudioApplication.normalized/feed/sync_idle`, `audio.ReplayInput`, `ConversationInput.feed` | **REPLACE WITH FRAMEWORK** | Native input transport, VAD and one aggregator start/stop strategy. Keep pre-roll only once in input/wake gate; preserve diagnostic replay as a test input, not a live controller. |
| `FrontendMonitor` endpoint/session policy and duplicate VAD decisions | **REMOVE AS REDUNDANT**; metrics **THIN ADAPTER** | Observe framework VAD/turn events. No second Silero invocation or idle clock. Keep annotated replay/scoring tools. |
| `audio.SileroVAD`, `PCMPreprocessor` conversion | **REPLACE WITH FRAMEWORK** for live I/O | Native Silero analyzer and transport resampler perform these functions. Keep file validation/statistics as diagnostics. Select one 48→16 kHz input converter; no second downmix/resample stage. |
| `voice_bridge.BrowserSink` pending set/flush/stop plus Session output queue | **REPLACE WITH FRAMEWORK / THIN TRANSPORT** | `BaseOutputTransport` owns delivery queue. Browser ACK/sample position is a transport-specific fence if retained; it must not introduce a second playback controller. |
| `pi_audio.run`, `PipeWirePlayback`, sidecar WebSocket | **REMOVE AS REDUNDANT / THIN ADAPTER** | In-process `BaseTransport` subclass owns explicit PipeWire targets and two pw-record/pw-play helpers. Eliminate separate Python peer, loopback PCM socket, duration-sleep ACK and its queue. |
| `VoiceRuntimeServices.open`, `_own_speech_lifecycle` | **KEEP AS THIN ADAPTER** | One composition `AsyncExitStack` owns framework/MCP/task child lifetimes. No custom shared reasoning/session manager. Keep MCP scope affinity and explicit credentials. |
| `SpeechClient` operation/sequence checks; `SpeechService`, `Operation`, MCP tools | **KEEP AS THIN ADAPTER / ME3-SPECIFIC LOGIC** | Required speech boundary. Simplify framing, retain bounded audio, protocol discovery, timeouts and resource release. Provider state belongs in these services, no duplicate direct Pipecat speech providers. |
| `OnitPrimaryReasoning`, `OnitTaskExecutor`, whole-job shield/`io_lock`; `onit_worker.run`, `Guard`, SDK monkey-patch, checkpoint-kill/restart narration shortcut | **REPLACE WITH FRAMEWORK** | One LangGraph worker with native state/checkpoint/interrupt/tool-node machinery. Remove Onit dependency/environment from target. Preserve scoped tool bodies and diagnostic evidence, not integration wrappers. |
| Proposed ME3 step engine/checkpoint/controller from prior review | **REMOVE AS REDUNDANT** before implementation | LangGraph is the authoritative durable runtime. Do not implement this proposed subsystem. |
| Artifact/project/search/GitHub tools, device/music adapters, scoped validation, credentials, `ReceiptStore` facts | **KEEP AS ME3-SPECIFIC LOGIC** | Frameworks cannot supply project authorization, hardware semantics, trusted postconditions or exactly-once external effects. Consolidate into one capability registry/broker; receipt schema migrates deliberately. |
| `Transcript`, `TranscriptPublisher`, `publish_state` | **KEEP AS THIN ADAPTER** | Event projections/export/reconnect presentation only. Remove duplicated semantic history/lifecycle decisions; read from framework/task events. Retain public UI format via projection during migration. |
| Reconnect result inbox, correlated tracing, delivery at safe conversational pause | **KEEP AS ME3-SPECIFIC LOGIC** | Native async function events help live delivery; process-restart delivery cursor and privacy policy still need small durable integration. Reconnect must never replay effects. |

Deletion targets include at least **18 named legacy classes**: the three Session classes, six harness classes, AudioApplication, ReplayInput/ConversationInput, three Onit parent classes, BrowserSink, VoiceRuntimeServices and ChatReasoning. Their useful adapter/tool behavior migrates to the boundary listed above; this is not permission to delete files wholesale. The production `MemorySink` path becomes fixture-only. No source was deleted in this review.

## 3. Native voice capability evidence and candidate selection

Installed framework source provides concrete substitution points:

- `processors/aggregators/llm_response_universal.py`: `LLMContextAggregatorPair` (2657), `LLMUserAggregator` (612), `LLMAssistantAggregator` (1616), `_handle_interruptions` (2020), function lifecycle handlers. Use one context; UI transcript is not model memory.
- `services/llm_service.py:1114`: `register_function`, explicit cancellation and timeouts. Async functions can continue across speech interruption. Durable task start returns a task ID promptly; do not keep a framework callback alive for hours as a checkpoint substitute.
- `services/groq/llm.py`: installed `GroqLLMService` supports GPT-OSS settings through native OpenAI-compatible service. No custom SSE parser needed in the final voice process.
- `services/stt_service.py:335,435`: `run_stt` and audio handling; separate receive task may emit final/interim transcription frames. `services/tts_service.py:708,1296`: `run_tts(text, context_id)` and interruption reset. `transports/base_output.py` owns PCM buffering, resampling and interruption; transport subclass still must implement device writes/flush truthfully.
- `turns/user_turn_strategies.py`: explicit VAD start and turn-analyzer stop strategies are available. Configure them explicitly; do not accidentally enable a second LLM completion classifier or inherited speculative/default strategies. Existing fixed-silence baseline is a native stop strategy profile; Smart Turn is the intended optimized policy only after measurement. With Nova disable provider end-of-turn authority, retain final transcription semantics.

| Final-design candidate | Native coverage and residual custom work | Final dependency/operation tradeoff | Selection |
|---|---|---|---|
| **Pipecat-native 1.13.0** | Replaces custom context, LLM loop, turn proposals, TTS aggregation/output lifecycle. Needs two MCP service subclasses, one hardware transport, engagement gate, policy and event projection. | Existing framework contains substantive reusable functionality, not just a queue wrapper. No SFU/service. Framework task/queue count can grow while application-owned controllers shrink. | **WINNER.** Meets required interfaces with one conversational owner and extensive actual source replacement. |
| LiveKit Agents | Source `AgentSession` owns chat/turn/interruption/tool/speech lifecycle; `start(room=NOT_GIVEN)` permits custom local I/O. It must also get two MCP STT/TTS plugins and PipeWire transport. | **A LiveKit server is not mandatory for custom local I/O.** Published source version 1.8.10 (main metadata/plugins not fully aligned) depends on livekit RTC, AV, local inference, telemetry and sounddevice. No measured lower ARM footprint or source deletion advantage over Pipecat. | Reject as target because no additional required capability or net custom adapter reduction was demonstrated, not because of migration effort. |
| Lean asyncio | Would retain/rebuild context aggregation, LLM/tool parsing, turn rules, interruption, segmentation/output timing and durable-task glue. | Could trim package footprint, but the required orchestration is then application code. No executable evidence of equal capability with less total complexity. | Reject; custom engines are exactly the redundancy this correction removes. |
| Current ME3 | Session + frontend controllers + Pipecat queue wrapper + Onit loop; useful baseline only. | Extra code/state sync without exploiting native functionality. | Replace target architecture; preserve only as rollback release until parity. |

External source evidence: [LiveKit AgentSession](https://raw.githubusercontent.com/livekit/agents/main/livekit-agents/livekit/agents/voice/agent_session.py), [dependency manifest](https://raw.githubusercontent.com/livekit/agents/main/livekit-agents/pyproject.toml), [version](https://raw.githubusercontent.com/livekit/agents/main/livekit-agents/livekit/agents/version.py). These mutable upstream sources are not a verified installed LiveKit release. Installed Pipecat is the stronger exact-version source reference. Neither framework supplies verified acoustic full duplex merely by handling interruption frames; PipeWire reference AEC and actual device trials remain mandatory.

## 4. Task runtime: reuse durable execution

**Select LangGraph StateGraph + MessagesState + ToolNode + a supported SQLite checkpointer**, one bounded model→tool→observation loop, no domain graphs, Deep Agents, subagents, managed graph server or LangSmith requirement. Dynamic planning is model-driven; the graph defines execution/control semantics, not predetermined task steps.

Review source versions: LangGraph main manifest **1.2.20**, SQLite checkpoint package **3.1.3**; these are proposed qualification versions, not installed or release-locked here. Manifest requires Python >=3.10, langchain-core, checkpoint, SDK/prebuilt, xxhash and Pydantic. SQLite package additionally pulls aiosqlite and sqlite-vec; vector search is not selected. ARM64 native wheel and lock compatibility are explicit gates. Do not hide these dependencies or claim fewer packages than today's tree. Freeze compatible published releases and hashes in the isolated prototype before application adoption.

LangGraph owns `get_state`/`aupdate_state`, checkpoint/thread identity, interrupted-node resume and streamed step events. Small graph wiring binds model and registry; `ToolNode` provides dispatch/schema/error handling, subject to broker approval and write serialization. Model provider adapter is framework-native where supported, e.g. ChatGroq; no custom planner/tool-dispatch loop. Credentials, caps, unsupported provider parameters and errors still need integration tests.

Start allocates a durable thread/task ID. Steer/pause/cancel arrives over one task-control channel: stop or drain the current model/read-only node, checkpoint at a safe boundary, apply a revision update and resume only after acknowledgement. Never mutate a checkpoint concurrently with an active stale run. A tool already dispatched can finish with an uncertain external outcome; cancellation does not reverse it. Approval uses native `interrupt()` before effect dispatch and resumes with a scoped decision. On restart recover from framework checkpoint, not serialized coroutines or saved speech transcripts. A node restarts from its beginning on resume: irreversible effects must never occur before its approval interrupt. Every effect uses a stable operation key and independent readback; Unknown is reconciled before retry. Preserve safety even if this requires an effect journal in addition to framework execution checkpoints.

**One task-state authority:** runtime checkpoint state stores requirements, revision, plan/observations and task lifecycle. Broker records only approval consumption/effect facts and checks the active revision fence; no parallel ME3 task status table. Control acknowledgement occurs after a safe fence is established, so an already-submitted action may precede accepted steering. Durable delivery cursor is a projection with stable event IDs, not another execution state owner.

| Runtime | Functionality reused / missing | Decision |
|---|---|---|
| Onit 0.1.5a0, installed source pin `3d014cfa...` | Model/tool loop, approval UI hook, safety queue exist. ME3 wraps a full-job call; no durable resumable per-step state contract at that boundary. Native hooks should not be mislabeled absent, but recovery requires substantial new glue. | Remove from final architecture; reuse safe tool implementations only. |
| OpenAI Agents SDK 0.24.0 published source | Tool loop, session/approval RunState machinery; good lean option. Persisting every safe task step and steering arbitrary running work would require more application lifecycle glue than the checkpoint runtime. | Not selected; no nesting or second agent runtime. |
| LangGraph | Checkpointed execution, state updates, interrupts/resume, tool dispatch and streamed progress directly match durable steering requirements. | Selected despite extra dependencies; replaces custom checkpoint/control engine. |
| Deep Agents | Adds filesystem/planning/subagent middleware on LangGraph. These overlap explicit ME3 scopes and are unnecessary to obtain persistence. | Do not add. |
| Custom ME3 step runtime | Requires writing the very planner/dispatcher/checkpoint manager being requested from mature tooling. | Superseded; do not implement. |

Sources: [LangGraph manifest](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/pyproject.toml), [ToolNode](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/prebuilt/langgraph/prebuilt/tool_node.py), [runtime state update](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/main.py), [interrupt/replay semantics](https://docs.langchain.com/oss/python/langgraph/interrupts), [SQLite implementation](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/aio.py), [SQLite dependencies](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-sqlite/pyproject.toml), [SDK human review](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals). Evidence is source/documentation, not a successful Pi prototype.

Important limitation: upstream AsyncSqliteSaver explicitly warns against production/high-write workloads and recommends a stronger DB. A single-user, one-active-task Pi is a bounded deployment exception to qualify, not a claim the warning does not apply. Test contention, fsync, termination/reopen and checkpoint retention with the actual storage. Do not add Postgres simply to rescue a poor fit; if SQLite cannot meet the declared bounded workload, stop at this gate and revise the runtime/storage decision rather than invent a second checkpoint engine.

## 5. Thinnest functional speech MCP integration

Two thin Pipecat service subclasses map native frames to project-owned MCP operations:

1. `McpSTTService(STTService)`: sends bounded PCM batches to a persistent active-engagement ASR operation and maps correlated intermediate/final events into native transcription frames. One provider receiver/event buffer in the ASR server; event polling must coexist with append/control. Current one-operation `busy` rule cannot block event reads behind audio writes. `finalize` is triggered by the single turn policy; provider-final segments are collected into that turn, not mistaken for another endpoint owner. No cloud upload during ordinary wake standby. Preserve the native audio/event path required by the aggregator/turn analyzer; prove finalization ordering rather than assuming any STT subclass automatically implements this contract.
2. `McpTTSService(TTSService)`: `run_tts(text, context_id)` opens/synthesizes/pulls validated PCM through the TTS server and yields native audio frames; native TTS aggregation owns sentence/chunk timing. Interruption maps context to MCP cancel/clear, rejects stale packets and immediately flushes physical output. Do not add custom sentence queues or direct-provider synthesis in the voice framework.

Keep stdio MCP as control/data transport initially. Contract-versioned batches of 64–96 ms reduce call frequency compared with 32 ms, but batch size is a benchmark variable, not a chosen latency win. Tail flush and provider event delivery must not wait for a full batch. MCP event read is a bounded pull/long-poll on the same connection, not an additional audio socket or third service. Retain explicit IDs/sequence/errors and bounded backpressure. Reuse current provider code inside the servers; replacing custom *conversation* networking does not delete speech provider validation.

Input downmix/resample happens once in native transport. Sherpa and VAD consume the same model-rate PCM; no second frontend pass. Output chooses one sample-rate conversion location at transport/device boundary. `BaseOutputTransport` owns the output queue; native PipeWire adapter owns only actual device writes and immediate stop. Initially use two owned CLI helpers (`pw-record`, `pw-play`) in the voice process; no PyAudio/PortAudio/Pulse-ALSA routing dependency is added solely to avoid two helpers. A direct binding may remove them later only if measured beneficial. Reaping/reopening pw-play on interruption is an adapter responsibility, not another playback scheduler. Device completion/latency remains unverified until acoustic tests.

Delivered-speech context is a required parity test: inspect native assistant aggregation against output consumption/word timestamps; if it overstates unheard text, add a thin context-transform hook using the same native context. Never solve this by retaining a second `Session.history`. Browser ACKs and PipeWire writes do not establish exactly what a person heard.

## 6. Net complexity accounting

Measured source scope: `session,harness,speech,servers,runtime,voice_bridge,audio,frontend,onit_tasks,onit_worker,remote,transcript.py` contains **8,962 physical lines, 44 class definitions and 279 function definitions including nested functions**. Hardware/device/music/UI asset modules are outside this scope and remain common to every candidate. All count differences below are design estimates, not implemented removals. Approximate future ranges include retained tools/provider/security/projection logic in this scope.

| Design | Approximate final custom source in scope | Owners / processes / loops | Additional machinery |
|---|---|---|---|
| Current baseline | 8,962 lines, 44 classes/279 functions measured | Session plus frontend turn bookkeeping and nested Onit loop; native 7 application processes/5 Python asyncio loops | One PCM WebSocket, custom input/segment/playback/Pi-output queues and ACK pending set plus framework internals |
| Pipecat-native + LangGraph | **3,500–5,200 lines estimated**, net reduction about 3,800–5,500; 18 legacy class removal targets, approximately 8–12 new adapter/wiring/policy components; retained classes are refactored, no exact future method count claimed | One voice owner + one task owner; **6 processes/4 Python loops** including one task worker; background native threads additional | Two real speech MCP children, two PipeWire helpers, task IPC. Native framework queues replace application schedulers; more internal queues may exist. |
| LiveKit-native + same task runtime | 3,700–5,500 estimated; comparable legacy owner removal, similar new adapter count | Also 6/4 with custom in-process PipeWire I/O; no SFU assumed | Additional RTC/AV/local-inference/native dependencies; no demonstrated further adapter deletion. Not measured heavier RSS. |
| Lean asyncio + same task runtime | 5,000–7,000 estimated; much of context/stream/tool/turn/playback logic must remain custom | Also 6/4; still one custom voice owner | Could reduce framework imports but reintroduces orchestration code and tests. Capability parity unproven. |

Winning estimate work breakdown: retained speech/providers/MCP 900–1,300; tools/security/verification migrated from current wrappers 700–1,000; UI/event projection/composition 700–1,100; hardware/wake/engagement/MCP frame glue 600–900; graph/control/policy wiring 600–900. Total 3,500–5,200. This estimate must be replaced by actual diff/AST counts after implementation; it must not be used as a promise that all 8,962 lines are redundant.

Process reduction: remove the Python Pi peer only, embed its I/O adapter in voice process; retain main voice (1), speech servers (2), graph worker (1), two PipeWire children (2). Existing audio daemons and on-demand browser/tool subprocess trees are extra. Isolating the task runtime preserves audio responsiveness and fault containment; it is justified even though one extra loop exists. No separate controller service, graph server, Redis, always-on browser, classifier or Onit process.

Critical no-tool voice path: remove loopback PCM WebSocket from capture/output. Keep two local speech MCP channels and three external inference connections (ASR, LLM, TTS). Task IPC is outside no-tool voice latency. No course requirement is satisfied nominally through a façade: speech network calls occur in the actual MCP servers.

Queue/state accounting distinguishes policy owners from native scheduling: current application has ASR input (32 frames), response text (2), playback (8), Pi output (8), browser pending (8) plus Pipecat/MCP/native transport queues. Target removes these application schedulers and introduces **no custom equivalent segment/playback queue**. It retains one bounded hardware ingestion boundary, one ASR provider-event buffer, native transport/TTS/frame queues, one control channel and persistent task events. Actual Pipecat per-processor queues/tasks are dynamic; an exact lower total queue count is not established. Every live PCM buffer must have an age/sample limit; application LOC reduction cannot justify unbounded framework queues.

Dependency effects: Pipecat remains but now supplies actual conversation functionality; remove Onit and its separate Python environment/compatibility stack; add LangGraph/core/checkpoint/prebuilt/SQLite dependencies. OpenAI SDK is already present under Pipecat. No guaranteed package-count or memory reduction: native ARM dependencies (notably sqlite-vec) and graph imports need lock/PSS testing. Expected CPU benefit is removal of duplicate preprocessing/scheduling and no extra classifier/model calls; external inference count remains three for a plain turn. Framework allocation/task overhead may offset savings, so p95/CPU/PSS remain measured release gates, not reasons to preserve redundant custom engines.

## 7. Focused proofs and no-regression gates

No new installation or runtime prototype was performed in this assessment. Source analysis is sufficient to select the target; these bounded proofs qualify it before migration:

1. Existing environment, zero API/device calls: capture source hashes and run current-tree offline tests; preserve results/failures. Then a native Pipecat mock pipeline fed PCM/transcript fixtures exercises MCP adapter lifecycle, partial/final gating, interruption, late PCM rejection and output-context consistency. Existing installed Pipecat can support this; no native service availability is assumed.
2. Isolated task environment **requires separate install/prototype authorization**: lock proposed compatible LangGraph/checkpoint releases and provider adapter; no managed services. Mock model/tool graph demonstrates native checkpoint, interrupt/resume, requirements update and forced-process restart at each approval/dispatch/result boundary. Check SQLite close/reopen, single-writer contention and dependency compatibility on ARM64. Do not patch a framework or create a custom checkpoint manager to force the result.
3. Physical Pi **requires attended audio authorization**: explicit AEC nodes, shared render reference, simultaneous I/O, double-talk, no clipped initial words, local device stop. All audible ME3 output must pass through the reference sink. No framework can bypass this proof.

Retain prior latency/resource targets as proposed gates; compare identical source-bound fixtures and authorized live samples with current baseline, then native architecture. Minimum: 50 matched turns/profile and 50 interruption positions; p50/p95/n/errors for speech-end→ASR-final/first audible, TTFT, TTS-first-PCM, local-loop lag and acoustic stop. Include quiet, pauses, corrections, echo-only and double-talk. Track total process PSS/RSS, native threads, swap and audio frame gaps under a live task and optional browser. No speed claim based on token throughput or framework-native labels.

Task qualification: >=20 unseen multi-step goals, >=30 steering/control sequences and >=20 crash windows. Zero unauthorized, duplicate or stale-revision effects; uncertain dispatch never auto-replays. Approval node replay must not repeat external writes. One graph owns task recovery; one native context owns dialogue. Cost includes all model/tool calls, repairs/retries and retrieval; no polling LLM progress calls. Tool schemas and permission tests remain general rather than fixed demonstration routes.

Rollback is an isolated baseline release/config, not two active controllers. Preserve state/receipt exports and a reviewed source manifest. Revert a bounded migration slice on lost cancellation, incorrect delivered context, premature turns, provider-final deadlock, device contention, failed recovery or measured resource/p95 gate regression. Never downgrade/delete ledgers to make an unknown effect retryable. If a framework fails a required contract, stop at that proof and revise the integration; do not quietly reintroduce the old Session/custom executor behind it.

## 8. Revised Phase 2–8 roadmap

Phases remain authorization gates; none has started through this report. The ordering is revised to avoid optimizing deleted code.

| Phase | Revised work and exit |
|---|---|
| **2 — native audio foundation and parity** | First current-tree offline regression + source manifest. Build isolated Pipecat-native transport/wake/turn/MCP mock contract slice with no Session controller. In parallel inspect actual Pi routing, then attended reference-AEC capture/render qualification. Do not tune Session/ReplayInput endpoint loops or the Pi sidecar queue; use baseline only for comparison. Exit: native input/output/interruption parity and physical AEC evidence, or explicit hardware blocker. |
| **3 — native speech MCP pipeline** | Functional `McpSTTService`/`McpTTSService`, versioned batching/events, native context/LLM/TTS/output. Compare fixed native silence vs Smart Turn, then provider alternatives only on failed gates. Retire live harness/Session/AudioApplication once parity passes; never stack them under the new controller. |
| **4 — native conversation/research/tools** | Register framework-native generic tools, runtime time/source provenance and retrieval; preserve device/music capabilities via shared broker. Delete custom ChatReasoning/Session tool loops and direct intent routes. Test context, delivered speech, budget/fallback and untrusted-content handling. |
| **5 — LangGraph task runtime** | Authorized isolated dependency qualification, then one worker with native checkpoint/interrupt/state update and event API. Retire Onit wrappers and abandon custom step engine. Steering is checkpoint/revision control, not a second planner. Exit includes restart/approval/unknown-effect tests and SQLite storage qualification. |
| **6 — generic effects and verification** | Bind same capability schemas to Pipecat and graph, constrained browser/files/commands/MCP/account APIs, scoped approval and independent outcome checks. No effectful tool until confinement and durable receipt gates pass. |
| **7 — full-system qualification** | Unseen goals, live concurrent voice/task, corrections, provider failures/reconnect, cost per verified success, all-process resource and acoustic tests. Replace LOC/dependency forecasts with actual manifest/diff/measurements. |
| **8 — deployment** | One primary user service owns voice/child lifecycles; existing PipeWire retained. Secrets, readiness, bounded restarts, graph recovery, artifact retention, rollback and physical Pi release qualification. No mandatory HPC or cloud framework host. |

No production functionality has been removed by this decision. The architecture we are selecting deliberately replaces redundant infrastructure rather than treating preservation of current code as a simplification criterion.
