# Modernization benchmark framework

> Current architecture and migration gates: PHASE1_REDUNDANCY_ELIMINATION.md. Earlier Session/narrow-ingress/custom-task candidate ordering below is historical and superseded. No new performance measurements were made during the correction.

This is the initial measurement contract, not a results report. Phase 0 ran no benchmarks and adds no latency or accuracy claims. The existing `docs/BENCHMARK.md` remains the canonical contract for the earlier coursework harness; this document defines the broader modernization comparison fields and provenance requirements.

## Evidence classes

Every row/result must be labeled `mock`, `replay`, `isolated_prototype`, `live_api`, or `physical_pi`; separately identify whether the outcome was independently verified. Do not pool classes. Preserve failures, cancellations, timeouts, and unknown outcomes.

## Required provenance

For every run record:

- UTC start/end timestamps and host timezone.
- Git commit plus dirty/untracked status. If no commit binds source, say so and include the source manifest/hash when one exists.
- Hardware, OS/kernel, architecture, Python, package/lock versions, Pi temperature/power/audio service state.
- Audio device identities/rates, geometry/distance, playback level, acoustic scenario, AEC/VAD/wake/endpoint configuration.
- Provider, endpoint/region, exact model/voice, route, request limits, token usage and estimated/actual cost where available.
- Corpus/task ID, expected result, sample count, run order, random seed, warm/cold state, retry count, and independent-verification method.
- Result class, evidence files, limitations, and whether any physical/external effect was allowed.

The schema example in `var/results/architecture_modernization/phase0-baseline.json` establishes machine-readable environment/provenance fields. Future run artifacts belong under ignored `var/results/architecture_modernization/` or a numbered `var/experiments/` directory.

## Metrics and event boundaries

| Metric | Start/end definition | Required companion data |
|---|---|---|
| Speech end → ASR final | Annotated acoustic end-of-speech to final ASR event | VAD endpoint error, input samples, partial/final timestamps, provider/network boundaries |
| Speech end → first audible response | Annotated speech end to verified first acoustic output at reference microphone | Keep server send, browser ACK, PipeWire write, and acoustic onset as separate timestamps |
| LLM first token | Request dispatch to first text delta/token at the adapter boundary | Model, endpoint, request size, stream flag, tool schema, time to headers/first event |
| TTS first audio | TTS request/open to first validated PCM packet | Text segment ID, MCP calls, network/provider timings, output sample count |
| Barge-in stop | Annotated user speech onset and confirmed VAD candidate to last acoustic assistant sample | Separate local browser stop, Session cancellation, provider cancellation, device stop, and acoustic end |
| Event-loop lag | Scheduled monotonic heartbeat delay while receiving audio and model work | p50/p95/max, CPU load, frame deadline misses, dropped/late audio |
| Resource profile | Steady and peak CPU, PSS/RSS, free memory, swap, temperature | Idle, wake listening, ASR, LLM, TTS, Onit/task, concurrent duplex phases |
| Task quality | Unseen task accepted only after independent postcondition checks | Task family, tool round trips, retries, approval, duplicate effects, unknown outcomes, time/cost |
| Grounding/date | Claims checked against timestamped retrieved sources and runtime clock | Citation coverage, source freshness, conflict handling, answer correctness |
| Human control | Steering/approval/pause/resume/cancel/recovery scenario outcomes | Whether task continues after speech interruption; final effect/state reconciliation |

For every continuous metric report sample count, p50, p95, max, and failures. Preserve per-sample values; do not report only aggregates. State the clock source and correlation IDs. Provider request count is not a billing control; report configured ceiling and provider usage separately.

## Comparison protocol

1. Freeze source, configuration, task/audio corpus, provider model, device placement, and acceptance criteria.
2. Verify all prerequisites without opening devices or making paid requests; confirm external-effect approvals and hard request limits before live trials.
3. Run a small smoke first, then repeated matched trials with randomized/rotated order where practical. Include cold/warm starts and all failure cases.
4. Keep task and speech controls matched; change one architecture boundary at a time only after baseline capture. For architecture combinations, report component versions and full topology.
5. Independently verify transcripts, factual claims, generated artifacts, device states, and external writes. A model's own statement is not verification.
6. Preserve raw sanitized event data, per-run JSON, commands, and limitations. Never fabricate unavailable acoustic, resource, cost, or provider metrics.

## Phase 0 baseline

No new benchmark run was executed. Existing artifact inventory and prior status are summarized in `PHASE0_BASELINE.md`; raw evidence remains in existing ignored experiment/results folders. The following are explicitly unavailable for the proposed end-to-end system: physical Pi speech-end/first-audible p50/p95; acoustic interrupt-to-last-sample; Pi loop-lag/resource traces during full duplex; frozen-profile live cost; unseen-task completion with independent checks; source-grounded research score; durable task steering/recovery success rate.

## Phase 1 acceptance targets

DECISIONS.md holds the proposed phase-specific release thresholds and candidate comparisons; none are measured. Pair full combinations A (current profile), B (Nova-3/Aura-2/Groq + local turn policy + Onit/controller), and C only if a named B gate fails (Flux as sole endpoint proposer or focused LangGraph worker). Hold tools/model/task corpus constant for harness comparisons. Capture endpoint/queue/MCP/network/first-token/PCM/acoustic boundaries, PSS for all children, context and task revisions, effect receipts, usage and failures. New results must include exact source/asset/package hashes and provider model IDs; phase0-baseline.json is an inventory, not a benchmark measurement schema. Use null plus reason for unavailable timing/cost fields.

## Post-Phase-1 challenge gates

The challenge report is authoritative for candidate ordering. No challenge benchmarks were run. Before the first behavioral change, bind the current source to a revision or hash manifest and run the offline regression suite with provider/device effects disabled. Preserve failed runs and explicitly classify IPC/environment failures. Baseline process inventory is seven native application processes plus existing audio daemons; include browser/tool descendants in PSS.

Voice: selected simplified Session/narrow Pipecat is the baseline; direct asyncio ingress is only a bounded A/B if measured frame-runner cost is material. Speech: compare Nova/local completion against Flux completion without stacked endpoint delay; AssemblyAI universal-3-6-pro only if a remaining gate justifies it. TTS stays Aura-2; Flux TTS and fixed-snapshot Sonic are challengers only on a named gate failure. Measure every candidate from common speech-end or text-ready to actual acoustic output, alongside provider/network/MCP spans. Cloud first-byte claims are not audible response measurements.

Task: compare one ME3 model-step worker versus unchanged Onit with common models, scopes and unseen tasks. Record cost per independently successful task, malformed/repair calls, tool rounds, interruption responsiveness, revision fencing and crash-window outcomes. SDK replacement is deferred. Detailed minimum sample counts, hold/revert conditions, approval invariants and Phase 2 order are in PHASE1_SIMPLIFICATION_DECISION.md. Current historical architecture combinations above are superseded where they prescribe Onit as the selected future runtime.

## Native architecture parity and complexity gates

Compare baseline versus Pipecat-native voice with real project MCP services; retain the same provider models, audio fixtures, tool scopes and device geometry. Map every old Session interruption/context/effect invariant to a framework integration test, including partial/final ordering, late PCM rejection, delivered-text context and task continuation across speech interruption. VADProcessor runs once before MCP STT; user aggregator owns turn acceptance; no duplicate VAD analyzer or LLM turn classifier. Measure bounded backlog age and native internal queue/task counts, not only custom queue deletion.

Replace source estimates after implementation with AST/diff counts over the report's fixed 12-module scope, count moved tools and new glue as custom code, and include all framework/runtime dependencies plus native threads/PSS. Native target is six application processes/four Python event loops with PipeWire CLI helpers, versus seven/five baseline; these are topology counts, not measured RSS savings.

LangGraph proof must validate a supported release lock, ARM64 (including sqlite-vec transitive wheel), checkpoint write/close/reopen, steering without active-state races, approval-node replay, partial external effects and receipt reconciliation. AsyncSqliteSaver's upstream production warning must be explicitly qualified under one active Pi task; no hidden Postgres or custom checkpoint manager. No performance claim based solely on framework source capabilities. Required prototypes and phase ordering are in PHASE1_REDUNDANCY_ELIMINATION.md.

## Phase 2 observed local results — 2026-11-09

Canonical methods, failures, commands and operator protocol: PHASE2_REPORT.md. Raw artifacts: var/experiments/022-native-voice-duplex/; summary: var/results/architecture_modernization/phase2-results.json. Evidence class: **mock / actual local MCP / x86_64 HPC**, not physical Pi or live inference.

Existing regression:469 passed in85.25s outside restricted sandbox IPC; initial sandbox attempt timed out at180s. Native Pipecat:3 scenarios passed; equivalent Session:2 passed. Transport:8 checks passed in3.46s at substituted process boundaries. No production code changed.

| Mock scenario | Sampled total PSS, MiB | Main CPU seconds | Loop-lag p50 / p95 / max, ms | Lag samples |
|---|---:|---:|---|---:|
| Context/tool/follow-up | 231.26 | 0.307 | 0.071 / 0.621 / 196.012 | 402 |
| Interruption | 230.63 | 0.088 | 0.072 / 0.270 / 24.994 | 421 |
| Background tool | 237.29 | 0.080 | 0.068 / 0.180 / 38.341 | 479 |

PSS sums proof plus two MCP children at one snapshot, not peak memory; sampled swap0. CPU excludes children. Lag includes startup,10ms heartbeat. The196ms maximum exceeds the proposed100ms bound and remains unresolved. This is not a paired improvement benchmark. Physical latency, acoustic interruption, Pi resources and AEC gates are PENDING_OPERATOR_VALIDATION. Live streaming and slow ASR-finalization cancellation are unqualified. PCM correction:1,920 bytes at48kHz mono PCM16 is20ms, superseding earlier10ms descriptions. Native VHQ Soxr buffering/flush needs later measurement.

## Existing-audio reuse and deferred live testing — 2026-11-09

Read-only corpus inventory found54 integrity-matching guided browser takes (435.923s), one18s reading, and a23.94s laptop AEC pair; no paired Pi raw/processed/reference session identified in this checkout. Canonical coverage/gaps and evidence limits: docs/evidence/audio-corpus-inventory.md. User explicitly authorized existing recordings for relevant new service evaluations, superseding historical other-provider consent restrictions; preserve original manifests. No upload, paid inference or later-phase implementation occurred.

Phase2 acquisition now includes separate noninterrupting corpus profiles, repeated true interruption trials, wake/negative exposure, continuity/resource logging, actual transcripts and reference/geometry metadata. Hardware gates remain PENDING_OPERATOR_VALIDATION; recordings enable offline comparisons but cannot qualify changed live AEC or final acoustics. Canonical commands: PHASE2_REPORT.md. Validation tools are tracked in scripts/phase2/ and tests/test_phase2_*.py; raw history remains ignored.
