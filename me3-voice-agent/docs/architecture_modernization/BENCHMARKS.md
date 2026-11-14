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

## Reported Pi trial: qualification failure — 2026-11-09

Source is operator report; artifacts not yet independently inspected. Second attempt saved raw/processed WAVs reported24.07s each, after a scratch-only diagnostic WAV fix; first attempt produced no diagnostic WAVs. Neither reference nor later4s direct playback was audible to the operator. Logged interruption cannot establish acoustic barge-in/AEC. input_failure=IncompleteReadError remains unclassified. No valid acoustic p50/p95, attenuation, false-interruption, duplex or Pi efficiency score. Preserve all artifacts under Pi var/results/phase2-pi-20261109-1446/aec-smoke-01/. Full observations/limitations and next steps: PHASE2_REPORT.md. No tests run while recording this checkpoint.

## Phase3 source-bound offline comparison — 2026-11-10

Canonical final data: var/results/architecture_modernization/phase3-final-benchmark.json; methods/tables/limits: PHASE3_REPORT.md. Four profiles,50 measured turns each plus one warm-up, identical384ms recorded PCM with mock ASR/LLM/TTS and real MCP, paced memory output. No provider costs or acoustic metrics. Full regression506 passed; baseline483.

Session/native32/native64/native96 first-memory-write p95:15.34/63.05/68.66/66.75ms. Native uses explicit50ms fixture turn-stop wait; no latency improvement claimed. Total per-turn tool calls22/22/17/15; keep32ms default because larger batches do not improve response time and live partial responsiveness is unqualified. Pure encode+argument JSON p95 is0.0072/0.0108/0.0149ms at1024/2048/3072-byte PCM (n1000 each); append MCP round-trip p95~1.31ms for native32 includes immediate mock provider/scheduling overhead. These are different measurements, not interchangeable estimates.

Startup and steady windows are distinct; native32 startup max176.32ms remains visible, steady p95~1.02ms/max11.09ms. Snapshots total232.24–234.45MiB PSS across main+two children, not peaks or isolated legacy/native deployment memory; no Pi RAM/CPU savings claim. Sampled swap0. Native private queue inspection is benchmark-only and version-scoped. Earlier overlapping/source-mutated exploratory runs are retained but excluded from final comparison. Physical Phase2 gates stay PENDING_OPERATOR_VALIDATION.


## Phase3 actual-entrypoint comparison — 2026-11-10

Canonical data: `var/results/architecture_modernization/phase3-entrypoint-legacy.json` and `phase3-entrypoint-native.json`; source accounting: `phase3-entrypoint-complexity.json`. Actual CLI dispatch, separate sequential processes, real project MCP children, mocked inference, identical64ms recorded PCM,50 measured turns plus one warm-up each. Both use explicit complete-utterance EOF with no extra silence floor. These results supersede the earlier50ms-biased fixture comparison for application latency only; earlier paced/batching data remains source-bound historical evidence.

Endpoint→final p50/p95: legacy3.795/5.860ms; native1.314/1.633ms. Endpoint→first-memory-output:7.729/10.107ms versus6.298/8.885ms. TTS-open→PCM:2.641/2.947ms versus1.995/2.458ms. Both12 MCP calls/turn; append RPC p95 legacy1.171/native0.975ms (102 samples each).32ms retained. Native40ms output versus legacy32ms; markers straddle an immediate sink invocation, not audible output. No live LLM/speech accuracy, token/cost or acoustic result.

Steady10ms heartbeat p95 legacy0.695/native0.613ms (n95/105); startup max878.4/173.7ms, distinct windows. Startup diagnostic single samples4.780/4.263s exclude initial benchmark/controller imports and include remaining lazy setup; not controlled cold-launch distributions. Sampled summed3-process PSS maxima224,308/260,283KiB; RSS281,948/317,532KiB; sampled swap0. Native uses more sampled memory here. CPU totals include setup, not Pi utilization.10ms queue samples show legacy input3/playback1, native admission1/processor queues up to3; not hard peak bounds. Full integrated working tree520 passed; baseline506. Detailed limitations, commands and logs: PHASE3_REPORT.md. Phase2 hardware gates remain PENDING_OPERATOR_VALIDATION.

## Phase4 native conversation checkpoint — 2026-11-10

Baseline522/final559 regression checks passed; final195.49s, one installed-framework warning, no skips.37 Phase4 cases plus existing native speech/legacy checks. Actual SDK/loopback SSE and two genuine mock-upstream speech MCP children validate repaired tool arguments, source retrieval/verification/follow-up and503/malformed-JSON failure recovery. No live Groq/Tavily/physical score.40 varied question inputs and no-spend manifest/evaluation runner prepared; correctness/citation scores remain null pending independent live review.

Canonical mock overhead run: var/results/architecture_modernization/phase4-native-qualified.json;50 measured turns plus warm-up, same64ms recorded PCM, explicit EOF, mock upstream and memory output. Endpoint→final p50/p951.336/1.490ms; first memory output6.647/17.924ms; mock LLM TTFT0.148/0.162ms; TTS→first PCM2.023/2.530ms; append IPC0.829/0.972ms(n102).12 MCP calls/turn,32ms unchanged. Steady lag p950.637ms/max3.509ms(n114), startup max166.09ms separated. Sampled3-process summed PSS max260,855KiB/RSS318,568KiB, swap0; no Pi or hard-bound interpretation.

Tail output p95 exceeds prior Phase3's8.885ms; first Phase4 run16.376ms, instrumented17.067ms, final source-bound17.924ms. Not a speedup or randomized paired model/physical comparison; corrected completion semantics/context/policy and host effects differ. Live TTFT/quality, real search latency, token costs and physical performance remain unmeasured. Source/dependency hashes and detailed limits: PHASE4_REPORT.md and phase4-results.json.676 staged application lines added, no new framework/process/dependency; legacy retained.

## Phase5 LangGraph task runtime — 2026-11-10

Reproduction: `PYTHONPATH=src var/cache/venv/bin/python scripts/benchmark_phase5_runtime.py --samples 10`. Raw machine record: `var/results/architecture_modernization/phase5-runtime.json`. Evidence class: repository-local mock/offline. CPython3.13.9, x86_64 HPC; one warm-up excluded;10 fixture tasks, each 3 scripted model responses and 2 task-local artifact operations;0 provider requests. Task-runtime startup after imports0.296s; task elapsed p50/p951.103/7.816s (n10, one slow sample dominates this small-sample tail); saver checkpoint writes p50/p9533.1/81.4ms (n290);5ms heartbeat lag p50/p950.075/0.927ms, max80.3ms (n3,465); RSS/PSS/swap102,604/84,947/0KiB; CPU user/system0.453/0.262s. One process plus aiosqlite worker thread; no MCP children in this isolated benchmark. The complete native integration test separately launches actual ASR/TTS MCP services with mock upstream inference.

These are not Pi estimates, live-provider latency, or evidence of improvement. The small task p95 is an order statistic over10 tasks. Checkpoint persistence produces29 writes per fixture task, so SQLite overhead is material for very short mock tasks; profile on the Pi with realistic provider/task durations and live voice concurrency before optimizing or replacing storage. Framework imports/process PSS here are host-specific. No physical CPU/RAM, swap, acoustic interruption, task cost or independently successful live-task measurements are available. Full-suite status and the approval/recovery failure cases are in PHASE5_REPORT.md.

## Phase5 independent review benchmark — 2026-11-10

Final-source record: var/results/architecture_modernization/phase5-review-qualified.json. Reproduce with `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python scripts/benchmark_phase5_runtime.py --samples 50 --output var/results/architecture_modernization/phase5-review-qualified.json`. x86_64 HPC, fixture model/local artifact tools, zero provider requests, one warm-up plus50 tasks. Startup0.360s after imports; task p50/p95/max0.862/1.604/3.343s. Checkpoint-call p50/p9528.5/56.1ms over1,450 calls;29 per task.5ms heartbeat lag p50/p95/max0.326/1.003/80.5ms(n8,795). RSS/PSS/swap104,652/83,946/0KiB. CPU user/system2.276/1.162s. One benchmark process and SQLite thread; no speech MCP subprocesses in this benchmark.

Two earlier50-task review records retain source hashes and distinct observations: p50/p951.024/1.154s and0.898/1.064s. Historical7.816s p95 did not recur, but the final3.343s maximum shows variability. Review runs overlapped offline regressions; no controlled before/after inference. fuseblk filesystem and FULL commits may contribute, but attribution remains unresolved. Saver calls include waits and may overlap; their sum is not exclusive elapsed time. Raw per-task/per-checkpoint timings are now preserved. No physical/live model performance, improvement or Pi estimate is established.

## Free-tier live pilot readiness snapshot (before execution) — 2026-11-11

No live-provider measurements exist yet. User authorized use of selected Groq/Tavily providers if the accounts are on free allowances. Both keys now load from `var/cache/phase5-secrets.env` (mode0600); secret values are not recorded. No requests were sent. Pilot remains capped at12 Groq and8 Tavily calls. Prior to execution, verify Groq account tier/limits and Tavily pay-as-you-go disabled; never rely on the request cap alone as a billing cap. OpenRouter, Brave, and Cerebras are later demo candidates only, not benchmark results or integrated fallbacks. See DECISIONS.md for current documented free-tier terms and sources.

## Phase4 live pilot attempt — 2026-11-11

Prepared four cases, but executed zero. Startup stopped during local project speech MCP `initialize` handshake timeout, before the case loop. Groq requests0; Tavily requests/credits0; model tokens0; no claims, latency, quality, attribution or provider metrics available. Configured limits were12 Groq and8 Tavily. Partial runner metadata and classified outcome: `var/results/architecture_modernization/phase4-live-pilot-20261111.json` and `var/results/architecture_modernization/phase4-live-pilot-outcome.json`. This is a local integration/startup failure, not a provider test. Repair and pass mock-only startup preflight before retrying live; do not widen the caps.

## Phase4 live pilot retry — 2026-11-11

ASR/TTS independent mock MCP preflight passed (2.121/2.074s, exit0, protocol2025-11-25, empty stderr). P4-02 completed:1.634s,2 Groq requests, 1.535s total observed TTFT sum, 0 Tavily searches; runtime date answer matched03:58:59Z/11:58:59 Asia/Manila. P4-11 stopped after5 Groq request-budget attempts and one429; elapsed5.496s, successful TTFTs0.612s/1.305s. It performed one Tavily basic search (1 credit) plus one direct original-source fetch (not a Tavily credit). Search/retrieval yielded official ECB source, but Groq failed before final response. Successful usage6,305 input/181 output tokens; failed request token usage unknown. P4-12/P4-17 not run. No retries, other APIs, or actions. The original8-second/local-MCP issue was managed sandbox IPC; unsandboxed subprocess diagnostics passed. 3 preflight tests and52 Phase4/native app regressions passed, one existing deprecation warning. Raw run and limitations: PHASE4_REPORT.md and `phase4-live-pilot-outcome.json`.


## Phase4 cumulative live-pilot continuation — 2026-11-11

P4-11 only:120.013s timeout,5 Groq requests succeeded,16,241 prompt/411 completion tokens (unique usage events),3 Tavily basic-search attempts and2 source-page fetches. No final answer/source was retained. Safe successful-response headers: token limit8,000; last remaining3,332; request limit1,000 (documented RPD), last remaining995. No provider 429 recurred, so these headers do not diagnose the earlier 429. Search/tool chain repeated three searches and two fetches. First-token event intervals included inserted pacing and are not pure provider TTFT. Cumulative after this attempt: Groq10/12; Tavily4/8; aggregate ME3 research HTTP operations7/8. The application aggregate cap counts source retrievals as well as Tavily calls. Pending cases P4-12/P4-17 not run. Full event/header measurements: ignored result `phase4-live-pilot-20261111-continuation01.jsonl`; summary/outcome in PHASE4_REPORT.md.


## Phase4 research reliability offline qualification — 2026-11-11

Python3.13.9/Linux x86_64, Pipecat1.13.0. Deterministic local MCP/mock provider scenarios; zero live requests. Focused6 tests passed; full suite594 passed with one existing deprecation warning in256.04s. Ruff passed. Tests verify tool result error/success in the subsequent native LLM context; repeat guard (same operation twice allowed, third blocked); explicit no-source failure speech with no factual text/citation; valid source/citation flow; model-call-budget terminal failure; and Groq header pacing/retry behavior. No user-facing latency, Pi CPU/RAM, or provider throughput was measured. Research answer buffering delays first speech until the final answer is citation-checked; quantify in a separately authorized live comparison. Detailed source hash and reproduction: `var/results/architecture_modernization/phase4-research-reliability.json`.

## Phase4 P4-11 bounded live canary — 2026-11-11

Evidence class: **live_api**, HPC only; mocked ASR/TTS and both project-owned MCP stdio services. Focused startup/native/MCP suite:71 passed (one existing deprecation warning); actual mock speech preflight: ASR1.963s and TTS1.863s, exit0, protocol2025-11-25. Canary timed out at120.015s with no answer. Five Groq attempts succeeded; deduplicated usage17,406 prompt/268 completion tokens. Four combined research HTTP operations were used: three Tavily searches and one failed source retrieval. The first retrieval yielded `research_http_failure`; all three searches had distinct operation fingerprints; a later retrieval was blocked at the cap. Five tool results reached subsequent contexts; retained sources0; final citations0; no factual answer could be independently checked. Successful-response first-token observations were1.002/0.470/2.179/46.391/33.557s including pacing, not provider-only latency. Last Groq header showed1,465 remaining TPM tokens and49.012s reset guidance; the sixth continuation did not authorize/send a request before timeout. Cumulative counters: Groq15/16; combined research11/11; Tavily searches7. No retry or further calls. Result and limitations: `PHASE4_REPORT.md`, `var/results/architecture_modernization/phase4-live-p4-11-canary-summary.json`.

## Phase4 research fix offline qualification — 2026-11-11

Evidence class: offline mock/local MCP, Python3.13.9/Pipecat1.13.0 on HPC. Full suite598 passed in256.32s; focused unit4 and MCP/SDK integration3 passed, one pre-existing deprecation warning. Source hashes and commands: `var/results/architecture_modernization/phase4-research-fix-offline.json`. Search snippet bound changed from2,000 to320 characters per lead, an upper bound of8,400 fewer characters for five leads on each later LLM request; no post-change provider token count or live latency was measured. Offline deadline tests confirmed terminal failure before an unaffordable paced request and after a deliberately delayed mocked model. No Pi CPU, acoustic or provider quality claim.

## Generic HTML extraction regression — 2026-11-11

Offline fixture tests6 passed; full existing-venv suite602 passed in255.30s, one existing deprecation warning. Ruff passed. Long-navigation fixture verifies navigation exceeding8KiB is excluded before applying the extracted-text cap; structured content, provenance/hash, absent article text, and a JavaScript placeholder are also checked. No live latency or source-quality rate was measured.

## Phase4 adaptive retrieval offline evaluation — 2026-11-11

Evidence class: offline fixtures/mock HTTP, HPC Python3.13.9/Pipecat1.13.0. Focused retrieval9 passed; full suite605 passed in253.02s, one existing warning; Ruff passed. Long-navigation fixture:73,700 HTML chars →200 semantic chars →200-char query excerpt/1 chunk,0 network requests. Repeated search cache:1 request for two identical calls. Basic Extract fallback:1 mocked HTTP request after local extraction inadequate. No live latency, p95, token usage, API credit, provider quality or semantic accuracy measurements. Exact source hashes/config and limits: `var/results/architecture_modernization/phase4-adaptive-retrieval-offline.json`.

## Phase4 autonomous reliability offline checkpoint — 2026-11-11

Existing-venv full suite605 passed in252.28s, one existing Pipecat warning; Ruff passed. No live provider calls. Adaptive retrieval fixture metrics are unchanged:73,700 HTML chars→200 extracted/evidence chars; two same-query calls→one mocked HTTP request; inadequate local HTML→one mocked Basic Extract request. No live Groq token, Tavily credit, p50/p95 voice answer, or Pi resource comparison was produced. Machine record: `var/results/architecture_modernization/phase4-autonomous-reliability-offline.json`.

## Native application intelligence offline evaluation — 2026-11-11

Evidence: deterministic mock on HPC, Python3.13.9/Pipecat1.13.0; real project ASR/TTS MCP stdio children, mocked ASR/TTS/LLM/research. Six turns through `NativeApplication` (direct, clarify, contextual research, ECB extraction failure, conflicting sources, historical). Separate startup3.74s. Response elapsed median20.0ms / nearest-rank p95 28.6ms (n=6); memory-output first PCM median15.9ms/max23.4ms. Stage median/max: ASR final0.84/1.48ms, mock LLM first token0.12/7.44ms, mock TTS first PCM2.58/5.21ms. These values reflect only local mocks/MCP serialization and are not live, audible or Pi latency. No event-loop/resource metrics.

Workload cost:13 model calls,7 tool calls,3 search and4 retrieve operations. Context payloads totalled34,460 serialized characters, approximately8,618 tokens by chars/4; this is only a proxy. Estimated mock provider cost$0, live costs unavailable. ECB no-body fixture correctly produced explicit uncertainty and no citation. The conflict fixture retained both differing statements and both citation IDs. Scripts selected the route and answer, so this is an integration contract, not model-quality evaluation. Baseline full regression605 passed/251.95s; post-change606 passed/261.01s, same one existing Pipecat deprecation warning. Changed Python Ruff checks pass; repository-wide Ruff still has four unrelated E401 findings and format-check lists ten unrelated files. Per-turn measurements, environment, commit and source hashes: `var/results/architecture_modernization/intelligence-offline.json`.

## Four-case Groq/Tavily live intelligence canary — 2026-11-11

Live provider, HPC, Pipecat-native app, mocked ASR/TTS inference and actual speech MCP stdio services. Offline preflight:68 relevant app/Phase4 tests pass; both MCP services preflighted with exit0. Of the four prepared prompts, INT-01 direct and INT-02 clarification passed; INT-03 fresh Bank Rate research failed safely after two search and two retrieval attempts returned no retained source; INT-04 contextual follow-up was not run after terminal failure. Case durations1.027s/1.681s/98.857s; the last reached the response-reserve guard, with five provider TTFT observations1.987/0.657/40.213/1.458/46.149s including pacing. No citation or factual answer was available to grade for INT-03.

New allocation usage7/12 Groq,4/6 combined research HTTP,2/6 Tavily Basic searches; cumulative total25/30,17/19,10/14 respectively. Deduplicated usage16,278 prompt and383 completion tokens. Mock speech; no audio-latency, Pi-resource or event-loop measure. The exact source HTTP status was not persisted; the symbolic error is `research_http_failure`. The preserved raw JSONL says budget exhausted incorrectly, while five model and two research slots remained. Evaluator terminal classifier was fixed afterward; focused regression1 passed. Details and corrected assessment: `PHASE4_REPORT.md`; raw JSONL and summary: `phase4-live-intelligence-canary-20261111.jsonl` and `phase4-live-intelligence-canary-summary-20261111.json`. Phase4 remains not live-qualified.

## Retrieval diagnostics and laptop browser preparation — 2026-11-11

Evidence class: offline project tests/source audit, HPC Python3.13.9/aiohttp3.14.5/Pipecat1.13.0; zero live API requests, no Windows browser/audio or Pi hardware opened. Focused Phase4 module **62 passed**, one pre-existing Pipecat deprecation warning. Checks cover classified HTTP/DNS/TLS/timeout/network failures without exception text, visible Tavily Basic Extract fallback outcomes, insufficient-content separation, safe status/origin/hash diagnostics through the native app, two failed source retrievals ending locally without another LLM request, and functional project ASR/TTS MCP startup. Changed-file Ruff check and format pass.

No latency, CPU, RSS, echo suppression, speech intelligibility, or barge-in score was measured. The two-retrieval fail-stop's only performance evidence is zero extra model requests after its threshold in a deterministic application fixture; this is not a live latency claim. Browser route composition does not prove actual Windows hardware. Laptop AEC and Pi PipeWire AEC remain physically unqualified.

Full existing-environment regression: **616 passed**, **0 failed**, one pre-existing `AudioContextTTSService` deprecation warning, wall time **258.54s**. Command: `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 var/cache/venv/bin/python -B -m pytest -q --basetemp=var/scratch/pytest-laptop-retrieval`. This exceeds the prior reported606-test suite because ten repository-local tests were added in intervening work; no comparison benchmark was run. Source hashes and environment: `var/results/architecture_modernization/retrieval-laptop-checkpoint-20261111.json`.

## Phase6 local action integration — 2026-11-11

- Environment: existing `var/cache/venv` Python3.13.9, Pipecat1.13.0, HPC, mocked device/music and provider paths; no live APIs or Pi hardware.
- Focused regression:57 passed across native fast-route/parser, DeviceActions and MusicController tests. Changed native browser capability/UI and legacy-route spot check:7 passed (one pre-existing Pipecat deprecation warning).
- Native fast-route test asserts RGB action uses the configured mock DeviceActions, adapter summary is returned, music pause uses MusicController, and `NativeLLM.calls == 0`. This validates a code-path/LLM-request count, not ASR-to-audible latency.
- Full regression after contextual follow-up routing coverage:617 passed,0 failed, one pre-existing Pipecat deprecation warning,274.84s with `--basetemp=var/scratch/phase6-full-final-20261111`. The unprivileged run encountered local-socket restrictions; the complete authorized offline loopback/stdio MCP run passed. No provider endpoints were used.
- Fast action benchmark:50 exact brightness commands through `NativeLLM.get_chat_completions`, mock device adapter; median0.113ms, nearest-rank p950.176ms, max12.425ms,0 provider calls. It excludes MCP, ASR, TTS synthesis/playback and audio. It does not establish voice latency.
- No CPU, PSS/RSS, event-loop lag, end-to-end audio, Pi, Spotify, weather, phone or physical readback values measured here. Machine-readable details/source hashes: `var/results/architecture_modernization/phase6-offline-20261111.json`.

## Phase6 general-purpose workspace actions — 2026-11-11

Evidence class: offline LangGraph/SQLite and local filesystem on HPC; mock model; no external calls or Pi. Final full regression **631 passed**,0 failed, one existing Pipecat deprecation warning,303.88s. Focused workspace recovery/security/routing:13 passed in the complete focused run; final steering/I/O-offload/crash focused set:3 passed. Ruff and format pass on changed files.

Thirty atomic file writes with durable SQLite operation identity and independent readback/hash verification: p50 **79.323 ms**, nearest-rank p95 **96.280 ms**, max **105.865 ms**; process CPU p50/p95 **2.073/2.183 ms**. Artifacts verified30/30. Measured path excludes model inference and complete graph planning loops. Process-lifetime max RSS **745,848 KiB** includes loaded Python/Pipecat/LangGraph stack; it is not incremental tool memory and cannot establish Pi capacity. A delayed-I/O fixture injects 150 ms into file persistence; a concurrent 20 ms event-loop tick passes after `asyncio.to_thread` offload. The prior synchronous exploratory run had one 2.1s maximum, but the sequential samples were not a controlled paired A/B; do not claim measured latency improvement. No voice/audio p50/p95, Pi CPU/RAM, event-loop production lag, site browser execution, or external effect was measured. See `var/results/architecture_modernization/phase6-workspace-benchmark-20261111.json` and `phase6-general-actions-20261111.json`.

## Phase6 independent security review — 2026-11-11

Evidence class: offline adversarial filesystem and subprocess tests on HPC, no live API or Pi. The pre-review baseline was 631 passing tests. After descriptor-relative workspace I/O and hard-link denial, the final exact-tree suite passed **633/633** in **314.51s**, with one existing Pipecat deprecation warning. Focused workspace/artifact/restart selection passed 18 tests after updating the crash fixture; the separately added hard-link case passed. Deterministic path-swap injection verified that the outside directory remained unwritten while atomic replacement and readback used the pinned original directory. Existing SIGKILL-after-replace recovery passed. Ruff check/format and `git diff --check` passed. No comparative latency, CPU/RAM, disk-quota, acoustic, or external-effect measurement was made in this review; previous workspace timing samples remain historical only.

Phase6 qualification/Phase7 preparation: focused multipage website plus path-swap, hard-link, and restart checks **5 passed** in 15.06s. Full regression after adding multipage artifact/link validation: **634 passed, 0 failed**, one existing Pipecat warning, 308.67s. The mock task wrote four actual files; every relative HTML href/src resolved locally; LangGraph readback verified all four. This is structural artifact validation, not browser/render validation. No browser executable was available; no live providers, audio devices, Pi, or external actions were used. No latency claim was added.

## Phase7 ASR append backpressure diagnostic — 2026-11-12

Evidence class: offline mocked HPC replay and fake Deepgram WebSocket, Python 3.13.9; no external providers, physical audio, laptop or Pi. Test command: `PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m pytest tests/test_native_speech.py tests/test_native_entrypoint.py tests/test_remote.py --basetemp=var/scratch/pytest-phase7-asr-final-reviewed`; **103 passed** in **156.04 s**, one existing Pipecat deprecation warning. Ruff check/format passed.

Same 9-second labeled PCM WAV, native Pipecat entrypoint, both project MCP subprocesses, mocked inference, one run each (n=1 per mode):

| Mode | Audio frames | VAD starts/stops | ASR streams/appends | Max append | Max input-frame gap | Max append queue | ASR final / turn complete |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 282 | 1 / 1 | 1 / 90 | 7.51 ms | 41.25 ms | 1 / 16 | 1 / 1 |
| Jittered MCP appends | 282 | 1 / 1 | 1 / 90 | 242.35 ms | 41.27 ms | 7 / 16 | 1 / 1 |

Jitter mode used ordinary 12–20 ms waits and selected 160–240 ms append spikes; flush-before-final took20.25 ms. Both turns yielded the expected transcript and answer, eight PCM output chunks, a valid WAV and terminal `turn_completed`. Existing sustained queue-saturation test continues to require explicit failure. Low-level service validation also wrote the exact recording through production `DeepgramASR`/`SpeechService` into a mock WebSocket, verified byte-for-byte order and a final transcript, and confirmed one SQLite reservation. Timeout, disconnect and incomplete-stream cases ended as failure/failure/unknown with safe codes; request cap blocked another reservation. Aborted native replay trace persisted frame/VAD/ASR/queue/ledger diagnostics without exception-message sentinel content.

These are deterministic software timing samples, not p95, live provider latency, acoustic capture, or audible playback. They demonstrate that a transient mocked append delay no longer serializes each audio frame behind MCP I/O. The `live-20261112-02` historical exact append code remains unknown because that run did not persist a partial trace. Machine summary: `var/results/architecture_modernization/phase7-asr-diagnostics-20261112.json`. No CPU/RAM, speaker, AEC, barge-in, or Raspberry Pi measurement was made.

## Phase7 native ASR readiness/backpressure recheck — 2026-11-13

Evidence is offline only: real project Pipecat/MCP path with mocked providers and the production Deepgram transport/SQLite logic behind a fake WebSocket. No external requests.

At the provider boundary, `SpeechService.open()` schedules one connection task and returns without waiting for the WebSocket handshake. Across injected handshake delays of 0.5/0.9/1.2/1.5 s, observed connection-ready times were 593/990/1,281/1,591 ms. The first MCP append acknowledgement took 0.05/0.05/0.23/0.11 ms; 50 PCM frames (51,200 bytes) were sent exactly once and in order before finalization. Peak pending bytes were 16,384/29,696/38,912/48,128 of the bounded64,000-byte provider queue. These measurements distinguish provider handshake delay from MCP append acknowledgement delay.

The native replay delay fixture injects latency into the MCP append call itself, not the provider WebSocket handshake. Baseline, jitter, and 200/600/900 ms append acknowledgements each processed282 frames, one VAD start/stop, mock final/LLM/TTS, `turn_completed`, and valid WAV. Maximum frame gaps were40.88/43.21/41.21/41.11/45.17 ms; peak native queue depths8/9/13/25/34 of40. A 1,200/1,500 ms append acknowledgement filled the40-slot queue at about1.125 s, then preserved a real VAD stop and emitted `asr_turn_aborted` (`queue_saturated`) rather than aborting the frame processor before endpointing. Each failure processed135 frames through the recorded pause, explicitly rejected subsequent audio (58,368 bytes), and did not produce a final answer/`turn_completed`. This is bounded overload behavior, not VAD starvation; no claim is made about actual provider/MCP tail latency.

Final affected suite: `tests/test_remote.py tests/test_native_speech.py tests/test_native_entrypoint.py tests/test_audio.py tests/test_intelligence_application.py` — **131 passed**, 1 existing Pipecat deprecation warning,248.84 s. The final corrected replay-only matrix passed **7/7** in87.46 s (`var/logs/phase7/native-asr-final-replay-check2.log`); the first renamed-fixture attempt failed only on parsing the new case label and is preserved in `native-asr-final-replay-check.log`. Ruff check and format pass. Run log: `var/logs/phase7/native-asr-offline-regression-final.log`; traces and controlled timing JSON are under `var/scratch/phase7-offline-regression-final/` (ignored, non-published).

No response-latency p50/p95 was measured. Existing reconnect tests exercise two user turns on one native session and preserve context; this is mocked application continuity, not live model reasoning or physical audio. No prompt edits were indicated.
## Phase7 live Nova-3 startup/append diagnostic — 2026-11-12

One authorized native Pipecat/MCP replay of the 9 s human WAV; downstream LLM/TTS mocked. Zero percentile claims (n=1). Preflight initialized both project MCP children in3.846 s with zero provider reservations. Input:63 frames, frame-gap p50 33.06 ms / max41.22 ms; one VAD start, no VAD stop/restart. ASR MCP open returned in6.94 ms. The first Deepgram append had not returned when the native 16-frame queue saturated563.68 ms after append start; cleanup cancelled the pending worker at590.12 ms. Result: no ASR final, no LLM/TTS, no turn completion. The local queue failure is proven; the provider WebSocket handshake outcome and usage/billing are unknown. Ledger records one `api.deepgram.com/nova-3` reservation with terminal `unknown`. See `PHASE7_PREPARATION.md`, partial trace `var/results/native-live-smoke/live-20261112-asr-01.jsonl`, sanitized summary `var/results/native-live-smoke/live-20261112-asr-01-summary.json`. These timings are HPC software/replay boundaries, not Pi or audible latency.

## Offline Phase7 Deepgram connection startup — 2026-11-12

Evidence class: deterministic mocked HPC replay / mocked WebSocket with real project `DeepgramASR`, `LiveTransport` SQLite authorization, Pipecat adapter and MCP subprocesses. Zero provider calls. Native replay n=1 per scenario; numbers below are observations, not percentile estimates.

| Scenario | Input frames | Max inter-frame gap | Peak append queue | Max append worker call | VAD start/stop | ASR final / turn complete | WAV |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 282 | 41.54 ms | 1/32 | 1.95 ms | 1/1 | yes/yes | valid, 10,284 B |
| Jittered MCP append | 282 | 41.04 ms | 7/32 | 242.71 ms | 1/1 | yes/yes | valid, 10,284 B |
| 200 ms startup stall | 282 | 41.00 ms | 6/32 | 202.42 ms | 1/1 | yes/yes | valid, 10,284 B |
| 600 ms startup stall | 282 | 41.68 ms | 18/32 | 603.93 ms | 1/1 | yes/yes | valid, 10,284 B |
| 1,200 ms startup stall | 79 before abort | 37.83 ms | 32/32 | 1,124.47 ms at saturation | 1/0 | no/no | none; explicit failure |

In the overflow case the queue rejected the next frame at1,127.69ms after VAD start; there was no VAD split/restart. The 32-frame queue therefore provides approximately1.05s total bounded audio headroom (one in-flight plus32 pending 32ms frames) and tested successful transient startup support through600ms, but intentionally fails a1.2s stall.

At the provider boundary, `asr_open` returned without waiting for mocked WebSocket readiness. Monotonic connect start→ready observations were **272.84ms** and **336.10ms** for the200ms injected delay across two runs, and **686.11ms** for600ms; saved first-append waits were **200.36ms** and **600.72ms**. Scheduler/host overhead varies, so the successful envelope is defined by bounded queue behavior rather than a strict handshake wall-time threshold. A forced startup timeout was classified `provider_timeout`/`asr_open` in **125.72ms** of test time and persisted ledger failure. Cancellation during a pending dispatched connection joined the task and persisted conservative `unknown`; repeated `start_background()` reused one task/authorization. All PCM reached the mock peer once and in order before finalization on successful paths. Timing JSON is retained under `var/results/phase7-asr-startup-20261112/provider-tests/`; full replay traces/WAVs are under `var/results/phase7-asr-startup-20261112/pytest/`.

### 2026-11-13 bounded campaign observations

Evidence: live provider ASR + mocked downstream for ASR-only; second full-live attempt aborted with unknown ASR outcome. One fixture/sample per live scenario; these are not percentile estimates.

| Metric | ASR-only live | Full-live attempt |
|---|---:|---:|
| ASR `asr_open` | 109.78 ms | 92.47 ms |
| Provider connection start→ready | 911.01 ms | no ready event before abort |
| First append→first provider send | 801.28 ms | no PCM send observed |
| Speech endpoint→ASR final | 591.80 ms | no final |
| Native frame-gap p50/p95/max | 33.07/33.99/112.09 ms | replay aborted mid-feed |
| Result | transcript correct; mocked downstream turn complete | 64 KB provider buffer filled; `asr_buffer_full`; ledger unknown |

ASR-only mock downstream endpoint→first output packet was 614.50 ms; this is not live Groq/TTS latency or speaker audibility. Cold replay process time was 13.285 s. After the failure, the bounded provider-side startup buffer was changed from 64,000 bytes (~2 s) to 160,000 bytes (5 s); fake transport validates 2.5 s handshake and fail-closed overflow. This change has no live performance qualification. Fixture and traces are documented in `PHASE7_PREPARATION.md`; machine data: `var/results/swift-campaign-20261113-e8795697/campaign-summary.json`.

### Fresh campaign connection instrumentation and full-live replay — 2026-11-13

Campaign `swift-live-20261113-19dbd8f963`: ASR-only live turn passed; full live reached ASR, Groq, and TTS but timed out before turn completion. Values are single runs, not distributions.

| Metric | ASR-only | Full-live |
|---|---:|---:|
| Speech end→ASR final | 1.042 s | ASR completed; endpoint event was evicted from 2,000-event ring |
| Speech end→first output packet | 1.065 s, mocked LLM/TTS | unavailable from preserved trace |
| Groq call time | mocked | 1.677 s ledger duration; first-token event not preserved |
| Aura requests | mocked | 4 segments, 3 success and 1 unknown |
| Replay terminal | `turn_completed`, valid mocked WAV | 15 s settle timeout; no `turn_completed` or output WAV |

The correct Groq answer appeared in logs; 4 TTS segment texts totaled477 characters. Logs showed the first bot playback start about2.1 s after Groq dispatch; this is server-side/in-memory packet evidence, not physical audibility. Actual Groq token usage was absent. The full-live trace had2,001 rows after high-volume PCM and playback events displaced the startup and ASR/LLM events. Post-run code now retains4,096 bounded events and has a30 s replay-settle deadline, both unqualified against live providers. The new `aiohttp.TraceConfig` records DNS, connector, upgrade, response and terminal timestamps; TCP and TLS are grouped by aiohttp, and these fields were not retained in the just-completed live trace due to the native diagnostic allowlist/ring issue, corrected offline afterward. Logs/traces/WAV: `var/logs/swift-live-20261113-19dbd8f963-*`, `var/results/swift-live-20261113-19dbd8f963/`.

Affected regression command (all provider inference mocked): `var/cache/venv/bin/python -B -m pytest -q --basetemp=var/scratch/phase7-asr-focused-final tests/test_remote.py tests/test_native_speech.py tests/test_native_entrypoint.py::test_labeled_nine_second_native_replay_with_delayed_mcp_appends` — **89 passed**, one existing Pipecat deprecation warning,122.83s. Ruff check and format passed on modified Python. A broader entrypoint sweep also exercised unrelated test setup failures (outputs under `/tmp` violate ME3's intentional `var/` artifact path constraint; some browser/task cases encountered `task_database_already_owned`); see final task report for distinction. No Pi CPU/RAM, live Nova-3 result, hardware audio, or audible speaker latency is inferred.

### 2026-11-13 replay-settlement follow-up

Evidence: one prior live full-turn attempt plus focused offline tests. The fourth Aura request was cancelled by the former15 s replay wait: request start to `unknown` ledger update1.933 s; first playback event to cancellation about120 ms. The runner traceback identifies `TimeoutError` while waiting for `application.settled`. This explains the missing final WAV/`turn_completed`; whether Aura would have completed uncancelled is unknown. The new30 s wait is a diagnostic allowance, not user-facing response latency.

The old process log places VAD stop at15:45:00.456 and first `Bot started speaking` at15:45:02.565: **2.109 s wall-clock endpoint-to-memory-output-start**, one sample. The first TTS segment began15:45:01.712; output start followed about0.853 s later. This is a backend/in-memory proxy from wall-clock logs, not a monotonic measurement or audible speaker result; no p50/p95 or complete-turn latency is available.

The focused offline run passed **6 tests** covering the actual replay timeout argument, two-segment TTS ordering, final PCM before `turn_completed`, valid WAV, late-error partial trace, cancellation/stale-result fencing, and retention of synthetic `asr_final`, `llm_usage`, and `tts_start` through2,890 playback events. Two additional mocked WebSocket tests passed for safe aiohttp stage capture and native diagnostics relay. Ruff passed. Logs: `var/logs/phase7-settle-followup-offline-20261113.log` and `var/logs/phase7-settle-trace-offline-20261113.log`. The prior affected full suite remains125 passed before this test-only assertion. No runtime source change or provider call occurred. The candidate campaign `swift-followup-20261113-4059fd4c00` has zero ledger rows. Its conservative executable limits are Nova2 ×30 s, Groq2 ×2,000 completion tokens and Aura4 ×500 characters, with research disabled. Aura's four-call cap limits all TTS input to2,000 characters without separate aggregate-character accounting and may support only one response like the previous four-segment answer. No live performance or prompt improvement is claimed.


### 2026-11-13 — authorized full-live replay `swift-followup-20261113-4059fd4c00`

Evidence class: **live_api, prerecorded WAV, native Pipecat/MCP; n=1; no physical speaker**. Zero-provider preflight took 3.269 s (ASR MCP init 1.606 s; TTS 1.597 s). Monotonic live spans:

| Boundary | Result | Notes |
|---|---:|---|
| Nova connection attempt→WebSocket ready | 0.911 s | DNS 0.072 s; connector establishment 0.547 s; upgrade/response 0.266 s, HTTP 101 |
| Speech end→ASR final | 0.553 s | One correct final transcript |
| Groq request→first content token | 0.834 s | 1,246 prompt/133 completion tokens |
| First TTS request→first PCM | 1.208 s | Aura-2 Thalia |
| Speech end→first PCM | 2.629 s | Backend proxy; no physical playback measurement |
| Speech end→first output packet | 2.632 s | Memory-sink proxy; above provisional 2 s target; no p50/p95 from n=1 |
| Speech end→replay abort | 34.394 s | Third sequential TTS segment was still active at runner deadline |

Nova finalized “why does ice float on water”; 78,848 PCM bytes were accepted and sent in order, zero discarded; peak provider buffer 33,792/160,000 bytes. The full Groq response was correct and 371 characters, with no research call. Aura segments were 54/145/172 characters: the first two succeeded; the third ended `unknown` after 11.794 s at replay cancellation. Ledger counts: Nova 1 success, Groq 1 success, Aura 2 success + 1 unknown, other providers 0. The trace logged 874 PCM chunks and 437 output packets, but no final WAV or `turn_completed`; this is a partial failure, not a passing full turn. Preflight, trace, process log, exit code and mode-0600 summary: `var/results/swift-followup-20261113-4059fd4c00-*`, `var/logs/swift-followup-20261113-4059fd4c00-full-live.log*`.

Estimated cost **US$0.011762** (not an invoice): Nova conservatively priced using 2.847 s ledger duration, Groq actual tokens, Aura all 371 submitted characters. Rates used: [Deepgram pricing](https://deepgram.com/pricing), [Groq GPT-OSS-120B pricing](https://console.groq.com/docs/model/openai/gpt-oss-120b). No p50/p95 or audible latency inference. After preserving the baseline, replay-only settlement was increased to 60 s and prompt ME3-VOICE-2 gives general concise-answer guidance; the prompt's live effect is unmeasured. Focused offline tests after changes: 10 passed plus one same-session browser reconnect test; Ruff passed. Full affected suite was 125 passed before these small changes and was not rerun.


### 2026-11-13 — TTS completion and delivery-rate evidence

| Evidence | PCM audio | Wall time | Ratio and interpretation |
|---|---:|---:|---|
| Historical live Aura segment 1 | 3.36 s | 5.274 s | 0.637× real time; completed |
| Historical live Aura segment 2 | 8.44 s | 15.898 s | 0.531× real time; completed |
| Historical live Aura segment 3 before settlement cancellation | 5.68 s | 11.794 s | 0.482× real time; incomplete/unknown |
| Offline mock through project TTS MCP | 0.128 s | 0.0531 s | 2.41× real time; four PCM pulls plus terminal pull |

Historical live PCM values are measured at native adapter delivery, not provider Flushed; provider-versus-MCP causality cannot be reconstructed from those old traces. New diagnostics record provider first/last PCM and flush alongside MCP pull latency and native delivery. Mock tts_next p50/p95 was 11.16/11.78 ms with 10 ms injected delay (five calls), about 1 ms additional local overhead in this fixture only. This does not predict live Deepgram or physical audible latency. The live 371-character answer and prior 477-character answer were not a paired prompt comparison; no completed ME3-VOICE-2 live response exists. Artifact: var/results/phase7/tts-delivery-offline-20261113.json. The affected suite passed 138 tests with local IPC; focused checks passed 8; Ruff clean.

### 2026-11-13 — Completed native live voice and ME3-VOICE-3 comparison

Evidence class: `live_api` native WAV replay with a memory output sink; **n=1 per prompt version**, not a latency distribution. Runs used Linux x86_64 HPC, Python3.13.9, Pipecat1.13.0, the same7.98 s mono PCM16/16 kHz fixture (`ice-16k.wav`, SHA-256 `e41b75292e4fc20b32f6dfe1dd6a76bc5f1a725660a7acb5948a210e8983137a`) and actual ASR/TTS MCP subprocesses. Source was not bound to a commit: Git resolves to parent `sandbox` HEAD `9338bc2`, and ME3 source/tests are untracked there. These figures do not represent physical speaker latency, laptop/Pi resources, or warm-start p50/p95.

| Metric | ME3-VOICE-2 baseline | ME3-VOICE-3 comparison |
|---|---:|---:|
| Model answer / TTS text | 338 / 336 chars; 3 sentences / 3 segments | 255 / 254 chars; 2 sentences / 2 segments |
| Correct direct answer; research calls | Yes; 0 | Yes; 0 |
| Prompt tokens / completion tokens | 1,279 / 79 | 1,291 / 63 |
| Speech endpoint → ASR final | 3.636 s | 0.558 s |
| Speech endpoint → first memory-output packet | 7.281 s | 4.156 s |
| ASR final → first memory-output packet | 3.645 s | 3.598 s |
| Speech endpoint → `turn_completed` | 55.155 s | 30.163 s |
| Synthesized audio duration / TTS wall | 19.84 / 50.610 s | 15.28 / 28.736 s |
| WAV and terminal event | valid / present | valid / present |

The V3 prompt reduced this matched response by one sentence, one TTS request and about24% of TTS characters; answer accuracy was preserved for this one stable fact. Time to first output did not materially improve after ASR finalization. The endpoint-to-first-packet delta mostly follows a single ASR latency outlier, so do not attribute it to the prompt. The response-end improvement coincides with fewer/shorter TTS segments; these two samples do not establish a general causal estimate.

Per-segment live timing (chars; provider PCM audio seconds; segment wall seconds; request→first provider PCM; MCP `tts_next` calls; client `tts_next` p50/p95/max):

- V2: `53; 2.84; 7.901; 2.729 s; 143; 1/259/2,732 ms`, `162; 9.96; 29.062; 2.441 s; 499; 1/280/2,442 ms`, `121; 7.04; 13.647; 3.378 s; 353; 1/272/3,380 ms`.
- V3: `142; 8.64; 17.963; 2.723 s; 433; 1/261/2,726 ms`, `112; 6.64; 10.773; 1.270 s; 333; 1/250/1,298 ms`.

All five segments completed and recorded provider `Flushed`. Median MCP pull latency was about1 ms; high-tail pulls waited for provider PCM and segment client/server pull sums closely matched provider streaming wall time. TTS output was slower than real time in both runs (V2 aggregate0.39×; V3 0.53×). This does not show a material local MCP serialization bottleneck. LLM generation and TTS synthesis overlapped: the first TTS request began25/56 ms after first token and before final stream usage. The comparison run recorded exactly one output-speaking start/stop pair and one final response stop; the baseline had an extra output stop/restart after a real3.67 s PCM gap, matching Pipecat's three-second output-queue idle fallback. The response still completed once.

Campaign `swift-tts-turn-20261113T102547Z-727f1dd9`: 2/2 Nova-3 success, 2/2 Groq success, 5/6 configured Aura-2 success, zero `unknown`; other providers0. Groq usage2,570 input/142 completion tokens; 15.96 submitted ASR seconds; 590 TTS characters; 35.12 seconds synthesized audio. Estimated cost **$0.0202** using the public rates already cited in `PHASE7_PREPARATION.md`; not an invoice or enforced dollar ceiling. No further calls were needed. The backend replay gate passes; laptop mic/speaker/UI, audible latency, AEC, persistent multi-turn and Pi qualification remain pending.

### 2026-11-13 — Native agent routing and task qualification (offline)

Evidence class: deterministic mocked providers through the real native application and both speech MCP subprocesses; no live calls. Exact routing rows are in `var/results/architecture_modernization/intelligence-offline.json`; five-turn task evidence and hashes are in `var/results/architecture_modernization/swift-agent-qualification-20261113T1939Z.json`.

| Evidence | Result |
|---|---|
| Seven-case routing matrix |7/7 pass: greeting, direct answer, clarify, contextual fresh research, safe failure on missing extraction, multi-source conflict, and direct historical answer |
| Model/tool/research totals |14 scripted model requests,7 tool calls,3 mock searches,4 mock retrieves,US$0 mock/live spend |
| Mocked response processing |n=7; median23.1 ms; nearest-rank p9536.1 ms; native app/MCP startup7.19 s reported separately |
| Same-session task |5 consecutive native turns; task started, steered to revision2, status/follow-ups; five `turn_completed` events |
| Actual local task execution |12 scripted LangGraph model calls,10 tool calls, five confined website files, independent artifact readback, links/assets/theme/project assertions passed |
| Visual render |Not run; no browser executable in this environment |
| Live provider / acoustic evidence in this session |0 calls; no new acoustic measurements |

The deterministic evaluation validates application dispatch and results, not Groq's real routing or planning quality. Conversation context survives a browser reconnect only while the app process remains alive; LangGraph task recovery after process termination is a separate tested persistence guarantee. Transcript history is not restored as chat after process restart. Simulated interruptions/wake/noise tests passed in the selected regression but do not establish physical barge-in, false activation rates or AEC.

First-output latency remains the clear measured weakness: the earlier live ME3-VOICE-3 replay was4.156 s from speech endpoint to first backend packet. TTS request-to-first PCM was2.730 s, including about2.402 s to WebSocket readiness and0.321 s from ready to first PCM. `tts_next` median was~1 ms; native PCM followed provider PCM by~2 ms. The provider transport/readiness and PCM-production time, not local MCP pull overhead, dominate. The two live prompt samples show ME3-VOICE-3 cut response text254 TTS characters and two segments vs336 characters/three segments, with correct answer preserved, but did not improve ASR-final→first-packet (3.598 vs3.645 s) and does not establish general reasoning quality. No local transport patch or unverified prewarm was made.

The official [Aura Speak streaming reference](https://developers.deepgram.com/reference/text-to-speech/speak-streaming) documents `speed` from0.7 to1.5. Swift has not tested or configured it; it could change speaking pace and full-duration latency but is not evidence of lower connection time or first-byte latency. Preserve naturalness until a separately authorized live comparison establishes benefit.

Final selected offline regression: **452 passed,0 failed,0 skipped,1 existing Pipecat deprecation warning in418.82 s**; log `var/logs/swift-agent-functional-regression-final-20261113.log`. After adding the greeting row, the two application-intelligence tests passed again and Ruff check/format remain clean. The prior 448/4 run exposed only stale test setup: missing run identity/temp ledger and missing dummy credential on mocked external bridge descriptors; all affected test files were rerun after correction. No production source changed and no provider ledger was touched.

Next live campaign proposal (not created or authorized here): Nova-3≤6 requests,20 s each (≤120 s); Groq≤8 requests and2,000 completion tokens/request (≤16,000 total); Aura-2≤16 requests and500 characters/request (≤8,000 total); no retries; Tavily0 until a separate durable campaign cap is available. Current input-token use is received after dispatch and is not pre-reserved; a proposed64k aggregate input-token ceiling remains monitored/planning-only until a fail-closed pre-dispatch guard exists. Use one fixed, initially empty campaign run ID; never reset old ledger rows.

### 2026-11-13 — Offline Aura socket-reuse benchmark and input-budget guard

Controlled three-segment replay with identical text/PCM and a fixed fake40 ms websocket handshake: prior per-segment setup made3 connections and took0.120997 s; the updated provider reused1 connection and took0.041024 s (79.97 ms setup-only reduction). Evidence: `tests/test_remote.py::test_tts_reuses_one_deepgram_connection_across_mcp_segments_and_reduces_setup_wall`; a separate localhost aiohttp fixture verified ordered independent Speak/Flush cycles and separate successful reservations. This is mock transport evidence; provider latencies and audible output were not measured after the change. Historical cold readiness remains about2.4 s and first live output about4.156 s; no claim that those improved. No connection reuse across a real provider session has yet been qualified.

Groq preflight size estimate now uses the exact serialized request body plus25% and at least128 tokens headroom, with per-request and aggregate ceilings reserved atomically in the existing SQLite ledger. Profile values32k/request and64k/campaign are estimated units, not provider-reported usage; failures and unknown requests consume the reservation. Offline results: affected suites284 passed; full suite704 passed, one existing Pipecat deprecation warning; Ruff check/format clean. No live requests, provider usage or spend.
