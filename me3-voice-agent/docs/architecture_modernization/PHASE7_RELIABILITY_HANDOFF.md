# Swift ME3: targeted native voice reliability handoff to Luna

Status: implementation checkpoint, updated 2026-11-15. The sections below retain the original evidence and acceptance requirements; each checkpoint labels its offline/live evidence. No live provider requests or physical audio tests were performed for the offline implementation checkpoints.

### Cached research answer delivery with real local inference (2026-11-15)

The existing research route is one question/tool-selection request, bounded
search and retrieval calls, then one cited final request. The final citation
repair added below is conditional and permits one correction only when a
retrieved-source answer omitted its citation/required format; it reuses the
same request-local evidence, disables tools, consumes the existing call/token
budget, and never repairs unsupported or invalid-source claims. Existing
source/passage identity, provenance and contradiction checks remain in place.
There is no evidence of an extra model request after a successfully finalized
answer; the earlier duplicate terminal-choice issue remains fixed.

Offline tests: **5 passed** for the one-repair path, request/call caps,
local-loopback configuration, and complete mocked native Michigan-shaped
answer. Ruff lint passed. `ruff format --check` still reports formatting drift
in pre-existing sections of `native_llm.py` and `test_phase4.py`; no whole-file
format was applied. No external providers were called.

An actual local Ollama Qwen 3 0.6B request was attempted through the native
Pipecat pipeline with the archived Japan Prime Minister's Office source made
available through a cached research client. The request reserved 13,585
estimated input tokens, produced no first model token/tool decision, made zero
cached search/retrieval calls, and ended at the native 110.023 s turn deadline.
There was one local `dispatched_unknown` ledger reservation, zero
`turn_completed`, and only the normal error fallback TTS segment. The local
ledger is isolated; no external provider reservation occurred. The last
observed Ollama runtime context was 4,096, materially below the request's
conservative estimate. CPU/runtime saturation or context mismatch is the
leading explanation, but the server-side cause is not proven. A separate
direct Ollama call had generated one 44-token source-related sentence in
27.1 s (1.75 tokens/s); this confirms local model execution only and is not a
native answer pass. The archived cache contains one Japan source and no
Michigan source. The requested six-case native acceptance was therefore not
achieved. Artifacts: `var/results/native-qwen-cache-37c25f0e3b90.json` and
`var/experiments/032-native-qwen-cached-research/`.

Next meaningful step is to provide a local runtime/model whose context and
CPU/GPU throughput can handle the actual native serialized request, and to
assemble genuine cached evidence for more than one topic. Do not raise
turn/provider timeouts or fabricate/cache-miss citations to make this offline
qualification appear to pass.

### Japan/Michigan final-answer and contradiction regressions (2026-11-15)

Reviewed Japan run `native-99a23d58112e4005a3e06a7d6afaa4a6`, session
`fe77dd90258341a6a51db2f5e7783d6b`, and Michigan run
`native-98b14f5a698f4cc481a1b2d1bc841841`, session
`91f2396e27c54242adec992eb969751a`. These were different failures. Japan
completed search and source retrieval, then dispatched its final model request
(estimated 20,633 input tokens under the 32k request estimate ceiling). A new
speech turn started while it was pending; neither an answer nor a terminal
turn event was archived. The evidence does not distinguish a slow provider
from supersession as the first cause. Michigan's final answer cited retained
S1 but was rejected as `contradictory_claim`. Its trace did not persist the
validator's contradiction codes or raw source/claim text, so the exact
predicate that rejected this live answer cannot be recovered.

Two narrow corrections address demonstrated failure modes without relaxing
provenance. `NativeLLM` now scopes numeric/date and polarity contradiction
checks to the claim's most relevant sentence(s), instead of treating an
unrelated date or negation anywhere in a long passage as a conflict. Retained
source/passage ID, evidence ownership, citation, and high-risk checks remain
required; lexical mismatch is not treated as semantic proof. Per-request LLM
terminal diagnostics now include sanitized request ID, owner turn, elapsed
time, and complete/canceled/superseded/failed outcome. When a new turn replaces
a still-pending answer, the application records that prior response as
interrupted and emits one `turn_interrupted`; response terminal outcomes are
deduplicated so late completion/failure cannot create a second terminal.
Sanitized contradiction codes are now projected to the native journal without
claim or article text.

Offline qualification: eight focused tests passed, covering (a) a long
Michigan-style passage with a supported claim plus unrelated date/negation
distractors, while still rejecting wrong date and polarity; (b) complete
mocked native search/retrieval/cited-answer/TTS delivery with exactly one
`turn_completed`; (c) a Japan-shaped pending-final request superseded by a
second utterance, then one complete contextual follow-up; and (d) citation,
passage, streaming and sanitized-diagnostic regressions. Ruff passed. The two
native subprocess integration cases required command-scoped local stdio IPC;
no external providers were contacted. Tests use mocked search/model/audio
boundaries, not the original live payloads. The Michigan historical false
rejection is consistent with the corrected whole-passage predicate but cannot
be proven because its diagnostic codes were dropped. The Japan regression
proves explicit cancellation/supersession and recovery; it does not establish
whether the historical provider would eventually have returned. One controlled
live qualification is reasonable after fresh authorization; live source
validation, provider latency, audible TTS quality, and physical playback remain
unverified.

### Recurrent live research stop after two failed fetches (2026-11-15)

Latest laptop turn `native-9bfbedd9586d4e4f962c5460f4b34f65` did select
research, returned five search leads, and retained official source `S1` with
2,371 evidence characters. On a later model round it tried an encyclopedia
page; that request returned HTTP 403 and its automatic alternate also returned
403. `NativeLLM` then terminated at `research_retrieval_failures_exhausted`
because its default failure limit was 2, although the same turn permits five
retrieval attempts and other search leads remained. This was a local guard
mismatch, not missing Tavily credentials, a failed initial search, or failure
to retain all evidence.

The failure guard now reads the existing five-fetch turn cap directly; there
is no separate configurable failure threshold that can drift from it. Search,
retrieval, model-call, evidence-token and durable provider budgets are
unchanged; failed fetches still consume retrieval slots and the guard still
stops after five failures. Two native replay regressions reproduce the same
sequence: retain S1, then fail a later page and its automatic alternate. One
path now answers with a validated citation to S1; the other retrieves a
remaining eligible lead and answers with its validated citation. Both assert
preserved evidence/leads in model context and `turn_completed`. The 24-case
research-only selection and both recovery cases passed; Ruff check passed. The
default sandbox attempt could not initialize local speech MCP stdio and failed
at `initialize`; tests passed with the previously qualified command-scoped
local IPC permission. No live provider calls were made during this fix. The
latest live ledger had one successful Nova request, three successful OpenRouter
requests and two successful Aura requests. A live research retest is still
required; this correction cannot guarantee source quality or model routing.

### Bounded structured-answer recovery (2026-11-15)

Latest browser run `native-660f8518b1ed4712a620e972a4eab0ca` selected live
research, completed one search and one source retrieval (`S1`; 2,371 projected
evidence characters), then failed the grounded final with
`invalid_structured_answer` (63 response characters, sanitized trace). The
first Japan utterance's search was cancelled when a new utterance began; it was
not a successful answer attempt. The final raw model payload is intentionally
not persisted, so the exact malformed shape/refusal cannot be established.

`NativeLLM` now accepts ordinary prose only when it explicitly cites a retained
source and the citation resolves to a retained, projected passage; the existing
claim validator still runs. Missing citations and malformed structures get at
most one correction continuation using that evidence, with tool selection
disabled. The correction consumes the normal model-call allowance, input-token
checks and durable provider reservation; it is not a transport retry. No
citation is invented and unsupported/contradictory output still fails closed.
Six focused offline verifier/SDK-stream checks passed; Ruff passed. No live
provider call was made. This supports cited natural-language responses, but
does not establish that the next live model output will validate or prove
semantic entailment; verify one live research turn before claiming that.

### Failed-source recovery correction (2026-11-15)

Native quick research now makes one automatic attempt from the next untried,
query-ranked search lead after an eligible retrieval failure (HTTP failure,
timeout/connect failure, oversized or insufficient/unsupported source content).
It preserves the failed primary's sanitized extraction diagnostics and records
the alternate's exact source ID/provenance for the existing citation verifier.
URL-policy failures do not trigger failover. An alternate consumes the existing
per-turn retrieval reservation and HTTP calls remain charged to the configured
ResearchClient request cap. When only primary and alternate direct fetches fit,
the primary's optional Tavily Extract fallback is deferred. After research has
started, unrelated device/tool calls are hidden and rejected for that turn.

Focused tests cover HTTP 403 plus failed Extract, oversized-source recovery,
alternate success, all available leads failing, request-cap preservation, and
rejection of an unrelated device action. Exact command with local MCP stdio IPC:
`PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m pytest -q tests/test_phase4.py -k 'native_retrieves_alternate_lead_before_research_continues or two_failed_source_retrievals_end_turn_without_another_model_request or retrieval_failure_preserves_original_and_extract_diagnostics or oversized_source_uses_one_bounded_basic_extract_fallback or retrieve_can_reserve_http_headroom_for_an_existing_alternate or retrieval_alternatives_exclude_failed' --tb=short` — **7 passed**, 71 deselected. Ruff check passed on the four affected Python files. The restricted IPC attempt timed out during MCP initialization; the command-scoped IPC run passed. No provider requests were made. Real Tavily availability, source quality, citation correctness, and browser delivery remain live-only.

### Evidence-to-answer forensic update (2026-11-14)

#### Operator retest after this offline checkpoint

Run `native-a10288ac2a2b439fbcfd638d5b9a71e6.jsonl`, session `698a174ac9a34ec18ec868cc3691ef3a`, re-used run ID `swift-texas-senate-6d9a0b2f`. Nova recognized “who won the us senate election in michigan”; search returned five leads; one source retrieval completed and the context reported one retained source. The third Groq request was recorded `provider_rate_limited` in the durable ledger. The operator interrupted shortly afterward; no final answer, citation-check event, `turn_completed`, or final-answer TTS was produced. The prior successful acknowledgment TTS is not a completed research answer. Ledger delta: +1 Nova success, +2 Groq success/+1 Groq failure, +1 Aura success; no uncertain requests. Do not count this as a citation test or repeat it as such.

At that checkpoint, the bridge journal projection did not preserve the newly added evidence diagnostics: `native_bridge._safe_event` only forwarded a narrower allowlist for `tool_completed`, and generic list fields from `llm_citation_check` were dropped. The raw run therefore showed retained-source counts but not source ID/hash/character count or parsed citation IDs; this was a diagnostics integration gap, not source loss. The bridge was stopped cleanly. No subsequent live requests were made.

**Journal projection correction (offline, 2026-11-14):** The defect was one layer after event creation: `NativeApplication` and `NativeLLM` emitted the provenance/citation fields, but `NativeEventJournal.write_new()` serializes only `_safe_event(row)`. That sanitizer intentionally discarded nested `tool_completed.diagnostics` except for its older research allowlist and discarded list-valued citation fields. `_safe_event` now projects only validated source ID, 16-character URL hash, extraction method/status, evidence character/chunk counts, and syntactically valid `citation_ids` / `retained_source_ids`. Article text, credentials, and raw model response fields remain omitted; journal serialization and sequence ordering are unchanged. The existing oversized-source research fixture now writes its emitted events through `NativeEventJournal` and reads the JSONL back, asserting both diagnostic groups and absence of article text. Final focused command: `PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m pytest -q --tb=short tests/test_native_entrypoint.py::test_native_event_projection_excludes_secrets_and_free_text tests/test_intelligence_application.py::test_search_keeps_retrieval_model_owned_and_oversized_source_falls_back` — **2 passed** using command-scoped local MCP IPC; Ruff check and format check passed for the three changed Python files. The focused research fixture still reports the existing Pipecat `AudioContextTTSService` deprecation warning. No live providers were called.

Trace `var/results/native-1994bd9175b34d42814baf5866c209f7.jsonl`, session `9e3628ec1eee478e85ba63c35456522b`, is the single Michigan browser turn after the stronger citation wording. It proves this sequence: `web_search` returned five discovery leads; the first `retrieve_source` failed with HTTP 403 and its Tavily Basic Extract fallback failed `source_too_large`; a second model-selected retrieval succeeded and retained one source; the following model context reported that source and no context compaction; the final research answer contained no citation recognized by the validator and was replaced with the safe `insufficient_verified_evidence` response. Thus the source was not absent from the context, and the guard did not reject a valid citation in the persisted evidence: it rejected an answer with no recognized citation. The guard’s decision is correct. The trace does not retain the extracted passage, its source ID/URL, or raw Groq answer, so source relevance and a nonstandard citation-formatting case cannot be resolved retrospectively.

Minimal sanitized diagnostics were added to existing `tool_completed` and `llm_citation_check` events: retrieved source ID, URL hash, extraction method/status, evidence character/chunk counts, retained source IDs, parsed citation IDs, invalid-citation flag and answer character count. Article text and provider payloads remain excluded. The existing integration fixture now asserts alternative retrieval provenance/evidence length and that `[S1]` is recognized while unrelated retained source `S0` is not substituted. Focused command: `PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba var/cache/venv/bin/python -B -m pytest -q --tb=short tests/test_intelligence_application.py::test_search_keeps_retrieval_model_owned_and_oversized_source_falls_back` — **1 passed** with command-scoped local MCP IPC; Ruff check and format passed on the four affected Python files. No live calls were made for this update. One existing Pipecat `AudioContextTTSService` deprecation warning remains.

Follow-up offline check (same date): no model-facing source-ID/parser mismatch reproduced. `ResearchClient.evidence_for_model` carries `source_id` and provenance metadata alongside bounded relevant text; retrieved evidence is query-filtered to at most 2,400 characters before model submission. `NativeLLM.observe` recognizes exact `[S<number>]` citations, buffers citations split between stream chunks, accepts only IDs in the retained source set, and rejects unknown IDs. Retrieved-evidence compaction preserves source ID, URL, evidence status and text hash. Existing checks `test_search_keeps_retrieval_model_owned_and_oversized_source_falls_back`, `test_stream_fencing_citations_and_budget` and `test_retrieved_evidence_compaction_preserves_citation_provenance` passed together: **3 passed** with local MCP IPC; no providers called. No code or limit changes were justified.

The final Groq request began at monotonic `2270391.261`, received response headers at `2270443.048` (51.787 s), and emitted its first token at `2270443.461` (52.200 s). This localizes the wait before response headers; it does not distinguish Groq service delay from network transit. The safe trace contains no usable quota-header values. There is no evidence that source compaction caused this delay: no `llm_context_compacted` event occurred. The four calls estimated 11,874 / 16,513 / 19,805 / 24,512 input tokens; provider usage was 1,843 / 2,762 / 3,373 / 4,218 input tokens (12,196 total), with 227 completion tokens total. The selected ignored backend profile actually enforced 32,000 estimated tokens per request and 500,000 aggregate, not the historically described 12,000 serialized-input limit. The estimator is an intentional fail-closed upper proxy: `serialized_request_bytes + max(128, ceil(bytes × 0.25))`; it is not a tokenizer and substantially overestimated this payload (the final estimate was 5.8× provider-reported prompt usage). Keep it as the existing pre-dispatch safety bound unless a tokenizer-compatible estimator is validated; do not tune it from one run. In this trace every provider-reported per-request input was under 12,000, but aggregate reported input was 12,196. The 19,609-byte final request exceeds 12,000 bytes. Thus “12k” could mean three materially different policies: conservative estimated tokens per request, provider-reported aggregate tokens (not enforceable before dispatch without a reserve policy), or serialized bytes. The active configuration implements none of those at 12k. Do not change limits until the operator identifies the intended unit/scope; then enforce that exact policy without calling bytes “tokens”.

The compacted request sizes grew from 9,499 to 19,609 bytes as tool results were added; actual prompt usage rose from 1,843 to 4,218 tokens. The retrieved source payload itself is already bounded/query-relevant, and the trace has no compaction event. This is expected tool-loop context growth, not proof of needless article duplication or truncation. No extraction/context simplification is justified from this trace alone.

**Remaining gate:** the previously missing source provenance and parsed citation IDs now survive sanitized journal projection, but source relevance and citation behavior still need a separately authorized live turn; the earlier failed run cannot answer them because raw article/model payloads are intentionally not persisted. The later turn was rate-limited before citation validation, so no further prompt-only change is justified. One bounded live bridge retest remains conditional on the operator confirming the intended 12,000-token interpretation and the selected profile enforcing it. Preserve the safe abstention and existing research/provider accounting.

## 0. Offline implementation checkpoint (2026-11-14)

### Browser/session integration correction

The recorded session `b4ed89d5f029400183e89d397ae8fe34` reproduces a browser contract defect without providers: Python accepts its interrupted, generated-only assistant message, while the old JavaScript validator rejects it with `Invalid delivery prefix`. The browser then displays `Updates unavailable. Showing the last report.` The validator now requires complete playback confirmation only for final messages, preserving prefix checks for interrupted/error/partial messages. The exact archived transcript rejects before and passes after the correction. The current saved snapshot (92 reports, one safely excluded invalid report) also passes the actual JavaScript validator.

Native `/` and `/api/snapshot` previously scanned/rendered reports synchronously on the audio event loop. Measured scans of the current report directory took 0.173–0.492 seconds. Both operations now run in `asyncio.to_thread`; the regression holds a scan pending while verifying the health endpoint remains responsive. This removes demonstrated blocking work, but does not establish the sole cause of the latest ingress-backlog disconnect.

One actual local scan/heartbeat comparison recorded a maximum event-loop gap of 0.5333 seconds synchronously versus 0.0110 seconds with the scan offloaded. Raw measurements: `var/results/phase7-report-scan-timing.json`. This is local report-I/O timing, not live voice or acoustic latency. TTS transport connection-stage numbers were already collected but discarded by the bridge's journal projection; the journal now retains allowlisted DNS, connection/upgrade and cancellation/failure timestamps. Arbitrary text, headers and credential-like fields remain excluded. This supplies missing evidence for a further diagnostic without changing provider timeouts or claiming provider reliability is fixed.

Delayed-append qualification also exposed false replay success: live recovery could clear a failed turn after subsequent silence. Replay now rejects any recorded `turn_error` before producing successful output, preserving the partial failure trace. Live turn recovery behavior is retained. Test task databases are isolated from the operator bridge; acknowledgment ordering is checked against actual answer synthesis, not early response-context reservation. No model/provider choices are live-qualified by these mocks.

Use the existing full `swift-laptop-sustained` profile for future operator sessions, retaining the complete registered tool set and research enablement. Updated limits: Nova 20, Groq 60, Aura 60, research 20; serialized model input 32k/request, 500k aggregate. These are finite session limits, not a hard dollar ceiling. Tavily is loaded from the standard private provider-key file when research is enabled. No further intermediate profile is needed.

Live-test non-redundancy gate: `swift-research-20261114T104052Z` never invoked research; it consumed Nova 2 success/1 unknown, Groq 2 success, Aura 1 success/2 unknown and failed before first PCM on the final TTS request. `swift-research-20261114T105236Z` also never invoked research and has zero provider reservations; it ended in standby with an ingress backlog. These are not two failed research answers. Do not repeat the same five-second TTS first-audio setup solely because request caps changed. A further live attempt must target a corrected boundary or collect missing provider-startup evidence; unknown reservations remain consumed.

Verification completed: a complete offline sweep finished **761 passed / 5 failed** in 472.23 s (`var/logs/phase7-session-reliability-final.log`, XML under `var/results/phase7-session-reliability.xml`). Corrections cover all five observed failures: a missing test-fixture schema field; two overload cases whose replay failure identity was lost; raw exception text leaking into a partial trace; and a cancellation-during-reservation race leaving an active ledger status. The last now joins authorization and records terminal `unknown` without dispatch or refund. A new deterministic real-SQLite regression proves the consumed slot cannot authorize again. Free-form exception messages are excluded; symbolic classes, stages and boundaries remain.

After production corrections, all seven affected suites ran to completion: **282 passed / 2 test-fixture failures** in 330.87 s (`var/logs/phase7-reliability-affected.log`). Those two fixtures were corrected without further production changes: match the symbolic budget-exhaustion code, and model terminal completion for an artificial reservation ID. The entire transport/ledger suite then passed **96/96** (`var/logs/phase7-ledger-reliability-final.log`). Thus every one of the **284 affected cases** has passing post-correction evidence across these runs; do not label this a single final full-suite PASS. Dashboard 17/17, bridge/protocol 23/23, and delayed-append/overload 3/3 also passed independently. Ruff and JavaScript syntax checks passed. One existing Pipecat deprecation warning remains. No provider, physical audio or research API request occurred. Live Aura first-audio reliability, current-search model decisions and acoustic stutter remain unqualified.

### Dynamic warm session and combined qualification (2026-11-14)

This checkpoint supersedes the two Luna tasks listed in the read-only review below; retain that review as the rationale. The native bridge already starts both MCP children, constructs the Groq client and starts LangGraph before wake, and retains those local resources across connected turns. The new wake callback starts the same configurable inactivity lease used by manual Arm. A wake with no speech now returns to standby; processing and speaking remain protected. Browser playback ACKs extend the lease through reported queued audio, so a long scheduled answer does not time out merely because backend TTS has completed. A valid next turn resets the lease; interruption clears the prior playback estimate. No speculative ASR stream, Groq inference or Aura synthesis is dispatched by wake/Arm. ASR and Aura provider sockets remain demand-driven and accounted for; their live cold/warm improvement remains **UNVERIFIED**.

The existing native portfolio integration case now runs **nine consecutive turns in one `NativeApplication`** using both real local speech MCP subprocesses, a scripted external model, mocked search/ASR/TTS providers and an actual confined LangGraph workspace: stable direct answer → slow research with one failed retrieval and one alternate source → website task creation → dark-theme/add-project correction → status/context follow-ups → interruption after TTS starts → recovery reply. Eight turns finish and the deliberately interrupted turn does not. The test asserts one search, source `S1`, one queued research cue, no progress TTS start after the answer starts, unchanged canonical user history, one start/steer receipt each, five verified files, valid local links, revised task status and no stale light-theme write. Files remain under `var/scratch/pytest-multiturn-ack-fenced/.../task_workspace/.../portfolio/`; `index.html`, `about.html`, `projects.html`, `contact.html` and `assets/site.css` were read back. Network/VPN/DNS/proxy restrictions were not involved in those local effects and remain a separate **CONDITIONAL** deployment concern.

The stronger case initially reproduced an actual cue-order defect: `tts_progress_started` followed `tts_response_started` (event positions 75 and 70). `ProgressTTSSpeakFrame` now carries its owning turn; `McpTTSService.process_frame` checks a callback against the current turn, settled/error state and whether that turn's answer already started. A delayed cue is suppressed before synthesis when the useful answer has won the queue. The corrected nine-turn case passed in 13.45 s; six focused acknowledgment/wake/playback-lease checks passed in 4.02 s, and four browser Arm/reconnect/idle checks passed in 18.24 s after isolating one test's SQLite path. Ruff check/format passed on the changed Python files. One earlier nine-turn attempt timed out after 25.50 s before its later passing reruns; that nondeterministic result is an unresolved reliability signal, not a production root cause established by this test. The existing Pipecat `AudioContextTTSService` deprecation warning remains.

Local setup timing from the same mocked native case: initial application/MCP readiness **4.441253 s**; calling `NativeState.start()` again with the same owner **0.0000107 s**. This proves local resource reuse and removes repeated MCP initialization from warm turns; it does not measure provider DNS/TLS/WebSocket startup, first useful audio, or audible latency. No new network prewarm or provider idle-policy change was made. The existing 20-second follow-up setting remains configurable; silence or browser keepalives do not extend it. Long background LangGraph work keeps its runtime resources, while the wake gate may return to standby after idle and can be activated again.

| Scenario | Offline result | Live/physical limit |
|---|---|---|
| Direct → research/retrieval recovery → website correction | **PASS**, one persistent scripted session and actual confined files | Autonomous Groq routing and Tavily source quality **UNVERIFIED** |
| Research cue versus final answer | **PASS** after reproduced order fix; late queued cue is suppressed | Aura cue cost/timing and audible overlap **UNVERIFIED** |
| Spoken interruption → next turn | **PASS** with mocked TTS and response identities; no completion for interrupted turn | Acoustic barge-in and stale speaker audio **UNVERIFIED** |
| Wake/Arm, repeated turns, inactivity/queued playback expiry, reconnect | **PASS** focused offline | Wake recall and speaker playback **CONDITIONAL** on operator test |
| Complete response latency and TTS smoothness | **UNVERIFIED** by this change | Historical backend first-packet proxy remains above the ≈2 s warm audible goal |

**Next gate:** one separately authorized short live browser session with a fixed run identity and checked provider/research caps. Observe direct answer, dated source-backed research/follow-up, local task receipt and correction, one interrupted answer and recovery; stop on unknown provider outcome or terminal TTS failure. Record speech end→first backend answer packet, browser scheduled onset, operator-heard onset, completion, and any PCM scheduling gaps separately. Do not call this a full-duplex/acoustic or Raspberry Pi pass.

### Fine-tuning review: multi-turn readiness (read-only, 2026-11-14)

**Verdict: CONDITIONAL.** The current implementation can run a persistent Pipecat session with real local LangGraph workspace effects, and older live trials completed individual Nova→Groq→Aura turns. The available evidence does **not** prove one autonomous, live-provider simple→research→task→interruption→recovery conversation. The latest 35-pass selection is focused boundary evidence, not that combined scenario. Git resolves to the parent `sandbox`; `git ls-files` and `git diff` returned no tracked ME3 paths, so this review used the current files, prior checkpoint and test log rather than claiming a meaningful Git diff. No tests or provider calls were run for this review.

**Actual native route inventory.** `native_app.py:register_tools` registers `conversation_tools.DEFINITIONS` plus `task_runtime.conversation_schemas`; research schemas are exposed only when live research is permitted and budget remains (or retrieved sources exist for verification). `native_app.py:try_fast_action` handles clear ME2 commands before Groq. The labels below describe evidence, not planned capability.

| Route from actual registry | Current path | Evidence and limit |
|---|---|---|
| Wake → Nova ASR → direct Groq reply → Aura/UI | Native Pipecat and both speech MCP children | Individual live replay turns completed; laptop sessions show wake/ASR/playback but recurring TTS pacing/errors. No full mixed-route persistent live pass. |
| Clock, clear lights/brightness/color, timer/alarm/reminder/weather, music controls | `fast_commands.py` → `try_fast_action` → ME3 action/music executor and receipt; ambiguous variants can clarify or reach Groq tools | Focused offline route/receipt tests; time and some device interactions observed live. Light/hardware outputs can be simulated or adapter acceptance; Spotify audibility and Pi GPIO readback unverified. `ACTION_TITLES` also lists temperature and demo call/message; these are not qualified real external effects. |
| Model-selected time, mock light, device action, music | `conversation_tools.DEFINITIONS` → Pipecat tool callback | Registered and argument-validated; real Groq autonomous selection across mixed turns remains unqualified. |
| Fresh research | `web_search`, `retrieve_source`, `verify_evidence`, dynamically advertised; Tavily/search and bounded local/Basic Extract path | Offline native search-failure/alternative-lead and grounding tests pass. Latest cited laptop research failed at retrieval; no complete live source-grounded answer or follow-up citation pass. |
| Complex task and correction | `start_task`, `task_status`, `steer_task` → durable LangGraph, confined workspace, progress/receipts | Five turns in one native application with actual local website writes, steering, stale revision rejection and five terminal events passed offline; voice router and providers were scripted/mock. Live Groq task choice and browser reconnect during active task remain unqualified. |

**Continuity and failure findings.** Request-local compaction preserves canonical messages and tool-call pairs (`conversation_context.py:prepare_context`, `native_llm.py`); response records/projected transcript distinguish generated, interrupted and completed output (`native_app.py`, `native_bridge.py`). Existing tests cover barge-in/late ASR rejection, fast-action receipts, task stale-write fencing, and same-connection recovery after a turn error. They do not combine these with live research/task routing and browser speaker interruption. `research_operation_with_ack` starts work before a delayed Pipecat progress frame, but only offline checks establish one cue and separate answer identity; synthesis of the cue adds provider cost and could compete with a fast-arriving final answer. TTS timeout diagnostics now survive failed streams, yet live speech smoothness and terminal reliability remain unqualified. Provider unknown outcomes remain consumed; neither retries nor fallbacks should replay partially spoken or consequential operations.

**Warm-session review.** The bridge retains the native application, context, MCP children and provider adapters across connected turns; `native_app.py:_arm_followup_timer` begins the 20-second follow-up window after settlement and checks listening/no active STT before standby. TTS reuses one socket only for the finite idle policy in `remote.py:_reuse_idle_seconds` (about four seconds with the default five-second timeout); ASR opens one accounted stream per utterance. There is no evidence that wake-triggered provider preconnection is safe, billable-free, or faster in the real environment. Preserve the current lifecycle; measure cold versus warm stage times before any prewarming change.

**Small Luna tasks, in order.**

1. **P0, insufficient combined evidence:** Add one offline native-browser scenario in a *single session* that goes direct fact → mocked current research with failed first retrieval and viable second lead → accepted website task → follow-up steering → interruption during spoken result → recovery. Assert canonical history/turn IDs, one progress cue, paired tool results, one verified artifact revision, no duplicated effect, and truthful final transcript. Reuse existing fixtures and local MCP children; no new harness. Scope: `tests/test_native_entrypoint.py` plus existing Phase 4/5 fixtures. This is the smallest integration proof before live spending.
2. **P0, credible demo risk:** Exercise progress cue and final answer overlap through the real native output queue with delayed mocked TTS. Assert cue never arrives after interruption or after the useful answer is already speaking; preserve distinct response IDs and `turn_completed`. If reproduced, fence the queued cue by turn/response generation in `native_app.py:research_operation_with_ack` / `native_speech.py:ProgressTTSSpeakFrame`. One focused native test; rollback by disabling only the delayed cue, leaving research dispatch untouched.
3. **P0, insufficient live evidence:** Use one separately authorized short live sequence: direct question, natural recent-event research plus “Which source supports that?”, then a safe local task request/correction in the same browser session. Require model-selected routes, actual retrieved-source support, task receipt/file verification, complete TTS and one terminal event per completed turn. Stop on unknown ledger outcome or terminal TTS failure. Before launch, verify the fixed profile, secrets presence and aggregate caps; do not inherit historical run allowances.
4. **P1, demonstrated latency risk:** For the same short sequence, compare speech-end→first backend answer packet, browser scheduled playback onset, and operator-heard onset; record ASR finalization, Groq TTFT, Aura connection reuse/setup, PCM gaps and browser queued audio. Target warm useful audio ≲2 s; do not count wake chime or research cue as a factual answer. Change TTS idle policy or browser buffering only after a reproducible boundary is identified. One matched cold/warm comparison is enough initially.

**Progressive gate:** First the combined offline scenario must show exact expected route/effect/terminal state at every step. Next a single live persistent browser sequence must produce accurate direct and sourced answers, a verified local artifact, contextual correction, and clean recovery without duplicate actions; record every failure rather than score cancellations as normal completion. Finally qualify acoustic barge-in and AEC separately on laptop, then Pi. The existing laptop AEC PASS applies only to its documented setup and does not prove this multi-turn sequence.

| Handoff task | Offline implementation status | Remaining qualification |
|---|---|---|
| 1. Canonical context | `prepare_context()` and NativeLLM budget compaction use request-local deep-copied projections; complete tool-call/result IDs are retained together. The canonical Pipecat context is not rewritten for provider budget enforcement. | Targeted tests cover unchanged canonical messages, tool pairing, budget compaction and actual native follow-up research. A live long-history turn remains unqualified. |
| 2. TTS timeout/evidence | Provider receive deadline sits inside the existing MCP call ceiling; failed/cancelled operation diagnostics are captured before cleanup. Receive wait is recorded even on timeout; safe provider and MCP fields reach the native journal. | Offline timeout, partial-stream and cancellation tests pass. Provider error behavior and live completion still need a separately authorized run. |
| 3. Response projection / speech | Generated-but-uncommitted interrupted output is projected as interrupted, not as completed playback. Markdown emphasis, citations, URLs and percent symbols are normalized at the TTS boundary while display text stays intact. | Formatting is unit-tested. Stutter and physical audio quality are not resolved or established by backend tests. |
| 4. Research | Failed/already retrieved leads are excluded from the next candidate set; one bounded Basic Extract fallback is available for eligible retrieval failures, with sanitized original/fallback diagnostics. Equivalent search calls share identity. | Mocked route/retrieval checks pass. Real Tavily retrieval, source sufficiency and end-user latency remain unqualified. |
| 5. Acknowledgment | A slow native search starts immediately; after 650 ms a single “I'll check that.” progress frame may use the existing Pipecat TTS output path. Immediate results suppress it; interruption cancels the pending operation/timer. Progress audio has a separate event identity and cannot complete or replace the final response. | Delayed native mock integration and lifecycle tests pass. This adds one possible TTS synthesis on a slow research turn. Durable task start already returns acceptance promptly and uses its ordinary model response; no separate synthetic task chime/ack was added. Live timing/audio remains unqualified. |
| 6. Cold/warm measurement | Existing bounded Aura connection reuse and connection-reuse/setup diagnostics are covered by the fake-peer test. In this run, three fake 40 ms handshakes measured 0.121282 s when cold versus 0.041148 s with one reused connection: 0.080135 s less setup wall time across three segments. | This is a deterministic offline setup-only result, not a Deepgram or end-to-end latency improvement. No live cold/warm call or TTS idle-policy change was made. |
| 7. Stutter telemetry | Browser playback acknowledgments now carry bounded schedule-gap, queued-audio, AudioContext state and arrival-time fields into the sanitized session journal. | Still lacks physical speaker-onset evidence and a complete browser underrun counter. Stutter is **not resolved**; packet/playback schedule proxies are now more diagnostic. |

Focused final command selected 28 pytest nodes from `tests/test_phase4.py`, `tests/test_native_entrypoint.py`, `tests/test_remote.py`, and `tests/test_speech.py`, with `-s -q --basetemp=var/scratch/pytest-phase7-targeted-final`; complete output is in `var/logs/phase7-reliability-targeted-final-20261114.log` — **35 cases passed** in 77.93 s (some nodes are parametrized). It ran with local IPC for offline native/MCP and fake-WebSocket paths. The one earlier failure in the same group was `task_database_already_owned`: a native research-loop test shared the default SQLite database. It now uses a unique `tmp_path` database and workspace, and its isolated rerun passed. A separate first attempt at the browser-journal case hit the sandbox's localhost-bind `PermissionError`; rerunning only that mock case with command-scoped local IPC passed. The final grouped run, including both cases, passed. Ruff check and format check passed for the 13 affected Python files. One existing Pipecat deprecation warning (`AudioContextTTSService`) remains.

Primary latency decision: the engineering target remains **end of user speech → first meaningful assistant audio heard**, approximately ≤2 s warm for a simple interaction. No measurement in this offline sprint reaches a speaker. The observed historical backend first-packet value of 4.156 s remains the only comparable live proxy in this evidence set; browser playback onset is a proxy too. Do not count the progress acknowledgment, TTS setup, or full-answer duration as the primary metric. For this sprint, no provider-side latency change is claimed and no additional timeout/buffer was added.

Readiness: **CONDITIONAL / NO-GO for claiming smooth or ≤2 s voice**, but offline backend lifecycle and focused research/context boundaries pass. A short, separately authorized live qualification is still needed to verify real source evidence, provider TTS completion and backend timing; a laptop trial is only appropriate after that live turn passes. Physical first-audible latency and stutter require operator/browser speaker observation. Provider profile, secrets-file selection and remaining campaign allowance were not validated here, so no bridge launch command is approved by this checkpoint.

## 1. Evidence anchors and preservation boundaries

Current architecture remains Pipecat conversation/output + LangGraph durable tasks + the two project-owned speech MCP subprocesses. Preserve fast ME2 routes, receipt/idempotency protection, workspace confinement, streaming overlap, cancellation fencing, legacy fallback and the completed laptop AEC evidence.

Read these raw artifacts only as needed to answer the named acceptance checks:

| Artifact | Observed behavior |
|---|---|
| `var/results/native-03f1c13c35f24baa8c217849dcc773b2.jsonl` | Session `426e49a010534ec6afa1d51e6692687e`; two operator commands: time succeeds, Texas research fails. |
| Same trace sequences 2643, 2689, 2810–2812 | Context compaction: 15,155→11,888; 12,524→11,659; finally 16,069→12,547, rejected against 12,000. |
| Same trace sequences 2675–2684 | Actual `retrieve_source` attempt fails with `research_http_failure`; subsequent search adds more context. No retained source or citation. |
| `var/results/voice-sessions/426e49a010534ec6afa1d51e6692687e.json` | Completed time exchange disappears after compaction; election exchange is renumbered. Confirmed canonical-history/projection defect. |
| `var/results/native-761c06234edd43acbf65a9565bf64acf.jsonl` | Session `4f11dbc5b3a242b388b0452b13591482`; ASR recognizes “what is twelve times twelve”; Groq answers correctly; TTS fails. |
| Same trace sequences 1478–1487 | `tts_next`, boundary `speech_mcp_client`, `provider_timeout`; interrupted response; application recovers. This is not a Groq timeout. |
| `var/results/voice-sessions/4f11dbc5b3a242b388b0452b13591482.json` | Generated arithmetic answer disappears from archive after failed speech. |
| `var/logs/swift-hpc-20261114T064441Z-2152-bridge.log` | Covers BOTH above browser sessions under one bridge/provider campaign; do not count them as independent allocations. |
| `var/results/native-39064f923ee447d6bd0e07d6edbebc84.jsonl` | Earlier time answer: 90 TTS characters, 6.88 s PCM, 23.80 s synthesis wall. Wall time alone cannot establish provider responsibility. |
| `var/results/native-f137efe14d2146238532967a37628f43.jsonl` | Earlier long operator trial identified as audibly smoother. Use its available delivery/configuration evidence as comparison; do not label every turn flawless. |
| `ME3_LAPTOP_AEC_QUALIFICATION_2026-11-13.md` | Preserve laptop AEC PASS and golden Thalia recording; no regeneration/requalification required by these changes. |

Already implemented, preserve rather than repeat: response-scoped TTS stop; common Markdown normalization; concise time `spoken_summary`; ASR onset buffer/ordered append worker; fail-once ASR-open fencing; ASR SQLite reservation off the event loop; separate transcript publisher and event task with joined shutdown. Recent focused checks passed. These do not establish overall live reliability.

Repository caution: Git currently resolves to the parent sandbox and reports ME3 source/tests as untracked. Preserve files in place; no staging, reset, cleanup, commit or history changes during this sprint.

## 2. Actual latency: do not optimize the wrong interval

### Primary acceptance metric and published reference points

Operator priority: **end of user speech → first meaningful assistant speech heard**, aiming for approximately **2 seconds or less** in warm, straightforward interactions. This is the headline metric; TTS request→first PCM, LLM TTFT and whole-answer duration are diagnostic submetrics.

Measure three explicitly named endpoints from the same speech-end reference:
1. First non-silent, meaningful response PCM sent in the backend output packet (backend proxy).
2. First meaningful audio scheduled on the browser AudioContext (playback proxy).
3. First meaningful response heard at the speaker (physical/operator measurement).

Use the last voiced input sample mapped to the session's monotonic timeline where available. VAD stop notification occurs after endpointing silence/processing; record it separately and label existing VAD-stop-based measurements accordingly. Do not start the main clock at ASR final, model dispatch, or TTS request. Do not compare unsynchronized browser/backend clock values by direct subtraction; map clocks or report intervals within each clock domain. Exclude silence-only packets and wake chimes. Report generic progress acknowledgment latency separately from the first useful answer; an early “I'll check that” must not disguise a slow answer.

Engineering goals for Swift, not externally standardized pass thresholds:
- Warm simple commands/questions: target ≤2.0 s speech-end→audible useful response; aim for p50 ≤1.5 s and p95 ≤2.0 s once enough trials exist. Under 1 s is a stretch improvement. Report every sample when the dataset is small; do not claim a representative p95 from two turns.
- Repeated >3 s simple-turn responses are a clear optimization concern; 3.5–4.3 s current backend proxies do not satisfy the two-second objective.
- Research/background tasks: target prompt acknowledgment around 1–2 s, report final grounded-answer latency independently; external retrieval can reasonably exceed 2 s. No invented immediate factual answer to meet the target.
- Report cold startup separately, and distinguish first packet, audio continuity, first useful answer and full spoken completion. A low onset time does not pass if speech then stutters or never completes.

Primary-source references checked for this handoff update:
- [LiveKit session metrics](https://docs.livekit.io/reference/agents/session-traces/) define conversational E2E latency from user speech stopping to reply playback beginning, matching the operator's priority.
- [LiveKit voice-agent guide](https://livekit.com/voice-agents) describes optimized production responses around one second. This is vendor guidance, not a universal reproducible SOTA benchmark.
- [Deepgram/NVIDIA deployment report, May 27 2026](https://deepgram.com/blog/voice-agents-deepgram-nvidia-nemotron) reports median below 700 ms and p90 below one second for an integrated deployment inside a customer VPC. Different hosting, models and measurement conditions mean Swift's distributed Nova/Groq/Aura/MCP path cannot inherit that result.

Conclusion for implementation priorities: a ~2 s goal is reasonable and less aggressive than these optimized vendor examples, but it requires reducing total endpointing/ASR-finalization plus generation/startup delay. There is no evidence for a mandatory one-second TTS floor, and no single comparable SOTA latency standard. Preserve accuracy, endpointing and interruption correctness while optimizing.

| Measured stage | Successful time command | Failed arithmetic command |
|---|---:|---:|
| VAD speech stop → ASR final | 2.149 s | 2.466 s |
| ASR final → TTS request | 0.019 s, fast route | 0.661 s, includes Groq |
| TTS request → first native PCM | 1.329 s | 1.170 s |
| Speech stop → first backend packet | 3.500 s | 4.299 s |
| PCM audio duration | 2.840 s | 2.160 s, potentially partial |
| First → last native PCM | 1.900 s | 4.683 s |
| Total TTS wall time | 3.279 s, completed | 10.895 s, failed |
| Browser packets / acknowledgments | 71 / 71 | 54 / 54 |

Older cold baseline: Aura WebSocket setup ~2.402 s, followed by ~0.321 s to first PCM. This shows a possible startup saving; it is not a promised future improvement.

There is NO fixed one-second TTS startup floor in the inspected application. `stop_frame_timeout_s`, ASR `ttfs_p99_latency` warnings, and the replay settle allowance are not delays inserted before first audio. Do not lower them to “remove one second.” Browser startup prebuffer is 120 ms and scheduling lead is 35 ms, not one second.

For the successful time TTS segment, new counters report receive wait 2.004 s and consumer suspension 0.221 s. For the failed arithmetic segment those provider counters were lost. Do not extrapolate the successful segment's attribution to the failed segment.

Physical first-audible latency is still unmeasured here. Browser AudioContext scheduled onset is a better software proxy than backend packet arrival, but neither proves speaker onset. Acknowledgment of a packet does not prove uninterrupted audible speech.

## 3. Implement in this order

### Task 1 — Preserve canonical history and stable response identity (P0, medium)

Problem: `conversation_context.prepare_context()` and NativeLLM's budget compactor call `context.set_messages()`. `native_bridge.context_view()` and response association count user messages to derive turn indices. Dropping history changes both transcript contents and response identity.

Files: `conversation_context.py`, `native_llm.py`; directly affected projection/correlation in `native_app.py`, `native_bridge.py`.

Implementation:
- Create a temporary request projection using the existing Pipecat context/adapter. Apply history selection, discovery compaction and evidence selection to that projection only. Keep the live context and its tool callbacks authoritative.
- Include complete assistant tool-call/result groups; never leave dangling calls or mismatched IDs. Keep current user input unchanged.
- Compact all eligible discovery result messages, not just the newest one (current discovery helper breaks after one match). Remove obsolete discovery verbosity before reducing supporting evidence.
- Preserve meaningful query-relevant sentences, source URL/ID, evidence type and temporal qualifiers. If evidence cannot fit meaningfully, return explicit insufficiency rather than blindly cutting to an arbitrary fragment.
- Keep actual serialized-request enforcement before dispatch. Do not raise caps as the fix.

Acceptance: extend existing native context-budget test with time → search → failed retrieval → second lead, then assert canonical messages, receipts, transcript turn IDs and prior time answer are unchanged after every projection. Verify projected tool-call pairing and under-cap dispatch or safe rejection. Existing compaction test currently rewards removal from the canonical context; correct that expectation.

Rollback: retain fail-closed rejection if projection building fails; do not revert to deleting live history.

### Task 2 — Fix TTS timeout ownership and preserve failed-stream evidence (P0, medium)

Problem: arithmetic audio stopped arriving; client `tts_next` expired ~5.04 s after last PCM. Provider socket read and outer MCP call use the same configured deadline. Their race hides the inner stage and diagnostics. `servers.next()` removes failed operations; cancellation/release can then find no diagnostics. `provider_receive_wait_seconds` currently excludes a failed await because accumulation is after successful receive.

Files: `speech.py:call/synthesize`, `servers.py:next/speech_tool`, `remote.py:Socket.receive/DeepgramTTS.synthesize`, existing journal sanitization.

Implementation:
- Establish one operation deadline: provider/server has a bounded inner allowance inside the existing MCP outer ceiling, leaving explicit time for sanitized error serialization and cleanup. Do not increase the outer ceiling.
- Snapshot terminal diagnostics after provider cleanup has set outcome, before removing them from reach. Preserve them through outer timeout/cancellation using the existing stderr/journal or bounded terminal-operation metadata; no monitoring service.
- Record receive wait in `finally`, including timeout; consumer suspension separately; connection reuse/setup, first/last PCM, last message type, whether `Flushed` arrived, cancellation origin, and bytes delivered. Never log audio, credentials or raw provider messages.
- Never infer successful completion from delivered byte count, browser acknowledgments or socket closure. Require normal provider terminal event and normal local drain.
- Do not retry synthesis after partial output. Keep the failed answer visible and allow the next turn without replaying speech.

Acceptance: extend existing TTS fake-peer/MCP timing test: (a) PCM then Flushed; (b) PCM then no Flushed; (c) disconnect after PCM; (d) user interruption while next is pending. Use short test deadlines, not five-second sleeps. Assert exact error stage/outcome, persisted terminal diagnostics, ordered partial PCM, no false `turn_completed`, no automatic synthesis replay, and clean next operation.

Decision after this check: if failed live evidence still lacks boundary attribution, fix evidence capture before another full-live attempt. Do not change buffers or providers based on missing measurements.

### Task 3 — Correct incomplete-response presentation and spoken formatting (P0, small–medium)

Problem: failed arithmetic response is absent from archive although generated text and audio delivery occurred. Existing response records already distinguish generation/TTS/playback, but projection depends on committed assistant messages. `[S1]` and bare URLs are not explicitly removed by `spoken_text()`.

Files: `native_bridge.py:context_view`, `native_app.py:response_records`, `native_speech.py:spoken_text`.

Implementation: project existing response records when committed context lacks the response. Preserve generated text as partial/failed/interrupted with its stable response ID and delivery counts. Never label it delivered or insert it as a successful answer into model context. Remove citation tokens/URLs and punctuation-only speech at the TTS boundary, keeping formatted UI evidence. Preserve the completed one-sentence correction separately from an interrupted earlier answer.

Acceptance: a two-sentence answer interrupted mid-playback, then a completed correction, both remain correctly identified in UI/archive; partial TTS failure remains visible; UI citations survive while TTS receives natural text. Reuse existing lifecycle, transcript and Markdown fixtures.

### Task 4 — Finish the actual research route, without a search loop (P0, medium)

Problem: search routing is live-proven, successful retrieval is not. HTTP failure diagnostics are generated by ResearchClient but dropped from persisted nested event data. Basic Extract fallback currently runs for insufficient HTML, not failed HTTP. After one fetch failure the model searched again instead of trying another available lead. Context then exceeded a conservative application budget.

Files: `research.py:retrieve/search/evidence_for_model`, `native_app.py:execute`, `native_llm.py:claim_tool_operation`, `native_bridge.py:_safe_event`; existing policy in `conversation_context.py`.

Implementation:
1. Preserve allowlisted HTTP status, host hash, stage, failure class and fallback outcome in the trace. The last run's status is unknown: do not assume 403, DNS or network restriction.
2. For explicitly retryable/inadequate retrieval cases, permit one existing Tavily Basic Extract fallback if budget permits. Never bypass unsafe-URL, SSRF or permission policy. Otherwise return compact failure plus existing viable untried lead choices.
3. Normalize equivalent search arguments for cache/loop checks. Do not add an LLM reranker or semantic router. Preserve deeper research for genuinely different questions.
4. For simple facts, implement search-evidence sufficiency only with explicit negative tests: direct relevant statement; identifiable source URL/provenance; requested event/date/scope matched; no known conflict; enough text for the claim. Search ranking or a government-looking host alone is insufficient. Label citations `search_result` versus `retrieved_page`, with the supporting excerpt retained. Never imply a page was fetched.
5. If insufficient, retrieve the selected source. After failure, try an already discovered viable alternative before another search. Stop with a specific inability when no path remains; do not consume the whole turn on equivalent searches.
6. Keep snippets compact before adding them to the request. Keep source evidence available for follow-ups without replaying every search body.

Acceptance: extend existing natural-current-event application scenario with: sufficient dated snippet → supported short answer; stale/ambiguous/conflicting snippet → retrieval; HTTP failure → one allowed fallback/alternate; no evidence → truthful inability. Assert calls, evidence type, provenance, request budget and no fabricated citation. Model choices in mocked transcripts are integration evidence only, not autonomous routing accuracy.

### Task 5 — Audible acknowledgment while work starts, not before it starts (P0 for the research/task demo, small–medium)

Problem: research can be silent for many seconds. Latest failed research produced its first packet only 11.19 s after endpoint, and that was an error response. Current native tool handler has no explicit deterministic research acceptance acknowledgment.

Files: native tool acceptance in `native_app.py`, existing Pipecat TTS/output framing and response records; no separate playback engine.

Implementation:
- On accepted research or a durable long task, start the underlying operation immediately. Emit at most one short progress utterance, e.g. “I'll check that,” without another LLM call. Do not wait for it to finish before executing.
- Use Pipecat's existing queued speak/context facilities supported by installed 1.13.0. Do not call `run_tts()` concurrently, open a second TTS controller, or play an independent browser clip over the final answer.
- Correlate progress with the owning request/task. Progress is not task completion and must not emit final `turn_completed`, arm idle expiry, or overwrite the final answer's response context.
- Suppress a queued acknowledgment if the useful answer arrives first. Do not acknowledge every direct fact, clock/light command or every internal search/fetch step.
- Interruption cancels pending progress audio; durable task work continues unless the user explicitly cancels the task. Task failure must update its card truthfully.
- Prefer an already available, suitable local acknowledgment asset only if it can use the same output/cancellation path; do not regenerate TTS or add assets during offline work. Otherwise account for the extra Aura synthesis and characters.

Acceptance: delayed search starts before acknowledgment audio ends; exactly one acknowledgment and one final answer; immediate result suppresses pending acknowledgment; interruption clears it; two concurrent task updates do not overlap audio; acknowledgment never completes the task/turn. Use existing mocked speech and delayed tool fixtures.

### Task 6 — Warm-up and latency reduction with a measured decision gate (P1, medium)

Already warm: native bridge owns persistent MCP children and provider adapters. Aura already reuses a socket across nearby segments. Do not build another warm-up service or send dummy ASR, Groq or TTS requests.

Specific warm-reuse issue: `DeepgramTTS._reuse_idle_seconds()` is `min(15, timeout_seconds * .8)`. Active profile omits timeout, default is five seconds, so the connection closes after FOUR seconds. Natural follow-ups can arrive within twenty seconds and still pay a cold TTS connection. Read timeout and connection idle lifetime are different concepts, but earlier stale-socket failures mean the current guard must not simply be deleted.

Implement in two bounded steps after Task 2:
- Expose existing connection-reused/setup timings through safe speech diagnostics and the journal; they currently lose useful nested fields. Compare cold request versus same-socket next segment and a six-second follow-up with identical text. Fake peers prove lifecycle/ordering, not network latency.
- Decouple idle-reuse policy from per-read deadline only if fake-peer tests establish safe stale-peer rejection and later authorized traces show setup repeats unnecessarily. Use a finite declared idle policy compatible with provider behavior and the 20-second conversation window. Do not invent KeepAlive messages or automatically replay text on a stale socket. Detect closed peers; unknown post-Speak outcomes still stop/reject replay.

Optional readiness overlap, only if setup remains a demonstrated major cost: overlap ONE accounted TTS connection preparation with accepted request/LLM work, then consume that same preparation for the first Speak. It needs explicit lifecycle/ledger ownership and cancellation; no unaccounted background sockets or double reservation. Defer a new MCP preparation method if safe existing reuse solves the issue. Warming when the user merely opens the webpage or while asleep is not justified.

ASR finalization is also material: latest endpoints took 2.15–2.47 s to final transcript, versus 0.56 s in an older successful baseline. Measure `CommitSpeechFrame`, accepted-audio flush, Finalize/CloseStream sends, final Results and terminal close. Determine whether local code waits for transport closure after a usable final transcript. Optimize only a reproduced unnecessary wait; do not publish provisional text as final, change VAD thresholds or weaken utterance boundaries merely to save time.

Acceptance: cold/warm benchmarks report setup and ready→PCM separately, same text/audio workload, no extra synthesis attempts, no stale audio, cancellation works. A warm result below one second is possible in principle; no such requirement or improvement is currently proven. Remove an optimization if it increases startup failures or breaks budget ownership.

### Task 7 — Explain and reduce stutter at the demonstrated boundary (P0, small initially)

Files: `remote.py`/`speech.py` counters from Task 2, `native_bridge.py` safe journal, `ui/dashboard.js:receiveAudio`; reuse existing browser audio tests.

Record only the missing discriminators: provider message wait and consumer pause; MCP next latency; queued audio seconds; browser arrival time, scheduled start, previous scheduled end, AudioContext state and underrun count/duration. Report intervals only between first and last audio of the same response. ACK counts alone are insufficient. Do not collect microphone audio by default.

The browser uses a 120 ms initial prebuffer, 35 ms scheduling lead and bounded sources. Do not increase them blindly. Use deterministic 40 ms packet playback fixtures with smooth delivery, jitter/bursts, a 300 ms gap, producer slower than realtime, and interruption. Assert order and clearing; measure actual scheduling holes. If holes come from scheduling despite sufficient queued PCM, fix that scheduling bug. If source audio is persistently slower than realtime, buffering only delays onset and cannot repair throughput; investigate connection/provider delivery using Task 2 evidence. Introduce adaptive rebuffering only if measured burst jitter is the cause and its added latency is explicit and bounded.

Compare the earlier smooth trial with the failing one using SAME fields, response lengths and cold/warm classification. Previous subjective smoothness is real operator evidence; it does not establish identical provider conditions. Do not replace Aura or change voice settings without a justified, separately approved comparison.

## 4. Limits, extensibility and recovery policy

- No general registry rewrite: existing `conversation_tools.DEFINITIONS`, validators, Pipecat registered handlers and LangGraph workspace tools already provide the required interfaces. Add only missing per-tool policy metadata where duplication is demonstrated.
- Current Groq request limit is an application estimate cap, not model context size. In the research trial actual prompt use was 5,412 tokens, versus 34,301 reserved estimates. Retain safe estimation until a compatible tokenizer is available/validated; do not divide bytes by four and call it enforcement. Do not refill historical reservations or raise approved caps silently.
- Separate actual tokens, estimate, remaining campaign and provider quotas in diagnostics. Deduplicate the duplicate `llm_usage` events when computing totals.
- Existing 2.1 s Groq pacing floor can add latency across tool continuations; log pacing delay separately from TTFT. Change it only with actual quota headers/evidence, not as part of TTS work.
- Research request limits are process-local. Before enabling automated research retries or longer multi-process campaigns, reuse the durable ledger pattern for research operations. Do not create a billing service.
- A 429 with a short explicit retry interval may justify one read-only retry within the same budget/deadline. Daily quota/credit exhaustion, unknown dispatch and partially spoken TTS do not justify automatic retries.
- Unknown device/write outcome: preserve receipt, inspect state, never duplicate the action. User speech interruption remains independent of background task cancellation.
- Active follow-up admits background speech; there is no proven speaker/addressee classifier. Preserve existing consequential-action approvals and push-to-talk/pause fallback. Do not claim AEC distinguishes professor commentary, or add a model without labeled evidence.
- Early ASR diagnostics reuse previous transport timestamps: reset/scope them at connection-attempt start. Test two streams; every reported readiness timestamp must belong to the current stream.

## 5. Minimal verification and live gate

Execution order: Task 1 → Task 2 + Task 7 → Task 3 → Task 4 → Task 5. Task 6 follows reliable completion, using its measurement gate; do not defer the stutter work behind optional warming. Tasks 1–5 and 7 are the next demonstration blockers.

Offline: run only named/extended context, transcript, TTS fake-peer, native browser and research boundary tests after their respective changes. Use command-scoped local IPC for actual MCP/socket tests; restricted runs previously failed with listener PermissionError or MCP initialize timeout. Preserve logs and classify those explicitly. No broad suite rerun or AEC requalification unless a change directly invalidates that evidence.

Proposed ONE later authorized persistent-session sequence:
1. Natural current-event question: hear one timely acknowledgment; actual Tavily use; sufficient attributed evidence or bounded retrieval/fallback; accurate concise answer; all TTS segments and terminal event complete.
2. “Which source supports that?” Context/source provenance retained; no unnecessary repeated search.
3. “What is twelve times twelve?” Direct correct answer after the research turn, normal speech completion; archive retains all exchanges.

First simulate one retrieval failure and one missing-Flushed failure offline. Do not spend live calls intentionally causing an outage. Stop live work on unknown outcomes or a terminal speech failure; no full-demo expansion.

Fresh campaign proposal only, NOT authorization: at most 3 ASR requests / 30 s input, 6 Groq requests / 2,000 completion tokens each with an explicit conservative aggregate input allowance, 8 Aura synthesis operations / 2,400 aggregate characters (acknowledgment included), 4 combined research operations; other providers off, no automatic retries. Verify practical accounting before requesting approval. Do not silently create a new run ID to replenish an existing allocation.

Capture stage timestamps above, exact response text and evidence type, source IDs, first/last PCM, Flushed, playback scheduling gaps, final turn outcome and ledger. Operator records audible stutter and estimated speaker onset; backend timestamps remain proxies. Preserve successful WAV/audio artifacts where existing capture supports them; do not claim a browser output WAV exists unless actually recorded.

Exit: no canonical-history deletion; research either supports its claim or fails specifically; progress acknowledgment does not delay work or conflict with final speech; TTS produces a confirmed complete turn or an accurately retained partial failure; next user turn works without replaying actions. Only then expand toward five-turn tasks and the physical demo.


### First-wake ASR timeout correction (2026-11-14)

Evidence: `laptop-20261114T093055Z`, native run `7e0018daaf674c58afaa39d818d283df`, session `8c1fce4e59ff49be92f7a600044981d4`. First utterance buffered 106,496 bytes; last pre-failure snapshot showed zero sent bytes and no connection-ready timestamp. Its Nova reservation remains `dispatched_unknown`, consumed. The trace does not establish why the real handshake failed.

Production correction: normalize the outer `_connect()` timeout in `remote.py` to a safe `ProviderError` at `asr_open`. Existing code already waits for readiness and flushes before finalization; no readiness-controller replacement was needed. Preserve ASR diagnostics after `SpeechService.finish()` cleanup, relay them through `SpeechClient` to the native event emitter, and retain connection state/stages plus `error_role` in browser traces. The UI can now identify speech-recognition connection timeout rather than use the generic response failure notice. No buffer/deadline increases, retries or live calls.

Verification: `python -m pytest -q tests/test_speech.py tests/test_remote.py tests/test_native_entrypoint.py -k 'outer_connection_deadline or pending_connection_flush_timeout or connection_timeout_is_explicit or pending_connection_cancellation or asr_browser_failure_trace or native_tts_event_journal or same_connection_asr_failure or native_feed_recovers or real_stdio_mcp_contract_and_cleanup or stdio_timeout_recovery or stdio_cancel_inflight_then_reuse or cancel_provider_in_flight'` passed **13** with command-scoped local IPC and isolated `/tmp` task/ledger storage. Two additional `test_deepgram_connection_starts_on_open_and_buffers_audio_once_in_order` cases (0.5 s and 4.5 s) passed. Raw handshake timeout and never-ready peers traverse the actual service/error/client code; real stdio startup/cancellation is tested separately with mock providers. This is not a live Deepgram qualification. Ruff and `node --check src/me3_voice/ui/dashboard.js` passed. Existing Pipecat AudioContextTTSService deprecation warning remains. The two recovery tests now isolate task storage rather than touch the running bridge database.

Next gate: preserve the old unknown reservation; inspect newly retained stage diagnostics in any separately authorized live attempt. Do not assert that error classification alone resolves intermittent external connection readiness.


Latest laptop evidence correction: native run `0d1854c7847d42cfb405352213bbef77`, session `3848235615244b6eb864e99ea965af91`. Boiling and 12×12 turns completed; backend endpoint→first packet was 4.082 s and 4.265 s, respectively. The fast light action was simulated and its speech was interrupted. Fourth utterance failed at seq5122 with `request_limit_reached` (old laptop profile: 5 Nova reservations already consumed); seq5135 secondary closed-operation poll overwrote the failure. Fixed first-failure retention/poll suppression in native_speech; fixed reuse of prior stream connection timestamps in remote diagnostics. Seven targeted tests and five existing ordering/admission/overload/stale-result checks passed with command-scoped IPC; Ruff passed. No live calls. User requested sustained testing without small turn-count quotas: new ignored `var/cache/swift-laptop-sustained-{backends,runtime}.json` validates with supported null ASR/Groq/TTS request caps, existing 12k serialized-input compaction guard, 1M aggregate conservative input guard, and 100 research operations. Provider quotas still apply. Historical run/campaign records remain unchanged. This prepares operator testing and does not itself dispatch providers.


### 2026-11-15 adaptive quick-research qualification

The live work reused campaign `swift-research-3q-20261115-8d7b31c4`, with the same durable OpenRouter ledger, 32k estimated-input/request and 500k aggregate estimate ceiling. Credential files were checked for required key names and mode 0600 without reading or printing values. ASR/TTS were mocked; OpenRouter GPT-OSS-120B and Tavily were real. No retries or fallback provider.

| Question / run suffix | Verbatim final output | Evidence | Calls / provider-reported tokens | Timing / terminal |
|---|---|---|---|---|
| “Who won the 2026 FIFA World Cup?” / `fifa-snippet-fix` | “Spain won the 2026 FIFA World Cup, defeating Argentina in the final on July 19, 2026 [S1].” | FIFA official champions snippet, S1; citation verifier true | 2 LLM; 1 search; 0 retrieval; 4,705 prompt / 221 completion tokens | Answer packet 8.28 s after speech endpoint; turn complete 8.62 s; PASS |
| “Which movie won Best Picture at the 2026 Academy Awards?” / `oscars-snippet-fix` | “I couldn't complete that request reliably, so I can't give a confident answer.” | Official Oscars search snippet S1 plus Wikipedia snippet S2; model chose S2 and validator flagged `numeric_anchor_conflict`; official page fetched as S3 | 3 LLM calls; 1 search; 1 retrieval; 7,471 prompt / 263 completion tokens. A fourth synthesis was denied before provider dispatch | Fallback packet 9.48 s after endpoint; terminal `turn_failed: input_token_budget_exhausted`; FAIL, safely abstained |
| “Who won the 2026 Nobel Peace Prize?” / earlier `nobel` | “Navanethem “Navi” Pillay won the 2026 Nobel Peace Prize [S1].” | Official Nobel Prize press release S1, source claim check true | 3 LLM; 1 search; 1 retrieval; 7,878 prompt / 265 completion tokens | Answer packet 12.30 s after endpoint; turn complete 12.64 s; PASS (earlier run, before current snippet-limit change) |

Other failed-run final emissions were exactly `I couldn't verify this answer from a retrieved source, so I won't guess.` for earlier FIFA and Oscar attempts. The earlier completed Oscar retry said `One Battle After Another won Best Picture at the 2026 Academy Awards (the 98th Oscars) [S1].`, but the selected extracted passage S1-P9 contained no winner; do not count that as a grounded pass. The latest Oscar attempt did not emit the rejected model claim; only the truthful fallback was spoken. Playback WAVs and packets in these runs are mock TTS, not Aura or physical speaker evidence.

Confirmed control-flow correction: when an unavailable `web_search` call was rejected, execution fell through into the search handler with no validated `args`, raising `UnboundLocalError`. The handler now terminates that branch and returns the existing `research_unavailable` result without dispatch. Quick factual search now promotes only the best-ranked result snippet as citation-eligible evidence; other leads remain discovery-only for retrieval. Focused offline native tests (five targeted cases, command-scoped local IPC) passed; Ruff passed. These checks are mocked and do not prove OpenRouter will select the top snippet consistently.

Campaign snapshot after the two latest calls: 29 successful OpenRouter reservations, 493,756 estimated input tokens reserved of 500,000, 6,244 remaining, zero unknown; 31 request slots remain but the token ceiling is binding. Provider-reported usage for the two latest turns was 12,176 prompt / 484 completion tokens, while the conservative ledger reserved 75,521 estimate tokens (includes prior Oscars recovery and is not actual usage). Do not dispatch another model call under this campaign. Tavily request accounting is process-local (`request_limit=20`) and is not an aggregate campaign ledger; recorded search/retrieval operation counts therefore must not be described as a durable remaining Tavily allowance.

Latency is backend replay only. For latest FIFA, the early 3.52 s endpoint→packet was the short acknowledgment; meaningful answer packet was 8.28 s after endpoint. Latest Oscar’s 2.97 s first packet was likewise acknowledgment; fallback packet was 9.48 s after endpoint. Earlier Nobel’s answer packet was 12.30 s. None measures human speaker audibility. The campaign’s 1.25× serialized-byte input estimator materially exceeds provider token use; it remains enforceable and must not be weakened to fit more calls.

Remaining work: one focused live Oscar/Nobel retest cannot fit safely in this campaign. The post-run local change reducing eligible quick-search evidence from two snippets to one is covered offline but has not been live-verified. Request a new bounded campaign before any additional provider call; keep research retrieval within the native request path. Full source-grounding reliability, campaign-wide Tavily accounting, real Aura delivery and physical speech remain unqualified.
