# Swift ME3 — Luna / Coding-Agent Resume Checkpoint (v2)

**As of:** 2026-11-13 (Asia/Manila)
**Revision:** v2 — adds wakeword/conversational gating and demo reliability as explicit P1; supersedes the first handoff for agent planning.
**Status:** Laptop Browser AEC **Test B PASS (qualified scope only)**; native full-live replay **BLOCKED** on an ASR opening deadline; **wakeword/multi-turn/background-speech behavior NOT YET QUALIFIED**; startup/latency optimization **PENDING MEASUREMENT**.

## Source-of-truth and constraints

- Repo used on HPC: `$ME3_ROOT` (operator has also referred to `$ME3_ROOT`; resolve aliases before assuming they are identical). Work from the verified repo root; inspect current branch, uncommitted changes, tests, and docs before editing. Do not overwrite work from another agent.
- Earlier runbook: `docs/architecture_modernization/OPERATOR_TEST_RUNBOOK.md`.
- Main intent: **reliable, low-latency full-duplex native Pipecat + LangGraph voice assistant** with two project-owned speech MCP services; few moving parts; laptop-first qualification then Raspberry Pi 5 4 GB physical validation.
- **No unapproved live Deepgram/Groq requests**, provider retries or credit spend. Do not print or expose API secrets. Default to read-only inspection, offline mocks/replays and focused regression tests. Before any proposed live call, present scope, estimated maximum requests, readiness gates and wait for explicit authorization.
- Avoid gratuitous code churn, new services or architectures, changing production defaults without evidence, hiding failures with blanket deadline increases, or asserting a test passed without real evidence.

## 0. Newly completed operator test — preserve as PASS

Test B laptop/browser AEC completed with USB mic/speaker and independent `/audio-probe` served at `http://127.0.0.1:8766/audio-probe`. Native voice bridge on 8765 served `/` but returned 404 for `/audio-probe`; native app does not currently expose that registered legacy diagnostic route.

**Evidence files on HPC:**

```text
var/results/aec/laptop_20261113/
  laptop-aec-test1-boiling/
    laptop-aec-off-1.webm
    laptop-aec-off-1.json
    laptop-aec-on-2.webm
    laptop-aec-on-2.json
  laptop-aec-test2-thalia/
    laptop-aec-off-3.webm
    laptop-aec-off-3.json
    laptop-aec-on-4.webm
    laptop-aec-on-4.json
var/results/aec/swift_aec_aura2_thalia.wav
var/results/aec/swift_aec_aura2_thalia.webm
```

All four captures decode with FFmpeg, are distinct by SHA-256, and are ~23.94–24.00 seconds based on WebM packet timestamps; `ffprobe format.duration=N/A` is not itself a failure. JSON reported correct AEC Off/On and disabled noise suppression/AGC. Level measurements (dBFS):

| Reference | Window | AEC Off | AEC On | On−Off |
|---|---|---:|---:|---:|
| Boiling | 0–8 speaker-only | −45.1 | −63.7 | **−18.6 dB** |
| Boiling | 8–16 double-talk | −22.8 | −24.2 | −1.4 dB |
| Boiling | 16–24 near-only | −26.1 | −27.8 | −1.7 dB |
| Thalia | 0–8 speaker-only | −42.6 | −65.1 | **−22.5 dB** |
| Thalia | 8–16 double-talk | −23.6 | −22.3 | +1.3 dB |
| Thalia | 16–24 near-only | −24.6 | −25.4 | −0.8 dB |

Operator listening of Thalia AEC On: **Thalia echo noticeably quieter; user speech clear and intelligible during double-talk; clear after playback stopped.** Therefore PASS strictly for **this laptop browser AEC setup**, not ERLE or native/Pi full-duplex qualification. Do not repeat the recordings without an identified need. See companion `ME3_LAPTOP_AEC_QUALIFICATION_2026-11-13.md` for scope, limitations, and full evidence facts.

## 1. P0 — Fix native ASR replay deadline inconsistency (offline first)

- Previous full-live replay `manual-full-20261113T010325Z` failed during Deepgram ASR `asr_open` at **~5.013 s**: `ProviderError: asr MCP call exceeded its configured deadline`.
- Deepgram request was reserved/ledger status `unknown`; Groq and TTS stages were not reached. Trace JSONL remains; there is no output WAV or normal completion. **Do not rerun live replay without explicit new authorization.**
- Known code-path mismatch: regular `NativeState` constructs `SpeechClient(timeout=max(p.timeout_seconds for p in self.backends.values()), ...)`, while `replay_native_application` builds `SpeechClient(backends=..., credentials=..., transcript=...)` without `timeout`, falling back to a **5-second default**, notwithstanding backend ASR configured for **35 seconds**. Per-call deadline is `min(self.timeout, self.backends[kind].timeout_seconds)`.
- Inspect actual code and config; fix just the inconsistent effective deadline propagation. Also investigate why `asr_open` took >5 seconds; a larger deadline alone is not evidence the underlying delay is solved.
- Add focused offline tests for correct precedence and timeout propagation, cancellation, retries/fail-closed provider ledger behavior, and regression. Preserve budget/retry restrictions.

## 2. P1 — Wakeword, hands-free multi-turn interaction and demo reliability (NEW IN v2)

**Operator requirement:** The upcoming demo involves *consecutive natural-language commands* and possible professor commentary (e.g., "Try asking it about the weather") while the assistant is nearby. Requiring "Hey Swift" before every turn creates avoidable false-reject risk. Conversely, keeping the microphone hard-muted except for manual button presses defeats the hands-free goal. **Do not make Pause/Resume a routine response to every sound.** Treat it as an occasional fallback for extended background discussion, alongside a one-click/manual Arm fallback if KWS fails. The main path should be automatic, low latency and robust.

**Initial engineering action is a read-only audit**, not immediate implementation or wakeword retraining. Inspect actual current native/legacy implementations of local "Hey Swift" KWS (including selected models/assets and confidence thresholds), frontend mic state, VAD, ASR partial/final handling, barge-in, state transitions, session reset/resume, tool execution and relevant UI controls. Report **already implemented / partial / absent / unverified** with exact source references. Avoid assuming mock browser UI proves physical KWS detection.

### Desired minimal conversational policy (proposed; tune from testing)

1. **SLEEP / IDLE:** Only local wakeword detection is eligible to arm a conversation. Do not open a billable ASR/LLM/TTS turn or execute a tool because of normal background speech in this state. There may be existing library initialization costs; measure them rather than assuming zero runtime cost.
2. **ARMED / LISTENING:** A single successful "Hey Swift" (or explicit operator Arm fallback) enables the next genuine user request. Keep a **multi-turn session** open for natural follow-ups after assistant responses; **do not require KWS each turn**. Begin with a configurable **~15–20 s convenient follow-up window** after assistant response and **~60 s session inactivity timeout** as *test candidates*, not fixed values. Investigate best semantics and actual reply-end/turn-end clocks; make state transitions deterministic. After the short follow-up window, preserve a graceful re-arm policy instead of unbounded unguarded listening.
3. **AUTOMATIC INPUT GATING:** VAD and audio thresholds should filter non-speech/noise *before* submitting ASR. Once an ASR transcript exists, use the lightest reliable **addressedness/intent gate** feasible in the current architecture to suppress obvious *meta-commentary*, e.g., professor to student: "Now ask Swift what time it is." Accept natural direct commands/questions without a new wakeword. Do not build a second full LLM classifier or make tool execution depend on brittle keywords as a default. First quantify errors, latency and confidence availability; adopt transcript-based rules or an existing response-stage check only if demonstrably beneficial. Uncertain utterances should fail conservatively for tool execution and allow immediate user recovery. No extra request to a paid provider solely to decide addressedness without approval.
4. **SPEAKING / BARGE-IN:** Preserve legitimate user interruption while rejecting echo/noise-induced and background-speaker interruptions as far as the available evidence permits. Audit when VAD *tentatively* ducks or interrupts versus when ASR confirms intended user speech, stale-audio fencing and cancellation. Avoid blindly cancelling and restarting output on every VAD spike. A single mic cannot identify speaker/target perfectly; document that limit.
5. **PAUSE / RESUME / MANUAL ARM:** Reuse existing browser/dashboard buttons or minimum additional UI state where possible. Manual Arm is a safety net for a missed KWS, not a requirement for normal commands. Pause Listening blocks new inbound turns/tool calls/accidental barge-in during extended discussion and Resume returns to the same conversation without restarting the backend. Make the UI status clearly reflect Sleep, Armed/Listening, Speaking, Paused. A short audible wake acknowledgment may be useful only if it doesn't add material latency; verify existing cues.
6. **EXPIRY / EXIT:** Use deterministic inactivity and End Conversation behavior. No session should remain indefinitely receptive to background commentary. Validate that changing between states cannot leak stale ASR events, execute stale tools, or leave a mic muted unexpectedly.

**Hard limits / non-goals:** Do not retrain/redeploy KWS merely based on fear of demo failures. No speaker diarization, voice biometrics, additional wakeword service, second orchestration engine, or new paid LLM gate without measured evidence that the simpler design cannot meet the need. One mic/VAD/ASR alone cannot perfectly identify the intended addressee; don't claim it can. Professor background instructions such as "Ask Swift about the weather" must be *explicitly tested* as non-target speech. Manual Pause is allowed for an unusually long discussion, not for each small noise or comment.

### Proposed offline-first qualification matrix (not achieved)

| Trial | Proposed acceptance target | Evidence required |
|---|---|---|
| Wakeword at actual demo distance/volume | >=19/20 natural wake attempts and a documented false-wake rate | Operator audio and timings; selected KWS model/threshold; laptop then Pi |
| Wake-once multi-turn | 10 consecutive direct follow-up commands without another wakeword, state timeouts as configured | Offline state tests, then live/browser operator qualification with separately approved calls |
| Background commentary | No unintended tool execution across a scripted 5-minute discussion, including professor's "now ask it X" instructions | Repeatable scripted audio/room trial; counts of false ASR submission vs false tool action |
| General background noise | No routine requirement to press Pause to continue; no repeated false wake/barge-in | Natural room noise trials and event traces |
| Manual fallback | 10/10 manual Arm/Resume attempts without restart; Pause blocks inbound tool execution | Mock/UI state tests and operator trials |
| Barge-in | 5 controlled real user interruptions without stale response or hangs | Native audio traces and operator listening; live provider budget requires approval |

**Deliverable:** Before extensive coding, produce one concise state/transition diagram or table showing which speech can reach ASR/LLM/tools in each state, existing versus missing functionality, precise test cases, and a minimal change plan. Implement in bounded slices **after P0 correctness**; prioritize ensuring a failed wake does not strand the demo. Add deterministic offline tests for voice gating, professor-commentary transcripts, false barge-in and resume/cancellation. Do not characterize these proposals as currently implemented or already qualified.

## 3. P1 — Preserve golden TTS fixture and investigate WAV duration handling

- `scripts/create_swift_thalia_aec_reference.py` generated **one** real Deepgram Aura-2 Thalia reference request (approx 602 characters), WAV + WebM, playable approximately **35 s** on both formats.
- Script displayed **`WAV: 44737.9s, 24000 Hz, 1 channel(s)`**, inconsistent with decoded/playback duration. Treat this as **suspected streaming WAV header/metadata parsing defect**, not confirmed production failure.
- Determine whether generator duration uses WAV header placeholders rather than decoded samples. Audit relevant production playback/TTS adapters for reliance on header-derived sample counts or duration (buffer sizing, pacing, stall/deadline measurement, streaming chunks, interruption, metrics). Add synthetic malformed/placeholder header unit tests and demonstrate safe decoding/duration handling; avoid indiscriminate production changes without a demonstrated shared issue.
- Existing project secrets key is `ME3_DEEPGRAM_API_KEY`; standalone generator originally expected `DEEPGRAM_API_KEY`, which operator temporarily mapped inside a subshell. Prefer aligned, secure key handling if modifying generator. Never print key, rotate it, or generate extra audio without approval.
- Preserve `var/results/aec/swift_aec_aura2_thalia.{wav,webm}` as private reusable reference/golden fixture; optionally add sidecar manifest (model, content hash, source text or hash, decoded duration, codec/sample rate, generation date), and short 5–10 s fixture if obtainable **from existing audio locally**, not another provider call by default.

## 4. P1 — Native `/audio-probe` diagnostics route regression

- `voice_bridge.py:build_app` switches to `native_bridge.build_native_app` for `engine == 'native'` before legacy route registration; native route table includes `/`, `/api/health`, `/api/snapshot`, `/api/voice/capabilities`, `/api/voice/ws`, but not `/audio-probe`.
- Legacy `voice_bridge.py` and `dashboard.py serve` serve `render_audio_probe`; standalone 8766 was a successful **operator workaround**, not a deployment recommendation.
- Consider the smallest safe native route exposure, with the same loopback-only checks, CSP, permissions and cache controls; ensure no live-provider startup, credential disclosure, or behavioral change to voice WS. Add offline HTTP route tests for native and legacy.

## 5. P1/P2 — Instrument and optimize startup and conversational latency

**Do not optimize based on simulated/offline event-loop numbers alone.** First report cold versus warmed startup and incremental pipeline timings with reproducible conditions and configuration.

Instrument monotonic stages:

```text
process startup / CLI → native bridge ready
MCP ASR/TTS subprocess initialization + discovery
Deepgram ASR stream connect/open (cold vs warm)
user speech end → VAD endpoint → ASR partial/final
ASR final → LLM request/first token
LLM first token → TTS initial request/first audio chunk
first audio → browser or physical speaker audible playback
end of user speech → first audible assistant audio (true E2E)
```

- Track per-stage and E2E p50/p95, startup-inclusive and warm-turn profiles, browser versus Pi hardware separately, cancellation and correctness alongside speed. Previous replays: 50 mocked turns per profile and ~196 ms startup-inclusive mock event-loop stall; not evidence of real-provider latency.
- Check sequential MCP startup and tool discovery; persistent/warm ASR connections versus reconnect per turn; 32 ms ASR frame contract (up to 96 ms batch allowed); approx 1.024 s endpoint-silence and 256–384 ms interruption gating from earlier planning (VERIFY live configuration); streaming LLM → TTS and playback overlap; needless per-chunk serialization and queuing.
- Consider changes only after demonstrating a bottleneck: warm or persistent MCP/ASR connections, safe overlap of LLM output and streaming TTS, endpoint tuning and queue reduction, without sacrificing AEC, response ordering, cancellation, reliable finalization or simple architecture.
- Target **under 2 s warm-session speech-end → first audible speech** as an **aspiration**, not an achieved result. No new infrastructure or performance claims without sound benchmarks.

## 6. Remaining separate qualification gates

- Native full-live ASR→Groq→Aura pipeline with predetermined small, user-approved provider budget, after P0 fix and mock gates; one previous full-live replay failed at `asr_open`.
- Browser-native real conversation, acoustic barge-in and interruption/cancellation; stand-alone browser AEC result does **not** qualify these automatically.
- **Wakeword/session/demo gate:** actual KWS recall and false positives at intended mic distance, automatic wake-once multi-turn behavior, professor background-speech rejection, low-noise wake/ASR gating and no operator mic-toggling for ordinary room sounds; manual Arm/Pause fallback. All still unqualified as of this handoff.
- Pi 5 hardware audio path: speaker audibility, PipeWire/WebRTC AEC reference routing, diagnostic microphone capture, cancellation under slow ASR finalization, `IncompleteReadError` resolution; needs operator physical evidence. Prior acoustic trial was **not qualified**.

## Required agent working style and deliverables

1. Start with **read-only inventory**: current revision, diff/dirty files, actual runtime defaults, test hierarchy, code paths, available AEC evidence; reconcile recent agent changes without clobbering work.
2. Give a short prioritized plan with measured/supported facts versus hypotheses. Tackle P0 correctness first. Then perform read-only wakeword/session-gating audit and propose minimal reliable behavior before broad latency work. Keep task scopes bounded.
3. Use **bounded code changes** with targeted unit/integration tests, then full offline regression as appropriate. Document exact tests/results, confirmed behavior versus remaining unknowns; no claims of live success from mocks.
4. For AEC, preserve original evidence; read JSON/WebM as needed and produce a repository-local report without rewriting recorded audio; respect privacy and `.gitignore` rules.
5. Explicitly separate zero-provider offline steps from live validation proposals; wait for approval before triggering any provider request or real TTS regeneration.
6. Produce updated checkpoint: implemented, tested, unresolved, next operator commands, risk, estimated live request budget if applicable, wakeword/demo gating findings, latency measurement results (if measured), and any required user decisions. No speculative 'PASS'.
7. Model-usage strategy: user treats the week as **100% finite capacity**. For each next slice recommend the lowest-cost/effort model likely to produce reliable output among Luna Medium/High, Terra Medium/High, Sol Medium/High, considering retries/rework; Luna High currently suitable for P0 async bug and the wake/session architecture audit, Luna Medium for contained tests/route/metadata patches. Terra or Sol only for a specific high-ROI stall, independent review or architecture-critical issue; no parallel agents editing the same files. Verify present usage/pricing assumptions instead of treating prior comparisons as absolute.
