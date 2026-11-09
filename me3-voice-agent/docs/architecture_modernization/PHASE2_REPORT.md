# Phase 2 — native contract proof and physical-audio checkpoint

Date: 2026-11-09. **Repository-local work is validated; physical Pi qualification is PENDING_OPERATOR_VALIDATION. Phase 2 is not fully accepted and Phase 3 has not started.** The user confirmed Pi is inaccessible from this environment and requested operator commands. No SSH connection, model API, model download, framework installation, GPIO action, service restart or Pi deployment was attempted. Production runtime and tests remain unchanged.

## Baseline and provenance

Raw historical artifacts live in ignored `var/experiments/022-native-voice-duplex/`. For GitHub/Pi handoff, the four executable prototypes are now canonical in tracked `scripts/phase2/`; the eight transport checks are in `tests/test_phase2_transport.py`. No production migration is implied. This is one bounded investigation: can a native Pipecat pipeline preserve the existing speech/control contracts while removing Session orchestration?

Parent Git HEAD is `9338bc205420747988f8b2254cae37701c13488a`; branch/main and untracked ME3 limitation remain as recorded in Phase 0. HEAD is not an ME3 revision. `baseline.json` binds 113 source/config/test/deployment files to SHA-256 and inventories 92 installed distributions. Key versions: Pipecat 1.13.0, MCP 2.3.0, aiohttp 3.14.5, Python 3.13.9 on x86_64 HPC; exact dependency versions are in that JSON. No `/dev/snd` is present. Host RAM/process totals are captured as shared-host inventory, not Pi resources or an application allocation.

Before any prototype execution, the existing suite was run:

| Run | Result | Artifact and limit |
|---|---|---|
| Sandbox, 180 s bound | Timed out, exit 124, after partial progress and an early failure marker | `baseline-tests.log`; incomplete suite, no final failure traceback. Do not count as passed. |
| Same source outside documented local IPC restriction, 600 s bound | **469 passed in 85.25 s**, exit 0; no skips reported | `baseline-tests-external.log`; mocks/offline, not live service or Pi qualification. |

Command: `env -u ME3_ASR_API_KEY -u ME3_LLM_API_KEY -u ME3_TTS_API_KEY PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 600s var/cache/venv/bin/python -B -m pytest -q --basetemp=var/scratch/pytest-phase2-baseline-external`. Stdout/stderr were redirected to the experiment log. The sandbox restriction and the incomplete attempt are preserved rather than interpreted as a product failure. No package workflow or lock changed.

## Executed native contract proof

`native_contract.py` composes installed native components: `STTService` → user context aggregator → `GroqLLMService` → `TTSService` → `BaseOutputTransport` → assistant aggregator. No `Session` or custom LLM/tool loop is used. The sole turn owner is the native aggregator with explicit VAD-start and speech-timeout-stop strategies. Fixture VAD events and a 50 ms test silence floor are injected annotations, not real VAD/wake accuracy or production tuning.

The two thin speech subclasses call the existing `SpeechClient`, which starts the actual project ASR and TTS MCP subprocesses. Both negotiated protocol `2025-11-25`. Audio crosses actual stdio MCP as PCM16/16 kHz. ASR append produces interim text, finish produces final text; TTS emits real mock-tone PCM. Only the completion-generator network boundary is replaced: the native Groq/OpenAI-compatible parser receives streamed text and split JSON tool-argument deltas. Model responses, transcript words and output sink are fixtures. This does not test Groq service access or model competence.

| Case | Observed assertions |
|---|---|
| Tool and contextual follow-up | One native function invocation, ordered tool receipt/context, two MCP ASR partials/finals, spoken segments through native TTS/output, follow-up history retained. Three mock completion requests. |
| Interrupt streamed answer | Native accepted onset broadcasts interruption; old output stops/drains; interrupted unspoken `Ready.` tail does not appear in context; follow-up succeeds without executing completed tool again. Three mock requests. |
| Background tool survives speech interruption | Native function registered with `cancel_on_interruption=False`; explicit streamed preamble can play while handler waits; speech interrupt does not cancel handler; result is injected into context and committed once; later input remains accepted. Two mock requests. This is live-process async behavior, not durable task recovery. |

All three cases pass. Each also reuses the ASR and TTS operation slots after pipeline cancellation. `native-final.log` and `native-final-result.json` retain events, context, tool identities, stream closure counts, output bytes, timings and resource snapshots. An initial two-case run is preserved separately.

`session_comparison.py` passes two equivalent legacy cases through the same actual mock MCP pair: remembered word/follow-up and interrupt/no repeated effect. Both use three mock model rounds and leave the Session output queue empty. Native and Session sinks have different chunk sizes and text segmentation; write counts or wall times must not be treated as a speed comparison. These are lifecycle parity fixtures, not equivalent model accuracy or acoustic tests.

## In-process audio adapter proof

`pipewire_transport.py` prototypes `BaseInputTransport` / `BaseOutputTransport` subclasses inside the voice process. It selects explicit nodes, owns `pw-record`/`pw-play`, uses one persistent native Soxr input converter, rejects capture backlog, stops only owned output, fences writes during interruption, and fails closed on device failure. It does not implement AEC: upstream source and reference sink must be correctly configured. It opens no device until explicitly enabled.

Eight tests pass (`transport-qualified.log`, 3.46 s): disabled device gates; explicit sink/stop/reopen; cancelled process-open ownership; failure without fallback; stale write prevented when interruption occurs during open; 48→16 kHz conversion and capture failure; sidecar/native PCM/target/owned-stop equivalence at substituted process boundaries; operator command without `--capture` creates no acquisition/output or devices. No test establishes actual PipeWire behavior.

Two problems were caught and corrected in the isolated prototype/tests:

- An output-open race could write an obsolete packet after interruption. A generation check after open prevents the write; its regression test passes.
- A capture test assumed a 1,920-byte PCM16 mono block at 48 kHz was 10 ms. It is **20 ms**. The fake supplied 2 s, not 1 s; the assertion was corrected. Earlier baseline documentation describing this block as 10 ms is superseded by this correction.

Native Soxr stream defaults to VHQ; the 2 s conversion fixture emitted 30,800 samples before abnormal EOF versus 32,000 input-duration-equivalent output samples. This is buffered/unflushed tail evidence, not a measured acoustic delay. Quality/buffering/flush policy needs explicit comparison in Phase 3. No second converter was added. Capture failure is explicit degraded operation; there is no fallback to a different mic, raw duplex, or implicit hardware action.

The first extended background fixture also failed: it waited for speech from a pending async function without providing any text to synthesize. The corrected fixture emits an explicit preamble, and tests native cancellation independence. `native-extended.log` preserves the timeout; async tool registration does not itself promise a spoken acknowledgment. Early native settings warnings were corrected by fully specifying unsupported settings as `None` in the adapters.

## Available measurements and missing gates

These are **HPC mock/software measurements**, not Pi or provider performance. Heartbeat interval was 10 ms and includes startup. Each resource sample contains the proof process and two actual speech MCP children (three visible owned processes); no task worker, PipeWire, browser, wake or real VAD load is included.

| Native fixture | Aggregate sampled PSS | Main-process CPU seconds | Loop lag p50 / p95 / max, ms (n) |
|---|---:|---:|---|
| Follow-up | 231.26 MiB | 0.307 | 0.071 / 0.621 / **196.012** (402) |
| Interruption | 230.63 MiB | 0.088 | 0.072 / 0.270 / 24.994 (421) |
| Background | 237.29 MiB | 0.080 | 0.068 / 0.180 / 38.341 (479) |

Sampled child/main swap was zero. PSS is a snapshot, not peak; CPU seconds exclude child CPU and are not CPU utilization. The first cold case's maximum loop lag exceeds the proposed 100 ms limit; it remains an unresolved observation rather than being hidden behind p95. Warm cases do not establish a Pi latency bound. Hardware process counts, Pi PSS/RSS/CPU/swap/temperature, echo leakage, false-interruption rate, first audible response, acoustic stop p50/p95 and live provider TTFT are **unavailable**. Existing baseline traces remain historical; they are not pooled into these fixtures.

| Gate | Status / practical limit |
|---|---|
| Current-tree offline baseline | PASS in working IPC environment; incomplete sandbox attempt retained. |
| Both functional project speech MCP services connected to native pipeline | PASS for current mock stdio contracts, partial/final input, tone output, cancellation/reuse. |
| Native context/tool/TTS/turn ownership and memory-output interruption | PASS for the three fixture scenarios only. |
| Native audio helper lifecycle/degraded mode and sidecar byte/target parity | PASS at substituted subprocess boundaries; physical parity not established. |
| Real input wake/VAD, simultaneous capture/render, AEC reference wiring, audible stop, Pi resources | **PENDING_OPERATOR_VALIDATION**. |
| Live ASR event timing, ASR-finalization under slow calls, interruption while ASR finish is blocked | NOT QUALIFIED. Current prototype awaits ASR finish in frame processing; concurrent finalization/event delivery and stale-final fencing need a Phase 3 contract. Do not claim it is already a production streaming adapter. |
| Delivered-word accuracy, long-duration state/resource bound, durable tasks | NOT QUALIFIED; no acoustic word timing, soak or LangGraph migration in this phase. |

The native proof replaces the need for `Session.reasoning/respond` orchestration, per-turn ASR/generation scheduling, two-segment synthesis queue and Session Playback worker **for demonstrated scenarios**. Native aggregation replaces `history` and its tested interrupted-tail behavior. The old input frame completion wrapper and duplicated endpoint policy are unnecessary in the proven native fixture. Device effects/receipt recovery, engagement expiry, actual wake, UI reconnect and arbitrary tool/provider failures still require parity checks. No Session, sidecar, existing test or production source was deleted or migrated.

### Precise legacy removal map, contingent on parity

| Current source | Native replacement / remaining gate |
|---|---|
| `src/me3_voice/session.py:508` — `Playback.start/enqueue/run/stop/close`, bounded `Playback.queue` | Native output/TTS scheduling and interruption. Memory-sink cancellation proven; acoustic stop pending. |
| `Session.begin_input/push_input/commit_input`, `recognize/recognize_stream`, `input_queue(maxsize=32)` | Input transport and MCP STT adapter; partial/final fixture passes, slow-final concurrency remains open. |
| `Session.reasoning`, `respond`, `segments(maxsize=2)` | Native LLM parser/tool dispatch, TTS text aggregation, context pair. Remove generic loop after provider/tool error parity; preserve ME3 authorization policy outside it. |
| `Session.interrupt`, `valid`, per-generation validity and speech state transitions | Native interruption plus adapter generation fencing. Fixture passes; actual wake/engagement and device timing remain gates. |
| `Session.acknowledge`, `history`, spoken segment tracking | Native assistant context aggregation. Unspoken-tail fixture passes; word-level audible context requires qualification. |
| `record_tool_outcome`, `reconcile_task_result`, `background_task_update` | Preserve effect receipts/verification and progress policy until later task-runtime parity; background cancellation fixture does not replace durable recovery. |
| `activate/open_followup/engagement_snapshot/tick`, UI projections | Retain ME3 engagement/interaction policy as thin integration, not a second conversational controller. Not covered by this proof. |

This table identifies removal candidates, not edits performed. The four scripts and one test module remain isolated validation tools, not production replacements.

## Operator procedure — run on Pi only

`ej@rpi5` is a placeholder, not an accessible target. No SSH commands are needed here. Fetch the published exact commit into the Pi checkout using its existing reviewed ME3-only synchronization procedure. The prototypes are tracked under `scripts/phase2/` and `tests/test_phase2_transport.py`; no manual transfer of ignored experiments is needed. Preserve the Pi parent HEAD/index, ME2, private configuration and `var/`; do not run a blind parent pull/reset. Run from ME3 root with its already installed environment. New captures remain private under ignored `var/results/`.

### 1. Read-only inventory and offline validation

```bash
mkdir -p var/results var/scratch
ME3_RUN="var/results/phase2-pi-$(date +%Y%m%d-%H%M%S)"
mkdir -m 700 "$ME3_RUN"
printf '%s\n' "$ME3_RUN"                 # retain this path for other terminals
uname -a > "$ME3_RUN/os.txt"
var/cache/venv/bin/python --version > "$ME3_RUN/python.txt"
wpctl status -n > "$ME3_RUN/wpctl-before.txt" 2>&1
pw-cli --version > "$ME3_RUN/pipewire-version.txt" 2>&1
pw-record --help > "$ME3_RUN/pw-record-help.txt" 2>&1
pw-play --help > "$ME3_RUN/pw-play-help.txt" 2>&1
pw-dump > "$ME3_RUN/graph-before.json"
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  var/cache/venv/bin/python -B -m me3_voice.readiness inventory --output "$ME3_RUN/inventory.json"
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B -m pytest -q --basetemp=var/scratch/pytest-phase2-pi > "$ME3_RUN/regression.log" 2>&1
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B scripts/phase2/native_contract.py "$ME3_RUN/native-contract.json" > "$ME3_RUN/native-contract.log" 2>&1
```

Inspect exit status after each test command and stop on failure; none of these commands opens audio devices. Return inventory/regression results before assuming native compatibility. Record actual USB mic/speaker, physical connections, mic distance, placement, speaker volume, power/cooling, and other active audio owners in a text file in the run directory.

### 2. Temporary reference AEC — Terminal A, attended

Use verified physical `node.name` values from the graph, not defaults or numeric IDs mistaken for names. Confirm existing ME2/audio activity is not being disrupted. The following generates a command, without loading a module:

```bash
read -r -p 'Physical microphone node.name: ' ME3_CAPTURE_NODE
read -r -p 'Physical speaker node.name: ' ME3_PLAYBACK_NODE
var/cache/venv/bin/python -B - "$ME3_CAPTURE_NODE" "$ME3_PLAYBACK_NODE" > "$ME3_RUN/aec-load.txt" <<'PY'
import re,sys
mic,speaker=sys.argv[1:]
assert mic!=speaker and all(re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}',v) and not v.startswith('me3.') for v in (mic,speaker))
print('load-module libpipewire-module-echo-cancel { library.name = aec/libspa-aec-webrtc audio.rate = 48000 audio.channels = 1 capture.props = { node.name = me3.phase2.aec.capture target.object = "'+mic+'" } source.props = { node.name = me3.phase2.aec.source } sink.props = { node.name = me3.phase2.aec.sink } playback.props = { node.name = me3.phase2.aec.playback target.object = "'+speaker+'" } }')
PY
cat "$ME3_RUN/aec-load.txt"
script -q -f "$ME3_RUN/aec-console.log" -c pw-cli
```

At the `pw-cli` prompt inspect `help`, then paste the generated single `load-module ...` line. Keep this terminal/client alive. If the WebRTC plugin fails or nodes are absent, stop and return the error; do not install services or use raw audio as a full-duplex fallback. No boot config is written. `script` only logs the interactive session; if unavailable, run `pw-cli` and save its messages manually.

PipeWire's [CLI documentation](https://docs.pipewire.org/page_man_pw-cli_1.html) specifies that module loading is local to the client; a one-shot command exits and cannot be assumed to retain the module. The [AEC module](https://docs.pipewire.org/page_module_echo_cancel.html) requires playback into its reference sink and capture from its processed source. These commands need installed-version validation; no claim of Pi execution is made.

### 3. Native acoustic acquisition — Terminal B

Use a nonprivate, clearly intelligible **mono PCM16 WAV at 16 or 48 kHz** already available on the Pi. An18s reference converted from the existing reading is prepared privately at `var/experiments/022-native-voice-duplex/reference-reading-16k.wav` on the development machine; transfer only that WAV and its hash/metadata privately if Pi lacks a suitable reference. It is not in GitHub. Alternatively, if the original reading is already on Pi and ffmpeg exists, convert locally without new inference:

```bash
ffmpeg -nostdin -hide_banner -loglevel error -i var/recordings/swift-asr-reading-1.webm \
  -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le -n "$ME3_RUN/reference-reading-16k.wav"
```

If neither file nor an existing local reference is available, report that transfer requirement; do not request a new spoken recording merely to create playback material. No download or inference is required. The script loops this reference; do not change geometry or volume during paired trials.

```bash
read -r -p 'Run directory printed in Terminal A: ' ME3_RUN
read -r -p 'Same physical microphone node.name: ' ME3_CAPTURE_NODE
read -r -p 'Local reference WAV path: ' ME3_REFERENCE_WAV
wpctl status -n > "$ME3_RUN/wpctl-aec.txt" 2>&1
pw-dump > "$ME3_RUN/graph-aec.json"
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B scripts/phase2/operator_native.py \
  --source me3.phase2.aec.source --sink me3.phase2.aec.sink --raw-mic "$ME3_CAPTURE_NODE" \
  --reference "$ME3_REFERENCE_WAV" --output "$ME3_RUN/trial-01" --capture
```

Follow the printed 0/8/16-second prompts: silence for echo-only exposure, then speak over playback, then speak alone. Reference stops on the first native confirmed onset. If it stops during the silent phase, record a possible false interruption; do not call it successful cancellation. This is an audio/VAD test without recognition or reasoning. `--capture` is mandatory; otherwise it opens nothing. Device failure stops the trial and requires text/mock degraded mode, not automatic substitution. Ctrl+C requests owned-child cleanup. The 24 s run is only a smoke; it cannot qualify false accepts per hour or interruption p95.

### 4. Verify active reference routing — Terminal C during playback

```bash
read -r -p 'Same run directory: ' ME3_RUN
pw-dump > "$ME3_RUN/graph-during.json"
pw-link -lI > "$ME3_RUN/links-during.txt" 2>&1
```

Verify links show **pw-play → me3.phase2.aec.sink → physical speaker**, **physical mic → AEC capture**, and **AEC source → native pw-record**. A source/sink name or successful module load alone does not prove this reference is active. Return the graph if interpretation is uncertain.

After the trial, in Terminal A use `unload-module` with the exact module variable returned by `load-module`, or `quit` to terminate this owned client and unload its module. Never destroy arbitrary graph IDs or restart shared PipeWire. Then capture `pw-dump` again as `graph-after.json` and confirm owned AEC nodes disappeared and existing routing is preserved.

Return the run directory's JSON/text/log files, `trial-01/raw.wav`, `processed.wav`, reference WAV (or its hash plus an available identical copy), and placement/volume/subjective observations. Raw taps are separate streams and are not a calibrated common acoustic clock; acoustic onset/stop needs annotation, and AEC attenuation alone does not establish intelligibility. The sampler records owned processes plus visible user PipeWire/WirePlumber PSS/RSS/swap/CPU ticks and loop lag; two diagnostic tap processes are additional to the proposed production topology. Preserve failed captures. Review actual results before more trials or declaring any hardware gate passed.

## Supplemental acquisition for extended offline development

**Use this before finishing the Pi visit.** Existing audio inventory and exact gaps: [audio-corpus-inventory.md](../evidence/audio-corpus-inventory.md). Existing54 guided human takes need not be rerecorded. First run the inventory tool on the Pi too; files remaining there may close additional gaps. Raw recordings stay private; return via the existing private transfer workflow, not GitHub.

Your agent should ask the operator to fix device geometry, state actual distance/volume/fan conditions, and keep the original reference WAV. Use a separate new output directory per trial. Run from ME3 root, with the temporary AEC already loaded and graph checked above:

```bash
# Inventory existing Pi audio first. Metadata/read-only; no inference or devices.
var/cache/venv/bin/python -B scripts/phase2/audio_inventory.py --output "$ME3_RUN/audio-inventory.json"

# Real Pi speech without speaker playback: read a representative subset of
# existing phrases, include pauses/names/numbers and a few unfamiliar sentences.
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B scripts/phase2/operator_native.py --profile speech --seconds 180 \
  --source me3.phase2.aec.source --sink me3.phase2.aec.sink --raw-mic "$ME3_CAPTURE_NODE" \
  --output "$ME3_RUN/quiet-near-01" --capture

# Echo-only: operator silent; reference continues. Repeat using overlap profile
# while reading known phrases, giving a sustained double-talk recording.
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B scripts/phase2/operator_native.py --profile echo --seconds 120 \
  --source me3.phase2.aec.source --sink me3.phase2.aec.sink --raw-mic "$ME3_CAPTURE_NODE" \
  --reference "$ME3_REFERENCE_WAV" --output "$ME3_RUN/echo-near-01" --capture
```

Repeat the last command with `--profile overlap` and a fresh `overlap-near-01` output. Repeat selected speech/echo/overlap at another intended condition, labeling volume/distance. These three corpus modes disable automatic native interruption to retain complete utterances/overlap; they **do not qualify barge-in**. No ASR or real task action runs.

For actual barge-in, repeat the original `--profile smoke`24s command with fresh trial directories, at least50 independent trials for the declared p95 gate. Keep the8s silence /8s overlap /8s user-only prompts. Preserve false stops rather than restarting until a passing sample is found. The agent can loop trials but must coordinate each onset with the operator. Do not use echo/overlap corpus mode for cancellation scoring.

Use `--profile speech --seconds 300` with a fresh directory for >=100 wake examples at natural levels/positions and annotated confusable phrases. This records material for later wake replay; the acquisition pipeline itself runs Silero, not the production wake detector. Use `--profile negative --seconds 7200` only in a consented environment with sufficient storage for the2h negative exposure gate. This records room sounds without playback; false-wake rates require subsequent labeled detector evaluation. Do not equate elapsed recording time with a passing detector score.

For continuous capture/render, an `echo` profile with `--seconds 1800` can collect a30min stress session after successful smoke. Reference keeps playing, automatic interruption is disabled, so it proves neither normal interruption policy nor wake reliability. Check returned logs for gaps/errors, graphs and actual audibility. This prototype is not yet Pi-qualified. Do not leave a failing or excessively loud trial running.

Two48k mono16-bit taps consume about11.52MB/min of raw PCM, or23.04MB/min while retaining both PCM and WAV copies, plus reference/logs. A2h recording can therefore need about2.8GB, more if additional evidence is saved; check free disk space first. WAV conversion streams chunks and resource snapshots stream to `resources.jsonl` with only100 recent samples retained in JSON. These are implementation bounds, not measured Pi RAM or disk performance. Stop on disk/device failure; preserve incomplete evidence.

**Return checklist:** all previous inventory/routing/native audio files plus every trial's `resources.jsonl`, full reference WAV/hash, actual condition notes, and a simple UTF-8 CSV with `trial_id,profile,condition,raw_speech_start_s,raw_speech_end_s,actual_words,raw_audible_stop_s,false_interrupt,first_words_cut_off,notes`. Leave unknown fields blank; software events are not acoustic timestamps. Annotate offsets separately for raw and processed files if clocks differ. Transcripts and precise annotations may be completed offline later; preserve intended phrase IDs/condition notes now. Return an independently checked transcript for quiet/overlap phrases and actual wake counts when available; scripts do not infer these. Failed trials and quiet/negative periods must remain included. Keep raw files private and never commit them.

The user authorized existing project recordings for relevant new service evaluations on2026-11-09; historical consent manifests remain unchanged. No paid service or next phase is authorized by the collection plan. Returned material supports extended offline work, but hardware changes and final acoustic qualification can still require another visit.

## Smallest remaining Phase 3 work and stop boundary

After returned physical data closes Phase 2 gates: turn these adapters into reviewed production interfaces, make ASR finalization/event reception nonblocking with stale-result fences, version batching/partial-event contracts, test actual streaming providers and cancel/clear behavior, and measure input backlog/Soxr and output acoustic timing. Integrate real wake/VAD/engagement policy and UI projections only through native ownership, with current Session retained as a rollback release until parity. No LangGraph migration, cloud-model replacement or whole-application rewrite was performed here.
