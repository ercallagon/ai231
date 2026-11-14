# Latency benchmark contract

Status: acoustic benchmark method remains proposed. HPC mock lifecycle events and file-VAD offsets now exist under var/results/; they are not Pi response, activation or barge-in latency measurements. No 20-trial acoustic baseline has been collected.

## Readiness qualification before agent/model optimization

User priority: qualify existing hardware and wake/AEC reliability first. Freeze the exact mic, output device, rate, routing, gain/volume, room, phrase/model/threshold, speaking distance and speaker position. Test the actual wired speaker, not only headphones. Keep device failures and missed activations in the results.

Start with stable simultaneous capture/playback and recovery, then positive wake attempts, non-wake challenge audio and assistant-only playback. Add wake immediately followed by a command, natural pauses, explicit/inactivity exit, and simultaneous user/assistant speech. Count input overflows/reopens, missing first words, assistant self-triggers and suppressed genuine user speech separately. AEC success requires preserved user speech, not just a silent processed recording. Attribute model error versus capture/routing error only with appropriate raw/processed/reference evidence.

Proposed initial qualification targets, not measured results or an agreed statistical guarantee: at least 98 detections in 100 real-speaker wake attempts across documented speakers/distances, zero observed false activations in at least one hour of non-wake challenge audio, and zero observed self-activations in a 30-minute assistant-playback test. Include unseen speakers/recordings after threshold tuning; do not tune and report on the same samples. Report sample counts, misses, exposure duration and uncertainty; zero observed events is not a zero underlying false-accept rate. Extend exposure/repetitions before any broader reliability claim. Fix the operating envelope after the first baseline; choose numerical latency targets only after measuring it.

End the hardware gate with one verdict: existing setup sufficient, sufficient under explicit operating constraints, or a specific deficient component needs replacement. Reuse the same investigation folder for follow-up trials; retain raw recordings and logs under var/. No physical readiness trials have been performed for ME3.

## Clock and events

Use a Pi-local monotonic clock for event differences. Add UTC wall time only for run identification. Do not subtract remote provider timestamps from Pi timestamps without a justified clock mapping.

Record run/session/turn IDs, revision, model IDs, configuration, Pi/OS/audio setup and network conditions. Define and instrument: wake detected, capture ready, actual user speech onset/end, endpoint decision, ASR final, LLM first token, TTS first chunk, first actual playback, playback stopped, tool request and verified completion.

ASR final time is not actual user speech end. Enqueued playback is not first audible output. Validate key acoustic boundaries using recordings with an appropriate playback reference/loopback or external measurement; report measurement uncertainty. Keep instrumentation overhead small and document the method.

## Primary metrics

| Metric | Definition |
|---|---|
| Activation | Wake-phrase acoustic end -> ready to accept input; also report detector-to-ready |
| Conversational response | User acoustic speech end -> first audible response |
| Barge-in | User acoustic speech onset during playback -> assistant audible output stops |
| Task completion | User acoustic speech end -> verified real action completion |
| Endpoint delay | User acoustic speech end -> system end-of-turn decision |

Report stage timings separately. Streaming stages overlap, so do not add their durations and call that end-to-end latency. Compare ME2/ME3 only where boundaries and conditions match.

## Initial protocol

Freeze a short script/config before collecting results. Start with at least 20 measured trials per scenario, report sample count, median, p95, failures and timeout rate; describe small-sample p95 as preliminary. Separate warm-up/cold-start from steady-state results. Keep failures in the report, rather than dropping them from the sample silently.

Scenarios: short question; multi-turn follow-up; brief mid-sentence pause; interruption during a long answer; real tool action; tool correction; quiet-room and speaker-playback conditions. Add controlled network failure and no-speech/noise cases for robustness, outside successful-latency summaries.

For activation, include intended wake phrases and a fixed non-wake audio set; report misses and false activations alongside latency. Fix speaker/mic position and playback volume. Do not claim general wake-word reliability from synthetic inputs.

Write raw JSONL, recordings and generated summaries under var/results/ and var/recordings/. Commit only a selected sanitized evidence report under docs/evidence/, with enough metadata and commands to reproduce it. Do not store private conversations or API keys in versioned evidence.

Choose performance targets after the first measured baseline. Do not present vendor latency figures as Pi end-to-end results.

## General-agent demonstration, clarified 2026-10-27

Include unfamiliar task compositions using configured tools, not only fixed device intents: retrieve information and save a cited report; compute from a local CSV and create a checked output; write/run a small program and inspect its result; interrupt or correct a longer job. Use bounded workspaces and actual accessible accounts. No training is required.

Separate first meaningful acknowledgment, first substantive answer and verified task completion. A filler or an "I'm working on it" message must not count as a completed answer or action. Report task success, correctness/artifact checks, tool failures and duplicate effects alongside latency. Freeze a reproducible task set for measurements, then demonstrate an additional unseen request to show generality without claiming guaranteed success on any possible task.

The optional Onit artifact route adds a cold worker-readiness interval and a tool/task span. Measure worker startup separately from user-turn latency when prewarming succeeds; if the task arrives before readiness, its wait belongs in the observed turn. `tool_duration_ms` covers the task executor only for completed, correlated turns. For a real task, retain the model-request count, tool-call count, artifact verification, failure and interruption outcome. The offline SSE fixture in `var/results/onit-task-offline-benchmark.json` tests Onit's agent/tool loop and local file creation; its warm task time is not real model inference, acoustic response or general task capability. The existing OpenRouter candidate failed one arithmetic correctness trial, so do not trade correctness for a short software interval.

The worker now collects Onit's per-model-request first generated chunk, model span and provider-reported completion count when available; absent token usage stays unavailable. The prewarm comparison in `var/results/onit-task-offline-metrics.json` and `onit-task-offline-prewarm.json` uses an in-memory SSE fixture, so its apparent first-token wait includes local SDK setup and **no network inference**. One later live artifact trial measured the Swift tool span but preceded this per-request instrumentation. It also exposed an older trace accounting error: `llm_request_total_ms` had included awaited tool execution on tool-call turns. Session now closes that request span at the tool-call event; keep the historical `onit-session-live-trial.jsonl` trace with its limitation, and use new traces for model-versus-tool comparisons.

## External benchmark applicability, reviewed 2026-10-27

The likely reference is Lin et al. (2025), [Full-Duplex-Bench](https://arxiv.org/html/2503.04721v1), arXiv:2503.04721. It evaluates pauses, backchannels, turn transitions and interruptions. Use it for conversational scenario design. Its original English/model-focused setup does not establish Pi acoustics or tool success. Its post-interruption response latency is distinct from our onset-to-audible-stop metric.

Recommended ME3 adaptation: freeze pause and correction scripts, distinguish listener acknowledgment from task cancellation, log premature turn-taking and missed responses, and retain the acoustic timing definitions above. Call modified protocols "FDB-inspired ME3 evaluation"; claim reproduction only with the specified version, data, timing rules and scorer. Do not present an adapted small trial set as a published leaderboard result.

The [upstream repository](https://github.com/DanielLin94144/Full-Duplex-Bench) now separates older static evaluation from newer real-time and tool-use versions; record the version and revision rather than cloning mutable main and calling every result "Lin 2025." [FDB-v3 (2026)](https://arxiv.org/abs/2604.04847) adds human disfluent speech and multi-step tool calls, making it a useful second reference for the professor's general-agent requirement.

The released [v3 pipeline](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/main/v3/README.md) uses mock task APIs and a LiveKit inference setup. Its task-completion timing identifies key information in the spoken response; ME3 separately requires verified action completion. Adapt scenario/argument checks to our runtime without making LiveKit or all benchmark dependencies part of the deployed Pi stack. Benchmark mock API success and actual task effects must be reported separately. Hosted judging or inference requires the same access/budget authorization as other live calls. No external benchmark/data download or execution has occurred.

## Pipeline observability informed by the production voice course

Use [LiveKit's stage definitions](https://docs.livekit.io/reference/agents/session-traces/) as an observability cross-check: transcription availability, end-of-turn decision, LLM time to first token, TTS time to first byte and tool duration. ME3 must additionally record local queue/decode/device delay and acoustic first playback/stop. Preserve spans with correlation IDs rather than adding overlapping stage durations. Cold model load and streaming lookahead belong in the configuration/run record. No LiveKit runtime or cloud logging is required to implement this method locally.

For configurable wake evaluation, report boost, threshold, tokenizer/model hashes, audio chunk size, leading silence/alignment and stream-reset policy. Keep tuning and held-out data separate. Synthetic variations from one synthesizer are correlated development cases, not independent real-speaker samples or a published wake benchmark.

## Implemented diagnostic trace prerequisite

`python -B -m me3_voice.cli demo --trace var/results/<new-name>.jsonl` exports the mock session's diagnostic events. `python -B -m me3_voice.telemetry <trace> --output var/results/<new-summary>.json` validates and summarizes them. No dashboard, cloud collector or provider account is needed. Use fresh paths; existing trace/summary evidence is preserved. Events exclude transcript/audio payloads and correlate by run, session, turn, clock, mode and TTS segment. The older demo JSON retains its mock conversation history separately.

Available durations are ASR request-to-final, LLM request-to-first-segment, per-segment TTS request-to-first-chunk, submitted endpoint-to-memory-sink-write, and interruption-request-to-mock-stop when both boundaries exist. These are software observations, not acoustic results, token-level TTFT or provider billing data. Missing boundaries remain unavailable; error/interruption/control-only outcomes stay visible. The finite exporter does not yet implement continuous bounded telemetry, cross-process clock mapping, audio alignment, WER scoring or cost accounting; finite p50/p95 software aggregation is now available.

The supplied reference's additive E2E formula and numerical scorecard are provisional examples. Collect direct acoustic intervals; retain stage overlaps, text aggregation delay, queue occupancy/underruns, audio device buffering and warm/cold state when those can actually be instrumented. Synthesis RTF and first-byte timing alone do not establish smooth or timely playback. No target or measured acoustic latency was changed by this review.

UI transcripts are a separate opt-in text projection, not diagnostic telemetry. Their publication/polling time is not ASR finalization, audible playback or acoustic barge-in latency. Draft/final/interrupted and sink-acknowledged segment tests establish lifecycle/display behavior only; exact words heard require real device progress/alignment. The current folder/report polling cadence must not be reported as model/audio latency. Keep real conversation text out of selected versioned evidence unless deliberately sanitized; the generated mock preview remains under ignored var/.

## Wake/VAD report scoring

Implemented diagnostic scoring is independent of ASR/TTS/LLM. `readiness replay` and `monitor` use the same FrontendMonitor; the latter accepts raw PCM stdin. Neither opens a microphone, performs AEC or drives task/playback behavior. The production conversation runtime remains separate. The monitor records up to 1,000 events/ranges, retaining counters/truncation flags, and up to 10,000 CPU-call times per model. Per-call timings exclude model startup, device/network buffering and report/UI delivery. Replay processing_rtf includes controller/preprocessing/report preparation after initialization; stdin RTF is unavailable because producer waiting is mixed in. These are CPU/input-clock observations, not acoustic latency.

Put human annotations under var/recordings/<trial>-labels.json, using schema version 1. For example, the following is an **incomplete template**, not ground truth:

```json
{
  "schema_version": 1,
  "complete": false,
  "input_sha256": "copy input_sha256 from the exact completed replay report",
  "audio_origin": "recorded",
  "split": "development",
  "wake_windows": [{"start": 0.0, "end": 2.0}],
  "speech_ranges": [{"start": 0.0, "end": 1.0}]
}
```

Replace example intervals with independently annotated values. Set complete=true only after annotating the entire clip. Use synthetic for generated voices, recorded for actual recordings; neither grants Pi verification. The hash is of the consumed original PCM bytes, excluding the WAV header; the report explicitly records that scope. Do not reuse the older whole-WAV hash receipts as this hash. Both development and qualification splits are supported, but separation is a data-collection obligation, not something the scorer can infer. Synthetic qualification is still not real-speaker qualification.

Wake windows are nonoverlapping **acceptable input-consumed detection intervals**, in seconds from clip start, with an inclusive start/exclusive end. Define tolerances before inspecting detector results. They are not speech annotations or acoustic delay. One event matches each window; unmatched/duplicate events count as false activations; unmatched windows are misses. Use independent clips for independent wake attempts because an active session deliberately suppresses further wakes. Only all-negative clips report exposure hours/false activations per hour; mixed clips report counts without inventing negative exposure. No dataset-wide rate or statistical reliability guarantee is implemented.

Speech ranges cover all audible human speech, including the wake phrase. Ranges must be sorted/nonoverlapping and within the clip. VAD precision/recall use the midpoint of each complete 512-sample frame against those ranges and the reported voiced ranges; the unscored tail is excluded. Undefined denominators stay null. This measures classification against annotations, not semantic end-of-turn quality. Pause/endpoint/inactivity behavior is visible separately in frontend_events.

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.readiness score var/results/wake-vad-trial.json --labels var/recordings/trial-labels.json --output var/results/wake-vad-trial-score.json
```

Scoring refuses mismatched origin/hash, incomplete/truncated/non-EOF reports and overlapping/out-of-bounds ranges. The score fingerprints both report and labels; the dashboard shows metrics only beside the matching report SHA-256. Current automated scoring checks use deliberately constructed test data, not collected real-speaker evidence. No real wake/VAD accuracy baseline has been established. See [Pi runbook](../deploy/rpi/README.md) for capture prerequisites and execution.

## Connected mock scenario and file verification

Use the existing CLI entry points, with fresh trace paths:

```bash
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.cli demo --scenarios --output var/results/scenarios.json --trace var/results/scenarios.jsonl
PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.cli replay var/recordings/trial.wav --mock-conversation --audio-origin recorded --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 --keywords-file var/cache/wake/hey-swift-keywords.txt --pace 1 --output var/results/application-trial.json --trace var/results/application-trial.jsonl
```

The deterministic cases now additionally cover immediate wake/command, a 320 ms mid-sentence pause, slow speech RPC timeout/recovery and a simulated disconnect after an actual MCP partial; earlier cases cover normal conversation/follow-up/explicit exit; interruption during reasoning; interruption after the first TTS chunk while synthesis is active; buffered playback interruption with completed mock effect, status follow-up and correction; controlled provider failure with recovery. Passed means the expected lifecycle occurred, including expected faults. Each case records its run ID, actual monotonic elapsed time, controlled delays/barriers, checks/errors and diagnostic spans. A 60-second blocked sink is cancelled, not waited out; it is an input fault/barrier rather than a measured speaker buffer. The independent connected file path and its unit test cover wake-once, pre-roll, follow-ups, inactivity and re-wake. A synthetic repeated-WAV test does not create independent speakers.

The mixed application report explicitly separates actual CPU frontend inference from mock ASR/reasoning/TTS/tasks and memory playback. Mock ASR script entries are selected when a stream opens, independent of the WAV's words; partials exercise transport/projection only. File replay --pace schedules a nominal sample timeline against the local monotonic clock and reports lateness; it is not an acoustic clock or performance target. Plain CLI replay without --mock-conversation remains diagnostic-only and starts no speech providers.

All trace durations use the local monotonic clock and run/session/turn identity. TTS spans additionally match segment_id; tool spans match tool_call_id. ASR's legacy asr_request_ms name now represents the stream's start-to-final span, including producer input/pacing. endpoint_to_asr_final_ms separately measures the commit-to-final span. Neither is standalone provider inference latency. `tool_duration_ms` observes the task call between Swift's start/completion events: currently an in-memory mock action or, when explicitly enabled, a confined Onit draft-artifact task. It is not a physical effect or the full user-to-task time. Finite aggregates group already-computed durations by mode/clock/metric, with count, p50, linearly interpolated p95 and max; no averaging across incompatible clocks/modes or summing overlapping stages. Small-sample percentiles are descriptive, not reliability guarantees. Missing boundaries remain unavailable; errors, interruption and control-only turns remain explicit. Acoustic response/barge-in stay null.

For later Pi real recordings, retain the existing exhaustive annotation schema above. Freeze room/device/routing/native rate/gain/volume, speaker position/distance, model/keyword hashes and thresholds before recording. Collect independent positives, confusables/no-wake, immediate command, follow-up/pause, explicit/inactivity/re-wake, assistant-only and double-talk clips; keep misses, capture faults, raw/processed/reference audio and failed trials. Annotate speech and acceptable consumed-offset wake windows before examining detector output, mark origin recorded and separate development from held-out qualification. Use readiness replay/score for classification; use connected application replay separately for lifecycle tests. Do not score mock ASR text as transcription accuracy or infer AEC/full-duplex acoustics from these software runs. The earlier 7/12 synthetic positives and 0/30 negatives remain limitations, not superseded by successful connected smoke tests.

### Natural interaction, proxies and adverse inputs

The extended runner records controlled ASR delay/RPC deadline and simulated-disconnect flags, plus pause-profile observations. A 320 ms scripted pause remains within one utterance. An 800 ms pause splits at 640 ms silence, while 1024 ms holds it together at an additional 384 ms endpoint wait. That is a known turn-taking tradeoff, not a solved semantic endpoint benchmark; use --endpoint-ms only for frozen-profile comparisons and retain failures. Do not reduce the silence wait to improve a latency number while cutting off users.

last_voiced_frame_to_sink_mock_ms starts at the latest VAD-positive file frame's consumption timestamp and ends at the first memory write. last_voiced_frame_to_verified_mock_task_ms ends at a verified in-memory task result, not a device effect. tts_first_chunk_to_sink_mock_ms observes packet dispatch/queueing. These are not physical speech-end, acoustic output, intelligibility or task-effect timestamps. They remain unavailable when their boundaries do not exist. The explicitly named speech_end_to_first_audible_response_ms, interruption_to_audio_stop_ms and verified_physical_task_completion_ms remain null until appropriate hardware/service effect measurement exists. Local sink stop is verified before deliberately delayed remote cleanup; no acoustic stop duration is inferred from it.

For background noise and speaker echo, distinguish waveform perturbations, simulated role labels, prerecorded real environments and actual simultaneous capture/render. Experiment 003 uses seeded white-noise/delayed-copy perturbations and the unchanged synthetic speech fixture through actual CPU readiness inference; it is not an AEC test or a room impulse response. Do not count its playback-like baseline as real assistant-only exposure, or tune thresholds and report on the same inputs. Pi acceptance still needs real noise-only, user-only, assistant-only and double-talk with raw/processed/reference capture, clipping/overrun checks and held-out speaker trials.

### Pipecat input adapter verification

Connected application replay now passes native PCM packets through a persistent Pipecat input pipeline before the same frontend/Session/MCP path. input_pipeline identifies pipecat-streaming-pcm and input_pipeline_frames counts acknowledged packets, not wake/VAD frames or audio quality. Only ingestion is acknowledged per packet; EOF separately drains the final response. Direct controller tests and Pipecat ingress tests compare preserved onset, follow-ups, timeout/re-wake, partial recognition and interruption/task history. No device/output transport, Pipecat-native turn analysis or acoustic timing is implied. Current commands/results are in PLAN.md. Compare source/clock/configuration and report packet cadence when interpreting software spans; one repeated synthetic replay cannot establish a framework latency improvement.

### Service/turn-policy instrumentation, 2026-10-28

turn_analysis_ms measures the actual optional CPU analyzer call between monotonic boundaries. It excludes wake/VAD and device latency; an async executor does not make capture nonblocking because the current file-ingestion adapter awaits its decision. llm_request_to_first_delta_ms measures each reasoning request to its first provider text delta, separately from the existing coherent first-segment/TTS spans. These boundaries are not acoustic response or guaranteed substantive-answer timing. An interrupted or failed request may have no matching end boundary.

Connected replay reports explicitly record per-role mock/external backend modes and silence/smart-turn policy. An external label means the adapter was configured; inspect turn outcomes before asserting successful service inference. Offline wire peers and dummy credentials are contract validation, not HPC-hosted model execution or live cloud performance. Request counters bound attempted connections/POSTs per provider instance/run, not billed dollars, tokens or a persistent quota. Code reads no real credential during offline tests.

Smart Turn smoke input is the existing generated phrase repeated around digital silence. Actual model completion on those two examples neither proves preservation of natural pauses nor supersedes the earlier synthetic wake misses. Use independent real utterances with incomplete clauses, hesitation, code-switching and corrections, annotate intended turn ends, and compare premature commits, extra wait, failure rate and CPU/RAM before adopting a policy. Record network/handshake time and compare persistent speech connections only once live access exists; current per-operation WebSockets prioritize cancellation fences. All acoustic measurement fields remain unavailable.

### DGX3 contract verification boundary

The Qwen2-Audio adapter buffers an utterance for final-only recognition; compare endpoint-to-ASR-final rather than claiming incremental transcription during input. Qwen3-TTS SSE audio is validated at 24 kHz and persistently resampled to 16 kHz. Software tests cover packet fragmentation, terminal completion, errors/disconnect and local cancellation; they do not establish server GPU abort or audible stop. Read-only health/schema/catalog probes are service reachability evidence, not inference, throughput or model latency. Authenticated DGX3 software smoke timings are now recorded in PLAN.md, including two actual-wake Pipecat turns. These small synthetic-file samples do not establish warm/cold behavior or recognition quality; collect independent comparable trials before changing buffering or endpoint defaults. Acoustic fields remain unavailable until Pi capture/render qualification.

### Laptop browser AEC A/B prerequisite

Canonical guided probe: src/me3_voice/ui/audio_probe.html; generated laptop copy: var/results/laptop-aec-check.html. Download the page and a speech reference (for example var/recordings/synthetic-speech.wav) to the Windows laptop using VS Code Remote Explorer's Download action. Open the HTML in current Chrome/Edge. If microphone access is unavailable on file URLs, place only these test files in a local test folder and run `py -m http.server 8081 --bind 127.0.0.1` there in local PowerShell; open http://127.0.0.1:8081/audio_probe.html. No SSH forwarding or HPC microphone access is involved.

The user explicitly requested laptop acoustic testing. The page requests capture only after Start, loops the selected local speech reference through the first 16 seconds, then pauses playback and ends capture at 24 seconds. On-screen prompts guide three 8-second phases: stay quiet, speak over playback, then speak alone. A short reference such as a 4-second MP3 repeats automatically; playback starts the phase timer, and the user only needs to press Play if the browser blocks autoplay. It releases tracks on stop. Save microphone WebM/MP4 and settings JSON for AEC off/on; keep the original reference file and record microphone/speaker identities, distance and volume separately. Use the same settings/geometry in both trials.

Reported echoCancellation can differ from the requested value. Noise suppression/AGC are requested off where supported; OS/device processing may persist. Browser handling of local media playback as an AEC reference is implementation-dependent. Retain failures and configuration mismatches; no guaranteed WebRTC AEC3 algorithm/version is claimed. This is qualitative A/B capture, not calibrated ERLE, synchronized raw/reference recording, acoustic latency, ME3 live device integration or Pi qualification. The page performs no upload/network model call; importing resulting files is a separate user action. Browser API references: https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation and https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia .

### Controlled ASR reading check

For an interpretable recognition trial, use the local capture page `var/results/asr-reading-check.html` generated from `src/me3_voice/ui/asr_probe.html`. Download/open it on the laptop in current Chrome/Edge, use the intended wired mic in a quiet room, and keep speaker playback silent. Read its fixed passage exactly when prompted. Save the audio and matching JSON under var/recordings. If file URL microphone access is blocked, serve only the downloaded page from a local laptop folder with `py -m http.server 8081 --bind 127.0.0.1` and open its localhost URL. No HPC microphone or SSH port forward is involved.

Then run `PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B scripts/evaluate_asr_reading.py var/recordings/swift-asr-reading-1.webm --output var/results/asr-reading-evaluation.json`, changing the basename to the downloaded file. The evaluator invokes actual DGX3 ASR through our MCP server with the configured per-run authorization. Report raw transcript, normalized word edits and WER together; manual listening and reading fidelity remain necessary. One fixed passage is a smoke check, not a representative accuracy estimate or acoustic latency benchmark. Use independent passages/speakers later for qualification.

### Browser bridge software boundaries

The loopback voice bridge writes a payload-free `var/results/voice-<run>.jsonl` and summary JSON after a browser session. Use the same run/session/turn/monotonic clock when pairing events. `endpoint_to_browser_send_ms` and `last_voiced_frame_to_browser_send_ms` measure server WebSocket-send initiation, while `interrupt_to_browser_stop_sent_ms` measures server stop signaling. Browser `onended` acknowledgments gate confirmed text but do not prove sound reached a listener. The browser may stop scheduled audio locally before the server stop event; that local-to-acoustic interval needs instrumented device capture. Keep `speech_end_to_first_audible_response_ms`, `interruption_to_audio_stop_ms` and verified physical task latency unavailable until synchronized mic/speaker/Pi observations exist.

### Stage metrics for tuning

The canonical trace/summary and Diagnostics Stage details now carry these per-turn software measurements. Durations are computed only from matching run/session/turn/clock identities; TTS segments are additionally matched by segment ID. Interrupted requests have no fabricated completion time.

| Stage | Current measurement | Interpretation and next measurement |
| --- | --- | --- |
| ASR input | `asr_input_duration_ms` = submitted 16 kHz mono PCM samples / 16; `asr_input_incremental` reports a stream opened at the Session speech boundary; `asr_partial_observed` records whether a partial arrived | Includes wake pre-roll, silence and possibly buffering inside a provider adapter. It is not spoken duration or proof that the model itself streams. Compare with `endpoint_to_asr_final_ms` and ASR correctness. Qwen2-Audio remains final-only at its service contract. |
| LLM | `llm_request_to_first_delta_ms` and `llm_request_total_ms` per reasoning request | First *text delta* is a client-observed content boundary, not necessarily one token. True tokens/second and time per output token require a trusted provider token count/tokenizer and final usage per request; leave them unavailable until the chosen reasoning endpoint exposes that contract. Tool rounds are separate requests. |
| TTS | `tts_first_chunk_ms` and `tts_render_total_ms` per speech segment | First chunk is decoded PCM delivered through our MCP speech path, not the first HTTP byte or audible sound. Total render ends on normal stream completion, independent of browser playback/queue drain. An interrupted segment has no total. |
| Turn-taking | Frontend wake/VAD/utterance/endpoint sample offsets, `endpoint_to_asr_final_ms`, `endpoint_to_browser_send_ms` | An annotated acoustic speech onset/end is needed to call a delay “VAD detection latency” or “EOU error.” Record true onset/end and acceptable pause boundaries in synchronized mic audio, then compute onset→VAD-positive, speech-end→commit, premature-end rate and late-end rate. |
| User experience | `endpoint_to_browser_send_ms`, `interrupt_to_browser_stop_sent_ms`, confirmed task receipt | These are software proxies only. The primary acceptance measures are speech-end→first audible response, confirmed interruption→acoustic stop, and verified task-effect latency, measured with synchronized audio/device evidence on the intended hardware. |

Read the stages as a waterfall, without summing overlapping spans. A long submitted clip with modest endpoint→ASR-final points to capture/endpoint policy or front-end waiting; a long endpoint→ASR-final points to final ASR buffering/service/network; a long LLM first-delta span points to reasoning queue, prompt/tool context, or transport; a long first-PCM span points to TTS startup/segmenting; a fast first PCM with slow browser send or acoustic onset points to buffering/output. Check recognition error, premature cutoffs, cancelled-turn rate and task correctness beside latency before tuning. Compare repeated trials by provider, audio duration, turn type and cold/warm state; report failures and p50/p95 with sample size. The UI keeps these details in Diagnostics; the Assistant conversation remains free of debug numbers.

The paced synthetic live-service browser-protocol trial under `var/experiments/007-browser-voice-path/` observed 2766.7 ms from endpoint to first browser-protocol PCM packet in one run; its first turn was interrupted and second completed. The recorded-input memory-sink trial saw 4390.4 ms endpoint-to-write in a different run/utterance. Do not compare them as a model ranking or acoustic latency. The Chrome fake-mic wake comparison under `var/experiments/008-browser-fake-aec/` varied on recheck and cannot qualify AEC or wake accuracy. Keep the current 1024 ms endpoint silence until held-out real pauses quantify premature cutoff versus extra wait; only then consider a shorter timeout or the optional Smart Turn gate.

## Guided reusable recording corpus

Use the canonical recording utility when a live trial reveals recognition problems. It collects reference-labelled laptop audio once, so developer comparisons can replay identical bytes instead of asking the speaker to reproduce the failure. The page never invokes inference or tools. It uses one authorized browser microphone stream for recording and the measured level meter; it plays no audio cues, uploads each clip directly to the project, releases the microphone when the set stops, and preserves earlier takes.

From the HPC project terminal, stop the user-owned voice bridge first if it occupies 8765, then launch:

```bash
cd $ME3_ROOT
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 var/cache/venv/bin/python -B -m me3_voice.recording_lab serve --port 8765
```

Keep VS Code port 8765 forwarded and open `http://127.0.0.1:8765/recording-lab` in Edge. The same page is also available from the live bridge at **Diagnostics → Audio → Guided recordings** after its source is reloaded. Close an active voice conversation before recording. Keep speakers silent and use the intended USB microphone. Select **Core transcription**, check the storage/evaluation consent, click **Start guided recording**, and allow microphone access. Read only the displayed phrase when “Speak now” appears; omit “Hey Swift” unless it is printed in that prompt. Countdown and clip endings are automatic; **Done with this phrase** is an optional early finish. Core contains six clips and takes roughly two minutes. Retake only a genuinely misread clip; preserve the original failure input for analysis.

The page confirms each successful upload. Completed recordings and `manifest.json` are stored under `var/recordings/guided/<batch-id>/`; no manual file transfer is needed. The manifest contains stable prompt/scenario IDs, expected spoken text, expected task kind where specified, initial/per-take browser audio settings, consent scope, take IDs, timestamps, duration, notes, file hashes and completion flags. Earlier takes are never overwritten. A stopped partial take is saved but excluded from replay by default. Completed clips can be resumed after refreshing the page. A failed upload retains the browser blob and offers retry plus an individual local download; refreshing before saving an unsent blob can lose that unsent take. No archive is generated.

The catalog in `src/me3_voice/recording_lab.py` is the single source for six sets:

| Set | Clips | Purpose |
|---|---:|---|
| Core transcription | 6 | Questions that previously became answers, full words/initials and a repeated question |
| Wake, pauses & follow-ups | 7 | Silence, wake-plus-command, trailing conjunction, natural pause and contextual follow-ups |
| Lights & music | 14 | RGB commands, artist correction, volume, transport controls, explicit stop correction and ambiguous music fragment |
| Stateful tasks | 9 | Display setpoint, timer/alarm, clock/timezone correction, weather and reminders |
| Complex agent requests | 5 | Bounded website request, progress question, correction, stopping speech separately from cancelling a task |
| ASR misses & control variants | 13 | Repeats the silent/music/time/weather cases that failed transcription, then adds explicit volume, relative volume, RGB, numeric brightness, and timer replacement phrases |

The full catalog contains 54 clips. Start with Core; use the supplemental set after reviewing the first five sets. All task phrases on this page are recordings, not instructions to execute. The complex set prepares inputs for the actual harness; recording a phrase does not implement or verify that action. The supplemental repeated phrases are separate takes, not a replacement for the original evidence.

### Developer replay

The evaluator uses our existing `SpeechClient`/ASR MCP and optionally the canonical `Session`; it does not introduce another harness or a second tool router. It verifies clip paths/hashes, decodes locally through ffmpeg, selects the latest complete take per prompt, preserves previous reports, checkpoints per-clip results, and requires a finite request allowance. Uploading does not start evaluation. Example after the user has finished a six-clip Core batch (replace the batch ID shown by the page):

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
var/cache/venv/bin/python -B -m me3_voice.recording_lab evaluate \
  var/recordings/guided/BATCH_ID \
  --backend-config var/cache/voice-backends-asr-check.json \
  --secrets-file var/cache/voice-backends.env \
  --max-requests 6 \
  --output var/results/guided-core-asr.json
```

Use a new output filename for each comparison. Add `--case capital` to isolate a failing clip with `--max-requests 1`. The report preserves raw ASR text, expected text, word edit counts, WER (unavailable for an empty reference), decoded input duration and observed ASR MCP time, including errors. An answer substituted for a question fails the word comparison. Punctuation/case normalization is permitted; semantic rewriting is not. No confidence score or real-speaker population accuracy is invented.

Add `--session` to pass each recorded clip through the existing session path in scenario order, keeping reasoning/TTS mocked by default. This checks connected direct routes and mock effects, not general real-model tool selection. Explicit `--live-reasoning` and/or `--live-tts` opt into the configured university endpoints within the same per-role request allowance; use `var/cache/voice-backends.json` and `--llm-secrets-file var/cache/dgx3-llm.env` when testing the university reasoning model. These optional live flags were not exercised by the recorder implementation pass. Saved assistant WAVs, task receipts, downstream error classification and monotonic session traces are available when the respective stages run. All task adapters remain mock-only: no Spotify, GPIO, calls/messages, Onit worker or arbitrary project writes. The complex agent set requires a separately authorized actual-harness trial to establish execution capability; the evaluator reports absent routes honestly.

Direct ASR replay bypasses wake/VAD. Session replay uses committed clips and does not recreate wall-clock pauses between separate recordings, real interruption during speech, wake detection or acoustic echo. Wake/pause clips can also be converted and supplied to the existing `me3_voice.cli replay` frontend for separate endpoint/wake analysis; preserve that distinction in reports. Neither the microphone meter nor browser AEC settings prove echo cancellation. A genuine capture/playback reference trial and eventual Pi 5 trial remain necessary for acoustic latency, music leakage and hardware effects.

### Coverage limits for recorded scenarios

`--session` can carry same-scenario clips through a canonical `Session` in order, with the real ASR MCP, mock music/device adapters, transcript/task receipts, and one session for correction/follow-up sequences. It is a sequential replay: each clip finishes processing before the next is submitted. The recorder does not capture assistant playback or a synchronized playback reference, so a `Stop speaking` phrase does not simulate actual barge-in timing or echo. Existing deterministic scenario tests exercise interruption/cancellation races separately. `--session` uses a generic `MockLLM` by default; it does not infer arbitrary natural-language tools. Exact direct routes can execute against mock adapters. Broader model tool selection requires an explicitly bounded live-reasoning run, and a real Onit run is not part of this evaluator. A recorded project request is therefore ASR evidence only unless separately tested through the harness.

## Course benchmark adapter status (2026-11-06)

Reference: `https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking`, pinned under ignored `var/cache/fd-benchmark-reference` at revision `293ba44e1a65645ed8f2bb6c8832e6564475e8e3`. The benchmark's real-time direct rig transports 16 kHz mono PCM over a local WebSocket; it does not open microphone/speaker devices. Maintained entry: `scripts/fdbench_entry.py`. Do not modify its scoring rules to compensate for ME3 behavior.

| Requirement | Current ME3 integration | Qualification |
|---|---|---|
|Build/run/cancel entry|Course `launch_assistant.py` loads ME3's `build(transport, log_dir)` and `audio_rates()`; runtime remains canonical ME3 AudioApplication/Session/SpeechClient/Onit executor.|`--check` reports READY. A first-run deadlock was traced to an invalid Transcript playback enum and fixed by adding the explicit `benchmark` delivery context. One disposable live clip connected after that fix. Pending-Onit cancellation/cleanup is not fully qualified because the smoke parent ended its owned process before the worker's 45 s deadline.|
|Input/output formats|16 kHz mono PCM16 input validated at adapter boundary; output rate reports 16 kHz and uses bounded Pipecat frames through `BenchmarkSink`.|One direct-feed clip reached ASR final and Onit dispatch. It yielded no response audio. No resampling/sample-rate discrepancy observed in the transport check.|
|Pipecat|HPC project and benchmark launcher both use Pipecat 1.13.0.|Compatible on Python 3.13.9 x86_64. Pi/aarch64 package compatibility remains untested.|
|Output routing and clear|ME3 sink queues `OutputAudioRawFrame`; stop pushes Pipecat `InterruptionFrame`, mapped to the benchmark's clear event.|Transport-level echo smoke passed, with aligned normalized correlation 1.0 at 64 ms offset. The real inference clip emitted no assistant PCM, so actual assistant output/interruption-clear propagation was not exercised. This is not acoustic AEC or playback evidence.|
|Playback completion|Adapter waits for transport frame-send callbacks and estimates queue-drain using PCM duration.|A software proxy only. It is not the browser's `onended`, a device ACK or acoustic silence. No output in the real trial means no completion semantics were measured.|
|Wake/session activation|Dataset clips generally omit “Hey Swift”; adapter performs an explicit benchmark-only activation at first direct-feed frame.|This leaves ordinary wake/session behavior intact. The benchmark runner keeps the socket/assistant across trials. ME3's 120 s extended session and the benchmark's default 6 s inter-trial silence do not reset each other. Do not run false-turn/side-talk score groups on this connection until a safe per-trial reset handshake/restart policy is added. Never activate every voiced clip: that would invalidate non-addressed-speech cases.|
|Instrumentation|Course launcher can report procfs CPU, memory and process stats; adapter writes payload-free startup/session event names.|`--check` detects resources, but the one audio smoke used `--no-stats`; no resource series. The single smoke is not a benchmark score.|

The first live-audio smoke initially produced no reply because the adapter had configured a file-style `wake_frame` but did not activate Session at first WebSocket audio. The benchmark-specific first-frame activation fixed that. A follow-up single clip (Candor `v1_turntaking_56`, 12.032 s total recording) showed activation, VAD onset, ASR finalization and `agent_route_dispatched`. The Onit task worker sent one university request body but no response headers arrived before the benchmark's 8-second answer window closed; the benchmark recorded no assistant speech (`TOR=0`, provisional “missed”). The diagnostic parent then reaped its own process with SIGTERM after a 12-second shutdown wait, before the provider's 45-second deadline. This establishes one live benchmark-window miss and a remaining teardown-harness issue, not a provider timeout result. Raw logs are under `var/experiments/020-endpoint-onit-benchmark/fdbench-real-audio-smoke-activated/`.

Endpoint and Onit details, live request accounting, the generated-project failure and owner action are in the 2026-11-06 section of [PLAN.md](PLAN.md). No complete class benchmark run, score, AEC result, user-speaker measurement or Pi result has been obtained.

## Class benchmark update — 2026-11-06

Reviewed upstream `a0bab59114aa31341e8a750c623ef60fc7ca8399` (previously293ba44). New stage marks, stage-latency block, WER support, model metadata and deck/report generation are appropriate for the supplied two-page template. The upstream direct-feed rig still carries no per-trial activation/reset handshake. `--reset-cmd` executes a shell command between trials but does not itself guarantee reconnection/readiness or active-versus-inactive ME3 policy. Six seconds of silence cannot reset the adapter's120-second engagement. Full mixed-trial scoring remains blocked pending an explicit tested reset policy; use separately started processes per controlled case, label active/inactive, and do not report the default mixed run as valid.

ME3's custom Session does not emit standard Pipecat transcription/LLM/TTS frames, so the upstream observer cannot infer all stages. `scripts/fdbench_entry.py` now emits documented marks for canonical endpoint, ASR completion, reasoning request, tool start/end and TTS request/first chunk, plus transport-send audio marks. Missing first-token marks remain unavailable: first speakable segment must not masquerade as TTFT. ASR text is not exported by these marks yet, so automatic WER remains unavailable. Audio-out marks reflect transport handoff, not acoustic onset. Same-machine monotonic values and session/turn IDs preserve correlation; cross-machine raw monotonic clocks must not be subtracted.

Adapter defaults now reuse the selected main secret file for LLM unless ME3_LLM_SECRETS_FILE is explicitly set; configured model labels replace hardcoded university/HPC claims. The unrelated four-request worker cap is removed; ME3_FDBENCH_REQUEST_LIMIT optionally supplies an explicit guard and backend/account limits still apply. No live allowance is granted by this change. `describe()` reports configured roles only, never credentials. Put reference checkout/dependencies and all generated datasets/runs under var/; use FDBENCH_PROJECT/FDBENCH_ENTRY explicitly rather than creating a new root source tree. Do not run a paid judge or new live comparisons without an applicable authorized budget.

## Latest pinned compatibility and Pi smoke gate — 2026-11-06

Rechecked upstream main at **9d7a824e0298b27a2c0f9f1f435be07125ae0391**, superseding the a0bab591 review. New changes exclude discarded speculative LLM requests from context summaries and distinguish benchmark-link routing from cloud/default routing. ME3 does not currently speculate on partial ASR, so it must not emit speculative requests/cancellations to manufacture those rows. On a VPN-connected Pi, the benchmark link and university/cloud route must be recorded separately; neither route label proves API authentication.

The current adapter provides explicit `ME3_FDBENCH_MOCK=1` software checks: canonical SpeechClient/MCP subprocesses, Session, VAD, task mocks and the upstream WebSocket transport, without keys, Onit installation, microphone or speaker. Real providers are selected only by explicit backend configuration when mock mode is absent. Mock metadata no longer claims Onit reasoning. Startup/teardown uses an AsyncExitStack so partial build failures close registered Session/worker/MCP resources. Queue/send/clear checks remain software evidence.

`ME3_FDBENCH_EXPORT_TEXT=1` opts into sending the matching authoritative final transcript in stage marks for class WER scoring. It uses only the current turn's final user projection, never another turn's history or a draft. The upstream mark contract truncates strings at500 characters; ME3 omits longer transcript text rather than submitting a truncated hypothesis as full WER input. Missing transcript marks mean unavailable WER. Provider-final text can be an incorrect recognition; measure it, do not replace it with the expected text. First useful text/Onit retrospective TTFT durations still do not supply a true realtime first-token mark. Exact model context token counts also remain unavailable unless instrumented at actual model request construction; do not infer them from UI transcript length.

### Reproducible setup and no-inference checks

Keep the benchmark clone, its separate scoring environment, datasets and runs under `var/`. Select the exact reviewed benchmark revision; install its own requirements in its own environment only if needed. Do not merge benchmark dependencies into ME3's lock. The ME3 environment supplies Pipecat1.13 and the official transport. From the ME3 root:

```bash
# Explicit entry and root avoid any dependency on the benchmark folder name/location.
export FDBENCH_ENTRY="$PWD/scripts/fdbench_entry.py"
export FDBENCH_PROJECT="$PWD"
export ME3_FDBENCH_MOCK=1
var/cache/venv/bin/python -B var/cache/fd-benchmark-reference/launch_assistant.py --check

# Supply an existing explicitly synthetic PCM16/16k/mono fixture, not private user audio.
var/cache/venv/bin/python -B scripts/check_fdbench_transport.py \
  --synthetic-wav var/recordings/synthetic-speech.wav \
  --output-dir var/results/fdbench-pi-mock-smoke
```

The maintained smoke runner requires a fresh output path and uses fresh Sessions for one active and one inactive case. It sends the existing synthetic clip over a local WebSocket with real pacing; mock ASR text does not come from the clip. It verifies raw reply bytes, actual transport `clear`, inactive silence and cleanup queue state. It does not simulate an acoustic interruption. Output and marks are collected with one host's monotonic clock. The exact upstream launcher `--check` passed on HPC; this is not a Pi or model-health result.

For the later Pi-to-laptop test use the pinned official launcher with `--url ws://LAPTOP_ADDRESS:8791` and a log directory under Pi `var/results/`. This socket is the private benchmark link; it does not route university inference. Check the local network/firewall and VPN behavior separately. No assistant_entry.py copy is required: set FDBENCH_ENTRY/FDBENCH_PROJECT as above. Transcript exports/recordings must be opt-in and credentials remain server-side.

**Do not proceed directly to full mixed-trial scoring.** The upstream runner still has no trial-start group/activation handshake. Its `--reset-cmd` runs after a variant before the queue status is saved, supplies no next-trial metadata and ignores a nonzero subprocess exit; it is not a sufficient readiness/reset protocol. A default6s gap does not reset ME3's120s benchmark engagement. Start separate fresh adapter processes for selected active and inactive cases and label their initial states; these are partial checks, not a full class score. Do not trigger activation from every voiced clip, reset midway through an interruption scenario, or change scoring/audio to improve results. Full-run readiness needs a tested controller that supplies trial identity, activation policy, reset completion and fresh-state acknowledgement, or an instructor-approved standalone-case protocol.

### Evidence and remaining limits

Offline real-transport evidence: `var/experiments/020-endpoint-onit-benchmark/latest-transport-smoke/result.json` — active290 input frames/3840 mock output bytes, wire clear observed; separate inactive290 frames/zero output bytes; both cleanup_complete with empty playback queues, zero live requests. Stage marks included endpoint, ASR final, request, TTS first chunk and audio handoff. Inactive cleanup may send a clear command; this is not a conversational interruption. `var/results/benchmark-latest-entry-check.txt` records the official launcher READY check. Relevant regressions cover current-final-only transcript export and failure after Session creation, with no residual asyncio tasks. Do not turn these checks into benchmark metric estimates or acoustic/Pi claims.

Before any separately authorized real-provider smoke, unset ME3_FDBENCH_MOCK and set ME3_BACKEND_CONFIG/ME3_SECRETS_FILE (and ME3_LLM_SECRETS_FILE only if separate) to the actual Pi-local files. Re-run --check and require the intended configured model labels; never count mock speech as live inference. ME3_FDBENCH_EXPORT_TEXT=1 is explicit benchmark transcript export, and should be removed after the trial. No acoustic accuracy or side-talk detection follows from a wake-disabled inactive Session emitting silence.
