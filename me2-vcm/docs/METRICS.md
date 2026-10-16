# Metric collection map

`swift.py evaluate --reset` collects automated evidence and generates
work/pi5_test_01/REPORT.md plus REPORT.json using the existing evaluation scripts.
`swift.py report` refreshes without repeating evaluation. Missing results stay
marked; historical evidence is not relabeled as Pi measurements.

| Metric/evidence | Method and output | Scope/limits |
| --- | --- | --- |
| Architecture, parameters, model bytes/hashes/gates | data/model_reference.json; actual ONNX hashes/sizes; doctor contracts | Champion (6s BC8-TCN-ASP) / Baseline (2s BC8); legacy profile IDs remain provenance; exp15 separate |
| Training seed, optimizer, steps, epochs, selected epoch, GPU hours | model_reference.json references committed TRAINING_ACCOUNTING.json/logs | Archived accounting; allocated wall time includes validation/host overhead |
| Dataset source, DOI, licence, recordings/hours/cohorts/groups | data/PROVENANCE.json and DATASET_LICENSE.md; report table | Original-waveform hours; declared groups are not verified humans |
| Full-manifest intent accuracy | injection/*_RESULTS.json: overall_intent | Processing failures remain incorrect in denominator |
| Real/synthetic accuracy and accepted joint success | real_intent, synthetic_intent, real_supported_accepted, synthetic_supported_accepted | Cohort uses frozen is_synthetic metadata, not filenames |
| Macro-F1, class precision/recall/F1 | macro_f1, families | Raw intent metrics differ from gated success |
| Confusion/errors per family | confusion, families; *_errors.jsonl | Separate processing-failure column |
| Intent+slot and accepted-correct command success | supported_joint, supported_accepted_joint | Supported-command denominator, not slot-only accuracy |
| Slot-head accuracy given true family | slot_accuracy_given_true_family | Independent of predicted intent; failed processing stays incorrect |
| Semantic success including rejection | semantic | Accepted-correct supported plus invalid correctly rejected; known-label denominator |
| OOS false accepts/FAR | oos_false_accept; processed_subsets.*.processed_oos_* | Full and processed denominators separate |
| Valid-command false rejects, wrong supported accepts | false_rejects, wrong_supported_accepts; families | Supported preprocessing failures remain false rejects |
| Unsupported value/temperature rejection | unsupported_false_accept, unsupported_temperature_false_accept; temperature_joint/accepted | n=0 means unavailable evidence, not 100% success |
| Coverage/reasons/long-input failures | processing_status, failures, unscored, duration_cohorts, *_errors.jsonl | Continuous-waveform retention is not manually verified complete-word retention |
| Paired errors | paired_supported | Same frozen rows; not independent-seed evidence |
| Feature preprocessing + inference p95 / RTF p95 | infer_p95_ms, per_clip_rtf_p95; *_timings.json | Features to ONNX return; no decode/resampling/capture/endpoint/actions/TTS; first inference included |
| Feature/model times | feature_ms, model_ms, infer_ms, audio_ms | Do not add separate percentiles or divide p95 by six seconds |
| Model-only p50/p95 and nominal RTF | champion_cost.json, baseline_cost.json, model_costs.json | Synthetic features, 30 warm-up + 200 measured, one intra/inter-op thread |
| CPU/RSS/temperature/clock | resources.json, automatic worker monitor | 0.5s sampling of evaluation worker including fetch/decode; not live assistant or exact peaks; absent sensors null |
| Revision/dependencies/command/device | source_revision.txt, ENVIRONMENT.json, DIAGNOSTICS.json, model hashes | Actual OS-provided lgpio import still required |
| Historical preprocessing/export parity | CHECKS.json and selected evidence; doctor contracts | New contract probes do not rerun all historical numerical parity |
| Speech-end to decision p50/p95 | swift.py live-latency; session latency.json | Detector final voiced-frame read time includes endpoint wait; not annotated acoustic truth |
| No-speech/accepted/rejected timing counts | live latency summary | Missing endpoint is not zero latency; acceptance is not correctness |
| Wake clip accuracy/P/R/F1/confusion | exp15 corrected historical reference in model_reference.json | Exposed, synthetic positives; not Pi streaming |
| Wake misses/false triggers per hour/raw-AEC/music robustness | Annotated acoustic trials plus negative exposure time | PENDING without evidence; cannot infer from clip FPR/unannotated wake counts |
| Official class acoustic results/resources | BENCHMARK.md; report --official RESULT_DIRECTORY imports metrics.json | Laptop playback into Pi mic; injection is supplementary |
| GPIO/display/timer, APIs, Spotify/fallback, TTS, communications | doctor plus guided demo observations | Availability probe is not physical success; each CALL/SMS needs terminal approval |
| Action initiation/completion, TTS onset, power/energy, live memory | Separate instrumented physical tests | PENDING; never equated with inference/decision latency |
| Fresh unseen speakers/fresh OS | New consented recordings/separate OS installation | Not supplied by shared-OS dry run; prior exposure disclosed |

Count/rate tables include descriptive Wilson 95% intervals. Correlated synthetic
voices, limited real speakers and repeated evaluation constrain interpretation;
intervals do not remove selection bias. No gate/weight changes occur during
collection. Keep full denominators/failures and raw files. Test is primary;
holdout is separate, not chosen based on the better result.

For live comparisons fix placement, gain, noise, thermal conditions and protocol.
Baseline/champion differ in capacity/frontend/training lineage: this is a system
comparison rather than an isolated architecture ablation. Freeze scored settings.
