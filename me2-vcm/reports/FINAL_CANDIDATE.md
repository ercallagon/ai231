# Final Submission Candidate Audit

Status: frozen models; final Pi 5 rerun pending. Read alongside
[RESULTS.md](RESULTS.md), [MODEL_SELECTION.md](MODEL_SELECTION.md), and
[the final handoff](../docs/FINAL_PI_VALIDATION.md). The containing Git commit,
not a historical run ID, identifies this candidate. No retraining, gate fitting,
model replacement, split change or automatic deployment is part of this update.

## Frozen Selection

Champion: six-second BC8-TCN-ASP, 488,264 learned parameters, gate 0.954.
Baseline: two-second BC8, 327,750 parameters, historical run_043, gate 0.875.
Wake: exp15 CNN-GRU, 204,738 parameters, gate 0.30. Calibration temperature is
already baked into each VCM ONNX; do not divide logits again. Configs, labels,
manifests and models are hashed in ARTIFACTS.json and data/model_reference.json.
No pretrained detector weights are used; optional Piper synthesis is separate.

BC8 is the repository's BC-ResNet-8-style width configuration, not eight layers:
12 broadcast blocks (stage repeats 2/2/4/4; widths 64/96/128/160), stem 128,
tail 256. Champion adds temporal stride-two pooling, a 128-channel projection,
three noncausal residual depthwise temporal blocks (kernel 5, dilations 1/3/9),
and masked attentive mean/std pooling. It is not a causal encoder, KV-cache
system or an unmodified paper model. The upstream ONNX profiler's 499,784 count
is not the training code's learned-parameter count; disclose counting scope.

## Received Pi Evidence

Uploaded injection: work/pi5_review/REPORT.md, test_RESULTS.json and
holdout_RESULTS.json in the private research workspace. Raw cost/environment
files were not included in that upload. Reported model-only Champion p95
89.18 ms / nominal RTF 0.01486 remains user-returned evidence, not a freshly
rerun host measurement. Do not confuse the earlier 48.31 ms two-second Baseline
measurement with Champion cost.

| Champion metric | Test | Holdout |
| --- | --- | --- |
| Raw intent, full manifest | 4466/4693 (95.16%) | 191/202 (94.55%) |
| Raw real intent | 649/824 (78.76%) | 87/96 (90.63%) |
| Raw synthetic intent | 3817/3869 (98.66%) | 104/106 (98.11%) |
| Accepted-correct supported intent + slot | 4046/4367 (92.65%) | 178/186 (95.70%) |
| Accepted-correct real supported | 508/777 (65.38%) | 78/86 (90.70%) |
| Accepted-correct synthetic supported | 3538/3590 (98.55%) | 100/100 (100%) |
| OOS false accepts | 14/326 | 1/16 |
| Processing failures | 16, all real | 0 |

Keep failures in full-manifest denominators and also report processed-subset
metrics. These sets have been inspected repeatedly; they are **previously
exposed evaluations**, not independent unseen confirmation. Real/synthetic
grouping follows frozen source metadata, not predictions. Neither high synthetic
accuracy nor 86 supported real holdout examples establishes broad generalization.

Initial class acoustic run: laptop runs/20261016-201135; 202 wake-present and
16 no-wake trials, seed 42, screened wake takes 2/3, ECHO Spot playback, two-second
gap, 1200 ms endpoint, preview actions. Preserved source file SHA-256:

| File | SHA-256 |
| --- | --- |
| report.md | 8e08ba3374b2699549b0a62d3434b76f119fef3661b30269b6620b1c46e7e95c |
| metrics.json | ebff6edd77b5826c6245910a9138293374ca51d0ecfe2c237c97b9e5837de4dc |
| trials.csv | 889f0b650ebbc9d5498ffef40a571c4b14207f8f87df822608f219ec13ad3a8d |
| pi_metrics.csv | e0019a21e767fd4cc2294e53e232dd24157ad6d033d0b0c7eeeddd150aa53b87 |

Raw user logs/recordings remain local, not publication assets. The uploaded class
outputs lack a contemporaneous model hash, Git revision and full calibration
record. Identity is consistent with the selected model path/size, not proven by
an execution-time hash. The new preflight closes that provenance gap prospectively.

- Official command decisions: 159/202 (78.71%); supported accepted-correct
  144/186 (77.42%): real 68/86, synthetic 76/100.
- Supported failures: 14 no-response, 9 zero-audio no-sustained-speech,
  19 inferred/rejected or other OOS outputs. Saved predictions cannot distinguish
  low-confidence rejection from OOS argmax without fuller diagnostics.
- Conditional slot accuracy 78/78 applies only after correct intent; it is not
  unconditional success across all slotted commands.
- Family success: COLOR 10/18, BRIGHTNESS 12/18, TIMER 13/18,
  ALARM 14/18, CREATE_REMINDER 14/18, TEMPERATURE 15/18.
- OOS false accepts 1/16. Three OOS no-responses are scored as rejection, not
  confirmed model rejection. No-wake fires 5/16 comprise four rejected decisions
  and one accepted MESSAGE; no service was executed in preview.
- Processed inference (input preparation + features + ONNX), n=176:
  mean 108.767 ms, p95 116.15 ms; actual-audio RTF p95 0.16412.
- Detector-estimated speech-end to prediction, n=176: p95 1326.24 ms.
  Endpoint wait is included; action/TTS and laptop log-delivery time are not.
- Runtime CPU mean 59.1%, RSS maximum 201.9 MB, temperature maximum 54 C,
  reported throttling flags zero. These are run-specific, not universal limits.

## Findings And Disposition

| Finding / metric | Addressed | Still required / tradeoff |
| --- | --- | --- |
| Endpoint dominates decision delay | Optional explicit 800 ms setting across public launchers; default preserved | Final Pi p95 target <1 s; may split pauses >800 ms |
| Model inference vs action latency confused | Distinct model-only, feature+ONNX, detector-end and external scopes | Do not call 116 ms speech-end latency |
| Missing vs zero timing conflated | Supplementary audit separates them | Initial nine zero-audio decisions were not missing fields |
| Early/multiple decisions | Full diagnostics and unique logs; audit preserves original scoring | 17 multi-decision trials, 20 extras, 24 negative external latencies need controlled repeat |
| Wake detection claim | Explicit wake events distinguished from proxy command receipt | Reported 92.1% is a proxy; explicit 182/202; screened takes bias wake recall |
| Capture/no-speech failures | Added peak/noise/onset diagnostics, not weakened onset gate | Pi reproduction with fixed gain/placement; noise alone not established cause |
| Speech/music isolation | Operator confirmed stable independent TTS across restarts and tablet volume changes | Not acoustic echo cancellation; loud music can still mask wake |
| GPIO/action path | Operator reported successful visible actions for 22 phrases | Not proof of actual calls/SMS or call-end recovery |
| Runtime rollback | Both models and 1200 ms endpoint retained | Do not overwrite working deployment automatically |
| Coverage and rejection | Full manifests, failures, source/family and accepted/raw scopes retained | Fresh unseen speakers absent; small OOS/no-wake samples uncertain |

Host tests show 800 ms retains a synthetic 0.6 s internal pause but ends before
speech after a 1.08 s pause; 1200 ms preserves that longer pause. Complete-word
retention is not manually established. Six seconds is capacity, not fixed wait.
The laptop wake-to-command gap is not included in speech-end decision timing.
No inference conclusion establishes real service latency or success.

The class scorer selects the first decision in its trial interval. Negative
external timing may reflect premature triggers or clock/playback/log attribution;
do not drop early events, rewrite scores or blame room noise without evidence.
Supplementary review_class_benchmark.py never changes the official score.

## Repository And Submission Audit

The incoming curated tree had 444 ME2 tracked files, 48,059,078 bytes, with no
tracked raw Parquet, user songs/voices, private profiles, caches or archives.
Original research artifacts are not moved/deleted. Historical contender evidence
is retained intentionally, not promoted. The shared Git repository's ME1 and
history are preserved. A restricted experiment allowlist reduces accidental
future staging of research runs; no broad staging command is needed.

Use scripts/prepare_submission_inventory.py --tracked to regenerate file sizes,
hashes and pattern-based sensitive-content checks. A pattern scan is not proof
of universal secret absence. Tests, inventory and a clean exported-tree contract
check must pass before publication; execution receipts stay under ignored work/.
No raw private diagnostics, credentials or recordings are submission evidence.

Host execution receipt: [FINAL_CHECKS.json](FINAL_CHECKS.json). The CPU suite
passed 258 tests with four explicit skips: two obsolete A0-export checks and
two unavailable raw-wake-data integration checks. Active frozen ONNX contracts
passed independently; 146 protected artifact hashes were unchanged. The initial
full invocation in an inference-only environment failed on absent torch/sklearn;
those dependency failures were rerun in the existing training environment with
GPUs disabled, not suppressed. Temporary CPU unit-test smoke fits are not new
experiments. Pi and complete-word listening verification remain separate.

Two-slide fields: source/DOI/licences, utterance and duration accounting,
architecture/parameters, seeds/splits/logs/checkpoints, comparable-size baseline,
test recognition/source/family/rejection metrics and reproducible commands are
in the repository. Pi cost/resource scope must identify its actual source. Final
controlled Pi endpoint/latency and acoustic A/B evidence are pending. No fresh-OS,
unseen-speaker, AEC or successful telephony recovery claim is justified.

## Freeze And Decision

Retain Champion as the published recognition default with Baseline selectable;
this is not a new model promotion. Final interaction settings are experimental
until measured. Require artifact integrity, complete denominators and no
unexplained scoring corruption or material retention regression. Sub-second
decision p95 is a target; no numerical accuracy/false-trigger acceptance cutoff
was pre-agreed, so do not invent one after seeing results. Return final evidence
before marking submission complete or populating final deck numbers. Afterwards
limit changes to evidence/reports/deck unless a material issue is explained with
its retesting requirements first.
