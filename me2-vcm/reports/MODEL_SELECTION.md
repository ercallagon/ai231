# Current recommendation: six-second model at 0.954

The user authorized this frozen candidate as the repository default. Champion Pi 5 acoustic results are separately summarized in PI5_ACOUSTIC_CHAMPION_20261017.md; run_043 remains the selectable fallback. Runtime integration and mock e2e checks passed; no working Pi installation was overwritten. The following records preserve the earlier validation decision and subsequent authorized refinement. Final 0.954 metrics are in RESULTS.md, not the historical 0.95 table below.

Current integration supports the selected two-input mel/mask graph, deterministic
full-context features and safe capacity/rejection handling. Host parity and mock
integration checks are complete; the Pi acoustic result validates Champion under
its reported protocol only. Call recovery, independent speakers and noisy/music
robustness remain unverified. Statements below that call mask integration pending
describe the earlier selection stage, not the current repository.
Exp15 remains the separately selected wake default after the final controlled
studies; see WAKE_EXP15_STREAMFT.md. Submission fields and remaining limitations
are collected in SUBMISSION_CHECKLIST.md.

## Historical model-selection and calibration record

## Authorized calibration revision, 2026-10-15

The original pre-test selection below remains preserved. After examining the
reported results, the user authorized a new development calibration using only
validation and the existing tune-OOS pool. This revision does not change the
checkpoint, features or temperature; it refines the global threshold beyond the
original search's 0.95 upper limit. Prior test/holdout exposure is disclosed, and
no new reserved predictions were accessed during this refinement.

The revised threshold is **0.954**, with temperature **0.6499874044563441** already
baked into the unchanged ONNX once. On the same 284 processed negative clips,
run_043 accepts 15 (5.28%), the original candidate policy accepts 16 (5.63%),
and the revised candidate policy accepts 13 (4.58%). On all 400 calibration
negatives the revised candidate accepts 13 (3.25%). There remain 116 capacity/
preprocessing rejections, separately accounted for.

Validation accepted-correct supported commands are 2,572/2,819 versus run_043's
2,519/2,819; real supported commands are 90/124 versus 76/124. Refinement costs
four correct accepts, including one real, relative to the original candidate
policy. Wrong supported accepts remain 14, unsupported-temperature accepts are
two, and no family with n>=20 loses more than five percentage points relative
to the baseline. The fixed global gate therefore passes the versioned offline
nomination criteria. No margin/intent-specific policy or weight retraining was
needed.

The six-second model is now the stronger validation-supported nominee for
submission and Pi integration. Formal deployment promotion remains pending its
two-input runtime adaptation and Pi verification. The test/holdout accepted-
command scores in the table below refer to the **original 0.95 gate** and must
not be attributed to the revised gate. Raw argmax intent scores are unaffected
by changing the confidence threshold. The 5% calibration limit is a point-estimate
development target, not a statistical guarantee of independent/live performance.

Evidence is preserved under work/vcm_six_second_calibration_20261015_v1: frozen
inputs/rules, sweep, original and new metrics, policy/config/labels, shared-subset
comparison, CPU parity checks and limitations. The earlier selection and reports,
champion weights and production runtime remain intact.

## Original pre-test selection and frozen evaluation

Verified 2026-10-15 from completed logs and saved evaluations. Training stopped
at epoch 72 under patience 20, selecting epoch 52. The candidate achieved
96.4750% overall validation intent and 82.9114% real-voice validation intent;
run_043 achieved 96.0301% and 74.0506% on the same 2,922 rows.

The validation-only selection worker froze run_043 before test/holdout inference.
The candidate passed weighted recognition, family-regression, export parity,
size and unsupported-temperature criteria, but failed the predeclared
classifier-conditional OOS criterion: 16/284 processed calibration negatives
were falsely accepted (5.6338%, limit 5%). Its operational rate was 16/400 (4%),
because 116 recordings were rejected by preprocessing/capacity. Those rejections
must not be presented as learned OOS performance.

Calibrated validation supported accepted-correct commands were 2,576/2,819 for
the candidate versus 2,519/2,819 for run_043. Real supported accepted-correct
were 91/124 versus 76/124. Wrong supported accepts decreased from 42 to 14;
validation OOS accepts were 2/76 for both. The candidate had 14 processing
failures, retained in the denominators.

Repeated frozen evaluations (historical exposure disclosed):

| Metric | run_043 | Six-second candidate |
| --- | ---: | ---: |
| Test intent, n=4,693 | 92.67% | 95.16% |
| Test real intent, n=824 | 62.74% | 78.76% |
| Test synthetic intent, n=3,869 | 99.04% | 98.66% |
| Test accepted-correct supported, n=4,367 | 3,922 (89.81%) | 4,049 (92.72%) |
| Test OOS false accepts, n=326 | 10 (3.07%) | 15 (4.60%) |
| Test wrong supported accepts | 41 | 45 |
| Test processing failures | 0 | 16 |
| Holdout intent, n=202 | 63.86% | 94.55% |
| Holdout real intent, n=96 | 25.00% | 90.63% |
| Holdout accepted-correct supported, n=186 | 110 (59.14%) | 179 (96.24%) |
| Holdout OOS false accepts, n=16 | 0 | 1 |

These results support the candidate's recognition direction, but do not override
the pre-test selection or justify threshold tuning on reserved results. They
also do not establish live acoustic performance. The six-second candidate remains
valuable comparison evidence, with mask-input deployment adaptation, actual Pi
cost and causal capture checks pending. Preserve its checkpoint and full logs.

The preserved deployment reference is run_043, a 327,750-parameter BC8 model.
The completed candidate is a scratch 488,264-parameter BC8 with temporal
convolutions and masked attentive statistics pooling, accepting up to six seconds.
Its capacity is not mandatory microphone listening time. It requires a two-input
inference adapter and is not yet a drop-in replacement for the deployed model.

The current experiment selects checkpoints by 0.70 times full validation intent
accuracy plus 0.30 times real-voice validation intent accuracy. Source-specific
sample counts, processing failures and calibrated decisions must accompany the
final comparison. Its architecture, input capacity, training weights and lineage
differ from run_043; this is a system comparison, not a controlled layer ablation.

Required comparison: raw intent, macro-F1, supported joint intent/slot success,
real/synthetic cohorts, per-family and paired errors, unsupported temperature
values, OOS false acceptance, valid-command rejection, and all preprocessing
failures. Report conditional OOS false acceptance among processed examples as
well as operational rates: capacity rejection must not be mistaken for learned
OOS discrimination. Complete-word retention remains manually unverified.

Checkpoint and rejection settings are frozen using validation before test and
holdout inference. The user explicitly authorized repeated evaluation of the
existing test/holdout sets on 2026-10-14, and requested test results in the final
assessment. These sets have prior project exposure and are not newly untouched.
The later user instruction required validation-only selection first; the worker
enforced that ordering. Thresholds and selection were not changed from these
evaluations. No independent newly unseen evaluation is currently available.

Promotion also requires ONNX output/decision parity, actual input/failure behavior,
CPU cost, deployment compatibility and Pi 5 measurements. Host latency is not
Pi latency. Wakeword selection remains separate. Neither a new high aggregate
score nor this draft constitutes promotion or submission readiness.
