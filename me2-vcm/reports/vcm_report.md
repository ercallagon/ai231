# Sealed VCM architecture and gate comparison

All policies were selected on development data and sealed BEFORE frozen-test processing.
Both candidates evaluated at the user's explicit request despite earlier screening failure.
No training, deployment, automatic promotion, model edits, final-test refitting or live holdout evaluation.

## Protocol and controls

Same 4,693-row frozen test for all models. Encoded audio and decoded16kHz mono PCM hashes verified for every row; no augmentation or membership changes.
2s/3s features use the existing center-crop/edge-pad implementation. Clip duration cohorts are file duration, not measured active speech or live endpointing.
The test was examined historically for previous runs. This is a fixed held-out comparison, not a claim of a newly pristine test.
The 202-row live Pi holdout was not used. Validation/challenge have repeated development exposure and policy search optimism.

Current baseline is preserved run_043 BC8, own run_042 fine-tuning lineage,327750 parameters. Both3s candidates are random-init seed42 on25366 identical training rows, matching source-weighted loss/SpecAug/optimization; both add TEMP REJECT supervision.
BC8 control:328007 parameters,70epochs,selected42. Conformer:662445 parameters,53epochs,selected32; two96-wide4-head Conformer blocks add gated slot residuals. Intent remains the shared BC8 pooled head.
Only the two3s arms isolate the added Conformer. Comparing with run_043 also changes window, initialization/training lineage and rejection supervision.

## Raw recognition

Raw metrics ignore acceptance gating. Legacy E2E treats a missing slot target as intent-only; known-slot E2E excludes unscored missing-slot rows.
Slot-only accuracy uses the ground-truth intent's head even when the intent was predicted incorrectly; it is NOT end-to-end accuracy.

| Model | Intent | Macro-F1 | Legacy intent+slot | Known-slot E2E | Slot-only |
|---|---:|---:|---:|---:|---:|
| run_043 (2s) | 92.67% | 91.28% | 92.24% | 92.24% | 97.68% |
| BC8 (3s) | 93.27% | 92.02% | 92.90% | 92.90% | 98.86% |
| BC8 + Conformer (3s) | 92.12% | 90.72% | 91.56% | 91.56% | 97.64% |

## Frozen gate outcomes

original = preserved deployment gate for run_043; previous screen gates for candidates. matched_joint = the same intent*slot score design for all models, no slot temperatures. optimized = highest preregistered validation balanced objective, NOT necessarily highest total recognition accuracy.
All thresholds/corrections were selected without test predictions. The optimized objective weights real supported success(.5), total semantic success(.25), audited invalid-slot rejection(.25). Equal tuning budget does not guarantee equal live risk.

| Model | Policy | Gated intent | Semantic success | Supported correct | Real supported correct | Wrong supported accepts | OOS accepts |
|---|---|---:|---:|---:|---:|---:|---:|
| run_043 (2s) | original | 90.60% | 90.30% | 3922/4367 | 369/777 | 41 | 10 |
| run_043 (2s) | matched_joint | 90.56% | 90.50% | 3932/4367 | 376/777 | 41 | 11 |
| run_043 (2s) | optimized | 89.88% | 89.86% | 3903/4367 | 373/777 | 42 | 12 |
| BC8 (3s) | original | 89.84% | 89.67% | 3896/4367 | 356/777 | 27 | 14 |
| BC8 (3s) | matched_joint | 90.90% | 90.86% | 3951/4367 | 400/777 | 32 | 13 |
| BC8 (3s) | optimized | 90.75% | 90.75% | 3948/4367 | 408/777 | 39 | 15 |
| BC8 + Conformer (3s) | original | 89.52% | 89.18% | 3874/4367 | 340/777 | 46 | 15 |
| BC8 + Conformer (3s) | matched_joint | 89.64% | 89.52% | 3888/4367 | 355/777 | 46 | 13 |
| BC8 + Conformer (3s) | optimized | 89.49% | 89.47% | 3889/4367 | 366/777 | 69 | 16 |

Semantic success means correct accepted supported intent+slot OR rejection of labeled OOS. Missing slot truth is unscored, not assumed correct. Wrong-supported-accepts are potentially incorrect actions; OOS rejects are evaluated as effective action rejection, including hard slot REJECT.

## OOS rejection

| Model | Policy | Precision | Recall | False accept rate | TP / FP / FN |
|---|---|---:|---:|---:|---|
| run_043 (2s) | original | 43.89% | 96.93% | 3.07% | 316 / 404 / 10 |
| run_043 (2s) | matched_joint | 44.43% | 96.63% | 3.37% | 315 / 394 / 11 |
| run_043 (2s) | optimized | 42.66% | 96.32% | 3.68% | 314 / 422 / 12 |
| BC8 (3s) | original | 41.27% | 95.71% | 4.29% | 312 / 444 / 14 |
| BC8 (3s) | matched_joint | 44.91% | 96.01% | 3.99% | 313 / 384 / 13 |
| BC8 (3s) | optimized | 45.01% | 95.40% | 4.60% | 311 / 380 / 15 |
| BC8 + Conformer (3s) | original | 41.03% | 95.40% | 4.60% | 311 / 447 / 15 |
| BC8 + Conformer (3s) | matched_joint | 41.96% | 96.01% | 3.99% | 313 / 433 / 13 |
| BC8 + Conformer (3s) | optimized | 43.12% | 95.09% | 4.91% | 310 / 409 / 16 |

OOS is the positive class here: precision is among rejected commands, recall is among true OOS. A low OOS precision can reflect rejecting valid commands. Frozen-test OOS rates are observed outcomes, not a newly tuned5% bound.

## Per-intent performance

Raw recall and optimized accepted-correct supported counts are below. All raw/gated precision, recall, F1 and20x20confusions are in RESULTS.json and exported CSV files.

| Intent | Raw recall043 / BC8 / Conformer | Optimized accepted-correct043 / BC8 / Conformer | Supported n |
|---|---|---|---:|
| PLAY_MUSIC | 90.78% / 90.78% / 90.07% | 124 / 125 / 123 | 141 |
| WEATHER | 90.78% / 92.91% / 90.07% | 123 / 127 / 124 | 141 |
| TIME | 94.33% / 92.20% / 95.04% | 127 / 127 / 125 | 141 |
| LIGHT_ON | 90.07% / 90.78% / 94.33% | 125 / 125 / 128 | 141 |
| LIGHT_OFF | 92.91% / 94.33% / 90.78% | 118 / 129 / 117 | 141 |
| PAUSE | 93.62% / 92.91% / 92.20% | 130 / 128 / 126 | 141 |
| LIST_REMINDERS | 94.33% / 94.33% / 92.91% | 131 / 130 / 131 | 141 |
| STOP | 90.07% / 91.49% / 88.65% | 125 / 122 / 121 | 141 |
| NEXT | 86.52% / 87.23% / 85.82% | 118 / 120 / 117 | 141 |
| VOLUME_UP | 78.72% / 81.56% / 80.14% | 100 / 107 / 108 | 141 |
| VOLUME_DOWN | 85.11% / 80.14% / 80.14% | 116 / 106 / 107 | 141 |
| CALL | 89.36% / 87.94% / 87.23% | 124 / 122 / 123 | 141 |
| MESSAGE | 93.43% / 94.16% / 90.51% | 126 / 125 / 124 | 137 |
| TIMER | 97.16% / 98.58% / 98.11% | 411 / 400 / 397 | 423 |
| ALARM | 95.27% / 96.45% / 95.04% | 380 / 391 / 378 | 423 |
| TEMPERATURE | 96.69% / 96.45% / 94.56% | 385 / 395 / 396 | 423 |
| BRIGHTNESS | 97.40% / 97.64% / 97.16% | 392 / 398 / 383 | 423 |
| COLOR | 92.91% / 95.98% / 96.69% | 348 / 367 / 357 | 423 |
| CREATE_REMINDER | 95.74% / 96.93% / 95.74% | 400 / 404 / 404 | 423 |
| OUT_OF_SCOPE | 82.82% / 82.21% / 77.30% | OOS: see rejection table | 0 |

## Slot heads

Ground-truth-intent head accuracy. Scalar slot calibration changes confidence, not argmax/classification. Three supported values per intent; candidate TEMPERATURE also has REJECT.

| Head | n | run_043 | BC8 | Conformer |
|---|---:|---:|---:|---:|
| TIMER | 423 | 100.00% | 99.76% | 98.82% |
| ALARM | 423 | 97.40% | 98.58% | 97.64% |
| TEMPERATURE | 423 | 98.58% | 98.82% | 98.11% |
| BRIGHTNESS | 423 | 97.40% | 99.05% | 96.45% |
| COLOR | 423 | 93.85% | 96.93% | 95.74% |
| CREATE_REMINDER | 423 | 98.82% | 100.00% | 99.05% |

Full head confusion matrices and slot accuracy conditional on correct intent are preserved in RESULTS.json. Missing test slot truth cannot establish unsupported-value safety; use the audited development unsupported cohort and future fresh recordings for that question.

## Paired comparisons

Candidate compared with run_043 under the same named policy design. Optimized policies may differ by the development-selected details. Also inspect candidate optimized versus unchanged run_043 original before proposing deployment.

| Candidate | Policy | Fixed / introduced errors | Semantic delta | Row bootstrap95% |
|---|---|---:|---:|---|
| BC8 (3s) | original | 81 / 111 | -0.64pp | [-1.21, -0.08]pp |
| BC8 (3s) | matched_joint | 101 / 84 | +0.36pp | [-0.19, +0.98]pp |
| BC8 (3s) | optimized | 120 / 78 | +0.89pp | [+0.30, +1.47]pp |
| BC8 + Conformer (3s) | original | 78 / 131 | -1.13pp | [-1.73, -0.55]pp |
| BC8 + Conformer (3s) | matched_joint | 72 / 118 | -0.98pp | [-1.56, -0.40]pp |
| BC8 + Conformer (3s) | optimized | 103 / 121 | -0.38pp | [-1.00, +0.21]pp |

Intervals are exploratory row bootstraps, not speaker-cluster estimates or multiple-policy-comparison correction. Single seed and repeated development selection limit reproducibility claims. Test results must not trigger policy refitting.

## Error patterns

run_043 (2s) strongest raw confusions: TEMPERATURE->OUT_OF_SCOPE: 11; COLOR->LIGHT_OFF: 11; ALARM->COLOR: 11; LIGHT_ON->LIGHT_OFF: 10; VOLUME_UP->VOLUME_DOWN: 8; OUT_OF_SCOPE->COLOR: 8; NEXT->OUT_OF_SCOPE: 8; VOLUME_UP->OUT_OF_SCOPE: 7.
BC8 (3s) strongest raw confusions: TEMPERATURE->OUT_OF_SCOPE: 15; OUT_OF_SCOPE->COLOR: 14; VOLUME_UP->OUT_OF_SCOPE: 12; VOLUME_DOWN->VOLUME_UP: 10; NEXT->OUT_OF_SCOPE: 10; VOLUME_DOWN->OUT_OF_SCOPE: 8; CALL->OUT_OF_SCOPE: 8; BRIGHTNESS->OUT_OF_SCOPE: 8.
BC8 + Conformer (3s) strongest raw confusions: TEMPERATURE->OUT_OF_SCOPE: 22; VOLUME_DOWN->VOLUME_UP: 12; OUT_OF_SCOPE->COLOR: 11; ALARM->COLOR: 11; VOLUME_UP->OUT_OF_SCOPE: 9; OUT_OF_SCOPE->LIGHT_ON: 9; CREATE_REMINDER->BRIGHTNESS: 9; OUT_OF_SCOPE->PAUSE: 8.

All policy row records include audio IDs, predicted/true slots, confidence, decision score, rejection reason, raw correctness and semantic correctness. They contain no raw audio and are not public-ready consent approvals.

## Efficiency

Same host, CPUExecutionProvider, one thread, batch1,50 timed forwards after10warmups. Pure model inference, not preprocessing/capture/action/TTS and NOT Raspberry Pi latency.

| Model | ONNX bytes | Host p50 | Host p95 |
|---|---:|---:|---:|
| run_043 (2s) | 1488732 | 13.16ms | 14.35ms |
| BC8 (3s) | 1490630 | 18.69ms | 22.71ms |
| BC8 + Conformer (3s) | 2914823 | 21.12ms | 21.93ms |

Optimized policies are evaluation-only postprocessors, not the currently deployed global confidence gate. Deployment requires a versioned adapter/export with once-only intent/slot temperatures, policy parity, accurate REJECT logging and physical Pi verification. Do not copy an optimized threshold into the old runtime alone.

## Development selection and train behavior

run_043 (2s): 176 development proposals. Selected policy: `{"kind":"oos","alpha":1.0,"slot_temperatures":{"TIMER":0.8906654560403103,"ALARM":1.496600479258843,"TEMPERATURE":1.9092762347014223,"BRIGHTNESS":1.3191373622200364,"COLOR":1.2019091240249598,"CREATE_REMINDER":1.5486161330540575},"thresholds":[0.9457986219024659,0.9546989530281675],"slot_oos_budget":5}`.
Validation supported/real correct 2457/2819 and 76/124; unsupported accepts 4/22. Historical challenge supported/real 663/975 and 127/439. Challenge was not used for policy choice.
BC8 (3s): 176 development proposals. Selected policy: `{"kind":"oos","alpha":1.0,"slot_temperatures":{},"thresholds":[0.8956228954887391,0.9703127661129796],"slot_oos_budget":6}`.
Validation supported/real correct 2555/2819 and 74/124; unsupported accepts 1/22. Historical challenge supported/real 669/975 and 134/439. Challenge was not used for policy choice.
Selected epoch42: augmented training intent 97.08%, raw validation intent 96.44%. Training has source weights/SpecAug, so this is not clean train-vs-test accuracy. Checkpoints were not chosen using final test.
BC8 + Conformer (3s): 176 development proposals. Selected policy: `{"kind":"oos","alpha":1.0,"slot_temperatures":{},"thresholds":[0.9094478948211671,0.9852316820956983],"slot_oos_budget":3}`.
Validation supported/real correct 2498/2819 and 81/124; unsupported accepts 0/22. Historical challenge supported/real 653/975 and 119/439. Challenge was not used for policy choice.
Selected epoch32: augmented training intent 96.15%, raw validation intent 95.38%. Training has source weights/SpecAug, so this is not clean train-vs-test accuracy. Checkpoints were not chosen using final test.

## Provenance and remaining work

Test source counts: `{"CommonVoice_en": 14, "FluentSpeechCommands": 144, "SLURP": 403, "SNIPS": 61, "TimersAndSuch": 2, "group_synthetic": 3590, "group_synthetic_oos": 29, "real_voice": 179, "synthetic_negative": 250, "xela_Multi-Sensor": 11, "xela_SET_TEMPERATURE_REAL": 10}`.
PRETEST_SEAL.json binds models/checkpoints/configs/mappings/policies/manifest/source. TEST_START.json records seal hash before feature generation. Development logits and search results, test logits, row errors, feature hashes, logs and protected26hash verification are retained.
Seven focused metric/normalization/gate tests passed before sealing. Empty-slot-only scoring is explicitly unscored rather than divide-by-zero. No physical Pi or hardware/action success is claimed.
No automatic champion decision is made by this report. Examine raw versus gated recognition, real cohorts, wrong accepted actions, OOS, candidate cost and fresh Pi evidence together. Do not select a different gate after reading this test table.
If a promising candidate is retained, next steps are a separately authorized runtime adapter/calibrated export, parity and fresh acoustic Pi tests; preserve run_043 until promotion is explicit. If future training uses these observed test errors, disclose that exposure and reserve a new independent final evaluation.
