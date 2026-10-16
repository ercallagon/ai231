# ME2 two-slide submission checklist

Reviewed 2026-10-15 against both pages of the uploaded me2-deck.pdf, SHA-256
305f13f6ca6418fbd64c33bcedfda200cb49fbc0ed1b21caee47aeb7e646ed4f.
The instructor-approved device is **Raspberry Pi 5, 4 GB**, not the template's
Pi 4. This is an evidence checklist, not certification of completed Pi testing.

## Six reviewer requirements

| Requirement | Available evidence | Remaining limitation |
| --- | --- | --- |
| Public MIT repository; one-command reproduction | README, LICENSE, run.sh/run.ps1, scripts/reproduce.py; clean-environment VCM result parity in CHECKS.json | MIT covers owned code, not model/data terms. Full public wake retraining is blocked by raw-recording access/consent. |
| Licensed/citable dataset; DOI/access terms | data/PROVENANCE.json, data/DATASET_LICENSE.md; DOI 10.57967/hf/10723; pinned HF revision | Source-specific restrictions and incomplete cloned-voice consent/licensing remain. Do not imply commercial or universally cleared rights. |
| Training logs and final checkpoint committed | experiments/selected/checkpoint.pt; full logs/training/selected.jsonl, history/config/source/environment; baseline checkpoint/logs; original exp15 checkpoint/log/parity | Raw data is not published. Checkpoint recovery does not resolve wake-data access. |
| Pi latency reproduced by posted script | scripts/benchmark_onnx_pi.py; docs/PI5.md; Champion acoustic benchmark summarized in PI5_ACOUSTIC_CHAMPION_20261017.md | Pi 5 acoustic run completed on Champion only. Baseline comparison is frozen injected test, not acoustic A/B. Decision p95 is detector-estimated; external timing contains negative values. |
| Held-out test; unseen speakers | Frozen manifests, explicit exposure history and reports/evaluation/ | Existing test/holdout were previously examined. Declared groups are not verified humans; see overlap caveats below. Wake positives are TTS. |
| Comparable-size baseline compared | run_043 327,750 versus selected 488,264 parameters; common frozen test, paired/family errors and selectable profiles | About 1.49x size ratio; capacity/preprocessing/training lineage differ. System comparison, not clean architectural ablation. Acoustic Baseline was not run. |

## Slide 1: model and training fields

| Field | Verified value/source |
| --- | --- |
| Model | Scratch BC8 backbone; 128-channel residual temporal blocks, kernel 5, dilations 1/3/9; masked attentive mean/std pooling; 256-dimensional embedding; intent and slot heads. See experiments/selected/model_source.py. |
| Audio/features | 16 kHz mono; FFT 400, hop 160; 80 x 600 log-mel plus 600-frame validity mask; maximum six-second retained input, not mandatory listening time. |
| Parameters/weights | Selected VCM 488,264 / ONNX 2,136,900 bytes; exp15 wake 204,738 / ONNX 825,493 bytes. Selected training checkpoint 6,138,121 bytes. Decimal MB and checkpoint size are distinct. |
| Labels | 19 supported intents plus OOS (20 logits); six slot heads, three supported values each (18 total), plus selected TEMPERATURE REJECT. |
| Objective | Weighted intent CE, label smoothing 0.05, plus 0.75 applicable slot CE; gradient clip 5. Not CTC or attention sequence decoding. |
| Optimizer/schedule | AdamW initial lr 0.0003, wd 0.01; ReduceLROnPlateau factor 0.5/patience 6; batch 32. |
| Compute/selection | One A100-SXM4-40GB, seed 42; 72 epochs, best epoch 52; 3.64645 allocated GPU-hours from logged epoch wall durations, including validation/host overhead. Node ID is not available in committed provenance. |
| Steps/loss | Derived 792 updates/epoch, 57,024 total; 41,184 through selected epoch. Best-epoch mean augmented objective 0.4076313. See experiments/selected/TRAINING_ACCOUNTING.json; steps are derived, not a saved counter. |
| Architecture diagram | Replace template causal encoder/KV-cache with wake -> dynamic capture -> BC8/TCN/pooling -> heads -> rejection -> action. The temporal network is not causal; no KV-cache or autoregressive decoding is implemented. |

## Dataset fields

Pinned source: airimonda/ai231-me2-voice-commands, revision
bfca0bbebfe17e00d6a685d1da573615d29ecdce. See data/PROVENANCE.json for hashes.

| Split | Manifest recordings | Source waveform hours | Real rows |
| --- | ---: | ---: | ---: |
| train | 25,366 | 14.4008 | 1,559 |
| validation | 2,922 | 1.8623 | 158 |
| tune_oos | 400 | 0.5652 | 400 |
| test | 4,693 | 2.6260 | 824 |
| holdout | 202 | 0.1791 | 96 |

Actual eligible training count is 25,330 after 36 recorded exclusions, not the
full manifest count. Hours are original audio, not padded duration or speech-only
time. Train has 559 declared pseudonymized groups, **not 559 verified humans**.
Confirm team/name/date and use a qualified speaker count instead of guessing.

### Lightweight manifest overlap check

This update compared nonempty encoded-audio hashes, decoded-PCM hashes and
declared group IDs directly from the committed manifests; no model evaluation
or split change was needed.

| Pair | Encoded/PCM hash intersections | Declared group intersection |
| --- | --- | ---: |
| train/validation | 0 / 0 | 17 |
| train/test | 0 / 0 | 1 |
| train/holdout | 0 / 0 | 0 |
| validation/test | 0 / 0 | 1 |
| validation/holdout | 0 / 0 | 0 |
| test/holdout | 0 / 0 | 0 |

All shared group IDs above occur on rows marked synthetic_negative. The sole
train/test shared group has 189 train, 11 validation and 50 test rows. Public
metadata does not establish whether it is a source placeholder or voice identity;
do not call it confirmed real-speaker leakage or claim full voice independence.
Hash nonoverlap does not establish speaker identity or exclude near-duplicates
and shared augmentation ancestry. Test/holdout are previously exposed evaluation.

## Accuracy and rejection fields available now

| Metric | Test | Holdout |
| --- | ---: | ---: |
| Selected VCM raw intent | 4,466/4,693 = 95.16% | 191/202 = 94.55% |
| Real intent | 649/824 = 78.76% | 87/96 = 90.63% |
| Synthetic intent | 3,817/3,869 = 98.66% | 104/106 = 98.11% |
| Accepted-correct supported intent + slot | 4,046/4,367 = 92.65% | 178/186 = 95.70% |
| OOS false accepts | 14/326 = 4.29% | 1/16 = 6.25% |
| Processing failures, full manifest retained | 16/4,693 | 0/202 |

Full denominators, processed-only counts, per-family and paired errors are in
RESULTS.md and reports/evaluation/. Raw intent accuracy is not gated joint success.

Exp15 corrected isolated clip test: 1,113 clips, 385 synthetic positives and
728 negatives. Historical fixed feature-crop recipe: accuracy 85.53%, precision
90.00%, recall 65.45%, F1 75.79%, negative-clip FPR 3.85%. Supplementary waveform-
first one-second recipe: accuracy 86.25%, recall 68.57%, F1 77.53%, FPR 4.40%.
These are distinct preprocessing views of unchanged weights, not Pi streaming
results. False wake events/hour cannot be inferred from clip FPR. Streaming
experiments are reported separately and did not justify replacing exp15.

## Pi measurements and one-command scope

- scripts/reproduce.py verify checks artifacts/contracts, not recognition.
- scripts/reproduce.py reproduce --fetch downloads about 305 MB of pinned
  test/holdout Parquet and evaluates both VCM profiles; no training data/models
  are downloaded by this step. It reports cohort/family/rejection/processing
  results and feature+ONNX timing. It bypasses wake/microphone/live endpointing.
- scripts/benchmark_onnx_pi.py measures model-forward p95 and nominal-capacity
  RTF on synthetic feature inputs, recording device, versions and intra/inter
  threads. It does not measure recognition, capture, services or TTS.
- docs/BENCHMARK.md documents the pinned official laptop-speaker -> Pi-microphone
  benchmark. Injection is supplementary. One-time SSH, levels, device selection
  and wake-take setup are still required; not a zero-manual-setup claim.
- Champion Pi 5 acoustic measurements and operator GPIO/audio observations are
  available; see PI5_ACOUSTIC_CHAMPION_20261017.md for scope and provenance gaps.
  Communication recovery and independent-speaker verification remain pending.
  Do not present detector timing as annotated speech-end/action latency.

## Slide 2 links and remaining items

Repository: https://github.com/ercallagon/ai231/tree/main/me2-vcm.
Dataset DOI: https://doi.org/10.57967/hf/10723, with data/DATASET_LICENSE.md.
Weights are downloadable in models/ and experiments/selected/checkpoint.pt;
MODEL_TERMS.md governs them separately from owned-code MIT. A tagged GitHub
release URL is not established by this checklist; confirm if literally required.

Remaining: confirm official author/team attribution (draft deck names Edgardo
Callagon Jr.), unavailable node ID, qualified human-speaker counts, dataset/voice
rights gaps, fresh real-speaker confirmation, and call-recovery verification.
Pi 5 Champion acoustic evidence is available with limitations in
PI5_ACOUSTIC_CHAMPION_20261017.md. No further retraining is required merely to
complete the deck. Preserve existing installation, accounts, wiring and rollback.
