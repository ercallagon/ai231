# Next architecture trial: genuine broadcasted BC-ResNet-8

Decision recorded 2026-10-08. This is an architecture experiment, not an
automatic deployment promotion. Pi models, thresholds, wake-word configuration,
tested GPIO wiring and transfer bundles are unchanged.

## Evidence reviewed

The available inventory contains 36 current run directories and 66 archived
JSON artifacts. Configs, checkpoint/result metadata, train histories, evaluation
metrics/confusions and recorded input hashes were reviewed alongside the
corrected dataset audit. The complete table and artifact fingerprints are in
`work/vcm_next_bc8/ALL_RUNS.md` and `all_run_inventory.json`. Incomplete runs and
unreadable historical JSON remain preserved, not silently treated as successes.
Some old effective overrides/source states cannot be reconstructed from current
configs. No AGENTS.md or work/PROJECT_HANDOFF.md was present in the inspected
workspace/ancestor locations; their absence is not permission to discard history.

Historical architecture families and conclusions:

| Trials | Actual architectural evidence | Conclusion |
|---|---|---|
| 005-007, 010-011, 014-015, 017 | A0/A1 ordinary dilated 2D convolutions, despite `ds_cnn` name | About 88-89% legacy intent; augmentation/OOS pilot did not establish an improvement |
| 008, 016, 018-019, 035 | Compact/wide TC-ResNet-style temporal residual models | Wider helped; later clean comparison does not justify returning to the weakest compact version |
| 009, 023, 028, 034 | CRNN; later bidirectional GRU with attention pooling | Competitive legacy intent, but weaker slot-fixed validation than A6/A10; recurrent export/runtime complexity |
| 012, 030 | A4 conventional full-3x3 residual CNN | Not paper BC-ResNet: no temporal broadcasting or SubSpectralNorm; cannot exclude genuine BC-ResNet |
| 013 | A5 real depthwise-separable CNN, about 51K parameters | 69.17% legacy test, 45.43% best val: repeating this capacity is uninformative |
| 021, 027, 032-033, 040 | A6 real 2D residual CNN, about 963K parameters | Strong quality reference, but corrected A10 is competitive at lower cost |
| 022, 029 | Lightweight MobileNet-style variants | No quality advantage demonstrated; do not add another similar mobile trial solely for diversity |
| 024 | A9 Transformer encoder called `conformer_tiny` | Not a full Conformer; weak existing result does not disprove the Conformer family, but adds little reason to prioritize it for Pi |
| 025, 031 GPU, 039 | A10 depthwise residual CNN with squeeze/excitation | Best current corrected efficiency/intent reference |
| 026 | A11 lightweight MatchboxNet-style temporal CNN | Prior result below finalists; larger literature variants would be separate hypotheses |
| 036-038 | CPU attempts/probe | Incomplete or screening only, not additional champion evidence |

Archived 10-intent champion s2v10 scored 93.88% intent and 92.55% strict E2E on
1,766 legacy test rows. s3v3 used a roughly 56.7K depthwise model with transfer
learning, slot weight 1.25 and light gain/shift/noise/reverb/far-field changes.
Those results support trying modest acoustically realistic augmentation later,
not claiming the same accuracy on today's 19-intent protocol. Their datasets,
label counts, transcription variants and score definitions differ.

## Corrected controls

Both controls completed 30 epochs with seed 42 on `hf_aligned_v1`.
Headline values below are raw argmax, not runtime-gated decisions.

| Model | Params | Best val intent / E2E | Test intent / E2E | Macro-F1 | Slot accuracy | OOS precision / recall | Host Torch mean |
|---|---:|---|---|---:|---:|---|---:|
| 039 A10 | 251,110 | 68.57 / 67.63% | 92.56 / 91.77% | 91.45% | 97.12% | 65.90 / 88.34% | 27.22 ms |
| 040 A6 | 962,702 | 69.60 / 68.49% | 91.95 / 91.63% | 90.82% | 98.03% | 62.33 / 85.28% | 41.31 ms |

Combined test has 4,693 rows, including 326 OOS; slot denominator is 2,538.
Slot accuracy uses the true intent's head; strict E2E also requires correct
intent and every available supported slot. Intent-only proxy rows do not prove
unsupported-slot rejection. Both training runs took about 12-15 minutes here,
excluding CPU evaluation. BC8 time is unknown, not extrapolated from parameter
count alone.

At the fixed 0.50 gate, A10 E2E becomes 90.67%; A6 becomes 90.60%. A10 accepts
10 OOS clips rather than 38, but rejects 345 valid commands rather than 149.
A6 accepts 13 OOS instead of 48, while rejecting 400 valid commands instead of
168. Rejection recall and precision are different tradeoffs: over-rejecting
valid commands generally hurts OOS precision, not OOS recall.

Persistent errors include volume up/down (A10 recalls 75.89/79.43%; A6
75.89/85.82%), light on/off, next/stop/media commands, and excessive rejection
of color commands. These contrasts can be lost by premature temporal/frequency
pooling; that is a hypothesis motivating the trial, not a proven cause.
Loss histories show substantially lower late training loss than validation
loss, consistent with a generalization gap. Neither control triggered early
stopping. Training loss alone does not establish training accuracy or diagnose
which acoustic condition caused an error.

The larger validation/test gap persists after correction. Validation is a
concentrated six-identity carve from HF train, not HF test or an HF val split;
487/1,171 rows belong to one speaker. Legacy supplemental test/holdout voice
leakage and five wrong cache entries were confirmed and fixed, but cache errors
alone could change total test accuracy by at most about 0.11 points. Do not
invent a single-cause explanation for the remaining distribution gap.

## Architecture and hypothesis

Trial: `run_041_hf_aligned_v1_bc8_s42`.
Source: `src/hey_swift/vcm/broadcast.py`, with the reference implementation's
BSD-3-Clause-Clear notice alongside it. This is an adaptation, not a pretrained
checkpoint or an exact replication of the paper's training protocol.

- Input: one 80x198 log-mel tensor.
- Stem: 5x5 convolution, 128 channels, frequency stride two, no time downsampling.
- Four stages: widths 64/96/128/160, repeat counts 2/2/4/4.
- Each block: frequency-depthwise 3x1 plus five-band SubSpectralNorm; frequency
  average feeds a temporal-depthwise 1x3 branch with dilation 1/2/4/8, BN,
  SiLU, pointwise projection and 0.1 dropout. Broadcast temporal residual back
  into the frequency-resolved branch, with identity where dimensions permit.
- Frequency-only strides in stages two and three; retain temporal resolution.
- Tail: depthwise 5x5, pointwise 256, BN/ReLU, global pooling.
- Shared embedding feeds 20-way intent/OOS and six independent 3-way slot heads.

Verified untrained preflight: 327,750 parameters; ONNX 1,448,636 bytes;
finite backward pass; ONNX checker and all-output Torch/ORT parity for batches
one and two. Untrained one-thread host ORT p50/p95: 12.84/13.20 ms. This is
neither a Pi measurement nor directly comparable to the controls' Torch timing.
Temporal preservation may cost more activations than parameter count suggests.

Hypothesis: broadcasted temporal context and frequency-local normalization
will preserve short discriminating words and slot cues more efficiently than
generic residual blocks. A6/A10 remain the defaults until evidence beats them.

## Controlled protocol

No dataset rebuild, legacy augmentation cache reuse, warm-start or test-derived
choice is permitted. The pinned HF revision, source-qualified manifests,
labels, clean finite float32 cache and all split hashes are identical to 039/040.
Protocol SHA-256:
`46cc113e2844bd3b593112cc3e0d582e43b89327236fb1dbcdd6776a64a38b9a`.
See `docs/vcm-hf-aligned.md` for complete subset/lineage checks and limitations.

Data: train 28,335; local val 1,171; official test 4,443; negative test 250;
calibration-only OOS 400; official Pi holdout 202, untouched. Supplemental
test/holdout voices and reserved numerals never enter training.
Exact audio/PCM, declared speech lineage and model-input overlap checks passed;
undeclared near-duplicates, clone identity equivalence and acoustic transcription
correctness are not independently proven.

Features: 16 kHz mono, two seconds, FFT 400, hop 160, 80 log-mels, 20-8,000 Hz,
same normalization/padding as controls. Only training receives online
SpecAugment; clean features are precomputed. No new waveform perturbations.
Optimizer: AdamW, LR 0.003, weight decay 0.01, two warmup epochs then cosine,
batch 256, four workers, seed 42, maximum 30 epochs, patience eight.
Loss: inverse-frequency weighted intent CE, label smoothing 0.05, plus 0.5
times the existing intent-masked slot CE. Unsupported/empty slot targets remain
unsupervised. Changing slot weight now would confound the architecture test.
Checkpoint selection retains the controls' best validation E2E rule.

Calibration is a separate matched procedure for BC8 and the two frozen controls:
fit one scalar temperature on validation NLL, then scan confidence thresholds
using validation in-scope correctness and reserved `tune_oos` speech. Require
at most 5% tune-OOS false accepts; among feasible thresholds maximize correctly
accepted validation intents, then minimize valid-command rejection. If none is
feasible, report failure rather than silently relaxing the target. This intent
criterion is not a claim of calibrated arbitrary-slot safety.
Temperature/threshold are frozen before each new calibrated test report;
test and holdout never fit them. Both raw and calibrated/gated results remain
separate. Temperature scaling does not improve argmax accuracy.
Baseline test results were already inspected to formulate hypotheses: the
official test is a known benchmark, not a never-seen discovery set. The reserved
holdout remains the independent final check.

## Selection and next decisions

Pre-registered material-gain targets: at least one percentage point validation
intent improvement without more than 0.5-point E2E loss, or comparable quality
with at least 25% matched-backend CPU cost reduction. These are screening
criteria, not statistical guarantees. Compare raw macro-F1 and all per-class
P/R/F1/confusions; scrutinize volume, lights, media and supported slots. Compare
calibrated decisions at the same tune-OOS constraint, including valid-command
false rejects, OOS false accepts, intent accuracy and strict E2E.

Record parameter/checkpoint/ONNX size, full train/validation histories and the
same Torch latency proxy as controls. Do not pick a champion from a tiny
single-seed test difference. If competitive, repeat corrected finalists with
a second seed and evaluate paired errors/uncertainty before promotion. No extra
training is queued automatically after this trial. A weak BC8 result retains
A10 as the efficiency reference, not a requirement to keep increasing capacity.

Deployment acceptance remains actual Pi 5 (4 GB) testing: ONNX shape/output
compatibility and parity, full assistant RSS, microphone endpointing, wake/chime
interaction, music interference and action timing. Aspirational budgets are
VCM p95 <=100 ms, full assistant RSS <1 GB, local end-of-speech-to-action p95
<=500 ms; these are targets, not measured results or API-response guarantees.
The experiment must not alter the currently tested wake model, wiring or speech
assets. Unsupported-value rejection needs a separately labeled slot-reject
experiment before claiming arbitrary-value command safety.

## Reproduce and monitor

All commands run from the repository root in the training environment:

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/validate_vcm_protocol.py --config configs/vcm_hf_aligned_v1_bc8.yaml
python scripts/vcm_bc8_trial.py preflight
python -m pytest -q --junitxml=work/vcm_next_bc8/regression.xml
python -u scripts/run_vcm_bc8_queue.py --detach
```

On a fresh checkout the immutable config is supplied; preparation is unnecessary
once the corrected protocol/cache are present. Follow the HF-aligned guide to
fetch/precompute missing data without committing WAVs. Inventory refresh only:
`python scripts/prepare_vcm_bc8_trial.py --inventory-only`.

The process-local factory bridge intentionally does not edit frozen shared
trainer/evaluator code or legacy architecture definitions. The config's `a4`
is a routing alias only; checkpoint architecture is `bc_resnet_broadcast8`.
**Use the adapter, not the generic training/evaluation CLI, for this candidate.**
Its eventual runtime artifact is standard ONNX; temperature is embedded into
the calibrated export's intent logits, not the slot outputs.

The persistent queue waits for the old corrected queue to finish; requires a
single GPU below 40% utilization, >=20,000 MiB free and working CUDA; checks
all GPUs and launches no SLURM jobs. It does not reserve a GPU exclusively or
terminate other users' work. If the safety gate cannot pass, it waits without
agent polling. The exact executable/GPU/command/config/protocol/preflight hashes
are recorded in `work/vcm_next_bc8/launch.json`.

```bash
CUDA_VISIBLE_DEVICES=<selected_gpu> PYTHONPATH=src \
  python -u scripts/vcm_bc8_trial.py train \
  --config configs/vcm_hf_aligned_v1_bc8.yaml
tail -f work/vcm_next_bc8/queue.log
tail -f work/vcm_next_bc8/run_041_hf_aligned_v1_bc8_s42.log
```

Outputs: fresh `experiments/vcm/run_041_hf_aligned_v1_bc8_s42/`, with immutable
config, checkpoint, result/history, source/data/environment/Git provenance,
raw evaluation metrics, calibration proof, separate calibrated results and ONNX.
Controls receive new `calibration.json` and `calibrated/` only; their raw
checkpoints/results are not overwritten. No holdout evaluation or Pi promotion
is performed by this queue. Logs survive closing the conversation; the process
does not survive a host reboot. Preserve/commit the untracked source before
submission; Git HEAD alone does not identify this dirty workspace.

## Primary research

- [BC-ResNet paper](https://arxiv.org/abs/2106.04140) and
  [official Qualcomm reference](https://github.com/Qualcomm-AI-research/bcresnet):
  genuine temporal broadcasting and sub-spectral normalization; Speech Commands
  results are not predictions for Hey Swift.
- [MatchboxNet](https://arxiv.org/abs/2004.08531): compact residual temporal
  separable convolutions; relevant alternative already screened in lite form.
- [Multi-scale DS-ResNet](https://www.isca-archive.org/interspeech_2020/yang20d_interspeech.pdf):
  related depthwise residual keyword spotting evidence, not an exact A10 replica.
- [Temperature scaling](https://arxiv.org/abs/1706.04599): validation-based
  confidence calibration, not guaranteed OOS recognition.
- [Home Assistant Speech-to-Phrase](https://www.home-assistant.io/blog/2025/02/13/voice-chapter-9-speech-to-phrase/)
  and [Rhasspy integration](https://www.home-assistant.io/integrations/rhasspy/):
  practical offline constrained-command assistants. Their Pi timings are
  pipeline-specific, not measurements of this classifier or our action loop.

Next separately controlled experiments, if warranted: group-stratified
multi-seed validation, supported-versus-unsupported slot rejection, and modest
realistic waveform augmentation. Do not combine these with this architecture
trial and then attribute their effects to BC8.
