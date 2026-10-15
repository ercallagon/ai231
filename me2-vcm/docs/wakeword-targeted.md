# Targeted wakeword trial

## Evidence and limits

The deployed exp15 WakeGRU has 204,738 parameters: three convolution blocks
(16/32/64 channels), GRU-128 and a 64-unit classifier. Its input is 80 x 100
mel/delta frames. The original ONNX, configuration, 0.30 threshold, sustain=2,
cooldown and hardware wiring remain unchanged.

Archived exp15 fixed-threshold clean test recall was 0.6602 with clip false
acceptance 0.03846. Historical best-test threshold reports selected a threshold
on test itself and are not deployable validation-selected comparisons. Historical
temporal proxies combined independent clips and do not establish stream recall.
The archived false-fires/hour replay also used different settings; it is not a
measurement of the current Pi loop. A separately reproduced exp15 checkpoint is
not the deployed checkpoint. This trial verifies the actual deployed parent
against the archived checkpoint using ONNX/Torch parity before training.

The local dataset contains train 14,818, validation 1,140 and test 1,143 clips,
including 400 mined training negatives. The preparation audit checks manifests,
metadata labels, byte/decoded-PCM fingerprints, and declared voice overlap.
Pseudo-identities for noise/mined clips do not prove speaker independence.
Near-duplicates, semantic lineage and representative long-stream Pi negatives
remain unverified. The first preparation correctly stopped on validation/test
PCM overlap; its incomplete stage is preserved in `work/wakeword_targeted/`.
Version 2 leaves training and original manifests unchanged and builds new
evaluation subsets excluding audio/declared speakers found in parent training,
validation overlap with original test, and within-evaluation duplicates. Exact
exclusions, reasons and denominators are recorded in the versioned audit.
Parent and candidates are rescored on the identical corrected evaluation view;
new scores must not be compared directly with historical denominators.
This cannot prove the complete training lineage of an archived checkpoint or
detect all near-duplicates. Original data is never edited.

The completed v2 PCM audit found 16 hashes shared between validation and test,
all positive TTS audio (69 rows across the two original splits). Removing 46
validation rows overlapping test, 42 additional within-validation duplicates,
and 30 within-test duplicates leaves validation 1,052 (326 wake / 726 negative)
and test 1,113 (385 wake / 728 negative). There is no exact train/evaluation PCM
overlap or conflicting PCM labels. Training retains 400 duplicate-surplus rows
to keep the continuation distribution intact; training multiplicity is a
remaining sampling limitation, not independent additional audio evidence.

## Bounded Experiment

Outputs live under `work/wakeword_targeted_v2/` and fresh
`experiments/wakeword/exp16_targeted_hardneg_s20261009/` and
`experiments/wakeword/exp17_targeted_confirm_s7/` directories.

The first trial fine-tunes the existing architecture with extra loss weight on
training negatives that the deployed parent scores highly. Negative weight is
`1 + 3 * min(parent_wake_probability / 0.30, 1)`; positives keep weight 1.
This tests targeted false-activation reduction without increasing model size.
AdamW uses LR 1e-4, weight decay 1e-4, batch 128, cosine decay, 12 maximum
epochs, patience 4, gradient clipping 5 and deterministic seed 20261009.

Precompute retains clean features plus one seeded low-pass/noise view per clip.
Training samples either view and applies stateless temporal shift/masking.
Validation and test perturbations are fixed stress views, not training examples.
This is not a matched augmentation-only ablation: improvement cannot be
attributed exclusively to hard-negative weighting without an unweighted
continuation control. No architecture superiority claim is intended.

Each checkpoint/policy is selected on validation only, maximizing wake recall
then F1 subject to <=1% validation clip false acceptance. Only a >=1 percentage
point recall improvement with <=2-point perturbed recall loss receives a second
seed (7). Otherwise the queue stops without inspecting test predictions.
After confirmation, selection is sealed before test evaluation. Reports include
fixed 0.30 and validation-selected policies, independent-clip runtime replay,
ONNX parity, file size and single-thread host latency. Clip FAR is not false
activations/hour. Real Pi microphone, music playback, far-field and multi-hour
negative-stream tests remain required before any promotion.

## Reproduction and Monitoring

Use the project's `onit` environment with `PYTHONPATH=src`,
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`:

```bash
python -u scripts/wakeword_targeted_trial.py prepare
pytest tests --junitxml=work/wakeword_targeted_v2/regression.xml
python scripts/seal_wakeword_targeted.py
python -u scripts/run_wakeword_targeted_queue.py --detach
```

Preparation/queue creation refuses existing staging/output directories. Fresh
artifacts preserve original hashes, source snapshots, cache fingerprints,
environment and Git provenance. Queue launch receipts record exact commands and
the physical GPU. Best/last checkpoints retain optimizer, scheduler and RNGs.

The persistent queue waits for the VCM discovery controller and its completion
report, then checks every 10 seconds for a GPU below 40% utilization with at
least 20,000 MiB free, confirms after 5 seconds, and probes CUDA. It never uses
SLURM or interrupts another job. The conservative memory gate is shared with
VCM, not an estimate of this small model's actual memory demand.

```bash
tail -f work/wakeword_targeted_v2/queue.log
tail -f work/vcm_oos_discovery/queue.log
tail -f work/wakeword_targeted_v2/exp16_targeted_hardneg_s20261009.log
```
