# Grouped validation and reproducible BC8 retraining

## Scope

Protocol `hf_grouped_v2b`; fresh trial `run_042_hf_grouped_v2b_bc8_s42`.
This corrects validation selection and training observability/reproducibility,
not the BC8 architecture or waveform augmentation recipe. No old config,
manifest, checkpoint or result is overwritten. Pi wake/VCM models, thresholds,
hardware wiring and bundles remain untouched. No SLURM.

## Split design and evidence

The old validation algorithm selected the largest speaker groups until reaching
10% of core training rows. Its comments said stratified/smallest-first, but its
code did neither. The resulting 1,171 clips contained only six identities and
one speaker contributed 487 rows. That is not a representative validation carve.
No HF-provided validation split exists; validation is local to this project.

V2 draws validation only from the authenticated, eligible **v1 training pool**.
The old six validation identities remain excluded globally and their unchanged
clips become a separate challenge set. This retains a harder real-voice test
instead of reusing it in training or hiding its poor results. It also means the
new training pool is smaller; this is not merely renaming existing metrics.

Use scikit-learn GroupShuffleSplit, seed 42, 4,096 candidate assignments with
10% of eligible positive speech identities selected. Select using metadata
only: equal-weight profile distances for intents, slots, source, HF subset,
and real/synthetic flags. No model predictions, loss or test audio inform the
selection. Hard constraints: 8.5-12.5% positive rows, largest positive speaker
at most 35% of validation, at least 30 non-synthetic positive rows, minimum ten
per available intent and five per available raw slot category (or the entire
category when fewer are available). Canonical supported-slot coverage is checked
again after assignment. Sources with >=30 clips and >=2 speech identities must
appear in both train and validation. Exact balancing is impossible with
indivisible groups. The first draft (`hf_grouped_v2`) failed this stronger source
coverage requirement and was not trained. Its artifacts and original source
snapshot remain preserved as preparation evidence; use finalized `v2b`, not
the draft config/protocol. Draft validation requires its saved source snapshot.

Supplemental clips follow their speaker. A speech-derived negative goes to a
split only when **all contributing voices** belong there; cross-boundary
composites are excluded, never randomly placed. Declared noise-only/near-silence
negatives without speech are assigned by seeded source-key hash. Eleven core
Speech Commands OOS clips lack declared speaker identity and are conservatively
excluded from both fitting and validation. This does not assert that they are
faulty audio, or that upstream metadata proves whether they are speech/noise.
All 47 exclusions (36 mixed-lineage composites, 11 unknown speaker identities)
and original rows remain recorded under `work/vcm_grouped_v2/`.
The finalized audit records total exclusions and individual reasons. Details
are in `work/vcm_grouped_v2/hf_grouped_v2b_excluded.csv`.

| Split | Clips | Role |
|---|---:|---|
| train | 25,366 | Eligible v1 training clips minus new val and explicit exclusions |
| val | 2,922 | New metadata-balanced validation; 32 selected speech identities |
| challenge | 1,171 | Unchanged old validation, still speaker-isolated from train/new val |
| test_core | 4,443 | Official HF test, byte-identical manifest |
| test_negatives | 250 | Official synthetic-negative test, byte-identical manifest |
| test | 4,693 | Unchanged combined test |
| holdout | 202 | Official Pi holdout, never accessed by this trial |
| tune_oos | 400 | Unchanged OOS calibration-only speech |

Audio bytes/PCM, declared speech identities/dependencies and exact model-input
overlap checks include the independent challenge. The new validator also resolves
manifest locators/labels/variations against pinned authoritative Parquet metadata.
All 20 labels and supported canonical slot values are represented in train/val.
Synthetic-heavy training remains synthetic-heavy validation; report real versus
synthetic and per-source/speaker scores, not only an easier aggregate number.
The challenge is a known historical challenge, not a pristine unseen benchmark.

## Feature reuse

Raw HF revision, labels, 16 kHz two-second 80x198 mel preprocessing and feature
values are unchanged. Reuse the authenticated v1 `mels.npy` and `index.json`
through repository-relative symlinks in `data/vcm/processed/hf_grouped_v2b/`.
Create new protocol-bound cache metadata; revalidate the original feature hashes
and finite tensors. Excluded rows may remain in the shared cache but are never
loaded by training. This avoids decoding audio or duplicating a 2.2-GB store just
because membership changed. A fresh checkout must prepare/validate v1 before v2.

## Training pipeline audit and changes

The historical trainer remains intact because v1 authenticates its source hash.
The new `train_repro.py` fixes the following in a separately versioned path:

- Historical logs only contained training loss, not training accuracy. New logs
  identify **augmented training accuracy** explicitly, plus validation intent,
  macro-F1, strict E2E, slots, per-class confusion and per-speaker/source groups.
  After training, evaluate the best checkpoint on clean training inputs once;
  this seen-example metric is diagnostic, never a substitute for validation.
- Historical initialization disabled cuDNN benchmarking, then later enabled it.
  V2 keeps benchmarking off, enables strict deterministic algorithms, fixes
  cuBLAS workspace configuration and disables TF32. Deterministic kernels can
  be slower; CUDA/PyTorch/hardware changes can still change numerical results.
- SpecAugment is keyed to experiment seed, epoch and sample index. Worker count
  or scheduling cannot silently change the augmentation for a particular sample.
  Shuffle uses an explicit Torch generator; workers get explicit seeded RNGs.
- Validation is clean, `model.eval()`, FP32 and gradient-free. No augmentation,
  dropout or optimizer updates apply to validation. BF16 is used for training
  only on supporting GPUs. No silent fallback from unavailable CUDA to CPU.
- Record the same configured weighted/smoothed intent+slot loss on train/val,
  and separately record unweighted intent NLL. The old train and val losses used
  different weighting/smoothing and should not be interpreted as identical
  objectives. Augmented training predictions still use changing model weights.
- Pinned-memory transfers use non-blocking CUDA copies; persistent workers reuse
  the clean memory-mapped features. No raw-WAV decoding enters the training loop.
- Preserve best and last checkpoints with optimizer/scheduler and RNG states,
  full history, config, dataset/code fingerprints, Git status and environment.
  State is retained for recovery; automatic resume is not implemented or claimed.

Keep the architecture, 30-epoch maximum, eight-epoch patience, seed 42, batch
256, AdamW LR 0.003/weight decay 0.01, two-epoch warmup/cosine, smoothing 0.05,
train-only SpecAugment and masked slot CE weight 0.5 unchanged. Select best raw
validation strict E2E; ties retain the earlier epoch. No extra architecture or
hyperparameter search is launched. A10 has not been retrained on v2, so v1 A10
and v2 BC8 do not form a controlled architecture comparison.

## Metrics and calibration

Headline intent accuracy/macro-F1 are raw argmax; slot accuracy uses the true
intent head; strict E2E requires intent plus applicable supported slots. Report
after-gate decision accuracy separately, with OOS false accepts and valid-command
false rejects. Preserve intent-only unsupported-value proxies as annotated;
three-way heads still do not establish arbitrary-value rejection.

Choose temperature using **new val only**, and confidence threshold using new
val plus `tune_oos`, with <=5% tune-OOS false accepts. Fit neither on challenge,
test nor holdout. Freeze choices before final known-test/challenge evaluation;
the queue will not recalibrate based on these results. Temperature is embedded
into calibrated ONNX intent logits; slot outputs are unchanged.

Compare v2 validation by source and the unchanged historical challenge, and
compare fixed-policy test results with v1 BC8 descriptively. A higher new val
score can reflect the new composition rather than a better model. Recommend a
second seed and an A10 v2 control before claiming small architecture gains.
Final deployment still needs live Pi real-voice/noise/music and action-latency
testing. Host timing and ONNX parity are not actual Pi verification.

## Reproduce and monitor

From the repository root with the training environment and v1 data/cache present:

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/prepare_vcm_grouped_protocol.py
python scripts/prepare_vcm_grouped_trial.py
python scripts/validate_vcm_protocol.py --config configs/vcm_hf_grouped_v2b_bc8.yaml
python -m pytest -q --junitxml=work/vcm_grouped_v2/regression.xml
python scripts/prepare_vcm_grouped_trial.py --seal-readiness
python -u scripts/run_vcm_grouped_bc8_queue.py --detach
```

Preparation refuses to overwrite existing outputs; validate instead when already
prepared. Existing raw Parquet downloads/features are reused; no WAVs need be
committed. Follow `docs/vcm-hf-aligned.md` to prepare v1 on a fresh checkout.
The queue uses one GPU, checks all devices, requires <40% utilization and >=20 GB
free across two readings plus working CUDA. Availability is a safety heuristic,
not exclusive reservation. Closing chat does not stop the persistent process;
a host reboot does. No other processes are interrupted and no SLURM is used.

Exact training command after a GPU passes the gate:

```bash
CUDA_VISIBLE_DEVICES=<selected_gpu> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=42 PYTHONPATH=src \
  python -u scripts/vcm_grouped_bc8_trial.py train \
  --config configs/vcm_hf_grouped_v2b_bc8.yaml
tail -f work/vcm_grouped_v2/queue.log
tail -f work/vcm_grouped_v2/run_042_hf_grouped_v2b_bc8_s42.log
```

Outputs live in `experiments/vcm/run_042_hf_grouped_v2b_bc8_s42/`; staging,
preservation fingerprints, source snapshots and regression evidence under
`work/vcm_grouped_v2/`. The queue completes calibration, known test/challenge
evaluation, CPU latency and trained ONNX export/parity, then stops. It never
evaluates holdout or promotes a Pi model automatically.

References: [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
Fixed seeds improve reproducibility within a fixed environment, not statistical
certainty across speakers or portability of exact floating-point results.
