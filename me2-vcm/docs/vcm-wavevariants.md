# Reproducible Training Waveform Trial

## Scope

Keep genuine BC8, corrected grouped v2b splits, features, labels, slots, source
loss weighting and deployment files unchanged. Warm-start both fresh runs from
run_043; do not overwrite its checkpoint or any historical results. No SLURM.

The previous waveform trials used historical protocols with known audit issues.
Run_043 uses clean cached features plus online SpecAugment, with no new waveform
views. Its real validation intent accuracy is about 74%, versus high aggregate
accuracy on synthetic-heavy evaluation. This motivates acoustic domain testing,
not another architecture or uncontrolled increase in TTS volume.

## Cache and Reproduction

`work/vcm_wavevariants_v1/` holds at most 6,000 float32 mel variants of existing
training parents, about 380 MB (363 MiB), plus a recipe/provenance manifest.
No generated WAVs or duplicate base cache are retained. Real training rows are
prioritized; remaining allocation round-robins intent/slot/source groups using
seeded source-key ordering. Each source receives one fixed view, generated from
seed 20261009 plus its SHA-256 source key and recipe version. The view is chosen
with probability 0.5 when that eligible parent is sampled; all other examples
remain clean before the identical online SpecAugment policy.

The validated cache has exactly 6,000 variants, including all 1,559 real-source
training clips, and uses 380,160,128 bytes for features. Twelve direct/replayed
feature checks and the actual model backward check passed. These are alternate
views of existing examples, not 6,000 new independent utterances; each epoch
still samples the same 25,366 training parents.

Four seeded profiles cover colored/harmonic interference, synthetic room
reflections plus low-pass microphone coloration, their combination, and gain
plus mild non-circular timing padding. Noise SNR is 15-30 dB; room tails are
0.10-0.25 seconds; low-pass cutoff is 3.5-7 kHz; gain is 0.65-1.25 with peak
safety scaling. Harmonic interference is a non-speech proxy, **not real music**.
No new voices, transcripts or slot values are generated.

Raw Parquet bytes and decoded PCM must match the authenticated parent manifest.
Every parent must be training-only, including declared speech identities;
variant PCM must not match protected split PCM. Shapes/dtypes/finiteness,
deterministic direct-vs-cached parity, original-clean-feature parity and actual
model/dataset backward execution are checked before queuing. Validation, test,
challenge, tune-OOS and holdout membership remain unchanged. Only train waveform
views are generated. The base feature cache is reused without modification.

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u scripts/prepare_vcm_wavevariants.py prepare
python -m pytest tests --junitxml=work/vcm_wavevariants_v1/regression.xml
python scripts/prepare_vcm_wavevariants.py seal
python -u scripts/run_vcm_wavevariants_queue.py --detach
```

Use the project's `onit` environment. Preparation refuses existing staging or
experiment paths. Source snapshots, recipes, environment, Git state, parent and
split/cache fingerprints are retained. Recreate at a new versioned path if any
source/data/recipe changes; never silently reuse stale views.

## Matched Experiment

1. `run_046_hf_grouped_v2b_bc8_aug_control_s42`: continuation with existing
   source-balanced intent loss and online SpecAugment, no waveform substitution.
2. `run_047_hf_grouped_v2b_bc8_wavevariants_s42`: identical continuation with
   the prepared waveform-view substitutions.

Both use seed 42, AdamW LR 1e-4, weight decay 0.01, batch 256, warmup 1, cosine
decay, 12 maximum epochs and patience 4. Slot loss remains 0.5; the real-source
target intent-loss mass remains 0.30. Existing deterministic trainer checkpoints,
best raw validation E2E selection, clean training diagnostic and per-source,
speaker and class metrics are retained. Calibration uses val and tune-OOS only.

Qualification compares against both matched control and frozen parent: real
validation intent improves >=2 points, or aggregate validation E2E improves
>=0.5 point without more than one real-validation example regression. In either
case, E2E and slots must not drop >0.5 point. The calibrated policy must meet
<=5% tune-OOS acceptance and lose <=1 point in accepted/correct validation intents
versus the best feasible control/parent policy. Criteria are fixed before runs.

Selection is sealed before any test predictions. Only a qualified waveform
trial receives paired control/candidate known-test and challenge evaluations,
calibrated evaluation, ONNX export/parity and host latency. Otherwise stop and
report a plateau without using test to choose the next recipe. Holdout is never
evaluated. No automatic Pi promotion or further runs.

## Scheduling and Limits

The persistent controller waits for the existing wakeword queue to complete
normally, then trains these two runs serially. All GPUs are checked every 10
seconds; one must be <40% utilized with >=20,000 MiB free, confirmed after 5
seconds with a CUDA probe. This avoids competing wakeword/VCM training during
contention. Logs/checkpoints persist independently of chat; normal completion
or error is visible in queue.log. A dependency failure deliberately blocks
training rather than bypassing validation.

```bash
tail -f work/vcm_wavevariants_v1/queue.log
tail -f work/vcm_wavevariants_v1/control.log
tail -f work/vcm_wavevariants_v1/wave.log
```

This one-seed screen does not establish statistical significance or Pi noise
robustness. Real validation has only 158 clips. A clean-validation plateau does
not prove augmentation cannot help noisy microphones. Generated noise/RIR is
not measured device playback or a substitute for representative real speech.
If promising, follow with a second seed and actual Pi latency/noise/music tests;
do not continue tuning against the repeatedly inspected frozen test.
