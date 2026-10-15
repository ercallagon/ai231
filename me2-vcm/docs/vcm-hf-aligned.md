# Corrected VCM data protocol

## Status and source of truth

Protocol `hf_aligned_v1` uses HF revision
`bfca0bbebfe17e00d6a685d1da573615d29ecdce` of
`airimonda/ai231-me2-voice-commands`. The existing 18 Parquet shards (4.68 GB)
are reused, SHA-256 verified against their pinned LFS objects. WAV payloads
are embedded in Parquet; separate raw WAV downloads/extraction are unnecessary.
The raw README, variations mapping, and license are also fingerprinted.

Historical manifests/configs/checkpoints/results are not overwritten.
The original preparation algorithms remain available only for explicit legacy
reproduction into fresh outputs, not as the default corrected training path.
Wake-word data, Pi models, thresholds, tested wiring, and bundles are unchanged.

## Evidence for the correction

See `work/vcm_hf_split_audit/REPORT.md` for the earlier subset audit and
`data/vcm/protocols/hf_aligned_v1/audit.json` for the full audio/metadata audit,
including source distributions, historical checkpoint metadata and hashes.

Confirmed legacy defects:

- All 5,856 supplemental synthetic clips entered training, including 1,612
  test-associated and 261 holdout-associated clips (29 and two voices).
- Four local validation TTS voices had 204 supplemental clips in training.
- 125 original training negatives contain local validation speech in their
  declared dependencies; 119 survived the legacy filename-based deduplication.
- 51 negative train/test filenames collide, but their audio differs. Filename
  deduplication incorrectly removed valid training examples. Five current test
  cache entries match training audio rather than the correct test payload.
- Old disjointness checks excluded synthetic voices and missed composites.
  Evaluation ignored configured manifest overrides; reproduction records did
  not prove the effective override inputs.
- The validation carve is largest-group-first, not intent-stratified: six
  identities, with 487/1,171 rows from one speaker. This is a local split, not
  an HF-provided validation split.

The full decoded-audio scan found three duplicate groups: one pair of training
reversed negatives (one removed), a repeated OOS clip within the official
holdout (both retained to preserve membership), and two identical noise assets
(reserved, not classification examples). No conflicting duplicate labels or
cross-boundary byte/PCM overlaps remain in the corrected fitting/eval manifests.

Completed slot-fixed runs 031-035 recorded the same legacy training manifest
hash and 30,487 training rows. Correct numeric-slot supervision did not repair
their data boundaries. Older run attribution is based on recorded input hashes,
configs, logs and checkpoint metadata where available; current configs alone
are not proof of historical inputs. Legacy test scores remain historical
results, not clean unseen-voice estimates. The five cache substitutions alone
can affect overall accuracy by at most about 0.11 percentage points; they do
not explain the large validation/test gap by themselves.

## Split contract

| Artifact | Rows | Role |
|---|---:|---|
| train | 28,335 | Core train minus local val; eligible supplemental synth; train-only Filipino supplement; filtered training negatives |
| val | 1,171 | Preserve historical core validation membership for this first corrected comparison; isolate its identities/lineage globally |
| test_core | 4,443 | Exact official HF test membership; final evaluation |
| test_negatives | 250 | Exact HF negative test subset; separate OOS report |
| test | 4,693 | Core plus negative test; report alongside separate components |
| holdout | 202 | Exact official Pi holdout; remains unevaluated by the queue |
| tune_oos | 400 | Reserved calibration-only speech; not fitted or used by the initial fixed-threshold trial |
| supplemental_test | 1,612 | Reserved test-voice examples; not fitted/tuned |
| supplemental_holdout | 261 | Reserved separately; do not append to the official 202-row holdout |
| numerals | 66,390 | Reserved OOS pool; no injection in this controlled baseline |
| noise | 88 | Reserved augmentation sources; not classifier training rows |

The corrected eligible training pool has 28,665 rows before filtering: remove
204 validation-voice supplemental clips, 125 validation-dependent negatives,
and one exact training duplicate, giving 28,335. The 1,873 supplemental
test/holdout clips never enter that eligible pool. True source identity is
`(HF revision, config, HF split, file)`, represented by a stable `hf:` key.
Each manifest includes its Parquet row-group/offset, audio and mono decoded-PCM
SHA-256, source metadata, and speech/noise dependency identities.

All actual holdout speaker identities are reserved, including the two extra
source identities in the rows beyond the README's primary three voices.
Training negatives are filtered using every contributing speech identity,
including all babble contributors, not their pseudo speaker ID.

## Labels and metrics

The 19 intent labels plus OOS retain their locked ordering. Six slot heads use
the same three supported values each; numeric class names are strings.
The 93 variation mappings are checked against the pinned variations CSV.

HF contains 158 empty-slot `other slot value` examples: 57 in corrected train
and 101 in unchanged local val. Their authoritative intent labels are retained
and their lack of slot supervision is explicit in the manifests/audit. They
are not silently relabeled to OOS or assigned a made-up supported slot.
The headline E2E metric requires intent and every available supported slot;
these proxy rows measure intent only. This is not evidence of arbitrary-slot
support/rejection. A later separately controlled four-way slot-reject trial
is warranted before claiming unsupported-value robustness.

Evaluation now honors the same configured manifests/cache as training and
checks checkpoint protocol identity. Headline accuracy/F1/confusion/OOS metrics
are raw argmax. The additional `gated` section applies the fixed 0.50 confidence
gate and reports decision accuracy/E2E/confusion separately. Slot accuracy uses
the ground-truth intent head; E2E additionally requires the correct intent.
Thresholds and checkpoints are selected without test/holdout access.

## Prepare and validate

From the repository root in a training environment:

```bash
pip install -e '.[data,cli,train,dev]' pyarrow
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/01_download_hf.py --verify
python scripts/prepare_vcm_protocol.py --name hf_aligned_v1
python scripts/prepare_vcm_protocol.py --name hf_aligned_v1 --cache --workers 4
python scripts/validate_vcm_protocol.py --config configs/vcm_hf_aligned_v1_a10.yaml
python scripts/validate_vcm_protocol.py --config configs/vcm_hf_aligned_v1_a6.yaml
python scripts/audit_vcm_feature_overlap.py
python -m pytest
```

For a fresh checkout, omit `--verify` on the first download. Existing protocol
and cache outputs are immutable: preparation refuses to overwrite them. If they
already exist, run validation rather than rebuilding. A changed preparation or
feature implementation requires a new protocol name, configs, cache, and tags.
Intermediate inventory and exclusion evidence live under `work/vcm_hf_alignment/`.
No training waveform perturbations are applied to evaluation data. This trial
precomputes deterministic clean mels and applies fresh train-only SpecAugment
online; it deliberately does not reuse legacy augmented stores.

The cache contains 34,801 clean feature tensors, shape `(34801,1,80,198)`,
float32, all finite. Construction verifies every source payload fingerprint
and performs 373 direct-source parity checks (one per used raw row group).
Cache/index/config/protocol fingerprints are checked before training/evaluation.
An additional model-input audit checks exact tensor overlap after mel cropping
and padding; all cross-split comparisons passed (`feature_overlap.json`).

## Retraining and monitoring

The first two corrected candidates use seed 42, identical data/features,
SpecAugment, optimizer, 30-epoch budget, patience eight, and 0.50 gate:

- `run_039_hf_aligned_v1_a10_s42`: depthwise-separable residual CNN with SE,
  about 251K parameters, efficiency candidate; run first.
- `run_040_hf_aligned_v1_a6_s42`: ResNet-style 2D CNN, about 963K parameters,
  quality candidate; run second.

The persistent queue uses one GPU at a time, checks all GPUs every 30 seconds,
requires utilization below 40% and at least 10,000 MiB free, and verifies CUDA
access. Availability is a heuristic, not an exclusive resource reservation.
It checks for equivalent completed protocol/architecture/seed trials, refuses
existing output directories, and holds a filesystem lock against duplicate
queue launch. No SLURM and no interruption of other jobs.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -u scripts/run_vcm_aligned_trials.py --detach
```

The queue issues this exact training command for each selected GPU/tag/config:

```bash
CUDA_VISIBLE_DEVICES=<selected_gpu> PYTHONPATH=src \
  python -u -m hey_swift.cli train vcm \
  --tag run_039_hf_aligned_v1_a10_s42 \
  --config configs/vcm_hf_aligned_v1_a10.yaml --seed 42
```

The A6 invocation substitutes its tag and config. Per-run launch receipts record
the actual Python executable, GPU, command, seed, and input hashes. Outputs live
in fresh `experiments/vcm/<tag>/` directories, including `config.yaml`,
`reproducibility.json`, `train.log`, `best.pt`, and `result.json` with history.
The queue evaluates core test, negative test and combined test after training,
then records a host-CPU latency proxy; it never evaluates Pi holdout.

```bash
tail -f work/vcm_hf_alignment/hf_aligned_v1/queue.log
tail -f work/vcm_hf_alignment/hf_aligned_v1/run_039_hf_aligned_v1_a10_s42.log
tail -f work/vcm_hf_alignment/hf_aligned_v1/run_040_hf_aligned_v1_a6_s42.log
nvidia-smi
```

Monitoring these logs does not require keeping an agent conversation active.
Do not relaunch a second queue if the first is waiting. Check the saved queue
PID and host processes first.

## Remaining uncertainty

The finalized follow-up validation protocol and fresh BC8 retraining are
documented in `docs/vcm-grouped-v2.md`. `hf_grouped_v2b` preserves this v1
validation as an independent challenge and keeps official test/holdout unchanged.
Use its dedicated entrypoint; historical v1 training source is not modified.

The next controlled architecture experiment and completed corrected-control
comparison are documented in `docs/vcm-bc8-trial.md`. It adds a genuine
broadcasted BC-ResNet without changing this protocol or the historical controls.

- Preserving the six-speaker validation carve isolates the data correction
  from a resplit, but does not fix its concentrated distribution. A new seeded,
  group-stratified validation protocol is the next separate experiment.
- Exact PCM fingerprints and declared lineage do not prove absence of all
  undeclared transformed/near-duplicate recordings. Human review/perceptual
  duplicate search remains useful; model-ready truncation can lose distinctions.
- Metadata/variation checks do not independently transcribe every recording.
  Acoustic label correctness still depends on the pinned HF annotation process.
- Filipino cloned-voice identities cannot be independently reconciled with
  all original speakers from available metadata; provenance is recorded.
- Shared environmental-noise sources are permitted by HF's negative protocol;
  this is not a held-out acoustic-environment benchmark.
- Historical configs/sources were not all committed/versioned. Recorded hashes
  and preserved artifacts support attribution, but cannot reconstruct missing
  earlier source states. Git HEAD/status plus current implementation hashes are
  recorded; commit the corrected source/manifests/configs before submission.
  Corrected source/config/test snapshots, a package freeze and a historical
  preservation proof are saved under `work/vcm_hf_alignment/hf_aligned_v1/`.
- Seed and deterministic cuDNN settings are recorded, but mixed precision,
  worker count, hardware and algorithm selection can affect reproducibility.
  Repeat promising clean runs with another seed before asserting small gains.
- Host CPU latency is not Pi latency. ONNX export/parity and actual Pi timing
  remain required; no corrected checkpoint is promoted automatically.
