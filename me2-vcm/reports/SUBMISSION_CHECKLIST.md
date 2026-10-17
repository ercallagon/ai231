# Submission Evidence

The six submission criteria below are mapped to verified repository evidence.
Raspberry Pi 5 (4 GB) is the documented instructor-approved substitute for Pi 4.

| Criterion | Evidence | Scope or limitation |
| --- | --- | --- |
| Public MIT repository; one-command reproduction | [README](../README.md), [LICENSE](../LICENSE), run.sh/run.ps1, scripts/reproduce.py; clean-environment VCM parity receipts | MIT covers owned source. Full public wake retraining is unavailable because raw-recording access/consent is incomplete. |
| Licensed and citable dataset | [PROVENANCE.json](../data/PROVENANCE.json), [DATASET_LICENSE.md](../data/DATASET_LICENSE.md); DOI 10.57967/hf/10723 | Source terms differ. Cloned-voice permission, source licensing and some speaker consent records are incomplete; a research-use label does not supply missing permission. |
| Training logs and final checkpoint committed | experiments/selected/checkpoint.pt, logs/training/selected.jsonl, config/source/environment/accounting; Baseline checkpoint/logs; exact exp15 checkpoint/log/parity | Raw recordings are not published. Checkpoint availability does not establish dataset rights or full wake-data reproduction. |
| Pi latency reproduced by posted script | scripts/benchmark_onnx_pi.py, docs/PI5.md and [Champion acoustic report](PI5_ACOUSTIC_CHAMPION_20261017.md) | Pi 5 measured feature+inference p95 is 116.5 ms; detector-estimated decision p95 is 926.7 ms. Acoustic comparison is Champion-only. |
| Held-out test and unseen speakers | Frozen manifests, encoded/PCM hashes, grouping and exposure records | Test/holdout were repeatedly exposed; fresh independent-speaker confirmation has not been collected. Declared groups are not verified humans; official wake positives are synthetic. |
| Comparable-size Baseline compared | 327,750 versus 488,264 parameters; same frozen injected test, cohort/family/paired errors, selectable profiles | Size ratio is about 1.49. Capacity, preprocessing and lineage differ, so this is a system comparison. No acoustic Baseline run exists. |

## Model and Training Fields

| Field | Verified value |
| --- | --- |
| Architecture | Scratch BC8 backbone; 12 broadcast residual blocks; three noncausal 128-channel depthwise temporal blocks, kernel 5, dilations 1/3/9; masked attentive mean/std pooling; 256-dimensional embedding; intent and slot heads. |
| Audio/features | Mono 16 kHz; FFT 400, hop 160; 80 x 600 log-mel plus 600-frame validity mask; maximum six-second retained input. |
| Parameters/weights | Champion 488,264 learned parameters / 2,136,900-byte trained ONNX; Baseline 327,750 / 1,488,732 bytes; wake exp15 204,738 / 825,493 bytes. Champion checkpoint 6,138,121 bytes. |
| Outputs | 19 supported intents plus OOS; six slot heads, three supported values each; Champion TEMPERATURE adds REJECT. |
| Objective | Weighted intent cross-entropy, label smoothing 0.05, plus 0.75 applicable slot cross-entropy; gradient clip 5. |
| Optimizer | AdamW initial LR 0.0003, weight decay 0.01; ReduceLROnPlateau factor 0.5/patience 6; batch 32. |
| Compute/selection | One A100-SXM4-40GB, seed 42; 72 epochs, selected epoch 52; 3.64645 allocated GPU-hours from logged epoch durations, including validation/host overhead. |
| Updates/loss | Derived 792 updates/epoch, 57,024 total, 41,184 through selected epoch; selected-epoch augmented mean objective 0.4076313. |
| Runtime | ONNX Runtime CPU; one intra-op and one inter-op thread; 800 ms endpoint, explicit 1200 ms override for longer pauses. |
| Attribution | Project author metadata names Edgardo Callagon Jr.; no additional team identity is asserted. |

The model is captured-utterance classification, not a causal encoder, KV-cache
or autoregressive/CTC decoder. Architecture is documented in
[architecture.md](../docs/architecture.md); update counts are derived, not a saved
optimizer counter. The external profiler's 499,784 count differs from the
488,264 learned parameters and is reported separately.

## Dataset Accounting

Pinned source: airimonda/ai231-me2-voice-commands, revision
bfca0bbebfe17e00d6a685d1da573615d29ecdce.
[PROVENANCE.json](../data/PROVENANCE.json) records manifest hashes and source terms.

| Split | Manifest recordings | Source waveform hours | Real rows |
| --- | ---: | ---: | ---: |
| Train | 25,366 | 14.4008 | 1,559 |
| Validation | 2,922 | 1.8623 | 158 |
| OOS calibration | 400 | 0.5652 | 400 |
| Test | 4,693 | 2.6260 | 824 |
| Holdout | 202 | 0.1791 | 96 |

Eligible training count is 25,330 after 36 recorded exclusions. Hours describe
original waveforms, not padding or speech-only duration. Train has 559 declared
pseudonymized groups; this is not a verified human-speaker count.

The committed-manifest check found zero encoded-audio and decoded-PCM hash
intersections for every train/validation/test/holdout pair. Declared group
intersections are train/validation 17, train/test 1 and validation/test 1;
all occur on synthetic_negative rows. The shared train/test group contains
189 train, 11 validation and 50 test rows. Metadata does not establish whether
it is a source placeholder or voice identity. Hash nonoverlap does not establish
voice independence or exclude near-duplicates/shared augmentation ancestry.
Train/holdout, validation/holdout and test/holdout have no shared group IDs.

## Results and Measurement Scope

[RESULTS.md](RESULTS.md) is the canonical Baseline/Champion frozen injected
comparison, including full-manifest failures, processed subsets, real/synthetic
cohorts, per-family errors and OOS denominators. [The acoustic report](PI5_ACOUSTIC_CHAMPION_20261017.md)
records Champion-only Pi playback results: 170/186 supported successes, 74/86
real and 96/100 synthetic; 2/16 OOS false accepts; 4/16 no-wake controls triggered
and were all rejected before action.

Feature+inference p95 is 116.5 ms, actual-audio RTF p95 0.1475 and
detector-estimated speech-end-to-decision p95 926.7 ms (194 processed decisions).
The latter includes endpoint waiting and uses the last voiced frame read time,
not annotated acoustic ground truth. Nine external latency values were negative;
the external 1.776 s p95 is not a valid command-end latency estimate.

Exp15 isolated clip test: 1,113 clips, 385 synthetic positives and 728 negatives;
historical feature-crop accuracy 85.53%, precision 90.00%, recall 65.45%,
F1 75.79%, negative-clip FPR 3.85%. Supplementary waveform-first one-second
processing yields accuracy 86.25%, recall 68.57%, F1 77.53%, FPR 4.40%.
These are distinct preprocessing views of unchanged weights, not Pi streaming
recall or false triggers/hour.

## Published Weights

[me2-weights-v1.0.0](https://github.com/ercallagon/ai231/releases/tag/me2-weights-v1.0.0)
publishes the frozen ONNX exports, checkpoints, labels/configuration, checksums,
terms and benchmark summaries from source commit
`9de6bb4eb0bf309b1166969748875e3d78cb49e7`.
The 14,041,136-byte archive and all 17 listed payload files match their recorded
SHA-256 values and tagged source. Frozen model/configuration/label/checkpoint
files also match the current tree. The release retains its original results
document; current results wording has changed without changing frozen evidence.
[WEIGHTS_RELEASE_20261017.json](WEIGHTS_RELEASE_20261017.json) records verification.
[MODEL_TERMS.md](../MODEL_TERMS.md) governs the weights separately from code MIT.

## Interpretation and Access Terms

Results characterize the recorded evaluation conditions and previously exposed
sets; they do not establish independent unseen-speaker performance. The public
reproduction command evaluates the frozen VCM models. Full wake retraining is
outside that public recipe because recording access and consent documentation
are incomplete. Source-specific voice/data permissions are described in
[DATASET_LICENSE.md](../data/DATASET_LICENSE.md); public availability supplies no
additional permission. [MODEL_TERMS.md](../MODEL_TERMS.md) applies separately to
weights. The [acoustic report](PI5_ACOUSTIC_CHAMPION_20261017.md) defines timing,
provenance and device scope. Historical receipts retain their original results.
