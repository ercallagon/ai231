# Publication and Evidence Status

Project: [ME2: Voice Controlled Smart Device](https://github.com/ercallagon/ai231/tree/main/me2-vcm).
Frozen weights are published as
[me2-weights-v1.0.0](https://github.com/ercallagon/ai231/releases/tag/me2-weights-v1.0.0),
whose tag remains at `9de6bb4eb0bf309b1166969748875e3d78cb49e7`.
[Release verification](WEIGHTS_RELEASE_20261017.json) records archive and payload
hashes. Use [MODEL_TERMS.md](../MODEL_TERMS.md) for weight-use terms. The containing
Git revision identifies the runtime and documentation.

## Frozen Artifacts

Champion is the scratch six-second BC8-TCN-ASP model at threshold 0.954.
Baseline is the complete two-second BC8 fallback at 0.875 (historical run_043).
Exp15 remains the wake default. Its recovered checkpoint, original log,
epoch-42 correction and ONNX parity are preserved. Later wake trials did not
justify replacement. Models, calibration, labels, manifests, checkpoints and
original training logs are unchanged by documentation cleanup.

The current public launcher's endpoint is 800 ms. The frozen model configuration
retains the historical 1200 ms capture entry; an explicit 1200 ms CLI override
provides the longer pause tolerance. Champion acoustic measurements were made
with an explicit 800 ms setting. Host launcher regressions verify default and
explicit override selection; the recorded acoustic measurements retain their
original protocol and scope.

## Verification Records

[VERIFICATION.md](VERIFICATION.md) describes completed host and device checks.
[PI5_ACOUSTIC_CHAMPION_20261017.md](PI5_ACOUSTIC_CHAMPION_20261017.md) records
Champion-only acoustic recognition, timing, resource usage and limitations.
Baseline/Champion comparisons use frozen injected test data.

CHECKS.json, FINAL_CHECKS.json and PUBLICATION_CHECKS.json are immutable
historical execution receipts. Their original dependency/authentication states
and test counts describe their own runs. They are not current publication
status. Updated review receipts remain under ignored work/ until reviewed.

Published historical logs replace private host prefixes with PROJECT_ROOT or
HPC_HOME. [PROVENANCE_REDACTIONS.json](PROVENANCE_REDACTIONS.json) ties published
and archived hashes; original local artifacts remain intact. The original
exp15 configuration's best_epoch 60 is preserved with an evidence-based
correction to epoch 42.

## Evidence Limits

[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) centralizes the outstanding
rights/access, independent-speaker, wake-data and cluster metadata evidence.
Test/holdout were repeatedly exposed; official wake positives are synthetic.
Acoustic results do not verify call recovery, music/AEC robustness or fresh-OS
installation. The source-code MIT licence does not grant underlying data,
voice or model permissions.
