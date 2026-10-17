# Submission Evidence Map

[SUBMISSION_CHECKLIST.md](../reports/SUBMISSION_CHECKLIST.md) maps the submission
criteria to source files and verified results. It records unresolved evidence
as explicit facts rather than assuming that every criterion is satisfied.

| Evidence | Canonical source |
| --- | --- |
| Model architecture and feature contracts | [architecture.md](architecture.md) |
| Training accounting and selected epoch | [TRAINING_ACCOUNTING.json](../experiments/selected/TRAINING_ACCOUNTING.json) |
| Dataset revision, split sizes and terms | [PROVENANCE.json](../data/PROVENANCE.json), [DATASET_LICENSE.md](../data/DATASET_LICENSE.md) |
| Baseline/Champion frozen comparison | [RESULTS.md](../reports/RESULTS.md) |
| Selection and calibration chronology | [MODEL_SELECTION.md](../reports/MODEL_SELECTION.md) |
| Champion-only Pi 5 acoustic results | [PI5_ACOUSTIC_CHAMPION_20261017.md](../reports/PI5_ACOUSTIC_CHAMPION_20261017.md) |
| Wake exp15 checkpoint/export | [original exp15](../experiments/wakeword/exp15_original/) |
| Hardware and measurement procedures | [hardware.md](hardware.md), [METRICS.md](METRICS.md) |
| Code/data/model rights | [LICENSE](../LICENSE), [MODEL_TERMS.md](../MODEL_TERMS.md) |
| Frozen weights download and integrity | [v1.0.0 release](https://github.com/ercallagon/ai231/releases/tag/me2-weights-v1.0.0), [verified hashes](../reports/WEIGHTS_RELEASE_20261017.json) |

The architecture is captured-utterance classification with intent/slot heads:
BC8, noncausal temporal CNN and masked attentive mean/std pooling. It uses no
causal encoder, KV cache, CTC or autoregressive decoder. Pi 5 is the documented
substitute for Pi 4. Learned parameters and the external profiler's count have
different scopes. Source metadata, not predictions, defines real/synthetic
cohorts. Frozen test/holdout exposure and the scope of each timing metric are
part of the reported evidence.
