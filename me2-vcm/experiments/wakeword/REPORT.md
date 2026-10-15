# Corrected Wakeword Test Checkpoint

Preserved exp15; threshold 0.30; CPU-only; no tuning, training or deployment.

Corrected test: 1113 clips, 385 synthetic TTS wake positives and 728 negatives.
Original test had 1143 clips; 30 duplicate wake clips were excluded. No raw data or original split was modified.

| Fixed preprocessing | Accuracy | Precision | Recall | F1 | Negative clip FPR |
|---|---:|---:|---:|---:|---:|
| historical_clip | 85.53% | 90.00% | 65.45% | 75.79% | 3.85% |
| one_second_window | 86.25% | 89.19% | 68.57% | 77.53% | 4.40% |

Primary historical_clip uses shared features on full PCM, then feature crop/pad. It reproduces every existing corrected clean cache row exactly.
Supplementary one_second_window crops/pads the waveform before shared features, matching the recent validation audit recipe.
Both recipes were sealed before inference; neither was selected based on the test result. Differences are preprocessing differences, not model improvements.

These are isolated clip-classification metrics, not complete streaming sustain/cooldown/RMS-gated wake detection.
Clip FPR is not false activations per hour. Live recall, latency, music/AEC robustness and real-speaker positive generalization remain pending.
Known PCM duplication was removed; near-duplicates and complete archived training lineage remain uncertain. Historical test exposure is disclosed; this is not a newly unseen final holdout.

## Reproduction

The original evaluation script and private recording/cache inputs remain in
the preserved local audit, not in a clean public checkout. The committed
corrected_test.json contains the resulting clip counts/metrics. Do not describe
that archived private workflow as publicly reproducible in one command.

The original exp15 Torch checkpoint has since been recovered under
exp15_original/, with parity against the unchanged deployed graph, original
training log and an epoch-42 selection correction. This recovery does not resolve
raw recording access/consent or create an independent real-speaker test.
Private host prefixes in published provenance/log copies are redacted; see
reports/PROVENANCE_REDACTIONS.json for original and published hashes.

For runnable public wake-only microphone, replay and CPU checks, use
docs/WAKE_STREAMFT_AB.md and scripts/wake_streamft_ab.py. Replay needs your own
authorized recording and labeled timestamps for recall/latency; event logging
alone does not reproduce the archived clip-classification metrics. No threshold
fitting is performed by that tool. Physical Pi verification remains pending.
