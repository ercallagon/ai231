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

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B work/wake_corrected_test_20261013_v1/evaluate.py
```

The script refuses an already sealed output directory. For reruns, copy this versioned script into a fresh sibling directory under work/.
PRETEST_SEAL.json, TEST_ROWS.json, predictions, feature arrays and RESULTS.json preserve the evaluation. No thresholds are fitted here.
