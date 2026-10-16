# Frozen threshold 0.954: measured results

The six-second model is the repository default by the user's explicit decision after validation-only gate refinement. This report covers frozen injected VCM evaluation; the separate Champion Pi 5 acoustic results are in PI5_ACOUSTIC_CHAMPION_20261017.md. Both test and holdout were previously exposed during research; no untouched-set or independent-speaker claim is made. No weights or gates were fitted during this evaluation.

| Split/cohort | run_043 intent | Selected intent | run_043 accepted intent+slot | Selected accepted intent+slot |
| --- | ---: | ---: | ---: | ---: |
| val/all | 2806/2922 (96.03%) | 2819/2922 (96.48%) | 2519/2819 (89.36%) | 2572/2819 (91.24%) |
| val/real | 117/158 (74.05%) | 131/158 (82.91%) | 76/124 (61.29%) | 90/124 (72.58%) |
| val/synthetic | 2689/2764 (97.29%) | 2688/2764 (97.25%) | 2443/2695 (90.65%) | 2482/2695 (92.10%) |
| test/all | 4349/4693 (92.67%) | 4466/4693 (95.16%) | 3922/4367 (89.81%) | 4046/4367 (92.65%) |
| test/real | 517/824 (62.74%) | 649/824 (78.76%) | 369/777 (47.49%) | 508/777 (65.38%) |
| test/synthetic | 3832/3869 (99.04%) | 3817/3869 (98.66%) | 3553/3590 (98.97%) | 3538/3590 (98.55%) |
| holdout/all | 129/202 (63.86%) | 191/202 (94.55%) | 110/186 (59.14%) | 178/186 (95.70%) |
| holdout/real | 24/96 (25.00%) | 87/96 (90.62%) | 11/86 (12.79%) | 78/86 (90.70%) |
| holdout/synthetic | 105/106 (99.06%) | 104/106 (98.11%) | 99/100 (99.00%) | 100/100 (100.00%) |
| tune_oos/all | 231/400 (57.75%) | 175/400 (43.75%) | 0/0 (not applicable) | 0/0 (not applicable) |
| tune_oos/real | 231/400 (57.75%) | 175/400 (43.75%) | 0/0 (not applicable) | 0/0 (not applicable) |
| tune_oos/synthetic | 0/0 (not applicable) | 0/0 (not applicable) | 0/0 (not applicable) | 0/0 (not applicable) |

## Rejection and full-manifest coverage

| Split/model | Processed / manifest | Failures | OOS false accepts / full OOS | Processed OOS false accepts / processed OOS | Wrong supported accepts | Valid-command false rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val/run043 | 2922/2922 | 0 | 2/76 (2.63%) | 2/76 | 42 | 258 |
| val/selected | 2908/2922 | 14 | 1/76 (1.32%) | 1/73 | 14 | 233 |
| test/run043 | 4693/4693 | 0 | 10/326 (3.07%) | 10/326 | 41 | 404 |
| test/selected | 4677/4693 | 16 | 14/326 (4.29%) | 14/318 | 41 | 280 |
| holdout/run043 | 202/202 | 0 | 0/16 (0.00%) | 0/16 | 2 | 74 |
| holdout/selected | 202/202 | 0 | 1/16 (6.25%) | 1/16 | 0 | 8 |
| tune_oos/run043 | 400/400 | 0 | 20/400 (5.00%) | 20/400 | 0 | 0 |
| tune_oos/selected | 284/400 | 116 | 13/400 (3.25%) | 13/284 | 0 | 0 |

False rejects include processing failures for supported commands; a processed-subset accuracy must never replace the full denominator. Unknown/unsupported slot supervision and rejection denominators are explicit in the machine-readable reports. Per-family errors/confusions and real/synthetic processed-subset counts are retained in reports/evaluation/*.json and families.csv.

On the identical 284 processed calibration negatives, run_043 accepts 15 (5.28%) and the revised candidate accepts 13 (4.58%). Its full-pool rate is 13/400 (3.25%); 116 input-capacity/preprocessing rejects are not learned OOS discrimination. A 5% development point estimate is not a guaranteed real-world bound.

## Efficiency and retention

Selected: 488,264 parameters, trained ONNX 2,136,900 bytes; baseline: 327,750 parameters, ONNX 1,488,732 bytes. The earlier 2,028,874-byte figure described the untrained preflight export, not the selected trained graph. One intra-op and one inter-op CPU thread are configured. Per-split reports contain host feature+ONNX p95 and actual-duration RTF; these are not Pi measurements. Model-only timing uses a separate script and nominal capacity. Capture, decoding, services and speech are excluded from injected timing.

The selected frontend retains the continuous input waveform up to 96,000 samples and includes its final partial frame. Over-capacity, flat and invalid inputs reject explicitly. This is waveform retention, not manually verified complete-word retention. Live capture uses pre-roll and the measured 800 ms endpoint default; `--end-silence-ms 1200` is an explicit longer-pause override. See the separate acoustic report for Pi microphone/latency scope. The baseline uses its historical two-second center-feature crop. Duration cohorts and all failures are reported rather than silently dropped.

## Limits and independent confirmation

Single training seed; repeated test/holdout exposure; gate refinement after that exposure used validation/tune only. Synthetic accuracy alone hides real-domain errors. Holdout gains are on a small, concentrated speaker set and do not establish population robustness. No fresh independent recordings are available. Reserve consented speakers absent from every development source, freeze this release before collection, and report every recording without tuning.

Wakeword exp15 remains a separate artifact. Corrected clip-test metrics use TTS positives and do not establish real-human wake recall or music/AEC robustness. See experiments/wakeword/ for exact source and metric scope.
