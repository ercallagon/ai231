# Final host-only wakeword comparison

Date: 2026-10-15. exp15 remains the published/default model. The candidate is
the frozen epoch31 random-init WakeGRU from this experiment, tau0.03. exp15
uses its unchanged deployed tau0.30. No weights, production code, dataset
splits, calibration or deployment were changed during these checks.

## Offline test: 1113 recordings

385 positives are TTS, 728 are negatives. This is previously exposed evaluation,
not a newly unseen holdout. There is no independently verified wake holdout.

| Metric | exp15 historical | exp15 waveform1s | candidate aligned2s |
|---|---:|---:|---:|
| Accuracy | 85.53% | 86.25% | 87.60% |
| Precision | 90.00% | 89.19% | 94.27% |
| Recall | 65.45% | 68.57% | 68.31% |
| F1 | 75.79% | 77.53% | 79.22% |
| False positive clips | 28/728 | 32/728 | 16/728 |
| Missed positive clips | 133/385 | 121/385 | 122/385 |

The gain is mainly precision/rejection, not a large positive-recall gain.
Paired accuracy: candidate wins84, loses61 vs historical exp15; exact two-sided
McNemar p0.06733. A single seed and exposed synthetic-positive test do not
establish a robust real-person improvement.

All rows remain in denominators. Boundaries are label-blind at evaluation;
no-speech -> non-detection and over-capacity -> an explicitly reported
maximum-energy continuous window for ALL labels. This is window recognition,
not proof of complete words; see test_clip.json for boundary flags.
The archived validation label-conditional failure policy was audited; a fresh
label-blind validation view reproduces273TP/15FP, with no refitting.

## Causal direct-window host replay of test

Uses the actual production process method, four overlapping windows,250ms hop,
sustain2, cooldown2s and RMS/nonwake gate. Candidate windows are2s; champion1s.
Each recording resets, with2s silent lead-in and3s tail; a recording is positive
if ANY event occurs. It is not verified temporal phrase localization or FA/hour.

| Metric | exp15 | candidate direct2s |
|---|---:|---:|
| Any-event accuracy | 76.01% | 77.54% |
| Precision | 60.57% | 61.86% |
| Positive clips with event | 338/385 (87.79%) | 352/385 (91.43%) |
| F1 | 71.69% | 73.79% |
| Negative clips with event | 220/728 (30.22%) | 217/728 (29.81%) |

The candidate offline tau0.03 does NOT bypass the unchanged live nonwake<0.90
gate: effective streaming wake probability must exceed0.10. No gate was
weakened or fitted using these results.

## Supplied development recordings and frontend consistency

Direct rolling2s input differs from the energy-positioned training frontend.
A separate SUPPLEMENTARY causal energy-positioning adapter was checked on
validation/development ONLY, after the direct-window evaluation. It only uses
the already captured2s buffer; no future samples, labels, new weights or gate
changes. This adapter has NOT been evaluated on test or selected/promoted.

| Development condition/path | exp15 | candidate direct2s | candidate causal-aligned2s |
|---|---:|---:|---:|
| Quiet raw, two recordings combined | 3 events | 0 | 5 |
| Quiet AEC, same recordings | 1 event | 0 | 2 |
| Speech-over-music raw, two recordings | 1 event | 1 | 4 |
| Speech-over-music AEC, same recordings | 1 event | 0 | 2 |
| Music-only raw, two recordings (~60s) | 0 events | 0 | 0 |
| Music-only AEC, same recordings (~60s) | 0 events | 0 | 2 |

These are EVENT COUNTS, not true positive recall: speech timing and utterance
identity are not manually confirmed, duplicates/events may occur, and AEC/raw
are correlated paths of the same recordings. The aligned adapter restores
speech events but produces one false event in each AEC music-only recording.
Its validation any-event recall293/326=89.88%, negative clip FPR162/726=22.31%.
Do not compare that FPR directly with isolated clip-classification FPR.

## Tone stress, parity and cost

18 synthetic ten-second cases, six sine frequencies x three durations:
exp15 triggered33 events in11 cases; direct2s candidate29 events in9 cases.
This does not establish rejection of real ring/hangup audio; recordings of
those specific failures remain unavailable. No calls/SMS were initiated.

Both have204738 parameters. ONNX bytes: exp15 825493, candidate825524.
One intra-op and one inter-op thread;100 measured calls after20 warmups:
host features+inference p95 2.241ms vs4.524ms; window-duration RTFp95 .00224
vs .00226. These are NOT Pi latency or total multiwindow hop cost.
Candidate checkpoint/ONNX parity passed batches1/2; silence and gate checks pass.

FAILED check: chunk-size invariance on positive training audio shifts a
candidate cooldown-boundary event by250ms. Shared production process accumulates
floating seconds; near exact2s cooldown, different addition order changes the
qualifying hop. A model-free constant-score probe also reproduces the issue
(1600-sample chunks events2.25/4.50/6.75/9.00s vs1733-sample chunks
2.25/4.50/6.50/8.50s). Original runtime was NOT edited. Eight-clip focused
baseline checks passed, while the shared-code synthetic probe demonstrates
that the root cause is not unique to the learned candidate.

## Recommendation

Do NOT promote. Keep exp15 and the initial GitHub commit unchanged.
The new model modestly improves offline precision and simulated any-event
recognition, but its actual streaming gate is not calibrated consistently,
music/tone rejection remains weak and real/AEC development results are mixed.
The training-only alignment hypothesis remains worth investigating; successful
offline crops alone do not justify replacing a working live model.

Next, if authorized: fix the sample-clock cooldown issue with regressions and
validate/calibrate the causally matched frontend on training/validation streams
including diverse hard music/beep negatives. Do not tune against the test
metrics above. Independent real-speaker and Pi checks remain necessary.

## Evidence

- evaluation_v1/REPORT.md: full frozen comparison and per-recording counts.
- evaluation_v1/{validation,test}_clip.json: metrics, cohorts, failure counts.
- evaluation_v1/test_stream.json: all causal test events.
- evaluation_v1/development_replay.json and synthetic_beeps.json.
- evaluation_v1/runtime_checks.json: failed/passing checks and measured cost.
- evaluation_v1/EVALUATION_FREEZE.json and COMPLETE.json (passed:false
  deliberately preserves the runtime failure; evaluation_complete:true).
- causal_alignment_check_v1/: supplementary validation/development-only checks.
- evaluate_frozen.py, finish_frozen_checks.py, check_causal_alignment.py and
  summarize_frozen.py preserve the exact procedures. The original evaluation
  stopped at its failing assertion; finish_frozen_checks.py completed independent
  checks without suppressing or relabeling that failure.
