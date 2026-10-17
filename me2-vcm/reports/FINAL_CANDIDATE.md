# Submission Candidate

Champion is the frozen six-second BC8-TCN-ASP model, with 488,264 learned
parameters and confidence threshold 0.954. Baseline is two-second BC8
(historical run_043), 327,750 parameters and threshold 0.875. Exp15 CNN-GRU,
204,738 parameters and threshold 0.30, remains the wake model.
[MODEL_SELECTION.md](MODEL_SELECTION.md) records selection and calibration;
[RESULTS.md](RESULTS.md) contains the frozen injected comparison.

## Measured Pi 5 Results

The [Champion-only acoustic report](PI5_ACOUSTIC_CHAMPION_20261017.md) records
the speaker-to-microphone benchmark. Baseline was compared on frozen injected
test data; no acoustic Baseline comparison was performed.

| Metric | Champion acoustic result |
| --- | --- |
| Intent accuracy | 184/202 (91.1%) |
| Accepted-correct supported intent + slot | 170/186 (91.4%) |
| Real / synthetic supported success | 74/86 (86.0%) / 96/100 (96.0%) |
| OOS false accepts | 2/16; real 0/10, synthetic 2/6 |
| No-wake controls | 4/16 triggered wake; all rejected before action |
| Processing/no-response | One zero-audio decision; seven no-response wake-present trials |
| Feature preparation + inference p95 | 116.5 ms, n=194 processed decisions |
| Actual-audio RTF p95 | 0.1475 |
| Detector-estimated speech-end to decision p95 | 926.7 ms, n=194; includes endpoint wait, excludes services/TTS |
| Runtime | Pi 5, 4 GB; ONNX Runtime CPU; one intra-op/inter-op thread; 800 ms endpoint |
| Resources | Peak RSS 201.7 MB; mean CPU 61.3%; peak temperature 51.2 C; no throttling observed |

The detector endpoint uses the last voiced frame read time, not annotated
acoustic ground truth. Nine external laptop-observed timings were negative,
so the external 1.776 s p95 is not a valid command-end latency estimate.
Four trial intervals had multiple decisions; the official scorer uses the
first decision. No failed or early trials were removed from official scoring.

The 2026-10-17 run used two screened wake takes, a two-second wake-to-command
gap and 29.3 dB SNR. It does not establish unbiased wake recall, false wakes/hour,
independent-speaker generalization or noise/music/AEC robustness. The export
lacks contemporaneous model/config hashes and Git revision, so exact loaded-byte
identity is not cryptographically established by that run. Source file hashes
are retained in the acoustic report.

## Implementation and Verification

The two-input mel/mask adapter, retained-input capacity guard, exact slot
translation and fail-closed OOS/unsupported-value routing are implemented.
Temperature scaling is baked into each ONNX once. The selected six-second
capacity is not a fixed listening time. Complete-word retention has not been
manually annotated.

The profile launcher supplies 800 ms endpoint silence instead of forwarding the
frozen YAML's historical 1200 ms value. Explicit 1200 ms remains available for
longer pauses. Host regressions verify the public launcher path and overrides
for both profiles. Models, gates and data are unchanged. The recorded acoustic
run supplied 800 ms explicitly and retains its original measurement scope.

[FINAL_CHECKS.json](FINAL_CHECKS.json) preserves the earlier 258-pass/four-skip
host receipt. [VERIFICATION.md](VERIFICATION.md) describes checkpoint/ONNX
parity, runtime contracts and operator-confirmed GPIO/audio evidence. New local
review receipts do not replace these immutable historical records. Host checks
use mocked peripherals/services and establish no successful real call/SMS.

## Evidence and Limits

The [submission evidence checklist](SUBMISSION_CHECKLIST.md) centralizes source
rights/consent, independent-speaker, public wake-data and cluster metadata gaps.
The complete injected comparison retains 16 Champion test processing failures
in the full denominator, real/synthetic cohorts, OOS rates and paired/family
errors. Test and holdout were repeatedly examined; later gate refinement used
validation/tune only. No fresh unseen confirmation is claimed.

Original experiment artifacts, full selected logs, checkpoints and manifests
are preserved. Earlier preparation and first-acoustic-run detail remain in
[the historical audit](https://github.com/ercallagon/ai231/blob/9de6bb4eb0bf309b1166969748875e3d78cb49e7/me2-vcm/reports/FINAL_CANDIDATE.md).
The comparable-model evidence is a system comparison, not an isolated
architecture ablation. Call-end recovery and fresh-OS installation are unverified.
