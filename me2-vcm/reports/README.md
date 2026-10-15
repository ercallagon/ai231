# ME2 evidence and current status

- [RESULTS.md](RESULTS.md): frozen six-second VCM at 0.954 versus run_043 at
  0.875, including test/holdout real/synthetic cohorts and full failure counts.
- [MODEL_SELECTION.md](MODEL_SELECTION.md): validation selection, calibration
  history and distinction between repository selection and Pi promotion.
- [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md): two-slide requirements,
  available values, provenance limitations and outstanding Pi measurements.
- [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md): current publication scope;
  older receipts are historical, not the present authentication/status check.
- [PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json): current staged-export
  regression and clean inference-only verification receipt.
- [VERIFICATION.md](VERIFICATION.md): host checks and pending hardware tests.
- [WAKE_EXP15_STREAMFT.md](WAKE_EXP15_STREAMFT.md): final short fine-tune;
  no replacement justified, exp15 retained. Earlier scratch frontend trials are
  under experiments/wakeword/final_streaming_v1/.

Machine-readable VCM evaluation is under evaluation/. Corrected exp15 isolated
clip metrics and recovered original checkpoint provenance are under
experiments/wakeword/. Wake event recall, negative-clip FPR and false events/hour
are different metrics. Official wake positives are synthetic; exposed evaluation
is not independent human-speaker confirmation. Pi recordings are not published.

Historical 20261013 comparisons remain archived and do not override current
defaults. Physical Pi 5 validation of this release is still pending.
