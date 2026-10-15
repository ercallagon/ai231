# Curated experiment evidence

The six-second `bc8_fullcontext_tcn_asp_s42` checkpoint at threshold 0.954 is
the repository default. Pi 5 verification is pending. The complete run_043
fallback is selectable. Source and measured reports, rather than the historical
draft descriptions below, define the current release.

`RUN_INDEX.csv` records 73 discovered run directories: 43 in the primary VCM
archive, three in the primary wakeword archive, and 27 additional trial
directories with checkpoints or result records. Names do not establish successful
completion: unknown completion, missing checkpoints and incompatible protocols are
explicitly marked. Other checkpoint naming conventions still require follow-up.

All original directories remain intact. The inventory contains no weights and
does not imply that a local archive reference is a public download URL. Exact
artifact evidence remains in the private preparation receipt until reviewed.

Retained sets:

- Final VCM: checkpoint, complete training logs, source/config, environment,
  manifests/provenance, calibration, evaluations, ONNX parity and Pi measurements.
- Comparable baseline: run_043 (327,750 parameters), subject to protocol and
  device comparison verification. The current candidate has 488,264 parameters.
- New candidate: retained as comparison evidence if not selected; otherwise
  referenced from the selected set without duplicating binaries.
- Wakeword: independent selection; exp15 is the preserved reference, with the
  previously staged gender-diverse candidate still requiring a selection record.

Earlier experiments span dataset revisions, slot-label fixes, different data
eligibility rules and calibration policies. Their headline scores must not be
ranked as one controlled experiment. Unknown fields remain unknown until backed
by an actual config, log or result artifact. Hashes of active checkpoints are
deferred until training finishes.
