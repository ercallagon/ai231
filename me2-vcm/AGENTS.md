# Project conventions

Canonical source lives under src/hey_swift; work/ is ignored staging space.
Preserve original experiments. The repository default is the frozen six-second
BC8/TCN/attentive-pooling model at threshold 0.954; run043 is a selectable fallback
at 0.875. Temperature is baked into each ONNX once. Do not apply it again.

No external pretrained models initialize either detector. Dataset and weight
terms are separate from the owned-code MIT licence. Never commit recordings,
credentials, account state, personal contact defaults or downloaded songs/voices.

Physical Pi 5 verification is pending. Host tests are not hardware verification.
Use mock services for checks; never place calls/SMS without explicit approval.
Keep test/holdout exposure disclosed. Do not fit a gate or model using those sets.
One GPU at a time for optional scratch retraining; no SLURM.
