# Project conventions

Canonical source lives under src/hey_swift; work/ is ignored staging space.
Preserve original experiments. The repository default is the frozen six-second
BC8/TCN/attentive-pooling model at threshold 0.954; run043 is a selectable fallback
at 0.875. Temperature is baked into each ONNX once. Do not apply it again.

No external pretrained models initialize either detector. Dataset and weight
terms are separate from the owned-code MIT licence. Never commit recordings,
credentials, account state, personal contact defaults or downloaded songs/voices.

Pi injection, a Champion-only Pi 5 acoustic benchmark at the 800 ms endpoint,
and operator GPIO/audio checks exist. The comparable-size Baseline is compared
on frozen injected test data, not acoustically. Call recovery and fresh
independent-speaker verification remain pending. Host tests are not hardware
verification. Follow docs/FINAL_PI_VALIDATION.md and the acoustic result report.
After final measurement, change evidence/reports/deck only unless a material
issue is documented together with the required implementation retest.
Use mock services for checks; never place calls/SMS without explicit approval.
Keep test/holdout exposure disclosed. Do not fit a gate or model using those sets.
One GPU at a time for optional scratch retraining; no SLURM.
