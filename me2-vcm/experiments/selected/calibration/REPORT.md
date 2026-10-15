# Six-second global-threshold refinement

Passes validation-only offline promotion criteria; Pi/runtime verification pending

Fixed checkpoint and temperature 0.6499874044563441; threshold 0.950 -> 0.954.
Calibration OOS 13/284 processed (4.58%), 13/400 operational (3.25%).
run_043 on identical 284 clips: 15/284 (5.28%); full pool 20/400.
Validation supported accepted-correct 2572/2819; real 90/124; wrong supported 14; unsupported temperature accepts 2.
Cost versus original candidate: 4 fewer supported correct accepts, including 1 real.
ONNX decision/logit parity checked on 12 validation rows; original model, policy, checkpoint and protected artifacts unchanged.

Only validation and existing tune-OOS predictions used. This is a versioned development calibration, with prior test/holdout exposure disclosed. No reserved predictions were read or recomputed. The original selection record is preserved; a revised offline nomination does not establish Pi readiness. Actual six-second runtime/mask handling, causal endpointing, physical Pi latency and live robustness remain pending.

The 5% criterion is a point-estimate development guardrail, not a statistical guarantee of real-world FAR. Reusing the same calibration pool introduces selection optimism. Independent negative recordings remain needed for a stronger claim.
