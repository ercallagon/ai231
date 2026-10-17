# Frozen Model Selection and Calibration

The repository default is **Champion (6s BC8-TCN-ASP)**, 488,264 learned
parameters, with frozen confidence threshold **0.954**. **Baseline (2s BC8)**,
327,750 parameters (historical run_043), remains selectable at **0.875**.
[RESULTS.md](RESULTS.md) contains the current injected comparison;
[the Pi report](PI5_ACOUSTIC_CHAMPION_20261017.md) covers Champion-only acoustic
measurements. Wake exp15 was selected separately.

## Training and Checkpoint Selection

The candidate trained from scratch with seed 42, stopping at epoch 72 under
patience 20 and selecting epoch 52. Checkpoints were ranked by 0.70 times full
validation intent accuracy plus 0.30 times real-voice validation intent accuracy.

On the same 2,922 validation rows, candidate intent accuracy was 96.4750%
(overall) and 82.9114% (real), versus Baseline 96.0301% and 74.0506%.
Capacity, preprocessing, training lineage and head definitions differ; this is
a frozen-system comparison rather than a controlled layer ablation.

## Original Policy and Subsequent Calibration

The original selection worker retained Baseline before the 2026-10-15 frozen
evaluation. The candidate's initial 0.95 policy failed the predeclared processed
OOS calibration limit: 16/284 false accepts (5.63%) against a 5% point-estimate
target. Its operational rate was 16/400 (4%), because 116 recordings were
rejected by capacity/preprocessing. Those rejects are not learned discrimination.
Its validation accepted-correct supported count was 2,576/2,819, versus
Baseline 2,519/2,819; real counts were 91/124 versus 76/124.

Test and holdout had already been examined in the project. After that exposure,
the calibration search was extended beyond its original 0.95 upper bound using
only validation and the existing tune-OOS pool. The checkpoint, features and
temperature remained fixed. No test/holdout predictions were used as calibration
inputs. This chronology does not make subsequent evaluations newly unseen.

The revised threshold is 0.954; temperature 0.6499874044563441 is baked into the
unchanged Champion ONNX exactly once. On the same 284 processed negatives,
Baseline accepts 15 (5.28%), the initial candidate policy 16 (5.63%) and the
revised policy 13 (4.58%). On the full 400-row pool, the revised policy accepts
13 (3.25%), with the 116 processing rejects separately reported.

The revised validation accepted-correct supported count is 2,572/2,819 versus
Baseline 2,519/2,819; real counts are 90/124 versus 76/124. Relative to the 0.95
policy, four correct accepts are lost, including one real example. Wrong
supported accepts remain 14; unsupported-temperature accepts are two. No family
with at least 20 examples loses more than five percentage points against
Baseline. The revised policy satisfies the versioned offline nomination rules;
the 5% development target is not an independent statistical guarantee.

## Current Evaluation Scope

Checkpoint, preprocessing and rejection settings were frozen before each
reported evaluation. Current test/holdout results at 0.954 retain all processing
failures in the full denominators and distinguish real/synthetic, supported
intent-plus-slot success, OOS acceptance and conditional processed rates.
Historical 0.95 results are preserved in the
[original selection record](https://github.com/ercallagon/ai231/blob/9de6bb4eb0bf309b1166969748875e3d78cb49e7/me2-vcm/reports/MODEL_SELECTION.md)
and immutable evaluation evidence; they are not current-policy scores.

The two-input mel/mask runtime, CPU ONNX parity and mocked route checks are
implemented. Pi 5 acoustic evidence supports Champion under the measured
protocol: 170/186 supported successes, with 2/16 OOS false accepts. No acoustic
Baseline run was performed. Complete-word retention has not been manually
annotated; call recovery, independent speakers and noisy/music/AEC robustness
have not been established. The 800 ms endpoint is measured device policy;
1200 ms remains available for longer pauses. Further model or gate development
requires a separately scoped experiment and independent confirmation.
