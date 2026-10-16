# Final exp15 Runtime-Window Fine-Tune

## Decision

retain exp15; close experimentation. Default remains exp15. The class acoustic
benchmark used pre-screened wake takes and does not establish unbiased Pi wake
recall or false triggers per hour; see the current Pi 5 acoustic report.

One fine-tune, no scratch run or architecture change. Original best checkpoint verified against deployed export; epoch42 selected historically, despite erroneous archived config best_epoch60. Provenance/parity errors are saved in PROVENANCE.json.
Training completed2epochs; best fine-tuned epoch1; initial-checkpoint improvement:False. GPU wall132.9s (0.0369GPUhours), peak reserved0.860GB.
Training eligibility:6005complete <=1s positives/7722negatives;973positive exclusions, preserved in audit, never negative labels. All evaluation utterances retained regardless of training eligibility.
Fresh epoch/index-seeded waveform positions and permitted non-wake TTS/confuser context; runtime1s crop BEFORE feature extraction. Half windows exercise actual48k->16k100ms-chunk resampling after deterministic upsampling. This is algorithmic parity, NOT a real microphone simulation. Partial overlaps are IGNORE; explicit neighbor labels logged; no fabricated word boundaries or partial-negative labels. Complete-word retention remains manually unverified.

## Matched Validation

Primary budget64.64992events/hour is anchored BEFORE training to baseline under shared policy at tau.30. It is an adversarial-corpus comparison, not acceptable real-room FAR. Raw/HP and original/generated cohorts must satisfy budgets separately. HP80 is a filtering proxy, NOT AEC.
One fresh1s scoring window every250ms; two consecutive fresh decisions; RMS/nonwake gates; integer2s cooldown. Stale-score test shows four-window max can reuse one spike as sustain. Both baseline/candidate disable that heuristic. GRU state resets each scoring call.

| Budget | Model | Threshold | Raw validation recall | HP validation recall | Raw/HP original events |
|---:|---|---:|---:|---:|---:|
|5.000|baseline|0.9000|34.05%|30.67%|1/1|
|5.000|candidate|0.9999|11.35%|11.96%|0/0|
|15.000|baseline|0.8300|41.41%|38.65%|3/3|
|15.000|candidate|0.9990|35.28%|33.13%|2/1|
|30.000|baseline|0.5900|56.44%|52.15%|7/5|
|30.000|candidate|0.9950|60.43%|56.75%|5/2|
|60.000|baseline|0.3500|70.25%|64.72%|14/8|
|60.000|candidate|0.9900|71.47%|67.48%|14/7|
|64.650|baseline|0.2800|73.93%|67.48%|16/11|
|64.650|candidate|0.9900|71.47%|67.48%|14/7|
|120.000|baseline|0.1100|84.66%|81.60%|28/28|
|120.000|candidate|0.9700|87.12%|80.67%|25/17|

## Frozen Exposed Test

| Model | Raw detected | HP detected | Raw/HP original false events | Raw rate95%CI /hour |
|---|---:|---:|---:|---:|
|baseline|263/385 (68.31%)|264/385 (68.57%)|25/21|81.54 (52.77..120.37)|
|candidate|250/385 (64.94%)|245/385 (63.64%)|34/15|110.90 (76.80..154.97)|

Full curves/policies froze before test scoring. All385official positives are synthetic and previously exposed; no fresh independent claim. Event counts cannot be mixed with clip TNs to fabricate accuracy/precision/F1. Original negative exposure~18.4min; generated17.1min reported separately. Existing automatic boundary-reference failures remain denominator misses.
Signed endpoint latencies use automated energy reference, not manual human endpoints; paired common-detection delays:
- raw: fixes24, regressions37, common226, median delay0.0ms, p950.0ms.
- hp80: fixes26, regressions45, common219, median delay0.0ms, p950.0ms.

## Actual Raw/AEC Development Replay

No annotated utterance timestamps: these are event counts, NOT true recall or detection latency. Pairs correlated and previously exposed. Never used as gradient/calibration inputs.

| Recording condition/path | Baseline fresh | Candidate fresh | Exp15 deployed |
|---|---:|---:|---:|
|music_only/raw/20261011_192226_57be1938|0|0|0|
|music_only/aec/20261011_192226_57be1938|0|0|0|
|speech_over_music/raw/20261011_194901_531c6356|0|0|0|
|speech_over_music/aec/20261011_194901_531c6356|0|0|0|
|quiet_speech/raw/20261011_204709_c0a63ffa|1|2|2|
|quiet_speech/aec/20261011_204709_c0a63ffa|0|0|0|
|quiet_speech/raw/20261011_210021_2027daaf|0|1|1|
|quiet_speech/aec/20261011_210021_2027daaf|0|1|1|
|music_only/raw/20261011_210156_0987295e|0|0|0|
|music_only/aec/20261011_210156_0987295e|0|0|0|
|speech_over_music/raw/20261011_210235_6e8f0942|0|0|1|
|speech_over_music/aec/20261011_210235_6e8f0942|0|0|1|

## Checks And Remaining Limits

- baseline: training-wave/runtime score parity maxerror0.0; silence/chunk checks PASS; host full-hop p952.119ms, RTF0.00847, ORT intra1/inter1. NOT Pi latency.
- candidate: training-wave/runtime score parity maxerror0.0; silence/chunk checks PASS; host full-hop p952.142ms, RTF0.00857, ORT intra1/inter1. NOT Pi latency.
- exp15_deployed: training-wave/runtime score parity maxerror0.0; silence/chunk checks PASS; host full-hop p958.828ms, RTF0.03531, ORT intra1/inter1. NOT Pi latency.
- Original exp15 initialization inherits historical split/exposure weaknesses; fine-tuning cannot erase them. Historical validation/test share17canonical Piper voice families. No real-speaker generalization claim.
- No verified real-room/music training negatives with independent ancestry were available; context uses permitted TTS/confusers and generated stress. Actual AEC contains nonlinear effects not represented by HP80; proper AEC recall/latency needs phrase timestamps.
- Source data/voice terms and access remain private-workspace limitations; no raw audio/credentials/state published. Public inference verification is reproducible; private wake fine-tuning is NOT advertised as publicly one-command reproducible.
- Independent confirmation: consenting fresh5speakers x10wakes in quiet and5x10over low music, with timestamps; at least30min no-wake music/household/beep recordings. Freeze policies first; do not tune on this confirmation set. About3zero-event negative hours are needed for a95%upper rate near1/hour.
- No Pi install, action, call/SMS, threshold replacement or historical-artifact overwrite occurred. No extra training is queued. See PI_TEST.md for wake-only comparisons and rollback.
