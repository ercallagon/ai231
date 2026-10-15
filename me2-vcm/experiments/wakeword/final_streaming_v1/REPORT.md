# Final streaming wakeword trial

Frozen validation selection; previously exposed synthetic-positive test; no physical Pi verification.
Default remains exp15. Both candidate models are trained from random initialization; no external detector weights.
Baseline calibrated BEFORE training. Its calibrated single-window policy is not the deployed four-window tau0.30 policy.
Strict training ancestry filtering means the new arms are matched to each other, not an exact historical exp15 data match.

| Model | Split/condition | Wake detected | FA events / negative hours | FA/hour | Energy-end p95ms |
|---|---|---:|---:|---:|---:|
| exp15_calibrated | validation/raw | 111/326 (34.05%) | 1/0.5319 | 1.88 | 656.8125 |
| exp15_calibrated | validation/hp80 | 100/326 (30.67%) | 1/0.5319 | 1.88 | 673.3375 |
| exp15_calibrated | test/raw | 107/385 (27.79%) | 1/0.5910 | 1.69 | 737.9875 |
| exp15_calibrated | test/hp80 | 108/385 (28.05%) | 1/0.5910 | 1.69 | 737.9125 |
| exp15_default | validation/raw | 289/326 (88.65%) | 40/0.5319 | 75.20 | 808.5124999999999 |
| exp15_default | validation/hp80 | 282/326 (86.50%) | 39/0.5319 | 73.32 | 826.3374999999999 |
| exp15_default | test/raw | 335/385 (87.01%) | 80/0.5910 | 135.36 | 1006.4125 |
| exp15_default | test/hp80 | 333/385 (86.49%) | 61/0.5910 | 103.21 | 977.6374999999997 |
| cmvn2s_v1 | validation/raw | 71/326 (21.78%) | 1/0.5317 | 1.88 | 1173.125 |
| cmvn2s_v1 | validation/hp80 | 51/326 (15.64%) | 1/0.5317 | 1.88 | 1259.5625 |
| cmvn2s_v1 | test/raw | 82/385 (21.30%) | 8/0.5908 | 13.54 | 1162.3375 |
| cmvn2s_v1 | test/hp80 | 66/385 (17.14%) | 8/0.5908 | 13.54 | 1279.796875 |
| pcen2s_v1 | validation/raw | 137/326 (42.02%) | 1/0.5317 | 1.88 | 1274.75 |
| pcen2s_v1 | validation/hp80 | 133/326 (40.80%) | 1/0.5317 | 1.88 | 1134.0749999999996 |
| pcen2s_v1 | test/raw | 139/385 (36.10%) | 3/0.5908 | 5.08 | 1175.5749999999998 |
| pcen2s_v1 | test/hp80 | 131/385 (34.03%) | 3/0.5908 | 5.08 | 1161.53125 |

Negative exposure combines original negative recordings and1024seconds of reproducible generated stress; raw/HP views are correlated. Reported zero-event upper bounds are in RESULTS.json.
No negative silent padding inflates exposure. Calibration enforces <=5FA/hour separately on original and generated negative cohorts, in both raw/HP conditions; no dilution by easy generated hours. Per-cohort rates are in RESULTS.json. Events/hour on this assembled corpus is not measured real-room FA/hour.
The5FA/hour target is empirical, NOT a statistical guarantee. Short negative exposure can yield a95% upper bound above5/hour even with zero observed events; separate cohort bounds are reported.
Positive matching uses recorded phrase intervals and allows1s after waveform end; misses and automatic boundary failures remain in denominators.
Energy-end latencies are automated estimates, not manually verified end-of-speech; signed latency can be negative.
HP80 is a deterministic filtering stress, NOT a simulation or verification of actual AEC.

## Development Raw/AEC Replay

Event counts are not true recall; phrase identities/timestamps have not been manually verified.
| Condition/path/recording | exp15 default | exp15 calibrated | CMVN | PCEN |
|---|---:|---:|---:|---:|
| music_only/raw/20261011_192226_57be1938 | 0 | 0 | 2 | 0 |
| music_only/aec/20261011_192226_57be1938 | 0 | 0 | 0 | 0 |
| speech_over_music/raw/20261011_194901_531c6356 | 0 | 0 | 0 | 2 |
| speech_over_music/aec/20261011_194901_531c6356 | 0 | 0 | 0 | 1 |
| quiet_speech/raw/20261011_204709_c0a63ffa | 2 | 0 | 0 | 2 |
| quiet_speech/aec/20261011_204709_c0a63ffa | 0 | 0 | 0 | 2 |
| quiet_speech/raw/20261011_210021_2027daaf | 1 | 0 | 2 | 2 |
| quiet_speech/aec/20261011_210021_2027daaf | 1 | 0 | 0 | 2 |
| music_only/raw/20261011_210156_0987295e | 0 | 0 | 0 | 0 |
| music_only/aec/20261011_210156_0987295e | 0 | 0 | 0 | 0 |
| speech_over_music/raw/20261011_210235_6e8f0942 | 1 | 0 | 0 | 0 |
| speech_over_music/aec/20261011_210235_6e8f0942 | 1 | 0 | 0 | 1 |

## Recommendation And Limitations

Validation nominee: pcen2s_v1. Recommendation: not supported for deployment; optional diagnostic Pi A/B only.
exp15 remains default. No automatic deployment or claim of successful Pi verification.
No complete human-source ancestry proof across distinct TTS engines; official positives are all synthetic and exposed test is not independent.
Insufficient real call/ring/music diversity and real positive speakers remain limitations. Generated stress is not actual call audio.
Both trials retain full original <=2s training waveforms with varied positions; over-capacity positives are excluded and enumerated, never silently cropped.
Fixed PCEN is motivated by https://arxiv.org/abs/1607.05666 ; no published checkpoint or learned frontend is loaded.
The integer sample clock fixes chunk-dependent cooldown; silence, chunk-size and vectorized/actual-gate parity checks passed.
Host features+ONNX costs with intra1/inter1, NOT Pi or total action latency:
- exp15_calibrated: p95 2.177ms, window RTF 0.00218, scoring-hop RTF 0.00871.
- cmvn2s_v1: p95 4.117ms, window RTF 0.00206, scoring-hop RTF 0.01647.
- pcen2s_v1: p95 4.196ms, window RTF 0.00210, scoring-hop RTF 0.01678.

Additional isolation disclosure: new training has zero known canonical voice/PCM overlap with reserved metadata. However, historical validation and test still share 17 canonical Piper voice families. We did not resplit or relabel those sets; they are exposed, correlated evaluations, not independent unseen-speaker confirmation. This is an additional promotion limitation. See ISOLATION.json.
