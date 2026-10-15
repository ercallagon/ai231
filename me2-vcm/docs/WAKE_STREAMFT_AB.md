# Final Exp15 Fine-Tune: Diagnostic Only

Decision: retain exp15; no replacement is recommended. This trial is closed.
Both baseline and candidate below use latest1s/hop250ms/two fresh consecutive
scores/cooldown2s. The deployed profile retains its historical four-window max.
Do not compare these as if they were identical policies. All thresholds frozen.

## Canonical Checkout (Wake Only, No Actions)

Activate the existing inference environment, then from the repository root:

```bash
python scripts/reproduce.py verify
python scripts/wake_streamft_ab.py --profile deployed --cost --output work/pi_exp15_deployed_cost.json
python scripts/wake_streamft_ab.py --profile baseline --cost --output work/pi_exp15_fresh_cost.json
python scripts/wake_streamft_ab.py --profile candidate --cost --output work/pi_exp15_streamft_cost.json
python -m sounddevice
python scripts/wake_streamft_ab.py --profile baseline --live --device 0 --output work/pi_baseline_live.json
# Ctrl+C, then keep identical microphone/gain/distance/music:
python scripts/wake_streamft_ab.py --profile candidate --live --device 0 --output work/pi_candidate_live.json
```

Use one microphone owner. These tests never load VCM, actions, GPIO, speech or
accounts and play no cue. They cannot place calls/SMS. Output files are exclusive:
use a fresh filename per session. Default assistant selection remains exp15.

## Digital Injection And AEC

```bash
python scripts/wake_streamft_ab.py --profile baseline --input path/to/raw.wav --output work/raw_baseline.json
python scripts/wake_streamft_ab.py --profile candidate --input path/to/raw.wav --output work/raw_candidate.json
python scripts/wake_streamft_ab.py --profile baseline --input path/to/aec.wav --output work/aec_baseline.json
python scripts/wake_streamft_ab.py --profile candidate --input path/to/aec.wav --output work/aec_candidate.json
```

Injection/replay is supplementary, not live speaker-to-microphone benchmarking.
The tool does not implement or configure AEC: use genuinely captured paired audio
or the verified processed source exposed by the Pi. Raw/AEC pairs are correlated.
Event counts need manually supplied utterance timestamps to become recall/latency.
Cost mode measures feature+ONNX scoring hops only, not capture or action latency;
it records model/config hashes, platform/Pi model and ORT intra1/inter1 settings.

## Rollback And Limits

Stop the diagnostic and run the usual assistant with wake profile exp15, or test
`--profile deployed`. No working installation is changed by the standalone tool.
The isolated transfer archive includes exp15's exact graph/config as rollback.
Physical Pi5 results remain pending. Official positives are TTS and test previously
exposed; no independent real-speaker claim. See the trial REPORT/DECISION evidence.
Public inference verification works without Torch; full wake retraining still
requires permitted private raw sources whose publication terms are unresolved.
