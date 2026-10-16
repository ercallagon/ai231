# Raspberry Pi 5 deployment and testing

For the final existing-install update and measurements use
[FINAL_PI_VALIDATION.md](FINAL_PI_VALIDATION.md). Initial Pi evidence is in
[FINAL_CANDIDATE.md](../reports/FINAL_CANDIDATE.md); Champion acoustic results are
summarized in [the Pi 5 report](../reports/PI5_ACOUSTIC_CHAMPION_20261017.md).
The class acoustic run covered Champion only; Baseline is compared on injected
frozen-test data.

Start with [the current SSH quickstart](QUICKSTART_PI.md). It is the canonical
new-user and separate-checkout dry-run procedure.

The short command set is:

```bash
./swift install
./swift setup
./swift doctor
./swift gpio  # optional physical LED/display check if enabled in setup
./swift doctor
./swift config set call-number
./swift run --model champion
./swift evaluate --reset
./swift live-latency --model champion
./swift report
```

The quickstart explains clone paths, OS prerequisites, private/session secrets,
Spotify phone authorization, Bluetooth pairing, real demo and rollback. Use
`--model baseline` for **Baseline (2s BC8)**. Its historical run_043 lineage and
the champion's selected profile remain in provenance; model files, hashes and
frozen gates are unchanged.

Setup defaults asset paths and offers automatic opt-in downloads for Soloist
and the reviewed Amy Piper voice. To replace one saved number later, run
`./swift config set call-number`; it asks in a hidden prompt, backs up the
private profile, and preserves all other settings. See the quickstart for all
editable fields and the exact update-after-install steps.

Automated comparison still uses work/pi5_test_01. Reset archives old test
artifacts; model/data/configuration state and working installations are not
deleted. The consolidated result is REPORT.md/REPORT.json. No live speaking
is needed for feature preprocessing + inference p95/RTF. Live speech-end
decision latency and physical actions are separate checks.

See [METRICS.md](METRICS.md) for exact metric definitions and evidence collection,
[DEPENDENCIES.md](DEPENDENCIES.md) for the committed ARM64 Python3.13 locks and
provisional optional uv path, and [BENCHMARK.md](BENCHMARK.md) for the official
class acoustic benchmark. Raspberry Pi 5 is the approved substitute for Pi 4.
Host tests and wheel downloads do not constitute successful Pi deployment.
