# Experimental Wakeword Pi 5 A/B

exp15 remains default. Both candidates are opt-in, not verified Pi deployments.
VCM model/calibration, wiring, account state and existing installation are unchanged.
Install/update a separate verified checkout; do not overwrite ~/swift-e2e automatically.
Use the existing documented Pi venv and dependencies in docs/PI5.md.

The current GitHub checkout already includes these optional graphs/configs and
the matching runtime. Follow [PI5.md](PI5.md); no separate transfer archive or
patch installer is required. Keep the existing working deployment untouched.
Historical selective archives were delivered separately and are not GitHub
release assets; their installation instructions are not prerequisites here.

From the ai231/me2-vcm checkout, with its venv active:

```bash
python scripts/reproduce.py verify
# Preferred first A/B: microphone and wake only, NO VCM/actions/calls/SMS.
python scripts/wake_test.py --config configs/wakeword.yaml --dry-run
python scripts/wake_test.py --config configs/wake_pcen2s_v1.yaml --dry-run
python scripts/wake_test.py --config configs/wakeword.yaml --device 0 --no-ack
# Ctrl+C, then candidate on the same microphone and matched playback level:
python scripts/wake_test.py --config configs/wake_pcen2s_v1.yaml --device 0 --no-ack
# Optional preview assistant only after wake-only tests; no real actions:
python scripts/run_assistant.py --profile selected --wake-profile exp15 --device 0 --offline
# Stop the previous runtime before starting B; one mic owner at a time.
python scripts/run_assistant.py --profile selected --wake-profile final --device 0 --offline
# Immediate model-policy rollback, no model file copying:
python scripts/run_assistant.py --profile selected --wake-profile exp15 --device 0 --offline
```

The historical `final` alias selects the experimental PCEN profile, not a promoted
champion. Exp15 remains the default. Use harmless light/timer commands, not real
calls/SMS. A/B repeat quiet, distance,
speech over music and music-only/beep conditions with matched speaker/sink levels.
Record missed wakes, false events with negative exposure seconds, and timestamps.
Do not interpret event counts as recall without manually labeled phrase intervals.
For CPU p95/RTF (no actions, no microphone required):

```bash
python scripts/benchmark_wake.py --config configs/wakeword.yaml --output work/pi_exp15_cost.json
python scripts/benchmark_wake.py --config configs/wake_pcen2s_v1.yaml --output work/pi_candidate_cost.json
# Optional replay of user-owned audio; logs events, not automatic accuracy:
python scripts/benchmark_wake.py --config configs/wake_pcen2s_v1.yaml --input YOUR_RECORDING.wav --output work/pi_candidate_replay.json
```

Above scoring-hop timing includes features and ONNX, not speech endpoint/action
latency. intra/inter threads=1; actual Pi model is recorded in JSON. Physical
latency, microphone, AEC, music interference and wiring effects remain unverified.
Never recalibrate from test/development or Pi A/B observations and call it an
independent final evaluation. Keep fresh consenting speakers for confirmation.

`--wake-profile exp15` restores exp15 weights/policy but retains the corrected
integer-clock source. To roll back source as well, use the preserved fbf01a6
checkout in a separate directory; do not overwrite the working installation.

Raw wake training audio, generator assets and private Pi recordings are not
published. Recipes require the documented permitted local datasets; there is no
claim that private wake training can be fetched/retrained publicly in one command.
VCM public one-command reproduction remains unchanged.

Results and limitations: experiments/wakeword/final_streaming_v1/REPORT.md.

Archived study regression result: 229 passed, 4 skipped; zero failures/errors.
Its separate selective apply/rollback rehearsal passed. These are not Pi tests.

## Windows Transfer

The preferred route is a GitHub checkout on Pi, as in PI5.md. To reach the Pi
from Windows, set your own connection details:

```powershell
$PiUser = 'YOUR_PI_USER'
$PiHost = 'YOUR_PI_HOST'
ssh "${PiUser}@${PiHost}"
```

Activate the existing compatible Pi venv if desired; do not recreate it or
modify account state. Use the canonical ai231/me2-vcm checkout, not an older
incompatible deployment. Owner-specific connection commands are kept in the
private delivery guide, not this public repository.

Additional no-hardware route/capture regression:
python scripts/test_local_pi_updates.py.

For this checkout, rollback means stop the test and resume the preserved working
installation. No transfer installer or file restoration is needed. Source-level
comparison with fbf01a6 should use a separate checkout rather than overwriting it.
