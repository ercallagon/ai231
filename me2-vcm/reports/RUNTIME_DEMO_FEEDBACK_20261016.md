# Live Demo Runtime Feedback

Publication source: the curated `me2-vcm/` checkout; parent commit `17edad5`.
Runtime marker: `pi-flow-v7-demo-controls-20261016`.

The operator reported successful light/color, timer, brightness, supported
temperature, weather, reminder and alarm commands on Pi 5. Remaining feedback
concerned spoken empty captures, communication prompts, post-dial monitoring,
model aliases, music masking the wake, and first-attempt Spotify fallback.
These are operator observations; this report does not turn them into an
automatically scored acoustic evaluation.

## Confirmed Issues

- The public launcher forced confirmation for every enabled call and message.
  Accepted live communication now routes automatically; an explicit
  `--confirm-communications` flag retains the optional approval mode.
- `ConfirmedCommunication` did not expose its underlying call backend.
  `wait_for_call_end` therefore failed after successful dialing. Backend access
  is now forwarded, and transient read failures retry with bounded backoff.
  Active/unknown calls keep capture paused. No call termination is attempted.
- Empty-capture rejection always produced speech. It is now console-only by
  default, with `--speak-no-speech` available. Other rejected spoken commands
  retain feedback; accepted-command and rejection policy is unchanged.
- Offline PLAY reset volume to the cap, overwriting lower listening levels.
  It now clamps excessive levels while preserving lower ones.
- The four CLI model names represented only two models. Public help presents
  Champion/Baseline while preserving historical aliases and artifact paths.

## Music and Speech

The live loop accepts `:volume 10`, `:quieter`, `:pause`, `:stop` and `:help`
followed by Enter. Controls act on the selected music player and share the
speech audio-focus lock. Buffered microphone audio is discarded by reopening
the wake stream after a terminal control. Global speaker/call gain is untouched.

For LIGHT_ON after music, the existing intended sequence is: mute music,
capture/classify, switch the LED on, speak "Lights on.", then restore the prior
music volume. A regression executes the real loop, router, scheduled-device
service and speech hooks with simulated capture, playback and hardware. It
checks the LED-before-speech order, one spoken response at zero music volume,
and restoration afterward. Playback resumes after feedback; a permanent stop
requires STOP. Actual Pi audibility and acoustic interference remain pending.

## Spotify: Improvements and Uncertainty

The root cause of the operator's first-attempt fallback remains unconfirmed.
Live startup now checks the saved-login daemon before listening, bounded to
15 seconds, without requesting playback or activation. PLAY uses automatic
activation with 25-second operation, 15-second startup, and 3-second control
budgets. An existing explicit phone authorization may still be needed for the
first login. `./swift spotify` now attempts saved-login activation without an
unconditional phone-selection prompt.

Fallback diagnostics include the failing phase. Existing uncertain-play and
confirmed-pause rules still prevent intentional overlap with local music.
An intentionally disabled Spotify integration does not receive online PLAY
attempts just because offline music or internet weather is enabled. Terminal volume control mitigates
music masking during a demo; no acoustic echo cancellation is implemented.

## Verification and Provenance

Host regression commands, using mocked devices/phone/player calls:

Results: **68 focused regression checks + 54 existing action/integration checks
passed (122 total)**. Frozen artifact integrity and finite CPU ONNX execution
passed for both VCM profiles and the existing wake export registry. These are
host results; the updated runtime has not yet been verified on physical Pi.

```bash
PYTHONPATH=src python -m pytest tests/test_demo_runtime_feedback.py tests/test_call_state_contract.py tests/test_public_workflow.py tests/test_music_capture_safety.py tests/test_runtime_optimizations.py tests/test_rpi_slot_translation.py -q
PYTHONPATH=src python -m pytest tests/test_actions.py tests/test_integration.py tests/test_selected_runtime.py tests/test_pi_runner.py scripts/test_pi_runtime_v6.py scripts/test_local_pi_updates.py -q
python scripts/reproduce.py verify
```

The full model/config/manifest hashes are checked through ARTIFACTS.json and
the existing frozen-profile tests. Some publication-source checksum entries
were already stale after previous documentation/fetch/workflow updates; they
are refreshed against current tracked source. Frozen model/config/data/log
hash changes are forbidden during that refresh.

No new training, calibration, test/holdout scoring, live call, SMS, OS audio
change or physical deployment occurs in this work. Thresholds remain Champion
0.954 and Baseline 0.875. Checkpoint/model bytes, exp15 and wiring are preserved.
The source backup and JUnit evidence are ignored intermediate files under
`work/runtime_feedback_20261016_v1/`.

See [the Pi checks and rollback](../docs/PI_RUNTIME_FEEDBACK.md) for update
commands, new behavior, remaining phone/Bluetooth limitations and targeted
hardware verification. No repeated injected evaluation is needed to establish
these interaction changes; verify them with the guided live Pi checks.
