# Pi Runtime Feedback and Verification

Runtime marker: `pi-flow-v7-demo-controls-20261016`.
Previous publication: `17edad5`.
The frozen models, mappings, confidence thresholds, wiring, private setup and
Spotify account state are retained. Host regression checks exercise mocked
services; Bluetooth call state, wake audibility and Spotify startup on the Pi
require the checks below.

## Findings and Changes

| Observation | Evidence and resulting behavior |
| --- | --- |
| Spoken empty-capture feedback interrupts the demo | The router spoke every rejection. Empty captures now log and return quietly; `--speak-no-speech` opts into speech. Other rejection feedback remains. |
| Calls/SMS ask for approval | The public launcher forced `HEY_SWIFT_CONFIRM_COMMUNICATIONS=1`. Live accepted commands now act automatically; `--confirm-communications` enables the old prompt. Only enabled services execute. |
| Successful dialing followed by stopped listening | The confirmation wrapper did not expose `.backend`, which the call guard required. It now forwards backend access. Transient call-state errors retry with bounded backoff; unknown states never count as idle. |
| Call service reported 21.506 seconds | Its timer included waiting for typed approval. Removing the prompt removes that component; actual Bluetooth/TTS time still needs Pi measurement. The reported inference was 0.111 seconds. |
| Four model choices | `selected` aliases Champion; `run043` aliases Baseline. Help now presents only the two public names; older commands still work. |
| Playback masks the wake phrase | Same-terminal music controls lower/pause/stop playback. Offline PLAY previously reset music to 25%; it now preserves lower volumes. This does not implement AEC or improve the wake model itself. |
| Spotify first attempt fell back | Root cause remains unconfirmed without its diagnostic. Startup now warms control readiness before listening (15-second bound); play uses saved-login activation with a 25-second budget. Fallback diagnostics include the failing phase. Existing pause-confirmation safety remains. |

## Update and Check on Pi

Stop the assistant with Ctrl+C. No installation, setup reset or data download is
needed for this source update:

```bash
cd ~/swift-clean
git status --short
git pull --ff-only origin main
cd me2-vcm
./swift run --help
./swift doctor
./swift spotify
./swift run --model champion --live
```

Run Spotify checks only if enabled. `./swift spotify` attempts activation using
this checkout's saved login. First-time authorization may still require choosing
Swift Pi on the phone; no password or saved login from another checkout is used.
Startup should show the v7 marker. Existing private profiles remain usable.

1. Wake without speaking; verify console rejection with no spoken timeout reply.
2. Test a light, timer and reminder to check normal routing and speech.
3. Play music and record the reported backend and any Spotify diagnostic/phase.
   Type `:volume 10` and Enter; check music becomes quieter and Hey Swift can be
   heard. Pause/resume and verify the lower offline level remains.
4. When deliberately testing a real call, confirm your configured recipient
   first. Check dialing needs no terminal approval, wake capture pauses, and
   listening resumes after the tablet actually ends the call. Remote hangup
   may still leave the tablet active; end it there if needed. The guard never
   sends a hangup or interprets beep counts as call state.
5. When deliberately testing a real message, confirm recipient and fixed message
   first. Verify a single send and receipt; check the phone before retrying an
   uncertain result. No automatic regression test calls or messages anyone.

If the guard cannot read call state, retain its `error_type` and `state_retry`
events. For a controlled read-only diagnostic, use the same private environment
through doctor, and inspect PipeWire locally:

```bash
./swift doctor
busctl --user --json=short call org.pipewire.Telephony /org/pipewire/Telephony org.ofono.Manager GetModems
wpctl status -n
```

GetModems can contain private device information. Review locally and share only
gateway paths and call `State` values. Do not paste phone/account identifiers.
Persistent Bluetooth loss still prevents safe automatic capture resumption.
Call loudness and actual call termination belong to phone/Bluetooth routing;
this update does not change global sink gain or terminate calls automatically.

## Rollback

Stop the assistant. The rollback restores only the six changed existing runtime
modules, keeping model/config/private state intact. Check for local edits first;
if `git diff --quiet` fails, preserve those edits before restoring:

```bash
cd ~/swift-clean
git diff --quiet -- me2-vcm/src/hey_swift && git restore --source=17edad5 --worktree -- me2-vcm/src/hey_swift/deployment.py me2-vcm/src/hey_swift/integration/runtime.py me2-vcm/src/hey_swift/integration/router.py me2-vcm/src/hey_swift/integration/call_guard.py me2-vcm/src/hey_swift/integration/e2e_loop.py me2-vcm/src/hey_swift/actions/services/music.py
cd me2-vcm
./swift run --model champion --live
```

The added terminal-control module remains unused after rollback. Git will show
the intentional reverted source edits; inspect them before a future update.
For the preserved two-second model instead, use `--model baseline`.
