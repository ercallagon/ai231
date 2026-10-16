# Class benchmark and supplementary injection

The current end-to-end execution sequence is
[FINAL_PI_VALIDATION.md](FINAL_PI_VALIDATION.md). Older example gaps below are
historical; the final controlled comparison explicitly uses a two-second gap.

## Endpoint trial and clean reruns (2026-10-16)

An explicit `--end-silence-ms 800` on the Pi benchmark launcher selects an
experimental 0.8-second endpoint wait. Omitting it retains the frozen profile's
capture policy (Champion: 1.2 seconds). Model weights, preprocessing, labels and
rejection thresholds do not change. Use the same explicit endpoint policy for
both models in a comparison; do not mix it with the earlier default-policy run.
This is input endpointing, not the model's six-second capacity or the laptop's
wake-to-command gap. A modeled 0.6-second internal pause is preserved with 800 ms;
a modeled 1.08-second pause is NOT. Retain 1200 ms for slower/paused commands.
Host simulations do not verify complete-word retention on the Pi.

Prepare Windows first with the Pi assistant stopped. Record wake takes, choose
the playback speaker, then calibrate the Pi microphone. `plughw:2,0` was the
operator's USB input; recheck device enumeration rather than assuming it is
portable. At **Sound check**, start the Pi listener and wait for Listening.
If replaying with an already running listener and unchanged calibrated audio,
skip the direct ALSA mic check; otherwise it competes for the same microphone.

Example on Pi, from this checkout (choose a new directory for a new session):

```bash
.venv/bin/python scripts/run_benchmark_assistant.py --profile champion --device 0 --end-silence-ms 800 --log-dir work/pi5_test_01/champion_echo_endpoint800_01
```

Use that exact absolute directory in the Windows benchmark's `--log-dir` and
retain `--wake-gap 2.0`. Keep speaker/placement/gain fixed after calibration.
For the operator's recorded development takes, `--wake-files` accepts
`runs/wake_probe_20261016_195853/wake_2.wav` and `wake_3.wav` with the same parent.
These takes were screened for triggering: disclose that selection, retain wake
misses in end-to-end scoring, and do not present the run as unbiased wake recall.
Do not change model/gate settings based on this previously exposed holdout.

The launcher now saves both a filtered `.log` (wake/decision events only) and a
separate `.diagnostics` file (full stdout/stderr, including capture noise floor,
threshold, no-speech peak input and maximum sustained onset). Give only the
filtered log directory to the benchmark. Do not concatenate diagnostics into
the score stream. Ctrl+C stops the wrapper without a Python traceback.

Known Windows compatibility workarounds for the reviewed upstream version:
create `.cache` before profiling, and in `vcmbench/pi.py`'s `SshLink.run`, keep
`input=input_bytes` but add
`stdin=(subprocess.DEVNULL if input_bytes is None else None)` to `subprocess.run`.
Preserve an exclusive timestamped backup before editing. This prevents the
no-input SSH subprocess from sharing interactive stdin while preserving data
uploads. The operator verified this got past the repeated SSH timeout. These
are local compatibility changes, not alterations to official scoring.

After a run, retain all official outputs and the two Pi logs. Supplementary
review of a copied Windows result directory:

```bash
.venv/bin/python scripts/review_class_benchmark.py work/copied_benchmark_run --output work/copied_benchmark_run/SWIFT_AUDIT.json
```

The audit never rewrites scores/trials or drops early events. It separates
zero-audio failures from absent timing fields, processed-only inference/RTF,
detector-estimated endpoint latency, negative external latencies, multiple
decisions and accepted versus rejected no-wake predictions. It refuses to
overwrite its output. Negative external latency can indicate a premature
decision or playback/log timestamp mismatch; full diagnostics are needed before
attributing it to room noise. Upstream selects the first decision in a trial
interval, so repeated decisions can also affect recognition scoring.

Rollback the endpoint experiment by omitting `--end-silence-ms` or explicitly
using 1200 for Champion. No installation/setup rerun, model replacement or
global audio-volume reset is required. Pi audibility, pause retention, actual
endpoint p95 and room-noise behavior remain physical verification tasks.

Reviewed upstream: https://github.com/airimonda/vcm-benchmark at commit
ab39857cf73dd067d068fed388c321ab160dc191. Its parser/schema host suite passed
174 checks. It expects laptop-speaker playback into the Pi microphone.
File injection is supplementary; --no-audio is restricted to simulation.

## Pi terminal

Install the sparse Pi checkout as described in PI5.md. If collecting automated
injection metrics too, run `scripts/test_pi.py --reset` before live trials;
reset archives the existing test folder, including any decision logs. Do not
reset during an active acoustic benchmark. Then run:

```bash
cd ~/swift-clean/me2-vcm
.venv/bin/python scripts/run_benchmark_assistant.py --profile champion --device 0 --log-dir work/pi5_test_01/official_champion_logs
```

The launcher uses preview actions, no synthesized speech and offline services;
it writes wake/decision records to the official_champion_logs directory. Each record is flushed
immediately after recognition and before services. Rejections log OUT_OF_SCOPE.
infer_ms covers input preparation, features and ONNX; audio_ms is actual model
waveform duration. A preprocessing failure has zero processed audio and should
not be mistaken for fast model inference. capture_audio_ms is separate.
The additive `speech_end_to_decision_ms` field measures from the detector's
last voiced frame read time to the completed intent/slot/rejection decision,
before service dispatch, debug-file writes or log delivery. It includes endpoint
waiting. No-speech decisions have null latency. This is a detector estimate,
not an independently annotated acoustic speech end or the laptop's observation
of the flushed record. Existing `infer_ms` keeps its original meaning.

After stopping the assistant with Ctrl+C, summarize its saved logs on the Pi:

```bash
.venv/bin/python scripts/summarize_runtime_latency.py work/pi5_test_01/official_champion_logs --output work/pi5_test_01/selected_live_latency.json
```

Retain this supplementary JSON with the official benchmark outputs. It reports
trial counts, p50/p95 in milliseconds, accepted/rejected timing subsets, missing
endpoints and rejection reasons. It does not change upstream benchmark scoring.
Use separate directories for each profile/session and report no-speech failures
alongside timed trials. See PI5.md for measurement scope and sample-size limits.

## Windows laptop

Use the existing benchmark checkout when present. For a new one:

```powershell
git clone https://github.com/airimonda/vcm-benchmark.git vcm-benchmark
cd vcm-benchmark
git checkout ab39857cf73dd067d068fed388c321ab160dc191
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
scp ej@192.168.100.103:~/swift-clean/me2-vcm/data/vcm/raw/data/holdout-00000-of-00001.parquet .\holdout_frozen.parquet
if ((Get-FileHash .\holdout_frozen.parquet -Algorithm SHA256).Hash -ne 'c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e') { throw 'Holdout checksum mismatch' }
.\.venv\Scripts\python.exe benchmark.py --mode ssh --host ej@192.168.100.103 --proc run_assistant.py --log-dir /home/ej/swift-clean/me2-vcm/work/pi5_test_01/official_champion_logs --wake-word "Hey Swift" --size full --seed 42 --wake-gap 0.8 --holdout .\holdout_frozen.parquet --model pi:~/swift-clean/me2-vcm/models/vcm/selected.onnx
```

Initial setup asks for speaker/microphone selection, SSH authentication, three
wake-word takes (or --wake-files), and a sound-level check. Complete these
prompts and freeze placement/gain with development audio before scored trials.
Do not use simulation as Pi evidence. A hotspot/shared LAN works after
dependencies/data are cached. Keep SSH keys and wake takes out of Git.

The Pi uses a new dedicated decision-log directory, preventing selection of an
older assistant log. --proc run_assistant.py monitors the assistant child,
rather than the log wrapper. Keep the laptop about one metre from the mic and
record distance, gain, angle, noise, Pi temperature and commit. Freeze levels
and wake-gap before scored trials. The 0.8-second gap is an upstream starting
point, not a verified Pi cue delay.

The upstream README/help mentions 196 holdout clips, but the pinned dataset
used here contains 202 (186 supported, 16 OOS). The command above passes the
pinned Parquet explicitly and checks its SHA:
c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e.

Retain metrics.json, report.md, trials.csv, pi_metrics.csv, pi_specs.json,
config.json, plan.json and the Pi decision log. Label results as previously
exposed class holdout, not pristine unseen-speaker confirmation. Report wake
false triggers, command OOS false accepts and recognition separately.

For run_043 repeat under the same frozen conditions with --profile baseline on
Pi and its ONNX path on the laptop; this changes capture policy as documented.
No test result may be used to adjust model weights or the frozen threshold.
No real calls/SMS or music playback is needed for benchmark execution.
