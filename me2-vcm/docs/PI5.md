# Raspberry Pi 5 installation and verification

Status: host-tested integration, **physical Pi 5 verification pending**.
Pi 5 (4 GB) is the instructor-approved replacement for the deck's Pi 4.
Keep the working ~/swift-e2e installation and its virtual environment intact.
The repository checkout is a separate test location; no installer copies files
over the working deployment or changes Spotify/account state.

## Install in a repository checkout

On Debian 13 / Python 3.13, install these system packages yourself:

```bash
sudo apt update
sudo apt install git python3-venv python3-dev python3-lgpio libportaudio2 libsndfile1 alsa-utils mpv
git clone --filter=blob:none --sparse https://github.com/ercallagon/ai231.git ~/ai231
cd ~/ai231
git -C ~/ai231 sparse-checkout set me2-vcm/src me2-vcm/scripts me2-vcm/configs me2-vcm/models me2-vcm/audio me2-vcm/data/manifests
cd ~/ai231/me2-vcm
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements-pi.txt -r requirements-eval.txt
.venv/bin/python -m sounddevice
arecord -l
wpctl status -n
```

This sparse checkout includes runtime source, scripts, configs, ONNX models,
cue, and manifest metadata. It omits experiment checkpoints, training logs,
tests, and unrelated ME1 files. The tracked checkout is about 20 MiB; injected
evaluation downloads its audio Parquet separately. The full-repository
`scripts/reproduce.py verify` checks files omitted from this sparse checkout,
so do not run it here. The measurement and assistant commands below load the
selected model contracts. Use --system-site-packages so apt's lgpio is visible.
The assistant creates its private integration/ runtime root and state/, logs/,
and tmp/ subdirectories automatically, including on a fresh sparse checkout.
If an older checkout reports a missing integration/state path, run
`git -C ~/ai231 pull --ff-only` before retrying. The immediate workaround is
`mkdir -p ~/ai231/me2-vcm/integration`; existing state is preserved.
No torch, pandas, typer, rich or huggingface_hub is needed for live inference.
The command installs PyArrow for injected evaluation; omit requirements-eval.txt
for a live-only setup.

These clone commands are for a fresh checkout. If ~/ai231 already exists,
inspect its Git status and sparse paths before changing it; do not clone over
local work. The source revision is saved with the test results. The capture
backend probes supported microphone rates and resamples to the model's 16 kHz
rate. Device numbers can change, so verify the input device instead of assuming
0 is stable.

## Test the selected model and fallback

```bash
cd ~/ai231/me2-vcm
.venv/bin/python scripts/run_assistant.py --profile selected --device 0 --offline
```

This selects mel [batch,1,80,600] and frame_mask [batch,1,600], threshold 0.954.
Temperature 0.6499874044563441 is baked into the graph once. Unsupported
temperature REJECT, OOS and low confidence cannot dispatch an action.
Default actions are previews with mock hardware. The energy boundary detector
uses no pretrained weights. Capture waits up to three seconds for sustained
onset, preserves 200 ms pre-roll, then ends dynamically after 1,200 ms silence.
It preserves internal pauses shorter than that hangover. This conservatively
accommodates the documented 1.06 s training pause; it adds about 1.2 s endpoint
waiting, before inference/service latency, and may still miss quiet speech.

Six seconds is maximum retained model input, including pre-roll/tail; longer
retained input is explicitly rejected, never center-cropped. The 7.4-second
capture safety deadline allows endpoint observation; hitting it rejects the
command as incomplete. Capacity is not a mandatory listening duration.
Complete-word retention and acoustics remain unverified on real Pi recordings.

Stop the test with Ctrl+C and select the complete repository fallback:

```bash
.venv/bin/python scripts/run_assistant.py --profile run043 --device 0 --offline
```

This selects the preserved two-second run_043 graph, labels, config, threshold
0.875 and legacy preprocessing. To return to the earlier working installation,
stop this test process and run your preserved command from ~/swift-e2e using
its existing .venv. No file restoration is needed because it was not modified.
Do not run two microphone listeners simultaneously.

The repo cue is an owned generated tone. It does not replace the Pi's locally
amplified audio/wake_chime.wav. Optional --speak requires piper-tts plus a
separately licensed voice configured privately. Offline MP3 playback requires
your own authorized files. Those assets and Spotify login state are excluded.
Inference and mock command handling work without internet after installation.
Weather/Spotify availability and real peripheral effects are separate checks.

## Preserved wiring

| Peripheral | Wiring |
| --- | --- |
| TM1637 | VCC physical 17; GND 6; CLK BCM23/physical16; DIO BCM24/physical18 |
| KY-016 | GND physical9; R BCM17/physical11; G BCM27/physical13; B BCM22/physical15 |

Both remain connected. configs/hardware.yaml preserves these pins. Installation
does not change wiring. No calls/SMS are part of verification.

## Collect measurements automatically

### Baseline versus champion: one command

Stop other assistant/benchmark processes first. Run on the Pi:

```bash
cd ~/ai231
git pull --ff-only
cd me2-vcm
.venv/bin/python scripts/test_pi.py --reset
```

This is the standard command for a clean initial run or a deliberate repeat.
It archives the entire existing work/pi5_test_01 directory under
work/pi5_test_01_previous_*/pi5_test_01, then recreates the active directory.
Nothing is deleted. Models, labels, thresholds, the virtual environment,
integration state, downloaded audio under data/, and ~/swift-e2e remain intact.
Only the previous test artifacts leave the active test directory. The runner
refuses symlinked work/results paths and a second concurrent test_pi.py run.
Stop older manually launched evaluations too; the lock covers this runner only.

The command evaluates exactly two frozen VCM profiles: baseline run043 and
current selected model selected, on both test and holdout. It does not run
training, calibration, wake detection, real actions, calls/SMS or TTS.
The cached pinned dataset is reused; missing files are downloaded and verified
(about 305 MB on first use). No live speaking is required. Keep the Pi's
power/cooling conditions fixed and avoid concurrent workloads.

The test manifest has 4,693 recordings and holdout has 202. These are previously
exposed evaluation sets. Only successfully processed recordings have inference
timings; retain full-manifest processing failures and timed counts alongside
p95. Use test as the primary reported split and holdout as a separate result,
rather than selecting whichever latency or recognition score is better.

The runner prints progress followed by a two-model timing table and DONE.
If it fails or you stop it with Ctrl+C, partial outputs and injection.log are
preserved. Inspect the error; the same --reset command archives those partial
outputs before a clean retry. Without --reset, an existing results directory
is refused. A non-Pi-5 machine is refused before any result directory reset.
If SSH disconnects, reconnect and inspect whether the earlier process is still
running; do not start overlapping tests.

### Results and timing definition

All new outputs are in work/pi5_test_01:

| File | Contents |
| --- | --- |
| SUMMARY.json | Baseline/champion p95, RTF, timed/manifest counts and failures, separately for test/holdout |
| injection/test_RESULTS.json | Full test recognition, intent/slot/rejection, real/synthetic results and timing metrics |
| injection/holdout_RESULTS.json | Separate holdout metrics |
| injection/test_selected_timings.json and test_run043_timings.json | Per-recording feature_ms, model_ms, infer_ms and audio_ms |
| injection/*_errors.jsonl | Predictions, acceptance and processing status per recording |
| injection/COMPLETE.json | Both splits/profiles completed, with frozen threshold records |
| injection.log | Progress and any errors |
| source_revision.txt, ENVIRONMENT.json | Revision, exact command, environment/dependency versions and archive location |

SUMMARY.json verifies result/model hashes to prevent confusing run043 with
selected. Its profiles.selected.test.infer_p95_ms and per_clip_rtf_p95 are the
recommended slide timing values. The full test_RESULTS.json contains these
under selected.infer_p95_ms and selected.per_clip_rtf_p95.

Timing starts immediately before feature preprocessing and stops when ONNX
session.run() returns model outputs. It excludes download, audio decode,
resampling, microphone capture, endpoint waiting, post-model decision mapping,
actions and TTS. Pinned waveforms are already 16 kHz mono. RTF is processing
time divided by each original waveform duration; its p95 is calculated from
per-recording ratios, not p95 time divided by six seconds. Timing includes the
first inference; no warm-up samples are discarded in injected evaluation.
Both models use one intra-op and one inter-op thread.

Suggested slide label: **Pi 5 CPU, 1 intra-op / 1 inter-op thread: feature
preprocessing + model inference p95 ___ ms; RTF p95 ___; N ___ processed test
recordings (___ attempted; ___ failures).** A full-manifest pass is preferable
to collecting 100 manual samples for this metric, but does not measure
independent repeated-device uncertainty. Model-only synthetic benchmarks are
optional diagnostics and are not needed for this standard comparison.

### Retrieve results

In Windows PowerShell:

```powershell
cd "C:\Users\EJ\Documents\ChatGPT\ME2-VCM 2"
scp -r ej@192.168.100.103:~/ai231/me2-vcm/work/pi5_test_01 .
```

Retain the whole results folder. Existing archived Pi evidence remains in
work/pi5_test_01_previous_* and is not mixed into the new summary. If a folder
with the same name already exists on Windows, rename/archive that local copy
before downloading a new run to avoid mixing old and new files.

### Optional live speech-end to decision latency

This is a separate metric. After automated evaluation, run:

```bash
.venv/bin/python scripts/run_benchmark_assistant.py --profile selected --device 0 --log-dir work/pi5_test_01/selected_logs
```

The launcher uses mock actions and no TTS/calls/SMS. Say Hey Swift, wait for
the cue and capture-ready message, then speak. Stop with Ctrl+C and summarize:

```bash
.venv/bin/python scripts/summarize_runtime_latency.py work/pi5_test_01/selected_logs --output work/pi5_test_01/selected_live_latency.json
```

Use all_speech_decisions.p50_ms/p95_ms and n, labeled **detector-estimated
speech-end to recognition decision**. This starts at the last voiced frame's
read time, not independently annotated acoustic ground truth, and ends at
prediction readiness before debug-file writes, log delivery or services.
It includes endpoint waiting, preprocessing, inference and slot/rejection
mapping. No-speech/legacy records have null latency, counted separately rather
than as zero. Rejected speech decisions remain in the main distribution;
accepted/rejected subsets are reported too. Acceptance is not correctness.
Use at least 30 trials for a preliminary check, preferably 100 or more for p95.

For another live session without repeating injection, use fresh live log and
summary names under the existing test directory; the summarizer refuses an
existing output. Old logs cannot recover the new timing field. Use
docs/BENCHMARK.md for the separate class acoustic benchmark, whose
laptop-observed latency has a different reference. Sparse checkout omits docs;
read the guide on GitHub.

Remaining physical checks include mic handoff/recovery, early/late speech after
the cue, long pauses, over-limit rejection, speech/music interference, GPIO/timer
effects, decision-to-action timing, resource/thermal behavior and optional
speech assets. Injection alone does not verify these.
