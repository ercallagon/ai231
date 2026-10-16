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
mkdir -p work/pi5_test_01
git -C ~/ai231 rev-parse HEAD > work/pi5_test_01/source_revision.txt
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

### Live speech-end to decision latency

Update the checkout with `git -C ~/ai231 pull --ff-only` and stop any other
microphone listener. In `~/ai231/me2-vcm`, run:

```bash
git -C ~/ai231 rev-parse HEAD > work/pi5_test_01/live_source_revision.txt
.venv/bin/python scripts/run_benchmark_assistant.py --profile selected --device 0 --log-dir work/pi5_test_01/selected_logs
```

This uses mock actions and no TTS/calls/SMS. Say Hey Swift, wait for the cue
and capture-ready message, then say a command. Collect repeated trials across
command families. Stop with Ctrl+C, then aggregate the saved decision logs:

```bash
.venv/bin/python scripts/summarize_runtime_latency.py work/pi5_test_01/selected_logs --output work/pi5_test_01/selected_live_latency.json
```

Use `all_speech_decisions.p50_ms`/`p95_ms` and `n` for slides, labeled
**detector-estimated speech-end to recognition decision**. The endpoint is the
last voiced frame's read time, not independently annotated acoustic ground
truth. This interval includes endpoint waiting, preprocessing, inference and
slot/rejection mapping; it excludes services, action completion and TTS.
The timestamp is taken when the prediction is ready, before optional debug-file
writes and decision-log delivery. The class benchmark's laptop-observed timing
is a separate measure and may use a different speech-end reference.

No-speech/legacy records have null latency and are counted separately, never
as zero. Rejected speech decisions remain in the main latency distribution;
accepted/rejected subsets are also reported. Counts and rejection reasons must
accompany the latency. Acceptance is not evidence of correct recognition.
Use at least 30 trials for an initial check, preferably 100 or more for p95;
small samples leave tail latency uncertain. Keep profiles, thermal conditions
and test sessions separate. Old logs cannot recover the new timing field.
For another session, use fresh log-directory and summary-output names. The
summarizer refuses to overwrite an existing summary.

### Inference cost and injected recognition

Model-only cost, without microphone or action side effects:

```bash
.venv/bin/python scripts/benchmark_onnx_pi.py --model models/vcm/selected.onnx --audio-seconds 6 --threads 1 --output work/pi5_test_01/selected_cost.json
.venv/bin/python scripts/benchmark_onnx_pi.py --model models/vcm/run043.onnx --audio-seconds 2 --threads 1 --output work/pi5_test_01/run043_cost.json
```

Injected-file recognition and features+ONNX p95/RTF:

```bash
.venv/bin/python scripts/reproduce.py reproduce --fetch --output work/pi5_test_01/injection
```

The sparse checkout includes the manifests; this command downloads about 305 MB
of pinned test/holdout audio on first use, verifies hashes, and reports all
processing failures. It bypasses wake detection, room acoustics and causal
capture. Both commands record one intra-op and one inter-op thread; neither
claims action or synthesized-speech latency. Model-only RTF uses nominal input
capacity; injected RTF uses each actual waveform duration. Preserve raw JSON.
If timing both profiles, use identical thermal, power and thread conditions.

### Fresh inference-latency run for slides

Live speech is not required for **feature preprocessing + model inference**
latency. Use the entire frozen test manifest (4,693 recordings) and report
holdout (202 recordings) separately. These counts are attempted recordings;
only successfully processed recordings have inference timings. Report the
timed count and full-manifest processing failures alongside p95. Do not choose
whichever split gives the smaller latency. For a brief manual latency check,
30 trials are preliminary and 100 or more are preferable; the complete test
run is the recommended measurement for submission.

For a fresh selected-model run, stop other assistant/benchmark processes and
use the following in one Pi SSH session. The unique directory preserves all
previous logs, including pi5_test_01. Cached pinned downloads are reused.

```bash
cd ~/ai231
git pull --ff-only
cd me2-vcm
mkdir -p work
PI_RUN=$(mktemp -d work/pi5_inference_XXXXXXXX)
git rev-parse HEAD > "$PI_RUN/source_revision.txt"
.venv/bin/python -m pip freeze > "$PI_RUN/packages.txt"
set -o pipefail
.venv/bin/python -u scripts/reproduce.py reproduce --profiles selected --splits test holdout --fetch --output "$PI_RUN/injection" 2>&1 | tee "$PI_RUN/injection.log"
echo "Results directory: $PI_RUN"
```

The script prints progress and ends with `Frozen evaluation complete:`; the
injection/COMPLETE.json file is written only after both splits finish. If the
command fails, retain the log and partial results; do not rerun into that output
directory. To compare the fallback too, include `run043` after `--profiles
selected` before starting a new run. Keep the Pi's power/cooling conditions
fixed and avoid concurrent workloads. A single manifest pass measures a
recording workload, not independent speakers or repeated-device uncertainty.

Retrieve the printed results directory with the scp command below, substituting its
actual path for work/pi5_test_01. The authoritative output fields are:

| File/field | Meaning |
| --- | --- |
| injection/test_RESULTS.json: selected.infer_p95_ms | Test feature preprocessing + ONNX p95, ms |
| injection/test_RESULTS.json: selected.per_clip_rtf_p95 | p95 of each processed clip's processing time / original waveform duration |
| injection/test_RESULTS.json: selected.infer_scope | Exact measurement scope |
| injection/test_RESULTS.json: selected.processed_subsets.all | Manifest/processed counts and failures |
| injection/test_selected_timings.json | One entry per processed recording; feature_ms, model_ms, infer_ms, audio_ms |
| injection/holdout_RESULTS.json and holdout_selected_timings.json | Separate holdout results and timing entries |

`infer_ms` starts immediately before feature preprocessing and stops when
`session.run()` returns model outputs. It excludes file download, audio decode,
resampling, microphone capture, endpoint waiting, post-model decision mapping,
action services and TTS. The pinned input waveforms are already 16 kHz mono.
RTF uses each original recording duration, not the model's six-second capacity;
compute its percentile from per-recording ratios, not p95 time / six seconds.
The timings include the first inference (no discarded warm-up); the separate
model-only cost script uses 30 warm-up and 200 measured passes. Record this
distinction instead of treating the two scripts as equivalent.

Suggested slide label: **Pi 5 CPU, 1 intra-op / 1 inter-op thread: feature
preprocessing + model inference p95 ___ ms; RTF p95 ___; N ___ processed test
recordings (___ attempted; ___ failures).** Speech-end to decision latency is
separate and still requires live/acoustic capture. No real action is executed
by this injection run.

From Windows, retrieve results using the existing LAN/SSH route:

```powershell
scp -r ej@192.168.100.103:~/ai231/me2-vcm/work/pi5_test_01 .
```

Use docs/BENCHMARK.md on GitHub for the separate class acoustic benchmark. The
sparse checkout omits documentation files. Remaining Pi
checks: mic handoff/recovery, early/late speech after the cue, long pauses,
over-limit rejection, speech/music interference, GPIO/timer display effects,
decision-to-action timing, resource/thermal behavior and optional speech assets.
