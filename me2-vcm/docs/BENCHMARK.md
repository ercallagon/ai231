# Class benchmark and supplementary injection

Reviewed upstream: https://github.com/airimonda/vcm-benchmark at commit
ab39857cf73dd067d068fed388c321ab160dc191. Its parser/schema host suite passed
174 checks. It expects laptop-speaker playback into the Pi microphone.
File injection is supplementary; --no-audio is restricted to simulation.

## Pi terminal

Install the sparse Pi checkout as described in PI5.md, then:

```bash
cd ~/ai231/me2-vcm
.venv/bin/python scripts/run_benchmark_assistant.py --profile selected --device 0 --log-dir work/pi5_test_01/selected_logs
```

The launcher uses preview actions, no synthesized speech and offline services;
it writes wake/decision records to the selected_logs directory. Each record is flushed
immediately after recognition and before services. Rejections log OUT_OF_SCOPE.
infer_ms covers input preparation, features and ONNX; audio_ms is actual model
waveform duration. A preprocessing failure has zero processed audio and should
not be mistaken for fast model inference. capture_audio_ms is separate.

## Windows laptop

Use the existing benchmark checkout when present. For a new one:

```powershell
git clone https://github.com/airimonda/vcm-benchmark.git vcm-benchmark
cd vcm-benchmark
git checkout ab39857cf73dd067d068fed388c321ab160dc191
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
scp ej@192.168.100.103:~/ai231/me2-vcm/data/vcm/raw/data/holdout-00000-of-00001.parquet .\holdout_frozen.parquet
if ((Get-FileHash .\holdout_frozen.parquet -Algorithm SHA256).Hash -ne 'c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e') { throw 'Holdout checksum mismatch' }
.\.venv\Scripts\python.exe benchmark.py --mode ssh --host ej@192.168.100.103 --proc run_assistant.py --log-dir /home/ej/ai231/me2-vcm/work/pi5_test_01/selected_logs --wake-word "Hey Swift" --size full --seed 42 --wake-gap 0.8 --holdout .\holdout_frozen.parquet --model pi:~/ai231/me2-vcm/models/vcm/selected.onnx
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

For run_043 repeat under the same frozen conditions with --profile run043 on
Pi and its ONNX path on the laptop; this changes capture policy as documented.
No test result may be used to adjust model weights or the frozen threshold.
No real calls/SMS or music playback is needed for benchmark execution.
