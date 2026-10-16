# Final Pi 5 Validation

Run these steps on the existing `~/swift-clean/me2-vcm` installation. They do
not replace private setup, songs, accounts, models or previous deployments.
Pi 5 replaces Pi 4 with instructor approval. No HPC-to-Pi connection is assumed.

## 1. Update Safely (Pi SSH)

Stop the running assistant with Ctrl+C first. From Windows use
`ssh ej@192.168.100.103`; macOS/Linux use the same command in Terminal.

```bash
cd ~/swift-clean
git status --short
git pull --ff-only origin main
cd me2-vcm
git rev-parse HEAD
.venv/bin/python -m pip check
```

If Git reports a conflict or refuses to fast-forward, stop and retain the output;
do not reset, clean, force-pull or rerun setup. This candidate adds no dependency.
For a new installation instead follow [QUICKSTART_PI.md](QUICKSTART_PI.md).
If a later update changes locks, follow its dependency instructions before testing.
Do not run `./swift install` or `setup` merely to update documentation/code.

## 2. Record Preflight

Use a fresh evidence folder, outside the resettable evaluation directory. Set
`EXPECTED_COMMIT` to the full published candidate SHA supplied with the release,
not an old value from another trial.

```bash
EXPECTED_COMMIT=PASTE_FULL_CANDIDATE_SHA
EVIDENCE=work/final_candidate_01
mkdir -p "$EVIDENCE"
.venv/bin/python scripts/record_submission.py --expected-commit "$EXPECTED_COMMIT" \
  --device 0 --end-silence-ms 800 --output "$EVIDENCE/PREFLIGHT.json"
./swift doctor
```

The recorder refuses an existing output, dirty tracked project files, mismatched
commit or protected asset hashes. It loads ONNX contracts and records public
environment, model/config hashes, thread settings and requested endpoint policy;
it does not read credentials or prove audible capture/GPIO effects. Device 0 is
this tested USB microphone, not a universal default. Confirm device identity in
doctor; never silently substitute another input. Retain doctor warnings.
If preflight fails or a required dependency is missing, stop before scored trials
and return the error. A weather network warning alone does not block offline
recognition; record it instead of claiming the API worked.

## 3. Injected Recognition And Model Cost

```bash
./swift evaluate --reset
./swift report
```

`--reset` archives previous `work/pi5_test_01` results before creating fresh ones;
it does not erase private setup or songs. Run it **before** acoustic testing, never
during a run. Cached verified Parquet is reused. A first fetch is about 305 MB.
The command evaluates frozen Baseline (2s BC8) and Champion (6s BC8-TCN-ASP),
benchmarks their model-only cost, refreshes diagnostics and creates `REPORT.md`.
Keep all failures/denominators. These are previously exposed test/holdout sets.
Injection reads complete files; it bypasses wake, microphone and causal endpointing.
Model-only p95/nominal RTF is not speech-end latency. Feature+ONNX timing has a
separate scope and actual-audio-duration RTF. Do not substitute one for another.

## 4. Live Endpoint Check (Pi)

```bash
./swift live-latency --model champion --end-silence-ms 800
```

Preview only: no real calls or SMS. Wait for listening, say Hey Swift, then the
command after the cue. Include short commands, all temperature values, and
commands with a natural internal pause. Keep failed/no-speech trials, not just
successful predictions. Ctrl+C stops. This measures detector-estimated speech
end to prediction, before action/TTS; it is not annotated acoustic ground truth.
800 ms is an optional interaction experiment, not a changed model/gate/default.
If it splits words/pauses, record that material failure; compare the preserved
1200 ms setting with `--end-silence-ms 1200` in separate logs, not mixed results.
Six seconds is model capacity, not mandatory listening time.

## 5. Acoustic Class Benchmark (Laptop + Pi)

Use [BENCHMARK.md](BENCHMARK.md) for the class repository and schema. Reviewed
upstream commit: `ab39857cf73dd067d068fed388c321ab160dc191`. Record your actual
benchmark revision and local diff. Windows uses `.venv\Scripts\python.exe`;
macOS uses `.venv/bin/python` (see [MACOS.md](MACOS.md)).

In the existing Windows benchmark checkout, PowerShell:

```powershell
cd C:\Users\EJ\vcm-benchmark
New-Item -ItemType Directory -Force .cache | Out-Null
git rev-parse HEAD
git diff -- vcmbench/pi.py
.\.venv\Scripts\python.exe benchmark.py --help
.\.venv\Scripts\python.exe benchmark.py --mode ssh --host ej@192.168.100.103 --proc run_assistant.py --log-dir /home/ej/swift-clean/me2-vcm/work/pi5_test_01/final_champion_800 --wake-word "Hey Swift" --wake-files runs/wake_probe_20261016_195853/wake_2.wav runs/wake_probe_20261016_195853/wake_3.wav --size full --seed 42 --wake-gap 2.0 --holdout .\holdout_frozen.parquet --model pi:~/swift-clean/me2-vcm/models/vcm/selected.onnx
```

Reuse verified wake takes 2 and 3 from `runs/wake_probe_20261016_195853`, the ECHO
speaker and the recorded two-second wake-to-command gap. Confirm those settings
in the wizard; do not assume reuse retains them. Keep the same gain, placement,
microphone and wake files for Baseline. Report that the takes were screened for
waking reliability: this comparison is not an unbiased wake-recall estimate.
Use the authorized pinned holdout/checksum from BENCHMARK.md; never fit settings
or thresholds to scored results. Keep all 202 wake-present + 16 no-wake trials.

The Pi assistant must be **stopped for the exclusive ALSA mic check**. Choose
`plughw:2,0` only after confirming the USB identity; `default` previously failed.
Record fresh room/speech level and peak readings with the actual ECHO setup;
aim for the benchmark's >=20 dB SNR without clipping. Then, at Sound check,
start this in the Pi SSH terminal and wait for `listening`:

```bash
cd ~/swift-clean/me2-vcm
.venv/bin/python scripts/run_benchmark_assistant.py --profile champion --device 0 \
  --end-silence-ms 800 --log-dir work/pi5_test_01/final_champion_800
```

Point the laptop log setting to
`/home/ej/swift-clean/me2-vcm/work/pi5_test_01/final_champion_800`.
Both unscored warmups must wake and yield nonzero processed audio and correct
semantic decisions before continuing. Otherwise stop and diagnose capture,
playback or timing; do not adjust frozen model rejection settings. Do not type
arbitrary input or change audio routing during the scored run.

Run Baseline separately with identical acoustic settings:

```bash
.venv/bin/python scripts/run_benchmark_assistant.py --profile baseline --device 0 \
  --end-silence-ms 800 --log-dir work/pi5_test_01/final_baseline_800
```

For the baseline laptop command, change only `--log-dir` to
`/home/ej/swift-clean/me2-vcm/work/pi5_test_01/final_baseline_800` and `--model`
to `pi:~/swift-clean/me2-vcm/models/vcm/run043.onnx`. All other flags stay fixed.
Select that directory in the laptop wizard. Restarting creates unique filenames;
do not combine different endpoints/profiles as one result. Keep full `.diagnostics`
beside filtered `.log` files. These preview commands do not actuate services.
No real calls/SMS are required. An optional live-action demo is a separate test
requiring operator consent and is not represented by benchmark predictions.

Windows SSH compatibility: the existing local `vcmbench/pi.py` fix supplies
`stdin=subprocess.DEVNULL` to `subprocess.run` when `input_bytes is None`, while
retaining `input=input_bytes`; retain/review that diff if SSH stalls. This is an
I/O fix, not a scoring change. `.cache` creation avoids the upstream profiling
directory error. Do not overwrite unrelated benchmark edits.

## 6. Retrieve And Summarize

The laptop run folder contains `report.md`, `metrics.json`, `trials.csv`,
`pi_metrics.csv`; preserve the run's config, plan and wake-file checksums too.
Copy each run into a distinct Pi folder (replace the timestamp with the actual
new folder, never the initial run):

```powershell
scp -r .\runs\ACTUAL_NEW_TIMESTAMP ej@192.168.100.103:~/swift-clean/me2-vcm/work/class_final_champion
```

On Pi:

```bash
.venv/bin/python scripts/review_class_benchmark.py work/class_final_champion \
  --output work/class_final_champion/SWIFT_AUDIT.json
./swift report --official work/class_final_champion
tar -czf work/final_candidate_evidence_01.tar.gz \
  work/final_candidate_01 work/pi5_test_01 work/class_final_champion
sha256sum work/final_candidate_evidence_01.tar.gz
```

Use fresh names on retries. Include the baseline laptop folder in the evidence
archive after its run. Keep this archive private until checked: runtime/service
diagnostics can contain personal utterance/action details. Do not archive
`work/user`, private profiles, Spotify state, keys, music or voice assets.

On the Windows laptop:

```powershell
scp ej@192.168.100.103:~/swift-clean/me2-vcm/work/final_candidate_evidence_01.tar.gz .
Get-FileHash .\final_candidate_evidence_01.tar.gz -Algorithm SHA256
```

Return that archive plus each laptop run folder for interpretation. The audit
does not rescore/drop early or extra events. Keep official and supplementary
metrics separate, including zero-audio failures, no-response, negative external
latencies, false accepts, raw/conditional slots and timing denominators.

## Acceptance And Rollback

Require matching frozen artifacts, no unexplained parser/timing corruption, full
failure accounting, and no material complete-command retention regression.
Sub-second detector-end-to-decision p95 is a **target**, not a measured promise.
No numerical accuracy or false-trigger pass threshold was pre-agreed; report
paired counts and uncertainty, not a newly invented post-hoc cutoff.
If an issue is material, preserve evidence and explain required retesting before
changing implementation. Keep the model/gate frozen and do not tune on holdout.

Select the preserved fallback without reinstalling or overwriting files:

```bash
./swift run --model baseline
./swift run --model champion --end-silence-ms 1200
```

After a satisfactory final run, only verified evidence/report/deck changes are
planned. Independent unseen-speaker confirmation and some call recovery behavior
remain limitations until separately measured; host checks cannot close them.
