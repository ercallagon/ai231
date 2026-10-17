# Pi 5 Verification and Reproduction

The completed Champion acoustic run is reported in
[PI5_ACOUSTIC_CHAMPION_20261017.md](../reports/PI5_ACOUSTIC_CHAMPION_20261017.md).
Baseline and Champion were compared on frozen injected test data.
The procedure below records a new run without replacing private setup,
music, accounts, earlier results or the working installation.

## 1. Update the Existing Pi Checkout

Connect from Windows PowerShell, macOS Terminal or Linux:

```bash
ssh -t USER@PI_ADDRESS
```

Stop the assistant with Ctrl+C, then run on the Pi:

```bash
cd ~/swift-clean
git status --short
git pull --ff-only origin main
cd me2-vcm
git rev-parse HEAD
.venv/bin/python -m pip check
./swift doctor
```

A refused fast-forward or tracked-file conflict requires inspection before
continuing; retain local edits and private state. This update adds no dependency.
Installation and first-run configuration are in [QUICKSTART_PI.md](QUICKSTART_PI.md).

## 2. Record the Run

Use a fresh evidence directory outside the resettable evaluation folder:

```bash
SOURCE_REVISION=$(git rev-parse HEAD)
EVIDENCE=work/verification_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EVIDENCE"
.venv/bin/python scripts/record_submission.py --expected-commit "$SOURCE_REVISION" \
  --end-silence-ms 800 --output "$EVIDENCE/PREFLIGHT.json"
```

The recorder checks the revision, clean tracked tree, frozen hashes and finite
ONNX contracts. It records environment, policy and threads without reading
private credentials. Device identity comes from doctor/setup; device index 0
is not universal. A successful import/contract probe does not prove audible
capture or physical action. Weather-network failure does not invalidate offline
recognition but remains a diagnostic failure.

## 3. Frozen Injected Comparison

```bash
./swift evaluate --reset
./swift report
```

Reset archives earlier results and reuses verified cached Parquet; first fetch
is about 305 MB. Run it before acoustic testing, never during a run.
It scores frozen Baseline/Champion test and holdout, measures model-only cost,
refreshes diagnostics and creates work/pi5_test_01/REPORT.md.
Injection bypasses wake, live capture and endpointing. Feature+ONNX timing and
model-only timing have different scopes; all processing failures remain in the
recognition denominators. These are previously exposed evaluation sets.

## 4. Live Endpoint Smoke Check

```bash
./swift live-latency --model champion
```

Wait for listening, say Hey Swift, then speak after the cue/capture-ready message.
Include short commands, temperature values and natural internal pauses; retain
failures. Ctrl+C summarizes the unique log directory. Default silence endpoint
is 800 ms. A separate `--end-silence-ms 1200` run provides longer pause tolerance.
Six seconds is input capacity, not mandatory listening time. This preview does
not place calls or send SMS. Decision timing ends before services/TTS and uses
a detector-estimated speech endpoint.

After stopping the default session with Ctrl+C, check the explicit override in
a separate session:

```bash
./swift live-latency --model champion --end-silence-ms 1200
```

Keep the two log directories separate. The effective endpoint is recorded with
runtime diagnostics; the model, configuration and frozen rejection threshold
remain those of the Champion profile.

## 5. Acoustic Benchmark

Follow [BENCHMARK.md](BENCHMARK.md) for the upstream checkout, pinned holdout
hash and recording setup. Reviewed upstream revision:
`ab39857cf73dd067d068fed388c321ab160dc191`.
Record the actual benchmark revision and any local SSH compatibility diff.

In Windows PowerShell, from the benchmark checkout:

```powershell
Set-Location "$HOME\vcm-benchmark"
New-Item -ItemType Directory -Force .cache | Out-Null
git rev-parse HEAD
git diff -- vcmbench/pi.py
.\.venv\Scripts\python.exe benchmark.py --mode ssh --host USER@PI_ADDRESS --proc run_assistant.py --log-dir /home/PI_USER/swift-clean/me2-vcm/work/pi5_test_01/acoustic_champion_01 --wake-word "Hey Swift" --size full --seed 42 --wake-gap 2.0 --holdout .\holdout_frozen.parquet --model pi:~/swift-clean/me2-vcm/models/vcm/selected.onnx
```

Replace USER/PI_ADDRESS/PI_USER with the intended account and host.
macOS uses `.venv/bin/python`; see [MACOS.md](MACOS.md).
Record/select wake takes during setup, before scoring, and preserve their hashes.
The reported 2026-10-17 run used two reliability-screened takes; that screening
limits wake-recall interpretation. Keep wake files, gap, gain, placement and
device routing fixed for a new frozen run.

Stop the Pi listener for the exclusive ALSA mic check. Confirm microphone
identity before choosing a hardware device; the tested USB input was
`plughw:2,0`. Record room/speech level and peaks; the upstream check targets
at least 20 dB SNR without clipping. At the laptop sound-check stage, start on Pi:

```bash
cd ~/swift-clean/me2-vcm
.venv/bin/python scripts/run_benchmark_assistant.py --profile champion --device 0 \
  --end-silence-ms 800 --log-dir work/pi5_test_01/acoustic_champion_01
```

Use the verified microphone index instead of 0 if setup reports another identity.
Both unscored warmups must wake, process nonzero audio and return the expected
semantics. Capture/playback problems are diagnosed before scoring; model gates
remain frozen. Preserve all 202 wake-present and 16 no-wake trials.

The official score uses the first decision per trial interval. Extra/early
decisions, no-response and zero-audio trials remain evidence. No acoustic
Baseline comparison is required to reproduce the published Champion result;
a separately labeled diagnostic Baseline run is supplementary.

Windows SSH workaround: the reviewed local upstream patch sets
`stdin=subprocess.DEVNULL` only when `input_bytes is None`, while preserving
`input=input_bytes`. It fixes terminal input handling, not scoring.
Record the diff; preserve other upstream edits.

## 6. Collect Evidence

Laptop output includes report.md, metrics.json, trials.csv and pi_metrics.csv.
Keep configuration, plan, wake hashes and effective playback settings with them.
To analyze a copied laptop result folder on Pi:

```bash
.venv/bin/python scripts/review_class_benchmark.py work/acoustic_champion_01 \
  --output work/acoustic_champion_01/SWIFT_AUDIT.json
./swift report --official work/acoustic_champion_01
```

Use fresh names for new runs; never merge profiles/endpoints. Keep raw diagnostic
logs alongside filtered decision logs. Share only reviewed result files, excluding
work/user, accounts, keys, songs and voice assets. The supplementary audit does
not alter the official score. [METRICS.md](METRICS.md) defines denominators and
timing scope.

## Rollback and Interpretation

Ctrl+C stops the listener. `./swift run --model baseline` selects the complete
fallback; `./swift run --model champion --end-silence-ms 1200` changes pause
tolerance without changing model rejection. Shared Bluetooth/audio routes remain
OS state. No installation or account restoration occurs automatically.

The recorded evidence supports Champion under the measured protocol. It does
not establish call recovery, independent-speaker generalization, music/AEC
robustness or a clean external command-end latency. Numerical acceptance
cutoffs must be declared before a new study, not fitted to scored results.
