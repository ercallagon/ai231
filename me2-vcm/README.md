# Hey Swift: ME2 voice commands on Raspberry Pi 5

This project runs Hey Swift -> command capture -> intent and slots -> rejection
-> preview/action -> idle listening. The repository default is the frozen
six-second **BC8 + temporal CNN + attentive statistics pooling** model at
confidence threshold **0.954**. Physical Pi 5 verification of this version is
**pending**. Raspberry Pi 5 is the instructor-approved substitute for Pi 4.

The selected classifier has 488,264 parameters and accepts complete recordings
up to six seconds. It uses deterministic NumPy log-mel features and a padding
mask. Six seconds is capacity: live capture uses onset, pre-roll and endpoint
silence. All detector weights were trained from scratch. There is no claim of
causal neural encoding, KV-cache, ASR decoding or continuous streaming inference.

The complete run_043 fallback (327,750 parameters, two-second features, threshold
0.875) remains selectable. Exp15 is the separate preserved wakeword model.

## Install and verify

From this directory in a new virtual environment:

```bash
python -m pip install -r requirements-eval.txt
python scripts/reproduce.py verify
```

Linux/macOS: `./run.sh verify`. Windows: `py scripts/reproduce.py verify` or
`.\run.ps1 verify`. Verification loads both ONNX contracts, checks artifact hashes
and runs synthetic inputs. It does not measure recognition or Pi hardware.

## One-command result reproduction

After dependency installation, this fetches only required pinned Parquet files,
verifies their size/hash and embedded PCM, then evaluates both frozen models:

```bash
./run.sh reproduce --fetch --output work/reproduction_unique
```

Windows equivalent:

```powershell
py scripts/reproduce.py reproduce --fetch --output work/reproduction_unique
```

Existing cached data can be reused with `--data-root PATH_TO_RAW_ROOT`; that folder
contains data/, supplemental_*/ and synthetic_negatives/. Output directories
must be new. Default splits are test and holdout. Use `--splits val tune_oos`
for validation accounting; none of these commands fits a gate or trains.

Results include full-manifest failures, processed-subset denominators,
real/synthetic breakdowns, per-family metrics, paired errors and timing scopes.
The public metadata preserves split membership/order and audio hashes, removes
unneeded identifying fields and pseudonymizes declared speaker groups.

## Frozen evaluation and limits

Both evaluation sets were repeatedly examined during this research. Results are
**previously exposed evaluation**, not independent untouched confirmation. Gate
0.954 was selected using validation and the existing OOS calibration pool after
that exposure; no test/holdout-driven gate fitting was performed.

Raw test intent accuracy is 95.16% (real 78.76%, synthetic 98.66%). Raw holdout
intent accuracy is 94.55% (real 90.63%, synthetic 98.11%). These raw intent
figures do not include the confidence gate or prove live acoustic performance.
Use reports/RESULTS.md for accepted intent-plus-slot success, rejection counts,
processing coverage, uncertainty and the exact frozen-policy version.

Dataset: airimonda/ai231-me2-voice-commands, pinned revision
bfca0bbebfe17e00d6a685d1da573615d29ecdce, DOI **10.57967/hf/10723**.
Code is MIT; data and model terms are separate. See data/DATASET_LICENSE.md,
MODEL_TERMS.md and THIRD_PARTY_NOTICES.md. No raw recordings, private account
state, songs or downloaded voices are published.

## Pi testing and rollback

See docs/PI5.md for installation, microphone checks, selectable profiles,
benchmarking and rollback. The working /home/ej/swift-e2e installation is not
modified by these commands.

```bash
python scripts/run_assistant.py --profile selected --device 0 --offline
python scripts/run_assistant.py --profile run043 --device 0 --offline
```

Actions default to preview/mock; calls/SMS are not required for testing.
Optional speech/music need separately obtained assets and private configuration.
The generated cue is owned by this project; preserve the working Pi's local cue.
Energy endpointing has no pretrained weights. The pause-tolerant default waits
1.2 seconds of silence, adding endpoint latency; its Pi behavior is unverified.

## Optional scratch retraining

Install requirements-train.txt on a suitable GPU host, then inspect the plan:

```bash
python scripts/train_selected.py --output work/retrain_unique --gpu 0
```

Add `--fetch --execute` to actually train from random initialization. This uses
the frozen train/validation membership, seed 42, original augmentation/loss and
120-epoch/patience-20 budget. It checks a single GPU before starting; no SLURM.
New weights require their own validation calibration. Do not assume threshold
0.954 transfers to newly trained weights. The published checkpoint reproduces
reported frozen-model results without retraining.

Full wakeword retraining is not publicly reproducible from the current archive:
recording publication/consent terms and the original PyTorch checkpoint remain
unresolved. Its exact deployed ONNX, feature source, configuration, corrected
metrics and a clearly distinct reproduction history are retained.

## Evidence

- experiments/selected/: selected checkpoint, architecture/frontend and calibration.
- experiments/baseline/: run_043 checkpoint, config and comparison evidence.
- experiments/wakeword/: exact artifact provenance and corrected wake evaluation.
- logs/training/: complete selected and baseline training logs.
- experiments/RUN_INDEX.csv: 73 discovered historical trial directories; incompatible
  protocols and unknown/incomplete results are marked rather than ranked together.
- reports/MODEL_SELECTION.md: original selection and authorized calibration revision.
- scripts/benchmark_onnx_pi.py: model-only p95/thread measurements.
- scripts/reproduce.py: injected-file feature/model recognition and RTF on host/Pi.
- docs/BENCHMARK.md: class laptop-speaker -> Pi-microphone benchmark.

Host regression tests use mocked calls, SMS, capture and GPIO. Actual Pi p95,
RTF, live wake/capture behavior and GPIO effects remain pending. Fresh independent
speaker recordings should be collected only after freezing this release, with
consent and no further tuning; they are not currently available.
