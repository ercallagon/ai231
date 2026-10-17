# ME2: Voice Controlled Smart Device (Hey Swift)

An offline, scratch-trained wake/intent/slot recognizer for Raspberry Pi 5 (4 GB):
**Hey Swift -> dynamic capture -> intent + slots -> safe rejection -> action -> listening**.
Internet services are optional and separate from offline recognition.

## Start Over SSH

Follow [the SSH quickstart](docs/QUICKSTART_PI.md) for prerequisites, a lean
`~/swift-clean/me2-vcm` checkout, private setup and opt-in asset downloads.
After installation, from the Pi project directory:

```bash
./swift setup
./swift doctor
./swift run --model champion
```

`run` previews; `run --live` executes enabled actions. Calls/SMS use configured
recipients automatically; add `--confirm-communications` for per-action consent.
Do not say communication commands unless intended. Empty capture is quiet.
Type `:quieter`, `:volume 10`, `:pause` or `:stop` for music control.
See [commands](docs/command-cheatsheet.md), [audio](docs/AUDIO.md),
[troubleshooting](docs/PI_RUNTIME_FEEDBACK.md), and [one-setting updates](docs/QUICKSTART_PI.md#change-one-setting-later).

## Frozen Models

| Component | Repository selection | Status |
| --- | --- | --- |
| Wake word | exp15 CNN + GRU, 204,738 parameters, threshold 0.30 | Preserved default; recent trials did not justify replacement |
| Voice commands | Champion: 6s BC8 + TCN + attentive statistics pooling, 488,264 learned parameters, threshold 0.954 | Frozen default; Pi 5 injected comparison and Champion acoustic benchmark recorded |
| Voice-command baseline | Baseline (2s BC8), 327,750 parameters, threshold 0.875 | Complete selectable fallback; historical run_043 lineage |

Frozen exports, checkpoints and matching configuration are published in the
[v1.0.0 weights release](https://github.com/ercallagon/ai231/releases/tag/me2-weights-v1.0.0).
The archive and all 17 payload files were checksum-verified against its source
tag; see [release verification](reports/WEIGHTS_RELEASE_20261017.json).
Weight use follows [MODEL_TERMS.md](MODEL_TERMS.md), separately from the code licence.

BC8 names this BC-ResNet-8-style width configuration, **not eight layers**:
the backbone has 12 broadcast-residual blocks. Champion adds three temporal
blocks and masked attentive mean/std pooling. See [architecture](docs/architecture.md).
Six seconds is capacity, not mandatory waiting. Historical IDs (`selected`,
`run_043`, `exp15`) remain provenance, not extra public VCM choices. No causal
encoder, KV-cache, ASR decoder or pretrained detector is claimed.

## Final Pi Measurements

Use [FINAL_PI_VALIDATION.md](docs/FINAL_PI_VALIDATION.md) for safe updates,
preflight/provenance, injected evaluation, acoustic comparison and retrieval.
The live default is 800 ms, measured in the Champion Pi 5 acoustic run. Use an
explicit longer endpoint (for example 1200 ms) when commands contain long pauses;
this trades responsiveness for pause tolerance and was not compared in a paired
endpoint experiment.

```bash
./swift evaluate --reset
./swift live-latency --model champion
./swift report
```

Do not reset results during a benchmark. Injection bypasses wake/capture/actions.
[METRICS.md](docs/METRICS.md) separates model-only, feature+ONNX, decision and
acoustic timing. The frozen-test comparison includes Baseline and Champion; the
class acoustic benchmark was Champion-only. See
[the acoustic report](reports/PI5_ACOUSTIC_CHAMPION_20261017.md) for results and
caveats. No acoustic Baseline comparison is claimed.
[Candidate audit](reports/FINAL_CANDIDATE.md) lists evidence and limitations.

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
The default fetch is about 305 MB of evaluation Parquet, with audio embedded;
it does not fetch training data or replace model files. Injection bypasses wake
detection, the microphone and live endpointing. It is supplementary to the
speaker-to-microphone class benchmark, not an acoustic test substitute.

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

## Rollback

Use the complete fallback without modifying the working installation:

```bash
./swift run --model baseline
./swift run --model champion --end-silence-ms 1200
```

Preview is the default. Calls/SMS are not required for benchmark testing.
The SSH workflow isolates private setup and inherited secrets; preserve any
locally adjusted cue. Omitting the endpoint flag uses the current 800 ms runtime default.

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
recording publication/consent terms remain unresolved. The original exp15
PyTorch checkpoint has been recovered and verified against the deployed ONNX;
it is retained with the feature source, configuration, corrected metrics and
a clearly distinct reproduction history. Its selected epoch is 42, not the
incorrect best_epoch 60 recorded in the preserved historical config.

## Evidence

- experiments/selected/: selected checkpoint, architecture/frontend and calibration.
- experiments/baseline/: run_043 checkpoint, config and comparison evidence.
- experiments/wakeword/: exact artifact provenance and corrected wake evaluation.
- logs/training/: complete selected and baseline training logs.
- experiments/RUN_INDEX.csv: historical trial inventory; incompatible
  protocols and unknown/incomplete results are marked rather than ranked together.
- reports/MODEL_SELECTION.md: validation selection and calibration history.
- reports/SUBMISSION_CHECKLIST.md: submission criteria, evidence and evaluation scope.
- reports/PUBLICATION_STATUS.md: current publication scope versus historical receipts.
- reports/PI5_ACOUSTIC_CHAMPION_20261017.md: Champion acoustic results, timing scopes
  and recorded conditions.
- scripts/benchmark_onnx_pi.py: model-only p95/thread measurements.
- scripts/reproduce.py: injected-file feature/model recognition and RTF on host/Pi.
- docs/BENCHMARK.md: class laptop-speaker -> Pi-microphone benchmark.

Verification comprises host regressions with mocked services/peripherals,
frozen Pi injection, Champion acoustic results at the 800 ms endpoint, and
operator GPIO/audio observations. The [candidate audit](reports/FINAL_CANDIDATE.md)
and [submission evidence](reports/SUBMISSION_CHECKLIST.md) define the measured
scope and interpretation.

Opt-in wake trials did not justify replacing exp15; see
[wake evidence](reports/WAKE_EXP15_STREAMFT.md). They are not extra public VCM
model choices. The [documentation index](docs/README.md) separates active guides
from preserved historical recipes. No further training is queued.
