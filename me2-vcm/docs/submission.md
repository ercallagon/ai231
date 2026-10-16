# Submission Readiness

> Historical research guide. Current frozen defaults, installation, rollback and
> reproduction commands are in README.md, docs/PI5.md and docs/BENCHMARK.md.
> Old artifact tags, package paths and readiness snapshots below are not current
> release instructions. Champion Pi 5 acoustic evidence is in
> ../reports/PI5_ACOUSTIC_CHAMPION_20261017.md; remaining limits are listed there.

## Current submission evidence, 2026-10-15

Use [the submission checklist](../reports/SUBMISSION_CHECKLIST.md),
[frozen results](../reports/RESULTS.md), [architecture](architecture.md),
[Pi 5 guide](PI5.md) and [publication status](../reports/PUBLICATION_STATUS.md).
The public repository exists and clean VCM result reproduction has passed.
The selected six-second model at 0.954, complete run_043 fallback at 0.875 and
exp15 wake graph are committed. Original exp15 checkpoint/provenance recovery
is included in the current update; recent trials did not replace exp15.
Actual Pi measurements, independent real-speaker confirmation and outstanding
dataset/voice rights limitations must not be represented as complete.

## Historical planning notes

Everything below records the earlier preparation state, including superseded
run.sh behavior, candidate choices and not-yet-published status. It is retained
for research history, not as current installation or submission instructions.

Public configuration, asset staging and professor retraining options are
documented in [public reproduction](public-reproduction.md).

Target: `https://github.com/ercallagon/ai231`, project directory `me2-vcm/`.
The authoritative source remains the existing sandbox project. The local Git
root is its parent sandbox; never stage the entire parent. Public publication
and one-command reproduction have not yet been verified.

## Six Deck Requirements

| Requirement | Prepared evidence | Remaining gate |
|---|---|---|
| Public repo / one-command reproduction | source package, pinned VCM fetch, versioned split preparation | allowlisted clean checkout, asset/wake data retrieval, tested scratch driver, actual push |
| Licensed/citable dataset | VCM DOI 10.57967/hf/10723 and pinned per-source terms | wake recordings/voice/music terms and consent; model distribution review |
| Logs/final checkpoint committed | immutable champion logs/checkpoint/configs/hashes | final selection, sanitized selected logs, explicit checkpoint exception and commit |
| Posted Pi latency script | CPU-only `scripts/measure_pi_onnx.py` | physical Pi 5 execution plus live speech-end-to-action benchmark |
| Held-out/unseen speakers | authenticated grouped-v2b manifests and lineage/PCM audit | publish reproducible audit, disclose legacy exposure, frozen official live evaluation |
| Comparable-size baseline | completed same-test comparison: run_043 327,750 parameters, 3s BC8 328,007, Conformer 662,445 | finalize selection; physical Pi efficiency and acoustic comparison |

Pi 5 replaces the template's Pi 4 wording by user agreement. Do not invent Pi 4
results, causal encoders, CTC objectives, KV caches or speech transcripts. The
system performs hierarchical intent and closed-set slot classification.

## Publication Policy

Include reviewed source, selected configs, tests, dependencies, documentation,
split fingerprints/recipes and small summary reports. Include only selected
checkpoint/log artifacts through an explicit exception after final review.
Historical runs remain preserved locally; do not publish all of `experiments/`.

Exclude intermediate work, raw WAV/Parquet, feature caches, backup deployment
trees, credential/account state, private recipient/device identifiers, archives
and unrelated parent files. Keep final unzipped Pi deployments in `deploy/rpi/`
and temporary release assembly/tars under `work/`. Model/voice assets require
separate verified fetch instructions and distribution terms.

`scripts/prepare_submission_inventory.py --output work/<unique-name>` produces
draft candidate-file hashes, baseline hashes and deck metric summaries. It does
not copy files, stage Git, scan all secrets, evaluate holdout, or authorize
publication. Candidate files still require privacy/licensing review. A later
clean target checkout under `work/` can receive the approved snapshot while
preserving ME1. Do not create a second authoritative project or initialize a
nested Git repository in the running source tree.

## Reproduction Gates

The current `run.sh` reproduces a historical A10 pipeline, not run_043. The
preserved run_043 refines run_042; record the full parent training lineage rather
than claiming run_043 was randomly initialized. The recent scratch trials
started randomly, with no external pretrained or champion-initialized weights.

The scratch trials and frozen VCM comparison completed on 2026-10-13. See
`work/experiments/vcm_final_comparison_20261013_v1/README.md`: all 4,693 test
rows were hash verified and policies frozen before inference. The 3s BC8
candidate improved raw intent to 93.27% and raw intent+slot to 92.90%, but is
not automatically promoted. Its optimized policy improved real supported
success while accepting more OOS. The Conformer configuration did not win.
Wake gate/metric review is under `work/wake_gate_audit_20261013_v1/`.
These local work paths are evidence locations, not promises that they are
included in a clean public checkout; curated summaries belong under reports/.

Two future entry points are needed: setup/inference from published trained
artifacts, and full training from random initialization with pinned datasets.
Saved-log evaluation does not substitute for rerunning a clean-checkout recipe.
Do not retrain just to prepare docs or change live jobs' protected scripts.

Blockers include a wakeword trial dependency on `../archive-me2/`, incomplete
wake-data retrieval/consent, speech/music assets currently under ignored work,
private contact defaults, and final runtime-config/label/model retrieval.
Pi pairing/accounts and physical wiring are explicit manual prerequisites.

## Metrics and Timing

Report raw intent, raw strict intent+slot, conditional slot accuracy, and gated
command correctness separately. For run_043's existing 4,693-row test report:
raw intent 92.67%, raw strict E2E 92.24%, conditional slot 97.68%; deployed gated
intent 90.60%, gated strict E2E 90.30%. These are offline results, not live Pi
accuracy. Wake clip FAR is not continuous false activations/hour. Validation
scores from new frontends/windows/slot targets require matched comparisons.

The new timing script measures feature extraction, ONNX forward and their sum,
reporting samples, mean, p50, p95 and RTF. It uses an explicit CPU session with
selected threads, leaving production session behavior unchanged. RTF divides by
the actual input duration; also record the fixed model window. It excludes
capture/endpoint waiting, routing, network services, TTS and physical actuation.
The official live benchmark supplies a different latency scope; present both,
not a host timing proxy as Pi latency or model timing as actuator latency.

After the selected runtime model/config exist on the current Pi installation:

```bash
cd /home/ej/swift-e2e
PYTHONPATH=src .venv/bin/python scripts/measure_pi_onnx.py \
  --model models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.onnx \
  --config configs/vcm_run_043_calibrated.yaml \
  --require-pi --threads 1 --iterations 200 --warmup 30 \
  --output work/pi_onnx_run043_threads1_unique.json
```

The script must first be included in a reviewed selective Pi update; it has not
been transferred/deployed automatically. Without `--audio`, the probe is
synthetic zero input and yields timing evidence only. For speech timing, add
`--audio <development-clip.wav>` using complete speech within the model window;
long input is rejected rather than silently truncated. Never use the official
holdout for setup or tuning. Use fresh output paths; existing results are refused.

Before final live testing, freeze model/config/mapping/gate hashes. Follow the
prepared benchmark readiness procedure, preserve previous baselines, disclose
historical test exposure and collect independent Pi development failures first.
If official holdout results inform changes, evaluate the revised system on
fresh independent data rather than claiming that holdout remained untouched.
