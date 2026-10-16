# Training

> Historical research guide. Current frozen defaults, installation, rollback and
> reproduction commands are in README.md, docs/PI5.md and docs/BENCHMARK.md.
> Old artifact tags, package paths and readiness snapshots below are not current
> release instructions. Champion Pi 5 acoustic results are reported in
> ../reports/PI5_ACOUSTIC_CHAMPION_20261017.md; device and licensing limits remain.

Reproducible training procedure for both subsystems.

## Current VCM Training

Use [the corrected HF-aligned preparation and training guide](vcm-hf-aligned.md)
for new experiments. The A0 procedures and experiment matrix below are
historical, not the current split contract. Existing models/results remain
untouched; new corrected A10/A6 trials train from scratch with fresh tags.

## Prerequisites

```bash
# data fetch, CLI and training dependencies (never needed for Pi deployment)
pip install -e ".[data,cli,train]"
```

## VCM — A0 baseline (first run)

```bash
python scripts/01_download_hf.py            # cache HF parquet (SHA-pinned)
python scripts/02_build_manifests.py        # -> data/vcm/manifests/ (committed)
python -m hey_swift.cli train vcm --tag a0_scratch
python -m hey_swift.cli eval vcm --tag a0_scratch
python scripts/03_export_onnx.py --tag a0_scratch
python scripts/04_parity_check.py --tag a0_scratch
```

Or in one shot: `./run.sh`.

**Locked hyperparameters** (`configs/vcm.yaml`): seed 42, 30 epochs, batch 64,
lr 1e-3, Adam, early-stop patience 6, 2.0 s window, 80-mel, `slot_reject: false`,
OOS τ = 0.5.

**What a run writes** (immutable, under `experiments/vcm/a0_scratch/`):
- `reproducibility.json` — seed, config, HF revision, split counts, disjointness proof
- `best.pt` — checkpoint
- `history.json` — per-epoch loss
- `eval_report.json` — intent acc, per-intent F1, OOS-FPR, slot acc (test + holdout)

## Experiment matrix (pre-registered)

| Tag | Arch | Params (est.) | Question |
|---|---|---:|---|
| **a0_scratch** | DS-CNN (32,64,128,256) | ~57k | Baseline: is current capacity enough for 19 intents? |
| a1_wider | DS-CNN (64,128,256,256) | ~123k | Does more capacity fix within-family separation? |
| a2_tcresnet | TC-ResNet w32 | ~131k | Does residual temporal conv help slot-at-tail? |

Each run answers one question. A re-run of a question gets a **new tag**
(immutable artifacts, D-008).

## Wake-word

```bash
python -m hey_swift.cli train wakeword --tag ww_base
python -m hey_swift.cli eval wakeword --tag ww_base
```

Reports accuracy, FAR (false accepts), FRR (false rejects) on the frozen test
split, plus a threshold scan.

## Carry-forward lessons (from prior training)

**Wake-word (from wakeword_exp15):**
- **Sustain 2 hops** — a single qualifying hop is noise; require 2 consecutive.
- **Cooldown 2.0 s** — suppresses double-fires on a sustained phrase.
- **RMS floor −60 dB** — ignores silence/mic-hiss hops before the model runs.
- **Cadence fix** — the original live runner mis-counted hops; the streaming
  `WakeWordDetector.detect()` contract (one hop in, fire-or-not out) removes it.
- **NumPy-only inference** — the Pi has no torch; the feature path + ONNX model
  are CPU-clean.

**VCM (from the 10-intent → 19-intent transition):**
- **Direct audio, not ASR→NLU** (D-009) — simpler, on-device, faster to converge.
- **2.0 s window** (D-004) — 1.5 s truncates slotted commands.
- **Learned OOS + confidence gate** (D-003) — two independent rejection signals.
- **From-scratch phase 1** (D-006) — no weight transfer; transfer is a later
  controlled experiment.
- **Frozen test/holdout never trained** (D-007) — eval integrity.

## Evaluation metrics

| Metric | Meaning |
|---|---|
| intent_acc | fraction of accepted in-scope utterances with correct intent |
| per-intent F1 | precision/recall/F1 per intent (catches within-family confusion) |
| OOS detection rate | true OOS correctly rejected |
| OOS false-positive rate | in-scope utterances wrongly rejected |
| slot_acc | correct slot value among slotted-intent utterances |
| FAR / FRR (wake) | false accepts / false rejects of the wake phrase |
