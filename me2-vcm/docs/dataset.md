# Dataset

Provenance, taxonomy, and splits for both subsystems.

## VCM dataset

**Source:** HuggingFace `airimonda/ai231-me2-voice-commands`
**Pinned revision:** `bfca0bbebfe17e00d6a685d1da573615d29ecdce`
(hard-coded in `scripts/01_download_hf.py`; a re-run pulls byte-identical data.)

### Intents (19 + OOS = 20-way head)

```
 0 PLAY_MUSIC       7 STOP            14 ALARM
 1 WEATHER          8 NEXT            15 TEMPERATURE
 2 TIME             9 VOLUME_UP       16 BRIGHTNESS
 3 LIGHT_ON        10 VOLUME_DOWN     17 COLOR
 4 LIGHT_OFF       11 CALL            18 CREATE_REMINDER
 5 PAUSE           12 MESSAGE         19 OUT_OF_SCOPE
 6 LIST_REMINDERS  13 TIMER
```

### Closed-set slots (3-way when `slot_reject: false`)

| Intent | Supported values |
|---|---|
| TIMER | `10s`, `30s`, `1m` |
| ALARM | `6AM`, `8AM`, `9PM` |
| TEMPERATURE | `18`, `22`, `26` (°C) |
| BRIGHTNESS | `20`, `60`, `100` (%) |
| COLOR | `red`, `blue`, `green` |
| CREATE_REMINDER | `drink_water`, `study`, `exercise` |

With `slot_reject: true`, a head retains `REJECT` only when that value is
explicitly listed in its slot set. The 158 empty-slot proxy rows in HF are thin/uneven
(COLOR = 0), which is why A0 runs 3-way.

### Historical Splits (Not Current Grouped Protocol)

| Split | Rows | Role |
|---|---:|---|
| train | (from HF train, minus val carve) | training |
| val | ~10% of train (speaker-stratified) | τ tuning, early-stop |
| test (frozen) | 4,443 | **eval-only, never trained** |
| holdout (untouched) | 202 | **eval-only, final verification** |
| numerals | 66,390 | all-OOS pressure/eval |

`scripts/02_build_manifests.py` writes the split CSVs + `splits.json` (with the
disjointness proof) to `data/vcm/manifests/`. **Only that directory is
committed.**

## Corrected VCM Protocol (Current)

The split description above is historical. The preserved run_043 uses
[the grouped-v2 guide](vcm-grouped-v2.md), versioned manifests under
`data/vcm/protocols/hf_grouped_v2b/`, and the matching source-aware feature cache.
Do not overwrite the historical `data/vcm/manifests/` directory or reuse its
filename-indexed caches for corrected trials. HF has no validation split;
the initial correction preserves the existing local carve while isolating
all its speech identities and dependencies across sources. Grouped-v2b then
selects validation by seeded speech-identity groups and metadata balance, not
audio/model predictions. It retains the old validation as a separate challenge.

| Current partition | Rows | Purpose |
|---|---:|---|
| train | 25,366 | training only |
| val | 2,922 | checkpoint/model development |
| challenge | 1,171 | independent old-validation challenge |
| tune_oos | 400 | rejection threshold selection |
| test | 4,693 | 4,443 core + 250 synthetic negatives; evaluation only |
| holdout | 202 | later live benchmark; not current trial selection |

Training duration is 14.4008 hours, calculated from authenticated `n_samples`
and `sample_rate`, with 287 declared speech identities. These include synthetic
reference identities; they are not 287 independently verified human speakers.
Exact PCM and declared lineage isolation do not prove absence of undeclared
near-duplicates or voice-clone equivalence. Historical test/holdout exposure in
legacy models must be disclosed; do not call every historical result pristine.

## Citation and Terms

VCM dataset DOI: `10.57967/hf/10723`. The pinned dataset `LICENSE.md` states that
the compilation is for research and education only, with per-source terms,
attribution and consent restrictions. Source-code MIT does not relicense the
dataset, its recordings, external voices/music or automatically the trained
weights. Resolve model distribution terms before a public model release.

## Wake-word dataset

**Source:** locally prepared "Hey Swift" positives + mined negatives (~1.2 GB,
16 kHz mono WAV).
**Model carried forward:** `wakeword_exp15_mined_lp` (~204,738 params,
ONNX inference). Manifests under `data/wakeword/manifests/` define train/val/test;
verify label conventions at the dataset/model boundary rather than assuming
positive is output index 1. The exp15 runtime interprets index 0 as wake.

The corrected evaluation audit uses 14,818 train, 1,052 validation and 1,113 test
rows after evaluation deduplication. Original manifests stay immutable. The audit
is currently under ignored `work/wakeword_targeted_v2/`; a reviewed reproducible
audit/recipe must be published before claiming full wake dataset reproduction.
Local raw audio and mined clips need source/license/consent documentation and a
fetch/generation procedure; the VCM DOI does not cover all wakeword recordings.

## Reproducibility contract

```
01_download_hf.py   ──▶  data/vcm/raw/          (gitignored, SHA-pinned)
02_build_manifests.py ─▶  data/vcm/manifests/   (COMMITTED)
configs/vcm.yaml    ──▶  schema + audio + training (COMMITTED)
```

Given the repo, the dataset is fully determined: the HF revision pins the
bytes, the manifests pin the splits, and the config pins the schema.
