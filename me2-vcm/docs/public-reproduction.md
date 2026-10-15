# Public Setup and Retraining

> Historical research guide. Current frozen defaults, installation, rollback and
> reproduction commands are in README.md, docs/PI5.md and docs/BENCHMARK.md.
> Old artifact tags, package paths and readiness snapshots below are not current
> release instructions. Physical Pi 5 verification of the six-second profile is pending.

## Status and Two Separate Workflows

One-command reproduction is a release gate, not complete yet. The GitHub target
is `ercallagon/ai231/me2-vcm`. Public artifacts, wake-data retrieval/consent and
full clean-environment installation are still pending. Nothing here downloads
unpublished models or claims that placeholder URLs are working release links.

1. **Use published trained artifacts:** install inference dependencies, fetch
   checksum-pinned model/config/labels and authorized audio assets, run preflight,
   then start the live loop. This does not train or require PyTorch on the Pi.
2. **Retrain for review:** fetch the pinned HF data, rebuild authenticated
   features, validate the unchanged protocol, train BC8 from random weights into
   a fresh tag, and review validation results before calibration/test/deployment.

The professor can choose the second option. It reproduces a controlled BC8
training experiment, not a claim of bit-identical run_043 weights. The current
run_043 includes run_042 training followed by source-balanced fine-tuning; its
full historical lineage remains separately documented. New experiments cannot
use external pretrained or champion initialization. Training is recommended on
an approved GPU host, not Pi CPU; no SLURM is used.

## Public Configuration Without Changing the Working Pi

`scripts/stage_public_source.py` creates a fresh review snapshot under `work/`.
Only its generated `src/hey_swift/integration/config.py` clears the three known
private contact/device defaults; the live sandbox source and Pi install remain
unchanged. It copies no model/audio/account state and rejects exact known
private values remaining in the snapshot. This is not a comprehensive secret
or consent audit, so the result is not automatically approved for publication.

`configs/public-runtime.env.example` lists private environment settings. Public
users must configure their own authorized recipients/paired device privately;
empty defaults prevent shipping the original user's phone/device identity.
Never use real calls/SMS as setup tests. Use preview actions and mock hardware.

## Runtime Asset Contract

`scripts/prepare_runtime_assets.py` takes a JSON manifest and a fresh staging
directory. Every asset requires exact path, SHA-256, size, reviewed terms,
`use_authorized: true` and exactly one local file or HTTPS URL. Network use is
explicit (`--allow-network`); embedded URL credentials are refused. Failed
stages remain for inspection; existing destinations are never overwritten.

```json
{
  "assets": [{
    "path": "integration/assets/music/my_owned_demo.mp3",
    "local_file": "PATH_TO_YOUR_OWNED_OR_LICENSED_MP3",
    "sha256": "REPLACE_WITH_64_CHARACTER_SHA256",
    "bytes": 12345,
    "terms_reference": "YOUR_LICENSE_OR_OWNERSHIP_RECORD",
    "use_authorized": true
  }]
}
```

```bash
python scripts/prepare_runtime_assets.py \
  --manifest YOUR_REVIEWED_ASSET_MANIFEST.json --output work/assets_unique
```

The example intentionally fails until real hashes/sizes/terms are supplied.
Use a manifest outside Git if it contains private source paths. The staged
receipt contains hashes and terms references, not saved accounts or credentials.
Asset preparation alone does not validate model/config/label compatibility;
run the existing deployment preflight on the assembled release afterwards.

Required assets include selected VCM ONNX + runtime labels/config, exp15 wake
ONNX + config, a cue, Amy ONNX + JSON when speech is enabled, and authorized
MP3s when demonstrating offline music. Current personal MP3s and private wake
recordings are not redistributed by the source snapshot. Do not overwrite the
Pi's locally amplified cue. Public users can provide their own licensed music.

Amy's [upstream model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/amy/medium/MODEL_CARD)
points to underlying dataset terms; the repository-level MIT label alone does
not resolve all voice distribution terms. Piper is the existing speech asset,
not initialization for the scratch wakeword/VCM classifiers. Public retrieval
URLs and licensing approvals are not invented here.

## Reviewer BC8 Retraining Commands

For the compact Git layout, first restore the separately published, hash-verified
metadata archive as described in compact-publication.md. This source tree does
not contain those manifests. After source and metadata are available, from
`ai231/me2-vcm` in an isolated Python environment:

```bash
python -m pip install -e '.[train,data,cli,dev]'
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/01_download_hf.py
python scripts/01_download_hf.py --verify
# The snapshot includes the frozen protocol metadata; do not regenerate splits.
# Use only if this feature-cache directory does NOT exist already:
python scripts/prepare_vcm_protocol.py --name hf_grouped_v2b --cache --workers 2
python scripts/validate_vcm_protocol.py --config configs/vcm_hf_grouped_v2b_bc8.yaml
# Safe plan only: no CUDA, dataset loading, training or files created.
python scripts/retrain_public_bc8.py --tag reviewer_bc8_s42_unique --gpu 0
# Actual training requires explicit opt-in and an approved available GPU:
python -u scripts/retrain_public_bc8.py --tag reviewer_bc8_s42_unique --gpu 0 --execute
```

The cache builder reads all listed partitions for deterministic feature
preparation/parity, but does not train on test/holdout or generate predictions.
The wrapper validates data first, checks the selected GPU twice 10 seconds
apart (<40% utilization, >=20 GiB free), exposes only that GPU, and invokes the
existing audited random-init trainer. It is not a resource reservation or an
autonomous queue; a failed gate needs a later explicit retry. It never runs final
test/holdout evaluation or promotes a model. A fresh tag is mandatory.

Logs, config, optimizer/RNG checkpoints, environment/Git records, per-class
validation and best/last checkpoints are retained under `experiments/vcm/<tag>/`.
After training, freeze selection and use the architecture-specific export and
calibration procedure; do not send BC8 through the historical generic exporter.
Keep validation, calibration, test and live holdout roles distinct.

These commands are prepared and imports/planning are tested, but the full HF
fetch/cache/retraining recipe has not been rerun in a fresh public installation.
Do not advertise it as verified one-command reproduction yet. Wake retraining
still needs a publishable, licensed data recipe and corrected audit independent
of the archived checkpoint. Separate baseline reproduction from new candidates.
