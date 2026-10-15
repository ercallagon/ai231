# Raspberry Pi 5 deployment bundle

For the run_043 v5 installation, use the selective code-only update described in
[`docs/rpi-runtime-v6.md`](../../docs/rpi-runtime-v6.md), not a fresh installation.
The updater checks baseline hashes, backs up affected files and preserves models,
calibration, GPIO pins, Spotify state, venv and the locally amplified chime.

Sandbox directory policy: `deploy/rpi/` holds final **unzipped** deployment files
and deployment documentation only. Downloadable `.tar.gz` files and their
checksum receipts stay under `work/`. Selective patches are assembled under
`work/rpi_patch_staging/`, then published here as complete unpacked releases.
Audits, regression logs, backups and installer rehearsals also remain under
`work/`. The selective packager refuses existing release names.

This directory is staged by `scripts/05_stage_rpi.py` (see `docs/raspberry-pi.md`).
It contains the lean, CPU-only artifacts needed to run Hey Swift on the Pi:

- `hey-swift.service` — systemd auto-start unit
- `models/wakeword.onnx` — wake-word detector (~204k params)
- `models/vcm_<tag>.onnx` — VCM intent+slot model
- `<tag>.labels.json` — intent order, slot sets, OOS τ
- `thresholds.json` — wake gates (threshold / sustain / cooldown / RMS)
- `run_live.py` — thin launcher

Nothing here is committed by default (models are gitignored); stage it fresh
after each export so the bundle always matches the latest tag.
