# Raspberry Pi Deployment Directory

For a legacy run_043 v5 installation, use the selective code-only update described in
[`docs/rpi-runtime-v6.md`](../../docs/rpi-runtime-v6.md), not a fresh installation.
The updater checks baseline hashes, backs up affected files and preserves models,
calibration, GPIO pins, Spotify state, venv and the locally amplified chime.

Sandbox directory policy: `deploy/rpi/` holds final **unzipped** deployment files
and deployment documentation only. Downloadable `.tar.gz` files and their
checksum receipts stay under `work/`. Selective patches are assembled under
`work/rpi_patch_staging/`, then published here as complete unpacked releases.
Audits, regression logs, backups and installer rehearsals also remain under
`work/`. The selective packager refuses existing release names.

The current submission uses the Git checkout and `./swift`, not a fabricated
standalone payload in this directory. The committed models/configs live under
`models/` and `configs/`. Start with
[the SSH quickstart](../../docs/QUICKSTART_PI.md), then
[final Pi validation](../../docs/FINAL_PI_VALIDATION.md).
Do not mix legacy bundle instructions with the fresh `swift-clean` workflow.
