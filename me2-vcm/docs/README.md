# Documentation Index

Current public workflow:

- [SSH quickstart](QUICKSTART_PI.md): install, private/session setup and one-setting updates.
- [Final Pi validation](FINAL_PI_VALIDATION.md): safe update, integrity, injected comparison, Champion acoustic run and evidence retrieval.
- [Commands](command-cheatsheet.md): supported intents/values and demo controls.
- [Metrics](METRICS.md): definitions, denominators and measurement scopes.
- [Class benchmark](BENCHMARK.md): pinned external schema and compatibility notes.
- [Architecture](architecture.md): actual implementation, not template claims.
- [Audio](AUDIO.md): independent speech/cue/music controls and troubleshooting.
- [Hardware](hardware.md): tested RGB/TM1637 wiring and supported adapter.
- [Windows](WINDOWS.md) / [macOS](MACOS.md): laptop companions.
- [Candidate audit](../reports/FINAL_CANDIDATE.md) / [submission evidence](../reports/SUBMISSION_CHECKLIST.md).

## Historical Material

[Submission planning](submission.md), [public-reproduction planning](public-reproduction.md)
and [compact-publication proposal](compact-publication.md) are archival references.
They are not installation or evaluation instructions. Their immutable original
versions are linked from those pages. Current reproduction starts in the root
README; current device setup starts in QUICKSTART_PI.md.

`raspberry-pi.md`, `rpi-e2e.md`, and `rpi-runtime-v5.md`/`rpi-runtime-v6.md`
preserve historical installation/patch evidence. They are not instructions to
replace the current private setup or selected model. Experimental wake A/B
guides are opt-in research diagnostics, not extra public VCM model choices.

## Artifact Integrity Maintenance

`python scripts/refresh_artifact_catalog.py` checks the release hashes without
writing files. After reviewing source/documentation changes, `--refresh` updates
their catalog entries and `--include PATH` records an explicit new source file.
The command refuses modified models, configurations, manifests, checkpoints,
training logs, audio and historical JSON/CSV reports. It does not stage, publish
or deploy files. Original experiment and execution receipts remain immutable.
