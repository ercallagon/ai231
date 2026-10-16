# Verification status

The selected six-second profile at threshold 0.954 is the repository default.
Champion Pi 5 acoustic verification is available in
PI5_ACOUSTIC_CHAMPION_20261017.md. This does not certify unseen speakers,
music/AEC robustness, call recovery or all physical actions.

Completed host checks:

- The original exp15 checkpoint has now been recovered and verified against
  the unchanged selected ONNX on batches 1/2/17. Original logs establish best
  epoch 42; the preserved config's best_epoch 60 is a historical metadata bug.
  See experiments/wakeword/exp15_original/CHECKPOINT_EPOCH_CORRECTION.md.
- Shared wake experiments and integer-sample cooldown regressions are retained.
  No candidate justified replacement; exp15 remains default. Optional profiles
  do not change VCM gates, working Pi state or hardware wiring.

- Exact selected and run_043 checkpoints load into their matching architectures:
  488,264 and 327,750 parameters. PyTorch/ONNX logits and all slot heads agree
  on short, intermediate and capacity-limit probes at rtol/atol 2e-4.
- Runtime/metric decisions agree on twelve processed examples per split/profile.
  Temperature is baked once; confidence is not rounded before acceptance.
- Full frozen evaluation at threshold 0.954 completed on validation, tune-OOS,
  test and holdout. No training or gate fitting occurred. Raw failures remain
  in full-manifest denominators. Those evaluation sets were previously exposed.
- Six new tests cover strict capacity, invalid/flat inputs, the 1.08 s pause,
  safe unsupported/OOS rejection and wake -> capture -> model -> router -> idle
  with mocked capture/services. Decision logging precedes service execution.
- Class benchmark parser/schema: 174 host checks passed at pinned commit
  ab39857cf73dd067d068fed388c321ab160dc191.
- Core/evaluation dependencies installed in a fresh local virtual environment.
  Python 3.13 aarch64 wheels for documented Pi core/audio/GPIO/evaluation
  dependencies were resolved/downloaded successfully. This checks wheel
  availability, not execution on that architecture.
- The operator installed the fresh `swift-clean` Pi checkout and `./swift doctor`
  passed actual imports for the core/audio/ONNX/GPIO stack and Piper; the
  documented RGB/display check was operator-confirmed. This is not evidence that
  optional Spotify, call or messaging actions were exercised.
- Optional retraining command was checked in planning mode only. This task
  did not launch training or use SLURM.

Full source regression and clean-export reproduction receipts are retained in
the task's work/finalize_submit/ directory; final commit audit records their
exact counts and results. Some hardware/data-dependent tests skip on HPC.

Remaining device checks: independent-speaker performance, wake false triggers
per hour, music/AEC robustness, complete-word retention through long pauses,
call-end recovery and fresh-OS reproducibility. The 2026-10-17 acoustic run
measured the 800 ms endpoint, six-second model behavior, processed inference and
detector-estimated decision latency under its stated conditions. The original
working Pi installation, chime, accounts and wiring are preserved.

Actual Pi p95, RTF, thread settings and scope are in
PI5_ACOUSTIC_CHAMPION_20261017.md. Do not conflate feature+inference with the
detector-estimated speech-end metric or external response timing. No fresh
independent speaker recordings are available. Wake training data/consent
and public raw-data retrieval remain incomplete; the original exp15 PyTorch
checkpoint itself has been recovered and verified, not substituted by a reproduction.

Current publication status and test receipts are described in
PUBLICATION_STATUS.md. CHECKS.json preserves the initial pre-publication
authentication result; it is not evidence that the public repository is absent.
PUBLICATION_CHECKS.json records the current staged-export regression counts,
clean inference-only contract checks and preserved model/config scope.
