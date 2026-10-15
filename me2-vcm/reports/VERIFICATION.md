# Verification status

The selected six-second profile at threshold 0.954 is the repository default.
Pi 5 hardware/acoustic verification is explicitly pending and is not a blocker
to this user-authorized initial source publication.

Completed host checks:

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
- Optional retraining command was checked in planning mode only. This task
  did not launch training or use SLURM.

Full source regression and clean-export reproduction receipts are retained in
the task's work/finalize_submit/ directory; final commit audit records their
exact counts and results. Some hardware/data-dependent tests skip on HPC.

Remaining device checks: stable 48 kHz mic acquisition/resampling, capture
handoff/recovery, pre-roll/quiet speech, new pause-tolerant endpoint latency,
six-second/over-limit behavior, wake false triggers during music, GPIO/timer
side effects, action responsiveness and optional speech assets. The original
working Pi installation, chime, accounts and wiring are preserved.

Actual Pi p95, RTF, thread settings and scope must be collected with the posted
commands. Until returned, no host timing is described as Pi performance. No
fresh independent speaker recordings are available. Wake training data/consent
and the selected exp15 original PyTorch checkpoint remain incomplete provenance.
