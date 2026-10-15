# Private-workspace recipe

Exact archived commands (not public raw-data reproduction):

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -B work/wake_exp15_streamft_20261015_v1/trial.py prepare
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -B work/wake_exp15_streamft_20261015_v1/trial.py train
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -B work/wake_exp15_streamft_20261015_v1/complete.py
```

Use a fresh work directory to reproduce; do not overwrite the recorded run.
Requires original private raw recordings, old frozen training manifests and
validation/test feature-cache construction from final_streaming_v1. Epoch
specifications are seed-generated; their exact hashes and PCM replay hashes
are recorded. Full original specifications and PCM cache remain private under
work/. No claim of publicly fetchable wake data or one-command wake retraining.
