# Archived Public-Reproduction Proposal

The earlier preparation proposal is preserved at
[commit 9de6bb4](https://github.com/ercallagon/ai231/blob/9de6bb4eb0bf309b1166969748875e3d78cb49e7/me2-vcm/docs/public-reproduction.md).
It describes a source-staging design and private research-workspace assumptions,
not the current installed workflow.

The current repository commits both VCM exports, selected checkpoints/logs and
permitted manifests. Use:

- [README: frozen result reproduction](../README.md#one-command-result-reproduction)
- [README: optional scratch retraining](../README.md#optional-scratch-retraining)
- [SSH quickstart](QUICKSTART_PI.md)
- [Dependency and platform contracts](DEPENDENCIES.md)
- [Dataset terms](../data/DATASET_LICENSE.md) and [model terms](../MODEL_TERMS.md)

Frozen VCM evaluation can be reproduced from pinned evaluation Parquet.
Full public wakeword retraining is not available because raw-recording access
and consent records are incomplete. This limitation is distinct from loading
the committed exp15 checkpoint/export or reproducing VCM results.
