# Compact Publication Policy

Keep the current source authoritative and preserve all local historical data,
results and deployments. Compact publication is a reviewed copy, not deletion
or a second authoritative repository. Do not stage the parent sandbox wholesale.

## Git Versus Release Artifacts

- Git: reviewed code, tests, selected/config-history documentation, dependencies,
  public configuration examples, artifact descriptors and split fingerprints.
- Metadata release: immutable grouped-v2b manifests/audit/reserved tables,
  required legacy base manifests and wake manifests/fingerprints. These are
  retained byte-for-byte in a deterministic archive with per-file checksums.
- Selected checkpoint/log evidence: explicit reviewed exception after final
  selection; do not publish every experiment or claim they are committed now.
- Runtime model binaries: immutable artifact URLs/checksums after selection and
  rights review. A preserved reference manifest is not automatic promotion.
- Never publish: private account state, raw private recordings, unlicensed music,
  feature caches, backup trees, sibling sandbox projects or credentials.

`scripts/prepare_compact_publication.py --output work/<fresh-name>` uses the
validated source review snapshot, omits large metadata from the Git proposal,
and creates an independent dataset metadata archive under work. Original files
and earlier snapshots are not modified. Existing destinations are refused.

## Versioned Artifact Catalog

The proposed `models/releases/run043_reference_v1.json` records actual file
sizes/hashes and the matching run_043 model/config/runtime labels, calibrated
temperature/gate, exp15 wake contract, wiring and selected checkpoint/log roles.
It declares no final selected champion. URLs are null and use authorization
false until immutable upload, rights/privacy review and final selection happen.
This intentionally blocks `prepare_runtime_assets.py`; do not bypass it by
inventing URLs or marking pending terms reviewed. A new champion gets a new
catalog version, retaining the original baseline descriptor.

The cue, Amy voice, owned/licensed music and wake raw-data recipe remain explicit
unresolved dependencies. Do not overwrite the current Pi's local chime/account
state or use a public example as the current tested Pi configuration.

## Reproduction Order

Moving metadata out of Git adds a required retrieval step: fetch the immutable
metadata archive, verify its outer SHA-256, inspect safe members, restore only
missing expected paths, and verify every member against METADATA_SHA256.json.
Existing files must match or abort; never overwrite changed manifests or silently
recompute different splits. Then follow the pinned raw-data fetch/cache/validation
and opt-in scratch-training steps in `public-reproduction.md`.

The compact proposal is not yet one-command reproducible: metadata/model URLs,
rights review, safe retrieval integration and a complete clean-checkout run are
pending. Do not run retraining until metadata and authenticated raw/cache inputs
are available. Independent Pi timing/live benchmark checks remain separate.
