# Frozen dataset metadata

Dataset: airimonda/ai231-me2-voice-commands.
Revision: bfca0bbebfe17e00d6a685d1da573615d29ecdce.
Citation: AI231 ME2 Voice Commands, Hugging Face dataset,
https://doi.org/10.57967/hf/10723 (cite the pinned revision as well).

This repository publishes metadata, not recordings. The dataset has mixed
source licences and research/education-only restrictions. Read DATASET_LICENSE.md;
group recordings must not be redistributed outside the class without speaker
consent. Supplemental cloned-voice consent/licensing gaps remain unresolved.

The hf_grouped_v2b manifests retain 25,366 training rows, 2,922 validation rows,
400 calibration OOS rows, 1,171 historical challenge rows, 4,693 test rows and
202 holdout rows. The selected classifier used 25,330 eligible training rows.
training_exclusions.json records all 36 exclusions; unsupported_overlay.json
preserves the audited unsupported-value handling. No new split was generated.

manifests/*.csv retain original row order, opaque audio IDs, intent/slot labels,
source, is_synthetic, hashes and Parquet offsets. Unneeded identifying fields
are removed and speaker_group is a deterministic pseudonym. PROVENANCE.json
records original and public manifest hashes, split counts/hours and declared
speaker-group counts. Those counts include synthetic voices and are not verified
counts of independent human speakers.

Raw audio remains embedded in the publisher's Parquet files. reproduce.py fetches
only files needed by the requested split and verifies publisher file hashes,
embedded audio bytes and decoded PCM. Existing caches can be reused. Datasets
and caches stay ignored under data/vcm/raw/ or a user-selected --data-root.

Test and holdout have been repeatedly inspected/evaluated during research. They
are previously exposed evaluation sets. The frozen 0.954 gate was refined only
on validation/tune-OOS after that exposure; no newly unseen confirmation is claimed.
Do not tune on test/holdout results. See docs/INDEPENDENT_CONFIRMATION.md for the
prospective consented-speaker plan.
