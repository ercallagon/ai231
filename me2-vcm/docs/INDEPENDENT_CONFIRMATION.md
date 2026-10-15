# Fresh speaker confirmation (not yet collected)

No fresh recordings or consent documents are available to this agent. Do not
describe the current class holdout as independently unseen: its results were
examined before this release. The following is a prospective confirmation plan,
not a measured result or another tuning opportunity.

Freeze this commit, model/config/label hashes, threshold 0.954, wake settings and
capture profile before collecting any new recordings. Recruit at least three
consenting speakers absent from all prior sources. Use private random aliases;
keep consent and identity documents outside Git.

For each speaker record every supported semantic combination: thirteen fixed
intents and eighteen intent/slot combinations (three values for each of six
slotted intents), plus at least twenty unrelated/unsupported utterances. Include
18/22/26-degree phrasing, short and naturally paused commands, quiet and moderate
music conditions. Do not selectively repeat only failures or omit rejects.

Reserve all recordings for this confirmation split. Do not add them to train,
validation, augmentation, speaker adaptation or calibration. Store original audio
privately, source/consent, timestamps, hashes, conditions, all decisions and
failures. Report recognition, correct accepted intent+slot, OOS false accepts,
false rejects and timing with denominators. If a later development iteration is
needed, retire this set as exposed and collect a different confirmation set.

This repository deliberately includes neither recordings nor fabricated reviewer
annotations. Real acoustic capture and Pi timing still require the actual device.
