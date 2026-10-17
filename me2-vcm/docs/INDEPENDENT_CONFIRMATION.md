# Fresh speaker confirmation (not yet collected)

Fresh independent-speaker confirmation recordings and their consent records
have not been collected. The existing class holdout was examined before this
release and does not provide independent unseen confirmation. The procedure
below specifies a separate confirmation study; no results from it are reported.

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
