# Model and dataset terms

The committed VCM and wakeword weights are provided for research, education and
class demonstration. They are not covered by the source-code MIT grant. No
commercial rights to underlying recordings, voices or datasets are granted.
Observe each source's terms in data/DATASET_LICENSE.md and provide attribution.
The dataset contains material with non-commercial terms and incomplete
consent/licensing documentation; this repository does not resolve those gaps.

No raw human recordings, cloned-voice reference recordings, music or Piper
weights are redistributed here. Public manifests omit identifying fields and
pseudonymize declared speaker groups; they retain the original split membership.
Downloading data from its publisher does not waive its access/consent obligations.

Selected VCM weights were trained from random initialization. The run043 fallback
was trained locally then fine-tuned from the project's run042, not an external
pretrained model. The preserved exp15 wake ONNX is published as its exact frozen
trained artifact. Its original PyTorch checkpoint was recovered and verified
against that export; a separately documented exp15 reproduction is not the same
checkpoint. The optional exp15_streamft_v1 experiment fine-tunes the project's
original exp15, not external pretrained weights. It did not justify replacing
the default and is retained only as research evidence and a diagnostic candidate.
