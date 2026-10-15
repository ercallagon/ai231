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
trained artifact; its original PyTorch checkpoint was not found in verified
sources. A separately documented exp15 reproduction is not the same checkpoint.
