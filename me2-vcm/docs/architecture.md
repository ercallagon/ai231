# Architecture

## Current Repository Selection

Hey Swift runs wake detection and command recognition locally on Raspberry Pi 5.
The repository defaults are exp15 WakeGRU and the frozen six-second
BC8/temporal-CNN/attentive-pooling VCM at threshold 0.954. The complete run_043
two-second profile remains a selectable fallback at 0.875. Existing working Pi
installations are not overwritten. Physical verification of the repository
selection remains separate from host regression tests.

```text
microphone -> continuous resampling to 16 kHz -> sliding wake detection
  -> wake accepted -> cue with capture suppression -> endpointed command capture
  -> retained-input duration guard -> 80-mel features + padding mask
  -> BC8 -> temporal CNN -> attentive pooling -> intent/slot heads
  -> calibrated confidence/OOS gate -> exact slot translation -> action router
  -> action feedback/TTS -> playback-tail guard -> fresh idle wake listening
```

The tested USB microphone captures at 48 kHz. This is a tested device property,
not a universal Pi microphone limitation. The loop probes supported rates and
resamples to the models' 16 kHz input rate.

## Models and Features

| Component | Architecture | Input | Parameters |
|---|---|---|---:|
| exp15 wake | CNN channels 16/32/64, GRU-128, FC-64, binary head | 1 s, static+delta 40-mel, `(batch,80,100)` | 204,738 |
| selected VCM | BC8 + residual temporal CNN + masked attentive statistics pooling | Up to 6 s, `(batch,1,80,600)` plus `(batch,1,600)` mask | 488,264 |
| run_043 VCM | BC-ResNet8 broadcast CNN, shared embedding, intent/OOS and six slot heads | 2 s, 80-mel, `(batch,1,80,198)` | 327,750 |

VCM uses FFT 400, hop 160, and NumPy/SciPy preprocessing shared with training.
Both profiles have 20 intent outputs: 19 supported intents plus OUT_OF_SCOPE.
Each has six slot heads with three supported values each; the selected
TEMPERATURE head adds REJECT. This explicit class does not guarantee that every
unsupported spoken value is recognized. Report measured rejection errors.

The selected backbone is followed by a 256-to-128 projection and three residual
depthwise temporal blocks (kernel 5, dilations 1/3/9). Masked attention computes
weighted mean and standard deviation; a 256-dimensional embedding feeds the
intent and slot classifiers. Temporal padding is symmetric, not causal. The
mask prevents padded frames contributing directly to pooling; it does not make
the encoder invariant to all padding context. No pretrained detector is loaded.
Loss is weighted intent cross-entropy plus applicable slot cross-entropy,
not CTC or attention sequence decoding.

BC8 means this repository's BC-ResNet-8-style width configuration, not eight
layers. The backbone has 12 broadcast-residual blocks: stage repeats 2/2/4/4
at 64/96/128/160 channels, with 128-channel stem and 256-channel tail.
Each block combines a frequency depthwise branch/SubSpectralNorm and a
mean-frequency temporal branch broadcast back to the frequency grid. This is
an implementation adaptation, not an assertion of an unchanged paper model.

Wake detection uses 1-second windows, a 0.25-second hop and overlapping scoring
windows. Preserved gates include probability 0.30, sustain 2, cooldown 2 seconds
and an RMS floor. It is streaming window classification, not a causal
Transformer or a KV-cache system.

The selected graph includes temperature scaling once, with temperature
0.6499874044563441 and confidence threshold 0.954. The run_043 graph separately
uses temperature 0.6475641442117414 and threshold 0.875.
Runtime applies softmax without a second temperature division. Intent accuracy,
conditional slot accuracy, raw strict E2E, and gated command success are distinct
metrics. See [submission readiness](submission.md).

## Runtime Safety

Capture endpoints on trailing silence or bounded limits; total capture length
and retained input length are different. Retained input exceeding capacity is
rejected rather than silently truncated. Complete-word retention is not manually
verified. The selected profile uses 200 ms pre-roll, sustained onset and a 1.2 s
silence endpoint; six seconds is capacity, not mandatory waiting time. A 7.4 s
capture safety deadline permits endpoint observation but rejects incomplete
captures. See [PI5.md](PI5.md) for current settings and separate live tests.

The router requires an accepted supported intent and an exact supported slot.
Numeric model slots accept canonical strings and legacy integers through explicit
translation; unsupported values fail closed. Cue/TTS capture suppression and
music ducking are half-duplex control, not acoustic echo cancellation. Experimental
AEC/PCEN recordings do not establish that deployment has verified AEC.

## Modules

- `wakeword/`: features, CNN/GRU training, sliding ONNX detector.
- `vcm/`: audited data protocol, broadcast model, training, evaluation and ONNX.
- `audio/`: capture/resampling, endpointing, recovery, playback and speech.
- `integration/`: live E2E loop, calibrated model selection, typed routing.
- `actions/services/`: offline/device actions and bounded external services.
- `hardware/`: mock hardware and the tested Pi GPIO adapter.

Network-dependent weather, Spotify and phone services are not offline model
inference. Their fallback/status behavior is documented separately. Preserve
RGB BCM17/27/22 and TM1637 BCM23/24; do not infer pin changes from legacy diagrams.

Historical architecture configurations remain available for reproducibility.
Three-second BC8/Conformer and scratch wake frontend trials are experimental,
with separate input/slot contracts and no automatic deployment.
