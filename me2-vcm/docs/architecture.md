# Architecture

## Preserved Deployment Reference

Hey Swift runs wake detection and command recognition locally on Raspberry Pi 5.
The preserved models are exp15 WakeGRU and
`run_043_hf_grouped_v2b_bc8_sourceft_s42`. New experiment results are candidates,
not automatic promotions. Physical verification of recent runtime fixes remains
separate from host regression tests.

```text
microphone -> continuous resampling to 16 kHz -> sliding wake detection
  -> wake accepted -> cue with capture suppression -> endpointed command capture
  -> complete-speech duration guard -> 80-mel features -> BC8 intent/slot heads
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
| run_043 VCM | BC-ResNet8 broadcast CNN, shared embedding, intent/OOS and six slot heads | 2 s, 80-mel, `(batch,1,80,198)` | 327,750 |

VCM uses FFT 400, hop 160, and NumPy/SciPy preprocessing shared with training.
Its 20 intent outputs represent 19 supported intents plus OUT_OF_SCOPE. The six
slot heads each have three supported values. Unsupported-slot rejection is a
known limitation of those heads, not proof that an invalid spoken value was
recognized correctly merely because the router rejects unknown outputs.

Wake detection uses 1-second windows, a 0.25-second hop and overlapping scoring
windows. Preserved gates include probability 0.30, sustain 2, cooldown 2 seconds
and an RMS floor. It is streaming window classification, not a causal
Transformer or a KV-cache system.

The deployed VCM graph includes temperature scaling once, with temperature
0.6475641442117414. Its matching configuration uses confidence threshold 0.875.
Runtime applies softmax without a second temperature division. Intent accuracy,
conditional slot accuracy, raw strict E2E, and gated command success are distinct
metrics. See [submission readiness](submission.md).

## Runtime Safety

Capture endpoints on trailing silence or bounded limits; total capture length
and retained speech length are different. Complete speech exceeding the model
window is rejected rather than silently truncated. See the current runtime
guide for capture settings, recovery and call guards.

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
