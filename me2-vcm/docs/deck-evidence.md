# Two-slide evidence map

The authoritative checklist is [SUBMISSION_CHECKLIST.md](../reports/SUBMISSION_CHECKLIST.md),
reviewed against the two-page professor-provided me2-deck.pdf on 2026-10-15.
The PDF and earlier draft slides are local inputs, not published presentation results.

Use [RESULTS.md](../reports/RESULTS.md) and reports/evaluation/ for the frozen
six-second VCM at threshold 0.954 versus run_043 at 0.875. Earlier 20261013
comparison reports describe historical candidates, not the current selection.
The original exp15 graph remains the wake default; its recovered checkpoint,
epoch-42 correction and original log are under experiments/wakeword/exp15_original/.

Selected VCM: 488,264 parameters, ONNX 2,136,900 bytes, 80 x 600 log-mel plus
padding mask. Wake: 204,738 parameters, ONNX 825,493 bytes, 80 x 100 features.
Training: one A100-SXM4-40GB, seed 42, 72 epochs, selected epoch 52;
3.64645 allocated GPU-hours from summed epoch wall durations, not kernel time.
See experiments/selected/TRAINING_ACCOUNTING.json for derived optimizer steps.

VCM train manifest: 25,366 recordings / 14.4008 source-audio hours; 25,330
eligible samples. Its 559 declared groups are NOT 559 verified human speakers.
Labels: 19 supported intents + OOS; six slot heads, 18 supported values total,
plus the TEMPERATURE reject class. Dataset DOI: 10.57967/hf/10723; rights are
source-specific and some consent/licensing documentation remains incomplete.

Replace the template's causal encoder/KV-cache/CTC diagram with the actual
BC8 -> temporal CNN -> masked attentive mean/std pooling -> classification heads.
This is captured-utterance classification, not causal autoregressive inference.

Pi 5 is the approved substitute for Pi 4. Champion acoustic accuracy and Pi
timings are reported in reports/PI5_ACOUSTIC_CHAMPION_20261017.md; injected-file
and acoustic scopes remain distinct. Test/holdout are previously exposed evaluation, and wake
official positives are synthetic; do not claim untouched real-speaker confirmation.
Author/team/date, node ID and a tagged weights release URL still need confirmation
where required. Downloadable committed weights and MODEL_TERMS.md are available.
