# Final matched streaming wakeword study

User authorized 2026-10-15. Preserve exp15, all prior artifacts, VCM default,
run043 fallback, Pi installation, wiring and Git history. No SLURM or external
pretrained detector weights. One GPU maximum after a twice-confirmed <40%
utilization / >=12GiB free check. No actual calls or SMS.

1. Back up runtime sources and protected hashes; use an integer sample clock
   for sustain/cooldown. Keep exp15 weights, features and default threshold.
2. Audit exact PCM/bytes, parent augmentation lineage and canonical voice
   families. Exclude training positives from voice families declared in reserved
   metadata. Exclude mined negatives derived from reserved parents. Do not
   rewrite historical manifests. Unknown human/source ancestry remains a limit.
3. Build validation streams before training. Negative recordings contribute
   their actual seconds, not silent padding, to event/hour denominators.
   Positives retain entire original recording; known TTS text supplies presence,
   while energy-estimated speech endpoints are NOT manually annotated truth.
   Report both waveform-end and energy-end latency with that qualification.
4. Freeze baseline streaming calibration BEFORE scratch training. Target <=5
   false events/hour on validation negatives, then maximize positive recall,
   minimize duplicate/outside events and latency. Include reject-all in the
   threshold grid; do not silently relax the rate target if all policies fail.
   Enforce <=5FA/hour separately for original negatives, generated stress and
   their combined exposure in BOTH raw/HP conditions; do not dilute difficult
   original recordings by adding easy generated hours.
   Thresholds >=0.10 and nonwake gate0.90; no offline/live gate mismatch.
5. Train exactly two matched arms: current static+delta per-frame CMVN (A),
   fixed-parameter PCEN+delta without across-frequency CMVN (B). Both WakeGRU
   204738 parameters, input80x200, seed20261015, batch128, AdamW0.0003,
   weight_decay0.0001, cosine80 epochs, minimum40, patience20, gradclip5.
   Training positives use entire <=2s waveform with varied zero/noise padding
   positions; never truncate positive speech. Over-capacity training positives
   are excluded and reported. No-speech and invalid records are excluded from
   positive training and remain failures in evaluation denominators.
   Both arms see identical window samples, augmentation draws and synthetic
   hard-negative seeds. Variants include causal HP/channel filtering, room/gain
   and waveform mixtures, plus synthetic tone/buzz/ring/music-like negatives.
   Generated stressors are NOT claimed to be real call/music captures or AEC.
6. Select using validation stream events/hour and wake recall; freeze checkpoint,
   frontend and policy before exposed test and real raw/AEC development replay.
   Do not use the replay recordings or test results for training or selection.
7. Export ONNX and verify logits/probabilities/stream events, chunk invariance,
   silence, full frontend parity, CPU p95/RTF with recorded threads. Actual Pi
   latency and acoustic behavior remain pending. Single-window scoring avoids
   counting the same historic max window as multiple pieces of evidence.
8. Curate experimental candidates, configs, logs, reports and selectable Pi
   commands. Keep exp15 default unless validation plus development evidence
   justifies candidate testing; do not claim hardware promotion before Pi tests.

PCEN motivation: Wang et al., Trainable Frontend For Robust and Far-Field
Keyword Spotting, https://arxiv.org/abs/1607.05666 . Here parameters are fixed,
not pretrained or learned: smoothing0.025, alpha0.98, delta2, power0.5,
epsilon1e-6. Filter state resets for each fully captured scoring window in BOTH
training and inference; this is causal-window processing, not a stateful encoder.

Official positive validation/test recordings are synthetic. Historical tests
have been exposed; this experiment cannot make them independent again.
Raw/AEC versions of one Pi recording are correlated, not separate speakers.
Manual word retention and real-speaker population recall remain unverified.
