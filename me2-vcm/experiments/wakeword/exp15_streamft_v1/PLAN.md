# Final exp15 runtime-window fine-tune

One existing scratch-trained exp15 checkpoint, no new architecture, scratch model,
SLURM, external pretrained weights, automatic Pi installation, real calls or SMS.
All originals, exp15, CMVN/PCEN and VCM selected/run043 artifacts remain unchanged.
Intermediates/archives stay here; expanded final payload alone may enter deploy/rpi/.

## Provenance And Permitted Data

Verify original best.pt d6d9069df46f01f1ed44a014d34cd37150b210e38a4428423f7e41b73c6254b0
against deployed ONNX on batches1/2/17 plus real training features. Exp15 ran60
epochs; log and saved best-associated history identify42 as last improvement.
The archived config's best_epoch60 is a trainer metadata bug, not a new checkpoint.
Exp15's historical training/exposure is inherited, not undone by clean fine-tuning.

Reuse only the previous audited14700training rows, known-zero training/reserved
PCM and canonical voice overlap, with all provenance/hash checks. Fine-tuning
positives must have complete original waveform <=16000samples and RMS>=.002.
Longer positives are EXCLUDED, not center-cropped or relabeled negative. Report
counts and duration limits. Full-waveform retention does not prove human word
retention. No validation/test/Pi recording enters gradient updates or hard mining.
Unknown-ancestry negatives remain excluded. Eligible negative context is existing
training TTS/confuser audio, NOT verified real room/music. Generated noise/tones
are supplements, not claims of actual call/music/AEC coverage.

## Frozen Window And Label Semantics

16kHz mono; exact1s waveform first, then existing80x100 Hann512/hop160 log-mel,
per-frame across-mel CMVN and delta. No whole-recording feature-center-crop.
Each epoch regenerates8192examples (half positives, half negatives) from seed
20261017/epoch/index. A positive is a complete known wake waveform placed anywhere
it fits within a fresh scoring window, with continuous low-level permitted
non-wake context before/after. A partial wake overlap is IGNORE, not positive or
negative, unless the original source itself has a verified non-wake label. No
unverified word timestamps or fabricated partial-phrase negative labels.
Neighboring250ms-grid window labels are logged; IGNORE windows do not enter loss.
Store epoch example specifications for reproducible replay; no feature-position
roll or fixed two-view cache. Half negative source sampling targets the top
quartile of training-only baseline wake scores; half stays broad.10percent of
negative examples are seeded non-speech tonal/noise stress. No mining on evaluation.

Half examples simulate a48kHz capture input by deterministic upsampling followed
by the exact ContinuousResampler48k->16k in100ms chunks BEFORE selecting1s windows.
Others use16k passthrough. This exercises the runtime algorithm, not a real48k mic
or acoustic simulation. Modest gain/channel augmentation is training-only.

## Single Bounded Training Recipe

Same204738parameter WakeGRU. Initialize original exp15 once. AdamW1e-5,
weight_decay1e-4, batch128, max6epochs, patience2, GPU-resident deadline20min,
clip_grad5, seed20261017. Freeze BN running statistics to avoid small balanced
fine-tune batches replacing historical statistics; affine parameters still train.
Epoch0 is eligible; never report a worse fine-tune as an improvement. No sweeps
or extensions beyond the cap. One GPU after two <40percent utilization and
>=12GiB free checks; finite20min maximum wait, no indefinite resource polling.

## Matched Streaming Policy And Calibration

Both original and candidate use latest1s window only,250ms hop, two FRESH
consecutive scoring decisions, RMS>-60dBFS, nonwake<.90, integer2s cooldown and
production streak reset after fire. Disable four-window max pooling: one retained
old spike cannot satisfy sustain repeatedly. No recurrent state across windows.
Keep shipped four-window/.30 policy as a separately labeled reference/fallback.

Before training, compute baseline validation curves on raw and HP80 (filter
proxy, NOT AEC). Primary empirical budget is the worst original/generated cohort
event rate of baseline under shared policy at shipped tau.30; freeze its numeric
value before training. Also report5/15/30/60/120events/hour curves. Each cohort
and raw/HP must meet a budget separately. Threshold grid .10..99 step.01 plus
.995/.999/.9999/reject-all; no data-dilution or silent padding in negative rates.
Checkpoint selection: worst raw/HP recall at primary budget, mean recall, mean
lower-budget worst recall, then fewer false events and lower endpoint p95.
Short negative exposure gives wide uncertainty; empirical rate is not room FAR.

## Freeze, Evaluation And Decision

Freeze best checkpoint/export/policy and all budget thresholds before reading
test scores. Test already exposed;385official positives synthetic only; historical
validation/test share17Piper voice families. No independent real-speaker claim.
Retain all evaluation utterances, no training exclusions applied to validation/test.
Actual paired raw/AEC Pi recordings are development evidence only; without verified
phrase timestamps report events, not fabricated recall/latency. Music-only event
counts/rates can be reported for their known labeled exposure, with uncertainty.

Nominate only if primary validation worst-condition recall improves >=2percentage
points without exceeding the anchored false-event budget. Exposed-test confirmation
can veto, never refit/select: at most2point recall regression on raw/HP versus
matched baseline, <=1.25x its original false events (or at most1 if baseline0),
and no extra music-only development events. These are descriptive host checks,
not sufficient proof of a real-world deployment champion. Keep exp15 fallback
and physical Pi5 verification pending. If not supported, retain exp15 and close
experimentation; no automatic extra run. Include latency, parity, CPUcost, coverage,
source grouping, exposure limitations, exact Pi commands and rollback evidence.
