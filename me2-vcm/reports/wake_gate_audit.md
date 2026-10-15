# Wakeword gate and metric audit

## Scope

Seven existing ONNX models audited CPU-only on identical1,052-row corrected
validation (326wake/726negative) and saved raw/AEC streaming development traces.
No new training, GPU job, model/config modification, wake test evaluation or
live holdout. Three gate tests pass through actual WakeWordDetector.process;
all saved default event counts AND timestamps reproduce exactly. All26 protected
fingerprints remain unchanged. Full sweeps, model/trace hashes and probabilities
are in RESULTS.json and per-model reports.

## Default gate comparison

threshold.30,sustain2,cooldown2s,RMS>-60dBFS,4 overlapping subwindows.

| Model | Clean recall | Clean clip FAR | HP recall | Music events raw/AEC |
|---|---:|---:|---:|---:|
| exp15 preserved | 81.60% | 3.86% | 77.30% | 0 /0 |
| Scratch logmel | 81.29% | 3.03% | 76.69% | 2 /1 |
| Scratch PCEN | 84.05% | 2.34% | 87.42% | 24 /6 |
| Existing exp16 | 82.82% | 2.20% | 82.21% | 3 /0 |
| Historical HP+tone | 84.66% | 3.86% | 84.66% | 3 /0 |
| Historical negative | 82.82% | 2.48% | 81.60% | 1 /0 |
| Historical negative+HP | 82.52% | 2.62% | 81.90% | 1 /0 |

Canonical metrics are NOT evidence that every recent model is inferior. PCEN
improves both recall and clip FAR substantially, particularly under HP stress.
However, the music regression at the actual default streaming gate is real,
not a label inversion or counting bug. Earlier "no improvement" statements
should mean "not demonstrated safe for promotion at that operating point."
Historical arms have different training lineage/recipes and must not be called
matched random-init architecture ablations. Only scratch logmel/PCEN share the
matched random-init recipe. Do not retroactively claim older own-model fine-tunes
were from scratch.

## Gate tuning opportunities

Compared14thresholds x4sustain values for each model, retaining the binary
nonwake condition, RMS eligibility and2s cooldown. Require clean clip recall
within1percentage point of exp15, clean/HP clip FAR within.5points of exp15.
This is a diagnostic screen, not live positive recall certification.

- Scratch PCEN: no zero-music gate under these canonical constraints. Its
  improved spectral proxy results do not solve real music false activation.
- Scratch logmel: threshold.30/sustain5 removes the three music events, but
  removes speech events on several raw/AEC paths too. Not a demonstrated upgrade.
- Historical HP+tone: threshold.40/sustain5 gives zero sampled music events,
  clean recall81.90%, HP recall83.13%. It produces events on some speech paths
  where baseline has none; it also loses an AEC speech-over-music path where
  baseline has an event. This is a promising tradeoff to annotate/test, not a
  promoted model. It is an older own-champion fine-tune, not a new scratch arm.
- Historical negative arms: sustain5 zeroes music but mostly suppresses speech
  traces, illustrating why zero false triggers alone is insufficient.

Only about60.06seconds of music per raw/AEC path were observed, from two captures.
Do not pool correlated raw and AEC views as independent duration. Baseline0events
does not establish zero long-run false activations; the approximate zero-event
Poisson upper95% rate is about180/hour for that short exposure. Such rates are
not reliable deployment estimates; collect hours of representative negatives.

## Confirmed implementation/presentation issues

1. Label0 is wake, label1 negative: recomputed confusion counts confirm correct
   orientation. Clip FAR=FP/(FP+TN), not FP/total and not false activations/hour.
   Baseline clean accuracy91.63% is a canonical-window validation statistic;
   baseline recall81.60% is a different statistic. Neither is live Pi recall.
2. HP stress is a high-pass surrogate, not actual AEC. Real AEC compatibility
   evidence comes from the supplied processed recordings only. PCEN is not AEC.
3. Speech traces have condition labels but no verified per-utterance intervals.
   Event counts cannot be treated as recall: extra events may be false or repeated
   activations, and count changes may conceal missed real utterances. No wake
   latency can be measured without reference onset/end annotations.
4. Sustain2 sees max scores over4overlapping windows. The same strong window can
   remain the maximum on consecutive hops. Therefore2qualifying hops are NOT
   two independent neural confirmations. Increasing sustain also changes
   detection opportunity and latency; our clip metrics do not model that effect.
5. For binary models, p(nonwake)<.90 means p(wake)>.10. At thresholds>=.30 this
   is redundant, not an independent OOS classifier. Temperature scaling for a
   fixed binary model is monotonic in wake probability; a threshold change gives
   the equivalent operating point, so it cannot improve score ordering itself.
6. WakeWordDetector.__init__ reads model.onnx rather than inference.onnx, while
   this project's YAML uses inference.onnx. Current fallback points to the SAME
   champion, and explicit onnx_path works, so current evaluation is correct.
   But YAML-only future model selection can be ignored. Constructor also does
   not read multiwin/n_subwin/subhop/gate_nonwake from YAML; actual defaults
   match today's YAML. Gate changes must be verified through actual runtime,
   not merely edits to ignored settings. Production remains unchanged here.

## Next evidence and decision

Keep exp15 deployed. Annotate the existing development speech recordings:
intended HeySwift intervals, EOS, and any unintended event intervals. Reassess
fixed before-evaluation gates for detection hit/miss, false triggers and delay,
including raw/AEC quiet/music cases. Collect fresh consenting real speakers,
beeps/rings and long music negatives separately from official holdout. Preserve
current gate until positive evidence supports a change. No automatic new training.

The evidence supports continued targeted gate/real-data work, particularly the
HP/tone direction, not a broad claim that all alternatives are inferior. It does
not establish a replacement champion or paper-ready live wake accuracy.

## Reproduction

From repository root, with the existing host environment:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B \
  work/wake_gate_audit_20261013_v1/test_gate.py
```

The same prefix ran audit.py. Output files are exclusive-create; use a fresh
versioned directory at the same depth for repetition. Original artifacts retained.
