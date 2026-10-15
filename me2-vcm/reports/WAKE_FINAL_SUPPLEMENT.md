# Wakeword Selection Supplement

The frozen PCEN model improves raw test recall from107/385 to139/385 versus
the equally calibrated single-window exp15 baseline, but has3 original-negative
events versus1 in about18.4minutes. This fails the predeclared empirical5/hour
guard; it is not proof of statistically worse real-world safety. The Poisson
confidence intervals overlap substantially. Keep exp15 default, with optional
diagnostic Pi A/B, not automatic promotion or further test-driven calibration.

The shipped exp15 four-window tau0.30 policy detects335/385 positives in this
constructed test stream but triggers80 times on original challenge negatives.
Those dense adversarial negatives are not a normal room; do not call that a
measured live Pi hourly rate. All official positive clips are synthetic.

Two-second models take longer per window, but the proposed policy scores ONE
window per hop rather than the default FOUR. Actual full-hop host CPU p95 is
8.470ms for default exp15 versus4.271ms for PCEN, intra1/inter1, with scoring-hop
RTF .03388 versus .01708. This is not Pi timing or end-of-speech/action latency.

Detection latency must be considered separately from CPU cost: longer context
and temporal scoring can delay the wake event even when CPU time is lower.
SUPPLEMENT.json records latency differences on commonly detected positive clips
as well as per-model uncertainty. Speech offsets are automated, not manual.

PCEN test false events occur at one hard_v3 and two solo-source intervals.
Negative transcripts/source ancestry are incomplete; do not invent their exact
spoken content. Unknown-ancestry training sources were excluded rather than
claiming filename IDs prove speaker independence. Reconstructing that provenance
and collecting independent consenting real speakers/real playback negatives is
more justified than blindly expanding TTS or adding layers now.

Evidence: experiments/wakeword/final_streaming_v1/{REPORT.md,RESULTS.json,
ERROR_ANALYSIS.json,SUPPLEMENT.json}. Test was previously exposed. No additional
training or policy tuning followed this frozen test evaluation. No deployment
or real calls/SMS were performed.
