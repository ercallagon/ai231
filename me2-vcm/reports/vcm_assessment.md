# Interpretation and recommendation

## 3s BC8 should not be ruled out

Frozen-test raw intent increases92.67% ->93.27%, macro-F1 91.28% ->92.02%,
intent+slot92.24% ->92.90%, and ground-truth-intent slot accuracy97.68% ->98.86%.
This is a useful compact candidate, not a demonstrated architecture failure.
However, window, training lineage and TEMP REJECT supervision also changed,
so the gain cannot be attributed exclusively to a3s window.

Its validation-selected optimized gate, fixed before test predictions, gives
90.75% semantic success versus unchanged run_04390.30%. Supported-real command
success improves369/777 (47.49%) ->408/777 (52.51%), with wrong supported accepts
41 ->39 and OOS accepts10/326 ->15/326 (3.07% ->4.60%).
The real-command gain matters, but the OOS cost is real too. Dataset-wide90%+
performance must NOT be described as90% live real-command success. The diverse
real test cohort remains a shared weakness.

Against unchanged deployed baseline, paired optimized BC8 fixes113 semantic
errors and introduces92, net+21. Overall semantic delta+0.45percentage points has
declared-speaker-bootstrap95% interval[-0.38,+1.38]pp, so an overall population
win is not established. Real supported success delta+5.02pp has exploratory
declared-speaker interval[+1.07,+7.46]pp. These intervals use metadata identities,
not independently verified voices; no multiple-comparison correction or multi-seed
architecture stability is claimed.

The matched joint diagnostic also favors BC8 on frozen semantic success90.86%
versus run_04390.50% under that SAME gate design, not just against the old gate.
Do not retrospectively switch from the preregistered optimized primary to the
diagnostic gate just because its final-test headline is slightly higher.

## Added Conformer did not justify promotion

Raw intent92.12%, macro-F1 90.72%, intent+slot91.56%, slot-only97.64% are all
below3s BC8. Optimized semantic success89.47% trails unchanged run_04390.30%.
Its real supported success366/777 is not better than369/777, wrong supported
accepts rise41 ->69, and OOS accepts10 ->16/326. These are meaningful adverse
tradeoffs for this specific selected checkpoint/policy.

Parameters662445 versus BC8328007 and ONNX2.91MB versus1.49MB add cost without
a demonstrated quality advantage. This is evidence against this particular
slot-residual Conformer recipe, not against every attention/Conformer architecture.

## Temperature and long speech

The3s models do NOT demonstrate a frozen-temperature recognition win:
accepted-correct supported temperature under original gates is400/423 run_043,
399/423 BC8,400/423 Conformer; optimized gates385/423,395/423,396/423 respectively.
Reference ORIGINAL remains400/423. These policies trade acceptance for safety.
The audited validation unsupported-temperature cohort improves strongly for the
candidates, but the frozen test has no ground-truth REJECT slot examples for
measuring that independently. Do not generalize a15-example development result
into guaranteed unsupported-value safety.

3s capacity may still solve the Pi's pre-inference length rejection for a natural
command retained beyond2s; that is a runtime capacity hypothesis requiring actual
endpointed acoustic examples. Offline fixed-centered file views cannot prove it.
Preserve fail-closed capture, unsupported values and no silent speech truncation.

## Practical next move

Retain3s BC8 as the next Pi A/B candidate. Keep run_043 deployed meanwhile.
Do not launch more architecture training merely because the test table is now
available. Next authorization should prepare a versioned calibrated export/
runtime adapter for the frozen selected policy, verify interfaces and capture
limits, then perform fresh quiet/music/temperature/unsupported-value Pi tests.

The evaluation-only postprocessor reproduces ALL42,237 decisions (3models x
3policies x4693rows) with zero mismatches and explicitly marks REJECT accepted=False,
while the actual router remains defense-in-depth. It is not automatically installed
and not a general-purpose untrusted-policy API; deployment preflight must validate
policy bounds/keys and exact model/config/labels bindings.
Synthetic ONNX probes confirm temperature is effectively applied once.
Postprocessor host p50 around0.02-0.05ms is negligible relative to model inference;
host ONNX p50 is13.16ms run_043,18.69ms BC8,21.12ms Conformer. None is Pi latency.

Freeze models/policies before the official live holdout benchmark. This observed
test must never be used for subsequent parameter tuning. If future development
uses these failures, document exposure and reserve a new independent final check.
