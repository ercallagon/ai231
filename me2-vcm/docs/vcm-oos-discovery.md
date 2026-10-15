# Bounded OOS architecture and optimization discovery

## Evidence and hypothesis

Historical architecture screens included conventional CNN, depthwise CNN,
TC-ResNet, residual 2D CNN, CRNN/attention, MobileNet, Matchbox and transformer
variants. Their old data/slot bugs and different validation protocols mean
their scores cannot be pooled as controlled comparisons. Corrected v1 A10/A6
and BC8 remained promising; no evidence justifies repeating every weak family.
Grouped-v2b BC8 run_042 and source-finetuned run_043 provide current controls.

run_043 improved real-validation intent from 70.89% to 74.05%, test raw E2E
from 91.88% to 92.24%, and valid-command OOS rejections from 159 to 76.
Raw OOS false accepts increased from 43 to 56. At the calibrated confidence
gate, run_043 still rejected 404/4,367 valid commands and accepted 10/326 OOS.
The independent real-heavy challenge is only 65.07% raw E2E. These verified
results motivate rejection separation, not a claim of reliable deployment.
HF test/challenge have already been inspected historically and are not blind.

The hypothesis is that a dedicated binary supported-command objective can
improve separation while preserving intent/slots. New head: Linear(256,1),
257 extra parameters, shared BC8 embedding, no extra audio frontend. Train on
the exact grouped-v2b pool and warm-start every screen from run_043. The main
intent loss retains run_043's source weighting, class weights and smoothing;
slots retain weight 0.5. Validity BCE gives equal weight to in/out-of-scope
means within a batch, not to the much more frequent valid class.

## Controlled sequence and stopping

1. CPU-only validation/tune scoring on run_042/043: maximum softmax confidence,
   probability of being in scope, and in-scope logsumexp (negative energy).
   Scores retain the learned OOS veto. This is an energy-score adaptation,
   not implementation of the paper's energy-margin training objective.
2. run_044: binary head, BCE weight 0.25, AdamW LR 0.0003, seed 42, 12-epoch
   cap/patience 4, warmup 1/cosine, unchanged batch/features/augmentation.
3. run_045: if 044 qualifies, reduce LR to 0.0001; otherwise reduce BCE weight
   to 0.10 if raw E2E lost >0.5 points, or increase it to 0.50 otherwise.
   Always start from the same run_043 parent; no chaining hidden extra training.
4. Only a qualifying winner gets run_046, same settings with initialization
   and augmentation/shuffle seed 7. Both share a seed-42 pretrained backbone;
   this checks fine-tuning stability, not full independently trained seeds.

Choose checkpoints/policy using validation E2E under <=5% empirical false
accepts on tune_oos. Compare to the best rescored existing control policy.
An essential gain is >=0.5 validation E2E points, while raw validation E2E
and slots lose <=0.5 points and real intent loses <=2 points. This is a
screening rule, not statistical significance. Speaker-level uncertainty,
small real sample (158), and repeated validation/tune reuse remain limitations.
Policies include confidence, scope probability, energy and validity; no test
rows are used to choose policy, epochs, hyperparameters or finalists.

Stop after two unsuccessful screens. Otherwise confirm one winner and seal
selection before test/challenge access. At most three new trainings; no endless
search, test-feedback optimization, SLURM, holdout use or automatic Pi promotion.
Even failed confirmation is reported rather than triggering more runs.
Threshold changes alone must be compared at matched false-accept budgets.

## Artifacts and deployment

Fresh configs live in configs/run_044*.yaml through run_046*.yaml. Checkpoints,
optimizer/scheduler/RNG states, raw/per-class/source metrics, policy metrics,
environment/Git/code/dataset hashes are preserved under experiments/vcm.
Staging, readiness, plan, source snapshots and historical fingerprints are
under work/vcm_oos_discovery. Writes of history/selection are atomic.

Only winner and confirmation get clean-train diagnostic, unchanged challenge,
core/negative/combined test, ONNX checker plus batch-1/2 parity, model size and
single-thread host latency. Never report host latency as Pi measurement.
Validity adds an ONNX output and new policy: current Pi runtime does NOT yet
consume that contract. A candidate is not a drop-in bundle replacement. If
selected, implement/verify its policy upstream before future packaging. Keep
existing bundles, models, wiring and thresholds untouched now.

Prefer host p95 <=25ms and model <=2MB; higher costs must be flagged in final
review, even if quality improves. Independent Pi speech/noise/music/action
tests remain required, including arbitrary unsupported slot values which
three-way slot heads do not inherently reject. No offline metric proves
foolproof behavior or action safety.

## Reproduce and monitor

Use the existing onit environment from repository root:

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/prepare_vcm_oos_discovery.py
python -m pytest tests --junitxml=work/vcm_oos_discovery/regression.xml
python scripts/prepare_vcm_oos_discovery.py --seal-readiness
python -u scripts/run_vcm_oos_discovery_queue.py --detach
tail -f work/vcm_oos_discovery/queue.log
tail -f work/vcm_oos_discovery/run_044_hf_grouped_v2b_bc8_validity_s42.log
```

Baseline scoring runs on one CPU thread. Each training independently waits
for one GPU <40% utilization and >=20,000 MiB free on two readings plus CUDA
availability. This heuristic is not an exclusive reservation. Logs persist
without chat polling, subject to host process/reboot policies. Failed stages
stop the queue; automatic resume is not claimed. Never overwrite partial runs.
Exact commands/GPU/config/checkpoint hashes are in *_launch.json.
The active controller was versioned to run_vcm_oos_discovery_queue_10s.py
at the user's request. It checks idle GPUs every 10 seconds, retaining the
five-second confirmation and unchanged safety gate. It reuses completed
baseline scores only after verifying checkpoint fingerprints; the original
30-second controller and authenticated experiment sources remain intact.
Controller PID/source fingerprint is recorded in polling_controller_10s.json.
The queue writes comparison.md and summary.json when it stops, including
controls, screens, finalist tests and deployment-cost flags.

Research supports trying related rejection techniques, not a guaranteed
improvement for this dataset: [Energy-based OOD detection](https://arxiv.org/abs/2010.03759)
and [Outlier Exposure](https://arxiv.org/abs/1812.04606).
No extra outlier data are added during this controlled stage.
