# BC8 source-balanced fine-tuning

## Hypothesis and fixed protocol

run_043_hf_grouped_v2b_bc8_sourceft_s42 refines run_042's best checkpoint.
The decision uses validation only: at epoch index 22, real-source intent
accuracy was 72.78% (158 clips), versus 97.54% on synthetic clips (2,764).
Training contains only 1,559 real-flagged rows out of 25,366. These flags
describe dataset provenance, not guaranteed recording quality; real-source
estimates are small and heterogeneous. Improving them is a hypothesis.

Keep the BC8 architecture (327,750 parameters), authenticated grouped-v2b
splits, cached features, augmentation, label/slot mapping, seed 42, batch 256,
class weights, and slot loss unchanged. No new audio or layers are introduced.
Multiply real-source intent CE by q*(1-p)/(p*(1-q)), with q=0.30 and training
real fraction p. This targets 30% source mass before class-weight interaction,
not exact observed loss contribution or a change in sampling. Slots remain
unweighted by source. Weighted training loss and ordinary validation loss are
different objectives; compare accuracy, E2E, NLL and clean-source breakdowns,
not their absolute loss values.

Reset AdamW optimizer and scheduler; LR 0.0003, 12 epochs maximum, warmup 1,
patience 5. Select checkpoint on raw validation E2E, exactly as the parent.
This is an optimization refinement, not evidence that a new architecture wins.
A matched continuation without source weights would be needed to attribute
gains exclusively to weighting rather than additional training.

## Selection and limits

Nominate only if real-validation intent improves by at least five percentage
points, overall validation E2E drops no more than 0.5 points, and slot accuracy
drops no more than 0.5 points. Inspect per-class/source errors and calibrated
OOS false accepts versus valid-command rejection at the same operating
constraint. The independent old-validation challenge checks whether the
improvement generalizes. Report frozen-test results, but do not tune from them.
Do not promote automatically or consume holdout. Pi latency remains unverified.

## Execution and monitoring

All commands run from repository root in the documented onit environment,
with PYTHONPATH=src, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.

```bash
python -u scripts/prepare_vcm_sourceft_trial.py
python -m pytest tests --junitxml=work/vcm_sourceft/regression.xml
python -u scripts/prepare_vcm_sourceft_trial.py --seal-readiness
python -u scripts/run_vcm_sourceft_queue.py --detach
```

The queue first waits for the entire run_042 pipeline, including evaluation
and ONNX parity. It then requires one GPU below 40% utilization with at least
20,000 MiB free on two checks and working CUDA. No SLURM or second concurrent
VCM job. Persistent logs/checkpoints survive closing the chat; process survival
still depends on host policy. Fresh output directories are required.

```bash
tail -f work/vcm_sourceft/queue.log
tail -f work/vcm_sourceft/run_043_hf_grouped_v2b_bc8_sourceft_s42.log
```

The training log appears only when training starts. Config, source snapshots,
environment and Git provenance live under work/vcm_sourceft; exact launch
command and final parent checkpoint hash are recorded in launch.json.
Training produces optimizer/RNG checkpoints and finetune_provenance.json.
The queue performs unchanged validation, calibration, independent challenge,
test partitions, CPU latency and ONNX export/parity, then stops.
