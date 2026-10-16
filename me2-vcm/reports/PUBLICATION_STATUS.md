# ME2 publication status, 2026-10-16

Project: **ME2: Voice Controlled Smart Device (Hey Swift)**, under
[ercallagon/ai231/me2-vcm](https://github.com/ercallagon/ai231/tree/main/me2-vcm).
The repository main revision before this update was
b6ab05bb580491a754be70c2f979ee48f5ca13e7, previously verified using git ls-remote.
Use the containing Git commit for the exact revision of this report; this file
does not invent a future commit hash or claim an unperformed push.

## Current selection and update scope

- Six-second BC8/TCN/attentive-pooling VCM at frozen threshold 0.954 remains
  repository default; complete run_043 at 0.875 remains selectable.
- Original exp15 remains wake default. The recovered Torch checkpoint, original
  log and epoch-42 correction are included; parity ties it to the selected ONNX.
- Controlled scratch CMVN/PCEN and short exp15 fine-tune evidence are retained
  as opt-in experiments, not promoted replacements. No additional trial is queued.
- Integer sample-clock cooldown fixes chunk-dependent wake timing. Original
  exp15 graph/gate is unchanged; keeping original weights is not a claim that
  the corrected runtime source is byte-identical to the earlier release.
- Current architecture/deck guidance, training accounting and submission
  checklist supersede stale planning snapshots. Historical results are preserved.
- The update is scoped to me2-vcm/. ME1, repository-root files and existing Git
  history are preserved. No Pi installation, accounts, model defaults, hardware
  wiring or protected training artifacts are changed automatically.

## Interpreting verification records

Publication copies of historical logs/metadata replace private host paths with
PROJECT_ROOT/HPC_HOME. PROVENANCE_REDACTIONS.json records archived and published
hashes; original local artifacts remain untouched. Logs retain all lines with
only those path substitutions; JSON formatting may change. Historical input
hashes continue to identify their archived originals, not sanitized derivatives.

CHECKS.json is an immutable receipt from the initial preparation. Its final
authentication-blocked note records that earlier attempt, not present repository
visibility: initial commits were subsequently published. VERIFICATION.md and
SUBMISSION_CHECKLIST.md describe current scope and missing device evidence.

Historical scripts/logs retain their original data/epoch semantics. In particular,
the exp15 config's best_epoch 60 is corrected by its log/history and parity record;
the original file is not rewritten to conceal the discrepancy. Earlier VCM
0.95-gate tables are not the current 0.954 policy results; use RESULTS.md.

Fresh-checkout inference contracts, relevant regressions and staged-file checks
are required before committing this update. The publication command's terminal
result and Git history establish whether commit/push succeeded; this report is
not a substitute for that outcome.

## Unresolved, explicitly not certified

Initial Pi injection/acoustic results and operator GPIO/audio observations have
been received; see FINAL_CANDIDATE.md for actual counts and provenance gaps.
Final controlled 800 ms endpoint/A-B and communications recovery remain pending.
Host parity and wheel availability do not establish Pi success.
Test/holdout were repeatedly exposed. Official wake positives are
synthetic; independent real-speaker wake performance is not established.
Wake raw-data retrieval and some dataset/voice consent/licensing remain incomplete.
No owned-code MIT grant overrides those source or model terms.

The posted Pi guide separates injected VCM recognition, model-only cost and
the official laptop-speaker-to-microphone benchmark. No real calls/SMS are needed.
