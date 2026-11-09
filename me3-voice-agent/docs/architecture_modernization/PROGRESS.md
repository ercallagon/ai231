# Modernization progress

Last updated: 2026-11-09

## Phase state

| Phase | State | Evidence / boundary |
|---|---|---|
| 0 — Baseline and evidence inventory | Complete | Read-only source/docs/artifact audit. No application/tests/providers/hardware invoked. Deliverables linked below. |
| 1 — Architecture and tooling decisions | Complete — design only | Proposed choices, model IDs, sources, contracts, targets and migration in `DECISIONS.md`; no implementation or live benchmarks. |
| 2 — Native contract and acoustic full duplex | Local checkpoint complete; PENDING_OPERATOR_VALIDATION | 469 baseline tests; 3 native scenarios, 2 Session comparisons, 8 transport checks passed. Physical gates remain open; see PHASE2_REPORT.md. |
| 3 — Speech MCP and latency | Not started | Existing software path audited; no new transport or measurements. |
| 4 — Conversational intelligence and research | Not started | Provider and grounding decisions remain open. |
| 5 — General agent and human steering | Not started | Onit task continuation/cancellation boundaries remain open. |
| 6 — Tools and real-world execution | Not started | External write/publish path is not authorized or implemented. |
| 7 — End-to-end testing and optimization | Not started | No Phase 0 tests run. |
| 8 — Deployment and qualification | Not started | Runbook contains reported software smoke only, not acoustic qualification. |

## Phase 0 deliverables

- `MASTER_PLAN.md` — user modernization directive and authorization boundary.
- `PHASE0_BASELINE.md` — source architecture, current evidence, gaps, permissions, and Phase 1 agenda.
- `DECISIONS.md` — evidence-qualified current decisions and unresolved choices.
- `BENCHMARKS.md` — benchmark contract for future reproducible comparisons; no new scores.
- `var/results/architecture_modernization/phase0-baseline.json` — machine-readable provenance/evidence inventory.

## Phase 1 checkpoint

Completed targeted official-source research and full-system proposed design. Changed documentation and machine-readable decision manifest only. No framework install, runtime edits, inference or Pi trials; no tests run. Phase 2 remains unauthorized.

## Post-Phase-1 independent challenge

Complete — architecture review only. Read installed Pipecat 1.13.0 and editable Onit 0.1.5a0 source plus targeted official speech/runtime references. Recommendation: keep simplified voice architecture and provisional speech/model selections; converge future tasks on one ME3 step executor with SQLite, gated before Onit retirement. No source edits, installs, inference, tests or device trials. Documentation consistency/JSON parsing only. Phase 2 remains unstarted. Historical recommendation superseded by PHASE1_REDUNDANCY_ELIMINATION.md.

## Final architecture correction

Complete — source-level assessment only. Current final target is Pipecat-native conversation plus one LangGraph task runtime. Explicitly supersedes preserving Session/narrow ingress and creating a custom durable step executor. Source inventory: 8,962 physical lines / 44 classes / 279 function definitions in the declared orchestration scope; proposed final 3,500–5,200 lines is an estimate, not an implemented reduction. No production code, installations, inference, prototypes or runtime regression executed. Phase 2 remains unstarted. SQLite/ARM64 and native speech-adapter parity require focused qualification.

## Next step

Execute the operator procedure in PHASE2_REPORT.md on the actual Pi and return the specified logs/audio/graph. No SSH access is available here; physical checks are PENDING_OPERATOR_VALIDATION. Review actual results before closing Phase 2. Phase 3 requires separate authorization.

## Phase 2 local checkpoint — 2026-11-09

Bounded prototypes live under ignored var/experiments/022-native-voice-duplex/. Provenance records 113 source hashes and dependencies. Actual ASR/TTS MCP children connect to native Pipecat with mock inference. Native context, streamed tool arguments, sequencing, interruption and a function surviving speech interruption pass isolated fixtures. In-process PipeWire transport and attended acquisition scripts are implemented; transport tests substitute physical processes. Production runtime, existing tests and lockfile remain unchanged.

Retained failures: initial sandbox suite timed out; external offline rerun passed469. Initial async fixture needed an explicit spoken preamble; a transport assertion had an incorrect PCM duration. Corrected fixtures pass. Startup-inclusive mock loop-lag maximum196ms remains unresolved. Slow ASR-finalization concurrency is unqualified. No Session/sidecar deletion, LangGraph migration, package install, provider call, SSH or Phase3 work. Canonical evidence and operator commands: PHASE2_REPORT.md. Structured checkpoint: var/results/architecture_modernization/phase2-results.json.

## Existing-audio reuse and deferred live testing — 2026-11-09

Read-only corpus inventory found54 integrity-matching guided browser takes (435.923s), one18s reading, and a23.94s laptop AEC pair; no paired Pi raw/processed/reference session identified in this checkout. Canonical coverage/gaps and evidence limits: docs/evidence/audio-corpus-inventory.md. User explicitly authorized existing recordings for relevant new service evaluations, superseding historical other-provider consent restrictions; preserve original manifests. No upload, paid inference or later-phase implementation occurred.

Phase2 acquisition now includes separate noninterrupting corpus profiles, repeated true interruption trials, wake/negative exposure, continuity/resource logging, actual transcripts and reference/geometry metadata. Hardware gates remain PENDING_OPERATOR_VALIDATION; recordings enable offline comparisons but cannot qualify changed live AEC or final acoustics. Canonical commands: PHASE2_REPORT.md. Validation tools are tracked in scripts/phase2/ and tests/test_phase2_*.py; raw history remains ignored.
