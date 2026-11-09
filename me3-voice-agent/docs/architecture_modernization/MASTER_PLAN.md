# ME3 End-to-End Architecture Modernization

## Mission

Modernize the existing Raspberry Pi 5 (4 GB) ME3 voice agent into a reliable, low-latency, full-duplex general-purpose agent. It should handle unfamiliar requests, research and execute tasks dynamically, accept human guidance while work is active, recover from failures, and independently verify outcomes. This covers the complete system, not only its agent harness.

## Scope

The architecture review and implementation may cover:

1. Pi audio devices, PipeWire, acoustic echo cancellation, and barge-in.
2. Wake-word detection, VAD, and turn completion.
3. Project-owned ASR/TTS MCP servers and streaming inference.
4. Conversational LLM, context, tool routing, and provider fallback.
5. Time-aware web research, evidence retrieval, and source verification.
6. General-purpose task harness, planning, execution, and human steering.
7. Durable task state, approvals, cancellation, and recovery.
8. Browser, coding, filesystem, MCP, and authorized external-service tools.
9. Service management, resource optimization, and deployment.

The proposed starting topology is local Pi audio/AEC, Sherpa-ONNX wake and Silero VAD, Pipecat plus Session, and two Pi-hosted speech MCP services, with one isolated task worker and lightweight durable state. Cloud services may host streaming speech and a fast tool-capable conversational model. University Qwen/vLLM over VPN is optional and must not be a basic runtime dependency. Onit is the existing harness baseline; LangChain Deep Agents/LangGraph and speech alternatives (Deepgram Flux, Sherpa/Moonshine, Piper, Smart Turn) are candidates only when evidence justifies a comparison. This is a hypothesis to evaluate, not an approved implementation decision.

## Design principles

- Prioritize verified correctness, human control, reliability, p95 latency, simplicity, Pi resource use, and provider cost.
- Keep one authoritative conversation controller and one general task execution path.
- Do not add fixed demonstration answers or task-specific agents. General capabilities should support unfamiliar tasks.
- Separate speech interruption from task cancellation. Let people steer ongoing work.
- Use runtime time/date, grounded retrieval, dynamic planning, independent verification, scoped approval, and safe recovery.
- Preserve the two required speech MCP services.
- Avoid unnecessary frameworks, services, state owners, model calls, subprocesses, and network hops.

## Evidence and research requirements

For every significant change, record the current behavior/limits; alternative and exact versions; primary source URLs; expected benefits and tradeoffs; Pi CPU/RAM, latency, reliability, cost, and integration complexity; measurement and rollback plan; and evidence class. Evidence classes are: measured locally, prototyped, independently reported, vendor-claimed, or unverified. Advertised capability is not ME3 evidence.

Prefer official documentation, maintained repositories, release notes, compatibility data, reproducible benchmarks, and relevant open-source Pi implementations. Compare complete architecture combinations, not isolated model leaderboards.

## Measurement requirements

Maintain reproducible evidence for:

- Speech-end to ASR-final and first audible response.
- LLM first token and TTS first audio.
- Barge-in and playback cancellation.
- p50/p95, sample counts, event-loop lag.
- CPU, PSS/RSS, memory headroom, and swap.
- Token/API cost and tool round trips.
- Unseen-task completion and independent verification.
- Context/date correctness and factual grounding.
- Human steering, approval, pause/resume, and recovery.
- External-write safety and duplicate-effect prevention.

Label mock, replay, isolated prototype, live API, and physical Pi evidence separately.

## Phases and authorization boundary

0. Baseline and evidence inventory.
1. End-to-end architecture and tooling decisions.
2. Audio and acoustic full duplex.
3. Speech MCP and latency.
4. Conversational intelligence and research.
5. General-purpose agent and human steering.
6. Tools and real-world execution.
7. End-to-end testing and optimization.
8. Deployment and final qualification.

Execute only the currently authorized phase. Stop at its checkpoint; do not advance phases implicitly. Reversible project-local changes are allowed within the authorized phase. Do not make destructive changes, external writes, production migrations, or significant paid API/HPC calls without permission.

## Phase 0 outcome

Phase 0 records the source/runtime architecture, tests, measurements, providers, permissions, and integration gaps. Its deliverables are `PHASE0_BASELINE.md`, `PROGRESS.md`, `DECISIONS.md`, `BENCHMARKS.md`, and the machine-readable `var/results/architecture_modernization/phase0-baseline.json`. No runtime behavior is changed. Phase 1 remains a separate decision checkpoint requiring review before implementation.

## Phase 1 directive extension — 2026-11-09

The user authorized Phase 1 only: complete explicit end-to-end tooling/model decisions using targeted official research; retain two coordinated voice/task loops; use Session for conversation, worker for bounded dynamic execution, and a durable controller for task revisions/approvals/recovery. Immediate controls must remain responsive and speech interruption must not cancel background tasks. Include generic search/fetch/browser/files/commands/MCP and authorized APIs with independent verification, temporal grounding and scoped approvals. Review only relevant protoVoice async delivery, steering, pause delivery, reconnect, barge and tracing patterns. Compare Onit with a focused alternative only on demonstrated gaps. Do not install frameworks, modify production runtime behavior or proceed to Phase 2. The resulting proposed architecture and acceptance criteria are in DECISIONS.md.

## Final architecture priority correction — 2026-11-09

The user reopened the final decision. Final correctness/full-duplex capability, elimination of redundant custom orchestration and reuse of mature framework functionality take priority over migration effort. Existing Session/Onit and the proposed custom step executor have no preservation preference. The current target and exact replacement/retention inventory are in PHASE1_REDUNDANCY_ELIMINATION.md: Pipecat-native conversation, two functional project speech MCP services, and one LangGraph task runtime/checkpoint authority. Retain custom hardware/adapters/security/independent verification. Do not recreate framework context, tool loops, playback or checkpoints under a new name. Prior Phase 1 and simplification controller choices are explicitly superseded. This correction authorizes assessment/documentation only, with no production migration, installation, inference, destructive work or phase advancement. Revised Phase 2–8 sequence includes native pipeline contract work before optimizing speech code slated for removal.
