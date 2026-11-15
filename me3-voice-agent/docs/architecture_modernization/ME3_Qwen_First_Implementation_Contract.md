# ME3 — Qwen First Implementation Contract (Phase 3 Closure)

## Goal
Make **one bounded, low-risk code change**: reconcile the browser voice-capabilities response between the **legacy Session** and **Pipecat-native** engines. Do not begin Phase 4 or LangGraph migration.

## Context
- Sol finished Phase 3 repository-local integration: native CLI/replay/browser entrypoints and both project-owned speech MCP services work in offline tests.
- **Legacy remains the default**; native is opt-in.
- Reported results: **520 working-tree tests passed**; isolated checkpoint `codex/me3-native-application` at `e9c2911` (not pushed).
- The checkpoint and shared working tree **may differ**.
- Pi acoustic/AEC, live-provider and rendered-browser qualification remain pending.
- **You are the only active writer for this task.**

## Instructions

### 1. Confirm the checkout safely
Read `docs/architecture_modernization/PHASE3_REPORT.md` and relevant bridge files. Check `git status --short`, `git rev-parse HEAD`, and relevant file contents **without changing Git state**. Do not assume the checkpoint commit is the current checkout. If the working tree or another agent's ongoing edits make ownership unclear, **stop and report**.

### 2. Fix only the capabilities contract, if still needed
Inspect:
- `src/me3_voice/voice_bridge.py` — legacy `capability()`
- `src/me3_voice/native_bridge.py` — native `capability()`
- Relevant existing tests, preferably `tests/test_native_entrypoint.py` or `tests/test_voice_bridge.py`

Verify whether Sol already corrected the mismatch. If not:
- Return a **stable, backward-compatible shared schema** from both engines, including a truthful `engine` value (`legacy` or `native`).
- Preserve existing fields and meanings used by clients (wake, devices, music, task-artifact access, capture/playback/AEC/ASR partial support where applicable).
- Never advertise unavailable capabilities as supported; avoid fake placeholders.
- Use a small shared helper **only if it actually removes duplication**.
- Do **not** change the separate static dashboard endpoint, change the default engine, or refactor unrelated files.
- Add focused tests covering both engine responses and backward compatibility.

### 3. Validate and report
Run focused tests, then the full relevant regression suite **if the shared repository is free of competing test workloads**. Report the exact commands, counts (pass/fail/skip), changed files, contract before/after, and any unresolved issues. Distinguish offline tests from physical Pi/live-provider/browser validation.

## Safety and boundaries
- No reset, checkout, rebase, stash, clean, force push, remote push, or edits to unrelated work.
- No paid APIs, HPC inference jobs beyond your normal coding model, Pi SSH, new packages, or framework migration.
- Do not commit unless separately authorized. Do not touch secrets or print credential-bearing URLs.
- If the issue is already fixed, make **no edits**; report evidence and the next smallest unresolved Phase 3 gate.
- **Stop after this one task.** Sol will review your diff before the next assignment.

## Response format
1. Git state and whether the fix was needed
2. Exact code changes and tests
3. Risks or blockers
4. Remaining Phase 3 gates
