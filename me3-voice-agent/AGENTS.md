# Project instructions

## Scope and truthfulness

Read README.md and docs/PLAN.md before changing the project. Preserve ME2 as a separate project. Read ME2 files for reference; do not import its repository, datasets, checkpoints or transfer bundles wholesale. Reuse selected action logic only after checking dependencies, wiring and behavior.

Distinguish proposed, implemented, locally tested, and Pi-verified states. Never claim a physical effect or acoustic result based on mocks. Do not spawn subagents unless the user explicitly requests delegation.

## Files and documentation

- Follow the directory contract in README.md. Keep the root small.
- Use one canonical implementation and one canonical document per topic. Edit in place. Do not create final_v2, latest_copy, dated source directories, or repeated handoff documents.
- Put generated files only under ignored var/: scratch/, experiments/, logs/, recordings/, results/, cache/, or exports/.
- Put exploratory trials under var/experiments/<number>-<topic>/ (for example, 001-wakeword-comparison). Each experiment has one clear question and a short README.md recording configuration, commands, findings and limitations. Keep exploratory scripts and raw output in that folder; reuse it for the same investigation rather than creating successive copies.
- Promote useful experiment code into src/me3_voice/ or scripts/, meaningful regression checks into tests/, and selected sanitized findings into docs/evidence/. Raw recordings and large generated output remain outside Git.
- Add scratch work only when needed. Remove only scratch files you created and no longer need; never delete user files, raw evidence, or results without authorization.
- Promote a result to versioned evidence only deliberately: add a compact, sanitized report under docs/evidence/ with command, revision, configuration and limitations. Keep raw data outside Git.
- Do not create export archives by default. If a deployment archive is explicitly needed, generate it under var/exports/ from a known revision; never maintain a second source tree.
- Update docs/PLAN.md after a meaningful milestone, blocker, or decision. Record actual commands and verification, not a transcript of all exploration.

## Implementation and security

Use one Python project under src/me3_voice/ and a single pyproject.toml dependency definition. Add a reproducible lockfile after selecting and verifying the package workflow. Keep provider adapters separate from session logic and MCP contracts.

Keep API keys in environment variables or a local ignored .env. Track only .env.example with empty values. Never print credentials. Do not read or copy unrelated credentials. Keep MCP services local by default; do not expose ports, tunnels, or public endpoints without authorization.

Do not change existing ME2 GPIO wiring, install conflicting audio services, overwrite Pi deployments, or enable boot services without checking existing state and recording the intended change. Default hardware actions to mock until explicit real-hardware testing is authorized.

## Git and validation

Inspect git status before changes. Do not overwrite unrelated work. Use codex/<topic> branches for implementation and stage explicit paths; do not use git add . or git add -A. Inspect staged diff and status before committing. Do not push or create a public repository without user authorization.

The user has deferred commits until more project structure is established. Continue authorized setup and implementation without requiring an initial commit. Do not stage or commit until the user asks. Before any later cross-checkout patch exchange, establish a shared base or a reviewed file manifest; do not invent a base revision.

Run checks appropriate to the change. Prioritize session transitions, cancellation, playback queue clearing, tool idempotency and timing correctness. Avoid tests that merely repeat constants. A completed task action must not execute again when speech is interrupted.

No runtime artifacts or dependencies are presently installed by this baseline. First implementation work must perform the Pi inventory and compatibility gate in docs/PLAN.md.
