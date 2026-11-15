# Swift ME3: local development handoff

Status and instructions verified against the ME3 workspace on 2026-11-15. This document is for continuing ME3 development in local Ubuntu on WSL2 and later on Raspberry Pi 5. The project is `ai231/me3-voice-agent/`; the parent coursework repository and ME2 are separate.

## 1. What the project is

Swift is a Pipecat-native voice assistant with a browser UI, two project-owned speech MCP subprocesses, a LangGraph task runtime, and configured remote model/search providers. The native browser path owns conversation state and real-time audio; LangGraph owns bounded background tasks and their local workspace/state. The legacy Session implementation remains a fallback. Supported short device/media commands use the existing fast route. Do not add another conversation controller or MCP dispatch layer.

Typical native turn:

```text
browser microphone → wake gate / VAD → Pipecat native pipeline
  → ASR MCP (Deepgram when enabled)
  → native LLM and registered tools (OpenRouter GPT-OSS-120B by current profile)
      → Tavily through ResearchClient when explicitly enabled
      → fast device/media commands or LangGraph tasks when selected
  → TTS MCP (Deepgram Aura-2 when enabled) → browser playback
```

The ASR and TTS MCP services are started as local stdio subprocesses by the runtime; operators should not launch standalone speech servers. Configuration and provider keys are loaded server-side. Native task storage defaults locally and can be relocated using the existing task-database/workspace configuration. Task files are confined to that selected workspace. No runtime dependency on HPC, VPN, or a Raspberry Pi is intended.

Useful code landmarks:

| Area | Location |
|---|---|
| Native session, routes, lifecycle, events | `src/me3_voice/native_app.py`, `native_pipeline.py`, `native_bridge.py` |
| Provider requests, structured research, context | `src/me3_voice/native_llm.py`, `research.py`, `conversation_context.py`, `remote.py` |
| Speech clients and project MCP services | `src/me3_voice/speech.py`, `native_speech.py`, `servers.py` |
| LangGraph task runtime and workspace tools | `src/me3_voice/task_runtime.py` |
| Browser UI | `src/me3_voice/ui/` |
| CLI and foreground launcher | `src/me3_voice/cli.py`, `voice_bridge.py`, `scripts/swift.sh` |
| Tests and runtime configuration templates | `tests/`, `configs/`, `pyproject.toml`, `uv.lock` |

## 2. Current progress and evidence

- The last broad offline baseline recorded 452 passing tests. This handoff session ran only focused research/native tests: **5 passed** with local stdio IPC permitted; Ruff passed. The broad suite was not rerun.
- Prior live-provider backend replays completed ASR → LLM → TTS turns. Those establish backend behavior only, not laptop speaker quality or Pi acoustics.
- In the latest OpenRouter/Tavily campaign, the FIFA 2026 question passed from one search and two OpenRouter calls, with a valid official FIFA search-snippet citation and one `turn_completed`. Speech inputs and outputs in that replay were mocked.
- The Nobel 2026 question previously passed with the official Nobel Prize press release and `turn_completed`; it predates the latest single-snippet change.
- The latest Academy Awards replay safely abstained after a source-validation rejection and an official-page retrieval. Its final synthesis was blocked by the campaign input-token cap, so there was no `turn_completed`. A prior completed Oscar response cited an extracted passage that did not contain the winner; it is not a grounded pass.
- Current campaign ledger: 29 successful OpenRouter reservations, 493,756 of 500,000 estimated input tokens reserved, zero unknown outcomes, and 6,244 estimated tokens remaining. Do not reuse it for another provider call. The Tavily limit in the runtime is process-local rather than a durable campaign-wide ledger.
- The latest focused change prevents an unavailable research tool callback from falling through into its handler. Quick factual search now marks only the best-ranked result snippet as citation-eligible; other leads remain available for retrieval. This passed offline tests but the single-snippet behavior still needs fresh live qualification.
- Backend endpoint-to-packet timings in recent mock-speech research replays were 8.28 s for the FIFA answer, 9.48 s for the Oscar fallback, and 12.30 s for the earlier Nobel answer. Shorter first packets were acknowledgments. These are replay/backend timings, not physical audible latency. Mock ASR/TTS, mocked packets, and valid mock WAVs do not qualify microphone recognition or Aura playback.
- Existing WSL/laptop browser and AEC evidence applies only to the tested setup. Raspberry Pi software smoke evidence exists for one Pi 5 installation; physical audio/AEC, acoustic latency, full-duplex behavior, and final deployment remain separate qualification gates.

Detailed current evidence is in `docs/PLAN.md`, `docs/architecture_modernization/PHASE7_PREPARATION.md`, `PHASE7_RELIABILITY_HANDOFF.md`, and `OPERATOR_TEST_RUNBOOK.md`. The two latest research replay artifacts are under ignored `var/results/` and logs under ignored `var/logs/` on HPC; they are not part of the source publication.

## 3. WSL2 setup

Use Ubuntu under WSL2, Python 3.11–3.13, Git, and Windows Chrome or Edge. Windows browser microphone access should use the loopback HTTP address `http://127.0.0.1:8765/`; allow microphone permission and select the intended Windows microphone and speaker. WSL localhost forwarding ordinarily exposes the loopback service to the Windows browser. If it does not, use VS Code’s local port forwarding for port 8765. Do not expose the bridge on a LAN interface.

From Ubuntu/WSL2:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv build-essential
git clone --filter=blob:none --sparse https://github.com/ercallagon/ai231.git ~/ai231
git -C ~/ai231 sparse-checkout set me3-voice-agent
cd ~/ai231/me3-voice-agent
bash scripts/setup.sh --dev
bash scripts/swift.sh check
bash scripts/swift.sh demo
```

`setup.sh` uses the committed `uv.lock` and project-local `var/cache/venv`. It installs the default KWS support and `--dev` test/lint tools; it does not open a microphone, start services, call providers, or configure hardware. Do not copy the HPC venv to WSL or Pi. Run the deterministic demo before enabling live providers.

To provision the existing Sherpa wake asset and tokenized phrase (explicit network download; no training):

```bash
bash scripts/swift.sh assets --phrase "HEY SWIFT" \
  --keywords-output var/cache/wake/hey-swift-keywords.txt
```

The script stores the release bundle, selected ONNX/model files, manifest and keyword file under ignored `var/cache/wake/`. It refuses to overwrite an existing keyword file. Reuse an existing asset set or choose a new keyword output path rather than deleting operator data. Keep its model README/license notice with the assets.

## 4. Credentials and ignored runtime state

Only commit empty `.env.example`; the application does not automatically load `.env`. Never copy provider key values into chat, source, a command-line argument, screenshot, or a published profile. Credential files are permission-protected (not encrypted) and belong in ignored `var/cache/`.

Create the OpenRouter key file with the existing masked helper:

```bash
PYTHONPATH=src var/cache/venv/bin/python -B \
  scripts/configure_provider_keys.py openrouter-gpt-oss-120b \
  --output var/cache/openrouter-gpt-oss-keys.env
```

For a local file containing only the Deepgram and Tavily keys, use the project’s secure writer and hidden terminal input (the values are never command arguments):

```bash
PYTHONPATH=src var/cache/venv/bin/python -B - <<'PY'
from getpass import getpass
from pathlib import Path
from me3_voice.credentials import write_secrets

write_secrets(
    Path("var/cache/provider-keys.env"),
    {
        "ME3_DEEPGRAM_API_KEY": getpass("Deepgram API key (hidden): "),
        "ME3_RESEARCH_TAVILY_API_KEY": getpass("Tavily API key (hidden): "),
    },
)
print("Saved the selected credential names to a private local file; no request sent.")
PY
```

Use separate files for OpenRouter (`--llm-secrets-file`) and ASR/TTS (`--secrets-file`); pass Tavily only via `--research-secrets-file` when research is enabled. The bridge verifies required credential names/presence, not that the provider accepts them.

The following HPC files are ignored runtime state and must not be published:

- `var/cache/provider-keys.env`, `openrouter-gpt-oss-keys.env`, other `*.env` secrets, and any files containing tokens.
- `var/cache/provider-requests.sqlite` and its `-wal`/`-shm` files. It contains durable request reservations/outcomes. If preserving ledger history locally is required, transfer the database and sidecars through an approved private channel while the bridge is stopped; do not reset it. The current research campaign is exhausted and should not be resumed.
- `var/cache/swift-laptop-sustained-{backends,runtime}.json` and other generated profiles. Recreate local profiles from committed templates/configs and a newly approved budget; do not treat HPC run IDs or limits as fresh local authorization.
- `var/cache/wake/` inference assets and keyword file. Redownload with the script above or transfer through a trusted private channel with their license/provenance manifest.
- `var/recordings/` human recordings, browser AEC samples, replay fixtures and generated audio. Do not publish these; transfer an individual recording only when authorized and necessary.
- `var/cache/venv`, package caches, task SQLite/workspace state, `var/logs/`, `var/results/`, and all generated experiment/cache files. Recreate the environment locally; preserve only artifacts needed for a specific investigation.

Use no history-rewriting or cleanup commands. Local state can include private speech and machine-specific details.

## 5. Runtime commands

### Offline checks

```bash
bash scripts/swift.sh check
bash scripts/swift.sh demo
PYTHONPATH=src var/cache/venv/bin/python -B -m pytest -q \
  tests/test_phase4.py::test_native_relevant_search_snippet_answers_without_page_fetch \
  tests/test_phase4.py::test_unavailable_research_tool_does_not_fall_through_to_handler
var/cache/venv/bin/ruff check src/me3_voice tests
```

The two focused tests exercise a mocked native model/MCP path and are not live accuracy checks. Project instructions call for focused tests after changes; do not repeatedly rerun a broad suite without a reason.

### Mock native browser bridge

Run from the ME3 root. With no backend profile the bridge uses the project’s mock defaults; the speech MCP children start on browser session startup. No credentials or live calls are needed.

```bash
mkdir -p var/logs var/results
umask 077
set -o pipefail
bash scripts/swift.sh bridge --engine native \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt \
  --trace-dir var/results \
  2>&1 | tee var/logs/wsl-native-mock-bridge.log
```

In another WSL terminal, `bash scripts/swift.sh health 8765`; open `http://127.0.0.1:8765/` in the Windows browser and start one session. Stop with Ctrl+C in the bridge terminal. Mock TTS may sound synthetic or unintelligible and does not qualify real speech.

### Live native browser profile, only after a new approved budget

Start from `configs/native-speech.provisional.json` and `configs/runtime.example.json`, copy each to a new ignored path under `var/cache/`, and explicitly edit the new files for the selected providers, a **fresh approved** request/campaign limit, per-request and aggregate input-token ceilings, research permission/request limit, and native engine. The committed provisional profile is fail-closed (`allow_live=false`, request limit 0); do not edit it in place. OpenRouter is the current normal LLM choice; `configs/groq-gpt-oss-120b.rollback.json` is an explicit rollback profile.

Only after approval of the exact allocation, generate a fresh run ID and launch with the existing private credentials and traces:

```bash
export ME3_REQUEST_RUN_ID="swift-local-$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(4))')"
export ME3_REQUEST_LEDGER="$PWD/var/cache/provider-requests.sqlite"
export ME3_DISABLE_PROVIDER_RETRIES=1
umask 077
set -o pipefail
bash scripts/swift.sh bridge --engine native \
  --backend-config var/cache/local-native-backends.json \
  --runtime-config var/cache/local-native-runtime.json \
  --secrets-file var/cache/provider-keys.env \
  --llm-secrets-file var/cache/openrouter-gpt-oss-keys.env \
  --research-secrets-file var/cache/research-keys.env \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt \
  --followup-seconds 20 --trace-dir var/results \
  2>&1 | tee "var/logs/${ME3_REQUEST_RUN_ID}-bridge.log"
```

The example command is not authorization and its filenames must exist locally. Check for a fresh campaign cap before speaking; SQLite enforces provider request/token reservations, while the dollar estimate and Tavily’s process-local cap are not aggregate financial controls. Never automatically retry an unknown outcome. Records are written beneath ignored `var/results/` and `var/logs/`; campaign reservations are in `var/cache/provider-requests.sqlite`. The bridge JSONL event journal is sanitized, but inspect raw stderr privately before sharing.

## 6. Raspberry Pi 5 continuation

Use `deploy/rpi/README.md` as the canonical Pi setup and deployment guide. Install a separate aarch64 venv on the Pi with `bash scripts/setup.sh --pi` (and `--dev` only for development); never copy the WSL/HPC environment. Run the documented mock software acceptance and inventory first. A Pi 5/Raspberry Pi OS 64-bit/Python 3.13 software smoke has been reported on one target, but microphone/speaker operation, PipeWire/AEC graph, full-duplex barge-in, physical latency, GPIO effects, live provider calls and final demo readiness require separate Pi-side qualification. Laptop browser AEC does not prove Pi acoustic AEC.

## 7. Next work, in order

1. In WSL2, reproduce the focused native research cases and mock browser session; verify local traces and clean shutdown.
2. Under a fresh separately approved provider budget, validate the one-snippet Oscars path and an independent current/research query. Improve reasoned relevance/claim support only with a reproduced error; preserve honest abstention.
3. Add durable campaign accounting for Tavily before multi-run research campaigns; the current Tavily cap resets per process.
4. Qualify one real OpenRouter/Tavily answer, then separate real ASR/TTS browser voice and physical audibility. Keep endpoint→first meaningful audio distinct from acknowledgment packet, provider first PCM, playback scheduling and complete-answer time.
5. Progress to same-session follow-ups, interruption/barge-in, speaker/AEC checks, then Pi hardware/audio qualification and professor demonstration evidence.

Do not switch providers, reset ledgers, publish credentials/recordings, push forcefully, or begin a deployment/architecture phase as a shortcut. Preserve the native Pipecat + LangGraph design and legacy fallback.
