# ME3 Operator Test Runbook

Run commands from the ME3 root on HPC unless marked **Windows** or **Pi**. Keep raw audio and outputs in ignored `var/`. Use a new `ME3_REQUEST_RUN_ID` for each separately authorized live run; never delete/reset `var/cache/provider-requests.sqlite`. Keys stay in `var/cache/provider-keys.env` (mode 0600), never source them into shell history or paste them into chat.

Two native Nova-3 → Groq → Aura-2 WAV replays have completed with valid WAV and `turn_completed` (see `docs/PLAN.md`); this is backend evidence, not browser acoustic evidence. An operator laptop AEC probe passed for one setup, but a real persistent browser conversation and Pi acoustics remain unqualified. Both MCP services are **stdio child processes started by replay/bridge**; do not launch standalone MCP commands. Run their initialization preflight in a terminal/environment that permits local subprocess IPC. No provider call below is authorized by this document: obtain a new explicit allocation before each live stage.

## Priority checklist

| Order | Test | Provider budget / gate | Status |
|---|---|---|---|
| A | Full native WAV replay | Previously passed twice; repeat only for a specific regression under new approval | Backend PASS |
| B | Laptop browser AEC A/B, mock-first | Zero provider calls; speakers on, headphones off | One operator setup passed; repeat only if devices/routing change |
| C | One live laptop conversation | Separate approval; use one-turn limits (1/1/3) and a fresh run ID | After A passes |
| D | Barge-in plus follow-up | Separate approval and a larger profile (suggested ceiling: ASR 3, LLM 3, TTS 9 segments × 500 chars; research 0) | Requires a newly prepared profile; one-turn profile is insufficient |
| E | Raspberry Pi audio/AEC | No cloud calls for physical probes; device-specific | Run existing qualification script only after Pi endpoints/AEC graph are known |

## A. Full real-provider replay (HPC) — already qualified

Two complete native Nova-3 → Groq → Aura-2 WAV replays passed earlier in Phase 7, so do not repeat the nine-second replay for laptop preparation. Their artifacts and request ledgers remain under ignored `var/results/` and `var/cache/`. A future regression replay needs a separate approved campaign and a newly validated backend profile; older private profiles do not satisfy the current live Groq input-token-ceiling validation. The browser test below is the next functional gate.

## B. Windows laptop microphone/speaker and browser AEC (mock-first)

**Prerequisites:** Windows with Chrome or Edge; same USB mic/speaker planned for Pi; headphones off; stable safe speaker volume; HPC VS Code Remote-SSH session; existing wake assets. No live keys/profile needed. AEC probe works on loopback/secure browser context and performs local capture/playback only.

HPC terminal 1, from project root (mock providers; starts loopback native bridge and its speech MCP children when needed):

```bash
bash scripts/swift.sh bridge --engine native \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt
```

In terminal 2: `bash scripts/swift.sh health 8765`. Require `bridge: ready`, `engine: native`; `speech_services: lazy` is expected until browser Start. To check both project-owned MCP children without inference, run:

```bash
PYTHONPATH=src var/cache/venv/bin/python -B - <<'PYCODE'
import asyncio
from me3_voice.configuration import load_backends
from me3_voice.speech import preflight_mock_speech
rows = asyncio.run(preflight_mock_speech(load_backends()))
print([(row['service'], row['ok'], row['exit_code']) for row in rows])
assert len(rows) == 2 and all(row['ok'] for row in rows)
PYCODE
```

In VS Code **Ports**, forward remote port `8765`; on Windows open `http://127.0.0.1:8765/`. Set Windows default input/output to the USB devices, allow microphone permission, confirm the dashboard shows mic activity. Browser AEC is requested by the UI (`echoCancellation` ideal true; NS/AGC ideal false); settings are not a measurement. Do not enable a second AEC stage.

For the existing A/B probe, open `/audio-probe` (or dashboard link). On Windows, choose an existing labeled speech recording as the reference (for example download `var/recordings/guided/664ebd3ed18d4724971d7a382aa11aef/boiling-e28411528d444783851e1f6345a3cf2f.webm` from the remote workspace using VS Code). This is a human speech reference, not assistant TTS. Run once with AEC Off and once On, same device placement/volume/reference. Follow page cues: 0–8 s quiet with playback, 8–16 s natural double-talk, 16–24 s playback stopped while continuing speech. Save both downloaded `.webm` and `.json` files locally; the probe does not upload them. Listen for residual echo, intelligibility, clipping, gain changes and dropped speech. AEC A/B is separate from application barge-in. The probe has no sample-synchronized loopback/ERLE score and cannot verify AEC effectiveness automatically.

Then test the mock native UI conversation and stop speaking during mock playback. UI/browser input/output confirms transport behavior only; mock TTS may not be intelligible and does not qualify live speech or AEC. Stop the bridge with **Ctrl+C in terminal 1**; confirm the terminal exits before another launch. Stop only your own forwarded port/session.

**Evidence and limits:** Save the AEC probe's downloaded `.webm` and `.json` files locally, plus screenshots of the dashboard/transcript and operator notes. The native bridge now writes one private `native-<application-run-id>.jsonl` per WebSocket connection under `--trace-dir` (default `var/results/`). Rows include session/turn IDs, monotonic timestamps and symbolic/numeric wake, VAD, ASR, LLM/tool, TTS, playback, interruption, error and completion events. Speech text, prompts, credentials and provider payloads are deliberately excluded. Reconnect creates a new run file with the same session ID; explicit End resets the session. The UI transcript remains in memory, so capture a screenshot if words matter. Capture process stderr separately:

```bash
mkdir -p var/logs var/results
umask 077
set -o pipefail
bash scripts/swift.sh bridge --engine native \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt \
  --trace-dir var/results \
  2>&1 | tee var/logs/laptop-native-mock-bridge.log
```

The private process log and sanitized event JSONL are complementary. Keep raw stderr local; review it before sharing because dependency diagnostics are not covered by the JSONL allowlist. Neither records raw microphone or speaker audio by default. No live provider calls occur when backend config is omitted. To find the latest session file after Start: `ls -lt var/results/native-*.jsonl | head`.

**Stop/fail:** no health response, missing microphone permission/activity, no selected speaker output, recording error, or audible discomfort. Save probe downloads, screenshots and exact browser/device/settings notes. Do not treat browser-reported AEC as acoustic pass.

## C–D. Live laptop conversation, then barge-in/multi-turn

Run C only after A passes and after a separate explicit approval. **Do not reuse the historical `native-live-approved-20261111-02` profile:** current live chat validation requires positive `max_input_tokens` and `max_input_tokens_total`, and previous run IDs may have consumed reservations. Prepare a new private `var/cache/<approved-run>-backends.json` with the approved Nova-3/Groq/Aura-2 limits, both Groq input ceilings, and `allow_live: true` only after approval. Keep research disabled in a matching runtime file. Validate both with `load_backends()`/`load_runtime()` before launch and check `request_ledger_summary()` shows zero rows for the fresh ID. In the HPC terminal, use that fixed approved ID, ledger path and retry-off setting (then verify both speech MCP subprocesses initialize before browser testing):

```bash
export ME3_REQUEST_RUN_ID="manual-laptop-$(date -u +%Y%m%dT%H%M%SZ)"
export ME3_REQUEST_LEDGER="$PWD/var/cache/provider-requests.sqlite"
export ME3_DISABLE_PROVIDER_RETRIES=1
```

Before the live bridge starts, verify the *newly approved* private files and empty run ledger without printing keys:

```bash
PYTHONPATH=src var/cache/venv/bin/python -B - <<'PYCODE'
import os
from pathlib import Path
from me3_voice.configuration import load_backends, load_runtime
from me3_voice.credentials import load_secrets
from me3_voice.remote import request_ledger_summary
run = os.environ['ME3_REQUEST_RUN_ID']
base = Path('var/cache') / run
b = load_backends(Path(str(base) + '-backends.json'))
r = load_runtime(Path(str(base) + '-runtime.json'))
assert r.engine == 'native' and not r.research.allow_live
assert [(b[k].protocol, b[k].request_limit) for k in ('asr','llm','tts')] == [
    ('deepgram-listen-v1', 1), ('chat-completions-v1', 1), ('deepgram-speak-v1', 3)]
assert [b[k].model for k in ('asr','llm','tts')] == [
    'nova-3', 'openai/gpt-oss-120b', 'aura-2-thalia-en']
assert all(b[k].allow_live for k in ('asr','llm','tts'))
assert b['llm'].max_input_tokens > 0 and b['llm'].max_input_tokens_total > 0
keys = load_secrets('var/cache/provider-keys.env', {'ME3_DEEPGRAM_API_KEY','ME3_GROQ_API_KEY'})
assert all(keys.get(k) for k in ('ME3_DEEPGRAM_API_KEY','ME3_GROQ_API_KEY'))
assert not request_ledger_summary(run)['requests']
print('New profile, credentials, research-disabled runtime and empty ledger: PASS')
PYCODE
```

Start:

```bash
umask 077
mkdir -p var/logs var/results
bash scripts/swift.sh bridge --engine native \
  --backend-config "var/cache/${ME3_REQUEST_RUN_ID}-backends.json" \
  --runtime-config "var/cache/${ME3_REQUEST_RUN_ID}-runtime.json" \
  --secrets-file var/cache/provider-keys.env \
  --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --keywords-file var/cache/wake/hey-swift-keywords.txt \
  --trace-dir var/results \
  2>&1 | tee "var/logs/${ME3_REQUEST_RUN_ID}-bridge.log"
```

Run `mkdir -p var/logs var/results` before launch. Forward port 8765 and open the existing UI. Speak one stable factual question; verify UI transcript/answer, actual USB speaker response and ledger. Preserve the bridge log, `var/results/native-<application-run-id>.jsonl`, screenshots, operator notes and the sanitized provider-ledger summary. Check `turn_completed` in JSONL and absence of `turn_error`; a browser playback acknowledgement is not acoustic proof. Stop after the approved turns or any failure/cap exhaustion; Ctrl+C bridge. Do not exceed the approved campaign.

D requires a separately approved budget and a newly created private backend profile under `var/cache/` with explicit `request_limit` values for Nova-3, Groq and Aura-2. Suggested maximum for one initial answer, one interruption, and one follow-up: **ASR 3, LLM 3, TTS 9 segments × 500 characters; retries off; research 0**. This is a proposed ceiling, not authorization. Do not edit/reuse a spent profile in place. Run the native bridge with this profile, a runtime with research disabled, a fresh run ID and the authorized ledger. Speak during assistant playback; pass only if old playback stops, the interruption is transcribed, the next answer reflects the latest utterance, and no stale audio resumes. Record all consumed segments/requests. Stop at any provider error or cap; never retry.

## E. Raspberry Pi Phase 2 audio/AEC (separate physical gate)

Requires operator access to Pi, actual device identifiers, PipeWire AEC module already configured/reviewed, and a known mono PCM16 reference WAV at 16 or 48 kHz. Do not infer Pi readiness from laptop tests. On Pi, from ME3 root, replace uppercase placeholders with verified `wpctl status -n` device/node names and use a new ignored output directory:

```bash
bash scripts/phase2/qualify_audio.sh PHYSICAL_MIC_NODE PHYSICAL_SPEAKER_NODE \
  me3.phase2.aec.source me3.phase2.aec.sink \
  var/results/native-live-smoke/boiling-question-16k.wav \
  var/results/phase2-qualification-NEW --capture
```

The script accepts only nonempty mono PCM16 WAV at 16 or 48 kHz; the fixture above is the existing 9-second 16 kHz WAV. Run only after actual mic/speaker nodes and the reviewed AEC source/sink exist. It first plays through the physical sink and AEC sink and requires the operator to type `audible` twice; inaudibility or CLI incompatibility stops before native smoke. Verify graph links show playback routed through the AEC reference sink, physical mic into AEC capture, AEC source into native capture. Follow prompts; preserve all output; unload only the temporary module you created. Phase 2 stays `PENDING_OPERATOR_VALIDATION` until diagnostic JSON, WAVs, graphs, resources and actual double-talk/barge-in observations are reviewed.

## Operator notes and handoff

For each manual acoustic turn, copy and fill:

```text
run ID / date / room:
OS + browser/version; USB mic + speaker; placement/distance; Windows or PipeWire route:
speaker volume; AEC requested / reported; noise suppression / AGC reported:
playback-only echo (none / slight / disruptive); double-talk words preserved / missed:
clipping, level changes, false interruptions; playback-stop observation (subjective ms or “not measured”):
UI transcript vs spoken words; UI answer vs heard answer; errors:
```

Provide the sanitized browser trace `var/results/native-*.jsonl`, matching `var/logs/*-bridge.log`, a sanitized provider-ledger summary for the approved campaign, laptop A/B settings JSON and operator notes. If a separately approved WAV replay is run, include its result JSON and WAV validity/duration. Never send secrets or the private key file. The latest recorded full offline regression is **704 passed**; this browser-preparation change was checked with focused tests only. Pi performance, new laptop voice acoustics and full-duplex barge-in remain unqualified.
