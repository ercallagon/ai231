# Phase 2 validation tools

These isolated tools validate native Pipecat integration and attended Pi audio;
they do not replace production Session or the sidecar. No inference provider is
required. Run from the ME3 project root with its existing environment.

Canonical commands, temporary AEC setup, routing checks, return artifacts and
rollback: [Phase 2 report](../../docs/architecture_modernization/PHASE2_REPORT.md).

- `native_contract.py OUTPUT.json`: three native scenarios, actual speech MCP children with mock providers.
- `session_comparison.py OUTPUT.json`: two legacy comparison scenarios with the same mock MCP services.
- `pipewire_transport.py`: native input/output prototype; devices disabled by default.
- `audio_inventory.py --output var/results/NEW.json`: read-only local audio/hash/manifest inventory; no uploads.
- `operator_native.py`: attended physical acquisition; requires explicit nodes, reference WAV and `--capture`.
- `../../tests/test_phase2_transport.py`: eight offline lifecycle/transport checks.

Use fresh JSON paths under ignored `var/results/`; output overwrite is refused.
Historical development logs remain private under `var/experiments/022-native-voice-duplex/`.
Hardware gates remain PENDING_OPERATOR_VALIDATION until actual results are reviewed.

For an extended offline interval, use the supplemental capture profiles and return checklist in the Phase2 report. Only smoke enables interruption; speech/echo/overlap/negative are explicitly labeled corpus modes.

## Offline repairs / native speech promotion — 2026-11-10

Canonical STT/TTS adapters now live in src/me3_voice/native_speech.py and composition in native_pipeline.py; native_contract.py imports them. Audio harness now records diagnostic WAV containers directly, separates intentional shutdown EOF, preserves cleanup failures and emits visible readiness/countdown cues. qualify_audio.sh is the minimal attended entrypoint: two confirmed audible WAV probes before native smoke. The script does not load AEC or change services. WAV preroll is not the24s trial clock. Phase2 remains PENDING_OPERATOR_VALIDATION. Phase3 source/tests/benchmarks/partial rollout: ../../docs/architecture_modernization/PHASE3_REPORT.md.
