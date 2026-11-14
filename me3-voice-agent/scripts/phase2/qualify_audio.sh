#!/usr/bin/env bash
# Attended qualification entrypoint. No module loading, SSH or service changes.
set -euo pipefail
if [[ "${1:-}" == "--help" || $# -ne 7 || "${7:-}" != "--capture" ]]; then
  echo 'From ME3 root: bash scripts/phase2/qualify_audio.sh PHYSICAL_MIC PHYSICAL_SPEAKER AEC_SOURCE AEC_SINK REFERENCE.wav var/results/NEW_RUN --capture'
  echo 'AEC must already be configured and its graph reviewed. Two audible probes must pass before the 24s native smoke. No devices open without all arguments and --capture.'
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi
ME3_AUDIO_MIC=$1
ME3_AUDIO_SPEAKER=$2
ME3_AUDIO_SOURCE=$3
ME3_AUDIO_SINK=$4
ME3_AUDIO_REFERENCE=$5
ME3_AUDIO_OUTPUT=$6
var/cache/venv/bin/python -B - "$ME3_AUDIO_REFERENCE" "$ME3_AUDIO_OUTPUT" <<'PY'
from pathlib import Path
import sys,wave
reference,output=map(Path,sys.argv[1:])
if not output.resolve().is_relative_to(Path('var').resolve()):
    raise SystemExit('Output must be under project var/')
with wave.open(str(reference),'rb') as w:
    if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate() not in (16000,48000) or not w.getnframes():
        raise SystemExit('Reference must be nonempty mono PCM16/16k or48k WAV')
output.mkdir(parents=True,exist_ok=False)
PY
wpctl status -n > "$ME3_AUDIO_OUTPUT/status-before.txt" 2>&1
pw-play --version > "$ME3_AUDIO_OUTPUT/pw-play-version.txt" 2>&1
pw-play --help > "$ME3_AUDIO_OUTPUT/pw-play-help.txt" 2>&1
pw-record --help > "$ME3_AUDIO_OUTPUT/pw-record-help.txt" 2>&1
pw-dump > "$ME3_AUDIO_OUTPUT/graph-before.json"
echo '[PHYSICAL SPEAKER PROBE] Listen now. Stop if output is uncomfortable.'
if ! timeout 20s pw-play --target="$ME3_AUDIO_SPEAKER" "$ME3_AUDIO_REFERENCE" > "$ME3_AUDIO_OUTPUT/physical-probe.log" 2>&1; then
  echo 'Physical playback command failed/timed out. STOP; return probe log.'
  exit 1
fi
read -r -p 'Was reference clearly audible? Type audible to proceed: ' ME3_AUDIO_HEARD
printf 'physical speaker: %s\n' "$ME3_AUDIO_HEARD" > "$ME3_AUDIO_OUTPUT/operator-observations.txt"
if [[ "$ME3_AUDIO_HEARD" != audible ]]; then
  echo 'Audibility unconfirmed. STOP; no native/acoustic qualification.'
  exit 1
fi
echo '[AEC REFERENCE PROBE] Listen now through the reviewed reference sink.'
if ! timeout 20s pw-play --target="$ME3_AUDIO_SINK" "$ME3_AUDIO_REFERENCE" > "$ME3_AUDIO_OUTPUT/aec-probe.log" 2>&1; then
  echo 'AEC reference command failed/timed out. STOP; return probe log and graph.'
  exit 1
fi
read -r -p 'Was reference clearly audible through AEC? Type audible: ' ME3_AUDIO_HEARD
printf 'AEC sink: %s\n' "$ME3_AUDIO_HEARD" >> "$ME3_AUDIO_OUTPUT/operator-observations.txt"
if [[ "$ME3_AUDIO_HEARD" != audible ]]; then exit 1; fi
# Runtime PCM streams require this capability; do not silently feed AU/WAV
# headers into a raw sample converter when an older CLI lacks --raw.
if ! grep -q -- '--raw' "$ME3_AUDIO_OUTPUT/pw-play-help.txt" || ! grep -q -- '--raw' "$ME3_AUDIO_OUTPUT/pw-record-help.txt"; then
  echo 'Installed CLI lacks --raw. Preserve successful WAV probes; native PCM streaming compatibility remains blocked. Do not upgrade/reconfigure services automatically.'
  exit 1
fi
printf 'mic=%s\nspeaker=%s\nsource=%s\nreference_sink=%s\n' "$ME3_AUDIO_MIC" "$ME3_AUDIO_SPEAKER" "$ME3_AUDIO_SOURCE" "$ME3_AUDIO_SINK" > "$ME3_AUDIO_OUTPUT/targets.txt"
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=var/cache/numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  var/cache/venv/bin/python -B scripts/phase2/operator_native.py \
  --raw-mic "$ME3_AUDIO_MIC" --source "$ME3_AUDIO_SOURCE" --sink "$ME3_AUDIO_SINK" \
  --reference "$ME3_AUDIO_REFERENCE" --output "$ME3_AUDIO_OUTPUT/smoke" --capture
pw-dump > "$ME3_AUDIO_OUTPUT/graph-after.json"
echo 'Capture saved; qualification remains PENDING_OPERATOR_VALIDATION until graphs, WAVs, diagnostics.json and native-audio.json are reviewed. Unload only the temporary AEC module you own using the runbook.'
