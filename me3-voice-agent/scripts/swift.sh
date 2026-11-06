#!/usr/bin/env bash
# Foreground commands only; exec preserves Ctrl+C and owns no unrelated process.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$PWD/var/cache/numba"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python_bin="$PWD/var/cache/venv/bin/python"
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$command" == help || "$command" == --help ]]; then
  echo 'Usage: bash scripts/swift.sh {check|demo|dashboard|bridge|preflight|network|assets|audio|health} [arguments]'
  echo 'bridge requires explicit backend, secrets and wake configuration; audio requires explicit devices.'
  exit 0
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'Missing environment. Run: bash scripts/setup.sh' >&2; exit 2
fi
case "$command" in
  check)
    var/cache/venv/bin/uv pip check --python "$python_bin"
    "$python_bin" -B -c 'import aiohttp, mcp, numpy, pipecat, soxr, sherpa_onnx, sentencepiece; print("Runtime imports OK; no devices opened")'
    exec "$python_bin" -B -m me3_voice.preflight "$@" ;;
  demo) exec "$python_bin" -B -m me3_voice.cli demo "$@" ;;
  dashboard) exec "$python_bin" -B -m me3_voice.dashboard serve "$@" ;;
  bridge) exec "$python_bin" -B -m me3_voice.voice_bridge "$@" ;;
  preflight) exec "$python_bin" -B -m me3_voice.preflight "$@" ;;
  network) exec "$python_bin" -B scripts/check_network.py "$@" ;;
  assets) exec "$python_bin" -B scripts/fetch_kws_assets.py "$@" ;;
  audio) exec "$python_bin" -B -m me3_voice.pi_audio "$@" ;;
  health)
    exec "$python_bin" -B - "$@" <<'PY'
import json
import sys
import urllib.request
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
if not 1 <= port <= 65535:
    raise SystemExit("Invalid port")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
        print(json.dumps(json.load(response), indent=2))
except Exception:
    raise SystemExit("Bridge health unavailable; inspect the foreground terminal and chosen port.")
PY
    ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
