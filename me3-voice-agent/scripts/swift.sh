#!/usr/bin/env bash
# Foreground commands only; exec preserves Ctrl+C and owns no unrelated process.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$PWD/var/cache/numba"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ -z "${ME3_MUSIC_CONFIG:-}" && -f "$PWD/var/cache/music.json" ]]; then
  export ME3_MUSIC_CONFIG="$PWD/var/cache/music.json"
fi
python_bin="$PWD/var/cache/venv/bin/python"
uv_bin="$PWD/var/cache/venv/bin/uv"
if [[ ! -x "$uv_bin" && -x "$PWD/var/cache/uv-bin/uv" ]]; then
  uv_bin="$PWD/var/cache/uv-bin/uv"
fi
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$command" == help || "$command" == --help ]]; then
  echo 'Usage: bash scripts/swift.sh {check|demo|dashboard|bridge|native-bridge|laptop-demo|setup|vpn-setup|vpn|preflight|network|assets|audio|health} [arguments]'
  echo 'Use "setup --interactive" for the guided production configuration wizard; "setup --status" is offline.'
  echo 'bridge requires explicit backend, secrets and wake configuration; audio requires explicit devices.'
  exit 0
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'Missing environment. Run: bash scripts/setup.sh' >&2; exit 2
fi
native_ready() {
  "$python_bin" -B - "$1" <<'PY'
import json
import hashlib
import os
import sys
import urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/health", timeout=0.5) as response:
        value = json.load(response)
except Exception:
    raise SystemExit(1)
ready = value.get("bridge") == "ready" and value.get("engine") == "native"
profile_id = os.environ.get("ME3_PROFILE_ID")
if ready and profile_id:
    profile = value.get("configuration_profile") or {}
    paths = (
        ("backend_sha256", os.environ.get("ME3_PROFILE_BACKEND_CONFIG", "")),
        ("runtime_sha256", os.environ.get("ME3_PROFILE_RUNTIME_CONFIG", "")),
    )
    expected = {
        key: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for key, path in paths
        if path
    }
    ready = profile.get("id") == profile_id and all(
        profile.get(key) == digest for key, digest in expected.items()
    )
raise SystemExit(0 if ready else 1)
PY
}
port_is_open() {
  "$python_bin" -B - "$1" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.3):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}
open_dashboard() {
  local url="http://127.0.0.1:${1}/"
  "$python_bin" -B - "$url" <<'PY'
import shutil
import subprocess
import sys
url = sys.argv[1]
for command in (("wslview", url), ("explorer.exe", url), ("xdg-open", url), ("gio", "open", url)):
    executable = shutil.which(command[0])
    if executable:
        try:
            subprocess.Popen((executable, *command[1:]), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            print(f"Opened Swift UI: {url}")
            raise SystemExit(0)
        except OSError:
            continue
print(f"No desktop browser opener is available. Open: {url}")
raise SystemExit(1)
PY
}
report_demo_bridge() {
  "$python_bin" -B - "$1" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/health", timeout=2) as response:
        health = json.load(response)
except (OSError, ValueError):
    raise SystemExit("Swift production bridge is unavailable. Start the configured Pi profile first.")
if health.get("bridge") != "ready" or health.get("engine") != "native" or health.get("voice_ready") is not True:
    raise SystemExit("Swift production bridge is not voice ready. Check its startup terminal.")
print("Running bridge revision:", health.get("source_revision") or "unreported by this running bridge")
print("Native voice ready; session:", health.get("voice_session"))
for role, quota in health.get("budget", {}).items():
    if role.endswith("_requests") and isinstance(quota, dict) and quota.get("limit") is not None:
        print(f"{role.removesuffix('_requests')}: {quota.get('remaining')} of {quota['limit']} requests left")
PY
}
case "$command" in
  check)
    "$uv_bin" pip check --python "$python_bin"
    "$python_bin" -B -c 'import aiohttp, mcp, numpy, pipecat, soxr, sherpa_onnx, sentencepiece; print("Runtime imports OK; no devices opened")'
    exec "$python_bin" -B -m me3_voice.preflight "$@" ;;
  demo)
    if [[ "${1:-}" == --report-only ]]; then shift; exec "$python_bin" -B -m me3_voice.cli demo "$@"; fi
    if [[ "${1:-}" == --connect ]] || { (( $# == 0 )) && grep -aq 'Raspberry Pi' /proc/device-tree/model 2>/dev/null; }; then
      if [[ "${1:-}" == --connect && $# != 1 ]]; then exit 2; fi
      report_demo_bridge 18770
      open_dashboard 18770 || :
      exit 0
    fi
    if [[ "${1:-}" == --offline ]]; then shift; fi
    if (( $# )); then exec "$python_bin" -B -m me3_voice.cli demo "$@"; fi
    # Keep the smoke report and transcript separate from the production session.
    mkdir -p var/results
    demo_workspace="$(mktemp -d "$PWD/var/results/swift-demo.XXXXXXXX")"
    cd "$demo_workspace"
    mkdir -p var/results
    printf 'Offline demo checkout revision: %s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    "$python_bin" -B -m me3_voice.cli demo
    exec "$python_bin" -B -m me3_voice.dashboard serve --open-browser ;;
  dashboard) exec "$python_bin" -B -m me3_voice.dashboard serve "$@" ;;
  bridge) exec "$python_bin" -B -m me3_voice.voice_bridge "$@" ;;
  laptop-demo)
    demo_port=18770
    demo_backend_config="configs/native-demo-backends.json"
    demo_runtime_config="configs/native-demo-runtime.json"
    export ME3_PROFILE_ID=swift-laptop-demo
    export ME3_PROFILE_BACKEND_CONFIG="$demo_backend_config"
    export ME3_PROFILE_RUNTIME_CONFIG="$demo_runtime_config"
    if native_ready "$demo_port"; then
      echo "Reusing the ready native Swift bridge on 127.0.0.1:${demo_port}."
      if ! open_dashboard "$demo_port"; then :; fi
      exit 0
    fi
    if port_is_open "$demo_port"; then
      echo "Port ${demo_port} is occupied by a bridge that does not match the current laptop-demo profile; no process was stopped." >&2
      exit 2
    fi
    # Each new production-demo launch gets a fresh private ledger. Production
    # limits come from the versioned configs; old run ledgers are untouched.
    umask 077
    mkdir -p "$PWD/var/results"
    demo_run_dir="$(mktemp -d "$PWD/var/results/phase6-laptop-demo-run-XXXXXXXX")"
    demo_run_name="$(basename -- "$demo_run_dir")"
    demo_run_id="swift-${demo_run_name}"
    demo_ledger="$demo_run_dir/provider-requests.sqlite"
    export ME3_REQUEST_RUN_ID="$demo_run_id" ME3_REQUEST_LEDGER="$demo_ledger"
    export ME3_RESEARCH_LEDGER="$demo_ledger"
    printf 'Swift laptop demo run: %s\nProvider ledger: %s\n' \
      "$ME3_REQUEST_RUN_ID" "$ME3_REQUEST_LEDGER" >&2
    bash "$0" native-bridge \
      --runtime-config "$demo_runtime_config" \
      --backend-config "$demo_backend_config" \
      --secrets-file var/cache/selected-provider-credentials.env \
      --llm-secrets-file var/cache/selected-provider-credentials.env \
      --research-secrets-file var/cache/selected-provider-credentials.env \
      --port "$demo_port" &
    bridge_pid=$!
    stop_demo_bridge() {
      if kill -0 "$bridge_pid" 2>/dev/null; then
        kill -INT "$bridge_pid" 2>/dev/null || true
        wait "$bridge_pid" 2>/dev/null || true
      fi
    }
    trap 'stop_demo_bridge; exit 130' INT
    trap 'stop_demo_bridge; exit 143' TERM
    ready=false
    for attempt in $(seq 1 60); do
      if ! kill -0 "$bridge_pid" 2>/dev/null; then
        wait "$bridge_pid" || exit $?
      fi
      if native_ready "$demo_port"; then ready=true; break; fi
      sleep 0.5
    done
    if [[ "$ready" != true ]]; then
      stop_demo_bridge
      echo 'Swift native bridge did not become ready within 30 seconds; see startup errors above.' >&2
      exit 1
    fi
    echo "Swift native bridge ready at http://127.0.0.1:${demo_port}/. Local clients and task stores were initialized at startup; remote providers were not probed or billed."
    if ! open_dashboard "$demo_port"; then :; fi
    set +e
    wait "$bridge_pid"
    bridge_status=$?
    set -e
    trap - INT TERM
    exit "$bridge_status" ;;
  native-bridge)
    # Supported native browser entrypoint. No backend descriptor means mocks;
    # live providers require an explicit backend profile and credentials.
    ME3_SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
    export ME3_SOURCE_REVISION
    native_state="${XDG_STATE_HOME:-$HOME/.local/state}/swift-me3"
    if [[ ! -e "$native_state" ]]; then
      (umask 077; mkdir -p "$native_state")
    fi
    if [[ -L "$native_state" || ! -d "$native_state" ]]; then
      echo 'Native task state must be a private, user-owned directory.' >&2; exit 2
    fi
    if [[ ! -e "$native_state/workspace" ]]; then
      (umask 077; mkdir "$native_state/workspace")
    fi
    for private_dir in "$native_state" "$native_state/workspace"; do
      if [[ -L "$private_dir" || ! -d "$private_dir" || \
            "$(stat -c '%u' "$private_dir")" != "$(id -u)" || \
            "$(stat -c '%a' "$private_dir")" != 700 ]]; then
        echo 'Native task state directories must be owned by this user with mode 700.' >&2; exit 2
      fi
    done
    export ME3_TASK_DATABASE="${ME3_TASK_DATABASE:-$native_state/tasks.sqlite}"
    export ME3_TASK_WORKSPACE="${ME3_TASK_WORKSPACE:-$native_state/workspace}"
    exec "$python_bin" -B -m me3_voice.voice_bridge --engine native \
      --kws-assets var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
      --keywords-file var/cache/wake/hey-swift-keywords.txt "$@" ;;
  vpn-setup) exec "$python_bin" -B scripts/configure_vpn.py "$@" ;;
  setup)
    case "${1:-}" in
      --interactive|--status) exec "$python_bin" -B -m me3_voice.setup_wizard "$@" ;;
      --github-token|--github-repository) exec "$python_bin" -B scripts/setup_github_write.py "$@" ;;
      *) exec "$python_bin" -B -m me3_voice.setup_wizard --interactive "$@" ;;
    esac ;;
  vpn) exec bash scripts/swift_vpn.sh "$@" ;;
  preflight) exec "$python_bin" -B -m me3_voice.preflight "$@" ;;
  network) exec "$python_bin" -B scripts/check_network.py "$@" ;;
  assets) exec "$python_bin" -B scripts/fetch_kws_assets.py "$@" ;;
  audio) exec "$python_bin" -B -m me3_voice.pi_audio "$@" ;;
  health)
    exec "$python_bin" -B - "$@" <<'PY'
import json
import sys
import urllib.request
port = int(sys.argv[1]) if len(sys.argv) > 1 else 18770
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
