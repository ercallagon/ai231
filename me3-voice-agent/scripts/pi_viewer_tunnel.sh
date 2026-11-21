#!/usr/bin/env bash
# WSL-side, loopback-only SSH tunnel to the Pi's existing WayVNC WebSocket.
set -euo pipefail

dashboard=false
if [[ "${1:-}" == --dashboard ]]; then
  dashboard=true
  shift
fi
host="${1:-${SWIFT_PI_HOST:-}}"
if [[ -z "$host" ]]; then
  echo 'Usage: ./scripts/swift.sh pi-viewer [--dashboard] user@pi-host' >&2
  exit 2
fi
[[ "$host" == *@* ]] || host="ej@$host"
if [[ $# -gt 1 ]]; then
  echo 'Usage: ./scripts/swift.sh pi-viewer [--dashboard] user@pi-host' >&2
  exit 2
fi

port="${SWIFT_PI_NOVNC_PORT:-15901}"
remote_port=5901
state_name=swift-me3-pi-viewer
if "$dashboard"; then
  port="${SWIFT_PI_DASHBOARD_PORT:-18771}"
  remote_port=18770
  state_name=swift-me3-pi-dashboard
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo 'The local tunnel port must be between 1024 and 65535.' >&2
  exit 2
fi

key="${SWIFT_PI_SSH_KEY:-$HOME/.ssh/id_ed25519_swift_pi}"
if [[ ! -r "$key" ]]; then
  echo "Pi SSH identity is unavailable at the configured path; set SWIFT_PI_SSH_KEY." >&2
  exit 2
fi

state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/$state_name"
mkdir -p -- "$state_dir"
if [[ -L "$state_dir" || ! -d "$state_dir" || "$(stat -c '%u' "$state_dir")" != "$(id -u)" ]]; then
  echo 'The private Pi-viewer state directory is not a safe user-owned directory.' >&2
  exit 2
fi
chmod 700 "$state_dir"
control="$state_dir/ssh-control"
target_file="$state_dir/ssh-target"

ssh_base=(
  ssh -F /dev/null -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes
  -o StrictHostKeyChecking=yes -o ConnectTimeout=10
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes
  -o ExitOnForwardFailure=yes -S "$control"
)

probe_tunnel() {
  if "$dashboard"; then
    python3 - "$port" <<'PY'
import json
import sys
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f"http://127.0.0.1:{sys.argv[1]}/api/health", timeout=3) as r:
    health = json.load(r)
if health.get("bridge") != "ready" or health.get("engine") != "native":
    raise SystemExit("The tunnel did not reach a ready Swift native bridge.")
PY
    return
  fi
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
request = (
    "GET /websockify HTTP/1.1\r\n"
    f"Host: 127.0.0.1:{port}\r\n"
    "Connection: Upgrade\r\n"
    "Upgrade: websocket\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
).encode("ascii")
with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
    connection.settimeout(3)
    connection.sendall(request)
    response = b""
    while b"\r\n\r\n" not in response and len(response) < 8192:
        block = connection.recv(1024)
        if not block:
            break
        response += block
if not response.startswith(b"HTTP/1.1 101 "):
    raise SystemExit("Pi WebSocket endpoint did not return HTTP 101.")
PY
}

print_viewer_url() {
  if "$dashboard"; then
    printf 'Open the Swift dashboard in your browser: http://127.0.0.1:%s/\n' "$port"
    echo 'This connection uses the microphone and speakers of the computer running the browser.'
  else
    printf 'noVNC viewer: https://novnc.com/noVNC/vnc.html?host=127.0.0.1&port=%s&encrypt=0&autoconnect=1&path=websockify&resize=scale&reconnect=1&reconnect_delay=1000&view_clip=0\n' "$port"
    echo 'Transport verified. The viewer must also support the Pi WayVNC authentication mode.'
  fi
  printf 'Close tunnel: ssh -S %q -O exit %q\n' "$control" "$host"
}

if [[ -S "$control" ]]; then
  old_host=""
  [[ ! -f "$target_file" ]] || old_host="$(<"$target_file")"
  if [[ -n "$old_host" ]] && "${ssh_base[@]}" -O check "$old_host" >/dev/null 2>&1; then
    if [[ "$old_host" == "$host" ]]; then
      if probe_tunnel; then
        echo "Pi viewer tunnel is already healthy on 127.0.0.1:$port."
        print_viewer_url
        exit 0
      fi
      echo 'The managed SSH tunnel is alive, but its remote service check failed; leaving it untouched.' >&2
      exit 1
    fi
    echo "Replacing this script's existing viewer tunnel to $old_host with $host."
    "${ssh_base[@]}" -O exit "$old_host" >/dev/null
    for _ in $(seq 1 10); do
      [[ ! -S "$control" ]] && break
      sleep 0.2
    done
  else
    rm -f -- "$control"
  fi
fi

if [[ -e "$control" ]]; then
  echo 'A stale control path remains; refusing to remove an unexpected file.' >&2
  exit 1
fi

if python3 - "$port" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.3):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
then
  echo "127.0.0.1:$port is occupied by a tunnel this script does not own. Set SWIFT_PI_DASHBOARD_PORT or SWIFT_PI_NOVNC_PORT to a free port for the selected mode." >&2
  exit 1
fi

printf '%s\n' "$host" >"$target_file"
chmod 600 "$target_file"
echo "Opening loopback-only Pi viewer tunnel to $host."
"${ssh_base[@]}" -f -M -N -L "127.0.0.1:$port:127.0.0.1:$remote_port" "$host"
if ! probe_tunnel; then
  "${ssh_base[@]}" -O exit "$host" >/dev/null 2>&1 || true
  exit 1
fi
print_viewer_url
