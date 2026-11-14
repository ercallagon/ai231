#!/usr/bin/env bash
# Project-local locked install; optional masked VPN setup. No services/devices started.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
  --help) echo 'Usage: bash scripts/setup.sh [--pi] [--dev] [--vpn-profile PRIVATE_OVPN_PATH]'; exit 0 ;;
esac
extras=(--extra kws)
vpn_profile=
while (($#)); do
  case "$1" in
    --pi) extras+=(--extra pi); shift ;;
    --dev) extras+=(--extra dev); shift ;;
    --vpn-profile)
      [[ $# -ge 2 && -n "$2" ]] || { echo '--vpn-profile requires a private profile path' >&2; exit 2; }
      vpn_profile="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
python_bin="${ME3_PYTHON:-python3}"
"$python_bin" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Use Python 3.11–3.13"'
mkdir -p var/cache var/logs var/results var/scratch var/recordings
if [[ ! -x var/cache/venv/bin/python ]]; then
  "$python_bin" -m venv var/cache/venv
fi
var/cache/venv/bin/python -m pip install --cache-dir var/cache/pip 'uv==0.13.4'
UV_PROJECT_ENVIRONMENT=var/cache/venv UV_PYTHON_DOWNLOADS=never \
  var/cache/venv/bin/uv sync --locked "${extras[@]}"
if [[ -n "$vpn_profile" ]]; then
  var/cache/venv/bin/python -B scripts/configure_vpn.py --profile "$vpn_profile"
fi
echo 'Installed editable application. Next: bash scripts/swift.sh check'
