#!/usr/bin/env bash
# Project-local, locked install. No system services, credentials or devices touched.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
  --help) echo 'Usage: bash scripts/setup.sh [--pi] [--dev]'; exit 0 ;;
esac
extras=(--extra kws)
for option in "$@"; do
  case "$option" in
    --pi) extras+=(--extra pi) ;;
    --dev) extras+=(--extra dev) ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
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
echo 'Installed editable application. Next: bash scripts/swift.sh check'
