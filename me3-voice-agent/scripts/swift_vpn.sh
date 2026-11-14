#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "$PWD/var/cache/venv/bin/python" -B -m me3_voice.vpn "$@"
