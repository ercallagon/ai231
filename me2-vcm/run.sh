#!/usr/bin/env bash
set -euo pipefail
ME2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ME2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -eq 0 ]; then set -- verify; fi
exec "${PYTHON:-python3}" "$ME2_ROOT/scripts/reproduce.py" "$@"
