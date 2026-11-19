#!/usr/bin/env bash
set -euo pipefail

# Spotify Web API OAuth client used by Swift to control a Soloist Connect
# receiver. This is a public client ID, not the separate Soloist daemon API key.
SPOTIFY_CLIENT_ID="aabc867a344a4600abfb25d4ebd419fa"
SPOTIFY_DEVICE_NAME="${SPOTIFY_DEVICE_NAME:-Swift Soloist}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Run this script to authorize ME3 to control a Spotify Connect device named '$SPOTIFY_DEVICE_NAME'."
  echo "It opens the existing PKCE flow; approve in Spotify and paste the callback URL into the hidden prompt."
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--help]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
cd -- "$project_root"

if [[ ! -f pyproject.toml || ! -f src/me3_voice/music.py ]]; then
  echo "Run this from the ME3 checkout; project files were not found." >&2
  exit 2
fi

python="$project_root/var/cache/venv/bin/python"
config="$project_root/var/cache/music.json"
spotify_secrets="$project_root/var/cache/spotify.env"
if [[ ! -x "$python" ]]; then
  echo "ME3 runtime is missing. Set it up with: bash scripts/setup.sh" >&2
  exit 2
fi

# The existing CLI refuses overwrite by default. Renew only when both existing
# Spotify files are regular, non-symlink files; preserve partial/unsafe state.
renew_args=()
if [[ -e "$config" || -e "$spotify_secrets" ]]; then
  if [[ -f "$config" && ! -L "$config" && -f "$spotify_secrets" && ! -L "$spotify_secrets" ]]; then
    renew_args=(--renew)
  else
    echo "Spotify setup is partial or unsafe; existing files were left untouched:" >&2
    echo "  $config" >&2
    echo "  $spotify_secrets" >&2
    exit 2
  fi
fi

export ME3_MUSIC_CONFIG=var/cache/music.json
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting Spotify PKCE setup for Connect receiver: $SPOTIFY_DEVICE_NAME"
echo "Register this redirect URI in the Spotify app: http://127.0.0.1:8876/callback"
exec "$python" -B -m me3_voice.music authorize \
  --client-id "$SPOTIFY_CLIENT_ID" \
  --device-name "$SPOTIFY_DEVICE_NAME" \
  "${renew_args[@]}"
