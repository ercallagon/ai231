#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run or provision Spotify Soloist on a Raspberry Pi/Linux host, including x86_64 WSL.

Usage: bash scripts/run_soloist_receiver.sh [--update] [--help]

First run downloads Spotify's official build for this CPU, installs it in
~/.local/bin, securely prompts for the Soloist API key, and saves it under
~/.config/swift-me3. It then starts the receiver in the foreground. Ctrl+C
stops it. --update downloads and replaces the installed binary first.
EOF
}

force_update=false
case "${1:-}" in
  "") ;;
  --update) force_update=true ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$EUID" == 0 ]]; then
  echo 'Run Soloist as the normal Pi/demo user, not root.' >&2
  exit 2
fi
for tool in curl tar find install; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required command: $tool" >&2
    exit 2
  fi
done

case "$(uname -m)" in
  aarch64|arm64) archive_name=soloist_release_arm64.tar.gz ;;
  armv7l|armv7) archive_name=soloist_release_arm32.tar.gz ;;
  x86_64|amd64) archive_name=soloist_release_x86_64.tar.gz ;;
  *) echo "Unsupported Soloist Linux architecture: $(uname -m)" >&2; exit 2 ;;
esac

home="${HOME:?HOME must be set}"
bin_dir="$home/.local/bin"
soloist_bin="$bin_dir/soloist"
config_dir="${XDG_CONFIG_HOME:-$home/.config}/swift-me3"
secret_file="$config_dir/soloist-api-key"
data_dir="${SOLOIST_DATA_DIR:-$home/.local/share/soloist}"
cache_dir="${SOLOIST_CACHE_DIR:-$home/.cache/soloist}"
device_name="${SOLOIST_DEVICE_NAME:-Swift Soloist}"
umask 077
mkdir -p "$bin_dir" "$config_dir" "$data_dir" "$cache_dir"
chmod 700 "$config_dir" "$data_dir" "$cache_dir"

if [[ "$force_update" == true || ! -x "$soloist_bin" ]]; then
  command -v mktemp >/dev/null 2>&1 || { echo 'Missing required command: mktemp' >&2; exit 2; }
  temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/swift-soloist.XXXXXXXX")"
  trap 'rm -rf -- "$temp_dir"' EXIT
  archive="$temp_dir/$archive_name"
  url="https://soloist-builds.spotifycdn.com/$archive_name"
  echo "Downloading Spotify's official Soloist build for $(uname -m)."
  curl --fail --location --retry 2 --proto '=https' --tlsv1.2 "$url" --output "$archive"
  tar --extract --gzip --file "$archive" --directory "$temp_dir" --no-same-owner --no-same-permissions
  candidate="$(find "$temp_dir" -type f -name soloist -print -quit)"
  if [[ -z "$candidate" ]]; then
    echo 'The official archive did not contain the Soloist executable.' >&2
    exit 1
  fi
  install -m 0755 -- "$candidate" "$soloist_bin"
  echo "Installed Soloist at $soloist_bin."
fi

if [[ -L "$secret_file" || ( -e "$secret_file" && ! -f "$secret_file" ) ]]; then
  echo "Refusing unsafe Soloist key file: $secret_file" >&2
  exit 2
fi
if [[ -n "${SOLOIST_API_KEY:-}" ]]; then
  api_key="$SOLOIST_API_KEY"
elif [[ -f "$secret_file" ]]; then
  chmod 600 "$secret_file"
  IFS= read -r api_key < "$secret_file" || true
else
  if [[ ! -t 0 ]]; then
    echo 'No Soloist API key found. Run interactively once or set SOLOIST_API_KEY.' >&2
    exit 2
  fi
  read -r -s -p 'Enter the Soloist API key from Spotify for Developers: ' api_key
  printf '\n'
  if [[ -z "$api_key" ]]; then
    echo 'The Soloist API key cannot be empty.' >&2
    exit 2
  fi
  printf '%s\n' "$api_key" > "$secret_file"
  chmod 600 "$secret_file"
  echo "Saved the key privately at $secret_file."
fi
if [[ -z "$api_key" ]]; then
  echo 'The Soloist API key is empty.' >&2
  exit 2
fi

if [[ -n "${temp_dir:-}" ]]; then
  rm -rf -- "$temp_dir"
  trap - EXIT
fi

"$soloist_bin" --version
echo "Starting '$device_name'. Keep this terminal open; press Ctrl+C to stop Soloist."
echo 'In Spotify, select this Connect device once it appears on the same local network.'
exec "$soloist_bin" \
  --verbose \
  --device-name "$device_name" \
  --api-key "$api_key" \
  --data-dir "$data_dir" \
  --cache-dir "$cache_dir"
