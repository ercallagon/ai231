#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download and run the official Windows Spotify Soloist build from WSL.

Usage: bash scripts/run_soloist_windows.sh [--update] [--help]

The launcher stores the Windows build under the Windows user's LocalAppData,
reuses the WSL-private Soloist API key, and runs Soloist in the foreground.
Stop an already-running WSL Soloist with Ctrl+C before starting this receiver.
EOF
}

force_update=false
case "${1:-}" in
  "") ;;
  --update) force_update=true ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

for tool in cmd.exe wslpath curl find; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "This launcher requires WSL interop and '$tool'." >&2
    exit 2
  fi
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
python_bin="$project_root/var/cache/venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "$python_bin" ]]; then
  echo 'This launcher requires the project Python environment or python3.' >&2
  exit 2
fi

windows_profile="$(cmd.exe /d /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')"
if [[ -z "$windows_profile" || "$windows_profile" == *'%'* ]]; then
  echo 'Could not resolve the Windows user profile through WSL interop.' >&2
  exit 2
fi
profile_path="$(wslpath -u "$windows_profile")"
if [[ ! -d "$profile_path" ]]; then
  echo 'The resolved Windows user profile is not mounted in WSL.' >&2
  exit 2
fi

install_dir="$profile_path/AppData/Local/SwiftSoloist"
archive="$install_dir/soloist_release_windows_x86_64.zip"
mkdir -p "$install_dir"
exe="$(find "$install_dir" -type f -iname soloist.exe -print -quit)"
if [[ "$force_update" == true || -z "$exe" ]]; then
  echo 'Downloading Spotify Soloist for Windows x86_64 from Spotify.'
  curl --fail --location --retry 2 --proto '=https' --tlsv1.2 \
    'https://soloist-builds.spotifycdn.com/soloist_release_windows_x86_64.zip' \
    --output "$archive"
  "$python_bin" -m zipfile -e "$archive" "$install_dir"
  exe="$(find "$install_dir" -type f -iname soloist.exe -print -quit)"
fi
if [[ -z "$exe" ]]; then
  echo 'The official Windows archive did not contain soloist.exe.' >&2
  exit 1
fi

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/swift-me3"
secret_file="$config_dir/soloist-api-key"
if [[ -L "$secret_file" || ! -f "$secret_file" ]]; then
  echo "Soloist key file not found safely at $secret_file. Run scripts/run_soloist_receiver.sh once first." >&2
  exit 2
fi
chmod 600 "$secret_file"
IFS= read -r api_key < "$secret_file" || true
if [[ -z "${api_key:-}" ]]; then
  echo 'The saved Soloist API key is empty.' >&2
  exit 2
fi

cd "$install_dir"
"$exe" --verbose --device-name "${SOLOIST_DEVICE_NAME:-Swift Soloist}" \
  --api-key "$api_key"
