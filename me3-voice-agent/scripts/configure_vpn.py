"""One-time masked setup for OpenVPN credentials; no VPN connection is started.

The auth file is plaintext protected by mode 0600, not encrypted at rest.
"""

import argparse
import getpass
import json
import os
import sys
import warnings
from pathlib import Path


def configure(profile: Path, config: Path):
    profile = profile.expanduser().absolute()
    if (
        profile.is_symlink()
        or not profile.is_file()
        or profile.stat().st_uid != os.getuid()
        or profile.stat().st_mode & 0o077
    ):
        raise ValueError("VPN profile must be a regular file with mode 0600")
    if b"client" not in [line.strip() for line in profile.read_bytes().splitlines()]:
        raise ValueError("Expected an OpenVPN client profile")
    directory = Path.home() / ".config/openvpn"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (
        directory.is_symlink()
        or directory.stat().st_uid != os.getuid()
        or directory.stat().st_mode & 0o077
    ):
        raise ValueError("~/.config/openvpn must be a real mode-0700 directory")
    auth = directory / "me3-auth.txt"
    config = Path(config).expanduser()
    if config.is_symlink():
        raise ValueError("VPN launcher config must not be a symlink")
    config = config.resolve()
    if not config.is_relative_to(Path("var/cache").resolve()):
        raise ValueError("VPN launcher config must be in project var/cache/")
    if auth.exists() or config.exists():
        raise FileExistsError("VPN auth/config exists; it was not overwritten")
    if not sys.stdin.isatty():
        raise ValueError("Use an interactive terminal for masked VPN credential entry")
    username = getpass.getpass("VPN username (hidden): ")
    password = getpass.getpass("VPN password (hidden): ")
    if any(not v or len(v) > 1024 or any(c in v for c in "\r\n\x00") for v in (username, password)):
        raise ValueError("Invalid VPN credential input; nothing saved")
    auth_tmp, config_tmp = auth.with_suffix(".tmp"), config.with_suffix(".tmp")
    try:
        fd = os.open(auth_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(username + "\n" + password + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(auth_tmp, auth)
        auth_tmp.unlink()
        config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(config_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"profile": str(profile), "auth_file": str(auth)}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(config_tmp, config)
        config_tmp.unlink()
    except BaseException:
        auth_tmp.unlink(missing_ok=True)
        config_tmp.unlink(missing_ok=True)
        raise
    print(f"Saved mode-0600 VPN auth and launcher settings to {config}.")
    print(
        "Auth is plaintext protected by Unix permissions, not encrypted at rest; no connection was started."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("var/cache/vpn-launch.json"))
    args = parser.parse_args()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            configure(args.profile, args.config)
    except (OSError, ValueError, getpass.GetPassWarning) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
