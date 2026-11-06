"""Explicit project-only secret files. No shell evaluation or automatic discovery."""

import os
import stat
from pathlib import Path

from .configuration import ConfigurationError

CACHE_ROOT = Path(__file__).resolve().parents[2] / "var/cache"
NAMES = {
    "ME3_ASR_API_KEY",
    "ME3_TTS_API_KEY",
    "ME3_LLM_API_KEY",
    "ME3_SPOTIFY_ACCESS_TOKEN",
    "ME3_SPOTIFY_REFRESH_TOKEN",
    "ME3_ONIT_GITHUB_TOKEN",
    "ME3_ONIT_TAVILY_API_KEY",
    "ME3_ONIT_HF_TOKEN",
}


def private_path(path):
    path = Path(path).absolute()
    if path.is_symlink():
        raise ConfigurationError("secret file must not be a symlink")
    try:
        path.resolve().relative_to(CACHE_ROOT.resolve())
    except ValueError:
        raise ConfigurationError(
            "secret files must be inside this project's ignored var/cache/"
        ) from None
    return path


def valid_key(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 4096
        and all(33 <= ord(c) <= 126 for c in value)
    )


def write_secrets(path, values):
    path = private_path(path)
    if not values or set(values) - NAMES or not all(valid_key(v) for v in values.values()):
        raise ConfigurationError("invalid project key names or values; no secret file written")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            for name, value in values.items():
                stream.write(name + "=" + value + "\n")
    except OSError:
        raise ConfigurationError(
            "cannot create private secret file; existing files are never overwritten"
        ) from None


def load_secrets(path, selected=NAMES):
    path = private_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd) as stream:
            status = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.getuid()
                or status.st_mode & 0o077
                or status.st_size > 16384
            ):
                raise ConfigurationError(
                    "secret file must be owned by this user, mode 0600, and at most 16 KiB"
                )
            values = {}
            for line in stream:
                name, separator, value = line.rstrip("\n").partition("=")
                if name not in NAMES or not separator or name in values or not valid_key(value):
                    raise ConfigurationError("invalid project secret file; values excluded")
                values[name] = value
            return {name: value for name, value in values.items() if name in selected}
    except (OSError, UnicodeError):
        raise ConfigurationError(
            "project secret file cannot be read securely; check path, owner and permissions"
        ) from None
