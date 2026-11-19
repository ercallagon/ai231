"""Explicit project-only secret files. No shell evaluation or automatic discovery."""

import os
import tempfile
import stat
from pathlib import Path

from .configuration import ConfigurationError

CACHE_ROOT = Path(__file__).resolve().parents[2] / "var/cache"
NAMES = {
    # Provider/model-qualified names are canonical for new profiles.
    "ME3_DEEPGRAM_API_KEY",
    "ME3_GROQ_API_KEY",
    "ME3_OPENROUTER_API_KEY",
    "ME3_QWEN_ASR_API_KEY",
    "ME3_QWEN_LLM_API_KEY",
    "ME3_QWEN_TTS_API_KEY",
    # Generic role names remain accepted only for existing installations.
    "ME3_ASR_API_KEY",
    "ME3_TTS_API_KEY",
    "ME3_LLM_API_KEY",
    "ME3_RESEARCH_TAVILY_API_KEY",
    "ME3_SPOTIFY_ACCESS_TOKEN",
    "ME3_SPOTIFY_REFRESH_TOKEN",
    "ME3_ONIT_GITHUB_TOKEN",
    "ME3_GITHUB_WRITE_TOKEN",
    "ME3_ONIT_TAVILY_API_KEY",
    "ME3_ONIT_HF_TOKEN",
}


def private_path(path, *, cache_root=None):
    path = Path(path).absolute()
    if path.is_symlink():
        raise ConfigurationError("secret file must not be a symlink")
    try:
        path.resolve().relative_to(Path(cache_root or CACHE_ROOT).resolve())
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


def update_secrets(path, updates, *, remove=(), cache_root=None):
    """Atomically append/update selected keys while preserving every other key."""
    path = private_path(path, cache_root=cache_root)
    remove = set(remove)
    if (
        not isinstance(updates, dict)
        or set(updates) - NAMES
        or remove - NAMES
        or not all(valid_key(value) for value in updates.values())
    ):
        raise ConfigurationError("invalid project key update; no secret file written")
    initial = None
    values = {}
    if path.exists():
        initial = path.stat(follow_symlinks=False)
        values = load_secrets(path, cache_root=cache_root)
    if set(updates) & remove:
        raise ConfigurationError("a key cannot be updated and removed in the same operation")
    values = {**values, **updates}
    for name in remove:
        values.pop(name, None)
    if not updates and not remove:
        return
    payload = "".join(f"{name}={value}\n" for name, value in values.items())
    if len(payload.encode("utf-8")) > 16384:
        raise ConfigurationError("updated credential file exceeds its size limit; no changes made")

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if initial is None:
            if path.exists() or path.is_symlink():
                raise ConfigurationError("secret file appeared during update; no changes made")
        else:
            current = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_uid != os.getuid()
                or (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino)
            ):
                raise ConfigurationError("secret file changed during update; no changes made")
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except ConfigurationError:
        raise
    except OSError:
        raise ConfigurationError("could not atomically update private credentials") from None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_secrets(path, selected=NAMES, *, cache_root=None):
    path = private_path(path, cache_root=cache_root)
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
