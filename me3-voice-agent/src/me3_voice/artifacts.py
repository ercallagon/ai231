"""Shared output boundary for generated project artifacts."""

import argparse
import re
from pathlib import Path


_PUBLIC_TASK_ARTIFACT_FILE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)$"
)
_TASK_WORKSPACE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,119}$")


def public_task_artifact_name(value: str) -> bool:
    """Validate a registered relative workspace path for task receipts/downloads."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
    ):
        return False
    parts = value.split("/")
    return (
        len(parts) <= 12
        and all(_TASK_WORKSPACE_SEGMENT.fullmatch(part) for part in parts)
        and _PUBLIC_TASK_ARTIFACT_FILE.fullmatch(parts[-1]) is not None
    )


def output_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(Path("var").resolve()):
        raise argparse.ArgumentTypeError("generated output must be under var/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def approved_project_root(value) -> Path:
    """Explicit existing output root: legacy sandbox parent or this project's var/."""
    candidate = Path(value)
    project = Path.cwd().resolve(strict=True)
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_dir()
        or candidate.resolve(strict=True) != candidate
        or not (candidate == project.parent or candidate.is_relative_to(project / "var"))
    ):
        raise ValueError(
            "project output root must be an existing canonical directory in project var/ or the sandbox parent"
        )
    return candidate
