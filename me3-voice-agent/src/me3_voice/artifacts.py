"""Shared output boundary for generated project artifacts."""

import argparse
from pathlib import Path


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
