"""Local demo defaults; environment variables can override each value."""
from pathlib import Path

DEFAULT_CALL_NUMBER = ""  # Public install requires private environment configuration.
DEFAULT_MESSAGE_NUMBER = ""  # Public install requires private environment configuration.
DEFAULT_PHONE_MAC = ""  # Public install requires private environment configuration.


def project_root() -> Path:
    """Find the repo/deployment root containing configs/ and src/."""
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


def runtime_root() -> Path:
    """Writable integration runtime root.

    Preferred deployment layout is <project>/integration. In this sandbox the
    prepared standalone bundle lives at <project>/work/integration. An explicit
    HEY_SWIFT_RUNTIME_ROOT overrides both.
    """
    import os
    override = os.environ.get("HEY_SWIFT_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    root = project_root()
    for candidate in (root / "integration", root / "work" / "integration"):
        if candidate.exists():
            return candidate.resolve()
    return (root / "integration").resolve()


def communication_values():
    """All integration entry points share ready-to-run recipient/device defaults."""
    import os
    return {
        "call_number": os.environ.get("VCM_CALL_NUMBER", DEFAULT_CALL_NUMBER),
        "message_number": os.environ.get("VCM_MESSAGE_NUMBER", DEFAULT_MESSAGE_NUMBER),
        "phone_mac": os.environ.get("VCM_PHONE_MAC", DEFAULT_PHONE_MAC),
    }


def spotify_paths(bundle_root):
    """Use bundled state unless standalone/auto reuse was explicitly selected.

    A fresh deployment with no standalone installation uses bundled paths.
    Explicit environment paths override the selected profile.
    """
    import os
    root = Path(bundle_root).resolve()
    project = root.parent
    # Check known installations without copying account state or credentials.
    candidates = [project, Path.home() / "vcm-modular-test"]
    selected_project = next((candidate for candidate in candidates
        if (candidate / "work/tools/soloist/soloist").is_file()
        and (candidate / "work/soloist-runtime/api-key").is_file()), project)
    standalone_binary = selected_project / "work/tools/soloist/soloist"
    standalone_data = selected_project / "work/soloist-runtime"
    profile = os.environ.get("VCM_SOLOIST_PROFILE", "bundled")
    if profile not in {"auto", "standalone", "bundled"}:
        raise ValueError("VCM_SOLOIST_PROFILE must be auto, standalone, or bundled.")
    if profile == "auto":
        profile = "standalone" if (standalone_binary.is_file()
            and (standalone_data / "api-key").is_file()) else "bundled"
    default_binary = standalone_binary if profile == "standalone" else root / "tools/soloist/soloist"
    default_data = standalone_data if profile == "standalone" else root / "state/soloist"
    binary = Path(os.environ.get("VCM_SOLOIST_BIN", str(default_binary))).expanduser().resolve()
    data = Path(os.environ.get("VCM_SOLOIST_DATA_DIR", str(default_data))).expanduser().resolve()
    key = Path(os.environ.get("VCM_SOLOIST_KEY_FILE", str(data / "api-key"))).expanduser().resolve()
    return {"profile": profile, "binary": binary, "data_dir": data, "key_file": key}
