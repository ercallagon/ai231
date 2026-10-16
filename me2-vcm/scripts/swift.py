#!/usr/bin/env python3
"""Public SSH workflow. Use --help for installation, setup, tests and reporting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hey_swift.deployment import main, SetupError

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped; existing deployment preserved.")
        raise SystemExit(130)
    except SetupError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        # Do not print configuration, commands or subprocess exceptions with secrets.
        print(f"Setup/runtime check failed ({type(exc).__name__}). Check prerequisites and private configuration.", file=sys.stderr)
        raise SystemExit(1)
