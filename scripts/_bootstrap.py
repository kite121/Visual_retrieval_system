"""Make ``src/`` importable when scripts are run as ``python scripts/<name>.py``."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = str(REPO_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
