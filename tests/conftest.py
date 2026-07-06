"""Pytest configuration.

Ensures the repo root is on ``sys.path`` so ``from src.memory.store import ...``
works whether or not the package is installed (e.g. on a fresh RunPod pod that
only ran ``pip install -r requirements``-style deps).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))