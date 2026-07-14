"""Pytest configuration.

Ensures the repo root is on ``sys.path`` so ``from src.memory.store import ...``
works whether or not the package is installed (e.g. on a fresh RunPod pod that
only ran ``pip install -r requirements``-style deps).
"""

import sys
from pathlib import Path

# Windows console defaults to the cp1252 charmap, which crashes on non-ASCII
# that third-party libs print verbatim (e.g. gliner2 emits a brain-emoji banner
# on model load -> UnicodeEncodeError under `pytest -s`). Force UTF-8 on the
# real std streams so local Windows runs don't need PYTHONUTF8=1 set by hand.
# (Under captured mode pytest already uses UTF-8 buffers; this covers `-s`.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))