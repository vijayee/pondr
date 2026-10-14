"""OCR re-encoder seam -- DeepSeek-OCR as the gist stage of icm-forget/1
(steal R10, unblocked: flash-attn is guarded in the model code, so the eager
path runs on plain transformers + torch cu130 in the DEDICATED .venv-icm-ocr
environment; probe: ~94% word/number precision at Small, ~84/81% at Tiny,
~6s per page on the RTX 5080, ~100s one-time model load).

ARCHITECTURE (the two hard contracts):
  * WRITE-PATH ONLY. Re-encoding happens in maintenance passes; the recall
    hot path NEVER touches this module (plan locality guarantee).
  * PROCESS SEPARATION: the main env never imports torch/transformers. This
    module talks to a DAEMON subprocess (scripts/icm/_ocr_worker.py) running
    the OCR venv's interpreter; stdin/stdout line-JSON; one daemon amortizes
    the model load across a whole maintenance pass. The daemon process is
    cleaned up on close; a dead daemon degrades to the heuristic gist
    (never crashes a maintenance pass).
  * DEFAULT OFF: stores keep the heuristic gist unless configured with an
    OCR backend -- house style; flag-gated.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading

_REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
OCR_ENV_PY = os.path.abspath(os.path.join(_REPO, ".venv-icm-ocr/Scripts/python.exe"))
WORKER_SCRIPT = os.path.abspath(os.path.join(_REPO, "scripts/icm/_ocr_worker.py"))


def heuristic_gist(text: str, max_chars: int = 220) -> str:
    """Self-contained copy of the deterministic first-drop gist (the OCR
    venv may have no wavedb, so icm.episodic is not importable here)."""
    marker = "[gist] "
    flat = " ".join(text.split())
    if len(marker) + len(flat) <= max_chars:
        return marker + flat
    cut = max_chars - len(marker)
    head = flat[:cut]
    if "." in head:
        cut = head.rindex(".") + 1
        head = flat[:cut]
    ellipsis = "" if head.rstrip().endswith((".", "!", "?")) else "..."
    return marker + head.rstrip(" ,;") + ellipsis


class OcrGistBackend:
    """Spawn-on-demand re-encode daemon wrapper."""

    def __init__(self, python: str = OCR_ENV_PY, worker: str = WORKER_SCRIPT,
                 load_timeout_s: float = 600.0, request_timeout_s: float = 180.0):
        self.python = python
        self.worker = worker
        self.load_timeout_s = load_timeout_s
        self.request_timeout_s = request_timeout_s
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._counter = 0
        self._failed = False  # sticky: dead daemon -> heuristic for the rest of the pass

    def _spawn(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        if self._failed:
            return False
        try:
            self._proc = subprocess.Popen(
                [self.python, self.worker],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1)
            import time

            t0 = time.time()
            load_line = None
            while time.time() - t0 < self.load_timeout_s:
                raw = self._proc.stdout.readline()
                if not raw:
                    raise RuntimeError("daemon stdout closed during load")
                if raw.startswith("WAVEDB_OCR_LOAD:"):
                    load_line = raw
                    break
            if load_line is None:
                raise RuntimeError("no load marker within timeout")
            json.loads(load_line[len("WAVEDB_OCR_LOAD:"):])
            return True
        except Exception:
            self._failed = True
            self.kill()
            return False

    def gist(self, text: str, mode: str = "small") -> str:
        with self._lock:
            if not self._spawn() or self._proc is None:
                return heuristic_gist(text)
            self._counter += 1
            rid = str(self._counter)
            try:
                self._proc.stdin.write(json.dumps({"text": text, "mode": mode, "id": rid}) + "\n")
                self._proc.stdin.flush()
                import time

                t0 = time.time()
                while time.time() - t0 < self.request_timeout_s:
                    line = self._proc.stdout.readline()
                    if not line:
                        raise RuntimeError("daemon closed stdout")
                    if not line.startswith("WAVEDB_OCR_RESULT:"):
                        continue  # stdout chatter from the model's own prints
                    resp = json.loads(line[len("WAVEDB_OCR_RESULT:"):])
                    if resp.get("id") == rid:
                        if "gist" in resp:
                            return resp["gist"]
                        raise RuntimeError(resp.get("error", "no gist"))
                raise TimeoutError("daemon response timed out")
            except Exception:  # noqa: BLE001
                import time

                time.sleep(0.5)
                return heuristic_gist(text)

    def kill(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.kill()