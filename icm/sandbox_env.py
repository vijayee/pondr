"""Sandbox environment with REAL, observed tool outcomes for Phase 0.

Everything here is real execution against real files under a scratch
directory (no shell, no network except the local model relay):
    probe_remote()   -- flaky: seeded ~35% failure rate, one retry by the
                        harness -> sometimes a DEGRADED success, sometimes
                        a hard failure
    write_manifest() -- real file write
    verify_manifest()-- real file read/verify

Event -> branch classification is DETERMINISTIC and PREDICTION-INDEPENDENT:
    clean      -> "desired"   (everything passed first try)
    degraded   -> "default"   (a step failed, retry recovered it)
    hardfail   -> "feared"    (retry also failed or the manifest verify broke)
    other      -> "other"     (file system surprise outside the flow)

If the agent did not bet "feared" and reality produced hardfail, the
congruence rule correctly stamps VIOLATED -- that is the semantics of a
missed prediction, not an observer judgment call.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil

BRANCH_FOR_EVENT = {
    "clean": "desired",
    "degraded": "default",
    "hardfail": "feared",
    "other": "other",
}

# datasets the agent may be asked to ingest; contents are fixed per seed
FIXTURES = {
    "fixture-a": {"files": 4, "rows": 40},
    "fixture-b": {"files": 6, "rows": 120},
    "fixture-c": {"files": 3, "rows": 25},
}


class SandboxEnv:
    """One sandbox per pilot run; sessions are subdirectories."""

    def __init__(self, root: str, seed: int = 7, probe_fail_p: float = 0.35):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._rng = random.Random(seed)
        self.probe_fail_p = probe_fail_p
        self._materialized: dict[str, str] = {}

    def wipe(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def materialize_fixture(self, name: str) -> str:
        """Real dataset files on disk; returns the session dir."""
        if name not in FIXTURES:
            raise KeyError(f"unknown fixture {name!r}")
        spec = FIXTURES[name]
        session_dir = os.path.join(self.root, name)
        shutil.rmtree(session_dir, ignore_errors=True)
        os.makedirs(session_dir, exist_ok=True)
        for i in range(spec["files"]):
            body = "\n".join(f"row-{name}-{i}-{r},value={rng_or(self._rng)}" for r in range(spec["rows"]))
            with open(os.path.join(session_dir, f"part-{i:02d}.csv"), "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
        digest = self.checksum(session_dir)
        self._materialized[name] = digest
        return session_dir

    def checksum(self, directory: str) -> str:
        h = hashlib.sha256()
        for fn in sorted(os.listdir(directory)):
            with open(os.path.join(directory, fn), "rb") as f:
                h.update(f.read())
            h.update(fn.encode("utf-8"))
        return h.hexdigest()[:16]

    def probe_remote(self) -> dict:
        """Flaky dependency check: REAL randomness the agent cannot see."""
        if self._rng.random() < self.probe_fail_p:
            return {"endpoint": "dep-a", "status": "error", "detail": "connection reset"}
        return {"endpoint": "dep-a", "status": "ok", "latency_ms": self._rng.randint(4, 60)}

    def write_manifest(self, session_dir: str, source_name: str) -> dict:
        digest = self._materialized.get(source_name)
        if digest is None:
            return {"status": "error", "detail": "fixture not materialized"}
        path = os.path.join(session_dir, "manifest.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"source": source_name, "sha": digest}, f)
        return {"status": "ok", "manifest": path, "sha": digest}

    def verify_manifest(self, session_dir: str, source_name: str) -> dict:
        path = os.path.join(session_dir, "manifest.json")
        if not os.path.exists(path):
            return {"status": "error", "detail": "manifest missing"}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("sha") != self._materialized.get(source_name):
            return {"status": "error", "detail": "checksum mismatch"}
        return {"status": "ok", "sha": data["sha"]}


def rng_or(rng: random.Random) -> int:
    return rng.randint(0, 999)


def classify_run(probe_events: list[dict], manifest_events: list[dict]) -> str:
    """Deterministic, prediction-independent event -> event-class -> branch."""
    if any(e.get("status") == "error" and e.get("fatal") for e in probe_events) \
            or any(e.get("status") == "error" for e in manifest_events):
        return "hardfail"
    if any(e.get("status") == "error" for e in probe_events):
        return "degraded"
    return "clean"