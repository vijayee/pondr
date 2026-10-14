"""Memory consult -- the tool-call surface of the episodic store (Phase 1).

This is what makes the store consultable by an off-the-shelf agent
(plan section 5 deliverable): an agent calls

    consult(memory, "what failed in the deploy run?", k=3)

and gets back a compact payload (the recalled episode texts + scores) it can
read from its context. NOTHING here is project-specific; the payload is
plain text + structured scores. Latency is measured by the caller (the E0
harness wraps this; per-plan E0 the memory may never exceed a per-turn
budget, and the budget includes THIS call's work, not the agent's own
generation).
"""

from __future__ import annotations

import time

CONSULT_VERSION = "icm-consult/1"


def consult(memory, question: str, k: int = 3) -> dict:
    """Recall episodes for a question; returns the consult payload.

    Returns:
        {"question", "episodes": [{"id", "text", "similarity"}...],
         "latency_ms", "version", "count"}
    """
    t0 = time.perf_counter()
    hits = memory.recall(question, k=k)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "question": question,
        "episodes": [{"id": rec["episode_id"], "text": rec.get("text", ""),
                      "fidelity": rec.get("fidelity", "full"),
                      "similarity": round(sim, 4)} for rec, sim in hits],
        "count": len(hits),
        "latency_ms": round(latency_ms, 2),
        "version": CONSULT_VERSION,
    }


def consult_block(payload: dict, max_chars: int = 1800) -> str:
    """Render a consult payload as plain text for an agent's context window;
    deterministically bounded (no prompt explosion)."""
    lines = [f"[memory consult: {payload['question']}"]
    used = len(lines[0])
    for ep in payload["episodes"]:
        line = f"[{ep['similarity']:.3}] [{ep['fidelity']}] {ep['text']}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)