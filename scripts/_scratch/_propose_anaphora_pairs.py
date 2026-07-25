"""Propose anaphora back-reference pairs for STRM CE fine-tune data (Task #122).
UNTRACKED scratch. Uses the LOCAL Ollama LLM (on-box -- onyx never leaves the
box) to PROPOSE candidate (query, target) pairs; a human verifies each before
it becomes gold. The final gold label is the human's judgment (same standard as
the 17 held-out hand-authored pairs); the LLM only surfaces candidates so the
human does not have to scan every turn pair.

PILOT: run on a small + medium + large session to validate proposal quality
before scaling to all 27 mineable sessions.

Run:
  PYTHONPATH=. python scripts/_scratch/_propose_anaphora_pairs.py [--all]
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_PATH = SCRATCH / "_trained_episodes_for_labeling.json"
OUT_PATH = SCRATCH / "_trained_gold_proposals.json"

URL = "http://127.0.0.1:11434/api/chat"
MODEL = "deepseek-v4-flash:cloud"  # flash for bulk labeling-proposal (human verifies)
TURN_TRUNC = 600
WINDOW = 30
STEP = 25
MIN_AGE = 3  # age = query - target - 1 >= 3

# Pilot set: small / medium / large (by user-turn count).
PILOT = ["35c9cc55", "b11be917", "75f12969"]


def call(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                obj = json.loads(resp.read().decode("utf-8", "replace"))
                return obj.get("message", {}).get("content", "")
        except Exception as e:  # noqa
            last = str(e)
            time.sleep(2)
    return f"[[CALL_FAILED: {last}]]"


def extract_pairs(text: str) -> list[dict]:
    """Robustly pull a JSON list out of the model output."""
    s = text
    # strip code fences
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1)
    # find the first [...] block
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for row in arr:
        if not isinstance(row, dict):
            continue
        q = row.get("query", row.get("query_idx"))
        t = row.get("target", row.get("target_idx"))
        if q is None or t is None:
            continue
        try:
            q = int(q); t = int(t)
        except Exception:
            continue
        out.append({"query": q, "target": t,
                    "reason": str(row.get("reason", row.get("why", "")))[:200]})
    return out


def build_prompt(name: str, turns: list[tuple[int, str]]) -> str:
    body = "\n".join(f"u{i:02d}: {txt}" for i, txt in turns)
    return (
        "You are labeling anaphora back-references in a single user's chat "
        "session (a coherent thread with one human user). Below are the USER "
        f"turns of session {name!r}, indexed u00..uNN.\n\n"
        f"USER TURNS:\n{body}\n\n"
        "TASK: identify pairs (query, target) where the QUERY turn refers BACK "
        "to, or resumes a topic from, an earlier TARGET turn. The signal is "
        "anaphora / conceptual continuity: the query says things like 'this', "
        "'that', 'the X we discussed', 'prior', 'earlier', 'as you said', or "
        "clearly continues a specific earlier subtopic. The target must be the "
        "SPECIFIC earlier turn the query points back to, not just any "
        "topically-related turn.\n"
        f"CONSTRAINTS: age = query_index - target_index - 1 must be >= {MIN_AGE} "
        "(target is at least 4 turns earlier). Do NOT pair duplicate or "
        "near-duplicate turns (retries where the user re-sent the same text). "
        "Prefer CLEAR back-references; quality over quantity. 2-8 pairs is "
        "fine for this window.\n"
        "OUTPUT: a JSON list of {\"query\": int, \"target\": int, \"reason\": "
        "\"one line\"}. No prose outside the JSON."
    )


def main() -> int:
    ep = json.loads(EP_PATH.read_text(encoding="utf-8"))
    by_prefix = {k[:8]: k for k in ep}
    run_all = "--all" in sys.argv
    targets = [by_prefix[p] for p in PILOT if p in by_prefix]
    if run_all:
        targets = list(ep.keys())

    proposals: dict[str, list[dict]] = {}
    for sid in targets:
        s = ep[sid]
        uturns = [(i, " ".join(t["text"].split()))
                  for i, t in enumerate(t for t in s["turns"] if t["role"] == "user")]
        seen: set[tuple[int, int]] = set()
        session_pairs: list[dict] = []
        # slide windows
        n = len(uturns)
        if n == 0:
            continue
        starts = list(range(0, max(1, n - 1), STEP))
        if starts[-1] + WINDOW < n and not run_all and n <= WINDOW:
            starts = [0]
        for st in starts:
            chunk = uturns[st:st + WINDOW]
            if len(chunk) < 6:
                continue
            chunk_trunc = [(i, txt[:TURN_TRUNC]) for i, txt in chunk]
            prompt = build_prompt(s["name"], chunk_trunc)
            content = call(prompt)
            pairs = extract_pairs(content)
            for p in pairs:
                q, t = p["query"], p["target"]
                if not (0 <= t < q < n):
                    continue
                if q - t - 1 < MIN_AGE:
                    continue
                if (q, t) in seen:
                    continue
                seen.add((q, t))
                session_pairs.append({"query": q, "target": t,
                                      "age": q - t - 1, "reason": p["reason"],
                                      "window_start": st})
        proposals[sid] = session_pairs
        print(f"[propose] {sid[:8]} {s['name'][:36]!r:38} -> {len(session_pairs)} pairs",
              flush=True)

    OUT_PATH.write_text(json.dumps(proposals, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    tot = sum(len(v) for v in proposals.values())
    print(f"\n[done] {tot} proposals across {len(proposals)} sessions -> {OUT_PATH}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())