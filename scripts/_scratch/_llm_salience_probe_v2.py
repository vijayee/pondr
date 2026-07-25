"""LLM-as-salience probe v2 -- PROMPT-TUNED (anti-recency + few-shot).

v1 (scripts/_scratch/_llm_salience_probe.py) found the signal: zero-shot
qwen3:8b window-RETRIEVAL lifts LOO-8 normal to top1=0.532 / in_top3=0.726
vs CE 0.294/0.588 -- the retrieval frame beats the forecast-head/CE apparatus
with NO training. v1 also found (via qwen3-coder:30b) that raw CAPACITY is not
the lever: the 30b coder REGRESSED (1/8 on 6b152fb5 where 8b got 5/8) due to a
strong RECENCY BIAS -- over-picking the most recent turns regardless of anaphora.
So the lever is the PROMPT, not model size. This v2 tests an anti-recency,
few-shot prompt on the same LOO-8 set to see if it lifts 8b past the 2/3 top1 bar.

PROMPT CHANGES vs v1:
  1. Explicit anti-recency: "The referent is often an EARLIER turn, not the most
     recent one. Do NOT default to recency -- recent turns are often only the
     current subtopic, while the anaphora target is the earlier turn that
     introduced the entity/topic the new turn's pronoun or implicit subject
     points back to."
  2. One worked few-shot example (synthetic, NOT from the gold sessions) showing
     a pronoun resolving to a MID-window turn, not the most recent.
  3. Restate "what does THIS turn refer BACK to" to anchor the backward-looking
     anaphora task.

Same gate, same gold, same window math as v1. MODEL env (default qwen3:8b),
WHICH=loo|heldout|both. UNTRACKED scratch. onyx PRIVATE -- nothing leaves the
box. No uploads.

Run: PYTHONPATH=. MODEL=qwen3:8b WHICH=loo python scripts/_scratch/_llm_salience_probe_v2.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_TRAINED = SCRATCH / "_trained_episodes_for_labeling.json"
GOLD_TRAINED = SCRATCH / "_trained_gold.json"
EP_HELDOUT = SCRATCH / "_heldout_episodes.json"
GOLD_HELDOUT = SCRATCH / "_heldout_gold.json"
OUT_RESULT = SCRATCH / "_llm_salience_result_v2.json"

URL = "http://127.0.0.1:11434/api/chat"
MODEL = os.getenv("MODEL", "qwen3:8b")
WHICH = os.getenv("WHICH", "loo")
BUDGET = 3
LOO_MIN_PAIRS = 6
MAX_TURN_CHARS = 600

# A synthetic few-shot example (NOT from any onyx session) demonstrating a
# pronoun resolving to a MID-window turn, not the most recent. Used to steer the
# model away from recency bias.
FEWSHOT = (
    "EXAMPLE (not from the real conversation):\n"
    "Prior turns:\n"
    "A: I'm thinking of adopting a border collie from the shelter.\n"
    "B: Collies need a lot of exercise, are you ready for that?\n"
    "C: The shelter said she's already house-trained.\n"
    "D: Exercise-wise I run every morning, so that's fine.\n"
    "E: How much did they ask for the adoption fee?\n\n"
    "New user turn: Is she good with kids though?\n"
    "Answer: A, C, D  (\"she\" refers back to A's collie / C's house-trained dog, "
    "NOT the most recent turn E about the fee.)\n\n"
)

SYS = ("You are a precise anaphora / coreference resolver for multi-turn "
       "conversation. The new turn often uses a pronoun (it, she, that, this, "
       "the one) or an implicit subject that points BACK to one specific earlier "
       "turn. Your job is to identify which earlier turns the new turn refers "
       "back to. Output only what is asked.")


def _user_turns(sess):
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def llm_rank(query_text, window_turns):
    """Anti-recency, few-shot prompt. Returns (idxs up to 3 most-relevant-first,
    raw content) or ([], "") on failure."""
    letters = [chr(ord("A") + i) for i in range(len(window_turns))]
    lines = []
    for L, txt in zip(letters, window_turns):
        t = (txt or "").strip().replace("\n", " ")
        if len(t) > MAX_TURN_CHARS:
            t = t[:MAX_TURN_CHARS] + "…"
        lines.append(f"{L}: {t}")
    window_block = "\n".join(lines)
    user = (
        f"{FEWSHOT}"
        f"Now the REAL conversation. Here are the last {len(window_turns)} user "
        f"turns, oldest first, each labeled with a letter:\n\n{window_block}\n\n"
        f"New user turn (the query):\n{query_text.strip()}\n\n"
        f"Task: which 3 prior turns does this new turn most refer BACK to? "
        f"The referent is often an EARLIER turn, not the most recent one -- do "
        f"NOT default to recency. Recent turns are often only the current "
        f"subtopic; the anaphora target is the earlier turn that introduced the "
        f"entity or topic that the new turn's pronoun or implicit subject points "
        f"back to. Pick the 3 turns the new turn most refers back to, "
        f"most-relevant first. Reply with EXACTLY 3 letters, comma-separated "
        f"(e.g. 'G, C, K'). Reply with only the letters, nothing else."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": user}],
        "stream": False, "options": {"temperature": 0.0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                obj = json.loads(resp.read().decode("utf-8", errors="replace"))
                content = obj.get("message", {}).get("content", "")
                hits = re.findall(r"\b([A-P])\b", content.upper())
                seen = set(); picked = []
                for h in hits:
                    if h in seen:
                        continue
                    seen.add(h); picked.append(h)
                    if len(picked) >= 3:
                        break
                idxs = []
                for h in picked:
                    i = ord(h) - ord("A")
                    if 0 <= i < len(window_turns):
                        idxs.append(i)
                return idxs, content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            time.sleep(2)
    return [], ""


def eval_set(episodes, gold_pairs, window, age_threshold, label):
    by_sid = {}
    for p in gold_pairs:
        by_sid.setdefault(p["session_id"], []).append(p)
    n_top1 = n_top3 = n_beats = 0
    n_in = 0
    breadths = []
    per_pair = []
    for sid, pairs in by_sid.items():
        sess = episodes.get(sid)
        if sess is None:
            continue
        ut = _user_turns(sess)
        for p in pairs:
            q = int(p["query"]); t = int(p["target"])
            if q >= len(ut) or t >= len(ut):
                continue
            age = q - t - 1
            lo = max(0, q - window)
            win_idx = list(range(lo, q))
            in_window = (age >= age_threshold and age <= window - 1 and t >= lo)
            if not in_window:
                continue
            window_turns = [ut[j] for j in win_idx]
            target_in_win = t - lo
            picked, raw = llm_rank(ut[q], window_turns)
            rank = None
            for pos, wpos in enumerate(picked, 1):
                if wpos == target_in_win:
                    rank = pos
                    break
            top1 = rank == 1
            in3 = rank is not None and rank <= BUDGET
            beats = rank is None or rank > 1
            breadth = min(BUDGET, len(picked))
            n_in += 1
            if top1: n_top1 += 1
            if in3: n_top3 += 1
            if beats: n_beats += 1
            breadths.append(float(breadth))
            per_pair.append({"sid": sid[:8], "q": q, "t": t, "age": age,
                             "picked": picked, "rank": rank, "top1": top1,
                             "in_top3": in3, "beats": beats, "breadth": breadth,
                             "raw": raw[:120]})
            sys.stdout.buffer.write(
                f"  [{label}] {sid[:8]} q{q:02d}->t{t:02d} picked={picked} "
                f"rank={rank} top1={top1}\n".encode())
            sys.stdout.buffer.flush()
    if not n_in:
        return None
    top1 = n_top1 / n_in; top3 = n_top3 / n_in; beats = n_beats / n_in
    med = _median(breadths)
    # gate degeneracy for LLM: beats == 1 - top1, breadth const 3 (see v1 memory)
    ship = (n_in >= 6 and top1 >= 2/3 and top3 >= 2/3 and beats <= 1/3 and med <= BUDGET)
    res = {"label": label, "n_in_window": n_in,
           "target_top1_rate": round(top1, 4),
           "target_in_top3_rate": round(top3, 4),
           "competitor_beats_rate": round(beats, 4),
           "median_breadth": round(med, 4), "ship": ship,
           "per_pair": per_pair}
    print(f"\n=== {label} LLM-SALIENCE v2 GATE (n={n_in}) model={MODEL} ===", flush=True)
    print(f"  target_top1   = {res['target_top1_rate']}  (need 0.667; v1 8b was 0.532)", flush=True)
    print(f"  target_in_top3= {res['target_in_top3_rate']}  (need 0.667; v1 8b was 0.726)", flush=True)
    print(f"  competitor_beats={res['competitor_beats_rate']}  (= 1 - top1 for LLM)", flush=True)
    print(f"  VERDICT: {'SHIP' if ship else 'HOLD'}", flush=True)
    return res


def main() -> int:
    out = {"model": MODEL, "which": WHICH, "prompt": "v2 anti-recency + few-shot"}
    if WHICH in ("heldout", "both"):
        ep = json.loads(EP_HELDOUT.read_text(encoding="utf-8"))
        g = json.loads(GOLD_HELDOUT.read_text(encoding="utf-8"))
        out["heldout"] = eval_set(ep, g["pairs"], int(g.get("window", 16)),
                                  int(g.get("age_threshold", 3)), "heldout-17")
    if WHICH in ("loo", "both"):
        ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
        g = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
        w = int(g.get("window", 16)); a = int(g.get("age_threshold", 3))
        by_sid = {}
        for p in g["pairs"]:
            by_sid.setdefault(p["session_id"], []).append(p)
        loo = []
        for sid, pairs in sorted(by_sid.items()):
            if len(pairs) >= LOO_MIN_PAIRS:
                loo.extend(pairs)
        out["loo"] = eval_set(ep, loo, w, a, "loo-normal")
    OUT_RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nwrote {OUT_RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())