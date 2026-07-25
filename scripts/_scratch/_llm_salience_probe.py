"""LLM-as-salience probe (consult #8 Fork A): can a window-aware LLM pick the
anaphoric referent where pairwise CE could not? This is the discriminating
experiment for the train/serve OBJECTIVE-MISMATCH hypothesis: the STRM heads
were trained to FORECAST (later_needed / relevance / surprise -- "will this be
needed in the future") but the serve gate tests RETRIEVAL ("which PAST turn does
THIS query refer back to"). A zero-shot LLM doing direct retrieval (no forecast
training) either clears the gate (-> the forecast-head architecture was the
wrong frame; retrieval is the right frame) or does not (-> the signal is not in
the window; deeper than architecture).

MODEL CHOICE. qwen3:8b (local, $0) is the dense sibling of the ternary Bonsai 8B
base. Phase 3c found dense qwen3:8b ~= ternary Bonsai 8B on adjudication (both
rubber-stamp hard calls), so qwen3:8b is the cheapest direct proxy for the
serve-time Bonsai 8B -- same family, same 8B capacity, free, live, no pod/tunnel.
Set MODEL env to compare ceilings (e.g. deepseek-v4-flash:cloud).

APPLES-TO-APPLES with the CE probe: same gold files, same _user_turns window
(16-turn, age>=age_threshold), same rank-then-budget gate. The LLM RANKS its
top-3 referent turns from the window; top1 = #1==target, in_top3 = target in the
LLM's 3. breadth=3 by construction (LLM returns 3). competitor_beats = target not
rank-1. This is the same gate the CE path HOLDs at 0.294 top1.

UNTRACKED scratch. onyx PRIVATE -- nothing leaves the box. No uploads.
Run: PYTHONPATH=. python scripts/_scratch/_llm_salience_probe.py
  (env: MODEL=qwen3:8b  default; WHICH=both|heldout|loo  default both)
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
OUT_RESULT = SCRATCH / "_llm_salience_result.json"

URL = "http://127.0.0.1:11434/api/chat"
MODEL = os.getenv("MODEL", "qwen3:8b")
WHICH = os.getenv("WHICH", "both")
BUDGET = 3
LOO_MIN_PAIRS = 6
MAX_TURN_CHARS = 600  # cap each turn's text in the prompt

# METRIC DEGENERACY (vs the cosine/CE probes). The LLM always returns a ranked
# top-3, so: breadth = min(BUDGET, len(picked)) is constant 3.0 (no cost axis),
# and beats = (rank is None or rank > 1) == NOT top1, so competitor_beats_rate
# == 1 - target_top1_rate exactly (verified: 0.7647 = 1-0.2353, 0.4677 = 1-0.5323).
# So the ship gate reduces, for the LLM, to: top1 >= 2/3 AND in_top3 >= 2/3.
# (For cosine, beats was a real cos-value axis and breadth a real cost axis; for
# the LLM both collapse. This is honest, not a bug -- reported so the gate is not
# read as more constraining than it is.)


def _user_turns(sess):
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def llm_rank(query_text, window_turns):
    """Ask the LLM to rank the 3 turns the query refers back to. Returns a list
    of up to 3 window-indices, most-relevant first (or [] on failure)."""
    letters = [chr(ord("A") + i) for i in range(len(window_turns))]
    lines = []
    for L, txt in zip(letters, window_turns):
        t = (txt or "").strip().replace("\n", " ")
        if len(t) > MAX_TURN_CHARS:
            t = t[:MAX_TURN_CHARS] + "…"
        lines.append(f"{L}: {t}")
    window_block = "\n".join(lines)
    user = (
        f"Here are the last {len(window_turns)} user turns from a conversation, "
        f"oldest first, each labeled with a letter:\n\n{window_block}\n\n"
        f"New user turn (the query):\n{query_text.strip()}\n\n"
        f"Task: anaphora resolution. Which 3 prior turns does this new turn most "
        f"refer back to? Think about pronouns, implicit subjects, and what the "
        f"new turn is actually building on -- not merely topically-similar turns. "
        f"Reply with EXACTLY 3 letters, most-relevant first, comma-separated "
        f"(e.g. 'G, C, K'). Reply with only the letters, nothing else."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise anaphora / "
             "coreference resolver for multi-turn conversation. You output only "
             "what is asked."},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                obj = json.loads(resp.read().decode("utf-8", errors="replace"))
                content = obj.get("message", {}).get("content", "")
                # parse letters out of the reply (take first 3 A-P tokens found)
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
        except (urllib.error.URLError, TimeoutError) as e:
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
            # target's position within window_turns
            target_in_win = t - lo
            picked, raw = llm_rank(ut[q], window_turns)
            # map picked window-positions back to absolute; rank of target
            rank = None
            for pos, wpos in enumerate(picked, 1):
                if wpos == target_in_win:
                    rank = pos
                    break
            top1 = rank == 1
            in3 = rank is not None and rank <= BUDGET
            beats = rank is None or rank > 1  # a competitor outranks the target
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
    ship = (n_in >= 6 and top1 >= 2/3 and top3 >= 2/3 and beats <= 1/3 and med <= BUDGET)
    res = {"label": label, "n_in_window": n_in,
           "target_top1_rate": round(top1, 4),
           "target_in_top3_rate": round(top3, 4),
           "competitor_beats_rate": round(beats, 4),
           "median_breadth": round(med, 4), "ship": ship,
           "per_pair": per_pair}
    print(f"\n=== {label} LLM-SALIENCE GATE (n={n_in}) model={MODEL} ===", flush=True)
    print(f"  target_top1   = {res['target_top1_rate']}  (need >= 0.667; CE was 0.294)", flush=True)
    print(f"  target_in_top3= {res['target_in_top3_rate']}  (need >= 0.667; CE was 0.588)", flush=True)
    print(f"  competitor_beats={res['competitor_beats_rate']}  (need <= 0.333)", flush=True)
    print(f"  median_breadth= {res['median_breadth']}", flush=True)
    print(f"  VERDICT: {'SHIP' if ship else 'HOLD'}", flush=True)
    return res


def main() -> int:
    out = {"model": MODEL, "which": WHICH}
    if WHICH in ("heldout", "both"):
        ep = json.loads(EP_HELDOUT.read_text(encoding="utf-8"))
        g = json.loads(GOLD_HELDOUT.read_text(encoding="utf-8"))
        r = eval_set(ep, g["pairs"], int(g.get("window", 16)),
                     int(g.get("age_threshold", 3)), "heldout-17")
        out["heldout"] = r
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
        r = eval_set(ep, loo, w, a, "loo-normal")
        out["loo"] = r
    OUT_RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nwrote {OUT_RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())