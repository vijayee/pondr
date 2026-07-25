"""Build 1 -- dense qwen teacher labels for SSM-state distillation.

Extends the C2 teacher (``_llm_salience_probe.py``) from a hard top-3 pick to a
DENSE per-window-turn soft relevance score in [0,1] -- the supervision signal
for training ``CrossSlotTransformerZHead`` over raw SSM states (Build 3). The
teacher reads each window turn's ORIGINAL text + the query (no state->text
decoder needed -- the original text IS the slot text) and scores EVERY window
turn, not just the top-3. Drives over EVERY turn q in [window, len(ut)) for each
TRAIN session (not just gold-pair turns) -> dense supervision.

The distillation crux: the teacher works over TEXT (which it can read); the
student (cross-slot transformer) works over the SSM STATE that produced that
text. Pairing these labels with the states (Build 2 + Build 3) forces the
student to learn that the state encodes the text's query-relevance -- the fair
test of whether the SSM carries anaphora signal at the right primitive.

SCOPE. LOO-8 first (8 sessions, ~350 qwen calls, ~30-60 min). ``SCOPE=trained``
expands to all 27 trained-gold sessions (Build 5). heldout-17 is the GATE -- no
teacher labels needed for it (graded by gold, not the teacher).

RESUME. Re-reads the output JSON on start and skips any (session, q) already
labeled, so an interrupted run loses no work.

NO engine change. onyx PRIVATE -- nothing leaves scripts/_scratch/. No uploads.
Per CLAUDE.md de-wonk at completion.

Run (qwen3:8b local ollama must be up at 127.0.0.1:11434):
  PYTHONPATH=. python scripts/_scratch/_dense_teacher_labels.py
  (env: MODEL=qwen3:8b default; SCOPE=loo8 default|trained; MAX_QTURN=0 = all)
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

URL = "http://127.0.0.1:11434/api/chat"
MODEL = os.getenv("MODEL", "qwen3:8b")
SCOPE = os.getenv("SCOPE", "loo8")          # loo8 (default) | trained (all 27)
MAX_QTURN = int(os.getenv("MAX_QTURN", "0"))  # 0 = label every q in [window, len(ut))
LOO_MIN_PAIRS = 6
MAX_TURN_CHARS = 600
NUM_PREDICT = 0            # 0 = no cap (qwen3 thinks first, then answers; a cap
                           # lets thinking eat the budget -> empty content). C2
                           # set no cap and worked; we match it.
REQUEST_TIMEOUT = 300      # raised from C2's 180: thinking + dense reply is longer

OUT_PATH = SCRATCH / (
    "_dense_teacher_labels_loo8.json" if SCOPE == "loo8"
    else "_dense_teacher_labels_trained27.json")


def _user_turns(sess):
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def _loo_sessions(gold):
    """LOO-8 normal set: sessions with >= LOO_MIN_PAIRS gold pairs."""
    by_sid = {}
    for p in gold["pairs"]:
        by_sid[p["session_id"]] = by_sid.get(p["session_id"], 0) + 1
    return sorted(sid for sid, n in by_sid.items() if n >= LOO_MIN_PAIRS)


def _trained_sessions(gold):
    """All 27 trained-gold sessions (Build 5 expansion)."""
    return sorted({p["session_id"] for p in gold["pairs"]})


def llm_dense_scores(query_text, window_turns):
    """Ask the LLM for a relevance score in [0,1] for EVERY window turn. Returns
    (scores, raw_content, ok). ``scores`` is a list[float] of len(window_turns)
    (0.0 for any letter the model omits; clamped to [0,1]) when ``ok``; on
    transport failure after 3 retries, returns ``(None, "", False)`` so the
    driver skips the turn (never records it done) rather than storing an
    all-zero label. Falls back to the C2 hard-pick logic (1.0 for the first 3
    letters it names, 0.0 else) if no letter:score pairs parse -- so a malformed
    dense reply still yields a usable label vector instead of silence."""
    letters = [chr(ord("A") + i) for i in range(len(window_turns))]
    lines = []
    for L, txt in zip(letters, window_turns):
        t = (txt or "").strip().replace("\n", " ")
        if len(t) > MAX_TURN_CHARS:
            t = t[:MAX_TURN_CHARS] + "…"
        lines.append(f"{L}: {t}")
    window_block = "\n".join(lines)
    # Prompt gotcha (learned in smoke): if the template says 'L:score' the model
    # emits the LITERAL letter "L" nine times instead of substituting A,B,C....
    # So we (1) name the actual letters, (2) give a concrete example using REAL
    # letters from this window, (3) say "the ACTUAL letter, not the word L", and
    # (4) ask for letter-order (not most-relevant-first) so every line is labeled
    # by position too -- a positional fallback can recover it if labels drop.
    ex_lines = "\n".join(f"{L}:0.0" for L in letters[:3]) + "\n..."
    user = (
        f"Here are the last {len(window_turns)} user turns from a conversation, "
        f"oldest first. Each is labeled with its letter "
        f"({', '.join(letters)}):\n\n{window_block}\n\n"
        f"New user turn (the query):\n{query_text.strip()}\n\n"
        f"Task: anaphora resolution. For EACH prior turn, output a relevance "
        f"score in [0.00, 1.00] for how much THIS new turn refers back to it "
        f"(pronouns, implicit subjects, what it builds on -- NOT mere topic "
        f"overlap).\n\n"
        f"Output EXACTLY {len(window_turns)} lines, one per letter, in this "
        f"order: {', '.join(letters)}.\n"
        f"Each line MUST start with the ACTUAL letter (not the word 'L'). "
        f"Example:\n{ex_lines}\n\n"
        f"Give every letter its own line with its score. Reply with only those "
        f"{len(window_turns)} lines, nothing else."
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
        "options": {"temperature": 0.0, "seed": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                obj = json.loads(resp.read().decode("utf-8", errors="replace"))
                content = obj.get("message", {}).get("content", "")
                return _parse_dense(content, len(window_turns)), content, True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(2)
    # All retries failed. Return None so the driver SKIPS this q (does not add
    # to `done`) -- an all-zero label would silently teach "nothing is relevant"
    # and, because it is recorded as done, never be retried. Skipping leaves it
    # unlabeled so a resume pass retries it.
    print(f"    [dense] qwen FAILED after 3 retries: {last_err}; skipping",
          file=sys.stderr, flush=True)
    return None, "", False


def _parse_dense(content, n):
    """Parse 'L:score' lines into a dense [n] vector (letter A->idx 0). Defaults
    0.0 for omitted letters; clamps to [0,1]. Three recovery paths so a slightly
    malformed reply still yields a usable label vector instead of silence:
      1. letter:score pairs (the intended format),
      2. positional: score-only lines in letter order (if labels dropped),
      3. C2 hard-pick: first 3 distinct A-P tokens get 1.0 (last resort)."""
    pairs = re.findall(r"\b([A-P])\s*[:\s]\s*([0-9]*\.?[0-9]+)", content.upper())
    if pairs:
        scores = [0.0] * n
        for L, s in pairs:
            i = ord(L) - ord("A")
            if 0 <= i < n:
                v = float(s)
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                # if a letter repeats, keep the first occurrence
                if scores[i] == 0.0:
                    scores[i] = v
        # require at least 2 distinct letters actually used, else the model may
        # have emitted the literal "L" for every line (the smoke bug) -- fall
        # through to positional.
        if sum(1 for i in range(n) if scores[i] != 0.0) >= 2 or len(pairs) >= n:
            return scores
    # Positional fallback: score-only lines (e.g. ":0.95" or bare "0.95"),
    # one per letter in order. Recovers when the model drops the letter label.
    bare = re.findall(r"(?m)^\s*[A-Za-z]?\s*[:\s]?\s*([0-9]*\.?[0-9]+)\s*$",
                      content)
    if len(bare) >= n:
        scores = [0.0] * n
        for i in range(n):
            v = float(bare[i])
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            scores[i] = v
        return scores
    # Last resort: C2 hard-pick -- first 3 distinct A-P tokens get 1.0, rest 0.0.
    hits = re.findall(r"\b([A-P])\b", content.upper())
    scores = [0.0] * n
    seen = set()
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        i = ord(h) - ord("A")
        if 0 <= i < n:
            scores[i] = 1.0
        if len(seen) >= 3:
            break
    return scores


def main() -> int:
    if not EP_TRAINED.exists() or not GOLD_TRAINED.exists():
        print(f"ERROR: need {EP_TRAINED} and {GOLD_TRAINED}", file=sys.stderr)
        return 1
    ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
    window = int(gold.get("window", 16))
    age_threshold = int(gold.get("age_threshold", 3))

    if SCOPE == "loo8":
        sids = _loo_sessions(gold)
    elif SCOPE == "trained":
        sids = _trained_sessions(gold)
    else:
        print(f"ERROR: SCOPE={SCOPE} not in (loo8, trained)", file=sys.stderr)
        return 1
    print(f"[dense] model={MODEL} scope={SCOPE} sessions={len(sids)} "
          f"window={window} age_threshold={age_threshold}", flush=True)
    print(f"[dense] OUT={OUT_PATH}", flush=True)

    # Resume: load any prior partial output and index already-labeled (sid,q).
    out = {"model": MODEL, "scope": SCOPE, "window": window,
           "age_threshold": age_threshold, "sessions": {}}
    done: set[tuple[str, int]] = set()
    if OUT_PATH.exists():
        try:
            prev = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            for sid, turns in prev.get("sessions", {}).items():
                for rec in turns:
                    done.add((sid, int(rec["q"])))
                out["sessions"][sid] = turns
            print(f"[dense] resumed {len(done)} labeled turns from {OUT_PATH}",
                  flush=True)
        except (json.JSONDecodeError, KeyError):
            print(f"[dense] could not parse prior {OUT_PATH}; starting fresh",
                  flush=True)

    n_calls = n_skipped = 0
    for sid in sids:
        sess = ep.get(sid)
        if sess is None:
            print(f"  [{sid[:8]}] no transcript in episodes; skip", flush=True)
            continue
        ut = _user_turns(sess)
        sess_turns = out["sessions"].setdefault(sid, [])
        # q must have a non-empty window: q >= 1 (window = turns before q).
        qmax = len(ut)
        if MAX_QTURN > 0:
            qmax = min(qmax, window + MAX_QTURN)
        for q in range(1, qmax):
            if (sid, q) in done:
                n_skipped += 1
                continue
            lo = max(0, q - window)
            win_idx = list(range(lo, q))
            if not win_idx:
                continue
            window_turns = [ut[j] for j in win_idx]
            scores, raw, ok = llm_dense_scores(ut[q], window_turns)
            if not ok:
                # transport failure: do NOT record as done -- resume retries it.
                continue
            n_calls += 1
            # scores[i] is the relevance of window turn win_idx[i] (= absolute
            # user-turn index). Store as {N: score} keyed by absolute turn index.
            score_map = {str(win_idx[i]): round(float(scores[i]), 4)
                         for i in range(len(win_idx))}
            sess_turns.append({"q": q, "window_lo": lo, "scores": score_map,
                               "raw_head": raw[:80]})
            done.add((sid, q))
            if n_calls % 10 == 0 or n_calls < 5:
                top = sorted(score_map.items(), key=lambda kv: kv[1],
                             reverse=True)[:3]
                print(f"  [{sid[:8]}] q{q:03d} done (n={n_calls}) "
                      f"top3={[(int(k), v) for k, v in top]}", flush=True)
            # checkpoint every 25 calls (a 30-60 min run survives interrupts)
            if n_calls % 25 == 0:
                OUT_PATH.write_text(json.dumps(out, ensure_ascii=False),
                                    encoding="utf-8")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n[dense] done. calls={n_calls} resumed_skipped={n_skipped} "
          f"total_labeled={len(done)}", flush=True)
    print(f"[dense] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())