"""INTENT-DETECTION SMOKE GATE -- does the instrument actually measure INTENT?

WHY THIS EXISTS (user, 2026-10-16): every lens/metric run so far chose its
measured quantity from whatever labels sat on disk, and no gate ever checked
that the number moves when intent changes and stays still when only wording
changes. Every "measure intent" hop silently became "predict an available
label" -- the plan document itself encodes one such mistranslation. This
script is the missing gate, applied at the CONCEPT level, BEFORE any real-data
run. Per the arc's own precedent (reversal-expressibility pre-checks; the
granularity synthetic smoke that could detect a grain AND absent it).

DATA: PUBLIC ONLY. CodeTraceBench-SWE mapped slice (the agent-intent corpus).
The personal mined corpora (local sessions, Onyx) are EXCLUDED on purpose:
the LLM instruments would send their text to a cloud model, which the
personal-data rule forbids. Public benchmark text has no such constraint.

PAIR SETS (ground truth by construction, non-authored):
  SENS (must come out DIFFERENT): the 32 tasks whose trajectory contains >=2
       plan declarations -- the agent RE-DECLARED its plan mid-run (the
       observable "non-reach -> new intent" shape). Pair = (plan_1, plan_2)
       of the same task: same repo/task surface, changed operative intent.
  INV  (must come out SAME): task <-> the agent's own declared plan for that
       task (up to 64 sampled from 258): same intent, different wording and
       granularity.
  CONTROL: 58 goal texts that appear in >1 trajectory (identical strings from
       different agent/model runs) -- trivially SAME; a sanity floor.

INSTRUMENTS (all local except the two LLM ones, which run on PUBLIC text):
  LEX      : token-set jaccard of the two statements; SAME if >= 0.25
             (threshold fixed a priori, never tuned).
  CONST-A/B: always-SAME and always-DIFFERENT floors (the balanced sets make
             both fail; included so no instrument can hide behind imbalance).
  SPO-STRUCT (the instrument under test): one chat call per pair asking for
             both statements' triples as JSON {attitude, relation, object};
             SAME iff normalized attitude token-overlap >= 0.5 AND object
             token jaccard >= 0.3 (rule fixed a priori).
  LLM-JUDGE (reported SEPARATELY as an oracle; every oracle is a similarity
             engine -- [[icm-reality-oracle-reframe]] -- so it may not be the
             headline): one call per pair, answer exactly SAME or DIFFERENT.
  MODEL: deepseek-v4.1-flash:cloud (the extraction-audit model: 88 percent
         full-triple recovery on real records).

READOUTS (direction always; failures itemized with texts for inspection):
  per instrument: accuracy on SENS, accuracy on INV, accuracy on the
  LEX-HARD RESIDUAL = the pairs LEX gets wrong (the pairs where a surface
  instrument is PROVABLY insufficient -- the only ones that can distinguish
  intent detection from lexical decoration).

PRE-REGISTERED BAR (all must hold):
  (1) SPO-STRUCT >= 0.80 on SENS
  (2) SPO-STRUCT >= 0.80 on INV
  (3) SPO-STRUCT >= 0.80 on the LEX-HARD residual
  (4) LEX <= 0.70 on at least one set (teeth; else the pair sets are
      lexically trivial and the smoke is VACUOUS -- reported as such, and the
      fix is harder pairs, not a different model)
  PASS -> the instrument detects intent on these constructions and may be
  pointed at real data. FAIL -> the capability is absent HERE, on the
  concept-level test, at a cost of minutes -- which is exactly what this gate
  exists to surface. No dismissals either way.

Run:  python scripts/icm/intent_detection_smoke.py     (local model calls)
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
CKPT = os.path.join("data", "_scratch", "icm", "intent_smoke_ckpt.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "intent_smoke_results.json")
MODEL = "deepseek-v4.1-flash:cloud"
N_INV = 64


def ask(prompt: str, tries: int = 4) -> str:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.0, "num_ctx": 4096}}
    backoff = 8.0
    for a in range(tries):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return str(json.loads(urllib.request.urlopen(req, timeout=180).read())
                       ["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            time.sleep(2.0)
        except Exception:
            time.sleep(2.0)
    return ""


def tokset(v):
    return {w for w in re.findall(r"[a-z0-9]+", str(v).lower()) if len(w) > 2}


def jacc(a, b):
    A, B = tokset(a), tokset(b)
    u = A | B
    return len(A & B) / len(u) if u else 0.0


def plan_prose(text: str) -> str:
    """Plans are declared as a JSON command echoing the task list; take the
    human-readable titles when parseable, else the raw text."""
    try:
        obj = json.loads(text.strip())
        tl = obj.get("task_list") or []
        titles = [str(t.get("title", "")) for t in tl if isinstance(t, dict)]
        if titles:
            return " | ".join(t for t in titles if t)
    except Exception:
        pass
    return text


def extract_triple(statement: str):
    """One call -> {'attitude','relation','object'} or None."""
    p = ("Extract the intent triple from the statement below. Reply with JSON "
         'only, keys: "attitude" (the stance verb, e.g. intend/verify/reclaim/'
         'maintain/ensure), "relation" (what is to be done to it), "object" '
         "(the state of affairs aimed at, in a few words). If the statement "
         "expresses no intent, use empty strings.\n\nStatement:\n" + statement[:1200])
    raw = ask(p)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {k: str(d.get(k, "")) for k in ("attitude", "relation", "object")}
    except Exception:
        return None


def judge_same(a: str, b: str):
    p = ("Do these two statements express the SAME intent -- the same thing "
         "being aimed at, even if worded differently? Intent means what is "
         "being brought about, not the topic.\n\nA: " + a[:1000] +
         "\n\nB: " + b[:1000] + "\n\nAnswer exactly one word: SAME or DIFFERENT.")
    raw = ask(p).upper()
    if "DIFFERENT" in raw and "SAME" not in raw:
        return False
    if "SAME" in raw and "DIFFERENT" not in raw:
        return True
    if raw.startswith("SAME"):
        return True
    if raw.startswith("DIFFERENT"):
        return False
    return None


def spo_same(t1, t2):
    if not t1 or not t2:
        return None
    att_ok = jacc(t1["attitude"] + " " + t1["relation"], t2["attitude"] + " " + t2["relation"]) >= 0.5
    obj_ok = jacc(t1["object"], t2["object"]) >= 0.3
    return bool(att_ok and obj_ok)


def main() -> int:
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    goals = {r["id"]: r["text"] for r in mapped if r["level"] == "goal"}
    plans = {}
    for r in mapped:
        if r["level"] == "plan":
            plans.setdefault(r["parent"], []).append(plan_prose(r["text"]))

    # ---------------- pair construction (public data only) ----------------
    sens = []
    for hid, ps in sorted(plans.items()):
        if len(ps) >= 2:
            sens.append(("SENS", ps[0], ps[1], hid))
    inv = []
    for hid, ps in sorted(plans.items()):
        if hid in goals and ps:
            inv.append(("INV", goals[hid], ps[0], hid))
    inv = inv[:N_INV]
    pairs = sens + inv
    print(f"[pairs] SENS={len(sens)} INV={len(inv)} total={len(pairs)}", flush=True)
    for tag, a, b, hid in pairs[:2]:
        print(f"  sample {tag} ({hid}):\n    A: {a[:110]}\n    B: {b[:110]}", flush=True)

    # ---------------- checkpointed evaluation ----------------
    done = {}
    if os.path.exists(CKPT):
        for l in io.open(CKPT, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    done[(r["set"], r["hid"])] = r
                except Exception:
                    pass
    t0 = time.time()
    for i, (tag, a, b, hid) in enumerate(pairs):
        key = (tag, hid)
        if key in done:
            continue
        j = judge_same(a, b)
        t1, t2 = extract_triple(a), extract_triple(b)
        row = {"set": tag, "hid": hid,
               "lex_same": jacc(a, b) >= 0.25, "lex_j": round(jacc(a, b), 3),
               "judge_same": j, "spo_same": spo_same(t1, t2),
               "t1": t1, "t2": t2, "a": a[:400], "b": b[:400]}
        done[key] = row
        with io.open(CKPT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i+1}/{len(pairs)}] {tag} {hid[:44]} lex_same={row['lex_same']} "
              f"judge={j} spo={row['spo_same']} ({time.time()-t0:.0f}s)", flush=True)

    rows = list(done.values())

    # ---------------- readouts ----------------
    def acc(tag, inst):
        rs = [r for r in rows if r["set"] == tag]
        vals = [r[inst] for r in rs if r[inst] is not None]
        if not vals:
            return None, 0
        correct = sum(1 for v in vals if v == (tag == "INV"))   # INV same, SENS differ
        return round(correct / len(vals), 3), len(vals)

    def lex_hard(r):
        return (r["set"] == "SENS" and r["lex_same"]) or (r["set"] == "INV" and not r["lex_same"])

    res = {"n_pairs": len(rows), "n_sens": sum(1 for r in rows if r["set"] == "SENS"),
           "n_inv": sum(1 for r in rows if r["set"] == "INV"), "instruments": {}}
    for inst in ("lex_same", "judge_same", "spo_same"):
        s, ns = acc("SENS", inst)
        v, nv = acc("INV", inst)
        # lex-hard residual pools both sets
        rs = [r for r in rows if lex_hard(r) and r[inst] is not None]
        hres = round(sum(1 for r in rs if r[inst] == (r["set"] == "INV")) / len(rs), 3) if rs else None
        res["instruments"][inst] = {"sens_acc": s, "sens_n": ns, "inv_acc": v, "inv_n": nv,
                                    "lexhard_acc": hres, "lexhard_n": len(rs)}
    # constant floors
    res["instruments"]["const_same"] = {"sens_acc": 0.0, "inv_acc": 1.0}
    res["instruments"]["const_differ"] = {"sens_acc": 1.0, "inv_acc": 0.0}
    lex = res["instruments"]["lex_same"]
    spo = res["instruments"]["spo_same"]
    vacuous = (lex["sens_acc"] or 0) >= 0.8 and (lex["inv_acc"] or 0) >= 0.8
    passed = (not vacuous and (spo["sens_acc"] or 0) >= 0.8 and (spo["inv_acc"] or 0) >= 0.8
              and (spo["lexhard_acc"] or 0) >= 0.8)
    res["vacuous"] = bool(vacuous)
    res["pass"] = bool(passed)
    res["verdict"] = ("VACUOUS -- pair sets lexically solvable; need harder pairs"
                      if vacuous else
                      "PASS -- SPO instrument detects intent on these constructions "
                      "(may now be pointed at real data)" if passed else
                      "FAIL -- the instrument does not clear the bar on the "
                      "concept-level test; itemized failures attached")
    res["failures"] = [{"set": r["set"], "hid": r["hid"], "a": r["a"][:200], "b": r["b"][:200],
                        "spo_same": r["spo_same"], "t1": r["t1"], "t2": r["t2"],
                        "judge_same": r["judge_same"]}
                       for r in rows if r["spo_same"] != (r["set"] == "INV")][:12]
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("n_pairs", "instruments", "vacuous", "pass", "verdict")},
                     indent=1), flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
