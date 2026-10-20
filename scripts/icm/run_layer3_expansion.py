"""LAYER-3 EXPERIMENT -- does a non-reach produce a better new intent?

PRE-REGISTERED DESIGN (written before the run; prepared fixture, not an organic
claim):

WORLD (icm/layer3_env.py): each session materializes a real dataset with ONE
hidden verification rule (header row / excluded file / name-inclusive hash /
duplicate rows). The agent submits a manifest spec {files, rows, hash_mode};
the verifier is mechanical. On a mismatch it returns a reason that NAMES the
violated rule. The first manifest is produced from the task + file metadata
only, so the rule is not knowable before the failure.

ARMS (on sessions whose FIRST attempt is violated):
  EXPANSION     the agent sees the failure reason, explains the rule, states a
                NEW INTENT, and submits a corrected manifest. Recorded in the
                capture as a new goal with `triggered_by` = the violated
                outcome (the causal-direction law, enforced by capture).
  RETRY         the same first spec is resubmitted (deterministic control).
  NO-EVIDENCE   the agent is told only that the attempt failed (no reason) and
                submits a revised manifest.

ORACLE: the sandbox verifier -- mechanical, no judge. The explanation is checked
against a rule-specific keyword only (alignment, not the outcome).

PRE-REGISTERED BAR:
  (1) POWER:       failed-first sessions >= 30 (of --sessions 40)
  (2) EXPANSION    >= 0.80
  (3) RETRY        <= 0.10
  (4) NO-EVIDENCE  <= 0.30
  (5) LIFT         expansion - no-evidence >= 0.40
  (6) LINK         100% of expansion goals carry triggered_by = the violated
                   outcome id
  (7) ALIGNMENT    >= 0.80 of explanations address the failure the verifier
                   actually reported (reason-based keywords, not the hidden
                   rule: a row-count error under the name-hash rule aligns to
                   rows)
  Report the first-attempt pass rate and a per-rule breakdown. PASS requires
  all. This tests the LOOP MECHANISM in a prepared world; organic applicability
  is not claimed.

Usage:
  python scripts/icm/run_layer3_expansion.py --sessions 40 --time-budget 480
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from icm.capture import IntentCapture  # noqa: E402
from icm.intent_keys import ollama_ask  # noqa: E402
from icm.layer3_env import RULES, Layer3Env, reason_keywords  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "layer3")
ROWS = os.path.join(OUTDIR, "rows.jsonl")
RESULTS = os.path.join(OUTDIR, "results.json")
LEDGER_DB = os.path.join(OUTDIR, "ledger")
MIRROR = os.path.join(OUTDIR, "capture")

TASK = "Ingest the dataset and produce a manifest that passes verification."

FIRST_PROMPT = """You are ingesting a dataset. Its files and line counts:
{meta}

Produce a manifest with:
- files: the number of files to include
- rows: the number of DATA rows
- hash_mode: "content" or "content+names"

Return JSON only:
{{"intent": "<one-line aim>", "spec": {{"files": N, "rows": N, "hash_mode": "content"|"content+names"}}}}"""

EXPAND_PROMPT = """You are ingesting a dataset. Its files and line counts:
{meta}

Your manifest was:
{spec}

Verification failed with: {reason}

Infer the rule from the failure, explain it in one sentence, state a NEW intent
(one line), and submit a corrected manifest.

Return JSON only:
{{"explanation": "<one sentence>", "intent": "<one-line new aim>", "spec": {{"files": N, "rows": N, "hash_mode": "content"|"content+names"}}}}"""

NOEVIDENCE_PROMPT = """You are ingesting a dataset. Its files and line counts:
{meta}

Your manifest was:
{spec}

Verification failed, but no failure details are available. State a new intent
(one line) and submit a revised manifest.

Return JSON only:
{{"explanation": "<one sentence>", "intent": "<one-line new aim>", "spec": {{"files": N, "rows": N, "hash_mode": "content"|"content+names"}}}}"""


def first_json(raw: str):
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def meta_text(meta: dict) -> str:
    return "; ".join(f"{f} ({n} lines)"
                     for f, n in zip(meta["files"], meta["line_counts"]))


def ask_json(prompt: str) -> dict | None:
    return first_json(ollama_ask(prompt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--time-budget", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    env = Layer3Env(os.path.join(OUTDIR, "world"), seed=args.seed)
    ledger = ICMLedger(LEDGER_DB)
    done: dict[str, dict] = {}
    if os.path.exists(ROWS):
        for line in io.open(ROWS, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["session"]] = r
            except Exception:
                continue
    print(f"[checkpoint] {len(done)} sessions", flush=True)
    fh = io.open(ROWS, "a", encoding="utf-8", newline="\n")
    t0 = time.time()
    try:
        for i in range(args.sessions):
            sid = f"l3-{i:03d}"
            if sid in done:
                continue
            rule = RULES[i % len(RULES)]
            meta = env.materialize(sid, rule)
            mt = meta_text(meta)
            first = ask_json(FIRST_PROMPT.format(meta=mt))
            spec1 = (first or {}).get("spec")
            check1 = env.check(sid, spec1)
            cap = IntentCapture(ledger, mirror_dir=MIRROR)
            g1 = cap.goal(str((first or {}).get("intent") or TASK),
                          desired="verification passes", epoch=1)
            s1 = cap.step(g1, "attempt 1: " + json.dumps(spec1)[:400])
            r1 = cap.retrieval(s1, "verify manifest", [])
            o1 = cap.outcome(r1, "desired" if check1["ok"] else "feared")
            row = {"session": sid, "rule": rule, "first_ok": bool(check1["ok"]),
                   "first_reason": check1["reason"]}
            if check1["ok"]:
                row.update({"expansion_ok": None, "retry_ok": None,
                            "noevidence_ok": None, "link_ok": None,
                            "aligned": None, "explanation": "",
                            "intent": str((first or {}).get("intent") or "")})
            else:
                exp = ask_json(EXPAND_PROMPT.format(meta=mt,
                                                    spec=json.dumps(spec1),
                                                    reason=check1["reason"]))
                exp_spec = (exp or {}).get("spec")
                exp_check = env.check(sid, exp_spec)
                exp_intent = str((exp or {}).get("intent") or
                                 "explain the failure and repair the manifest")
                explanation = str((exp or {}).get("explanation") or "")
                g2 = cap.goal(exp_intent, desired="verification passes",
                              epoch=2, triggered_by=o1)
                s2 = cap.step(g2, "expansion: " + explanation[:300])
                r2 = cap.retrieval(s2, "verify revised manifest", [])
                cap.outcome(r2, "desired" if exp_check["ok"] else "feared")
                noev = ask_json(NOEVIDENCE_PROMPT.format(
                    meta=mt, spec=json.dumps(spec1)))
                noev_check = env.check(sid, (noev or {}).get("spec"))
                link_ok = ledger.get("icapture", g2).get("triggered_by") == o1
                aligned = any(k in explanation.lower()
                              for k in reason_keywords(check1["reason"]))
                row.update({"expansion_ok": bool(exp_check["ok"]),
                            "retry_ok": bool(check1["ok"]),
                            "noevidence_ok": bool(noev_check["ok"]),
                            "link_ok": bool(link_ok), "aligned": bool(aligned),
                            "explanation": explanation,
                            "intent": exp_intent})
            cap.close()
            done[sid] = row
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            fh.flush()
            print(f"  [{len(done)}/{args.sessions}] {sid} {rule} "
                  f"first={row['first_ok']} "
                  f"exp={row.get('expansion_ok')} noev={row.get('noevidence_ok')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if args.time_budget and (time.time() - t0) >= args.time_budget:
                print(f"[budget] stopping after {len(done)} sessions", flush=True)
                break
    finally:
        fh.close()
        ledger.close()

    rows = [done[f"l3-{i:03d}"] for i in range(args.sessions)
            if f"l3-{i:03d}" in done]
    failed = [r for r in rows if not r["first_ok"]]
    complete = len(rows) == args.sessions
    exp = [r for r in failed if r["expansion_ok"] is not None]
    if exp:
        exp_rate = round(statistics.mean(r["expansion_ok"] for r in exp), 3)
        retry_rate = round(statistics.mean(r["retry_ok"] for r in exp), 3)
        noev_rate = round(statistics.mean(r["noevidence_ok"] for r in exp), 3)
        link_rate = round(statistics.mean(r["link_ok"] for r in exp), 3)
        align_rate = round(statistics.mean(r["aligned"] for r in exp), 3)
    else:
        exp_rate = retry_rate = noev_rate = link_rate = align_rate = None
    per_rule = {}
    for rule in RULES:
        rs = [r for r in failed if r["rule"] == rule and r["expansion_ok"] is not None]
        if rs:
            per_rule[rule] = {
                "n": len(rs),
                "expansion": round(statistics.mean(r["expansion_ok"] for r in rs), 3),
                "no_evidence": round(statistics.mean(r["noevidence_ok"] for r in rs), 3),
            }
    checks = {
        "power_failed_ge_30": len(failed) >= 30,
        "expansion_ge_0.80": (exp_rate or 0) >= 0.80,
        "retry_le_0.10": (retry_rate if retry_rate is not None else 1) <= 0.10,
        "no_evidence_le_0.30": (noev_rate if noev_rate is not None else 1) <= 0.30,
        "lift_ge_0.40": ((exp_rate or 0) - (noev_rate or 0)) >= 0.40,
        "link_rate_1.0": link_rate == 1.0,
        "alignment_ge_0.80": (align_rate or 0) >= 0.80,
    }
    res = {"sessions": len(rows), "complete": complete,
           "first_pass_rate": round(statistics.mean(r["first_ok"] for r in rows), 3)
           if rows else None,
           "failed_first": len(failed), "expansion_rate": exp_rate,
           "retry_rate": retry_rate, "noevidence_rate": noev_rate,
           "link_rate": link_rate, "alignment_rate": align_rate,
           "per_rule": per_rule, "checks": checks,
           "pass": all(checks.values()) and complete,
           "verdict": ("INCOMPLETE -- resume" if not complete else
                       "PASS -- non-reach produces a reaching new intent"
                       if all(checks.values()) else
                       "FAIL -- see the failing row")}
    with io.open(RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps({**res, "rows": rows}, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return (0 if (complete and res["pass"]) else (2 if complete else 3))


if __name__ == "__main__":
    sys.exit(main())
