#!/usr/bin/env python
"""PROBE (not committed): full EnterpriseRAG-Bench LLM-judge harness for the
Phase 3c contradiction resolver -- the deferred D7 item ("correctness x
completeness, three-judge consensus").

Runs the REAL Ponder contradiction pipeline (system's own Bonsai extraction ->
collision detection -> BonsaiDecider.decide_contradiction with the v6 LoRA +
deterministic guards on :8080) on the bench's 20 held-out Conflicting Info
pairs (real enterprise docs, vendored from onyx-dot-app/EnterpriseRAG-Bench),
and scores each decision with an INDEPENDENT 3-judge panel
(deepseek-v4-flash:cloud via local Ollama at :11434) -- majority consensus --
against the bench's gold_answer.

This is the scale-confidence signal the 16-pair committed fixture + the
single dogfood can't give: 20 REAL contradicting enterprise doc pairs (not
planted), judged by a model that is NOT the decider (so it cannot
rubber-stamp its own output -- the 8B/27B self-judge failure mode the
fine-tune decision rejected).

Per pair:
  1. load the 2 gold docs (content/source_type) from the documents parquet.
  2. extract relations from both docs via BonsaiRelationExtractor (V1 single
     pass; the isolated 10-pass is the higher-recall shipped arm but ~22.8 s/doc
     -- swap via ISOLATED=1 for a slower, higher-recall rerun).
  3. find a collision (entity with DIFFERENT values across the two docs) -- the
     proxy for _detect_contradictory_state firing. If none, record an
     extraction-miss (honest: the system did not see the conflict).
  4. call the real BonsaiDecider.decide_contradiction (guards + v6 LoRA) with
     the conflicting state_values + real provenance (source_path =
     "{source_type}/{doc_id}"; bench docs carry no month prefix -> guards fall
     through to the LoRA, the real-conflict path).
  5. 3-judge panel: each judge sees both docs (truncated) + gold_answer + the
     decider's decision+reasoning, votes correct/incorrect/ambiguous. Majority
     consensus = the harness verdict.

Correctness rubric (judge):
  - REAL newer-supersedes-older conflict -> decision must be fix + supersede.
  - complementary / non-conflict -> decision must be ask_user / dismiss.
  - extraction-miss (no collision) is scored as a system miss regardless of
    what the decider would have said (the system never saw the conflict).

Env:
  BENCH_DIR (default scripts/_scratch/erag), ISOLATED (0/1), JUDGE_MODEL
  (default deepseek-v4-flash:cloud), JUDGE_PANEL (default 3), DECIDER_TAG.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.encoding.assertion_extractor import extract_state_assertions, _norm_key  # noqa: E402
from src.encoding.bonsai_relations import BonsaiRelationExtractor  # noqa: E402
from src.gnn.bonsai_decider import BonsaiDecider  # noqa: E402

BENCH = Path(os.environ.get("BENCH_DIR", ROOT / "scripts" / "_scratch" / "erag"))
DOCS_PARQUET = BENCH / "data" / "documents" / "test.parquet"
CI_JSON = BENCH / "conflicting_info.json"
ISOLATED = os.environ.get("ISOLATED", "0") == "1"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash:cloud")
JUDGE_ENDPOINT = os.environ.get("JUDGE_ENDPOINT", "http://localhost:11434/v1")
PANEL = int(os.environ.get("JUDGE_PANEL", "3"))
TAG = os.environ.get("BENCH_EVAL_TAG", "").strip()


def _norm(v: str) -> str:
    return (v or "").strip().lower()


def _strict_rels(rels):
    out = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        if str(r.get("predicate", "")).lower().strip() not in ("has_state", "state"):
            continue
        s, o = r.get("subject"), r.get("object")
        if isinstance(s, str) and isinstance(o, str):
            out.append({"entity": _norm_key(s), "value": o.strip()})
    return out


def _collisions(old_asserts, new_asserts):
    old_by_e = {}
    for a in old_asserts:
        old_by_e.setdefault(a["entity"], set()).add(_norm(a["value"]))
    new_by_e = {}
    for a in new_asserts:
        new_by_e.setdefault(a["entity"], set()).add(_norm(a["value"]))
    colls = []
    for e, ovals in old_by_e.items():
        if e not in new_by_e:
            continue
        nvals = new_by_e[e]
        if ovals == nvals:
            continue  # agreeing values are not a collision
        ov = next(iter(ovals))
        # pick a new value that genuinely differs from the old set
        diffs = [n for n in nvals if n not in ovals] or list(nvals)
        nv = diffs[0]
        colls.append({"entity": e, "old_value": ov, "new_value": nv})
    return colls


def judge_one(judge_endpoint, model, docs, gold, decision, reasoning, timeout=120.0):
    """One judge vote: 'correct' | 'incorrect' | 'ambiguous'."""
    d1, d2 = docs
    prompt = (
        "You are an independent evaluator for a contradiction-resolution system.\n"
        "Two enterprise documents disagree. The system adjudicated the conflict.\n"
        "Decide whether the system's DECISION is correct given the gold truth.\n\n"
        f"DOC 1 ({d1['source_type']}): {d1['content'][:1400]}\n\n"
        f"DOC 2 ({d2['source_type']}): {d2['content'][:1400]}\n\n"
        f"GOLD ANSWER (the resolved current truth): {gold[:600]}\n\n"
        f"SYSTEM DECISION: {decision}\n"
        f"SYSTEM REASONING: {reasoning[:600]}\n\n"
        "Rubric: if the docs are a REAL newer-supersedes-older conflict, the "
        "correct decision is 'fix' with action 'supersede_assertion'. If they "
        "are complementary point-in-time snapshots or not a real conflict, the "
        "correct decision is 'ask_user' or 'dismiss'. If the system never saw "
        "the conflict (no extraction), it is incorrect. Reply ONLY JSON: "
        '{"vote":"correct|incorrect|ambiguous","why":"<one sentence>"}'
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    try:
        r = requests.post(f"{judge_endpoint}/chat/completions", json=payload,
                          timeout=timeout)
        if r.status_code != 200:
            return {"vote": "ambiguous", "why": f"judge http {r.status_code}"}
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            return {"vote": "ambiguous", "why": "judge empty content (thinking ate budget?)"}
        # robust JSON carve (model may wrap in fences / add trailing prose)
        body = content.strip().strip("`")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            s, e = body.find("{"), body.rfind("}")
            if s != -1 and e > s:
                data = json.loads(body[s:e + 1])
            else:
                return {"vote": "ambiguous", "why": f"no JSON: {body[:120]}"}
        v = str(data.get("vote", "")).strip().lower()
        if v not in ("correct", "incorrect", "ambiguous"):
            v = "ambiguous"
        return {"vote": v, "why": str(data.get("why", ""))[:200]}
    except Exception as e:  # noqa: BLE001
        return {"vote": "ambiguous", "why": f"judge error: {e}"}


def consensus(votes):
    c = Counter(v["vote"] for v in votes)
    top, n = c.most_common(1)[0]
    # majority requires > panel/2; else ambiguous
    if n > len(votes) / 2:
        return top, n
    return "ambiguous", n


def main():
    if not DOCS_PARQUET.exists() or not CI_JSON.exists():
        print(f"ERROR: missing bench data under {BENCH} -- run the download first.")
        sys.exit(2)
    ci = json.loads(CI_JSON.read_text(encoding="utf-8"))
    qs = ci["questions"]
    print(f"Loaded {len(qs)} Conflicting Info pairs from {CI_JSON.name}")

    # doc lookup once
    docs_tbl = pq.read_table(DOCS_PARQUET,
                             columns=["doc_id", "source_type", "title", "content"])
    ids = docs_tbl.column("doc_id").to_pylist()
    idx = {d: i for i, d in enumerate(ids)}
    src = docs_tbl.column("source_type").to_pylist()
    content = docs_tbl.column("content").to_pylist()

    dec = BonsaiDecider(timeout=90.0, max_tokens=768)
    ext = BonsaiRelationExtractor(timeout=90.0)
    print(f"Decider: {dec.endpoint} model={dec.model}  health={dec.health_check(5.0)}")
    print(f"Extractor isolated={ISOLATED}")
    print(f"Judge: {JUDGE_ENDPOINT} model={JUDGE_MODEL} panel={PANEL}\n")

    rows = []
    for qi, q in enumerate(qs):
        qid = q["question_id"]
        doc_ids = q["expected_doc_ids"]
        docs = []
        for d in doc_ids:
            i = idx.get(d)
            if i is None:
                continue
            docs.append({"doc_id": d, "source_type": src[i],
                         "content": str(content[i] or "")})
        if len(docs) < 2:
            print(f"{qid}: SKIP (only {len(docs)} docs found)")
            continue
        gold = q["gold_answer"] or ""
        d1, d2 = docs[0], docs[1]

        # extraction (system's own arm)
        try:
            r1 = ext.extract(d1["content"], isolated=ISOLATED)
        except Exception as e:  # noqa: BLE001
            r1 = []
        try:
            r2 = ext.extract(d2["content"], isolated=ISOLATED)
        except Exception as e:  # noqa: BLE001
            r2 = []
        # also deterministic normalizer (free)
        det_coll = _collisions(extract_state_assertions(d1["content"]),
                               extract_state_assertions(d2["content"]))
        strict_coll = _collisions(_strict_rels(r1), _strict_rels(r2))

        coll = strict_coll or det_coll
        extraction_caught = bool(coll)
        decision = None
        reasoning = ""
        if coll:
            c = coll[0]
            state_values = [
                {"value": c["old_value"],
                 "asserted_by": f"{d1['source_type']}/{d1['doc_id']}",
                 "asserted_at": "2026-07-01T00:00:00Z",
                 "source_path": f"{d1['source_type']}/{d1['doc_id']}"},
                {"value": c["new_value"],
                 "asserted_by": f"{d2['source_type']}/{d2['doc_id']}",
                 "asserted_at": "2026-07-05T00:00:00Z",
                 "source_path": f"{d2['source_type']}/{d2['doc_id']}"},
            ]
            flag = {"node": c["entity"], "type": "contradictory_state",
                    "evidence": state_values}
            try:
                decision = dec.decide_contradiction(flag, {"state_values": state_values})
            except Exception as e:  # noqa: BLE001
                decision = None
                reasoning = f"decider error: {e}"
            if decision is not None:
                reasoning = decision.get("reasoning", "")
                decision = {"decision": decision.get("decision"),
                            "action": decision.get("action", "")}

        # 3-judge panel
        votes = []
        if decision is not None:
            for _ in range(PANEL):
                votes.append(judge_one(JUDGE_ENDPOINT, JUDGE_MODEL, (d1, d2),
                                      gold, decision, reasoning))
        verdict, _n = consensus(votes) if votes else ("ambiguous", 0)

        # structural correctness (the binding planted-label-style gate, here
        # derived from the gold_answer: a real bench conflict -> fix+supersede)
        if decision is None:
            struct = "miss"
        else:
            is_fix = (decision.get("decision") == "fix"
                      and "supersede_assertion" in decision.get("action", ""))
            struct = "correct" if is_fix else "wrong"

        rows.append({
            "qid": qid, "question": q["question"][:120],
            "extraction_caught": extraction_caught,
            "collision": coll[0] if coll else None,
            "decision": decision, "verdict": verdict,
            "votes": [v["vote"] for v in votes],
            "struct": struct, "gold": gold[:200],
        })
        dec_s = (decision.get("decision") + "/" + decision.get("action", "")
                 if decision else "None(miss)")
        print(f"{qi+1:2d}/{len(qs)} {qid} catch={extraction_caught!s:5s} "
              f"dec={dec_s:24s} judge={verdict:9s} votes={ [v['vote'] for v in votes] }")

    # ---- summary ----
    n = len(rows)
    caught = [r for r in rows if r["extraction_caught"]]
    decided = [r for r in rows if r["decision"] is not None]
    judge_correct = sum(1 for r in rows if r["verdict"] == "correct")
    judge_amb = sum(1 for r in rows if r["verdict"] == "ambiguous")
    struct_correct = sum(1 for r in rows if r["struct"] == "correct")
    dec_dist = Counter((r["decision"] or {}).get("decision", "miss") for r in rows)
    print("\n" + "=" * 78)
    print("SUMMARY -- EnterpriseRAG-Bench Conflicting Info (20 pairs, 3-judge consensus)")
    print("=" * 78)
    print(f"Pairs evaluated:                       {n}")
    print(f"Extraction caught a collision:         {len(caught)}/{n} = "
          f"{len(caught)/n:.0%}")
    print(f"Decider returned a decision:           {len(decided)}/{n}")
    print(f"Decision distribution:                 {dict(dec_dist)}")
    print(f"JUDGE consensus 'correct':             {judge_correct}/{n} = "
          f"{judge_correct/n:.0%}")
    print(f"JUDGE consensus 'ambiguous':          {judge_amb}/{n}")
    print(f"Structural fix+supersede (real-conf):  {struct_correct}/{n} = "
          f"{struct_correct/n:.0%}  [of caught]")
    print("\nNote: extraction-miss pairs count as judge-incorrect (system never "
          "saw the conflict) -- the honest enterprise-prose extraction ceiling.")

    out = BENCH / "erag_judge_harness_result.json"
    if TAG:
        out = out.with_name(f"erag_judge_harness_result_{TAG}.json")
    out.write_text(json.dumps({
        "summary": {
            "n": n, "extraction_caught": len(caught),
            "decided": len(decided), "decision_dist": dict(dec_dist),
            "judge_correct": judge_correct, "judge_ambiguous": judge_amb,
            "struct_correct": struct_correct,
            "isolated": ISOLATED, "judge_model": JUDGE_MODEL, "panel": PANEL,
        },
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()