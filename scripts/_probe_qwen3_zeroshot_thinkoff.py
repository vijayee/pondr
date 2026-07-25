#!/usr/bin/env python
"""PROBE (not committed): same 16-pair eval as _probe_bonsai_zeroshot_eval.py,
but against Ollama qwen3:8b via /api/chat with think=False.

Why this exists: the production extractor/decider POST to /v1/chat/completions,
which does NOT pass Ollama's `think` flag, and qwen3:8b defaults to thinking-on.
With max_tokens=768 the reasoning eats the whole budget -> finish_reason=length,
empty content (reproduced deterministically on P7). The Bonsai 8B/27B runs had
thinking OFF (--reasoning-budget 0), so the like-for-like comparison requires
thinking off here. Ollama's /v1 ignores `think:false` (verified), so we drive
/api/chat directly and reuse the PRODUCTION parsers (_parse_relations, the
decider's _parse_json_object) + the production prompts for fidelity.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoding.assertion_extractor import extract_state_assertions, _norm_key
from src.encoding.bonsai_relations import BonsaiRelationExtractor, BONSAI_RELATION_PROMPT
from src.gnn.bonsai_decider import _FENCE_RE
from src.training.prompts import bonsai_contradiction_decision_prompt

FIXTURE = ROOT / "tests" / "fixtures" / "enterpriserag" / "semantic_pairs.json"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = os.environ.get("BONSAI_EVAL_MODEL", "qwen3:8b")


def _chat(prompt: str) -> str:
    """One /api/chat call, think=False, format=json. Returns raw content or ''."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 768},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return d.get("message", {}).get("content") or ""
    except Exception as e:
        print(f"    /api/chat error: {e}")
        return ""


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


def _relaxed_rels(rels):
    out = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        if not {"subject", "predicate", "object"} <= r.keys():
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
        if e in new_by_e:
            nvals = new_by_e[e]
            if ovals != nvals:
                ov = next(iter(ovals))
                nv = next(iter(n for n in nvals if n not in ovals) or iter(nvals))
                colls.append({"entity": e, "old_value": ov, "new_value": nv})
    return colls


def _extract(text: str):
    """Mirror BonsaiRelationExtractor.extract via /api/chat think=False."""
    prompt = BONSAI_RELATION_PROMPT.format(text=text)
    content = _chat(prompt)
    return BonsaiRelationExtractor._parse_relations(content) if content else []


def _decide(flag, retrieved_context):
    """Mirror BonsaiDecider.decide_contradiction via /api/chat think=False."""
    flagged_entity = str(flag.get("node", ""))
    prompt = bonsai_contradiction_decision_prompt(flagged_entity, retrieved_context)
    content = _chat(prompt)
    if not content:
        return None
    # reuse the decider's parser
    body = content.strip()
    fence = _FENCE_RE.match(body)
    if fence:
        body = fence.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(body[start:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(data, dict) or "decision" not in data:
        return None
    decision = str(data.get("decision", "")).strip()
    if decision not in ("fix", "ask_user", "dismiss"):
        return None
    return {
        "decision": decision,
        "action": str(data.get("action", ""))[:1000],
        "reasoning": str(data.get("reasoning", ""))[:1000],
    }


def main():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} pairs from {FIXTURE.name}")
    print(f"Endpoint: {OLLAMA}  Model: {MODEL}  think=False\n")

    rows = []
    for p in pairs:
        pid = p["id"]
        conflict = p["conflict"]
        ob, nb = p["old_doc"]["body"], p["new_doc"]["body"]
        det_coll = _collisions(extract_state_assertions(ob), extract_state_assertions(nb))
        rels_old = _extract(ob)
        rels_new = _extract(nb)
        strict_coll = _collisions(_strict_rels(rels_old), _strict_rels(rels_new))
        relaxed_coll = _collisions(_relaxed_rels(rels_old), _relaxed_rels(rels_new))

        adjudication = None
        adjudication_correct = None
        state_values = [
            {"value": p["old_value"], "asserted_by": p["old_doc"]["source_path"], "asserted_at": "2026-07-14"},
            {"value": p["new_value"], "asserted_by": p["new_doc"]["source_path"], "asserted_at": "2026-07-15"},
        ]
        flag = {"node": p["entity_hint"], "type": "contradictory_state", "evidence": state_values}
        adjudication = _decide(flag, {"state_values": state_values})
        if adjudication is not None:
            if conflict:
                ok = (adjudication.get("decision") == "fix"
                      and "supersede_assertion" in adjudication.get("action", ""))
            else:
                ok = not (adjudication.get("decision") == "fix"
                          and "supersede_assertion" in adjudication.get("action", ""))
            adjudication_correct = ok

        rows.append({
            "id": pid, "conflict": conflict,
            "det_catch": bool(det_coll), "bonsai_strict_catch": bool(strict_coll),
            "bonsai_relaxed_catch": bool(relaxed_coll),
            "adjudication": adjudication, "adjudication_correct": adjudication_correct,
        })
        adj_s = "None(FAIL)" if adjudication is None else f"{adjudication.get('decision')}/{adjudication.get('action','')[:36]}"
        print(f"{pid:34s} det={bool(det_coll)!s:5s} bStrict={bool(strict_coll)!s:5s} bRelax={bool(relaxed_coll)!s:5s} adj={adj_s}")

    conf = [r for r in rows if r["conflict"]]
    neg = [r for r in rows if not r["conflict"]]
    det_r = sum(r["det_catch"] for r in conf) / len(conf)
    bstr_r = sum(r["bonsai_strict_catch"] for r in conf) / len(conf)
    brel_r = sum(r["bonsai_relaxed_catch"] for r in conf) / len(conf)
    adjudged = [r for r in conf if r["adjudication"] is not None]
    adjud_none = [r for r in conf if r["adjudication"] is None]
    adjud_correct = sum(1 for r in adjudged if r["adjudication_correct"])
    neg_adjudged = [r for r in neg if r["adjudication"] is not None]
    neg_falsefix = sum(1 for r in neg_adjudged if not r["adjudication_correct"])

    print("\n" + "=" * 78)
    print("SUMMARY (qwen3:8b, think=False, /api/chat)")
    print("=" * 78)
    print(f"Conflict pairs: {len(conf)}   Negative pairs: {len(neg)}")
    print(f"DETERMINISTIC catch (recall):            {det_r:.2%}  ({sum(r['det_catch'] for r in conf)}/{len(conf)})")
    print(f"BONSAI strict has_state catch (recall):   {bstr_r:.2%}  ({sum(r['bonsai_strict_catch'] for r in conf)}/{len(conf)})  [what production lifts]")
    print(f"BONSAI relaxed any-predicate catch:       {brel_r:.2%}  ({sum(r['bonsai_relaxed_catch'] for r in conf)}/{len(conf)})  [latent capability]")
    print(f"  schema-adherence gap (relaxed - strict): {(brel_r - bstr_r):+.2%}")
    print(f"BONSAI strict false-positives on negatives: {sum(r['bonsai_strict_catch'] for r in neg)}/{len(neg)}")
    print(f"BONSAI relaxed false-positives on negatives: {sum(r['bonsai_relaxed_catch'] for r in neg)}/{len(neg)}")
    print(f"\nADJUDICATION (zero-shot decide_contradiction, ground-truth conflicts):")
    print(f"  decided: {len(adjudged)} / {len(conf)}   returned-None(fail): {len(adjud_none)}")
    print(f"  correct (fix + supersede_assertion): {adjud_correct}/{len(adjudged)} = {(adjud_correct/len(adjudged) if adjudged else 0):.2%}")
    print(f"  NEGATIVES -- false auto-fix (fix+supersede on a non-conflict): {neg_falsefix}/{len(neg_adjudged)}")

    out = ROOT / "scripts" / "_scratch" / "bonsai_zeroshot_eval_result_qwen3_8b_thinkoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": {
            "n_conflict": len(conf), "n_negative": len(neg),
            "det_recall": det_r, "bonsai_strict_recall": bstr_r, "bonsai_relaxed_recall": brel_r,
            "schema_adherence_gap": brel_r - bstr_r,
            "bonsai_strict_fp": sum(r["bonsai_strict_catch"] for r in neg),
            "bonsai_relaxed_fp": sum(r["bonsai_relaxed_catch"] for r in neg),
            "adjudication_decided": len(adjudged), "adjudication_none": len(adjud_none),
            "adjudication_correct_rate": (adjud_correct / len(adjudged) if adjudged else 0),
            "negatives_false_fix": neg_falsefix,
        },
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote result -> {out}")


if __name__ == "__main__":
    main()