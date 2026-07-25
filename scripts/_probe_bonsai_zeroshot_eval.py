#!/usr/bin/env python
"""PROBE (not committed): Bonsai zero-shot contradiction eval -- Phase 3c fine-tune hinge.

Three measurements over tests/fixtures/enterpriserag/semantic_pairs.json (16 pairs:
4 field controls, 9 paraphrased, 3 negatives):

  (1) EXTRACTION -- does Bonsai catch the conflict, and under what predicate?
      - det:        deterministic normalizer (extract_state_assertions) catch-rate
      - bonsai_strict:  Bonsai relations filtered to {has_state, state} (what the
                    PRODUCTION encoder lifts -- extract_state_assertions only
                    lifts these). Measures the shipped Bonsai contribution.
      - bonsai_relaxed: Bonsai relations with ANY predicate (subject=entity,
                    object=value). Measures Bonsai's latent semantic capability
                    ignoring schema adherence. The gap (relaxed - strict) is the
                    schema-adherence gap a LoRA fine-tune (or a relaxed filter)
                    would close -- the opportunity.

  (2) ADJUDICATION -- decide_contradiction is INDEPENDENT of the extraction
      schema (it takes a flag + state_values + provenance and returns a
      decision). So we adjudicate ALL 13 conflict pairs from GROUND-TRUTH
      values (fixture old/new), not just the Bonsai-caught subset. Correct =
      decision=="fix" and action contains "supersede_assertion". This is the
      fine-tune hinge for the DECIDER.

  (3) NEGATIVES -- feed each negative pair's values as a flag and check Bonsai
      does NOT auto-fix (complementary -> dismiss/ask; same-value / different-
      entity are not real conflicts).

Collision (catch) = shared normalized entity with DIFFERENT values across
old+new (the proxy for _detect_contradictory_state firing).

Requires the Bonsai 8B server at localhost:8080/v1 (pre-warmed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoding.assertion_extractor import extract_state_assertions, _norm_key
from src.encoding.bonsai_relations import BonsaiRelationExtractor
from src.gnn.bonsai_decider import BonsaiDecider

FIXTURE = ROOT / "tests" / "fixtures" / "enterpriserag" / "semantic_pairs.json"


def _norm(v: str) -> str:
    return (v or "").strip().lower()


def _strict_rels(rels):
    """Only has_state/state predicates (what the production encoder lifts)."""
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
    """ANY predicate: subject=entity, object=value (latent semantic capability)."""
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


def main():
    # Env overrides so the same harness can target any OpenAI-compatible server
    # (e.g. Ollama's /v1) without touching production config. Default = the
    # Bonsai server via config (localhost:8080/v1).
    endpoint = os.environ.get("BONSAI_EVAL_ENDPOINT") or None
    model = os.environ.get("BONSAI_EVAL_MODEL") or None
    dec = BonsaiDecider(model=model, endpoint=endpoint)
    if not dec.health_check(timeout=8.0):
        ep = dec.endpoint
        print(f"ERROR: server not reachable at {ep}/models -- start it first.")
        sys.exit(2)
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} pairs from {FIXTURE.name}")
    print(f"Endpoint: {dec.endpoint}  Model: {dec.model}\n")

    ext = BonsaiRelationExtractor(model=model, endpoint=endpoint, timeout=90.0)
    dec = BonsaiDecider(model=model, endpoint=endpoint, timeout=90.0, max_tokens=768)

    rows = []
    for p in pairs:
        pid = p["id"]
        cat = p["category"]
        conflict = p["conflict"]
        ob, nb = p["old_doc"]["body"], p["new_doc"]["body"]

        det_coll = _collisions(extract_state_assertions(ob), extract_state_assertions(nb))

        try:
            rels_old = ext.extract(ob)
        except Exception as e:
            rels_old = []; print(f"  [{pid}] extract(old) error: {e}")
        try:
            rels_new = ext.extract(nb)
        except Exception as e:
            rels_new = []; print(f"  [{pid}] extract(new) error: {e}")

        strict_coll = _collisions(_strict_rels(rels_old), _strict_rels(rels_new))
        relaxed_coll = _collisions(_relaxed_rels(rels_old), _relaxed_rels(rels_new))

        # adjudication from GROUND-TRUTH values (independent of extraction)
        adjudication = None; adjudication_correct = None
        if conflict:
            state_values = [
                {"value": p["old_value"], "asserted_by": p["old_doc"]["source_path"], "asserted_at": "2026-07-14"},
                {"value": p["new_value"], "asserted_by": p["new_doc"]["source_path"], "asserted_at": "2026-07-15"},
            ]
            flag = {"node": p["entity_hint"], "type": "contradictory_state", "evidence": state_values}
            adjudication = dec.decide_contradiction(flag, {"state_values": state_values})
            if adjudication is not None:
                ok = (adjudication.get("decision") == "fix"
                      and "supersede_assertion" in adjudication.get("action", ""))
                adjudication_correct = ok
        else:
            # negative: feed its values; a fix+supersede_assertion would be a FALSE fix
            state_values = [
                {"value": p["old_value"], "asserted_by": p["old_doc"]["source_path"], "asserted_at": "2026-07-14"},
                {"value": p["new_value"], "asserted_by": p["new_doc"]["source_path"], "asserted_at": "2026-07-15"},
            ]
            flag = {"node": p["entity_hint"], "type": "contradictory_state", "evidence": state_values}
            adjudication = dec.decide_contradiction(flag, {"state_values": state_values})
            if adjudication is not None:
                # correct = NOT a fix+supersede_assertion (no auto-tombstone on a non-conflict)
                adjudication_correct = not (adjudication.get("decision") == "fix"
                                            and "supersede_assertion" in adjudication.get("action", ""))

        rows.append({
            "id": pid, "category": cat, "conflict": conflict,
            "det_catch": bool(det_coll), "bonsai_strict_catch": bool(strict_coll),
            "bonsai_relaxed_catch": bool(relaxed_coll),
            "adjudication": adjudication, "adjudication_correct": adjudication_correct,
            "correct_decision": p.get("correct_decision"),
        })
        adj_s = "-"
        if adjudication is None: adj_s = "None(FAIL)"
        else: adj_s = f"{adjudication.get('decision')}/{adjudication.get('action','')[:36]}"
        print(f"{pid:34s} det={bool(det_coll)!s:5s} bStrict={bool(strict_coll)!s:5s} bRelax={bool(relaxed_coll)!s:5s} adj={adj_s}")

    # ---- summary ----
    conf = [r for r in rows if r["conflict"]]
    neg = [r for r in rows if not r["conflict"]]
    def rate(rs, k): return sum(r[k] for r in rs) / len(rs) if rs else 0.0
    det_r = rate(conf, "det_catch")
    bstr_r = rate(conf, "bonsai_strict_catch")
    brel_r = rate(conf, "bonsai_relaxed_catch")
    adjudged = [r for r in conf if r["adjudication"] is not None]
    adjud_none = [r for r in conf if r["adjudication"] is None]
    adjud_correct = sum(1 for r in adjudged if r["adjudication_correct"])
    neg_adjudged = [r for r in neg if r["adjudication"] is not None]
    neg_falsefix = sum(1 for r in neg_adjudged if not r["adjudication_correct"])

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Conflict pairs: {len(conf)}   Negative pairs: {len(neg)}")
    print(f"DETERMINISTIC catch (recall):        {det_r:.2%}  ({sum(r['det_catch'] for r in conf)}/{len(conf)})")
    print(f"BONSAI strict has_state catch (recall): {bstr_r:.2%}  ({sum(r['bonsai_strict_catch'] for r in conf)}/{len(conf)})  [what production lifts]")
    print(f"BONSAI relaxed any-predicate catch:     {brel_r:.2%}  ({sum(r['bonsai_relaxed_catch'] for r in conf)}/{len(conf)})  [latent capability]")
    print(f"  schema-adherence gap (relaxed - strict): {(brel_r - bstr_r):+.2%}  <- opportunity a fine-tune / relaxed filter would close")
    print(f"BONSAI strict false-positives on negatives: {sum(r['bonsai_strict_catch'] for r in neg)}/{len(neg)}")
    print(f"BONSAI relaxed false-positives on negatives: {sum(r['bonsai_relaxed_catch'] for r in neg)}/{len(neg)}")
    print()
    print(f"ADJUDICATION (zero-shot decide_contradiction, ground-truth conflicts):")
    print(f"  decided: {len(adjudged)} / {len(conf)}   returned-None(fail): {len(adjud_none)}")
    print(f"  correct (fix + supersede_assertion): {adjud_correct}/{len(adjudged)} = {(adjud_correct/len(adjudged) if adjudged else 0):.2%}")
    print(f"  NEGATIVES -- false auto-fix (fix+supersede on a non-conflict): {neg_falsefix}/{len(neg_adjudged)}")

    out = ROOT / "scripts" / "_scratch" / "bonsai_zeroshot_eval_result.json"
    tag = os.environ.get("BONSAI_EVAL_TAG", "").strip()
    if tag:
        out = out.with_name(f"bonsai_zeroshot_eval_result_{tag}.json")
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