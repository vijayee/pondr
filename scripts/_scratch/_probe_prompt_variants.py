#!/usr/bin/env python
"""PROBE (not committed): test the user's ranking hypothesis.

Hypothesis: the "Extract AT MOST 6 ... prefer the salient few" instruction in
BONSAI_RELATION_PROMPT is a salience RANKING that pushes has_state out of the
top 6 on decision-framed docs (decides/concerns/involves rank higher). If a
prompt change alone lifts strict has_state catch off 0 on the zero-shot 8B
(no adapter), the prompt -- not the model -- is the binding constraint, and
the fix is cheaper than retraining.

Three variants, identical HTTP shape to the production extractor
(src/encoding/bonsai_relations.py:146): POST /chat/completions with
response_format {json_object}, max_tokens 768, temperature, single user msg.

  V1 ORIGINAL  -- verbatim BONSAI_RELATION_PROMPT (control; should reproduce 0)
  V2 STATE-FIRST -- has_state listed FIRST + the "at most 6 / prefer salient
                   few" line REMOVED (lift the ranking suppression)
  V3 STATE-ONLY -- a dedicated has_state-only extraction prompt (no competing
                   predicates, no cap) -- upper bound for what the model can do
                   on this axis zero-shot

Runs all three on the held-out 16-pair fixture against the live zero-shot 8B,
prints per-variant strict/relaxed catch + negatives FP, writes a JSON result.
Reuses the harness logic from scripts/_probe_bonsai_zeroshot_eval.py.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.encoding.assertion_extractor import _norm_key

FIXTURE = ROOT / "tests" / "fixtures" / "enterpriserag" / "semantic_pairs.json"

# ---- prompt variants -------------------------------------------------------

V1_ORIGINAL = """Extract relationships from this conversation.
Return ONLY valid JSON, no other text.
Extract AT MOST 6 of the most important relations -- prefer the salient few
over exhaustively listing every mention, or the response may truncate.
Relation types:
- explains(Person, Concept): A explains a concept to someone.
- decides(Person, Decision): a person decides or chooses something.
- expresses(Person, Tone): a person expresses a tone/emotion.
- questions(Person, Concept): a person asks about a concept.
- suggests(Person, Concept): a person suggests an idea/option.
- concerns(Episode, Topic): the episode is about a topic.
- involves(Episode, Entity): the episode involves an entity (tool/team/etc).
- contradicts(Statement, Statement): one statement contradicts another.
- has_state(Entity, Value): an ENTITY's current state/value/choice -- the
  subject is a tool/team/ticket/policy/project, NOT a person. NOT a topic.
  Use for explicit "the team chose X", "status: Y", "X is now Z",
  "switched to W", "we use X", "the framework is X"; Value is the literal
  value, not a topic. Extract EVERY such state, even if other relations exist.
- follows_up_on(Episode, Episode): an episode follows up on a prior one.
Conversation:
{text}
Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""

V2_STATE_FIRST = """Extract relationships from this conversation.
Return ONLY valid JSON, no other text.
Extract every relation you can find -- do not cap or rank them; list them all.
Relation types (extract has_state FIRST and ALWAYS -- it is the priority):
- has_state(Entity, Value): an ENTITY's current state/value/choice -- the
  subject is a tool/team/ticket/policy/project, NOT a person. NOT a topic.
  Use for explicit "the team chose X", "status: Y", "X is now Z",
  "switched to W", "we use X", "the framework is X"; Value is the literal
  value, not a topic. ALWAYS extract this for every entity state you see.
- decides(Person, Decision): a person decides or chooses something.
- expresses(Person, Tone): a person expresses a tone/emotion.
- questions(Person, Concept): a person asks about a concept.
- suggests(Person, Concept): a person suggests an idea/option.
- explains(Person, Concept): A explains a concept to someone.
- concerns(Episode, Topic): the episode is about a topic.
- involves(Episode, Entity): the episode involves an entity (tool/team/etc).
- contradicts(Statement, Statement): one statement contradicts another.
- follows_up_on(Episode, Episode): an episode follows up on a prior one.
Conversation:
{text}
Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""

V3_STATE_ONLY = """Extract ONLY the entity-state facts from this conversation.
Return ONLY valid JSON, no other text.
For each entity (tool, team, ticket, policy, project, system, framework,
database, service, etc -- NEVER a person), extract its current state, value,
or choice as a has_state relation. Use the literal value (e.g. "Postgres",
"React", "blue", "v2", "approved"), not a topic. Extract one has_state per
distinct entity-value pair you can find. Do not extract any other relation
type.
Examples:
  "we switched to Postgres"        -> has_state(database, Postgres)
  "the build status is red"         -> has_state(build, red)
  "the team chose React"            -> has_state(framework, React)
  "status: green"                   -> has_state(status, green)
Conversation:
{text}
Return JSON:
{{"relations": [{{"subject": "...", "predicate": "has_state", "object": "..."}}]}}"""

VARIANTS = [
    ("V1_original", V1_ORIGINAL),
    ("V2_state_first", V2_STATE_FIRST),
    ("V3_state_only", V3_STATE_ONLY),
]

# ---- HTTP (mirrors BonsaiRelationExtractor.extract) -----------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _salvage_json(content: str):
    """Find the first balanced {...} object in content and parse it.

    The ternary 8B often emits a reasoning prefix / prose before the JSON, so
    a bare json.loads(content) fails. Scan for the first '{', then walk with
    brace+string tracking to find its matching '}'.
    """
    if not content:
        return None
    start = content.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(content)):
        c = content[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = content[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    # try the next '{'
                    return _salvage_json(content[i + 1:])
    return None


def _extract(endpoint: str, model: str | None, prompt: str, text: str) -> list[dict]:
    import requests
    url = f"{endpoint}/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt.format(text=text)}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 768,
    }
    if model:
        payload["model"] = model
    r = requests.post(url, json=payload, timeout=120.0)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"] or ""
    content = _FENCE_RE.sub(lambda m: m.group(1), content).strip()
    # The ternary model often emits a reasoning prefix / prose before the JSON
    # object (or trailing prose). Salvage the first balanced {...} block.
    obj = _salvage_json(content)
    if obj is None:
        return []
    rels = obj.get("relations") if isinstance(obj, dict) else None
    return rels if isinstance(rels, list) else []


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
                cand = next((n for n in nvals if n not in ovals), None)
                nv = cand if cand is not None else next(iter(nvals))
                colls.append({"entity": e, "old_value": ov, "new_value": nv})
    return colls


def main() -> int:
    endpoint = os.environ.get("BONSAI_EVAL_ENDPOINT") or "http://localhost:8080/v1"
    model = os.environ.get("BONSAI_EVAL_MODEL") or None
    # health
    import requests
    try:
        requests.get(f"{endpoint}/models", timeout=8.0).raise_for_status()
    except Exception as e:
        print(f"ERROR: server not reachable at {endpoint}/models ({e})")
        return 2

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} pairs from {FIXTURE.name}")
    print(f"Endpoint: {endpoint}  Model: {model}\n")

    results = {}
    for name, prompt in VARIANTS:
        print(f"==== {name} ====")
        rows = []
        for p in pairs:
            ob, nb = p["old_doc"]["body"], p["new_doc"]["body"]
            try:
                ro = _extract(endpoint, model, prompt, ob)
            except Exception as e:
                ro = []
            try:
                rn = _extract(endpoint, model, prompt, nb)
            except Exception as e:
                rn = []
            strict = _collisions(_strict_rels(ro), _strict_rels(rn))
            relaxed = _collisions(_relaxed_rels(ro), _relaxed_rels(rn))
            rows.append({
                "id": p["id"], "conflict": p["conflict"],
                "strict": bool(strict), "relaxed": bool(relaxed),
                "n_rels_old": len(ro), "n_rels_new": len(rn),
            })
            print(f"  {p['id']:34s} strict={bool(strict)!s:5s} relax={bool(relaxed)!s:5s} "
                  f"rels(old={len(ro)},new={len(rn)})")
        conf = [r for r in rows if r["conflict"]]
        neg = [r for r in rows if not r["conflict"]]
        s_recall = sum(r["strict"] for r in conf) / len(conf) if conf else 0
        r_recall = sum(r["relaxed"] for r in conf) / len(conf) if conf else 0
        s_fp = sum(r["strict"] for r in neg)
        r_fp = sum(r["relaxed"] for r in neg)
        avg_rels = (sum(r["n_rels_old"] + r["n_rels_new"] for r in rows) /
                    (2 * len(rows))) if rows else 0
        results[name] = {
            "strict_recall": s_recall,
            "relaxed_recall": r_recall,
            "strict_fp_neg": s_fp,
            "relaxed_fp_neg": r_fp,
            "avg_rels": avg_rels,
            "rows": rows,
        }
        print(f"  -> strict has_state catch: {sum(r['strict'] for r in conf)}/{len(conf)} "
              f"({s_recall:.2%})   relaxed: {sum(r['relaxed'] for r in conf)}/{len(conf)} "
              f"({r_recall:.2%})   neg FP strict={s_fp} relaxed={r_fp}   "
              f"avg_rels/doc={avg_rels:.1f}\n")

    out = Path(__file__).parent / "prompt_variants_result.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote -> {out}")
    print("\nHEADLINE: strict has_state catch on conflict pairs (the 8B ships 0/13 zero-shot):")
    for name in results:
        print(f"  {name:18s} {int(results[name]['strict_recall']*len(pairs)):<3d}/"
              f"{len([r for r in pairs if r['conflict']])}  "
              f"(FP neg={results[name]['strict_fp_neg']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())