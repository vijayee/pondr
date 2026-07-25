#!/usr/bin/env python
"""PROBE (not committed): test the isolation-by-class design.

V3 (scripts/_scratch/_probe_prompt_variants.py) proved that a dedicated
has_state-only prompt lifts strict has_state catch 0->12/13 zero-shot, because
has_state no longer loses the salience race against decides/concerns/involves
for the "at most 6" slots. The user's generalization: isolate the prompt for
EACH class of relation we want to extract -- one focused single-predicate
pass per class, then merge. No class ever competes with another.

This probe tests whether isolation GENERALIZES:
  (a) every class still emits when isolated (no class goes silent),
  (b) the merged strict has_state catch holds at ~12/13 (the V3 win transfers),
  (c) the merged graph is at least as rich as the V1 single-pass (more total
      relations, and each class present).

Compares:
  V1   -- the original single-pass BONSAI_RELATION_PROMPT (control)
  ISO  -- N isolated single-predicate passes, merged

Same HTTP shape as the production extractor (POST /chat/completions,
response_format {json_object}, max_tokens 768, temperature 0.3), with the
same JSON-salvage the ternary 8B needs (it emits a reasoning prefix).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.encoding.assertion_extractor import _norm_key

FIXTURE = ROOT / "tests" / "fixtures" / "enterpriserag" / "semantic_pairs.json"

# ---- original single-pass prompt (control) --------------------------------

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

# ---- isolated per-class prompts -------------------------------------------

# Each class: (predicate_name, signature, extraction_directive). The directive
# tells the model exactly what counts as this predicate and to extract every
# instance, emitting the exact predicate string.
CLASSES = [
    ("has_state",
     "has_state(Entity, Value)",
     "an ENTITY's current state/value/choice. The subject is a tool, team, "
     "ticket, policy, project, system, framework, database, or service -- "
     "NEVER a person, NEVER a topic. Use for explicit 'the team chose X', "
     "'status: Y', 'X is now Z', 'switched to W', 'we use X', 'the framework "
     "is X'. Value is the literal value (e.g. Postgres, React, red, v2), not "
     "a topic. Extract one has_state per distinct entity-value pair."),
    ("decides",
     "decides(Person, Decision)",
     "a person decides, chooses, picks, or commits to a course of action or "
     "option. The subject MUST be a person. The object is the decision/choice."),
    ("expresses",
     "expresses(Person, Tone)",
     "a person expresses a tone or emotion (e.g. frustrated, optimistic, "
     "concerned, enthusiastic). Subject is a person; object is the tone."),
    ("questions",
     "questions(Person, Concept)",
     "a person asks a question about a concept or topic. Subject is a person; "
     "object is the concept being asked about."),
    ("suggests",
     "suggests(Person, Concept)",
     "a person suggests, proposes, or recommends an idea or option. Subject "
     "is a person; object is the suggested idea/option."),
    ("explains",
     "explains(Person, Concept)",
     "a person explains a concept to someone. Subject is a person; object is "
     "the concept being explained."),
    ("concerns",
     "concerns(Episode, Topic)",
     "the conversation/episode is about a topic. Subject is the episode (use "
     "'episode'); object is the topic."),
    ("involves",
     "involves(Episode, Entity)",
     "the conversation/episode involves an entity (tool, team, service, "
     "person). Subject is the episode (use 'episode'); object is the entity."),
    ("contradicts",
     "contradicts(Statement, Statement)",
     "one statement in the conversation contradicts another. Subject and "
     "object are the two contradicting statements (short quotes or summaries)."),
    ("follows_up_on",
     "follows_up_on(Episode, Episode)",
     "the conversation follows up on a prior episode. Both subject and object "
     "are episodes."),
]

_ISO_TEMPLATE = """Extract ONLY __SIG__ relations from this conversation.
Return ONLY valid JSON, no other text.
A __PRED__ relation is: __DIRECTIVE__
Extract EVERY __PRED__ relation you can find in the conversation. Do NOT
extract any other relation type. Emit the predicate as the exact string
"__PRED__".
Conversation:
{text}
Return JSON:
{{"relations": [{{"subject": "...", "predicate": "__PRED__", "object": "..."}}]}}"""


def iso_prompt(pred: str, sig: str, directive: str) -> str:
    return (_ISO_TEMPLATE
            .replace("__SIG__", sig)
            .replace("__PRED__", pred)
            .replace("__DIRECTIVE__", directive))

# ---- HTTP + salvage (mirrors production BonsaiRelationExtractor.extract) ---

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _salvage_json(content: str):
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
    import requests
    try:
        requests.get(f"{endpoint}/models", timeout=8.0).raise_for_status()
    except Exception as e:
        print(f"ERROR: server not reachable at {endpoint}/models ({e})")
        return 2

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} pairs from {FIXTURE.name}")
    print(f"Endpoint: {endpoint}  Model: {model}")
    print(f"Isolated classes: {[c[0] for c in CLASSES]}\n")

    iso_prompts = [(pred, iso_prompt(pred, sig, d)) for pred, sig, d in CLASSES]

    # ---- V1 single-pass (control) ----
    print("==== V1 original single-pass ====")
    v1_rows = []
    for p in pairs:
        ob, nb = p["old_doc"]["body"], p["new_doc"]["body"]
        ro = _extract(endpoint, model, V1_ORIGINAL, ob)
        rn = _extract(endpoint, model, V1_ORIGINAL, nb)
        strict = _collisions(_strict_rels(ro), _strict_rels(rn))
        relaxed = _collisions(_relaxed_rels(ro), _relaxed_rels(rn))
        v1_rows.append({"id": p["id"], "conflict": p["conflict"],
                        "strict": bool(strict), "relaxed": bool(relaxed),
                        "n_old": len(ro), "n_new": len(rn),
                        "preds_old": sorted({str(r.get("predicate","")) for r in ro if isinstance(r,dict)}),
                        "preds_new": sorted({str(r.get("predicate","")) for r in rn if isinstance(r,dict)})})
    conf = [r for r in v1_rows if r["conflict"]]
    neg = [r for r in v1_rows if not r["conflict"]]
    v1_strict = sum(r["strict"] for r in conf)
    v1_relax = sum(r["relaxed"] for r in conf)
    v1_fp = sum(r["strict"] for r in neg)
    v1_avg = (sum(r["n_old"] + r["n_new"] for r in v1_rows) / (2 * len(v1_rows)))
    v1_preds = sorted({p for r in v1_rows for p in r["preds_old"] + r["preds_new"]})
    print(f"  strict has_state: {v1_strict}/{len(conf)}  relaxed: {v1_relax}/{len(conf)}  "
          f"neg FP={v1_fp}  avg_rels/doc={v1_avg:.1f}")
    print(f"  predicates emitted (union): {v1_preds}\n")

    # ---- ISO isolated per-class, merged ----
    print("==== ISO isolated per-class (merged) ====")
    iso_rows = []
    # per-class emission tally across all docs
    class_emit = {pred: 0 for pred, _, _ in CLASSES}
    for p in pairs:
        ob, nb = p["old_doc"]["body"], p["new_doc"]["body"]
        merged_old, merged_new = [], []
        for pred, prompt in iso_prompts:
            ro = _extract(endpoint, model, prompt, ob)
            rn = _extract(endpoint, model, prompt, nb)
            # force-normalize predicate to the exact class name (the prompt asks
            # for it, but the model sometimes paraphrases the predicate string)
            for r in ro:
                if isinstance(r, dict):
                    r["predicate"] = pred
            for r in rn:
                if isinstance(r, dict):
                    r["predicate"] = pred
            class_emit[pred] += sum(1 for r in ro if isinstance(r, dict)) + \
                               sum(1 for r in rn if isinstance(r, dict))
            merged_old += ro
            merged_new += rn
        strict = _collisions(_strict_rels(merged_old), _strict_rels(merged_new))
        relaxed = _collisions(_relaxed_rels(merged_old), _relaxed_rels(merged_new))
        iso_rows.append({"id": p["id"], "conflict": p["conflict"],
                         "strict": bool(strict), "relaxed": bool(relaxed),
                         "n_old": len(merged_old), "n_new": len(merged_new)})
        print(f"  {p['id']:34s} strict={bool(strict)!s:5s} relax={bool(relaxed)!s:5s} "
              f"rels(old={len(merged_old)},new={len(merged_new)})")
    conf = [r for r in iso_rows if r["conflict"]]
    neg = [r for r in iso_rows if not r["conflict"]]
    iso_strict = sum(r["strict"] for r in conf)
    iso_relax = sum(r["relaxed"] for r in conf)
    iso_fp = sum(r["strict"] for r in neg)
    iso_avg = (sum(r["n_old"] + r["n_new"] for r in iso_rows) / (2 * len(iso_rows)))

    print(f"\n  strict has_state: {iso_strict}/{len(conf)}  relaxed: {iso_relax}/{len(conf)}  "
          f"neg FP={iso_fp}  avg_rels/doc={iso_avg:.1f}")
    print(f"  per-class emission (rels across all 32 docs):")
    for pred, _, _ in CLASSES:
        print(f"    {pred:14s} {class_emit[pred]:3d}")
    silent = [pred for pred, n in class_emit.items() if n == 0]
    print(f"  silent classes (0 emitted): {silent if silent else 'none'}")

    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"{'metric':40s} {'V1 single':>10s} {'ISO merged':>10s}")
    print(f"{'strict has_state catch (conflicts)':40s} {v1_strict:>7d}/13 {iso_strict:>7d}/13")
    print(f"{'relaxed catch (conflicts)':40s} {v1_relax:>7d}/13 {iso_relax:>7d}/13")
    print(f"{'neg strict FP':40s} {v1_fp:>10d} {iso_fp:>10d}")
    print(f"{'avg rels/doc':40s} {v1_avg:>10.1f} {iso_avg:>10.1f}")
    print(f"{'distinct predicates emitted':40s} {len(v1_preds):>10d} {len([p for p,n in class_emit.items() if n>0]):>10d}")

    out = Path(__file__).parent / "isolate_classes_result.json"
    out.write_text(json.dumps({
        "v1": {"strict": v1_strict, "relaxed": v1_relax, "fp": v1_fp,
               "avg": v1_avg, "preds": v1_preds, "rows": v1_rows},
        "iso": {"strict": iso_strict, "relaxed": iso_relax, "fp": iso_fp,
                "avg": iso_avg, "class_emit": class_emit, "rows": iso_rows},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())