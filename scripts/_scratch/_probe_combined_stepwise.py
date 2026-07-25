#!/usr/bin/env python
"""PROBE (not committed): single-request stepwise prompt vs 10-pass isolation.

Two questions from the user:
  (1) Can we batch the 10 isolated per-class prompts as STEPS in ONE prompt,
      instead of 10 round-trips? (chatbot latency: 10 requests/doc is a lot)
  (2) Remove any number cap (the "at most 6 / prefer salient few" directive)
      and see the quality of the output.

Three strategies, all on the zero-shot 8B (no adapter), timed per doc:

  V1       -- original single-pass BONSAI_RELATION_PROMPT (control; 1 request)
  ISO      -- 10 isolated per-class passes, merged (10 requests; proven 11/13)
  V5_STEP  -- ONE request: a stepwise prompt that walks each class as an
              explicit step, NO cap anywhere, returns one merged JSON

Reports per-strategy: strict has_state catch, relaxed catch, neg FP,
avg rels/doc, per-class emission, AND per-doc wall time (avg/p50/p95) so we
can see if 10 requests is too slow for a chatbot and whether the stepwise
single request holds the quality.

Same HTTP shape as production (POST /chat/completions,
response_format {json_object}, max_tokens 768 -- but V5_STEP uses 1536 since
it must emit all 10 classes in one go; noted in output).
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

# ---- V1 original (control) -----------------------------------------------

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

# ---- isolated per-class (the proven 10-pass design) -----------------------

CLASSES = [
    ("has_state", "has_state(Entity, Value)",
     "an ENTITY's current state/value/choice. Subject is a tool, team, ticket, "
     "policy, project, system, framework, database, or service -- NEVER a "
     "person, NEVER a topic. Use for 'the team chose X', 'status: Y', 'X is "
     "now Z', 'switched to W', 'we use X'. Value is the literal value, not a "
     "topic. One has_state per distinct entity-value pair."),
    ("decides", "decides(Person, Decision)",
     "a person decides, chooses, picks, or commits to a course of action. "
     "Subject MUST be a person; object is the decision/choice."),
    ("expresses", "expresses(Person, Tone)",
     "a person expresses a tone/emotion (frustrated, optimistic, concerned). "
     "Subject is a person; object is the tone."),
    ("questions", "questions(Person, Concept)",
     "a person asks a question about a concept. Subject is a person; object "
     "is the concept being asked about."),
    ("suggests", "suggests(Person, Concept)",
     "a person suggests, proposes, or recommends an idea/option. Subject is a "
     "person; object is the suggested idea/option."),
    ("explains", "explains(Person, Concept)",
     "a person explains a concept to someone. Subject is a person; object is "
     "the concept being explained."),
    ("concerns", "concerns(Episode, Topic)",
     "the conversation/episode is about a topic. Subject is 'episode'; object "
     "is the topic."),
    ("involves", "involves(Episode, Entity)",
     "the conversation/episode involves an entity (tool, team, service, "
     "person). Subject is 'episode'; object is the entity."),
    ("contradicts", "contradicts(Statement, Statement)",
     "one statement contradicts another. Subject and object are the two "
     "contradicting statements (short quotes or summaries)."),
    ("follows_up_on", "follows_up_on(Episode, Episode)",
     "the conversation follows up on a prior episode. Both subject and object "
     "are episodes."),
]

_ISO_TEMPLATE = """Extract ONLY __SIG__ relations from this conversation.
Return ONLY valid JSON, no other text.
A __PRED__ relation is: __DIRECTIVE__
Extract EVERY __PRED__ relation you can find. Do NOT extract any other type.
Emit the predicate as the exact string "__PRED__".
Conversation:
{text}
Return JSON:
{{"relations": [{{"subject": "...", "predicate": "__PRED__", "object": "..."}}]}}"""


def iso_prompt(pred, sig, d):
    return (_ISO_TEMPLATE.replace("__SIG__", sig)
            .replace("__PRED__", pred).replace("__DIRECTIVE__", d))


# ---- V5 stepwise single-request (the new design under test) ---------------
# One request: walk each class as an explicit step, no cap, return one JSON.

def build_stepwise_prompt() -> str:
    steps = []
    for i, (pred, sig, d) in enumerate(CLASSES, 1):
        steps.append(f"Step {i} -- {sig}: {d} Extract every {pred} you find.")
    steps_block = "\n".join(steps)
    return (
        "Extract relations from this conversation by working through EVERY "
        "step below. Return ONLY valid JSON at the end, no other text.\n"
        "Do NOT cap the number of relations. Do NOT prefer the salient few. "
        "Extract EVERY relation you can find across ALL steps -- there is no "
        "limit. Emit each predicate as the EXACT string given.\n\n"
        f"{steps_block}\n\n"
        "Conversation:\n{text}\n\n"
        "After working through all steps, return ALL extracted relations as "
        "one JSON object (relations from every step merged together):\n"
        "{{\"relations\": [{{\"subject\": \"...\", \"predicate\": \"...\", "
        "\"object\": \"...\"}}]}}"
    )


# ---- HTTP + salvage (mirrors production) -----------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _salvage_json(content: str):
    if not content:
        return None
    start = content.find("{")
    if start < 0:
        return None
    depth = 0; in_str = False; esc = False
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
                try:
                    return json.loads(content[start:i + 1])
                except Exception:
                    return _salvage_json(content[i + 1:])
    return None


def _extract(endpoint, model, prompt, text, max_tokens=768):
    import requests
    url = f"{endpoint}/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt.format(text=text)}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=180.0)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    content = r.json()["choices"][0]["message"]["content"] or ""
    content = _FENCE_RE.sub(lambda m: m.group(1), content).strip()
    obj = _salvage_json(content)
    rels = obj.get("relations") if isinstance(obj, dict) else None
    return (rels if isinstance(rels, list) else []), dt


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


def _collisions(old_a, new_a):
    old_e = {}
    for a in old_a:
        old_e.setdefault(a["entity"], set()).add(_norm(a["value"]))
    new_e = {}
    for a in new_a:
        new_e.setdefault(a["entity"], set()).add(_norm(a["value"]))
    colls = []
    for e, ov in old_e.items():
        if e in new_e:
            nv = new_e[e]
            if ov != nv:
                colls.append({"entity": e, "old": next(iter(ov)),
                              "new": next(iter(n for n in nv if n not in ov) or iter(nv))})
    return colls


def _force_pred(rels, pred):
    for r in rels:
        if isinstance(r, dict):
            r["predicate"] = pred


def run_strategy(name, endpoint, model, pairs, extract_fn, max_tokens=768):
    """extract_fn(endpoint, model, body) -> (rels, dt). Returns rows+timings."""
    print(f"\n==== {name} (max_tokens={max_tokens}) ====")
    rows = []
    times = []
    class_emit = {pred: 0 for pred, _, _ in CLASSES}
    for p in pairs:
        ob, nb = p["old_doc"]["body"], p["new_doc"]["body"]
        ro, t1 = extract_fn(endpoint, model, ob)
        rn, t2 = extract_fn(endpoint, model, nb)
        times += [t1, t2]
        strict = _collisions(_strict_rels(ro), _strict_rels(rn))
        relaxed = _collisions(_relaxed_rels(ro), _relaxed_rels(rn))
        rows.append({"id": p["id"], "conflict": p["conflict"],
                     "strict": bool(strict), "relaxed": bool(relaxed),
                     "n_old": len(ro), "n_new": len(rn)})
        for r in ro + rn:
            if isinstance(r, dict):
                pr = str(r.get("predicate", "")).lower()
                if pr in class_emit:
                    class_emit[pr] += 1
        print(f"  {p['id']:34s} strict={bool(strict)!s:5s} relax={bool(relaxed)!s:5s} "
              f"rels(old={len(ro)},new={len(rn)})")
    return rows, times, class_emit


def summarize(name, rows, times, class_emit, max_tokens):
    conf = [r for r in rows if r["conflict"]]
    neg = [r for r in rows if not r["conflict"]]
    strict = sum(r["strict"] for r in conf)
    relax = sum(r["relaxed"] for r in conf)
    fp = sum(r["strict"] for r in neg)
    avg_rels = (sum(r["n_old"] + r["n_new"] for r in rows) / (2 * len(rows))) if rows else 0
    times_s = sorted(times)
    n = len(times_s)
    p50 = times_s[n // 2] if n else 0
    p95 = times_s[min(int(n * 0.95), n - 1)] if n else 0
    print(f"  -> strict has_state: {strict}/{len(conf)}  relaxed: {relax}/{len(conf)}  "
          f"neg FP={fp}  avg_rels/doc={avg_rels:.1f}")
    print(f"  -> per-doc wall time (s): avg={sum(times)/n:.2f}  p50={p50:.2f}  p95={p95:.2f}  "
          f"max={max(times):.2f}")
    print(f"  -> per-class emission: " +
          " ".join(f"{p}={class_emit[p]}" for p, _, _ in CLASSES))
    return {"strict": strict, "relaxed": relax, "fp": fp, "avg_rels": avg_rels,
            "t_avg": sum(times) / n, "t_p50": p50, "t_p95": p95, "t_max": max(times),
            "class_emit": class_emit, "max_tokens": max_tokens, "rows": rows}


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

    stepwise = build_stepwise_prompt()
    iso_prompts = {pred: iso_prompt(pred, sig, d) for pred, sig, d in CLASSES}

    # V1: 1 request/doc, max_tokens 768
    def v1_fn(ep, m, body):
        return _extract(ep, m, V1_ORIGINAL, body, max_tokens=768)
    v1_rows, v1_t, v1_emit = run_strategy("V1 original", endpoint, model, pairs, v1_fn, 768)
    v1 = summarize("V1 original", v1_rows, v1_t, v1_emit, 768)

    # ISO: 10 requests/doc, max_tokens 768 each
    def iso_fn(ep, m, body):
        t_all = 0.0
        merged = []
        for pred, _, _ in CLASSES:
            rels, dt = _extract(ep, m, iso_prompts[pred], body, max_tokens=768)
            t_all += dt
            _force_pred(rels, pred)
            merged += rels
        return merged, t_all
    iso_rows, iso_t, iso_emit = run_strategy("ISO 10-pass", endpoint, model, pairs, iso_fn, 768)
    iso = summarize("ISO 10-pass", iso_rows, iso_t, iso_emit, 768)

    # V5 stepwise: 1 request/doc, max_tokens 1536 (needs room for all 10 classes)
    def v5_fn(ep, m, body):
        return _extract(ep, m, stepwise, body, max_tokens=1536)
    v5_rows, v5_t, v5_emit = run_strategy("V5 stepwise (1 req)", endpoint, model, pairs, v5_fn, 1536)
    v5 = summarize("V5 stepwise", v5_rows, v5_t, v5_emit, 1536)

    # ---- comparison table ----
    print("\n" + "=" * 86)
    print("COMPARISON (zero-shot 8B, no adapter)")
    print("=" * 86)
    print(f"{'metric':36s} {'V1(1req)':>12s} {'ISO(10req)':>12s} {'V5(1req)':>12s}")
    print(f"{'strict has_state catch (/13)':36s} {v1['strict']:>10d}/13 {iso['strict']:>10d}/13 {v5['strict']:>10d}/13")
    print(f"{'relaxed catch (/13)':36s} {v1['relaxed']:>10d}/13 {iso['relaxed']:>10d}/13 {v5['relaxed']:>10d}/13")
    print(f"{'neg strict FP':36s} {v1['fp']:>12d} {iso['fp']:>12d} {v5['fp']:>12d}")
    print(f"{'avg rels/doc':36s} {v1['avg_rels']:>12.1f} {iso['avg_rels']:>12.1f} {v5['avg_rels']:>12.1f}")
    print(f"{'per-doc time avg (s)':36s} {v1['t_avg']:>12.2f} {iso['t_avg']:>12.2f} {v5['t_avg']:>12.2f}")
    print(f"{'per-doc time p95 (s)':36s} {v1['t_p95']:>12.2f} {iso['t_p95']:>12.2f} {v5['t_p95']:>12.2f}")
    print(f"{'per-doc time max (s)':36s} {v1['t_max']:>12.2f} {iso['t_max']:>12.2f} {v5['t_max']:>12.2f}")
    print(f"{'requests per doc':36s} {'1':>12s} {'10':>12s} {'1':>12s}")
    print(f"{'max_tokens':36s} {768:>12d} {768:>12d} {1536:>12d}")

    out = Path(__file__).parent / "combined_stepwise_result.json"
    out.write_text(json.dumps({"v1": v1, "iso": iso, "v5_stepwise": v5},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())