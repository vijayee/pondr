"""S-P-O extraction audit over the EXISTING Phase-0 corpus.

Tests the hypothesis "intent is propositionally structured (subject, attitude,
embedded predicate-relation, object)" by asking the live model to decompose
harvested expectation statements + intent aims, and measuring the LEAK:
residue = statements that do not fully decompose (state-intents, avoid-intents,
underspecified intents, other).

Caveats reported by construction:
  * instrument-induced bias: our scaffold prompts intent statements in
    propositional shape, so high decomposition success may partly reflect the
    instrument, not intrinsic intent structure;
  * synthetic statements are TEMPLATES -- they act as a control: an auditor
    that struggles on templates is disqualified.

Uses the local Ollama relay only. Read-only over the ledgers.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

SCRATCH = os.path.join("data", "_scratch", "icm")
LIVE_DBS = {"agent_severity": os.path.join(SCRATCH, "phase0agent"),
            "sandbox_real": os.path.join(SCRATCH, "phase0sandbox")}

SPO_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "attitude": {"type": "string"},
        "relation": {"type": "string"},
        "object": {"type": "string"},
        "residue": {"type": ["string", "null"]},
    },
    "required": ["subject", "attitude", "relation", "object", "residue"],
}

PROMPT = """Decompose this intent statement into its propositional structure.
Statement: {statement}
Intent (the agent's own stated WHY): {aim}

Roles:
- subject: who holds the intent. If it is the acting agent itself and unstated, write "SELF".
- attitude: the outer stance predicate (e.g. intend, desire, fear, avoid, maintain).
- relation: the embedded content's own predicate (what should actually happen).
- object: the state of affairs the content is about.
- residue: null if the statement fully decomposes. If it cannot, one of:
  "state_intent" (object is the agent's own ongoing state),
  "avoid_intent" (the core content is a negation/avoidance),
  "underspecified" (no determinate object yet),
  "other" (explain-free label; use sparingly).

Reply ONLY as JSON with exactly those five keys."""


def audit_statement(policy: DeepSeekAgentPolicy, statement: str, aim: str) -> dict:
    prompt = PROMPT.format(statement=statement, aim=aim)
    rec = policy._chat_json(prompt)
    if rec.get("subject") and rec.get("relation") and rec.get("object") and rec.get("residue") is None:
        full = True
    else:
        full = False
    return {"full": full, "subject_self": rec.get("subject") == "SELF",
            "attitude": rec.get("attitude"), "relation": rec.get("relation"),
            "object": rec.get("object"), "residue": rec.get("residue")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-db", type=int, default=50)
    ap.add_argument("--control", type=int, default=50)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    policy = DeepSeekAgentPolicy(model=args.model)
    out = {"model": args.model, "per_db": args.per_db, "tiers": {}}
    for tier, db in LIVE_DBS.items():
        with ICMLedger(db) as led:
            pairs = []
            for e in led.records("expectation"):
                iid = e["intent_id"]
                intent = led.get("intent", iid)
                aim = (intent or {}).get("aim", "")
                pairs.append((e["exp_id"], e["statement"], aim))
        # sample deterministically: every len//N-th offset, spread across sessions
        step = max(1, len(pairs) // args.per_db)
        sample = pairs[::step][:args.per_db]
        results = []
        for i, (exp_id, statement, aim) in enumerate(sample):
            t0 = time.time()
            try:
                r = audit_statement(policy, statement, aim)
            except Exception as e:  # relay hiccups recorded, not fatal
                r = {"full": False, "error": f"{type(e).__name__}: {e}"}
            r["exp_id"] = exp_id
            r["latency"] = round(time.time() - t0, 1)
            results.append(r)
        full_n = sum(1 for r in results if r.get("full"))
        self_n = sum(1 for r in results if r.get("subject_self"))
        residues = Counter(r.get("residue") or "none" for r in results)
        attitudes = Counter((r.get("attitude") or "?").lower().split()[0]
                            if r.get("attitude") else "?" for r in results)
        errors = sum(1 for r in results if "error" in r)
        out["tiers"][tier] = {
            "n": len(results), "full_extraction": full_n,
            "full_rate": round(full_n / len(results), 3),
            "subject_self_rate": round(self_n / len(results), 3),
            "residue": dict(residues), "attitudes": dict(attitudes.most_common(8)),
            "relay_errors": errors, "detail": results,
        }
        print(f"[{tier}] n={len(results)} full={full_n} self={self_n} "
              f"residues={dict(residues)} errors={errors}", flush=True)

    # synthetic control: templated statements from the synthetic-pilot templates
    from icm.pilot import ARCHETYPES
    ctl = []
    for name, tpl in list(ARCHETYPES.items()):
        for variant in ("aim", "desired"):
            if len(ctl) >= args.control:
                break
            statement = (f"{tpl['aim']} will yield: {tpl['desired']}"
                         if variant == "aim" else f"{tpl['aim']} (check) will yield: {tpl['desired']}")
            try:
                r = audit_statement(policy, statement, tpl["aim"])
            except Exception as e:
                r = {"full": False, "error": str(e)}
            ctl.append(r)
    out["tiers"]["synthetic_control"] = {
        "n": len(ctl), "full_extraction": sum(1 for r in ctl if r.get("full")),
        "full_rate": round(sum(1 for r in ctl if r.get("full")) / max(1, len(ctl)), 3),
        "subject_self_rate": round(sum(1 for r in ctl if r.get("subject_self")) / max(1, len(ctl)), 3),
        "errors": sum(1 for r in ctl if "error" in r),
    }
    path = os.path.join(SCRATCH, "spo_audit.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "detail"}
                      for k, v in out["tiers"].items()}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())