"""S-P-O extraction pass (from docs/Intent_Subject_Predicate_Object_Schema.md).

Shared by the imposed-grammar audit (spo_audit.py) and the freeform arm
(scaffold A/B arm B): arm A extracts from schema-led declarations; arm B
extracts from raw prose produced WITHOUT any leading schema. The two arms
share this pass so extraction differences are attributable to the SOURCE,
not to the extractor.
"""

from __future__ import annotations

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

PROMPT = """Decompose this text into its propositional intent structure, if it has one.
Text: {statement}
(Stated WHY, if the agent gave one: {aim})

Roles:
- subject: who holds the intent. If it is the acting agent itself and unstated, write "SELF".
- attitude: the outer stance predicate (e.g. intend, desire, fear, avoid, maintain).
- relation: the embedded content's own predicate (what should actually happen).
- object: the state of affairs the content is about.
- residue: null if the text fully decomposes. If it cannot, one of:
  "state_intent" (object is the agent's own ongoing state),
  "avoid_intent" (the core content is a negation/avoidance),
  "underspecified" (no determinate object yet),
  "other".

Reply ONLY as JSON with exactly those five keys."""


# Long-prose variant (arm B = freeform): freeform text is intent + reasons +
# plan + hedging mixed, so "the intent" must be defined BEFORE extraction or
# the decomposer silently picks whatever suits it.
PRIMARY_PROMPT = """The text below is an agent's free-form note before acting: it may mix
the primary intent with reasons, plans, contingencies, and hedging.

Text: {statement}

Extract the PRIMARY intended outcome as ONE proposition (the thing the note is
FOR, not the method or the reasoning). Then fill:
- subject: who holds it ("SELF" if the acting agent and unstated; empty string ""
  only if genuinely unnameable).
- attitude: the outer stance predicate.
- relation: the embedded predicate of that primary proposition.
- object: its state of affairs.
- residue: null if the primary proposition is fully determinate; else one of
  "state_intent" / "avoid_intent" / "underspecified" / "other".
- propositions: how many DISTINCT candidate intents the text mentions (integer).

Reply ONLY as JSON with keys subject, attitude, relation, object, residue, propositions."""


def extract_spo(policy, statement: str, aim: str) -> dict:
    """Pass-2 extraction via the policy's structured-output channel (arm A)."""
    rec = policy._chat_json(PROMPT.format(statement=statement, aim=aim))
    full = bool(rec.get("subject") and rec.get("relation") and rec.get("object")
                and rec.get("residue") is None)
    return {"full": full, "subject_self": rec.get("subject") == "SELF",
            "attitude": rec.get("attitude"), "relation": rec.get("relation"),
            "object": rec.get("object"), "residue": rec.get("residue")}


def extract_primary(policy, statement: str, aim: str) -> dict:
    """Arm-B extraction: primary-proposition contract for freeform prose."""
    rec = policy._chat_json(PRIMARY_PROMPT.format(statement=statement, aim=aim))
    full = bool(rec.get("subject") and rec.get("relation") and rec.get("object")
                and rec.get("residue") is None)
    subs = (rec.get("subject") or "")
    try:
        n_props = int(rec.get("propositions", 1))
    except (TypeError, ValueError):
        n_props = None
    return {"full": full, "subject_self": subs.strip().upper() == "SELF",
            "subject_raw": subs, "attitude": rec.get("attitude"),
            "relation": rec.get("relation"), "object": rec.get("object"),
            "residue": rec.get("residue"), "propositions": n_props}