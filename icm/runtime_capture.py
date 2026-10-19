"""Capture a LIVE Pondr session into the ICM ledger -- the real-data test bed.

WHY (docs/ICM_intent_store_spec.md sec 4 item 5): every corpus we hold lacks
per-memory intent variation; the only honest source of it is the store running
behind actual work, where the SAME memory is retrieved under DIFFERENT intents
over time. This adapter is that seam for Pondr's own runtime. It is OPT-IN:
nothing in the runtime changes unless a caller wraps an orchestrator.

WHAT IS RECORDED (six-field record, fields 1/3/5 + the declared part of 4):
  intent      the user prompt verbatim, DECLARED at capture time; optional
              qualifier / requirements / epoch are recorded as declared and are
              never inferred here.
  memory      the episode ids the runtime actually retrieved this turn
              (result["retrieved_episodes"]), recorded at the moment of use.
  epoch       declared run/order when given, else capture time.
  key         the write-time judge key, when an IntentKeyer is supplied.
  variation   the same memory id appearing under different intent ids across
              turns -- see variation_index().

WHAT IS NOT RECORDED, DELIBERATELY:
  outcome     a chat answer is NOT a mechanical outcome. This adapter never
              computes a congruence label from prose. If a caller holds a
              mechanical observation (tool exit status, filesystem state, test
              result), it declares a prediction with IntentCapture.goal()/
              step() and records the observation with outcome(); that path is
              the existing rule-computed one and is unchanged.
  usefulness  no label is derived from "the runtime retrieved it" -- what the
              old retriever chose is not ground truth.

HONESTY: a capture failure never loses the user's answer. The turn result
carries {"icm_capture": {"status": "error", ...}} and the query returns
normally. A failure inside the runtime itself is NOT swallowed.

DEFAULT OFF: constructed only where a caller asks for it; no runtime wiring.
"""

from __future__ import annotations

from icm.capture import IntentCapture


def _retrieved_ids(result) -> list[str]:
    """Episode ids from the runtime's result; order preserved, duplicates gone."""
    episodes = result.get("retrieved_episodes") if isinstance(result, dict) else None
    ids: list[str] = []
    for ep in episodes or []:
        eid = ep.get("episode_id") if isinstance(ep, dict) else ep
        if isinstance(eid, str) and eid and eid not in ids:
            ids.append(eid)
    return ids


class SessionCapture:
    """Wrap an orchestrator's query() and record intents + memory links."""

    def __init__(self, ponder, ledger, mirror_dir=None, keyer=None):
        self.ponder = ponder
        self.capture = IntentCapture(ledger, mirror_dir=mirror_dir, keyer=keyer)

    def query(self, user_prompt: str, *, qualifier=None, requirements=None,
              epoch=None, **query_kwargs) -> dict:
        declaration_error = None
        intent_id = None
        try:
            intent_id = self.capture.intent(
                user_prompt, level="goal", qualifier=qualifier,
                requirements=requirements, epoch=epoch)
        except Exception as e:  # declaration failure must not block the turn
            declaration_error = e

        result = self.ponder.query(user_prompt, **query_kwargs)
        out = dict(result) if isinstance(result, dict) else {"result": result}
        if intent_id is None:
            out["icm_capture"] = {"status": "error", "stage": "declare",
                                  "error": str(declaration_error)}
            return out
        try:
            memory_units = _retrieved_ids(result)
            retrieval_id = self.capture.retrieval(
                intent_id, user_prompt, memory_units, k=len(memory_units))
            out["icm_capture"] = {
                "status": "recorded",
                "intent_id": intent_id,
                "retrieval_id": retrieval_id,
                "memory_units": memory_units,
                "outcome": "unmeasured",
            }
        except Exception as e:  # the answer is already built; never lose it
            out["icm_capture"] = {"status": "error", "stage": "retrieval",
                                  "intent_id": intent_id, "error": str(e)}
        return out

    def close(self) -> str:
        return self.capture.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def variation_index(ledger) -> dict:
    """Memory id -> the declared intents it was retrieved under.

    This is the census that answers "has real work produced the flip shape?":
    `with_variation` counts memory units retrieved under 2+ DISTINCT intents.
    It makes no claim about usefulness (there is no mechanical outcome here).
    """
    intents: dict[str, dict] = {}
    for rec in ledger.records("icapture"):
        if rec.get("level") in ("goal", "step"):
            intents[rec["id"]] = rec
    units: dict[str, list[dict]] = {}
    for rec in ledger.records("icapture"):
        if rec.get("level") != "retrieval":
            continue
        intent_id = rec.get("intent_id")
        if intent_id is None:
            continue
        irec = intents.get(intent_id, {})
        for memory_id in rec.get("memory_units") or rec.get("retrieved") or []:
            rows = units.setdefault(memory_id, [])
            if any(r["intent_id"] == intent_id for r in rows):
                continue
            rows.append({
                "intent_id": intent_id,
                "text": irec.get("text", ""),
                "qualifier": irec.get("qualifier", ""),
                "epoch": irec.get("epoch"),
                "key_status": irec.get("key_status"),
                "intent_key": irec.get("intent_key"),
            })
    with_variation = sum(1 for rows in units.values() if len(rows) >= 2)
    return {"memory_units": len(units), "with_variation": with_variation,
            "units": units}
