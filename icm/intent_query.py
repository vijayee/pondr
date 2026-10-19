"""Two-stage intent query path -- dense candidates, deterministic filter, key re-rank.

ARCHITECTURE (measured; addenda 22-23, docs/ICM_intent_store_spec.md sec 1):
  stage 1  dense similarity generates candidates (measured 45x chance on real
           intents, addenda 17-19);
  stage 2  an entity/type FILTER, then an ordering derived from the intent
           key's epoch reference (judge-read at write time). Addendum 23:
           60/60 filesystem-verified trials, chance 1/3, dense@1 0.433.

PRE-REGISTERED BEHAVIOUR (written before implementation; no fitting on runs):
  (1) NO SCORE BLENDING. Dense orders candidates in stage 1 and supplies the
      fallback ranking; stage 2 changes the order only by FILTERING and by
      ordering on the key's epoch reference. Requirement/qualifier overlap is
      never added to a dense score (addendum 22: fusion cancels).
  (2) DETERMINISM: same inputs produce the same ranking (stable sorts only,
      no sampling, no ties broken randomly).
  (3) FALLBACKS ARE EXPLICIT, never silent. `fallback` in the result is one of
      None / "no_key" / "empty_filter" / "no_epoch", and when it is set the
      ranking IS the dense ranking (the spec's stated fallback is dense@1,
      which is untested at scale -- here it is at least visible).
  (4) REPRODUCTION: on a constructed pool the stage-2 order must reproduce
      the addendum-23 mechanism -- entity/type filter, then
      EARLIEST / MIDDLE / LATEST ordering over candidate epoch -- including
      picking the correct record when dense@1 picks a wrong one. This is
      checked in tests/test_icm_intent_query.py; the filesystem-verified
      60-trial harness scripts/icm/twobytwo_stage2.py remains the regression
      of the underlying claim.

CANDIDATE SHAPE (minimal): {"id": str, "text": str}; optional "epoch",
"type", "entity", "intent_key", "meta". Query-side metadata:
{"type": ..., "entity": ..., "entities": [...], "qualifier": ...}.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np

from icm.intent_keys import ORDINALS

# Words that identify the KIND of thing rather than WHICH thing, plus function
# words. The entity filter must narrow on instances ("alpha"), not on the
# artifact noun ("checksum"), or it filters nothing.
GENERIC = frozenset("""
a an and are as at be been before by current data dataset datasets dated for
from has have in into is it its last latest mean middle new now of old on or
over past previous record records report reported same state stored than that
the then there these this those through to under up value values was were
which with within
""".split())

FALLBACKS = (None, "no_key", "empty_filter", "no_epoch")


def content_tokens(text) -> set[str]:
    """Lowercase alnum tokens of length >= 3 that are not generic."""
    out, cur = set(), []
    for ch in str(text or "").lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                word = "".join(cur)
                if len(word) >= 3 and word not in GENERIC:
                    out.add(word)
                cur = []
    if cur:
        word = "".join(cur)
        if len(word) >= 3 and word not in GENERIC:
            out.add(word)
    return out


def entities_of(key: dict | None, query_meta: dict | None = None) -> set[str]:
    """Instance tokens the query names: requirement subjects/objects + object.

    Explicit query_meta entities are added. When query_meta names a TYPE, the
    type's own tokens are removed: the type filter already narrows on the
    artifact kind, and leaving those tokens in the entity set would let every
    record of the same kind pass (measured failure in the first unit run).
    """
    toks: set[str] = set()
    if key:
        toks |= content_tokens(key.get("object", ""))
        for req in key.get("requirements") or []:
            if not isinstance(req, dict):
                continue
            toks |= content_tokens(req.get("subject", ""))
            toks |= content_tokens(req.get("object", ""))
    meta = query_meta or {}
    if meta.get("type"):
        toks -= content_tokens(meta["type"])
    if meta.get("entity"):
        toks |= content_tokens(meta["entity"])
    for ent in meta.get("entities") or []:
        toks |= content_tokens(ent)
    return toks


def _sort_key(value):
    """Total-order key for an epoch value; missing epochs sort last."""
    if value is None:
        return (2, 0.0)
    if isinstance(value, bool):
        return (2, 0.0)
    if isinstance(value, (int, float)):
        return (0, float(value))
    text = str(value).strip()
    if not text:
        return (2, 0.0)
    try:
        return (0, float(text))
    except ValueError:
        pass
    try:
        return (0, _dt.datetime.fromisoformat(text).timestamp())
    except ValueError:
        return (1, text)


def _coerce_epoch(value):
    k = _sort_key(value)
    return k[1] if k[0] == 0 else None


def _epoch_domain(value) -> str:
    """Which comparable family an epoch value belongs to."""
    if isinstance(value, bool) or value is None:
        return "missing"
    if isinstance(value, (int, float)):
        return "number"
    text = str(value).strip()
    if not text:
        return "missing"
    try:
        float(text)
        return "number"
    except ValueError:
        pass
    try:
        _dt.datetime.fromisoformat(text)
        return "date"
    except ValueError:
        return "text"


def _haystack(cand: dict) -> set[str]:
    parts = [cand.get("text", ""), cand.get("type", ""), cand.get("entity", "")]
    meta = cand.get("meta")
    if isinstance(meta, dict):
        parts.extend(str(v) for v in meta.values())
    key = cand.get("intent_key")
    if isinstance(key, dict):
        parts.append(key.get("object", ""))
        for req in key.get("requirements") or []:
            if isinstance(req, dict):
                parts.append(req.get("subject", ""))
                parts.append(req.get("object", ""))
    toks: set[str] = set()
    for p in parts:
        toks |= content_tokens(p)
    return toks


def _epoch_order(cands: list[dict], epoch_ref: dict) -> list[dict] | None:
    """Order candidates by the key's epoch reference. None = unusable ref."""
    kind = (epoch_ref or {}).get("kind", "none")
    value = (epoch_ref or {}).get("value", "")
    if kind == "ordinal" and value in ORDINALS:
        ordered = sorted(cands, key=lambda c: _sort_key(c.get("epoch")))
        present = [_coerce_epoch(c.get("epoch")) for c in ordered]
        present = [v for v in present if v is not None]
        if not present:
            return None
        if value == "LATEST":
            # descending, but MISSING epochs stay LAST (do not reverse them
            # to the head -- that would rank records with no epoch first)
            ordered = sorted(cands, key=lambda c: (
                _sort_key(c.get("epoch"))[0] != 0,
                -(_coerce_epoch(c.get("epoch")) or 0.0)))
        elif value == "MIDDLE":
            median = present[len(present) // 2]
            pairs = list(zip(ordered, [_coerce_epoch(c.get("epoch"))
                                       for c in ordered]))
            pairs.sort(key=lambda p: (p[1] is None,
                                      abs(p[1] - median) if p[1] is not None else 0.0,
                                      p[1] if p[1] is not None else 0.0))
            ordered = [p[0] for p in pairs]
        return ordered
    if kind == "absolute" and value:
        target = _coerce_epoch(value)
        if target is None:
            return None
        target_domain = _epoch_domain(value)
        domains = [_epoch_domain(c.get("epoch")) for c in cands]
        if not any(d == target_domain for d in domains):
            # the key names a date and the pool carries run numbers (or vice
            # versa): ordering them would be numerically meaningless, so the
            # honest answer is "no usable epoch reference"
            return None
        pairs = list(zip(cands, [_coerce_epoch(c.get("epoch")) for c in cands],
                         domains))
        pairs.sort(key=lambda p: (p[2] != target_domain,
                                  p[1] is None,
                                  abs(p[1] - target) if p[1] is not None else 0.0,
                                  p[1] if p[1] is not None else 0.0))
        return [p[0] for p in pairs]
    return None


class TwoStageRetriever:
    """Dense generation + deterministic filter + key re-rank. No score fusion."""

    def __init__(self, embed, keyer=None):
        # embed: callable [texts] -> 2D float array (normalisation is applied
        # defensively here, so an unnormalised embedder still yields cosine).
        self.embed = embed
        self.keyer = keyer

    def _cosine(self, intent_text: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        q = np.asarray(self.embed([intent_text]), dtype=float)[0]
        m = np.asarray(self.embed(texts), dtype=float)
        if m.ndim == 1:
            m = m.reshape(1, -1)

        def unit(v):
            n = math.sqrt(float(np.dot(v, v)))
            return v / n if n > 0 else v

        q = unit(q)
        return np.array([float(np.dot(unit(row), q)) for row in m])

    def query(self, intent_text: str, candidates: list[dict],
              query_key: dict | None = None,
              query_meta: dict | None = None) -> dict:
        cands = [dict(c) for c in candidates]
        meta = query_meta or {}
        key_status = "supplied" if query_key is not None else "missing"
        if query_key is None and self.keyer is not None:
            res = self.keyer.key_for(intent_text, meta.get("qualifier", ""))
            query_key, key_status = res.key, res.status

        texts = [str(c.get("text", "")) for c in cands]
        scores = self._cosine(intent_text, texts)
        dense_order = [int(i) for i in np.argsort(-scores, kind="stable")]
        stage1_ids = [cands[i].get("id") for i in dense_order]

        def dense_ranked(fallback: str) -> dict:
            return {
                "ranked": [cands[i] for i in dense_order],
                "top": [cands[i] for i in dense_order[: meta.get("k", 5)]],
                "stage1_ids": stage1_ids,
                "filtered_ids": [],
                "fallback": fallback,
                "key": query_key,
                "key_status": key_status,
                "entities": [],
                "type_filter": meta.get("type"),
                "epoch_ref": (query_key or {}).get("epoch_ref"),
                "stage2_applied": False,
            }

        if query_key is None:
            return dense_ranked("no_key")

        entities = entities_of(query_key, meta)
        qtype = meta.get("type")

        def passes(c: dict) -> bool:
            if qtype is not None and c.get("type") != qtype:
                return False
            if entities and not (_haystack(c) & entities):
                return False
            return True

        filtered = [cands[i] for i in dense_order if passes(cands[i])]
        if not filtered:
            return dense_ranked("empty_filter")

        epoch_ref = query_key.get("epoch_ref") or {"kind": "none", "value": ""}
        ordered = _epoch_order(filtered, epoch_ref)
        if ordered is None:
            ranked = filtered + [cands[i] for i in dense_order if not passes(cands[i])]
            fallback = "no_epoch"
            stage2 = False
        else:
            ranked = ordered + [cands[i] for i in dense_order if not passes(cands[i])]
            fallback = None
            stage2 = True

        k = meta.get("k", 5)
        return {
            "ranked": ranked,
            "top": ranked[:k],
            "stage1_ids": stage1_ids,
            "filtered_ids": [c.get("id") for c in filtered],
            "fallback": fallback,
            "key": query_key,
            "key_status": key_status,
            "entities": sorted(entities),
            "type_filter": qtype,
            "epoch_ref": epoch_ref,
            "stage2_applied": stage2,
        }
