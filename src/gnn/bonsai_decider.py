"""Bonsai-in-consolidation: the deploy-time decider for three consolidation
actions, all driven by the local 8B Bonsai model (the "subconscious").

The consolidation loop (``consolidate.py``) clusters episodes (DiffPool GNN)
and then, at APPLY time, performs three Bonsai-gated actions:

1. **Abstract gist generation** -- synthesize one paragraph abstracting a
   cluster into a semantic memory (``M:NNNN``), embed it, and index it so the
   memory becomes a retrieval candidate.
2. **Ontology promotion** -- gate the ontology head's entity->class typing
   proposals through Bonsai: accept the typing, propose a NEW narrower class
   under an existing parent, or reject.
3. **Identity-drift anomaly decision** -- for an ``identity_drift`` rule-flag,
   retrieve-then-decide ``fix``/``ask_user``/``dismiss``.

The Oracle/DeepSeek is the *training-data teacher only* (it labels the pairs
Bonsai is later fine-tuned on). At DEPLOY the decider is Bonsai -- local,
zero-cost, the speed proposition. This module is the thin HTTP client to the
Bonsai ``llama-server`` (OpenAI-compatible, ``config.bonsai_endpoint``); it
mirrors ``BonsaiRelationExtractor``'s request + parse pattern but is a separate
object (the decider returns str / dict, not triples, so it does not extend the
link-prediction ``verifier`` callable).

All three actions are independently gated: the caller passes ``decider=None``
(or sets ``bonsai_decider_enabled=False``) and the cold-start path stays
record-only and byte-identical to today -- no HTTP, no fabricated decision.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import requests

from ..config import config
from ..training.prompts import (
    bonsai_anomaly_decision_prompt,
    bonsai_contradiction_decision_prompt,
    bonsai_doc_kind_prompt,
    bonsai_gist_prompt,
    bonsai_typing_prompt,
)

__all__ = ["BonsaiDecider"]


# Matches a ```json ... ``` (or bare ```) fenced block. Small models sometimes
# wrap "ONLY JSON" output in fences anyway. Mirrors bonsai_relations._FENCE_RE.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

# Control chars stripped defensively from a gist before it is stored as a
# content value + vector-metadata string. Keeps tab/newline (legitimate prose)
# but drops the C0 control range that would corrupt downstream JSON / display.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Calendar-month abbreviations. Used by the complementary-temporal deterministic
# guard (see ``_deterministic_non_conflict``) to recognize month-named
# point-in-time records (``docs/jan-status.md`` / ``docs/jul-status.md``): two
# such records carrying different values are complementary snapshots, not a
# supersession. Lowercased; the guard lowercases the path token before lookup.
_MONTH_ABBREVS = frozenset(
    ("jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec")
)

# Phase 3c Sec 7.11: the doc-kind taxonomy. Written at ingest
# (``Document.doc_kind``); the complementary-temporal guard fires when BOTH
# asserting sources are ``point_in_time_snapshot``. ``classify_doc_kind``
# validates the model's label against this set and returns ``None`` on a
# missing / out-of-vocab label (caller writes the cold-start ``"other"``).
_DOC_KIND_LABELS = frozenset(
    ("point_in_time_snapshot", "decision_update", "plan", "reference", "other")
)


def _month_prefix(path: str) -> str:
    """Return the lowercased first ``-``-delimited token of ``path``'s basename
    if it is a calendar-month abbreviation, else ``""``.

    ``"docs/jan-status.md"`` -> ``"jan"``; ``"docs/db-pick-v1.md"`` -> ``""``;
    ``"doc_000123_sec_002"`` -> ``""`` (production doc/section ids carry no
    month token -- the guard keys off ``source_path`` which
    ``_gather_entity_context`` resolves from the doc id, not the id itself).
    """
    if not isinstance(path, str) or not path:
        return ""
    base = path.rsplit("/", 1)[-1]
    tok = base.split("-", 1)[0].lower()
    return tok if tok in _MONTH_ABBREVS else ""


class BonsaiDecider:
    """Deploy-time Bonsai decider for abstract gist / ontology typing / anomaly.

    Talks to an OpenAI-compatible ``llama-server`` (Bonsai) over HTTP. Plain
    ``response_format: {"type": "json_object"}`` chat completions -- NO
    tool-calling dependency. Constructible offline (the HTTP call is lazy, one
    per ``gist``/``verify_typing``/``decide_anomaly`` call).

    Each decision method returns ``None`` / a rejected verdict on HTTP or
    parse failure rather than raising, so the consolidator can fall back to
    the honest cold-start path (placeholder abstract / record-only) without a
    try/except wrapper at every call site. ``RuntimeError`` is reserved for
    ``_post_json`` itself (unexpected server state) and is caught by the
    caller's best-effort contract.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = 60.0,
        max_tokens: int = 768,
    ):
        self.model = model or config.bonsai_model
        self.endpoint = (endpoint or config.bonsai_endpoint).rstrip("/")
        self.temperature = (
            temperature if temperature is not None else config.bonsai_temperature
        )
        self.timeout = timeout
        self.max_tokens = max_tokens

    # ---- public decisions ------------------------------------------------

    def health_check(self, timeout: float = 3.0) -> bool:
        """True iff the Bonsai server responds to ``GET {endpoint}/models``.

        The live-test skip guard: mirrors ``tests/test_bonsai_relations.py``.
        Never raises -- a down server is a normal cold-start condition.
        """
        try:
            r = requests.get(f"{self.endpoint}/models", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def gist(self, source_episodes: list[dict]) -> Optional[str]:
        """Synthesize one paragraph abstracting ``source_episodes``.

        ``source_episodes`` is a list of ``{id, summary, text?}`` dicts; the
        caller pre-caps the list (``abstract_gist_max_episodes``). Returns the
        gist string, or ``None`` on HTTP/parse failure (caller falls back to
        the placeholder abstract). Control chars are stripped defensively
        since the gist becomes a stored content value + vector-metadata.
        """
        if not source_episodes:
            return None
        prompt = bonsai_gist_prompt(source_episodes)
        data = self._post_json(prompt)
        if data is None:
            return None
        raw = data.get("gist") if isinstance(data, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            return None
        return _CTRL_RE.sub("", raw.strip())

    def verify_typing(
        self, entity: str, candidate_class: str, retrieved_context: dict
    ) -> Optional[dict]:
        """Decide an ontology typing proposal.

        Returns ``{"accept": bool, "new_class": Optional[str], "parent":
        Optional[str], "reasoning": str}``, or ``None`` on HTTP/parse failure
        (caller records the proposal without writing). ``new_class``/``parent``
        are normalized to ``None`` when empty/``"null"`` strings so the caller's
        truthiness checks are unambiguous. The caller verifies ``parent``
        exists in the seed ontology before creating any class.
        """
        prompt = bonsai_typing_prompt(entity, candidate_class, retrieved_context)
        data = self._post_json(prompt)
        if not isinstance(data, dict) or "accept" not in data:
            return None
        accept = bool(data.get("accept"))
        new_class = _opt_str(data.get("new_class"))
        parent = _opt_str(data.get("parent"))
        # If a new class is proposed, a parent is required; without one the
        # class would orphan -- the caller treats this as a rejection.
        if new_class and not parent:
            accept = False
        return {
            "accept": accept,
            "new_class": new_class,
            "parent": parent,
            "reasoning": str(data.get("reasoning", ""))[:1000],
        }

    def decide_anomaly(self, flag: dict, retrieved_context: dict) -> Optional[dict]:
        """Decide what to DO about an identity-drift flag.

        ``flag`` is the ``{node, type, evidence}`` record from
        ``anomaly_rules.flag_identity_drift``; ``retrieved_context`` is the
        radius-1 neighborhood the consolidator gathered (the same context the
        Oracle teacher saw, so the decision is reproducible). Reuses the
        existing ``bonsai_anomaly_decision_prompt`` (prompts.py:142). Returns
        ``{"decision": "fix"|"ask_user"|"dismiss", "action": str, "reasoning":
        str}`` or ``None`` on failure (caller records the flag only).
        """
        flagged_entity = str(flag.get("node", ""))
        anomaly_type = str(flag.get("type", "identity_drift"))
        prompt = bonsai_anomaly_decision_prompt(
            flagged_entity, retrieved_context, anomaly_type
        )
        data = self._post_json(prompt)
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

    def decide_contradiction(self, flag: dict, retrieved_context: dict) -> Optional[dict]:
        """Decide what to DO about a ``contradictory_state`` flag (Phase 3c D3).

        Mirror of ``decide_anomaly`` for the fact-level contradiction path:
        ``flag`` is the ``{node, type:"contradictory_state", evidence}`` record
        from ``_detect_contradictory_state``; ``retrieved_context`` carries
        the conflicting ``state_values`` WITH provenance (``asserted_by`` /
        ``asserted_at`` / ``source_path``), gathered by
        ``_gather_entity_context``. Uses
        ``bonsai_contradiction_decision_prompt``. Returns
        ``{"decision": "fix"|"ask_user"|"dismiss", "action": str, "reasoning":
        str}`` or ``None`` on failure (caller records the flag only -- honest
        cold-start, no fabricated decision). The conservative dispatcher in
        ``_apply`` auto-applies ONLY a ``fix`` whose ``action`` contains
        ``supersede_assertion`` AND ``forgetting_enabled``; any other ``fix``
        -> ``ask_user`` (record-only).

        A deterministic pre-filter (``_deterministic_non_conflict``) runs BEFORE
        the HTTP call and short-circuits the clear non-conflict cases the 8B
        decider capacity-boundedly false-fixes (see ``docs/Phase 3c.md`` Sec 7:
        the 8B and 27B both rubber-stamp complementary-temporal pairs into
        ``fix+supersede_assertion`` -- a silent false tombstone). These guards
        are correct-by-construction, not learned behavior; they mirror the
        always-on ``extract_state_assertions`` normalizer and cost one HTTP call
        fewer when they fire. Real conflicts (two distinct values, non-month-
        named sources) bypass the pre-filter and hit Bonsai verbatim -- the
        fine-tuned adapter still adjudicates the genuine-conflict path.
        """
        flagged_entity = str(flag.get("node", ""))
        pre = _deterministic_non_conflict(
            retrieved_context.get("state_values")
            if isinstance(retrieved_context, dict) else None
        )
        if pre is not None:
            return pre
        prompt = bonsai_contradiction_decision_prompt(
            flagged_entity, retrieved_context
        )
        data = self._post_json(prompt)
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

    def classify_doc_kind(self, doc_text: str) -> Optional[str]:
        """Zero-shot doc-kind tag (Phase 3c Sec 7.11).

        One Bonsai HTTP call at ingest: ``doc_text`` -> one of the five
        ``_DOC_KIND_LABELS``. Mirrors ``decide_contradiction``'s shape --
        ``self._post_json(prompt)`` then validate the label against the
        vocabulary. Returns the label string, or ``None`` on HTTP / parse
        failure OR an out-of-vocabulary / missing label (the caller writes the
        cold-start ``"other"`` default -- NO fabricated label, byte-identical
        to a not-wired ingest). The complementary-temporal guard later fires
        on ``point_in_time_snapshot`` (semantic) instead of a filename month-
        prefix (fragile on real enterprise docs).

        No deterministic pre-filter here (unlike ``decide_contradiction``):
        the tagger is a single-label classifier, not an adjudicator, so there
        is no non-conflict case to short-circuit. ``doc_text`` is capped by
        the caller (``_doc_text`` in the ingestion pipeline) to the Bonsai
        text cap before this call.
        """
        if not isinstance(doc_text, str) or not doc_text.strip():
            return None
        prompt = bonsai_doc_kind_prompt(doc_text)
        data = self._post_json(prompt)
        if not isinstance(data, dict):
            return None
        kind = str(data.get("doc_kind", "")).strip().lower()
        if kind not in _DOC_KIND_LABELS:
            return None
        return kind

    # ---- HTTP + parse helpers --------------------------------------------

    def _post_json(self, prompt: str) -> Optional[dict]:
        """POST a single-user chat completion, return the parsed JSON object.

        Returns ``None`` on any HTTP failure, non-JSON body, or missing content
        (the decision methods translate that into their own ``None`` /
        rejected-verdict fallback). Never raises: a down server, a non-200
        response, a non-JSON body, a missing ``choices[0].message.content``,
        or an unparseable content string are ALL normal cold-start
        conditions that route to ``None`` so the consolidator falls back to
        the honest record-only / placeholder path without a try/except at
        every call site. (``gist``/``verify_typing``/``decide_anomaly`` still
        keep their own ``except Exception`` guards in ``_apply`` so a stub
        decider that raises cannot break a dream pass.)
        """
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        try:
            outer = resp.json()
        except json.JSONDecodeError:
            return None
        try:
            content = outer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        return self._parse_json_object(content)

    @staticmethod
    def _parse_json_object(content: str) -> Optional[dict]:
        """Parse the model's JSON content into a dict.

        Strips accidental ``` fences; failing that, falls back to the outermost
        ``{...}`` span (handles trailing prose). Returns ``None`` if no JSON
        object can be carved out -- the caller treats that as a missed
        decision (record-only / placeholder), never a crash.
        """
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
                    data = json.loads(body[start : end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None
        return data if isinstance(data, dict) else None


def _opt_str(v) -> Optional[str]:
    """Normalize a model string field to ``Optional[str]``: ``None`` for empty
    / ``"null"`` / non-string; stripped otherwise."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s or s.lower() == "null":
        return None
    return s


def _deterministic_non_conflict(state_values) -> Optional[dict]:
    """Deterministic guards for cases that are non-conflicts by construction.

    Returns a ``{"decision", "action", "reasoning"}`` dict to short-circuit
    ``decide_contradiction`` BEFORE the Bonsai HTTP call, or ``None`` to let the
    LLM adjudicate. Two guards, both correct-by-construction (neither can
    false-dismiss a real conflict nor false-tombstone a non-conflict):

    (1) EQUAL VALUES -- every live state value is the same string. A
        contradiction requires two DIFFERENT values; agreeing values are not a
        conflict regardless of source or newness -> ``dismiss`` + ``no_action``.
        This is defense-in-depth: ``_detect_contradictory_state`` only flags
        DISTINCT values, so the production decider is never handed an all-equal
        pair -- but if a future caller or a direct test invokes the decider on
        one, this guard correctly dismisses instead of asking the LLM (which the
        8B capacity-boundedly false-fixes, see ``docs/Phase 3c.md`` Sec 7).

    (2) COMPLEMENTARY TEMPORAL -- two DIFFERENT values whose asserting sources
        are BOTH point-in-time snapshots. Two snapshots carrying different
        values are states at different dates -- both true at their respective
        times, not a supersession -> ``ask_user`` (conservative, NON-MUTATING:
        the ``_apply`` dispatcher only auto-applies a ``fix``+
        ``supersede_assertion``, so ``ask_user`` surfaces the pair to the
        human rather than auto-tombstoning a possibly-complementary value).
        This is the guard that closes the N14 false-tombstone -- the one
        production-real negative the fine-tune AND the 27B both fail.

        The complementary signal is recognized in TWO ways (either fires the
        guard), so it is production-sound on real enterprise docs:
        (a) SEMANTIC (Phase 3c Sec 7.11, primary): both sources' ``doc_kind``
            == ``"point_in_time_snapshot"`` -- a content-derived tag written at
            ingest by ``classify_doc_kind``. This fires on docs that carry NO
            month in their names (a Jira ticket, ``Q1-report-final.pdf``), the
            exact case the bench found the filename guard inert on.
        (b) FILENAME (fallback, defense-in-depth): both sources' path carries
            a month-name prefix (``jan-status.md`` / ``jul-status.md``). Kept so
            the guard still fires when ``doc_kind`` is absent -- cold-start
            (no tagger wired), pre-7.11 docs, and the committed month-named
            fixtures -- byte-identical to pre-7.11 on those.

    Source-path / doc-kind resolution: ``_gather_entity_context`` enriches each
    ``state_value`` with ``source_path`` AND ``doc_kind`` (resolved from the
    doc/section ``asserted_by`` id). The guard keys off ``doc_kind`` first
    (semantic), then ``source_path`` with an ``asserted_by`` fallback (the eval
    harness passes the source_path directly as ``asserted_by``). Episode-id /
    ``None`` provenance carries neither -> the guard falls through to the LLM
    (status quo). A ``decision_update`` doc_kind is a REAL conflict -- it
    bypasses this guard and hits Bonsai verbatim.
    """
    if not isinstance(state_values, list) or len(state_values) < 2:
        return None
    vals: list[str] = []
    for v in state_values:
        if not isinstance(v, dict):
            continue
        vals.append(str(v.get("value", "")).strip())
    if len(vals) < 2:
        return None

    # Guard 1: all live values identical -> never a conflict.
    if len(set(vals)) == 1:
        return {
            "decision": "dismiss",
            "action": "no_action",
            "reasoning": (
                f"Deterministic guard (equal values): all {len(vals)} live "
                f"state values are identical ('{vals[0]}'). A contradiction "
                f"requires two DIFFERENT values; agreeing values are not a "
                f"conflict regardless of source or newness -> dismiss, "
                f"no_action."
            ),
        }

    # Guard 2: complementary temporal -- fires when BOTH asserting sources are
    # point-in-time snapshots. SEMANTIC primary (both doc_kind ==
    # "point_in_time_snapshot", content-derived, fires on non-month-named docs)
    # OR FILENAME fallback (both paths carry a month-name prefix, kept for
    # cold-start / pre-7.11 / month-named fixtures). Two snapshots with different
    # values are complementary, not a supersession -> ask_user (non-mutating;
    # do not auto-tombstone). A decision_update / absent doc_kind + no month
    # prefix falls through to the LLM (real version-suffixed decision docs).
    kinds: list[str] = []
    months: list[str] = []
    for v in state_values:
        if not isinstance(v, dict):
            continue
        kinds.append(str(v.get("doc_kind") or "").strip().lower())
        path = v.get("source_path") or v.get("asserted_by")
        months.append(_month_prefix(path))
    # Consider the first two distinct-value sources (the conflicting pair).
    if len(kinds) >= 2:
        k0, k1 = kinds[0], kinds[1]
        m0, m1 = months[0], months[1]
        semantic = (k0 == k1 == "point_in_time_snapshot")
        filename = bool(m0 and m1)
        if semantic or filename:
            signal = ("doc_kind=point_in_time_snapshot"
                      if semantic else f"month-prefix {m0}-/{m1}-")
            return {
                "decision": "ask_user",
                "action": "no_action",
                "reasoning": (
                    f"Deterministic guard (complementary temporal): two "
                    f"DIFFERENT live state values ('{vals[0]}' vs "
                    f"'{vals[1]}') asserted by point-in-time snapshots "
                    f"({signal}). These are states at different dates, both "
                    f"true at their respective times -- not a supersession "
                    f"-> ask_user (do not auto-tombstone a possibly-"
                    f"complementary value)."
                ),
            }
    return None