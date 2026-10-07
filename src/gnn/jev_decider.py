"""R8: the Jev decision-model backend (Ollama ``/v1/systemone``).

A thin adapter that routes Pondr's CLOSED-VOCAB decision call sites to a
decision model (``clef-flash`` / ``nimble`` / ``tev1``) served by Ollama
>=0.35.1's raw ``/v1/systemone`` endpoint (TypeSafe Jev API -- named, typed
questions: ``noul`` true/false+prob, ``choice`` options+probs, ``score``
rubric levels; 1-64 questions per request; no SDK in ``src`` -- plain HTTP,
mirroring ``BonsaiDecider``'s plain-``requests`` pattern instead of pulling
``typesafe-sdk``).

Scope discipline (the survey's call-site split): this adapter overrides ONLY
the five decision methods whose output vocabularies are closed AND that emit
no free-form generation:

- ``judge_dedup_pairs``   -- 4-action ``choice`` per candidate pair (A1)
- ``verify_fidelity``     -- one ``noul`` corruption verdict (fade validate)
- ``decide_anomaly``      -- 3-way ``choice`` fix/ask_user/dismiss
- ``decide_contradiction``-- same, after the deterministic pre-filter
- ``classify_doc_kind``   -- 5-label ``choice`` (Sec 7.11)

Everything else is INHERITED from ``BonsaiDecider`` and keeps hitting the
Bonsai llama-server: ``gist`` / ``consolidate_gist`` / ``author_scene``
(generation, stays Bonsai 8B per the survey), and ``verify_typing`` /
``judge_task_lifecycle`` (their shapes are NOT closed-vocab: verify_typing can
PROPOSE a new ontology class and the lifecycle gate emits a generated task
label -- generation-shaped, stays Bonsai until a two-pass split is wanted).
So ``JevDecider`` is a *subclass* (``isinstance(JevDecider(...), BonsaiDecider)``
holds) with a second HTTP surface: ``self.endpoint`` remains the Bonsai
server for inherited methods (ctor default), while decision methods go to
``self.jev_endpoint``.

Decision safety notes:

- Jev answers carry probabilities, not text reasons -- verdict ``reason``
  fields become a compact ``"jev <type> p=<prob>"`` line. ``judge_dedup_pairs``
  keeps interface compatibility (dedup.py needs ``eid`` + ``action``; the
  reason is display-only telemetry).
- ``decide_anomaly``/``decide_contradiction`` return ``action="no_action"``:
  the consolidation ``_apply`` dispatcher auto-applies ONLY a ``fix`` whose
  action contains ``supersede_assertion``, so a Jev ``fix`` routes to
  ``ask_user`` (record-only). NEVER a silent auto-write -- conservative by
  construction with a closed-vocab model that cannot emit action code.
- ``judge_dedup_pairs`` chunks at 64 questions (the Ollama per-request cap);
  ANY failed chunk returns ``None`` for the whole call (the parent's
  all-or-nothing defer contract -- never a partial application).
- All failures (HTTP / parse / malformed answer) return ``None`` or drop, so
  the callers' cold-start fallbacks (defer / record-only / placeholder)
  behave identically to a down Bonsai.

``make_decider`` is the DI factory every wiring site uses:
``config.decision_backend == "bonsai"`` (default) -> a plain ``BonsaiDecider``
(byte-identical to pre-R8); any other value is treated as the Ollama model
name and routed to ``JevDecider``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from ..config import config
from ..observability.llm_telemetry import record_llm_call
from .bonsai_decider import (
    _DOC_KIND_LABELS,
    _deterministic_non_conflict,
    BonsaiDecider,
)

__all__ = ["JevDecider", "make_decider"]

# Ollama per-request caps (Jev API / TypeSafe System One): 1-64 questions per
# request and a 64 KiB body. ``judge_dedup_pairs`` chunks its candidate pool at
# the question cap; the body cap is the server's problem (a 400 routes to the
# same ``None``-defer cold-start path as a down server, documented in the
# method).
_JEV_MAX_QUESTIONS = 64

# The A1 dedup 4-action vocabulary (mirrors BonsaiDecider.judge_dedup_pairs).
_DEDUP_ACTIONS = ("store", "update", "merge", "skip")

# The anomaly/contradiction 3-action vocabulary (mirrors decide_anomaly /
# decide_contradiction).
_FIX_DISPATCH = ("fix", "ask_user", "dismiss")

# Coarse criteria descriptions for the dedup 4-action choice. The Bonsai
# prompt's wording (``bonsai_dedup_prompt``) carries the semantics; here each
# option's criterion is the one-line definition the Jev model scores against.
_DEDUP_CRITERIA = {
    "store": "The episode and the candidate are about different facts; keep both.",
    "update": "The episode is a NEWER version of the candidate; the old record should be superseded.",
    "merge": "The episode and the candidate together state one combined fact.",
    "skip": "The episode adds nothing beyond the candidate (a duplicate); discard it.",
}

# Criteria for the fix/ask_user/dismiss adjudication choice (mirrors the
# decision prompts: a fix mutates the graph, ask_user defers to the human,
# dismiss resolves that nothing is wrong).
_DISPATCH_CRITERIA = {
    "fix": "The record is wrong and should be corrected automatically.",
    "ask_user": "The evidence is ambiguous or the correction is risky; surface it to the user.",
    "dismiss": "The evidence shows nothing is actually wrong.",
}


def _jev_prob(answer: Any) -> Optional[float]:
    """Coerce a Jev probability field to a bounded float, or ``None``.

    ``noul`` answers carry ``{"noul": 0.997}``; ``choice`` answers carry
    ``probabilities: {option: p}`` plus the chosen option's own value. A
    malformed / missing number -> ``None`` (the caller formats "p=?" rather
    than crashing).
    """
    if not isinstance(answer, dict):
        return None
    v = answer.get("noul")
    if v is None:
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            return None
        v = probs.get(answer.get("choice"))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return min(max(f, 0.0), 1.0)


def _reason_line(kind: str, label: str, p: Optional[float]) -> str:
    """The compact display-only reason string for a Jev verdict."""
    ps = f"{p:.3f}" if p is not None else "?"
    return f"jev {kind}={label} p={ps}"


class JevDecider(BonsaiDecider):
    """The R8 Jev decision-model adapter over ``BonsaiDecider``.

    Inherited methods (generation + open-vocab judges) keep calling the Bonsai
    llama-server via ``self.endpoint``; the five overridden closed-vocab
    decision methods call the Ollama ``/v1/systemone`` endpoint instead.
    ``pause_gate`` / ``_last_usage`` / telemetry behavior carry over: the Jev
    client checks ``pause_gate`` before HTTP (same async-distill yielding as
    ``_post_json``) and maps Jev's ``{input_tokens, output_tokens}`` usage
    onto the OpenAI-style keys the ``@record_llm_call`` decorator reads.
    """

    def __init__(
        self,
        jev_model: Optional[str] = None,
        jev_endpoint: Optional[str] = None,
        jev_timeout: float = 30.0,
        corruption_threshold: float = 0.5,
        *args,
        **kwargs,
    ):
        """See ``BonsaiDecider.__init__`` for the inherited endpoints.

        All Bonsai ctor args pass through unchanged, so inherited methods read
        the same ``config.bonsai_*`` values a plain ``BonsaiDecider`` would.
        ``jev_model`` defaults to ``config.decision_backend`` (the backend is
        literally the Ollama model name); ``jev_endpoint`` defaults to
        ``config.decision_endpoint`` (the local Ollama base URL -- NOT the
        Bonsai server). ``corruption_threshold`` is the ``verify_fidelity``
        noul cutoff for ``corruption=True`` (0.5 balanced; a misjudged
        ``clean`` OVERWRITES the verbatim while a misjudged ``corrupt`` only
        defers, so lower = more conservative -- the caller may tune it down).
        """
        super().__init__(*args, **kwargs)
        self.jev_model = jev_model or config.decision_backend
        self.jev_endpoint = (jev_endpoint or config.decision_endpoint).rstrip("/")
        self.jev_timeout = jev_timeout
        self.corruption_threshold = corruption_threshold

    # ---- Jev HTTP ---------------------------------------------------------

    def jev_health_check(self, timeout: float = 3.0) -> bool:
        """True iff something answers ``GET {jev_endpoint}/v1/models`` (Ollama's
        OpenAI-compat listing). Live-test skip guard; never raises."""
        try:
            r = requests.get(f"{self.jev_endpoint}/v1/models", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _score_systemone(
        self, state: dict, questions: dict
    ) -> Optional[dict]:
        """One POST to ``/v1/systemone``; return the ``answers`` dict or None.

        Never raises (same cold-start contract as ``_post_json``): a down
        model, non-200, non-JSON body, a missing ``answers`` key, or a
        NOT-JSON-SERIALIZABLE state (the callers hand in radius-1 graph
        material that may carry non-primitive values) all return ``None`` and
        the caller defers. On success ``self._last_usage`` is set with Jev's
        ``{input_tokens, output_tokens}`` MAPPED onto the OpenAI-style keys
        (``prompt_tokens`` / ``completion_tokens`` + ``total_tokens``) the
        ``@record_llm_call`` decorator reads.
        """
        if self.pause_gate is not None:
            self.pause_gate()
        self._last_usage = None
        url = f"{self.jev_endpoint}/v1/systemone"
        payload = {"model": self.jev_model, "state": state, "questions": questions}
        try:
            body = json.dumps(payload)
        except (TypeError, ValueError):
            return None
        try:
            resp = requests.post(url, data=body, timeout=self.jev_timeout,
                                 headers={"Content-Type": "application/json"})
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        try:
            outer = resp.json()
        except json.JSONDecodeError:
            return None
        if not isinstance(outer, dict) or "answers" not in outer:
            return None
        usage = outer.get("usage")
        if isinstance(usage, dict):
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")
            self._last_usage = {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": (in_tok + out_tok)
                if isinstance(in_tok, int) and isinstance(out_tok, int)
                else None,
            }
        return outer["answers"] if isinstance(outer["answers"], dict) else None

    # ---- overridden closed-vocab decision methods -------------------------

    @record_llm_call("judge_dedup_pairs")
    def judge_dedup_pairs(
        self,
        new_episode_summary: str,
        new_entities: list,
        new_topics: list,
        candidates: list[dict],
    ) -> Optional[list[dict]]:
        """A1 dedup: one 4-action ``choice`` question per candidate.

        Chunks at the 64-question Ollama cap -- each chunk carries the SAME
        new-episode state with only that chunk's candidates indexed, so
        question keys stay chunk-local. ANY failed chunk -> ``None`` for the
        WHOLE call (the parent's all-or-nothing defer: dedup.py applies
        verdicts as a batch, never partially). Verdict shape
        ``{"eid", "action", "reason"}`` matches the parent exactly; the
        reason is a compact probability line (Jev returns no free text).
        """
        if not candidates:
            return None
        state = {
            "new_episode": {
                "summary": str(new_episode_summary),
                "entities": list(new_entities),
                "topics": list(new_topics),
            }
        }
        out: list[dict] = []
        for start in range(0, len(candidates), _JEV_MAX_QUESTIONS):
            chunk = candidates[start : start + _JEV_MAX_QUESTIONS]
            questions: dict = {}
            for i, cand in enumerate(chunk):
                eid = str(cand.get("eid", ""))
                questions[f"pair_{i}"] = {
                    "type": "choice",
                    "instructions": (
                        f"Candidate pair {i} (eid={eid}): is the candidate a "
                        f"duplicate/older-version/merge-partner of the new "
                        f"episode, or a distinct fact?"
                    ),
                    "criteria": _DEDUP_CRITERIA,
                }
            answers = self._score_systemone(state, questions)
            if answers is None:
                return None
            for i, cand in enumerate(chunk):
                a = answers.get(f"pair_{i}")
                choice = a.get("choice") if isinstance(a, dict) else None
                if choice not in _DEDUP_ACTIONS:
                    continue  # malformed verdict -> drop, never fabricate
                out.append({
                    "eid": cand.get("eid", ""),
                    "action": choice,
                    "reason": _reason_line("dedup", choice, _jev_prob(a)),
                })
        return out

    @record_llm_call("verify_fidelity")
    def verify_fidelity(self, blurb: str, narrative: str) -> Optional[dict]:
        """Fade validated-compaction: one ``noul`` corruption verdict.

        ``corruption`` is True when the Jev model puts >=
        ``self.corruption_threshold`` probability on the narrative CHANGING a
        fact's meaning (the worker then defers; only compression/paraphrase
        applies in place). Malformed / missing noul answer -> ``None`` (the
        worker defers, honest cold-start -- never auto-apply an unjudged
        gist).
        """
        if not isinstance(blurb, str) or not blurb.strip():
            return None
        if not isinstance(narrative, str) or not narrative.strip():
            return None
        state = {"source_blurb": blurb, "candidate_gist": narrative}
        questions = {
            "corrupt": {
                "type": "noul",
                "instructions": (
                    "Does the candidate gist CORRUPT the source blurb? "
                    "Corruption means a fact's MEANING changed (inverted, "
                    "swapped, fabricated, or attributed to the wrong entity). "
                    "Mere compression, paraphrase, or dropped detail is NOT "
                    "corruption -- answer true ONLY for a meaning change."
                ),
            }
        }
        answers = self._score_systemone(state, questions)
        if answers is None:
            return None
        a = answers.get("corrupt")
        p = _jev_prob(a)
        if not isinstance(a, dict) or p is None:
            return None
        corruption = p >= self.corruption_threshold
        return {
            "corruption": corruption,
            "reason": _reason_line("corrupt", "true" if corruption else "false", p),
        }

    @record_llm_call("decide_anomaly")
    def decide_anomaly(self, flag: dict, retrieved_context: dict) -> Optional[dict]:
        """Identity-drift adjudication: a 3-way ``choice`` (fix/ask_user/dismiss).

        ``action`` is always ``"no_action"`` -- with a closed-vocab model that
        cannot emit ``supersede_assertion`` code, the consolidation ``_apply``
        dispatcher routes any ``fix`` to ``ask_user`` (record-only). Never a
        fabricated action. Same ``None``-on-failure contract as the parent.
        """
        return self._decide_three_way(
            "decide_anomaly", "identity_drift", flag, retrieved_context
        )

    @record_llm_call("decide_contradiction")
    def decide_contradiction(self, flag: dict, retrieved_context: dict) -> Optional[dict]:
        """``contradictory_state`` adjudication.

        The parent's deterministic non-conflict pre-filter runs FIRST (BEFORE
        any HTTP, byte-identical guards -- the complementary-temporal and
        equal-values cases the small deciders rubber-stamp); only a real
        conflict reaches the Jev 3-way choice. See ``decide_anomaly`` for the
        ``action="no_action"`` note.
        """
        pre = _deterministic_non_conflict(
            retrieved_context.get("state_values")
            if isinstance(retrieved_context, dict) else None
        )
        if pre is not None:
            return pre
        return self._decide_three_way(
            "decide_contradiction", "contradictory_state", flag
        )

    @record_llm_call("classify_doc_kind")
    def classify_doc_kind(self, doc_text: str) -> Optional[str]:
        """Zero-shot doc-kind tag: a 5-label ``choice`` over the Sec 7.11 vocab.

        The answer's ``choice`` is validated against ``_DOC_KIND_LABELS``
        (Jev constrains it to the criteria options, but the guard is kept --
        mirrors the parent). ``None`` on failure / out-of-vocab -> the caller
        writes the cold-start ``"other"``.
        """
        if not isinstance(doc_text, str) or not doc_text.strip():
            return None
        state = {"document": doc_text}
        criteria = {k: k for k in sorted(_DOC_KIND_LABELS)}
        questions = {
            "doc_kind": {
                "type": "choice",
                "instructions": (
                    "Which kind of document is this? point_in_time_snapshot="
                    "a state at a specific date (report, status, snapshot); "
                    "decision_update=records a decision that replaces an "
                    "earlier one; plan=forward-looking intent; reference="
                    "stable background material; other=none of these."
                ),
                "criteria": criteria,
            }
        }
        answers = self._score_systemone(state, questions)
        if answers is None:
            return None
        a = answers.get("doc_kind")
        if not isinstance(a, dict):
            return None
        kind = str(a.get("choice", "")).strip().lower()
        if kind not in _DOC_KIND_LABELS:
            return None
        return kind

    # ---- shared 3-way adjudication ---------------------------------------

    def _decide_three_way(
        self,
        method: str,
        anomaly_type: str,
        flag: dict,
        retrieved_context: Optional[dict] = None,
    ) -> Optional[dict]:
        """Shared body for ``decide_anomaly`` / ``decide_contradiction``.

        One ``choice`` question; the evidence (the flag's record + the
        retrieved radius-1 context) goes in the ``state`` verbatim-ish (JSON)
        so the model sees the same material the Bonsai prompt would carry.
        Returns the parent's ``{decision, action: "no_action", reasoning}``
        shape, or ``None`` on HTTP/parse/missing answer.
        """
        ctx = retrieved_context if retrieved_context is not None else {}
        state = {
            "flag": {
                "entity": str(flag.get("node", "")),
                "type": str(flag.get("type", anomaly_type)),
                "evidence": flag.get("evidence"),
            },
            "retrieved_context": ctx,
        }
        questions = {
            "decision": {
                "type": "choice",
                "instructions": (
                    f"A {anomaly_type} flag was raised on an entity. Decide "
                    f"what to do: fix it automatically, surface it to the "
                    f"user, or dismiss it as a false alarm."
                ),
                "criteria": _DISPATCH_CRITERIA,
            }
        }
        answers = self._score_systemone(state, questions)
        if answers is None:
            return None
        a = answers.get("decision")
        if not isinstance(a, dict):
            return None
        decision = str(a.get("choice", "")).strip()
        if decision not in _FIX_DISPATCH:
            return None
        return {
            "decision": decision,
            "action": "no_action",
            "reasoning": _reason_line(method, decision, _jev_prob(a)),
        }


def make_decider(decision_backend: Optional[str] = None) -> BonsaiDecider:
    """The DI factory every decider wiring site uses (R8).

    ``decision_backend=None`` reads ``config.decision_backend`` (the serve CLI
    sets it from ``--decision-backend`` before ``build_ponder``). ``"bonsai"``
    (the default) -> a plain ``BonsaiDecider`` -- byte-identical to pre-R8.
    Any other value is treated as an Ollama Jev model name (``clef-flash`` /
    ``nimble`` / ``tev1`` / ``tev1:0.8b``) and routed to ``JevDecider``;
    the returned subclass still satisfies every ``BonsaiDecider`` DI seam
    (pause_gate install in ``distill_worker``, ``DedupJudge._decider``,
    ``SceneAuthoringWorker``'s author, the DreamWorker / ConsolidationWorker
    deciders).

    A down / unpulled model is NOT an error here (construction is always
    offline-safe; the first HTTP call routes to ``None`` -> the caller's
    cold-start defer). The serve entrypoint warns once at startup when the
    health check fails.
    """
    backend = (
        decision_backend
        if decision_backend is not None
        else config.decision_backend
    )
    if not isinstance(backend, str) or not backend.strip() or backend == "bonsai":
        return BonsaiDecider()
    return JevDecider(jev_model=backend.strip())