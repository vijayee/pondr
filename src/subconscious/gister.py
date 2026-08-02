"""The structured-gist producer for the fade consolidation loop.

The gist-on-forgetting loop (``consolidation_worker.py``) calls this when a fade
anchor crosses the recallability threshold (``cos < cos_gist + epsilon``). It
produces a STRUCTURED gist -- the maximem_synap_sdk ``key_extractions`` shape,
not a flat string -- by COMPOSING two existing Bonsai modules (neither returns
"narrative + facts" alone):

  - ``BonsaiDecider.consolidate_gist(blurb, prior_gist, count)`` -- the NARRATIVE
    (one tight paragraph; prior-baseline-merge on the second+ pass so
    gist-of-gist preserves fidelity). Cold-start safe (returns None).
  - ``BonsaiRelationExtractor.extract(blurb, isolated=False)`` -- the FACTS
    (relation triples; ``has_state`` triples are entity->value). NOT cold-start
    safe at this layer (raises) -- wrapped ``try/except -> []`` mirroring
    ``HippocampalEncoder._extract_relations`` (encoder.py:128).
  - ``extract_state_assertions(blurb, [], relations)`` -- deterministic, no-model
    entity->value claims (the Synap ``key_extractions`` analog; runs even with
    the server down).

Cold-start honest: if the narrative is ``None`` (Bonsai down / parse fail) the
gister returns ``None`` -- the worker skips the consolidation, the anchor stays
R4, retried next sweep. Deterministic facts alone do not re-key the anchor (the
narrative is the embed handle + recalled content), so a facts-only result is not
a usable consolidation. No fabricated gist, no silent stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..encoding.assertion_extractor import extract_state_assertions
from ..encoding.bonsai_relations import BonsaiRelationExtractor
from ..gnn.bonsai_decider import BonsaiDecider


@dataclass
class StructuredGist:
    """A structured gist = narrative + extracted facts.

    ``narrative`` is the embed handle AND the recalled content (replaces the
    verbatim blurb in place). ``facts`` (relation triples) +
    ``state_assertions`` (entity->value claims) are the structured sidecar --
    staged on the anchor for the future R4 -> long-term-memory pull (the
    ``fact_sink`` in ``ConsolidationWorker`` is the hook for the graph write).
    ``consolidation_count`` is carried for observability (the depth of
    gist-of-gist).
    """

    narrative: str
    facts: list[dict] = field(default_factory=list)
    state_assertions: list[dict] = field(default_factory=list)
    consolidation_count: int = 0


class BonsaiGister:
    """Compose the Bonsai narrative + fact extractors into a structured gist.

    Reuses ``config.bonsai_*`` via the two existing clients (the local Bonsai
    llama-server -- NOT Ollama/OracleClient). No cache in v1 (Bonsai has none
    today; re-gisting the same blurb is rare because consolidation mutates the
    blurb in place, so the next pass gists the gist, not the original).
    """

    def __init__(self, extractor: BonsaiRelationExtractor,
                 decider: BonsaiDecider) -> None:
        self.extractor = extractor
        self.decider = decider

    def gist(self, blurb: str, prior_gist: Optional[str],
             count: int) -> Optional[StructuredGist]:
        """Produce a structured gist for ``blurb``, or ``None`` on cold-start.

        ``prior_gist`` is the anchor's existing narrative (None for the first
        consolidation, the prior gist for gist-of-gist). ``count`` is the
        consolidation depth (fed to the prompt). Returns ``None`` when the
        narrative could not be produced (Bonsai down / parse fail) -- the caller
        skips the consolidation; the anchor stays R4 and is retried next sweep.
        """
        narrative = self.decider.consolidate_gist(blurb, prior_gist, count)
        if narrative is None:
            return None
        # Facts are carried from the SOURCE blurb (richer signal than the
        # compressed narrative) -- the long-term-memory pull sidecar.
        return self.shape(narrative, blurb, count)

    def shape(self, narrative: str, fact_source: str,
              count: int) -> StructuredGist:
        """Build a ``StructuredGist`` from a given narrative (fact-extraction half).

        Used by ``gist`` (facts from the source blurb) AND by
        ``ConsolidationWorker.resolve("edit")`` to rebuild a gist from the user's
        corrected narrative with facts extracted from that corrected text (the
        new truth). ``resolve("accept")`` does NOT call this -- it reuses the
        stored gist directly (already shaped with facts from the original blurb
        at defer time). Pure composition -- no LLM call (narrative in hand).
        """
        try:
            relations = self.extractor.extract(fact_source, isolated=False)
        except Exception:  # noqa: BLE001 - cold-start: facts degrade to []
            relations = []
        if not isinstance(relations, list):
            relations = []
        try:
            state_assertions = extract_state_assertions(fact_source, [], relations)
        except Exception:  # noqa: BLE001 - pure fn, but guard anyway
            state_assertions = []
        return StructuredGist(
            narrative=narrative,
            facts=relations,
            state_assertions=state_assertions,
            consolidation_count=count,
        )


def default_gister() -> BonsaiGister:
    """Construct a ``BonsaiGister`` from ``config.bonsai_*`` defaults.

    Lazy convenience for ``build_ponder`` -- both clients read
    ``config.bonsai_endpoint`` / ``bonsai_model`` / ``bonsai_temperature`` in
    their own ``__init__``; the HTTP call is lazy (one per ``gist``), so this is
    import-safe and constructible offline.
    """
    return BonsaiGister(BonsaiRelationExtractor(), BonsaiDecider())