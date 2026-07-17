"""Doc-kind taggers for the ingestion pipeline (Phase 3c Sec 7.11 + deferred step).

Two adapters satisfying the pipeline's ``doc_kind_tagger`` contract
(``classify_doc_kind(section_texts: list[str]) -> Optional[str]``):

- ``BonsaiDocKindTagger``: wraps a ``BonsaiDecider``; joins the section texts via
  the same recipe ``UnifiedIngestionPipeline._doc_text`` uses (byte-identical to
  the Sec 7.11 zero-shot HTTP path) and calls ``decider.classify_doc_kind``.
- ``BackboneDocKindTagger``: the deferred step -- a local forward pass through
  the trained ``DocKindHead`` on the frozen backbone (no :8080 contention).

``build_doc_kind_tagger`` centralizes the policy: prefer the trained head when
its checkpoint exists; fall back to the Bonsai adapter when a decider is
supplied; else ``None`` (the pipeline leaves ``doc_kind`` at the cold-start
``"other"``, byte-identical to pre-7.11).

The interface widens from Sec 7.11's ``classify_doc_kind(text: str)`` to
``classify_doc_kind(section_texts: list[str])`` so the head gets the section
SEQUENCE it steps the SSM over. ``BonsaiDecider.classify_doc_kind(text)`` is NOT
changed -- the Bonsai adapter joins the sections back to the identical text.
The previous single-string duck-typed taggers (tests) ignore their arg, so
the widen is back-compatible for them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Protocol

# Cap on the joined text handed to the Bonsai tagger (mirrors
# pipeline._BONSAI_TEXT_CAP). Kept here so the Bonsai adapter reproduces the
# exact text the Sec 7.11 HTTP path sent, without depending on a parsed object.
_BONSAI_TEXT_CAP = 8000

# Default trained-checkpoint paths. The backbone path mirrors
# runtime.DEFAULT_BACKBONE_PATH -- duplicated here so ingestion does not depend
# on the serve-only runtime module (the CLI owns ingest; build_ponder does not
# ingest and does not load the head).
DEFAULT_DOC_KIND_HEAD_PATH = "data/training/doc_kind_head/best.pt"
DEFAULT_BACKBONE_PATH = (
    "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
)


class DocKindTagger(Protocol):
    """Pipeline contract: tag a doc's semantic kind from its section texts.

    ``section_texts`` is the per-section chunk text (``heading + "\\n" +
    content`` or just ``content``) in document order -- the same texts
    ``UnifiedIngestionPipeline`` embeds for the per-chunk vector path. Returns
    one of the 5 Sec 7.11 labels, or ``None`` on failure / out-of-vocab (the
    caller writes the cold-start ``"other"``, no fabricated label).
    """

    def classify_doc_kind(self, section_texts: list[str]) -> Optional[str]: ...


def join_section_texts(section_texts: list[str], cap: int = _BONSAI_TEXT_CAP) -> str:
    """Join per-section texts the way ``_doc_text`` does: strip each, ``\\n\\n``-join, cap.

    Byte-identical to ``UnifiedIngestionPipeline._doc_text(parsed)`` when
    ``section_texts`` is the pipeline's per-section chunk list. Factored here so
    the Bonsai adapter reproduces the exact text the Sec 7.11 HTTP path sent,
    without depending on a ``parsed`` object.
    """
    out = [s.strip() for s in section_texts]
    return "\n\n".join(out)[:cap]


# ── temporal feature (Phase 4: attack the mean-pool blind spot) ──
#
# The head's mean-pool over per-section step outputs DISCARDS which section
# carries the date that distinguishes "state AS OF T" (point_in_time_snapshot)
# from "decision MADE ON T" (decision_update) from "will do by T" (plan). This
# extracts a doc-level temporal signal vector concatenated with the pooled
# embedding so the linear head sees the date + its framing directly. Pure +
# stable: no reference "now" (a snapshot's "as of 2026-03" date is past/present
# at train time but the FEATURE only records that a date + an "as of" phrase
# co-occur -- invariant across train/serve, unlike an any_future_date dim which
# would flip as calendar time advances).

_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                       # 2026-03-31
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                 # 3/31/26, 03/31/2026
    re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
               r"[a-z]*\s+\d{2,4}\b", re.IGNORECASE),           # 31 March 2026
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
               r"[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", re.IGNORECASE),  # March 31, 2026
]
_AS_OF_RE = re.compile(r"\b(?:as\s+of|as\s+at|as-of)\b", re.IGNORECASE)
# decision_update signal -- stems catch decide/decides/decided/decision.
_DECISION_RE = re.compile(
    r"\b(?:decid\w*|decision|ADR|switched|adopted|approved|supersed\w*|"
    r"policy\s+change|will\s+(?:switch|move|migrate|adopt|deprecate))\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(?:roadmap|plan(?:ned|ning)?|will|going\s+to|sprint|OKRs?|objective|"
    r"proposal|milestone|upcoming|next\s+quarter|target(?:ing)?)\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^\s{0,3}#+\s", re.MULTILINE)  # a markdown heading line

# Dimension count -- the head's first Linear widens by this when the feature is
# on. Keep stable; the checkpoint persists + validates it.
TEMPORAL_FEAT_DIM = 6


def _find_dates(text: str) -> list[re.Match]:
    matches: list[re.Match] = []
    for pat in _DATE_PATTERNS:
        matches.extend(pat.finditer(text))
    # Order by position so "first date" is well-defined.
    matches.sort(key=lambda m: m.start())
    return matches


def extract_temporal_features(section_texts: list[str]) -> list[float]:
    """Doc-level temporal signal vector (``TEMPORAL_FEAT_DIM`` floats in [0,1]).

    Dims:
      0 has_any_explicit_date   -- any ISO/slash/month date in the doc.
      1 has_as_of_phrase        -- "as of"/"as at" (the snapshot signal: a record
                                    of state AS OF a date).
      2 has_decision_phrase     -- decid*/decision/ADR/switched/adopted/approved
                                    (the decision_update signal -- the direct
                                    counter-signal that separates a decision
                                    record from a status snapshot).
      3 n_explicit_dates_norm    -- min(date_count, 3) / 3 (date density; snapshots
                                    + decisions carry one, plans/roadmaps carry
                                    several).
      4 first_date_in_heading   -- the first date occurs on a markdown heading line
                                    (snapshots + decisions often date-stamp the
                                    heading: "# Q1 2026 status (as of 03-31)").
      5 has_plan_phrase         -- roadmap/plan/will/OKR/milestone (the plan signal).

    Pure + dependency-free (regex only), so it runs identically at train and serve
    (no train/serve skew) and unit-tests without torch. The doc is the
    ``\\n\\n``-joined section texts (same join the Bonsai path uses).
    """
    text = join_section_texts(section_texts) if section_texts else ""
    if not text:
        return [0.0] * TEMPORAL_FEAT_DIM
    dates = _find_dates(text)
    has_date = 1.0 if dates else 0.0
    has_as_of = 1.0 if _AS_OF_RE.search(text) else 0.0
    has_decision = 1.0 if _DECISION_RE.search(text) else 0.0
    n_dates_norm = min(len(dates), 3) / 3.0
    # first_date_in_heading: does the LINE containing the first date start with
    # a markdown heading marker (#)? Snapshots/decisions often date-stamp the
    # heading ("# Q1 2026 status (as of 03-31)").
    first_in_heading = 0.0
    if dates:
        line_start = text.rfind("\n", 0, dates[0].start()) + 1
        line_end = text.find("\n", dates[0].end())
        line = text[line_start:line_end if line_end != -1 else len(text)]
        if _HEADING_RE.match(line):
            first_in_heading = 1.0
    has_plan = 1.0 if _PLAN_RE.search(text) else 0.0
    return [has_date, has_as_of, has_decision, n_dates_norm, first_in_heading, has_plan]


class BonsaiDocKindTagger:
    """Sec 7.11 zero-shot tagger, adapted to the section-texts interface.

    Wraps a ``BonsaiDecider`` (whose ``classify_doc_kind(text)`` is the text
    primitive, unchanged). Joins ``section_texts`` via ``join_section_texts``
    (byte-identical to ``_doc_text``) and delegates. A failed HTTP / parse /
    out-of-vocab result returns ``None`` from the decider -> caller writes
    ``"other"`` (same as Sec 7.11).
    """

    def __init__(self, decider) -> None:
        self._decider = decider

    def classify_doc_kind(self, section_texts: list[str]) -> Optional[str]:
        if not section_texts:
            return None
        return self._decider.classify_doc_kind(join_section_texts(section_texts))


class BackboneDocKindTagger:
    """Deferred step: local forward pass through the trained ``DocKindHead``.

    Holds a loaded ``DocKindHead`` (on the frozen backbone) and a bge-small
    embedder. Tags a doc by embedding its sections, stepping the SSM, pooling
    the final state, and reading the 5-class head -- no HTTP, no :8080
    contention. Returns ``None`` on empty sections -> caller writes ``"other"``.

    When the head was trained with the temporal feature (``head.feat_dim > 0``,
    Phase 4), the feature is computed from the SAME filtered ``section_texts``
    via ``extract_temporal_features`` and passed into ``classify`` -> ``forward``
    -- byte-identical to the train/eval path (no train/serve skew). A feat-less
    head (``feat_dim == 0``) skips the feature (the original path).
    """

    def __init__(self, head, embedder) -> None:
        self._head = head
        self._embedder = embedder

    def classify_doc_kind(self, section_texts: list[str]) -> Optional[str]:
        if not section_texts:
            return None
        feat = None
        if getattr(self._head, "feat_dim", 0) > 0:
            import torch
            fv = extract_temporal_features(section_texts)
            feat = torch.tensor(fv, dtype=torch.float32).unsqueeze(0)   # [1, k]
        return self._head.classify(section_texts, self._embedder, feat=feat)


def build_doc_kind_tagger(
    *,
    head_path: Optional[str] = DEFAULT_DOC_KIND_HEAD_PATH,
    backbone_path: str = DEFAULT_BACKBONE_PATH,
    device: str = "auto",
    embedder_source: str = "on-demand",
    bonsai_decider=None,
    verbose: bool = True,
) -> Optional[DocKindTagger]:
    """Construct the best available doc-kind tagger (head > Bonsai > None).

    Prefers the trained ``BackboneDocKindTagger`` when the head checkpoint
    exists (local forward pass, no :8080 contention). Falls back to
    ``BonsaiDocKindTagger`` when a ``bonsai_decider`` is supplied. Returns
    ``None`` when neither is available -- the pipeline then leaves ``doc_kind``
    at the cold-start ``"other"`` default (byte-identical to pre-7.11). A
    head-load failure prints a warning and falls through to the Bonsai fallback
    rather than aborting the ingest.

    Heavy deps (torch, the backbone checkpoint, ``sentence_transformers``) are
    imported lazily INSIDE the head branch so this module imports cheaply and
    the no-tag (cold-start) path never pulls them.
    """
    if head_path and Path(head_path).exists():
        try:
            from ..subconscious.configs import BackboneConfig
            from ..subconscious.training.routing_training import (
                build_embedder, load_backbone, load_doc_kind_head,
            )
            backbone = load_backbone(backbone_path, BackboneConfig(), device=device)
            head = load_doc_kind_head(head_path, backbone, device=device)
            embedder = build_embedder(embedder_source)
            if verbose:
                print(f"doc-kind tagger: trained head on frozen backbone "
                      f"({head_path})")
            return BackboneDocKindTagger(head, embedder)
        except Exception as exc:  # noqa: BLE001 -- best-effort; fall back to Bonsai
            if verbose:
                print(f"warning: doc-kind head load failed ({exc}); "
                      f"falling back to Bonsai/None")
    if bonsai_decider is not None:
        if verbose:
            print("doc-kind tagger: Bonsai zero-shot (no trained head checkpoint)")
        return BonsaiDocKindTagger(bonsai_decider)
    return None