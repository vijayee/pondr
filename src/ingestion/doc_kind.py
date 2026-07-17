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
    """

    def __init__(self, head, embedder) -> None:
        self._head = head
        self._embedder = embedder

    def classify_doc_kind(self, section_texts: list[str]) -> Optional[str]:
        if not section_texts:
            return None
        return self._head.classify(section_texts, self._embedder)


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