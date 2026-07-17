"""DocKindHead: a 5-class doc-kind classifier on the trained JGSBackbone.

Phase 3c Sec 7.11 deferred step -- the "first real downstream job for the
trained SSM". Sec 7.11 shipped semantic doc-kind tagging at ingest via a
zero-shot Bonsai HTTP call (``BonsaiDecider.classify_doc_kind``). The labels it
writes to ``content/doc/{doc_id}/doc_kind`` are this head's training data. This
head replaces the HTTP call at ingest with a local forward pass through the
frozen shared backbone (no :8080 contention).

Architecture (the confirmed "section sequence via inject + pool state" path):
a doc is a sequence of sections. The head embeds each section (bge-small,
384-d), steps the SSM over the section embeddings on the **serving path** (the
same ``JGSInstance.step`` loop ``SSMChunker.compress_episodes`` runs via
``WorkingMemory.inject``), pools the final layer's recurrent state via
``state.mean(dim=1)`` (mirroring ``DecomposedGate._pool`` in ``gate.py``), and
applies a 5-class linear head. This reuses a production serving path and keeps
section structure -- it does NOT use the pretraining-only ``forward_seq`` and
does NOT crush the doc to a single embedding.

The shared backbone is frozen and held via ``object.__setattr__`` (inherited
from ``JGSInstance``), so ``head.state_dict()`` EXCLUDES the ~19.5M backbone
params -- the checkpoint is lean (instance projections + LoRA + state_lora +
the classifier head). The loader (``routing_training.load_doc_kind_head``)
pairs with ``load_backbone`` so a serving/ingest path stands up the trained
head on the trained frozen backbone.

The 5 labels (``LABELS``) are the Sec 7.11 taxonomy, in a FIXED canonical order
so the head's logits map to a stable index. The checkpoint persists this order
and the loader validates it. The guard (``bonsai_decider.py``) consumes the
returned label STRING and is order-agnostic, so this list is LOCAL to the head
-- no subconscious->gnn import coupling.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .configs import INSTANCE_CONFIGS, InstanceConfig
from .instance import JGSInstance


class DocKindHead(JGSInstance):
    """5-class doc-kind classifier on the shared frozen JGSBackbone.

    Owns one classifier head on top of the ``JGSInstance`` base. The shared
    backbone is frozen during head training (Phase 2a weights); only the
    instance-owned params (input/output projections + LoRA, state_lora, the
    decomposed gate -- the gate is unused here but constructed by the base) and
    the classifier head train. ``head.parameters()`` already excludes the
    backbone (stored via ``object.__setattr__``), so an ``AdamW(head.parameters(),
    ...)`` optimizer naturally leaves the backbone alone.
    """

    # Canonical 5-class label order (the Sec 7.11 taxonomy). The checkpoint
    # persists this; the loader validates it on load. The complementary-temporal
    # guard keys off ``"point_in_time_snapshot"``; ``"decision_update"`` is a
    # real conflict that bypasses the guard; ``"other"`` is the cold-start /
    # tagger-failure / not-wired default.
    LABELS: tuple[str, ...] = (
        "point_in_time_snapshot",
        "decision_update",
        "plan",
        "reference",
        "other",
    )

    def __init__(self, backbone, config: Optional[InstanceConfig] = None):
        cfg = config or INSTANCE_CONFIGS["doc_kind"]
        super().__init__(backbone, cfg)
        d = cfg.d_model  # 384 -- the SSM recurrent-state dim we pool
        self.head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, len(self.LABELS)),
        )

    def forward(self, section_embeddings: list[Tensor]) -> Tensor:
        """Step the SSM over the section embeddings, pool the final state, classify.

        Args:
            section_embeddings: one ``[1, 384]`` (or ``[384]``) bge-small embedding
                per section, in document order.

        Returns:
            Logits ``[1, len(LABELS)]``.

        Resets the recurrent state first (each doc is independent -- no
        cross-doc memory), then steps each section embedding through the serving
        path (``JGSInstance.step``), then pools the LAST layer's recurrent state
        ``[1, d_state=16, d_model=384]`` via ``mean(dim=1)`` -> ``[1, 384]``
        (mirrors ``DecomposedGate._pool`` in ``gate.py``). The last layer is the
        most-abstracted summary -- the natural pooled representation of the doc.
        """
        if not section_embeddings:
            raise ValueError("DocKindHead.forward called with no section embeddings")
        self.reset_state(1)
        for emb in section_embeddings:
            self.step(emb)
        # After the loop self.state holds the final per-layer recurrent states
        # (detached per-step by JGSInstance -- no BPTT across sections). Pool the
        # last layer's [1, d_state, d_model] -> [1, d_model].
        state = self.state[-1]
        pooled = state.mean(dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def classify(self, section_texts: list[str], embedder) -> Optional[str]:
        """Tag a doc given its section texts -> one of ``LABELS``, or ``None``.

        Embeds each section text via ``embedder`` (bge-small, 384-d), runs
        ``forward``, and returns ``LABELS[argmax]``. Returns ``None`` for empty
        section text so the caller writes the cold-start ``"other"`` default
        (same contract as ``BonsaiDecider.classify_doc_kind`` -> no fabricated
        label). The caller owns the embedder (injected), keeping this module
        torch-only (no ``sentence_transformers`` import here -- mirrors
        ``RetrievalGate.route_text``).
        """
        section_texts = [s for s in section_texts if s and s.strip()]
        if not section_texts:
            return None
        self.eval()
        device = next(self.parameters()).device
        vecs = embedder.encode(section_texts)
        embs = [
            torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for v in vecs
        ]
        logits = self.forward(embs)
        idx = int(logits.argmax(dim=-1).item())
        return self.LABELS[idx]