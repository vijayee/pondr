"""DocKindHead: a 5-class doc-kind classifier on the trained JGSBackbone.

Phase 3c Sec 7.11 deferred step -- the "first real downstream job for the
trained SSM". Sec 7.11 shipped semantic doc-kind tagging at ingest via a
zero-shot Bonsai HTTP call (``BonsaiDecider.classify_doc_kind``). The labels it
writes to ``content/doc/{doc_id}/doc_kind`` are this head's training data. This
head replaces the HTTP call at ingest with a local forward pass through the
frozen shared backbone (no :8080 contention).

Architecture (section sequence via the serving path + reduced step outputs):
a doc is a sequence of sections. The head embeds each section (bge-small,
384-d), steps the SSM over the section embeddings on the **serving path** (the
same ``JGSInstance.step`` loop ``SSMChunker.compress_episodes`` runs via
``WorkingMemory.inject``), reduces the per-section STEP OUTPUTS (the learned
``output_proj`` readout, 256-d -- the same signal ``RetrievalGate`` classifies
on) to one ``[1, 256]`` doc vector, and applies a 5-class linear head. The
reduction is either a MEAN over sections (the original head) or a learned
ADDITIVE ATTENTION over sections (Phase 5, ``attention_readout=True`` -- finds
the date-bearing section instead of averaging it away). This reuses a
production serving path and keeps section structure -- it does NOT use the
pretraining-only ``forward_seq`` and does NOT crush the doc to a single
embedding. (An earlier design pooled the raw recurrent state via
``state.mean(dim=1)``; it mode-collapsed on real enterprise prose and was
replaced -- see ``forward``.)

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

    # Attention readout bottleneck dim (Phase 5). Single-head additive
    # attention over the per-section step outputs: a learned key projection +
    # a learned query; softmax over sections -> a weighted sum (vs the equal-
    # weight mean-pool). Lets the head FIND the date-bearing section instead
    # of averaging it away. Small (16.5k params) vs the 567k head.
    ATTN_DIM = 64

    def __init__(self, backbone, config: Optional[InstanceConfig] = None,
                 feat_dim: int = 0, attention_readout: bool = False):
        cfg = config or INSTANCE_CONFIGS["doc_kind"]
        super().__init__(backbone, cfg)
        d = cfg.output_dim  # 256 -- the step-output readout dim we pool
        # feat_dim > 0 widens the head's first Linear to accept the temporal
        # feature vector concatenated with the pooled embedding (Phase 4). 0
        # = no feature (the original head). Persisted in the checkpoint +
        # validated on load so a feat-trained head can't be loaded into a
        # feat-less head (or vice versa) -- a mismatch would silently feed
        # zeros/garbage into the Linear.
        self.feat_dim = int(feat_dim)
        # Phase 5 attention-over-sections readout. When False the head uses the
        # original mean-pool (the A/B baseline + the old-checkpoint load path).
        # When True a learned additive attention readout replaces it -- same
        # [1, d] pooled shape, so the feat concat + head Linear are unchanged.
        # Persisted in the checkpoint + validated on load (the strict state_dict
        # check in load_doc_kind_head catches a mean-pool ckpt loaded into an
        # attention head as a mismatch).
        self.attention_readout = bool(attention_readout)
        if self.attention_readout:
            # score_i = query . tanh(W_key @ section_i); softmax over i -> weight.
            # attn_query inits to ZEROS so attention starts UNIFORM (== mean-pool
            # at init) and learns to diverge -- a clean A/B starting point with
            # no random-init luck dependency.
            self.attn_key = nn.Linear(d, self.ATTN_DIM, bias=True)
            self.attn_query = nn.Parameter(torch.zeros(self.ATTN_DIM))
        self.head = nn.Sequential(
            nn.Linear(d + self.feat_dim, 128), nn.GELU(),
            nn.Linear(128, len(self.LABELS)),
        )

    def forward(self, section_embeddings: list[Tensor],
                feat: Optional[Tensor] = None) -> Tensor:
        """Step the SSM over the section embeddings, pool the step outputs, classify.

        Args:
            section_embeddings: one ``[1, 384]`` (or ``[384]``) bge-small embedding
                per section, in document order.
            feat: optional ``[1, feat_dim]`` temporal feature vector (Phase 4)
                concatenated with the pooled embedding. ``None`` is
                backward-compatible: a feat-trained head (``feat_dim > 0``) fed
                ``feat=None`` gets a zeros vector (so old callers / a missing
                feature at serve fall back to the embedding-only signal, not a
                shape error); a feat-less head (``feat_dim == 0``) ignores it.

        Returns:
            Logits ``[1, len(LABELS)]``.

        Resets the recurrent state first (each doc is independent -- no
        cross-doc memory), then steps each section embedding through the serving
        path (``JGSInstance.step``). We reduce the per-section STEP OUTPUTS (the
        learned ``output_proj`` readout, ``[1, 256]`` each) to one ``[1, 256]``
        doc vector -- NOT the raw recurrent state. The step output is the same
        signal ``RetrievalGate`` (Phase 2b, val 0.826) classifies on; pooling
        the raw recurrent state (an earlier design) mode-collapsed on real
        bge-small embeddings of similar enterprise prose (the frozen state was
        not linearly separable for the subtle doc-kind distinctions). The step
        output is the backbone's learned readout, which is. The reduction is
        either a MEAN over sections (the original head -- equal weight, dilutes
        the date-bearing section) or a learned ADDITIVE ATTENTION over sections
        (Phase 5, ``attention_readout=True`` -- finds the date-bearing section
        instead of averaging it away; root cause #3). The temporal feature
        (Phase 4) is concatenated AFTER the reduction -- it re-injects the
        date/framing signal the mean discards (which section carries the date
        that distinguishes a snapshot "as of T" from a decision "made on T").
        """
        if not section_embeddings:
            raise ValueError("DocKindHead.forward called with no section embeddings")
        self.reset_state(1)
        outputs = []
        for emb in section_embeddings:
            out, _pred, _decision = self.step(emb)   # [1, output_dim=256]
            outputs.append(out)
        # Stack per-section step outputs -> [1, N, d], then reduce to [1, d].
        stacked = torch.stack(outputs, dim=1)
        if self.attention_readout:
            # Phase 5: additive attention over sections. score_i = query . key_i
            # -> softmax over sections -> weighted sum. Finds the date-bearing
            # section instead of averaging it away (root cause #3). attn_query
            # zeros-init => uniform weights => == mean-pool at init.
            keys = torch.tanh(self.attn_key(stacked))       # [1, N, ATTN_DIM]
            scores = (keys * self.attn_query).sum(-1)       # [1, N]
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [1, N, 1]
            pooled = (stacked * weights).sum(dim=1)         # [1, d]
        else:
            # Mean over sections -> [1, output_dim]. outputs differ per section
            # (the readout of the SSM state after absorbing that section); the
            # mean is a doc-level summary that retains per-section signal.
            pooled = stacked.mean(dim=1)
        if self.feat_dim > 0:
            if feat is None:
                # Backward-compat: zeros so a feat-trained head still runs from a
                # caller that did not compute the feature (no shape error; the
                # head falls back to the embedding-only signal).
                feat = torch.zeros(1, self.feat_dim, dtype=pooled.dtype,
                                   device=pooled.device)
            pooled = torch.cat([pooled, feat.to(pooled.dtype).to(pooled.device)], dim=-1)
        return self.head(pooled)

    @torch.no_grad()
    def classify(self, section_texts: list[str], embedder,
                 feat: Optional[Tensor] = None) -> Optional[str]:
        """Tag a doc given its section texts -> one of ``LABELS``, or ``None``.

        Embeds each section text via ``embedder`` (bge-small, 384-d), runs
        ``forward``, and returns ``LABELS[argmax]``. Returns ``None`` for empty
        section text so the caller writes the cold-start ``"other"`` default
        (same contract as ``BonsaiDecider.classify_doc_kind`` -> no fabricated
        label). The caller owns the embedder (injected), keeping this module
        torch-only (no ``sentence_transformers`` import here -- mirrors
        ``RetrievalGate.route_text``). ``feat`` is the optional temporal feature
        (Phase 4); the caller computes it from the SAME section_texts.
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
        logits = self.forward(embs, feat=feat)
        idx = int(logits.argmax(dim=-1).item())
        return self.LABELS[idx]