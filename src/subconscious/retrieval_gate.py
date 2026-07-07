"""RetrievalGate: the first trained JGS instance (Phase 2b).

The subconscious router. Given a query's embedding (and the gate's recurrent
state), it predicts — *before any retrieval* — which domain(s) to query, which
pathway to use, which meta-skills are required, what model size is needed, and
whether conscious deliberation is required. Trained supervised on Oracle JEPA
routing pairs (``training/routing_training.py``), then personalized by
outcome-based REINFORCE.

Architecture vs the ``docs/Phase 2b.md`` draft (see the doc's §0): the draft put
five ``nn.Linear(512, …)`` heads on a cached ``gate.last_output`` attribute.
Reality: the shared backbone runs in **384-dim embedding space** and the instance
``step()`` returns an **output_dim=256** tensor (the ``output_proj`` result).
The heads consume that 256-dim output directly, and ``forward`` returns it
explicitly — there is no ``last_output`` cache (a hidden-state attribute the
trainer reads after the fact is an anti-pattern; the trainer gets the tensor
from the return value).

The text embedder is **injected** (``route_text``), matching the
``Embedder`` Protocol in ``routing.py``. The subconscious package stays
torch-only — no ``sentence_transformers`` import here. The integration layer
(``src/retrieval/retriever.py``) injects the real bge-small embedder.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .configs import INSTANCE_CONFIGS, InstanceConfig
from .gate import GateContext, GateDecision
from .instance import JGSInstance
from .routing import (
    AVAILABLE_DOMAINS,
    Embedder,
    META_SKILLS,
    MODEL_SIZES,
    PATHWAYS,
    RoutingDecision,
)


class RetrievalGate(JGSInstance):
    """The subconscious router. First trained JGS instance.

    Owns five routing heads on top of the shared ``JGSInstance`` base. The
    shared backbone is frozen during instance training (Phase 2a weights); only
    the instance-owned params (input/output projections + LoRA, the decomposed
    gate) and the five routing heads train. ``gate.parameters()`` already
    excludes the backbone (stored via ``object.__setattr__``), so an
    ``AdamW(gate.parameters(), …)`` optimizer naturally leaves the backbone
    alone — the trainer also freezes it explicitly for grad-flow safety.
    """

    def __init__(self, backbone, config: Optional[InstanceConfig] = None):
        cfg = config or INSTANCE_CONFIGS["retrieval_gate"]
        super().__init__(backbone, cfg)
        d = cfg.output_dim  # 256 — the instance step output the heads consume

        # ── Routing heads (trained on Oracle pairs from Phase 1d) ──
        # Domain / pathway / skill share a 256→vocab hidden; model-size and
        # deliberation are smaller (their vocab is tiny). All consume the
        # instance ``output`` (output_dim=256), NOT d_model=384/512.
        self.domain_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(),
            nn.Linear(256, len(AVAILABLE_DOMAINS)),
        )
        self.pathway_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(),
            nn.Linear(256, len(PATHWAYS)),
        )
        self.skill_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(),
            nn.Linear(256, len(META_SKILLS)),
        )
        self.model_size_head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, len(MODEL_SIZES)),
        )
        self.deliberation_head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        prompt_embedding: Tensor,
        context: Optional[GateContext] = None,
    ) -> tuple[dict[str, Tensor], GateDecision, Tensor]:
        """Run the instance step + the five routing heads.

        Args:
            prompt_embedding: ``[batch, input_dim=384]``.
            context: optional ``GateContext`` (3 features: entity_recency,
                topic_recency, query_complexity). Zeros when ``None``.

        Returns:
            ``(logits, gate_decision, output)`` where ``logits`` is
            ``{"domain", "pathway", "skill", "model_size", "deliberation"}``
            each ``[batch, vocab]`` (deliberation is ``[batch, 1]``),
            ``gate_decision`` is the decomposed-gate output, and ``output`` is
            the instance ``step()`` output ``[batch, output_dim=256]`` (returned
            explicitly so the trainer never needs a cached attribute).
        """
        output, _predicted, gate_decision = self.step(prompt_embedding, context)
        logits = {
            "domain": self.domain_head(output),
            "pathway": self.pathway_head(output),
            "skill": self.skill_head(output),
            "model_size": self.model_size_head(output),
            "deliberation": self.deliberation_head(output),
        }
        return logits, gate_decision, output

    def route(
        self,
        prompt_embedding: Tensor,
        context: Optional[GateContext] = None,
    ) -> RoutingDecision:
        """Inference: forward → decode logits → ``RoutingDecision`` (batch=1)."""
        logits, gate_decision, _output = self.forward(prompt_embedding, context)
        return self.decode_batch(logits, gate_decision)[0]

    def route_text(
        self,
        prompt: str,
        embedder: Embedder,
        context: Optional[GateContext] = None,
    ) -> RoutingDecision:
        """Embed ``prompt`` via the injected ``embedder`` then ``route``.

        The embedder is the caller's responsibility (the integration layer
        passes the real bge-small ``VectorSearch`` embedder; tests pass a
        deterministic stub). Keeping it out of the constructor preserves the
        package's torch-only import surface.
        """
        vec = embedder.encode([prompt])[0]
        emb = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # [1, 384]
        device = next(self.parameters()).device
        # Move the context's feature tensor to the gate's device too (a CPU
        # context against a CUDA gate would mismatch inside DecomposedGate).
        if context is not None and context.features is not None:
            context.features = context.features.to(device)
        return self.route(emb.to(device), context)

    # ── decoding helpers ──

    def decode_batch(
        self,
        logits: dict[str, Tensor],
        gate_decision: GateDecision,
    ) -> list[RoutingDecision]:
        """Decode batched head logits → one ``RoutingDecision`` per row.

        ``logits["domain"|"pathway"|"skill"|"model_size"]`` are ``[batch, vocab]``
        and ``logits["deliberation"]`` is ``[batch, 1]``. Domain: multi-label at
        0.3, falling back to the single argmax per row so no route is vacuous.
        Skills: multi-label at 0.5 (may be empty — auxiliary). Pathway /
        model_size: argmax. The gate returns ONE ``GateDecision`` (its live
        contract is one decision per step), so every row shares that decision's
        confidence — batched evaluation uses the per-row logits for the discrete
        choices and the shared gate confidence for the ``confidence`` field.
        """
        batch = logits["domain"].shape[0]
        dom_probs = torch.sigmoid(logits["domain"])          # [B, n_domains]
        path_idx = torch.softmax(logits["pathway"], dim=-1).argmax(dim=-1)
        skill_probs = torch.sigmoid(logits["skill"])
        size_idx = torch.softmax(logits["model_size"], dim=-1).argmax(dim=-1)
        delib = (torch.sigmoid(logits["deliberation"]).squeeze(-1) > 0.5)

        decisions: list[RoutingDecision] = []
        for b in range(batch):
            active = (dom_probs[b] > 0.3).nonzero(as_tuple=True)[0].tolist()
            if not active:
                active = [int(dom_probs[b].argmax().item())]
            domains = [AVAILABLE_DOMAINS[i] for i in active]
            skill_active = (skill_probs[b] > 0.5).nonzero(as_tuple=True)[0].tolist()
            meta_skills = [META_SKILLS[i] for i in skill_active]
            decisions.append(RoutingDecision(
                domains=domains,
                pathway=PATHWAYS[int(path_idx[b].item())],
                meta_skills=meta_skills,
                model_size=MODEL_SIZES[int(size_idx[b].item())],
                needs_deliberation=bool(delib[b].item()),
                confidence=float(gate_decision.confidence),
                gate_decision=gate_decision,
            ))
        return decisions