"""Working Memory: continuous awareness via a JGSInstance whose state persists.

Phase 2c. The single behavioral difference from the Retrieval Gate's
``JGSInstance`` is that the recurrent state is **NOT reset between queries**.
``JGSInstance.reset_state`` zeros the state (the 2b trainer calls it per batch);
Working Memory calls it only on an explicit session reset (``WorkingMemory.reset``),
never per query. The state therefore *persists across queries* — the engine has
"presence": the activated subset of long-term memory plus attention (Cowan
embedded-processes model; see docs/Ponder Engine Chat Facts.md §1, chat [002]).

State is the instance's own ``self.state``: a list of 4 per-layer tensors
``[batch=1, d_state=16, d_model=384]``, detached after each ``step()`` (no BPTT,
by ``JGSInstance`` construction). It carries forward across queries.

Memory injection: retrieved episodes are stepped into the SSM as *embeddings*
(the episode summary embedding), not text. The state carries the *gist*; the
primary chunk (Task 2) carries the *detail* of the most-relevant episodes. The
chat: "the SSM state is not a context window. It's a dynamical system whose
current activation pattern is the memory in use" ([002]).

``WorkingMemoryState`` is a type alias to the shipped ``JGSSnapshot``
(``state_serializer.py``) — same fields (``state_tensors`` / ``input_count`` /
``timestamp`` / ``metadata``). We reuse it rather than duplicate a dataclass; the
serializer's round-trip is the WM session save/load path.

This module imports torch (it lives in the torch-only ``subconscious`` package).
The text embedder is *injected* (the ``Embedder`` Protocol from ``routing.py``);
the package stays free of a ``sentence_transformers`` hard dep.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor

from .configs import INSTANCE_CONFIGS, InstanceConfig
from .instance import JGSInstance
from .routing import Embedder
from .state_serializer import JGSSnapshot

# Working Memory state == a JGS snapshot (state tensors + bookkeeping). Reused,
# not duplicated — the serializer already round-trips this exact shape.
WorkingMemoryState = JGSSnapshot


class WorkingMemory(JGSInstance):
    """Continuous-awareness SSM instance. State persists across queries.

    The recurrent state evolves with each ``update``/``inject`` call and is NOT
    zeroed between them. Only an explicit ``reset()`` (a session boundary) zeros
    it. ``snapshot()`` returns detached clones so callers can serialize/restore
    a session without aliasing the live state.
    """

    def __init__(
        self,
        backbone,
        config: Optional[InstanceConfig] = None,
        embedder: Optional[Embedder] = None,
        decay_alpha: float = 1.0,
    ) -> None:
        cfg = config or INSTANCE_CONFIGS["working_memory"]
        super().__init__(backbone, cfg)
        # Injected embedder (bge-small, 384-dim) — may be None if the caller
        # steps the instance manually with pre-computed embeddings (tests do
        # this). Keeping it optional preserves a torch-only import surface.
        self._embedder = embedder
        self.decay_alpha = float(decay_alpha)
        self._input_count = 0
        self._metadata: dict[str, object] = {}

    # ── state evolution ──

    def update(
        self,
        input_embedding: Tensor,
        retrieved_embeddings: Optional[list[Tensor]] = None,
    ) -> WorkingMemoryState:
        """Step the SSM with the query embedding, then inject each retrieved
        episode embedding as a step. State evolves in place; NOT reset.

        Args:
            input_embedding: ``[1, 384]`` (or ``[384]``) — the query embedding.
            retrieved_embeddings: optional list of ``[1, 384]`` episode-summary
                embeddings to absorb as gist after the query step.

        Returns:
            A detached ``WorkingMemoryState`` snapshot (clones; caller-independent
            of the live state).
        """
        self.step(input_embedding)
        self._input_count += 1
        if retrieved_embeddings:
            for emb in retrieved_embeddings:
                self.inject(emb)
        return self.snapshot()

    def inject(self, embedding: Tensor) -> None:
        """One SSM step with ``embedding`` without incrementing ``input_count``.

        Used to absorb retrieved episodes (and, in the chunker, secondary chunks)
        into the recurrent state as gist. Does not reset; mutates ``self.state``.
        """
        self.step(embedding)

    def _apply_decay(self) -> None:
        """Post-step forget factor on the recurrent state.

        ``decay_alpha < 1.0`` multiplies the recurrent state after a step, giving
        a faster forgetting rate than the SSM dynamics alone. Default ``1.0``
        means rely on the ReferenceSSM dynamics (the chat's graceful forgetting
        is a *feature* of the SSM, [002]). This is a WM-state-tensor lever only;
        the chat's saturation / "don't overweight indefinitely" concern
        (diminishing-returns, LLM-mediated importance, boost decay) is an
        edge-level / graph concern that belongs to Phase 3 GNN consolidation —
        NOT this knob (docs/Phase 2c.md §13).
        """
        if self.decay_alpha != 1.0 and self.state is not None:
            self.state = [self.decay_alpha * s for s in self.state]

    def step(self, input_embedding: Tensor, context=None):  # type: ignore[override]
        """Wrap ``JGSInstance.step`` to apply ``decay_alpha`` after each step.

        The SSM step already mixes the new input into the state; ``decay_alpha``
        is an additional global forget factor applied post-step (not a second EMA
        — that would double-apply). Returns the same ``(output, predicted, gate
        decision)`` triple as the base instance.
        """
        result = super().step(input_embedding, context)
        self._apply_decay()
        return result

    # ── snapshot / restore / reset ──

    def snapshot(self, metadata: Optional[dict[str, object]] = None) -> WorkingMemoryState:
        """Detached clones of the current state + bookkeeping.

        Requires the instance to have been stepped at least once (or
        ``reset_state`` called) so ``self.state`` is not ``None``. The returned
        tensors are clones — mutating them or the live state afterward does not
        affect the other.
        """
        if self.state is None:
            raise ValueError(
                "WorkingMemory.state is None — call reset() or update() before "
                "snapshotting so the per-layer shapes/device/dtype are known."
            )
        meta = dict(self._metadata)
        if metadata:
            meta.update(metadata)
        return WorkingMemoryState(
            state_tensors=[t.detach().cpu().contiguous().clone() for t in self.state],
            input_count=self._input_count,
            timestamp=time.time(),
            metadata=meta,
        )

    def restore(self, snapshot: WorkingMemoryState) -> None:
        """Load a snapshot into the live state (session resume).

        Restored tensors are moved to the instance's parameter device/dtype. The
        bookkeeping (``input_count`` / ``metadata``) is restored too, so a
        resumed session continues from the same awareness point.
        """
        if not snapshot.state_tensors:
            raise ValueError("cannot restore an empty snapshot (no state tensors)")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.state = [t.to(device=device, dtype=dtype).contiguous() for t in snapshot.state_tensors]
        self._input_count = snapshot.input_count
        self._metadata = dict(snapshot.metadata)

    def reset(self) -> None:
        """Explicit session-boundary reset → zeros. NOT called per query.

        This is the only place Working Memory zeros its state. Per-query resets
        would defeat the whole point (no persistence / no presence).
        """
        self.reset_state(1)
        self._input_count = 0
        self._metadata = {}

    # ── bookkeeping ──

    @property
    def input_count(self) -> int:
        return self._input_count

    def set_metadata(self, key: str, value: object) -> None:
        """Set a WM metadata field (e.g. ``active_domains``, ``last_query_type``).

        Metadata is carried in snapshots so a resumed session keeps the
        awareness bookkeeping (what domains are active, what the last query was).
        """
        self._metadata[key] = value

    def get_metadata(self, key: str, default: object = None) -> object:
        return self._metadata.get(key, default)

    def embed(self, texts: list[str]) -> list[Tensor]:
        """Embed texts via the injected embedder → ``[1, 384]`` tensors.

        Convenience for callers that have text (query / episode summaries) rather
        than pre-computed embeddings. Raises if no embedder was injected.
        """
        if self._embedder is None:
            raise RuntimeError("WorkingMemory.embed requires an embedder at construction")
        device = next(self.parameters()).device
        vecs = self._embedder.encode(texts)
        return [
            torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for v in vecs
        ]