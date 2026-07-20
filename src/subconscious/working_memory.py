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
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .configs import INSTANCE_CONFIGS, InstanceConfig
from .instance import JGSInstance
from .pin_tag import PinTag
from .routing import Embedder
from .state_serializer import JGSSnapshot

# Working Memory state == a JGS snapshot (state tensors + bookkeeping). Reused,
# not duplicated — the serializer already round-trips this exact shape.
WorkingMemoryState = JGSSnapshot


@dataclass(frozen=True, eq=False)
class RingSlot:
    """One entry in the STRM ring buffer: a step output plus its provenance.

    ``y`` is the step output vector (``[1, output_dim]`` — for the working_memory
    instance, ``output_dim=256``), detached+cloned so the slot is independent of
    the live computation graph and of later steps. ``source_id`` / ``text`` map
    the slot back to the event/episode that produced it, so the context-builder
    (Phase 3) can attend over slot vectors and return the *text* of the selected
    slots rather than a continuous vector. Provenance is optional — recalled
    episodes injected without a source id carry ``None`` (the slot is still
    selectable; it just has no text to surface).

    The slot vector dim is config-driven (``output_dim``), NOT a hardcoded 384:
    the buffer is dimension-agnostic and stores whatever the step emits.

    ``pinned`` (Phase 4 Step 3) marks a slot whose input was a salience-fired
    recall re-injected with the pin tag (``pin=True``). It is per-slot
    bookkeeping ONLY — the pin itself is an input-side token-type embedding
    (``PinTag``), not this flag; we never name a slot flag ``u_t`` (that is the
    SSM input vector). Default ``False``; carried into the replay JSONL so a
    retention surrogate can ask whether pinned slots stay relevant over K steps.
    The ring is in-memory only (NOT in ``snapshot()``/checkpoints), so adding a
    field is checkpoint-backward-compatible.

    ``h`` (state-trajectory rewire) is the per-layer SSM recurrent state that
    produced this slot's ``y``: a list of ``n_layers`` (4) fp16 tensors, each
    ``[1, d_state=16, d_model=384]``, captured post-step/post-decay at
    ``WorkingMemory.step``. This is the SSM *state trajectory* the original
    vision wants a transformer to attend over — ``y`` alone (the output readout)
    was shown NOT to carry query-relevance signal (Probe 4a), so the
    state-trajectory path stores the actual recurrent state, not just the
    readout. fp16 bounds the memory cost (~48 KB/slot, ~750 KB for K=16);
    ``LatentDynamicsHead.project`` casts to fp32 internally, so the projection
    is lossless. ``None`` for slots constructed without a state (tests, partial
    constructions) — ``slot_z`` none-guards. Like ``y``, it is in-memory only
    (NOT in ``snapshot()``/checkpoints), so the field is checkpoint-backward-
    compatible and defaults to ``None`` for any existing positional constructor.
    """

    y: Tensor
    source_id: Optional[str]
    text: Optional[str]
    pinned: bool = False
    h: Optional[list[Tensor]] = None


def slot_z(slot: RingSlot, head) -> Optional[Tensor]:
    """Project a ring slot's stored recurrent state ``h`` to the 384-d ``z_i``.

    The state-trajectory rewire's unit of attention: ``z_i = head.project(h)``
    where ``head`` is a ``LatentDynamicsHead`` (or anything exposing
    ``project(state_tensors) -> [1, 384]`` — last SSM layer, mean over
    ``d_state``). Returns ``None`` when the slot carries no state (``h is None``:
    a test/partial construction, or a cold-start slot) so callers can none-guard
    rather than crash. ``project`` casts to fp32 internally, so fp16 storage is
    lossless here. Kept loose-typed (no ``LatentDynamicsHead`` import) to avoid a
    ``working_memory`` <-> ``latent_dynamics_head`` import cycle.
    """
    if slot.h is None:
        return None
    return head.project(slot.h)


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
        ring_capacity: Optional[int] = None,
        pin_tag: Optional[PinTag] = None,
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
        # STRM ring buffer of recent step outputs with provenance. Capacity is
        # config-driven (``InstanceConfig.ring_capacity``); the ``ring_capacity``
        # kwarg overrides the config for tests. 0 = OFF: no buffer is allocated
        # and step() does no extra work, so the shipped Phase 2c path is
        # byte-identical. K>0 retains the last K slots (FIFO).
        self._ring_capacity = int(ring_capacity) if ring_capacity is not None else int(cfg.ring_capacity)
        self._ring: deque[RingSlot] = deque(maxlen=self._ring_capacity)
        # Phase 4 Step 3: the pin tag. Owned here so a pinned injection
        # (``pin=True``) can add the token-type embedding to ``u_{t+1}`` before
        # the SSM step. Default-initialized (deterministic non-zero) when no
        # trained checkpoint is supplied — a faithful non-stub, NOT a zero
        # vector (see ``pin_tag.py``). Loaded from a checkpoint when wired via
        # ``build_ponder``. The off-path (``pin=False``, the default everywhere
        # except the salience re-inject) never reads this, so it is
        # byte-identical to pre-Step-3.
        self._pin_tag = pin_tag if pin_tag is not None else PinTag()

    # ── state evolution ──

    def update(
        self,
        input_embedding: Tensor,
        retrieved_embeddings: Optional[list[Tensor]] = None,
        source_id: Optional[str] = None,
        text: Optional[str] = None,
        retrieved_sources: Optional[list[tuple[Optional[str], Optional[str]]]] = None,
        pin: bool = False,
    ) -> WorkingMemoryState:
        """Step the SSM with the query embedding, then inject each retrieved
        episode embedding as a step. State evolves in place; NOT reset.

        Args:
            input_embedding: ``[1, 384]`` (or ``[384]``) — the query embedding.
            retrieved_embeddings: optional list of ``[1, 384]`` episode-summary
                embeddings to absorb as gist after the query step.
            source_id / text: optional provenance for the query step — the
                episode id and source text of the input. Carried into the ring
                buffer (when ``ring_capacity > 0``) so the context-builder can map
                a selected slot back to its text. Ignored when the ring is OFF.
            retrieved_sources: optional parallel list of ``(source_id, text)``
                tuples, one per ``retrieved_embeddings`` entry, carrying each
                recalled episode's provenance into its ring slot. ``None`` (the
                default) means injected recalls carry ``None`` provenance.
            pin: Phase 4 Step 3 — when ``True`` AND the ring is ON, add the pin
                tag (a token-type embedding) to each stepped input before the
                SSM step so ``W_A`` retains the episode over the next K steps,
                and mark the resulting ring slots ``pinned=True``. Default
                ``False`` (the prompt-driven path) is byte-identical to
                pre-Step-3. The salience re-inject (Step 5) is the only caller
                that sets ``pin=True``. Ignored when the ring is OFF (no salience
                fires without the ring, so pin is a no-op there — pinned by
                ``test_k0_pin_is_noop``).

        Returns:
            A detached ``WorkingMemoryState`` snapshot (clones; caller-independent
            of the live state).
        """
        self.step(input_embedding, source_id=source_id, text=text, pin=pin)
        self._input_count += 1
        if retrieved_embeddings:
            if retrieved_sources is not None and len(retrieved_sources) != len(retrieved_embeddings):
                raise ValueError(
                    f"retrieved_sources length ({len(retrieved_sources)}) must match "
                    f"retrieved_embeddings length ({len(retrieved_embeddings)}) — a mismatch "
                    "would silently drop or misalign episode steps."
                )
            srcs = retrieved_sources if retrieved_sources is not None else [None] * len(retrieved_embeddings)
            for emb, src in zip(retrieved_embeddings, srcs):
                sid, txt = src if isinstance(src, tuple) else (None, None)
                self.inject(emb, source_id=sid, text=txt, pin=pin)
        return self.snapshot()

    def inject(self, embedding: Tensor, source_id: Optional[str] = None, text: Optional[str] = None, pin: bool = False) -> None:
        """One SSM step with ``embedding`` without incrementing ``input_count``.

        Used to absorb retrieved episodes (and, in the chunker, secondary chunks)
        into the recurrent state as gist. Does not reset; mutates ``self.state``.
        ``source_id`` / ``text`` carry the recalled episode's provenance into the
        ring buffer (when ``ring_capacity > 0``); ignored when the ring is OFF.
        ``pin`` (Phase 4 Step 3) adds the pin tag to the input before the SSM
        step and marks the slot ``pinned=True``; see ``update`` for the full
        semantics. Default ``False`` is byte-identical to pre-Step-3.
        """
        self.step(embedding, source_id=source_id, text=text, pin=pin)

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

    def step(self, input_embedding: Tensor, context=None, source_id=None, text=None, pin: bool = False):  # type: ignore[override]
        """Wrap ``JGSInstance.step`` to apply ``decay_alpha`` and record the ring.

        The SSM step already mixes the new input into the state; ``decay_alpha``
        is an additional global forget factor applied post-step (not a second EMA
        — that would double-apply). When ``ring_capacity > 0``, the step output is
        detached+cloned into the ring buffer with its provenance. The ring append
        is strictly post-step and post-decay; it never touches the state
        computation, so the K=0 path is byte-identical to Phase 2c.

        Phase 4 Step 3: when ``pin=True`` AND the ring is ON, the pin tag (a
        token-type embedding owned by ``self._pin_tag``) is ADDED to
        ``input_embedding`` BEFORE ``super().step``. ``d_model`` stays 384 (an
        ADD, not a concat — a concat would break ``W_A``'s ``Linear(384 -> 16)``).
        The pin is gated on the ring so K=0 + ``pin=True`` stays byte-identical
        (no salience fires without the ring; pinned by ``test_k0_pin_is_noop``).
        The slot is recorded with ``pinned=pin`` so the replay JSONL can flag
        pin-tagged recalls for a retention surrogate.

        Returns the same ``(output, predicted, gate decision)`` triple as the base
        instance (the triple is unchanged; the ring records ``output`` and, when
        the state-trajectory path is in use, the per-layer recurrent state ``h``
        that produced it).
        """
        if pin and self._ring_capacity > 0:
            input_embedding = self._pin_tag(input_embedding)
        result = super().step(input_embedding, context)
        self._apply_decay()
        if self._ring_capacity > 0:
            output, _predicted, _decision = result
            # State-trajectory rewire: snapshot the per-layer recurrent state
            # (post-step, post-decay) alongside the readout. self.state is the
            # live detached [1,16,384]-per-layer list set by JGSInstance.step; we
            # clone to fp16 so the slot is independent of later steps and the
            # memory cost is bounded. A read of self.state only — it never
            # touches the state computation, so K=0 (which skips this block) and
            # the state evolution are byte-identical to Phase 2c.
            h = (
                [s.detach().to(torch.float16).clone() for s in self.state]
                if self.state is not None
                else None
            )
            self._ring.append(
                RingSlot(output.detach().clone(), source_id, text, pinned=pin, h=h)
            )
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
        self._ring.clear()

    # ── STRM read-out: ring buffer + live state ──

    @property
    def ring_capacity(self) -> int:
        """Configured ring capacity (K). 0 = OFF (no buffer, byte-identical)."""
        return self._ring_capacity

    def ring_buffer(self) -> list[RingSlot]:
        """Read-only snapshot of the current ring contents (oldest-first).

        Returns a list (not the internal deque) so callers can iterate without
        the ring mutating under them. Empty when ``ring_capacity == 0`` or before
        any step. The slot tensors are detached clones; mutating them does not
        affect the live state (and the live state does not affect them).
        """
        return list(self._ring)

    def state_tensors(self) -> list[Tensor]:
        """Live per-layer recurrent state for read-only head use.

        Returns ``self.state`` directly — the live, on-device per-layer state
        (one ``[1, d_state=16, d_model=384]`` tensor per SSM layer). The tensors
        are already detached (the SSM detaches after each step — no BPTT), so a
        head treating this as a frozen input feature gets gradients into its own
        params only, not into the (frozen) backbone. This is the head-facing
        accessor; it is distinct from ``snapshot()``, which detaches+clones+CPU-
        copies for serialization.

        Contract: READ-ONLY. Do not mutate the returned list or tensors — the
        SSM steps in place against this state. Raises if ``state`` is ``None``
        (call ``reset()`` or ``update()`` first so shapes/device/dtype are known).
        """
        if self.state is None:
            raise ValueError(
                "WorkingMemory.state is None — call reset() or update() before "
                "reading state_tensors so the per-layer shapes/device/dtype are known."
            )
        return self.state

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