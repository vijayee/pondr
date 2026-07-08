# ADR 006: Working Memory = JGSInstance recurrent state (persistent across queries)

**Status:** Accepted (Phase 2c)
**Date:** 2026-07-07

## Context

Phase 2c needs "continuous awareness" — the engine should not start from zero on
every query. The original Phase 2c draft specified a flat `np.ndarray(8192,)` EMA
state updated as `(1-α)·state + α·embed`, separate from the trained backbone (see
`docs/Phase 2c.md` §0 alignment table).

The real codebase has none of that imagined infrastructure. What it does have, from
Phase 2a, is a trained JEPA-Gated SSM backbone (`backbone_final.pt`, 19.5M params,
ReferenceSSM, `d_model=384`, `d_state=16`) and the `JGSInstance` framework
(`src/subconscious/instance.py`) whose recurrent state is a list of 4 per-layer
tensors `[1, 16, 384]`, detached after each step (no BPTT). The Retrieval Gate
(Phase 2b) is a `JGSInstance` whose state is **reset per query**.

## Decision

Working Memory is a `JGSInstance` (`src/subconscious/working_memory.py`,
`WorkingMemory(JGSInstance)`) configured with `INSTANCE_CONFIGS["working_memory"]`
(rank 8, two context features: `input_novelty`, `state_saturation`). The single
behavioral difference from the Retrieval Gate: **the recurrent state is NOT reset
between queries.** `reset_state` is called only on an explicit `WorkingMemory.reset()`
(a session boundary), never per query.

Retrieved episodes are injected as **embedding-steps** (the episode-summary
embedding stepped into the SSM), not text. The state carries the *gist*; the
primary chunk (Task 2) carries the *detail* of the most-relevant episodes. This
matches the chat framing: "the SSM state is not a context window. It's a
dynamical system whose current activation pattern is the memory in use."

`WorkingMemoryState` is a **type alias** to the shipped `JGSSnapshot`
(`state_serializer.py`) — same fields (`state_tensors` / `input_count` /
`timestamp` / `metadata`). We reuse the serializer's round-trip as the WM
session save/load path rather than duplicate a dataclass.

## Rationale

- **Use the trained backbone's dynamics.** A parallel numpy EMA is a separate
  state machine that ignores the trained SSM. The whole point of training the
  2a backbone is to use its dynamics; mapping WM onto the `JGSInstance` recurrent
  state does that for free (no new training — 2c adds no training cost).
- **Fixed-dimension regardless of conversation length.** The recurrent state is
  `4 × [1, 16, 384]` (~24,576 floats) regardless of how many episodes were
  absorbed. This preserves the draft's "8,192 floats, fixed regardless of
  conversation length" *intent* without the imagined flat vector.
- **Presence vs. per-query reset.** The 2b Retrieval Gate resets per query
  because its job is to classify a single query's pathway. Working Memory's job
  is the opposite — to carry awareness forward — so it must NOT reset. The
  difference is one line (don't call `reset_state` per query) but it is the
  defining behavioral property of the subsystem.
- **Inject as embeddings, not text.** The state is a gist; the detail lives in
  the primary chunk text. Injecting text would require a text encoder the
  backbone doesn't have; injecting the summary embedding reuses the injected
  `Embedder` Protocol and keeps the subconscious package torch-only.
- **`decay_alpha` defaults to 1.0.** The SSM step already mixes the new input
  into the state; a second EMA (`(1-α)·state + α·embed`) would double-apply.
  `decay_alpha < 1.0` is a post-step forget factor `state ← decay_alpha * state`
  for tuning faster forgetting — off by default (rely on SSM dynamics). This is
  a WM-state-tensor lever only; the chat's saturation / "don't overweight
  indefinitely" concern is an edge-level / graph concern that belongs to Phase 3
  GNN consolidation, NOT this knob (docs/Phase 2c.md §13).

## Consequences

- The WM state is the trained backbone's recurrent state — no new training, no
  new params for WM itself (the LoRA adapter `working_memory` already exists from
  2a's instance framework).
- Serialization reuses `state_serializer` (`JGSSnapshot` ↔ JSON/base64 blob,
  element-exact round-trip). Session save/load is file-first (`data/sessions/`)
  with optional WaveDB-backed per-user persistence (`store.save_jgs_state`).
- The save *trigger* policy is intentionally not wired here (the caller decides
  when to save) — documented open in `docs/Phase 2c.md` §15.
- `gate.parameters()` excludes the backbone (inherited `object.__setattr__`
  storage), so any future WM training (REINFORCE) never touches the 2a backbone.

## Alternatives considered

- **Flat numpy EMA (the draft).** Rejected: ignores the trained SSM, parallel
  state machine, imagined infrastructure that doesn't exist.
- **Reset per query (like the Retrieval Gate).** Rejected: defeats the purpose —
  no persistence, no presence.
- **A second EMA on top of the SSM step.** Rejected: double-applies the input
  mix. `decay_alpha` provides forgetting without double-application.