# ADR 007: SSM Chunking — compress into SSM state, primary/secondary split, EXPAND scope

**Status:** Accepted (Phase 2c)
**Date:** 2026-07-07

## Context

The generation model (Bonsai, local llama-server) has a finite context window.
Naively presenting all retrieved episodes as full text either overflows the
window or forces truncation — which silently drops episodes the model never
knows existed. The original Phase 2c draft proposed hierarchical LLM
summarization. EXPAND was also double-specified: as a chunking-level mechanism
here (load full text of a compressed episode on demand) and as a
metacognition-level mechanism in `docs/Ponder Engine Phases.md` Phase 4a
(uncertainty-triggered EXPAND / ADMIT GAP / TOOL USE PLAN).

## Decision

`SSMChunker` (`src/subconscious/ssm_chunker.py`) splits the ranked episode list
(retriever output, already sorted by score) into:

- **primary chunks**: the most-relevant episodes, kept as **full text** (the
  detail), bounded by `max_primary_chunks` (5) and `max_primary_tokens` (4096,
  `len//4` estimate).
- **compressed state**: the remaining episodes, stepped into a *separate,
  ephemeral* `WorkingMemory` compressor as summary embeddings (the gist). The
  formatter exposes the compressed section as the **union of their topics**
  (text), NOT the raw SSM state vector (Bonsai consumes text, not state).

`ChunkedContext` retains the compressed episode dicts (`secondary_episodes`) so
EXPAND can resolve them in-memory before hitting the store. `expandable_ids` is
exactly the compressed set.

**EXPAND** (`SSMChunker.expand` + `ExpandHandler`) is the **chunking-level**
loader: load the full text of a compressed episode on demand and inject it into
working memory as an embedding-step. The **trigger** logic — when to auto-EXPAND
mid-generation on low decoder confidence — is explicitly **Phase 4a**. This ADR
records the handoff so the two phases don't re-implement each other.

## Rationale

- **Compress into SSM state, not LLM-summarize.** An extra LLM call per query is
  latency + cost + nondeterminism. Stepping summary embeddings into the trained
  SSM is deterministic, free (no new training), and reuses the 2a backbone's
  dynamics. The gist is recoverable on demand via EXPAND.
- **Primary/secondary split, not hierarchical summarization.** A flat split
  (top-N full, rest gist) is simpler, deterministic, and matches how humans read
  ("you remember the gist of everything; the exact words of almost nothing;
  EXPAND is going back to the source"). Hierarchical summarization would chain
  LLM calls and compound error.
- **Compressed section = topic union (text), not the state tensor.** Bonsai
  consumes text. Feeding it a 96KB float tensor is meaningless to a text model.
  The topic union tells the model *what* is available compressed, and EXPAND
  lets it pull detail on demand.
- **Ephemeral compressor.** The chunker owns its own `WorkingMemory` (fresh
  state per `chunk()` call) so compressing episodes into gist does not pollute
  the user's persistent working memory. The user's WM is updated separately by
  the orchestrator.
- **EXPAND scope split.** The chunking-level loader (load full text + inject
  into WM) is self-contained and testable now. The metacognitive trigger (when
  the model is unsure and should auto-EXPAND) depends on decoder confidence
  signals that don't exist until Phase 4a. Implementing the trigger now would
  require faking a signal — a de-wonk "stubbed/disabled" flag. So this phase
  ships the loader; Phase 4a adds the trigger.

## Consequences

- Context windows become elastic: primary full text + compressed gist grows
  with the episode count without growing the token budget linearly.
- `expand()` distinguishes three cases: primary id → `EpisodeNotExpandable`
  (already full text), compressed id → resolve from `secondary_episodes` then
  the store, unknown id → `EpisodeNotFound`.
- `ExpandHandler` counts EXPAND invocations (`expand_count`) as the
  Presentation Gate outcome signal (something important was compressed → should
  have been primary; chat [130], docs/Ponder Engine Chat Facts.md §3.1). This
  feeds the deferred learned gate's `ReplayBuffer`.
- The chunker uses the codebase's `len(text)//4` token estimate (no tokenizer
  dep), consistent with `HippocampalRetriever.build_context_string`.

## Alternatives considered

- **LLM hierarchical summarization.** Rejected: extra LLM calls, nondeterminism,
  compounding error.
- **Truncate silently.** Rejected: drops episodes the model never knows existed
  — the user can't ask "did you consider X?" meaningfully.
- **Implement the EXPAND trigger now.** Rejected: requires faking a decoder
  confidence signal that doesn't exist until Phase 4a. Ship the loader; defer
  the trigger (honest, not stubbed).