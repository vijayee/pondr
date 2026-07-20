# Phase 2c: Working Memory & Presentation — Implementation Plan

**Ponder Engine · Architecture v2.0 · July 2026**

> **Terminology:** "Phase 2c" here is the same body of work that the older Phase 1b/1c/2b
> docs and `src/` comments call **"Phase 2.5."** This doc standardizes on "Phase 2c" to match
> `docs/Ponder Engine Phases.md`. No new training cost is added by this phase (see §9.1).

---

## 0. Alignment Notes — doc vs. the real codebase

The original draft of this doc was written against an imagined API. This section records every
correction, the way `docs/Phase 2b.md` §0 does. Implementation follows the **Reality** column.

| Original draft said | Reality (this doc corrects to) |
|---|---|
| Package layout `ponder/engine/memory/…`, `ponder/models/jgs_backbone.py` | Real code is `src/`. New files go under `src/subconscious/` and `src/retrieval/`; modifications to existing `src/retrieval/retriever.py`, `src/generation/mode_a.py`, `src/subconscious/configs.py`. |
| `WorkingMemoryState.vector: np.ndarray (8192,) float32`; `input_embedding: (4096,)` | Real backbone is **torch**, `d_model=384`, instance `output_dim=256`, recurrent state = `list` of 4 per-layer tensors `[batch, d_state=16, d_model=384]`. There is no 8192-dim or 4096-dim tensor anywhere. Working Memory state = the `JGSInstance` recurrent state (config-driven, ~24,576 floats = 4×16×384), **persisted across queries** (the key 2c behavior — *not* reset per query like the Retrieval Gate). The "8,192 floats, fixed regardless of conversation length" *intent* is preserved: the recurrent state is fixed-dimension regardless of conversation length. |
| `backbone.load_lora("working_memory", rank=8)`, `backbone.embed(text)`, `backbone.ssm_step(state, embedding)` | None of these exist. LoRA lives in `src/subconscious/lora.py` (`StateLoRA`), instantiated by `JGSInstance` from `INSTANCE_CONFIGS`. Embedding is via the **injected `Embedder` Protocol** (`encode(list[str]) -> list[list[float]]`, 384-dim bge-small-en-v1.5) — the caller injects it; the subconscious package stays torch-only. The SSM step is `JGSInstance.step(input_embedding, context)`; there is no numpy `ssm_step`. |
| Everything in `np.ndarray` | Everything in `torch.Tensor`. Serialization flattens tensors to bytes (and back) via `StateSerializer`. |
| `presentation_gate` is a declared instance (`backbone.load_lora("presentation_gate", rank=4)`) | `presentation_gate` is **not** in `INSTANCE_CONFIGS` (`src/subconscious/configs.py` declares 8 cognitive instances; presentation is not one). There is also **no supervised training data** for it (no Oracle pairs — only outcome signals, which are not wired live yet). Per the draft's own Risk Register, plan() falls back to a heuristic when gate confidence is low. This doc makes the heuristic the **real** production strategy-selection logic (episode-count + query-specificity thresholds from config) and wires `record_outcome` to a `ReplayBuffer` for a **future** learned gate. No untrained/dead params. The learned JGS Presentation Gate is **deferred** (no data to train it on) — documented, not faked. Mirrors Phase 2b's "routed but not executed" honesty. |
| `WorkingMemory` config three-way mismatch (doc: rank 8 only; Phase 2a doc: stale 512/512/512; code: 384/256/rank-8/2-context-features) | Use the real code: `INSTANCE_CONFIGS["working_memory"]` = `input_dim=384, output_dim=256, d_state=16, lora_rank=8, gate_config=GateConfig(num_context_features=2)`. The 2 context features are `input_novelty` and `state_saturation` (not the Retrieval Gate's `entity_recency/topic_recency/query_complexity`). |
| `HippocampalRetriever(store, retrieval_gate, working_memory, planner_model)` | Real: `HippocampalRetriever(store, planner=None, auto_load_index=False, retrieval_gate=None, embedder=None)` — there is **no `working_memory` constructor param** (Phase 2b §0 records this). Working Memory is composed at the orchestrator layer (new `src/orchestrator.py`), not threaded into the retriever. |
| `QueryPlanner` class with `plan(user_prompt, working_memory, conversation_history)` | There is no `QueryPlanner` class. Planning is `retriever.planner.plan(prompt, conversation_history) -> dict` (Bonsai llama-server), returning `{entities, entity_mode, temporal_intent, …}` with **no `domains` key**. Prompt compression (Task 5) is a pre-step that produces a **text** prompt ≤ `bonsai_max_input` chars; Bonsai consumes text, not a state vector. |
| `ContextFormatter`, `end_state`, `consumer` ("openai_chat"/"anthropic"/"generic_llm") | No `ContextFormatter` exists. The real formatter is `HippocampalRetriever.build_context_string` (`src/retrieval/retriever.py`, summary-only, `len//4` token estimate, hard cutoff at `max_context_tokens=4000`, `[Episode ID | Date] + Entities/Topics/Tone/Summary`). The only live consumer is the local Bonsai model. This doc keeps the `consumer` parameter for forward-compat but produces one format now (the structured format in §6.2). |
| Token counts via `episode.token_count` | No tokenizer exists in `src/` (only the `len(text)//4` estimate). Reuse that estimate everywhere; no new tokenizer dep. |
| EXPAND as a Presentation/chunking mechanism (Task 4) | EXPAND is double-specified: here (chunking-level — load full text of a compressed episode on demand) and in `Ponder Engine Phases.md` Phase 4a (metacognition-level — uncertainty-triggered EXPAND/ADMIT GAP/TOOL USE PLAN). This doc implements the **chunking-level** EXPAND (the compressed→full-text loader + working-memory injection). The **trigger** logic (when to auto-EXPAND) is Phase 4a. Noted explicitly so the two phases don't re-implement each other. |
| `DecomposedGate(ValueHead(400_000), CostHead(500_000), …)` ~1.5M params | The real `DecomposedGate` (`src/subconscious/gate.py`) is ~484K params; head sizes are code-defined, not the draft's numbers. The Presentation Gate (deferred-learned) would reuse this framework **if** it becomes a JGS instance later. The heuristic planner needs no gate. |
| YAML config `ponder/config/phase2_config.yaml` | Config is dataclass-based (`src/config.py`, `src/subconscious/configs.py`), not YAML. Add a `Phase2cConfig` dataclass + a `presentation_gate` entry to `INSTANCE_CONFIGS`. |
| Cost table has no Phase 2c row | Correct: 2c adds **no training cost**. The backbone is already trained (Phase 2a); Working Memory and the SSM Chunker are runtime-only (no training); the Presentation Gate is heuristic (learned gate deferred). The only compute is local CPU/GPU inference, already budgeted. This doc states that explicitly. |

### What does **not** exist yet (greenfield for 2c)

`src/subconscious/working_memory.py`, `ssm_chunker.py`, `presentation_gate.py`,
`state_serializer.py`, `src/retrieval/chunked_context.py`, `expand_handler.py`,
`src/orchestrator.py` — none exist. `INSTANCE_CONFIGS["presentation_gate"]` does not exist.
Confirmed by glob.

---

## 1. Overview

Phase 2c deploys three coupled subsystems that give the engine **continuous awareness** and
**adaptive context presentation**:

| Subsystem | What it does | Why it matters |
|---|---|---|
| **Working Memory** | A `JGSInstance` whose recurrent state **persists across queries** (not reset per query) and absorbs retrieved episodes as embedding-steps | The system no longer starts from zero on each query — it has *presence* |
| **SSM Chunking** | Compresses less-relevant retrieved episodes into the SSM recurrent state instead of raw text | Context windows are finite; this makes them *elastic* |
| **Presentation Gate** | Decides **how to present** retrieved context along **two axes** — (a) *chunking strategy* (direct / chunked / summary_only + chunk counts) and (b) *end state* (return results directly / format for another consumer / synthesize via LLM / extract structured data) | The system *decides how to present*, not just *what to retrieve* — and not every retrieval ends in an LLM call |

They are built and tested together: Working Memory provides the state the Presentation Gate
reads, and SSM Chunking is the mechanism the Presentation Gate orchestrates. The Presentation
Gate's **end-state axis** (b) is the chat's "do we always want the LLM to process the
results?" answer — sometimes the results *are* the answer (direct return, no LLM), sometimes
they are context for another consumer (format), sometimes they need LLM reasoning (synthesize),
sometimes they need transformation to structured data (extract). Per the chat, this is an
**explicit API with a heuristic default**: the caller may specify the end state; when they
don't, the gate picks a default; a caller override is recorded as a training signal for the
deferred learned gate.

### Prerequisites (status)

- [x] Phase 1b — `retrieve()`, context-string builder, graph traversal, vector search
- [x] Phase 2a — shared JEPA-Gated SSM backbone trained and loadable (`backbone_final.pt`,
      19.5M params, ReferenceSSM backend, `d_model=384, d_state=16, pred_dim=384`)
- [x] Phase 2b — Retrieval Gate instance trained (best val 0.826) and routing via
      `retrieve_with_routing`
- [x] LoRA adapter framework from 2a (`StateLoRA`, `INSTANCE_CONFIGS["working_memory"]` rank 8)
- [ ] `presentation_gate` instance config — **added by this phase** (Task 7), heuristic-only

---

## 2. File Map

All paths corrected to the real `src/` layout. `[MODIFY]` = existing file changed.

```plaintext
src/subconscious/
├── working_memory.py            # NEW — WorkingMemory(JGSInstance), persistent state
├── ssm_chunker.py                # NEW — SSMChunker, ChunkedContext, compress_episodes, expand
├── presentation_gate.py         # NEW — PresentationGate + PresentationPlan + EndStatePlan
│                                #        (heuristic chunking axis §5.2-5.4 + end-state axis §5.6)
├── state_serializer.py          # NEW — flatten/unflatten torch recurrent state <-> bytes
├── configs.py                    # [MODIFY] — add INSTANCE_CONFIGS["presentation_gate"]
└── (instance.py, backbone.py, gate.py, lora.py, ssm.py — used as-is)
src/retrieval/
├── retriever.py                  # [MODIFY] — add build_with_chunking(); reuse build_context_string
├── chunked_context.py            # NEW — ChunkedContext dataclass + ChunkedContextFormatter
├── expand_handler.py             # NEW — ExpandHandler (chunking-level EXPAND)
└── end_state.py                 # NEW — direct/format/synthesize/extract dispatch + extract handler
src/generation/
└── mode_a.py                     # [MODIFY] — add generate_with_working_memory()
src/
├── orchestrator.py               # NEW — PonderOrchestrator: composes WM + chunker + gate + retriever
└── config.py                     # [MODIFY] — add Phase2cConfig dataclass
tests/
├── test_working_memory.py        # NEW
├── test_ssm_chunker.py           # NEW
├── test_presentation_gate.py     # NEW — chunking axis (§5.2-5.4) + end-state axis (§5.6)
├── test_end_state.py            # NEW — direct/format/synthesize/extract dispatch + override buffer
├── test_chunked_context.py       # NEW
├── test_expand_handler.py        # NEW — focused ExpandHandler unit tests
└── test_orchestrator.py          # the Task-8 integration scenarios live here
                                  # (folded in to avoid duplicating the shared
                                  # _orchestrator/_ep/_Stub* helpers; see §10)
docs/adr/
├── 006-working-memory-design.md  # NEW
└── 007-ssm-chunking-strategy.md  # NEW
```

---

## 3. Task 1 — Working Memory Core

**File:** `src/subconscious/working_memory.py`

### 3.1 Design (corrected)

`WorkingMemory` is a `JGSInstance` configured with `INSTANCE_CONFIGS["working_memory"]`
(rank 8, 2 context features: `input_novelty`, `state_saturation`). The single behavioral
difference from the Retrieval Gate: **the recurrent state is NOT reset between queries.**
`JGSInstance.reset_state` zeros the state (used by the 2b trainer per-batch); Working Memory
calls it **only** on an explicit session reset, never per query.

State is the instance's own `self.state: list[Tensor]` — 4 per-layer tensors
`[batch=1, d_state=16, d_model=384]`, detached after each `step()` (no BPTT, by `JGSInstance`
construction). This carries forward across queries.

```python
@dataclass
class WorkingMemoryState:
    """A serializable snapshot of the continuous recurrent state + bookkeeping."""
    state_tensors: list[torch.Tensor]   # 4 × [1, 16, 384], detached clones
    input_count: int
    timestamp: float                     # time.time() at last update
    metadata: dict[str, Any]             # {"active_domains": [...], "last_query_type": "..."}
```

> No flat `(8192,)` vector — the recurrent state is the structured per-layer tensor list. The
> fixed-dimension-regardless-of-conversation-length property is preserved (the shape is
> config-driven, not length-driven). Serialization (Task 6) flattens these tensors.

```python
class WorkingMemory(JGSInstance):
    """Continuous-awareness SSM instance. State persists across queries."""

    def __init__(self, backbone, config=INSTANCE_CONFIGS["working_memory"],
                 embedder: Optional[Embedder] = None):
        super().__init__(backbone, config)
        self._embedder = embedder     # injected; may be None (caller steps manually)
        self._input_count = 0
        self._metadata: dict[str, Any] = {}

    def update(self, input_embedding: Tensor,          # [1, 384]
               retrieved_embeddings: Optional[list[Tensor]] = None
               ) -> WorkingMemoryState:
        """Step the SSM with the query, then inject each retrieved episode as a step.
        State evolves in place; NOT reset. Returns a detached snapshot."""

    def inject(self, embedding: Tensor) -> None:
        """One SSM step with `embedding` without incrementing input_count."""

    def snapshot(self) -> WorkingMemoryState:
        """Detached clones of the current state + bookkeeping."""

    def reset(self) -> None:
        """Explicit session-boundary reset → zeros. NOT called per query."""
        self.reset_state(1)
        self._input_count = 0
        self._metadata = {}
```

### 3.2 Memory injection protocol (corrected)

Retrieved episodes are injected **as embedding-steps**, not text:

```
query     → Embedder.encode → Tensor[1,384] → instance.step (state evolves)
episode 1 → embed(summary)  → Tensor[1,384] → instance.step (state absorbs)
episode 2 → embed(summary)  → Tensor[1,384] → instance.step
...
final recurrent state encodes: query + all retrieved episodes (gist)
```

The generation model receives the Working Memory **snapshot** (flattened/serialized for the
context preamble) **in addition to** the primary chunk text. The state carries the *gist*;
the primary chunk carries the *detail* of the most-relevant episodes.

### 3.3 Decay

The SSM step provides the primary state evolution (ReferenceSSM dynamics). An optional
explicit forget factor `decay_alpha` (default **1.0** = rely on SSM dynamics) is applied
post-step as `state ← decay_alpha * state` on the recurrent tensors when a faster forgetting
rate is desired. The draft's numpy EMA `(1-α)·state + α·embed` is **not** used — the SSM step
already mixes the new input into the state; a second EMA would double-apply. `decay_alpha`
is a config knob for tuning forgetting, off by default.

### 3.4 Acceptance criteria

- [ ] `WorkingMemoryState` round-trips through `StateSerializer` (Task 6) byte-faithfully.
- [ ] State shape is unchanged after any number of updates (4 × `[1, 16, 384]`).
- [ ] `update(q)` then `inject(m1)`, `inject(m2)` produces a state ≠ `update(q)` alone.
- [ ] Deterministic given the same input sequence (no random noise; ReferenceSSM is deterministic).
- [ ] `reset()` returns state to zeros; `update` does **not** reset (state persists across
      consecutive `update` calls — verified by asserting state[−1] changes monotonically).
- [ ] 100 consecutive updates (CPU, ReferenceSSM, batch=1) complete in < 50ms (CPU-realistic;
      the draft's <10ms assumed numpy; torch CPU is slower — see §9.2).
- [ ] `gate.parameters()` excludes the backbone (inherited from `JGSInstance` —
      `object.__setattr__` storage), so working-memory training/REINFORCE never touches 2a.

**Test file:** `tests/test_working_memory.py`

---

## 4. Task 2 — SSM Chunker

**File:** `src/subconscious/ssm_chunker.py`

### 4.1 ChunkedContext (corrected)

```python
@dataclass
class ChunkedContext:
    primary_chunks: list[dict]            # full-text episodes: {episode_id, text, date, entities, topics, tones}
    compressed_state: WorkingMemoryState  # SSM state encoding all secondary episodes
    chunk_map: dict[str, int]            # episode_id → 0 (primary idx) or -1 (compressed)
    expandable_ids: set[str]             # episode IDs available for EXPAND (the compressed ones)
    total_episodes: int
    primary_token_count: int             # len//4 estimate, summed over primary
    compressed_episode_count: int
```

> `compressed_state` is a `WorkingMemoryState` (torch tensors), **not** `np.ndarray(8192,)`.

### 4.2 SSMChunker (corrected)

```python
class SSMChunker:
    def __init__(self, backbone, embedder: Embedder, config: Phase2cConfig):
        self.backbone = backbone
        self.embedder = embedder                 # injected, 384-dim
        self._compressor = WorkingMemory(backbone, INSTANCE_CONFIGS["working_memory"], embedder)
        self.max_primary_tokens = config.ssm_chunker.max_primary_tokens   # 4096
        self.max_primary_chunks = config.ssm_chunker.max_primary_chunks   # 5

    def chunk(self, episodes: list[dict],         # retriever output dicts (have "text"/"episode_id")
              presentation_plan: PresentationPlan) -> ChunkedContext: ...

    def compress_episodes(self, episodes: list[dict]) -> WorkingMemoryState:
        """Embed each episode summary, step the compressor SSM sequentially.
        Returns the final recurrent state (gist of all secondary episodes)."""

    def expand(self, episode_id: str, store) -> dict:
        """EXPAND: load full text of a compressed episode on demand (chunking-level).
        Raises EpisodeNotExpandable if episode_id is in primary_chunks (already full text),
        EpisodeNotFound if unknown. The trigger logic (when to auto-EXPAND) is Phase 4a."""
```

### 4.3 Algorithm

```
Input: episodes (already sorted by retrieval relevance — GraphTraversal._score_candidates),
       presentation_plan, max_primary_tokens, max_primary_chunks
1. primary_chunks = []; token_count = 0
2. For each episode in order (up to presentation_plan.primary_chunk_count):
   a. tok = len(episode["text"]) // 4
   b. if token_count + tok <= max_primary_tokens AND len(primary_chunks) < max_primary_chunks:
        → primary_chunks.append(...); chunk_map[id] = idx; token_count += tok
   c. else: → secondary_pool.append(episode)
3. compressed_state = compress_episodes(secondary_pool)
4. expandable_ids = {ids in secondary_pool}; return ChunkedContext
```

### 4.4 Acceptance criteria

- [ ] Primary chunks never exceed `max_primary_tokens` (len//4) or `max_primary_chunks` count.
- [ ] `compressed_state.state_tensors` is always 4 × `[1, 16, 384]`.
- [ ] `expand()` returns full text for any id in `expandable_ids`.
- [ ] `expand()` raises `EpisodeNotExpandable` for primary ids; `EpisodeNotFound` for unknown ids.
- [ ] Chunking 100 episodes (CPU) completes in < 200ms; compressing 50 episodes < 300ms
      (torch CPU, batch=1 per step — see §9.2 for why the draft's <50ms/<100ms are unrealistic).

**Test file:** `tests/test_ssm_chunker.py`

---

## 5. Task 3 — Presentation Gate

**File:** `src/subconscious/presentation_gate.py`

### 5.1 Why heuristic (not a trained JGS instance)

The draft designs a trained JGS Presentation Gate. There is **no supervised training data**
for it (no Oracle pairs) and outcome signals (EXPAND frequency, unused-primary count, user
satisfaction) are **not wired live** yet. Per the draft's own Risk Register, plan() should
fall back to a heuristic when gate confidence is low — which is always, until outcome data
exists. Rather than instantiate an untrained JGS instance whose learned heads are dead params
(a de-wonk "weird/dead" flag), this phase makes the heuristic the **real** strategy-selection
logic and wires `record_outcome` to a `ReplayBuffer` for a future learned gate. The learned
JGS Presentation Gate is **deferred** to when outcome signals are live. This mirrors Phase 2b's
"routed but not executed / honestly flagged `supported`" precedent.

### 5.2 PresentationPlan

```python
@dataclass
class PresentationPlan:
    strategy: str            # "direct" | "chunked" | "summary_only"
    primary_chunk_count: int
    primary_chunk_size: int  # max tokens per primary chunk (len//4)
    compressed_chunk_count: int
    expand_threshold: float  # confidence threshold for auto-EXPAND (Phase 4a trigger reads this)
    rationale: str           # human-readable, for debugging

@dataclass
class PresentationOutcome:
    expand_count: int
    unused_primary_count: int
    user_satisfaction: float
```

### 5.3 PresentationGate (heuristic)

```python
class PresentationGate:
    """Heuristic presentation planner + outcome buffer for a future learned gate."""
    def __init__(self, config: Phase2cConfig, embedder: Optional[Embedder] = None):
        self.cfg = config.presentation_gate
        self.replay = ReplayBuffer(capacity=config.replay_capacity)  # for future training

    def plan(self, query: str, retrieved_episodes: list[dict],
             working_memory: Optional[WorkingMemoryState] = None,
             retrieval_gate_pathway: Optional[str] = None) -> PresentationPlan:
        """Heuristic: direct if episodes <= direct_max (3) AND query is specific;
        summary_only if episodes >= summary_only_min (20) OR query is a summarization;
        else chunked. Chunk count = min(episodes, max_primary_chunks). Deterministic."""

    def record_outcome(self, plan: PresentationPlan, outcome: PresentationOutcome) -> None:
        """Store (plan, outcome) in the replay buffer. No-op learning until a learned gate
        is added in a later phase; the buffer is the training-data seed."""
```

### 5.4 Strategy selection

| Strategy | When (heuristic) | What happens |
|---|---|---|
| `direct` | episodes ≤ `direct_max_episodes` (3) and query is specific (not a summarization verb) | No chunking; all episodes as full text |
| `chunked` | `direct_max` < episodes < `summary_only_min` (5–19) and a specific query | Top-N primary (full text), rest compressed into SSM state |
| `summary_only` | episodes ≥ `summary_only_min` (20) OR query is a summarization | All compressed; only SSM state + topic summary passed |

Query-specificity is a cheap heuristic (presence of summarization keywords — "summarize",
"overview", "everything about" — flips toward `summary_only`).

### 5.5 Acceptance criteria

- [ ] `plan()` returns a valid `PresentationPlan` for any input (incl. empty episodes).
- [ ] `direct` when episodes ≤ 3 and no summarization keyword.
- [ ] `chunked` for 5 < episodes < 20 and a specific query.
- [ ] `summary_only` when episodes ≥ 20 OR a summarization keyword present.
- [ ] `primary_chunk_count` ≤ `max_primary_chunks`.
- [ ] `record_outcome()` appends to the buffer (verify buffer length grows).
- [ ] `plan()` is deterministic for identical inputs.
- [ ] `plan()` (heuristic, no model) completes in < 1ms.

**Test file:** `tests/test_presentation_gate.py`

### 5.6 End-state routing axis (direct / format / synthesize / extract)

The chunking `strategy` above is **axis (a)**. The chat (`[144]`/`[146]`) adds a second,
orthogonal **axis (b)**: *what to do with the retrieved results*. Not every retrieval
should end in an LLM call. Four end states:

| End state | When | LLM call? | Output |
|---|---|---|---|
| `direct` | The results *are* the answer ("What did Alice say about X?" → return the episode; "Show me the conversation about Y") | **No** | Episodes/summaries returned as-is |
| `format` | Results are context for another consumer (a different model, a code generator, a tool) | No (formatting only) | A formatted context string/structure for the named consumer |
| `synthesize` | Results need reasoning across episodes ("Why did we choose X over Y?") | Yes | Bonsai (or a larger model) generates a response |
| `extract` | Results need transformation to structured data ("List all decisions as JSON"; "Create a dependency graph") | Cheap local extractor (not Bonsai) | Structured data (list / graph / table) |

**Why an explicit API with a heuristic default (not a learned router).** Per `[145]`/`[146]`:
the feedback signal for end-state routing is weak (the difference between `direct` and
`synthesize` is latency/cost/format, not correctness — the user may be satisfied with
either), so inferring an unobservable preference from implicit signals is fragile. The
chat's resolution: **the API is the interface, JEPA is the optimization.** The caller may
specify `end_state` explicitly; when they don't, a heuristic picks a default; a caller
**override** is recorded and becomes the training signal for the future learned gate
(same `ReplayBuffer` as the chunking outcomes — one buffer, two label fields).

```python
EndState = str  # "direct" | "format" | "synthesize" | "extract"

@dataclass
class EndStatePlan:
    end_state: EndState
    format_spec: Optional[dict] = None    # for "format": {"consumer": "bonsai"|"claude"|..., "purpose": ..., "max_tokens": ...}
    extract_schema: Optional[dict] = None # for "extract": {"type": "list"|"graph"|"table", "item_type": ...}
    model_size: Optional[str] = None      # for "synthesize": "bonsai" (only live model now)
    jepa_default: bool = True             # True if the gate picked it; False if the caller overrode
    rationale: str = ""

class PresentationGate:
    # ... (chunking axis from §5.2-5.4 unchanged) ...

    def plan_end_state(self, query: str, retrieved_episodes: list[dict],
                       working_memory: Optional[WorkingMemoryState] = None,
                       caller_end_state: Optional[EndState] = None) -> EndStatePlan:
        """Heuristic default when caller_end_state is None; else honor the caller.
        Heuristic: 'direct' for show/who/when factual lookups with ≤3 episodes;
        'extract' for list/graph/json verbs; 'synthesize' for why/how/compare reasoning
        or >3 episodes; 'format' only when a non-bonsai consumer is named. Deterministic."""

    def record_override(self, query: str, episodes: list[dict],
                        jepa_predicted: EndState, caller_chose: EndState) -> None:
        """Caller overrode the default → push (query, episodes-sig, predicted, chose) to the
        ReplayBuffer as a training signal for the deferred learned end-state router. No-op
        learning until that learned head exists; the buffer is the seed."""
```

**Integration with the orchestrator.** The orchestrator's `query()` (§8.1) accepts an
optional `end_state` (+ `format_spec` / `extract_schema` / `model_size`). For `direct` it
returns **without any model call**; for `format` it returns the formatted context (no model);
for `extract` it uses a **cheap local span/regex extractor** (the same one §7 prompt
compression uses — GLiNER if available, else a regex/titlecase fallback), **not** Bonsai;
only `synthesize` calls the generation model. This is the first Hippo retrieval path that can
answer without invoking Bonsai.

**Acceptance criteria (axis b)**

- [ ] `plan_end_state()` returns a valid `EndStatePlan` for any input (incl. empty episodes).
- [ ] Heuristic default is deterministic for identical inputs.
- [ ] `caller_end_state` is always honored when provided (`jepa_default=False`).
- [ ] `record_override()` is called only on an actual override and appends to the buffer.
- [ ] `direct` for a show/who/when factual lookup with ≤3 episodes (no LLM).
- [ ] `extract` for a list/graph/json verb.
- [ ] `synthesize` for a why/how/compare query or >3 episodes.
- [ ] `plan_end_state()` (heuristic, no model) completes in < 1ms.

**Test file:** `tests/test_presentation_gate.py` (extended)

---

## 6. Task 4 — Mode A Generation with Chunking

**Files:** `src/retrieval/chunked_context.py` (NEW), `src/retrieval/expand_handler.py` (NEW),
`src/retrieval/retriever.py` [MODIFY], `src/generation/mode_a.py` [MODIFY]

### 6.1 ChunkedContextFormatter

```python
class ChunkedContextFormatter:
    def format_for_llm(self, chunked: ChunkedContext,
                       consumer: str = "bonsai",                 # only live consumer now
                       working_memory: Optional[WorkingMemoryState] = None) -> str:
        """Produce the context string for the generation model.
        Sections: [RETRIEVED CONTEXT — PRIMARY] full text + metadata;
        [COMPRESSED CONTEXT — SUMMARY] topic list derived from secondary episodes' topics
        (NOT the raw state vector); [WORKING MEMORY STATE] active domains/recent topics
        from working_memory.metadata. EXPAND instructions appended."""
```

### 6.2 Context format

```plaintext
[RETRIEVED CONTEXT — PRIMARY]
The following conversations are directly relevant to your query:

--- Episode ep_001 (2026-06-15) ---
Entities: Alice, Postgres
Topics: database performance, indexing
Tone: technical
[full text]

[COMPRESSED CONTEXT — SUMMARY]
The following topics are available in compressed form. If you need specific details,
use EXPAND(episode_id) to retrieve full text.
Compressed topics: <union of secondary episodes' topics>

[WORKING MEMORY STATE]
Current conversation focus: <working_memory.metadata["last_query_type"]>
Active domains: <working_memory.metadata["active_domains"]>
```

### 6.3 ExpandHandler (chunking-level EXPAND)

```python
class ExpandHandler:
    def handle_expand(self, episode_id: str, chunked: ChunkedContext,
                      working_memory: WorkingMemory, store,
                      embedder: Embedder) -> tuple[str, WorkingMemoryState]:
        """Load full text of a compressed episode (via SSMChunker.expand) and inject it
        into working memory as a step. Returns (full_text, updated_snapshot).
        The trigger logic (when to auto-EXPAND) is Phase 4a — this is the loader only."""
```

### 6.4 Retriever integration

Add to `HippocampalRetriever`:

```python
def build_with_chunking(self, query, episodes: list[dict],
                        presentation_plan: PresentationPlan,
                        working_memory: WorkingMemory,
                        consumer: str = "bonsai") -> tuple[str, ChunkedContext]:
    """1. SSMChunker.chunk(episodes, plan) → ChunkedContext
       2. ChunkedContextFormatter.format_for_llm(chunked, consumer, working_memory.snapshot())
       3. return (context_string, chunked)  # chunked is kept for later EXPAND"""
```

`retrieve()` and `retrieve_with_routing()` are **unchanged** (back-compat).

### 6.5 Mode A integration

Add to `ModeAGenerator`:

```python
def generate_with_working_memory(self, prompt, conversation_history=None,
                                 session: Optional[PonderOrchestrator] = None) -> dict:
    """Full 2c pipeline (see Task 8). Returns
    {response, route, retrieved_episodes, context_used, chunked, working_memory_state,
     presentation_plan, supported}. For ssm_direct/process_exec/tool_plan returns
     supported=False (honest — not faked), mirroring generate_with_routing."""
```

`generate()` and `generate_with_routing()` are **unchanged**.

### 6.6 Acceptance criteria

- [ ] `format_for_llm()` produces the three sections for the Bonsai consumer.
- [ ] Primary chunks appear in full with episode metadata.
- [ ] Compressed section lists **topics** (from secondary episodes), not a raw state vector.
- [ ] Working-memory section shows active domains / recent topics from metadata.
- [ ] `ExpandHandler.handle_expand` loads full text and updates the working-memory snapshot.
- [ ] Context string never exceeds `max_context_tokens` (hard cap, len//4).
- [ ] Integration: retrieve → build_with_chunking → context string (no LLM call in the test).

---

## 7. Task 5 — Prompt Compression for Query Planning

**File:** `src/retrieval/retriever.py` [MODIFY] (or a new `src/retrieval/prompt_compress.py`)

### 7.1 Problem (corrected)

The draft says "compress the prompt through the SSM before Bonsai sees it" and "pass the
compressed state + key entities to Bonsai." But **Bonsai consumes text, not a state vector.**
So compression = produce a **text** prompt ≤ `bonsai_max_input` (2000 chars) that preserves
the planning-relevant signal (entities + recent focus), not a state vector.

> **Mechanism note (vs. the planning chat).** The chat's sketch compresses the prompt by
> chunking it, stepping each chunk through the SSM, and **decoding a summary from the SSM
> state**. That decode step is not implementable here: the Phase 2a backbone is a JEPA
> predictor with no text-decoder head (it predicts latent targets, not tokens). So this
> task extracts key spans (GLiNER / regex) + a WM preamble + tail truncation to produce the
> same *output* the chat intended — a ≤2000-char text summary Bonsai can plan from — without
> requiring a decoder that does not exist. The SSM state still influences the result, via the
> WM-metadata preamble (not as a vector fed to Bonsai). If a decoder head is added later, the
> SSM-decode-the-summary path from the chat can be reconsidered.

### 7.2 Solution

```python
def compress_prompt_for_planning(prompt: str, working_memory: Optional[WorkingMemoryState],
                                  embedder: Optional[Embedder] = None,
                                  config: Phase2cConfig = ...) -> str:
    """If len(prompt) <= short_prompt_threshold (500 chars): return prompt unchanged.
       Else: build a compressed text prompt = a working-memory preamble
       (active domains + recent topics from working_memory.metadata, if present)
       + key spans extracted from the prompt (GLiNER if available, else a cheap
       titlecase/regex span extractor) + a tail-truncated copy of the prompt,
       hard-capped at bonsai_max_input (2000 chars). Returns text, not a vector."""
```

The planner is then called with the compressed text. The hard cap (2000 chars) prevents
Bonsai context overflow. (The SSM state is not fed to Bonsai; it influences the *preamble*
text, which is what Bonsai can read.)

### 7.3 Acceptance criteria

- [ ] Prompts ≤ 500 chars pass through byte-identical.
- [ ] Prompts > 500 chars are compressed to ≤ `bonsai_max_input` (2000) chars.
- [ ] Compressed planning yields the same `entity_mode` / `temporal_intent` as uncompressed
      for ≥ 90% of test cases (a regression guard against over-compression).
- [ ] Compression adds < 30ms (GLiNER path may be slower — gate GLiNER behind a flag; the
      cheap span extractor is the default so the latency target holds).
- [ ] Bonsai never receives > 2000 chars (hard cap enforced by truncation).

---

## 8. Task 6 — Orchestrator Integration + State Serializer

**Files:** `src/orchestrator.py` (NEW), `src/subconscious/state_serializer.py` (NEW)

### 8.1 PonderOrchestrator

There is no `PonderOrchestrator` yet. Create one that **composes** the existing retriever +
Mode-A + the new 2c components. It owns the `WorkingMemory` instance (the cross-query state
holder the retriever does not have a constructor slot for — see §0).

```python
class PonderOrchestrator:
    def __init__(self, store, retriever: HippocampalRetriever,
                 backbone, embedder, mode_a: ModeAGenerator,
                 config: Phase2cConfig):
        self.working_memory = WorkingMemory(backbone, embedder=embedder)
        self.ssm_chunker = SSMChunker(backbone, embedder, config)
        self.presentation_gate = PresentationGate(config, embedder)
        self.expand_handler = ExpandHandler(...)
        self.formatter = ChunkedContextFormatter()
        self.state_serializer = StateSerializer()
        self.sessions_dir = config.session.state_dir

    def query(self, user_prompt: str, consumer: str = "bonsai",
              conversation_history: list[dict] | None = None,
              end_state: Optional[EndState] = None,
              format_spec: Optional[dict] = None,
              extract_schema: Optional[dict] = None,
              model_size: Optional[str] = None) -> dict:
        # 1. embed prompt; working_memory.update(prompt_emb)
        # 2. retrieval_gate.route_text(prompt, embedder, ctx from WM snapshot)
        # 3. compress_prompt_for_planning(prompt, WM snapshot); planner.plan(...)
        # 4. retriever.retrieve(structured_query)  (or retrieve_with_routing)
        # 5. for ep in episodes: working_memory.inject(embed(ep summary))
        # 6. presentation_gate.plan(prompt, episodes, WM snapshot, route.pathway)        # axis (a) chunking
        # 6b. es = presentation_gate.plan_end_state(prompt, episodes, WM snapshot,         # axis (b) end state
        #                                            caller_end_state=end_state)
        #     if caller overrode: presentation_gate.record_override(...) → ReplayBuffer
        # 7. ssm_chunker.chunk(episodes, plan) → ChunkedContext
        # 8. context = formatter.format_for_llm(chunked, consumer, WM snapshot)
        # 9. dispatch by es.end_state:
        #    - "direct":   return {type:"direct", episodes, ...}                # NO LLM call
        #    - "format":   return {type:"format", context, format_spec, ...}     # NO LLM call
        #    - "extract":  return {type:"extract", data, schema, ...}             # cheap span extractor, NOT Bonsai
        #    - "synthesize": response = mode_a._complete(context, conversation_history[-10:])
        #    return {response, route, retrieved_episodes, context_used, chunked,
        #            working_memory_state: WM.snapshot(), presentation_plan, end_state_plan, supported}
```

For `ssm_direct`/`process_exec`/`tool_plan` pathways, return `supported=False` (no
process/tool/System-2 infra — honest, mirroring 2b). `graph_retrieve`/`conscious_deliberation`
run the full pipeline. The `direct`/`format` end states and (via a cheap local extractor) the
`extract` end state return **without invoking Bonsai** — the first Hippo retrieval paths that
answer without the generation model (the "database you can talk to" behavior from the chat).
`synthesize` is the only end state that calls Bonsai.

### 8.2 StateSerializer

```python
class StateSerializer:
    def serialize(self, wm_state: WorkingMemoryState) -> bytes:
        """Flatten the 4 per-layer [1,16,384] tensors (detached, cpu, float32) + bookkeeping
        to a portable bytes blob (torch.save to a BytesIO, or a structured pickle)."""
    def deserialize(self, blob: bytes) -> WorkingMemoryState:
        """Inverse. Round-trip is byte/element faithful."""
```

### 8.3 Session management

```python
def save_session(self, session_id: str) -> Path:
    """Persist WM snapshot to data/sessions/{session_id}.pt (file-based; offline-testable).
    Optional: also store.put(f"session/{id}/wm_state", blob) if a store is configured."""
def load_session(self, session_id: str) -> bool:
    """Restore WM state from disk. Returns False if no saved session."""
```

File-based first (so tests need no WaveDB); WaveDB-backed persistence is optional.

### 8.4 Acceptance criteria

- [ ] `query("What did Alice say about Postgres?")` returns a valid response dict.
- [ ] WM state evolves across multiple queries in one session (state[−1] differs Q1→Q2).
- [ ] Session save/load round-trip preserves WM state (element-equal tensors).
- [ ] Response includes the chunked context + working_memory_state + presentation_plan.
- [ ] Pipeline latency excluding the LLM call < 300ms typical (CPU, ReferenceSSM; the
      draft's <200ms assumed numpy — see §9.2).
- [ ] **No regression** on Phase 1b retrieval: `retrieve()` unchanged; the 1b retrieval tests
      pass unmodified.

---

## 9. Task 7 — Configuration

**Files:** `src/config.py` [MODIFY], `src/subconscious/configs.py` [MODIFY]

Config is dataclass-based (not YAML). Add:

```python
# src/subconscious/configs.py — add to INSTANCE_CONFIGS:
"presentation_gate": InstanceConfig(
    lora_rank=4, output_dim=256, input_dim=384, d_state=16,
    gate_config=GateConfig(num_context_features=2),   # placeholder for the deferred learned gate
    note="heuristic-only in 2c; learned JGS gate deferred until outcome signals are live",
),
```
(The heuristic planner doesn't use this; it exists so a future learned gate has a home and
the instance framework stays consistent. Documented, not dead — `INSTANCE_CONFIGS` is a
registry of declared instances, and presentation is now declared.)

```python
# src/config.py — add:
@dataclass
class Phase2cConfig:
    working_memory: WMConfig = WMConfig(decay_alpha=1.0, lora_rank=8)
    ssm_chunker: ChunkerConfig = ChunkerConfig(max_primary_tokens=4096, max_primary_chunks=5)
    presentation_gate: PGConfig = PGConfig(
        direct_max_episodes=3, chunked_min_episodes=5, summary_only_min_episodes=20,
        expand_threshold=0.5)
    prompt_compression: PCConfig = PCConfig(short_prompt_threshold=500, bonsai_max_input=2000)
    session: SessionConfig = SessionConfig(state_dir="data/sessions/", auto_save_interval=300)
    replay_capacity: int = 1000
```

### 9.1 Cost note (corrects the missing cost-table row)

**Phase 2c adds no training cost.** The backbone is already trained (2a, $61.92).
Working Memory and the SSM Chunker are runtime-only (no training step). The Presentation Gate
is heuristic (the learned gate is deferred). The only compute is local inference, already
covered by the Phase 2a/2b hardware budget. A note is added to `Ponder Engine Phases.md`
§Cost Summary: "Phase 2c: no training cost (runtime-only; backbone reused from 2a)."

### 9.2 Performance targets (corrected)

The draft's latency targets (WM update <1ms, chunk 50 episodes <50ms, full pipeline <200ms)
were set against the imagined numpy EMA. The real path is **torch CPU, batch=1 per SSM step,
ReferenceSSM**. Realistic targets:

| Metric | Draft | Corrected (CPU, ReferenceSSM) |
|---|---|---|
| WM single update | <1ms | < 2ms |
| WM update + 10 injections | <10ms | < 25ms |
| SSM chunking 50 episodes | <50ms | < 300ms |
| Presentation Gate plan() | <20ms | < 1ms (heuristic, no model) |
| Full pipeline (excl. LLM) | <200ms | < 300ms |
| Session save + load | <100ms | < 150ms |

GPU (if available) meets the draft targets; CPU targets above are the test/baseline.

---

## 10. Task 8 — Integration Tests

**File:** `tests/test_orchestrator.py` (the scenarios were folded into the
orchestrator test module rather than a separate `tests/integration/` file, to
avoid duplicating the shared `_orchestrator` / `_ep` / `_StubPlanner` /
`_StubModeA` / `_StubEmbedder` helpers across two files). The EXPAND mechanism
is additionally pinned at the handler unit level in
`tests/test_expand_handler.py`.

| Test | Description | Expected |
|---|---|---|
| `test_short_query_direct` | "What was the Python async throughput?" + 2 episodes | `direct`, both primary, no compression |
| `test_broad_query_chunked` | "What have we discussed about performance?" + 12 episodes | `chunked`, 5 primary, 7 compressed |
| `test_summarization_summary_only` | "Summarize everything about databases" + 30 episodes | `summary_only`, all compressed |
| `test_working_memory_continuity` | Q2 references Q1 | WM state after Q2 ≠ after Q1; Q2 retrieval includes Q1-relevant episodes |
| `test_expand_mechanism` | Chunked → EXPAND(ep_id) | `ExpandHandler` returns full text; WM snapshot updated |
| `test_session_persistence` | save → new orchestrator → load | WM state element-equal after round-trip |
| `test_prompt_compression` | 2000-char prompt → planning | Bonsai-path receives ≤2000 chars; planning fields preserved |
| `test_presentation_gate_buffer` | record outcomes → buffer grows | buffer length increases (no fake learning) |
| `test_direct_return_no_llm` | `end_state="direct"` + 3 episodes | returns `{type:"direct", episodes}`, **no Bonsai call** (mode_a not invoked) |
| `test_format_for_consumer` | `end_state="format"`, `format_spec={"consumer":"claude","max_tokens":4000}` | returns formatted context string, **no LLM call** |
| `test_extract_to_json` | `end_state="extract"`, `extract_schema={"type":"list","item_type":"decision"}` | returns structured list from episodes (cheap extraction or span match, **no synthesis LLM**) |
| `test_synthesize_calls_bonsai` | `end_state="synthesize"` (or unset) | `mode_a._complete` invoked once; response returned |
| `test_end_state_override_records_to_buffer` | caller passes `end_state="extract"` while gate default was `"synthesize"` | `record_override` called; override ReplayBuffer grows by 1 (the training signal per [146]) |

All tests offline: ReferenceSSM + stub embedder + tmp_path store (mirrors
`tests/test_retriever.py` / `tests/test_retrieval_gate.py`). No GLiNER/Bonsai/WaveDB-on-pod
required for the unit suite; GLiNER/Bonsai paths are skip-gated.

---

## 11. Task 9 — Documentation (ADRs)

Create `docs/adr/` (does not exist yet):

- `006-working-memory-design.md` — why map WM onto the `JGSInstance` recurrent state (uses the
  trained 2a backbone's dynamics; fixed-dim regardless of conversation length) rather than a
  flat numpy EMA (parallel state machine that ignores the trained SSM); why state persists
  across queries (presence) vs the 2b per-query reset; why inject memories as embedding-steps
  (gist, not text); why `decay_alpha` defaults to 1.0 (SSM dynamics already mix; avoid
  double-applying an EMA).
- `007-ssm-chunking-strategy.md` — why compress into SSM state rather than LLM-summarize
  (no extra LLM call, deterministic, uses trained backbone); why primary/secondary split
  rather than hierarchical summarization; EXPAND scope — chunking-level loader here,
  metacognitive trigger in Phase 4a (explicit handoff, no overlap).

---

## 12. Task order & dependencies

```
Task 1  Working Memory Core            (no deps; uses JGSInstance + Embedder)
  ↓
Task 2  SSM Chunker                    (deps Task 1 — compressor is a WorkingMemory)
  ↓
Task 3  Presentation Gate              (deps Task 1 — reads WM snapshot; heuristic, no model)
  ↓
Task 4  Mode A with Chunking           (deps Tasks 2, 3)
  ↓
Task 5  Prompt Compression             (deps Task 1 — reads WM metadata)
  ↓
Task 6  Orchestrator + StateSerializer (deps Tasks 1-5)
  ↓
Task 7  Configuration                  (parallel with 1-6; dataclasses)
  ↓
Task 8  Integration Tests              (deps Task 6)
  ↓
Task 9  ADRs                           (parallel with Task 8)
```

---

## 13. Risk register (corrected)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `presentation_gate` learned gate has no data | High | Suboptimal presentation | Heuristic is the real logic; learned gate deferred; outcome buffer seeds future training. |
| WM state drift over long sessions | Medium | Degraded retrieval | `decay_alpha` knob; add a drift-detection test (state-norm bound). |
| ReferenceSSM dynamics don't actually "forget" gracefully | Medium | Old info dominates | `decay_alpha < 1.0` is a **WM-state-tensor** tuning lever (post-step forget factor on the recurrent state). Note: the chat's *saturation / "don't overweight indefinitely"* concern (diminishing-returns, saturation detection, LLM-mediated importance, boost decay) is an **edge-level / graph** concern that belongs to Phase 3 GNN consolidation, **not** to `decay_alpha` on the SSM state — they are different forgetting mechanisms and must not be conflated. Mamba3 backend remains unavailable (build fails), so ReferenceSSM is the working path for 2c too. |
| EXPAND latency | Low | UX | Expandable-episode full text cached in `ChunkedContext.expandable_ids` → store lookup; expand is one store.get + one SSM step. |
| Prompt compression over-compresses → wrong planning | Medium | Wrong retrieval | 90%-same-planning regression test (§7.3); GLiNER gated behind a flag; cheap span extractor is default; fall back to truncation if entities sparse. |
| End-state routing feedback signal is weak (chat [146]) | High | Can't *learn* the end-state router from outcomes | Use an **explicit API with a heuristic default** (§5.6), not a learned router. The feedback loop for end-state choice is ambiguous (satisfaction doesn't distinguish `direct` from `synthesize`; the real difference is latency/cost/format, which are unobservable to the model). Caller overrides feed a `ReplayBuffer` that only seeds a *future* learned router once enough overrides accumulate; the heuristic stays the production default. This is the explicit-API-with-override principle from the chat, not a faked learned gate. |

---

## 14. Definition of Done

- [x] All unit tests pass (Tasks 1-5).
- [x] All integration tests pass (Task 8).
- [x] WM state persists across queries within a session (verified).
- [x] SSM chunking splits episodes into primary/compressed correctly.
- [x] Presentation Gate selects the appropriate strategy for the 8 test scenarios.
- [x] **End-state routing** dispatches correctly: `direct`/`extract`/`format` return **without an LLM call**; only `synthesize` calls Bonsai; a caller override is recorded in the override `ReplayBuffer` (§5.6 / §8.1).
- [x] EXPAND loads full text of compressed episodes and updates WM.
- [x] Prompt compression activates for long prompts and preserves planning accuracy (90% test).
- [x] Full pipeline latency < 300ms (CPU, excl. LLM).
- [x] **No regression** on Phase 1b/2b retrieval + routing tests.
- [x] Session save/load round-trip preserves state.
- [x] ADRs 006/007 written.
- [x] All public APIs documented (Google-style docstrings).
- [x] **de-wonk clean** (no CRITICAL/HIGH/MEDIUM): no untrained-dead-params, no faked
      `ssm_direct`/`process_exec`/`tool_plan`/learned-gate, no TODO/stub left in scope.

---

## 15. Next Phase Handoff (Phase 3a: GNN Consolidator)

1. **WM state format** — the GNN Consolidator reads WM state to prioritize consolidation of
   what's currently "in awareness."
2. **SSM chunking infrastructure** — the GNN's semantic-memory abstractions are stored in the
   same `WorkingMemoryState` / serialized-tensor format.
3. **Presentation Gate outcome signals** — EXPAND frequency (from the `ExpandHandler` log)
   feeds the GNN's salience scoring: frequently-expanded episodes should have higher salience.
4. **Learned Presentation Gate** — the outcome `ReplayBuffer` populated in 2c is the training
   data for the deferred learned gate (Phase 3a+ or a dedicated 2c.1).
5. **Learned end-state router** — the **override `ReplayBuffer`** (§5.6, populated whenever a
   caller overrides the gate's end-state default) is the seed for the deferred *learned*
   end-state router. Because the end-state feedback signal is weak (chat [146], §13 risk row),
   this learned router is deferred far out — the explicit API + heuristic default stays the
   production interface. The buffer is the only training signal available, so it must be
   collected now even though learning is not.