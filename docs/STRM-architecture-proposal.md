# STRM — Short-Term Read-Only Memory Primitive

**Architecture proposal — v0.3 (draft for review, not a shipped design)**

STRM = **S**hort-**T**erm **R**ead-**O**nly **M**emory. The name states the core
finding: the `WorkingMemory` recurrent state is **write-only relative to LTM**
(verified across `src/` — nothing outside `working_memory.py` reads it), and the
STRM heads are the **read-only** read-out of that state (they read `state_t` /
`y_t`; they never write the state). Two different "only"s on two interfaces.
The old acronym (JEPA/SSM/Transformer) is retired in favor of STRM now that the
latent-dynamics head dropped JEPA for a linear predictor (v0.3, below).

**Status: de-risked 2026-07-19.** The one core assumption the whole design rested
on — that SSM forgetting is predictable enough for a learned probe to read off
the hidden state — is now verified by the §6.1 / §9 probes. Both gates PASSED.
The result that changes the design: the **latent-dynamics head ships LINEAR, not
JEPA** (see §6.2). This doc is still written so the parts that stand if a later
gate fails are separable from the parts that don't.

**What changed in v0.3 (2026-07-19): probe results + the linear-not-JEPA pivot.**
- **0a recoverability — GO.** Probe `P(state_t, u_i)` AUC val = 0.810 (gate 0.75),
  beats the free `k`-baseline (0.732). State carries lag-independent "which anchor
  was forgotten" info, not just older=more-forgotten; the decay curve grows
  monotonically with `k`. The recoverability head is viable; its labels are
  decoder `D`'s reconstruction error `e(i,t)` (free, no Oracle).
- **0b latent-dynamics — GO, ship LINEAR.** Linear `z_{t+1}=Az_t+b` R²=0.297 over
  constant-mean (gate 0.15); linear surprise-AUC (L2 residual) = 0.7625 (gate
  0.70). An EMA JEPA predictor `g` trained with the contrastive anti-collapse
  loss scored surprise-AUC = 0.565 (cosine, its native distance) — *underperforms*
  the closed-form linear baseline. Latent variance stayed bounded (no collapse).
  Because the backbone is frozen, collapse cannot occur, so the JEPA EMA /
  stop-grad / anti-collapse machinery solves a problem that does not exist in v1.
  The latent-dynamics head ships as `z_{t+1}=Az_t+b`; the upgrade path if its R²
  ceiling binds is a light **MSE-trained MLP**, not JEPA. JEPA only earns its
  place for a future *generative* rollout (predict `k>1` steps ahead for
  imagination), which is a different feature than the surprise signal — past v1.

**What changed in v0.2:** the head layer is expanded from two heads to four, to
cover the full set of behaviors from the source conversation (relevance,
forgetting-prediction, latent-dynamics/world-model, STM→LTM graduation), and a
new §3 maps each behavior to **what Ponder already ships** vs. what is genuinely
net-new. The headline of that mapping: Ponder has rich LTM-side memory machinery
(forgetting, anomaly, consolidation, presentation, retrieval) — but **none of it
reads the `WorkingMemory` recurrent state**. The SSM state is write-only relative
to LTM. STRM's contribution is the STM-side, state-conditioned analogs that close
that gap, plus the wiring between the two sides.

---

## 0. Scope

- **In scope:** a short-term memory (STM) primitive that streams an activity feed,
  maintains a compressed state, builds a bounded context block from it, predicts
  its own latent dynamics, **decides what graduates from STM into long-term
  memory**, and emits a self-aware signal when information needed for factual
  accuracy has been compressed out — so a recall can be issued against LTM.
- **Out of scope (covered elsewhere):** long-term memory itself (Ponder's
  ingestion + episodic store + GNN consolidation + contradiction/citation
  machinery). The consumer (LLM/agent) that reads the context block. STRM produces
  a bounded context block, graduation decisions, and recall pointers; it does not
  store episodes or reason.

STRM is **not greenfield**. The SSM substrate already ships as `WorkingMemory`
(`src/subconscious/working_memory.py`, Phase 2c): a `JGSInstance` whose recurrent
state persists across queries, with `inject()` for absorbing retrieved episodes
and `decay_alpha` for a post-step forget factor. State is 4 per-layer tensors
`[1, d_state=16, d_model=384]`, detached per step (no BPTT). STRM extends
`WorkingMemory` with what it currently lacks — and the central finding of this
proposal is that **`WorkingMemory`'s state is currently never read by anything
outside its own module** (verified across `src/`; see §3). Everything STRM adds is
a read-out of that state.

---

## 1. Purpose

Give a stateless consumer a **seamless, self-aware short-term memory**: one
component that

1. ingests a streaming activity feed and maintains a compressed running state;
2. predicts its own next latent state (a world model over memory) — prediction
   error is the STM-side **surprise** signal;
3. builds a **bounded** context block from that state — sized to a target window,
   regardless of how much experience has streamed through;
4. **decides what graduates** from STM into LTM before it is compressed away
   (the "remembering" decision);
5. **knows when it has lost information needed for factual accuracy** and emits a
   pointer so the missing material can be recalled from LTM before the consumer
   commits to an answer.

"Seamless" means: the consumer never manages memory layout, eviction, refresh
timing, or what to consolidate. It hands STRM activity and importance signals and
receives a context block plus, occasionally, a recall request and a graduation
hint.

---

## 2. Problems it solves

| Problem | What goes wrong today | STRM's answer |
|---|---|---|
| **Unbounded stream vs. fixed window** | Activity streams grow without limit; the consumer's window is fixed. Naive truncation drops the wrong things; Ponder's `PresentationGate` picks a chunking strategy by keyword heuristics, not by state content. | A learned context-builder emits a **fixed-budget** block; the relevance head decides what fills the budget. This is the "deferred learned gate" `PresentationGate` reserves replay buffers for (`presentation_gate.py:13-21`). |
| **Opaque STM forgetting** | `WorkingMemory` forgets via SSM dynamics + a scalar `decay_alpha`, but nothing can tell *what* it has forgotten or *when* a needed fact has been compressed out. Ponder's forgetting machinery is LTM-side only. | A **recoverability head** estimates, per anchor, whether the fact is still in the state — the STM-side analog of Ponder's LTM salience/prune heads. |
| **No STM-side surprise signal** | Ponder's `AnomalyHead` (`heads.py:205-273`) is a 9-label *classifier* over GAT node embeddings detecting structural corruption in LTM. It is not a next-state predictor and is not SSM-conditioned. There is no "the STM just did something unpredicted" signal. | A **latent-dynamics head** predicts `z_{t+k}` from `z_t`; prediction error is STM-side surprise — the genuinely predictive piece (a next-state predictor, not a label classifier), and the STM sibling of the LTM anomaly head. Ships **linear** in v1 per the §6.2 / 0b probe; JEPA is reserved for a future generative rollout. |
| **No STM→LTM graduation** | Ponder's consolidator (`consolidate.py`) operates **entirely within the LTM graph** (DiffPool abstraction, salience-prune, supersede). There is no gate that decides what the STM writes out to LTM before it is lost; Thread-2 ingestion is undiscriminating. | A **graduation head** scores state contents for LTM-write priority — the "remembering" decision, the inverse of the recoverability flag. |
| **Wasteful / latency-coupled refresh** | Retrieval is always externally triggered by a user prompt (`retrieval_gate.py:126-146` embeds the *prompt*, not the state). Fixed-interval refresh fetches on a clock whether or not anything was lost. | Refresh is **self-triggered** by the salience signal; LTM is kept warm by a parallel continuous-ingestion thread so a recall is a cheap read. |
| **Reactive-only factual guards** | Existing contradiction/citation guards (Phase 3c, Bonsai decider, doc-kind snapshot guard) catch errors *after* the fact. | Salience is a **proactive** "about to be wrong" signal — the sibling of the reactive guards, earlier in the loop. |

---

## 3. What Ponder already ships vs. what STRM adds

This is the heart of the user's question. Verified across `src/` (excluding
training and tests): **nothing outside `working_memory.py` reads the
`WorkingMemory` recurrent state tensors.** The only things that ever leave WM are
(a) the `metadata` dict (`active_domains`, `last_query_type` — string labels the
orchestrator set itself, not derived from the state) and (b) `wm_episode_ids`, a
set of episode ids used only to re-order consolidation scoring centers
(`consolidate.py:163`). The SSM state is **write-only relative to LTM**.

| Behavior (from source conversation) | Ponder already has (LTM-side) | Does it read WM state? | STRM adds (STM-side) |
|---|---|---|---|
| **Relevance scoring** | `SalienceHead` (GNN, over episode subgraphs); kind-aware rerank (Phase 2c) | No — graph embeddings only | Relevance head over the `y_t` ring buffer, query-conditioned |
| **Forgetting prediction** | `SalienceHead`-gated prune (`consolidate.py:617-644`); utility decay (`forgetting.py`); `supersede_assertion` tombstone, Bonsai-gated (`store.py:1076-1105`) | No | Recoverability head reading `state_t` directly |
| **Latent dynamics / world model** (predict `z_{t+k}`) | *Nothing.* `AnomalyHead` is a classifier, not a predictor. (`DecomposedGate` does pool a `predicted_future` (`gate.py:113-116`) — but that is the gate's own internal state, not a WM read-out.) | No | **Latent-dynamics head** — predict `z_{t+k}` from `z_t`; prediction error = STM surprise. Genuinely new. Ships **linear** (`z_{t+1}=Az_t+b`) in v1 per the 0b probe — a closed-form map already clears the surprise gate and beats an EMA JEPA predictor; the frozen backbone means collapse cannot occur. JEPA's EMA/stop-grad/anti-collapse machinery is deferred to a future generative rollout (§6.2). |
| **Graduation (STM→LTM write gate)** | LTM-internal abstraction only (DiffPool clusters → `M:` nodes, `semantic_memory.py:63-109`); no STM→LTM promotion path exists | No | **Graduation head** — score state contents for LTM-write priority before compression loses them |
| **Predictability / surprise** | `AnomalyHead` (9 structural-corruption labels, classifier over GAT node emb, `heads.py:205-273`) | No — graph side | STM surprise from the latent-dynamics head's prediction error; interlocks with the LTM anomaly head (§5.4) |
| **Consumer importance tags** | **Already shipped as `llm_signal`** (`important/routine/satisfied/frustration/correction`, `forgetting.py:58-64`) modulating retrieval boost | n/a (caller-supplied) | Route `llm_signal` into the STM too (graduation + relevance bias). **Do not rebuild this — reuse it.** |
| **Context selection / presentation** | `PresentationGate` — heuristic, keyword-based, reads WM *metadata strings* only, explicitly defers a learned gate (`presentation_gate.py:13-21`, replay buffers reserved) | No (metadata only) | Context-builder Transformer over the ring buffer — the learned gate `PresentationGate` already anticipates |
| **Retrieval gating** | `RetrievalGate` + `BonsaiQueryPlanner`; query = **prompt embedding**, never state; externally triggered only (`retrieval_gate.py:126-146`) | No | State-conditioned retrieval query + **self-triggered** recall on salience |

**Bottom line:** Ponder has the LTM side of every one of these behaviors, driven
by trained GNN heads and Bonsai. STRM does not duplicate any of it. STRM adds the
**STM-side, state-conditioned analogs** (which are absent precisely because the
WM state is write-only today) and the **wiring** that lets the two sides talk:
STM surprise feeds into the anomaly pipeline; STM graduation feeds the
consolidator's input; STM salience triggers retrieval; the context-builder
replaces the heuristic `PresentationGate`. The consumer-importance channel
(`llm_signal`) already exists and is shared.

---

## 4. Components

Grounded in shipped code. Shapes from `src/subconscious/configs.py`
(`d_model=384`, `d_state=16`) and `working_memory.py` (4 per-layer state tensors
`[1, 16, 384]`, detached per step).

### 4.1 SSM — `ReferenceSSM` via `WorkingMemory` (exists)

Role: streaming lossy buffer. Each step consumes `u_t ∈ [1, 384]` and produces
`y_t ∈ [1, 384]` plus updated `state_t ∈ [1, 16, 384]` per layer (4 layers).

Why it fits:

- **Already selective.** `ReferenceSSM._step` (`ssm.py:98-107`) uses an
  input-dependent retention gate `gate = σ(W_A x)` and
  `new_state = g·b + (1-g)·state`. Content-dependent forgetting is what makes
  recoverability *predictable* (a learned function of content, not uniform decay).
  We do not need Mamba for selectivity — we already have it.
- **Working per-step API.** `step()` is implemented and used live (Phase 2b+).
  The Mamba3 backends raise `NotImplementedError` on `step()` and the CUDA build
  fails on this box — a *downgrade* for the streaming path. `ReferenceSSM` is the
  right backend. (Swap candidate if the probe fails: Mamba2, not Mamba3 — mature
  kernels, working step path.)
- **Already persists state across queries** and **already absorbs recalled
  episodes** via `inject()`. The re-injection mechanism STRM needs is shipped.

### 4.2 Four diagnostic/predictive heads (new)

The heads' job in STRM is **flagging, predicting, and gating** — not orchestrating.
They emit signals that other components act on. Four heads, with deliberately
different read access:

- **Relevance head** — reads slot content `y_t` + the current query → per-slot
  `r_i ∈ [0,1]`. Biases the context-builder. The STM analog of `SalienceHead`.
- **Recoverability head** — reads **`state_t` directly** (its own projection) +
  an anchor → estimate of whether that anchor is still recoverable. **Must** read
  state, not `y_t`: `y_t` is already the lossy read-out
  (`W_C(flat(state)) + D·x`, `ssm.py:105-106`) and may have dropped exactly the
  information whose presence we are asking about. This is the "predict forgetting
  via reconstruction error" head from the source conversation
  (`forget(t) = D(g(z_{t+k}), z_t)`), trained self-supervised (§6.2).
- **Latent-dynamics head** — predicts `ẑ_{t+k}` from `z_t` (a learned transition
  / world model over the memory state). Prediction error
  `‖ẑ_{t+k} − z_{t+k}‖` is the **STM-side surprise** signal. This is the one
  behavior Ponder has no analog for on the STM side. **It ships LINEAR in v1**
  (`ẑ_{t+1} = A z_t + b`, fit closed-form): the 0b probe showed a linear map
  already clears the surprise gate (surprise-AUC 0.7625) and *beats* an EMA JEPA
  predictor (cosine surprise-AUC 0.565). Because the backbone is frozen, the
  latent `z_t` is a fixed, non-collapsing input — there is no encoder being
  trained that could collapse, so the JEPA anti-collapse machinery (EMA target
  encoder, stop-grad, contrastive negatives) solves a problem that cannot occur
  here and is deferred. The honest upgrade path if linear's R² ceiling binds is a
  light **MSE-trained MLP** (non-linear, same L2 objective the surprise signal
  wants); JEPA only earns its place for a future *generative* rollout predicting
  `k>1` steps ahead for imagination, which is a different feature than the
  surprise signal and past v1 (§6.2).
- **Graduation head** — reads `state_t` + the slot's content + an
  `llm_signal`-derived importance input → a score for "write this to LTM before
  it is compressed out." This is the "remembering" decision. It feeds the
  consolidator's input queue; the consolidator itself (LTM-internal abstraction,
  prune, supersede) is unchanged and reused as-is.

**Salience signal** = `recoverability(anchor) < θ ∧ relevance(anchor) > φ` → emit
an LTM pointer. Optionally gated by surprise: a high-surprise step raises the
bar for trusting the recoverability estimate (we are less sure of state contents
right after a prediction miss). The thing that *judges state contents* reads the
state; the thing that *assembles context* reads the published outputs.

### 4.3 Context-builder Transformer (new)

Role: turn the ring buffer of recent `y_t` into a bounded context block, using
the relevance head's scores as an attention bias. This is the learned `PresentationGate`
that `presentation_gate.py:13-21` explicitly defers and reserves replay buffers
for — STRM consumes those buffers as training signal.

```
M ∈ [1, K, 384]            # ring buffer of recent step outputs y_t
q_raw ∈ [1, 384]           # query, from current input/task
r ∈ [1, K]                 # per-slot relevance (from the relevance head)
q     = q_raw @ W_q        # [1, d_head]   (project query into attention space)
K_mat = M @ W_k            # [1, K, d_head]
V     = M @ W_v            # [1, K, d_head]
logits = (q @ K_matᵀ)/√d + λ·r          # [1, K]
attn   = softmax(logits, -1)            # [1, K]
ctx    = attn.unsqueeze(1) @ V          # [1, d_head] — INTERNAL training surrogate
                                #   (see below: serve path is discrete top-m, not ctx)
```

**The output is discrete, not the continuous `ctx` vector.** This is the
resolution of the obvious question the equation raises: the math produces a
continuous `[1, d_head]` vector, but the seam consumes and produces **text** and
the consumer is a text LLM (it cannot eat a raw 384-dim vector without a whole
soft-prompt research layer). So the context-builder is a **learned
selector/reranker**, not a vector compressor: `attn` produces selection weights
over ring-buffer slots; the selected slots map back to their **source episode
text** (the ring buffer carries provenance — slot → source `episode_id` / event —
not just `y_t`); the consumer-facing output is the bounded **text** of the
selected episodes, same shape as `PresentationGate`'s primary/compressed split.
The continuous `ctx = attn @ V` is the *internal* selection signal used during
training (differentiable surrogate for the discrete pick); at serve time it is
discarded and the discrete selection is what ships.

Three consequences made explicit:

- **The ring buffer carries provenance.** Each slot is `(y_t, source_id, text)`,
  not just `y_t`. Without provenance the selector cannot map back to text.
- **Hard top-m is the primary serve path, not an optional variant.** Soft
  attention trains the selector; hard top-m (keep top-m of `q·K + λr`, mask the
  rest to `-inf`) is what produces the discrete selection the consumer needs.
  "Start soft, add hard" is a training schedule, not a serve-time choice.
- **Continuous soft-prompt output is out of scope for v1.** Feeding `ctx` as a
  soft prompt / prefix to a consumer that accepts continuous inputs is a
  different consumer interface and a separate research direction; the seam here
  is text-in/text-out, so v1 is discrete selection only.

Small model: it attends over `K` vectors (~dozens to low hundreds), not tokens —
a 20-200M-parameter router, not a language model. It does **not** touch `state_t`
directly; it reads `y_t` through the SSM's trained `W_C`. (Upgrade path if
`W_C`'s read-out doesn't preserve what the builder needs: add a `proj(state_t)`
head trained jointly — see §7.)

**Where it plugs in (the seam, verified):** today the context the LLM receives is
**prompt-assembled, not a tool call**. The chain is `PresentationGate.plan`
(heuristic, `orchestrator.py:323`) → `SSMChunker.chunk` (`orchestrator.py:336`)
→ `ChunkedContextFormatter.format_for_llm` (`end_state.py:225-228`) → injected
into the user message as `Context from past conversations:\n{context}\n\nUser:
{user_prompt}` (`orchestrator.py:373`). The context-builder replaces the
**plan producer** at the first seam: emit a `PresentationPlan`-like object at
`orchestrator.py:323` and let the existing chunker + formatter keep working.
This is the smallest seam *and* the one the codebase itself designates —
`PresentationGate`'s `ReplayBuffer` (`presentation_gate.py:178-179`) is already
collecting `(plan, outcome)` pairs "for the deferred learned gate"
(`presentation_gate.py:168`), so the training data is already being gathered. Two
deeper seams exist if the builder wants to emit its own primary/compressed split
(replace `SSMChunker.chunk`, `ssm_chunker.py:64-71` `ChunkedContext`) or the final
string (replace `format_for_llm`); start at the first.

### 4.4 Ring buffer (new)

A temporal window of the last `K` step outputs. Each slot is a triple
`(y_t ∈ [1, 384], source_id, text)` — the vector plus **provenance** back to the
event/episode that produced it (required so the context-builder can map a
selected slot back to text; see §4.3). Append/pop per step. **`K` is not
`d_state`.** `d_state = 16` is the SSM's channel depth at one instant — how many
parallel memory tracks the state holds *right now*, already collapsed to 384 by
`W_C` when producing `y_t`; it does not appear in the buffer. `K` is the temporal
lookback — how many recent timesteps the Transformer can reach. Independent knob.
Crank `d_state` → richer per-moment compression; crank `K` → deeper history
visible to the builder. They answer different questions.

### 4.5 LTM pointer, graduation, and re-injection (partly exists)

Three things leave STRM for Ponder:

- **Recall pointer** (on salience) → Ponder services it from warm LTM. The
  returned episode is encoded and **re-injected as the next `u_{t+1}`** via
  `WorkingMemory.inject()` — which already exists and already does this. The SSM
  does not care that the input is a recall rather than a fresh event.
- **Graduation writes** (from the graduation head) → handed to the existing
  consolidator as prioritized input. The consolidator's LTM-internal abstraction,
  salience-prune, and supersede logic are reused unchanged.
- **Surprise events** (from the latent-dynamics head, high prediction error) →
  routed to the anomaly pipeline as an STM-side signal complementing the LTM
  `AnomalyHead` (§5.4).

**Pinning (new, small):** a recalled episode can itself be compressed out on the
next tick → re-fetch loop. Add a learned token-type embedding to the recalled
`u_{t+1}` (a "refresh" tag, same dimension as the input embedding, so `d_model`
is unchanged) so the input-dependent `W_A` gate tends to retain it over the next
`K` steps. Standard cache-pin pattern.

### 4.6 Two-thread architecture (wiring, partly exists)

- **Thread 1 — STM (hot, low-latency):** `WorkingMemory` + ring buffer + four
  STRM heads + context-builder. Runs per activity event. Latency budget: ms.
- **Thread 2 — Ponder continuous ingestion (async, eventually consistent):** the
  existing doc-ingestion pipeline (TEXT+MD/PDF/Code/DOCX/Web/email parsers,
  salience, doc-kind, Bonsai, 3c citation/contradiction) pointed at the **live
  activity feed**. Always running; populates/updates LTM in the background. Also
  receives STRM's graduation writes and surprise events.

Why two threads: it **decouples STM latency from consolidation latency**. STRM
never blocks on Bonsai/GLiNER/extraction (the ingestion bottleneck your notes put
at ~24s/conv). When salience fires, LTM is already warm; the recall is a cheap
read. The two threads are **parallel consumers of one activity stream**; they
share the input tap, not state. Keep the SSM state and Ponder's graph deliberately
un-entangled — STRM communicates with Ponder via the three message types above,
not by sharing tensors.

### 4.7 Delivery path & tool interlock (verified)

How context reaches the consumer today, so STRM's outputs land in the right place:

- **Prompt assembly is primary.** The LLM always receives a pre-assembled
  context block in its user prompt — `Context from past conversations:\n{context}
  \n\nUser: {user_prompt}` (`orchestrator.py:373`) — built by
  `ChunkedContextFormatter.format_for_llm` into `[RETRIEVED CONTEXT — PRIMARY]`
  (full text, hard-capped at `max_tokens`), `[COMPRESSED CONTEXT — SUMMARY]`
  (secondary episodes as a **topic-union only**, plus the list of `EXPAND` ids),
  and `[WORKING MEMORY STATE]` (the WM `metadata` strings). The model can answer
  from this with zero tool calls. The `direct`/`format`/`extract` end states
  bypass the LLM; only `synthesize` calls it.
- **Tools are a secondary refinement loop**, gated by
  `self_chat_tool_loop_enabled` (default on): `expand` pulls the full text of a
  compressed gist by id (`tools.py:73-91`), `search_memory` re-retrieves
  mid-generation with a refined query (`tools.py:92-118`), and `record_feedback`
  has the LLM rate each cited unit 1-5 *after* answering (`tools.py:44-72`).
- **STRM salience is the internal, pre-emptive version of the external tools.**
  `search_memory` is the LLM noticing "I'm missing something" and re-retrieving;
  STRM's salience signal fires that same recall *before* the LLM has to notice —
  the recall is re-injected into STM (`WorkingMemory.inject`) and shows up in the
  next prompt-assembly, so the LLM never sees the gap. `expand` is the LLM
  pulling detail on a gist it was shown; STRM's relevance head decides which gists
  to show uncompressed in the first place. STRM does not remove the tools — it
  reduces how often the LLM needs them.
- **`record_feedback` could label the relevance head — but the raw ratings
  aren't logged today.** The 1-5 per-unit judgments are reduced to a compounded
  boost multiplier at `content/unit_boost/{unit_id}` (`store.py:707-749`); only
  the multiplier survives, not the rating. Phase 2a adds a raw-rating JSONL tap
  before the reduction so the relevance head can train on real labels; until
  then, synthetic labels.

`prompt_compress._wm_preamble` / `compress_prompt_for_planning`
(`prompt_compress.py:97-174`) is a *separate, earlier* compression for the
**Bonsai planner** input, not the consumer LLM — do not confuse the two seams.

---

## 5. Life-cycle of intended use

Per activity event (one step):

1. **Encode** the event to `u_t ∈ [1, 384]` (bge-small, the injected `Embedder`).
   If the consumer supplied an `llm_signal` (importance tag) for this event, fold
   it into `u_t` as a token-type embedding (same mechanism as the recall pin
   tag) so the SSM gate and the graduation head both see it.
2. **Predict (before stepping):** the latent-dynamics head predicts the next
   state `ẑ_{t+1}` from the current `z_t`. (Horizon `k=1` to start, per §7;
   longer horizons score against `z_{t+k}` `k` steps later and need a small
   pending-prediction buffer.)
3. **Step the SSM:** `WorkingMemory.step(u_t)` → new `state_{t+1}`, new `y_t`.
4. **Score surprise:** `surprise_t = ‖ẑ_{t+1} − z_{t+1}‖`. If high, route an
   STM-surprise event to the anomaly pipeline and raise the salience bar this
   step (lower confidence in the recoverability estimate).
5. **Append** `y_t` to the ring buffer; pop the oldest. Buffer stays
   `[1, K, 384]`.
6. **STRM heads:**
   - relevance head → `r ∈ [1, K]`;
   - recoverability head → per-anchor recoverability from `state_t`;
   - graduation head → per-slot LTM-write priority (high-priority slots queued
     for Thread 2 before they are lost).
7. **Build context:** the context-builder attends over the buffer with `r` as
   bias → emits a bounded block of size `C`. This block replaces the heuristic
   `PresentationGate` output.
8. **Emit** the block to the consumer (and, for inspection, salience/surprise
   flags).
9. **Salience check:** for each anchor, if
   `recoverability < θ ∧ relevance > φ` (and surprise is not high enough to
   suppress), emit an LTM pointer.
10. **If a pointer was emitted** (next step, async): Ponder services it from warm
    LTM → returns an episode embedding → `WorkingMemory.inject(episode)` with the
    refresh tag → the recalled gist is now in state and the buffer.

Session boundaries: `WorkingMemory.reset()` on explicit session end only — never
per query (existing contract). `snapshot()` / `restore()` is the session
save/load path, unchanged.

**Freshness watermark (Thread 2 lag):** if Thread 2 lags by `Δ`, a fact younger
than `Δ` that gets compressed out of STM is unrecallable — it is neither in STM
nor yet in LTM. The salience signal must carry a freshness check, and the check
must **tell the consumer, not silently drop the pointer.** Suppressing the
pointer alone is a silent false negative: the consumer assumes the fact does not
exist, which is worse than admitting uncertainty. So for anchors younger than the
watermark, emit a typed **stale-uncertain** signal to the consumer ("I may know
this but have not finished ingesting it") alongside the suppressed pointer — the
consumer can then choose to wait, re-ask, or proceed with a stated gap, rather
than being lied to by omission. With async-distill + the GLiNER GPU path, `Δ`
should be manageable; define it explicitly rather than pretending it is zero.

---

## 6. Training

### 6.1 De-risking probe (first, cheap, gates everything)

The whole design rests on one assumption: **SSM forgetting is predictable enough
that a probe can estimate recoverability from `state_t`.** Test it on the
**already-trained** backbone — no retrain.

1. Run real activity streams through `WorkingMemory`, logging `u_1..u_T` and
   `state_t` at each step.
2. Train a **recovery decoder** `D(state_t) → û_i` that reconstructs a past input
   `u_i` from a later state. Reconstruction error `e(i, t)` is the ground-truth
   forgetting signal. (This is exactly `forget(t) = D(g(z_{t+k}), z_t)` from the
   source conversation.)
3. Train a **lightweight probe** `P(state_t, anchor_i) → ê(i, t)` to predict the
   error without doing the recovery.
4. Measure probe AUC.

Decision: decent AUC → the salience mechanism is viable, build STRM. Poor AUC *and*
discretization is the suspected cause → consider a Mamba2 swap. Poor AUC with no
obvious fix → simplify to fixed-interval refresh and stop. The probe also answers
"do we need Mamba" for free: it tests whether `ReferenceSSM`'s selectivity
suffices in practice.

### 6.2 The four heads (none are JEPA in v1)

- **Recoverability head** — supervised on the §6.1 labels (you generate them from
  your own SSM). A supervised probe with generated labels, not a self-supervised
  objective. Simpler and likely better than a collapse-prone self-supervised
  objective for this specific job. The labels come from a reconstruction-error
  world model (decoder `D`'s `e(i,t)`), but the head itself is a supervised
  regressor.
- **Latent-dynamics head** — ships **linear** in v1, not JEPA. This is the
  outcome of the 0b probe, not an a priori choice. The probe ran the de-risk
  ordering the original v0.2 spec called for (linear baseline first, then EMA
  JEPA only if linear beat mean) and got a decisive result: linear
  `z_{t+1}=Az_t+b` R²=0.297 (beats mean, gate 0.15) AND its L2-residual
  surprise-AUC = 0.7625 (gate 0.70), while the EMA JEPA predictor `g` trained
  with `jepa_contrastive_loss` (cosine + logsumexp negatives + 0.1·MSE) scored
  surprise-AUC = 0.565 in its native cosine distance — *below* the linear
  baseline. Four reasons JEPA loses here: (1) the surprise signal is naturally an
  L2 residual (magnitude of the miss), but the JEPA contrastive loss optimizes
  cosine direction (scale-invariant) — a predictor can have low cosine loss and
  poor magnitude calibration, which is exactly what `g` showed (its L2 MSE was
  83× worse than linear); (2) the logsumexp negatives push the prediction *away*
  from the other batch targets, but consecutive `z_{t+1}` states are similar, so
  the anti-collapse term fights the prediction objective; (3) the backbone is
  frozen, so there is no encoder that could collapse — the entire reason JEPA
  exists (collapse-resistant self-supervision) does not apply; (4) JEPA needs an
  EMA target encoder, stop-grad, temperature, negative count, and a training loop,
  vs. a closed-form one-shot ridge fit with no hyperparameters. So v1 ships the
  linear map: `min ‖A z_t + b − z_{t+1}‖²` (ridge, closed-form). The honest
  upgrade path if R²=0.30 becomes limiting is a light **MSE-trained MLP** —
  non-linear dynamics under the same L2 objective the surprise signal wants; the
  probe's `g` did poorly *because it was JEPA-trained*, not because MLPs are bad.
  **JEPA is reserved for a future generative rollout** (predict `k>1` steps
  ahead for imagination/rollout), where a learned latent dynamics model is the
  point and the collapse-resistant objective earns its place — that is a
  different feature than the v1 surprise signal and past v1.
  **Gate ordering (already run):** linear baseline first (Phase 0b step 1) — if it
  had not beaten mean, the latent-dynamics head would be dropped before any
  predictor work. It beat mean, so step 2 (EMA JEPA) ran and lost to linear. The
  v1 head is the linear map; EMA/stop-grad/anti-collapse is not built.
- **Relevance head** — supervised on query-anchor pairs. The `record_feedback`
  tool (`tools.py:44-72`) has the LLM rate each cited unit 1-5, but today those
  ratings are reduced to a compounded boost multiplier (`store.record_feedback`,
  `store.py:707-749`) and the **raw 1-5 judgments are not persisted** — only the
  multiplier at `content/unit_boost/{unit_id}` survives. So Phase 2a must add a
  raw-rating JSONL tap (write `{unit_id, rating, query, slot}` *before* the
  reduction) before this head can train on real labels; until then, fall back to
  synthetic labels à la `scripts/generate_jepa_training_data.py`.
- **Graduation head** — two stages, v1 cheap then v2 accurate. **v1 proxy:**
  graduation score = the slot's integrated relevance over its lifetime in the
  buffer (`∫ r_i dt` — a fact that stayed relevant for many steps is probably
  worth keeping). This breaks the circular dependency for free: "would have been
  needed later" is approximated by "was relevant for a long time," which needs no
  replay and no downstream pipeline. Ship v1 as the heuristic baseline.
  **v2 replay:** supervise on "would this fact have been useful later?" labels
  from replaying logged streams and marking which compressed-out facts later
  triggered a salience recall or a consumer `search_memory`/`expand`. This is the
  true credit-assignment signal but depends on the whole downstream pipeline
  (relevance → context-builder → consumer behavior), so the labels are noisy and
  slow — the long pole. Reuses the `PresentationGate` replay buffers
  (`outcome_buffer`, `override_buffer`) as a seed source. Train v2 only once the
  pipeline exists and v1 is shipping.

### 6.3 Context-builder Transformer

Train on context quality: does the block it produces let a downstream consumer
answer correctly? Reuse the existing contradiction/citation eval sets (Phase 3c,
ERAG-Bench) — the "factual accuracy" goal already has ground-truth labels in the
repo. The `PresentationGate` override buffer (caller overrode the heuristic) is
seed supervision for the learned gate.

### 6.4 Stage B — joint fine-tune (optional, only if Stage A plateaus)

Unfreeze the backbone and jointly train SSM + heads + builder with the combined
objective. Risk: catastrophic forgetting of the pretrained representations the
2a/2b/doc-kind work relies on. Only if Stage A's frozen backbone is the
bottleneck.

### 6.5 Honest training caveats

- **None of the v1 heads are JEPA.** Three of the four heads (relevance,
  recoverability, graduation) are supervised probes that happen to use
  reconstruction-error-derived labels or framing; the fourth (latent-dynamics) is
  a **linear** next-state predictor per the 0b probe. The original v0.2 framing
  called the dynamics head "real JEPA" needing EMA/stop-grad/anti-collapse; the
  probe retired that — a closed-form linear map beats the JEPA predictor on the
  surprise signal, and the frozen backbone means collapse cannot occur. Do not
  drag EMA target encoders into any of the four heads in v1. JEPA only re-enters
  if a future generative rollout (predict `k>1` for imagination) is built — a
  separate feature, past v1.
- **The dependency half of salience is the hard part.** Recoverability is
  well-defined and labelable. "Needed for factual accuracy" is query-dependent
  and harder. Start with the weaker proxy
  `salience = recoverability_drop on recently-high-relevance slots`, validate,
  then upgrade to true dependency modeling.
- **Graduation labels are the long pole — so ship the v1 proxy first.** v2
  replay labels require replay over long streams to know what *would have been*
  needed later, and depend on the whole downstream pipeline (circular). The v1
  proxy (integrated relevance over slot lifetime) breaks the circle and ships
  without replay. Do not block the "remembering" feature on the long pole.
- **Multi-objective balancing.** Stage B has at least four losses (the four
  heads) + context quality. Expect tuning, not a clean single-objective train.
- **Thresholds exist.** `θ`, `φ`, `C`, `K`, `Δ`, the surprise gate, and the
  graduation priority cutoff are a real tuning surface. "Seamless" is a
  direction (minimize the surface), not a state. Any version claiming zero
  thresholds is selling something.

---

## 7. Open questions (decide before implementation, not during)

- **Does `y_t` suffice, or does the builder need `state_t`?** Start with `y_t`
  (zero new params, uses trained `W_C`). Add a `proj(state_t)` head jointly only
  if the builder can't produce good context from `y_t`. Local change, not a
  redesign.
- **Soft attention vs hard top-m.** Start soft. Add hard selection only if
  discrete recall is required downstream.
- **Latent-dynamics horizon `k`.** Short `k` (1-2) is easy to train but a weak
  world model; long `k` is more useful but harder to learn and more collapse-prone.
  Start short, lengthen only if the surprise signal is too noisy at short `k`.
- **Surprise → anomaly interlock.** Should STM surprise feed the existing
  `AnomalyHead` (retrain it to accept an STM feature), or stay a separate signal
  consumed by the orchestrator? Separate is cheaper; integrated is more
  principled. Decide after the §6.1 probe.
- **Sizing:** `K` (lookback), `C` (output budget), `θ`/`φ` (salience thresholds),
  `Δ` (freshness watermark). Pick initial values from probe data, not by gut.
- **Anchor set source.** High-`r` slots only, or an externally supplied
  task-anchor set? Self-contained by default, optional external anchors.
- **Thread 2 reuse vs. new path.** Can the existing doc-ingestion pipeline
  consume the activity feed as-is, or does it need an "activity ingest" adapter?
  Probably reuse, but the activity feed is not a document — verify the parsers do
  something sensible on event-stream input.

---

## 8. What STRM is not

- Not a reasoning engine. It builds context and emits signals; the consumer
  reasons.
- Not a single fixed latent vector that "bypasses the context window." The output
  is a bounded block at a chosen budget, not a 384-float bottleneck. (The
  one-vector framing from the source conversation is explicitly rejected — it is
  information-bottlenecked and loses precision recall.)
- Not long-term memory. Ponder owns LTM; STRM points at it and graduates into it.
- Not a replacement for the reactive contradiction/citation guards or the LTM
  anomaly/salience/consolidation heads. It is their STM-side sibling and feeds
  them.
- Not greenfield. The SSM substrate (`WorkingMemory`), re-injection (`inject()`),
  the persistence/snapshot contract, the `llm_signal` importance channel, and the
  entire LTM consolidation/anomaly/forgetting/retrieval stack already ship. STRM
  adds the read-out of the WM state (ring buffer + four heads + context-builder)
  and the wiring that lets that read-out drive the existing LTM machinery.

---

## 9. First move (done 2026-07-19)

The §6.1 recoverability probe and the §6.2 dynamics de-risk ran on the trained
`WorkingMemory` backbone — no retraining, on infrastructure that already exists.
Both gates PASSED:

- **0a recoverability — GO.** Probe AUC val = 0.810 (gate 0.75), beats the free
  `k`-baseline (0.732). The recoverability head is viable; its labels are
  decoder `D`'s reconstruction error `e(i,t)`.
- **0b latent-dynamics — GO, ship LINEAR.** Linear `z_{t+1}=Az_t+b` R²=0.297 over
  constant-mean; linear surprise-AUC (L2 residual) = 0.7625 (gate 0.70), beating
  the EMA JEPA predictor (cosine surprise-AUC 0.565). The latent-dynamics head
  ships as the linear map; JEPA is deferred to a future generative rollout.

Everything else is downstream of those numbers, and both said GO. Next move is
Phase 2 — the four heads, starting with the recoverability head (cheapest: labels
are already free from 0a) and the latent-dynamics head (now linear, the cheapest
of the four — a closed-form ridge fit, no training loop).