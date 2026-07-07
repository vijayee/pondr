# Phase 2b: Retrieval Gate Instance — Implementation Plan for Claude Code

## Overview

**Goal:** Train the first JGS instance — the subconscious router that decides where
to look, which pathway to use, what model size is needed, and whether conscious
deliberation is required. This is the component that makes the ponder engine feel
like it anticipates your needs before you finish asking.

**What "done" looks like:** A trained Retrieval Gate instance that receives a prompt
embedding (and an optional `GateContext`) and outputs a routing decision: which
domain(s) to query, which pathway to use (`ssm_direct`, `graph_retrieve`,
`process_exec`, `tool_plan`, `conscious_deliberation`), what meta-skills are
required, what model size is needed, and whether conscious deliberation is
necessary. The gate learns from outcomes — successful routes are reinforced,
delegation surprises are penalized, overkill is penalized.

**Prerequisite:** Phase 2a complete — `JGSBackbone` trained and validated on
temporal-chain sequences (`backbone_final.pt`, 19.5M params, `ReferenceSSM`
backend, `d_model=384`), instance-config templates defined. Phase 1b retrieval
pipeline operational. Oracle-generated JEPA routing pairs — **generated this
phase** via `deepseek-v4-flash:cloud` (the Phase 1d run only produced a 5-example
validate slice; the full 5,000 were generated for 2b, with `tool_plan` templates
added after a first run came back with `tool_plan=0/5000`).

**Status:** Implemented and tested. Code in `src/subconscious/{routing,
retrieval_gate}.py`, `src/subconscious/training/routing_training.py`; integration
in `src/retrieval/retriever.py` + `src/generation/mode_a.py`; training CLI
`scripts/train_retrieval_gate.py`; offline tests `tests/test_retrieval_gate.py`
(13 passing, no model download / no Ollama / no WaveDB on the gate path).

---

## 0. Alignment Notes (doc-vs-reality corrections)

This doc was originally written speculatively against an imagined API. The
implementation differs in many places. The table records every correction so the
doc and the code agree; the sections below reflect the **corrected** design.

| Original doc said | Reality (corrected) |
|---|---|
| `d_model=512`; heads `nn.Linear(512, …)` | `BackboneConfig.d_model=384`; instance `output_dim=256` — the heads consume `step()`'s `output` (256-dim), **not** 512; `pred_dim=384` |
| `get_embedding(prompt)` = "text-embedding-3-small or local" | Local **bge-small-en-v1.5** (384-dim) via an **injectable `Embedder`** Protocol (`encode(list[str]) -> list[list[float]]`); **no OpenAI** |
| `planner_model="gpt-4o-mini"` | Local **Bonsai llama-server** (`config.bonsai_*`, `prism-ml/Ternary-Bonsai-8B-gguf`); the planner's plan has **no `domains` key** |
| external `ssm_state = torch.zeros(1,16,512)` threaded into `route()` | `JGSInstance` manages its **own** recurrent state (`instance.state`, per layer `[batch,16,384]`); no `WorkingMemory` object exists yet — gate state **resets per query** in 2b; `GateContext`'s 3 features default to **zeros** (a Phase 2.5 hook) |
| `gate.last_output` cached attribute | `forward()` returns the `output` tensor explicitly — **no cached attribute** (an anti-pattern the implementation deliberately avoids) |
| checkpoint = bare `gate.state_dict()` | Real checkpoint is a wrapper `{"gate": gate.state_dict(), "val_accuracy", "epoch"}` (the gate `state_dict()` excludes the backbone — it's not a submodule) |
| `HippocampalRetriever(store, retrieval_gate, working_memory, planner_model)` | Real: `(store, planner=None, auto_load_index=False, retrieval_gate=None, embedder=None)`; `retrieve()` returns **`list[dict]`** (unchanged, for back-compat); new `retrieve_with_routing()` returns the routing dict |
| `ModeAGenerator` model-size ladder (`_get_model_for_size`, `model.size`) | No ladder exists; real `__init__(retriever, model, endpoint, …)`; predicted size is **recorded** in the route, generation uses the single Bonsai model now |
| 13 `META_SKILLS`, 7 `AVAILABLE_DOMAINS` (incl. `general`, `process_adaptation`, `process_invention`, `verification`, `delegation_judgment`, `cross_domain`) | Oracle's actual emitted vocab: **8 skills**, **6 domains** (no `general`); constants align to the real labels the Oracle produces |
| `ReplayEntry(prompt, decision, outcome, timestamp)` (generic value/cost entry) | 2b uses a **routing-specific** `RoutingReplayEntry(embedding, context, decision, outcome)` — the outcome trainer re-runs `forward` on the stored embedding, so the prompt string isn't kept |
| "5,000+ pairs from Phase 1d" prerequisite | Generated this phase via `deepseek-v4-flash:cloud` (1d only had 5). Two coverage issues surfaced and were fixed in sequence: (1) a `tool_plan` template gap (`tool_plan=0/5000`) → added tool-plan templates + regenerated; (2) the generator's **tiny vocabulary** produced only ~177 unique query→route mappings across the 5000 records (minority pathways had 2–9 unique examples each) → **expanded the vocab** (entities 8→40, topics 5→25, tones 3→10, etc.) + regenerated for genuine diversity |

**Routed but not executed (honest, not faked):** `process_exec` / `tool_plan` /
`ssm_direct` pathways are **routed** (the gate predicts them) but not yet
**executable** end-to-end — no stored-process / tool / System-2 / working-memory
infrastructure exists. The integration returns the route plus a `supported=False`
flag and empty results; it never fakes a response. The model-size ladder is
likewise **recorded but not selected** — generation uses the single configured
Bonsai model.

---

## 1. What Phase 2b Delivers

| Artifact | Description | Consumer |
|---|---|---|
| **Retrieval Gate instance** | Trained JGS instance that routes queries before retrieval | Real-time query pipeline |
| **Domain router** | Predicts domain(s) (multi-label) from the prompt embedding | Future domain-aware traversal (recorded now; 1b traversal is domain-agnostic) |
| **Pathway selector** | Chooses `ssm_direct` / `graph_retrieve` / `process_exec` / `tool_plan` / `conscious_deliberation` | Retrieval orchestrator |
| **Model size predictor** | Predicts required model size (1B–175B) | Future delegation ladder (recorded now) |
| **Deliberation gate** | Decides whether System 2 needs to engage | Conscious/subconscious split |
| **Outcome-based learning** | Gate weights updated from routing outcomes (REINFORCE) | Continuous improvement |
| **Integration with Phase 1b** | `retrieve_with_routing` + `generate_with_routing` (opt-in; back-compat preserved) | End-to-end system |

---

## 2. Architecture

### 2.1 The Retrieval Gate in Context

```plaintext
User: "What was I frustrated about last week?"
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                 RETRIEVAL GATE (Subconscious)                 │
│                                                              │
│  Input: prompt embedding [batch, 384]  (local bge-small)      │
│         + optional GateContext (3 features, zeros in 2b)      │
│                                                              │
│  step() → output [batch, 256]  (instance output_dim)         │
│  5 heads over output → logits                                 │
│                                                              │
│  Predicts:                                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Domain(s): database (multi-label, sigmoid > 0.3)      │    │
│  │ Pathway:   graph_retrieve (argmax softmax)           │    │
│  │ Skills:    [factual_recall, basic_synthesis]         │    │
│  │ Model size:3B  (argmax)                              │    │
│  │ Deliberation: NOT NEEDED (sigmoid > 0.5)             │    │
│  └──────────────────────────────────────────────────────┘    │
│  Returns (logits, gate_decision, output) — no cached attr    │
└────────────────────────────┬─────────────────────────────────┘
                             │  RoutingDecision
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                 RETRIEVAL PIPELINE (Phase 1b, unchanged)       │
│                                                              │
│  graph_retrieve / conscious_deliberation (supported):         │
│    Bonsai plans query → Graph traversal → context built       │
│    → Bonsai model synthesizes response                       │
│  ssm_direct / process_exec / tool_plan (unsupported):         │
│    supported=False, results=[] — surfaced honestly           │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 The Retrieval Gate as a JGS Instance

The Retrieval Gate is the first JGS instance trained on the shared backbone from
Phase 2a. It subclasses `JGSInstance` and adds five routing heads over the
instance `output` (256-dim). The shared backbone is frozen; only the
instance-owned params (input/output projections + LoRA, the decomposed gate) and
the five heads train. Because the backbone is stored via `object.__setattr__`
(not registered as a submodule), `gate.parameters()` **excludes** the backbone —
so `AdamW(gate.parameters(), …)` naturally leaves it alone (the trainer also
freezes it explicitly for grad-flow safety).

```python
class RetrievalGate(JGSInstance):
    """The subconscious router. First trained JGS instance.

    Owns five routing heads on top of the shared JGSInstance base. The shared
    backbone is frozen during instance training (Phase 2a weights); only the
    instance-owned params (input/output projections + LoRA, the decomposed
    gate) and the five routing heads train. gate.parameters() already excludes
    the backbone (stored via object.__setattr__), so an AdamW(gate.parameters(),
    …) optimizer naturally leaves the backbone alone.
    """

    def __init__(self, backbone, config: Optional[InstanceConfig] = None):
        cfg = config or INSTANCE_CONFIGS["retrieval_gate"]
        super().__init__(backbone, cfg)
        d = cfg.output_dim  # 256 — the instance step output the heads consume

        # ── Routing heads (trained on Oracle pairs) ──
        # Domain / pathway / skill share a 256→vocab hidden; model-size and
        # deliberation are smaller (their vocab is tiny). All consume the
        # instance `output` (output_dim=256), NOT d_model=384/512.
        self.domain_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, len(AVAILABLE_DOMAINS)))
        self.pathway_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, len(PATHWAYS)))
        self.skill_head = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Linear(256, len(META_SKILLS)))
        self.model_size_head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Linear(128, len(MODEL_SIZES)))
        self.deliberation_head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, prompt_embedding: Tensor,
                context: Optional[GateContext] = None
                ) -> tuple[dict[str, Tensor], GateDecision, Tensor]:
        """Run the instance step + the five routing heads.

        Args:
            prompt_embedding: [batch, input_dim=384].
            context: optional GateContext (entity_recency, topic_recency,
                query_complexity). Zeros when None (Phase 2.5 will populate).

        Returns:
            (logits, gate_decision, output). logits is
            {"domain","pathway","skill","model_size","deliberation"} each
            [batch, vocab] (deliberation is [batch, 1]); gate_decision is the
            decomposed-gate output; output is the instance step() output
            [batch, output_dim=256] (returned explicitly — no cached attr).
        """
        output, _predicted, gate_decision = self.step(prompt_embedding, context)
        logits = {
            "domain":      self.domain_head(output),
            "pathway":     self.pathway_head(output),
            "skill":       self.skill_head(output),
            "model_size":  self.model_size_head(output),
            "deliberation": self.deliberation_head(output),
        }
        return logits, gate_decision, output

    def route(self, prompt_embedding: Tensor,
              context: Optional[GateContext] = None) -> RoutingDecision:
        """Inference: forward → decode logits → RoutingDecision (batch=1)."""
        logits, gate_decision, _output = self.forward(prompt_embedding, context)
        return self.decode_batch(logits, gate_decision)[0]

    def route_text(self, prompt: str, embedder: Embedder,
                   context: Optional[GateContext] = None) -> RoutingDecision:
        """Embed prompt via the injected embedder, then route.

        The embedder is the caller's responsibility (the integration layer
        passes the real bge-small VectorSearch embedder; tests pass a
        deterministic stub). Keeping it out of the constructor preserves the
        package's torch-only import surface.
        """
        vec = embedder.encode([prompt])[0]
        emb = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # [1, 384]
        device = next(self.parameters()).device
        return self.route(emb.to(device), context)
```

Decoding (`decode_batch`) is batched: domain multi-label at sigmoid `> 0.3` with
an **argmax fallback** so no route is vacuous; pathway argmax softmax; skills
multi-label at `> 0.5` (may be empty — auxiliary); model_size argmax; deliberation
sigmoid `> 0.5`. The gate returns ONE `GateDecision` (its live contract is one
decision per step), so every row in a batch shares that decision's confidence —
batched evaluation uses per-row logits for the discrete choices and the shared
gate confidence for the `confidence` field.

### 2.3 Routing Decision Data Model

`src/subconscious/routing.py` defines the vocab constants (matching the Oracle's
**real** emitted labels), the discrete `RoutingDecision`, the `RoutingOutcome`
with its `reward()`, and a routing-specific `RoutingReplayEntry`.

```python
# Vocab — aligned to the labels the Oracle actually emits (not the original
# 13-skill / 7-domain spec).
AVAILABLE_DOMAINS = ["database", "coding", "robotics", "economics",
                     "ai_architecture", "personal"]          # 6 (no "general")
PATHWAYS = ["ssm_direct", "graph_retrieve", "process_exec",
            "tool_plan", "conscious_deliberation"]           # 5
META_SKILLS = ["factual_recall", "basic_synthesis", "pattern_recognition",
               "decomposition", "process_selection", "creative_synthesis",
               "security_analysis", "tradeoff_analysis"]     # 8
MODEL_SIZES = ["1B", "3B", "8B", "70B", "175B"]               # 5

@dataclass
class RoutingDecision:
    domains: list[str]                       # ["database", "coding"]
    pathway: str                            # "graph_retrieve"
    meta_skills: list[str]                  # ["factual_recall", "basic_synthesis"]
    model_size: str                         # "3B"
    needs_deliberation: bool                # False
    confidence: float                       # 0.89
    gate_decision: Optional[GateDecision]   # raw gate output for learning

@dataclass
class RoutingOutcome:
    user_accepted: bool
    user_corrected: bool
    had_to_delegate: bool
    model_was_overkill: bool
    response_fast: bool
    had_to_expand: bool

    def reward(self) -> float:
        """Doc §3.3 weighting: +1 accepted, −1 corrected, −0.3 delegate,
        −0.1 overkill, +0.1 fast, −0.3 expand."""
        return (+1.0 * self.user_accepted - 1.0 * self.user_corrected
                - 0.3 * self.had_to_delegate - 0.1 * self.model_was_overkill
                + 0.1 * self.response_fast - 0.3 * self.had_to_expand)

@dataclass
class RoutingReplayEntry:
    embedding: Tensor          # the prompt embedding [1, 384] (re-run forward on it)
    context: Optional[GateContext]
    decision: RoutingDecision
    outcome: Optional[RoutingOutcome] = None
    filled: bool = False
```

The decision is **discrete and detached** (no logits carried) — the outcome
trainer re-runs `forward` on the stored embedding rather than caching logits, so
the replay entry stays small and the graph is rebuilt fresh each update.

---

## 3. Training Procedure

### 3.1 Training Data

**Source:** JEPA routing pairs (`data/training/jepa/routing_pairs.jsonl`),
5,000 examples generated via `deepseek-v4-flash:cloud` (local Ollama, Ollama
credits — no OpenAI). The generator (`scripts/generate_jepa_training_data.py`)
covers all five pathways.

**Label provenance (important methodology note):** the `pathway` label is the
**template intent** (`expected_pathways[0]`), NOT the Oracle's pathway choice.
The Oracle is a reliable labeler for `domains`, `meta_skills`, `model_size`, and
`needs_deliberation`, but its **pathway** label collapses to `graph_retrieve`
(it re-labels most `tool_plan`/`process_exec`/`ssm_direct` intents as "needs
retrieval") — a gate trained on the Oracle's pathway routes everything to
`graph_retrieve` (verified: 100% majority-class collapse, even on training data,
even with inverse-frequency class weighting). The template's expected pathway IS
the routing intent by construction ("Plan the sequence of tool calls…" =
`tool_plan`), so it is cleaner pathway ground truth. The Oracle's original
pathway choice is preserved in each record as `oracle_pathway` for audit. The
other four fields stay Oracle-labeled. The generator's `to_record` writes this
split directly, so regeneration is consistent.

**Diversity journey (three fixes):**

1. **`tool_plan` template gap.** The first run came back with `tool_plan=0/5000`
   — the generator's templates never expressed a tool-plan intent, so the
   Oracle was never biased toward it. Closed by adding two tool-plan templates
   (`_TASKS` / `_OBJECTIVES` vocab) + regenerating.
2. **Vocabulary diversity.** The regenerated 5000 were only ~177 **unique**
   query→route mappings (the generator's tiny vocabulary produced heavy
   duplication; minority pathways had 2–9 unique examples each). Training on
   that would memorize exact strings, not generalize. Closed by **expanding the
   vocab** (entities 8→40, topics 5→25, tones 3→10, events 4→12, decisions 3→13,
   problems 3→12, tasks 5→20, objectives 4→15) so the cross-product yields
   ~956 unique queries + regenerating.
3. **Pathway-label collapse.** Even with balanced templates, the Oracle's
   pathway labels were ~61% `graph_retrieve`, and a gate trained on them
   collapsed to a constant `graph_retrieve`/`3B` predictor (0% recall on every
   minority class *on the training set itself*, so not a generalization issue —
   a training-fit failure). Inverse-frequency class weighting did NOT break the
   collapse (the minority classes are too thin: `process_exec` ~3 unique). Closed
   by switching the `pathway` training label to the template intent (above),
   which is balanced by construction. The resulting gate routes diversely
   (best val 0.826; per-class recall on unique train: graph_retrieve 96%,
   conscious_deliberation 84%, tool_plan 100%, ssm_direct 98%; `process_exec`
   remains the weak class at 0/3 — only 3 unique examples, unlearnable, a
   documented limitation).

The earlier runs are kept for reference: `routing_pairs_4class_backup.jsonl`
(first 4-class run), `routing_pairs_177unique_backup.jsonl` (tool_plan-covered,
pre-vocab-expansion), and `routing_pairs_oraclepathway_backup.jsonl` (the
expanded-vocab set with the Oracle's original — collapsed — pathway labels,
before the template-intent re-label).

**Format:**
```json
{
  "query": "What did Alice say about the WAL config?",
  "route": {
    "domains": ["database"],
    "pathway": "graph_retrieve",
    "oracle_pathway": "graph_retrieve",
    "meta_skills": ["factual_recall", "basic_synthesis"],
    "model_size": "3B",
    "needs_deliberation": false,
    "confidence": 0.85,
    "reasoning": "…"
  },
  "expected_pathways": ["graph_retrieve"],
  "cost": 0.0
}
```

**Malformed records:** the Oracle occasionally returns a slightly-off schema
(`route.domain` singular, out-of-vocab `pathway`/`model_size`, or a non-dict
`route`). `load_routing_pairs` **drops** such records rather than silently
degrading them to default labels (a silent wrong label is worse than a smaller
dataset): it requires `query` to be a str, `route` to be a dict, `domains` to be
a list, and `pathway`/`model_size` to be in-vocab. ~0.6% of records are dropped.

**Split:** 80% train / 20% validation, deterministic via the configured seed,
**after dedup by query** (see §open notes) so the same query cannot land in both
splits.

### 3.2 Supervised Training on JEPA Pairs

```python
def train_retrieval_gate_supervised(gate, backbone, train_data, val_data,
                                    embedder, config, device=None,
                                    progress_cb=None):
    """Train the Retrieval Gate on JEPA routing pairs.

    Pathway label = template intent (expected_pathways[0]); domains/skills/
    model_size/deliberation = Oracle labels (see §3.1 provenance).

    The backbone is frozen. Only the gate's routing heads and instance-specific
    parameters (projections, LoRA, decomposed gate) train. Embeddings are
    computed ONCE for all queries up front (the pairs are query strings, so the
    embedding is fixed); state is reset per batch (no BPTT across batches).
    """
    for param in backbone.parameters():        # explicit freeze (grad safety)
        param.requires_grad = False
    optimizer = torch.optim.AdamW(gate.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    train_emb = _embed_all(embedder, [ex["query"] for ex in train_data], device)
    val_emb   = _embed_all(embedder, [ex["query"] for ex in val_data], device)

    best_val = 0.0
    log = []
    for epoch in range(config.epochs):
        order = torch.randperm(len(train_data), generator=g).tolist()
        total_loss = 0.0
        for start in range(0, len(train_data), config.batch_size):
            idx = order[start:start + config.batch_size]
            batch = [train_data[i] for i in idx]
            emb = train_emb[idx]                       # [B, 384]
            gate.reset_state(len(idx))                 # reset state per batch
            logits, _, _ = gate.forward(emb)           # batched
            targets = _routing_targets(batch, device)
            loss = _routing_loss(logits, targets, config.skill_loss_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        val_acc = evaluate_routing(gate, val_data, val_emb, device)
        log.append({"epoch": epoch, "train_loss": total_loss, "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            torch.save(gate.state_dict(), ckpt_dir / "best.pt")  # heads + instance
        if progress_cb: progress_cb(epoch, total_loss, val_acc)
    torch.save(gate.state_dict(), ckpt_dir / "final.pt")
    write_json(ckpt_dir / "train_log.json", log)
    return {"best_val": best_val, "log": log}
```

`_routing_loss` sums: domain BCE-with-logits (multi-label) + pathway cross-entropy
+ `skill_loss_weight × skill` BCE (auxiliary, default 0.5) + model_size
cross-entropy + deliberation BCE. Checkpoints are a wrapper
`{"gate": gate.state_dict(), "val_accuracy", "epoch"}` — the gate `state_dict()`
holds the heads + instance params only; the backbone is excluded (not a
submodule). The train CLI (`scripts/train_retrieval_gate.py`) **dedups by query
before the split** so the same query can't land in both train and val (the
Oracle cache replays identical query strings → duplicates would otherwise leak).

### 3.3 Outcome-Based Learning (Personalization)

After supervised pretraining, the gate learns from real outcomes — this is what
makes the subconscious personalized. `OutcomeBasedTrainer` uses a replay buffer of
`RoutingReplayEntry` and trains offline with a REINFORCE-style policy gradient.

```python
class OutcomeBasedTrainer:
    """Fine-tunes the Retrieval Gate from actual routing outcomes.

    REINFORCE: −reward · log p(chosen) summed across the five heads. The replay
    entry stores the embedding (not the prompt string) so forward is re-run
    fresh. train_from_outcomes is a NO-OP until len(buffer) >= min_buffer.
    """

    def __init__(self, gate, config=None):
        self.gate = gate
        cfg = config or RetrievalGateTrainingConfig()
        self.buffer = []   # list[RoutingReplayEntry] (bounded by capacity)
        self.optimizer = torch.optim.AdamW(gate.parameters(), lr=cfg.online_lr,
                                           weight_decay=0.01)

    def record_outcome(self, embedding, context, decision, outcome):
        self.buffer.append(RoutingReplayEntry(embedding, context, decision,
                                              outcome, filled=True))
        if len(self.buffer) > cfg.replay_buffer_capacity:
            self.buffer.pop(0)

    def train_from_outcomes(self, batch_size=None):
        if len(self.buffer) < cfg.min_buffer:
            return 0.0                        # no-op below min_buffer
        batch = sample(self.buffer, batch_size)
        self.optimizer.zero_grad()
        total = 0.0
        for entry in batch:
            gate.reset_state(1)
            logits, _, _ = gate.forward(entry.embedding, entry.context)
            total += _reinforce_loss(logits, entry.decision, entry.outcome.reward())
        total.backward(); self.optimizer.step()
        return total.item()
```

`_reinforce_loss` reinforces the **chosen** discrete decisions: for pathway and
model_size (softmax) it adds `−reward · log p(chosen)`; for domain, skill, and
deliberation (sigmoid) it adds `−reward · log σ(logit_chosen)` for each chosen
label. The reward weighting (`RoutingOutcome.reward()`) is the doc's §3.3
weighting: `+1` accepted, `−1` corrected, `−0.3` delegate, `−0.1` overkill, `+0.1`
fast, `−0.3` expand.

### 3.4 Training Configuration

```python
@dataclass
class RetrievalGateTrainingConfig:
    d_model: int = 384          # = embedder dim (bge-small), NOT 512
    d_state: int = 16
    lora_rank: int = 4
    # Supervised (Oracle pairs)
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    skill_loss_weight: float = 0.5
    # Outcome-based
    online_lr: float = 1e-5
    replay_buffer_capacity: int = 1000
    outcome_batch_size: int = 32
    min_buffer: int = 50        # no-op until this many outcomes
    # Hardware
    dtype: str = "float32"     # bf16/autocast still unfixed in the 2a path;
                               # gate params are small → fp32 is fine
    device: str = "auto"
    val_fraction: float = 0.2
    seed: int = 0
    backbone_path: str = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
    checkpoint_dir: str = "data/training/routing_gate"
```

### 3.5 Backbone Loader

```python
def load_backbone(path, config=BackboneConfig(), device="auto", map_location="cpu"):
    """Strict-load the frozen Phase 2a backbone.

    torch.load → ckpt["backbone"] → JGSBackbone(config).load_state_dict(strict=True)
    (raises RuntimeError on any missing/unexpected key). All params frozen, .eval().
    """
```

---

## 4. Integration with the Retrieval Pipeline

### 4.1 Updated HippocampalRetriever (backward-compatible)

`HippocampalRetriever.__init__` gains **optional** `retrieval_gate` and
`embedder` params. Existing behavior is unchanged: `retrieve()` still returns
`list[dict]` so `ModeAGenerator.generate()` keeps working. The routing path is
opt-in via `retrieve_with_routing`.

```python
class HippocampalRetriever:
    def __init__(self, store, planner=None, auto_load_index=False,
                 retrieval_gate=None, embedder=None):
        ...
        self.gate = retrieval_gate
        self._route_embedder = embedder
        # If no embedder passed but a VectorSearch index was auto-loaded, reuse
        # its embedder (the real bge-small) for routing.
        if self.gate is not None and self._route_embedder is None \
                and self.vector_search is not None:
            self._route_embedder = self.vector_search
        self._outcome_trainer = None   # lazily built on first record_outcome

    def retrieve_with_routing(self, prompt, conversation_history=None,
                               use_semantic=True) -> dict:
        """Retrieve with the gate consulted first.

        Returns {"type", "route", "results", "context", "supported"}.

        - graph_retrieve / conscious_deliberation (supported): run the existing
          retrieve() pipeline + build_context_string. The gate's predicted
          domains are recorded in the route but do NOT filter traversal — the
          Phase 1b traversal is domain-agnostic (it scores on entities/topics/
          tones), so filtering by domain here would be theater. Domain-aware
          traversal is a later phase; the route carries the domains for it.
        - ssm_direct / process_exec / tool_plan (unsupported): no executor wired
          yet → supported=False, results=[], context=None. Honest, not faked.

        Raises RuntimeError if no gate / no embedder was configured.
        """
        if self.gate is None: raise RuntimeError("requires a retrieval_gate")
        if self._route_embedder is None: raise RuntimeError("requires an embedder")
        route = self.gate.route_text(prompt, self._route_embedder)
        if route.pathway in ("graph_retrieve", "conscious_deliberation"):
            results = self.retrieve(prompt, conversation_history=conversation_history,
                                    use_semantic=use_semantic)
            context = self.build_context_string(results) if results else None
            return {"type": route.pathway, "route": route,
                    "results": results, "context": context, "supported": True}
        return {"type": route.pathway, "route": route,
                "results": [], "context": None, "supported": False}

    def record_outcome(self, prompt, route, outcome):
        """Push (embedding, context, decision, outcome) to the outcome trainer.
        No-op unless a gate + embedder are configured."""
        if self.gate is None or self._route_embedder is None: return
        from ..subconscious.training.routing_training import OutcomeBasedTrainer
        if self._outcome_trainer is None:
            self._outcome_trainer = OutcomeBasedTrainer(self.gate)
        emb = torch.tensor(self._route_embedder.encode([prompt])[0],
                           dtype=torch.float32).unsqueeze(0)
        self._outcome_trainer.record_outcome(emb, None, route, outcome)
```

### 4.2 Updated Mode A Generator (backward-compatible)

`generate()` is unchanged. New `generate_with_routing` is opt-in:

```python
class ModeAGenerator:
    def generate_with_routing(self, prompt, conversation_history=None,
                               max_context_tokens=None) -> dict:
        """Generate using the subconscious Retrieval Gate (Phase 2b, opt-in).

        - supported (graph_retrieve / conscious_deliberation): build context
          from retrieved episodes + complete via the local Bonsai endpoint
          (reuses _complete). The gate's predicted model_size is recorded in the
          result for the future model-size ladder — generation itself still
          uses the single configured Bonsai model (the ladder is a later phase;
          we don't fake a multi-model selection).
        - unsupported (ssm_direct / process_exec / tool_plan): response is None,
          supported=False — surfaced honestly, not faked.

        Returns {"response", "route", "retrieved_episodes", "model_used",
        "context_used", "supported"}.
        """
        r = self.retriever.retrieve_with_routing(
            prompt, conversation_history=conversation_history)
        route = r["route"]
        if not r["supported"]:
            return {"response": None, "route": route, "retrieved_episodes": [],
                    "model_used": None, "context_used": None, "supported": False}
        episodes = r.get("results", [])
        context = r.get("context") or self.retriever.build_context_string(
            episodes, max_context_tokens)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if conversation_history: messages.extend(conversation_history[-10:])
        messages.append({"role": "user",
                         "content": f"Context from past conversations:\n{context}\n\nUser: {prompt}"})
        return {"response": self._complete(messages), "route": route,
                "retrieved_episodes": episodes, "model_used": self.model,
                "context_used": context, "supported": True}
```

---

## 5. Testing Strategy

`tests/test_retrieval_gate.py` is **fully offline**: CPU `ReferenceSSM`, the
deterministic `stub` embedder (no `sentence_transformers` download), no Ollama,
no WaveDB on the gate path. The integration test uses a tmp_path WaveDB store
with a stub planner (mirrors `tests/test_retriever.py`). The backbone-load test
is gated on the Phase 2a checkpoint existing locally so the suite runs on a
fresh clone. 13 tests, all passing.

```python
# Contract / shape
def test_forward_returns_five_logit_heads_with_correct_vocab():
    gate = RetrievalGate(JGSBackbone(BackboneConfig()))
    gate.reset_state(3)
    logits, gate_decision, output = gate.forward(torch.randn(3, 384))
    assert logits["domain"].shape == (3, len(AVAILABLE_DOMAINS))     # 6
    assert logits["pathway"].shape == (3, len(PATHWAYS))            # 5
    assert logits["skill"].shape == (3, len(META_SKILLS))           # 8
    assert logits["model_size"].shape == (3, len(MODEL_SIZES))      # 5
    assert logits["deliberation"].shape == (3, 1)
    assert output.shape == (3, INSTANCE_CONFIGS["retrieval_gate"].output_dim)  # 256

def test_route_returns_valid_decision():
    dec = RetrievalGate(JGSBackbone(BackboneConfig())).route(torch.randn(1, 384))
    assert all(d in AVAILABLE_DOMAINS for d in dec.domains)
    assert dec.pathway in PATHWAYS and dec.model_size in MODEL_SIZES
    assert 0.0 <= dec.confidence <= 1.0 and len(dec.domains) >= 1   # argmax fallback

# Param isolation: gate.parameters() EXCLUDES the backbone
def test_gate_parameters_exclude_backbone():
    bb = JGSBackbone(BackboneConfig()); gate = RetrievalGate(bb)
    assert sum(p.numel() for p in gate.parameters()) < sum(p.numel() for p in bb.parameters())
    assert all(p.requires_grad for p in bb.parameters())   # backbone not owned by gate

# Supervised step: 2 distinct examples overfit → loss decreases
def test_supervised_step_runs_and_decreases_loss():
    gate = RetrievalGate(JGSBackbone(BackboneConfig()))
    for p in gate.backbone.parameters(): p.requires_grad = False
    train = [_example(...), _example(...)] * 4
    cfg = RetrievalGateTrainingConfig(epochs=8, batch_size=4, device="cpu")
    losses = []; train_retrieval_gate_supervised(gate, ..., progress_cb=lambda e,t,v: losses.append(t))
    assert losses[-1] < losses[0]   # overfit 2 examples

# Outcome trainer: no-op below min_buffer; runs above it
def test_outcome_trainer_noop_below_min_buffer(): ...
def test_outcome_trainer_runs_above_min_buffer(): ...

# Integration (tmp_path WaveDB store + stub planner, untrained gate)
def test_retrieve_with_routing_contract(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Alice"], summary="WAL config"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({...}),
                                retrieval_gate=RetrievalGate(JGSBackbone(BackboneConfig())),
                                embedder=build_embedder("stub"))
    result = retr.retrieve_with_routing("What did Alice say about the WAL config?")
    assert set(result.keys()) == {"type", "route", "results", "context", "supported"}
    if result["supported"]: assert isinstance(result["results"], list)
    else: assert result["results"] == [] and result["context"] is None

def test_retrieve_unchanged_without_gate(tmp_path):   # retrieve() still list[dict]
def test_record_outcome_noop_without_gate(tmp_path):  # no gate → silent no-op

# Backbone strict-load (gated on the 2a checkpoint existing locally)
def test_backbone_frozen_after_load_backbone():
    if not Path(BACKBONE_PATH).exists(): pytest.skip("Phase 2a checkpoint absent")
    bb = load_backbone(BACKBONE_PATH, BackboneConfig(), device="cpu")
    assert all(not p.requires_grad for p in bb.parameters())
    assert sum(p.numel() for p in bb.parameters()) == 19_518_016
```

The original doc's semantic assertions ("database in domains" after training,
"routes complex to deliberation") assume a trained, semantically-meaningful gate.
Those are **not** unit tests — they're validation-smoke checks run after training
on the 5k pairs (see §6), and they're only meaningful once the gate has actually
seen real (bge-small) embeddings. The offline suite instead verifies the
**contract** (shapes, vocab, param isolation, back-compat, honest unsupported
flag) and a **supervised-step overfit** check, which is the strongest claim
possible without a trained checkpoint or a model download.

---

## 6. Verification

- **5k pairs**: `wc -l data/training/jepa/routing_pairs.jsonl` ≈ 5000;
  `python scripts/validate_training_data.py --data-dir data/training/` all ✅;
  pathway distribution non-degenerate across **all 5** pathways;
  **unique-query diversity** — count distinct query→route mappings (should be
  in the thousands after the vocab expansion, not ~177); per-pathway unique
  counts healthy (each pathway ≥ tens of unique examples); eye-check 1–2 raw
  records.
- **Tests**: `python -m pytest -q tests/test_retrieval_gate.py` green; full
  suite `python -m pytest -q` still green (existing suite + 13 new).
- **Training**:
  `python scripts/train_retrieval_gate.py --pairs data/training/jepa/routing_pairs.jsonl --embed-source on-demand --device auto --dtype float32`
  converges (train loss ↓, val accuracy > majority-class baseline); `best.pt`
  saved; load it back via `gate.load_state_dict` strict (0 missing/unexpected);
  **SCP checkpoint + train log to `data/pod_runs/phase2b/` local** before any
  pod stop (pod disk is ephemeral — see `runpod-community-pod-disk-wipe`).
- **Smoke**: load the trained gate, `route_text` the validate-slice queries —
  outputs are valid vocab + plausible.

---

## 7. Open Notes

- **bf16/autocast dtype-mix bug** in `pretrain.py` / `ReferenceSSM` is still
  unfixed (deferred 2a work); 2b gate training uses `--dtype float32` (gate
  params small, fp32 fine) — not in scope for 2b.
- `scripts/train_backbone.py` remains uncommitted (deferred 2a) — not touched.
- **Gate context features** (entity_recency, topic_recency, query_complexity)
  default to **zeros** in 2b — assembling them from the store/conversation is a
  Phase 2.5 concern; supervised training uses zeros (documented, not faked).
- **`process_exec` / `tool_plan` / `ssm_direct`** and the **model-size ladder**
  are routed + recorded but not executed in 2b (infra is later phases); the
  integration is honest about this via the `supported` flag.
- Oracle generation runs **locally** (Ollama at `localhost:11434`, `:cloud`
  routes to ollama.com on Ollama credits); gate training is device-auto (local
  CPU first; the kept-running 2a L4 pod is available if CPU is too slow).
- The first 5k run (4-class, `tool_plan=0`) is backed up at
  `data/training/jepa/routing_pairs_4class_backup.jsonl`; the second (tool-plan
  covered but only ~177 unique) at `routing_pairs_177unique_backup.jsonl`. The
  production data is the third run (expanded vocab, genuine diversity).
- **Training split must dedup.** Because the first two runs were
  duplication-heavy, the train/val split in `scripts/train_retrieval_gate.py`
  operates on the records as loaded; if a future run is again duplication-heavy,
  dedup by query before splitting to avoid train→val leakage. The expanded-vocab
  run is expected to be near-unique so this is a non-issue, but the dedup
  discipline is noted.