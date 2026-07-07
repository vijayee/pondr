# Phase 2a: JEPA-Gated SSM Backbone — Implementation Plan for Claude Code

## Overview

**Goal:** Train the shared JEPA-Gated SSM backbone that all cognitive instances will use. This is the foundational neural component of the ponder engine — a single set of SSM+JEPA weights that serves as the "laws of physics" for the cognitive universe, with instance-specific states, gates, and LoRA adapters providing specialization.

**What "done" looks like:** A trained JGSBackbone module (~480M params, ~0.90 GB) that can be loaded once in GPU memory and called by any number of JGSInstance objects. Each instance maintains its own state vector, decomposed gate, input/output projections, and LoRA adapters. The backbone has been pre-trained on diverse cognitive state sequences and validated on held-out prediction tasks.

**Prerequisite:** Phase 1d complete (Oracle routing pairs for the Phase 2b Retrieval Gate are available). Backbone pre-training data is **not** from Phase 1d — it is extracted from the surviving encoded corpora's `follows` turn-chains plus a 384-dim embedding backfill. See §0.2 for the corrected data path.

**Duration estimate:** 7-10 days (5-7 days for backbone training, 2-3 days for integration and validation).

---

## 0. Alignment Notes (2026-07-06) — read before any code below

The original draft of this doc (§§1–10 below) was written before Phase 1a–1d landed and
contains several claims that the implemented system has since disproven. **The prose and
code blocks in §§1–10 are kept as design intent; the corrections below are authoritative
where they conflict.** Every correction ties to a concrete lesson from Phase 1a–1d.

### 0.1 Mamba3 is real (shipped Mar 2026) — but the doc's *API* for it is fictional

`mamba_ssm.Mamba3` exists (`state-spaces/mamba`, v2.3.2+, `pip install mamba-ssm --no-build-isolation`,
build from source with `MAMBA_FORCE_BUILD=TRUE`). The doc's usage does **not** match the real
library and must be replaced:

- **Real signature:** `Mamba3(d_model, d_state, headdim, is_mimo, mimo_rank, chunk_size, dtype=...)`.
  The doc's `n_layers=24`, `expand=2`, `dt_rank="auto"` are Mamba1/Mamba2 kwargs, **not Mamba3's**.
  Layers are handled by stacking blocks ourselves, not via an `n_layers` arg.
- **No `.step(input, state)` recurrent API** and **no `.with_lora(A, B)`** — both are invented in
  §2.3. Real recurrent decode goes through `inference_params`; LoRA must be applied by us
  (PEFT-style on the in/out projections, or a hand-rolled low-rank delta on the SSM input/output).
- **CUDA-only.** Triton (prefill) + TileLang (MIMO) + CuTe DSL (decode `step()`) kernels. No CPU
  build, no AMD ROCm for Mamba3 (ROCm only covers Mamba1/2). The `step()` decode path is
  **"only tested on H100"** per `mamba_ssm/modules/mamba3.py`.
- **Implication for 2a:** pre-training uses the **bulk-sequence Triton prefill path** (process whole
  `follows`-chains as sequences), which runs on Ampere/Ada — **no H100 required for training**. The
  per-step `step()` path is only needed for *live instance inference* (Phase 2b+), so its H100-only
  status does not gate 2a.
- **Dev loop:** all module + extraction + unit-test code is written and verified **locally on CPU**
  using the pure-PyTorch reference ([`rishikksh20/mamba3-pytorch`](https://github.com/rishikksh20/mamba3-pytorch) —
  all 3 Mamba3 innovations, runs anywhere PyTorch runs) behind a **pluggable SSM interface**. The
  official `mamba_ssm.Mamba3` is swapped in only on the training pod. This makes the code real and
  CPU-testable without GPU spend.

### 0.2 The pre-training data prerequisite does NOT exist as the doc describes

The doc §3.1 says training data = "Oracle-generated cognitive state sequences (10M+ examples) from
Phase 1d." **Phase 1d produced no such thing.** It produced GNN labels, Bonsai query/relation pairs,
JEPA *routing* pairs (those are for the **Phase 2b Retrieval Gate instance**, not the shared
backbone), gate scalar labels, and code-aware examples — no temporal `(state_t, state_{t+1})`
sequences. The doc's §9 line "Training uses Oracle-generated sequences, not live WaveDB data" is
doubly wrong: those sequences were never generated, *and* the real surviving corpora are the
obvious substrate.

**Corrected data path (no Oracle, no spend):**
1. The surviving corpora have **`follows` chains** — *intra-session turn chains within a
   conversation* (each conversation = one session; `start_session` resets the chain; the encoder
   writes `(ep_N, follows, ep_{N-1})` so forward-in-time traversal is `.out("follows")` from a
   chain start). Verified on the surviving DialogSum DB: 5,002 episodes, ~80% of sampled episodes
   have a next turn.
2. **Embeddings are NOT persisted** in the surviving DB (0/272 sampled). Backfill them with the
   **local** `BAAI/bge-small-en-v1.5` sentence-transformer (`src/retrieval/vector_search.py`'s
   embedder) — **384-dim, not the 1536 a draft script assumed.** `scripts/build_vector_index.py`
   already persists `content/ep/{eid}/embedding` via `store.set_summary_embedding`; run it (or
   embed-on-demand in the extraction script). No OpenAI, no Oracle.
3. **Extract** `(emb_t, emb_{t+1})` pairs by walking each `follows` chain forward, forward + reverse
   pairs. Real API: `sample_episode_centers(store, n=None)` for ids, `.out("follows").execute_sync()`
   → `.vertices` (+ `result.close()` in finally) to walk, `store.get_episode(eid).summary_embedding`
   for the 384-dim vector. **~4,000–8,000 pairs** (chains are per-conversation turns, not cross-
   conversation), not 10M.

### 0.3 Right-size the backbone to the data — the doc's 480M/100k-step plan was for data that doesn't exist

The doc's "~480M params, 100k steps, A100 80GB, 48h, ~$62" was sized for "10M+ examples." With a few
thousand transition pairs, a 480M/24-layer model would massively overfit and 100k steps is absurd.
**2a trains a modest backbone (~30–60M params, d_model 256, a handful of layers) for a few thousand
steps**, which fits in <8GB VRAM and trains in minutes-to-an-hour. See §3.1 for the corrected config.

### 0.4 Hardware — modest GPU, not A100/H100

Single **secure RunPod A5000 24GB pod at $0.27/hr** (Ampere sm_80; Triton prefill supported). Run as
**one session**: spin up → build `mamba-ssm` from source → backfill embeddings → extract pairs →
train → save checkpoint → **SCP checkpoint + backfilled DB to local** → delete pod. No stop/start,
which sidesteps the RunPod disk-wipe-on-stop/start lesson (container disk is ephemeral on both
community and secure clouds — see project memory `runpod-community-pod-disk-wipe`). Total well under
$1. H100 is not needed for 2a.

### 0.5 De-wonk corrections to the doc's own code blocks

The §2/§3/§6 code blocks have concrete bugs beyond the fictional API. Fixed in implementation:
- **Gate param math is ~20× off.** `value_head`'s first Linear is `512·16·2 = 16384 → 512` ≈ 8.4M
  params, not the "~400K" claimed; the gate is far larger than "~1.5M". The §6 test
  `assert 2_000_000 < instance_params < 3_000_000` would fail against the doc's own design.
  Right-sized in implementation.
- **`JGSBackbone`/`JGSInstance` don't subclass `nn.Module`**, so `backbone.parameters()`,
  `torch.compile(backbone)`, and `instance.parameters()` (used in §3.1/§6) don't exist. Fixed.
- **`_threshold_modifier` builds a fresh `nn.Linear` inside `forward` every call** — unregistered,
  never in the optimizer, so the "learned threshold modulation" is never learned. Moved to a
  registered module.
- **`torch.cuda.amp.autocast(dtype=config.dtype)`** is deprecated and `config.dtype="bfloat16"` is a
  string, not a `torch.dtype`. Replaced with `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
- **No device handling anywhere** — `self.state = torch.zeros(...)` is always CPU. This is the exact
  Phase 1c GLiNER-on-CPU bug (model loaded with no device). All modules use `.to(device)` with
  auto-detected CUDA.

### 0.6 What 2a does NOT do (unchanged from §9, restated for clarity)

No specific gate is trained (gate training is 2b–7b). No cognitive function is deployed. The
per-step `step()` recurrent inference path is **not** exercised in 2a — only bulk-sequence
pre-training of the shared SSM+JEPA weights. Live instance inference arrives in 2b.

---

## 1. What Phase 2a Delivers

Artifact	Description	Consumer
**JGSBackbone**	Shared Mamba3 SSM + JEPA Predictor weights (~480M params, ~0.90 GB)	All JGS instances (Phases 2b-7b)
**DecomposedGate**	Reusable gate architecture with value/cost/decision heads (~1.5M params)	All JGS instances
**JGSInstance**	Base class for all cognitive functions with state, gate, projections, LoRA	All JGS instances
**LoRA adapter framework**	Instance-specific low-rank adaptation of SSM transition kernels	All JGS instances
**Pre-training validation**	Held-out prediction accuracy metrics	Quality measurement
**Backbone checkpoint**	Saved model weights for all downstream training	Phases 2b-7b

---

## 2. Architecture

### 2.1 The JEPA-Gated SSM Primitive

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                   JEPA-GATED SSM PRIMITIVE                        │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐               │
│  │   Mamba3 SSM        │  │   JEPA Predictor    │               │
│  │   (~370M params)    │  │   (~110M params)    │               │
│  │                     │  │                     │               │
│  │   Recurrent state   │  │   Predictive coding │               │
│  │   maintenance.      │  │   in embedding      │               │
│  │   State evolves     │  │   space. Predicts   │               │
│  │   with each input.  │  │   future states.    │               │
│  │                     │  │                     │               │
│  │   Weights: SHARED   │  │   Weights: SHARED   │               │
│  └──────────┬──────────┘  └──────────┬──────────┘               │
│             │                        │                           │
│             └────────────┬───────────┘                           │
│                          │                                       │
│                 ┌────────▼────────┐                              │
│                 │  DECOMPOSED     │                              │
│                 │  GATE           │                              │
│                 │  (~1.5M params) │                              │
│                 │                 │                              │
│                 │  Value head:    │                              │
│                 │  "How good?"   │                              │
│                 │                 │                              │
│                 │  Cost head:     │                              │
│                 │  "How hard?"    │                              │
│                 │                 │                              │
│                 │  Decision head: │                              │
│                 │  "Pursue or     │                              │
│                 │   inhibit?"     │                              │
│                 │                 │                              │
│                 │  Weights:       │                              │
│                 │  INSTANCE-      │                              │
│                 │  SPECIFIC       │                              │
│                 └─────────────────┘                              │
│                                                                  │
│  Input projection:  INSTANCE-SPECIFIC (~100K params)              │
│  Output projection: INSTANCE-SPECIFIC (~100K params)              │
│  LoRA adapters:     INSTANCE-SPECIFIC (~50K params)              │
│                                                                  │
│  Total shared:      ~480M params (~0.90 GB)                      │
│  Total per instance: ~2.5M params (~10 MB)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Why Mamba3

Mamba3 is the latest generation of state space models with several advantages for this architecture:

Feature	Benefit for JGS
**Selective state spaces**	Better at filtering relevant vs. irrelevant information in the continuous state
**Improved training efficiency**	Faster convergence on the diverse pre-training tasks
**Extended context handling**	Better compression of long sequences into the fixed-dimension state
**Stable recurrent dynamics**	More predictable state evolution across different instance configurations
**Native LoRA support**	Built-in low-rank adaptation of transition kernels without modifying base weights

The architecture is not tightly coupled to Mamba3 — any SSM with a continuous hidden state and LoRA support would work. Mamba3 is chosen for its maturity, library support, and performance characteristics at the time of implementation.

### 2.3 Instance Architecture

```python
class JGSBackbone:
    """
    Shared SSM and JEPA weights. Loaded once in GPU memory.
    All instances call this. Stateless — all state is in the instances.
    """
    
    def __init__(self, config: BackboneConfig):
        # Mamba3 SSM: recurrent state maintenance
        self.ssm = Mamba3(
            d_model=config.d_model,        # 512
            n_layers=config.n_layers,      # 24
            d_state=config.d_state,        # 16
            expand=config.expand,          # 2
            dt_rank=config.dt_rank,        # "auto"
        )  # ~370M params, ~0.70 GB
        
        # JEPA Predictor: predictive coding in embedding space
        self.predictor = JEPAPredictor(
            d_model=config.d_model,        # 512
            n_layers=config.pred_layers,   # 12
            pred_dim=config.pred_dim,      # 256
        )  # ~110M params, ~0.20 GB
        
        # Total shared: ~0.90 GB
    
    def forward(self, input_embedding, state, instance):
        """
        Called by every instance.
        
        Args:
            input_embedding: instance-specific input [batch, input_dim]
            state: instance-specific state [batch, d_state, d_model]
            instance: the JGSInstance making the call
        
        Returns:
            new_state, predicted_future, output
        """
        # Project instance-specific input into shared space
        projected_input = instance.input_proj(input_embedding)
        
        # Apply instance-specific LoRA adapter to SSM dynamics
        adapted_ssm = self.ssm.with_lora(
            instance.ssm_lora_A, 
            instance.ssm_lora_B
        )
        
        # Shared SSM dynamics with instance-specific modulation
        new_state = adapted_ssm.step(projected_input, state)
        
        # Shared JEPA prediction
        predicted_future = self.predictor(new_state, action_embedding=None)
        
        # Project back to instance-specific space
        output = instance.output_proj(new_state)
        
        return new_state, predicted_future, output


class JGSInstance:
    """
    One cognitive function. Has its OWN state, OWN gate, OWN projections,
    OWN LoRA adapters. Uses SHARED backbone.
    
    Total instance cost: ~2.5M params (~10 MB)
    """
    
    def __init__(self, backbone: JGSBackbone, config: InstanceConfig):
        self.backbone = backbone
        self.config = config
        
        # Instance-OWNED state: [d_state, d_model] = [16, 512]
        # In fp16: ~16 KB
        self.state = torch.zeros(
            config.batch_size or 1, 
            config.d_state, 
            config.d_model
        )
        
        # Instance-OWNED decomposed gate (~1.5M params, ~6 MB)
        self.gate = DecomposedGate(config.gate_config)
        
        # Instance-OWNED non-linear projections (~200K params, ~0.8 MB)
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )
        
        # Instance-OWNED LoRA adapters (~50K params, ~0.2 MB)
        self.ssm_lora_A = nn.Parameter(
            torch.randn(config.d_state, config.lora_rank) * 0.01
        )
        self.ssm_lora_B = nn.Parameter(
            torch.zeros(config.lora_rank, config.d_state)
        )
    
    def step(self, input_embedding, action_embedding=None):
        """One forward step. Returns output, prediction, and gate decision."""
        new_state, predicted_future, output = self.backbone.forward(
            input_embedding, self.state, self
        )
        
        # Gate decides: excite or inhibit?
        decision = self.gate(
            self.state, 
            predicted_future, 
            self.get_context()
        )
        
        # Update state
        self.state = new_state
        
        return output, predicted_future, decision
```

### 2.4 Decomposed Gate Architecture

```python
class DecomposedGate(nn.Module):
    """
    Three sub-modules: value head, cost head, decision head.
    ~1.5M params total. Instance-specific.
    """
    
    def __init__(self, config: GateConfig):
        super().__init__()
        
        # Value head: estimates reward of predicted future state
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model * config.d_state * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )  # ~400K params
        
        # Cost head: estimates effort to reach predicted future state
        self.cost_head = nn.Sequential(
            nn.Linear(config.d_model * config.d_state * 2 + config.context_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )  # ~500K params
        
        # Decision head: combines value, cost, and context
        self.decision_head = nn.Sequential(
            nn.Linear(2 + config.context_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # [inhibit_score, excite_score]
        )  # ~500K params
        
        # Learnable base threshold
        self.base_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.num_context_features, 128),
            nn.GELU(),
            nn.Linear(128, config.context_dim)
        )  # ~100K params
    
    def forward(self, state, predicted_future, context):
        """Returns GateDecision with value, cost, ratio, pursue/inhibit, confidence."""
        context_vec = self.context_encoder(context.to_vector())
        
        value = self.value_head(
            torch.cat([state.flatten(), predicted_future.flatten()])
        )
        cost = self.cost_head(
            torch.cat([state.flatten(), predicted_future.flatten(), context_vec])
        )
        
        ratio = value / (cost + 1e-8)
        
        decision_input = torch.cat([ratio, cost, context_vec])
        logits = self.decision_head(decision_input)
        
        inhibit_score = torch.sigmoid(logits[0])
        excite_score = torch.sigmoid(logits[1])
        
        threshold = self.base_threshold * self._threshold_modifier(context)
        
        pursue = (excite_score > inhibit_score) & (excite_score > threshold)
        
        return GateDecision(
            value_estimate=value.item(),
            cost_estimate=cost.item(),
            ratio=ratio.item(),
            inhibit_score=inhibit_score.item(),
            excite_score=excite_score.item(),
            threshold=threshold.item(),
            pursue=pursue.item(),
            confidence=abs(excite_score.item() - inhibit_score.item()),
        )
    
    def _threshold_modifier(self, context):
        """Learned threshold modulation based on context. Range [0.3, 1.7]."""
        context_vec = self.context_encoder(context.to_vector())
        modifier = torch.sigmoid(
            nn.Linear(context_vec.shape[-1], 1).to(context_vec.device)(context_vec)
        )
        return 0.3 + 1.4 * modifier
```

---

## 3. Training Procedure

### 3.1 Phase 1: Pre-Train Shared Backbone (Self-Supervised)

The shared SSM+JEPA weights are pre-trained on diverse prediction tasks to establish general-purpose temporal dynamics. This is the largest training investment — done once, used by all instances.

**Training data:** `follows`-chain state-transition pairs extracted from the surviving encoded
corpora (DialogSum 5,002 eps + Samsum 2,384 eps), **not** Oracle output. Each pair is
`(emb_t, emb_{t+1})` where the embeddings are the 384-dim `BAAI/bge-small-en-v1.5` summary
embeddings backfilled into the store. Chains are *intra-conversation turn sequences*
(`(ep_N, follows, ep_{N-1})`; walk forward with `.out("follows")`). Forward + reverse pairs →
**~4,000–8,000 pairs total.** See §0.2 and `scripts/extract_backbone_sequences.py`.

**Self-supervised tasks:**

Task	Description	Loss
**Next-state prediction**	Given state_t, predict state_{t+1}	JEPA contrastive: minimize distance to positive target, maximize distance to negative samples
**Outcome prediction**	Given state_t and action embedding, predict outcome state	Trained on recorded (state, action, outcome) triples
**Interpretation prediction**	Given state_t and interpretation embedding, predict resulting state	Trained on recorded (state, interpretation, outcome) triples
**Masked state reconstruction**	Randomly mask dimensions of state_t, predict full state_{t+1}	MSE on masked dimensions

**JEPA contrastive loss:**

```python
def jepa_contrastive_loss(predicted, actual, negative_samples, temperature=0.1):
    """
    JEPA contrastive loss.
    
    Minimize distance between predicted and actual future state.
    Maximize distance between predicted and negative samples.
    """
    # Positive pair: predicted should be close to actual
    pos_dist = F.cosine_similarity(predicted, actual, dim=-1)
    pos_loss = -pos_dist.mean()
    
    # Negative pairs: predicted should be far from negatives
    neg_dist = F.cosine_similarity(
        predicted.unsqueeze(1),      # [batch, 1, dim]
        negative_samples.unsqueeze(0), # [1, num_neg, dim]
        dim=-1
    )
    neg_loss = torch.logsumexp(neg_dist / temperature, dim=-1).mean()
    
    return pos_loss + neg_loss
```

**Training configuration:**

```python
@dataclass
class BackboneTrainingConfig:
    # Model
    d_model: int = 512
    n_layers: int = 24
    d_state: int = 16
    expand: int = 2
    dt_rank: str = "auto"
    pred_layers: int = 12
    pred_dim: int = 256
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 100_000
    gradient_accumulation: int = 4
    
    # JEPA-specific
    temperature: float = 0.1
    num_negative_samples: int = 16
    target_ema_decay: float = 0.996  # EMA decay for target encoder
    
    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Hardware
    dtype: str = "bfloat16"
    compile: bool = True  # torch.compile for Mamba3
```

**Training loop:**

```python
def train_backbone(config: BackboneTrainingConfig, data_loader, val_loader):
    """
    Pre-train the shared JGS backbone.
    """
    backbone = JGSBackbone(config)
    target_backbone = JGSBackbone(config)  # EMA target for JEPA
    target_backbone.load_state_dict(backbone.state_dict())
    
    optimizer = torch.optim.AdamW(
        backbone.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    
    # Learning rate schedule
    scheduler = cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.total_steps
    )
    
    backbone = backbone.to(dtype=config.dtype)
    if config.compile:
        backbone = torch.compile(backbone)
    
    for step in range(config.total_steps):
        batch = next(data_loader)
        
        # Accumulate gradients
        total_loss = 0
        for micro_step in range(config.gradient_accumulation):
            with torch.cuda.amp.autocast(dtype=config.dtype):
                loss = compute_jepa_loss(
                    backbone, target_backbone, batch, config
                )
            (loss / config.gradient_accumulation).backward()
            total_loss += loss.item()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update target encoder (EMA)
        update_ema(target_backbone, backbone, config.target_ema_decay)
        
        # Logging
        if step % 100 == 0:
            val_loss = validate(backbone, target_backbone, val_loader, config)
            print(f"Step {step}: train_loss={total_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Checkpoint
        if step % 5000 == 0:
            save_checkpoint(backbone, optimizer, step, f"checkpoint_{step}.pt")
    
    # Final save
    save_checkpoint(backbone, optimizer, config.total_steps, "backbone_final.pt")
    return backbone
```

**Training hardware:** single secure RunPod **A5000 24GB** pod (Ampere sm_80, $0.27/hr), bulk-
sequence Triton prefill. One session: build `mamba-ssm` from source → backfill embeddings → extract
pairs → train → save checkpoint → SCP checkpoint + backfilled DB to local → delete pod. Well under
$1. See §0.4. (The doc's original "A100 80GB / 48h / $62" was sized for the 10M-example dataset
that doesn't exist; the right-sized backbone on a few thousand pairs needs <8GB and trains in
minutes-to-an-hour.)

### 3.2 Phase 2: Initialize Gates (Rule-Based Priors)

Each instance's gate is initialized with rule-based thresholds derived from cognitive science literature. These are not the final thresholds — they are good priors that give the gates reasonable starting behavior before gradient-based fine-tuning.

```python
def initialize_gate_priors(gate: DecomposedGate, instance_type: str):
    """
    Initialize gate thresholds with rule-based priors.
    
    These are starting points, not final values.
    Fine-tuned by gradient descent in Phase 3.
    """
    priors = {
        "retrieval_gate": {
            "base_threshold": 0.5,
            "context_features": ["entity_recency", "topic_recency", "query_complexity"],
        },
        "uncertainty_detector": {
            "base_threshold": 0.7,
            "context_features": ["error_magnitude", "noise_level", "novelty"],
        },
        "aspirational_model": {
            "base_threshold": 0.5,
            "context_features": ["goal_alignment", "expected_value", "urgency"],
        },
        "self_model": {
            "base_threshold": 0.6,
            "context_features": ["domain_density", "fact_specificity", "retrieval_confidence"],
        },
        "common_sense_resolver": {
            "base_threshold": 0.6,
            "context_features": ["ambiguity_magnitude", "context_coherence", "historical_frequency"],
        },
        "disturbance_detector": {
            "base_threshold": 0.7,
            "context_features": ["error_magnitude", "noise_level", "novelty"],
        },
        "intuition_module": {
            "base_threshold": 0.5,
            "context_features": ["sunk_cost", "novelty", "recent_reward_rate", "pattern_familiarity"],
        },
    }
    
    prior = priors.get(instance_type, priors["retrieval_gate"])
    gate.base_threshold = nn.Parameter(torch.tensor(prior["base_threshold"]))
```

### 3.3 Phase 3: Joint Fine-Tuning (Delayed Reward + Replay Buffer)

Gates and projections are fine-tuned using delayed reward signals. This happens in later phases (2b-7b) for each specific instance, but the framework is established here.

```python
class ReplayBuffer:
    """
    Stores gate decisions and outcomes for offline training.
    
    Key insight: gate decisions at time t are validated by outcomes at time t+k.
    The replay buffer bridges this temporal gap without requiring BPTT.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, entry: ReplayEntry):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(entry)
    
    def sample(self, batch_size: int = 32) -> list[ReplayEntry]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


def train_gate_offline(gate, replay_buffer, optimizer):
    """
    One training step using the replay buffer.
    
    Gradients flow from outcome back through the gate,
    but not through the SSM state (avoids BPTT across many steps).
    """
    batch = replay_buffer.sample(batch_size=32)
    
    total_loss = 0
    
    for entry in batch:
        decision = gate(entry.state, entry.predicted_outcome, entry.context)
        
        actual_reward = entry.outcome.reward
        actual_cost = entry.outcome.effort
        optimal_decision = actual_reward > actual_cost
        
        # Value loss: did we predict the reward correctly?
        value_loss = F.mse_loss(
            decision.value_estimate, 
            torch.tensor(actual_reward)
        )
        
        # Cost loss: did we predict the effort correctly?
        cost_loss = F.mse_loss(
            decision.cost_estimate, 
            torch.tensor(actual_cost)
        )
        
        # Decision loss: did we make the right pursue/inhibit choice?
        decision_loss = F.binary_cross_entropy(
            decision.excite_score,
            torch.tensor(float(optimal_decision))
        )
        
        total_loss += value_loss + cost_loss + decision_loss
    
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss.item()
```

---

## 4. Instance Configuration Templates

Each cognitive function gets a specific configuration. These are defined here and used in Phases 2b-7b.

```python
INSTANCE_CONFIGS = {
    "retrieval_gate": InstanceConfig(
        name="retrieval_gate",
        input_dim=512,
        output_dim=256,
        d_model=512,
        d_state=16,
        lora_rank=4,  # Fast routing
        gate_config=GateConfig(
            num_context_features=3,  # entity_recency, topic_recency, query_complexity
            context_dim=128,
        ),
    ),
    "working_memory": InstanceConfig(
        name="working_memory",
        input_dim=512,
        output_dim=512,
        d_model=512,
        d_state=16,
        lora_rank=8,  # Rich state
        gate_config=GateConfig(
            num_context_features=2,  # input_novelty, state_saturation
            context_dim=128,
        ),
    ),
    "uncertainty_detector": InstanceConfig(
        name="uncertainty_detector",
        input_dim=512,
        output_dim=256,
        d_model=512,
        d_state=16,
        lora_rank=4,
        gate_config=GateConfig(
            num_context_features=3,  # error_magnitude, noise_level, novelty
            context_dim=128,
        ),
    ),
    "aspirational_model": InstanceConfig(
        name="aspirational_model",
        input_dim=512,
        output_dim=256,
        d_model=512,
        d_state=16,
        lora_rank=6,
        gate_config=GateConfig(
            num_context_features=3,  # goal_alignment, expected_value, urgency
            context_dim=128,
        ),
    ),
    "self_model": InstanceConfig(
        name="self_model",
        input_dim=512,
        output_dim=256,
        d_model=512,
        d_state=16,
        lora_rank=4,
        gate_config=GateConfig(
            num_context_features=3,  # domain_density, fact_specificity, retrieval_confidence
            context_dim=128,
        ),
    ),
    "common_sense_resolver": InstanceConfig(
        name="common_sense_resolver",
        input_dim=512,
        output_dim=512,
        d_model=512,
        d_state=16,
        lora_rank=6,  # Flexible dynamics
        gate_config=GateConfig(
            num_context_features=3,  # ambiguity_magnitude, context_coherence, historical_frequency
            context_dim=128,
        ),
    ),
    "disturbance_detector": InstanceConfig(
        name="disturbance_detector",
        input_dim=512,
        output_dim=512,
        d_model=512,
        d_state=16,
        lora_rank=4,  # Fast, bursty
        gate_config=GateConfig(
            num_context_features=3,  # error_magnitude, noise_level, novelty
            context_dim=128,
        ),
    ),
    "intuition_module": InstanceConfig(
        name="intuition_module",
        input_dim=512,
        output_dim=256,
        d_model=512,
        d_state=16,
        lora_rank=8,  # Slow, accumulative
        gate_config=GateConfig(
            num_context_features=4,  # sunk_cost, novelty, recent_reward_rate, pattern_familiarity
            context_dim=128,
        ),
    ),
}
```

---

## 5. Project Structure (Additions)

```plaintext
hippocampal-memory/
├── src/
│   └── subconscious/                  # NEW
│       ├── __init__.py
│       ├── backbone.py                # JGSBackbone (Mamba3 + JEPA)
│       ├── instance.py               # JGSInstance base class
│       ├── gate.py                    # DecomposedGate
│       ├── lora.py                    # LoRA adapter utilities
│       ├── configs.py                 # Instance configuration templates
│       └── training/
│           ├── __init__.py
│           ├── pretrain.py            # Backbone pre-training loop
│           ├── jepa_loss.py           # JEPA contrastive loss
│           ├── replay_buffer.py       # Delayed reward replay buffer
│           └── gate_training.py       # Gate fine-tuning utilities
├── tests/
│   ├── test_backbone.py               # NEW
│   ├── test_instance.py               # NEW
│   ├── test_gate.py                   # NEW
│   └── test_lora.py                   # NEW
├── scripts/
│   ├── train_backbone.py              # NEW
│   └── validate_backbone.py           # NEW
└── checkpoints/
    └── backbone/                      # NEW — saved model weights
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**`tests/test_backbone.py`:**

```python
def test_backbone_forward():
    """Backbone produces valid output shapes."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    
    state = torch.zeros(1, config.d_state, config.d_model)
    input_emb = torch.randn(1, 512)
    
    # Create a minimal instance for testing
    instance = JGSInstance(backbone, INSTANCE_CONFIGS["retrieval_gate"])
    
    new_state, predicted, output = backbone.forward(input_emb, state, instance)
    
    assert new_state.shape == state.shape
    assert predicted.shape == (1, config.pred_dim)
    assert output.shape == (1, 256)

def test_backbone_stateful():
    """Backbone maintains state across multiple steps."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    instance = JGSInstance(backbone, INSTANCE_CONFIGS["working_memory"])
    
    state = torch.zeros(1, config.d_state, config.d_model)
    
    # Process a sequence
    states = []
    for _ in range(10):
        input_emb = torch.randn(1, 512)
        state, _, _ = backbone.forward(input_emb, state, instance)
        states.append(state.clone())
    
    # State should change over time
    assert not torch.allclose(states[0], states[-1])
```

**`tests/test_instance.py`:**

```python
def test_instance_independence():
    """Two instances with the same backbone have independent states."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    
    inst1 = JGSInstance(backbone, INSTANCE_CONFIGS["retrieval_gate"])
    inst2 = JGSInstance(backbone, INSTANCE_CONFIGS["uncertainty_detector"])
    
    # Process different inputs through each
    _, _, _ = inst1.step(torch.randn(1, 512))
    _, _, _ = inst2.step(torch.randn(1, 512))
    
    # States should be different
    assert not torch.allclose(inst1.state, inst2.state)

def test_instance_parameter_count():
    """Instance adds ~2.5M parameters."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    
    backbone_params = sum(p.numel() for p in backbone.parameters())
    
    instance = JGSInstance(backbone, INSTANCE_CONFIGS["retrieval_gate"])
    instance_params = sum(
        p.numel() for p in instance.parameters() 
        if p is not None
    )
    
    # Instance should add roughly 2.5M params
    assert 2_000_000 < instance_params < 3_000_000
```

**`tests/test_gate.py`:**

```python
def test_gate_decision():
    """Gate produces valid decision with all fields populated."""
    config = GateConfig(num_context_features=3, context_dim=128)
    gate = DecomposedGate(config)
    
    state = torch.randn(1, 16, 512)
    predicted = torch.randn(1, 256)
    context = GateContext(
        error_magnitude=0.8,
        noise_level=0.1,
        novelty=0.5,
    )
    
    decision = gate(state, predicted, context)
    
    assert 0 <= decision.value_estimate
    assert 0 <= decision.cost_estimate
    assert 0 <= decision.confidence <= 1
    assert isinstance(decision.pursue, bool)

def test_gate_threshold_adaptation():
    """Gate threshold adapts to context."""
    config = GateConfig(num_context_features=3, context_dim=128)
    gate = DecomposedGate(config)
    
    state = torch.randn(1, 16, 512)
    predicted = torch.randn(1, 256)
    
    # High noise → higher threshold
    high_noise = GateContext(error_magnitude=0.3, noise_level=0.9, novelty=0.1)
    low_noise = GateContext(error_magnitude=0.3, noise_level=0.1, novelty=0.1)
    
    decision_high = gate(state, predicted, high_noise)
    decision_low = gate(state, predicted, low_noise)
    
    # High noise should produce higher threshold
    assert decision_high.threshold > decision_low.threshold
```

**`tests/test_lora.py`:**

```python
def test_lora_parameter_efficiency():
    """LoRA adapters add minimal parameters."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    
    full_params = sum(p.numel() for p in backbone.ssm.parameters())
    
    # LoRA A and B matrices
    lora_A = nn.Parameter(torch.randn(16, 4) * 0.01)
    lora_B = nn.Parameter(torch.zeros(4, 16))
    
    lora_params = lora_A.numel() + lora_B.numel()
    
    # LoRA should be <0.1% of full parameters
    assert lora_params / full_params < 0.001

def test_lora_modulates_without_changing_base():
    """LoRA adapters modulate dynamics without changing base weights."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    
    # Snapshot base weights
    base_weights = {
        name: param.clone() 
        for name, param in backbone.ssm.named_parameters()
    }
    
    # Apply LoRA
    lora_A = nn.Parameter(torch.randn(16, 4) * 0.01)
    lora_B = nn.Parameter(torch.zeros(4, 16))
    adapted = backbone.ssm.with_lora(lora_A, lora_B)
    
    # Base weights should be unchanged
    for name, param in backbone.ssm.named_parameters():
        assert torch.allclose(param, base_weights[name])
```

### 6.2 Integration Tests

```python
def test_full_pipeline():
    """End-to-end: backbone + instance + gate + step."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    instance = JGSInstance(backbone, INSTANCE_CONFIGS["retrieval_gate"])
    
    # Simulate a sequence of inputs
    outputs = []
    decisions = []
    
    for _ in range(20):
        input_emb = torch.randn(1, 512)
        output, predicted, decision = instance.step(input_emb)
        outputs.append(output)
        decisions.append(decision)
    
    # Outputs should vary
    assert not torch.allclose(outputs[0], outputs[-1])
    
    # Decisions should be valid
    assert all(isinstance(d.pursue, bool) for d in decisions)

def test_multiple_instances_shared_backbone():
    """Multiple instances share backbone without interference."""
    config = BackboneTrainingConfig()
    backbone = JGSBackbone(config)
    
    inst1 = JGSInstance(backbone, INSTANCE_CONFIGS["retrieval_gate"])
    inst2 = JGSInstance(backbone, INSTANCE_CONFIGS["uncertainty_detector"])
    inst3 = JGSInstance(backbone, INSTANCE_CONFIGS["self_model"])
    
    # Process different inputs
    _, _, d1 = inst1.step(torch.randn(1, 512))
    _, _, d2 = inst2.step(torch.randn(1, 512))
    _, _, d3 = inst3.step(torch.randn(1, 512))
    
    # Each instance should have independent state
    assert not torch.allclose(inst1.state, inst2.state)
    assert not torch.allclose(inst2.state, inst3.state)
    assert not torch.allclose(inst1.state, inst3.state)
```

---

## 7. Checkpoint Criteria

Phase 2a is complete when:

- [ ] `JGSBackbone` class implemented with Mamba3 SSM + JEPA Predictor
- [ ] `DecomposedGate` class implemented with value/cost/decision heads
- [ ] `JGSInstance` base class implemented with state, gate, projections, LoRA
- [ ] LoRA adapter framework working (modulates SSM without changing base weights)
- [ ] Backbone pre-trained on `follows`-chain state-transition pairs from the surviving corpora (384-dim bge-small embeddings), forward + reverse
- [ ] JEPA contrastive loss implemented and validated
- [ ] Target encoder EMA update working correctly
- [ ] Validation loss decreasing and stable on held-out data
- [ ] Instance configuration templates defined for all 8 cognitive functions
- [ ] Gate initialization with rule-based priors working
- [ ] Replay buffer infrastructure in place for delayed reward training
- [ ] Backbone checkpoint saved and loadable
- [ ] All unit tests pass
- [ ] Integration test: multiple instances share backbone without interference
- [ ] VRAM usage verified: backbone ~0.90 GB, per instance ~10 MB
- [ ] Training cost within budget (single A5000 24GB session, well under $1; no H100 required)
- [ ] CPU dev loop verified: all unit + integration tests pass locally with the pure-PyTorch Mamba3 reference behind the pluggable SSM interface
- [ ] Pod checkpoint + backfilled DB SCP'd to local before pod teardown (disk-wipe safeguard)

---

## 8. Implementation Order

1. **Mamba3 setup** — Dev: pin the pure-PyTorch Mamba3 reference (`rishikksh20/mamba3-pytorch`) behind a pluggable SSM interface for CPU unit tests. Pod: `MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation git+https://github.com/state-spaces/mamba.git` on the A5000, verify the real `mamba_ssm.Mamba3` forward pass on CUDA. Note: `step()` is H100-tested — 2a uses the bulk-sequence prefill path, not `step()`.
2. **JGSBackbone** — Implement shared SSM+JEPA with forward method
3. **DecomposedGate** — Implement value/cost/decision heads with context encoder
4. **JGSInstance** — Implement base class with state, gate, projections, LoRA
5. **LoRA adapters** — Implement low-rank adaptation of SSM transition kernels
6. **JEPA loss** — Implement contrastive loss with negative sampling
7. **Pre-training loop** — Implement training loop with EMA target, gradient accumulation
8. **Instance configs** — Define configuration templates for all 8 cognitive functions
9. **Gate priors** — Implement rule-based initialization for each gate type
10. **Replay buffer** — Implement replay buffer for delayed reward training
11. **Run pre-training** — Execute on A100, monitor loss, save checkpoints
12. **Validation** — Evaluate on held-out prediction tasks
13. **Tests** — Unit tests for backbone, instance, gate, LoRA; integration tests
14. **VRAM profiling** — Verify memory budget fits in allocation

---

## 9. What Phase 2a Does NOT Do

- **Does not train any specific gate.** Gate training happens in Phases 2b-7b for each instance.
- **Does not deploy any cognitive function.** Instances are created in later phases.
- **Does not integrate with the retrieval pipeline.** That's Phase 2b (Retrieval Gate).
- **Does not require the full memory graph** for *training*, but DOES read the surviving WaveDB
  corpora to extract `follows`-chain state-transition pairs + backfill embeddings (§0.2). No Oracle
  calls. The per-step `step()` recurrent inference path is not exercised in 2a (bulk-sequence
  pre-training only); live instance inference arrives in 2b.

---

## 10. Next Phase

After Phase 2a checkpoint is met, proceed to **Phase 2b: Retrieval Gate Instance** which trains the first JGS instance — the subconscious router that predicts domain, pathway, meta-skills, model size, and deliberation need before any retrieval or generation occurs.

---

Begin with step 1. Report after each step. If Mamba3 installation fails or CUDA compatibility issues arise, report the exact error. If VRAM usage exceeds budget during backbone training, we'll adjust batch size or gradient accumulation before continuing.