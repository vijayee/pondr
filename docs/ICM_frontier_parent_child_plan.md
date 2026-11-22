# Frontier Parent / Small Child — Implementation Plan

Status: downstream proposal, not started.

Related documents:

- `docs/ICM_frontier_parent_child_architecture.md`
- `docs/ICM_predicate_intent_model.md`
- `docs/ICM_predicate_intent_model_plan.md`
- `docs/ICM_intent_store_spec.md`
- `docs/ICM_final_report.md`
- `ops/WARP_DYNAMICS_EXPERIMENTS.md`
- `ops/SPATIAL_THOUGHT_MAPPING.md`

---

## 1. Goal

Build a downstream system where:

```text
frontier parent
    ↓
knowledge substrate
procedural reasoning substrate
intent/geometry substrate
geometry operator
    ↓
small looped Discernment child
    ↓
accessible runtime AI
```

The parent is used once to compile durable artifacts.  
The child is used many times to execute them.

The target is not general frontier equivalence. The target is:

```text
parent-level performance
on domains covered by the substrate
at child-level cost
```

---

## 2. Non-goals

- Do not try to replicate the parent’s full parameter count.
- Do not treat memory as a generic vector database replacement.
- Do not assume raw geometry transfer works across model sizes.
- Do not use score blending between intent, geometry, and dense retrieval.
- Do not make reasoning geometry a hidden latent.
- Do not promise frontier-level open-world judgment.

---

## 3. Artifacts the parent must produce

| Artifact | Contents | Consumer |
|---|---|---|
| Knowledge substrate | Documents, code, APIs, domain facts | Memory index |
| Procedural reasoning substrate | Structured reasoning traces | Child distillation |
| Intent/geometry substrate | Predicate frames, orientation, qualifiers, warp trajectories | Memory keys |
| Geometry operator | Warp-generating operator or initialization prior | Child training |
| Evaluation harness | Benchmarks, controls, audit metrics | All stages |

---

## 4. Workstreams

### Workstream 1 — Define the substrate schemas

Create schemas for:

```text
knowledge record
reasoning trace
intent frame
geometry record
memory record
outcome record
```

Deliverable:

```text
docs/ICM_parent_child_substrate_schemas.md
```

### Workstream 2 — Generate structured reasoning traces

Use the parent to emit:

```text
intent
plan
retrieval query
evidence
decision
verification
outcome
```

Example:

```text
intent:
  diagnose(build_failure)

steps:
  read log
  reproduce
  isolate module
  inspect diff
  patch
  test
  verify
```

Deliverable:

```text
icm/parent_reasoning_trace.py
data/_scratch/icm/parent_reasoning_traces.jsonl
```

### Workstream 3 — Build the knowledge and procedure memory

Store two kinds of memory:

```text
factual memory:
    code, docs, APIs, domain facts

procedural memory:
    reasoning traces, plans, debugging procedures
```

Each record carries:

```text
content key
reasoning key
intent frame
geometry key
audit record
```

Deliverable:

```text
icm/parent_child_memory.py
```

### Workstream 4 — Extract and inherit geometry

Measure parent geometry on a fixed probe corpus:

```text
warp profile
delta-warp trajectory
memory-write subspaces
principal subspaces
```

Then align the child to the parent:

```text
same dimension      -> Procrustes alignment
smaller child       -> PCA / CCA / learned adapter
```

Deliverable:

```text
icm/geometry_inheritance.py
```

### Workstream 5 — Train the small looped Discernment child

Train the child with:

```text
task loss
+ reasoning-trace distillation
+ geometry-distillation loss
+ memory-retrieval training
+ loop-schedule training
```

Architecture:

```text
small Discernment core
+ adaptive loops
+ memory interface
+ tool interface
```

Deliverables:

```text
stage5_encoder/parent_child_distillation.py
stage5_encoder/train_child_with_substrate.py
```

### Workstream 6 — Add memory retrieval training

The child must learn:

```text
when to retrieve
what intent to emit
which geometry key to use
how to combine evidence
when to stop
```

Deliverable:

```text
stage5_encoder/memory_retrieval_head.py
```

### Workstream 7 — Add outcome qualification

After execution, compare:

```text
intent frame
outcome frame
```

and record:

```text
useful
violated
irrelevant
expanded_intent
```

Deliverable:

```text
icm/parent_child_outcome_qualifier.py
```

### Workstream 8 — Evaluate parent vs child

Compare arms:

| Arm | System |
|---|---|
| 1 | frontier parent alone |
| 2 | small child alone |
| 3 | small child + factual memory |
| 4 | small child + factual memory + procedural memory |
| 5 | small child + full substrate + inherited geometry |
| 6 | small child + full substrate + loops + geometry |

Metrics:

```text
code correctness
debugging accuracy
reasoning benchmark score
retrieval precision
candidate recall@10
analogical top1
latency
cost per query
memory size
audit-record size
```

Deliverable:

```text
scripts/icm/parent_child_eval.py
```

### Workstream 9 — Test geometry inheritance controls

Arms:

```text
random init
parent-geometry init
synthetic anti-aligned geometry
parent geometry, frozen
parent geometry, unfrozen
```

Measure:

```text
steps to selective memory
steps to parent-compatible warp trajectory
final reasoning quality
inherited blind spots
```

Deliverable:

```text
scripts/icm/geometry_inheritance_eval.py
```

### Workstream 10 — Security, privacy, and accessibility

Implement:

```text
local memory
access control
audit records
offline child inference
substrate export/import
```

Deliverable:

```text
docs/ICM_parent_child_security_plan.md
```

### Workstream 11 — Alien sampling for artificial curiosity

Use the parent’s reasoning routes to learn two scores:

```text
coherence(route):
    whether the route is a viable procedure

availability(route):
    how often the model or model family has used it
```

Then sample:

```text
high-coherence, low-availability routes
```

The child executes candidate routes in a sandbox and verifies outcomes before
promoting them to useful memory. This is an active-learning and curriculum
mechanism, not a claim of general intuition.

Evaluate against:

```text
dense similarity
intent + geometry routing
high-availability retrieval
random recombination
alien sampling without coherence
alien sampling without availability
```

Metrics:

```text
route novelty
outcome success rate
route diversity
analogical recall
loop count
compute cost
```

Deliverables:

```text
docs/ALIEN_SAMPLING.md
scripts/icm/alien_scoring.py
scripts/icm/alien_eval.py
data/_scratch/icm/alien_routes.jsonl
```

The source paper and full experiment protocol are in `docs/ALIEN_SAMPLING.md`.

---

## 5. Training pipeline

```text
1. Train or select parent.
2. Generate reasoning traces on a probe corpus.
3. Build factual and procedural memory.
4. Extract parent geometry.
5. Align parent geometry to child dimension.
6. Initialize or regularize child with inherited geometry.
7. Train child with task loss + trace distillation.
8. Train child to query memory.
9. Train loop schedule.
10. Evaluate against parent.
```

---

## 6. Runtime pipeline

```text
user query
    ↓
child emits intent + geometry
    ↓
retrieve content key
    ↓
retrieve reasoning key
    ↓
merge candidates deterministically
    ↓
child reasons over retrieved substrate
    ↓
tool use / verification
    ↓
answer or action
    ↓
outcome written to memory
```

---

## 7. Acceptance criteria

### Minimum useful result

```text
child + memory
    outperforms child alone
    on coding and reasoning tasks
```

### Strong result

```text
child + memory + inherited geometry
    approaches parent
    on target domains
```

### Production-ready result

```text
child + substrate
    runs locally
    retrieves audited memory
    executes inherited procedures
    at much lower cost than parent
```

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Parent reasoning is too tacit to externalize | Focus first on coding and debugging, where procedures are explicit |
| Geometry does not transfer across dimensions | Use adapters and test anti-aligned controls |
| Child lacks capacity | Start at 125M; scale only after substrate works |
| Memory retrieval is noisy | Keep content and reasoning keys separate; use deterministic filters |
| Child overfits parent style | Use multiple trace sources and diverse domains |
| Loops overthink | Use adaptive depth and early exit |
| Inherited blind spots propagate | Measure suppressions and include blind-spot audits |

---

## 9. First concrete milestone

Build a small, closed-domain prototype:

```text
domain:
  build debugging

parent:
  strong external model or large Discernment model

child:
  125M looped Discernment encoder

substrate:
  build logs, code, patch traces, debugging procedures
```

Success:

```text
child + substrate
    solves more build failures
    than child alone
```

That is the first proof that inherited procedure and memory can recover part
of the parent’s capability.

---

## 10. Bottom line

The downstream project is:

```text
frontier parent as compiler
memory substrate as product
small child as executor
```

If the substrate captures:

```text
knowledge
procedure
intent
geometry
```

then a small model can inherit much of the parent’s practical capability
without inheriting its inference cost.
