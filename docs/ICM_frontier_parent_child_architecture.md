# Frontier Parent / Small Child Reasoning Inheritance

Status: downstream architecture proposal, not implemented.

Related work:

- `docs/ICM_predicate_intent_model.md`
- `docs/ICM_predicate_intent_model_plan.md`
- `docs/ICM_intent_store_spec.md`
- `docs/ICM_final_report.md`
- GrassmannFlow stage-6 vector-index prototype
- Discernment BERT-style encoder plan
- `ops/WARP_DYNAMICS_EXPERIMENTS.md`
- `ops/SPATIAL_THOUGHT_MAPPING.md`

---

## 1. Thesis

A frontier-scale model is expensive to train and expensive to serve. The
proposal here is to use it **once** as a compiler of durable substrates, then
let many smaller models inherit and execute those substrates cheaply.

```text
frontier parent
    ↓
knowledge substrate
reasoning substrate
intent/geometry substrate
    ↓
many small looped Discernment children
```

The parent is not the product.

The product is:

```text
small child
+ shared memory
+ inherited reasoning geometry
+ tools
```

That reframes a frontier model from “a giant model everyone queries” into “a
one-time compiler of knowledge, procedure, and geometry.”

---

## 2. Why this is worth doing

Most people do not need frontier-level general intelligence on every query.
They need:

```text
coding help
domain knowledge
personal context
debugging
planning
retrieval
```

A small model with the right memory can outperform a frontier model that lacks
that context.

The missing piece is not always knowledge. It is often:

```text
reasoning procedure
```

So the downstream system must inherit both:

| Gap | How this architecture addresses it |
|---|---|
| Knowledge breadth | External memory substrate |
| Reasoning depth | Loops in the child + inherited procedures |
| Analogical recall | Intent and geometry keys |
| Auditability | Explicit memory records and geometry emission |
| Accessibility | Small model, local memory, cheap inference |

---

## 3. Roles

### 3.1 Frontier parent

The parent is the expensive model used once to produce durable artifacts.

It can be:

- a frontier model trained by us,
- a strong external model used as a teacher,
- or a large Discernment parent.

Its job is not to serve every user. Its job is to compile:

```text
knowledge
procedures
reasoning traces
intent frames
geometry operators
```

### 3.2 Memory substrate

The memory substrate stores what the parent learned or reasoned about.

It contains:

```text
documents
code
API references
domain rules
past decisions
debugging procedures
reasoning traces
intent frames
geometry keys
```

This is the child’s external knowledge.

### 3.3 Small child

The child is a cheap, looped Discernment model.

It does not need to store all world knowledge internally.

It needs enough capacity to:

```text
emit intent
query memory
retrieve relevant procedures
execute reasoning loops
verify outcomes
```

The child is the runtime executor, not the knowledge store.

---

## 4. What is transferred from parent to child

The parent produces four substrates.

### 4.1 Knowledge substrate

This is explicit knowledge:

```text
facts
documentation
code
APIs
domain rules
past outcomes
```

This is stored in memory.

### 4.2 Procedural reasoning substrate

This is the part most likely to be lost in ordinary distillation.

The parent emits structured reasoning traces:

```text
intent
plan
retrieval query
evidence considered
decision
verification
outcome
```

Example:

```text
intent:
  diagnose(build_failure)

steps:
  1. read error log
  2. reproduce failure
  3. isolate failing module
  4. inspect dependency change
  5. patch
  6. run tests
  7. verify outcome
```

A child can inherit this procedure even if it does not inherit the parent’s
full world knowledge.

### 4.3 Intent/geometry substrate

Each reasoning trace carries an intent frame and reasoning geometry:

```text
predicate frame
orientation
qualifier
warp trajectory
```

This supports analogical recall:

```text
pouring a drink
    ->
pour out the oil in the pan
```

The surface content differs, but the reasoning shape matches.

### 4.4 Geometry-generating operator

The strongest inheritance object is not the parent’s raw eigenspace.

It is the operator that generates useful geometry.

```text
operator first, geometry second
```

The child should inherit a way to shape its reasoning space, not merely a copy
of the parent’s current subspace.

---

## 5. What is not transferred

This architecture is powerful, but it has honest limits.

The child still loses:

```text
implicit priors
broad common sense
unusual analogies
frontier-level judgment
```

Memory can reduce those gaps, but cannot eliminate them.

The realistic claim is:

> A small child with inherited memory and geometry can match a frontier
> parent on the domains the substrate covers.

Not:

> A small child with retrieval is generally equal to a frontier model.

---

## 6. Architecture

```text
frontier parent
    ↓
knowledge substrate
procedural reasoning substrate
intent/geometry substrate
geometry operator
    ↓
shared memory index
    ↓
small looped Discernment child
    ↓
user / agent runtime
```

A more detailed runtime view:

```text
child emits intent + reasoning geometry
        ↓
memory retrieval by content key
        ↓
memory retrieval by reasoning key
        ↓
candidate procedures and facts
        ↓
child executes looped reasoning
        ↓
tool use / verification
        ↓
answer or action
        ↓
outcome written back to memory
```

The same geometry is used at write time and query time.

---

## 7. Child training process

The child is trained with:

```text
task loss
+ reasoning-trace distillation
+ geometry-distillation loss
+ memory-retrieval training
+ loop-schedule training
```

### 7.1 Task loss

Normal next-token or sequence prediction.

### 7.2 Reasoning-trace distillation

The child learns to reproduce the parent’s structured procedure:

```text
intent
plan
query
evidence
decision
verification
```

This is stronger than answer-only distillation because the child learns
*how* to reason, not just what to output.

### 7.3 Geometry distillation

The child is regularized to produce a reasoning geometry compatible with the
parent:

```text
warp profile
delta-warp trajectory
memory-write geometry
```

This transfers the shape of reasoning, not the facts.

### 7.4 Memory-retrieval training

The child learns when and how to query memory:

```text
what intent to emit
which geometry key to use
when to retrieve
how to combine retrieved evidence
```

### 7.5 Loop training

The child uses loops to recover reasoning depth:

```text
loop count 1, 2, 4, 8
```

with adaptive gating so the model does not overthink simple queries.

---

## 8. Memory schema

Each memory record stores two keys:

```text
content key:
    raw subject/object embedding

reasoning key:
    predicate frame
    semantic role types
    orientation
    warp trajectory
```

This gives the child two retrieval modes:

```text
exact recall:
    find the same content

analogical recall:
    find the same reasoning shape
```

The downstream value is the second mode.

---

## 9. Example: inherited debugging procedure

Parent reasoning trace:

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

geometry:
  predicate: diagnose
  object: build_failure
  orientation: same
  warp trajectory: verify-compare-decide
```

Later, the child encounters a similar build failure.

It retrieves:

```text
diagnose(build_failure)
```

and receives:

```text
the procedure
the relevant code
likely failure modes
verification steps
```

The child does not need the parent’s full world knowledge. It needs enough
capacity to execute the inherited procedure.

---

## 10. Why this amortizes frontier cost

The parent is expensive once.

The children are cheap many times.

```text
cost of parent
    /
number of children and users
```

This makes a frontier model economically defensible as infrastructure rather
than as a per-query service.

The parent becomes:

```text
knowledge curator
reasoning-trace generator
geometry donor
memory builder
```

The child becomes:

```text
cheap executor
```

---

## 11. Security and accessibility

This architecture supports the downstream goals:

| Goal | How this architecture helps |
|---|---|
| Efficiency | Small child, local memory, amortized parent |
| Security | Memory can stay local; child can run offline |
| Accessibility | Cheap hardware can execute the child |
| Auditability | Memory writes and reasoning traces are explicit |
| Privacy | Personal memory does not require a frontier query |
| Portability | Substrate can be shared without sharing the parent |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Child lacks capacity to use inherited procedures | Start with 125M-scale children; measure task ceiling |
| Geometry transfer fails across sizes | Use Procrustes/PCA/adapter alignment before assuming portability |
| Parent blind spots are inherited | Measure inherited suppressions; test with anti-aligned controls |
| Memory retrieval fails | Keep exact recall and reasoning-key recall separate |
| Child overfits parent reasoning style | Train on diverse traces and multiple parents |
| Loops cause overthinking | Use adaptive loop depth and early exit |
| Memory becomes a security liability | Use local storage, access control, and audit records |

---

## 13. Relationship to existing work

| Existing work | Role here |
|---|---|
| ICM | Intent-conditioned memory and deterministic retrieval |
| Predicate-first intent model | Mechanical intent extraction |
| Stage-6 geometry index | Orientation-aware memory keys |
| Discernment encoder | Purpose-built child architecture |
| Warp dynamics | Geometry inheritance and reasoning-state measurement |
| Looped transformers | Recover reasoning depth at fixed parameter count |
| WaveDB | Durable storage for the substrate |

---

## 14. Success criteria

A successful downstream system would show:

```text
child + memory + geometry
    approximates parent
    on target domains
```

while providing:

```text
lower latency
lower cost
local execution
auditable retrieval
```

The key benchmark is not general frontier equivalence. It is:

> Does the inherited child perform like the parent on the domains the
> substrate covers?

---

## 15. Open questions

1. How much frontier reasoning is procedural enough to externalize?
2. What is the minimum child size that can execute inherited procedures?
3. Can geometry transfer survive a dimension reduction from parent to child?
4. How much parent knowledge must remain in memory versus in child weights?
5. Can multiple children share one memory substrate without interference?
6. How do we prevent inherited blind spots from becoming systemic?
7. What is the best loop schedule for cheap reasoning?

---

## 16. Bottom line

The frontier parent is a one-time compiler.

The child is a cheap executor.

The memory substrate is the durable product.

```text
frontier intelligence
    ↓
compiled knowledge, procedure, and geometry
    ↓
many small accessible children
```

That is the downstream project.
