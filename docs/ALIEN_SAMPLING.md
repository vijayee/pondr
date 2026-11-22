# Alien Sampling for Artificial Curiosity

Status: proposed experiment, not implemented.

Source paper:

- [The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research Directions](https://arxiv.org/html/2603.01092)

Related work:

- `SEMANTIC_CARTOGRAPHY.md`
- `related/ICM/DISCERNMENT_RELATION.md`
- `related/ICM/ICM_predicate_intent_model.md`
- `related/ICM/ICM_frontier_parent_child_architecture.md`
- `related/ICM/ICM_frontier_parent_child_plan.md`
- `stage6_index/RESULTS.md`

---

## 1. Purpose

Test whether a model can discover useful new reasoning procedures by
sampling combinations that are:

```text
coherent
but
underexplored
```

This is a candidate mechanism for **artificial curiosity**, not a claim of
general artificial intuition.

The paper’s useful decomposition is:

```text
coherence   = whether an idea is plausible
availability = whether the current community is likely to produce it
```

The proposed Discernment mapping is:

```text
intuition  = coherence model
curiosity  = availability model
```

The exploration objective is therefore:

```text
score(route) = coherence(route) - λ × availability(route)
```

where higher scores favor valid but underused reasoning directions.

---

## 2. What the source paper establishes

The paper decomposes papers into conceptual units and clusters those units
into a shared vocabulary of idea atoms. It then learns:

1. a **coherence model**, which scores whether atom combinations form a viable
   research direction;
2. an **availability model**, which estimates whether existing author
   communities are positioned to produce those combinations.

Sampling high-coherence, low-availability combinations produces more diverse
research portfolios while remaining competitive on quality evaluations.

Reported results include:

- a more diverse idea portfolio than frontier LLM ideation baselines,
- competitive per-idea quality under human, LLM, and downstream evaluation,
- recovery of 2025 combinations from pre-2025 components at **35.9× chance**,
- improved recovery of strictly novel combinations to **3.24× chance** when
  unavailability is weighted.

These results support guided exploration. They do **not** establish a general
mechanism for intuition, and they are specific to the paper’s corpus and
availability definition.

---

## 3. Mapping to Discernment and ICM

| Paper concept | Discernment/ICM counterpart |
|---|---|
| Idea atom | Intent primitive, semantic role type, H-Net chunk, or reasoning landmark |
| Coherence model | A model of whether an intent or reasoning-route combination is viable |
| Availability model | A model of how frequently a route has been used by this model or model family |
| Alien sampler | A retrieval policy that selects coherent but underused procedures |
| Outcome evaluation | ICM intent-vs-outcome qualification |

The crucial addition is that retrieval no longer means only:

```text
retrieve the most familiar or most similar procedure
```

It can also mean:

```text
retrieve a coherent but underused procedure
```

This creates two retrieval modes:

```text
intuitive recall:
    select the most coherent known procedure

curious recall:
    select a coherent but underexplored procedure
```

---

## 4. Artificial curiosity versus novelty gating

The existing surprise gate asks:

> Is this memory surprising enough to store?

Alien sampling asks a different question:

> Is this reasoning direction coherent but underexplored?

These are not equivalent.

A random surprising memory may be incoherent. A familiar but useful procedure
may have low surprise but high coherence. Alien sampling deliberately searches
between those extremes.

This makes alien sampling a **retrieval policy**, not merely a novelty
detector.

---

## 5. Proposed mechanism

### 5.1 Reasoning primitives

Use a bounded vocabulary of reasoning primitives rather than raw tokens:

```text
verify
compare
isolate
inspect
patch
test
retrieve
```

These can be extracted from:

- intent frames,
- predicate-first intent records,
- learned H-Net chunks,
- reasoning traces,
- semantic GPS records.

### 5.2 Coherence model

Estimate whether a combination of primitives forms a viable reasoning route.

Training signal:

```text
parent reasoning traces that led to successful outcomes
```

The coherence model should learn route-level compatibility, not just text
similarity.

### 5.3 Availability model

Estimate how often a route or combination has been used.

Availability is relative to a specified community:

```text
this child
this model family
this repository
this deployment history
```

It is not an absolute measure of human novelty.

### 5.4 Alien sampler

Sample routes that maximize:

```text
coherence - λ × availability
```

Candidates are then executed and verified. Successful routes are written back
to memory. Failed routes remain labeled as failed or unresolved rather than
being silently promoted to useful memory.

---

## 6. Experiment E-ALIEN-1

### 6.1 Goal

Determine whether alien sampling lets a small Discernment child discover
useful reasoning procedures that dense similarity and intent-only retrieval
miss.

### 6.2 Initial scale

```text
parent: 8B open-weight model
children: 125M, 500M, and 1B looped Discernment models
domain: code debugging and analogical reasoning
```

Do not begin with a frontier-scale parent or a 27B child.

### 6.3 Corpus

Build a route corpus containing:

```text
intent frame
reasoning primitives
route manual
landmarks
turn rules
geometry trajectory
outcome
usage count
verification result
```

Sources:

1. Parent-generated coding and debugging traces.
2. Existing ICM `icapture` records.
3. FewRel and analogical-reasoning pairs.
4. Synthetic but mechanically verifiable route combinations.

### 6.4 Baselines

Compare:

| Arm | Retrieval policy |
|---|---|
| 1 | Dense similarity |
| 2 | Intent + geometry routing |
| 3 | High-availability retrieval |
| 4 | Random recombination of valid primitives |
| 5 | Alien sampling without coherence |
| 6 | Alien sampling without availability |
| 7 | Full alien sampling |

### 6.5 Ablations

Test:

```text
λ = 0
λ = 0.25
λ = 0.5
λ = 1.0
λ = 2.0
```

Also compare:

- fixed versus learned availability,
- token-level versus chunk-level primitives,
- frozen versus jointly trained coherence and availability models,
- Procrustes-only versus learned geometry adapters.

### 6.6 Metrics

Measure:

```text
task accuracy
route novelty
outcome success rate
route diversity
analogical recall
loop count
latency
compute cost
audit-record size
```

Novelty alone is not success. A candidate must also be useful and verifiable.

### 6.7 Temporal validation

Use a frozen early-period corpus to score routes, then test whether the
sampler recovers useful routes that appear later.

This mirrors the paper’s strongest evaluation: do not merely generate
plausible text; test whether the search policy discovers directions that were
actually useful later.

### 6.8 Success criteria

Minimum useful result:

```text
alien sampling finds valid routes missed by dense similarity
without reducing outcome success rate
```

Strong result:

```text
alien sampling improves both route diversity and outcome success
while reducing loop count or search compute
```

Failure conditions:

- low-availability routes are mostly incoherent,
- novelty rises but task accuracy collapses,
- the sampler only rediscovers high-frequency routes,
- route diversity increases without useful outcomes,
- the model overfits parent-specific reasoning patterns or blind spots.

---

## 7. Implementation workstreams

### Workstream 1 — Route schema

Define a machine-readable route record containing:

```json
{
  "intent": "...",
  "primitives": ["verify", "compare", "isolate"],
  "geometry": "...",
  "outcome": "...",
  "usage_count": 0,
  "verified": false
}
```

Deliverable:

```text
ops/route_schema.json
```

### Workstream 2 — Corpus construction

Generate parent and child reasoning traces, then convert them to route
records.

Deliverable:

```text
data/_scratch/icm/alien_routes.jsonl
```

### Workstream 3 — Coherence and availability scorers

Implement:

```text
coherence(route)
availability(route)
alien_score(route)
```

Start with simple learned or deterministic baselines before training larger
models.

Deliverable:

```text
stage6_index/alien_scoring.py
```

### Workstream 4 — Sampler and evaluator

Sample candidates, execute them in a sandbox, verify outcomes, and write
qualified routes back to memory.

Deliverable:

```text
stage6_index/alien_sampler.py
stage6_index/alien_eval.py
```

### Workstream 5 — Geometry and chunk ablations

Compare:

```text
raw token primitives
H-Net chunk primitives
intent-frame primitives
```

This tests whether learned semantic chunks make better route atoms than fixed
tokens.

---

## 8. Relationship to the frontier-parent architecture

The parent can be used to compile the initial route vocabulary and coherence
model. The child can then explore underused combinations locally.

```text
parent:
    supplies coherent route primitives and initial geometry

child:
    samples unfamiliar combinations
    executes them
    verifies outcomes
```

This makes frontier cost amortizable: the parent supplies the map, while many
small children explore different underused regions of it.

---

## 9. Risks and limits

| Risk | Mitigation |
|---|---|
| Availability is mistaken for absolute novelty | Always report the community and time window used to define availability |
| Low-availability routes are mostly invalid | Require coherence and outcome verification |
| Sampler inherits parent blind spots | Include anti-aligned controls and independent route sources |
| Curiosity becomes random recombination | Constrain candidates with route schemas and execution checks |
| Model overfits parent style | Evaluate on held-out tasks and out-of-domain routes |
| New routes are unverifiable | Abstain instead of promoting them to useful memory |

The source paper’s availability model is community-relative. Our version must
be equally explicit: “underexplored” means underexplored by this model, this
family, or this corpus, not universally novel.

---

## 10. Bottom line

Alien sampling gives Discernment a candidate mechanism for artificial
curiosity:

```text
intuition:
    coherence model

curiosity:
    availability model
```

The experiment is not to prove that a model can be intuitively creative. It is
to test whether a small model can find **coherent but underexplored reasoning
routes** more effectively than dense similarity alone.

If it works, the memory substrate becomes not only a store of past procedures
but also a generator of new, verifiable ones.
