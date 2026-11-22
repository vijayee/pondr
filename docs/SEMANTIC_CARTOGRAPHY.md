# Semantic Cartography — Maps, Routes, and Model-to-Model Navigation

Status: research program proposal, not implemented.

Related Discernment artifacts:

- `PAPER_DRAFT.md`
- `DISCOVERIES.md`
- `RESEARCH_NOTES.md`
- `ROADMAP.md`
- `ops/WARP_DYNAMICS_EXPERIMENTS.md`
- `ops/SPATIAL_THOUGHT_MAPPING.md`
- `stage2/warp_profile.py`
- `stage6_index/PAPER.md`
- `stage6_index/RESULTS.md`
- `stage5_encoder/E2_ENCODER_RUNG0_PLAN.md`

External evidence:

- [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)
- [From Tokens to Bytes I: Scaling Hierarchical Networks](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1)
- [From Tokens to Bytes II: Learning from Raw Data](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2)

Downstream application notes:

- `related/ICM/SEMANTIC_CARTOGRAPHY.md`
- `related/ICM/ICM_predicate_intent_model.md`
- `related/ICM/ICM_frontier_parent_child_architecture.md`
- `related/ICM/DISCERNMENT_RELATION.md`
- Original Pondr copies: `pondr/docs/`

---

## 1. The claim

Semantic cartography is the empirical study of how models represent, measure,
align, and navigate meaning.

The key distinction is:

```text
a map is not a language
a route through the map is
```

A map tells you what the semantic space looks like.  
A route tells you how to move through it.

A model does not need to hand another model its whole brain. It needs to hand
it:

```text
start point
destination
route
landmarks
turn rules
arrival check
```

That is not raw geometry. That is a structured navigation language.

---

## 2. Why this is now more than a metaphor

Discernment already has the instruments needed to make this measurable:

| Discernment instrument | Cartographic role |
|---|---|
| Warp profile | Terrain profile of the representation space |
| Delta-warp trajectory | Route through that terrain |
| Principal-subspace resonance | Landmark alignment |
| Plücker / wedge coordinates | Orientation and direction |
| Projection features | Orientation-invariant similarity |
| Intent frames | Route labels |
| Reasoning traces | Roads and paths |
| Memory substrate | Persistent terrain |
| Audit ledger | Survey log |
| Outcome qualification | Arrival check |

The important shift is that these are no longer only interpretability tools.
They can be used as:

```text
measurement instruments
transfer keys
navigation instructions
```

That makes semantic cartography an empirical research program rather than a
poetic analogy.

### 2.1 Dynamic chunking as learned landmarks

Recent H-Net results provide direct external evidence that semantic
boundaries can be learned end-to-end.

H-Net replaces fixed BPE tokenization with dynamic, context-dependent
chunking. In the controlled scaling ladder:

```text
1-stage H-Net vs BPE Transformer:
    validation: 1.28× compute efficiency
    OLMES:      2.06× compute efficiency
```

A larger two-stage H-Net reportedly reaches SmolLM3’s final average score
after roughly 1T tokens, compared with about 11T tokens for the matched
open-source baseline.

The key cartographic interpretation is:

> H-Net’s learned chunk boundaries are candidate landmarks.

A route manual does not need to refer only to hand-labeled tokens. It can
refer to learned, content-dependent units such as:

```text
variable name
dependency diff
error signature
code block
```

H-Net’s recursive hierarchy also provides a concrete version of an atlas:

```text
bytes → chunks → higher-level abstractions
```

This suggests that route manuals should be written over learned semantic
chunks rather than fixed token sequences.

The H-Net distillation appendix is additionally relevant to model-to-model
translation. It aligns a byte-level H-Net to a frozen tokenized model using:

```text
boundary prediction
hidden-state matching
dimension adapters
```

That supports the idea that geometry translation can be learned rather than
relying only on rigid Procrustes alignment.

H-Net does not prove universal cross-model semantic alignment. It shows that
a learned interface can approximate another model’s internal representation
space. Cross-model route transfer still needs shared landmarks, alignment,
and verification.

---

## 3. Core objects

### 3.1 Chart

A chart is the measured geometry of a model’s representation space.

It includes:

```text
principal subspaces
covariance structure
warp profile
orientation metrics
landmark states
```

A chart answers:

> What does this model’s semantic terrain look like?

### 3.2 Route

A route is a sequence of transitions through a chart.

It includes:

```text
reasoning steps
state transitions
intermediate landmarks
conditional turn rules
```

A route answers:

> How does the model move through this terrain?

### 3.3 Landmark

A landmark is a stable intermediate state that a model can recognize during
reasoning.

Examples:

```text
error signature
isolated module
dependency diff
failing test
passing test
```

Landmarks make a route navigable by another model.

### 3.4 Turn rule

A turn rule is a conditional transition.

Example:

```text
if error is dependency-related:
    inspect dependency diff
```

Turn rules are what make a route executable rather than merely descriptive.

### 3.5 Arrival check

An arrival check verifies that the destination was reached.

Example:

```text
all tests pass
diff is minimal
no unrelated regressions
```

Without an arrival check, a route is only a story.

### 3.6 Atlas

An atlas is a collection of charts plus the transition maps between them.

This is the natural object for:

```text
multi-model transfer
domain adaptation
model families
```

---

## 4. Semantic GPS instructions

The central inter-model artifact is a structured route record:

```json
{
  "origin": {
    "intent": "diagnose(build_failure)",
    "state": "current_context"
  },
  "destination": {
    "outcome": "tests_pass",
    "verification": "run_test_suite"
  },
  "route": [
    "read_error_log",
    "reproduce_failure",
    "isolate_module",
    "inspect_dependency_diff",
    "apply_patch",
    "run_tests"
  ],
  "landmarks": [
    "error_signature",
    "failing_module",
    "dependency_diff",
    "passing_tests"
  ],
  "turn_rules": [
    {
      "if": "dependency_error",
      "then": "inspect_dependency_diff"
    }
  ],
  "geometry": {
    "trajectory": "verify-compare-decide",
    "orientation": "same"
  },
  "arrival_check": {
    "condition": "all_tests_pass",
    "action": "stop"
  }
}
```

This is the “GPS driving instructions” object.

It is not:

```text
here is my map
```

It is:

```text
start here
take this route
watch for this landmark
turn when this condition holds
stop when this test passes
```

### 4.1 Procrustes alignment

Procrustes alignment is the process of finding the best rigid transformation
that makes one model’s coordinate system match another’s.

Plain English:

> It rotates one map until it lines up with another map.

Two models can encode the same meaning in different internal bases. Procrustes
alignment finds the rotation that makes those bases correspond.

Given paired hidden states:

```text
X = child points
Y = parent points
```

it solves:

```text
minimize || XQ − Y ||
subject to QᵀQ = I
```

where `Q` is an orthogonal rotation or reflection.

If the models have different dimensions, first project both into a shared
subspace using PCA, CCA, or a learned adapter.

Important caveat for orientation-sensitive geometry:

```text
standard Procrustes allows reflection
```

For Plücker or signed-wedge work, you may need rotation-only alignment or a
check that:

```text
det(Q) > 0
```

so the handedness of the route is preserved.

### 4.2 Route manuals, not just directions

A real semantic route is rarely a fixed list of steps. It is closer to a
manual than to a driving direction.

A useful route record contains:

```text
normal path
exceptions
fallbacks
loop guards
stop conditions
verification
```

Example:

```json
{
  "steps": [
    {
      "do": "read_error_log",
      "unless": "no_log_available"
    },
    {
      "do": "inspect_dependency_diff",
      "if": "dependency_error"
    },
    {
      "do": "apply_patch",
      "after": "failure_is_reproduced"
    },
    {
      "do": "run_tests",
      "after": "patch_is_applied"
    }
  ],
  "fallbacks": [
    {
      "when": "diff_not_found",
      "do": "inspect_recent_commits"
    }
  ],
  "loop_guard": {
    "max_repetitions": 3
  },
  "arrival_check": {
    "condition": "all_tests_pass",
    "action": "stop"
  }
}
```

That is why the inter-model object is better described as:

```text
a route manual
```

than as:

```text
a fixed route
```

The route tells the child what to do in the common case.  
The manual tells it what to do when the terrain changes.

---

## 5. Map vs route vs language

| Layer | Meaning |
|---|---|
| Map | Static geometry of semantic space |
| Route | A path through that geometry |
| Manual | A route with conditionals, fallbacks, and loop guards |
| Landmark | A recognizable intermediate state |
| Turn rule | A conditional transition |
| Arrival check | A verification condition |
| Language | A transferable route manual |

The central insight is:

> A map becomes language when it is paired with a traversal protocol.

That is the difference between:

```text
cartography
```

and

```text
navigation
```

---

## 6. Why this matters for training efficiency

Semantic cartography could reduce training cost by replacing brute-force
parameter scaling with:

```text
1. inherited charts
2. transferable routes
3. shared memory substrates
4. reusable reasoning procedures
```

Instead of every model learning every route from scratch:

```text
train one parent
    ↓
survey its semantic terrain
    ↓
extract routes
    ↓
let children inherit and execute those routes
```

This changes the economics of training.

The expensive object becomes:

```text
one-time charting
```

not:

```text
per-model rediscovery
```

---

## 7. What semantic cartography is not

It is not:

```text
just vector search
just interpretability
just latent-space visualization
just knowledge distillation
```

The distinction is:

| Field | Question |
|---|---|
| Interpretability | What is the model doing? |
| Representation learning | How are meanings encoded? |
| Vector search | Which items are similar? |
| Semantic cartography | How can one model navigate another model’s meaning space? |

The last question is new.

---

## 8. Research questions

1. Can semantic spaces be charted consistently across models?
2. Can routes be transferred between models as executable instructions?
3. Can geometry alignment make one model’s map navigable by another?
4. Can inherited routes reduce the number of loops or tokens a child needs?
5. Can small models execute routes discovered by large models?
6. Can routes reveal failure modes before they occur?
7. Can an atlas of charts support multi-domain reasoning?
8. Can semantic cartography reduce training cost without reducing capability?

---

## 9. Minimum viable experiment

Use an 8B open-weight parent and a 125M–1B Discernment child.

Parent emits:

```text
intent frame
reasoning route
landmarks
turn rules
warp trajectory
outcome
```

Child is trained to:

```text
read the route
align its geometry
retrieve by intent
execute the route
verify the outcome
```

Compare arms:

| Arm | Transfer method |
|---|---|
| 1 | Natural-language explanation only |
| 2 | Reasoning trace only |
| 3 | Geometry only |
| 4 | Intent + trace |
| 5 | Intent + trace + geometry |
| 6 | Full semantic GPS record |

Metrics:

```text
task accuracy
loop count
token count
latency
reasoning quality
transfer success
```

The key question:

> Does the semantic GPS record let the child solve the task with less
> computation than the child alone?

---

## 10. Training-efficiency hypothesis

The main hypothesis is:

> Inherited routes reduce the amount of search a model must perform.

Without inherited routes:

```text
input
    ↓
search many possible reasoning paths
    ↓
choose one
    ↓
execute
```

With inherited routes:

```text
input
    ↓
emit intent + geometry
    ↓
retrieve route
    ↓
adapt route to current content
    ↓
execute
    ↓
verify
```

The saving is not that calculation disappears.  
The saving is that search is replaced by retrieval.

---

## 11. Discernment-specific consequence

Discernment is especially suited to this program because its memory is already:

```text
explicit
auditable
orientation-aware
state-conditioned
```

That gives it a natural substrate for:

```text
charts
routes
landmarks
turn rules
arrival checks
```

A standard transformer can be interpreted after the fact.  
Discernment can potentially emit its navigation record directly.

That is the architectural advantage:

> The model does not merely have a semantic geometry.  
> It can expose the geometry and the route as first-class objects.

---

## 12. Human analogy

Humans do not share raw brain states.

They share:

```text
maps
recipes
procedures
driving directions
```

Semantic cartography is the model equivalent.

A parent model would not hand over its brain.  
It would hand over a compressed, structured, navigable experience.

---

## 13. Bottom line

Semantic cartography is the study of:

```text
maps
routes
landmarks
turn rules
arrival checks
```

in semantic space.

The central object is not a raw embedding.  
It is a transferable navigation record.

The field becomes practical when maps are:

```text
measured
aligned
routed
verified
inherited
```

That is the research program.
