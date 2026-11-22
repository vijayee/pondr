# ICM Predicate-First Intent Model

Status: proposal, not implemented.

Related work:

- `docs/ICM_intent_store_spec.md`
- `docs/ICM_final_report.md`
- `docs/ICM_predicate_intent_model_plan.md`
- `docs/ICM_frontier_parent_child_architecture.md`
- `docs/ICM_frontier_parent_child_plan.md`
- `docs/SEMANTIC_CARTOGRAPHY.md`
- `docs/ALIEN_SAMPLING.md`
- `icm/spo_extract.py`
- `icm/intent_keys.py`
- `icm/intent_query.py`
- GrassmannFlow stage-6 vector-index prototype
- Discernment BERT-style encoder plan (`stage5_encoder/E2_ENCODER_RUNG0_PLAN.md`)

---

## 1. Summary

The current ICM intent record is close, but it separates concepts that are
easier to learn if they are reduced to one rule:

> The operative intent is the predicate acting on the object.

This document replaces the current `subject / attitude / relation / object /
qualifier` layout with a simpler predicate-first frame:

```text
speech_act
subject
predicate
object
qualifier
polarity
parent_intent
```

The important distinction is:

| Layer | Meaning | Example |
|---|---|---|
| Speech act | What the utterance is doing communicatively | inform, command, request |
| Action intent | What action the utterance is about | acquire, verify, fix, deploy |

Most statements default to `inform`. Imperative verb-only utterances default
to `command`. When a full subject-predicate-object clause is present and the
predicate is an action, the intent is:

```text
predicate(object)
```

---

## 2. Why this is simpler than the current ICM layout

The current ICM record uses:

```text
subject
attitude
relation
object
qualifier
residue
```

That is workable, but `attitude` and `relation` are doing overlapping work.
In practice:

```text
attitude  -> speech act
relation  -> predicate
residue   -> abstention
```

So the cleaner mapping is:

| Current ICM field | Predicate-first field |
|---|---|
| `attitude` | `speech_act` |
| `relation` | `predicate` |
| `object` | `object` |
| `qualifier` | `qualifier` |
| `residue` | `abstention` |
| — | `polarity` |
| — | `parent_intent` |

The change is not a rewrite. It is a reduction.

---

## 3. Intent frame schema

A write-time intent record becomes:

```json
{
  "speech_act": "command",
  "subject": "SELF",
  "predicate": "verify",
  "object": "archived_version",
  "qualifier": "before_patch",
  "polarity": "positive",
  "parent_intent": null
}
```

Fields:

| Field | Meaning | Example |
|---|---|---|
| `speech_act` | Communicative intent | `inform`, `command`, `request`, `question`, `promise` |
| `subject` | Who holds the intent | `SELF`, `Microsoft`, `user` |
| `predicate` | The action or state verb | `acquire`, `verify`, `fix`, `deploy` |
| `object` | What the predicate acts on | `LinkedIn`, `archived_version`, `login_bug` |
| `qualifier` | Temporal, conditional, or version marker | `before_patch`, `if_tests_pass`, `archived` |
| `polarity` | Positive, negative, or avoid | `positive`, `negative`, `avoid` |
| `parent_intent` | Link to a higher-level goal | intent ID or `null` |

---

## 4. Default extraction rules

### 4.1 Declarative statement

```text
"Microsoft acquired LinkedIn."
```

Extract:

```text
speech_act: inform
subject:    Microsoft
predicate:  acquired
object:     LinkedIn
```

The communicative intent is `inform`.  
The action intent is `acquire(LinkedIn)`.

### 4.2 Imperative verb alone

```text
"Stop."
```

Extract:

```text
speech_act: command
subject:    SELF
predicate:  stop
object:     listener
```

For addressee-directed verbs:

```text
stop, listen, look, wait
```

the object is the listener.

For task-directed verbs:

```text
run, build, test
```

the object is an implicit task, not the listener.

### 4.3 Full SPO with an action predicate

```text
"Fix the login bug."
```

Extract:

```text
speech_act: command
subject:    SELF
predicate:  fix
object:     login_bug
```

Intent:

```text
fix(login_bug)
```

### 4.4 Negation or avoidance

```text
"Do not enable the flag."
```

Extract:

```text
speech_act: command
subject:    SELF
predicate:  enable
object:     flag
polarity:   negative
```

Intent:

```text
avoid(enable(flag))
```

### 4.5 Conditional or temporal qualifier

```text
"Verify the archived version before the patch."
```

Extract:

```text
speech_act: command
subject:    SELF
predicate:  verify
object:     archived_version
qualifier:  before_patch
```

### 4.6 Multi-requirement intent

```text
"Fix the login bug, preserve the old API, and add a test."
```

Extract three frames:

```text
fix(login_bug)
preserve(old_api)
add(test)
```

with one shared parent:

```text
goal: stabilize_login
```

---

## 5. Intent hierarchies

Hierarchies are important, but they do not need to be free-form.

Represent them as a tree of intent frames:

```text
goal:
  deploy_the_app
    step:
      if_tests_pass
        action:
          deploy
```

Each frame carries a `parent_intent` pointer. The model does not need to infer
an entire plan as one object. It only needs to emit:

```text
frame + parent_link
```

That is tractable for a structured-prediction head.

---

## 6. Intent vs outcome

This is the part that qualifies memory.

If the intent is:

```text
verify(archived_version)
```

then the desired outcome is:

```text
archived_version_verified
```

So memory qualification becomes:

```text
compare intent frame to outcome frame
```

Example:

```text
Intent:
  predicate: verify
  object:    archived_version

Outcome:
  predicate: verified
  object:    archived_version
  result:    true
```

Qualification rule:

```text
if outcome matches intent:
    memory is useful

if outcome does not match intent:
    memory is violated or irrelevant

if a new intent appears after failure:
    record intent expansion
```

This gives ICM a mechanical target for its non-reach loop.

---

## 7. Reasoning geometry

The predicate-first intent model pairs naturally with the stage-6 geometry
work.

A memory unit is stored with both:

```text
document frame: [subject_embedding, object_embedding]
intent frame:   [subject_embedding, predicate_embedding, object_embedding,
                qualifier_embedding]
```

The reasoning geometry is the orientation-aware structure of those frames:

```json
{
  "frame": "[subject_embedding, object_embedding]",
  "predicate": "predicate_embedding",
  "qualifier": "qualifier_embedding",
  "orientation": "same"
}
```

The model must emit this geometry explicitly. It should not be a hidden latent
that only the retrieval layer can inspect.

### 7.1 Exact recall is not the goal

The purpose of these geometric keys is **not** to find the exact same memory
again.

Exact recall asks:

```text
Have I seen this same content before?
```

Thought-process recall asks:

```text
Have I reasoned like this before?
```

Those are different retrieval problems.

A generic ANN index optimizes surface similarity. The downstream Pondr
subconscious needs structural similarity of reasoning.

### 7.2 Example: pouring a drink vs pouring oil from a pan

```text
"pouring a drink"
```

and

```text
"Pour out the oil in the pan."
```

have different surface content:

```text
drink  ≠ oil
glass  ≠ pan
```

But they share the same reasoning shape:

```text
pour(liquid, out of a container)
```

A content-only vector index may treat them as distant because the nouns and
containers differ. A reasoning-geometry key should recognize that both
statements use:

```text
predicate:      pour
object type:    liquid
source role:    container
direction:      source -> destination
reasoning mode: transfer a liquid out of a container
```

That is the memory the artificial subconscious should surface.

### 7.3 Why Plücker coordinates alone are not sufficient

Plücker coordinates are excellent at encoding orientation:

```text
[A, B] vs [B, A]
```

So they can distinguish:

```text
pour from a glass
```

from:

```text
pour into a glass
```

But a raw subject/object Plücker key is still content-sensitive:

```text
[drink, glass]
```

and

```text
[pan, oil]
```

are different subspaces. Raw Plücker coordinates alone do not automatically
connect the two thoughts.

To match the thought process, the key needs an abstraction layer:

```text
predicate class
semantic role types
orientation
reasoning-state trajectory
```

So the useful key is not raw Plücker alone. It is:

```text
abstract intent frame
+ orientation geometry
+ reasoning-state trajectory
```

### 7.4 Content key vs reasoning key

Each memory should therefore carry two keys:

```text
content key:
    raw subject/object embedding

reasoning key:
    predicate frame
    semantic role types
    orientation
    warp / reasoning trajectory
```

The subconscious can then retrieve in two modes:

```text
exact recall      -> content key
analogical recall -> reasoning key
```

That distinction is the point of the geometry layer.

---

## 8. Downstream design: the artificial subconscious

This model supports Pondr’s downstream goal:

> Pondr acts as an artificial subconscious that pulls memory based on intent
> and reasoning geometry, then supplies relevant context to a model that can
> emit its own intent and reasoning geometry when storing memory.

Operationally, that means:

```text
model emits intent + reasoning geometry
        ↓
write-time keyer
        ↓
WaveDB graph + vector + KV
        ↓
query-time intent/geometry router
        ↓
candidate memory
        ↓
deterministic filter + epoch re-rank
        ↓
context returned to the model
```

The same structure is used at write time and query time:

- write time: store memory under the emitted intent and geometry,
- query time: retrieve memory using the current intent and geometry.

That symmetry is the point.

The subconscious therefore has two retrieval modes:

```text
exact recall:
    retrieve the same content

analogical recall:
    retrieve a memory produced by the same reasoning shape
```

The downstream value is the second mode.

---

## 9. Relationship to existing ICM code

The current ICM implementation already has the right bones:

| Existing file | Role in this model |
|---|---|
| `icm/spo_extract.py` | Extracts subject, attitude, relation, object, residue |
| `icm/intent_keys.py` | Builds write-time intent keys |
| `icm/intent_query.py` | Two-stage dense candidate + deterministic re-rank |
| `icm/capture.py` | Records intent, outcome, memory units, epoch, labels |

The predicate-first model mainly changes the labels:

```text
attitude -> speech_act
relation -> predicate
residue  -> abstention
```

and adds:

```text
polarity
parent_intent
reasoning geometry
```

---

## 10. Non-goals

- Do not infer unstated intent beyond the declared rules.
- Do not replace the deterministic filter with a learned relevance lens.
- Do not blend dense and structural scores.
- Do not make reasoning geometry a hidden latent.
- Do not force every statement into an operative action intent; allow
  abstention.

---

## 11. Bottom line

The predicate-first model reduces ICM intent to:

```text
speech_act + predicate(object) + qualifier
```

Memory qualification then becomes:

```text
intent frame vs outcome frame
```

The geometric layer’s job is not exact nearest-neighbor recall. Its job is:

```text
find memories produced by the same reasoning shape
```

That is what makes the downstream artificial subconscious useful: it can
connect “pouring a drink” to “pour out the oil in the pan” because both share
the same predicate, role types, orientation, and reasoning mode.
