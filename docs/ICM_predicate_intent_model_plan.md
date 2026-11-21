# ICM Predicate-First Intent Model — Implementation Plan

Status: not started.

Related documents:

- `docs/ICM_predicate_intent_model.md`
- `docs/ICM_intent_store_spec.md`
- `docs/ICM_final_report.md`
- `icm/spo_extract.py`
- `icm/intent_keys.py`
- `icm/intent_query.py`
- `icm/capture.py`

---

## 1. Goal

Build an intent extractor and memory-qualification layer where:

```text
intent = speech_act + predicate(object) + qualifier
```

and where memory usefulness is judged by comparing:

```text
intent frame vs outcome frame
```

The downstream target is Pondr’s artificial-subconscious design:

```text
model emits intent + reasoning geometry
        ↓
memory is stored under that intent and geometry
        ↓
later queries retrieve memory using the current intent and geometry
        ↓
relevant context is returned to the model
```

---

## 2. Why this is worth doing

ICM already measured that:

- intent is the binding constraint in retrieval,
- dense similarity alone is weak on counterfactual pairs,
- a judge-built intent key plus deterministic filtering can reach 100% on the
  filtered git-counterfactual pool,
- real intents are multi-requirement, averaging 4.63 requirements.

The missing piece is a cheap, auditable, mechanical intent extractor that does
not require a language-model judge on every write.

The predicate-first model provides that.

---

## 3. Current baseline numbers to beat or match

| Task | Current result |
|---|---:|
| ICM user-originated before/after direction | 0.963 |
| ICM filtered git counterfactual accuracy | 1.000 |
| Dense similarity on the same counterfactual pool | 0.242 |
| Stage-6 FewRel relation top1, prompted MPNet | 0.683 |
| Stage-6 FewRel relation top1, supervised rerank | 0.707 |
| Stage-6 cross-domain bidirectional max top1 | 0.606 |

The first target is to match the ICM direction result while beating the
stage-6 relation baseline.

---

## 4. Architecture

### 4.1 Extraction

Input:

```text
"Verify the archived version before the patch."
```

Output:

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

### 4.2 Reasoning geometry

The same extractor also emits:

```json
{
  "frame": "[subject_embedding, object_embedding]",
  "predicate_embedding": "...",
  "qualifier_embedding": "...",
  "orientation": "same"
}
```

### 4.3 Memory qualification

For each memory write, compare:

```text
intent frame
outcome frame
```

and record one of:

```text
useful
violated
irrelevant
expanded_intent
```

---

## 5. Workstreams

### Workstream 1 — Formalize the grammar

Produce a small typed grammar for:

```text
speech_act
subject
predicate
object
qualifier
polarity
parent_intent
abstention
```

The grammar should cover:

- declarative statements,
- imperatives,
- negation,
- conditionals,
- temporal qualifiers,
- version qualifiers,
- multi-requirement intents,
- implicit objects,
- underspecified fragments.

Deliverable:

```text
docs/ICM_predicate_intent_grammar.md
```

### Workstream 2 — Build a rule-based weak labeler

Implement a deterministic parser that applies the grammar to ICM corpora and
FewRel examples.

Output:

```text
intent frame
confidence
abstention flag
```

The labeler does not need to be perfect. It needs to be auditable and
consistent enough to train the neural extractor.

Deliverable:

```text
icm/predicate_intent_labeler.py
```

### Workstream 3 — Add structured heads to the Discernment encoder

Fine-tune the BERT-style Discernment encoder with:

```text
speech_act head
subject span head
predicate span head
object span head
qualifier head
polarity head
parent-link head
intent embedding head
```

Training loss:

```text
L = L_speech_act
  + L_span
  + L_predicate
  + L_object
  + L_qualifier
  + L_polarity
  + L_parent_link
  + L_intent_contrastive
  + L_abstention
```

Deliverable:

```text
stage5_encoder/predicate_intent_head.py
stage5_encoder/train_predicate_intent.py
```

### Workstream 4 — Build the training corpus

Combine:

1. FewRel
   - subject / relation / object labels.

2. ICM `icapture` records
   - declared intents, outcomes, memory units, epochs.

3. ICM git-counterfactual pairs
   - before/after versions of the same artifact.

4. CodeTraceBench issues
   - multi-requirement intents.

5. Rule-generated synthetic statements
   - cheap, controllable, auditable.

Deliverable:

```text
data/_scratch/icm/predicate_intent_corpus.jsonl
```

### Workstream 5 — Integrate with the stage-6 geometry index

Use the extracted intent frame to produce:

```text
subject/object 2-frame
predicate embedding
qualifier embedding
orientation
```

Each memory carries two keys:

```text
content key:
    raw subject/object embedding

reasoning key:
    predicate frame
    semantic role types
    orientation
    reasoning trajectory
```

Then route retrieval by geometry:

```text
same orientation       -> raw-concat ANN
swapped orientation    -> raw-concat on reversed frame
orientation-invariant  -> projection ANN
explicit wedge target  -> signed-wedge ANN
unknown orientation    -> max(forward raw, backward raw)
known predicate        -> SPO / POS / OSP graph scan
```

The reasoning key is for analogical recall, not exact recall:

```text
exact recall:
    retrieve the same content

analogical recall:
    retrieve a memory with the same reasoning shape
```

Example target pair:

```text
"pouring a drink"
"Pour out the oil in the pan."
```

Both should map to:

```text
pour(liquid, out of a container)
```

even though the surface objects differ.

Deliverable:

```text
icm/predicate_geometry_index.py
```

### Workstream 6 — Wire into ICM’s two-stage retriever

Replace or augment the dense stage-1 candidate generator with the
predicate-geometry index.

Keep ICM’s existing deterministic stages:

```text
stage 1: predicate-geometry candidate generation
stage 2: entity/type filter
stage 3: epoch / ancestry re-rank
```

Do not blend scores.

Deliverable:

```text
icm/intent_query.py [modify]
```

### Workstream 7 — Add memory qualification

At outcome time, compare:

```text
intent frame
outcome frame
```

Record:

```text
useful
violated
irrelevant
expanded_intent
```

Deliverable:

```text
icm/intent_outcome_qualifier.py
```

### Workstream 8 — Evaluate

Evaluate on:

1. ICM git-counterfactual pairs.
2. ICM user-originated before/after phrasings.
3. CodeTraceBench real issues.
4. FewRel relation retrieval.
5. Cross-domain FewRel splits.

Metrics:

- intent extraction F1,
- counterfactual direction accuracy,
- candidate recall@10,
- end-to-end two-stage accuracy,
- FewRel relation top1,
- abstention calibration,
- latency,
- audit-record size.

### Workstream 9 — Analogical reasoning retrieval benchmark

Build a benchmark that separates content similarity from reasoning-shape
similarity.

Positive pairs:

```text
same predicate, different objects
```

Example:

```text
"pouring a drink"
"Pour out the oil in the pan."
```

Both share:

```text
predicate:   pour
object type: liquid
source role: container
direction:   outward
reasoning:    transfer a liquid out of a container
```

Negative pairs:

```text
same object, different predicate
```

Example:

```text
"oil in the pan"
"heat the oil in the pan"
```

The object is the same, but the reasoning shape is different.

Arms:

| Arm | Retrieval key |
|---|---|
| 1 | raw content embedding |
| 2 | Plücker / wedge only |
| 3 | predicate frame only |
| 4 | predicate frame + orientation |
| 5 | predicate frame + warp trajectory |
| 6 | predicate frame + orientation + warp trajectory |

Metric:

```text
analogical top1
analogical recall@5
content-confusion rate
```

Deliverable:

```text
data/_scratch/icm/analogical_reasoning_pairs.jsonl
```

The key question is:

> Does the reasoning key retrieve the positive analogical pair more often
> than the raw content key?

---

## 6. Acceptance criteria

### Minimum useful result

```text
ICM before/after direction accuracy  > 0.963
FewRel relation top1                > 0.683
candidate recall@10                 near current dense baseline
```

### Strong result

```text
ICM counterfactual direction accuracy > 0.963
FewRel relation top1                 > 0.707
cross-domain mean top1               > 0.606
analogical top1                      > raw-content baseline
abstention calibrated on ambiguous fragments
```

### Production-ready result

```text
intent extraction F1 high enough to replace the judge-built key
memory qualification mechanically reproducible
audit trail exact
query latency acceptable for the subconscious layer
```

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Imperative verb-only utterances are ambiguous | Use verb-frame defaults; allow implicit-object abstention |
| Multi-requirement intents are hard | Emit a list of frames, not one summary |
| Hierarchy is too complex | Use parent links only, not free-form plans |
| Model over-extracts intent | Allow `underspecified` and `other` residues |
| Geometry becomes a hidden latent | Require explicit geometry emission at write and query time |
| Neural extractor replaces determinism | Keep deterministic filter and re-rank unchanged |
| Analogical retrieval overgeneralizes | Use negative pairs with the same object but a different predicate |

---

## 8. Next concrete step

1. Write `docs/ICM_predicate_intent_grammar.md`.
2. Implement the rule-based weak labeler.
3. Run it over the existing ICM corpora.
4. Compare extracted intent frames with the current judge-built keys.
5. Only then fine-tune the Discernment encoder.

The first milestone is not neural. It is:

> A deterministic predicate-first intent labeler that reproduces the ICM
> before/after direction result on a held-out split.
