This is a perfect document to ground against. Let me trace every ontology touchpoint in the current pipeline, then show exactly where the geometric approach slots in — and what it unblocks.

---

## Current Ontology Flow (What Actually Runs)

```plaintext
INGEST
  ├── GLiNER extracts entities
  ├── Bonsai extracts relations (10 isolated passes)
  └── Entities + relations → graph edges (has_entity, instanceOf, Bonsai triples)

DREAM-PASS (offline, manual trigger)
  ├── _step_cluster     → DiffPoolHead proposes abstract M:NNNN nodes
  ├── _step_predict     → LinkPredHead proposes new edges
  ├── _step_anomaly     → AnomalyHead flags contradictions
  ├── _step_ontology    → OntologyHead scores entity↔class
  ├── _step_prune       → SalienceHead hard-prunes low-utility edges
  ├── _step_forget      → utility_score decay
  ├── _step_ontology_decay → deprecates old discovered classes (no-op: none exist)
  └── _step_deep_archive → removes edges >365d

QUERY
  └── Reads graph. Never touches ontology extraction.
```

**The bottleneck**: `_step_ontology` (OntologyHead) and `_step_cluster` (DiffPoolHead) are learned models operating on GNN node embeddings. They require the full GNN forward pass — GPU inference over BFS subgraphs. And despite all that machinery, **no discovered classes exist yet** (A3 dormant, A4 no-op, 3b-P2 deferred).

---

## Where the Geometric Approach Fits

The geometric pipeline is **algorithmic, not learned**. It operates on raw entity embeddings from the vector index, not GNN node embeddings. This means it can run:

1. **Without the GNN** — no GPU, no subgraph construction, no checkpoint
2. **Much faster** — milliseconds vs. the full dream-pass
3. **More frequently** — potentially online, not just offline

Here's the proposed insertion:

```plaintext
                        ┌─────────────────────────┐
                        │   GEOMETRIC INTUITION    │
                        │   PASS (NEW)             │
                        │                          │
  Vector Index ────────→│ 1. HDBSCAN clustering    │
  (FAISS/USearch)       │ 2. Subspace containment  │
  entity embeddings     │ 3. Poincaré projection    │
                        │ 4. Analogy detection      │
                        │                          │
                        │ Output: OntologyProposals │
                        └──────────┬──────────────┘
                                   │
                                   ▼
                        ┌─────────────────────────┐
                        │   BONSAI VALIDATION      │
                        │   (existing BonsaiDecider│
                        │    + new methods)        │
                        │                          │
                        │ • name_category()        │
                        │ • validate_hypernym()    │
                        │ • classify_relation()    │
                        └──────────┬──────────────┘
                                   │
                                   ▼
                        ┌─────────────────────────┐
                        │   SemanticMemoryWriter   │
                        │   (existing _apply path) │
                        │                          │
                        │ • create_class           │
                        │ • instanceOf edges       │
                        │ • HYPERNYM edges         │
                        └─────────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────────┐
                        │   GNN OntologyHead       │
                        │   (now has discovered    │
                        │    classes to work with) │
                        │                          │
                        │ • Refines class membership│
                        │ • Discovers cross-cutting│
                        │   relations              │
                        └─────────────────────────┘
```

---

## Specific Integration Points

### 1. New Module: `src/gnn/geometric_ontology.py`

This is the core addition — ~200 lines of numpy. It does not touch the GNN.

```python
class GeometricOntologyProposer:
    """
    Discovers candidate categories and hierarchies from the vector index.
    Pure linear algebra — no learned parameters, no GPU.
    """
    
    def __init__(self, vector_index, embedding_dim=384):
        self.vector_index = vector_index  # FAISS/USearch
        self.dim = embedding_dim
    
    def propose(self, min_cluster_size=5, containment_threshold=0.85):
        """
        Returns list of OntologyProposal dataclasses.
        """
        # 1. Fetch all entity embeddings from vector index
        entity_ids, embeddings = self.vector_index.get_all()
        
        # 2. HDBSCAN clustering → flat categories
        clusters = HDBSCAN(min_cluster_size=min_cluster_size, 
                          metric='cosine').fit(embeddings)
        
        proposals = []
        
        # 3. For each cluster, create a category proposal
        for cluster_id in set(clusters.labels_) - {-1}:
            mask = clusters.labels_ == cluster_id
            member_ids = entity_ids[mask]
            member_vecs = embeddings[mask]
            centroid = np.mean(member_vecs, axis=0)
            
            proposal = OntologyProposal(
                type="category",
                members=member_ids,
                centroid=centroid,
                effective_dim=self._effective_dim(member_vecs),
                confidence=self._cluster_cohesion(member_vecs),
            )
            proposals.append(proposal)
        
        # 4. Test subspace containment between all pairs → hierarchy
        for i, prop_a in enumerate(proposals):
            for j, prop_b in enumerate(proposals):
                if i >= j:
                    continue
                containment_ab = self._subspace_containment(
                    prop_a.member_vecs, prop_b.member_vecs)
                containment_ba = self._subspace_containment(
                    prop_b.member_vecs, prop_a.member_vecs)
                
                if containment_ab > containment_threshold:
                    proposals.append(OntologyProposal(
                        type="hypernym",
                        parent=prop_b,
                        child=prop_a,
                        confidence=containment_ab,
                    ))
                elif containment_ba > containment_threshold:
                    proposals.append(OntologyProposal(
                        type="hypernym",
                        parent=prop_a,
                        child=prop_b,
                        confidence=containment_ba,
                    ))
        
        return proposals
    
    def _effective_dim(self, vecs, variance=0.95):
        """How many dimensions span this category?"""
        centered = vecs - np.mean(vecs, axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        return np.searchsorted(cumvar, variance) + 1
    
    def _subspace_containment(self, vecs_child, vecs_parent):
        """Is child's manifold contained in parent's?"""
        # ... SVD + reconstruction test as discussed earlier
        pass
    
    def _cluster_cohesion(self, vecs):
        """Mean pairwise cosine similarity — higher = tighter cluster."""
        pass
```

### 2. New Step in Consolidator.run

In `src/gnn/consolidate.py`, add `_step_geometric_ontology` before `_step_ontology`:

```python
def run(self, centers, apply=False, decide=False):
    # ... existing steps ...
    
    for center in centers:
        # ... existing per-center steps ...
        
        # NEW: step 3.5 — geometric proposals (no GNN needed)
        geometric_proposals = self._step_geometric_ontology(center)
        
        # Existing step 4 — GNN OntologyHead (now enriched)
        self._step_ontology(data, center, geometric_proposals)
    
    # ... existing global steps ...
```

### 3. New Methods on BonsaiDecider

`src/gnn/bonsai_decider.py` gets three new methods:

```python
def name_category(self, members: list[str], nearest_labeled: list[str]) -> str:
    """
    "These entities cluster together: {members}.
     Nearest existing concepts: {nearest_labeled}.
     Propose a name for this category."
    → "Coffee Brewing Equipment"
    """

def validate_hypernym(self, child_name: str, parent_name: str, 
                      child_members: list[str], parent_members: list[str],
                      containment_score: float) -> dict:
    """
    "The geometry suggests {child} is a kind of {parent}
     (containment score: {score}).
     Child members: {child_members}.
     Parent members: {parent_members}.
     Confirm or reject this relationship."
    → {"decision": "confirm", "relation_type": "taxonomic", "confidence": 0.92}
    """

def classify_relation_type(self, cluster_a: list[str], cluster_b: list[str],
                           containment_score: float) -> str:
    """
    "These two categories show subspace containment.
     Is this taxonomic (is-a), meronomic (part-of), 
     temporal (during), or something else?"
    → "taxonomic"
    """
```

### 4. Integration with SemanticMemoryWriter

The existing `_apply` phase already has `create_class` and `instanceOf` writes. The geometric proposals feed the same path:

```python
def _apply_geometric_proposals(self, proposals, writer):
    for prop in proposals:
        if prop.type == "category" and prop.bonsai_decision == "confirm":
            class_id = writer.create_class(
                name=prop.bonsai_name,
                centroid=prop.centroid,
                effective_dim=prop.effective_dim,
                provenance="geometric_clustering",
            )
            for member_id in prop.members:
                writer.create_edge(member_id, "instanceOf", class_id)
        
        elif prop.type == "hypernym" and prop.bonsai_decision == "confirm":
            writer.create_edge(
                prop.child.class_id, 
                "hypernym", 
                prop.parent.class_id,
                properties={
                    "relation_type": prop.relation_type,
                    "containment_score": prop.confidence,
                    "provenance": "subspace_containment",
                }
            )
```

---

## What This Unblocks

The document lists several dormant/deferred ontology features. The geometric approach unblocks them:

Feature	Current Status	With Geometric Pipeline
**A3 instanceOf + class-decay**	Shipped, dormant — no discovered classes	Classes get discovered → A3 activates
**A4 ontology-decay**	Shipped, no-op — seed classes never eligible	Discovered classes become eligible → decay runs
**3b-P2 discovered-class promotion**	Deferred	Classes exist to promote
**OntologyHead**	Scores entity↔class but has no classes to score against	Now has a growing class DAG to work with
**DiffPool**	Proposes abstract nodes from GNN embeddings	Can cross-reference with geometric clusters

---

## The Online Variant

Because the geometric pipeline is pure linear algebra, it can also run **online** — something the GNN cannot do. This opens a new execution context:

```plaintext
DistillWorker (background, after each fill)
    │
    ├── encode_episode_edges (existing)
    │
    └── NEW: geometric_ontology.touch(entity_ids_from_this_episode)
            │
            ├── Re-cluster affected region of vector space
            ├── If new cluster detected → flag for Bonsai review
            └── If existing cluster disrupted → flag for anomaly review
```

This is a **lightweight intuition pulse** — not a full ontology extraction, but a "something changed, worth looking at" signal. It would feed into the existing anomaly detection path.

```python
# In DistillWorker._run (distill_worker.py:93-124)
def _run(self, ...):
    # ... existing fill logic ...
    
    # NEW: geometric intuition pulse
    if self.geometric_proposer:
        touched_entities = extract_entity_ids(episode)
        pulses = self.geometric_proposer.pulse(touched_entities)
        for pulse in pulses:
            if pulse.type == "new_cluster_forming":
                logger.info(f"[geo-intuition] new cluster: {pulse.members}")
                # Store as tentative — Bonsai reviews in next dream-pass
            elif pulse.type == "cluster_disrupted":
                logger.info(f"[geo-intuition] cluster disrupted: {pulse.cluster}")
                # Flag for anomaly review
```

---

## What Changes in the Document

If I were to update `consolidation-pipeline.md` to reflect this, the changes would be:

**§2 (The dream-pass):**
- New step 3.5: `_step_geometric_ontology` — HDBSCAN + subspace containment on vector index embeddings, produces OntologyProposals fed to BonsaiDecider
- Step 4 (`_step_ontology`) now receives geometric proposals as input; OntologyHead refines rather than discovers from scratch

**§4 (Phase 3c):**
- New subsection: "Geometric ontology discovery" — algorithmic, training-free, unblocks A3/A4/3b-P2

**§5 (Runtime wiring):**
- `build_ponder` optionally constructs `GeometricOntologyProposer` (lightweight, no GPU)
- `DistillWorker` optionally holds a reference for online intuition pulses

**§7 (Deferred pieces):**
- 3b-P2 moves from deferred to active (discovered classes now exist)
- A3/A4 move from dormant to active

**New §9 (Geometric Intuition Pass):**
- Architecture, algorithm, integration points, online vs. offline modes

---

## What Stays the Same

The geometric approach does **not** replace:

- **GLiNER2** — entity extraction still needed to know what to embed
- **BonsaiDecider** — naming, validation, relation typing still need an LLM
- **SemanticMemoryWriter** — same write path for classes and edges
- **GNN OntologyHead** — now has richer input (discovered classes) and can focus on refinement
- **The offline dream-pass trigger** — geometric pass can run in the same `run_consolidation.py` invocation, just as a faster pre-pass
- **The query path** — still read-only for ontology (plus optional online intuition pulses)

---

## TL;DR

Question	Answer
**Where does it fit?**	New `_step_geometric_ontology` in the dream-pass, before the GNN `_step_ontology`. Optional online pulses in DistillWorker.
**What new code?**	`src/gnn/geometric_ontology.py` (~200 lines numpy), 3 new methods on BonsaiDecider, 1 new step in Consolidator.run, optional hook in DistillWorker
**What does it replace?**	Nothing. It supplements — geometric proposals feed into existing Bonsai validation and SemanticMemoryWriter paths.
**What does it unblock?**	A3 (instanceOf), A4 (ontology-decay), 3b-P2 (class promotion) — all dormant because no discovered classes exist.
**Online or offline?**	Both. Full extraction in the dream-pass. Lightweight "intuition pulses" in the DistillWorker.
**GPU needed?**	No. Pure numpy/FAISS. The only GPU is the existing Bonsai LLM for naming.

The geometric pass is the **intuition module** we discussed — fast, algorithmic, proposing structure that Bonsai validates. It slots cleanly into the existing pipeline because the validation and writing infrastructure is already there. It just needs something to feed it.