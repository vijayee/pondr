# Ponder Engine: Complete Phased Implementation Plan

**Architecture Version 2.0 — July 2026**

---

## Phase Map

```plaintext
Phase 1: Foundation ────── Encoding, Storage, Retrieval, Training Data
    │
Phase 2: Subconscious ──── JEPA-Gated SSM Backbone, Retrieval Gate, 
    │                       Working Memory, SSM Chunking, Presentation Gate
    │
Phase 3: Consolidation ─── GNN Consolidator, Forgetting System, 
    │                       Reconsolidation Counting, Ontology Decay
    │
Phase 4: Metacognition ──── Uncertainty Detector, Aspirational Model,
    │                       Self-Model, Common Sense Resolver, EXPAND Mechanism
    │
Phase 5: Evaluation ─────── ConvoMem, EverMemBench, CloneMem Benchmarks
    │
Phase 6: Procedural ─────── Process Observer, Process Executor,
    │                       Delegation Ladder, Graph-Native Optimizer,
    │                       Failure Procedures
    │
Phase 7: Curiosity ──────── Disturbance Detector, Intuition Module,
    │                       Curiosity Cascade, Self-Generated Training
    │
Phase 8: Ecosystem ──────── Process Marketplace, Domain Sharing,
                            Federated Improvement
```

---

## Phase 1: Foundation

### Phase 1a: Encoding Pipeline

**Goal:** Build a working pipeline that consumes raw conversation text and produces structured episodes stored in WaveDB.

**Duration:** 2-3 days

**What "done" looks like:** A Python script that takes a conversation file, runs GLiNER2 + GLiNER-Decoder + Bonsai extraction, and stores the result as an Episode in WaveDB. A test suite that verifies extraction quality against known examples.

**Key deliverables:**
- `HippocampalStore` — WaveDB wrapper with encode/decode
- `GLiNERExtractor` — GLiNER2 (stable) + GLiNER-Decoder (open discovery)
- `BonsaiRelationExtractor` — structured relation extraction
- `HippocampalEncoder` — orchestrator: extract → create episode → store
- Seed ontology (376 classes, 165 properties, multi-parent DAG)
- 20 hand-crafted sample conversations with expected extraction labels
- Unit tests for all components
- Extraction quality: entity recall >70%, topic recall >70%, tone recall >70%

**Episode model includes downstream fields with safe defaults:**
- `retrieval_count`, `ltp_phase`, `consolidation_window_start`
- `utility_decay_rate`, `retrieval_timestamps`, `saturation_flags`
- `summary_embedding` (for vector search)

**Configuration includes Phase 2-4 placeholders:**
- SSM state dimensions, JEPA backbone model
- GNN hidden dimensions, gate configurations
- Forgetting system parameters (saturation threshold, boost half-life, min decay rate)

**Developmental stage:** All extraction components at INFANT.

---

### Phase 1b: Storage & Retrieval

**Goal:** Populate WaveDB with real conversation corpora, build the graph traversal engine, implement the query planner, and deliver working Mode A generation.

**Duration:** 5-7 days

**Prerequisite:** Phase 1a complete.

**Key deliverables:**

**Corpus ingestion at scale:**
- 1,000+ conversations from DialogSum encoded in WaveDB
- 500+ conversations from SAMSum encoded
- Ingestion report with extraction quality metrics at scale
- Resume support for interrupted processing

**Graph traversal engine:**
- Entity queries (union and intersection modes)
- Topic queries (union)
- Tone queries (union)
- Temporal chain queries (forward and backward via `follows` edges)
- Temporal filter queries (today, this_week, last_week, this_month)
- Scoring: entity matches × 10 + topic matches × 5 + tone matches × 3 + recency × 0.1

**Query planner:**
- Bonsai-based NL → structured query conversion
- Correct entity_mode selection (union vs. intersection)
- Temporal intent detection (temporal_after, temporal_before, temporal_filter)

**Vector search:**
- FAISS index over episode summary embeddings
- Semantic fallback when graph traversal returns <3 results

**Mode A generator:**
- Context window adapter for any LLM API
- Structured context format: entities, topics, tones explicitly labeled
- Token-counted context building with hard cutoff

**Retrieval end-state API:**
- `retrieve()` accepts `end_state` parameter: "direct", "format", "synthesize", "extract"
- `consumer` parameter: "openai_chat", "anthropic", "generic_llm", "code_agent", "tool", "human", "downstream"
- JEPA provides default when not specified; overrides become training signal
- `ContextFormatter` produces output in the format appropriate for each consumer

**Context building strategy (Phase 1b):**
- Fixed: top 5 episodes, full text, truncate at token limit
- No SSM chunking yet (Phase 2c)
- No semantic abstractions yet (Phase 3)

**Oracle labeling infrastructure:**
- Prompts for all five GNN tasks (salience, clustering, link prediction, anomaly detection, ontology refinement)
- Subgraph extraction from populated graph
- Ready for Phase 1d execution

**What Phase 1b delivers to the user:**
A database you can talk to. The system retrieves relevant episodes, builds context, and feeds it to an LLM. The LLM responds as if it remembers everything. But there's no subconscious routing, no consolidation, no uncertainty detection, no procedural memory.

**Known limitations at Phase 1b:**
- Bonsai plans blind (no conversation context for pronoun resolution)
- No multi-domain query support
- Fixed context strategy (always top 5, always full text)
- Crude scoring (heuristic weights, not learned)
- No document-level retrieval (documents returned as individual sections)
- No cross-document deduplication

**Developmental stage:** Retrieval components at INFANT.

---

### Phase 1c: Retrieval Refinements

**Goal:** Address the known scaling limitations of Phase 1b before moving to learned components.

**Duration:** 2-3 days

**Note:** Can run in parallel with Phase 1d. Training data generation does not depend on retrieval refinements.

**Key deliverables:**

**Document-level retrieval:**
- `Document` nodes aggregate their sections in the graph
- Retrieval can return documents or sections depending on query specificity
- Context building presents "Document X, sections 3, 7, 12 are relevant"

**Temporal indexing:**
- Timestamp range queries for long chains ("what happened in June 2025?")
- `follows` edges for short-range ("what happened next?")
- Both mechanisms coexist

**Entity salience tracking:**
- Track entity mention frequency
- Prioritize high-salience entities in retrieval scoring
- Foundation for ontology decay (Phase 3)

**Conversation context for Bonsai:**
- Pass last 2-3 conversation turns to Bonsai for pronoun resolution
- Improves entity extraction accuracy on ambiguous references

---

### Phase 1d: Training Data Generation

**Goal:** Generate labeled training data for all downstream learned components.

**Duration:** 3-5 days (mostly Oracle API time)

**Note:** Can run in parallel with Phase 1c. Does not depend on retrieval refinements.

**Key deliverables:**

**GNN training dataset:**
- 4,000+ labeled subgraphs from processed corpora
- Labels for all five tasks: salience, clustering, link prediction, anomaly detection, ontology refinement
- Oracle-generated with validation

**Bonsai query planning pairs:**
- 5,000-10,000 (prompt, structured_query) pairs
- Covers entity_mode edge cases, temporal queries, multi-entity queries

**Bonsai relation extraction pairs:**
- 2,000+ (conversation_text, relations) pairs

**JEPA routing pairs:**
- 5,000+ (prompt, optimal_route) pairs
- Covers domain routing, pathway selection, model sizing

**Gate training examples:**
- 50,000 each for Uncertainty Detector, Aspirational Model, Self-Model, and Common Sense Resolver gates
- Oracle-labeled "is this a gap?", "should I commit?", "should I say I don't know?", "is this ambiguous?"

**Synthetic code-aware examples:**
- Generated training data that includes code entity types
- Ensures models allocate representational capacity for code structure
- Even before real code is parsed

**Total Oracle cost:** ~$20

---

## Phase 2: Subconscious Layer

### Phase 2a: JEPA-Gated SSM Backbone

**Goal:** Train the shared SSM+JEPA backbone that all cognitive instances will use.

**Duration:** 5-7 days

**Key deliverables:**

**Shared backbone training:**
- Mamba SSM (~370M params) + JEPA Predictor (~110M params)
- Total: ~480M params, ~0.90 GB
- Trained on Oracle-generated cognitive state sequences (10M+ examples)
- Self-supervised tasks: next-state prediction, outcome prediction, interpretation prediction
- JEPA contrastive loss: minimize distance to positive target, maximize distance to negative samples

**LoRA adapter framework:**
- Instance-specific low-rank adaptation matrices (~50K params each)
- Modulate SSM dynamics without changing base weights
- Different ranks for different temporal signatures:
  - Disturbance Detector: rank 4 (fast, bursty)
  - Intuition Module: rank 8 (slow, accumulative)
  - Common Sense Resolver: rank 6 (flexible)
  - Retrieval Gate: rank 4 (fast routing)
  - Working Memory: rank 8 (rich state)

**Decomposed gate framework:**
- Value head (~400K params): estimates reward
- Cost head (~500K params): estimates effort
- Decision head (~500K params): combines into excite/inhibit
- Context encoder (~100K params): scalar features → vector
- Learnable base threshold with context-dependent modulation
- Total per gate: ~1.5M params

**Training hardware:** A100 80GB, Lambda Labs, 48 hours, ~$62

**Developmental stage:** Backbone at INFANT.

---

### Phase 2b: Retrieval Gate Instance

**Goal:** Train the first JGS instance — the subconscious router.

**Duration:** 3-4 days

**Key deliverables:**

**Retrieval Gate:**
- Predicts: domain(s), pathway, meta-skills, model size, deliberation need
- Pathways: ssm_direct | graph_retrieve | process_exec | tool_plan | conscious_deliberation
- Trained on Oracle-generated routing pairs (5,000 examples)
- Learns from outcomes: successful routes reinforced, delegation surprises penalized, overkill penalized

**Domain routing:**
- Routes to correct domain graph(s) based on prompt + SSM state
- Multi-domain queries: "Compare database performance with robotics actuator torque" → routes to both
- Cross-domain traversal via cross-graph edges

**Pathway selection:**
- "What is X?" → ssm_direct (no retrieval needed)
- "What did Alice say?" → graph_retrieve
- "Review this PR" → process_exec (if process exists)
- "Design a new X" → conscious_deliberation

**Model sizing:**
- 1B: factual recall
- 3B: basic synthesis
- 8B: reasoning, process execution
- 70B: creative design, security analysis
- 175B: novel research

**Training hardware:** RTX 4090, Vast.ai spot, 24 hours, ~$6.50

**Developmental stage:** Retrieval Gate at INFANT → CHILD (when routing accuracy >75% vs Oracle).

---

### Phase 2c: Working Memory & Presentation

**Goal:** Deploy the SSM instance that maintains continuous awareness, plus SSM chunking and the JEPA presentation gate.

**Duration:** 3-4 days

**Key deliverables:**

**Working Memory:**
- Continuous hidden state — 8,192 floats, ~16 KB
- State evolves with each input
- Retrieved memories injected as embeddings (not text)
- Old information decays gracefully (exponential moving average)
- No fixed token limit — state dimension is constant, information capacity is dynamic

**SSM chunking for context building:**
- Retrieved episodes divided into chunks
- Primary chunk: full text (most relevant episodes)
- Secondary chunks: compressed into SSM state
- Generation model receives: primary chunk + SSM state summary
- EXPAND capability for any compressed chunk

**JEPA presentation gate:**
- Extends Retrieval Gate with presentation strategy
- Predicts: chunk count, chunk size, primary vs. compressed assignment
- Trained from outcomes: EXPAND frequency, unused primary episodes, user satisfaction
- "What was the Python async throughput?" → direct, no chunking
- "What have we discussed about performance?" → chunked, top 3 primary, rest compressed

**Mode A generation with chunking:**
- Context builder uses presentation plan from JEPA
- Generation model receives structured primary context + SSM state summary
- EXPAND triggers load full text of compressed chunks on demand

**Prompt compression before query planning:**
- Very long prompts compressed by SSM before Bonsai sees them
- Bonsai plans from compressed state, not raw text
- Prevents context window overflow in query planning

**Developmental stage:** Working Memory at INFANT.

---

## Phase 3: Consolidation & Forgetting

### Phase 3a: GNN Consolidator

**Goal:** Deploy the graph neural network for dream-state memory processing.

**Duration:** 5-7 days

**Key deliverables:**

**Static GNN with 5 task-specific heads:**
- Salience scoring (GAT): learned structural importance
- Subgraph summarization (DiffPool): collapse related episodes into semantic memories
- Link prediction (GAE/SEAL): discover implicit edges
- Anomaly detection: flag structural violations
- Ontology refinement: predict missing subClassOf edges

**Training:**
- Oracle-generated salience labels + link prediction examples
- GNN backbone: GAT pre-trained on OGB benchmarks (general graph knowledge)
- Task heads trained on Oracle-labeled memory graphs
- Training hardware: RTX 4090, Vast.ai spot, 48 hours, ~$13

**Consolidation loop:**
- Nightly dream-state pass
- GNN scores all nodes/edges → detects clusters → abstracts semantic memories
- Predicts missing edges → auto-accepts high-confidence, proposes medium for Bonsai
- Detects anomalies → wakes Bonsai for verification
- Refines ontology → suggests new subClassOf edges
- Prunes low-salience edges → archives, never deletes

**Semantic memory storage:**
- Abstracted memories stored in HBTrie
- Linked to source episodes via `abstracts` edges
- Source episodes marked as "abstracted" (still retrievable, not in default queries)

**Cross-document deduplication:**
- GNN detects similar document sections across different documents
- Creates cross-document semantic memories
- Reduces context bloat from redundant ingested content

**Note on temporal continuity:** The initial GNN is a stateless function. After collecting data on failure modes (cluster flapping, prediction miscalibration, false positive anomalies, ontology oscillation), SSM-augmented instances with temporal memory of past consolidation decisions will be added. This prevents premature optimization — temporal continuity is built only for failure modes that actually occur.

**Developmental stage:** GNN at INFANT.

---

### Phase 3b: Forgetting System

**Goal:** Deploy the complete forgetting system with retrieval-weighted persistence.

**Duration:** 3-4 days

**Key deliverables:**

**Retrieval-weighted persistence:**
- Every retrieval reduces edge's decay rate
- Diminishing returns: 20th retrieval provides far less boost than 1st
- Absolute floor: 0.1% per day minimum decay (nothing is immortal)

**Saturation detection:**
- >5 retrievals in 24 hours → stop boosting, slight decay increase
- Breaks the frustration loop: user keeps asking because answer isn't sticking

**LLM-mediated signals:**
- `[IMPORTANT]` → stronger persistence boost
- `[ROUTINE]` → normal diminishing returns
- `[FRUSTRATION]` → no boost, slight decay increase
- `[CORRECTION]` → triggers reconsolidation of old fact

**Boost decay:**
- Retrieval boost has ~7-day half-life
- Old retrievals matter less than recent ones
- If user stops asking, edge returns to baseline decay

**Reconsolidation counting:**
- Track `reconsolidation_count` and `consolidation_window_start`
- 3 retrievals across 15+ days → late-phase LTP
- Late-phase LTP: additional 70% reduction in decay rate

**Active forgetting:**
- User-triggered: "Forget what I said about Postgres"
- Edge state → "deprecated", validity_end → now()
- Still retrievable via historical queries

**Reconsolidation on contradiction:**
- New fact contradicts old → old superseded, new current
- `supersedes` edge links versions
- Both preserved, default queries return current only

**Ontology decay:**
- Categories not seen in 30+ days → deprecated
- Entities reassigned to parent class
- Prevents ontology bloat from one-off entities

**Entity salience scoring:**
- Entities scored by mention frequency, recency, and structural position
- Low-salience entities deprioritized in retrieval
- Foundation for entity archiving

**Developmental stage:** Forgetting system at INFANT.

---

## Phase 4: Metacognition

### Phase 4a: Uncertainty Detector

**Goal:** Deploy the instance that knows when the system doesn't know.

**Duration:** 3-4 days

**Key deliverables:**

**Uncertainty Detector (JGS instance):**
- Flags three conditions:
  1. Routing uncertainty (Retrieval Gate confidence < threshold)
  2. Novel entities (retrieved context contains entities not in ontology)
  3. Unresolved contradictions (multiple sources disagree without resolution)
- Gate trained on Oracle-labeled "is this a gap?" pairs (50,000 examples)
- Thresholds learned, not hardcoded

**EXPAND mechanism:**
- Uncertainty Detector → EXPAND → HBTrie load → SSM injection
- Three levels of response:
  1. Missing detail → EXPAND from HBTrie
  2. Missing memory → ADMIT GAP with specific reason
  3. Missing capability → TOOL USE PLAN → delegate to larger model

**Empty result handling:**
- Graph traversal returns nothing → Uncertainty Detector classifies why
- Unknown entity: "I don't have any information about X"
- Unknown topic: "I don't have any conversations about Y"
- Known but not in combination: "I know about those separately, but not together"

**Training hardware:** RTX 4090, Vast.ai spot, 10 hours, ~$2.70

**Developmental stage:** Uncertainty Detector at INFANT → CHILD (when precision >80% vs Oracle).

---

### Phase 4b: Aspirational Model

**Goal:** Deploy the three-temporal-mode aspiration system.

**Duration:** 3-4 days

**Key deliverables:**

**Present mode (encoding gate):**
- Modulates encoding strength based on aspirational match
- "I'm curious about this → encode strongly"
- Implemented through JEPA salience scoring (already in Phase 2)

**Future mode (prospective memory):**
- Sets triggers that fire when conditions are met
- "Alert me when you encounter information about X"
- Trigger storage, condition matching, pre-retrieval check
- When trigger fires, goal state injected into SSM state

**Past mode (retrieval-weighted persistence):**
- Already implemented in Phase 3b forgetting system
- Aspirational Model provides the "why" — this matters because it aligns with goals

**Gate training:**
- Oracle-labeled "should I commit?" pairs (50,000 examples)
- Training hardware: RTX 4090, Vast.ai spot, 10 hours, ~$2.70

**Developmental stage:** Aspirational Model at INFANT.

---

### Phase 4c: Self-Model

**Goal:** Deploy the instance that estimates knowledge boundaries.

**Duration:** 2-3 days

**Key deliverables:**

**Self-Model (JGS instance):**
- Learned knowledge boundary thresholds
- "Do I know enough to answer this?"
- Different thresholds for different domains:
  - Sparse but complete domain (3 episodes about niche topic) → confident
  - Dense but incomplete domain (100 episodes but missing specific fact) → not confident
- Gate trained on Oracle-labeled "should I say I don't know?" pairs (50,000 examples)

**Calibration:**
- "I don't know" precision >80% vs Oracle
- Learned from Oracle feedback: "You should have known this" vs. "You were right to admit uncertainty"

**Training hardware:** RTX 4090, Vast.ai spot, 10 hours, ~$2.70

**Developmental stage:** Self-Model at INFANT → CHILD (when precision >80%).

---

### Phase 4d: Common Sense Resolver

**Goal:** Deploy the instance that resolves ambiguity before committing to action.

**Duration:** 3-4 days

**Key deliverables:**

**Common Sense Resolver (JGS instance):**
- Detects ambiguity in inputs, queries, and retrieved results
- Generates candidate interpretations from the world model graph
- Gate evaluates each interpretation: value, cost, coherence with context
- Selects best interpretation or asks for clarification
- Verification loop: predicted outcome vs. actual → updates SSM state
- "I need to get to the bank" → resolves to financial institution or riverbank based on context

**Gate configuration:**
- ambiguity_magnitude > threshold → resolve before proceeding
- Threshold adapts to context coherence and historical frequency
- LoRA adapter: rank 6 (flexible dynamics)

**Training hardware:** RTX 4090, Vast.ai spot, 10 hours, ~$2.70

**Developmental stage:** Common Sense Resolver at INFANT → CHILD (when resolution accuracy >80% vs Oracle).

---

## Phase 5: Evaluation

### Phase 5: Benchmark Evaluation

**Goal:** Quantify performance against standard benchmarks.

**Duration:** 5-7 days

**Benchmarks:**

Benchmark	What It Tests	Target
**ConvoMem**	75K QA pairs, 6 evidence categories, 100 personas	>80% recall on single-session, >60% on cross-session
**EverMemBench**	Multi-party, 250-day span, 1M+ tokens	>40% on multi-hop (baseline is 26% with oracle evidence)
**CloneMem**	1-3 year temporal reasoning	>50% on temporal comparison

**Metrics tracked:**
- Encoding accuracy: >90% entity F1, >85% relation F1
- Retrieval recall: >80% single-session, >60% cross-session
- Retrieval precision: >85%
- Retrieval latency: <50ms for graph traversal + HBTrie load
- Routing accuracy: >90% correct domain routing
- Consolidation quality: >70% of GNN-predicted edges validated by Bonsai
- Forgetting accuracy: <5% of pruned edges later needed
- Context efficiency: graph-based context ≤50% size of full-history for equivalent recall
- Uncertainty calibration: >80% precision on "I don't know" decisions
- Delegation efficiency: >80% of queries handled by ≤8B model

**Developmental transitions during evaluation:**
- Extraction: INFANT → CHILD (F1 > 0.85)
- Retrieval Gate: INFANT → CHILD (routing accuracy > 0.75 vs Oracle)
- Self-Model: INFANT → CHILD ("I don't know" precision > 0.80)
- GNN: INFANT → CHILD (predicted edges accepted > 0.70)

---

## Phase 6: Procedural Memory

### Phase 6a: Process Observer & Executor

**Goal:** Deploy procedural memory — stored processes that can be executed and delegated.

**Duration:** 5-7 days

**Key deliverables:**

**Process graph ontology:**
- Process nodes with typed properties (name, description, triggers, domains, min_model_size)
- Step nodes with typed properties (instruction, tool, delegate_to, on_failure, depends_on)
- Execution nodes linking processes to episodes where they ran
- Version nodes linked by `supersedes` edges
- Failure procedure nodes as first-class process elements

**Process Observer:**
- Watches for repeated multi-step task patterns
- After 3 observations → proposes stored process via Bonsai
- Process stored as subgraph in WaveDB

**Process Executor:**
- Executes stored processes step by step
- Handles dependencies between steps
- Delegates steps to larger models when they exceed current capability
- Handles failure modes: delegate, skip, ask_user, abort

**Delegation Ladder:**
- 1B → 3B → 8B → 70B → 175B
- Each level handles what it can, delegates up what it can't
- Process defines delegation rules per step
- Retrieval Gate learns optimal routing over time

**Failure procedures:**
- Structured failure responses as first-class process elements
- Failure classification: missing_info | wrong_approach | tool_error | model_error | scope_too_large
- Recovery paths: delegate_to_larger_model | ask_user | reduce_scope | try_alternative | report_and_abort
- Failure handler monitoring: detect when failure handling itself is failing
- Silent scope drift detection, hallucinated completion detection, delegation loop detection

**Meta-processes:**
- Strategy templates for handling novelty
- Decision points, not fixed steps
- "Can this task be broken into independent sub-tasks? If yes, parallel decomposition."

---

### Phase 6b: Graph-Native Process Optimizer

**Goal:** Deploy the SkillOpt-inspired optimization loop for procedural memory.

**Duration:** 5-7 days

**Key deliverables:**

**Graph-native optimization loop:**
- Rollout evidence from memory graph (past executions, not fresh rollouts)
- Bonsai reflection: analyze failure minibatches, analyze success minibatches
- Structural edits: add_step, modify_delegation, add_failure_handler, modify_step, remove_step
- Merge: deduplicate, resolve conflicts, rank by support count
- Bounded edits: structural edit budget (not text token budget)
- Validation gate: held-out episodes from memory graph
- Versioning: old process → superseded, new process → current

**Intuition-driven edit budget:**
- JGS Intuition Module sizes the edit budget
- Strong evidence, low risk → more edits
- Thin evidence, critical process → fewer edits
- Learned from optimization outcomes
- Replaces SkillOpt's fixed cosine schedule

**Process metadata lifecycle:**
- Aggressive cleanup after each optimization cycle
- Execution nodes → summarized to aggregate stats → discarded
- Old edits → discarded (keep only last cycle)
- Old cycles → discarded
- Steady state: ~10-20 nodes per process regardless of age
- Grace periods: 7 days for execution nodes, 1 day for rejected edits

**Failure procedure optimization:**
- Failure handlers optimized as aggressively as success paths
- Disturbance Detector monitors failure-handling failures
- Silent scope drift, hallucinated completion, delegation loops → trigger optimization

---

## Phase 7: Curiosity & Self-Improvement

### Phase 7a: Disturbance Detector

**Goal:** Deploy the instance that detects when things need attention.

**Duration:** 4-5 days

**Key deliverables:**

**Disturbance Detector (JGS instance):**
- Monitors process executions for failure patterns
- Compares perception to world model prediction
- Prediction error > adaptive threshold → disturbance registered
- Disturbance carries: direction, pattern signature, location, novelty, error magnitude
- Novelty and prediction error as orthogonal signals

**Process absence detection:**
- Detects when user is doing something manually, repeatedly, with no stored process
- Disturbance type: PROCESS_ABSENCE
- Triggers Process Observer to propose new process

**Failure handling monitoring:**
- Detects when failure handling itself is failing
- Silent scope drift, hallucinated completion, delegation loops, premature completion
- These are the most dangerous failures because they're invisible

**Gate configuration:**
- error > adaptive_threshold → fire
- Threshold adapts to noise level
- LoRA adapter: rank 4 (fast, bursty dynamics)

**Training hardware:** RTX 4090, Vast.ai spot, 10 hours, ~$2.70

---

### Phase 7b: Intuition Module

**Goal:** Deploy the instance that evaluates whether action is worth taking.

**Duration:** 4-5 days

**Key deliverables:**

**Intuition Module (JGS instance):**
- Compressed outcome history: pattern_signature → expected_outcome_valence
- Maps disturbance signatures to expected valence
- Bypasses retrieval — accesses compressed mapping directly
- "Based on past experience, is pursuing this likely to be worth it?"

**Gate configuration:**
- predicted_reward / predicted_effort > adaptive_threshold → pursue
- Threshold adapts to: sunk cost, novelty, recent reward rate, pattern familiarity
- "Push beyond the limit" dynamic: stalled progress + high novelty → threshold falls

**Edit budget sizing:**
- Same gate, different output: how many edits to apply
- Strong evidence, low risk → more edits
- Thin evidence, critical process → fewer edits
- Learned from optimization outcomes

**Training:**
- Delayed reward + replay buffer
- Value loss, cost loss, decision loss
- Training hardware: RTX 4090, Vast.ai spot, 10 hours, ~$2.70

---

### Phase 7c: Curiosity Cascade Integration

**Goal:** Wire the Disturbance Detector, Intuition Module, and Aspirational Model into a recurrent circuit.

**Duration:** 3-4 days

**Key deliverables:**

**Meta-gate:**
- Fast triage: which cascade stages does this input need?
- ~50K params, runs before any stage
- Routine greeting → no stages. Complex question → all three.

**Recurrent circuit:**
- Disturbance Detector → Intuition Module → Aspirational Model
- Feedback loops between stages (gated to prevent infinite loops)
- Disturbance Detector → Common Sense Resolver: "I detected a disturbance, but it's ambiguous"
- Intuition Module → Disturbance Detector: "I'm inhibiting, but my history is thin — re-examine"

**Full cascade algorithm:**
1. Meta-gate: which stages?
2. Common Sense (if needed): resolve ambiguity
3. Disturbance Detection (if needed): find anomalies
4. Intuition (if needed): evaluate pursuit worth
5. Aspirational Model: commit to exploration or optimization

**Self-generated training:**
- Retrieval successes → positive salience signal
- Retrieval failures → training examples for missing edges
- Bonsai verifications → training examples for GNN anomaly detector
- User corrections → reconsolidation training examples
- Delegation outcomes → Retrieval Gate routing updates
- Process execution outcomes → process success rate updates

---

## Phase 8: Ecosystem

### Phase 8a: Process Marketplace

**Goal:** Enable processes to be published, discovered, imported, and improved across users.

**Duration:** 5-7 days

**Key deliverables:**

**Process registry:**
- Semantic versioning (major.minor.patch)
- Dependency declarations (domain graphs, sub-processes, tools, models)
- Compatibility ranges
- Transfer records (how well does this process transfer across domains?)
- Optimization provenance (what evidence produced this version?)

**Process package manager:**
- Import with dependency resolution
- Merge local adaptations with upstream changes
- Update notifications: "v3 is available. Your local adaptations are compatible."
- Rollback support

**Process absence detection (marketplace-aware):**
- Disturbance Detector monitors for patterns that match published processes
- "You've manually deployed Kubernetes 5 times. Published process 'k8s_deploy v3' handles this with 94% success rate."
- Suggests import, not forces it

**Feedback loop:**
- User imports process → adapts to their environment → publishes improvements
- Publisher's optimization loop continues → new versions
- User gets update notifications → merges or forks
- Marketplace learns which processes transfer well, which publishers are reliable

---

### Phase 8b: Domain Graph Sharing

**Goal:** Enable domain knowledge to be exported, imported, and composed.

**Duration:** 3-4 days

**Key deliverables:**

**Domain export/import:**
- Export domain graph as portable WaveDB instance
- Import with cross-graph edge linking
- Composable: link domains without merging

**Federated improvement:**
- Shared processing models improve from aggregate experience
- Personal graphs stay private
- Federated learning at the architectural level

**Cognitive style configurations:**
- Gate configurations as exportable cognitive styles
- "Systematic" style: high threshold, slow adaptation
- "Creative" style: low latent inhibition, high ambiguity tolerance
- Different tasks get different cognitive styles, not just different knowledge

---

## Cost Summary

Phase	Component	GPU	Provider	Hours	Cost
1a	GLiNER2 fine-tune	RTX 4090	Vast.ai spot	4	$1.08
1a	Bonsai relations (Oracle gen)	A100 80GB	RunPod Community	4	$6.56
1d	Oracle API (all training data)	—	DeepSeek API	—	~$20.00
2a	Shared SSM+JEPA backbone	A100 80GB	Lambda Labs	48	$61.92
2b	Retrieval Gate training	RTX 4090	Vast.ai spot	24	$6.48
2b	Bonsai query planning	A100 80GB	RunPod Community	10	$16.40
3a	GNN implementation	RTX 4090	Vast.ai spot	16	$4.32
3a	GNN training	RTX 4090	Vast.ai spot	32	$8.64
4a	Uncertainty Detector gate	RTX 4090	Vast.ai spot	10	$2.70
4b	Aspirational Model gate	RTX 4090	Vast.ai spot	10	$2.70
4c	Self-Model gate	RTX 4090	Vast.ai spot	10	$2.70
4d	Common Sense Resolver gate	RTX 4090	Vast.ai spot	10	$2.70
7a	Disturbance Detector gate	RTX 4090	Vast.ai spot	10	$2.70
7b	Intuition Module gate	RTX 4090	Vast.ai spot	10	$2.70
**TOTAL**				**198**	**~$142**

---

## Hardware Requirements (Single User, Full System)

Component	Hardware	VRAM/RAM
GLiNER2 + GLiNER-Decoder	CPU	~8 GB RAM
Bonsai (8B ternary)	GPU	~2.15 GB
JEPA-Gated SSM backbone (~480M)	GPU	~0.90 GB
8× instance gates + states (~20M total)	GPU	~0.08 GB
GNN (~200M)	GPU	~0.40 GB
**Total GPU**	**RTX 5060 Ti (16 GB)**	**~3.5 GB used**
WaveDB	CPU + disk	50-100 MB LRU + disk
Vector Index	CPU	~100 MB - 1 GB
Delegation models (70B+)	Cloud API	Per-use cost

**Total inference cost: $0 (local). Total hardware: ~$400-500 (GPU).**

---

## Developmental Stage Transitions

Component	INFANT → CHILD	CHILD → ADOLESCENT	ADOLESCENT → ADULT
**Extraction**	F1 > 0.85	Oracle consultation rate < 0.2	Independent accuracy > 0.90
**Retrieval Gate**	Routing accuracy > 0.75 vs Oracle	Oracle consultation rate < 0.15	Independent accuracy > 0.90
**GNN**	Predicted edges accepted > 0.70	Oracle consultation rate < 0.2	Independent accuracy > 0.85
**Uncertainty Detector**	"I don't know" precision > 0.80	Oracle consultation rate < 0.2	Independent accuracy > 0.90
**Self-Model**	"I don't know" precision > 0.80	Oracle consultation rate < 0.2	Independent accuracy > 0.90
**Common Sense Resolver**	Resolution accuracy > 0.80 vs Oracle	Oracle consultation rate < 0.2	Independent accuracy > 0.90
**Intuition Module**	Gate decisions match Oracle > 0.75	Oracle consultation rate < 0.2	Independent accuracy > 0.85
**Disturbance Detector**	Detection recall > 0.80	False positive rate < 0.2	Independent accuracy > 0.90

---

## What Each Phase Delivers to the User

Phase	User Experience
**1a**	Nothing user-facing. Internal extraction pipeline.
**1b**	A database you can talk to. Remembers conversations. Retrieves relevant context. Responds via any LLM.
**1c**	Better retrieval. Documents returned as documents. Pronoun resolution works.
**2**	Subconscious routing. System anticipates domains. Faster responses for routine queries. SSM chunking handles large context.
**3**	Memory consolidates during idle. Semantic abstractions appear. Forgetting is graceful — old info fades, important info persists.
**4**	System knows when it doesn't know. "I don't have information about that." Prospective memory: "You asked me to remind you about X." Resolves ambiguity before acting.
**5**	Quantified performance. Benchmarks validate the architecture.
**6**	Procedural memory. System learns your workflows. Executes processes. Delegates to larger models when needed. Failure handling is explicit and reliable.
**7**	Curiosity. System detects when processes need improvement. Evaluates whether improvement is worth the effort. Optimizes automatically.
**8**	Ecosystem. Import processes from others. Publish your own. Domain knowledge is transferrable. The system improves across users.

---

*This plan is designed to be built incrementally. Each phase produces a working system. Phase 1b alone — encoding + retrieval + Mode A generation — is a functional memory system you can talk to. Each subsequent phase adds capability without requiring redesign of earlier components. The architecture separates knowledge from processing, making domain expertise a transferrable, composable resource. The subconscious router learns from every interaction, becoming personalized over time. The ponder engine doesn't think — it ensures that when it's time to think, everything needed is already in awareness.*