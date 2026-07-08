"""Oracle prompts for Phase 1d training-data generation.

Each prompt is a pure function that takes structured input and returns a
formatted prompt string. Keeping prompts as functions (not inline literals in
the generators) makes them versionable, unit-testable, and independent of the
Oracle API-calling code in ``src/training/oracle_labeling.py: OracleClient``.

Ported from ``docs/Phase 1d.md`` §3. The prompt *bodies* are verbatim from the
doc; only the module scaffolding is new. Every prompt instructs the Oracle to
``Return ONLY valid JSON`` so the client's ``response_format: json_object`` +
salvage parser can recover the payload.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# GNN TRAINING DATA
# ═══════════════════════════════════════════════════════════════


def gnn_salience_prompt(subgraph_json: str) -> str:
    """Prompt for GNN salience scoring labels."""
    return f"""You are labeling a memory graph for GNN training.
Score each node and edge by structural importance (0.0-1.0).

HIGH salience (>0.7):
- Bridge nodes connecting otherwise-separate clusters
- Episodes containing major decisions
- Temporal chain anchors (first episode in a sequence)
- Nodes with unique information not available through other paths

MEDIUM salience (0.3-0.7):
- Episodes with moderate entity/topic overlap with other episodes
- Nodes that are part of active temporal chains but not anchors

LOW salience (<0.3):
- Routine conversations with no decisions
- Nodes with redundant information available through other paths
- Peripheral entities mentioned only once

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"node_scores": {{"ep_001": {{"salience": 0.92, "reason": "..."}}, ...}},
 "edge_scores": {{"edge_001": {{"salience": 0.80, "reason": "..."}}, ...}}}}"""


def gnn_cluster_prompt(subgraph_json: str) -> str:
    """Prompt for GNN cluster detection labels."""
    return f"""You are labeling a memory graph for GNN training.
Identify groups of episodes that should be abstracted into semantic memories.

A valid cluster has:
- Shared entities (at least 2 entities in common)
- Shared topics (at least 1 topic in common)
- Temporal proximity (within 7 days of each other)
- Coherent theme (the episodes tell a connected story)

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"clusters": [
    {{"name": "WaveDB initial development (June 20-24)",
     "episodes": ["ep_001", "ep_002", "ep_003", "ep_004"],
     "abstracted_summary": "Decided on HBTrie architecture...",
     "coherence_score": 0.89}}
]}}"""


def gnn_link_prediction_prompt(subgraph_json: str) -> str:
    """Prompt for GNN link prediction labels (positive + negative edges).

    SEAL/GAE need explicit negative edges, not just positives — a link-prediction
    head trained on positive-only labels collapses to "predict 1 everywhere."
    So the Oracle emits both ``predicted_edges`` (edges that SHOULD exist but
    don't) and ``negative_edges`` (node pairs that plausibly COULD share an edge
    but should NOT — unrelated entities/topics, non-adjacent episodes with no
    shared context). Phase 3a Task 3 added the negative-edges field.
    """
    return f"""You are labeling a memory graph for GNN training.
Identify edges that SHOULD exist but are not explicitly in the graph, AND node
pairs that should NOT be linked (negative examples for link prediction).

Look for POSITIVE edges (predicted_edges):
- Entities that co-occur in similar contexts but have no direct edge
- Episodes that share topics/entities but aren't linked
- Hierarchical relationships implied by usage patterns
- Causal relationships implied by temporal order
- Contradictions between statements in different episodes

Look for NEGATIVE edges (negative_edges) — pairs that plausibly COULD share an
edge given their types/proximity but should NOT:
- Entities from unrelated domains that happen to appear in the subgraph
- Episodes far apart in time with no shared entities/topics
- Topic/entity pairs with no semantic relationship

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"predicted_edges": [
    {{"subject": "Postgres", "predicate": "related_to", "object": "performance",
     "confidence": 0.82, "evidence": "Both appear in episodes about..."}}
],
 "negative_edges": [
    {{"subject": "Postgres", "predicate": "related_to", "object": "WaveDB",
     "confidence": 0.10, "evidence": "No shared context; unrelated domains"}}
]}}"""


def gnn_anomaly_prompt(subgraph_json: str) -> str:
    """Prompt for GNN anomaly detection labels."""
    return f"""You are labeling a memory graph for GNN training.
Flag structural anomalies — patterns that don't fit a well-formed memory graph.

Anomaly types:
- ORPHAN_DECISION: Decision node with no 'madeBy' edge
- MISSING_TEMPORAL: Gap in follows chain
- CONTRADICTION: Two edges that cannot both be true
- TYPE_VIOLATION: Edge connecting incompatible types
- ISOLATED_CLUSTER: Subgraph with no external connections
- DUPLICATE_DECISION: Same decision appears to be made twice

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"anomalies": [
    {{"type": "MISSING_TEMPORAL", "severity": "warning",
     "description": "ep_007 follows ep_004 but ep_005-006 also follow ep_004",
     "involved_nodes": ["ep_004", "ep_005", "ep_007"]}}
]}}"""


def gnn_ontology_prompt(subgraph_json: str, current_ontology: str) -> str:
    """Prompt for GNN ontology refinement labels."""
    return f"""You are labeling a memory graph for GNN training.
Suggest missing subClassOf edges and misclassified entities.

CURRENT ONTOLOGY:
{current_ontology}

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"suggested_edges": [
    {{"child": "DEBOUNCED", "parent": "WALSyncMode", "confidence": 0.90,
     "evidence": "Discussed alongside IMMEDIATE and ASYNC as alternatives"}}
],
 "misclassified": [
    {{"entity": "WaveDB", "current_class": "Application",
     "suggested_class": "Database", "confidence": 0.85}}
]}}"""


# ═══════════════════════════════════════════════════════════════
# BONSAI TRAINING DATA
# ═══════════════════════════════════════════════════════════════


def bonsai_query_planning_prompt(conversation_text: str, question: str) -> str:
    """Prompt for generating Bonsai query planning training pairs."""
    return f"""You are generating training data for a query planner that converts
natural language questions into structured memory queries.

CONVERSATION:
{conversation_text}

HYPOTHETICAL QUESTION: {question}

The memory graph stores episodes with these attributes:
- entities: [Person, Project, Technology, Concept]
- topics: [database_design, configuration, graph_database, performance,
           decision_making, ai_architecture, api_design, security]
- tones: [frustrated, excited, curious, neutral]
- decisions: specific choices made

Query parameters:
- entities: list of entities to search for
- topics: list of topics to filter by
- tones: list of emotional tones to filter by
- entity_mode: "intersection" (ALL entities) or "union" (ANY entity)
- temporal_after: keyword to find anchor and follow chain forward, or null
- temporal_before: keyword to find anchor and follow chain backward, or null
- temporal_filter: "today", "this_week", "last_week", "this_month", or null
- date_from: ISO date for start of range, or null
- date_to: ISO date for end of range, or null
- limit: max episodes to return (default 5)

IMPORTANT RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union"
- "What did Alice and Bob disagree about?" → entities=["Alice", "Bob"], entity_mode="intersection"
- "What happened after morphisms?" → temporal_after="morphism"
- "Why did we choose X over Y?" → topics=["decision_making"], entities=["X", "Y"], entity_mode="union"

Return ONLY valid JSON:
{{"question": "{question}",
 "query": {{"entities": [], "topics": [], "tones": [], "entity_mode": "union",
           "temporal_after": null, "temporal_before": null,
           "temporal_filter": null, "date_from": null, "date_to": null,
           "limit": 5}},
 "reasoning": "Brief explanation of why these parameters were chosen"}}"""


def bonsai_relation_extraction_prompt(conversation_text: str) -> str:
    """Prompt for generating Bonsai relation extraction training pairs."""
    return f"""Extract relationships from this conversation. Return ONLY valid JSON.

Relation types:
- explains(Person, Concept): someone explains something
- decides(Person, Decision): someone makes a decision
- expresses(Person, Tone): someone expresses an emotion
- questions(Person, Concept): someone asks about something
- suggests(Person, Concept): someone proposes an idea
- concerns(Episode, Topic): the conversation is about a topic
- involves(Episode, Entity): an entity participates
- contradicts(Statement, Statement): one statement contradicts another
- follows_up_on(Episode, Episode): this conversation continues from another

CONVERSATION:
{conversation_text}

Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""


# ═══════════════════════════════════════════════════════════════
# JEPA ROUTING DATA
# ═══════════════════════════════════════════════════════════════


def jepa_routing_prompt(prompt: str, available_domains: str,
                        available_pathways: str) -> str:
    """Prompt for generating JEPA routing training pairs."""
    return f"""You are generating training data for a subconscious router that
decides how to handle a user's query before any retrieval or generation.

USER QUERY: {prompt}

AVAILABLE DOMAINS:
{available_domains}

AVAILABLE PATHWAYS:
{available_pathways}

MODEL SIZES: 1B, 3B, 8B, 70B, 175B

META-SKILLS: factual_recall, basic_synthesis, pattern_recognition,
             decomposition, process_selection, creative_synthesis,
             security_analysis, tradeoff_analysis

Decide:
1. Which domain(s) to query?
2. Which pathway to use?
3. What meta-skills are required?
4. What model size is needed?
5. Is conscious deliberation needed?

Return ONLY valid JSON:
{{"domains": ["database"],
 "pathway": "graph_retrieve",
 "meta_skills": ["factual_recall", "basic_synthesis"],
 "model_size": "3B",
 "needs_deliberation": false,
 "confidence": 0.89,
 "reasoning": "Brief explanation"}}"""


# ═══════════════════════════════════════════════════════════════
# GATE TRAINING DATA
# ═══════════════════════════════════════════════════════════════


def uncertainty_detector_prompt(context: str, query: str,
                                retrieval_results: str) -> str:
    """Prompt for Uncertainty Detector gate training."""
    return f"""You are generating training data for an uncertainty detector.
Given a query, the retrieved context, and what the system knows, determine
whether the system should flag uncertainty.

CONTEXT (what the system knows):
{context}

USER QUERY: {query}

RETRIEVAL RESULTS:
{retrieval_results}

Should the system flag uncertainty? Consider:
- Is the retrieved context sufficient to answer the query?
- Are there novel entities not in the ontology?
- Are there contradictions in the retrieved results?
- Is the routing confidence low?

Return ONLY valid JSON:
{{"should_flag": true/false,
 "uncertainty_type": "routing_uncertainty|novel_entity|unresolved_contradiction|none",
 "confidence": 0.0-1.0,
 "reasoning": "Brief explanation"}}"""


def aspirational_model_prompt(goal_context: str, candidate_action: str) -> str:
    """Prompt for Aspirational Model gate training."""
    return f"""You are generating training data for an aspirational model.
Given the agent's current goals and a candidate action, determine whether
the agent should commit to this action.

CURRENT GOALS AND CONTEXT:
{goal_context}

CANDIDATE ACTION: {candidate_action}

Consider:
- Does this align with known goals?
- Is the expected value worth the effort?
- Is this a novel opportunity or a routine action?
- Should a prospective trigger be set?

Return ONLY valid JSON:
{{"should_commit": true/false,
 "encoding_strength": 0.0-1.0,
 "set_prospective_trigger": true/false,
 "trigger_condition": "description or null",
 "reasoning": "Brief explanation"}}"""


def self_model_prompt(knowledge_state: str, query: str) -> str:
    """Prompt for Self-Model gate training."""
    return f"""You are generating training data for a self-model that estimates
its own knowledge boundaries.

KNOWLEDGE STATE:
{knowledge_state}

USER QUERY: {query}

Should the system say "I don't know" or attempt to answer?
Consider:
- Is the knowledge in this domain dense or sparse?
- Is the specific fact likely to be known?
- Would answering risk hallucination?

Return ONLY valid JSON:
{{"should_say_dont_know": true/false,
 "confidence_in_answer": 0.0-1.0,
 "knowledge_boundary_hit": true/false,
 "reasoning": "Brief explanation"}}"""


# ═══════════════════════════════════════════════════════════════
# CODE-AWARE SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════


def code_aware_synthetic_prompt(domain: str, code_ontology_fragment: str) -> str:
    """Prompt for generating synthetic code-aware training examples."""
    return f"""You are generating synthetic training data for a memory system
that needs to learn about code structure before any real code is parsed.

DOMAIN: {domain}

CODE ONTOLOGY (available types):
{code_ontology_fragment}

Generate a realistic conversation about software development that includes
code artifacts. Then extract structured triples using the code ontology types.

Return ONLY valid JSON:
{{"conversation": "User: ... Assistant: ...",
 "extracted_entities": ["auth.py", "authenticate_user", "JWT", ...],
 "extracted_topics": ["security", "api_design"],
 "extracted_relations": [
    {{"subject": "auth.py", "predicate": "contains", "object": "authenticate_user"}},
    {{"subject": "authenticate_user", "predicate": "calls", "object": "validate_token"}}
 ],
 "code_artifacts": [
    {{"type": "File", "name": "auth.py"}},
    {{"type": "Function", "name": "authenticate_user", "defined_in": "auth.py"}}
 ]}}"""