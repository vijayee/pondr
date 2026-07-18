# Phase 1d: Training Data Generation — Implementation Plan for Claude Code

## Overview

**Goal:** Generate all labeled training data needed for downstream learned components using the Oracle (DeepSeek). This phase transforms the populated WaveDB from Phase 1b into a complete training dataset for the JEPA-Gated SSM backbone (Phase 2), the GNN Consolidator (Phase 3), and all instance-specific gates (Phase 4).

**What "done" looks like:** A directory of training data files covering five task categories, with quality validation reports. Every downstream component has the labeled examples it needs to begin training. The Oracle is no longer needed for initial training — components can train independently from this point forward.

**Prerequisite:** Phase 1b complete. Populated WaveDB with 1,500+ encoded episodes. Oracle labeling infrastructure (prompts, subgraph extraction) in place. Phase 1c is NOT a prerequisite — training data generation can run in parallel with retrieval refinements.

**Duration estimate:** 3-5 days (mostly Oracle API time, not implementation time).

---

## 1. What Phase 1d Delivers

Artifact	Count	Consumer	Est. Oracle Cost
**GNN training subgraphs**	4,000+ labeled subgraphs	Phase 3: GNN Consolidator	~$7.00
**Bonsai query planning pairs**	5,000-10,000 (prompt, query) pairs	Phase 2: Bonsai fine-tuning	~$3.50
**Bonsai relation extraction pairs**	2,000+ (text, relations) pairs	Phase 2: Bonsai fine-tuning	~$1.75
**JEPA routing pairs**	5,000+ (prompt, route) pairs	Phase 2: Retrieval Gate	~$3.50
**Uncertainty Detector gate examples**	50,000 labeled decisions	Phase 4: Uncertainty Detector	~$0.80
**Aspirational Model gate examples**	50,000 labeled decisions	Phase 4: Aspirational Model	~$0.80
**Self-Model gate examples**	50,000 labeled decisions	Phase 4: Self-Model	~$0.80
**Synthetic code-aware examples**	2,000+ mixed examples	All phases (code ontology)	~$1.00
**Quality validation reports**	Per-dataset metrics	All phases	—
**TOTAL**			**~$20.00**

---

## 2. Project Structure (Additions)

```plaintext
hippocampal-memory/
├── data/
│   └── training/                        # NEW — all generated training data
│       ├── gnn/
│       │   ├── subgraphs.jsonl           # 4,000+ labeled subgraphs
│       │   ├── salience_labels.jsonl
│       │   ├── cluster_labels.jsonl
│       │   ├── link_prediction_labels.jsonl
│       │   ├── anomaly_labels.jsonl
│       │   └── ontology_labels.jsonl
│       ├── bonsai/
│       │   ├── query_planning_pairs.jsonl
│       │   └── relation_extraction_pairs.jsonl
│       ├── jepa/
│       │   └── routing_pairs.jsonl
│       ├── gates/
│       │   ├── uncertainty_detector.jsonl
│       │   ├── aspirational_model.jsonl
│       │   └── self_model.jsonl
│       ├── code_aware/
│       │   └── synthetic_examples.jsonl
│       └── reports/
│           ├── gnn_quality.json
│           ├── bonsai_quality.json
│           ├── jepa_quality.json
│           └── gates_quality.json
├── scripts/
│   ├── generate_gnn_training_data.py     # NEW
│   ├── generate_bonsai_training_data.py  # NEW
│   ├── generate_jepa_training_data.py    # NEW
│   ├── generate_gate_training_data.py    # NEW
│   ├── generate_code_aware_data.py       # NEW
│   └── validate_training_data.py         # NEW
└── src/
    └── training/
        ├── __init__.py
        ├── oracle_labeling.py            # UPDATED: full implementation
        ├── prompts.py                    # NEW: all Oracle prompts
        └── validators.py                # NEW: quality validation
```

---

## 3. Oracle Prompt Library (`src/training/prompts.py`)

All Oracle prompts in one place, versioned and testable:

```python
"""Oracle prompts for training data generation.

Each prompt is a function that takes structured input and returns
a formatted prompt string. This keeps prompts versionable, testable,
and independent of the API calling code.
"""

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
    """Prompt for GNN link prediction labels."""
    return f"""You are labeling a memory graph for GNN training.
Identify edges that SHOULD exist but are not explicitly in the graph.

Look for:
- Entities that co-occur in similar contexts but have no direct edge
- Episodes that share topics/entities but aren't linked
- Hierarchical relationships implied by usage patterns
- Causal relationships implied by temporal order
- Contradictions between statements in different episodes

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"predicted_edges": [
    {{"subject": "Postgres", "predicate": "related_to", "object": "performance",
     "confidence": 0.82, "evidence": "Both appear in episodes about..."}}
]}}"""


# NOTE: anomaly labels are NO LONGER produced by an Oracle prompt. The 6-type
# ``gnn_anomaly_prompt`` schema below was superseded in Phase 3a Task 3 by an
# Oracle-FREE injection path: ``anomaly_injector.inject_anomalies`` plants
# synthetic anomalies into a clean subgraph and ``anomaly_rules`` (the canonical
# ``ANOMALY_TYPES`` tuple + ``enrich_subgraph`` + ``node_label_vectors``) detects
# them and emits the per-node 9-type label vector the ``AnomalyHead`` trains on
# (see ``src/gnn/heads.py:AnomalyHead`` and ``src/gnn/anomaly_rules.py``). The
# live generator (``scripts/generate_gnn_training_data.py``) imports
# ``inject_anomalies`` + ``anomaly_rules`` and never calls an Oracle for
# anomalies; the snippet kept below this comment is the HISTORICAL 6-type prompt
# for reference only and is NOT used.
def gnn_anomaly_prompt(subgraph_json: str) -> str:
    """HISTORICAL — superseded by the Oracle-FREE injection path (see comment
    above). Kept for reference; do NOT call in the live generator."""
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
```

---

## 4. Oracle API Client (`src/training/oracle_labeling.py`)

```python
"""Oracle API client for training data generation.

Handles batching, rate limiting, retries, cost tracking, and validation.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


@dataclass
class OracleConfig:
    """Configuration for Oracle API calls."""
    model: str = "deepseek-chat"           # or "gpt-4o" for OpenAI
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay: float = 2.0              # seconds between retries
    batch_delay: float = 0.5              # seconds between batches (rate limiting)
    cost_per_1k_input: float = 0.001      # $0.001 per 1K input tokens (adjust for model)
    cost_per_1k_output: float = 0.002     # $0.002 per 1K output tokens


@dataclass
class OracleResult:
    """Result of an Oracle API call."""
    prompt: str
    response: dict
    input_tokens: int
    output_tokens: int
    cost: float
    latency_seconds: float
    retries: int
    cached: bool = False


class OracleClient:
    """
    Manages Oracle API calls with batching, caching, and cost tracking.
    """
    
    def __init__(self, config: OracleConfig = None):
        self.config = config or OracleConfig()
        self.client = OpenAI()  # or DeepSeek client
        self.cache = {}         # prompt_hash → OracleResult
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_tokens = 0
    
    def generate(self, prompt: str, 
                 response_format: str = "json_object") -> OracleResult:
        """
        Call the Oracle with caching and retry logic.
        
        Caching: identical prompts return cached results.
        Retries: up to max_retries with exponential backoff.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        # Check cache
        if prompt_hash in self.cache:
            cached = self.cache[prompt_hash]
            cached.cached = True
            return cached
        
        # Call Oracle with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                start = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": response_format},
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                
                elapsed = time.time() - start
                
                # Parse response
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Track costs
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = (
                    input_tokens / 1000 * self.config.cost_per_1k_input +
                    output_tokens / 1000 * self.config.cost_per_1k_output
                )
                
                result = OracleResult(
                    prompt=prompt,
                    response=data,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    latency_seconds=elapsed,
                    retries=attempt,
                )
                
                # Cache and track
                self.cache[prompt_hash] = result
                self.total_cost += cost
                self.total_calls += 1
                self.total_tokens += input_tokens + output_tokens
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    time.sleep(delay)
        
        raise RuntimeError(f"Oracle call failed after {self.config.max_retries} retries: {last_error}")
    
    def generate_batch(self, prompts: list[str], 
                       response_format: str = "json_object",
                       progress_callback=None) -> list[OracleResult]:
        """
        Generate results for a batch of prompts.
        
        Includes rate limiting between calls and progress reporting.
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, response_format)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(prompts), result)
            
            # Rate limiting
            if i < len(prompts) - 1:
                time.sleep(self.config.batch_delay)
        
        return results
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "cached_calls": sum(1 for r in self.cache.values() if r.cached),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 2),
            "cache_size": len(self.cache),
        }
```

---

## 5. GNN Training Data Generator (`scripts/generate_gnn_training_data.py`)

```python
"""Generate GNN training data from the populated WaveDB.

Usage:
    python scripts/generate_gnn_training_data.py \
        --db ./data/memory_db \
        --output data/training/gnn/ \
        --num-subgraphs 4000 \
        --subgraph-radius 3 \
        --batch-size 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleClient, OracleConfig
from src.training.prompts import (
    gnn_salience_prompt,
    gnn_cluster_prompt,
    gnn_link_prediction_prompt,
    gnn_anomaly_prompt,  # HISTORICAL — anomalies now Oracle-FREE (see below)
    gnn_ontology_prompt,
)
# Oracle-FREE anomaly labels (the live path, replaces the gnn_anomaly_prompt
# Oracle call for the anomaly task):
from src.gnn.anomaly_injector import inject_anomalies
from src.gnn.anomaly_rules import ANOMALY_TYPES, enrich_subgraph, node_label_vectors


def main():
    parser = argparse.ArgumentParser(description="Generate GNN training data")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    parser.add_argument("--output", default="data/training/gnn/", help="Output directory")
    parser.add_argument("--num-subgraphs", type=int, default=4000, help="Number of subgraphs to label")
    parser.add_argument("--subgraph-radius", type=int, default=3, help="BFS radius for subgraph extraction")
    parser.add_argument("--batch-size", type=int, default=10, help="Oracle calls per batch")
    parser.add_argument("--limit", type=int, help="Max subgraphs (overrides --num-subgraphs)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    store = HippocampalStore(args.db)
    oracle = OracleClient(OracleConfig())
    
    # ── 1. Extract subgraphs ──
    print("Extracting subgraphs from memory graph...")
    subgraphs = extract_subgraphs(store, args.num_subgraphs, args.subgraph_radius)
    print(f"Extracted {len(subgraphs)} subgraphs")
    
    # ── 2. Generate labels for each task ──
    # NOTE: anomalies are Oracle-FREE. ``inject_anomalies`` plants synthetic
    # anomalies and ``anomaly_rules`` detects them -> per-node 9-type label
    # vector (NOT the historical ``gnn_anomaly_prompt`` Oracle call). The
    # live ``scripts/generate_gnn_training_data.py`` does the injection pass
    # alongside the Oracle-driven salience/cluster/linkpred/ontology tasks.
    tasks = [
        ("salience", gnn_salience_prompt, "salience_labels.jsonl"),
        ("clusters", gnn_cluster_prompt, "cluster_labels.jsonl"),
        ("link_prediction", gnn_link_prediction_prompt, "link_prediction_labels.jsonl"),
        ("anomalies", gnn_anomaly_prompt, "anomaly_labels.jsonl"),  # superseded — see note above
        ("ontology", gnn_ontology_prompt, "ontology_labels.jsonl"),
    ]
    
    all_stats = {}
    
    for task_name, prompt_fn, output_file in tasks:
        print(f"\n{'='*60}")
        print(f"Generating {task_name} labels...")
        print(f"{'='*60}")
        
        results = []
        stats = Counter()
        
        # Resume support
        start_idx = 0
        checkpoint_file = output_dir / f"{task_name}_checkpoint.json"
        if args.resume and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                start_idx = checkpoint["last_index"]
                results = checkpoint["results"]
                print(f"Resuming from index {start_idx}")
        
        for i in range(start_idx, len(subgraphs), args.batch_size):
            batch = subgraphs[i:i + args.batch_size]
            
            # Build prompts
            prompts = []
            for sg in batch:
                if task_name == "ontology":
                    prompts.append(prompt_fn(
                        json.dumps(sg),
                        json.dumps(get_current_ontology(store))
                    ))
                else:
                    prompts.append(prompt_fn(json.dumps(sg)))
            
            # Call Oracle
            batch_results = oracle.generate_batch(
                prompts,
                progress_callback=lambda done, total, result: print(
                    f"  {task_name}: {done}/{total} batches "
                    f"(cost: ${oracle.total_cost:.2f})"
                )
            )
            
            # Store results
            for sg, result in zip(batch, batch_results):
                results.append({
                    "subgraph_id": sg["id"],
                    "labels": result.response,
                    "cost": result.cost,
                })
                stats["labeled"] += 1
                stats["total_cost"] += result.cost
            
            # Checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "last_index": i + len(batch),
                    "results": results,
                }, f)
            
            # Progress
            print(f"  {task_name}: {min(i + args.batch_size, len(subgraphs))}/{len(subgraphs)} "
                  f"subgraphs labeled (${stats['total_cost']:.2f})")
        
        # Write output
        output_path = output_dir / output_file
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        all_stats[task_name] = dict(stats)
        print(f"  {task_name}: {stats['labeled']} labels, ${stats['total_cost']:.2f}")
    
    # ── 3. Quality report ──
    report = {
        "total_subgraphs": len(subgraphs),
        "tasks": all_stats,
        "oracle_stats": oracle.get_stats(),
        "elapsed_seconds": time.time() - start_time,
    }
    
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"GNN training data generation complete.")
    print(f"Total cost: ${oracle.total_cost:.2f}")
    print(f"Total calls: {oracle.total_calls}")
    print(f"Output: {output_dir}")
    
    store.close()


def extract_subgraphs(store, num_subgraphs, radius):
    """
    Extract subgraphs from the memory graph.
    
    Strategy:
    1. Score all episodes by structural interest (decisions, entities, tones)
    2. Select the top N as centers
    3. BFS from each center to extract subgraph
    """
    # Get all episodes
    all_eps = _get_all_episode_ids(store)
    
    # Score by interest
    scored = []
    for ep_id in all_eps:
        ep = store.get_episode(ep_id)
        if not ep:
            continue
        
        interest = 0.0
        if ep.decisions:
            interest += len(ep.decisions) * 3
        if ep.entities:
            interest += len(ep.entities) * 0.5
        if any(t in ["frustrated", "excited"] for t in ep.tones):
            interest += 2
        
        scored.append((interest, ep_id))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Select centers (stratified: high-interest + random sample)
    top_n = min(num_subgraphs // 2, len(scored))
    centers = [ep_id for _, ep_id in scored[:top_n]]
    
    # Add random sample for diversity
    remaining = [ep_id for _, ep_id in scored[top_n:]]
    if remaining:
        import random
        random.shuffle(remaining)
        centers.extend(remaining[:num_subgraphs - len(centers)])
    
    # Extract subgraphs
    subgraphs = []
    for center in centers:
        sg = _bfs_subgraph(store, center, radius)
        if sg and len(sg["nodes"]) >= 3:  # Minimum viable subgraph
            subgraphs.append(sg)
    
    return subgraphs[:num_subgraphs]


def _bfs_subgraph(store, center_id, radius):
    """Extract subgraph via BFS from center node."""
    visited = set()
    frontier = {center_id}
    
    for _ in range(radius):
        next_frontier = set()
        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)
            neighbors = _get_neighbors(store, node_id)
            next_frontier.update(neighbors)
        frontier = next_frontier - visited
    
    nodes = []
    for node_id in visited:
        ep = store.get_episode(node_id)
        if ep:
            nodes.append({
                "id": ep.id,
                "type": "Episode",
                "entities": ep.entities,
                "topics": ep.topics,
                "tones": ep.tones,
                "decisions": ep.decisions,
                "timestamp": ep.timestamp,
            })
    
    if not nodes:
        return None
    
    return {
        "id": f"subgraph_{center_id}",
        "center": center_id,
        "radius": radius,
        "nodes": nodes,
        "node_count": len(nodes),
    }


def _get_neighbors(store, node_id):
    """Get all neighbor nodes."""
    neighbors = set()
    
    # Follows edges
    result = store.graph.query().vertex(node_id).out("follows").execute()
    neighbors.update(r.id for r in result)
    result = store.graph.query().vertex(node_id).in_("follows").execute()
    neighbors.update(r.id for r in result)
    
    # Entity edges
    result = store.graph.query().vertex(node_id).out("has_entity").execute()
    for r in result:
        entity_eps = store.graph.query().vertex(r.id).in_("in_episode").execute()
        neighbors.update(e.id for e in entity_eps)
    
    return list(neighbors)


def _get_all_episode_ids(store):
    result = store.graph.query().has("predicate", "has_entity").execute()
    return list(set(r.subject for r in result))


def get_current_ontology(store):
    """Get current ontology state for ontology refinement prompts."""
    # Query all subClassOf edges
    result = store.graph.query().has("predicate", "subClassOf").execute()
    classes = {}
    for r in result:
        if r.subject not in classes:
            classes[r.subject] = []
        classes[r.subject].append(r.object)
    return {"classes": classes}


if __name__ == "__main__":
    main()
```

---

## 6. Bonsai Training Data Generator (`scripts/generate_bonsai_training_data.py`)

```python
"""Generate Bonsai training data from the populated WaveDB.

Usage:
    python scripts/generate_bonsai_training_data.py \
        --db ./data/memory_db \
        --output data/training/bonsai/ \
        --num-query-pairs 5000 \
        --num-relation-pairs 2000
"""

import argparse
import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleClient, OracleConfig
from src.training.prompts import (
    bonsai_query_planning_prompt,
    bonsai_relation_extraction_prompt,
)


def main():
    parser = argparse.ArgumentParser(description="Generate Bonsai training data")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    parser.add_argument("--output", default="data/training/bonsai/", help="Output directory")
    parser.add_argument("--num-query-pairs", type=int, default=5000)
    parser.add_argument("--num-relation-pairs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    store = HippocampalStore(args.db)
    oracle = OracleClient(OracleConfig())
    
    # ── 1. Query planning pairs ──
    print("Generating query planning pairs...")
    
    # Get episodes to use as conversation context
    episodes = _get_all_episodes(store)
    random.shuffle(episodes)
    episodes = episodes[:args.num_query_pairs]
    
    # Generate hypothetical questions for each episode
    query_pairs = []
    
    for i in range(0, len(episodes), args.batch_size):
        batch = episodes[i:i + args.batch_size]
        
        prompts = []
        for ep in batch:
            # Generate 1-2 hypothetical questions per episode
            questions = _generate_hypothetical_questions(ep)
            for q in questions:
                prompts.append(bonsai_query_planning_prompt(ep["full_text"], q))
        
        results = oracle.generate_batch(prompts)
        
        for ep, result in zip(batch, results):
            query_pairs.append({
                "conversation_id": ep["id"],
                "conversation_text": ep["full_text"],
                "training_pair": result.response,
            })
        
        print(f"  {min(i + args.batch_size, len(episodes))}/{len(episodes)} "
              f"episodes processed (${oracle.total_cost:.2f})")
    
    # Write
    with open(output_dir / "query_planning_pairs.jsonl", "w") as f:
        for pair in query_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"  Generated {len(query_pairs)} query planning pairs")
    
    # ── 2. Relation extraction pairs ──
    print("\nGenerating relation extraction pairs...")
    
    # Use different episodes for relation extraction
    rel_episodes = _get_all_episodes(store)
    random.shuffle(rel_episodes)
    rel_episodes = rel_episodes[:args.num_relation_pairs]
    
    relation_pairs = []
    
    for i in range(0, len(rel_episodes), args.batch_size):
        batch = rel_episodes[i:i + args.batch_size]
        
        prompts = [bonsai_relation_extraction_prompt(ep["full_text"]) for ep in batch]
        results = oracle.generate_batch(prompts)
        
        for ep, result in zip(batch, results):
            relation_pairs.append({
                "conversation_id": ep["id"],
                "conversation_text": ep["full_text"],
                "relations": result.response.get("relations", []),
            })
        
        print(f"  {min(i + args.batch_size, len(rel_episodes))}/{len(rel_episodes)} "
              f"episodes processed (${oracle.total_cost:.2f})")
    
    with open(output_dir / "relation_extraction_pairs.jsonl", "w") as f:
        for pair in relation_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"  Generated {len(relation_pairs)} relation extraction pairs")
    
    # ── 3. Quality report ──
    report = {
        "query_planning_pairs": len(query_pairs),
        "relation_extraction_pairs": len(relation_pairs),
        "oracle_stats": oracle.get_stats(),
    }
    
    with open(output_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nBonsai training data generation complete.")
    print(f"Total cost: ${oracle.total_cost:.2f}")
    
    store.close()


def _get_all_episodes(store):
    """Get all episodes with full text."""
    result = store.graph.query().has("predicate", "has_entity").execute()
    episode_ids = list(set(r.subject for r in result))
    
    episodes = []
    for ep_id in episode_ids:
        ep = store.get_episode(ep_id)
        if ep and ep.full_text:
            episodes.append({
                "id": ep.id,
                "full_text": ep.full_text,
                "entities": ep.entities,
                "topics": ep.topics,
                "tones": ep.tones,
                "decisions": ep.decisions,
            })
    
    return episodes


def _generate_hypothetical_questions(episode):
    """Generate hypothetical questions a user might ask about this episode."""
    questions = []
    
    # Entity-based questions
    for entity in episode.get("entities", [])[:2]:
        questions.append(f"What did {entity} say?")
        questions.append(f"What was {entity}'s opinion?")
    
    # Topic-based questions
    for topic in episode.get("topics", [])[:1]:
        questions.append(f"What did we discuss about {topic}?")
    
    # Tone-based questions
    for tone in episode.get("tones", [])[:1]:
        questions.append(f"What was I {tone} about?")
    
    # Decision-based questions
    if episode.get("decisions"):
        questions.append("What did we decide?")
        questions.append("Why did we make that decision?")
    
    # Temporal questions
    questions.append("What happened after this conversation?")
    
    # Limit to 3 questions per episode
    return questions[:3]


if __name__ == "__main__":
    main()
```

---

## 7. JEPA Routing Data Generator (`scripts/generate_jepa_training_data.py`)

```python
"""Generate JEPA routing training data.

Usage:
    python scripts/generate_jepa_training_data.py \
        --db ./data/memory_db \
        --output data/training/jepa/ \
        --num-pairs 5000
"""

import argparse
import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleClient, OracleConfig
from src.training.prompts import jepa_routing_prompt


AVAILABLE_DOMAINS = """
- database: WaveDB, Postgres, HBTrie, SQL, configuration, performance
- coding: Python, Rust, Dart, tree-sitter, AST parsing, code review
- robotics: actuators, sensors, inverse kinematics, control policies
- economics: Spark Ledger, monetary theory, QE, zk-SNARKs
- ai_architecture: neural networks, cognitive systems, memory models
- personal: user preferences, relationships, emotional patterns
"""

AVAILABLE_PATHWAYS = """
- ssm_direct: Answer from working memory. No retrieval needed.
- graph_retrieve: Query the memory graph. Standard retrieval.
- process_exec: Execute a stored process.
- tool_plan: Plan a multi-step tool use strategy.
- conscious_deliberation: Engage System 2 for complex reasoning.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate JEPA routing data")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    parser.add_argument("--output", default="data/training/jepa/", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    store = HippocampalStore(args.db)
    oracle = OracleClient(OracleConfig())
    
    # Generate diverse query patterns
    queries = _generate_diverse_queries(args.num_pairs)
    
    routing_pairs = []
    
    for i in range(0, len(queries), args.batch_size):
        batch = queries[i:i + args.batch_size]
        
        prompts = [
            jepa_routing_prompt(q, AVAILABLE_DOMAINS, AVAILABLE_PATHWAYS)
            for q in batch
        ]
        
        results = oracle.generate_batch(prompts)
        
        for query, result in zip(batch, results):
            routing_pairs.append({
                "query": query,
                "route": result.response,
            })
        
        print(f"  {min(i + args.batch_size, len(queries))}/{len(queries)} "
              f"queries processed (${oracle.total_cost:.2f})")
    
    with open(output_dir / "routing_pairs.jsonl", "w") as f:
        for pair in routing_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"\nGenerated {len(routing_pairs)} routing pairs")
    print(f"Total cost: ${oracle.total_cost:.2f}")
    
    store.close()


def _generate_diverse_queries(num_queries):
    """Generate diverse query patterns for routing training."""
    
    templates = [
        # Factual recall (should route to ssm_direct or 1B model)
        ("What is {entity}?", ["ssm_direct"]),
        ("When did we discuss {topic}?", ["graph_retrieve"]),
        ("Who was involved in {topic}?", ["graph_retrieve"]),
        
        # Basic synthesis (should route to 3B model)
        ("What did {entity} say about {topic}?", ["graph_retrieve"]),
        ("Why was I {tone} about {topic}?", ["graph_retrieve"]),
        ("What happened after {event}?", ["graph_retrieve"]),
        
        # Process execution (should route to process_exec)
        ("Review this code for security issues", ["process_exec"]),
        ("Deploy the latest changes", ["process_exec"]),
        ("Run the test suite and report failures", ["process_exec"]),
        
        # Complex reasoning (should route to conscious_deliberation)
        ("Why did we choose {entity_a} over {entity_b}?", ["conscious_deliberation"]),
        ("What are the implications of {decision}?", ["conscious_deliberation"]),
        ("Design a new approach for {problem}", ["conscious_deliberation"]),
        
        # Cross-domain (should route to multiple domains)
        ("Compare {domain_a} performance with {domain_b} reliability", ["graph_retrieve"]),
        ("How does {domain_a} architecture influence {domain_b} design?", ["conscious_deliberation"]),
    ]
    
    entities = ["Alice", "Bob", "WaveDB", "Postgres", "Python", "HBTrie", "WAL", "API"]
    topics = ["database_design", "configuration", "performance", "security", "api_design"]
    tones = ["frustrated", "excited", "curious"]
    events = ["morphisms", "the optimizer", "the refactor", "the deployment"]
    decisions = ["using WaveDB", "the DEBOUNCED choice", "the cost-based optimizer"]
    problems = ["sync mode configuration", "async performance", "encryption API"]
    domains = ["database", "robotics", "economics"]
    
    queries = []
    for _ in range(num_queries):
        template, _ = random.choice(templates)
        
        query = template.format(
            entity=random.choice(entities),
            entity_a=random.choice(entities),
            entity_b=random.choice(entities),
            topic=random.choice(topics),
            tone=random.choice(tones),
            event=random.choice(events),
            decision=random.choice(decisions),
            problem=random.choice(problems),
            domain_a=random.choice(domains),
            domain_b=random.choice(domains),
        )
        
        queries.append(query)
    
    return queries


if __name__ == "__main__":
    main()
```

---

## 8. Gate Training Data Generator (`scripts/generate_gate_training_data.py`)

```python
"""Generate gate training data for Uncertainty Detector, Aspirational Model, and Self-Model.

Usage:
    python scripts/generate_gate_training_data.py \
        --db ./data/memory_db \
        --output data/training/gates/ \
        --num-examples 50000
"""

import argparse
import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleClient, OracleConfig
from src.training.prompts import (
    uncertainty_detector_prompt,
    aspirational_model_prompt,
    self_model_prompt,
)


def main():
    parser = argparse.ArgumentParser(description="Generate gate training data")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    parser.add_argument("--output", default="data/training/gates/", help="Output directory")
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    store = HippocampalStore(args.db)
    oracle = OracleClient(OracleConfig())
    
    examples_per_gate = args.num_examples // 3
    
    # ── Uncertainty Detector ──
    print("Generating Uncertainty Detector training data...")
    _generate_gate_data(
        oracle, store, output_dir,
        "uncertainty_detector.jsonl",
        uncertainty_detector_prompt,
        examples_per_gate,
        args.batch_size,
        _build_uncertainty_inputs
    )
    
    # ── Aspirational Model ──
    print("\nGenerating Aspirational Model training data...")
    _generate_gate_data(
        oracle, store, output_dir,
        "aspirational_model.jsonl",
        aspirational_model_prompt,
        examples_per_gate,
        args.batch_size,
        _build_aspirational_inputs
    )
    
    # ── Self-Model ──
    print("\nGenerating Self-Model training data...")
    _generate_gate_data(
        oracle, store, output_dir,
        "self_model.jsonl",
        self_model_prompt,
        examples_per_gate,
        args.batch_size,
        _build_self_model_inputs
    )
    
    print(f"\nGate training data generation complete.")
    print(f"Total cost: ${oracle.total_cost:.2f}")
    
    store.close()


def _generate_gate_data(oracle, store, output_dir, filename, prompt_fn, 
                        num_examples, batch_size, input_builder):
    """Generic gate data generator."""
    
    inputs = input_builder(store, num_examples)
    results = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        prompts = [prompt_fn(**inp) for inp in batch]
        batch_results = oracle.generate_batch(prompts)
        
        for inp, result in zip(batch, batch_results):
            results.append({
                "input": inp,
                "label": result.response,
            })
        
        print(f"  {min(i + batch_size, len(inputs))}/{len(inputs)} "
              f"examples (${oracle.total_cost:.2f})")
    
    with open(output_dir / filename, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"  Generated {len(results)} examples")


def _build_uncertainty_inputs(store, count):
    """Build inputs for uncertainty detector training."""
    episodes = _get_all_episodes(store)
    random.shuffle(episodes)
    
    inputs = []
    for ep in episodes[:count]:
        # Build context from related episodes
        related = _get_related_episodes(store, ep["id"], limit=5)
        context = "\n".join(r["summary"] for r in related)
        
        # Generate a query about this episode
        query = _generate_query(ep)
        
        # Simulate retrieval results (sometimes empty, sometimes partial)
        if random.random() < 0.3:
            retrieval = "No results found."
        elif random.random() < 0.5:
            retrieval = f"Found {len(related)} partially relevant episodes."
        else:
            retrieval = f"Found {len(related)} relevant episodes:\n" + \
                       "\n".join(r["summary"] for r in related[:3])
        
        inputs.append({
            "context": context,
            "query": query,
            "retrieval_results": retrieval,
        })
    
    return inputs


def _build_aspirational_inputs(store, count):
    """Build inputs for aspirational model training."""
    episodes = _get_all_episodes(store)
    random.shuffle(episodes)
    
    inputs = []
    for ep in episodes[:count]:
        goal_context = f"Recent topics: {', '.join(ep.get('topics', []))}. "
        goal_context += f"Recent entities: {', '.join(ep.get('entities', [])[:5])}."
        
        actions = [
            f"Encode this episode about {ep.get('topics', ['unknown'])[0]}",
            f"Set a reminder to follow up on {ep.get('entities', ['this'])[0]}",
            f"Explore more about {ep.get('topics', ['this'])[0]}",
            "Skip encoding this routine conversation",
        ]
        
        inputs.append({
            "goal_context": goal_context,
            "candidate_action": random.choice(actions),
        })
    
    return inputs


def _build_self_model_inputs(store, count):
    """Build inputs for self-model training."""
    episodes = _get_all_episodes(store)
    random.shuffle(episodes)
    
    inputs = []
    for ep in episodes[:count]:
        # Knowledge state: what the system knows about this domain
        topic = ep.get("topics", ["unknown"])[0] if ep.get("topics") else "unknown"
        entity_count = len(_get_episodes_by_topic(store, topic))
        
        if entity_count > 10:
            knowledge_state = f"Dense knowledge: {entity_count} episodes about {topic}."
        elif entity_count > 3:
            knowledge_state = f"Moderate knowledge: {entity_count} episodes about {topic}."
        else:
            knowledge_state = f"Sparse knowledge: {entity_count} episodes about {topic}."
        
        # Query that may or may not be answerable
        if random.random() < 0.4:
            query = f"What is the exact {topic} configuration we used?"
        else:
            query = f"What did we discuss about {topic}?"
        
        inputs.append({
            "knowledge_state": knowledge_state,
            "query": query,
        })
    
    return inputs


def _get_all_episodes(store):
    result = store.graph.query().has("predicate", "has_entity").execute()
    episode_ids = list(set(r.subject for r in result))
    
    episodes = []
    for ep_id in episode_ids[:1000]:  # Limit for performance
        ep = store.get_episode(ep_id)
        if ep:
            episodes.append({
                "id": ep.id,
                "summary": ep.summary,
                "entities": ep.entities,
                "topics": ep.topics,
                "tones": ep.tones,
                "decisions": ep.decisions,
            })
    
    return episodes


def _get_related_episodes(store, episode_id, limit=5):
    """Get episodes related to the given episode."""
    ep = store.get_episode(episode_id)
    if not ep or not ep.entities:
        return []
    
    related_ids = set()
    for entity in ep.entities[:3]:
        result = store.graph.query().vertex(f"E:{entity}").in_("in_episode").execute()
        related_ids.update(r.id for r in result)
    
    related_ids.discard(episode_id)
    
    episodes = []
    for rid in list(related_ids)[:limit]:
        rep = store.get_episode(rid)
        if rep:
            episodes.append({"id": rep.id, "summary": rep.summary})
    
    return episodes


def _get_episodes_by_topic(store, topic):
    result = store.graph.query().vertex(f"T:{topic}").in_("has_topic").execute()
    return [r.id for r in result]


def _generate_query(episode):
    """Generate a query about an episode."""
    if episode.get("entities"):
        return f"What did {episode['entities'][0]} say?"
    if episode.get("topics"):
        return f"What did we discuss about {episode['topics'][0]}?"
    return "What was this conversation about?"


if __name__ == "__main__":
    main()
```

---

## 9. Code-Aware Synthetic Data Generator (`scripts/generate_code_aware_data.py`)

```python
"""Generate synthetic code-aware training examples.

Usage:
    python scripts/generate_code_aware_data.py \
        --output data/training/code_aware/ \
        --num-examples 2000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.oracle_labeling import OracleClient, OracleConfig
from src.training.prompts import code_aware_synthetic_prompt
from src.memory.ontology import SEED_ONTOLOGY


CODE_ONTOLOGY_FRAGMENT = """
CodeArtifact: File, Module, Package, Class, Interface, Function, Method, 
  Constructor, Property, Variable, Parameter, Type, Decorator, Lambda
VersionControl: Commit, Branch, Tag, PullRequest, Merge, Conflict
Issue: Bug, Feature, Regression, BreakingChange
Test: UnitTest, IntegrationTest, Mock, Fixture, Coverage
Architecture: Service, Microservice, Middleware, Adapter, Factory, Proxy
API: Endpoint, Route, Controller, Guard, DTO, Schema
Data: Database, Table, Column, Index, ForeignKey, Migration, Transaction
Configuration: EnvironmentVariable, ConfigFile, Secret, FeatureFlag
Infrastructure: Container, Pod, Cluster, LoadBalancer, CDN
Deployment: Pipeline, Stage, Job, Artifact, Rollback
Observability: Log, Metric, Alert, Incident, PostMortem
"""

DOMAINS = [
    "authentication system with JWT tokens",
    "database migration and schema design",
    "API rate limiting and throttling",
    "CI/CD pipeline configuration",
    "microservice deployment with Kubernetes",
    "error handling and logging strategy",
    "code review and testing workflow",
    "configuration management across environments",
    "performance optimization and profiling",
    "security audit and vulnerability patching",
]


def main():
    parser = argparse.ArgumentParser(description="Generate code-aware data")
    parser.add_argument("--output", default="data/training/code_aware/", help="Output directory")
    parser.add_argument("--num-examples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    oracle = OracleClient(OracleConfig())
    
    examples = []
    
    for i in range(0, args.num_examples, args.batch_size):
        batch_size = min(args.batch_size, args.num_examples - i)
        
        prompts = []
        for _ in range(batch_size):
            import random
            domain = random.choice(DOMAINS)
            prompts.append(code_aware_synthetic_prompt(domain, CODE_ONTOLOGY_FRAGMENT))
        
        results = oracle.generate_batch(prompts)
        
        for result in results:
            examples.append(result.response)
        
        print(f"  {min(i + args.batch_size, args.num_examples)}/{args.num_examples} "
              f"examples (${oracle.total_cost:.2f})")
    
    with open(output_dir / "synthetic_examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\nGenerated {len(examples)} code-aware synthetic examples")
    print(f"Total cost: ${oracle.total_cost:.2f}")


if __name__ == "__main__":
    main()
```

---

## 10. Training Data Validator (`scripts/validate_training_data.py`)

```python
"""Validate generated training data for quality and completeness.

Usage:
    python scripts/validate_training_data.py --data-dir data/training/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--data-dir", default="data/training/", help="Training data directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    all_valid = True
    
    # ── GNN data ──
    print("=" * 60)
    print("Validating GNN training data...")
    gnn_dir = data_dir / "gnn"
    
    for task in ["salience", "cluster", "link_prediction", "anomaly", "ontology"]:
        file_path = gnn_dir / f"{task}_labels.jsonl"
        if not file_path.exists():
            print(f"  ❌ Missing: {file_path}")
            all_valid = False
            continue
        
        with open(file_path) as f:
            lines = f.readlines()
        
        if len(lines) < 1000:
            print(f"  ⚠️  {task}: Only {len(lines)} examples (target: 1000+)")
        else:
            print(f"  ✅ {task}: {len(lines)} examples")
        
        # Check parseability
        parse_errors = 0
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                if "labels" not in data and "response" not in str(data):
                    parse_errors += 1
            except json.JSONDecodeError:
                parse_errors += 1
        
        if parse_errors > 0:
            print(f"     ⚠️  {parse_errors} parse errors")
    
    # ── Bonsai data ──
    print("\nValidating Bonsai training data...")
    bonsai_dir = data_dir / "bonsai"
    
    for task in ["query_planning_pairs", "relation_extraction_pairs"]:
        file_path = bonsai_dir / f"{task}.jsonl"
        if not file_path.exists():
            print(f"  ❌ Missing: {file_path}")
            all_valid = False
            continue
        
        with open(file_path) as f:
            lines = f.readlines()
        
        print(f"  ✅ {task}: {len(lines)} examples")
    
    # ── JEPA data ──
    print("\nValidating JEPA training data...")
    jepa_file = data_dir / "jepa" / "routing_pairs.jsonl"
    
    if jepa_file.exists():
        with open(jepa_file) as f:
            lines = f.readlines()
        print(f"  ✅ routing_pairs: {len(lines)} examples")
        
        # Check route diversity
        pathways = Counter()
        for line in lines:
            data = json.loads(line)
            route = data.get("route", {})
            pathway = route.get("pathway", "unknown")
            pathways[pathway] += 1
        
        print(f"     Pathway distribution: {dict(pathways)}")
    else:
        print(f"  ❌ Missing: {jepa_file}")
        all_valid = False
    
    # ── Gate data ──
    print("\nValidating gate training data...")
    gates_dir = data_dir / "gates"
    
    for gate in ["uncertainty_detector", "aspirational_model", "self_model"]:
        file_path = gates_dir / f"{gate}.jsonl"
        if not file_path.exists():
            print(f"  ❌ Missing: {file_path}")
            all_valid = False
            continue
        
        with open(file_path) as f:
            lines = f.readlines()
        
        print(f"  ✅ {gate}: {len(lines)} examples")
    
    # ── Code-aware data ──
    print("\nValidating code-aware data...")
    code_file = data_dir / "code_aware" / "synthetic_examples.jsonl"
    
    if code_file.exists():
        with open(code_file) as f:
            lines = f.readlines()
        print(f"  ✅ synthetic_examples: {len(lines)} examples")
    else:
        print(f"  ❌ Missing: {code_file}")
        all_valid = False
    
    # ── Summary ──
    print(f"\n{'=' * 60}")
    if all_valid:
        print("✅ All training data validated.")
    else:
        print("❌ Some training data is missing or invalid.")
    
    # Total size
    total_size = sum(
        f.stat().st_size for f in data_dir.rglob("*.jsonl") if f.is_file()
    )
    print(f"Total training data size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
```

---

## 11. Checkpoint Criteria

Phase 1d is complete when:

- [ ] GNN training data: 4,000+ subgraphs labeled across all five tasks
- [ ] GNN labels parseable and validated (no JSON errors)
- [ ] Bonsai query planning pairs: 5,000+ (prompt, query) pairs
- [ ] Bonsai relation extraction pairs: 2,000+ (text, relations) pairs
- [ ] JEPA routing pairs: 5,000+ with diverse pathway distribution
- [ ] Uncertainty Detector gate examples: 50,000+ labeled decisions
- [ ] Aspirational Model gate examples: 50,000+ labeled decisions
- [ ] Self-Model gate examples: 50,000+ labeled decisions
- [ ] Code-aware synthetic examples: 2,000+ with code ontology types
- [ ] All output files are valid JSONL (no parse errors)
- [ ] Quality validation report generated
- [ ] Total Oracle cost within budget (~$20)
- [ ] Oracle cache populated (reduces cost for re-runs)

---

## 12. Implementation Order

1. **Oracle client** — `oracle_labeling.py` with caching, retries, cost tracking
2. **Prompt library** — `prompts.py` with all Oracle prompts
3. **GNN training data** — largest dataset, run first (longest Oracle time)
4. **Bonsai training data** — query planning + relation extraction
5. **JEPA routing data** — diverse query patterns
6. **Gate training data** — three gates, can run in parallel
7. **Code-aware synthetic data** — smallest dataset, run last
8. **Validation** — `validate_training_data.py` across all outputs
9. **Quality report** — aggregate statistics, cost breakdown

---

## 13. What Phase 1d Does NOT Do

- **Does not train any models.** This phase only generates training data.
- **Does not require Phase 1c.** Can run in parallel with retrieval refinements.
- **Does not modify the memory graph.** Read-only access to WaveDB.
- **Does not require the Oracle after completion.** All training data is cached to disk.

---

## 14. Next Phase

After Phase 1d checkpoint is met, proceed to **Phase 2a: JEPA-Gated SSM Backbone** which trains the shared SSM+JEPA weights on the Oracle-generated cognitive state sequences. The training data from Phase 1d is the input to all downstream training phases.

---

Begin with step 1. The Oracle client is the foundation — get caching and cost tracking right before generating any data. Run the GNN data generation first as it's the largest and most expensive. Monitor Oracle costs throughout. If costs exceed budget, reduce `--num-examples` for gate data (gates can train on fewer examples).