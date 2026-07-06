# Phase 1b: Storage & Retrieval — Implementation Plan for Claude Code

## Overview

**Goal:** Populate WaveDB with real conversation corpora, build the graph traversal engine for pattern completion, implement the query planner, and deliver working Mode A generation — the context window adapter that lets any LLM use the memory system.

**What "done" looks like:** A corpus of 1,000+ conversations encoded in WaveDB. A query planner that converts natural language questions into structured graph queries. A graph traversal engine that executes those queries and returns ranked episodes. A Mode A generator that builds context strings from retrieved episodes and produces responses via any LLM API. An integration test demonstrating end-to-end retrieval: "What was I frustrated about?" → relevant episodes → context → LLM response.

**Prerequisite:** Phase 1a complete. Encoding pipeline operational. WaveDB store working. Sample conversations passing extraction quality thresholds.

**Duration estimate:** 5-7 days of focused implementation.

---

## 1. What Phase 1b Delivers

Artifact	Description	Consumer
**Populated WaveDB**	1,000+ episodes from DialogSum + SAMSum + hand-crafted conversations	All subsequent phases
**Graph Traversal Engine**	Pattern completion via Gremlin-style queries against the graph layer	Retrieval pipeline
**Query Planner**	Bonsai-based NL → structured query conversion	Retrieval pipeline
**Vector Index**	FAISS index over episode summary embeddings	Semantic search
**Mode A Generator**	Context window adapter for any LLM	End-to-end demonstration
**Corpus processing reports**	Extraction quality metrics at scale	Quality measurement
**Oracle labeling scripts**	Prompts and pipeline for GNN training data generation	Phase 1d

**What the user gets:** A database you can talk to. The system retrieves relevant episodes, builds context, and feeds it to an LLM. The LLM responds as if it remembers everything. But there's no subconscious routing, no consolidation, no uncertainty detection, no procedural memory — those come in later phases.

---

## 2. Updated Project Structure

```plaintext
hippocampal-memory/
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                    # Updated with Phase 2-4 placeholders
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py                 # Updated with persistence fields
│   │   ├── episode.py               # Updated with downstream fields
│   │   └── ontology.py              # Seed ontology (unchanged from 1a)
│   ├── encoding/
│   │   ├── __init__.py
│   │   ├── gliner_extractor.py
│   │   ├── bonsai_relations.py
│   │   └── encoder.py
│   ├── retrieval/                   # NEW
│   │   ├── __init__.py
│   │   ├── query_planner.py         # Bonsai NL → structured query
│   │   ├── graph_traversal.py       # Pattern completion engine
│   │   ├── vector_search.py         # FAISS semantic search
│   │   └── retriever.py             # Orchestrator
│   ├── generation/                  # NEW
│   │   ├── __init__.py
│   │   └── mode_a.py                # Context window adapter
│   └── training/                    # NEW (prep for Phase 1d)
│       ├── __init__.py
│       └── oracle_labeling.py       # Oracle prompts for GNN data
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_episode.py
│   ├── test_store.py
│   ├── test_gliner_extractor.py
│   ├── test_bonsai_relations.py
│   ├── test_encoder.py
│   ├── test_query_planner.py        # NEW
│   ├── test_graph_traversal.py      # NEW
│   ├── test_retriever.py            # NEW
│   └── test_mode_a.py               # NEW
├── scripts/
│   ├── process_corpus.py            # Updated for scale
│   ├── build_vector_index.py        # NEW
│   └── generate_training_data.py    # NEW (Oracle labeling)
├── data/
│   ├── sample_conversations.jsonl
│   ├── test_corpus/
│   └── corpora/                     # NEW — downloaded datasets
└── notebooks/
    ├── extraction_quality.ipynb
    └── retrieval_quality.ipynb       # NEW
```

---

## 3. Updated Data Models

### 3.1 Episode — Add Downstream Fields (`src/memory/episode.py`)

The Episode model is the contract between Phase 1a and everything that follows. Add fields that later phases will populate, with safe defaults for Phase 1b:

```python
@dataclass
class Episode:
    # ── Existing fields (Phase 1a) ──
    id: str
    timestamp: str
    summary: str
    full_text: str
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    tones: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    follows: Optional[str] = None
    salience: float = 0.5
    state: str = "current"
    validity_start: Optional[str] = None
    validity_end: Optional[str] = None
    
    # ── New: Persistence & retrieval tracking (Phase 2-4) ──
    retrieval_count: int = 0              # Reconsolidation counting
    ltp_phase: str = "early"             # "early" | "late"
    consolidation_window_start: Optional[str] = None
    utility_decay_rate: float = 0.01     # Base decay rate
    retrieval_timestamps: list[str] = field(default_factory=list)
    saturation_flags: int = 0
    
    # ── New: Embedding for vector search ──
    summary_embedding: Optional[list[float]] = None
```

### 3.2 Configuration — Add Phase 2-4 Placeholders (`src/config.py`)

```python
@dataclass
class Config:
    # WaveDB
    db_path: str = os.getenv("HIPPOCAMPAL_DB_PATH", "./data/memory_db")
    lru_memory_mb: int = 100
    wal_sync_mode: str = "debounced"
    
    # Phase 1a: Extraction
    gliner2_model: str = "fastino/gliner2-base-v1"
    gliner_decoder_model: str = "knowledgator/gliner-decoder-base-v1.0"
    extraction_threshold: float = 0.3
    bonsai_model: str = os.getenv("BONSAI_MODEL", "gpt-4o-mini")
    bonsai_temperature: float = 0.1
    
    # Phase 1a: Encoding defaults
    episode_salience_default: float = 0.5
    discovery_buffer_threshold: int = 10
    
    # Phase 1b: Retrieval
    default_retrieval_limit: int = 5
    max_context_tokens: int = 4000
    embedding_model: str = "text-embedding-3-small"  # For vector search
    vector_index_type: str = "faiss"                  # faiss | usearch
    
    # Phase 1b: Mode A generation
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    generation_temperature: float = 0.7
    
    # Phase 2+: JEPA-Gated SSM (not yet active)
    ssm_state_dim: int = 512
    jepa_backbone_model: str = "mamba-2.8b"
    
    # Phase 3+: GNN Consolidator (not yet active)
    gnn_hidden_dim: int = 256
    
    # Phase 4+: Instance-specific gates (not yet active)
    gate_hidden_dim: int = 128
    
    # Forgetting system (not yet active)
    saturation_threshold: int = 5
    boost_half_life_days: float = 7.0
    min_decay_rate: float = 0.001
    
    # Paths
    data_dir: Path = Path("./data")
    sample_conversations: Path = Path("./data/sample_conversations.jsonl")
    corpora_dir: Path = Path("./data/corpora")
```

### 3.3 Store — Add Persistence for New Fields (`src/memory/store.py`)

Update `encode_episode` and `get_episode` to handle the new Episode fields:

```python
def encode_episode(self, episode: Episode):
    # ... existing HBTrie writes ...
    
    # ── New: persistence-related fields ──
    self.db.put_sync(f"ep/{episode.id}/retrieval_count", str(episode.retrieval_count))
    self.db.put_sync(f"ep/{episode.id}/ltp_phase", episode.ltp_phase)
    self.db.put_sync(f"ep/{episode.id}/decay_rate", str(episode.utility_decay_rate))
    
    # ── New: embedding ──
    if episode.summary_embedding:
        self.db.put_sync(
            f"ep/{episode.id}/embedding",
            json.dumps(episode.summary_embedding)
        )
    
    # ... existing graph writes ...

def get_episode(self, episode_id: str) -> Episode | None:
    # ... existing HBTrie reads ...
    
    # ── New: persistence fields ──
    retrieval_count = self.db.get_sync(f"ep/{episode_id}/retrieval_count") or b"0"
    ltp_phase = self.db.get_sync(f"ep/{episode_id}/ltp_phase") or b"early"
    decay_rate = self.db.get_sync(f"ep/{episode_id}/decay_rate") or b"0.01"
    
    return Episode(
        # ... existing fields ...
        retrieval_count=int(retrieval_count.decode() if isinstance(retrieval_count, bytes) else retrieval_count),
        ltp_phase=ltp_phase.decode() if isinstance(ltp_phase, bytes) else ltp_phase,
        utility_decay_rate=float(decay_rate.decode() if isinstance(decay_rate, bytes) else decay_rate),
    )
```

---

## 4. Corpus Ingestion at Scale

### 4.1 Updated Corpus Processor (`scripts/process_corpus.py`)

Extend the Phase 1a script to handle real datasets with progress tracking, error recovery, and quality metrics:

```python
"""Process conversation corpora at scale with progress tracking and quality metrics.

Usage:
    python scripts/process_corpus.py \
        --input data/corpora/dialogsum.jsonl \
        --db ./data/memory_db \
        --limit 1000 \
        --report ingestion_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.encoding.encoder import HippocampalEncoder


def main():
    parser = argparse.ArgumentParser(description="Process a conversation corpus")
    parser.add_argument("--input", required=True, help="JSONL file with conversations")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    parser.add_argument("--limit", type=int, help="Max conversations to process")
    parser.add_argument("--report", help="Output path for ingestion quality report")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    store = HippocampalStore(args.db)
    encoder = HippocampalEncoder(store)
    
    # Track metrics
    metrics = {
        "conversations_processed": 0,
        "episodes_encoded": 0,
        "errors": 0,
        "entity_counts": Counter(),
        "topic_counts": Counter(),
        "tone_counts": Counter(),
        "decision_counts": Counter(),
        "start_time": time.time(),
    }
    
    # Resume support
    start_line = 0
    if args.resume:
        checkpoint = _load_checkpoint(args.db)
        if checkpoint:
            start_line = checkpoint["last_line"]
            metrics = checkpoint["metrics"]
            encoder.episode_counter = checkpoint["episode_counter"]
            encoder.last_episode_id = checkpoint["last_episode_id"]
            print(f"Resuming from line {start_line}")
    
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if args.limit and metrics["conversations_processed"] >= args.limit:
                break
            
            try:
                conv = json.loads(line)
                turns = conv.get("turns", [])
                
                if not turns:
                    continue
                
                episodes = encoder.encode_conversation(turns)
                metrics["conversations_processed"] += 1
                metrics["episodes_encoded"] += len(episodes)
                
                # Track extraction quality
                for ep in episodes:
                    metrics["entity_counts"].update(ep.entities)
                    metrics["topic_counts"].update(ep.topics)
                    metrics["tone_counts"].update(ep.tones)
                    metrics["decision_counts"].update(ep.decisions)
                
            except Exception as e:
                metrics["errors"] += 1
                print(f"Error at line {i}: {e}")
            
            # Progress + checkpoint
            if metrics["conversations_processed"] % 100 == 0:
                elapsed = time.time() - metrics["start_time"]
                rate = metrics["conversations_processed"] / elapsed if elapsed > 0 else 0
                print(f"Processed {metrics['conversations_processed']} conversations "
                      f"({metrics['episodes_encoded']} episodes) "
                      f"at {rate:.1f} conv/sec "
                      f"({metrics['errors']} errors)")
                
                _save_checkpoint(args.db, i + 1, metrics, encoder)
    
    # Final report
    metrics["elapsed_seconds"] = time.time() - metrics["start_time"]
    metrics["entity_counts"] = dict(metrics["entity_counts"].most_common(50))
    metrics["topic_counts"] = dict(metrics["topic_counts"].most_common(50))
    metrics["tone_counts"] = dict(metrics["tone_counts"].most_common(50))
    metrics["decision_counts"] = dict(metrics["decision_counts"].most_common(50))
    
    if args.report:
        with open(args.report, "w") as f:
            json.dump(metrics, f, indent=2)
    
    print(f"\nDone. {metrics['conversations_processed']} conversations, "
          f"{metrics['episodes_encoded']} episodes, "
          f"{metrics['errors']} errors, "
          f"{metrics['elapsed_seconds']:.0f}s elapsed")
    
    store.close()


def _save_checkpoint(db_path, line, metrics, encoder):
    checkpoint = {
        "last_line": line,
        "metrics": metrics,
        "episode_counter": encoder.episode_counter,
        "last_episode_id": encoder.last_episode_id,
    }
    with open(f"{db_path}/.checkpoint.json", "w") as f:
        json.dump(checkpoint, f)


def _load_checkpoint(db_path):
    try:
        with open(f"{db_path}/.checkpoint.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()
```

### 4.2 Dataset Preparation Scripts

Create a script to download and prepare the corpora:

```bash
# scripts/download_corpora.sh
#!/bin/bash
# Download DialogSum, SAMSum, and Memory-Traces from HuggingFace

python -c "
from datasets import load_dataset

# DialogSum
ds = load_dataset('knkarthick/dialogsum')
ds['train'].to_json('data/corpora/dialogsum_train.jsonl', orient='records', lines=True)

# SAMSum
ds = load_dataset('samsum')
ds['train'].to_json('data/corpora/samsum_train.jsonl', orient='records', lines=True)

# Memory-Traces
ds = load_dataset('Cossale/memory-traces')
ds['train'].to_json('data/corpora/memory_traces_train.jsonl', orient='records', lines=True)

print('Corpora downloaded and converted to JSONL.')
"
```

---

## 5. Graph Traversal Engine (`src/retrieval/graph_traversal.py`)

This is the core retrieval mechanism — pattern completion via graph traversal:

```python
"""Pattern completion via graph traversal over the WaveDB Graph layer."""

from typing import Optional
from datetime import datetime, timedelta
from ..memory.store import HippocampalStore


class GraphTraversal:
    """
    Executes structured queries against the memory graph.
    
    This is the hippocampal pattern completion mechanism:
    partial cue → graph traversal → full reconstruction.
    
    NOTE: The WaveDB graph query API calls in this module (e.g., .vertex(), .in_(), 
    .out(), .has(), .execute()) should be adapted to match the actual WaveDB Python 
    bindings API. The patterns are correct; the syntax may need adjustment.
    """
    
    def __init__(self, store: HippocampalStore):
        self.store = store
    
    def retrieve(self, query_plan: dict) -> list[dict]:
        """
        Execute a structured query.
        
        query_plan:
        {
            "entities": ["Alice", "WaveDB"],
            "topics": ["database_design"],
            "tones": ["frustrated"],
            "entity_mode": "union",        # "intersection" | "union"
            "temporal_after": "morphism",  # keyword or null
            "temporal_before": null,
            "temporal_filter": "last_week", # "today" | "this_week" | "last_week" | "this_month" | null
            "limit": 5
        }
        
        Returns list of:
        {
            "episode_id": "ep_0047",
            "score": 18.5,
            "summary": "...",
            "text": "...",
            "timestamp": "...",
            "entities": [...],
            "topics": [...],
            "tones": [...],
        }
        """
        entities = query_plan.get("entities", [])
        topics = query_plan.get("topics", [])
        tones = query_plan.get("tones", [])
        entity_mode = query_plan.get("entity_mode", "union")
        temporal_after = query_plan.get("temporal_after")
        temporal_before = query_plan.get("temporal_before")
        temporal_filter = query_plan.get("temporal_filter")
        limit = query_plan.get("limit", 5)
        
        # 1. Find candidate episodes
        candidates = self._find_candidates(entities, topics, tones, entity_mode)
        
        # 2. Apply temporal filter
        if temporal_filter:
            candidates = self._filter_temporal(candidates, temporal_filter)
        
        # 3. Handle temporal chain queries
        if temporal_after:
            candidates = self._follow_temporal_chain(candidates, temporal_after, "forward")
        elif temporal_before:
            candidates = self._follow_temporal_chain(candidates, temporal_before, "backward")
        
        # 4. Score and rank
        scored = self._score_candidates(candidates, entities, topics, tones)
        
        # 5. Load full episodes from HBTrie
        results = []
        for score, ep_id in scored[:limit]:
            ep = self.store.get_episode(ep_id)
            if ep:
                results.append({
                    "episode_id": ep.id,
                    "score": score,
                    "summary": ep.summary,
                    "text": ep.full_text,
                    "timestamp": ep.timestamp,
                    "entities": ep.entities,
                    "topics": ep.topics,
                    "tones": ep.tones,
                })
        
        return results
    
    def _find_candidates(self, entities, topics, tones, entity_mode):
        """
        Find candidate episodes matching the query.
        
        Entities: INTERSECTION or UNION depending on entity_mode.
        Topics: UNION (match ANY topic).
        Tones: UNION (match ANY tone).
        """
        all_eps = self._get_all_episode_ids()
        candidates = set(all_eps)
        
        # Filter by entities
        if entities:
            if entity_mode == "intersection":
                entity_sets = []
                for entity in entities:
                    eps = set(self._get_episodes_by_entity(entity))
                    if eps:
                        entity_sets.append(eps)
                if entity_sets:
                    candidates &= entity_sets[0]
                    for s in entity_sets[1:]:
                        candidates &= s
                else:
                    return []
            else:  # union
                entity_eps = set()
                for entity in entities:
                    entity_eps |= set(self._get_episodes_by_entity(entity))
                if entity_eps:
                    candidates &= entity_eps
                else:
                    return []
        
        # Filter by topics (union)
        if topics:
            topic_eps = set()
            for topic in topics:
                topic_eps |= set(self._get_episodes_by_topic(topic))
            if topic_eps:
                candidates &= topic_eps
            else:
                return []
        
        # Filter by tones (union)
        if tones:
            tone_eps = set()
            for tone in tones:
                tone_eps |= set(self._get_episodes_by_tone(tone))
            if tone_eps:
                candidates &= tone_eps
            else:
                return []
        
        return list(candidates)
    
    def _score_candidates(self, candidate_ids, entities, topics, tones):
        """
        Score candidates by match quality + recency.
        
        Entity matches: 10 points each
        Topic matches: 5 points each
        Tone matches: 3 points each
        Recency bonus: 0.1 * episode_number
        
        NOTE: These are heuristic weights. Phase 3 replaces this with
        GNN-learned salience scoring.
        """
        scored = []
        for ep_id in candidate_ids:
            ep = self.store.get_episode(ep_id)
            if not ep:
                continue
            
            score = 0.0
            
            if entities:
                ep_entities = set(ep.entities)
                query_entities = set(entities)
                score += len(ep_entities & query_entities) * 10
            
            if topics:
                ep_topics = set(ep.topics)
                query_topics = set(topics)
                score += len(ep_topics & query_topics) * 5
            
            if tones:
                ep_tones = set(ep.tones)
                query_tones = set(tones)
                score += len(ep_tones & query_tones) * 3
            
            ep_num = int(ep.id.split("_")[1]) if "_" in ep.id else 0
            score += ep_num * 0.1
            
            scored.append((score, ep_id))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    
    def _get_all_episode_ids(self) -> list[str]:
        """
        Get all episode IDs from the graph.
        
        NOTE: Adapt to actual WaveDB graph query API. The pattern is:
        find all nodes that have outgoing 'has_entity' edges (all episodes
        have at least one entity). If the API uses a different query pattern,
        adjust accordingly.
        """
        result = self.store.graph.query() \
            .has("predicate", "has_entity") \
            .execute()
        return list(set(r.subject for r in result))
    
    def _get_episodes_by_entity(self, entity: str) -> list[str]:
        """Get episode IDs mentioning an entity."""
        result = self.store.graph.query() \
            .vertex(f"E:{entity}") \
            .in_("in_episode") \
            .execute()
        return [r.id for r in result]
    
    def _get_episodes_by_topic(self, topic: str) -> list[str]:
        """Get episode IDs with a topic."""
        result = self.store.graph.query() \
            .vertex(f"T:{topic}") \
            .in_("has_topic") \
            .execute()
        return [r.id for r in result]
    
    def _get_episodes_by_tone(self, tone: str) -> list[str]:
        """Get episode IDs with an affective tone."""
        result = self.store.graph.query() \
            .vertex(f"A:{tone}") \
            .in_("has_tone") \
            .execute()
        return [r.id for r in result]
    
    def _filter_temporal(self, candidate_ids, temporal_filter):
        """Filter episodes by time range."""
        now = datetime.now()
        
        if temporal_filter == "today":
            cutoff = now - timedelta(days=1)
        elif temporal_filter == "this_week":
            cutoff = now - timedelta(days=7)
        elif temporal_filter == "last_week":
            cutoff = now - timedelta(days=14)
            start = now - timedelta(days=7)
        elif temporal_filter == "this_month":
            cutoff = now - timedelta(days=30)
        else:
            return candidate_ids
        
        filtered = []
        for ep_id in candidate_ids:
            ep = self.store.get_episode(ep_id)
            if not ep:
                continue
            ts = datetime.fromisoformat(ep.timestamp)
            
            if temporal_filter == "last_week":
                if start <= ts <= cutoff:
                    filtered.append(ep_id)
            else:
                if ts >= cutoff:
                    filtered.append(ep_id)
        
        return filtered
    
    def _follow_temporal_chain(self, candidate_ids, keyword, direction):
        """
        Follow temporal chain from anchor episode.
        
        1. Find the episode containing the keyword
        2. Follow 'follows' edges forward or backward
        
        NOTE: This uses 'follows' edges for short-range traversal.
        Phase 1c adds timestamp range queries for long chains.
        """
        anchor_id = None
        for ep_id in candidate_ids:
            ep = self.store.get_episode(ep_id)
            if ep and keyword.lower() in ep.summary.lower():
                anchor_id = ep_id
                break
        
        if not anchor_id:
            return candidate_ids
        
        chain_ids = [anchor_id]
        current = anchor_id
        
        for _ in range(5):
            if direction == "forward":
                result = self.store.graph.query() \
                    .vertex(current) \
                    .in_("follows") \
                    .execute()
            else:
                result = self.store.graph.query() \
                    .vertex(current) \
                    .out("follows") \
                    .execute()
            
            if not result:
                break
            current = result[0].id
            chain_ids.append(current)
        
        return chain_ids
```

---

## 6. Query Planner (`src/retrieval/query_planner.py`)

```python
"""Bonsai-based query planning: natural language → structured graph query."""

import json
import openai


BONSAI_QUERY_PROMPT = """Convert this question into a structured memory query.
Return ONLY valid JSON, no other text.

The memory graph stores episodes with these attributes:
- entities: [Person, Project, Technology, Concept]
- topics: [database_design, configuration, graph_database, performance, 
           decision_making, ai_architecture, api_design, security]
- tones: [frustrated, excited, curious, neutral]
- decisions: specific choices made (e.g., "use_hbtrie", "add_optimizer")
- temporal: episodes linked by "follows" edges

Query parameters:
- entities: list of entities to search for
- topics: list of topics to filter by
- tones: list of emotional tones to filter by
- entity_mode: "intersection" (episodes containing ALL entities) or 
               "union" (episodes containing ANY entity)
- temporal_after: if the question asks "what happened after X", the 
                  keyword to find the anchor episode, or null
- temporal_before: if the question asks "what led up to X", the keyword, 
                   or null
- temporal_filter: "today", "this_week", "last_week", "this_month", or null
- limit: max episodes to return (default 5)

IMPORTANT RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union" 
  (NOT intersection — "Alice and I" means episodes involving Alice)
- "What did Alice say about databases?" → entities=["Alice"], 
  topics=["database_design"], entity_mode="union"
- "What happened after we implemented morphisms?" → temporal_after="morphism"
- "Why did we choose X over Y?" → topics=["decision_making"], 
  entities=["X", "Y"], entity_mode="union"
- If the question is about a specific person's opinion, entity_mode is 
  "union" (episodes involving that person)
- If the question is about when two specific things were discussed 
  TOGETHER, entity_mode is "intersection"

Question: {prompt}

Return ONLY valid JSON:
{{"entities": [], "topics": [], "tones": [], "entity_mode": "union", 
  "temporal_after": null, "temporal_before": null, 
  "temporal_filter": null, "limit": 5}}"""


class BonsaiQueryPlanner:
    """Converts natural language questions into structured query parameters."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
    
    def plan(self, prompt: str) -> dict:
        """Plan a query from a natural language prompt."""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": BONSAI_QUERY_PROMPT.format(prompt=prompt)
            }],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        return json.loads(response.choices[0].message.content)
```

---

## 7. Vector Search (`src/retrieval/vector_search.py`)

```python
"""FAISS-based semantic similarity search over episode embeddings."""

import json
import numpy as np
import faiss
from ..memory.store import HippocampalStore


class VectorSearch:
    """
    Semantic similarity search over episode summary embeddings.
    Complementary to graph traversal — answers "what's similar to this?"
    """
    
    def __init__(self, store: HippocampalStore, embedding_dim: int = 1536):
        self.store = store
        self.embedding_dim = embedding_dim
        self.index = None
        self.episode_ids = []
    
    def build_index(self):
        """Build FAISS index from all episode embeddings in the store."""
        embeddings = []
        self.episode_ids = []
        
        all_eps = self._get_all_episode_ids()
        
        for ep_id in all_eps:
            emb = self._get_embedding(ep_id)
            if emb is not None:
                embeddings.append(emb)
                self.episode_ids.append(ep_id)
        
        if not embeddings:
            return
        
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        print(f"Built vector index with {len(embeddings)} embeddings")
    
    def search(self, query_embedding: list[float], k: int = 10) -> list[dict]:
        """Search for episodes semantically similar to the query."""
        if self.index is None:
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, min(k, len(self.episode_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.episode_ids):
                results.append({
                    "episode_id": self.episode_ids[idx],
                    "score": float(score),
                })
        
        return results
    
    def _get_all_episode_ids(self) -> list[str]:
        result = self.store.graph.query() \
            .has("predicate", "has_entity") \
            .execute()
        return list(set(r.subject for r in result))
    
    def _get_embedding(self, episode_id: str) -> np.ndarray | None:
        emb_str = self.store.db.get_sync(f"ep/{episode_id}/embedding")
        if not emb_str:
            return None
        
        if isinstance(emb_str, bytes):
            emb_str = emb_str.decode()
        
        emb = json.loads(emb_str)
        return np.array(emb, dtype=np.float32)
```

---

## 8. Retriever Orchestrator (`src/retrieval/retriever.py`)

```python
"""Orchestrates the full retrieval pipeline: plan → traverse → load."""

from .query_planner import BonsaiQueryPlanner
from .graph_traversal import GraphTraversal
from .vector_search import VectorSearch
from ..memory.store import HippocampalStore


class HippocampalRetriever:
    """
    Full retrieval pipeline.
    
    1. Bonsai plans the query (NL → structured parameters)
    2. Graph traversal finds candidate episodes
    3. Vector search provides semantic fallback
    4. Results are loaded from HBTrie and ranked
    
    Phase 1b context strategy: Fixed top 5 episodes, full text, hard cutoff
    at token limit. Phase 2.5 adds SSM chunking and JEPA presentation gating.
    """
    
    def __init__(self, store: HippocampalStore, 
                 planner_model: str = "gpt-4o-mini"):
        self.store = store
        self.planner = BonsaiQueryPlanner(model=planner_model)
        self.traversal = GraphTraversal(store)
        self.vector_search = VectorSearch(store)
    
    def retrieve(self, prompt: str, use_semantic: bool = True) -> list[dict]:
        """
        Retrieve relevant episodes for a prompt.
        
        Args:
            prompt: Natural language question
            use_semantic: Whether to fall back to semantic search if graph
                         traversal returns few results
        
        Returns:
            List of episode dicts with scores, summaries, text, metadata
        """
        query_plan = self.planner.plan(prompt)
        results = self.traversal.retrieve(query_plan)
        
        if use_semantic and len(results) < 3:
            semantic_results = self._semantic_fallback(prompt, query_plan)
            
            existing_ids = {r["episode_id"] for r in results}
            for sr in semantic_results:
                if sr["episode_id"] not in existing_ids:
                    results.append(sr)
                    existing_ids.add(sr["episode_id"])
            
            results.sort(key=lambda r: r["score"], reverse=True)
        
        return results
    
    def _semantic_fallback(self, prompt, query_plan):
        """Fall back to semantic search."""
        query_embedding = self._embed(prompt)
        semantic_results = self.vector_search.search(query_embedding, k=10)
        
        results = []
        for sr in semantic_results:
            ep = self.store.get_episode(sr["episode_id"])
            if ep:
                results.append({
                    "episode_id": ep.id,
                    "score": sr["score"] * 0.5,
                    "summary": ep.summary,
                    "text": ep.full_text,
                    "timestamp": ep.timestamp,
                    "entities": ep.entities,
                    "topics": ep.topics,
                    "tones": ep.tones,
                })
        
        return results
    
    def build_context_string(self, episodes: list[dict], max_tokens: int = 4000) -> str:
        """
        Build a context string from retrieved episodes for Mode A generation.
        
        Structured format is more efficient than raw text — the LLM doesn't
        have to infer that Alice is a person or that the tone was frustrated.
        
        Phase 1b strategy: Fixed top N episodes, full text, hard cutoff.
        Phase 2.5 adds SSM chunking with primary/compressed split.
        """
        parts = [
            "You have access to relevant past conversations.",
            "Each is formatted as [Episode ID | Date]: Summary with metadata.",
            "Use this context to answer the user's question. If the context",
            "doesn't contain the answer, say so rather than guessing.",
            "",
        ]
        token_count = len("\n".join(parts)) // 4
        
        for ep in episodes:
            chunk = (
                f"[{ep['episode_id']} | {ep['timestamp']}]\n"
                f"Entities: {', '.join(ep['entities'])}\n"
                f"Topics: {', '.join(ep['topics'])}\n"
                f"Tone: {', '.join(ep['tones'])}\n"
                f"Summary: {ep['summary']}\n"
                f"\n"
            )
            chunk_tokens = len(chunk) // 4
            if token_count + chunk_tokens > max_tokens:
                break
            parts.append(chunk)
            token_count += chunk_tokens
        
        return "\n".join(parts)
    
    def _embed(self, text: str) -> list[float]:
        """Embed text using the configured embedding model."""
        import openai
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
```

---

## 9. Mode A Generator (`src/generation/mode_a.py`)

```python
"""Mode A: Context window adapter for any LLM."""

import openai
from ..retrieval.retriever import HippocampalRetriever


class ModeAGenerator:
    """
    Context window adapter.
    
    Retrieves relevant episodes, builds a context string,
    feeds it to any LLM API. The LLM never knows about the
    memory system — it just receives curated context.
    """
    
    def __init__(self, retriever: HippocampalRetriever,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
    
    def generate(self, prompt: str, 
                 conversation_history: list[dict] = None,
                 max_context_tokens: int = 4000) -> dict:
        """
        Generate a response using retrieved context.
        
        Returns:
        {
            "response": "You were frustrated about...",
            "retrieved_episodes": [...],
            "context_used": "...",
            "model": "gpt-4o-mini",
        }
        """
        episodes = self.retriever.retrieve(prompt)
        context = self.retriever.build_context_string(episodes, max_context_tokens)
        
        messages = []
        
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant with access to past conversations. "
                       "Use the provided context to answer the user's question accurately. "
                       "If the context doesn't contain the answer, say so."
        })
        
        if conversation_history:
            messages.extend(conversation_history[-10:])
        
        messages.append({
            "role": "user",
            "content": f"Context from past conversations:\n{context}\n\nUser: {prompt}"
        })
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        
        return {
            "response": response.choices[0].message.content,
            "retrieved_episodes": episodes,
            "context_used": context,
            "model": self.model,
        }
```

---

## 10. Oracle Labeling Scripts (`src/training/oracle_labeling.py`)

Prepare the pipeline for Phase 1d (GNN training data generation):

```python
"""Oracle labeling pipeline for GNN training data generation.

This script is used in Phase 1d. Phase 1b sets up the infrastructure:
- Prompts for the Oracle
- Subgraph extraction from the populated graph
- Label parsing and validation
"""

ORACLE_GNN_LABELING_PROMPT = """
You are labeling a memory graph for GNN training. Given a subgraph of 
episodes and their relationships, produce training labels for five tasks.

SUBGRAPH:
{nodes_and_edges_as_json}

TASK 1: SALIENCE SCORING
Score each node and edge by structural importance (0.0-1.0).
HIGH salience: bridge nodes, decision-containing episodes, temporal chain anchors
LOW salience: redundant nodes, routine conversations, peripheral entities

TASK 2: CLUSTER DETECTION
Identify groups of episodes that should be abstracted into semantic memories.
A cluster has: shared entities, shared topics, temporal proximity, coherent theme.

TASK 3: LINK PREDICTION
Identify edges that SHOULD exist but are not explicitly in the graph.
Look for: entities that co-occur in similar contexts, causal chains, 
hierarchical relationships, contradictions between episodes.

TASK 4: ANOMALY DETECTION
Flag structural anomalies. Types:
- ORPHAN_DECISION: Decision with no 'madeBy' edge
- MISSING_TEMPORAL: Gap in follows chain
- CONTRADICTION: Two edges that cannot both be true
- TYPE_VIOLATION: Edge connecting incompatible types
- ISOLATED_CLUSTER: Subgraph with no external connections
- DUPLICATE_DECISION: Same decision appears to be made twice

TASK 5: ONTOLOGY REFINEMENT
Suggest missing subClassOf edges and misclassified entities.

Return ONLY valid JSON with all five label sets.
"""


class OracleLabelingPipeline:
    """
    Prepares training data for the GNN consolidator.
    
    Phase 1b: Infrastructure setup.
    Phase 1d: Full execution with Oracle API calls.
    """
    
    def __init__(self, store):
        self.store = store
    
    def extract_subgraph(self, center_episode_id: str, radius: int = 3) -> dict:
        """
        Extract a subgraph centered on an episode for Oracle labeling.
        
        The subgraph includes:
        - The center episode
        - All episodes within `radius` hops (via follows, has_entity, has_topic)
        - All edges between included nodes
        """
        visited = set()
        frontier = {center_episode_id}
        
        for _ in range(radius):
            next_frontier = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                neighbors = self._get_neighbors(node_id)
                next_frontier.update(neighbors)
            
            frontier = next_frontier - visited
        
        nodes = []
        for node_id in visited:
            ep = self.store.get_episode(node_id)
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
        
        return {"nodes": nodes, "edges": []}
    
    def _get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbors of a node in the graph."""
        pass
```

---

## 11. Testing Strategy

### 11.1 Query Planner Tests (`tests/test_query_planner.py`)

```python
def test_plan_simple_affect_query():
    """Plans a simple affect-based query."""
    planner = BonsaiQueryPlanner()
    plan = planner.plan("What was I frustrated about?")
    
    assert "frustrated" in plan["tones"]
    assert plan["entity_mode"] == "union"

def test_plan_entity_decision_query():
    """Plans an entity + decision query with correct entity_mode."""
    planner = BonsaiQueryPlanner()
    plan = planner.plan("What did Alice and I decide about the database?")
    
    assert "Alice" in plan["entities"]
    assert plan["entity_mode"] == "union"
    assert "decision_making" in plan["topics"] or "database_design" in plan["topics"]

def test_plan_temporal_query():
    """Plans a temporal chain query."""
    planner = BonsaiQueryPlanner()
    plan = planner.plan("What happened after we implemented morphisms?")
    
    assert plan["temporal_after"] is not None
    assert "morphism" in plan["temporal_after"].lower()

def test_plan_cross_entity_intersection():
    """Plans a true intersection query for two specific people."""
    planner = BonsaiQueryPlanner()
    plan = planner.plan("What did Alice and Bob disagree about?")
    
    assert "Alice" in plan["entities"]
    assert "Bob" in plan["entities"]
    assert plan["entity_mode"] == "intersection"
```

### 11.2 Graph Traversal Tests (`tests/test_graph_traversal.py`)

```python
def test_retrieve_by_entity(tmp_path):
    """Retrieves episodes by entity."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn("Alice said HBTrie is great", "Yes, HBTrie is excellent")
    encoder.encode_turn("Bob prefers Postgres", "Postgres has its strengths")
    encoder.encode_turn("Alice and Bob discussed databases", "They reached consensus")
    
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "entities": ["Alice"],
        "entity_mode": "union",
        "limit": 5,
    })
    
    assert len(results) >= 2
    assert all("Alice" in r["entities"] for r in results)

def test_retrieve_by_tone(tmp_path):
    """Retrieves episodes by emotional tone."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn("This is so frustrating", "I understand your frustration")
    encoder.encode_turn("This is exciting news", "I'm excited too")
    
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "tones": ["frustrated"],
        "entity_mode": "union",
        "limit": 5,
    })
    
    assert len(results) >= 1
    assert "frustrated" in results[0]["tones"]

def test_temporal_chain(tmp_path):
    """Follows temporal chain from anchor episode."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn("Let's implement morphisms", "Good idea")
    encoder.encode_turn("Morphisms are working", "Excellent")
    encoder.encode_turn("Now let's add the optimizer", "On it")
    
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "temporal_after": "morphism",
        "limit": 5,
    })
    
    assert len(results) >= 2

def test_entity_intersection(tmp_path):
    """Intersection: episodes containing ALL specified entities."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn("Alice said X", "Response about Alice only")
    encoder.encode_turn("Bob said Y", "Response about Bob only")
    encoder.encode_turn("Alice and Bob discussed Z", "Response about both")
    
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "entities": ["Alice", "Bob"],
        "entity_mode": "intersection",
        "limit": 5,
    })
    
    assert len(results) == 1
    assert "Alice" in results[0]["entities"] and "Bob" in results[0]["entities"]
```

### 11.3 Retriever Integration Tests (`tests/test_retriever.py`)

```python
def test_retrieve_end_to_end(tmp_path):
    """Full retrieval pipeline: plan → traverse → load."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn(
        "I'm frustrated with the WAL config",
        "Let me explain the three sync modes: IMMEDIATE, DEBOUNCED, ASYNC."
    )
    encoder.encode_turn(
        "What about encryption?",
        "We support AES-256-GCM with symmetric and asymmetric key modes."
    )
    encoder.encode_turn(
        "The Python async performance is terrible",
        "The async put is bottlenecked by asyncio marshalling. put_many achieves 299K ops/sec."
    )
    
    retriever = HippocampalRetriever(store)
    results = retriever.retrieve("What was I frustrated about?")
    
    assert len(results) > 0
    frustrations = [r for r in results if "frustrated" in r.get("tones", [])]
    assert len(frustrations) > 0

def test_build_context_string(tmp_path):
    """Context string is built correctly from retrieved episodes."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn("Alice said HBTrie is great", "Yes, HBTrie is excellent")
    
    retriever = HippocampalRetriever(store)
    results = retriever.retrieve("What did Alice say?")
    context = retriever.build_context_string(results)
    
    assert "Alice" in context
    assert "HBTrie" in context
    assert "ep_0001" in context
```

### 11.4 Mode A Tests (`tests/test_mode_a.py`)

```python
def test_generate_with_memory(tmp_path):
    """Mode A generates a response using retrieved context."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    encoder.encode_turn(
        "I'm frustrated with the WAL config",
        "Let me explain the three sync modes."
    )
    
    retriever = HippocampalRetriever(store)
    generator = ModeAGenerator(retriever)
    
    result = generator.generate("What was I frustrated about?")
    
    assert "response" in result
    assert len(result["retrieved_episodes"]) > 0
    assert "WAL" in result["response"] or "config" in result["response"].lower()
```

---

## 12. Build Vector Index Script (`scripts/build_vector_index.py`)

```python
"""Build FAISS vector index from encoded episode embeddings.

Usage:
    python scripts/build_vector_index.py --db ./data/memory_db
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.retrieval.vector_search import VectorSearch


def main():
    parser = argparse.ArgumentParser(description="Build vector index")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    args = parser.parse_args()
    
    store = HippocampalStore(args.db)
    vector_search = VectorSearch(store)
    vector_search.build_index()
    
    import faiss
    faiss.write_index(vector_search.index, f"{args.db}/vector_index.faiss")
    
    with open(f"{args.db}/vector_index_ids.json", "w") as f:
        json.dump(vector_search.episode_ids, f)
    
    print(f"Vector index saved. {len(vector_search.episode_ids)} episodes indexed.")
    store.close()


if __name__ == "__main__":
    main()
```

---

## 13. Datasets

### Pre-existing (download before starting)

Dataset	URL	Size	Use
**DialogSum**	`https://huggingface.co/datasets/knkarthick/dialogsum`	13,460 dialogues	Primary corpus for population
**SAMSum**	`https://huggingface.co/datasets/samsum`	16,369 dialogues	Secondary corpus
**Memory-Traces**	`https://huggingface.co/datasets/Cossale/memory-traces`	27,449 conversations	Salience labels (Phase 2), additional corpus

### Processing Order

1. **Sample conversations** (20) — already encoded in Phase 1a. Use for development.
2. **DialogSum** (first 1,000) — primary corpus. Diverse topics, good extraction test.
3. **SAMSum** (first 500) — messenger-style. Tests informal language extraction.
4. **Memory-Traces** (first 500) — has salience/emotion labels. Valuable for Phase 2.

---

## 14. Checkpoint Criteria

Phase 1b is complete when:

- [ ] Episode model updated with downstream fields (retrieval_count, ltp_phase, etc.)
- [ ] Config updated with Phase 2-4 placeholders
- [ ] Store persists and retrieves new Episode fields
- [ ] 1,000+ conversations from DialogSum encoded in WaveDB without errors
- [ ] 500+ conversations from SAMSum encoded
- [ ] Ingestion report shows extraction quality metrics at scale
- [ ] `BonsaiQueryPlanner` correctly converts NL questions to structured queries
- [ ] `GraphTraversal` executes entity, topic, tone, temporal, and intersection queries
- [ ] `GraphTraversal` follows temporal chains forward and backward
- [ ] `VectorSearch` builds FAISS index and returns semantic search results
- [ ] `HippocampalRetriever` orchestrates plan → traverse → load end-to-end
- [ ] `ModeAGenerator` produces responses using retrieved context
- [ ] Integration test: "What was I frustrated about?" returns relevant episodes
- [ ] Integration test: "What did Alice and I decide?" uses entity_mode="union"
- [ ] Integration test: "What happened after morphisms?" follows temporal chain
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Oracle labeling infrastructure in place (prompts, subgraph extraction)

---

## 15. Known Limitations (Addressed in Later Phases)

These are documented so the implementer knows they are deferred, not forgotten:

Limitation	Impact	Fix
**Bonsai plans blind**	No conversation context for pronoun resolution. "What did he say about it?" will fail.	Phase 1c: Pass last 2-3 turns to Bonsai. Phase 2: SSM state provides entity context.
**Fixed context strategy**	Always top 5 episodes, full text, hard cutoff at token limit. Later episodes silently dropped.	Phase 2.5: SSM chunking + JEPA presentation gate.
**No document-level retrieval**	Documents returned as individual sections, not as aggregated documents with relevant sections highlighted.	Phase 1c.
**Crude scoring**	Heuristic weights (entity×10, topic×5, tone×3, recency×0.1). Not learned.	Phase 3: GNN salience scoring.
**No multi-domain query support**	Cross-domain questions route to dominant domain only.	Phase 2: Retrieval Gate with multi-domain routing.
**No temporal indexing for long chains**	`follows` edges only. No timestamp range queries for "what happened in June 2025?"	Phase 1c.
**No entity salience tracking**	All entities treated equally in scoring. One-off entities bloat the ontology.	Phase 1c (tracking) + Phase 3 (decay).
**No cross-document deduplication**	Redundant ingested content bloats context.	Phase 3: GNN cross-document semantic memories.
**No subconscious routing**	Always retrieves. No ssm_direct pathway.	Phase 2: Retrieval Gate.
**No consolidation**	No semantic abstractions. No link prediction. No anomaly detection.	Phase 3: GNN Consolidator.
**No uncertainty detection**	Can't say "I don't know" with calibrated confidence. Empty results handled by semantic fallback only.	Phase 4: Uncertainty Detector.
**No procedural memory**	Can't execute or optimize stored processes.	Phase 6.

---

## 16. Implementation Order

1. **Update data models** — Episode fields, Config placeholders, Store persistence
2. **Download corpora** — DialogSum, SAMSum, Memory-Traces
3. **Process DialogSum** — 1,000 conversations through encoding pipeline
4. **Process SAMSum** — 500 conversations
5. **Build vector index** — FAISS index over all encoded episodes
6. **Implement Query Planner** — Bonsai NL → structured query
7. **Implement Graph Traversal** — Entity, topic, tone, temporal, intersection queries
8. **Implement Retriever** — Orchestrator: plan → traverse → load
9. **Implement Mode A Generator** — Context window adapter
10. **Write tests** — Unit tests for each component, integration tests
11. **Run integration tests** — End-to-end retrieval quality validation
12. **Set up Oracle labeling** — Prompts and subgraph extraction for Phase 1d

---

## 17. Next Phase

After Phase 1b checkpoint is met, proceed to **Phase 1c: Retrieval Refinements** which addresses the known limitations above: document-level retrieval, temporal indexing, entity salience tracking, and conversation context for Bonsai. Phase 1c takes 2-3 days and produces a more robust retrieval system before moving to learned components in Phase 2.

---

Begin with step 1. Report after each step. If the WaveDB graph query API differs from what's assumed in the traversal code, adapt to the actual API — the patterns are correct, the syntax may need adjustment. If extraction quality degrades at scale (compared to Phase 1a sample conversations), investigate and tune.