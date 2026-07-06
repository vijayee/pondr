# Phase 1c: Retrieval Refinements — Implementation Plan for Claude Code

## Overview

**Goal:** Address the known scaling limitations of Phase 1b before moving to learned components. This phase makes the retrieval system robust enough to handle real-world usage patterns: documents returned as documents, temporal queries that span months, entity salience that prevents ontology bloat, and pronoun resolution that makes conversation feel natural.

**What "done" looks like:** A retrieval system where "What did he say about it?" works because Bonsai has conversation context. Where "What happened in June 2025?" returns results without walking a 500-episode `follows` chain. Where documents are returned as documents with relevant sections highlighted. Where frequently-mentioned entities rank higher than one-off mentions.

**Prerequisite:** Phase 1b complete. Graph traversal engine working. Mode A generation operational. Known limitations documented.

**Duration estimate:** 2-3 days of focused implementation.

---

## 1. What Phase 1c Delivers

Artifact	Description	Consumer
**Document-level retrieval**	Documents returned as aggregated nodes with relevant sections highlighted	Retrieval pipeline, Mode A generation
**Temporal indexing**	Timestamp range queries for long chains; coexists with `follows` edges	Graph traversal
**Entity salience tracking**	Entity mention frequency, recency, and structural position tracked	Retrieval scoring, ontology decay (Phase 3)
**Conversation context for Bonsai**	Last 2-3 turns passed to query planner for pronoun resolution	Query planner
**Updated context builder**	Document-aware context formatting	Mode A generation

---

## 2. Updated Project Structure

Only files that change or are added:

```plaintext
hippocampal-memory/
├── src/
│   ├── retrieval/
│   │   ├── graph_traversal.py       # UPDATED: temporal indexing, entity salience
│   │   ├── query_planner.py         # UPDATED: conversation context
│   │   ├── retriever.py             # UPDATED: document-level retrieval
│   │   └── document_retriever.py    # NEW: document-aware retrieval
│   └── memory/
│       └── store.py                 # UPDATED: entity salience persistence
├── tests/
│   ├── test_graph_traversal.py      # UPDATED: temporal indexing tests
│   ├── test_query_planner.py        # UPDATED: context-aware planning tests
│   ├── test_retriever.py            # UPDATED: document retrieval tests
│   └── test_document_retriever.py   # NEW
└── scripts/
    └── compute_entity_salience.py    # NEW: batch salience computation
```

---

## 3. Refinement 1: Document-Level Retrieval

### 3.1 Problem

When a 100-page PDF is ingested, it becomes 200 individual section nodes in the graph. A query matching 15 sections returns 15 separate results. The context builder treats each as an independent episode. The LLM receives a wall of text with no indication that sections 3, 7, and 12 are all from the same document.

### 3.2 Solution

Documents are first-class nodes in the graph. Sections link to their parent document via `has_section` edges. Retrieval can return documents (with relevant sections highlighted) or individual sections depending on query specificity.

### 3.3 Document Retriever (`src/retrieval/document_retriever.py`)

```python
"""Document-aware retrieval that aggregates sections into documents."""

from ..memory.store import HippocampalStore


class DocumentRetriever:
    """
    Wraps the graph traversal engine with document-aware result aggregation.
    
    When a query matches multiple sections from the same document,
    returns the document as a single result with relevant sections
    highlighted, rather than returning each section separately.
    """
    
    def __init__(self, store: HippocampalStore):
        self.store = store
    
    def aggregate_results(self, raw_results: list[dict]) -> list[dict]:
        """
        Aggregate raw section-level results into document-level results.
        
        For each result, check if it's a document section.
        If multiple sections from the same document match, merge them
        into a single document result with section highlights.
        Non-document results (conversation episodes) pass through unchanged.
        """
        # Separate document sections from regular episodes
        doc_sections = []
        regular_eps = []
        
        for r in raw_results:
            if self._is_document_section(r["episode_id"]):
                doc_sections.append(r)
            else:
                regular_eps.append(r)
        
        # Group document sections by parent document
        doc_groups = {}
        for section in doc_sections:
            doc_id = self._get_parent_document(section["episode_id"])
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "document_id": doc_id,
                    "title": self._get_document_title(doc_id),
                    "sections": [],
                    "best_score": 0,
                    "entities": set(),
                    "topics": set(),
                }
            
            group = doc_groups[doc_id]
            group["sections"].append(section)
            group["best_score"] = max(group["best_score"], section["score"])
            group["entities"].update(section.get("entities", []))
            group["topics"].update(section.get("topics", []))
        
        # Build document-level results
        doc_results = []
        for doc_id, group in doc_groups.items():
            # Sort sections by score
            group["sections"].sort(key=lambda s: s["score"], reverse=True)
            
            # Build summary from top sections
            section_summaries = []
            for s in group["sections"][:5]:  # Top 5 sections
                section_summaries.append(
                    f"Section '{s.get('heading', 'Untitled')}': {s['summary']}"
                )
            
            doc_results.append({
                "episode_id": doc_id,
                "type": "document",
                "score": group["best_score"],
                "summary": f"Document: {group['title']}\n" + 
                          f"Relevant sections ({len(group['sections'])} matched):\n" +
                          "\n".join(f"  - {s}" for s in section_summaries),
                "text": self._build_document_context(group),
                "timestamp": self._get_document_timestamp(doc_id),
                "entities": list(group["entities"]),
                "topics": list(group["topics"]),
                "tones": [],
                "matched_sections": len(group["sections"]),
                "total_sections": self._get_document_section_count(doc_id),
            })
        
        # Merge document results with regular episodes, re-sort by score
        all_results = doc_results + regular_eps
        all_results.sort(key=lambda r: r["score"], reverse=True)
        
        return all_results
    
    def _is_document_section(self, node_id: str) -> bool:
        """Check if a node is a document section."""
        # Query: does this node have a 'child_of' edge? (sections link to parent)
        result = self.store.graph.query() \
            .vertex(node_id) \
            .out("child_of") \
            .execute()
        return len(result) > 0
    
    def _get_parent_document(self, section_id: str) -> str:
        """Get the parent document ID for a section."""
        result = self.store.graph.query() \
            .vertex(section_id) \
            .out("child_of") \
            .execute()
        return result[0].id if result else section_id
    
    def _get_document_title(self, doc_id: str) -> str:
        """Get document title from HBTrie."""
        title = self.store.db.get_sync(f"doc/{doc_id}/title")
        if title:
            return title.decode() if isinstance(title, bytes) else title
        return doc_id
    
    def _get_document_timestamp(self, doc_id: str) -> str:
        """Get document timestamp."""
        ts = self.store.db.get_sync(f"doc/{doc_id}/created_at")
        if ts:
            return ts.decode() if isinstance(ts, bytes) else ts
        return ""
    
    def _get_document_section_count(self, doc_id: str) -> int:
        """Get total number of sections in a document."""
        result = self.store.graph.query() \
            .vertex(doc_id) \
            .out("has_section") \
            .execute()
        return len(result)
    
    def _build_document_context(self, group: dict) -> str:
        """Build full text context for a document result."""
        parts = [f"Document: {group['title']}"]
        parts.append(f"Relevant sections: {len(group['sections'])} of "
                    f"{self._get_document_section_count(group['document_id'])}")
        parts.append("")
        
        for s in group["sections"][:5]:
            heading = s.get("heading", "Untitled")
            text = s.get("text", s.get("summary", ""))
            parts.append(f"--- {heading} ---")
            parts.append(text[:500])  # Truncate long sections
            parts.append("")
        
        return "\n".join(parts)
```

### 3.4 Integration with Retriever (`src/retrieval/retriever.py`)

Update the `HippocampalRetriever.retrieve` method to use the document retriever:

```python
class HippocampalRetriever:
    def __init__(self, store, planner_model="gpt-4o-mini"):
        # ... existing init ...
        self.document_retriever = DocumentRetriever(store)  # NEW
    
    def retrieve(self, prompt: str, use_semantic: bool = True) -> list[dict]:
        # ... existing retrieval logic ...
        
        # NEW: Aggregate document sections into document results
        results = self.document_retriever.aggregate_results(results)
        
        return results
```

---

## 4. Refinement 2: Temporal Indexing

### 4.1 Problem

`follows` edges work for short-range traversal ("what happened next?"). For "what happened in June 2025?", walking a 500-episode chain is O(n). The graph needs timestamp range queries.

### 4.2 Solution

Store episode timestamps as indexed properties in the graph. Add timestamp range queries to the graph traversal engine. Both mechanisms coexist: `follows` for short-range, timestamp indexes for long-range.

### 4.3 Graph Traversal Updates (`src/retrieval/graph_traversal.py`)

Add timestamp range query support:

```python
class GraphTraversal:
    # ... existing methods ...
    
    def retrieve(self, query_plan: dict) -> list[dict]:
        # ... existing logic ...
        
        # NEW: Check for explicit date range before temporal filter
        date_from = query_plan.get("date_from")
        date_to = query_plan.get("date_to")
        
        if date_from or date_to:
            candidates = self._filter_date_range(candidates, date_from, date_to)
        elif temporal_filter:
            candidates = self._filter_temporal(candidates, temporal_filter)
        
        # ... rest of existing logic ...
    
    def _filter_date_range(self, candidate_ids, date_from, date_to):
        """
        Filter episodes by explicit date range.
        
        This is O(n) in candidate count but O(1) per episode —
        much faster than walking a follows chain for long ranges.
        """
        filtered = []
        
        for ep_id in candidate_ids:
            ep = self.store.get_episode(ep_id)
            if not ep:
                continue
            
            ts = datetime.fromisoformat(ep.timestamp)
            
            if date_from and ts < datetime.fromisoformat(date_from):
                continue
            if date_to and ts > datetime.fromisoformat(date_to):
                continue
            
            filtered.append(ep_id)
        
        return filtered
    
    def _get_episodes_in_range(self, date_from: str, date_to: str) -> list[str]:
        """
        Get all episodes within a date range.
        
        Uses HBTrie prefix scan if timestamps are stored with sortable keys,
        or falls back to graph query with timestamp filtering.
        
        NOTE: For optimal performance, timestamps should be stored in the
        HBTrie with a sortable key format (e.g., ep/2025-06/...) to enable
        prefix scans. This is a store-level optimization for Phase 1c.
        """
        # If timestamps are indexed in HBTrie with date-based prefixes:
        # Scan ep/2025-06/* for June 2025 episodes
        # This is O(results) not O(total episodes)
        
        # Fallback: graph query with timestamp range
        # (Implementation depends on WaveDB graph query API support
        # for range predicates on properties)
        pass
```

### 4.4 Query Planner Update (`src/retrieval/query_planner.py`)

Add date range support to the Bonsai prompt:

```python
BONSAI_QUERY_PROMPT = """...existing prompt...

ADDITIONAL QUERY PARAMETERS:
- date_from: ISO date string for start of range (e.g., "2025-06-01"), or null
- date_to: ISO date string for end of range, or null
- Use date_from/date_to for explicit date queries like "what happened in June 2025?"
- Use temporal_filter for relative queries like "last week"
- Do NOT set both date_from/date_to and temporal_filter in the same query

EXAMPLES:
- "What happened in June 2025?" → date_from="2025-06-01", date_to="2025-06-30"
- "What did we discuss last week?" → temporal_filter="last_week"
- "What happened between March and May?" → date_from="2025-03-01", date_to="2025-05-31"

...rest of existing prompt...

Return ONLY valid JSON:
{{"entities": [], "topics": [], "tones": [], "entity_mode": "union", 
  "temporal_after": null, "temporal_before": null, 
  "temporal_filter": null, "date_from": null, "date_to": null,
  "limit": 5}}"""
```

---

## 5. Refinement 3: Entity Salience Tracking

### 5.1 Problem

All entities are treated equally in retrieval scoring. "Alice" (mentioned 200 times) gets the same score boost as "the barista at the coffee shop" (mentioned once). One-off entities bloat the ontology and dilute retrieval quality.

### 5.2 Solution

Track entity mention frequency, recency, and structural position. Use these to weight entity matches in retrieval scoring. Foundation for ontology decay in Phase 3.

### 5.3 Entity Salience Store (`src/memory/store.py` — additions)

```python
class HippocampalStore:
    # ... existing methods ...
    
    def increment_entity_salience(self, entity: str, episode_id: str):
        """
        Increment salience for an entity when it appears in a new episode.
        
        Called during encoding (Phase 1a encoder should call this).
        For Phase 1c, we compute salience in batch from existing data.
        """
        current = self.db.get_sync(f"entity/{entity}/mention_count")
        count = int(current.decode() if isinstance(current, bytes) else current) if current else 0
        self.db.put_sync(f"entity/{entity}/mention_count", str(count + 1))
        self.db.put_sync(f"entity/{entity}/last_mentioned", episode_id)
    
    def get_entity_salience(self, entity: str) -> float:
        """
        Get salience score for an entity.
        
        Salience = mention_count * recency_factor * structural_factor
        
        For Phase 1c: mention_count only. Recency and structural factors
        added in Phase 3 when GNN salience scoring is available.
        """
        count_str = self.db.get_sync(f"entity/{entity}/mention_count")
        if not count_str:
            return 0.0
        
        count = int(count_str.decode() if isinstance(count_str, bytes) else count_str)
        
        # Normalize: log-scale to prevent dominant entities from
        # completely drowning out less frequent ones
        return min(1.0, 0.1 + 0.3 * (count ** 0.5) / 10)
    
    def get_top_entities(self, limit: int = 100) -> list[tuple[str, float]]:
        """Get the highest-salience entities."""
        # Scan entity/* keys from HBTrie
        # (Implementation depends on HBTrie prefix scan API)
        pass
```

### 5.4 Salience-Weighted Scoring (`src/retrieval/graph_traversal.py` — update)

```python
def _score_candidates(self, candidate_ids, entities, topics, tones):
    """
    Score candidates by match quality + recency.
    
    Phase 1c: Entity matches weighted by entity salience.
    Phase 3: Full GNN-learned salience scoring replaces this.
    """
    scored = []
    for ep_id in candidate_ids:
        ep = self.store.get_episode(ep_id)
        if not ep:
            continue
        
        score = 0.0
        
        # Entity matches — weighted by salience
        if entities:
            ep_entities = set(ep.entities)
            query_entities = set(entities)
            for entity in (ep_entities & query_entities):
                salience = self.store.get_entity_salience(entity)
                score += 10 * (0.5 + 0.5 * salience)  # Range: 5-10 per match
        
        # Topic matches
        if topics:
            ep_topics = set(ep.topics)
            query_topics = set(topics)
            score += len(ep_topics & query_topics) * 5
        
        # Tone matches
        if tones:
            ep_tones = set(ep.tones)
            query_tones = set(tones)
            score += len(ep_tones & query_tones) * 3
        
        # Recency bonus
        ep_num = int(ep.id.split("_")[1]) if "_" in ep.id else 0
        score += ep_num * 0.1
        
        scored.append((score, ep_id))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
```

### 5.5 Batch Salience Computation (`scripts/compute_entity_salience.py`)

```python
"""Compute entity salience from existing encoded episodes.

Usage:
    python scripts/compute_entity_salience.py --db ./data/memory_db
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore


def main():
    parser = argparse.ArgumentParser(description="Compute entity salience")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    args = parser.parse_args()
    
    store = HippocampalStore(args.db)
    
    # Scan all episodes, count entity mentions
    entity_counts = Counter()
    entity_last_episode = {}
    
    all_eps = _get_all_episode_ids(store)
    
    for ep_id in all_eps:
        ep = store.get_episode(ep_id)
        if not ep:
            continue
        
        for entity in ep.entities:
            entity_counts[entity] += 1
            entity_last_episode[entity] = ep_id
    
    # Store salience data
    for entity, count in entity_counts.items():
        store.db.put_sync(f"entity/{entity}/mention_count", str(count))
        store.db.put_sync(f"entity/{entity}/last_mentioned", 
                         entity_last_episode.get(entity, ""))
    
    print(f"Computed salience for {len(entity_counts)} entities.")
    print(f"Top 20 entities:")
    for entity, count in entity_counts.most_common(20):
        print(f"  {entity}: {count} mentions")
    
    store.close()


def _get_all_episode_ids(store):
    result = store.graph.query() \
        .has("predicate", "has_entity") \
        .execute()
    return list(set(r.subject for r in result))


if __name__ == "__main__":
    main()
```

---

## 6. Refinement 4: Conversation Context for Bonsai

### 6.1 Problem

Bonsai plans queries from the current prompt alone. "What did he say about it?" has no entity to extract. The query planner needs conversation context to resolve pronouns and implicit references.

### 6.2 Solution

Pass the last 2-3 conversation turns to Bonsai as context. The prompt includes recent messages so Bonsai can resolve "he" → "Bob" and "it" → "the WAL configuration."

### 6.3 Query Planner Update (`src/retrieval/query_planner.py`)

```python
BONSAI_QUERY_PROMPT_WITH_CONTEXT = """Convert this question into a structured memory query.
Return ONLY valid JSON, no other text.

RECENT CONVERSATION (for context):
{conversation_context}

CURRENT QUESTION: {prompt}

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
- temporal_after: keyword to find anchor and follow chain forward, or null
- temporal_before: keyword to find anchor and follow chain backward, or null
- temporal_filter: "today", "this_week", "last_week", "this_month", or null
- date_from: ISO date for start of range, or null
- date_to: ISO date for end of range, or null
- limit: max episodes to return (default 5)

IMPORTANT: Use the RECENT CONVERSATION to resolve pronouns and implicit references.
- If the user says "he" or "she", identify the person from recent context.
- If the user says "it" or "that", identify the topic from recent context.
- If the user says "we discussed", the entities are the people in the conversation.

IMPORTANT RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union" 
- "What did Alice say about databases?" → entities=["Alice"], 
  topics=["database_design"], entity_mode="union"
- "What happened after we implemented morphisms?" → temporal_after="morphism"
- "Why did we choose X over Y?" → topics=["decision_making"], 
  entities=["X", "Y"], entity_mode="union"
- If the question is about a specific person's opinion, entity_mode is "union"
- If the question is about when two specific things were discussed TOGETHER, 
  entity_mode is "intersection"

Return ONLY valid JSON:
{{"entities": [], "topics": [], "tones": [], "entity_mode": "union", 
  "temporal_after": null, "temporal_before": null, 
  "temporal_filter": null, "date_from": null, "date_to": null,
  "limit": 5}}"""


class BonsaiQueryPlanner:
    """Converts natural language questions into structured query parameters."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
    
    def plan(self, prompt: str, conversation_history: list[dict] = None) -> dict:
        """
        Plan a query from a natural language prompt.
        
        Args:
            prompt: The user's question
            conversation_history: Recent conversation turns for context
        """
        # Build context from recent history
        context = ""
        if conversation_history:
            recent = conversation_history[-6:]  # Last 3 exchanges (6 messages)
            context = "\n".join(
                f"{msg['role']}: {msg['content']}" 
                for msg in recent
            )
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": BONSAI_QUERY_PROMPT_WITH_CONTEXT.format(
                    conversation_context=context if context else "(no prior context)",
                    prompt=prompt
                )
            }],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        return json.loads(response.choices[0].message.content)
```

### 6.4 Retriever Update (`src/retrieval/retriever.py`)

Pass conversation history through to the planner:

```python
class HippocampalRetriever:
    def retrieve(self, prompt: str, 
                 conversation_history: list[dict] = None,
                 use_semantic: bool = True) -> list[dict]:
        """
        Retrieve relevant episodes for a prompt.
        
        Args:
            prompt: Natural language question
            conversation_history: Recent conversation for pronoun resolution
            use_semantic: Whether to fall back to semantic search
        """
        # Pass conversation history to planner for context
        query_plan = self.planner.plan(prompt, conversation_history)
        
        # ... rest of existing retrieval logic ...
```

### 6.5 Mode A Generator Update (`src/generation/mode_a.py`)

Pass conversation history to the retriever:

```python
class ModeAGenerator:
    def generate(self, prompt: str, 
                 conversation_history: list[dict] = None,
                 max_context_tokens: int = 4000) -> dict:
        # Pass history to retriever for pronoun resolution
        episodes = self.retriever.retrieve(prompt, conversation_history)
        
        # ... rest of existing generation logic ...
```

---

## 7. Testing Strategy

### 7.1 Document Retrieval Tests (`tests/test_document_retriever.py`)

```python
def test_aggregate_document_sections(tmp_path):
    """Multiple sections from the same document are aggregated."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    
    # Setup: create a document with sections in the graph
    # (Requires document ingestion — use store directly for test)
    doc_id = "doc_001"
    store.graph.insert_sync(doc_id, "type", "Document")
    store.graph.insert_sync(doc_id, "has_section", "sec_001")
    store.graph.insert_sync(doc_id, "has_section", "sec_002")
    store.graph.insert_sync("sec_001", "child_of", doc_id)
    store.graph.insert_sync("sec_002", "child_of", doc_id)
    store.db.put_sync(f"doc/{doc_id}/title", "Test Document")
    
    # Create raw results with two sections from the same document
    raw_results = [
        {"episode_id": "sec_001", "score": 15.0, "summary": "Section 1 content",
         "text": "Full text 1", "timestamp": "2025-06-01", 
         "entities": ["Alice"], "topics": ["database_design"], "tones": []},
        {"episode_id": "sec_002", "score": 10.0, "summary": "Section 2 content",
         "text": "Full text 2", "timestamp": "2025-06-01",
         "entities": ["Bob"], "topics": ["configuration"], "tones": []},
        {"episode_id": "ep_001", "score": 20.0, "summary": "Regular episode",
         "text": "Full text", "timestamp": "2025-06-01",
         "entities": ["Alice"], "topics": ["test"], "tones": ["curious"]},
    ]
    
    doc_retriever = DocumentRetriever(store)
    aggregated = doc_retriever.aggregate_results(raw_results)
    
    # Should have 2 results: 1 document (aggregated) + 1 regular episode
    assert len(aggregated) == 2
    
    doc_result = [r for r in aggregated if r.get("type") == "document"][0]
    assert doc_result["matched_sections"] == 2
    assert "Alice" in doc_result["entities"]
    assert "Bob" in doc_result["entities"]
    
    ep_result = [r for r in aggregated if r.get("type") != "document"][0]
    assert ep_result["episode_id"] == "ep_001"

def test_single_section_not_aggregated(tmp_path):
    """A single section from a document is still returned as a document result."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    
    doc_id = "doc_001"
    store.graph.insert_sync(doc_id, "type", "Document")
    store.graph.insert_sync(doc_id, "has_section", "sec_001")
    store.graph.insert_sync("sec_001", "child_of", doc_id)
    store.db.put_sync(f"doc/{doc_id}/title", "Test Document")
    
    raw_results = [
        {"episode_id": "sec_001", "score": 15.0, "summary": "Section 1",
         "text": "Full text", "timestamp": "2025-06-01",
         "entities": ["Alice"], "topics": ["test"], "tones": []},
    ]
    
    doc_retriever = DocumentRetriever(store)
    aggregated = doc_retriever.aggregate_results(raw_results)
    
    # Single section still becomes a document result
    assert len(aggregated) == 1
    assert aggregated[0].get("type") == "document"
    assert aggregated[0]["matched_sections"] == 1
```

### 7.2 Temporal Indexing Tests (`tests/test_graph_traversal.py` — additions)

```python
def test_date_range_query(tmp_path):
    """Retrieves episodes within an explicit date range."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    # Encode episodes with known timestamps
    # (Override timestamps for testing)
    ep1 = encoder.encode_turn("June discussion", "Response 1")
    ep2 = encoder.encode_turn("July discussion", "Response 2")
    ep3 = encoder.encode_turn("August discussion", "Response 3")
    
    # Manually set timestamps for testing
    store.db.put_sync(f"ep/{ep1.id}/ts", "2025-06-15T10:00:00")
    store.db.put_sync(f"ep/{ep2.id}/ts", "2025-07-15T10:00:00")
    store.db.put_sync(f"ep/{ep3.id}/ts", "2025-08-15T10:00:00")
    
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "date_from": "2025-06-01",
        "date_to": "2025-07-31",
        "limit": 5,
    })
    
    # Should find June and July episodes, not August
    result_ids = {r["episode_id"] for r in results}
    assert ep1.id in result_ids
    assert ep2.id in result_ids
    assert ep3.id not in result_ids

def test_date_range_and_entity_combined(tmp_path):
    """Date range combined with entity filter."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    # Encode episodes
    ep1 = encoder.encode_turn("Alice said X in June", "Response")
    ep2 = encoder.encode_turn("Bob said Y in June", "Response")
    ep3 = encoder.encode_turn("Alice said Z in August", "Response")
    
    store.db.put_sync(f"ep/{ep1.id}/ts", "2025-06-15T10:00:00")
    store.db.put_sync(f"ep/{ep2.id}/ts", "2025-06-15T10:00:00")
    store.db.put_sync(f"ep/{ep3.id}/ts", "2025-08-15T10:00:00")
    
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "entities": ["Alice"],
        "entity_mode": "union",
        "date_from": "2025-06-01",
        "date_to": "2025-07-31",
        "limit": 5,
    })
    
    # Should find only Alice's June episode
    result_ids = {r["episode_id"] for r in results}
    assert ep1.id in result_ids
    assert ep2.id not in result_ids  # Bob, not Alice
    assert ep3.id not in result_ids  # August, not June
```

### 7.3 Context-Aware Planning Tests (`tests/test_query_planner.py` — additions)

```python
def test_pronoun_resolution_with_context():
    """Bonsai resolves pronouns using conversation context."""
    planner = BonsaiQueryPlanner()
    
    history = [
        {"role": "user", "content": "What did Bob say about the WAL config?"},
        {"role": "assistant", "content": "Bob said the WAL config needed better documentation."},
    ]
    
    plan = planner.plan("What did he suggest we do about it?", history)
    
    # "he" should resolve to "Bob", "it" should resolve to WAL config
    assert "Bob" in plan["entities"]
    assert "configuration" in plan["topics"]

def test_implicit_reference_with_context():
    """Bonsai resolves implicit references from context."""
    planner = BonsaiQueryPlanner()
    
    history = [
        {"role": "user", "content": "I'm worried about the database performance."},
        {"role": "assistant", "content": "The Python async bindings are the main bottleneck."},
    ]
    
    plan = planner.plan("How do we fix that?", history)
    
    # "that" should resolve to the performance issue / Python async
    assert "performance" in plan["topics"] or "Python" in plan["entities"]

def test_no_context_still_works():
    """Planner works without conversation history."""
    planner = BonsaiQueryPlanner()
    plan = planner.plan("What was I frustrated about?")
    
    assert "frustrated" in plan["tones"]
```

### 7.4 Entity Salience Tests (`tests/test_graph_traversal.py` — additions)

```python
def test_salience_weighted_scoring(tmp_path):
    """High-salience entities get higher match scores."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    # Encode episodes
    encoder.encode_turn("Alice said X", "Response")  # Alice mentioned once
    encoder.encode_turn("Bob said Y", "Response")     # Bob mentioned once
    
    # Artificially boost Alice's salience
    store.db.put_sync("entity/Alice/mention_count", "50")
    store.db.put_sync("entity/Bob/mention_count", "1")
    
    traversal = GraphTraversal(store)
    
    # Query for both
    results = traversal.retrieve({
        "entities": ["Alice", "Bob"],
        "entity_mode": "union",
        "limit": 5,
    })
    
    # Alice's episode should score higher than Bob's
    alice_result = [r for r in results if "Alice" in r["entities"]][0]
    bob_result = [r for r in results if "Bob" in r["entities"]][0]
    assert alice_result["score"] > bob_result["score"]
```

---

## 8. Checkpoint Criteria

Phase 1c is complete when:

- [ ] `DocumentRetriever` aggregates multiple sections from the same document into a single result
- [ ] Document results show relevant section count and summaries
- [ ] Non-document results (conversation episodes) pass through unchanged
- [ ] `GraphTraversal` supports `date_from` and `date_to` range queries
- [ ] Date range queries can be combined with entity, topic, and tone filters
- [ ] `BonsaiQueryPlanner` accepts optional conversation history
- [ ] Pronoun resolution works: "What did he say about it?" resolves correctly with context
- [ ] Implicit reference resolution works: "How do we fix that?" resolves correctly
- [ ] Planner still works without conversation history (backward compatible)
- [ ] Entity salience is computed and stored for all entities
- [ ] Retrieval scoring weights entity matches by salience
- [ ] High-salience entities rank higher than low-salience entities in results
- [ ] `scripts/compute_entity_salience.py` runs successfully on the populated database
- [ ] All existing tests still pass (no regressions)
- [ ] All new tests pass

---

## 9. Implementation Order

1. **Entity salience** — `compute_entity_salience.py` + store methods + scoring update
2. **Temporal indexing** — date range queries in graph traversal + query planner update
3. **Conversation context** — query planner context support + retriever + Mode A updates
4. **Document retrieval** — `DocumentRetriever` + integration with retriever
5. **Tests** — new tests for all four refinements + regression check on existing tests
6. **Integration test** — full pipeline with all refinements active

---

## 10. What Phase 1c Does NOT Do

These remain for later phases:

- **SSM chunking** — Phase 2.5
- **JEPA presentation gating** — Phase 2.5
- **GNN salience scoring** — Phase 3 (replaces heuristic entity salience)
- **Ontology decay** — Phase 3 (uses entity salience tracking from this phase)
- **Cross-document deduplication** — Phase 3
- **Multi-domain routing** — Phase 2
- **Uncertainty detection** — Phase 4

---

## 11. Next Phase

After Phase 1c checkpoint is met, proceed to **Phase 1d: Training Data Generation** which uses the Oracle to label subgraphs for GNN training, generate Bonsai query planning pairs, and prepare all training data needed for Phase 2 (JEPA-Gated SSM) and Phase 3 (GNN Consolidator).

---

Begin with step 1. Report after each step. The entity salience computation should be run on the existing populated database from Phase 1b. If any refinement causes a regression in the Phase 1b integration tests, pause and investigate before continuing.