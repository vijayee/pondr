# Phase 1c: Retrieval Refinements — Implementation Plan for Claude Code

## Overview

**Goal:** Address the known scaling limitations of Phase 1b before moving to learned
components. This phase makes the retrieval system robust enough to handle real-world usage
patterns: temporal queries that span months, entity salience that prevents ontology bloat,
pronoun resolution that makes conversation feel natural, and (conditional on a document
ingestion path being built) documents returned as documents with relevant sections
highlighted.

**What "done" looks like:** "What did he say about it?" works because Bonsai has
conversation context. "What happened in June 2025?" returns results without walking a
500-episode `follows` chain. Frequently-mentioned entities rank higher than one-off
mentions. Document-level aggregation is designed and gated on a document ingestion path.

**Prerequisite:** Phase 1b complete. Graph traversal engine working. Mode A generation
operational. Known limitations documented.

**Duration estimate:** 2-3 days of focused implementation.

---

## 0. Reality check from Phase 1b — read before implementing anything

Phase 1b shipped retrieval code that diverges from the API this doc originally assumed.
**Every code snippet below was rewritten to match the shipped API.** Do not paste code from
older versions of this doc. The non-obvious facts:

### 0.1 WaveDB graph API (shipped in `src/retrieval/graph_traversal.py`)

- `GraphLayer.query()` → a `GraphQuery` builder: `.vertex(id)`, `.out(p)`, `.in_(p)`,
  `.has(p, v)`, `.limit(n)`, then **`.execute_sync()`** → a `GraphResult` whose
  **`.vertices` is a `list[str]`** of vertex ids, with `.count`.
- **There is no `.execute()` and no `r.subject` / `r.id` row model.** Results are a flat
  list of vertex-id strings. Iterate `for vid in result.vertices:` and always
  `result.close()` in a `finally` (the result wraps a heap-allocated C struct; see
  `_exec_vertices` in `graph_traversal.py`).
- **`.has(predicate, value)` requires a concrete value.** "All episodes with predicate P
  regardless of value" is NOT expressible via the builder. Use
  `store.db.create_read_stream(start=..., end=...)` over a `memory/pos/{P}/` (or
  `memory/spo/{S}/`) prefix and parse the last `/`-component(s) as the subject/object —
  exactly the pattern `list_sessions` / `list_session_episodes` in `store.py` and
  `_user_for_session` / `_hydrate` in `graph_traversal.py` use.
- Graph subtree prefix is **`memory`**: SPO keys `memory/spo/{s}/{p}/{o}`, POS keys
  `memory/pos/{p}/{o}/{s}`.
- `GraphLayer` exposes both `insert_sync(s, p, o)` (single triple, immediate) and
  `expand_triple(s, p, o) -> list[dict]` (returns ops to splice into a parent
  `db.batch_sync`). `store.encode_episode` uses `expand_triple` so content + index share
  one atomic batch. Tests that set up raw triples may use either; `insert_sync` is fine for
  isolated test fixtures, `expand_triple` + `db.batch_sync` for anything that must be
  atomic with content.
- Predicates are **snake_case**: `has_entity`, `has_topic`, `has_tone`, `has_decision`,
  `in_episode`, `in_session`, `has_session`, `has_episode`, `follows`, `follows_session`,
  `at_time`, `state`, `validity_start`, `subClassOf`. The ontology registry's camelCase
  names (`hasEntity`…) never appear in the SPO index. Node id conventions: `E:{entity}`,
  `T:{topic}`, `A:{tone}`, `D:{decision}`, `ep_000001` (6-digit), `S:0001`, `U:{user}`.
- `follows` points from an episode to the one **before** it (`ep2 follows ep1` ⇒ ep2 is
  later). Forward-in-time = `in_("follows")`, backward = `out("follows")` (see
  `_follows_neighbors`).

### 0.2 Key layout (shipped `store.py`)

- Content lives under **`content/ep/{eid}/{field}`** (summary, text, ts, salience, state,
  retrieval_count, ltp_phase, decay_rate, saturation_flags, retrieval_timestamps,
  consolidation_window_start, embedding). **There is no top-level `ep/` or `doc/` or
  `entity/` namespace.** Older snippets in this doc that wrote `ep/{id}/ts` or
  `doc/{id}/title` were wrong.
- System counters live under `content/system/...` (e.g. `content/system/episode_counter`).
- New namespaces introduced by Phase 1c (entity salience, documents) are defined here under
  `content/...` to stay in the HBTrie content tree and be point-lookupable via `get_sync`.

### 0.3 `get_sync` reliability — the load-bearing caveat

`get_sync` was observed to be **broken on DBs built by unsorted ingestion under WaveDB
0.1.13** (~22-27% hit rate on timestamp keys; wrong separator keys in internal bnodes — a
symptom of the split-orphan bug). The split-orphan bug (Bug A) is **fixed in WaveDB 0.1.14**,
which should restore `get_sync` correctness on unsorted-built DBs, but this has **not been
re-verified** on a freshly-ingested unsorted DB.

What IS verified:
- The **compact corpora** (DialogSum, SAMSum) were reloaded by `scripts/compact_corpus_db.py`
  with **lexically-sorted keys** into WaveDB 0.1.14 → `get_sync` is confirmed correct on
  them (the entire shipped retrieval pipeline — `get_episode` does 11 `get_sync` point
  reads per episode — works against these DBs).
- `create_read_stream(start=, end=)` scans are reliable on every DB we have tested
  (sorted- and unsorted-built) and yield `(key, value)` tuples.

**Consequences for Phase 1c:**
- For **iterating** over entities/episodes/documents, use `create_read_stream` prefix scans,
  never `get_sync` in a loop.
- For **point lookups** of a single known key (the salience read during scoring, a document
  title), `get_sync` is fine **on the compact corpora**. For any **new namespace written by
  live encoding** (unsorted writes), either (a) write the new keys in a **sorted batch**
  (collect → sort → `batch_sync`, the `compact_corpus_db.py` pattern) so `get_sync` is
  safe, or (b) read them via scans. Treat `get_sync` on unsorted-written keys as
  unverified.
- The batch salience script (§5) writes salience keys in a sorted batch → safe for
  `get_sync` reads during scoring.

### 0.4 LLM pieces use the local Bonsai server — NO OpenAI spend

The query planner and Mode A generator talk to the **local Bonsai llama-server** at
`config.bonsai_endpoint` (`http://localhost:8080/v1` by default) via its OpenAI-compatible
`/chat/completions` API, using **`requests`** — not the `openai` package, and not
`gpt-4o-mini`. Model is `config.bonsai_model` (`prism-ml/Ternary-Bonsai-8B-gguf`). The
shipped `BonsaiQueryPlanner` constructor is
`BonsaiQueryPlanner(model=None, endpoint=None, temperature=None, timeout=...)` and defaults
to those config fields; `plan()` calls `plan_via_server` (posts via `requests`) and falls
back to `plan_rule_based` (deterministic, no server) on any failure. The shipped
`HippocampalRetriever` is `HippocampalRetriever(store, planner, auto_load_index=...)` — it
takes a **planner instance**, not a model name. The shipped `ModeAGenerator.generate`
**already accepts** `conversation_history` and forwards the last 10 turns to the LLM.

### 0.5 `get_episode` is content-only — retrieval hydrates graph fields

`store.get_episode(eid)` returns content fields only (id/timestamp/summary/full_text/
salience/state + the persisted downstream fields). It does **not** populate
`entities`/`topics`/`tones`/`decisions`/`user_id`/`session_id`. `GraphTraversal._hydrate`
fills those by scanning `memory/spo/{eid}/` once and bucketing by predicate, then resolves
the user via one `memory/pos/has_session/{sess}/` scan. Retrieval results are these
**hydrated dicts** (`episode_id`, `summary`, `text`, `timestamp`, `entities`, `topics`,
`tones`, `decisions`, `session_id`, `user_id`, `follows`, `score`). Scoring (§5) operates
on the hydrated dicts, not on `get_episode().entities` (which is always `[]`).

### 0.6 Encoder API

`HippocampalEncoder.encode_turn(user_message, assistant_response) -> Episode` **requires an
open session** (call `start_session(user_id, ...)` first, or use `encode_conversation`).
It sets `timestamp` to `datetime.now().isoformat()` internally and writes
`content/ep/{eid}/ts`. Offline tests that need deterministic timestamps should either (a)
construct an `Episode` directly with an explicit `timestamp` and call
`store.encode_episode(episode)` (the pattern the 1b test suite uses — no encoder, no
session, no GLiNER/Bonsai), or (b) encode then overwrite `content/ep/{eid}/ts` via
`store.db.put_sync`. Do not call `encode_turn` without a session.

---

## 1. What Phase 1c Delivers

| Artifact | Description | Consumer |
|---|---|---|
| **Temporal indexing** | Timestamp range queries (`date_from`/`date_to`) for long chains; coexists with `follows` edges and the existing `temporal_filter` buckets | Graph traversal, query planner |
| **Entity salience tracking** | Entity mention frequency tracked and persisted; weighted into retrieval scoring | Retrieval scoring, ontology decay (Phase 3) |
| **Conversation context for Bonsai** | Last 2-3 turns passed to the query planner for pronoun / implicit-reference resolution | Query planner, retriever, Mode A |
| **Document-level retrieval** *(conditional)* | Documents returned as aggregated nodes with relevant sections highlighted | Retrieval pipeline, Mode A generation |

The first three operate on the **existing conversational corpus** encoded in Phase 1a/1b
(DialogSum/SAMSum). Document-level retrieval (§3) depends on a **document ingestion path
that 1a/1b did not build** — see §3.1 for the prerequisite and the deferral decision.

---

## 2. Updated Project Structure

Only files that change or are added:

```plaintext
hippocampal-memory/
├── src/
│   ├── retrieval/
│   │   ├── graph_traversal.py       # UPDATED: date-range filter, salience-weighted scoring
│   │   ├── query_planner.py         # UPDATED: conversation_context param + date_from/date_to
│   │   ├── retriever.py             # UPDATED: conversation_history threading
│   │   └── document_retriever.py    # NEW (conditional on §3.1 prerequisite)
│   ├── memory/
│   │   └── store.py                 # UPDATED: entity salience persistence (sorted-batch)
│   └── generation/
│       └── mode_a.py                # UPDATED: pass conversation_history to retriever
├── tests/
│   ├── test_graph_traversal.py      # UPDATED: date-range + salience-weighted scoring tests
│   ├── test_query_planner.py        # UPDATED: context-aware planning tests
│   ├── test_retriever.py            # UPDATED: conversation_history threading
│   └── test_document_retriever.py   # NEW (conditional)
└── scripts/
    └── compute_entity_salience.py   # NEW: batch salience from the POS index
```

---

## 3. Refinement 1: Document-Level Retrieval (CONDITIONAL — see 3.1)

### 3.1 Prerequisite & deferral decision

**Phase 1a/1b only encoded conversations.** The encoding pipeline
(`HippocampalEncoder` → `Episode.from_extraction` → `store.encode_episode`) models one
episode per conversational turn, scoped to a User → Session → Episode hierarchy. There is
**no document ingestion pipeline**, no `has_section` / `child_of` edges, and no `doc/`
keys in any existing database. DialogSum and SAMSum are conversational corpora.

Refinement 1 therefore **cannot be exercised on the current corpus**. Two options:

- **(A) Build a document ingestion path as a Phase 1c sub-step** (a `DocumentEncoder`
  that splits a PDF/markdown document into section `Episode`s, writes
  `content/doc/{doc_id}/...` content keys, and links sections to the document via
  `has_section` / `child_of` triples in the same atomic `batch_sync`), then implement
  `DocumentRetriever` against it.
- **(B) Defer Refinement 1** to a future phase that introduces document ingestion, and
  ship Phase 1c as the three conversational refinements (temporal, salience, context).

**Recommended: (B) defer, unless the user explicitly wants documents in 1c.** The three
conversational refinements are the high-value, immediately-testable work. The design below
is kept so a future document-ingestion phase can implement it verbatim. The implementation
order (§9) puts document retrieval last and marks it optional.

### 3.2 Problem (when document ingestion exists)

When a 100-page PDF is ingested, it becomes ~200 section nodes in the graph. A query
matching 15 sections returns 15 separate results. The context builder treats each as an
independent episode. The LLM receives a wall of text with no indication that sections 3,
7, and 12 are all from the same document.

### 3.3 Solution (when document ingestion exists)

Documents are first-class nodes in the graph. Sections link to their parent document via
`child_of` edges (and the parent has `has_section` edges to each section). Retrieval can
return documents (with relevant sections highlighted) or individual sections depending on
query specificity. Content keys: `content/doc/{doc_id}/title`,
`content/doc/{doc_id}/created_at`, and section content stays under
`content/ep/{section_id}/...` (a section IS an episode with extra edges).

### 3.4 Document Retriever (`src/retrieval/document_retriever.py`)

```python
"""Document-aware retrieval that aggregates sections into documents.

CONDITIONAL: only meaningful once a document ingestion path writes `has_section` /
`child_of` edges and `content/doc/{doc_id}/...` keys. See docs/Phase 1c.md §3.1.
"""

from ..memory.store import HippocampalStore


def _b2s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


class DocumentRetriever:
    """Wraps the graph traversal engine with document-aware result aggregation.

    When a query matches multiple sections from the same document, returns the
    document as a single result with relevant sections highlighted, rather than
    returning each section separately.
    """

    def __init__(self, store: HippocampalStore):
        self.store = store

    def aggregate_results(self, raw_results: list[dict]) -> list[dict]:
        """Aggregate raw section-level results into document-level results.

        Non-document results (conversation episodes — no `child_of` edge) pass
        through unchanged.
        """
        doc_sections: list[dict] = []
        regular_eps: list[dict] = []
        for r in raw_results:
            if self._is_document_section(r["episode_id"]):
                doc_sections.append(r)
            else:
                regular_eps.append(r)

        # Group document sections by parent document.
        doc_groups: dict[str, dict] = {}
        for section in doc_sections:
            doc_id = self._get_parent_document(section["episode_id"])
            group = doc_groups.get(doc_id)
            if group is None:
                group = {
                    "document_id": doc_id,
                    "title": self._get_document_title(doc_id),
                    "sections": [],
                    "best_score": 0.0,
                    "entities": set(),
                    "topics": set(),
                }
                doc_groups[doc_id] = group
            group["sections"].append(section)
            group["best_score"] = max(group["best_score"], section.get("score", 0.0))
            group["entities"].update(section.get("entities", []))
            group["topics"].update(section.get("topics", []))

        doc_results: list[dict] = []
        for doc_id, group in doc_groups.items():
            group["sections"].sort(key=lambda s: s.get("score", 0.0), reverse=True)
            total = self._get_document_section_count(doc_id)
            section_summaries = [
                f"Section '{s.get('heading', 'Untitled')}': {s.get('summary', '')}"
                for s in group["sections"][:5]
            ]
            doc_results.append({
                "episode_id": doc_id,
                "type": "document",
                "score": group["best_score"],
                "summary": (
                    f"Document: {group['title']}\n"
                    f"Relevant sections ({len(group['sections'])} matched):\n"
                    + "\n".join(f"  - {s}" for s in section_summaries)
                ),
                "text": self._build_document_context(group, total),
                "timestamp": self._get_document_timestamp(doc_id),
                "entities": sorted(group["entities"]),
                "topics": sorted(group["topics"]),
                "tones": [],
                "matched_sections": len(group["sections"]),
                "total_sections": total,
            })

        all_results = doc_results + regular_eps
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return all_results

    # ── graph helpers (use the REAL API: execute_sync + .vertices) ──

    def _exec_vertices(self, query) -> list[str]:
        result = query.execute_sync()
        try:
            return list(result.vertices)
        finally:
            result.close()

    def _is_document_section(self, node_id: str) -> bool:
        """A document section has an outgoing `child_of` edge to its parent doc."""
        vids = self._exec_vertices(
            self.store.graph.query().vertex(node_id).out("child_of")
        )
        return len(vids) > 0

    def _get_parent_document(self, section_id: str) -> str:
        vids = self._exec_vertices(
            self.store.graph.query().vertex(section_id).out("child_of")
        )
        return vids[0] if vids else section_id

    def _get_document_section_count(self, doc_id: str) -> int:
        """Count sections via the POS index of `has_section`.

        `has_section` POS keys: `memory/pos/has_section/{section_id}/{doc_id}`.
        Each key is one section; counting keys = section count. (Using a scan,
        not `.out("has_section").execute_sync()`, so a missing/zero result is
        distinguishable and the count is exact.)
        """
        start = "memory/pos/has_section/"
        end = "memory/pos/has_section/\x7f"
        return sum(1 for _ in self.store.db.create_read_stream(start=start, end=end))

    # ── content helpers (point lookups via get_sync — safe on sorted-built DBs) ──

    def _get_document_title(self, doc_id: str) -> str:
        return _b2s(self.store.db.get_sync(f"content/doc/{doc_id}/title")) or doc_id

    def _get_document_timestamp(self, doc_id: str) -> str:
        return _b2s(self.store.db.get_sync(f"content/doc/{doc_id}/created_at"))

    def _build_document_context(self, group: dict, total_sections: int) -> str:
        parts = [
            f"Document: {group['title']}",
            f"Relevant sections: {len(group['sections'])} of {total_sections}",
            "",
        ]
        for s in group["sections"][:5]:
            heading = s.get("heading", "Untitled")
            text = s.get("text") or s.get("summary", "")
            parts.append(f"--- {heading} ---")
            parts.append(text[:500])
            parts.append("")
        return "\n".join(parts)
```

### 3.5 Integration with Retriever (`src/retrieval/retriever.py`)

Only wired in when document ingestion exists. Guard the construction so the retriever
works unchanged on conversation-only corpora:

```python
class HippocampalRetriever:
    def __init__(self, store, planner, auto_load_index=True):
        # ... existing init (takes a planner INSTANCE, not a model name) ...
        # Conditional: only aggregate when a DocumentRetriever is configured.
        self.document_retriever = None  # set externally only if §3.1 path (A) is taken

    def retrieve(self, prompt, conversation_history=None, use_semantic=True) -> list[dict]:
        # ... existing retrieval logic (now threads conversation_history to planner) ...
        results = self.retrieve_with_plan(query_plan)
        if self.document_retriever is not None:
            results = self.document_retriever.aggregate_results(results)
        return results
```

---

## 4. Refinement 2: Temporal Indexing

### 4.1 Problem

`follows` edges work for short-range traversal ("what happened next?"). For "what happened
in June 2025?", walking a 500-episode chain is O(n) and capped at `_MAX_FOLLOWS_HOPS = 5`.
The graph needs timestamp range queries.

### 4.2 Solution

Add `date_from` / `date_to` (ISO date strings) to the query plan and a `_filter_date_range`
helper in `GraphTraversal`. Both mechanisms coexist: `follows` chain + `temporal_filter`
buckets for relative/short-range, `date_from`/`date_to` for absolute long-range. The
filter is O(candidates) with O(1) per episode (one `get_episode` per candidate, same as
the existing `_filter_temporal`).

> **Store-level optimization (deferred):** prefix-scannable timestamp keys
> (`content/ep_by_ts/{YYYY-MM}/{eid}`) would make range queries O(results) instead of
> O(candidates). Not required for 1c — the candidate set after axis filtering is already
> small. Noted as a future store-level optimization; do not block 1c on it.

### 4.3 Graph Traversal Updates (`src/retrieval/graph_traversal.py`)

Add to `retrieve` (after the `temporal_after`/`temporal_before` chain block, before the
`temporal_filter` bucket block):

```python
        # ── NEW: absolute date-range filter (Phase 1c) ──
        date_from = query_plan.get("date_from")
        date_to = query_plan.get("date_to")
        if date_from or date_to:
            candidates = self._filter_date_range(candidates, date_from, date_to)
            if not candidates:
                return []

        if temporal_filter:
            candidates = self._filter_temporal(candidates, temporal_filter)
            if not candidates:
                return []
```

And the helper:

```python
    def _filter_date_range(
        self,
        candidates: set[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> set[str]:
        """Keep candidates whose timestamp falls in [date_from, date_to].

        ``date_from`` / ``date_to`` are ISO date or datetime strings. Either may be
        None (one-sided range). Comparison is on parsed datetimes; a bare date
        ("2025-06-01") is parsed at midnight. O(candidates), O(1) per episode —
        one ``get_episode`` each, same cost shape as ``_filter_temporal``.
        """
        lo = self._parse_dt(date_from) if date_from else None
        hi = self._parse_dt(date_to) if date_to else None
        out: set[str] = set()
        for eid in candidates:
            ep = self.store.get_episode(eid)
            if not ep or not ep.timestamp:
                continue
            ts = self._parse_dt(ep.timestamp)
            if ts is None:
                continue
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts > hi:
                continue
            out.add(eid)
        return out

    @staticmethod
    def _parse_dt(s: str) -> Optional[datetime]:
        """Parse an ISO date/datetime; return None on failure."""
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None
```

`date_from`/`date_to` and `temporal_filter` are mutually exclusive in the prompt (§4.4);
the code tolerates both being set (date range applied first, then bucket) but the planner
should not emit both.

### 4.4 Query Planner Update (`src/retrieval/query_planner.py`)

1. Add `date_from` and `date_to` to `_default_plan()` (both `None`).
2. Add a `BONSAI_QUERY_PROMPT` addendum describing the two fields + mutual-exclusion rule
   (see §6.3 for the full updated prompt, which also adds the conversation-context slot).
3. `_parse_plan` already passes through unknown keys to the default plan — extend it to
   preserve `date_from` / `date_to` / `conversation_context` is NOT a plan field (it's an
   input to the planner, not emitted in the plan).

---

## 5. Refinement 3: Entity Salience Tracking

### 5.1 Problem

All entities are treated equally in retrieval scoring (shipped `_score_candidates`:
`+ _W_ENTITY * |matched entities|`). "Alice" (mentioned 200 times) gets the same per-match
boost as "the barista at the coffee shop" (mentioned once). One-off entities bloat the
ontology and dilute retrieval quality.

### 5.2 Solution

Track entity mention frequency. Weight entity matches in scoring by salience. Foundation
for ontology decay in Phase 3. Salience is **computed in batch from the existing POS
index** (no per-encode increment — see §0.3 on why unsorted incremental writes are
unreliable for `get_sync`).

### 5.3 Salience key layout & persistence (`src/memory/store.py` — additions)

Salience keys live under `content/entity/{entity}/...` (HBTrie content namespace,
point-lookupable via `get_sync` on the sorted-built compact corpora):

- `content/entity/{entity}/mention_count` — int, number of distinct episodes mentioning
  the entity.
- `content/entity/{entity}/last_mentioned` — the episode id of the most recent mention.

> **Slash caveat:** entity strings are assumed `/`-free and NUL-free — the encoder makes
> the same assumption for its `memory/spo/{eid}/has_entity/E:{entity}` keys. If an entity
> may contain `/`, sanitize it (replace `/` and `\x00` with a safe token) before building
> the key, and apply the same sanitization in the scoring read path. The batch script (§5.5)
> should assert/fail on `/`-bearing entities rather than silently producing malformed keys.

```python
    # ── Entity salience (Phase 1c) ──

    def get_entity_salience(self, entity: str) -> float:
        """Salience score in [0, 1]. Phase 1c: mention_count only (log-scaled).

        Recency and structural factors are Phase 3 (GNN salience). Returns 0.0
        for an unknown entity. ``get_sync`` is a point lookup on a single known
        key — safe on the sorted-built compact corpora (see docs/Phase 1c.md §0.3).
        """
        count_str = _b2s(self.db.get_sync(f"content/entity/{entity}/mention_count"))
        if not count_str:
            return 0.0
        try:
            count = int(count_str)
        except ValueError:
            return 0.0
        # Log-scale: 0 mentions -> 0; 1 -> ~0.1; 100 -> ~0.4; capped at 1.0.
        # Keep the curve gentle so a dominant entity doesn't fully drown out
        # less-frequent ones.
        return min(1.0, 0.1 + 0.3 * (count ** 0.5) / 10.0)

    def write_entity_salience_batch(self, counts: dict[str, int],
                                    last_ep: dict[str, str]) -> None:
        """Persist salience for all entities in ONE sorted batch_sync.

        Writes ``content/entity/{entity}/mention_count`` and
        ``content/entity/{entity}/last_mentioned``. Keys are SORTED before
        submission so ``get_sync`` reads on them are reliable (see §0.3). Called
        by ``scripts/compute_entity_salience.py`` — do NOT call per-encode.
        """
        ops: list[dict] = []
        for entity in sorted(counts):
            if "/" in entity or "\x00" in entity:
                continue  # skip malformed-key entities (slash caveat)
            ops.append({"type": "put",
                        "key": f"content/entity/{entity}/mention_count",
                        "value": str(counts[entity])})
            ops.append({"type": "put",
                        "key": f"content/entity/{entity}/last_mentioned",
                        "value": last_ep.get(entity, "")})
        if ops:
            self.db.batch_sync(ops)
```

### 5.4 Salience-Weighted Scoring (`src/retrieval/graph_traversal.py` — update)

Update `_score_candidates` to weight each entity match by salience. This layers onto the
**shipped** method (which takes `hydrated: list[dict]` and uses `r["entities"]`, NOT
`get_episode().entities`):

```python
    def _score_candidates(
        self,
        hydrated: list[dict],
        entities: list[str],
        topics: list[str],
        tones: list[str],
    ) -> list[dict]:
        """Heuristic ranker: axis-match counts × weights + recency ordinal.

        Phase 1c: entity matches are weighted by per-entity salience
        (``store.get_entity_salience``), so a high-salience entity match is worth
        up to 2× a low-salience one. Phase 3 GNN salience replaces this.
        """
        ent_set = {e.lower() for e in entities}
        top_set = {t.lower() for t in topics}
        ton_set = {a.lower() for a in tones}

        # Recency = rank by timestamp within the result set (newest = highest).
        times: list[tuple[str, datetime]] = []
        for r in hydrated:
            try:
                times.append((r["episode_id"], datetime.fromisoformat(r["timestamp"])))
            except (ValueError, TypeError):
                times.append((r["episode_id"], datetime.min))
        times.sort(key=lambda x: x[1])
        recency_rank = {eid: i for i, (eid, _) in enumerate(times)}

        for r in hydrated:
            score = 0.0
            # Entity matches — salience-weighted.
            matched = {e.lower() for e in r["entities"]} & ent_set
            for e in r["entities"]:
                if e.lower() in matched:
                    salience = self.store.get_entity_salience(e)
                    score += _W_ENTITY * (0.5 + 0.5 * salience)  # range _W_ENTITY/2 .. _W_ENTITY
            score += _W_TOPIC * len({t.lower() for t in r["topics"]} & top_set)
            score += _W_TONE * len({a.lower() for a in r["tones"]} & ton_set)
            score += _W_RECENCY * recency_rank[r["episode_id"]]
            r["score"] = score
        return hydrated
```

> Note: `get_entity_salience` is a `get_sync` point lookup per matched entity per
> candidate. For the result-set sizes after axis filtering (≤ `limit`, default 5) this is
> a handful of lookups — negligible. If a future phase expands the candidate set before
> scoring, batch-prefetch salience for the query's entities once (the query entities are
> known before scoring) and pass a `{entity: salience}` map into `_score_candidates`.

### 5.5 Batch Salience Computation (`scripts/compute_entity_salience.py`)

Computes salience from the **POS index of `has_entity`** in ONE scan — not per-episode
hydration, and not the broken `.has("predicate", "has_entity")` builder query (which is
not expressible — see §0.1).

```python
"""Compute entity salience from existing encoded episodes.

Scans the POS index of `has_entity` once (memory/pos/has_entity/{E:entity}/{eid})
to count, per entity, how many distinct episodes mention it, then persists
salience via store.write_entity_salience_batch (one sorted batch_sync).

Usage:
    python scripts/compute_entity_salience.py --db ./data/memory_db
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore


def _strip_prefix(vid: str, prefix: str) -> str:
    return vid[len(prefix):] if vid.startswith(prefix) else vid


def _iter_entity_episode_pairs(store: HippocampalStore):
    """Yield (entity, episode_id) for every has_entity triple.

    POS key = memory/pos/has_entity/{E:entity}/{eid}. One scan over the whole
    has_entity POS subtree — O(total has_entity triples), no per-episode work.
    """
    start = "memory/pos/has_entity/"
    end = "memory/pos/has_entity/\x7f"
    for k, _ in store.db.create_read_stream(start=start, end=end):
        # k = memory/pos/has_entity/{E:entity}/{eid}
        parts = k.split("/", 4)
        if len(parts) < 5:
            continue
        entity_node, eid = parts[3], parts[4]
        yield _strip_prefix(entity_node, "E:"), eid


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute entity salience")
    ap.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    args = ap.parse_args()

    store = HippocampalStore(args.db)

    counts: Counter[str] = Counter()
    last_ep: dict[str, str] = {}
    n_triples = 0
    for entity, eid in _iter_entity_episode_pairs(store):
        counts[entity] += 1
        last_ep[entity] = eid  # last-seen wins; scan order is trie order
        n_triples += 1

    store.write_entity_salience_batch(dict(counts), last_ep)

    print(f"Scanned {n_triples} has_entity triples; salience for {len(counts)} entities.")
    print("Top 20 entities:")
    for entity, count in counts.most_common(20):
        print(f"  {entity}: {count} episodes")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 6. Refinement 4: Conversation Context for Bonsai

### 6.1 Problem

Bonsai plans queries from the current prompt alone. "What did he say about it?" has no
entity to extract. The query planner needs conversation context to resolve pronouns and
implicit references.

### 6.2 Solution

Pass the last 2-3 conversation turns to Bonsai as context. The prompt includes recent
messages so Bonsai can resolve "he" → "Bob" and "it" → "the WAL configuration."

### 6.3 Query Planner Update (`src/retrieval/query_planner.py`)

Three changes to the **shipped** `BonsaiQueryPlanner`:

1. `plan(prompt)` → `plan(prompt, conversation_history=None)`. Thread the history into
   `plan_via_server` and `plan_rule_based`.
2. Add a `{conversation_context}` slot to `BONSAI_QUERY_PROMPT` and the date-range fields
   from §4.4.
3. `plan_rule_based` gains an optional context-aware pre-pass: if `conversation_history`
   is present and the prompt contains a pronoun (`he`/`she`/`it`/`that`/`we`), scan the
   last few turns for capitalized names / topic keywords and inject them into the rule-based
   plan. (Bonsai does this in the server path via the prompt; the rule-based fallback
   needs an explicit heuristic so offline tests can assert pronoun resolution.)

```python
BONSAI_QUERY_PROMPT = """Convert this question into a structured memory query.
Return ONLY valid JSON, no other text.

RECENT CONVERSATION (for context, use it to resolve pronouns and implicit references):
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
- temporal_after: keyword to find an anchor and follow the chain forward, or null
- temporal_before: keyword to find an anchor and follow the chain backward, or null
- temporal_filter: "today", "this_week", "last_week", "this_month", or null
- date_from: ISO date for start of an ABSOLUTE range (e.g., "2025-06-01"), or null
- date_to: ISO date for end of an ABSOLUTE range (e.g., "2025-06-30"), or null
- limit: max episodes to return (default 5)

Use the RECENT CONVERSATION to resolve pronouns and implicit references:
- "he" / "she" → the person from recent context.
- "it" / "that" → the topic/entity from recent context.
- "we discussed" → the people in the conversation as entities.

ABSOLUTE vs RELATIVE time:
- "What happened in June 2025?" → date_from="2025-06-01", date_to="2025-06-30"
- "What did we discuss last week?" → temporal_filter="last_week"
- "What happened between March and May?" → date_from="2025-03-01", date_to="2025-05-31"
- Do NOT set both date_from/date_to and temporal_filter in the same query.

IMPORTANT RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union"
- "What did Alice say about databases?" → entities=["Alice"],
  topics=["database_design"], entity_mode="union"
- "What happened after we implemented morphisms?" → temporal_after="morphism"
- "Why did we choose X over Y?" → topics=["decision_making"],
  entities=["X", "Y"], entity_mode="union"
- Specific person's opinion → entity_mode="union"
- Two things discussed TOGETHER → entity_mode="intersection"

Return ONLY valid JSON:
{{"entities": [], "topics": [], "tones": [], "entity_mode": "union",
  "temporal_after": null, "temporal_before": null,
  "temporal_filter": null, "date_from": null, "date_to": null,
  "limit": 5}}"""
```

The planner (matching the shipped constructor signature — no `gpt-4o-mini`, no `openai`):

```python
class BonsaiQueryPlanner:
    """Converts natural language questions into structured query parameters.

    Talks to the LOCAL Bonsai llama-server at config.bonsai_endpoint via
    requests (OpenAI-compatible /chat/completions). NOT OpenAI. Falls back to
    plan_rule_based on any server failure.
    """

    def __init__(self, model=None, endpoint=None, temperature=None, timeout=30.0):
        self.model = model or config.bonsai_model
        self.endpoint = (endpoint or config.bonsai_endpoint).rstrip("/")
        self.temperature = temperature if temperature is not None else config.bonsai_temperature
        self.timeout = timeout

    def plan(self, prompt: str, conversation_history: list[dict] | None = None) -> dict:
        """Plan a query; fall back to rule-based on any server failure."""
        try:
            return self.plan_via_server(prompt, conversation_history)
        except Exception:
            return self.plan_rule_based(prompt, conversation_history)

    def plan_via_server(self, prompt: str,
                        conversation_history: list[dict] | None = None) -> dict:
        context = self._format_context(conversation_history)
        content = BONSAI_QUERY_PROMPT.format(
            conversation_context=context or "(no prior context)",
            prompt=prompt,
        )
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Bonsai {url} HTTP {resp.status_code}: {resp.text}")
        outer = resp.json()
        return self._parse_plan(outer["choices"][0]["message"]["content"])

    @staticmethod
    def _format_context(conversation_history: list[dict] | None) -> str:
        if not conversation_history:
            return ""
        recent = conversation_history[-6:]  # last ~3 exchanges
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)
```

(`_parse_plan` strips code fences and merges the parsed JSON over `_default_plan()`, which
must now include `date_from`/`date_to` as `None`.)

### 6.4 Retriever Update (`src/retrieval/retriever.py`)

Thread `conversation_history` from `retrieve` into `planner.plan` (shipped `retrieve` has
no `conversation_history` param — this adds it):

```python
    def retrieve(self, prompt: str,
                 conversation_history: list[dict] | None = None,
                 use_semantic: bool = True) -> list[dict]:
        query_plan = self.planner.plan(prompt, conversation_history)
        # ... existing retrieve_with_plan / semantic-fallback logic ...
```

### 6.5 Mode A Generator Update (`src/generation/mode_a.py`)

The shipped `ModeAGenerator.generate(prompt, conversation_history=None,
max_context_tokens=None)` already accepts `conversation_history` and forwards the last 10
turns to the LLM. The **only** 1c change is to also pass it to the retriever so the
**planner** can use it for pronoun resolution:

```python
        episodes = self.retriever.retrieve(
            prompt, conversation_history=conversation_history
        )
```

---

## 7. Testing Strategy

All offline tests use `tmp_path`, construct `Episode` directly (no encoder, no GLiNER,
no Bonsai), call `store.encode_episode`, and scan `memory/spo/...` / `content/ep/...` via
`store.db.create_read_stream(start=..., end="...\x7f")`. Keep the NUL-free scan assertion
(`not any("\x00" in k for k, _ in stream)`). Use `_scoped_episode(eid, user, session, ts)`
from `tests/test_session_user.py` as the reusable builder. Inject a literal `query_plan`
dict (no planner) for traversal tests; use `plan_rule_based` for planner tests.

### 7.1 Temporal Indexing Tests (`tests/test_graph_traversal.py` — additions)

```python
def test_date_range_query(tmp_path):
    """Retrieves episodes within an explicit absolute date range."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    for eid, ts in [("ep_000001", "2025-06-15T10:00:00"),
                    ("ep_000002", "2025-07-15T10:00:00"),
                    ("ep_000003", "2025-08-15T10:00:00")]:
        store.encode_episode(_scoped_episode(eid, "u", "S:0001", ts,
                                             summary=f"{eid} discussion"))
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "date_from": "2025-06-01", "date_to": "2025-07-31", "limit": 5,
    })
    ids = {r["episode_id"] for r in results}
    assert "ep_000001" in ids and "ep_000002" in ids and "ep_000003" not in ids
    store.close()


def test_date_range_and_entity_combined(tmp_path):
    """Absolute date range combined with an entity filter."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    store.encode_episode(_scoped_episode("ep_000001", "u", "S:0001",
                                         "2025-06-15T10:00:00",
                                         entities=["Alice"], summary="Alice June"))
    store.encode_episode(_scoped_episode("ep_000002", "u", "S:0001",
                                         "2025-06-15T10:00:00",
                                         entities=["Bob"], summary="Bob June"))
    store.encode_episode(_scoped_episode("ep_000003", "u", "S:0001",
                                         "2025-08-15T10:00:00",
                                         entities=["Alice"], summary="Alice August"))
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "entities": ["Alice"], "entity_mode": "union",
        "date_from": "2025-06-01", "date_to": "2025-07-31", "limit": 5,
    })
    ids = {r["episode_id"] for r in results}
    assert "ep_000001" in ids
    assert "ep_000002" not in ids   # Bob, not Alice
    assert "ep_000003" not in ids   # August, outside the range
    store.close()
```

### 7.2 Entity Salience Tests (`tests/test_graph_traversal.py` — additions)

```python
def test_salience_weighted_scoring(tmp_path):
    """High-salience entity matches score higher than low-salience ones."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    store.encode_episode(_scoped_episode("ep_000001", "u", "S:0001",
                                         "2025-01-01T00:00:00",
                                         entities=["Alice"], summary="Alice said X"))
    store.encode_episode(_scoped_episode("ep_000002", "u", "S:0001",
                                         "2025-01-02T00:00:00",
                                         entities=["Bob"], summary="Bob said Y"))
    # Persist salience via the sorted-batch store method (NOT raw put_sync, so the
    # keys are sorted-written and get_sync-safe).
    store.write_entity_salience_batch(
        counts={"Alice": 50, "Bob": 1},
        last_ep={"Alice": "ep_000001", "Bob": "ep_000002"},
    )
    traversal = GraphTraversal(store)
    results = traversal.retrieve({
        "entities": ["Alice", "Bob"], "entity_mode": "union", "limit": 5,
    })
    by_ent = {r["episode_id"]: r for r in results}
    assert by_ent["ep_000001"]["score"] > by_ent["ep_000002"]["score"]
    store.close()
```

### 7.3 Context-Aware Planning Tests (`tests/test_query_planner.py` — additions)

These exercise `plan_rule_based` (no server). The rule-based pronoun pre-pass must inject
entities/topics from `conversation_history` when the prompt contains a pronoun.

```python
def test_pronoun_resolution_with_context():
    """Rule-based planner resolves pronouns from conversation history."""
    planner = BonsaiQueryPlanner()
    history = [
        {"role": "user", "content": "What did Bob say about the WAL config?"},
        {"role": "assistant", "content": "Bob said the WAL config needed better docs."},
    ]
    plan = planner.plan("What did he suggest we do about it?", history)
    assert "Bob" in plan["entities"]
    # "it" -> WAL config -> configuration topic (rule-based heuristic).
    assert "configuration" in plan["topics"]


def test_no_context_still_works():
    """Planner works without conversation history (backward compatible)."""
    planner = BonsaiQueryPlanner()
    plan = planner.plan("What was I frustrated about?")
    assert "frustrated" in plan["tones"]
```

### 7.4 Document Retrieval Tests (`tests/test_document_retriever.py` — conditional)

Only when §3.1 path (A) is taken. Uses `store.graph.insert_sync(s, p, o)` (valid API —
see §0.1) for fixture setup and `content/doc/{doc_id}/...` content keys:

```python
def test_aggregate_document_sections(tmp_path):
    store = HippocampalStore(str(tmp_path / "test_db"))
    doc_id = "doc_001"
    store.graph.insert_sync(doc_id, "has_section", "sec_001")
    store.graph.insert_sync(doc_id, "has_section", "sec_002")
    store.graph.insert_sync("sec_001", "child_of", doc_id)
    store.graph.insert_sync("sec_002", "child_of", doc_id)
    store.db.batch_sync([
        {"type": "put", "key": f"content/doc/{doc_id}/title", "value": "Test Document"},
        {"type": "put", "key": f"content/doc/{doc_id}/created_at", "value": "2025-06-01"},
    ])
    raw = [
        {"episode_id": "sec_001", "score": 15.0, "summary": "Section 1",
         "text": "Full 1", "timestamp": "2025-06-01",
         "entities": ["Alice"], "topics": ["database_design"], "tones": []},
        {"episode_id": "sec_002", "score": 10.0, "summary": "Section 2",
         "text": "Full 2", "timestamp": "2025-06-01",
         "entities": ["Bob"], "topics": ["configuration"], "tones": []},
        {"episode_id": "ep_000001", "score": 20.0, "summary": "Regular ep",
         "text": "Full", "timestamp": "2025-06-01",
         "entities": ["Alice"], "topics": ["test"], "tones": ["curious"]},
    ]
    out = DocumentRetriever(store).aggregate_results(raw)
    assert len(out) == 2  # 1 document + 1 regular episode
    doc = [r for r in out if r.get("type") == "document"][0]
    assert doc["matched_sections"] == 2
    assert "Alice" in doc["entities"] and "Bob" in doc["entities"]
    ep = [r for r in out if r.get("type") != "document"][0]
    assert ep["episode_id"] == "ep_000001"
    store.close()
```

---

## 8. Checkpoint Criteria

Phase 1c is complete when:

- [ ] `GraphTraversal` supports `date_from` and `date_to` absolute range queries
- [ ] Date range queries combine with entity, topic, and tone filters
- [ ] `BonsaiQueryPlanner.plan` accepts optional `conversation_history` (backward compatible)
- [ ] Pronoun / implicit-reference resolution works in `plan_rule_based` with context
- [ ] `HippocampalRetriever.retrieve` threads `conversation_history` to the planner
- [ ] `ModeAGenerator.generate` passes `conversation_history` to the retriever
- [ ] `scripts/compute_entity_salience.py` runs on the populated database (POS-index scan)
- [ ] Entity salience is persisted via `store.write_entity_salience_batch` (sorted batch)
- [ ] Retrieval scoring weights entity matches by `get_entity_salience`
- [ ] High-salience entities rank higher than low-salience entities in results
- [ ] All existing Phase 1b tests still pass (no regressions)
- [ ] All new tests pass
- [ ] de-wonk pass clean (Hippo CLAUDE.md gate) over all new/modified files
- [ ] *(Conditional, only if §3.1 path A taken)* `DocumentRetriever` aggregates sections

---

## 9. Implementation Order

1. **Entity salience** — `compute_entity_salience.py` (POS-index scan) +
   `store.write_entity_salience_batch` + `store.get_entity_salience` + scoring update in
   `graph_traversal.py`. Run the script on the existing populated compact corpus from
   Phase 1b.
2. **Temporal indexing** — `date_from`/`date_to` in `_default_plan()` + `_filter_date_range`
   in `graph_traversal.py` + prompt addendum.
3. **Conversation context** — `plan(prompt, conversation_history)` + rule-based pronoun
   pre-pass + retriever threading + Mode A retriever call update.
4. **Tests** — new tests for refinements 1-3 + regression check on the Phase 1b suite.
5. **Integration test** — full pipeline with all three conversational refinements active
   (on the pod, against the compact corpus).
6. **(Optional, conditional on §3.1 path A)** Document ingestion path + `DocumentRetriever`
   + `test_document_retriever.py`.
7. **de-wonk** — run the de-wonk skill over all new/modified files; loop until a round
   produces no new CRITICAL/HIGH/MEDIUM issues.

---

## 10. What Phase 1c Does NOT Do

These remain for later phases:

- **Document ingestion pipeline** — required for §3; not built in 1a/1b; deferred unless
  the user opts into path (A) this phase.
- **Prefix-scannable timestamp index** (`content/ep_by_ts/{YYYY-MM}/...`) — noted §4.2;
  not needed for 1c candidate-set sizes.
- **SSM chunking** — Phase 2.5
- **JEPA presentation gating** — Phase 2.5
- **GNN salience scoring** — Phase 3 (replaces heuristic entity salience)
- **Ontology decay** — Phase 3 (uses entity salience tracking from this phase)
- **Cross-document deduplication** — Phase 3
- **Multi-domain routing** — Phase 2
- **Uncertainty detection** — Phase 4

---

## 11. Next Phase

After Phase 1c checkpoint is met, proceed to **Phase 1d: Training Data Generation** which
uses the Oracle to label subgraphs for GNN training, generate Bonsai query planning pairs,
and prepare all training data needed for Phase 2 (JEPA-Gated SSM) and Phase 3 (GNN
Consolidator).

---

Begin with step 1. Report after each step. The entity salience computation should be run
on the existing populated compact corpus from Phase 1b (sorted-built, so `get_sync` is
safe). If any refinement causes a regression in the Phase 1b integration tests, pause and
investigate before continuing. If a WaveDB bug surfaces, fix it in the WaveDB repo, commit,
push, and republish per the 0.1.14 flow (no `Co-Authored-By`).