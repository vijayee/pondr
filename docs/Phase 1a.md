# Phase 1a: Encoding Pipeline — Implementation Plan for Claude Code

## Overview

**Goal:** Build a working pipeline that consumes raw conversation text and produces structured episodes stored in WaveDB. This is the foundation of the entire hippocampal memory architecture.

**What "done" looks like:** A Python script that takes a conversation file, runs GLiNER2 + GLiNER-Decoder + Bonsai extraction, and stores the result as an Episode in WaveDB. A test suite that verifies extraction quality against known examples.

**Duration estimate:** 2-3 days of focused implementation.

> **Status note (2026-07-05):** The User / Session / Episode hierarchy,
> persisted global `episode_counter` / `session_counter`, and the
> `at_time` / `follows_session` edges were added as task #17 *after* the
> code samples below were written. The Episode / Encoder / Store samples
> further down (§2.1, §3, §6, §9) reflect the **pre-hierarchy** model and
> are kept for the encoding-pipeline narrative; the
> [User / Session / Episode hierarchy](#user--session--episode-hierarchy-task-17-added-2026-07-05)
> section below supersedes them for user/session/episode scoping. Retrieval
> over these edges is Phase 1b (Gremlin-style graph traversal) — see
> `docs/Phase 1b.md` and `docs/Ponder Engine Phases.md`.

---

## Architecture Context: Where Phase 1a Sits

The hippocampal memory system is built in seven phases. Phase 1a is the
**extraction** phase — it turns raw conversation text into structured episodes
stored in WaveDB. The later phases read what 1a produces, so 1a must emit
episodes whose schema is forward-compatible with them, and it must store them
atomically (content + graph index together) so downstream phases never see a
half-written episode.

### The v2 seven-phase plan

| Phase | Name | What it adds |
|-------|------|--------------|
| **1a** | Extraction | GLiNER-Decoder + GLiNER2 + Bonsai → Episode → WaveDB; **atomic content+graph writes via `GraphLayer.expand_triple` in one `WaveDB.batch_sync`** (this plan) |
| **1b** | Storage refinement | Subtree layout finalization, supersession predicates (`state` / `validity_*` / `supersedes`) driven by retrieval, reconsolidation |
| **1c** | Retrieval | Cue → graph traversal → episode reactivation; Bonsai query planning |
| **1d** | Training data | Oracle labeling for GNN / SSM / gate training sets |
| **2** | Shared Backbone + Retrieval Gate | JEPA-gated SSM shared encoder; salience gate decides what persists |
| **3** | GNN Consolidator + Forgetting | Graph consolidation, utility decay, saturation, forgetting |
| **4** | Instance-Specific Gates | Per-instance gating of retrieval / persistence |
| **5** | Evaluation | F1 / recall benchmarks across axes |
| **6** | Process Learning + Delegation | Learn recurring workflows; delegate sub-tasks |
| **7** | Self-Generated Training | Agent generates its own training data |

### Developmental stages

Each component matures through stages tracked against F1 thresholds:

- **INFANT** — component exists, below threshold (where Phase 1a's extractors start)
- **CHILD** — F1 > 0.85 on its axis (the Phase 1a transition target)
- later stages defined in Phase 5

### What Phase 1a enables

- **1b** — episodes are already in WaveDB with `state` / `validity_start` / `validity_end` fields and an atomic write path, ready for supersession predicates and subtree reorganization.
- **1c** — `entities` / `topics` / `tones` / `decisions` / `relations` / `follows` are graph-indexed, so cue-to-traversal retrieval works directly on 1a output.
- **1d** — the 20 hand-labeled sample conversations + extraction results become oracle labels for downstream training.
- **2** — `salience` and `retrieval_count` / `ltp_phase` / `utility_decay_rate` fields are present for the JEPA gate to read and update.
- **3** — `saturation_flags`, `retrieval_timestamps`, `consolidation_window_start` are reserved on the Episode so the consolidator can populate them without a schema migration.
- **4 / 5 / 6 / 7** — a stable, queryable episode graph to gate, evaluate, and learn over.

### What Phase 1a does NOT do

- **No JEPA salience** — `salience` is a static default (0.5); the JEPA gate (Phase 2) computes it.
- **No retrieval-weighted persistence** — episodes persist once; retrieval does not yet boost salience or update `retrieval_count` / `retrieval_timestamps`.
- **No reconsolidation counting** — `supersedes` / `validity_end` are modeled but not yet driven by retrieval.
- **No ontology evolution** — the seed ontology is loaded once; GLiNER-Decoder discoveries are buffered, not promoted.
- **No consolidation** — no GNN, no forgetting, no saturation logic.
- **No subconscious routing** — every episode is encoded the same way; no gating of what enters long-term memory.

Phase 1a's `store.py` writes content and graph indices through **one atomic
`WaveDB.batch_sync`** using `GraphLayer.expand_triple` (shipped in WaveDB
0.1.4) — an episode is either fully stored or not at all. Driving supersession
predicates from retrieval, and any subtree reorganization, are deferred to
Phase 1b.

---

## User / Session / Episode hierarchy (task #17, added 2026-07-05)

The encoding pipeline is scoped to a **User → Session → Episode** chain so a
chat history is first-class in the graph, not just a flat list of episodes.

**Classes** (registered in `src/memory/ontology.py`):
- `User subClassOf Person` — the agent's owner / a persona. Existing
  `Person`-typed relations (`madeBy`, `explains`, `pairs_on`, …) accept a
  `User` too.
- `Session subClassOf Event` — one chat / conversation. A `User` owns
  `Session`s; a `Session` contains `Episode`s.

**Edges** (`CONVERSATIONAL_PROPERTIES`):
- `has_session` (User → Session), `has_episode` (Session → Episode),
  `in_session` (Episode → Session), `follows_session` (Session → Session).
- `follows` (Episode → Episode) chains episodes *within* a session;
  `follows_session` chains a user's sessions across chats (cross-chat
  temporal order).

**Literal data edges** — `at_time`, `started_at`, `ended_at`, `state`,
`validity_start`, `validity_end` — are written as graph triples but are
**intentionally NOT registered** in `CONVERSATIONAL_PROPERTIES`: literal
timestamps have no class-typed range, so they are data, not structure. See
the comment in `ontology.py` (`CONVERSATIONAL_PROPERTIES` block).

**Persisted counters** (`content/system/...` in HBTrie):
- `episode_counter`, `session_counter` — global, monotonic; allocate ids
  `ep_NNNN` / `sess_NNNN`.
- `last_session/{user_id}` — per-user pointer to the most recent session,
  used to build the `follows_session` chain.

**Session-scoped encoder** (`src/encoding/encoder.py`):
`HippocampalEncoder(store, user_id=...)` exposes `start_session()` /
`encode_turn(user, assistant)` / `end_session()`; `last_episode_id` resets
per session so the `follows` chain is intra-session. Each conversation in
`scripts/process_corpus.py` is one session (`encode_conversation` → one
session), and `--user` is required.

**Scope decision:** retrieval over these edges (list a user's episodes,
episodes in a time range, rehydrate a session's scope) is **Phase 1b** — the
Gremlin-style `src/retrieval/graph_traversal.py` engine and Phase 1c
timestamp range queries. No Python recall helpers were added to the store
for #17; #17's job is to *write* the hierarchy scaffolding, not read it
back.

---

## 1. Project Setup

### 1.1 Directory Structure

```plaintext
hippocampal-memory/
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py          # WaveDB wrapper
│   │   ├── episode.py        # Episode data model
│   │   └── ontology.py       # Seed ontology (conversation + code)
│   └── encoding/
│       ├── __init__.py
│       ├── gliner_extractor.py
│       ├── bonsai_relations.py
│       └── encoder.py        # Orchestrator
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_episode.py
│   ├── test_store.py
│   ├── test_gliner_extractor.py
│   ├── test_bonsai_relations.py
│   └── test_encoder.py
├── scripts/
│   ├── process_corpus.py     # Batch process a conversation corpus
│   └── generate_training_data.py  # Oracle labeling for GNN (Phase 1d)
├── data/
│   ├── sample_conversations.jsonl   # 20 hand-crafted test conversations
│   └── test_corpus/                 # Small corpus for integration testing
└── notebooks/
    └── extraction_quality.ipynb
```

### 1.2 Dependencies (`pyproject.toml`)

```toml
[project]
name = "hippocampal-memory"
version = "0.1.0"
description = "Brain-inspired memory architecture for AI agents"
requires-python = ">=3.10"
dependencies = [
    "wavedb>=0.1.4",  # GraphLayer.expand_triple (atomic content+graph batch) shipped in 0.1.4
    "gliner2",
    "gliner",
    "openai>=1.0.0",
    "pydantic>=2.0",
    "numpy",
    "python-dotenv",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "black",
    "ruff",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

### 1.3 Configuration (`src/config.py`)

```python
"""Central configuration for the hippocampal memory system."""

import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # WaveDB
    db_path: str = os.getenv("HIPPOCAMPAL_DB_PATH", "./data/memory_db")
    lru_memory_mb: int = 100
    wal_sync_mode: str = "debounced"
    
    # GLiNER
    gliner2_model: str = "fastino/gliner2-base-v1"
    gliner_decoder_model: str = "knowledgator/gliner-decoder-base-v1.0"
    extraction_threshold: float = 0.3
    
    # Bonsai (small LLM for relations and query planning)
    bonsai_model: str = os.getenv("BONSAI_MODEL", "gpt-4o-mini")
    bonsai_temperature: float = 0.1
    
    # Encoding
    episode_salience_default: float = 0.5
    discovery_buffer_threshold: int = 10  # promote label after N occurrences
    
    # ── Phase 2+ (Shared Backbone + Retrieval Gate) — placeholders, unused in 1a ──
    ssm_state_dim: int = 512
    jepa_backbone_model: str = "mamba-2.8b"

    # ── Phase 3+ (GNN Consolidator) — placeholder, unused in 1a ──
    gnn_hidden_dim: int = 256

    # ── Phase 4+ (Instance-Specific Gates) — placeholder, unused in 1a ──
    gate_hidden_dim: int = 128

    # ── Forgetting system (Phase 3+) — placeholders, unused in 1a ──
    saturation_threshold: int = 5
    boost_half_life_days: float = 7.0
    min_decay_rate: float = 0.001

    # Paths
    data_dir: Path = Path("./data")
    sample_conversations: Path = Path("./data/sample_conversations.jsonl")

config = Config()
```

---

## 2. Data Models

### 2.1 Episode (`src/memory/episode.py`)

```python
"""Episode data model — the atomic unit of episodic memory."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Episode:
    """
    One complete conversational exchange (user message + assistant response).
    
    This is the atomic unit of encoding. It's the smallest unit that contains
    all information needed for retrieval: who, what, how felt, what decided,
    what next.
    """
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

    # ── Downstream-system fields (populated by Phase 1a, read by Phase 2-4) ──
    # Present from the start so the store schema never has to migrate later.
    # Phase 1a writes safe defaults; later phases update them on retrieval /
    # consolidation. See "What Phase 1a Enables" below.
    retrieval_count: int = 0
    ltp_phase: str = "early"  # "early" | "late" — long-term potentiation stage
    consolidation_window_start: Optional[str] = None
    utility_decay_rate: float = 0.01
    retrieval_timestamps: list[str] = field(default_factory=list)
    saturation_flags: int = 0
    
    def __post_init__(self):
        if self.validity_start is None:
            self.validity_start = self.timestamp
    
    @classmethod
    def from_extraction(cls, episode_id: str, user_message: str, 
                        assistant_response: str, extracted: dict,
                        relations: list[dict], follows: Optional[str] = None):
        """Create an Episode from extraction results."""
        full_text = f"User: {user_message}\nAssistant: {assistant_response}"
        timestamp = datetime.now().isoformat()
        
        # Simple summary: first 200 chars of assistant response
        summary = assistant_response[:200]
        if len(assistant_response) > 200:
            summary += "..."
        
        return cls(
            id=episode_id,
            timestamp=timestamp,
            summary=summary,
            full_text=full_text,
            entities=extracted.get("entities", []),
            topics=extracted.get("topics", []),
            tones=extracted.get("tones", []),
            decisions=extracted.get("decisions", []),
            relations=relations,
            follows=follows,
        )
```

### 2.2 Seed Ontology (`src/memory/ontology.py`)

Include the full conversation + code ontology from our discussion. The file should contain:

The seed merges four halves — a **conversational** taxonomy (Episode-side:
entities, events, tones, topics), a **code** taxonomy (code-artifact side),
a **development** taxonomy (language-agnostic code semantics, dev artifacts,
security), and a **business/organizational** taxonomy (stakeholders,
requirements, workflows, regulations, organizations, roles) — into one
**multi-parent DAG**: a class may have several `subClassOf` parents, which is
exactly what the Graph layer stores. The halves overlap on a few names
(`Database`, `Field`, `Cache`, `Configuration`, `Entity`, `Query`, `Schema`,
`Proxy`, `Middleware`, `Migration`, `Patch`, `Conflict`, `Queue`, `Runbook`,
`Vulnerability`, `Standard`, `README`, `Changelog`, `License`, `Event`,
`Constant`); those become multi-parent nodes rather than collisions.

```python
"""Seed ontology for the hippocampal memory system.

Stored in the WaveDB Graph layer at init as ``subClassOf`` triples and evolved
through discovery (GLiNER-Decoder invents labels → buffered → promoted by
Bonsai entailment) and periodic GNN refinement.

The seed merges four taxonomies into one **multi-parent DAG**:
**CONVERSATIONAL** (Episode-side entities, events, tones, topics), **CODE**
(code-artifact structure, VCS, issues, testing, architecture, API, data,
config, ops, observability, quality, process), **DEVELOPMENT**
(language-agnostic code semantics, algorithms, data structures, control flow,
error handling, concurrency, documents, security), and **BUSINESS**
(stakeholders, requirements, workflows, regulations, organizations, roles).
A class may have several ``subClassOf`` parents — exactly what the Graph layer
stores. The four halves overlap on a handful of names (``Database``, ``Field``,
``Cache``, ``Configuration``, ``Entity``, ``Query``, ``Schema``, ``Proxy``,
``Middleware``, ``Migration``, ``Patch``, ``Conflict``, ``Queue``, ``Runbook``,
``Vulnerability``, ``Standard``, ``README``, ``Changelog``, ``License``,
``Event``, ``Constant``); those become multi-parent nodes rather than
collisions. Totals: **376 classes, 165 properties, acyclic, 18 multi-parent
overlap nodes**.

Merge notes (resolved, flagged for review):
* Multi-parent DAG — a class may have several subClassOf parents (e.g.
  Database is Project-side AND Data-side; Field is CodeArtifact-side AND
  Data-side; Cache under API/Data/Infrastructure; Configuration is a Topic AND
  a parent of config artifacts).
* The source code ontology had a duplicate ``implements`` key (Class→Interface
  and Commit→Feature). Kept both: ``implements`` = Class→Interface and
  ``implements_feature`` = Commit→Feature.
* Property domain/range types originally dangling (Repository, Team, Message,
  Permission, Value, plus the types introduced by the development/business
  extensions) are defined as classes, so every relation endpoint is a real
  node. Episode (the root all has* properties hang off) is declared explicitly.
* Beyond the code-artifact taxonomy, the seed also covers development
  (paradigms, algorithms, data structures, control flow, error handling,
  concurrency, documents, knowledge, communication, security) and
  business/organizational concepts (stakeholders, requirements, business
  rules, workflows, products, markets, regulations, organizations, roles) so
  conversations about code in any language, the development process, and the
  business/org context around it all index against a shared vocabulary.
"""

from typing import Any

# ── Conversational (Episode-side) taxonomy ──
CONVERSATIONAL_CLASSES = {
    # Episode is the root: every recorded turn is an instance of Episode, and
    # the has* properties hang off it. Declared explicitly so it's a real node.
    "Episode":      [],
    "Entity":       ["Person", "Project", "Technology", "Concept"],
    "Project":      ["Database", "Application", "Library"],
    "Technology":   ["Protocol", "Language", "Framework"],
    "Event":        ["Decision", "Explanation", "Question", "Conflict"],
    "AffectiveTone": ["Frustrated", "Excited", "Curious", "Neutral"],
    "Topic": [
        "DatabaseDesign", "Configuration", "Performance",
        "Security", "APIDesign", "AIArchitecture",
    ],
    "Statement": [],  # leaf; referenced by the `contradicts` property
}

CONVERSATIONAL_PROPERTIES = {
    "hasEntity":   {"domain": "Episode",   "range": "Entity"},
    "hasTopic":     {"domain": "Episode",   "range": "Topic"},
    "hasTone":      {"domain": "Episode",   "range": "AffectiveTone"},
    "hasDecision":  {"domain": "Episode",   "range": "Decision"},
    "madeBy":       {"domain": "Decision",  "range": "Person"},
    "about":        {"domain": "Decision",  "range": "Topic"},
    "explains":     {"domain": "Person",    "range": "Concept"},
    "contradicts":  {"domain": "Statement", "range": "Statement"},
    "follows":      {"domain": "Episode",   "range": "Episode"},
    "supersedes":   {"domain": "Episode",   "range": "Episode"},  # reconsolidation
    "subClassOf":   {"domain": "Entity",    "range": "Entity"},    # taxonomy edges
}

# ── Code taxonomy (from the design discussion) ──
CODE_CLASSES = {
    # Code structure (AST-level artifacts).
    "CodeArtifact": [
        "File", "Module", "Package",
        "Class", "Interface", "Trait", "Mixin", "Enum", "Struct",
        "Function", "Method", "Constructor", "Destructor",
        "Property", "Attribute", "Field",
        "Variable", "Constant", "Parameter",
        "Type", "Generic", "Union", "Alias",
        "Decorator", "Annotation",
        "Lambda", "Closure", "Generator",
        "Expression", "Statement", "Block",
    ],
    # Version control.
    "VersionControl": [
        "Repository",
        "Commit", "Branch", "Tag", "Release",
        "PullRequest", "MergeRequest", "Patch",
        "Merge", "Rebase", "CherryPick",
        "Conflict", "Diff", "Blame",
        "Fork", "Clone", "Remote",
        "Stash", "Worktree",
    ],
    # Issue tracking.
    "Issue": [
        "Bug", "Feature", "Enhancement", "Task",
        "TechnicalDebt", "Refactor",
        "PerformanceIssue", "SecurityVulnerability",
        "Regression", "BreakingChange",
        "Deprecation", "Migration",
    ],
    # Testing.
    "Test": [
        "UnitTest", "IntegrationTest", "EndToEndTest",
        "PerformanceTest", "SecurityTest", "RegressionTest",
        "Mock", "Stub", "Fixture", "TestSuite",
        "Coverage", "Assertion",
    ],
    # Architecture & design.
    "Architecture": [
        "DesignPattern", "ArchitecturalPattern",
        "Component", "Service", "Microservice",
        "Monolith", "Plugin", "Middleware",
        "Layer", "Tier", "Boundary",
        "Adapter", "Facade", "Proxy", "Bridge",
        "Factory", "Singleton", "Observer", "Strategy",
    ],
    # API.
    "API": [
        "Endpoint", "Route", "Controller",
        "Middleware", "Guard", "Interceptor",
        "Request", "Response", "DTO", "Schema",
        "Query", "Mutation", "Subscription",
        "REST", "GraphQL", "gRPC", "WebSocket",
        "RateLimit", "Throttle", "Cache",
    ],
    # Data.
    "Data": [
        "Database", "Table", "Collection",
        "Column", "Field", "Index",
        "PrimaryKey", "ForeignKey", "Constraint",
        "Query", "Migration", "Seed",
        "Schema", "Model", "Entity", "Relation",
        "Transaction", "Lock", "Deadlock",
        "Cache", "Session", "Connection",
    ],
    # Configuration.
    "Configuration": [
        "EnvironmentVariable", "ConfigFile",
        "Secret", "Credential", "APIKey",
        "FeatureFlag", "Toggle",
        "Profile", "BuildConfig", "Value",
    ],
    # Operations / DevOps.
    "Infrastructure": [
        "Server", "Container", "Pod", "Cluster",
        "LoadBalancer", "Proxy", "CDN",
        "Database", "Queue", "Cache",
        "Volume", "Network", "Firewall",
        "DNS", "Certificate",
    ],
    "Deployment": [
        "Pipeline", "Stage", "Job", "Step",
        "Build", "Test", "Deploy", "Rollback",
        "Artifact", "Image", "Registry",
        "Environment", "Namespace",
        "HelmChart", "Manifest", "Template",
    ],
    "Observability": [
        "Log", "Metric", "Trace", "Span",
        "Alert", "Incident", "Runbook",
        "Dashboard", "Monitor", "SLO",
        "Error", "Warning", "Debug",
    ],
    # Quality & process.
    "Quality": [
        "Lint", "Format", "StyleGuide",
        "CodeReview", "Audit",
        "Complexity", "Duplication",
        "Documentation", "README", "Changelog",
        "License", "Dependency",
        "Vulnerability", "CVE", "Patch",
    ],
    "Process": [
        "Sprint", "Milestone", "Roadmap",
        "Estimate", "StoryPoint",
        "Standup", "Retrospective",
        "OnCall", "IncidentResponse",
        "PostMortem", "RCA",
    ],
}

CODE_PROPERTIES = {
    # ── Code structure ──
    "contains":          {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "defined_in":        {"domain": "CodeArtifact", "range": "File"},
    "declared_in":       {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "calls":             {"domain": "Function",     "range": "Function"},
    "imports":           {"domain": "File",         "range": "Module"},
    "exports":           {"domain": "Module",       "range": "CodeArtifact"},
    "inherits":          {"domain": "Class",        "range": "Class"},
    "implements":        {"domain": "Class",        "range": "Interface"},
    "implements_feature": {"domain": "Commit",     "range": "Feature"},  # split dup
    "overrides":         {"domain": "Method",      "range": "Method"},
    "uses":              {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "instantiates":      {"domain": "CodeArtifact", "range": "Class"},
    "decorates":         {"domain": "Decorator",    "range": "CodeArtifact"},
    "annotates":         {"domain": "Annotation",   "range": "CodeArtifact"},
    "type_of":           {"domain": "Variable",     "range": "Type"},
    "returns":           {"domain": "Function",     "range": "Type"},
    "accepts":           {"domain": "Function",     "range": "Parameter"},
    "raises":            {"domain": "Function",     "range": "Type"},
    "catches":           {"domain": "Block",        "range": "Type"},

    # ── Version control ──
    "commits":           {"domain": "Branch",       "range": "Commit"},
    "parents":           {"domain": "Commit",       "range": "Commit"},
    "branches_from":     {"domain": "Branch",       "range": "Branch"},
    "merges_into":       {"domain": "Branch",       "range": "Branch"},
    "tags":              {"domain": "Commit",       "range": "Tag"},
    "releases":          {"domain": "Tag",          "range": "Release"},
    "resolves":          {"domain": "Merge",        "range": "Conflict"},
    "cherry_picks":      {"domain": "Commit",       "range": "Commit"},
    "reverts":           {"domain": "Commit",       "range": "Commit"},

    # ── Issue tracking ──
    "fixes":             {"domain": "Commit",       "range": "Bug"},
    "introduces":        {"domain": "Commit",       "range": "Bug"},
    "regresses":         {"domain": "Commit",       "range": "Regression"},
    "refactors":         {"domain": "Commit",       "range": "Refactor"},
    "addresses":         {"domain": "Commit",       "range": "Issue"},
    "closes":            {"domain": "PullRequest",  "range": "Issue"},
    "blocks":            {"domain": "Issue",        "range": "Issue"},
    "depends_on_issue":  {"domain": "Issue",        "range": "Issue"},
    "duplicates":        {"domain": "Issue",        "range": "Issue"},

    # ── Testing ──
    "tests":             {"domain": "Test",        "range": "CodeArtifact"},
    "covers":            {"domain": "TestSuite",    "range": "CodeArtifact"},
    "mocks":              {"domain": "Test",         "range": "CodeArtifact"},
    "asserts":            {"domain": "Test",         "range": "Assertion"},
    "fails":              {"domain": "Test",         "range": "Bug"},
    "regression_tests":  {"domain": "RegressionTest", "range": "Bug"},

    # ── Architecture & design ──
    "depends_on":         {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "depends_on_module":   {"domain": "Module",      "range": "Module"},
    "depends_on_service":  {"domain": "Service",     "range": "Service"},
    "owns":                {"domain": "Team",        "range": "Service"},
    "communicates_with":   {"domain": "Service",     "range": "Service"},
    "proxies":             {"domain": "Proxy",       "range": "Service"},
    "balances":            {"domain": "LoadBalancer", "range": "Service"},
    "caches":              {"domain": "Cache",       "range": "Data"},
    "queues":              {"domain": "Queue",        "range": "Message"},
    "subscribes":          {"domain": "Service",      "range": "Event"},
    "publishes":           {"domain": "Service",      "range": "Event"},

    # ── API ──
    "routes_to":          {"domain": "Route",        "range": "Controller"},
    "handles":            {"domain": "Controller",   "range": "Endpoint"},
    "guards":             {"domain": "Guard",        "range": "Route"},
    "intercepts":         {"domain": "Interceptor",  "range": "Request"},
    "validates":          {"domain": "Middleware",    "range": "Schema"},
    "rate_limits":        {"domain": "RateLimit",    "range": "Endpoint"},
    "authenticates":      {"domain": "Guard",        "range": "Credential"},
    "authorizes":         {"domain": "Guard",        "range": "Permission"},

    # ── Data ──
    "persists":           {"domain": "Repository",   "range": "Entity"},
    "maps_to":            {"domain": "Entity",       "range": "Table"},
    "columns":            {"domain": "Table",        "range": "Column"},
    "references":         {"domain": "ForeignKey",   "range": "PrimaryKey"},
    "indexes":           {"domain": "Index",         "range": "Column"},
    "constrains":         {"domain": "Constraint",   "range": "Column"},
    "migrates":           {"domain": "Migration",    "range": "Schema"},
    "seeds":              {"domain": "Seed",         "range": "Table"},
    "transacts":          {"domain": "Transaction",   "range": "Database"},
    "locks":              {"domain": "Transaction",  "range": "Table"},

    # ── Configuration ──
    "configures":         {"domain": "ConfigFile",          "range": "CodeArtifact"},
    "sets":               {"domain": "EnvironmentVariable", "range": "Value"},
    "secrets":            {"domain": "Secret",       "range": "Credential"},
    "flags":              {"domain": "FeatureFlag",   "range": "Feature"},
    "profiles":           {"domain": "Profile",      "range": "Environment"},

    # ── Deployment ──
    "builds":             {"domain": "Pipeline",     "range": "Artifact"},
    "deploys_to":         {"domain": "Pipeline",     "range": "Environment"},
    "runs_on":            {"domain": "Job",          "range": "Infrastructure"},
    "produces":           {"domain": "Job",          "range": "Artifact"},
    "rolls_back":         {"domain": "Deploy",       "range": "Deploy"},
    "contains_stage":     {"domain": "Pipeline",     "range": "Stage"},
    "contains_job":       {"domain": "Stage",        "range": "Job"},
    "contains_step":      {"domain": "Job",          "range": "Step"},

    # ── Observability ──
    "logs":               {"domain": "CodeArtifact", "range": "Log"},
    "emits":              {"domain": "CodeArtifact", "range": "Metric"},
    "traces":             {"domain": "CodeArtifact", "range": "Span"},
    "alerts_on":          {"domain": "Monitor",      "range": "Metric"},
    "triggers":           {"domain": "Alert",        "range": "Incident"},
    "resolved_by":        {"domain": "Incident",     "range": "Runbook"},
    "caused_by":          {"domain": "Incident",     "range": "Deploy"},
    "postmortem_for":     {"domain": "PostMortem",   "range": "Incident"},

    # ── Quality ──
    "lints":              {"domain": "Lint",         "range": "CodeArtifact"},
    "formats":            {"domain": "Format",       "range": "CodeArtifact"},
    "reviews":            {"domain": "CodeReview",   "range": "PullRequest"},
    "documents":         {"domain": "Documentation", "range": "CodeArtifact"},
    "changelogs":         {"domain": "Changelog",    "range": "Release"},
    "depends_on_lib":     {"domain": "Module",       "range": "Dependency"},
    "vulnerable_in":      {"domain": "Vulnerability", "range": "Dependency"},
    "patches_vuln":       {"domain": "Patch",        "range": "Vulnerability"},

    # ── Process ──
    "scheduled_in":       {"domain": "Issue",        "range": "Sprint"},
    "milestoned_in":      {"domain": "Issue",        "range": "Milestone"},
    "estimated_at":       {"domain": "Issue",        "range": "Estimate"},
    "discussed_in":       {"domain": "Issue",        "range": "Standup"},
    "retrospected_in":    {"domain": "Sprint",       "range": "Retrospective"},
    "action_item_from":   {"domain": "Task",         "range": "Retrospective"},

    # ── Cross-cutting (code ↔ conversation) ──
    "discusses":          {"domain": "Episode",      "range": "CodeArtifact"},
    "modifies":           {"domain": "Episode",      "range": "File"},
    "produces_commit":    {"domain": "Episode",      "range": "Commit"},
    "reviews_code":       {"domain": "Episode",      "range": "PullRequest"},
    "debates":            {"domain": "Episode",      "range": "Issue"},
    "decides_on":         {"domain": "Episode",      "range": "Architecture"},
    "troubleshoots":      {"domain": "Episode",      "range": "Bug"},
    "incident_response":  {"domain": "Episode",      "range": "Incident"},
    "pairs_on":           {"domain": "Person",       "range": "CodeArtifact"},
    "owns_code":          {"domain": "Person",       "range": "CodeArtifact"},
    "reviews_work_of":    {"domain": "Person",       "range": "Person"},
}


# ── Development: language-agnostic code semantics, dev artifacts, security ──
DEVELOPMENT_CLASSES = {
    # Programming paradigms (language-agnostic).
    "Paradigm": [
        "ObjectOriented", "Functional", "Procedural",
        "Declarative", "Reactive", "EventDriven",
    ],
    # Algorithms & complexity (abstract CS concepts).
    "Algorithm": [
        "Sorting", "Search", "DynamicProgramming",
        "Greedy", "DivideAndConquer", "Heuristic",
        "ComplexityClass",
    ],
    "ComplexityClass": [
        "Constant", "Logarithmic", "Linear",
        "Quadratic", "Exponential",
    ],
    # Abstract data structures (semantic; concrete storage under Data).
    "DataStructure": [
        "Array", "List", "Tree", "Graph",
        "HashTable", "Set", "Map",
        "Stack", "Queue", "Heap",
        "Trie", "Tuple", "Record",
    ],
    # Control flow concepts (semantic, not AST nodes).
    "ControlFlow": [
        "Conditional", "Loop", "Recursion",
        "Iteration", "SwitchCase", "Return",
    ],
    # Error / exception handling.
    "ErrorHandling": ["Exception", "Retry", "Fallback", "Recovery"],
    # Concurrency & async.
    "Concurrency": [
        "Thread", "Coroutine", "Async", "Await",
        "Semaphore", "Mutex", "Future", "Promise",
        "Actor", "Pool",
    ],
    # Documents & knowledge artifacts.
    "Document": [
        "Specification", "DesignDoc", "RFC", "ADR",
        "Wiki", "Manual", "Tutorial", "Playbook",
        "Runbook", "README", "Changelog", "License",
    ],
    "Knowledge": [
        "BestPractice", "AntiPattern", "LessonLearned",
        "Pattern", "Convention", "Standard",
    ],
    # Communication / messaging.
    "Communication": ["Message", "Notification", "Channel", "Signal", "Event"],
    # Security & access control.
    "Security": [
        "Permission", "Privilege", "AccessToken", "Scope",
        "ACL", "Identity", "Principal", "Threat", "Vulnerability",
    ],
}

DEVELOPMENT_PROPERTIES = {
    # ── Algorithms & complexity ──
    "uses_algorithm":     {"domain": "Function",     "range": "Algorithm"},
    "has_complexity":     {"domain": "Algorithm",    "range": "ComplexityClass"},
    # ── Documents & knowledge ──
    "describes":          {"domain": "DesignDoc",    "range": "Architecture"},
    "decides":            {"domain": "ADR",          "range": "Decision"},
    "references_doc":     {"domain": "Document",     "range": "CodeArtifact"},
    "documented_in":      {"domain": "CodeArtifact", "range": "Document"},
    "follows_practice":   {"domain": "CodeArtifact", "range": "BestPractice"},
    "avoids":             {"domain": "CodeArtifact", "range": "AntiPattern"},
    # ── Communication ──
    "delivers":           {"domain": "Channel",      "range": "Message"},
    "notifies":           {"domain": "Notification", "range": "Stakeholder"},
    "consumes":           {"domain": "Service",      "range": "Message"},
    "broadcasts":         {"domain": "Service",      "range": "Event"},
    # ── Security & access control ──
    "grants":             {"domain": "Role",          "range": "Permission"},
    "scoped_to":          {"domain": "AccessToken",  "range": "Scope"},
    "identifies":         {"domain": "Credential",   "range": "Identity"},
    "authenticates_with": {"domain": "Principal",    "range": "Credential"},
    "protects":           {"domain": "Security",      "range": "CodeArtifact"},
    "threatens":          {"domain": "Threat",       "range": "CodeArtifact"},
}


# ── Business & organizational concepts ──
BUSINESS_CLASSES = {
    # People / stakeholders in the business context.
    "Stakeholder": ["Customer", "EndUser", "Sponsor", "ProductOwner", "Champion"],
    # Requirements engineering.
    "Requirement": [
        "FunctionalRequirement", "NonFunctionalRequirement",
        "AcceptanceCriterion", "UserStory", "UseCase",
    ],
    # Business rules & governance.
    "BusinessRule": [],
    "Regulation": ["Compliance", "Standard", "Policy", "Law", "SLA"],
    # Workflows / process modeling.
    "Workflow": ["TaskStep", "Transition", "Action", "Trigger", "Gate", "Lane"],
    # Product & market.
    "Product": [],
    "Market": ["Segment", "Competitor", "Trend"],
    "Domain": [],
    "KPI": [],
    # Organization & people structure.
    "Organization": [
        "Company", "Department", "Team",
        "Squad", "Tribe", "Chapter", "Guild",
    ],
    "Role": [
        "Architect", "Engineer", "Manager", "ProductManager",
        "Designer", "QAEngineer", "DevOpsEngineer", "Analyst", "Lead",
    ],
}

BUSINESS_PROPERTIES = {
    # ── Requirements ↔ delivery ──
    "requests":          {"domain": "Stakeholder", "range": "Feature"},
    "defines":           {"domain": "Stakeholder", "range": "Requirement"},
    "specifies":          {"domain": "Requirement", "range": "Feature"},
    "validated_by":      {"domain": "Requirement", "range": "AcceptanceCriterion"},
    "implemented_by":    {"domain": "Feature",     "range": "Commit"},
    # ── Organization & people ──
    "member_of":         {"domain": "Person",       "range": "Team"},
    "leads":              {"domain": "Person",       "range": "Team"},
    "reports_to":        {"domain": "Person",       "range": "Person"},
    "assigned_to":       {"domain": "Task",          "range": "Person"},
    "responsible_for":   {"domain": "Role",          "range": "CodeArtifact"},
    "employed_by":       {"domain": "Person",       "range": "Organization"},
    "owns_team":         {"domain": "Organization", "range": "Team"},
    # ── Business rules, workflow, compliance ──
    "automates":         {"domain": "Workflow",     "range": "Process"},
    "governs":           {"domain": "BusinessRule", "range": "Process"},
    "applies_to":        {"domain": "BusinessRule", "range": "Domain"},
    "complies_with":     {"domain": "CodeArtifact", "range": "Standard"},
    "complies":          {"domain": "Process",      "range": "Regulation"},
    "measured_by":       {"domain": "KPI",          "range": "Metric"},
}


def _merge(classes_parts, properties_parts):
    """Merge taxonomy halves into one multi-parent DAG.

    Subclass lists are unioned (multi-parent). Any subclass name with no
    explicit entry is auto-created as a leaf, so every subClassOf target is a
    real class.
    """
    classes = {}
    for part in classes_parts:
        for name, subs in part.items():
            classes.setdefault(name, set()).update(subs)
    for subs in list(classes.values()):
        for c in subs:
            classes.setdefault(c, set())
    properties = {}
    for part in properties_parts:
        properties.update(part)
    return {
        "classes": {k: {"subclasses": sorted(v)} for k, v in classes.items()},
        "properties": properties,
    }


SEED_ONTOLOGY = _merge(
    [CONVERSATIONAL_CLASSES, CODE_CLASSES, DEVELOPMENT_CLASSES, BUSINESS_CLASSES],
    [CONVERSATIONAL_PROPERTIES, CODE_PROPERTIES, DEVELOPMENT_PROPERTIES, BUSINESS_PROPERTIES],
)
# → 376 classes, 165 properties, acyclic, 18 multi-parent overlap nodes.
```

---

## 3. WaveDB Store (`src/memory/store.py`)

```python
"""WaveDB wrapper for the hippocampal memory system.

Graph layer (``memory`` subtree) = hippocampal index (sparse pointers).
HBTrie (``content/`` subtree) = neocortical store (content).

Each episode is written as ONE atomic ``WaveDB.batch_sync`` that merges content
puts with graph-index ops from ``GraphLayer.expand_triple`` — content and index
share a single transaction / WAL record, so encoding is all-or-nothing.
"""

from wavedb import WaveDB, WaveDBConfig, GraphLayer
from .episode import Episode
from .ontology import SEED_ONTOLOGY


class HippocampalStore:
    """Wraps WaveDB for hippocampal memory operations."""
    
    def __init__(self, db_path: str, config: dict = None):
        cfg = WaveDBConfig(
            lru_memory_mb=config.get("lru_memory_mb", 100) if config else 100,
            wal_sync_mode=config.get("wal_sync_mode", "debounced") if config else "debounced",
        )
        self.db = WaveDB(db_path, config=cfg)
        self.graph = GraphLayer("memory", self.db)
        self._seed_ontology()
    
    def _seed_ontology(self):
        """Store seed ontology in the graph layer."""
        for parent, info in SEED_ONTOLOGY["classes"].items():
            for child in info.get("subclasses", []):
                try:
                    self.graph.insert_sync(child, "subClassOf", parent)
                except Exception:
                    pass  # Already exists
    
    def encode_episode(self, episode: Episode):
        """Store episode content + graph index in ONE atomic batch.

        Content (HBTrie, ``content/ep/...``) and graph index (``memory``
        subtree) are written through a single ``WaveDB.batch_sync`` so an
        episode is either fully stored or not at all — no content without its
        index entries, no index entries without their content. Graph triples
        are expanded into root-namespace ops via ``GraphLayer.expand_triple``
        (shipped in WaveDB 0.1.4) and spliced into the same batch.
        """
        ops: list[dict] = []
        eid = episode.id

        # ── HBTrie: content (neocortical store), root namespace under content/ ──
        ops += [
            {"type": "put", "key": f"content/ep/{eid}/summary", "value": episode.summary},
            {"type": "put", "key": f"content/ep/{eid}/text", "value": episode.full_text},
            {"type": "put", "key": f"content/ep/{eid}/ts", "value": episode.timestamp},
            {"type": "put", "key": f"content/ep/{eid}/salience", "value": str(episode.salience)},
            {"type": "put", "key": f"content/ep/{eid}/state", "value": episode.state},
            # Downstream-system fields (defaults at encode time; Phase 2-4
            # update them on retrieval / consolidation). Persisted now so the
            # store schema is stable from the start.
            {"type": "put", "key": f"content/ep/{eid}/retrieval_count", "value": str(episode.retrieval_count)},
            {"type": "put", "key": f"content/ep/{eid}/ltp_phase", "value": episode.ltp_phase},
            {"type": "put", "key": f"content/ep/{eid}/decay_rate", "value": str(episode.utility_decay_rate)},
        ]

        # ── Graph: sparse pointers (hippocampal index) via expand_triple ──
        # expand_triple returns root-namespace ops (the "memory/" subtree prefix
        # is already prepended by the C helper) — splice into the SAME batch_sync
        # so content and graph indices share one atomic transaction / WAL record.
        for entity in episode.entities:
            ops += self.graph.expand_triple(eid, "has_entity", f"E:{entity}")
            ops += self.graph.expand_triple(f"E:{entity}", "in_episode", eid)
        for topic in episode.topics:
            ops += self.graph.expand_triple(eid, "has_topic", f"T:{topic}")
        for tone in episode.tones:
            ops += self.graph.expand_triple(eid, "has_tone", f"A:{tone}")
        for decision in episode.decisions:
            ops += self.graph.expand_triple(eid, "has_decision", f"D:{decision}")
        for rel in episode.relations:
            ops += self.graph.expand_triple(rel["subject"], rel["predicate"], rel["object"])
        if episode.follows:
            ops += self.graph.expand_triple(eid, "follows", episode.follows)

        # State tracking
        ops += self.graph.expand_triple(eid, "state", episode.state)
        if episode.validity_start:
            ops += self.graph.expand_triple(eid, "validity_start", episode.validity_start)

        self.db.batch_sync(ops)

    def get_episode(self, episode_id: str) -> Episode | None:
        """Load episode from HBTrie."""
        summary = self.db.get_sync(f"content/ep/{episode_id}/summary")
        if not summary:
            return None

        text = self.db.get_sync(f"content/ep/{episode_id}/text") or b""
        ts = self.db.get_sync(f"content/ep/{episode_id}/ts") or b""
        salience_str = self.db.get_sync(f"content/ep/{episode_id}/salience") or b"0.5"
        state = self.db.get_sync(f"content/ep/{episode_id}/state") or b"current"
        retrieval_count_raw = self.db.get_sync(f"content/ep/{episode_id}/retrieval_count") or b"0"
        ltp_phase_raw = self.db.get_sync(f"content/ep/{episode_id}/ltp_phase") or b"early"
        decay_rate_raw = self.db.get_sync(f"content/ep/{episode_id}/decay_rate") or b"0.01"

        # Decode bytes to strings
        summary = summary.decode() if isinstance(summary, bytes) else summary
        text = text.decode() if isinstance(text, bytes) else text
        ts = ts.decode() if isinstance(ts, bytes) else ts
        state = state.decode() if isinstance(state, bytes) else state
        retrieval_count_str = retrieval_count_raw.decode() if isinstance(retrieval_count_raw, bytes) else retrieval_count_raw
        ltp_phase = ltp_phase_raw.decode() if isinstance(ltp_phase_raw, bytes) else ltp_phase_raw
        decay_rate_str = decay_rate_raw.decode() if isinstance(decay_rate_raw, bytes) else decay_rate_raw

        return Episode(
            id=episode_id,
            timestamp=ts,
            summary=summary,
            full_text=text,
            salience=float(salience_str.decode() if isinstance(salience_str, bytes) else salience_str),
            state=state,
            retrieval_count=int(retrieval_count_str),
            ltp_phase=ltp_phase,
            utility_decay_rate=float(decay_rate_str),
        )
    
    def close(self):
        self.db.close()
```

---

## 4. GLiNER Extractor (`src/encoding/gliner_extractor.py`)

```python
"""GLiNER-based entity extraction for hippocampal memory.

Two models:
- GLiNER-Decoder: open discovery (invents labels freely)
- GLiNER2: stable extraction against evolved schema
"""

from collections import defaultdict
from gliner import GLiNER
from gliner2 import GLiNER2


class GLiNERExtractor:
    """
    Extracts entities, topics, tones, and decisions from conversation text.
    
    GLiNER-Decoder discovers new entity types.
    GLiNER2 extracts against the stable schema.
    """
    
    def __init__(self, gliner2_model: str = "fastino/gliner2-base-v1",
                 gliner_decoder_model: str = "knowledgator/gliner-decoder-base-v1.0",
                 threshold: float = 0.3):
        self.discoverer = GLiNER.from_pretrained(gliner_decoder_model)
        self.extractor = GLiNER2.from_pretrained(gliner2_model)
        self.threshold = threshold
        self.discovery_buffer = defaultdict(list)
        self.promotion_threshold = 10
    
    def extract(self, text: str) -> dict:
        """
        Extract structured information from conversation text.
        
        Returns:
        {
            "entities": ["Alice", "WaveDB", "HBTrie"],
            "topics": ["database_design", "configuration"],
            "tones": ["frustrated", "curious"],
            "decisions": ["use_hbtrie"],
            "discovered": [
                {"text": "WAL config", "label": "technical concept"},
            ]
        }
        """
        stable = self._extract_stable(text)
        discovered = self._extract_open(text)
        self._buffer_discoveries(discovered)
        
        return {**stable, "discovered": discovered}
    
    def _extract_stable(self, text: str) -> dict:
        """Extract using GLiNER2 against the stable schema."""
        schema = {
            "entities": {
                "person": "Full name of a person mentioned",
                "project": "Software project or system name",
                "technology": "Technical tool, framework, or concept",
            },
            "topics": {
                "database_design": "Database architecture or storage discussion",
                "configuration": "Config, WAL, encryption, or settings",
                "graph_database": "Graph queries, traversals, or morphisms",
                "performance": "Benchmarks, throughput, or optimization",
                "decision_making": "Choosing between options",
                "ai_architecture": "AI, neural networks, or cognitive systems",
                "api_design": "API, bindings, async, or interface design",
                "security": "Encryption, keys, or authentication",
            },
            "tones": {
                "frustrated": "Frustration, confusion, or annoyance",
                "excited": "Excitement, enthusiasm, or satisfaction",
                "curious": "Questions or exploration",
                "neutral": "Neutral or factual statements",
            },
            "decisions": {
                "decision": "A specific choice or commitment made",
            },
        }
        
        result = self.extractor.extract(text, schema=schema)
        
        entities = []
        for category in ["person", "project", "technology"]:
            entities.extend(result.get("entities", {}).get(category, []))
        
        topics = list(result.get("topics", {}).keys())
        tones = list(result.get("tones", {}).keys())
        decisions = result.get("decisions", {}).get("decision", [])
        
        return {
            "entities": list(set(entities)),
            "topics": topics,
            "tones": tones,
            "decisions": decisions,
        }
    
    def _extract_open(self, text: str) -> list[dict]:
        """Open discovery using GLiNER-Decoder."""
        entities = self.discoverer.predict_entities(
            text, labels=["label"], threshold=self.threshold
        )
        return [{"text": e["text"], "label": e["label"]} for e in entities]
    
    def _buffer_discoveries(self, discovered: list[dict]):
        """Buffer discovered labels for potential promotion."""
        for item in discovered:
            label = item["label"]
            self.discovery_buffer[label].append(item["text"])
    
    def get_promotion_candidates(self) -> list[str]:
        """Get labels that have crossed the promotion threshold."""
        return [
            label for label, examples in self.discovery_buffer.items()
            if len(examples) >= self.promotion_threshold
        ]
```

---

## 5. Bonsai Relation Extractor (`src/encoding/bonsai_relations.py`)

```python
"""Bonsai-based relation extraction for hippocampal memory.

Bonsai (Prism-ML Ternary-Bonsai-8B) is an OpenAI-compatible small LLM served by
the Prism fork of llama.cpp's ``llama-server``. The server runs on the RunPod
GPU pod — there is no local llama-server — so this client talks to a
configurable HTTP endpoint (``config.bonsai_endpoint``, override via
``BONSAI_ENDPOINT`` or the constructor). When the encoding pipeline runs on the
same pod as ``llama-server`` the endpoint is ``http://localhost:8080/v1``;
when run remotely it is the pod's public URL.

The connection is opened lazily (on the first ``extract`` call), so the class is
constructible offline and the module imports without the server present. HTTP
and parse failures surface verbatim per the plan's process instruction.
"""

from __future__ import annotations

import json
import re

import requests

from ..config import config


BONSAI_RELATION_PROMPT = """Extract relationships from this conversation.
Return ONLY valid JSON, no other text.

Relation types:
- explains(Person, Concept): someone explains something
- decides(Person, Decision): someone makes a decision
- expresses(Person, Tone): someone expresses an emotion
- questions(Person, Concept): someone asks about something
- suggests(Person, Concept): someone proposes an idea
- concerns(Episode, Topic): the conversation is about a topic
- involves(Episode, Entity): an entity participates in the conversation
- contradicts(Statement, Statement): one statement contradicts another
- follows_up_on(Episode, Episode): this conversation continues from another

Conversation:
{text}

Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""


# Matches a ```json ... ``` (or bare ```) fenced block. The model is told to
# return ONLY JSON, but small models sometimes wrap output in fences anyway.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class BonsaiRelationExtractor:
    """Extracts structured relations from conversation text.

    Talks to an OpenAI-compatible ``llama-server`` (Bonsai) over HTTP. The
    server is not assumed to be local — the endpoint is configurable and
    defaults to ``config.bonsai_endpoint`` (``BONSAI_ENDPOINT`` env var).
    """

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or config.bonsai_model
        self.endpoint = (endpoint or config.bonsai_endpoint).rstrip("/")
        self.temperature = temperature if temperature is not None else config.bonsai_temperature
        self.timeout = timeout

    def extract(self, text: str) -> list[dict]:
        """Extract relations as (subject, predicate, object) triples.

        Raises ``RuntimeError`` with the exact server response if the request
        fails or the model returns non-JSON, so the caller can log the raw
        output rather than silently dropping relations.
        """
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": BONSAI_RELATION_PROMPT.format(text=text)}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Bonsai request to {url} failed: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Bonsai endpoint {url} returned HTTP {resp.status_code}: {resp.text}"
            )

        try:
            outer = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bonsai returned non-JSON body: {resp.text}") from e

        try:
            content = outer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Bonsai response missing choices[0].message.content: {outer}") from e

        return self._parse_relations(content)

    @staticmethod
    def _parse_relations(content: str) -> list[dict]:
        """Parse the model's JSON content into a list of relation dicts.

        Strips accidental ``` fences and, failing that, falls back to the
        outermost ``{...}`` span. Raises with the raw content if no JSON object
        can be recovered — that is a real extraction failure, not an empty
        result, and the caller needs the raw text to debug it.
        """
        body = content.strip()
        fence = _FENCE_RE.match(body)
        if fence:
            body = fence.group(1).strip()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            start, end = body.find("{"), body.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(body[start : end + 1])
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from e
            else:
                raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from None

        relations = data.get("relations", []) if isinstance(data, dict) else []
        if isinstance(data, list):
            relations = data

        return [
            r for r in relations
            if isinstance(r, dict) and {"subject", "predicate", "object"} <= r.keys()
        ]
```

---

## 6. Encoder Orchestrator (`src/encoding/encoder.py`)

```python
"""Orchestrates the full encoding pipeline."""

from datetime import datetime
from ..memory.store import HippocampalStore
from ..memory.episode import Episode
from .gliner_extractor import GLiNERExtractor
from .bonsai_relations import BonsaiRelationExtractor


class HippocampalEncoder:
    """Orchestrates extraction and storage of conversation episodes."""
    
    def __init__(self, store: HippocampalStore,
                 gliner2_model: str = "fastino/gliner2-base-v1",
                 gliner_decoder_model: str = "knowledgator/gliner-decoder-base-v1.0",
                 bonsai_model: str = "gpt-4o-mini"):
        self.store = store
        self.gliner = GLiNERExtractor(
            gliner2_model=gliner2_model,
            gliner_decoder_model=gliner_decoder_model,
        )
        self.bonsai = BonsaiRelationExtractor(model=bonsai_model)
        self.episode_counter = 0
        self.last_episode_id = None
    
    def encode_turn(self, user_message: str, assistant_response: str) -> Episode:
        """Encode a single conversation turn."""
        self.episode_counter += 1
        episode_id = f"ep_{self.episode_counter:04d}"
        full_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # 1. Extract entities, topics, tones, decisions
        extracted = self.gliner.extract(full_text)
        
        # 2. Extract relations
        relations = self.bonsai.extract(full_text)
        
        # 3. Create episode
        episode = Episode.from_extraction(
            episode_id=episode_id,
            user_message=user_message,
            assistant_response=assistant_response,
            extracted=extracted,
            relations=relations,
            follows=self.last_episode_id,
        )
        
        # 4. Store
        self.store.encode_episode(episode)
        self.last_episode_id = episode_id
        
        return episode
    
    def encode_conversation(self, turns: list[tuple[str, str]]) -> list[Episode]:
        """Encode a full conversation (list of (user, assistant) turns)."""
        episodes = []
        for user_msg, assistant_msg in turns:
            ep = self.encode_turn(user_msg, assistant_msg)
            episodes.append(ep)
        return episodes
```

---

## 7. Sample Conversations (`data/sample_conversations.jsonl`)

Create 20 hand-crafted conversations that cover diverse scenarios. Each line is a JSON object with a conversation and expected extraction. These serve as both development fixtures and test cases.

```jsonl
{"id": "conv_001", "turns": [["I'm thinking about the storage layer for our new database project.", "You should look at HBTrie — it gives you hierarchical key paths that map naturally to how data is organized."], ["That's interesting. How does it compare to a regular B+tree?", "An HBTrie is a B+tree where each level is itself a B+tree. A path like users/alice/name traverses three levels of B+trees, one per path component."]], "expected_entities": ["HBTrie"], "expected_topics": ["database_design"], "expected_tones": ["curious"], "expected_decisions": []}

{"id": "conv_002", "turns": [["The WAL config is confusing. Why are there three modes?", "IMMEDIATE does fsync after every write — safest but slowest at ~1K ops/sec. DEBOUNCED batches fsync every 250ms — the sweet spot at ~300K ops/sec. ASYNC buffers to kernel — fastest at ~400K ops/sec but you can lose data on power failure."], ["Okay, I'll go with DEBOUNCED then.", "Good choice. That gives you high durability with excellent throughput."]], "expected_entities": [], "expected_topics": ["configuration"], "expected_tones": ["frustrated", "curious"], "expected_decisions": ["use_debounced"]}

{"id": "conv_003", "turns": [["Bob thinks we should use Postgres instead of WaveDB.", "WaveDB gives us hierarchical paths natively. Postgres would require an ORM layer to get the same structure. What's Bob's specific concern?"], ["He says Postgres has better tooling and more people know it.", "That's fair, but we decided on WaveDB for the HBTrie architecture. The tooling gap is closing — we have Python, Node.js, and Dart bindings now."]], "expected_entities": ["Bob", "WaveDB", "Postgres"], "expected_topics": ["database_design", "decision_making"], "expected_tones": ["curious"], "expected_decisions": ["use_wavedb_over_postgres"]}

{"id": "conv_004", "turns": [["The graph layer is working! I just ran the benchmarks.", "What are the numbers?"], ["2.6 million reads per second. That's incredible!", "That's better than we projected. The MVCC fast-path and sharded LRU cache are paying off. This changes what we can build on top of it."]], "expected_entities": ["WaveDB"], "expected_topics": ["graph_database", "performance"], "expected_tones": ["excited"], "expected_decisions": []}

{"id": "conv_005", "turns": [["We need Python bindings with async support.", "The C work pool can handle that. We can expose promise-based async operations that drive the worker pool and marshal results back to the asyncio loop."], ["Can we get batched operations too? Individual async puts are slow.", "Yes — put_many forwards to a single C batch call. It's 15-25x faster than individual puts. We're seeing 299K ops/sec for batched puts."]], "expected_entities": ["Python", "WaveDB"], "expected_topics": ["api_design", "configuration"], "expected_tones": ["curious"], "expected_decisions": []}

{"id": "conv_006", "turns": [["The encryption API is too complex. Why do I need to manage keys manually?", "We could add a key derivation function. That way you provide a passphrase and we derive the AES-256 key from it using PBKDF2."], ["Yes please, that would help a lot.", "I'll add it to the roadmap. For now, the symmetric mode with a 32-byte key is the simplest path."]], "expected_entities": ["WaveDB"], "expected_topics": ["configuration", "security"], "expected_tones": ["frustrated"], "expected_decisions": []}

{"id": "conv_007", "turns": [["Graph queries are slow on large datasets. We need to optimize.", "What kind of queries are slow?"], ["Intersections with multiple Has filters. The optimizer isn't reordering them by selectivity.", "We need a cost-based optimizer. Similar to what Postgres does, but for graph traversals. It should reorder Has filters by selectivity and sort Intersect children by estimated cardinality."], ["Let's add it to the roadmap.", "Done. I've created an issue for the cost-based optimizer. We can use statistics from PSO index scans to estimate cardinality."]], "expected_entities": ["Bob", "WaveDB", "Postgres"], "expected_topics": ["graph_database", "performance", "decision_making"], "expected_tones": ["curious"], "expected_decisions": ["add_cost_based_optimizer"]}

{"id": "conv_008", "turns": [["Morphisms are working! I can define reusable query fragments.", "That's perfect for the common traversal patterns. What does the syntax look like?"], ["You define a morphism with a name and a traversal, then use Follow to invoke it. It's like stored procedures but for graphs.", "So we can define 'alice_decisions' as a morphism that finds all episodes with Alice that contain decisions, and then just call Follow('alice_decisions')?"], ["Exactly. It compresses multi-step traversals into single named pathways.", "This is going to make complex queries so much cleaner."]], "expected_entities": ["Alice", "WaveDB"], "expected_topics": ["graph_database", "api_design"], "expected_tones": ["excited"], "expected_decisions": []}

{"id": "conv_009", "turns": [["What if WaveDB's graph layer IS a hippocampal index?", "That's a fascinating connection. The HBTrie would be the neocortical store — the actual content. And the graph layer would be the sparse pointers that reconstruct memories."], ["Exactly! And graph traversal is pattern completion. A partial cue triggers a traversal that reactivates the full memory.", "So the context window problem — the fixed token limit — goes away. Working memory is just the set of currently activated graph nodes. No fixed buffer."], ["This could replace RAG entirely. Not retrieval into a context window, but retrieval AS the context.", "Write this up. This is the architecture we should build."]], "expected_entities": ["Alice", "WaveDB"], "expected_topics": ["ai_architecture", "database_design"], "expected_tones": ["excited", "curious"], "expected_decisions": []}

{"id": "conv_010", "turns": [["The async put is only hitting 13K ops/sec. That's terrible.", "The bottleneck is the asyncio marshalling. Each call crosses the C/Python boundary, which adds overhead. The C work pool is fast — it's the Python side that's slow."], ["Can we batch the Python-side calls?", "That's exactly what put_many does. It forwards to a single C batch call. We're seeing 299K ops/sec for batched puts — that's a 23x improvement."], ["Why didn't I know about put_many?", "It's in the README. I should have mentioned it earlier. For throughput-sensitive workloads, always use the batched helpers."]], "expected_entities": ["Python", "WaveDB"], "expected_topics": ["performance", "configuration"], "expected_tones": ["frustrated"], "expected_decisions": []}
```

Create 10 more conversations covering: code-specific discussions, multi-entity interactions, emotional arcs, decision chains, and contradictions/reconsolidation scenarios.

---

## 8. Testing Strategy

### 8.1 Unit Tests

**`tests/test_episode.py`:**
```python
def test_episode_creation():
    """Episode can be created with minimal fields."""
    ep = Episode(id="ep_001", timestamp="2026-07-03T10:00:00", 
                 summary="Test", full_text="User: Hi\nAssistant: Hello")
    assert ep.id == "ep_001"
    assert ep.state == "current"
    assert ep.salience == 0.5

def test_episode_from_extraction():
    """Episode.from_extraction creates a valid episode from extraction results."""
    ep = Episode.from_extraction(
        episode_id="ep_001",
        user_message="Hello",
        assistant_response="Hi there!",
        extracted={"entities": ["User"], "topics": ["test"], "tones": ["neutral"], "decisions": []},
        relations=[],
    )
    assert ep.full_text == "User: Hello\nAssistant: Hi there!"
    assert "Hi there!" in ep.summary
```

**`tests/test_store.py`:**
```python
def test_encode_and_retrieve_episode(tmp_path):
    """Episode can be stored and retrieved from WaveDB."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    ep = Episode(id="ep_001", timestamp="2026-07-03T10:00:00",
                 summary="Test episode", full_text="User: Hi\nAssistant: Hello",
                 entities=["User"], topics=["test"], tones=["neutral"])
    
    store.encode_episode(ep)
    loaded = store.get_episode("ep_001")
    
    assert loaded is not None
    assert loaded.summary == "Test episode"
    assert loaded.state == "current"
    store.close()

def test_graph_triples_stored(tmp_path):
    """Encoding stores triples in the graph layer."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    ep = Episode(id="ep_001", timestamp="2026-07-03T10:00:00",
                 summary="Test", full_text="User: Hi\nAssistant: Hello",
                 entities=["Alice"], topics=["database_design"], 
                 tones=["curious"], decisions=["use_hbtrie"],
                 relations=[{"subject": "Alice", "predicate": "explains", "object": "HBTrie"}])
    
    store.encode_episode(ep)
    
    # Verify graph triples exist
    # (Use WaveDB graph query API to check)
    store.close()
```

**`tests/test_gliner_extractor.py`:**
```python
def test_extract_entities():
    """GLiNER extracts entities from conversation text."""
    extractor = GLiNERExtractor()
    text = "Alice suggested using HBTrie for the WaveDB storage layer."
    result = extractor.extract(text)
    
    assert "Alice" in result["entities"]
    assert "WaveDB" in result["entities"] or "HBTrie" in result["entities"]

def test_extract_tones():
    """GLiNER extracts emotional tones."""
    extractor = GLiNERExtractor()
    text = "I'm so frustrated with this configuration. It's incredibly confusing."
    result = extractor.extract(text)
    
    assert "frustrated" in result["tones"]

def test_extract_topics():
    """GLiNER extracts topics."""
    extractor = GLiNERExtractor()
    text = "The WAL sync modes need better documentation. DEBOUNCED vs ASYNC is unclear."
    result = extractor.extract(text)
    
    assert "configuration" in result["topics"]

def test_extract_decisions():
    """GLiNER extracts decisions."""
    extractor = GLiNERExtractor()
    text = "I've decided to go with DEBOUNCED for the WAL sync mode."
    result = extractor.extract(text)
    
    assert len(result["decisions"]) > 0

def test_open_discovery():
    """GLiNER-Decoder discovers entity types not in the schema."""
    extractor = GLiNERExtractor()
    text = "The Kubernetes deployment uses Helm charts for the staging environment."
    result = extractor.extract(text)
    
    # Should discover labels like "deployment tool", "infrastructure", etc.
    assert len(result["discovered"]) > 0
```

**`tests/test_bonsai_relations.py`:**
```python
def test_extract_explains_relation():
    """Bonsai extracts 'explains' relations."""
    bonsai = BonsaiRelationExtractor()
    text = "User: What is DEBOUNCED? Assistant: DEBOUNCED batches fsync calls every 250ms."
    relations = bonsai.extract(text)
    
    explains_rels = [r for r in relations if r["predicate"] == "explains"]
    assert len(explains_rels) > 0

def test_extract_decides_relation():
    """Bonsai extracts 'decides' relations."""
    bonsai = BonsaiRelationExtractor()
    text = "User: I'll go with DEBOUNCED then. Assistant: Good choice."
    relations = bonsai.extract(text)
    
    decides_rels = [r for r in relations if r["predicate"] == "decides"]
    assert len(decides_rels) > 0
```

### 8.2 Integration Tests

**`tests/test_encoder.py`:**
```python
def test_encode_single_turn(tmp_path):
    """Encoder processes a single conversation turn end-to-end."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    ep = encoder.encode_turn(
        "I'm frustrated with the WAL config. Why are there three modes?",
        "IMMEDIATE is safest but slowest, DEBOUNCED is the sweet spot, ASYNC is fastest."
    )
    
    assert ep.id == "ep_0001"
    assert len(ep.entities) > 0
    assert "frustrated" in ep.tones or "curious" in ep.tones
    assert "configuration" in ep.topics
    
    # Verify stored
    loaded = store.get_episode("ep_0001")
    assert loaded is not None
    assert loaded.summary == ep.summary
    
    store.close()

def test_encode_conversation_chain(tmp_path):
    """Multiple turns form a follows chain."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    turns = [
        ("What's HBTrie?", "HBTrie is a hierarchical B+tree..."),
        ("How does it compare to B+tree?", "Each level is itself a B+tree..."),
        ("I'll use it then.", "Great choice."),
    ]
    
    episodes = encoder.encode_conversation(turns)
    
    assert len(episodes) == 3
    assert episodes[0].follows is None
    assert episodes[1].follows == "ep_0001"
    assert episodes[2].follows == "ep_0002"
    
    store.close()

def test_encode_sample_conversations(tmp_path):
    """All 20 sample conversations encode without errors."""
    import json
    
    store = HippocampalStore(str(tmp_path / "test_db"))
    encoder = HippocampalEncoder(store)
    
    with open("data/sample_conversations.jsonl") as f:
        for line in f:
            conv = json.loads(line)
            episodes = encoder.encode_conversation(conv["turns"])
            assert len(episodes) == len(conv["turns"])
    
    store.close()
```

### 8.3 Extraction Quality Tests

**`tests/test_extraction_quality.py`:**
```python
def test_extraction_matches_expected():
    """Extraction quality against hand-labeled sample conversations."""
    import json
    
    extractor = GLiNERExtractor()
    bonsai = BonsaiRelationExtractor()
    
    with open("data/sample_conversations.jsonl") as f:
        for line in f:
            conv = json.loads(line)
            
            # Combine all turns into one text for extraction
            full_text = " ".join(
                f"User: {u} Assistant: {a}" for u, a in conv["turns"]
            )
            
            result = extractor.extract(full_text)
            
            # Check entity recall
            expected_entities = set(conv.get("expected_entities", []))
            extracted_entities = set(result["entities"])
            entity_recall = len(expected_entities & extracted_entities) / len(expected_entities) if expected_entities else 1.0
            
            # Check topic recall
            expected_topics = set(conv.get("expected_topics", []))
            extracted_topics = set(result["topics"])
            topic_recall = len(expected_topics & extracted_topics) / len(expected_topics) if expected_topics else 1.0
            
            # Check tone recall
            expected_tones = set(conv.get("expected_tones", []))
            extracted_tones = set(result["tones"])
            tone_recall = len(expected_tones & extracted_tones) / len(expected_tones) if expected_tones else 1.0
            
            # Log results (don't assert — this is a quality measurement, not a pass/fail)
            print(f"{conv['id']}: entity_recall={entity_recall:.2f}, "
                  f"topic_recall={topic_recall:.2f}, tone_recall={tone_recall:.2f}")
```

---

## 9. Corpus Processing Script (`scripts/process_corpus.py`)

```python
"""Process a conversation corpus through the encoding pipeline.

Usage:
    python scripts/process_corpus.py --input data/corpus.jsonl --db ./data/memory_db
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import HippocampalStore
from src.encoding.encoder import HippocampalEncoder


def main():
    parser = argparse.ArgumentParser(description="Process a conversation corpus")
    parser.add_argument("--input", required=True, help="JSONL file with conversations")
    parser.add_argument("--db", default="./data/memory_db", help="WaveDB path")
    parser.add_argument("--limit", type=int, help="Max conversations to process")
    args = parser.parse_args()
    
    store = HippocampalStore(args.db)
    encoder = HippocampalEncoder(store)
    
    processed = 0
    episodes = 0
    
    with open(args.input) as f:
        for line in f:
            if args.limit and processed >= args.limit:
                break
            
            conv = json.loads(line)
            turns = conv.get("turns", [])
            
            if not turns:
                continue
            
            eps = encoder.encode_conversation(turns)
            processed += 1
            episodes += len(eps)
            
            if processed % 100 == 0:
                print(f"Processed {processed} conversations ({episodes} episodes)")
    
    print(f"Done. {processed} conversations, {episodes} episodes stored.")
    store.close()


if __name__ == "__main__":
    main()
```

---

## 10. Datasets to Download

### Pre-existing (download before starting)

Dataset	URL	Size	Use
**DialogSum**	`https://huggingface.co/datasets/knkarthick/dialogsum`	13,460 dialogues	Integration testing, SSM pre-training (Phase 3)
**SAMSum**	`https://huggingface.co/datasets/samsum`	16,369 dialogues	Integration testing, SSM pre-training (Phase 3)
**Memory-Traces**	`https://huggingface.co/datasets/Cossale/memory-traces`	27,449 conversations	JEPA training (Phase 3), salience labels
**kniv-corpus-en**	`https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en`	45K examples	GLiNER2 fine-tuning (Phase 2)

### To generate (Oracle-assisted)

Dataset	How to Generate	Use
**Bonsai query planning pairs**	Oracle prompt (see below)	Bonsai fine-tuning (Phase 2)
**Bonsai relation extraction pairs**	Oracle prompt (see below)	Bonsai fine-tuning (Phase 2)
**GNN training subgraphs**	Oracle labeling of corpus graphs	GNN training (Phase 2)

---

## 11. Prompts for Generating Training Data

### 11.1 Bonsai Query Planning Pairs

```
You are generating training data for a query planner that converts natural 
language questions into structured memory queries. Given a conversation and 
a hypothetical question a user might later ask about it, output the query 
parameters that would retrieve the relevant memories.

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
- limit: max episodes (default 5)

RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union"
- "What did Alice and Bob disagree about?" → entities=["Alice", "Bob"], entity_mode="intersection"
- "What happened after morphisms?" → temporal_after="morphism"

Return ONLY valid JSON:
{{"entities": [], "topics": [], "tones": [], "entity_mode": "union", "temporal_after": null, "limit": 5}}
```

### 11.2 Bonsai Relation Extraction Pairs

```plaintext
Extract relationships from this conversation. Return ONLY valid JSON.

Relation types:
- explains(Person, Concept): someone explains something
- decides(Person, Decision): someone makes a decision
- expresses(Person, Tone): someone expresses an emotion
- questions(Person, Concept): someone asks about something
- suggests(Person, Concept): someone proposes an idea
- concerns(Episode, Topic): the conversation is about a topic
- involves(Episode, Entity): an entity participates
- contradicts(Statement, Statement): one statement contradicts another

CONVERSATION:
{conversation_text}

Return: {{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
```

---

## 12. Checkpoint Criteria

Phase 1a is complete when:

- [ ] `HippocampalStore` can encode and retrieve episodes from WaveDB
- [ ] Episode content + graph index written in ONE atomic `WaveDB.batch_sync` via `GraphLayer.expand_triple` (no partial writes)
- [ ] `GLiNERExtractor` extracts entities, topics, tones, and decisions from conversation text
- [ ] `GLiNERExtractor` discovers entity types not in the schema (open discovery)
- [ ] `BonsaiRelationExtractor` extracts structured relations from conversation text
- [ ] `HippocampalEncoder` orchestrates the full pipeline: extract → create episode → store
- [ ] All 20 sample conversations encode without errors
- [ ] Extraction quality: entity recall > 70%, topic recall > 70%, tone recall > 70% on sample conversations
- [ ] `follows` chains are correctly maintained across multi-turn conversations
- [ ] All unit tests pass
- [ ] Integration test: encode 100 conversations from DialogSum without errors
- [ ] `scripts/process_corpus.py` can process a JSONL file of conversations
- [ ] Developmental stage for extraction components set to INFANT
- [ ] Metrics tracked: entity F1, topic F1, tone F1, relation accuracy
- [ ] Transition criteria defined: F1 > 0.85 → CHILD stage

---

## 13. Implementation Order

1. **Project setup** — `pyproject.toml`, directory structure, `config.py`
2. **Data models** — `episode.py`, `ontology.py`
3. **WaveDB store** — `store.py` with encode/decode
4. **GLiNER extractor** — `gliner_extractor.py` with both models
5. **Bonsai relations** — `bonsai_relations.py`
6. **Encoder orchestrator** — `encoder.py`
7. **Sample conversations** — `sample_conversations.jsonl` (20 conversations)
8. **Tests** — unit tests for each component, integration test for encoder
9. **Corpus processing script** — `process_corpus.py`
10. **Quality measurement** — run extraction quality tests, iterate on prompts until thresholds met

---

Begin with step 1. Report after each step. If any model download fails or API call errors, report the exact error. If extraction quality is below threshold, we'll iterate on the GLiNER schema and Bonsai prompt together.