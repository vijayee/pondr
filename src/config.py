"""Central configuration for the hippocampal memory system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Runtime configuration.

    Mutable values can be overridden via environment variables so the same code
    runs locally (light editing) and on a RunPod GPU pod (heavy inference).
    """

    # ── WaveDB ──
    db_path: str = os.getenv("HIPPOCAMPAL_DB_PATH", "./data/memory_db")
    lru_memory_mb: int = 100
    wal_sync_mode: str = "debounced"

    # ── GLiNER ──
    # GLiNER2: stable extraction against the evolved schema (Fastino). Both
    # GLiNER models are transformer-based and GPU-intended (the extractor
    # module docstring: "both models are heavy (GPU)"). They previously loaded
    # with no device -> CPU, the ~20s/conv ingestion bottleneck (CPU pegged
    # 95%, GPU ~0 on the 2026-07-06 pod). gliner_device="auto" now moves them
    # to CUDA when available, with an OOM-safe per-model CPU fallback (the 8B
    # Bonsai server can fill the 5080's 16GB VRAM, leaving no room).
    gliner2_model: str = "fastino/gliner2-base-v1"
    # GLiNER-Decoder: open discovery, invents labels freely (Knowledgator).
    gliner_decoder_model: str = "knowledgator/gliner-decoder-base-v1.0"
    # Device for the GLiNER models: "cpu" (default -- keeps every existing
    # caller byte-identical), "auto" (CUDA if available, else CPU; the SERVING
    # entrypoint's path), or an explicit "cuda"/"cuda:0". The CUDA move is
    # OOM-safe -> per-model CPU fallback.
    gliner_device: str = "cpu"
    # Log per-stage GLiNER extraction timing ([gliner-timing] stable/open/total)
    # to stderr. Off by default; enable to measure the CPU-vs-CUDA bottleneck.
    gliner_timing: bool = False
    # Extraction threshold. The matching spans sit far below the model's
    # "natural" 0.3 on CPU: a sweep over the 20 sample conversations showed
    # topic recall 0.09 at 0.3 / 0.22 at 0.05 / 0.27 at 0.03, with ZERO
    # garbage spans leaking through down to 0.01 (the count_lstm_v2 count-pred
    # is well-behaved here, so lowering the cutoff admits more true positives
    # without flooding noise). 0.05 clears both quality-test floors
    # (mean_topic > 0.2 and non-empty decisions) on CPU. On GPU the matching
    # spans score ABOVE 0.3, so 0.05 admits strictly more true positives there
    # too -- one threshold works for both backends, no `if cuda` branching.
    # See docs/Phase 1a.md "Extraction threshold -- revisit".
    extraction_threshold: float = 0.05

    # ── Bonsai ──
    # Prism-ML Ternary Bonsai (1.58-bit / Q2_0 ternary, Qwen3-based). The Q2_0
    # ternary kernels live only in the Prism fork of llama.cpp, so Bonsai is
    # served on the RunPod GPU pod by that fork's `llama-server` (OpenAI-
    # compatible). The Python side talks to its HTTP endpoint — no llama-cpp-python
    # build needed. See infra/runpod for the serving image. BONSAI_MODEL can be
    # swapped to the 4B/1.7B GGUFs for a speed/quality tradeoff.
    bonsai_model: str = os.getenv("BONSAI_MODEL", "prism-ml/Ternary-Bonsai-8B-gguf")
    bonsai_endpoint: str = os.getenv("BONSAI_ENDPOINT", "http://localhost:8080/v1")
    bonsai_temperature: float = 0.1
    bonsai_n_ctx: int = 4096

    # ── Encoding ──
    episode_salience_default: float = 0.5
    discovery_buffer_threshold: int = 10  # promote a discovered label after N occurrences

    # ── Phase 1b: Retrieval ──
    # Graph-traversal + query-planner defaults. Zero OpenAI spend: the LLM pieces
    # (query planner, Mode A generation) use the local Bonsai llama-server at
    # bonsai_endpoint; vector-search embeddings use a local sentence-transformers
    # model on the pod, not the OpenAI embeddings API.
    default_retrieval_limit: int = 5
    max_context_tokens: int = 4000
    # Local embedding model (sentence-transformers), 384-dim. Swappable to a
    # larger BGE/e5 model for quality at the cost of pod memory.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_index_type: str = "faiss"  # faiss | usearch
    # Native in-DB vector index (WaveDB VectorLayer, wavedb>=0.2.2). When True
    # AND the installed wavedb exposes VectorLayer, HippocampalStore opens a
    # FLAT/COSINE index over episode summary embeddings and maintains it live
    # (insert on encode, delete on forget/supersede), closing the FAISS-sidecar
    # "not updated live / can't delete" caveats. When False (or old wavedb) the
    # retriever falls back to the FAISS VectorSearch path. dim must match the
    # embedding model above (bge-small = 384).
    vector_index_enabled: bool = True
    embedding_dim: int = 384

    # ── Phase 1b: Mode A generation ──
    # Default to the local Bonsai server's model (same GGUF as bonsai_model) so
    # generation costs nothing; override GENERATION_MODEL for a different backend.
    generation_model: str = os.getenv("GENERATION_MODEL", "prism-ml/Ternary-Bonsai-8B-gguf")
    generation_temperature: float = 0.7

    # ── Phase 1d: Oracle (training-data labeling) ──
    # The Oracle labels training data (GNN subgraphs, Bonsai query/relation pairs,
    # JEPA routing, gate decisions, code-aware synthetics). It is DeepSeek served
    # by the user's LOCAL Ollama instance — OpenAI-compatible at /v1 — NOT OpenAI.
    # The ``:cloud`` tag routes inference to ollama.com (Ollama credits, not local
    # compute); token counts are tracked but ``$`` cost is left at 0 (set
    # ``OracleConfig.cost_per_1k_*`` to meter credits). Talks to the same
    # OpenAI-compatible /chat/completions API the Bonsai client uses, so the
    # OracleClient mirrors src/encoding/bonsai_relations.py (requests, no SDK dep).
    oracle_model: str = os.getenv("ORACLE_MODEL", "deepseek-v4-pro:cloud")
    oracle_endpoint: str = os.getenv("ORACLE_ENDPOINT", "http://localhost:11434/v1")
    oracle_temperature: float = 0.1
    oracle_max_tokens: int = 32768  # DeepSeek-v4-pro is a REASONING model: the `reasoning` CoT shares the max_tokens budget with `content`, so a small cap truncates content (even to a single inner fragment). DeepSeek supports ~1M context; 32768 output leaves comfortable headroom. Raise via --oracle-max-tokens if a task still truncates.
    oracle_max_retries: int = 3
    oracle_retry_delay: float = 2.0       # base seconds between retries (exp backoff)
    oracle_batch_delay: float = 0.0       # throttle between calls (0 = no throttle)
    oracle_timeout: float = 120.0         # :cloud routing can be slow

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
    # Phase 3b: master gate. When True the default-query filters exclude
    # deprecated/superseded episodes and deprecated/superseded/archived edges,
    # and the consolidation dream pass applies utility decay. When False the
    # system behaves as if forgetting were never deployed (everything current).
    # Default True: the filters are a no-op until something is actually
    # deprecated, so a fresh corpus is unaffected.
    forgetting_enabled: bool = True

    # ── Phase 3c: entity-state assertions + citation ──
    # ``assertion_extraction_enabled`` gates the production writer of
    # ``(E:entity, state, value)`` edges -- the deterministic normalizer
    # (src/encoding/assertion_extractor.py) + Bonsai ``has_state`` relations,
    # which unblock the dormant A2 ``contradictory_state`` anomaly resolver.
    # Default True: the deterministic normalizer is inert on a corpus with no
    # explicit state claims (zero ``state`` edges -> detector never fires -> no
    # tombstones -> byte-identical to today), and the Bonsai half degrades to
    # empty when no server is wired, so a cold start is unchanged.
    assertion_extraction_enabled: bool = True
    # ``citation_resolution_enabled`` gates resolving ``Document.citations``
    # (literal strings) to Document nodes via title/URL match
    # (``find_document_by_title_or_url``) + emitting email
    # ``in_reply_to``/``references`` provenance edges. Default True: unresolved
    # literals are kept as-is (byte-identical) and email provenance is only
    # written when the parser supplied reply maps, so a corpus with no
    # citations / no email is unchanged.
    citation_resolution_enabled: bool = True
    # 10-pass isolated per-class relation extraction (Phase 3c async-distill).
    # False (default) = the V1 single-pass BONSAI_RELATION_PROMPT (byte-identical
    # to pre-async). True = one focused single-predicate pass per class, merged
    # -- lifts strict has_state catch 0 -> 11/13 zero-shot (every class emits,
    # no salience race for the "at most 6" slots) at the cost of 10 HTTP
    # round-trips (~22.8 s/doc, untenable on the sync path -> only enable
    # behind async_distill_enabled so it runs on the background worker). See
    # docs/Phase 3c.md Sec 7 + memory pondr-bonsai-zeroshot-eval-finetune-warranted.
    bonsai_isolation_extraction: bool = False
    # Async episode distillation (Phase 3c). False (default) = the synchronous
    # _persist_exchange (encode blocks the response, byte-identical to today).
    # True = the response returns immediately; the episode is written as a stub
    # (content + vector index) on the main thread and a single-worker background
    # FIFO fills the graph edges (extraction + 10-pass Bonsai) in the gaps
    # between turns (foreground-priority yielding). Live-dogfood passed
    # 2026-07-16 against the real 8B: response 7.8 s << 22.8 s fill, stub
    # content-retrievable immediately, has_state assertion edges
    # (E:entity, state, value) fire after the fill (the Bonsai assertion arm
    # goes no-op -> live). Default off (opt-in via serve_ponder --async-distill);
    # compose with bonsai_isolation_extraction (async hides the 22.8 s, isolation
    # provides the 11/13 has_state quality).
    async_distill_enabled: bool = False

    # ── Phase 2c+: feedback-driven salience ──
    # When True, after a synthesizing turn the consumer (the external LLM, or
    # Ponder's own Bonsai self-chat) reports per-unit usefulness via the
    # ``record_feedback`` tool, and a per-unit boost (default 1.0, a no-op)
    # reweights retrieval scoring on the next query (the "learning over time"
    # differentiator vs stateless RAG). Default True: a fresh corpus reads
    # boost=1.0 everywhere (no-op), so the loop is inert until a unit is judged.
    feedback_salience_enabled: bool = True
    # Max consecutive results of one kind (section/document/episode) in the
    # top-K after kind-aware diversity rerank. 0 disables the cap (pure score
    # sort -- the pre-2c+ behavior). Independent of feedback_salience_enabled.
    kind_diversity_cap: int = 3

    # ── Phase 2c+: self-chat full agent loop ──
    # When True, the Bonsai self-chat synthesize path runs a multi-turn tool
    # loop (``run_tool_loop``): the model may call ``expand`` / ``search_memory``
    # mid-generation to ground its answer beyond the pre-retrieved context, and
    # ``record_feedback`` for salience (gated by ``feedback_salience_enabled``).
    # When False, ``_synthesize`` is byte-identical to the one-shot path (one
    # ``mode_a._complete`` + ``_dispatch_feedback``) -- the A/B regression
    # guard. Default True: a live probe confirmed the 8B Bonsai emits native,
    # parseable ``tool_calls`` (``finish_reason:"tool_calls"``), so the loop is
    # the primary path; the structured-JSON fallback stays as a safety net.
    self_chat_tool_loop_enabled: bool = True
    # Loop bound: each turn is one Bonsai call + N tool dispatches (~2.6s/call
    # warm on the 5080). 4 leaves headroom for an expand + a search + a clean
    # answer turn inside the 4K context.
    self_chat_tool_loop_max_iters: int = 4

    # ── Paths ──
    data_dir: Path = Path("./data")
    sample_conversations: Path = Path("./data/sample_conversations.jsonl")
    corpora_dir: Path = Path("./data/corpora")

    # ── Document ingestion (task #17) ──
    # Cold, content-addressed blob store for document section bodies -- a
    # SECOND WaveDB instance separate from the hot memory store (db_path) so
    # large chunk bodies never flush the memory store's 100MB LRU. The store
    # defaults to a sibling of db_path when these are unset; these make the
    # paths explicit + overridable.
    document_db_path: str = "./data/document_db"
    document_store_lru_mb: int = 16
    # Ingestion knobs (structure-based chunking leaf sizing + blob hashing).
    ingestion: "IngestionConfig" = field(default_factory=lambda: IngestionConfig())


# ── Phase 2c: Working Memory & Presentation ──
# Config is dataclass-based (not YAML), matching the rest of the codebase. These
# knobs are runtime-only — Phase 2c adds NO training cost (the backbone is reused
# from 2a; Working Memory + SSM Chunker are runtime-only; the Presentation Gate is
# heuristic, learned gate deferred). See docs/Phase 2c.md §9.


@dataclass
class WMConfig:
    """Working Memory recurrent-state knobs (runtime, no training)."""
    decay_alpha: float = 1.0   # post-step state forget factor; 1.0 = rely on SSM dynamics
    lora_rank: int = 8        # matches INSTANCE_CONFIGS["working_memory"]


@dataclass
class ChunkerConfig:
    """SSM Chunker primary/compressed split (runtime)."""
    max_primary_tokens: int = 4096   # len(text)//4 estimate, summed over primary
    max_primary_chunks: int = 5      # cap on primary (full-text) episodes


@dataclass
class PGConfig:
    """Presentation Gate heuristic thresholds (axis a: chunking strategy)."""
    direct_max_episodes: int = 3        # ≤ this + specific query → direct (no chunking)
    chunked_min_episodes: int = 5       # above direct_max, below summary_only → chunked
    summary_only_min_episodes: int = 20  # ≥ this OR summarization verb → summary_only
    expand_threshold: float = 0.5        # confidence threshold for auto-EXPAND (Phase 4a reads it)


@dataclass
class PCConfig:
    """Prompt compression for query planning (Task 5)."""
    short_prompt_threshold: int = 500   # prompts ≤ this pass through byte-identical
    bonsai_max_input: int = 2000         # hard cap (chars) for the planner prompt


@dataclass
class SessionConfig:
    """Working-Memory session persistence (file-first; WaveDB-backed is optional)."""
    state_dir: str = "data/sessions/"
    auto_save_interval: int = 300   # seconds; the save TRIGGER policy is still open (memory)


@dataclass
class Phase2cConfig:
    """Top-level Phase 2c config. All runtime; no training cost."""
    working_memory: WMConfig = field(default_factory=WMConfig)
    ssm_chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    presentation_gate: PGConfig = field(default_factory=PGConfig)
    prompt_compression: PCConfig = field(default_factory=PCConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    replay_capacity: int = 1000   # shared by chunking + end-state outcome/override buffers


# ── Phase 3a: GNN Consolidator ──
# Config is dataclass-based, matching the rest of the codebase. The GNN is the
# one Phase-3a component with a training cost (Task 4, pod/RTX 4090); the rest
# is runtime + the consolidation loop. See docs/Phase 3a.md §10.


@dataclass
class GNNConfig:
    """GAT backbone + head architecture knobs (Task 2)."""
    hidden_dim: int = 128        # GAT hidden; sized to the memory graph (~1e4-1e5 nodes)
    num_heads: int = 4           # GAT attention heads per layer
    num_layers: int = 3          # GAT layers
    dropout: float = 0.1
    # Node-feature dim. Episodes use the 384-dim embedder vector; other node
    # kinds use a type-onehot ∪ optional embedding projected into this dim.
    node_feature_dim: int = 384
    # Predicate-vocabulary size for edge_attr onehot (snake_case graph predicates
    # + the open Bonsai-relation bucket hashed into the last slot).
    predicate_vocab_size: int = 32
    ogb_pretrain: bool = True    # §1.3 decision 1: OGB-pretrain-then-transfer
    ogb_dataset: str = "ogbn-arxiv"


@dataclass
class ConsolidationConfig:
    """Nightly dream-state loop thresholds + candidate-sampling knobs (Task 6).

    All fields are exposed as CLI flags in ``scripts/run_consolidation.py`` so a
    consolidation run is tunable for efficacy and measurable against alternatives.
    The threshold knobs (accept/bonsai/prune) are also sweepable from a SINGLE run
    via the ``score_distributions`` histograms in the report (no re-run needed);
    the strategy/budget knobs change WHICH pairs get scored, so comparing those
    requires a re-run per value.
    """
    accept_threshold: float = 0.85       # auto-accept predicted edges above this
    bonsai_propose_threshold: float = 0.60  # propose to Bonsai between this and accept
    prune_salience_below: float = 0.15   # archive edges where BOTH endpoints below this
    dry_run_default: bool = True         # --dry-run is the default; --apply mutates
    wm_prioritized: bool = True          # score what's "in awareness" first
    # Ontology candidate sampling. The old code reused a link-pred cap of 16 here,
    # but the real entity x class space is ~1512 x 377, so that cap sampled ~16
    # pairs and missed every true class (0 proposals despite the head scoring
    # true classes 0.93-0.98). "all" scores every pair (the honest, complete
    # option -- chunked to bound memory); "topk" prefilters by embedding dot
    # product then scores the top-k classes per entity (fast); "rotation" is the
    # legacy deterministic slice kept as a comparison baseline.
    ontology_strategy: str = "all"      # "all" | "topk" | "rotation"
    ontology_topk: int = 10              # classes per entity for "topk"
    ontology_candidate_budget: int = 16  # cap for "rotation" (legacy behavior)
    # Link-prediction candidate budget (splits the old shared max_candidates cap;
    # link-pred is O(N^2) over the subgraph, so this stays small by design).
    linkpred_candidate_budget: int = 16
    # Histogram collection: record every score >= this into 0.1-width bins so the
    # report supports a threshold sweep without re-running. 0.0 collects all
    # (bins are just 10 int counts per category -- tiny).
    score_collect_bar: float = 0.0
    # Anomaly head subgraph bound (the giant-subgraph data-quality fix). The
    # other 4 heads stay radius-3 uncapped (their Oracle labels are cached and
    # cache-keyed by the node set; bounding them = cache miss = paid re-label).
    # Anomaly is Oracle-free (inject->detect), so its subgraph is bounded in
    # isolation: radius-2 keeps the sibling-episode comparison set (ep -has_entity-
    # E: -has_entity- ep_sibling is 2 hops) and fanout_cap stops the entity hub from
    # flooding to thousands of unrelated episodes. radius-3 + None cap reproduces
    # the prior giant behavior (degenerate guard in consolidate.py).
    anomaly_subgraph_radius: int = 2
    anomaly_fanout_cap: Optional[int] = 64
    # Phase 3b forgetting dream-pass knobs. The decay math itself lives in
    # ``src/memory/forgetting.py`` (canonical constants); these are the
    # consolidation-side thresholds. ``utility_prune_below`` is the soft-archive
    # cutoff: a current edge whose composed ``utility_score`` drops below this is
    # set to ``state='archived'`` (excluded from default queries, NOT deleted).
    utility_prune_below: float = 0.1
    # Anomaly -> reconsolidation auto-resolve cutoff (Phase 3b step 8). A
    # ``contradictory_state`` anomaly whose head score is >= this is handed to
    # the resolver (which confirms >=2 distinct entity ``state`` values in the
    # graph, finds the entity's source episodes, and supersedes the oldest by
    # the latest). Below this the flag is record-only -- the head over-fires
    # on the giant subgraph, so low-confidence contradictions are NOT auto-
    # superseded (honest: the resolver is best-effort; the data model carries
    # no value->episode provenance, so the resolver assumes the latest-asserting
    # episode is the current truth).
    anomaly_resolve_threshold: float = 0.8
    # ── Phase 3c: contradiction adjudication ──
    # A ``contradictory_state`` anomaly whose head score is >= this is handed to
    # the Bonsai ``decide_contradiction`` adjudicator (when a decider is wired
    # AND ``bonsai_decider_enabled`` AND ``forgetting_enabled``). Below this --
    # or with no decider -- the flag is record-only (still in
    # ``report["anomalies"]`` for observability, no mutation). Mirrors
    # ``anomaly_resolve_threshold``; the 3b no-decider episode-supersede path
    # stays intact (the decider loop is ADDITIVE under ``decider_active``).
    contradiction_resolve_threshold: float = 0.8
    # Ontology decay (Phase 3b step 9). A DISCOVERED class (runtime-invented
    # label promoted via Bonsai -- a deferred path) whose ``last_seen`` is older
    # than this many days is deprecated (``content/class/{c}/state =
    # "deprecated"``). Seed classes are NEVER decay-eligible (they have no
    # ``content/class/`` entry -- the seed writes only ``subClassOf`` graph
    # triples), so this is a no-op on the seed-only ontology today; the
    # mechanism ships so promotion lands into a decay-ready namespace.
    ontology_decay_days: int = 30
    # A1 deep-archive tier. A soft-archived edge (``state='archived'``) whose
    # ``archived_at`` is older than this many days is PHYSICALLY removed: the live
    # graph edge is deleted and a recoverable ``archive/edge/...`` JSON record is
    # written (reusing the 3a hard-prune format), then the orphaned sidecar +
    # consumed ``content/archived_edge/`` index entry are deleted. ``None`` (via
    # ``--deep-archive-days 0``) disables the sweep. Soft-archive (in-place,
    # excluded from default queries) always ships; this is the deep tier.
    deep_archive_days: Optional[int] = 365
    # ── Bonsai-in-consolidation (the deploy-time decider). The 8B Bonsai
    # (localhost:8080/v1) is the SUBCONSCIOUS decider for three consolidation
    # actions: (1) abstract gist generation, (2) ontology promotion (entity
    # -> class instanceOf typing + new-class creation), (3) the identity_drift
    # anomaly decision (fix/ask_user/dismiss). The Oracle/DeepSeek stays the
    # TRAINING-DATA teacher only; at deploy the decider is Bonsai. All three
    # gate on BOTH a wired ``decider`` AND ``bonsai_decider_enabled`` so
    # ``--no-bonsai`` disables even when wired, and a cold start (no decider
    # / dry-run / server down) stays record-only + byte-identical to today.
    bonsai_decider_enabled: bool = True
    # Cap on the number of source episodes fed to the gist prompt (the 8B's
    # 4096 ctx). A gist is a summary-of-summaries; 8 is plenty and bounds the
    # HTTP round-trip size.
    abstract_gist_max_episodes: int = 8
    # Min ontology-proposal confidence gated through Bonsai. 0.0 = EVERY
    # proposal above ``accept_threshold`` goes to Bonsai (Bonsai IS the
    # deploy-time decider, per the user's "all three together" scope). Raise
    # to skip Bonsai on very-high-confidence auto-accepts (one HTTP call per
    # proposal at 0.0 -- acceptable for a nightly dream pass; see
    # ``--ontology-bonsai-threshold`` for the escape hatch).
    ontology_bonsai_threshold: float = 0.0


@dataclass
class ArchiveConfig:
    """Archive subtree for pruned/abstracted content (never deleted)."""
    subtree: str = "archive/"   # e.g. archive/edge/..., archive/ep/{eid}/...


@dataclass
class LabelGenConfig:
    """Oracle label regeneration knobs (Task 3 — the run is Bonsai-gated)."""
    num_subgraphs: int = 4000
    subgraph_radius: int = 3
    neg_edge_ratio: float = 1.0  # negatives per positive for link prediction
    # Anomaly head subgraph bound (mirror of ConsolidationConfig's fields; the
    # generator and the consolidator are separate flows so the two configs each
    # carry the anomaly radius/cap). The generator writes these into
    # quality_report.json; the trainer reads them so the bounded subgraph used to
    # extract == the bounded subgraph used to train. Defaults match the
    # consolidation defaults (radius=2, cap=64).
    anomaly_subgraph_radius: int = 2
    anomaly_fanout_cap: Optional[int] = 64


@dataclass
class Phase3aConfig:
    """Top-level Phase 3a config."""
    gnn: GNNConfig = field(default_factory=GNNConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    labels: LabelGenConfig = field(default_factory=LabelGenConfig)
    checkpoint_dir: str = "data/pod_runs/phase3a/"   # gitignored


@dataclass
class IngestionConfig:
    """Document/record ingestion knobs (task #17, RAG-replacement pillar).

    Structure-based chunking leaf sizing (the chat's HierarchicalChunker gap):
    ``max_section_tokens`` sub-splits an oversized section on paragraph
    boundaries so each leaf is one-embedding-pass + retrieval-sized (default
    512 = the embedder's cap); ``min_section_tokens`` merges a too-small leaf
    into its parent (default 64). ``semantic_split_threshold`` is reserved for
    the Phase-2 embedding-based semantic-boundary splitter (needs the lazy
    embedder) and is a no-op until set. ``blob_hash_algo`` is the
    content-addressed key for the cold store (``sha256[:16]``).
    """

    max_section_tokens: int = 512
    min_section_tokens: int = 64
    semantic_split_threshold: Optional[float] = None
    blob_hash_algo: str = "sha256[:16]"


config = Config()