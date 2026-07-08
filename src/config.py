"""Central configuration for the hippocampal memory system."""

import os
from dataclasses import dataclass, field
from pathlib import Path


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
    # GLiNER2: stable extraction against the evolved schema (Fastino, CPU).
    gliner2_model: str = "fastino/gliner2-base-v1"
    # GLiNER-Decoder: open discovery, invents labels freely (Knowledgator).
    gliner_decoder_model: str = "knowledgator/gliner-decoder-base-v1.0"
    extraction_threshold: float = 0.3

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

    # ── Paths ──
    data_dir: Path = Path("./data")
    sample_conversations: Path = Path("./data/sample_conversations.jsonl")
    corpora_dir: Path = Path("./data/corpora")


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


config = Config()