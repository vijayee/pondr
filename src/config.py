"""Central configuration for the hippocampal memory system."""

import os
from dataclasses import dataclass
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


config = Config()