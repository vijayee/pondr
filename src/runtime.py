"""Runtime entrypoint: build a live ``PonderOrchestrator`` on the TRAINED models.

This is the wiring the SSM/JEPA consistency audit (2026-07-15) found missing.
The trained Phase 2a backbone (``backbone_final.pt``) and the trained Phase 2b
RetrievalGate (``phase2b/best.pt``) existed on disk but were never loaded into a
live query path -- every ``PonderOrchestrator`` construction lived in tests and
passed a FRESH ``JGSBackbone(BackboneConfig())``, so the SSM/JEPA ran at runtime
on untrained random weights. ``load_backbone`` / ``load_retrieval_gate`` were
training-only.

``build_ponder`` is the factory that closes that gap: it loads the trained
backbone (frozen) + the trained gate, wires the real bge-small embedder, the
retriever (with the gate injected + the store's vector index auto-loaded), the
Mode A generator against the Bonsai endpoint, and -- by default -- a
``HippocampalEncoder`` so each exchange is persisted as a learnable episode
("the system learns from use"; live-encode is the product vision). The
GLiNER device is threaded through so live-encode runs GLiNER on CUDA (the
~20s/conv CPU bottleneck) with an OOM-safe CPU fallback.

Pure DI: every collaborator is constructed here and passed into the
orchestrator; ``mode_a`` is injectable so tests can stub the LLM and exercise
the wiring offline (no Bonsai). The caller owns the store's lifetime
(``orch.store.close()`` when done -- see ``scripts/serve_ponder.py``).
"""

from __future__ import annotations

from typing import Optional

from .config import Phase2cConfig, config
from .encoding.encoder import HippocampalEncoder
from .generation.mode_a import ModeAGenerator
from .memory.store import HippocampalStore
from .orchestrator import PonderOrchestrator
from .retrieval.query_planner import BonsaiQueryPlanner
from .retrieval.retriever import HippocampalRetriever
from .subconscious.configs import BackboneConfig
from .subconscious.training.routing_training import (
    build_embedder,
    load_backbone,
    load_retrieval_gate,
)

# Default trained-checkpoint paths (the artifacts the audit found unwired).
DEFAULT_BACKBONE_PATH = (
    "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
)
DEFAULT_GATE_PATH = "data/pod_runs/phase2b/best.pt"


def build_ponder(
    db_path: Optional[str] = None,
    *,
    backbone_path: str = DEFAULT_BACKBONE_PATH,
    gate_path: str = DEFAULT_GATE_PATH,
    embedder_source: str = "on-demand",
    bonsai_endpoint: Optional[str] = None,
    device: str = "auto",
    gliner_device: str = "auto",
    gliner_timing: bool = True,
    live_encode: bool = True,
    user_id: str = "ponder",
    mode_a: Optional[ModeAGenerator] = None,
    config_override: Optional[Phase2cConfig] = None,
    relevance_head_path: Optional[str] = None,
    graduation_proxy: bool = False,
    ring_capacity: int = 0,
) -> PonderOrchestrator:
    """Build a live ``PonderOrchestrator`` on the TRAINED backbone + gate.

    Args:
        db_path: WaveDB path for the hippocampal store. Defaults to
            ``config.db_path`` (``$HIPPOCAMPAL_DB_PATH`` / ``./data/memory_db``).
        backbone_path: Phase 2a backbone checkpoint (``backbone_final.pt``).
        gate_path: Phase 2b RetrievalGate checkpoint (``best.pt`` / ``final.pt``).
        embedder_source: ``"on-demand"`` (real bge-small-en-v1.5, 384-dim -- the
            embeddings the gate was trained on, so routing is meaningful) or
            ``"stub"`` (deterministic hash; shape-only, for offline wiring tests).
        bonsai_endpoint: override for the Mode A / planner LLM endpoint; defaults
            to ``config.bonsai_endpoint`` (the local 8B server).
        device: device for the backbone + gate (``"auto"`` -> CUDA if available).
        gliner_device: device for the GLiNER models when ``live_encode`` (same
            ``"auto"`` semantics; CUDA move is OOM-safe -> CPU fallback).
        gliner_timing: log per-stage GLiNER extraction timing to stderr.
        live_encode: persist each exchange as a new episode via
            ``HippocampalEncoder`` (default True -- the product vision; the
            system learns from use). False -> no encoder, no GLiNER import.
        user_id: the user the encoder attributes live-encoded episodes to.
        mode_a: inject a Mode A generator (tests stub the LLM); ``None`` builds
            a real ``ModeAGenerator`` against ``bonsai_endpoint``.
        config_override: inject a ``Phase2cConfig``; ``None`` builds a default
            (with ``session.state_dir`` set under the store's parent dir).
        relevance_head_path: optional STRM Phase 2a relevance-head checkpoint
            (``best.pt`` from ``scripts/train_relevance_head.py``). When set,
            ``load_relevance_head`` builds the head and it is attached to the
            orchestrator (it scores each WM ring slot's relevance to the query;
            Phase 3's context-builder consumes ``r_i``). ``None`` (default) ->
            no relevance head at serve (byte-identical to pre-2a).
        graduation_proxy: when True, attach the STRM Phase 2d v1 graduation
            proxy (``GraduationProxyV1`` -- the parameter-free ``integral(r_i
            dt)`` heuristic baseline the v2 head must beat). No checkpoint, no
            training -- the proxy reads the 2a ``r_i`` stream. Default False
            (byte-identical to pre-2d). Full graduation -> LTM promotion is
            Phase 4; this round only makes the proxy attachable + the flag
            plumbed.
        ring_capacity: WM ring buffer capacity K (default 0 = OFF, byte-
            identical to Phase 2c). The STRM 2a relevance head + the 2d
            graduation replay logger need the ring ON to populate per-slot
            state; pass K>0 (e.g. 16) to turn it on at serve. Provenance
            (episode_id + summary) is threaded into each recalled-episode
            inject so the ring slots carry ``source_id``/``text``.

    Returns:
        A ready ``PonderOrchestrator`` whose retriever gate is the TRAINED
        gate on the TRAINED frozen backbone. The caller closes the store.
    """
    store = HippocampalStore(db_path or config.db_path)

    # Trained backbone (frozen) + trained gate on that shared backbone. The
    # gate's state_dict excludes the backbone (object.__setattr__), so
    # load_retrieval_gate reuses this frozen backbone rather than reloading.
    backbone = load_backbone(backbone_path, BackboneConfig(), device=device)
    gate = load_retrieval_gate(gate_path, backbone, device=device)

    # Real bge-small embedder: 384-dim, matching the gate's training embeddings
    # AND the backbone's d_model. Injected into the retriever (for routing) and
    # the orchestrator (for WM embed / chunking / live-encode).
    embedder = build_embedder(embedder_source)

    endpoint = bonsai_endpoint
    planner = BonsaiQueryPlanner(endpoint=endpoint)
    retriever = HippocampalRetriever(
        store,
        planner=planner,
        auto_load_index=True,
        retrieval_gate=gate,
        embedder=embedder,
    )

    # Phase 1c Refinement 1: attach the document-aware aggregator so multi-
    # section hits surface as one document result. Guarded by a cheap
    # ``has_section`` POS probe -- conversation-only corpora (no document
    # edges) skip aggregation entirely (``document_retriever`` stays ``None``
    # -> retrieval is byte-identical to the pre-1c path).
    from .retrieval.document_retriever import DocumentRetriever, store_has_documents
    if store_has_documents(store):
        retriever.document_retriever = DocumentRetriever(store)

    gen = mode_a if mode_a is not None else ModeAGenerator(retriever, endpoint=endpoint)

    # Live-encode (default on): the encoder runs GLiNER on the resolved device
    # (CUDA w/ OOM-safe CPU fallback) so the exchange becomes a retrievable
    # episode. The orchestrator auto-starts one session per instance and
    # best-effort-persists, so no manual session lifecycle is needed here.
    encoder: Optional[HippocampalEncoder] = None
    if live_encode:
        encoder = HippocampalEncoder(
            store,
            user_id,
            gliner_device=gliner_device,
            gliner_timing=gliner_timing,
        )

    cfg = config_override or Phase2cConfig()
    if config_override is None:
        # Put session state beside the memory DB so one data dir holds everything.
        cfg.session.state_dir = _sessions_dir_for(db_path or config.db_path)

    # STRM Phase 2a relevance head (optional). Loaded only when a checkpoint
    # path is given (default off, mirroring --doc-kind-ensemble). The head reads
    # a ring slot + a query at serve -- no backbone -- so it loads independently
    # of the frozen backbone above. Full r_i -> context-builder consumption is
    # Phase 3; this round only makes the head loadable + the flag plumbed.
    relevance_head = None
    if relevance_head_path:
        from .subconscious.relevance_head import load_relevance_head
        relevance_head = load_relevance_head(relevance_head_path, device=device)

    # STRM Phase 2d v1 graduation proxy (optional, parameter-free). Attached
    # only when graduation_proxy is True (default off). No checkpoint -- the
    # proxy is the integral(r_i dt) heuristic the v2 head must beat. Full
    # graduation -> LTM promotion is Phase 4; this round only attaches it.
    graduation_proxy_head = None
    if graduation_proxy:
        from .subconscious.graduation_head import GraduationProxyV1
        graduation_proxy_head = GraduationProxyV1()

    orch = PonderOrchestrator(
        store=store,
        retriever=retriever,
        backbone=backbone,
        embedder=embedder,
        mode_a=gen,
        config=cfg,
        user_id=user_id,
        encoder=encoder,
        relevance_head=relevance_head,
        graduation_proxy=graduation_proxy_head,
        ring_capacity=ring_capacity,
    )
    return orch


def _sessions_dir_for(db_path: str) -> str:
    """Sibling ``sessions`` dir next to the memory DB (one data dir for all)."""
    import os

    parent = os.path.dirname(db_path.rstrip("/").rstrip("\\"))
    return os.path.join(parent or ".", "sessions")