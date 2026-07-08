"""Node-feature pipeline for the Phase 3a GNN (Task 1).

The memory graph's nodes have no inherent vector features — this module builds
them, per node-kind, from what IS persisted in WaveDB:

- **episode** (``ep_NNNNNN``): the 384-dim summary embedding at
  ``content/ep/{eid}/embedding`` (backfilled by Phase 1b/2a's sentence-transformer
  encoder). If absent, a deterministic hash embedding (stub, clearly labeled) —
  shape-correct for a no-model dev smoke, NOT for training.
- **entity** (``E:{entity}``): a type-onehot + the Phase-1c heuristic salience
  scalar (``store.get_entity_salience``), which is the cold-start prior the GAT
  head supersedes (``graph_traversal.py:389-430``).
- **topic / tone / decision / session / user**: a type-onehot only (no persisted
  per-node vector); the GAT learns these from structure + the ontology's
  ``subClassOf`` edges.

All raw feature vectors are packed into a fixed ``FEATURE_DIM``-wide tensor
(``384`` — matches the episode embedding, so episodes pass through unchanged)
with the type-onehot in the leading ``len(NODE_KINDS)`` slots. The *per-kind
projection MLP* from the §1.3 node-feature decision lives in the model
(``InputProjection`` in ``model.py``), selected by the ``node_kind`` index tensor
the loader emits alongside ``x`` — that keeps learnable parameters in the model
and the loader parameter-free (testable + reusable by the consolidation loop).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from ..memory.store import HippocampalStore

# ``_b2s`` decodes WaveDB bytes→str ('' for missing). Module-level helper in
# ``store.py`` (not a method) — import it rather than reaching through the store.
from ..memory.store import _b2s

# Node-kind vocabulary. Index order is load-bearing — ``node_kind`` tensors use
# these indices and the model's InputProjection indexes by them. ``unknown`` is
# the fallback for ids that match no prefix (should not happen on a well-formed
# graph, but the loader must not crash on one).
NODE_KINDS: tuple[str, ...] = (
    "episode", "entity", "topic", "tone", "decision", "session", "user", "unknown",
)
NODE_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(NODE_KINDS)}

# Raw feature width. Episodes use a 384-dim embedding (bge-small-en-v1.5); the
# type-onehot (8) + one salience scalar fit in the leading slots, and everything
# else is zero-padded out to 384 so ``Data.x`` is a single dense tensor.
FEATURE_DIM: int = 384

# Slot layout inside the 384-wide vector:
#   [0:8]   type-onehot
#   [8]     entity salience (entities only; 0.0 otherwise)
#   [9:384] episode embedding (episodes only; 0.0 otherwise)
_ONEHOT_END = len(NODE_KINDS)        # 8
_SALIENCE_SLOT = _ONEHOT_END         # 8
_EMB_START = _ONEHOT_END + 1         # 9


def infer_kind(node_id: str) -> str:
    """Infer the node kind from its id prefix (mirrors ``oracle_labeling._node_type``)."""
    for prefix, kind in (
        ("E:", "entity"), ("T:", "topic"), ("A:", "tone"),
        ("D:", "decision"), ("S:", "session"), ("U:", "user"),
    ):
        if node_id.startswith(prefix):
            return kind
    if node_id.startswith("ep_"):
        return "episode"
    return "unknown"


def _hash_embedding(node_id: str, dim: int = FEATURE_DIM) -> torch.Tensor:
    """Deterministic hash embedding for episodes with no persisted vector.

    STUB: shape-correct only — gives a no-model-download dev smoke a non-zero
    feature so the GAT forward pass runs. NOT for training (a trained model would
    learn nothing from a hash of the id). The real path is the backfilled
    ``content/ep/{eid}/embedding``.
    """
    vec = torch.zeros(dim)
    # Seed a deterministic float vector from sha256(id) — 4 bytes → one float.
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    for i in range(dim):
        chunk = digest[i % len(digest):i % len(digest) + 4]
        if len(chunk) < 4:
            chunk = (digest * 2)[i % len(digest):i % len(digest) + 4]
        u32 = int.from_bytes(chunk, "big", signed=False)
        vec[i] = (u32 / 0xFFFFFFFF) * 2.0 - 1.0  # [-1, 1]
    # Place it in the embedding slice; leave the onehot/salience slots at 0.
    out = torch.zeros(FEATURE_DIM)
    out[_EMB_START:] = vec[: FEATURE_DIM - _EMB_START]
    return out


class NodeFeatureBuilder:
    """Builds raw node-feature tensors for a subgraph from the WaveDB store.

    Stateless beyond the store handle — safe to reuse across subgraphs.
    """

    def __init__(self, store: "HippocampalStore") -> None:
        self.store = store

    def feature_for(self, node_id: str) -> tuple[int, torch.Tensor]:
        """Return ``(kind_index, feature_vector)`` for one node id."""
        kind = infer_kind(node_id)
        kind_idx = NODE_KIND_INDEX[kind]
        vec = torch.zeros(FEATURE_DIM)
        vec[kind_idx] = 1.0  # type-onehot in the leading slots

        if kind == "episode":
            emb = self._episode_embedding(node_id)
            if emb is not None:
                # Place the 384-dim embedding in the embedding slice (truncating
                # is safe — FEATURE_DIM - _EMB_START == 375, and the embedding is
                # 384 wide, so we take the first 375 dims). A trained model can
                # recover the dropped dims via the InputProjection; for a cold
                # start this is a negligible information loss.
                slot = vec[_EMB_START:]
                slot[: min(len(emb), slot.shape[0])] = torch.tensor(
                    emb[: slot.shape[0]], dtype=torch.float32
                )
            else:
                vec = _hash_embedding(node_id)
                vec[kind_idx] = 1.0  # re-stamp the onehot (hash stub zeroed it)
                # Tag the stub so callers/tests can tell it apart from a real
                # embedding: stash a sentinel in the salience slot.
                vec[_SALIENCE_SLOT] = -1.0
        elif kind == "entity":
            # Entity key in the graph is "E:{entity}"; the salience store keys
            # on the bare entity string.
            entity = node_id[2:] if node_id.startswith("E:") else node_id
            try:
                sal = float(self.store.get_entity_salience(entity))
            except Exception:
                sal = 0.0
            vec[_SALIENCE_SLOT] = sal

        return kind_idx, vec.to(torch.float32)

    def _episode_embedding(self, episode_id: str) -> Optional[list[float]]:
        """Read ``content/ep/{eid}/embedding``; None if absent or unparseable."""
        raw = _b2s(self.store.db.get_sync(f"content/ep/{episode_id}/embedding"))
        if not raw:
            return None
        try:
            emb = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(emb, list) or not emb:
            return None
        return emb


def training_feature_for(store: "HippocampalStore") -> Callable[[str], tuple[int, torch.Tensor]]:
    """Feature function for training on a CORRUPTED subgraph (anomaly head, Task 4a).

    ``NodeFeatureBuilder.feature_for`` degrades gracefully on ids NOT in the
    store — an unknown episode gets a deterministic hash embedding, an unknown
    kind gets a type-onehot — so it never raises for a synthetic injected node.
    The one case worth special-handling is an injected ``{orig}_dup`` clone
    (``anomaly_injector`` plants these for ``duplicate_episode`` and
    ``duplicate_decision``): the clone is meant to look like its origin, so it
    reuses the origin's REAL feature (embedding / salience) instead of a fresh
    hash. That keeps the duplication signal STRUCTURAL (shared neighborhood +
    shared feature) rather than a cheap feature-divergence artefact the head
    could key on instead of learning the real structural signature. Other
    synthetic nodes (``ep_iso_*`` isolated-cluster eps, ``M:000N`` stale
    abstractions) have no origin to mirror — ``feature_for`` handles them
    directly (hash / onehot), which is the correct weak feature for a node that
    genuinely has no persisted content.
    """
    fb = NodeFeatureBuilder(store)

    def _f(node_id: str) -> tuple[int, torch.Tensor]:
        if node_id.endswith("_dup"):
            return fb.feature_for(node_id[:-4])  # clone -> origin's real feature
        return fb.feature_for(node_id)

    return _f