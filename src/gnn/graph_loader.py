"""WaveDB memory-graph → PyG ``Data`` loader (Phase 3a Task 1).

Reuses the Phase-1d subgraph BFS (``OracleLabelingPipeline.extract_subgraph``)
so the loader and the label generator walk the SAME subgraph for a given center
+ radius — a training example and its label are over the same node/edge set by
construction. The loader's only new job is to attach node features
(``features.NodeFeatureBuilder``) and emit ``torch_geometric.data.Data``.

The BFS follows node-to-node predicates in BOTH directions, so the emitted
``edge_index`` already contains both orientations (``ep→E:has_entity`` and
``E→ep:in_episode``) — GAT message passing sees a bidirectional graph without
needing a reverse-edge copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch_geometric.data import Data

from .features import FEATURE_DIM, NodeFeatureBuilder, infer_kind, NODE_KIND_INDEX
from ..training.oracle_labeling import OracleLabelingPipeline

if TYPE_CHECKING:
    from ..memory.store import HippocampalStore


# Predicate vocabulary for ``edge_attr`` onehots. The loader only ever traverses
# the 9 node-to-node predicates below (``KNOWN_PREDICATES``), so each maps to a
# fixed index. ``PREDICATE_VOCAB`` reserves 32 slots (configurable via
# ``GNNConfig.predicate_vocab_size``) so ontology-refinement / Bonsai-relation
# edges the GNN later scores can be hashed in without reindexing.
KNOWN_PREDICATES: tuple[str, ...] = (
    "has_entity", "has_topic", "has_tone", "has_decision",
    "in_episode", "has_session", "in_session", "follows", "follows_session",
)
_PREDICATE_INDEX: dict[str, int] = {p: i for i, p in enumerate(KNOWN_PREDICATES)}
PREDICATE_VOCAB: int = 32


def _predicate_index(pred: str) -> int:
    """Fixed slot for a known predicate; hashed slot in [9, 32) for an unknown one."""
    idx = _PREDICATE_INDEX.get(pred)
    if idx is not None:
        return idx
    # Hash unknown predicates (Bonsai relations, future ontology edges) into the
    # tail of the vocab without colliding with the known slots.
    tail = PREDICATE_VOCAB - len(KNOWN_PREDICATES)
    if tail <= 0:
        return len(KNOWN_PREDICATES)  # fall back to the first "other" slot
    h = hash(pred) % tail
    return len(KNOWN_PREDICATES) + h


def _predicate_onehot(pred: str, vocab_size: int = PREDICATE_VOCAB) -> torch.Tensor:
    vec = torch.zeros(vocab_size)
    vec[_predicate_index(pred)] = 1.0
    return vec


def data_from_subgraph(
    subgraph: dict,
    feature_for: Callable[[str], tuple[int, torch.Tensor]],
    predicate_vocab_size: int = PREDICATE_VOCAB,
) -> Data:
    """Convert an ``extract_subgraph`` dict into a PyG ``Data``.

    ``feature_for(node_id) -> (kind_index, feature_vector)`` is supplied by
    ``NodeFeatureBuilder.feature_for`` (store-backed) or a stub in tests. This
    function is pure (no store) so it is unit-testable without WaveDB.

    The ``Data`` carries:
    - ``x``            [N, FEATURE_DIM] raw node features
    - ``node_kind``    [N] long  — kind index per node
    - ``node_depth``   [N] long  — hop distance from the center
    - ``edge_index``   [2, E] long — oriented subject→object
    - ``edge_attr``    [E, predicate_vocab_size] — predicate onehot
    - ``node_id``      list[str] — the original node ids (same order as rows)
    - ``center_idx``   long — index of the center in ``node_id``
    """
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]
    if not nodes:
        # Empty subgraph (center not in graph): emit a trivial 1-node Data so
        # downstream code doesn't crash on a zero-dim tensor.
        nodes = [{"id": subgraph["center"], "type": "unknown", "depth": 0}]

    id_to_idx: dict[str, int] = {}
    node_ids: list[str] = []
    x_rows: list[torch.Tensor] = []
    kinds: list[int] = []
    depths: list[int] = []
    for n in nodes:
        nid = n["id"]
        # The extractor's "type" may differ from infer_kind on edge cases; the
        # feature builder drives the kind, so use it as the source of truth.
        kind_idx, vec = feature_for(nid)
        id_to_idx[nid] = len(node_ids)
        node_ids.append(nid)
        x_rows.append(vec)
        kinds.append(kind_idx)
        depths.append(int(n.get("depth", 0)))

    x = torch.stack(x_rows, dim=0) if x_rows else torch.zeros(0, FEATURE_DIM)
    node_kind = torch.tensor(kinds, dtype=torch.long)
    node_depth = torch.tensor(depths, dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    attrs: list[torch.Tensor] = []
    for e in edges:
        s, p, o = e["subject"], e["predicate"], e["object"]
        if s not in id_to_idx or o not in id_to_idx:
            # An edge to a node the BFS didn't keep (shouldn't happen — the BFS
            # adds both endpoints — but be defensive): skip rather than crash.
            continue
        src.append(id_to_idx[s])
        dst.append(id_to_idx[o])
        attrs.append(_predicate_onehot(p, predicate_vocab_size))

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.stack(attrs, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, predicate_vocab_size)

    center_idx = id_to_idx.get(subgraph["center"], 0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_kind = node_kind
    data.node_depth = node_depth
    data.node_id = node_ids
    data.center_idx = torch.tensor(center_idx, dtype=torch.long)
    data.predicate = subgraph.get("predicate", "")  # blank; subgraphs are center+radius
    return data


class WaveDBGraphLoader:
    """Loads a PyG ``Data`` subgraph from the WaveDB store.

    Thin wrapper: run the 1d BFS via ``OracleLabelingPipeline``, then attach
    features via ``NodeFeatureBuilder`` and convert to ``Data``.
    """

    def __init__(
        self,
        store: "HippocampalStore",
        radius: int = 3,
        predicate_vocab_size: int = PREDICATE_VOCAB,
        fanout_cap: Optional[int] = None,
    ) -> None:
        self.store = store
        self.radius = radius
        self.predicate_vocab_size = predicate_vocab_size
        # Per-node BFS fanout cap (see OracleLabelingPipeline._get_neighbors):
        # ``None`` = uncapped = the prior 10,680-node-giant behavior. A set cap
        # bounds the high-degree entity hubs so radius-2 subgraphs don't flood
        # to ~5,000 unrelated episodes. Default ``None`` keeps every existing
        # caller on the uncapped path (no behavior change).
        self.fanout_cap = fanout_cap
        self._pipe = OracleLabelingPipeline(store)
        self._features = NodeFeatureBuilder(store)

    def load(
        self,
        center_id: str,
        radius: Optional[int] = None,
        fanout_cap: Optional[int] = None,
    ) -> Data:
        """Load the radius-``r`` subgraph around ``center_id`` as PyG ``Data``.

        ``radius``/``fanout_cap`` override the instance defaults when given; both
        fall back to the instance value (and the instance ``fanout_cap`` defaults
        to ``None`` = uncapped). The loader and the Phase-1d label generator walk
        the same bounded subgraph for a given center + radius + cap.
        """
        r = self.radius if radius is None else radius
        c = self.fanout_cap if fanout_cap is None else fanout_cap
        sub = self._pipe.extract_subgraph(center_id, radius=r, fanout_cap=c)
        return data_from_subgraph(
            sub,
            self._features.feature_for,
            predicate_vocab_size=self.predicate_vocab_size,
        )

    def episode_centers(self, limit: Optional[int] = None) -> list[str]:
        """Episode ids usable as subgraph centers (delegates to the 1d sampler)."""
        from ..training.oracle_labeling import sample_episode_centers
        return sample_episode_centers(self.store, n=limit)