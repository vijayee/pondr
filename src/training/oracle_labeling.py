"""Oracle labeling infrastructure for Phase 1d GNN training (infra only).

Phase 1b Phase G builds the *plumbing* for oracle-labeled subgraph training
examples — no live oracle (Bonsai) calls happen here. The pipeline:

1. ``extract_subgraph(center_id, radius=3)`` — BFS over the memory graph from a
   center node (typically an episode id), following node-to-node predicates
   (``has_entity`` / ``has_topic`` / ``has_tone`` / ``has_decision`` /
   ``in_episode`` / ``has_session`` / ``in_session`` / ``follows`` /
   ``follows_session``) in BOTH directions, up to ``radius`` hops. Returns
   ``{"nodes": [...], "edges": [...]}`` with node types inferred from the id
   prefix (``E:``/``T:``/``A:``/``D:``/``S:``/``U:``/``ep_``).
2. ``ORACLE_GNN_LABELING_PROMPT`` — the prompt that a future phase sends to the
   local Bonsai server to label a subgraph (relevance / salience / link targets
   for the GNN). Defined here, not invoked in 1b.

``scripts/generate_training_data.py`` is a thin runner that extracts subgraphs
for a sample of episodes and writes them to JSONL.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..memory.store import HippocampalStore

# Node-to-node predicates traversed by the BFS. Literal-valued predicates
# (at_time, state, validity_start, ended_at) are excluded — their objects are
# timestamps/strings, not graph nodes, so they'd inject non-node "neighbors".
_NODE_PREDICATES = (
    "has_entity",
    "has_topic",
    "has_tone",
    "has_decision",
    "in_episode",
    "has_session",
    "in_session",
    "follows",
    "follows_session",
)

# Prompt for the oracle (local Bonsai) to label a subgraph for GNN training.
# Slot {subgraph_json} receives the extract_subgraph output. Not invoked in 1b.
ORACLE_GNN_LABELING_PROMPT = (
    "You are labeling a memory subgraph for a graph neural network that ranks "
    "which past episodes to recall for a future query.\n\n"
    "Given a subgraph centered on an episode, output JSON with a `labels` list. "
    "Each label is an object with:\n"
    '  - "node": the node id,\n'
    '  - "relevance": float in [0, 1] (how relevant this node is to the center '
    "episode's topic),\n"
    '  - "salience": float in [0, 1] (how memorable / surprising this node is),\n'
    '  - "should_recall": boolean (whether an episode node should be retrieved '
    "given the center).\n\n"
    "Subgraph (JSON):\n{subgraph_json}\n\n"
    "Return ONLY the JSON object."
)


def _node_type(node_id: str) -> str:
    """Infer the node type from its id prefix."""
    for prefix, kind in (
        ("E:", "entity"), ("T:", "topic"), ("A:", "tone"),
        ("D:", "decision"), ("S:", "session"), ("U:", "user"),
    ):
        if node_id.startswith(prefix):
            return kind
    if node_id.startswith("ep_"):
        return "episode"
    return "unknown"


class OracleLabelingPipeline:
    """Extracts subgraphs for oracle labeling (Phase 1d GNN training prep)."""

    def __init__(self, store: HippocampalStore) -> None:
        self.store = store
        self.graph = store.graph

    def _get_neighbors(self, node_id: str) -> list[tuple[str, str, str]]:
        """Return ``(neighbor_id, predicate, direction)`` for all node-to-node
        edges incident to ``node_id``. direction is "out" (node→neighbor) or
        "in" (neighbor→node) — recorded so the caller can orient the edge.
        """
        out: list[tuple[str, str, str]] = []
        for pred in _NODE_PREDICATES:
            for direction, query in (
                ("out", self.graph.query().vertex(node_id).out(pred)),
                ("in", self.graph.query().vertex(node_id).in_(pred)),
            ):
                result = query.execute_sync()
                try:
                    neighbors = list(result.vertices)
                finally:
                    result.close()
                for nb in neighbors:
                    out.append((nb, pred, direction))
        return out

    def extract_subgraph(self, center_id: str, radius: int = 3) -> dict:
        """BFS from ``center_id`` up to ``radius`` hops over node-to-node edges.

        Returns ``{"center": center_id, "radius": radius, "nodes": [...],
        "edges": [...]}`` where each node is ``{"id", "type", "depth"}`` (depth
        = hop distance from the center) and each edge is ``{"subject",
        "predicate", "object"}`` with directions normalized so subject→object
        matches the stored triple orientation.
        """
        nodes: dict[str, dict] = {
            center_id: {"id": center_id, "type": _node_type(center_id), "depth": 0}
        }
        edges: dict[tuple[str, str, str], None] = {}  # ordered dedup set
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(center_id, 0)])

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            if depth >= radius:
                continue
            for nb, pred, direction in self._get_neighbors(node_id):
                # Orient the edge so subject→object reflects stored direction:
                # "out" means node_id → nb; "in" means nb → node_id.
                if direction == "out":
                    edges[(node_id, pred, nb)] = None
                else:
                    edges[(nb, pred, node_id)] = None
                if nb not in nodes:
                    nodes[nb] = {"id": nb, "type": _node_type(nb), "depth": depth + 1}
                if nb not in visited:
                    queue.append((nb, depth + 1))

        return {
            "center": center_id,
            "radius": radius,
            "nodes": list(nodes.values()),
            "edges": [{"subject": s, "predicate": p, "object": o}
                      for (s, p, o) in edges],
        }

    def build_labeling_prompt(self, center_id: str, radius: int = 3) -> str:
        """Render the oracle labeling prompt for a subgraph (not sent in 1b)."""
        import json
        sub = self.extract_subgraph(center_id, radius=radius)
        return ORACLE_GNN_LABELING_PROMPT.format(subgraph_json=json.dumps(sub))


def sample_episode_centers(store: HippocampalStore, n: Optional[int] = None) -> list[str]:
    """Return up to ``n`` episode ids to use as subgraph centers (all if None)."""
    ids: set[str] = set()
    for k, _ in store.db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
        parts = k.split("/", 3)
        if len(parts) >= 3 and parts[2]:
            ids.add(parts[2])
    centers = sorted(ids)
    return centers if n is None else centers[:n]