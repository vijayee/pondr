"""Structural anomaly detectors for the GNN anomaly head (Phase 3a Task 3).

The anomaly head is trained on **injected corruption labels**, not Oracle
labels (spec §2 of ``docs/Phase 3a Task 3 - sharded labeling design.md``): the
DialogSum corpus is anomaly-free by construction, so a clean corpus can't teach
a multi-label BCE head (it would collapse to "predict 0" and couldn't be
F1-evaluated — no positives). The injector (``anomaly_injector.py``) corrupts a
clean enriched subgraph; the rule detectors here label each corruption
deterministically — **zero Oracle calls** for the head. The head learns the
structural signature of each corruption; at deploy these same rules run as the
ground-truth backstop, the head is a cheap pre-filter that flags candidates for
the rules + the Bonsai decider (spec §2.5).

This module owns the canonical **9-type taxonomy** (``ANOMALY_TYPES``) —
``heads.py`` imports it so the head's output slots and the training labels stay
aligned by construction. The 9 types supersede the 6-type Oracle-prompt schema
that previously lived in ``heads.py`` (the ``madeBy``-artifact orphan and the
vague "contradiction" were dropped; the rest were concretized to the real
lifelong-memory failure modes — see the spec §2 table).

**IDENTITY_DRIFT** is a review-flag, NOT a head label (spec §2 / §9): "one node
name, two different referents" is genuinely semantic — no rule can decide it,
the only clean signal (type-level ``subClassOf`` incompatibility) is too rare
to train on, and the naive "disjoint topic neighborhoods" heuristic over-fires
on every legitimately multifaceted entity. So it is emitted as a flag-for-review
(``flag_identity_drift``) routed to the Bonsai decider, deliberately over-firing.

Detector contract — all detectors are PURE functions of an **enriched
subgraph dict**::

    {"center": str,
     "nodes": [{"id", "type", "depth", "summary"?: str, ...}],
     "edges": [{"subject", "predicate", "object"}]}

An edge's ``object`` may be a node id OR a literal string (for ``state`` and
other data edges the injector plants or ``enrich_subgraph`` surfaces).
``OracleLabelingPipeline.extract_subgraph`` produces node-to-node edges only;
``enrich_subgraph`` (below) hydrates episode summaries and surfaces the
data/extra predicates the rules need (``state``, ``abstracts``, ``supersedes``,
``subClassOf``). Both the training pre-inject path and the deploy pre-detect
path produce the same enriched shape, so the detectors take no store handle —
they're unit-testable from a hand-built dict.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from ..memory.ontology import SEED_ONTOLOGY

if TYPE_CHECKING:
    from ..memory.store import HippocampalStore

# ── Canonical 9-type taxonomy (source of truth; heads.py imports this) ──
# Index order is load-bearing: the head's 9 output slots correspond to these
# in order. Supersedes the 6-type Oracle-prompt schema (orphan_decision,
# missing_temporal, contradiction, type_violation, isolated_cluster,
# duplicate_decision). Dropped: missing_temporal (replaced by the more
# concrete broken_follows), contradiction (replaced by contradictory_state).
# Added: duplicate_episode, detached_episode, broken_follows,
# stale_abstraction.
ANOMALY_TYPES: tuple[str, ...] = (
    "contradictory_state",
    "duplicate_episode",
    "duplicate_decision",
    "orphan_decision",
    "detached_episode",
    "broken_follows",
    "type_violation",
    "isolated_cluster",
    "stale_abstraction",
)
ANOMALY_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(ANOMALY_TYPES)}

# The identity-drift review-flag is NOT a head label (no index in
# ANOMALY_TYPES). It is routed to the Bonsai decider, not trained against.
IDENTITY_DRIFT_FLAG: str = "identity_drift"

# Node-to-node predicates the BFS traverses (mirrors
# ``oracle_labeling._NODE_PREDICATES``). Used for degree-based detectors
# (orphan_decision, detached_episode) and the connected-components walk
# (isolated_cluster excludes ``subClassOf`` taxonomy edges, which are seeded
# and would otherwise connect everything).
LINK_PREDICATES: frozenset[str] = frozenset({
    "has_entity", "has_topic", "has_tone", "has_decision",
    "in_episode", "has_session", "in_session", "follows", "follows_session",
})

# Node-kind → ontology class. The encoder keys nodes by id prefix only (no
# per-instance class is stored), so the rule can only check the coarse class.
# Subclass nuance (Person is an Entity subclass) is NOT resolved here — a
# cold-start limitation, documented; the head learns from structure too.
_KIND_CLASS: tuple[tuple[str, str], ...] = (
    ("ep_", "Episode"), ("E:", "Entity"), ("T:", "Topic"), ("A:", "AffectiveTone"),
    ("D:", "Decision"), ("S:", "Session"), ("U:", "User"),
)
_NODE_PREFIXES: tuple[str, ...] = tuple(p for p, _ in _KIND_CLASS) + ("M:",)

# Precomputed property domain/range table from the seed ontology (declared
# relations only), keyed by the GRAPH predicate form. The ontology registry
# uses camelCase keys (``hasEntity``/``hasTopic``) but the graph stores
# snake_case predicates (``has_entity``/``has_topic``) — except ``subClassOf``,
# which is camelCase in BOTH. Normalize so a graph predicate looks up its
# domain/range directly. ``state`` / ``abstracts`` / ``supersedes`` are NOT
# declared (data/extra edges) → never trigger type_violation, by design.
def _to_graph_predicate(name: str) -> str:
    """Ontology property key → graph predicate form.

    ``subClassOf`` and ``instanceOf`` are camelCase in the graph (kept as-is);
    everything else is camelCase→snake_case (``hasEntity``→``has_entity``).
    Already-snake keys (``has_session``, ``defined_in``) pass through unchanged.
    """
    if name in ("subClassOf", "instanceOf"):
        return name
    # Insert ``_`` before each uppercase letter (not at start), then lowercase.
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).lower()


_DECLARED_PROPERTIES: dict[str, dict[str, str]] = {
    _to_graph_predicate(name): spec
    for name, spec in SEED_ONTOLOGY["properties"].items()
}

# Default token-overlap threshold for duplicate detection (Jaccard on the
# normalized token sets of two summaries / decision texts). The injector
# clones exactly → Jaccard 1.0; 0.9 keeps it robust to a one-token clone
# suffix without over-firing on merely-similar episodes.
DUPLICATE_JACCARD_THRESHOLD: float = 0.9


def _kind_class(node_id: str) -> Optional[str]:
    """Ontology class for a node id from its prefix, or ``None`` if unknown."""
    for prefix, cls in _KIND_CLASS:
        if node_id.startswith(prefix):
            return cls
    return None


def _is_node_id(value: str) -> bool:
    """True if ``value`` looks like a node id (a known prefix), else a literal."""
    return value.startswith(_NODE_PREFIXES)


def _tokens(text: str) -> frozenset[str]:
    """Lowercased whitespace/punctuation-split token set for overlap comparison."""
    # Cheap, dependency-free tokenizer. Not linguistic — only has to agree on
    # both sides of a duplicate comparison, which it does.
    return frozenset(re.findall(r"[0-9a-z]+", (text or "").lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets (0.0 if both empty)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Enrichment — produces the enriched subgraph dict the detectors consume
# ═══════════════════════════════════════════════════════════════════════════

def enrich_subgraph(store: "HippocampalStore", subgraph: dict) -> dict:
    """Hydrate a raw ``extract_subgraph`` dict into the enriched form detectors
    consume — used by BOTH the training pre-inject path and the deploy
    pre-detect path, so the detectors stay pure dict-functions.

    Adds:
    - ``summary`` on each episode node (via ``store.get_episode``) — needed by
      duplicate_episode.
    - Data/extra graph edges incident to subgraph nodes that ``extract_subgraph``
      doesn't surface (it only walks ``LINK_PREDICATES``): ``abstracts`` (M: →
      ep_), ``supersedes`` (M: → M:), ``subClassOf`` (taxonomy), and ``state``
      graph edges if any exist. These feed stale_abstraction, type_violation,
      and contradictory_state respectively.

    Returns the SAME dict (mutated in place) for convenience.
    """
    graph = store.graph
    node_ids = {n["id"] for n in subgraph["nodes"]}
    extra_edges: dict[tuple[str, str, str], None] = {
        (e["subject"], e["predicate"], e["object"]): None
        for e in subgraph["edges"]
    }

    # ── episode summaries (duplicate_episode) ──
    # ``store.get_episode`` returns an ``Episode`` dataclass (attribute access),
    # not a dict; ``GraphTraversal.hydrate_episode`` (used by the generator)
    # returns a dict. Both carry ``summary`` — use ``getattr`` to be agnostic.
    for node in subgraph["nodes"]:
        if node.get("type") == "episode" and "summary" not in node:
            ep = store.get_episode(node["id"])
            if ep is not None:
                node["summary"] = getattr(ep, "summary", "") or ""

    # ── surface data/extra graph edges for each subgraph node ──
    # Iterate the NODES LIST (deterministic order), not the node_ids set —
    # set iteration order is hash-seed-dependent, which would make the
    # enriched edge order (and thus training-data determinism) vary across
    # processes.
    for node in subgraph["nodes"]:
        nid = node["id"]
        for pred in ("abstracts", "supersedes", "subClassOf", "state"):
            for direction, query in (
                ("out", graph.query().vertex(nid).out(pred)),
                ("in", graph.query().vertex(nid).in_(pred)),
            ):
                result = query.execute_sync()
                try:
                    neighbors = list(result.vertices)
                finally:
                    result.close()
                for nb in neighbors:
                    if direction == "out":
                        edge = (nid, pred, nb)
                    else:
                        edge = (nb, pred, nid)
                    # Phase 4 (D2): edge-currentness on the ``state`` branch.
                    # A fact-level tombstone marks an assertion edge's sidecar
                    # ``state="superseded"`` (``supersede_assertion``); the graph
                    # edge itself is NOT deleted (MVCC). Without this filter the
                    # tombstoned OLD value would still be surfaced here and
                    # ``_detect_contradictory_state`` would keep flagging a
                    # contradiction that was already adjudicated -- the
                    # tombstone would not *resolve* it. ``is_edge_current``
                    # returns True for an edge with no sidecar (episode-level
                    # ``(eid, state, "current")`` + injector-planted edges),
                    # so the 3b no-sidecar path is unchanged. Only the
                    # ``state`` predicate carries assertion tombstones; the
                    # other three predicates have no per-edge sidecar.
                    if pred == "state" and not store.is_edge_current(*edge):
                        continue
                    extra_edges[edge] = None

    subgraph["edges"] = [
        {"subject": s, "predicate": p, "object": o}
        for (s, p, o) in extra_edges
    ]
    return subgraph


# ═══════════════════════════════════════════════════════════════════════════
# The 9 rule detectors — each returns a list of findings
# {"node": str, "type": str, "evidence": str}
# ═══════════════════════════════════════════════════════════════════════════

def _detect_contradictory_state(subgraph: dict) -> list[dict]:
    """A node carrying >1 distinct live ``state`` value (facts changed over
    time). Edge shape: ``(node, state, literal)``. The injector plants two
    ``state`` edges with different values on one entity."""
    by_node: dict[str, set[str]] = {}
    for e in subgraph["edges"]:
        if e["predicate"] == "state" and not _is_node_id(e["object"]):
            by_node.setdefault(e["subject"], set()).add(e["object"])
    findings = []
    for nid, values in by_node.items():
        if len(values) > 1:
            findings.append({"node": nid, "type": "contradictory_state",
                             "evidence": f"distinct states: {sorted(values)}"})
    return findings


def _detect_duplicate_episode(subgraph: dict) -> list[dict]:
    """Two episode nodes with near-identical summaries (re-import / cross-device
    sync). Pairwise Jaccard ≥ ``DUPLICATE_JACCARD_THRESHOLD`` on summary token
    sets. O(E²) over the subgraph's episodes — bounded by the subgraph size."""
    eps = [(n["id"], n.get("summary", "")) for n in subgraph["nodes"]
           if n.get("type") == "episode"]
    sigs = [(nid, _tokens(s)) for nid, s in eps]
    findings = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            nid_a, ta = sigs[i]
            nid_b, tb = sigs[j]
            if not ta or not tb:
                continue
            if _jaccard(ta, tb) >= DUPLICATE_JACCARD_THRESHOLD:
                findings.append({"node": nid_a, "type": "duplicate_episode",
                                 "evidence": f"duplicate of {nid_b}"})
                findings.append({"node": nid_b, "type": "duplicate_episode",
                                 "evidence": f"duplicate of {nid_a}"})
    return findings


def _detect_duplicate_decision(subgraph: dict) -> list[dict]:
    """Two decision nodes (``D:``) with near-identical text (re-decide /
    re-ingest). The signature is the node's ``text`` field if present (the
    injector / a future encoder writes decision text there), else the id
    suffix after ``D:``; pairwise Jaccard ≥ threshold. The id suffix alone is a
    weak signal (slugs diverge), so deploy text-bearing decisions are what makes
    this fire — the head still learns the signature from injected clones."""
    decs = []
    for n in subgraph["nodes"]:
        if n.get("type") == "decision" or n["id"].startswith("D:"):
            text = n.get("text") or n["id"][2:]
            decs.append((n["id"], text))
    sigs = [(nid, _tokens(text)) for nid, text in decs]
    findings = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            nid_a, ta = sigs[i]
            nid_b, tb = sigs[j]
            if not ta or not tb:
                continue
            if _jaccard(ta, tb) >= DUPLICATE_JACCARD_THRESHOLD:
                findings.append({"node": nid_a, "type": "duplicate_decision",
                                 "evidence": f"duplicate of {nid_b}"})
                findings.append({"node": nid_b, "type": "duplicate_decision",
                                 "evidence": f"duplicate of {nid_a}"})
    return findings


def _incident_link_edges(subgraph: dict) -> dict[str, int]:
    """Per-node count of incident ``LINK_PREDICATES`` edges (degree on the
    structural predicates). Used by orphan_decision / detached_episode."""
    deg: dict[str, int] = {n["id"]: 0 for n in subgraph["nodes"]}
    for e in subgraph["edges"]:
        if e["predicate"] in LINK_PREDICATES:
            deg.setdefault(e["subject"], 0)
            deg.setdefault(e["object"], 0)
            deg[e["subject"]] = deg.get(e["subject"], 0) + 1
            deg[e["object"]] = deg.get(e["object"], 0) + 1
    return deg


def _detect_orphan_decision(subgraph: dict) -> list[dict]:
    """A ``D:`` node with zero incident link-predicate edges (partial ingest —
    the encoder crashed after creating the decision node but before linking
    it). The injector deletes the ``has_decision`` edge from a decision."""
    deg = _incident_link_edges(subgraph)
    return [{"node": n["id"], "type": "orphan_decision",
             "evidence": "degree 0 on link predicates"}
            for n in subgraph["nodes"]
            if n["id"].startswith("D:") and deg.get(n["id"], 0) == 0]


def _detect_detached_episode(subgraph: dict) -> list[dict]:
    """An ``ep_`` node with zero incident link-predicate edges (partial ingest —
    an episode recorded but never linked to its entities/topics). The injector
    strips an episode's link edges."""
    deg = _incident_link_edges(subgraph)
    return [{"node": n["id"], "type": "detached_episode",
             "evidence": "degree 0 on link predicates"}
            for n in subgraph["nodes"]
            if n["id"].startswith("ep_") and deg.get(n["id"], 0) == 0]


def _detect_broken_follows(subgraph: dict) -> list[dict]:
    """A ``follows`` edge whose object is not a node in the subgraph (dangling
    reference — the target was deleted, or the edge was rewired to a bad id).
    The injector rewires a ``follows`` edge to a removed id."""
    node_ids = {n["id"] for n in subgraph["nodes"]}
    findings = []
    for e in subgraph["edges"]:
        if e["predicate"] == "follows" and e["object"] not in node_ids:
            findings.append({"node": e["subject"], "type": "broken_follows",
                             "evidence": f"follows -> missing {e['object']}"})
    return findings


def _detect_type_violation(subgraph: dict) -> list[dict]:
    """An edge whose predicate is a DECLARED ontology property whose
    domain/range doesn't match the endpoint kinds (ontology drift over years).
    Cold-start checks the coarse ``_kind_class`` (no subclass closure) — a
    documented limitation; the head learns from structure too. Both endpoints
    are flagged. Literal objects are skipped (data edges aren't declared
    properties, so they never reach here)."""
    findings = []
    for e in subgraph["edges"]:
        prop = _DECLARED_PROPERTIES.get(e["predicate"])
        if prop is None:
            continue
        s_cls = _kind_class(e["subject"])
        if s_cls is None or s_cls != prop["domain"]:
            findings.append({"node": e["subject"], "type": "type_violation",
                             "evidence": f"{e['predicate']} domain "
                                         f"{prop['domain']} != {s_cls}"})
        if _is_node_id(e["object"]):
            o_cls = _kind_class(e["object"])
            if o_cls is None or o_cls != prop["range"]:
                findings.append({"node": e["object"], "type": "type_violation",
                                 "evidence": f"{e['predicate']} range "
                                             f"{prop['range']} != {o_cls}"})
    return findings


def _detect_isolated_cluster(subgraph: dict) -> list[dict]:
    """A connected component (over link predicates, EXCLUDING the seeded
    ``subClassOf`` taxonomy) that doesn't contain the center — a detached
    life-domain cluster. The injector detaches a component by removing its
    bridge edges to the center's component. All nodes in a non-center
    component are flagged (often legitimate — hence routed to Bonsai too)."""
    node_ids = [n["id"] for n in subgraph["nodes"]]
    node_set = set(node_ids)
    adj: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for e in subgraph["edges"]:
        if e["predicate"] == "subClassOf":
            continue  # taxonomy edges are seeded; don't let them connect clusters
        if e["predicate"] not in LINK_PREDICATES:
            continue  # only structural edges define component membership
        s, o = e["subject"], e["object"]
        if s in node_set and o in node_set:
            adj.setdefault(s, set()).add(o)
            adj.setdefault(o, set()).add(s)

    # BFS connected components.
    seen: set[str] = set()
    center = subgraph.get("center")
    findings = []
    for nid in node_ids:
        if nid in seen:
            continue
        comp: list[str] = []
        stack = [nid]
        seen.add(nid)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if center and center not in comp and len(comp) >= 2:
            # A component (a real cluster, ≥2 nodes) not containing the center
            # is isolated from the query focal point — flag every node in it.
            # Single-node components are orphan_decision / detached_episode's
            # job, not isolated_cluster (a cluster is ≥2 nodes by definition).
            for c in comp:
                findings.append({"node": c, "type": "isolated_cluster",
                                 "evidence": f"component size {len(comp)}, "
                                             f"no path to center"})
    return findings


def _detect_stale_abstraction(subgraph: dict) -> list[dict]:
    """An ``abstracts`` edge from an ``M:`` node whose target episode is not in
    the subgraph (the source was deleted — consolidator re-ingest dogfooding
    3a's own writes). The injector points ``abstracts`` at a dead ep id."""
    node_ids = {n["id"] for n in subgraph["nodes"]}
    findings = []
    for e in subgraph["edges"]:
        if e["predicate"] == "abstracts" and e["object"] not in node_ids:
            findings.append({"node": e["subject"], "type": "stale_abstraction",
                             "evidence": f"abstracts -> missing {e['object']}"})
    return findings


# Detectors in ANOMALY_TYPES index order.
_DETECTORS: tuple = (
    _detect_contradictory_state,  # 0
    _detect_duplicate_episode,    # 1
    _detect_duplicate_decision,   # 2
    _detect_orphan_decision,      # 3
    _detect_detached_episode,     # 4
    _detect_broken_follows,       # 5
    _detect_type_violation,       # 6
    _detect_isolated_cluster,     # 7
    _detect_stale_abstraction,    # 8
)


def detect_anomalies(subgraph: dict) -> list[dict]:
    """Run all 9 structural detectors on an enriched subgraph.

    Returns a flat list of findings ``{"node", "type", "evidence"}``. A node
    may carry multiple findings (e.g. an orphan decision that is also a
    duplicate). Order follows ``ANOMALY_TYPES``.
    """
    findings: list[dict] = []
    for det in _DETECTORS:
        findings.extend(det(subgraph))
    return findings


def node_label_vectors(subgraph: dict) -> dict[str, list[int]]:
    """Per-node multi-label vector for head training: ``{node_id: [type_idx,
    ...]}`` aligned to ``ANOMALY_TYPES``. Nodes with no finding get an empty
    list (the trainer masks unlabeled nodes — spec §7)."""
    labels: dict[str, list[int]] = {n["id"]: [] for n in subgraph["nodes"]}
    for f in detect_anomalies(subgraph):
        idx = ANOMALY_TYPE_INDEX[f["type"]]
        if idx not in labels[f["node"]]:
            labels[f["node"]].append(idx)
    for nid in labels:
        labels[nid].sort()
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# IDENTITY_DRIFT — review-flag for the Bonsai decider (NOT a head label)
# ═══════════════════════════════════════════════════════════════════════════

def flag_identity_drift(subgraph: dict) -> list[dict]:
    """Over-firing review-flag: an entity whose episodes' topic sets are
    pairwise disjoint (Jaccard 0) — "one node name, two different referents".

    Deliberately over-fires on legitimately multifaceted entities (a person who
    is a coder AND a parent has disjoint topic neighborhoods). That is why this
    is a FLAG-FOR-REVIEW routed to the Bonsai decider (spec §2.5), NOT a head
    label: the only clean signal (type-level ``subClassOf`` incompatibility) is
    too rare to train on, and this heuristic can't decide — Bonsai can, with
    retrieved context.

    Returns findings ``{"node", "type": "identity_drift", "evidence"}``.
    """
    # entity -> set of episode nodes (via has_entity / in_episode). A
    # has_entity edge (ep, has_entity, E:x) and its reverse in_episode edge
    # (E:x, in_episode, ep) describe the SAME entity-episode link, so the
    # episodes must be DEDUPED -- otherwise each episode is counted twice,
    # the topic set list contains the same set twice, and a pair's Jaccard
    # with itself is 1.0 (not 0), so "pairwise disjoint" is never true and
    # the flag could never fire on real (bidirectional) data. The training
    # path only saw this masked because the injector plants drift with a
    # controlled edge set; deploy runs over the real bidirectional graph.
    node_ids = {n["id"] for n in subgraph["nodes"]}
    ent_eps: dict[str, set[str]] = {}
    for e in subgraph["edges"]:
        if e["predicate"] == "has_entity" and e["subject"] in node_ids:
            ent_eps.setdefault(e["object"], set()).add(e["subject"])
        elif e["predicate"] == "in_episode" and e["object"] in node_ids:
            # in_episode: (entity, in_episode, episode) — the reverse orientation
            ent_eps.setdefault(e["subject"], set()).add(e["object"])

    # episode -> set of topics (via has_topic).
    ep_topics: dict[str, set[str]] = {}
    for e in subgraph["edges"]:
        if e["predicate"] == "has_topic" and e["subject"] in node_ids:
            ep_topics.setdefault(e["subject"], set()).add(e["object"])

    findings = []
    for ent, eps in ent_eps.items():
        if len(eps) < 2:
            continue
        topic_sets = [ep_topics.get(ep, set()) for ep in eps]
        # Pairwise disjoint among the non-empty topic sets.
        nonempty = [ts for ts in topic_sets if ts]
        if len(nonempty) < 2:
            continue
        all_disjoint = all(
            _jaccard(nonempty[i], nonempty[j]) == 0.0
            for i in range(len(nonempty)) for j in range(i + 1, len(nonempty))
        )
        if all_disjoint:
            findings.append({
                "node": ent, "type": IDENTITY_DRIFT_FLAG,
                "evidence": f"disjoint topic neighborhoods across {len(eps)} episodes",
            })
    return findings


__all__ = [
    "ANOMALY_TYPES", "ANOMALY_TYPE_INDEX", "IDENTITY_DRIFT_FLAG",
    "LINK_PREDICATES", "DUPLICATE_JACCARD_THRESHOLD",
    "enrich_subgraph", "detect_anomalies", "node_label_vectors",
    "flag_identity_drift",
]