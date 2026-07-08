"""Corruption injector for the GNN anomaly head's training labels (Task 3).

The DialogSum corpus is anomaly-free by construction, so the anomaly head
can't be trained on it (a multi-label BCE head would collapse to "predict 0").
``anomaly_rules.py`` provides the pure rule detectors; THIS module corrupts a
clean enriched subgraph to plant each of the 9 anomaly types, then the rules
label the corruption deterministically — the **closed loop**:

    inject → rule-detect recovers exactly what was planted

Zero Oracle calls for the head (spec §2 / §3). The head learns the structural
signature of each corruption; at deploy the rule detector is the ground-truth
backstop and the head is a cheap pre-filter.

Each ``_inject_*`` helper works on a deep-copied enriched subgraph dict (the
same contract ``anomaly_rules`` consumes: ``{center, nodes, edges}`` with
optional ``summary``/``text`` on nodes and literal objects on data edges). It
mutates the copy and appends ``{node, type}`` ground-truth records to
``planted``. Injections are designed to be **surgical** — they plant one
anomaly without creating collateral labels where possible, so the round-trip
test can assert both recall (every planted label is recovered) and, for the
surgical types, precision (no spurious labels on untouched nodes).

Deterministic via a seeded ``random.Random`` (reproducible dev runs).
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from .anomaly_rules import ANOMALY_TYPES, LINK_PREDICATES, detect_anomalies

# A planted-anomaly ground-truth record: the (node, type) the injector intends
# the rule detector to recover. ``type`` is one of ``ANOMALY_TYPES``.


def _nodes_of(subgraph: dict, prefix: str) -> list[dict]:
    """Nodes whose id starts with ``prefix`` (e.g. ``"ep_"``, ``"D:"``)."""
    return [n for n in subgraph["nodes"] if n["id"].startswith(prefix)]


def _edges_with(subgraph: dict, predicate: str) -> list[dict]:
    """All edges with the given predicate (returns the live list refs)."""
    return [e for e in subgraph["edges"] if e["predicate"] == predicate]


def _add_node(subgraph: dict, nid: str, **fields) -> dict:
    """Append a node dict (type inferred by ``anomaly_rules``-style prefix)."""
    for prefix, typ in (
        ("ep_", "episode"), ("E:", "entity"), ("T:", "topic"),
        ("A:", "tone"), ("D:", "decision"), ("S:", "session"),
        ("U:", "user"), ("M:", "semantic_memory"),
    ):
        if nid.startswith(prefix):
            node = {"id": nid, "type": typ, "depth": 9, **fields}
            break
    else:
        node = {"id": nid, "type": "unknown", "depth": 9, **fields}
    subgraph["nodes"].append(node)
    return node


def _add_edge(subgraph: dict, s: str, p: str, o: str) -> None:
    """Append an edge (no dedup — the detector is idempotent over dup edges)."""
    subgraph["edges"].append({"subject": s, "predicate": p, "object": o})


def _remove_edge(subgraph: dict, s: str, p: str, o: str) -> None:
    """Remove the first ``(s, p, o)`` edge (the one the injector planted on)."""
    for i, e in enumerate(subgraph["edges"]):
        if e["subject"] == s and e["predicate"] == p and e["object"] == o:
            del subgraph["edges"][i]
            return


# ═══════════════════════════════════════════════════════════════════════════
# Per-type injectors
# ═══════════════════════════════════════════════════════════════════════════

def _inject_contradictory_state(sub, rng, planted) -> None:
    """Plant two distinct live ``state`` values on one entity."""
    ents = _nodes_of(sub, "E:")
    if not ents:
        # Fall back to any non-episode node; the rule keys by subject.
        ents = [n for n in sub["nodes"] if not n["id"].startswith("ep_")]
    if not ents:
        return
    n = ents[rng.randrange(len(ents))]
    _add_edge(sub, n["id"], "state", "alive")
    _add_edge(sub, n["id"], "state", "dead")
    planted.append({"node": n["id"], "type": "contradictory_state"})


def _inject_duplicate_episode(sub, rng, planted) -> None:
    """Clone an episode (same summary, mirrored neighborhood) → duplicate."""
    eps = [n for n in _nodes_of(sub, "ep_") if n.get("summary")]
    if not eps:
        return
    orig = eps[rng.randrange(len(eps))]
    clone_id = f"{orig['id']}_dup"
    if any(n["id"] == clone_id for n in sub["nodes"]):
        return
    clone = {"id": clone_id, "type": "episode", "depth": 9,
             "summary": orig["summary"]}
    sub["nodes"].append(clone)
    # Mirror orig's link edges so the clone is connected (not detached / not
    # an isolated single node) — isolating duplicate_episode as the only label.
    for e in list(sub["edges"]):
        if e["subject"] == orig["id"] and e["predicate"] in LINK_PREDICATES:
            _add_edge(sub, clone_id, e["predicate"], e["object"])
        elif e["object"] == orig["id"] and e["predicate"] in LINK_PREDICATES:
            _add_edge(sub, e["subject"], e["predicate"], clone_id)
    planted.append({"node": orig["id"], "type": "duplicate_episode"})
    planted.append({"node": clone_id, "type": "duplicate_episode"})


def _inject_duplicate_decision(sub, rng, planted) -> None:
    """Clone a decision (same text, linked to the same episode) → duplicate."""
    decs = _nodes_of(sub, "D:")
    if not decs:
        return
    orig = decs[rng.randrange(len(decs))]
    text = orig.get("text") or orig["id"][2:]
    clone_id = f"{orig['id']}_dup"
    if any(n["id"] == clone_id for n in sub["nodes"]):
        return
    _add_node(sub, clone_id, text=text)
    # Link the clone via has_decision from every episode that links the orig,
    # so the clone isn't an orphan — isolating duplicate_decision.
    link_eps = [e["subject"] for e in _edges_with(sub, "has_decision")
               if e["object"] == orig["id"]]
    if link_eps:
        for ep in link_eps:
            _add_edge(sub, ep, "has_decision", clone_id)
    else:
        # No existing link to mirror — link to any episode so the clone isn't
        # an orphan (keeps the duplicate label clean).
        eps = _nodes_of(sub, "ep_")
        if eps:
            _add_edge(sub, eps[0]["id"], "has_decision", clone_id)
    planted.append({"node": orig["id"], "type": "duplicate_decision"})
    planted.append({"node": clone_id, "type": "duplicate_decision"})


def _inject_orphan_decision(sub, rng, planted) -> None:
    """Strip a decision's link edges → degree 0 → orphan_decision."""
    decs = _nodes_of(sub, "D:")
    if not decs:
        return
    n = decs[rng.randrange(len(decs))]
    # Remove every incident link-predicate edge on this decision.
    sub["edges"] = [e for e in sub["edges"]
                    if not ((e["subject"] == n["id"] or e["object"] == n["id"])
                            and e["predicate"] in LINK_PREDICATES)]
    planted.append({"node": n["id"], "type": "orphan_decision"})


def _inject_detached_episode(sub, rng, planted) -> None:
    """Strip an episode's link edges → degree 0 → detached_episode.

    Picks an episode that is NOT the center (the center stays the focal point).
    """
    eps = [n for n in _nodes_of(sub, "ep_") if n["id"] != sub.get("center")]
    if not eps:
        return
    n = eps[rng.randrange(len(eps))]
    sub["edges"] = [e for e in sub["edges"]
                    if not ((e["subject"] == n["id"] or e["object"] == n["id"])
                            and e["predicate"] in LINK_PREDICATES)]
    planted.append({"node": n["id"], "type": "detached_episode"})


def _inject_broken_follows(sub, rng, planted) -> None:
    """Rewire a ``follows`` edge to a non-existent target → broken_follows."""
    fedges = _edges_with(sub, "follows")
    if not fedges:
        return
    e = fedges[rng.randrange(len(fedges))]
    dead = "ep_999999"
    _remove_edge(sub, e["subject"], "follows", e["object"])
    _add_edge(sub, e["subject"], "follows", dead)
    planted.append({"node": e["subject"], "type": "broken_follows"})


def _inject_type_violation(sub, rng, planted) -> None:
    """Insert a wrong-domain edge ``(E:X, has_decision, D:Y)`` → type_violation
    on the entity subject (``has_decision`` domain is Episode)."""
    ents = _nodes_of(sub, "E:")
    decs = _nodes_of(sub, "D:")
    if not ents or not decs:
        return
    s = ents[rng.randrange(len(ents))]["id"]
    o = decs[rng.randrange(len(decs))]["id"]
    _add_edge(sub, s, "has_decision", o)
    planted.append({"node": s, "type": "type_violation"})


def _inject_isolated_cluster(sub, rng, planted) -> None:
    """Add a detached 3-node component (ep + 2 entities) → isolated_cluster
    on all three. New nodes only — the existing graph is untouched, so no
    collateral labels."""
    ep = f"ep_iso_{rng.randrange(10**6):06d}"
    e1, e2 = f"E:iso_{rng.randrange(10**6):06d}", f"E:iso2_{rng.randrange(10**6):06d}"
    _add_node(sub, ep)            # no summary → not a duplicate candidate
    _add_node(sub, e1)
    _add_node(sub, e2)
    _add_edge(sub, ep, "has_entity", e1)
    _add_edge(sub, ep, "has_entity", e2)
    for nid in (ep, e1, e2):
        planted.append({"node": nid, "type": "isolated_cluster"})


def _inject_stale_abstraction(sub, rng, planted) -> None:
    """Add an ``M:`` node whose ``abstracts`` target is dead → stale_abstraction."""
    mid = f"M:000{rng.randrange(1, 10)}"
    if any(n["id"] == mid for n in sub["nodes"]):
        mid = f"M:000{rng.randrange(10, 99)}"
    _add_node(sub, mid)
    _add_edge(sub, mid, "abstracts", "ep_999999")  # dead target
    planted.append({"node": mid, "type": "stale_abstraction"})


_INJECTORS = {
    "contradictory_state": _inject_contradictory_state,
    "duplicate_episode": _inject_duplicate_episode,
    "duplicate_decision": _inject_duplicate_decision,
    "orphan_decision": _inject_orphan_decision,
    "detached_episode": _inject_detached_episode,
    "broken_follows": _inject_broken_follows,
    "type_violation": _inject_type_violation,
    "isolated_cluster": _inject_isolated_cluster,
    "stale_abstraction": _inject_stale_abstraction,
}


def inject_anomalies(
    subgraph: dict,
    *,
    seed: int = 0,
    types: Optional[list[str]] = None,
) -> tuple[dict, list[dict]]:
    """Corrupt a clean enriched subgraph, planting one of each requested type.

    Returns ``(corrupted_subgraph, planted)`` where ``planted`` is the ground-
    truth list of ``{node, type}`` records the rule detector should recover.
    ``types=None`` injects all 9 ``ANOMALY_TYPES``; ``types=[]`` injects NONE
    (a clean-subgraph true-negative record — used by the generator's random-
    subset policy, spec §2.5). An explicit non-empty list injects exactly those
    types. A type whose precondition isn't met (e.g. no episode to duplicate) is
    silently skipped — its record is simply absent from ``planted``.

    The input subgraph is NOT mutated (a deep copy is corrupted and returned).
    """
    rng = random.Random(seed)
    corrupted = copy.deepcopy(subgraph)
    planted: list[dict] = []
    inject_types = list(ANOMALY_TYPES) if types is None else types
    for t in inject_types:
        injector = _INJECTORS.get(t)
        if injector is not None:
            injector(corrupted, rng, planted)
    return corrupted, planted


def round_trip(subgraph: dict, *, seed: int = 0) -> dict:
    """Inject then rule-detect EACH type in isolation; return a recall report.

    Each type is planted on a FRESH deep copy of the clean subgraph (no cross-
    type interactions) so recall is a clean per-injector/per-detector check.
    Used by the training pipeline to self-check that the rules recover what
    was planted (spec §8: inject → rule-detect recovers exactly what was
    planted). ``recall`` = fraction of planted ``(node, type)`` records
    recovered across all isolated injections; ``missed`` is the list NOT
    recovered (a rule-detector bug signal).

    (Production training calls ``inject_anomalies`` directly with a random
    SUBSET of types per subgraph — multiple types in one copy can interact,
    e.g. a ``type_violation`` edge re-linking an orphaned node — which is fine
    for training labels but not for this isolated unit check.)
    """
    all_planted: list[tuple[str, str]] = []
    all_missed: list[tuple[str, str]] = []
    for i, t in enumerate(list(ANOMALY_TYPES)):
        corrupted, planted = inject_anomalies(subgraph, seed=seed + i, types=[t])
        detected = {(f["node"], f["type"]) for f in detect_anomalies(corrupted)}
        for p in planted:
            pair = (p["node"], p["type"])
            all_planted.append(pair)
            if pair not in detected:
                all_missed.append(pair)
    return {
        "planted": all_planted,
        "missed": all_missed,
        "recall": (len(all_planted) - len(all_missed)) / max(1, len(all_planted)),
    }


__all__ = ["inject_anomalies", "round_trip"]