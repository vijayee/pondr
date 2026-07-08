"""Sharded Oracle labeling for the GNN heads (Phase 3a Task 3).

Decouples subgraph size from one Oracle call's output budget (spec §0 of
``docs/Phase 3a Task 3 - sharded labeling design.md``): the full radius-3
subgraph (≈10K nodes on the dense DialogSum corpus) is kept for what the GNN
trains on, while the *labeling* is sharded across many calls so each call's
output fits in ``oracle_max_tokens``. Recombination merges the shard outputs
back into ONE per-subgraph JSONL record matching the existing label schemas, so
``train_gnn.py`` and ``validators.py`` are unchanged (spec §4).

This module owns the four Oracle-labelable heads' sharding:

- **salience** — node shards; the Oracle scores only the shard's nodes, and edge
  scores are computed in code from endpoint node salience (halves the shard
  count; the head trains on node MSE anyway — ``SalienceHead.loss``).
- **link-pred** — candidate non-edge pair shards (same-kind, non-edge, capped).
- **ontology** — entity/topic pair shards (candidate ``subClassOf`` pairs).
- **cluster** — self-supervised by default; one optional episode-only Oracle call
  for weak supervision (gated by ``--oracle-cluster-supervision`` in the
  generator). No sharding — the episode-only context is small.

The **anomaly** head is Oracle-FREE (spec §2): the corpus is anomaly-free, so
labels come from ``anomaly_injector`` + ``anomaly_rules``. This module's
``build_anomaly_labels`` orchestrates inject → detect → label over an enriched
subgraph and emits the per-subgraph record in the existing ``anomaly_labels``
schema (with extra keys the trainer reads). The structural detectors themselves
live in ``anomaly_rules.py`` (also the deploy backstop).

What lives here, per spec §5: shard construction, local-context prompt builders
(shards carry local context + a one-line global summary, NOT the full 4 MB),
recombination, candidate-pair samplers, partial-label masking helpers.

The flow with the existing ``run_batches`` machinery (spec §5): a shard IS the
``item`` — ``build_prompt(shard, idx)`` renders the local-context prompt,
``to_shard_record(shard, result, idx)`` tags the result with
``(subgraph_id, shard_idx)``, and a ``recombine_*`` pass groups shards by
``subgraph_id`` and merges into the final per-subgraph JSONL record. The on-disk
Oracle cache makes resume free.
"""

from __future__ import annotations

import json
from typing import Optional

from .anomaly_injector import inject_anomalies
from .anomaly_rules import (
    ANOMALY_TYPES,
    _is_node_id,  # canonical node-id check (prefix list must stay in one place)
    detect_anomalies,
    flag_identity_drift,
    node_label_vectors,
)

# ── defaults (spec §1 / §2 / §7) ──
DEFAULT_SHARD_SIZE = 500                 # salience nodes / link-pred pairs per shard
DEFAULT_MAX_CANDIDATE_PAIRS = 500        # cap for link-pred + ontology candidate pairs
MIN_LABELED_FRACTION = 0.5               # drop a subgraph below this salience coverage


# ═══════════════════════════════════════════════════════════════════════════
# Global summary — the one-line global awareness every shard carries
# ═══════════════════════════════════════════════════════════════════════════

def global_summary(subgraph: dict) -> dict:
    """One-line global awareness for the shard prompts (spec §1).

    Total counts + the top shared entities/topics by episode-degree. Cheap to
    compute over the full subgraph once; sent with every shard so the Oracle has
    global awareness without the 4 MB subgraph. Deterministic: ties break by id.
    """
    ent_deg: dict[str, int] = {}
    top_deg: dict[str, int] = {}
    for e in subgraph["edges"]:
        if e["predicate"] == "has_entity":
            ent_deg[e["object"]] = ent_deg.get(e["object"], 0) + 1
        elif e["predicate"] == "has_topic":
            top_deg[e["object"]] = top_deg.get(e["object"], 0) + 1

    def top5(d: dict[str, int]) -> list[dict]:
        return [{"id": k, "episodes": v}
                for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]

    return {
        "total_nodes": len(subgraph["nodes"]),
        "total_edges": len(subgraph["edges"]),
        "top_shared_entities": top5(ent_deg),
        "top_shared_topics": top5(top_deg),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Candidate-pair samplers (deterministic — reproducible dev runs + cache-friendly)
# ═══════════════════════════════════════════════════════════════════════════

def sample_link_pred_candidates(
    subgraph: dict,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> list[tuple[str, str]]:
    """Same-kind non-edge node pairs, deterministic sorted take, capped (mirrors
    ``consolidate._sample_candidate_pairs``). Variety across the corpus comes
    from different subgraphs, not from randomness within one — reproducible dev
    runs and identical prompts on resume (Oracle cache hits).
    """
    node_ids = sorted(n["id"] for n in subgraph["nodes"])
    kind = {n["id"]: n.get("type") for n in subgraph["nodes"]}
    existing: set[tuple[str, str]] = set()
    for e in subgraph["edges"]:
        s, o = e["subject"], e["object"]
        if _is_node_id(s) and _is_node_id(o):
            existing.add((s, o))
            existing.add((o, s))
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(node_ids):
        ka = kind.get(a)
        for b in node_ids[i + 1:]:
            if kind.get(b) != ka:
                continue
            if (a, b) in existing or (b, a) in existing:
                continue
            pairs.append((a, b))
            if len(pairs) >= max_candidate_pairs:
                return pairs
    return pairs


def sample_ontology_candidates(
    subgraph: dict,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> list[tuple[str, str]]:
    """Entity (``E:``) ∪ topic (``T:``) pairs — candidate ``subClassOf``
    children→parents. Deterministic sorted take, capped. Small (entities+topics
    are a fraction of the 10K nodes) → typically 1–2 Oracle shards.
    """
    ets = sorted(n["id"] for n in subgraph["nodes"]
                 if n["id"].startswith("E:") or n["id"].startswith("T:"))
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(ets):
        for b in ets[i + 1:]:
            pairs.append((a, b))
            if len(pairs) >= max_candidate_pairs:
                return pairs
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Shard construction
# ═══════════════════════════════════════════════════════════════════════════

def _induced_edges(subgraph: dict, shard_ids: set[str]) -> list[dict]:
    """Edges among the shard's nodes ∪ the center — both endpoints must be node
    ids in the set (data edges with literal objects aren't "among" nodes, and
    aren't needed for salience scoring)."""
    return [e for e in subgraph["edges"]
            if e["subject"] in shard_ids and _is_node_id(e["object"])
            and e["object"] in shard_ids]


def shard_nodes(subgraph: dict, shard_size: int = DEFAULT_SHARD_SIZE) -> list[dict]:
    """Split a subgraph's nodes into shards of ≤ ``shard_size``.

    Deterministic: nodes are sorted by id before chunking, so re-runs produce
    identical shards (and identical Oracle prompts → cache hits on resume). Each
    shard carries the center, the shard's nodes, the induced edges (both
    endpoints in shard ∪ {center}), and the global summary — everything
    ``build_salience_shard_prompt`` needs.
    """
    center = subgraph["center"]
    nodes = sorted(subgraph["nodes"], key=lambda n: n["id"])
    summary = global_summary(subgraph)
    total_nodes = len(subgraph["nodes"])
    total_edges = len(subgraph["edges"])
    shards: list[dict] = []
    for start in range(0, len(nodes), shard_size):
        chunk = nodes[start:start + shard_size]
        shard_ids = {n["id"] for n in chunk} | {center}
        shards.append({
            "subgraph_id": center,
            "task": "salience",
            "shard_idx": len(shards),
            "center": center,
            "nodes": chunk,
            "edges": _induced_edges(subgraph, shard_ids),
            "global": summary,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        })
    return shards


def shard_pairs(
    subgraph: dict,
    pairs: list[tuple[str, str]],
    task: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> list[dict]:
    """Shard candidate pairs into chunks; each shard carries the pairs + the
    local context (the pair nodes' content + the global summary). Used by both
    link-pred and ontology (``task`` discriminates the prompt builder)."""
    center = subgraph["center"]
    summary = global_summary(subgraph)
    node_map = {n["id"]: n for n in subgraph["nodes"]}
    total_nodes = len(subgraph["nodes"])
    total_edges = len(subgraph["edges"])
    shards: list[dict] = []
    for start in range(0, len(pairs), shard_size):
        chunk = pairs[start:start + shard_size]
        shard_node_ids = sorted({a for a, _ in chunk} | {b for _, b in chunk})
        shard_nodes = [node_map[nid] for nid in shard_node_ids if nid in node_map]
        shards.append({
            "subgraph_id": center,
            "task": task,
            "shard_idx": len(shards),
            "center": center,
            "pairs": chunk,
            "nodes": shard_nodes,
            "global": summary,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        })
    return shards


def build_link_pred_shards(
    subgraph: dict,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> list[dict]:
    """Candidate non-edge pair shards for the link-prediction head."""
    pairs = sample_link_pred_candidates(subgraph, max_candidate_pairs)
    return shard_pairs(subgraph, pairs, "link_prediction", shard_size)


def build_ontology_shards(
    subgraph: dict,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> list[dict]:
    """Entity/topic pair shards for the ontology-refinement head."""
    pairs = sample_ontology_candidates(subgraph, max_candidate_pairs)
    return shard_pairs(subgraph, pairs, "ontology", shard_size)


def episode_only_context(subgraph: dict) -> dict:
    """Episode nodes + their shared entities/topics/timestamps (small — fits in
    one Oracle call). For the optional ``--oracle-cluster-supervision`` weak
    supervision call (spec §2). No sharding needed — the episode-only context is
    small even on a 10K-node radius-3 subgraph (only the episode nodes, a small
    fraction of the whole). Returns ONE shard-shaped dict (``shard_idx`` 0).
    """
    eps = [n for n in subgraph["nodes"] if n.get("type") == "episode"]
    ep_ids = {ep["id"] for ep in eps}
    ent_deg: dict[str, int] = {}
    top_deg: dict[str, int] = {}
    for e in subgraph["edges"]:
        if e["predicate"] == "has_entity" and e["subject"] in ep_ids:
            ent_deg[e["object"]] = ent_deg.get(e["object"], 0) + 1
        elif e["predicate"] == "has_topic" and e["subject"] in ep_ids:
            top_deg[e["object"]] = top_deg.get(e["object"], 0) + 1

    def top(d: dict[str, int], n: int = 20) -> list[str]:
        return [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    return {
        "subgraph_id": subgraph["center"],
        "task": "cluster",
        "shard_idx": 0,
        "center": subgraph["center"],
        "episodes": [{"id": n["id"], "summary": n.get("summary", ""),
                      "timestamp": n.get("timestamp", "")} for n in eps],
        "shared_entities": top(ent_deg),
        "shared_topics": top(top_deg),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Local-context prompt builders (shards carry local context, NOT the full 4 MB)
# ═══════════════════════════════════════════════════════════════════════════

def build_salience_shard_prompt(shard: dict) -> str:
    """Render the local-context salience prompt for one node shard. Instructs
    the Oracle to score ONLY this shard's nodes (no inventing nodes outside the
    set) relative to the center."""
    nodes_json = json.dumps(
        [{"id": n["id"], "type": n.get("type"), "summary": n.get("summary", "")}
         for n in shard["nodes"]],
        ensure_ascii=False,
    )
    edges_json = json.dumps(
        [{"subject": e["subject"], "predicate": e["predicate"], "object": e["object"]}
         for e in shard["edges"]],
        ensure_ascii=False,
    )
    g = shard["global"]
    return f"""You are labeling a memory graph for GNN training.
Score each node by structural importance (0.0-1.0) relative to the center node.

CENTER: {shard["center"]}
GLOBAL SUMMARY: {json.dumps(g, ensure_ascii=False)} (the full subgraph has
{shard["total_nodes"]} nodes / {shard["total_edges"]} edges; this shard covers
{len(shard["nodes"])} of them).

Score ONLY these {len(shard["nodes"])} nodes — do NOT invent nodes outside this set.

SHARD NODES:
{nodes_json}

INDUCED EDGES (among shard nodes + center):
{edges_json}

HIGH salience (>0.7): bridge nodes, decision-bearing episodes, temporal anchors,
nodes with unique information not reachable through other paths.
MEDIUM (0.3-0.7): moderate overlap, part of active but non-anchoring chains.
LOW (<0.3): redundant/peripheral nodes, info available through other paths.

Return ONLY valid JSON — a bare float per node, no reasons, no prose:
{{"node_scores": {{"<node_id>": 0.0}}}}"""


def build_link_pred_shard_prompt(shard: dict) -> str:
    """Local-context link-prediction prompt for one candidate-pair shard."""
    pairs_json = json.dumps(
        [{"subject": a, "object": b} for a, b in shard["pairs"]],
        ensure_ascii=False,
    )
    nodes_json = json.dumps(
        [{"id": n["id"], "type": n.get("type"), "summary": n.get("summary", "")}
         for n in shard["nodes"]],
        ensure_ascii=False,
    )
    g = shard["global"]
    return f"""You are labeling a memory graph for GNN training.
For each candidate node pair, decide if an edge SHOULD exist (positive) or should
NOT (negative), using the local context + global summary to judge.

CENTER: {shard["center"]}
GLOBAL SUMMARY: {json.dumps(g, ensure_ascii=False)}

CANDIDATE PAIRS ({len(shard["pairs"])}):
{pairs_json}

PAIR NODE CONTEXT:
{nodes_json}

POSITIVE edges (predicted_edges): pairs that SHOULD be linked but aren't — shared
context, causal/temporal order, co-occurring entities/topics, implied hierarchy.
NEGATIVE edges (negative_edges): pairs that plausibly COULD share an edge given
type/proximity but should NOT — unrelated domains, no shared context, far apart.

Return ONLY valid JSON — subject/object only, no predicate/confidence/evidence/prose:
{{"predicted_edges": [{{"subject": "...", "object": "..."}}],
 "negative_edges": [{{"subject": "...", "object": "..."}}]}}"""


def build_ontology_shard_prompt(shard: dict, current_ontology_json: str) -> str:
    """Local-context ontology prompt for one entity/topic pair shard."""
    pairs_json = json.dumps(
        [{"child": a, "parent": b} for a, b in shard["pairs"]],
        ensure_ascii=False,
    )
    nodes_json = json.dumps(
        [{"id": n["id"], "type": n.get("type")} for n in shard["nodes"]],
        ensure_ascii=False,
    )
    return f"""You are labeling a memory graph for GNN training.
Suggest missing subClassOf edges among the candidate entity/topic pairs, and flag
misclassified entities.

CURRENT ONTOLOGY:
{current_ontology_json}

CANDIDATE PAIRS ({len(shard["pairs"])}) — child → parent:
{pairs_json}

PAIR NODES:
{nodes_json}

Return ONLY valid JSON — child/parent (and entity/suggested_class) only, no confidence/evidence/prose:
{{"suggested_edges": [{{"child": "...", "parent": "..."}}],
 "misclassified": [{{"entity": "...", "suggested_class": "..."}}]}}"""


def build_cluster_episode_prompt(ctx: dict) -> str:
    """Episode-only cluster prompt for the optional weak-supervision call."""
    eps_json = json.dumps(ctx["episodes"], ensure_ascii=False)
    return f"""You are labeling a memory graph for GNN training.
Identify groups of episodes that should be abstracted into semantic memories.

A valid cluster: shared entities/topics, temporal proximity (within 7 days),
coherent theme.

EPISODES (episode-only context — {len(ctx["episodes"])} of them):
{eps_json}

SHARED ENTITIES: {", ".join(ctx["shared_entities"]) or "(none)"}
SHARED TOPICS: {", ".join(ctx["shared_topics"]) or "(none)"}

Return ONLY valid JSON:
{{"clusters": [{{"name": "...", "episodes": ["ep_..."], "abstracted_summary": "...",
   "coherence_score": 0.0}}]}}"""


# ═══════════════════════════════════════════════════════════════════════════
# run_batches glue + recombination
# ═══════════════════════════════════════════════════════════════════════════

def to_shard_record(shard: dict, result, idx: int) -> dict:
    """The ``run_batches`` ``to_record``: tag a shard's Oracle result with its
    ``(subgraph_id, shard_idx)`` so recombination can group shards back together.
    """
    return {
        "subgraph_id": shard["subgraph_id"],
        "shard_idx": shard["shard_idx"],
        "task": shard["task"],
        "labels": result.response,
        "cost": result.cost,
    }


def group_by_subgraph(shard_records: list[dict]) -> dict[str, list[dict]]:
    """Group shard records by ``subgraph_id``, each group sorted by ``shard_idx``
    so recombination merges shards in the deterministic order they were built."""
    out: dict[str, list[dict]] = {}
    for rec in shard_records:
        out.setdefault(rec["subgraph_id"], []).append(rec)
    for sid in out:
        out[sid].sort(key=lambda r: r["shard_idx"])
    return out


def _salience_value(v) -> Optional[float]:
    """Pull a numeric salience from either ``{"salience": x}`` or a bare number."""
    if isinstance(v, dict):
        s = v.get("salience")
        return float(s) if isinstance(s, (int, float)) else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def recombine_salience(
    shard_records: list[dict],
    subgraphs_by_id: dict[str, dict],
) -> list[dict]:
    """Merge salience shards → per-subgraph labels in the existing schema
    ``{"node_scores": {...}, "edge_scores": {...}}`` (spec §2 / §4).

    - ``node_scores``: union of every shard's ``node_scores`` (a node scored by
      two shards — shouldn't happen given disjoint shards — keeps the first).
    - ``edge_scores``: computed IN CODE from endpoint node salience (mean of the
      two endpoints). Halves the shard count (the head trains on node MSE anyway).
      Edges with an unlabeled endpoint are skipped (partial-label honest).

    Returns one ``{"subgraph_id", "labels", "cost"}`` record per subgraph.
    """
    grouped = group_by_subgraph(shard_records)
    records: list[dict] = []
    for sid, recs in grouped.items():
        node_scores: dict = {}
        cost = 0.0
        for r in recs:
            ns = (r["labels"] or {}).get("node_scores") or {}
            for nid, v in ns.items():
                if nid not in node_scores:
                    node_scores[nid] = v
            cost += r.get("cost", 0.0)
        edge_scores: dict = {}
        sub = subgraphs_by_id.get(sid)
        if sub is not None:
            for e in sub["edges"]:
                s, o = e["subject"], e["object"]
                if not _is_node_id(o):
                    continue  # data edge (literal object): no node endpoint to score
                vs = _salience_value(node_scores.get(s))
                vo = _salience_value(node_scores.get(o))
                if vs is None or vo is None:
                    continue  # partial: one endpoint unlabeled → skip (honest)
                key = f"{s}|{e['predicate']}|{o}"
                edge_scores[key] = {
                    "salience": (vs + vo) / 2.0,
                    "reason": "mean of endpoint node salience",
                }
        records.append({
            "subgraph_id": sid,
            "labels": {"node_scores": node_scores, "edge_scores": edge_scores},
            "cost": cost,
        })
    return records


def recombine_link_pred(shard_records: list[dict]) -> list[dict]:
    """Concat each shard's ``predicted_edges`` + ``negative_edges`` → one record
    per subgraph in the existing ``link_prediction_labels`` schema."""
    grouped = group_by_subgraph(shard_records)
    records: list[dict] = []
    for sid, recs in grouped.items():
        predicted: list = []
        negative: list = []
        cost = 0.0
        for r in recs:
            labels = r["labels"] or {}
            predicted.extend(labels.get("predicted_edges") or [])
            negative.extend(labels.get("negative_edges") or [])
            cost += r.get("cost", 0.0)
        records.append({
            "subgraph_id": sid,
            "labels": {"predicted_edges": predicted, "negative_edges": negative},
            "cost": cost,
        })
    return records


def recombine_ontology(shard_records: list[dict]) -> list[dict]:
    """Concat each shard's ``suggested_edges`` + ``misclassified`` → one record
    per subgraph in the existing ``ontology_labels`` schema."""
    grouped = group_by_subgraph(shard_records)
    records: list[dict] = []
    for sid, recs in grouped.items():
        edges: list = []
        misclass: list = []
        cost = 0.0
        for r in recs:
            labels = r["labels"] or {}
            edges.extend(labels.get("suggested_edges") or [])
            misclass.extend(labels.get("misclassified") or [])
            cost += r.get("cost", 0.0)
        records.append({
            "subgraph_id": sid,
            "labels": {"suggested_edges": edges, "misclassified": misclass},
            "cost": cost,
        })
    return records


def recombine_cluster(shard_records: list[dict]) -> list[dict]:
    """One record per subgraph (single episode-only call); pass ``clusters``
    through in the existing ``cluster_labels`` schema."""
    grouped = group_by_subgraph(shard_records)
    records: list[dict] = []
    for sid, recs in grouped.items():
        clusters: list = []
        cost = 0.0
        for r in recs:
            clusters.extend((r["labels"] or {}).get("clusters") or [])
            cost += r.get("cost", 0.0)
        records.append({
            "subgraph_id": sid,
            "labels": {"clusters": clusters},
            "cost": cost,
        })
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Partial-label masking helpers (spec §7)
# ═══════════════════════════════════════════════════════════════════════════

def salience_coverage(node_scores: dict, subgraph: dict) -> dict:
    """Report salience label coverage for partial-label masking (spec §7).

    The trainer masks unlabeled nodes rather than treating them as salience=0
    (which would silently teach "low salience = everything I couldn't label").
    Returns ``{"labeled", "unlabeled", "fraction"}``. A subgraph below
    ``MIN_LABELED_FRACTION`` is dropped from training (logged) — see
    ``meets_min_labeled_fraction``.
    """
    labeled = set(node_scores)
    all_ids = {n["id"] for n in subgraph["nodes"]}
    unlabeled = sorted(all_ids - labeled)
    frac = len(labeled) / len(all_ids) if all_ids else 1.0
    return {"labeled": sorted(labeled), "unlabeled": unlabeled, "fraction": frac}


def meets_min_labeled_fraction(coverage: dict, min_fraction: float = MIN_LABELED_FRACTION) -> bool:
    """True if a subgraph's salience coverage meets the ``min_labeled_fraction``
    threshold (default 0.5); subgraphs below are dropped from training."""
    return coverage["fraction"] >= min_fraction


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly labels — Oracle-FREE (inject → detect → label), spec §2
# ═══════════════════════════════════════════════════════════════════════════

def build_anomaly_labels(
    enriched_subgraph: dict,
    *,
    seed: int = 0,
    types: Optional[list[str]] = None,
) -> dict:
    """Oracle-free anomaly labels for one subgraph (spec §2): inject corruption(s)
    → detect with the structural rules → emit per-node multi-label vectors
    aligned to ``ANOMALY_TYPES``, plus the IDENTITY_DRIFT review-flags for the
    Bonsai decider. Zero Oracle calls; runs over the full subgraph in code.

    ``types=None`` injects all 9 ``ANOMALY_TYPES`` (a type whose precondition
    isn't met — e.g. ``stale_abstraction`` on a subgraph with no ``M:`` node —
    is silently skipped, so it's simply absent from the labels). The per-subgraph
    RANDOM-SUBSET training policy (spec §2.5: "production calls inject_anomalies
    with a random subset per subgraph") is the GENERATOR's job — it picks a
    seeded subset per subgraph and passes it as ``types=`` here. Keeping this
    wrapper deterministic given ``(subgraph, seed, types)`` is what lets the
    trainer reproduce the corrupted graph from those three values.

    ``enriched_subgraph`` is an ALREADY-enriched subgraph (summaries hydrated +
    data edges surfaced — ``anomaly_rules.enrich_subgraph``); the generator does
    the store-backed enrichment once per subgraph, then this is pure.

    The trainer reproduces the corrupted graph deterministically from
    ``(subgraph_id, seed, types)`` (the injector is seed-deterministic) — so the
    per-node label vector + the corrupted structure are kept in sync without
    serializing a 10K-node corrupted subgraph per JSONL record.

    Returns ``{"subgraph_id", "labels", "cost": 0.0}`` where ``labels`` carries:
    - ``anomalies``: the rule findings (keeps ``validators.validate_gnn`` happy —
      ``GNN_LABEL_KEYS["anomaly"] == {"anomalies"}``).
    - ``planted``: the ground-truth injected records (audit; the Oracle's
      "correct" decision is checkable against these — spec §2.5).
    - ``node_labels``: per-node multi-label vectors for the head trainer
      (``anomaly_rules.node_label_vectors``).
    - ``identity_drift``: the review-flags routed to the Bonsai decider
      (``anomaly_rules.flag_identity_drift``), NOT a head label.
    - ``seed`` / ``types``: so the trainer can reproduce the corrupted graph.
    """
    corrupted, planted = inject_anomalies(enriched_subgraph, seed=seed, types=types)
    findings = detect_anomalies(corrupted)
    node_labels = node_label_vectors(corrupted)
    drift = flag_identity_drift(corrupted)
    return {
        "subgraph_id": enriched_subgraph.get("center"),
        "labels": {
            "anomalies": findings,
            "planted": planted,
            "node_labels": node_labels,
            "identity_drift": drift,
            "seed": seed,
            "types": types,
        },
        "cost": 0.0,
    }


__all__ = [
    # defaults
    "DEFAULT_SHARD_SIZE", "DEFAULT_MAX_CANDIDATE_PAIRS", "MIN_LABELED_FRACTION",
    # shard construction + samplers
    "global_summary",
    "sample_link_pred_candidates", "sample_ontology_candidates",
    "shard_nodes", "shard_pairs",
    "build_link_pred_shards", "build_ontology_shards", "episode_only_context",
    # prompt builders
    "build_salience_shard_prompt", "build_link_pred_shard_prompt",
    "build_ontology_shard_prompt", "build_cluster_episode_prompt",
    # run_batches glue + recombination
    "to_shard_record", "group_by_subgraph",
    "recombine_salience", "recombine_link_pred",
    "recombine_ontology", "recombine_cluster",
    # partial-label masking
    "salience_coverage", "meets_min_labeled_fraction",
    # anomaly (Oracle-free)
    "build_anomaly_labels",
]