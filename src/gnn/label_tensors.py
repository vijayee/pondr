"""Pure label-record -> head-target tensor builders (Phase 3a Task 4a).

Each function maps a freshly-loaded PyG ``Data`` (from
``graph_loader.data_from_subgraph``) plus one label record's ``labels`` dict to
the target tensor(s) a head trains against. Pure: no store, no model — unit-
testable in isolation. A head with no usable labels for a subgraph returns a
sentinel (``None`` / all-False mask) so the trainer SKIPS that head's loss on
that example honestly — no fabrication, no silent "label = 0 everywhere" that
would teach the wrong thing.

Endpoint resolution: the link-pred / ontology labels carry ``subject``/
``object`` / ``child`` / ``parent`` that are usually FULL node ids (the sharded
generator shows the Oracle full ids in the candidate-pair prompt) but may be
bare names (the radius-1 one-call path, or Oracle paraphrase). We resolve
against ``data.node_id`` by exact id first, then by stripped-prefix bare-name
match; unresolved endpoints are SKIPPED and counted (the caller logs the total
once per run — no silent truncation).

Negative sampling: the link-pred record carries Oracle ``negative_edges``;
when it doesn't (old positive-only PoC data), we sample same-kind non-edge
negatives IN CODE so the BCE head doesn't collapse to "predict 1". The ontology
head is a TWO-encoder pair classifier (entity emb from the episode backbone +
class emb from the taxonomy encoder over the live class DAG), so
``ontology_target`` unifies ``suggested_edges`` (child=entity, parent=class) and
``misclassified`` (entity, suggested_class) into ``(entity_row, class_row)``
positives and samples ``(entity, other class)`` negatives in-code (seeded) --
an entity not labeled with class X is a structural true negative. Both samplers
are seeded -> deterministic, reproducible runs.
"""

from __future__ import annotations

import random
from collections import namedtuple
from typing import Optional

import torch

from .anomaly_rules import ANOMALY_TYPES
from .features import infer_kind
from .sharded_labeling import _salience_value

# A scoreable pair-set result for the link-pred and ontology heads.
# ``edge_index``/``labels`` are None when no usable pairs survived resolution
# (the trainer skips that head's loss on this example). ``skipped`` is always
# an int — the count of label endpoints that didn't resolve to subgraph nodes.
_PairTarget = namedtuple("_PairTarget", ["edge_index", "labels", "skipped"])

# Prefixes whose strip yields a "bare name" (entity/topic/...). Episodes
# (``ep_``) have no bare-name form — their id IS the name — so they stay whole.
_PREFIXES: tuple[str, ...] = ("E:", "T:", "A:", "D:", "S:", "U:", "M:")


def _strip_prefix(node_id: str) -> str:
    """Strip a kind prefix to get the bare name (episodes keep their full id)."""
    for p in _PREFIXES:
        if node_id.startswith(p):
            return node_id[len(p):]
    return node_id


def _build_name_index(node_ids: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """``(id->row, bare-name->first-row)`` for endpoint resolution."""
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    name_to_idx: dict[str, int] = {}
    for i, nid in enumerate(node_ids):
        bare = _strip_prefix(nid)
        if bare and bare not in name_to_idx:
            name_to_idx[bare] = i
    return id_to_idx, name_to_idx


def _resolve(name: Optional[str], id_to_idx: dict[str, int],
             name_to_idx: dict[str, int]) -> Optional[int]:
    """Map a label endpoint (full id OR bare name) to a subgraph row, or None."""
    if not name:
        return None
    if name in id_to_idx:
        return id_to_idx[name]
    if name in name_to_idx:               # label gave a bare name ("Alice")
        return name_to_idx[name]
    bare = _strip_prefix(name)
    if bare != name and bare in name_to_idx:
        return name_to_idx[bare]
    return None


def salience_target(data, labels: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node salience target + labeled-mask from ``labels["node_scores"]``.

    ``node_scores`` keys are full node ids (the salience shard prompt shows the
    Oracle full ids); values are ``{"salience": x}`` or a bare number
    (``_salience_value`` handles both). ``mask`` is True only on nodes the
    Oracle actually scored — the trainer computes MSE on ``logits[mask]`` only,
    so unscored nodes don't get taught as "salience 0" (the partial-label
    honesty rule, spec §7). Returns ``(target[N] float32, mask[N] bool)``; an
    all-False mask means "no salience labels this subgraph" (trainer skips).
    """
    node_scores = (labels or {}).get("node_scores") or {}
    ids = data.node_id
    n = len(ids)
    target = torch.zeros(n, dtype=torch.float32)
    mask = torch.zeros(n, dtype=torch.bool)
    for i, nid in enumerate(ids):
        v = _salience_value(node_scores.get(nid))
        if v is not None:
            target[i] = v
            mask[i] = True
    return target, mask


def _existing_pairs(data) -> set[tuple[int, int]]:
    """Both orientations of every edge in ``data.edge_index`` (for exclusion)."""
    pairs: set[tuple[int, int]] = set()
    ei = data.edge_index
    if ei.shape[1] == 0:
        return pairs
    srcs, dsts = ei[0].tolist(), ei[1].tolist()
    for s, o in zip(srcs, dsts):
        pairs.add((s, o))
        pairs.add((o, s))
    return pairs


def _sample_pair_negatives(
    data,
    n: int,
    seed: int,
    exclude: set[tuple[int, int]],
    candidate_kinds: tuple[str, ...],
    same_kind: bool,
) -> list[tuple[int, int]]:
    """Seeded in-code negative-pair sampler (link-pred fallback + ontology).

    Picks pairs among the subgraph's nodes of ``candidate_kinds`` that are NOT
    in ``exclude`` (positives) and NOT an existing edge, so a "negative" is a
    plausible-but-absent pair. ``same_kind`` constrains to same-kind pairs
    (link-pred mirrors the generator's same-kind candidate sampler); ontology
    allows cross-kind E:/T: pairs (``subClassOf`` can relate either). Deterministic
    via ``seed`` -> reproducible. Returns up to ``n`` pairs (fewer if the
    candidate pool is small — honest, not padded with junk).
    """
    ids = data.node_id
    if len(ids) < 2 or n <= 0:
        return []
    kind = [infer_kind(nid) for nid in ids]
    cand_idx = [i for i, k in enumerate(kind) if k in candidate_kinds]
    if len(cand_idx) < 2:
        return []
    existing = _existing_pairs(data)
    rng = random.Random(seed)
    out: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n * 40
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        a = rng.choice(cand_idx)
        b = rng.choice(cand_idx)
        if a == b:
            continue
        if same_kind and kind[a] != kind[b]:
            continue
        if (a, b) in exclude or (b, a) in exclude:
            continue
        if (a, b) in existing or (b, a) in existing:
            continue
        exclude.add((a, b))
        exclude.add((b, a))
        out.append((a, b))
    return out


def linkpred_pairs(data, labels: dict, *, seed: int = 0) -> _PairTarget:
    """Link-prediction target edges from ``predicted_edges`` + ``negative_edges``.

    Positives (label 1) = ``predicted_edges``; negatives (label 0) =
    ``negative_edges``. Endpoints resolve via ``_resolve`` (full id then bare
    name); unresolved or self-loops are skipped + counted. If the record has NO
    usable Oracle negatives, same-kind non-edge negatives are sampled in-code
    (seeded) so the BCE head can't collapse to "predict 1". Returns a
    ``_PairTarget``; ``edge_index`` is None when zero usable pairs survived.
    """
    labels = labels or {}
    id_to_idx, name_to_idx = _build_name_index(data.node_id)
    src: list[int] = []
    dst: list[int] = []
    lbl: list[float] = []
    skipped = 0

    for item in labels.get("predicted_edges") or []:
        s = _resolve(item.get("subject"), id_to_idx, name_to_idx)
        o = _resolve(item.get("object"), id_to_idx, name_to_idx)
        if s is None or o is None or s == o:
            skipped += 1
            continue
        src.append(s); dst.append(o); lbl.append(1.0)

    neg_items = labels.get("negative_edges") or []
    n_neg_resolved = 0
    for item in neg_items:
        s = _resolve(item.get("subject"), id_to_idx, name_to_idx)
        o = _resolve(item.get("object"), id_to_idx, name_to_idx)
        if s is None or o is None or s == o:
            skipped += 1
            continue
        src.append(s); dst.append(o); lbl.append(0.0)
        n_neg_resolved += 1

    # No USABLE negatives (Oracle provided none, OR every provided negative failed
    # endpoint resolution) -> sample same-kind non-edges in-code so BCE can't
    # collapse to "predict 1". Only falls back when there are positives to
    # balance; a positives-only record with no resolvable negatives of its own
    # is the collapse risk this guards.
    if n_neg_resolved == 0 and src:
        exclude = {(s, o) for s, o in zip(src, dst)} | {(o, s) for s, o in zip(src, dst)}
        kinds = tuple(set(infer_kind(nid) for nid in data.node_id))
        for s, o in _sample_pair_negatives(data, n=len(src), seed=seed,
                                           exclude=exclude, candidate_kinds=kinds,
                                           same_kind=True):
            src.append(s); dst.append(o); lbl.append(0.0)

    if not src:
        return _PairTarget(None, None, skipped)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    labels_t = torch.tensor(lbl, dtype=torch.float32)
    return _PairTarget(edge_index, labels_t, skipped)


def class_vocab(labels_dir) -> list[str]:
    """Distinct class names seen across ``ontology_labels.jsonl``.

    Scans every record's ``suggested_edges[].parent`` and
    ``misclassified[].suggested_class`` and returns the sorted distinct class
    names. These are the train-time candidate classes -- the set the labels
    actually exercise. (Deploy can score an entity against the FULL live class
    DAG, including classes not seen during labeling -- the taxonomy encoder
    generalizes to them via their parent edges; ``class_vocab`` only bounds the
    labeled training signal.) A missing file -> empty list (a head whose labels
    weren't regenerated is just never trainable -- honest, not an error).
    """
    from pathlib import Path
    import json as _json

    p = Path(labels_dir) / "ontology_labels.jsonl"
    names: set[str] = set()
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            labels = rec.get("labels") or {}
            for item in labels.get("suggested_edges") or []:
                name = item.get("parent")
                if name:
                    names.add(name)
            for item in labels.get("misclassified") or []:
                name = item.get("suggested_class")
                if name:
                    names.add(name)
    return sorted(names)


def _sample_ontology_negatives(
    data,
    class_rows: list[int],
    n: int,
    seed: int,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Seeded in-code sampler of (entity_row, other_class_row) negative pairs.

    An entity not labeled with class X is a true negative for the (entity, X)
    pair -- structural and free (no Oracle negatives needed). Picks pairs among
    the subgraph's E:/T: nodes and the class DAG's rows that are NOT in
    ``exclude`` (the positives). Deterministic via ``seed`` -> reproducible.
    Returns up to ``n`` pairs (fewer if the candidate pool is small -- honest,
    not padded with junk).
    """
    if n <= 0 or not class_rows:
        return []
    ids = data.node_id
    ent_rows = [i for i, nid in enumerate(ids)
                if nid.startswith("E:") or nid.startswith("T:")]
    if not ent_rows:
        return []
    rng = random.Random(seed)
    out: list[tuple[int, int]] = []
    out_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = n * 40
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        e = rng.choice(ent_rows)
        c = rng.choice(class_rows)
        pair = (e, c)
        if pair in exclude or pair in out_set:
            continue
        out_set.add(pair)
        out.append(pair)
    return out


def ontology_target(
    data,
    labels: dict,
    class_index: dict[str, int],
    *,
    seed: int = 0,
) -> _PairTarget:
    """Entity->class typing target pairs from ``suggested_edges`` + ``misclassified``.

    Both label kinds are entity->class typing labels (``suggested_edges``:
    child=entity, parent=class; ``misclassified``: entity, suggested_class) --
    unify them into ``(entity_row, class_row)`` positive pairs. The entity
    resolves via ``data.node_id`` (full id then bare name, via ``_resolve`` --
    entities ARE in the episode subgraph via ``in_episode``); the class resolves
    via ``class_index`` (class NAME -> taxonomy-encoder row, built by the trainer
    from the live class DAG). Unresolved entities or classes not in the live DAG
    are SKIPPED and counted (honest, not fabricated into a pair from a string).

    Negatives (label 0) are sampled in-code over ``(entity_row, other_class_row)``
    pairs not in the positives (seeded, reproducible) so the pair-classifier BCE
    can't collapse to "predict 1". Returns a ``_PairTarget``; ``edge_index`` is
    ``[2, P]`` with row 0 = entity row in ``data.node_id``, row 1 = class row in
    the taxonomy encoder. ``edge_index`` is None when zero usable pairs survived
    (no resolvable positives AND no sampleable negatives).
    """
    labels = labels or {}
    id_to_idx, name_to_idx = _build_name_index(data.node_id)
    ent_rows: list[int] = []
    cls_rows: list[int] = []
    lbl: list[float] = []
    skipped = 0
    seen: set[tuple[int, int]] = set()

    def _add(entity_name, class_name) -> None:
        nonlocal skipped
        if not entity_name or not class_name:
            skipped += 1
            return
        e = _resolve(entity_name, id_to_idx, name_to_idx)
        if e is None:
            skipped += 1
            return
        c = class_index.get(class_name)
        if c is None:
            # Class not in the live DAG (e.g. an Oracle-invented name that
            # hasn't been promoted) -- can't form a scoreable pair: skip+count.
            skipped += 1
            return
        if (e, c) in seen:
            return  # dedupe the same entity->class pair across the two kinds
        seen.add((e, c))
        ent_rows.append(e)
        cls_rows.append(c)
        lbl.append(1.0)

    for item in labels.get("suggested_edges") or []:
        _add(item.get("child"), item.get("parent"))
    for item in labels.get("misclassified") or []:
        _add(item.get("entity"), item.get("suggested_class"))

    n_pos = len(ent_rows)
    negatives = _sample_ontology_negatives(
        data, list(set(class_index.values())), n=n_pos, seed=seed, exclude=seen,
    )
    for e, c in negatives:
        ent_rows.append(e)
        cls_rows.append(c)
        lbl.append(0.0)

    if not ent_rows:
        return _PairTarget(None, None, skipped)
    edge_index = torch.tensor([ent_rows, cls_rows], dtype=torch.long)
    labels_t = torch.tensor(lbl, dtype=torch.float32)
    return _PairTarget(edge_index, labels_t, skipped)


def anomaly_target(data, labels: dict) -> torch.Tensor:
    """Per-node multi-label anomaly target ``[N, len(ANOMALY_TYPES)]`` float32.

    Built from ``labels["node_labels"]`` (``{node_id: [type_idx, ...]}``,
    aligned to ``ANOMALY_TYPES`` — the generator stores this from
    ``anomaly_rules.node_label_vectors`` over the corrupted subgraph). Nodes
    with an empty list stay all-zero — TRUE NEGATIVES, a valid target (NOT
    masked: the head must learn to predict 0 on clean nodes). A clean subgraph
    (``types=[]`` -> no injection -> empty ``node_labels``) is an all-zero
    true-negative example, KEPT not skipped. Always returns a tensor (never
    None) — there is no "no anomaly labels" case; absence of anomalies IS the
    label.
    """
    n = len(data.node_id)
    n_types = len(ANOMALY_TYPES)
    target = torch.zeros(n, n_types, dtype=torch.float32)
    node_labels = (labels or {}).get("node_labels") or {}
    for i, nid in enumerate(data.node_id):
        for tidx in node_labels.get(nid, []):
            if 0 <= int(tidx) < n_types:
                target[i, int(tidx)] = 1.0
    return target


def split_centers(
    subgraph_ids: list[str], val_fraction: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Seeded shuffle split of subgraph centers into (train, val).

    Deterministic via ``seed`` (reproducible runs). Disjoint, covers all ids.
    Guarantees at least 1 val center when there are >=2 centers (so a tiny dev
    store still exercises the val path); with 1 center, val is empty (the
    trainer records ``val_metric=None`` honestly for every head).
    """
    ids = sorted(subgraph_ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    if n_val == 0 and len(shuffled) >= 2:
        n_val = 1
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


__all__ = [
    "salience_target", "linkpred_pairs", "ontology_target", "class_vocab",
    "anomaly_target", "split_centers",
]