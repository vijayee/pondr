"""Nightly dream-state consolidation loop (Phase 3a Task 6).

The loop reads the memory graph, runs the GNN, and consolidates: score →
cluster → abstract → predict → verify → anomaly → ontology → prune. It is
**dry-run by default** (``ConsolidationConfig.dry_run_default = True``); only
``dry_run=False`` (the script's ``--apply``) mutates the store. Bonsai
verification of medium-confidence proposals is via a caller-supplied
``verifier`` callable — when ``None``, proposals in the "propose" band are
recorded as unverified and NOT accepted (honest, not faked).

The loop reuses, by construction:
- ``WaveDBGraphLoader`` (Task 1) for the subgraphs the GNN scores — same loader
  the GNN was trained on, so train/serve skew is zero.
- ``SemanticMemoryWriter`` (Task 5) for abstractions + edge archive.
- The 1d ``OracleClient`` HTTP pattern — a verifier built on it (caller-side)
  validates proposals the same way 1d validated labels.

With an UNTRAINED model (random weights — the dev-slice default when no
checkpoint is supplied), the loop runs end-to-end and produces a shape-correct
report, but the scores are meaningless. The real run loads a trained checkpoint
(Task 4, pod). Nothing here fakes that: the report carries a ``trained`` flag.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from .graph_loader import WaveDBGraphLoader
from .heads import ANOMALY_TYPES
from .model import GNNModel
from .semantic_memory import SemanticMemoryWriter
from .taxonomy_graph import build_taxonomy_data

if TYPE_CHECKING:
    from ..config import ConsolidationConfig
    from ..memory.store import HippocampalStore

log = logging.getLogger(__name__)

# A verifier answers "is this proposal consistent with the memory graph / world
# knowledge?" — True to accept, False to reject. ``proposal`` is a dict the loop
# builds (subject/predicate/object + evidence); the caller's Bonsai-backed
# verifier turns it into a prompt and calls the Oracle/Bonsai endpoint.
Verifier = Callable[[dict], bool]

# Pair-scoring chunk size: the ontology "all" strategy scores up to
# ~1512 entities x 377 classes = ~570k pairs at once. A [570k, 256] concat is a
# ~580 MB transient; chunking to 50k keeps the peak ~50 MB without changing the
# result (the head is stateless across pairs).
_SCORE_CHUNK = 50_000
# Histogram resolution: 100 width-0.01 buckets over [0,1] so the threshold sweep
# is exact at the values that matter (0.05/0.15/0.85 are 0.01-boundaries) and the
# salience cliff in [0.05, 0.10] is resolved. 100 ints/category is ~800 bytes.
_HIST_BINS = 100


def _chunked(seq, n: int):
    """Yield successive ``n``-sized slices of ``seq`` (a list, not a tensor)."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Consolidator:
    """One pass of dream-state consolidation over the memory graph."""

    def __init__(
        self,
        store: "HippocampalStore",
        model: Optional[GNNModel] = None,
        loader: Optional[WaveDBGraphLoader] = None,
        config: Optional["ConsolidationConfig"] = None,
        verifier: Optional[Verifier] = None,
        dry_run: Optional[bool] = None,
        device: str = "cpu",
        allow_untrained_apply: bool = False,
    ) -> None:
        from ..config import Phase3aConfig
        self.store = store
        pa = Phase3aConfig()
        self.cfg = config or pa.consolidation
        self.dry_run = self.cfg.dry_run_default if dry_run is None else dry_run
        self.verifier = verifier
        self.device = device
        # Untrained model is the dev-slice default; the real run passes a loaded
        # checkpoint. ``trained`` records which so the report is honest.
        self.model = model if model is not None else GNNModel(
            hidden_dim=128, num_heads=4, num_layers=3,
            predicate_vocab_size=32, num_clusters=16,
        )
        self.model.to(self.device)
        self.model.eval()
        self.trained = model is not None
        self.loader = loader or WaveDBGraphLoader(store, radius=3)
        self.writer = SemanticMemoryWriter(store)
        self.allow_untrained_apply = allow_untrained_apply
        # Candidate budgets live in self.cfg (linkpred_candidate_budget /
        # ontology_*). See ConsolidationConfig for the per-knob rationale.
        # Lazy-cached class embeddings for the ontology head (two-encoder pair
        # classifier: entity emb from the backbone, class emb from the taxonomy
        # encoder over the live class DAG). Built on first _step_ontology call.
        # The class DAG is fixed for the run; the (eval, no_grad) class_emb is
        # reused across subgraphs. ``_class_names`` aligns class_emb rows to
        # bare class names for the recorded proposals.
        self._class_emb: Optional[torch.Tensor] = None
        self._class_names: Optional[list[str]] = None

    # ── main pass ──

    def run(
        self,
        centers: Optional[list[str]] = None,
        limit: Optional[int] = None,
        wm_episode_ids: Optional[set[str]] = None,
    ) -> dict:
        """Run one consolidation pass; return a report dict.

        ``centers``: explicit subgraph centers (episode ids). If None, the
        loader's episode centers are used (optionally WM-prioritized: when
        ``cfg.wm_prioritized`` and ``wm_episode_ids`` are given, centers whose
        episodes are WM-resident come first). ``limit`` caps the number of
        subgraphs scored (the full corpus is the real run; the dev slice uses a
        small limit).
        """
        if centers is None:
            centers = self.loader.episode_centers(limit=limit)
        if self.cfg.wm_prioritized and wm_episode_ids:
            centers = self._wm_first(centers, wm_episode_ids)
        if limit is not None:
            centers = centers[:limit]

        report: dict = {
            "dry_run": self.dry_run, "trained": self.trained,
            "subgraphs_scored": 0, "abstracts": [], "edges_proposed": [],
            "edges_accepted": [], "edges_unverified": [], "anomalies": [],
            "ontology_proposed": [], "pruned": [], "verifier_calls": 0,
            "verifier_accepted": 0,
            # Binned score histograms (100 width-0.01 buckets, [0,1]) so a reader
            # can sweep accept/bonsai/prune thresholds from ONE run without
            # re-running (exact at 0.01-boundaries like 0.05/0.15/0.85).
            # ontology/linkpred hold the raw sigmoid head scores;
            # salience_endpoint holds the per-edge MAX endpoint salience (the
            # binding constraint for the "prune if BOTH endpoints < thr" rule:
            # prune iff this < thr). Buckets are int counts; tiny vs the
            # per-proposal lists.
            "score_distributions": {
                "ontology": [0] * _HIST_BINS, "linkpred": [0] * _HIST_BINS,
                "salience_endpoint": [0] * _HIST_BINS,
            },
        }

        for center in centers:
            data = self.loader.load(center)
            if data.x.shape[0] < 2:
                continue
            report["subgraphs_scored"] += 1
            with torch.no_grad():
                out = self.model(data)
            self._step_cluster(data, out, report)
            self._step_predict(data, out, report)
            self._step_anomaly_bounded(data, out, center, report)
            self._step_ontology(data, out, report)
            self._step_prune(data, out, report)

        # Apply phase: only when not dry-run. (Per-subgraph steps above only
        # RECORD proposals; mutations happen here so a dry run never writes.)
        if not self.dry_run:
            if not self.trained:
                # An untrained model's salience is ≈ random → the prune step
                # would archive ~every edge, and clusters/edges are meaningless.
                # Refuse to mutate unless the caller explicitly opted in via
                # ``allow_untrained_apply=True`` (the script wires this to
                # ``--force-untrained``). The report still records proposals.
                if not getattr(self, "allow_untrained_apply", False):
                    report["apply_skipped"] = "untrained model — pass allow_untrained_apply=True to force"
                    log.warning("apply skipped: model is untrained (random salience would prune ~every edge); "
                                "pass allow_untrained_apply=True to force")
                else:
                    self._apply(report)
            else:
                self._apply(report)

        report["verifier_validation_rate"] = (
            report["verifier_accepted"] / report["verifier_calls"]
            if report["verifier_calls"] else None
        )
        return report

    # ── per-step logic (record-only; _apply mutates) ──

    def _step_cluster(self, data, out, report: dict) -> None:
        """DiffPool clusters → propose an abstract per cluster with ≥2 episodes."""
        assign = out["diffpool"]  # [N, C]
        node_ids = data.node_id
        clusters = assign.argmax(dim=-1).tolist()
        by_cluster: dict[int, list[int]] = {}
        for idx, c in enumerate(clusters):
            by_cluster.setdefault(c, []).append(idx)
        for c, idxs in by_cluster.items():
            eps = [node_ids[i] for i in idxs if node_ids[i].startswith("ep_")]
            if len(eps) >= 2:
                report["abstracts"].append({
                    "center": data.node_id[int(data.center_idx)],
                    "cluster": c, "episodes": eps,
                    "coherence": float(assign[idxs, c].mean()),
                })

    def _step_predict(self, data, out, report: dict) -> None:
        """Link-pred on sampled non-edge pairs → accept / propose / skip."""
        node_ids = data.node_id
        existing = {(int(s), int(o)) for s, o in data.edge_index.t().tolist()}
        cands = self._sample_candidate_pairs(data, existing)
        if not cands:
            return
        pair_index = torch.tensor(cands, dtype=torch.long).t().contiguous()
        with torch.no_grad():
            # Score candidate pairs with the link-pred head (dot product of
            # projected embeddings).
            link_scores = self.model.linkpred(out["node_emb"], pair_index)
        for (s, o), score in zip(cands, link_scores.tolist()):
            proposal = {
                "subject": node_ids[s], "predicate": "related_to",
                "object": node_ids[o], "confidence": float(score),
                "center": data.node_id[int(data.center_idx)],
            }
            if score >= self.cfg.accept_threshold:
                report["edges_proposed"].append(proposal)
            elif score >= self.cfg.bonsai_propose_threshold and self.verifier:
                report["verifier_calls"] += 1
                ok = self.verifier(proposal)
                if ok:
                    report["verifier_accepted"] += 1
                    report["edges_accepted"].append(proposal)
                else:
                    report["edges_unverified"].append(proposal)
            elif score >= self.cfg.bonsai_propose_threshold:
                # No verifier configured: record as unverified, do NOT accept.
                report["edges_unverified"].append(proposal)
        self._accumulate_hist(
            report["score_distributions"]["linkpred"],
            link_scores, self.cfg.score_collect_bar)

    def _step_anomaly_bounded(self, data, out, center, report: dict) -> None:
        """Run the anomaly step on the SAME bounded subgraph the head trained on.

        The anomaly head is the ONE head bounded in isolation (giant-subgraph
        data-quality fix): it trained on radius-``cfg.anomaly_subgraph_radius`` +
        ``cfg.anomaly_fanout_cap`` subgraphs, so serving it on the radius-3 giant
        ``data``/``out`` (which the other 4 steps use) would re-introduce the
        ``duplicate_episode`` flood the bound was meant to remove -- train/serve
        skew. So this loads a second bounded subgraph for ``center`` and forwards
        it, then flags anomalies on THAT output.

        Degenerate guard: when ``anomaly_subgraph_radius >= 3`` AND the cap is
        None (uncapped), the bounded subgraph IS the radius-3 giant -- the same
        graph ``data`` already loaded. Skip the redundant second load+forward and
        flag on the existing ``out`` (preserves the prior behavior exactly when a
        caller configures the old bound, e.g. ``--anomaly-radius 3
        --anomaly-fanout-cap 0``).
        """
        radius = self.cfg.anomaly_subgraph_radius
        cap = self.cfg.anomaly_fanout_cap
        if radius >= 3 and cap is None:
            # Bounded subgraph == the radius-3 giant already loaded -> reuse it.
            self._step_anomaly(data, out, report)
            return
        anom_data = self.loader.load(center, radius=radius, fanout_cap=cap)
        if anom_data.x.shape[0] < 2:
            # Too small to score (e.g. an isolated center under a tight cap) ->
            # no anomalies this center; honest, not faked.
            return
        with torch.no_grad():
            anom_out = self.model(anom_data)
        self._step_anomaly(anom_data, anom_out, report)

    def _step_anomaly(self, data, out, report: dict) -> None:
        """Flag nodes whose anomaly logits exceed 0.5 on any type."""
        anomaly = out["anomaly"]  # [N, len(ANOMALY_TYPES)]
        flags = (anomaly > 0.5).nonzero(as_tuple=False).tolist()
        for n, t in flags:
            report["anomalies"].append({
                "node": data.node_id[n], "type": ANOMALY_TYPES[t],
                "score": float(anomaly[n, t]),
            })

    def _step_ontology(self, data, out, report: dict) -> None:
        """Score entity vs candidate classes; record high-confidence typing proposals.

        The ontology head is a TWO-encoder pair classifier: entity embeddings come
        from the episode backbone (``out["node_emb"]`` at the entity rows), class
        embeddings from the taxonomy encoder over the live class DAG. For each
        entity in the subgraph we score it against a set of candidate classes
        (selected by ``cfg.ontology_strategy``) and record an ``entity -> class``
        typing proposal above the accept threshold.

        Candidate strategies (``cfg.ontology_strategy``):
          * ``all`` -- score every entity x class pair (chunked to bound memory).
            The honest, complete option: surfaces the head's real high-confidence
            typings. (The dry-run that found 0 proposals was the old rotation cap
            sampling ~16 pairs and missing every true class; the head scores true
            classes 0.93-0.98 when actually asked.)
          * ``topk`` -- cheap embedding dot-product prefilter, then score the top
            ``cfg.ontology_topk`` classes per entity with the trained head. Fast;
            catches the true classes because both encoders encode the typing
            signal (a heuristic prefilter -- can miss a pair the trained MLP
            ranks high but the dot product ranks low; use ``all`` for exact).
          * ``rotation`` -- legacy deterministic ``(k*7+j*3)%n_classes`` slice
            bounded by ``cfg.ontology_candidate_budget`` (the old behavior). Kept
            as a comparison baseline.

        HONEST scope (deferred, not faked): (a) class->class ``subClassOf``
        refinement -- no class->class labels exist, so this head does NOT propose
        taxonomy edges (it scores entity->class typing, not class hierarchy); (b)
        new-class creation is a Bonsai-gated consolidation ACTION, not a head
        output; (c) Bonsai gating on the ontology step is not wired (only
        link-pred calls the verifier today) -- proposals are RECORDED only; ``_apply``
        writes no ontology edges. The vision-complete path (Bonsai-gated promotion
        of entity->class to a real ``instanceOf``/membership edge, + new-class
        creation) is future work; the cold-start loop records proposals honestly.
        """
        if self._class_emb is None:
            tax_data, _ = build_taxonomy_data(self.store)
            tax_data = tax_data.to(self.device)
            with torch.no_grad():
                self._class_emb = self.model.encode_taxonomy(tax_data)
            self._class_names = list(tax_data.node_id)

        node_ids = data.node_id
        ent_idx = [i for i, nid in enumerate(node_ids) if nid.startswith("E:")]
        n_classes = self._class_emb.shape[0]
        if not ent_idx or n_classes == 0:
            return

        pairs = self._ontology_candidates(out["node_emb"], ent_idx, n_classes)
        if not pairs:
            return

        node_emb = out["node_emb"]
        accept = self.cfg.accept_threshold
        hist = report["score_distributions"]["ontology"]
        # Score in chunks to bound the ~580 MB [P,256] concat transient when "all"
        # scores 1512 x 377 = ~570k pairs at once. Bin each chunk's scores straight
        # into the histogram (streaming) -- avoids holding ~570k floats x 3
        # subgraphs as a Python list; the head is stateless across pairs so the
        # chunked result equals the all-at-once result.
        for chunk in _chunked(pairs, _SCORE_CHUNK):
            pair_index = torch.tensor(chunk, dtype=torch.long).t().contiguous()
            with torch.no_grad():
                scores = self.model.ontology(node_emb, self._class_emb, pair_index)
            self._accumulate_hist(hist, scores, self.cfg.score_collect_bar)
            for (ei, ci), score in zip(chunk, scores.tolist()):
                if score >= accept:
                    report["ontology_proposed"].append({
                        "entity": node_ids[ei], "class": self._class_names[ci],
                        "confidence": float(score),
                    })

    def _ontology_candidates(
        self, node_emb: torch.Tensor, ent_idx: list[int], n_classes: int
    ) -> list[tuple[int, int]]:
        """Build the (entity_row, class_row) pairs to score, per ``ontology_strategy``."""
        strategy = self.cfg.ontology_strategy
        if strategy == "all":
            return [(ei, ci) for ei in ent_idx for ci in range(n_classes)]
        if strategy == "topk":
            return self._ontology_topk_candidates(node_emb, ent_idx, n_classes)
        if strategy == "rotation":
            return self._ontology_rotation_candidates(ent_idx, n_classes)
        raise ValueError(
            f"unknown ontology_strategy {strategy!r} (expected all|topk|rotation)")

    def _ontology_topk_candidates(
        self, node_emb: torch.Tensor, ent_idx: list[int], n_classes: int
    ) -> list[tuple[int, int]]:
        """Prefilter classes per entity by embedding dot product; take top-k.

        ``entity_emb @ class_emb.T`` -> ``[n_ent, n_classes]``; top-k per row. The
        trained head then scores only those ~``n_ent * topk`` pairs (vs ``all``'s
        ``n_ent * n_classes``). ~1 s; catches the true classes because both
        encoders encode the typing signal.
        """
        k = min(self.cfg.ontology_topk, n_classes)
        ent_rows = torch.tensor(ent_idx, dtype=torch.long, device=node_emb.device)
        ent_emb = node_emb[ent_rows]                       # [n_ent, H]
        with torch.no_grad():
            sims = ent_emb @ self._class_emb.t()            # [n_ent, n_classes]
        top = torch.topk(sims, k, dim=1).indices.tolist()  # [n_ent, k]
        return [(ei, ci) for ei, cls_list in zip(ent_idx, top) for ci in cls_list]

    def _ontology_rotation_candidates(
        self, ent_idx: list[int], n_classes: int
    ) -> list[tuple[int, int]]:
        """Legacy deterministic ``(k*7+j*3)%n_classes`` slice, budget-capped.

        Reproduces the pre-knob behavior exactly (kept as a comparison baseline).
        """
        budget = self.cfg.ontology_candidate_budget
        per_entity = max(1, budget // len(ent_idx))
        pairs: list[tuple[int, int]] = []
        for k, ei in enumerate(ent_idx):
            for j in range(per_entity):
                pairs.append((ei, (k * 7 + j * 3) % n_classes))
                if len(pairs) >= budget:
                    break
            if len(pairs) >= budget:
                break
        return pairs

    def _step_prune(self, data, out, report: dict) -> None:
        """Record low-salience edges for archival pruning."""
        sal = out["salience"]  # [N]
        # Prune an edge if BOTH endpoints score below the prune threshold (a
        # high-salience endpoint keeps the edge alive — it's still a useful link
        # for that node). The binding constraint is the MAX endpoint salience
        # (prune iff max(s,o) < thr), so the histogram records per-edge max --
        # that lets prune-fraction be swept across thresholds from one run.
        thr = self.cfg.prune_salience_below
        if data.edge_index.shape[1] > 0:
            edge_max = sal[data.edge_index].max(dim=0).values  # [E]
            self._accumulate_hist(
                report["score_distributions"]["salience_endpoint"],
                edge_max, self.cfg.score_collect_bar)
        for s, o in data.edge_index.t().tolist():
            if float(sal[s]) < thr and float(sal[o]) < thr:
                report["pruned"].append({
                    "subject": data.node_id[s], "object": data.node_id[o],
                    "salience_s": float(sal[s]), "salience_o": float(sal[o]),
                })

    # ── apply (mutates; only when not dry-run) ──

    def _apply(self, report: dict) -> None:
        # Abstracts: one semantic memory per proposed cluster.
        for ab in report["abstracts"]:
            self.writer.create_abstract(
                ab["episodes"], summary=f"Abstract of {ab['episodes']}",
            )
        # Accepted edges: write as graph triples (predicate 'related_to').
        for e in report["edges_accepted"]:
            ops = self.store.graph.expand_triple(
                e["subject"], e["predicate"], e["object"]
            )
            self.store.db.batch_sync(ops)
        # Pruned edges: archive + remove.
        for p in report["pruned"]:
            # The subgraph's edge_attr carries the predicate; but the report
            # recorded subject/object only. Recover the predicate from the
            # graph by scanning the subject's out-edges for the object.
            pred = self._find_predicate(p["subject"], p["object"])
            if pred:
                self.writer.archive_edge(p["subject"], pred, p["object"],
                                          reason="low salience (prune)")

    def _find_predicate(self, subject: str, object_: str) -> Optional[str]:
        """Recover the predicate of a stored ``(subject, ?, object)`` triple."""
        from .graph_loader import KNOWN_PREDICATES
        for pred in KNOWN_PREDICATES:
            r = self.store.graph.query().vertex(subject).out(pred).execute_sync()
            try:
                if object_ in list(r.vertices):
                    return pred
            finally:
                r.close()
        return None

    # ── helpers ──

    @staticmethod
    def _accumulate_hist(acc: list[int], scores: torch.Tensor, bar: float) -> None:
        """Bin ``scores`` (clipped to [0,1]) into ``len(acc)`` equal buckets, in place.

        The bin count is taken from ``len(acc)`` so callers pick the resolution
        (the report uses 100 width-0.01 buckets so thresholds like 0.05/0.15/0.85
        are exact 0.01-boundaries and the salience cliff in [0.05, 0.10] is
        resolved -- 10 width-0.1 bins cannot). Bucket i covers
        ``[i/n, (i+1)/n)`` for n = ``len(acc)``. Scores below ``bar`` are dropped
        (``bar`` is the collection cutoff, not a bin boundary).

        Note: scores are clipped to [0,1]. The salience head emits RAW logits (can
        be slightly negative); clipping folds that tiny negative tail into bucket
        0, which is correct for any prune threshold > 0 (negatives are prunable).
        """
        n = len(acc)
        if n == 0 or scores.numel() == 0:
            return
        s = scores.detach().flatten().clamp(0.0, 1.0)
        s = s[s >= bar]
        if s.numel() == 0:
            return
        idx = (s * n).long().clamp(0, n - 1)
        counts = torch.bincount(idx, minlength=n)
        for b in range(n):
            acc[b] += int(counts[b])

    def _wm_first(self, centers: list[str], wm_ids: set[str]) -> list[str]:
        """Order centers so WM-resident episodes come first (stable)."""
        return sorted(centers, key=lambda c: (c not in wm_ids, c))

    def _sample_candidate_pairs(
        self, data, existing: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Sample non-edge node pairs to score for link prediction.

        Same-kind pairs are more likely meaningful (entity-entity, episode-
        episode) than cross-kind, so bias the sample toward them. Caps at
        ``cfg.linkpred_candidate_budget``.
        """
        node_ids = data.node_id
        n = data.x.shape[0]
        if n < 2:
            return []
        kind = data.node_kind.tolist()
        same_kind: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if kind[i] == kind[j] and (i, j) not in existing and (j, i) not in existing:
                    same_kind.append((i, j))
        # Deterministic take (no Math.random in workflows — but this is a normal
        # module; still, keep it deterministic for reproducible dev runs).
        return same_kind[: self.cfg.linkpred_candidate_budget]


__all__ = ["Consolidator", "Verifier"]