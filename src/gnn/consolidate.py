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

if TYPE_CHECKING:
    from ..config import ConsolidationConfig
    from ..memory.store import HippocampalStore

log = logging.getLogger(__name__)

# A verifier answers "is this proposal consistent with the memory graph / world
# knowledge?" — True to accept, False to reject. ``proposal`` is a dict the loop
# builds (subject/predicate/object + evidence); the caller's Bonsai-backed
# verifier turns it into a prompt and calls the Oracle/Bonsai endpoint.
Verifier = Callable[[dict], bool]


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
        # Max candidate non-edge pairs scored per subgraph (keeps the predict
        # phase tractable — all-pairs is O(N²)).
        self.max_candidates_per_subgraph = 16

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
            self._step_anomaly(data, out, report)
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
        """Propose subClassOf edges for high ontology-score pairs (Bonsai-gated)."""
        node_ids = data.node_id
        # Score entity/topic pairs as candidate subClassOf children→parents.
        et_idx = [i for i, nid in enumerate(node_ids)
                  if nid.startswith("E:") or nid.startswith("T:")]
        if len(et_idx) < 2:
            return
        pairs = []
        for i in range(len(et_idx)):
            for j in range(i + 1, len(et_idx)):
                pairs.append((et_idx[i], et_idx[j]))
                if len(pairs) >= self.max_candidates_per_subgraph:
                    break
            if len(pairs) >= self.max_candidates_per_subgraph:
                break
        if not pairs:
            return
        pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        with torch.no_grad():
            scores = self.model.ontology(out["node_emb"], pair_index)
        for (s, o), score in zip(pairs, scores.tolist()):
            if score >= self.cfg.accept_threshold:
                report["ontology_proposed"].append({
                    "child": node_ids[s], "parent": node_ids[o],
                    "confidence": float(score),
                })

    def _step_prune(self, data, out, report: dict) -> None:
        """Record low-salience edges for archival pruning."""
        sal = out["salience"]  # [N]
        # Prune an edge if BOTH endpoints score below the prune threshold (a
        # high-salience endpoint keeps the edge alive — it's still a useful link
        # for that node).
        thr = self.cfg.prune_salience_below
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

    def _wm_first(self, centers: list[str], wm_ids: set[str]) -> list[str]:
        """Order centers so WM-resident episodes come first (stable)."""
        return sorted(centers, key=lambda c: (c not in wm_ids, c))

    def _sample_candidate_pairs(
        self, data, existing: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Sample non-edge node pairs to score for link prediction.

        Same-kind pairs are more likely meaningful (entity-entity, episode-
        episode) than cross-kind, so bias the sample toward them. Caps at
        ``max_candidates_per_subgraph``.
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
        return same_kind[: self.max_candidates_per_subgraph]


__all__ = ["Consolidator", "Verifier"]