"""Training loop for the Phase 3a 5-head GNN (Task 4a).

Consumes the regenerated GNN labels (``data/training/gnn/*_labels.jsonl`` from
Task 3) and trains ``GNNModel`` (GAT backbone + 5 heads). Pod-ready + CPU-
testable: float32, batch-size-1 (``GNNModel.forward`` doesn't read
``data.batch``, so one radius-3 subgraph per step is the realistic shape for
10K-node graphs), ASCII-only logging.

Two training topologies (``cfg.head``):
- ``all``    -- one joint multi-task run: per step, sum every head loss that has
  usable labels for this subgraph (heads with no labels this step are skipped,
  not zeroed). Saves ``all.pt`` + one self-contained ``{head}.pt`` per head
  (all carry the same full state_dict -- consolidation inference loads by head
  name). The cheap CPU-dev default.
- ``<one>``  -- train that head only. With ``--backbone-checkpoint``, load a
  full state_dict, FREEZE the GAT backbone, and refine just the head on top of
  the shared features (mirrors the 2b frozen-backbone gate pattern). Without
  it, cold-start the backbone + that head. Saves one ``{head}.pt``.

Checkpoint format: a RAW ``model.state_dict()`` (strict-loadable), matching
``scripts/run_consolidation.py:_load_model`` which does ``torch.load`` +
``load_state_dict(state)``. Metadata (step, per-head val metrics, config,
wall-clock) is written to a sidecar ``{head}.pt.meta.json`` -- NOT wrapped into
the ``.pt`` (the consolidation loader expects a bare state_dict).

Store-backed pod path (Path A, user decision -- see ADR 010): the compact
corpus DB is SCP'd to the pod and opened read-only here. The loader's BFS is
the SAME walk the label generator used (``OracleLabelingPipeline.extract_subgraph``),
so a training example and its labels are over the same node/edge set by
construction -- zero train/serve skew, no tensor-persistence layer needed.

Anomaly head: the label record carries ``seed`` + ``types`` (+ ``node_labels``)
from the generator's injection-based labeling. The trainer REPRODUCES the
corrupted subgraph deterministically from ``(subgraph_id, seed, types)``:
``extract_subgraph`` -> ``enrich_subgraph`` -> ``inject_anomalies`` ->
``data_from_subgraph(corrupted, training_feature_for(store))``. Synthetic
injected nodes (``{orig}_dup``, ``ep_iso_*``, ``M:000N``) aren't in the store;
``training_feature_for`` reuses the origin's feature for ``_dup`` clones and
lets ``feature_for`` degrade gracefully (hash / onehot) for the rest.
"""

from __future__ import annotations

import copy
import json
import random as _random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from ..config import Phase3aConfig
from ..training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers
from .anomaly_injector import inject_anomalies
from .anomaly_rules import ANOMALY_TYPES, enrich_subgraph
from .features import training_feature_for
from .graph_loader import WaveDBGraphLoader, data_from_subgraph
from .heads import AnomalyHead, DiffPoolHead, LinkPredHead, OntologyHead, SalienceHead
from .label_tensors import (
    anomaly_target, class_vocab, linkpred_pairs, ontology_target, salience_target,
    split_centers,
)
from .model import GNNModel
from .taxonomy_graph import build_taxonomy_data

HEADS: tuple[str, ...] = ("salience", "diffpool", "link_prediction", "ontology", "anomaly")
HEAD_CHOICES: tuple[str, ...] = ("all", "salience", "link_prediction", "ontology", "cluster", "anomaly")

# Map a --head choice to the model submodule that head trains. ``cluster`` trains
# the ``diffpool`` head (the cluster-assignment head); the user-facing name stays
# ``cluster`` to match the label-file stem ``cluster_labels.jsonl``.
_HEAD_MODULE: dict[str, str] = {
    "salience": "salience", "link_prediction": "linkpred",
    "ontology": "ontology", "cluster": "diffpool", "anomaly": "anomaly",
}

# Per-head val metric + whether higher is better (for best-val tracking).
_HIGHER_BETTER: dict[str, bool] = {
    "salience": False, "diffpool": False,        # MAE / entropy -- lower better
    "link_prediction": True, "ontology": True, "anomaly": True,  # AUC / acc / F1
}


@dataclass
class GNNTrainConfig:
    """Knobs for ``train_gnn``. Defaults mirror ``GNNConfig`` / the spec cold start."""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    predicate_vocab_size: int = 32
    num_clusters: int = 16
    lr: float = 1e-3
    epochs: int = 20
    device: str = "auto"
    dtype: str = "float32"          # float32-only (bf16/autocast unfixed in the 2a SSM path)
    val_fraction: float = 0.1
    seed: int = 0
    checkpoint_dir: str = field(default_factory=lambda: Phase3aConfig().checkpoint_dir)
    head: str = "all"
    backbone_checkpoint: Optional[str] = None
    ogb_pretrain: bool = False      # pod-only, deferred -- see _ogb_pretrain


# ═══════════════════════════════════════════════════════════════════════════
# device / dtype
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    # GNN is float32-only. The bf16/autocast dtype-mix bug is in the 2a SSM path
    # (GNN-independent), and fp32 is fine for the cold start, so every requested
    # dtype resolves to float32. WARN on a non-fp32 request rather than silently
    # downgrading, so a user who passes --dtype bf16 expecting a speedup doesn't
    # misread their own wall-clock (mirrors routing_training._resolve_dtype).
    if name not in ("float32", "fp32", "auto"):
        print(f"  NOTE: dtype '{name}' requested but the GNN trains float32-only "
              "(bf16/autocast is unfixed in the 2a SSM path; the GNN is independent "
              "of it but kept fp32 for the cold-start baseline). Using float32.",
              file=sys.stderr, flush=True)
    return torch.float32


# ═══════════════════════════════════════════════════════════════════════════
# label IO
# ═══════════════════════════════════════════════════════════════════════════

def _load_labels(path: Path) -> dict[str, dict]:
    """Read a ``{subgraph_id, labels}`` JSONL file into ``{subgraph_id: labels}``.

    Missing file -> empty dict (a head whose labels weren't regenerated is just
    never trainable -- honest, not an error). Parse errors are skipped (the
    generator's validators already gate the file; a malformed line shouldn't
    abort a 4000-subgraph run).
    """
    out: dict[str, dict] = {}
    path = Path(path)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("subgraph_id")
            if sid:
                out[sid] = rec.get("labels") or {}
    return out


def _read_radius(labels_dir: Path, fallback: int) -> int:
    """Read the generation radius from ``quality_report.json`` (fallback if absent)."""
    p = Path(labels_dir) / "quality_report.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                rep = json.load(f)
            r = rep.get("radius")
            if isinstance(r, int) and r > 0:
                return r
        except (json.JSONDecodeError, OSError):
            pass
    return fallback


# ═══════════════════════════════════════════════════════════════════════════
# OGB pretrain-then-transfer (pod-only, DEFERRED)
# ═══════════════════════════════════════════════════════════════════════════

def _ogb_pretrain(model: GNNModel, device: torch.device) -> int:
    """Pod-only OGB pretrain-then-transfer (spec sec 1.3 decision 1). DEFERRED.

    The cold-start trainer uses DIRECT training on the Hippo graph (the spec's
    fallback path). OGB pretrain-then-transfer -- pretrain the GAT backbone on
    ogbn-arxiv, transfer the GATConv weights into the Hippo model -- is an
    optional quality lever for the pod run. It is NOT implemented in this
    CPU-testable trainer: it needs the [gnn] extra + the ogbn-arxiv dataset on
    the pod, and a mini-batch pretrain loop whose transfer logic (arxiv has no
    edge features, so layer 0's edge weights aren't transferable) deserves its
    own verified run, not untested code shipped blind. This function LAZY-probes
    for ogb (never imported at module top, so CPU dev doesn't need it) and fails
    LOUDLY when toggled, so the flag can't silently no-op.

    To implement (trigger: the Task 4b pod run, once direct-train baselines are
    measured): ``from ogb.nodeproppred import PygNodePropPredDataset``; load
    ogbn-arxiv; pretrain a matching-GAT (same hidden_dim/num_heads/num_layers,
    NO edge_dim on layer 0) with ``NeighborLoader`` + node-label CE; per-layer
    ``model.layers[i].load_state_dict(temp[i].state_dict(), strict=False)`` (the
    lin_edge weight on layer 0 has no source -> stays at init). Return the count
    of layers whose lin_src/lin_dst transferred.
    """
    try:
        import ogb  # noqa: F401  -- probe only, never used at module top
        present = True
    except Exception:
        present = False
    raise RuntimeError(
        "OGB pretrain-then-transfer is a pod-only step, not yet wired in the "
        "trainer (spec sec 1.3 decision 1: OGB-pretrain is an optional lever; "
        "direct-train is the cold-start fallback, which is what --ogb-pretrain "
        "OFF does). "
        + ("ogb is installed but the transfer loop is not implemented"
           if present else "ogb is NOT installed (pip install '.[gnn]' on the pod)")
        + ". See src/gnn/train.py:_ogb_pretrain docstring for the implementation plan."
    )


# ═══════════════════════════════════════════════════════════════════════════
# per-step build + forward
# ═══════════════════════════════════════════════════════════════════════════

def _build_inputs(cfg, sid, store, loader, pipe, feat_for, labels, radius):
    """Build the clean + corrupted ``Data`` for one center + its per-head labels.

    The clean heads (salience / link / cluster / ontology) train on the CLEAN
    subgraph (their labels were generated on it); the anomaly head trains on the
    REPRODUCED corrupted subgraph. So a joint (``all``) step builds BOTH when
    anomaly labels exist for this center; a single-head step builds only what it
    needs (one store walk, not two).
    """
    sal_lbl = labels["salience"].get(sid)
    link_lbl = labels["link_prediction"].get(sid)
    ont_lbl = labels["ontology"].get(sid)
    an_lbl = labels["anomaly"].get(sid)

    need_clean = cfg.head in ("all", "salience", "link_prediction", "cluster", "ontology")
    need_anom = cfg.head in ("all", "anomaly")

    clean_data = loader.load(sid, radius=radius) if need_clean else None

    corrupted_data = None
    if need_anom and an_lbl is not None:
        sub = pipe.extract_subgraph(sid, radius=radius)
        enriched = enrich_subgraph(store, copy.deepcopy(sub))
        corrupted, _ = inject_anomalies(
            enriched, seed=an_lbl.get("seed", 0), types=an_lbl.get("types"),
        )
        corrupted_data = data_from_subgraph(
            corrupted, feat_for, predicate_vocab_size=cfg.predicate_vocab_size,
        )
    return clean_data, corrupted_data, sal_lbl, link_lbl, ont_lbl, an_lbl


def _forward(model, cfg, clean_data, corrupted_data, sal_lbl, link_lbl, ont_lbl,
             an_lbl, device, seed, tax_data, class_index):
    """Run the backbone + the heads selected by ``cfg.head`` that have usable
    labels this step. Returns ``{head: {pred, target, ...}}`` for the heads with
    usable labels (the trainer sums their losses / accumulates their val preds;
    absent heads are simply skipped).

    ``tax_data`` is the class DAG ``Data`` (built once, on ``device``) and
    ``class_index`` maps class name -> taxonomy row; both are used only by the
    ontology head, whose class embeddings come from the taxonomy encoder (run
    here WITH grad each step so the encoder trains alongside the head).
    """
    out: dict[str, dict] = {}

    if clean_data is not None:
        clean_data = clean_data.to(device)
        node_emb = model.encode(
            clean_data.x, clean_data.edge_index,
            getattr(clean_data, "edge_attr", None), clean_data.node_kind,
        )
        if cfg.head in ("all", "salience") and sal_lbl is not None:
            tgt, mask = salience_target(clean_data, sal_lbl)
            if mask.any():
                out["salience"] = {
                    "pred": model.salience(node_emb), "target": tgt, "mask": mask,
                }
        if cfg.head in ("all", "cluster"):
            assign = model.diffpool(node_emb, clean_data.edge_index)
            out["diffpool"] = {"assign": assign, "edge_index": clean_data.edge_index}
        if cfg.head in ("all", "link_prediction") and link_lbl is not None:
            pt = linkpred_pairs(clean_data, link_lbl, seed=seed)
            if pt.edge_index is not None:
                out["link_prediction"] = {
                    "pred": model.linkpred(node_emb, pt.edge_index.to(device)),
                    "target": pt.labels.to(device), "skipped": pt.skipped,
                }
        if cfg.head in ("all", "ontology") and ont_lbl is not None:
            pt = ontology_target(clean_data, ont_lbl, class_index, seed=seed)
            if pt.edge_index is not None:
                # Two-encoder pair classifier: entity emb from the episode
                # backbone (node_emb), class emb from the taxonomy encoder
                # (recomputed each step WITH grad so it trains). The taxonomy
                # encoder is cheap (377 nodes, 2 layers) vs. an episode radius-3
                # subgraph, so recomputing per step is negligible.
                class_emb = model.encode_taxonomy(tax_data)
                out["ontology"] = {
                    "pred": model.ontology(node_emb, class_emb, pt.edge_index.to(device)),
                    "target": pt.labels.to(device), "skipped": pt.skipped,
                }

    if corrupted_data is not None:
        corrupted_data = corrupted_data.to(device)
        c_emb = model.encode(
            corrupted_data.x, corrupted_data.edge_index,
            getattr(corrupted_data, "edge_attr", None), corrupted_data.node_kind,
        )
        if cfg.head in ("all", "anomaly"):
            tgt = anomaly_target(corrupted_data, an_lbl or {}).to(device)
            out["anomaly"] = {"pred": model.anomaly(c_emb), "target": tgt}

    return out


def _losses_from_outputs(model, out: dict) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    if "salience" in out:
        d = out["salience"]
        dev = d["pred"].device
        m = d["mask"].to(dev)
        # salience_target builds target+mask on CPU (default device); move target
        # to the pred's device BEFORE indexing -- indexing a CPU tensor with a
        # cuda mask raises "indices should be on the same device". The old
        # ``d["target"][m].to(dev)`` indexed first (crash) then moved (never ran).
        tgt = d["target"].to(dev)
        losses["salience"] = SalienceHead.loss(d["pred"][m], tgt[m])
    if "diffpool" in out:
        d = out["diffpool"]
        losses["diffpool"] = model.diffpool.loss(d["assign"], d["edge_index"])
    if "link_prediction" in out:
        d = out["link_prediction"]
        losses["link_prediction"] = LinkPredHead.loss(d["pred"], d["target"])
    if "ontology" in out:
        d = out["ontology"]
        losses["ontology"] = OntologyHead.loss(d["pred"], d["target"])
    if "anomaly" in out:
        d = out["anomaly"]
        losses["anomaly"] = AnomalyHead.loss(d["pred"], d["target"])
    return losses


def _accumulate_val(out: dict, acc: dict) -> None:
    """Collect per-head (pred, target) tensors across val centers (CPU).

    The salience mask stays on CPU here (it is NOT moved to the pred's device,
    unlike the loss path in ``_losses_from_outputs``). This is deliberate, not
    an oversight: ``cuda_tensor[cpu_bool_mask]`` is tolerated (the mask is
    auto-moved to the tensor's device), and ``d["target"][m]`` is cpu[cpu] -- so
    both indexings here are valid on GPU. The loss path moves the mask to cuda
    because it indexes the CPU ``target`` there, and ``cpu_tensor[cuda_mask]``
    RAISES ("indices should be on the same device") -- the GPU crash that fix
    repaired. Don't "unify" the two paths by moving the mask here too: doing so
    without also moving the target would turn ``d["target"][m]`` into the exact
    cpu[cuda] crash. The asymmetry is correct."""
    if "salience" in out:
        d = out["salience"]
        m = d["mask"]
        if m.any():
            acc["salience"][0].append(d["pred"][m].detach().cpu())
            acc["salience"][1].append(d["target"][m].detach().cpu())
    if "link_prediction" in out:
        d = out["link_prediction"]
        acc["link_prediction"][0].append(d["pred"].detach().cpu())
        acc["link_prediction"][1].append(d["target"].detach().cpu())
    if "ontology" in out:
        d = out["ontology"]
        acc["ontology"][0].append(d["pred"].detach().cpu())
        acc["ontology"][1].append(d["target"].detach().cpu())
    if "anomaly" in out:
        d = out["anomaly"]
        acc["anomaly"][0].append(d["pred"].detach().cpu())
        acc["anomaly"][1].append(d["target"].detach().cpu())
    if "diffpool" in out:
        acc["diffpool"].append(out["diffpool"]["assign"].detach().cpu())


def _metrics_from_accumulators(acc: dict) -> dict[str, Optional[float]]:
    m: dict[str, Optional[float]] = {}
    if acc["salience"][0]:
        m["salience"] = SalienceHead.metric(torch.cat(acc["salience"][0]),
                                            torch.cat(acc["salience"][1]))
    else:
        m["salience"] = None
    if acc["link_prediction"][0]:
        m["link_prediction"] = LinkPredHead.metric(torch.cat(acc["link_prediction"][0]),
                                                   torch.cat(acc["link_prediction"][1]))
    else:
        m["link_prediction"] = None
    if acc["ontology"][0]:
        m["ontology"] = OntologyHead.metric(torch.cat(acc["ontology"][0]),
                                            torch.cat(acc["ontology"][1]))
    else:
        m["ontology"] = None
    if acc["anomaly"][0]:
        m["anomaly"] = AnomalyHead.metric(torch.cat(acc["anomaly"][0]),
                                          torch.cat(acc["anomaly"][1]))
    else:
        m["anomaly"] = None
    if acc["diffpool"]:
        m["diffpool"] = sum(DiffPoolHead.metric(a) for a in acc["diffpool"]) / len(acc["diffpool"])
    else:
        m["diffpool"] = None
    return m


def _is_better(head: str, new: float, old: Optional[float]) -> bool:
    if old is None:
        return True
    return (new > old) if _HIGHER_BETTER[head] else (new < old)


def _build_optimizer(model: GNNModel, cfg: GNNTrainConfig) -> torch.optim.Optimizer:
    """Optimizer over exactly the params that will receive gradients.

    ``all``: everything trainable. ``<one>``: the selected head + the backbone
    (the backbone is frozen when a ``--backbone-checkpoint`` was loaded, so its
    params drop out via ``requires_grad``). The ontology head's class embeddings
    come from the taxonomy encoder, so a ``--head ontology`` run must also train
    it (the taxonomy encoder is NOT frozen by a backbone checkpoint -- only
    ``input_proj`` + ``layers`` are). Other heads aren't called in single-head
    mode -> no gradients -> excluded so they don't clutter the optimizer state.
    """
    if cfg.head == "all":
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        head_mod = getattr(model, _HEAD_MODULE[cfg.head])
        head_ids = {id(p) for p in head_mod.parameters()}
        backbone_ids = {id(p) for p in model.input_proj.parameters()} | \
                       {id(p) for p in model.layers.parameters()}
        extra_ids: set[int] = set()
        if cfg.head == "ontology":
            # The taxonomy encoder produces the ontology head's class side;
            # it must train in a --head ontology run (it is NOT part of the
            # frozen backbone).
            extra_ids |= {id(p) for p in model.taxonomy.parameters()}
        params = [p for p in model.parameters() if p.requires_grad
                  and (id(p) in head_ids or id(p) in backbone_ids
                       or id(p) in extra_ids)]
    return torch.optim.AdamW(params, lr=cfg.lr)


# ═══════════════════════════════════════════════════════════════════════════
# main loop
# ═══════════════════════════════════════════════════════════════════════════

def train_gnn(
    cfg: GNNTrainConfig,
    store,
    labels_dir,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
) -> dict[str, Any]:
    """Train the GNN. Returns a summary dict (per-head val metrics, checkpoints,
    honest notes). See module docstring for the data flow + checkpoint format."""
    labels_dir = Path(labels_dir)
    device = _resolve_device(cfg.device)
    _resolve_dtype(cfg.dtype)  # float32 always; NOTE on a non-fp32 request
    torch.manual_seed(cfg.seed)
    _random.seed(cfg.seed)

    if cfg.head not in HEAD_CHOICES:
        raise ValueError(f"cfg.head must be one of {HEAD_CHOICES}, got {cfg.head!r}")

    labels = {
        "salience": _load_labels(labels_dir / "salience_labels.jsonl"),
        "link_prediction": _load_labels(labels_dir / "link_prediction_labels.jsonl"),
        "ontology": _load_labels(labels_dir / "ontology_labels.jsonl"),
        "cluster": _load_labels(labels_dir / "cluster_labels.jsonl"),
        "anomaly": _load_labels(labels_dir / "anomaly_labels.jsonl"),
    }
    radius = _read_radius(labels_dir, fallback=3)

    # Centers: union for `all`, the selected head's centers otherwise.
    if cfg.head == "all":
        centers = sorted(set().union(*(set(v) for v in labels.values())))
    else:
        centers = sorted(labels[cfg.head])
    valid = set(sample_episode_centers(store))
    dropped = [c for c in centers if c not in valid]
    centers = [c for c in centers if c in valid]
    if dropped:
        print(f"  NOTE: {len(dropped)} label center(s) not in this store -- skipped "
              f"(e.g. {dropped[:3]})", file=sys.stderr, flush=True)
    if not centers:
        raise RuntimeError(
            f"no usable training centers for head={cfg.head!r} under {labels_dir} "
            "(the label files are empty or none of their subgraph_ids are episodes "
            "in this store)."
        )

    train_ids, val_ids = split_centers(centers, cfg.val_fraction, cfg.seed)

    # Model + optional backbone load + freeze.
    model = GNNModel(
        hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
        dropout=cfg.dropout, predicate_vocab_size=cfg.predicate_vocab_size,
        num_clusters=cfg.num_clusters,
    ).to(device)
    if cfg.backbone_checkpoint:
        state = torch.load(cfg.backbone_checkpoint, map_location=device, weights_only=True)
        # strict=False: a backbone checkpoint from before the taxonomy encoder
        # existed (e.g. the GPU ``all.pt`` from #125) has NO ``taxonomy.*`` keys.
        # Those stay at init (random) and train fresh in this run. Any OTHER
        # missing/unexpected key means the checkpoint is genuinely incompatible
        # -- fail loud rather than silently loading a partial backbone.
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("taxonomy.")]
        assert not bad_missing, (
            f"backbone checkpoint {cfg.backbone_checkpoint} missing unexpected "
            f"keys (not the taxonomy encoder): {bad_missing}")
        assert not unexpected, (
            f"backbone checkpoint {cfg.backbone_checkpoint} has unexpected keys: "
            f"{unexpected}")
        for p in model.input_proj.parameters():
            p.requires_grad = False
        for p in model.layers.parameters():
            p.requires_grad = False
        # The taxonomy encoder is NOT part of the frozen backbone -- it trains
        # in this run (it had no weights in the old checkpoint to freeze anyway).
        print(f"  backbone loaded from {cfg.backbone_checkpoint} (frozen) "
              f"-- refining head={cfg.head}", file=sys.stderr, flush=True)
    if cfg.ogb_pretrain:
        _ogb_pretrain(model, device)  # raises (deferred pod step) -- loud, not silent

    optimizer = _build_optimizer(model, cfg)
    pipe = OracleLabelingPipeline(store)
    loader = WaveDBGraphLoader(store, radius=radius, predicate_vocab_size=cfg.predicate_vocab_size)
    feat_for = training_feature_for(store)

    # Taxonomy DAG for the ontology head's class side. Built ONCE (the live class
    # set is fixed for the run; the taxonomy encoder weights change each step but
    # the graph structure doesn't). ``class_index`` maps each class name seen in
    # the labels to its row in the taxonomy Data so ``ontology_target`` can form
    # (entity_row, class_row) pairs. Classes in the labels but NOT in the live
    # DAG (an Oracle-invented name that hasn't been Bonsai-promoted) are absent
    # from ``class_index`` -> ``ontology_target`` skips them + counts (honest).
    # Only built for runs that actually score ontology (all / ontology) -- a
    # salience-only run doesn't pay the ~754-query class-DAG enumeration.
    if cfg.head in ("all", "ontology"):
        tax_data, name_to_row = build_taxonomy_data(store)
        tax_data = tax_data.to(device)
        class_names = class_vocab(labels_dir)
        class_index = {name: name_to_row[name]
                       for name in class_names if name in name_to_row}
        dropped_classes = [n for n in class_names if n not in name_to_row]
        if dropped_classes:
            print(f"  NOTE: {len(dropped_classes)} ontology class name(s) not in the "
                  f"live DAG -- skipped (e.g. {dropped_classes[:3]}). These are "
                  f"Oracle-suggested classes not yet Bonsai-promoted to a seed-"
                  f"anchored subClassOf edge.", file=sys.stderr, flush=True)
    else:
        tax_data = None
        class_index = {}

    # Which heads to track best-val for. `all` tracks all 5; a single-head run
    # tracks its head (cluster -> diffpool) only.
    tracked = list(HEADS) if cfg.head == "all" else [
        "diffpool" if cfg.head == "cluster" else cfg.head]
    best_val: dict[str, Optional[float]] = {h: None for h in tracked}
    best_step: dict[str, int] = {h: -1 for h in tracked}
    final_val: dict[str, Optional[float]] = {h: None for h in tracked}
    head_steps: dict[str, int] = {h: 0 for h in HEADS}
    total_skipped: dict[str, int] = {"link_prediction": 0, "ontology": 0}

    # Checkpoint dir + a save helper set up BEFORE the loop so we can write a
    # best-so-far checkpoint at the end of every epoch. A 20-epoch pod run that
    # only saves at the very end loses everything if the SSH session dies (SIGHUP)
    # or the box reboots at epoch 19; per-epoch saves leave the latest state on
    # disk so a crash never loses more than one epoch. The post-loop save below
    # is the guaranteed-final overwrite (identical path, full wall-clock in meta).
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    param_count = sum(p.numel() for p in model.parameters())

    def _save(head_name: str, step_now: int, wall_now: float) -> Path:
        state_dict = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        path = ckpt_dir / f"{head_name}.pt"
        torch.save(state_dict, path)
        meta = {
            "head": head_name, "step": step_now, "epochs": cfg.epochs,
            "radius": radius, "device": str(device), "dtype": "float32",
            "param_count": param_count, "wall_clock_s": wall_now,
            "config": {
                "hidden_dim": cfg.hidden_dim, "num_heads": cfg.num_heads,
                "num_layers": cfg.num_layers, "lr": cfg.lr, "seed": cfg.seed,
                "val_fraction": cfg.val_fraction, "head": cfg.head,
                "backbone_checkpoint": cfg.backbone_checkpoint,
            },
            "val_metrics": {h: final_val.get(h) for h in tracked},
            "best_val": {h: best_val[h] for h in tracked},
            "best_step": {h: best_step[h] for h in tracked},
            "skipped_endpoints": total_skipped,
        }
        with open(ckpt_dir / f"{head_name}.pt.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def _save_all(step_now: int, wall_now: float) -> list[Path]:
        # `all` -> all.pt + one self-contained .pt per head (consolidation loads
        # by head name); a single-head run -> only that head's .pt (NOT all.pt,
        # which would imply a joint run). Mirrors the pre-refactor branch.
        if cfg.head == "all":
            paths = [_save("all", step_now, wall_now)]
            for h in ("salience", "link_prediction", "ontology", "cluster", "anomaly"):
                paths.append(_save(h, step_now, wall_now))
            return paths
        return [_save(cfg.head, step_now, wall_now)]

    t0 = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        _random.shuffle(train_ids)
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        for sid in train_ids:
            cdata, andata, sal_lbl, link_lbl, ont_lbl, an_lbl = _build_inputs(
                cfg, sid, store, loader, pipe, feat_for, labels, radius)
            out = _forward(model, cfg, cdata, andata, sal_lbl, link_lbl, ont_lbl,
                           an_lbl, device, seed=cfg.seed + step,
                           tax_data=tax_data, class_index=class_index)
            for h in ("link_prediction", "ontology"):
                if h in out:
                    total_skipped[h] += out[h]["skipped"]
            for h in out:
                head_steps[h] += 1
            losses = _losses_from_outputs(model, out)
            if not losses:
                continue  # no usable labels this center -> skip step honestly
            total = sum(losses.values())
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_sum += float(total.item())
            epoch_n += 1
            step += 1

        # End-of-epoch validation.
        if val_ids:
            acc = {"salience": ([], []), "link_prediction": ([], []),
                   "ontology": ([], []), "anomaly": ([], []), "diffpool": []}
            model.eval()
            with torch.no_grad():
                for sid in val_ids:
                    cdata, andata, sal_lbl, link_lbl, ont_lbl, an_lbl = _build_inputs(
                        cfg, sid, store, loader, pipe, feat_for, labels, radius)
                    out = _forward(model, cfg, cdata, andata, sal_lbl, link_lbl,
                                   ont_lbl, an_lbl, device, seed=cfg.seed + step,
                                   tax_data=tax_data, class_index=class_index)
                    _accumulate_val(out, acc)
            val_metrics = _metrics_from_accumulators(acc)
        else:
            val_metrics = {h: None for h in HEADS}

        for h in tracked:
            final_val[h] = val_metrics.get(h)
            v = val_metrics.get(h)
            if v is not None and _is_better(h, v, best_val[h]):
                best_val[h] = v
                best_step[h] = step

        # Per-epoch checkpoint: overwrite all.pt (+ per-head) with the current
        # state so a crash/SIGHUP at epoch N still leaves the epoch-N checkpoint
        # on disk. Cheap (a handful of small saves, overwritten each epoch, vs.
        # a multi-minute epoch) and the meta carries this epoch's val_metrics +
        # the best-so-far tracking (same fields the final save will write).
        _save_all(step, time.time() - t0)

        if progress_cb:
            mean_train = epoch_loss_sum / max(1, epoch_n)
            # Representative val for the print: the trained head's, else NaN.
            rep = final_val[tracked[0]] if tracked else None
            progress_cb(step, mean_train, float(rep) if rep is not None else float("nan"))

    # ── checkpoints ── (raw state_dict, sidecar meta JSON). One final overwrite
    # so the on-disk meta carries the exact final wall-clock; the per-epoch saves
    # above already wrote the same paths each epoch (so a crash before here still
    # leaves a usable checkpoint -- the whole point of the per-epoch saves).
    wall = time.time() - t0
    paths = _save_all(step, wall)

    # ── honest notes ──
    notes: list[str] = []
    selected = list(HEADS) if cfg.head == "all" else [
        "diffpool" if cfg.head == "cluster" else cfg.head]
    for h in selected:
        if head_steps[h] == 0:
            pretty = "cluster (diffpool)" if h == "diffpool" else h
            notes.append(
                f"{pretty}: no usable labels in any subgraph -- val_metric=None, "
                "this head is UNTRAINED (the label file was empty or no subgraph "
                "yielded a scoreable target).")
    if total_skipped["link_prediction"]:
        notes.append(f"link_prediction: {total_skipped['link_prediction']} label "
                     "endpoint(s) didn't resolve to subgraph nodes (skipped, not truncated).")
    if total_skipped["ontology"]:
        notes.append(f"ontology: {total_skipped['ontology']} label endpoint(s) "
                     "didn't yield a scoreable entity->class pair (entity not in "
                     "subgraph, or class not in the live DAG) -- skipped, not truncated.")
    if not val_ids:
        notes.append("no validation centers (val_fraction too small for the center "
                     "count) -- all val metrics are None.")

    return {
        "head": cfg.head,
        "param_count": param_count,
        "train_centers": len(train_ids),
        "val_centers": len(val_ids),
        "epochs": cfg.epochs,
        "steps": step,
        "radius": radius,
        "device": str(device),
        "dtype": "float32",
        "wall_clock_s": wall,
        "best_val": {h: best_val[h] for h in tracked},
        "best_step": {h: best_step[h] for h in tracked},
        "final_val": {h: final_val[h] for h in tracked},
        "head_steps": head_steps,
        "skipped_endpoints": total_skipped,
        "checkpoints": [str(p) for p in paths],
        "notes": notes,
    }


__all__ = ["GNNTrainConfig", "train_gnn", "HEADS", "HEAD_CHOICES"]