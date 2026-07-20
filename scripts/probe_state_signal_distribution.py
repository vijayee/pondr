"""Phase 0a: WHERE is the state-signal loss -- the readout or the backbone?

Phase B GATE 1 ([[pondr-strm-phaseb-ht-gate-no-go]]) returned NO-GO: a
``ZRelevanceHead`` trained on ``z_i = LatentDynamicsHead.project(slot.h)`` (the
mean-pool of the last SSM layer's 16 ``d_state`` channels) reached top-3 recall
0.285 == random, and the no-train baseline showed ``bge(query, slots_z)`` top-3
= 0.196 with across-slot std 0.00169 = 0.068x the doc baseline (0.0248) -- the
projected recurrent state is near-constant across 15 different doc inputs.

This probe isolates the cheapest root cause BEFORE any backbone retraining:

  ``LatentDynamicsHead.project`` is a FIXED, PARAMETER-FREE mean over the 16
  ``d_state`` channels of the last layer (``latent_dynamics_head.py:62-91``).
  If those 16 channels carry opposing-sign signal, the MEAN cancels to
  near-constant while individual channels (or a learned projection) vary. The
  signal may be in the state and killed by the mean-pool -- a learned readout
  (Phase 0b) could recover it WITHOUT retraining the backbone.

For ~200 queries x 15 slots it drives the WorkingMemory exactly as
``generate_relevance_data.build_records`` does (``wm.reset()`` then one
``wm.step(doc_bge)`` per candidate doc), captures the raw per-slot ``slot.h``
(the 4 per-layer ``[1,16,384]`` fp16 recurrent state shipped in Phase A), and
measures on FOUR representations:

  - ``z_mean_last`` [384]   -- the mean-pool (reproduce 0.00169 / 0.196).
  - ``z_chan_last`` [16,384] -- per-channel last layer. The DECISIVE test: for
    each of the 16 channels, ``cosine(query, channel[K,384])`` top-3 recall. If
    ANY channel climbs toward the 0.938 doc baseline, the mean-pool cancelled
    real signal -> Phase 0b (learned readout) -> skip Phase 1.
  - ``z_flat_last`` [6144]  -- flattened last layer. Across-slot std ratio only
    (a raw cosine vs the 384-d query is dimensionally impossible; a learned
    readout is exactly Phase 0b).
  - ``z_flat_all`` [24576]  -- flattened all 4 layers. Across-slot std ratio only.

GATE 0a decision (no training):
  - GO (-> 0b): any channel top-3 climbs materially above 0.196, OR any
    flat-rep std ratio climbs well above 0.068x. The state carries signal the
    mean-pool destroyed.
  - NO-GO (-> Phase 1): every channel top-3 ~0.196 AND every std ratio ~0.068x.
    The state truly collapsed under the JEPA objective -- warm-start the backbone.

Reuses ``load_questions`` / ``build_doc_index`` / ``open_docs_table`` / ``get_doc``
/ ``embed_doc`` from ``scripts/generate_relevance_data.py`` and
``_cos_top3_recall`` / ``_split_queries`` / ``_wilson_ci95`` from the relevance
training path. No new model code, no training, no live-orchestrator touch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import HierarchicalChunker  # noqa: E402
from src.ingestion.parsers import MarkdownParser  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.latent_dynamics_head import LatentDynamicsHead  # noqa: E402
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _split_queries,
    _wilson_ci95,
)
from src.subconscious.training.routing_training import (  # noqa: E402
    _resolve_device,
    build_embedder,
    load_backbone,
)
from src.subconscious.working_memory import WorkingMemory  # noqa: E402

from scripts.generate_relevance_data import (  # noqa: E402
    DEFAULT_BACKBONE_PATH,
    DEFAULT_DOCS_PARQUET,
    DEFAULT_QUESTIONS_PARQUET,
    GOLD_CATEGORIES,
    build_doc_index,
    embed_doc,
    get_doc,
    load_questions,
    open_docs_table,
)


# The Phase B baseline numbers this probe must reproduce (sanity) -- from
# [[pondr-strm-phaseb-ht-gate-no-go]]: bge(query, slots_z) top-3 = 0.196,
# across-slot std of slots_z = 0.00169 = 0.068x the doc across-slot std (0.0248).
DOC_BASELINE_TOP3 = 0.938
MEANPOOL_TOP3 = 0.196
MEANPOOL_STD_RATIO = 0.068


def _cos_top3_recall(qvec, doc_vecs, labels):
    """qvec [D], doc_vecs [K,D], labels [K] (1=gold) -> (recall, hit) or None."""
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    sims = d @ q
    gold_idx = [i for i, l in enumerate(labels) if l == 1]
    n_gold = len(gold_idx)
    if n_gold == 0:
        return None
    k_top = min(3, len(sims))
    top = set(np.argsort(-sims)[:k_top].tolist())
    n_in = sum(1 for i in gold_idx if i in top)
    return n_in / n_gold, n_in == n_gold


def _mean_top3(records, rep_key):
    """Mean top-3 recall + Wilson CI over the val split for one representation.

    ``rep_key`` names a per-slot np.ndarray field on each record (``query``,
    ``z_mean``, ``z_chan_c{cc}``, ``doc``); the field's last axis must match the
    query dim (384) for a cosine to be defined.
    """
    cfg = RelevanceTrainingConfig()
    train_idx, val_idx = _split_queries(len(records), cfg.val_fraction, cfg.seed)
    val = [records[i] for i in val_idx]
    recalls, hits = [], 0
    for rec in val:
        qv = rec["query"]
        dv = rec[rep_key]
        lab = rec["labels"]
        r = _cos_top3_recall(qv, dv, lab)
        if r is None:
            continue
        recalls.append(r[0])
        if r[1]:
            hits += 1
    if not recalls:
        return None
    mean_top3 = sum(recalls) / len(recalls)
    hit_rate = hits / len(recalls)
    ci = _wilson_ci95(hit_rate, len(recalls))
    return {"mean_top3": mean_top3, "hit_rate": hit_rate, "ci": ci, "n": len(recalls)}


def _median(xs):
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 0a: measure where the SSM state-signal loss is "
                    "(readout mean-pool vs backbone collapse). No training.")
    p.add_argument("--questions-parquet", default=DEFAULT_QUESTIONS_PARQUET)
    p.add_argument("--docs-parquet", default=DEFAULT_DOCS_PARQUET)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH)
    p.add_argument("--max-queries", type=int, default=200)
    p.add_argument("--neg-per-query", type=int, default=14)
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    questions_path = Path(args.questions_parquet)
    docs_path = Path(args.docs_parquet)
    backbone_path = Path(args.backbone)
    for pth, label in ((questions_path, "questions"), (docs_path, "documents"),
                       (backbone_path, "backbone")):
        if not pth.exists():
            print(f"ERROR: {label} not found at {pth}", file=sys.stderr)
            return 1

    print(f"Loading questions from {questions_path} (max {args.max_queries})",
          flush=True)
    questions = load_questions(str(questions_path), GOLD_CATEGORIES, args.max_queries)
    if not questions:
        print("ERROR: no gold-bearing questions loaded", file=sys.stderr)
        return 1
    gold_doc_ids = {d for q in questions for d in q["expected_doc_ids"]}
    print(f"  {len(questions)} questions, {len(gold_doc_ids)} unique gold doc_ids",
          flush=True)

    print(f"Indexing documents parquet {docs_path} (doc_id column only)", flush=True)
    doc_idx, all_doc_ids = build_doc_index(str(docs_path))
    print(f"  {len(all_doc_ids)} docs indexed", flush=True)
    print("Memory-mapping documents parquet title+content (lazy)", flush=True)
    docs_tbl = open_docs_table(str(docs_path))

    print(f"Loading frozen backbone from {backbone_path}", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  backbone: {sum(p.numel() for p in backbone.parameters()):,} params (frozen)",
          flush=True)
    print("Loading embedder (bge-small, on-demand)", flush=True)
    embedder = build_embedder("on-demand")

    dev = _resolve_device(args.device)
    parser = MarkdownParser()
    chunker = HierarchicalChunker()

    # Rebuild the SAME candidate plans the generator builds (seed 0, neg=14 ->
    # K=15), so the slots match the Phase B traces. One WM instance reused
    # across queries with reset() between them (mirrors build_records:267-308).
    rng = np.random.default_rng(args.seed)
    non_gold = [d for d in all_doc_ids if d not in gold_doc_ids]
    plans: list[tuple[dict, list[str]]] = []
    max_k = 1
    for q in questions:
        gold = q["expected_doc_ids"]
        gold_set = set(gold)
        k_neg = min(args.neg_per_query, len(non_gold))
        neg_ids = list(rng.choice(non_gold, size=k_neg, replace=False))
        cand = list(gold) + [d for d in neg_ids if d not in gold_set]
        rng.shuffle(cand)
        plans.append((q, cand))
        max_k = max(max_k, len(cand))

    wm = WorkingMemory(backbone, embedder=embedder, ring_capacity=max_k)
    ld_head = LatentDynamicsHead()  # parameter-free project; matches the generator

    doc_cache: dict[str, torch.Tensor] = {}

    def doc_vec(doc_id: str) -> torch.Tensor | None:
        if doc_id in doc_cache:
            return doc_cache[doc_id]
        tc = get_doc(docs_tbl, doc_idx, doc_id)
        if tc is None:
            return None
        v = embed_doc(doc_id, tc[0], tc[1], parser, chunker, embedder, dev)
        doc_cache[doc_id] = v
        return v

    # Per-query capture: the raw slot.h (4 layers x [1,16,384] fp16) + the doc
    # bge input + the query bge + labels. We materialize all four
    # representations per slot so the analysis is offline (no second WM pass).
    D_STATE = 16
    D_MODEL = 384
    N_LAYERS = 4
    records: list[dict] = []
    t0 = time.time()
    for qi, (q, cand_ids) in enumerate(plans):
        cand_vecs: list[tuple[str, torch.Tensor]] = []
        for d in cand_ids:
            v = doc_vec(d)
            if v is not None:
                cand_vecs.append((d, v))
        if not cand_vecs:
            continue
        wm.reset()
        for d, v in cand_vecs:
            wm.step(v, source_id=d, text=d)
        ring = wm.ring_buffer()
        if any(s.h is None for s in ring):
            raise RuntimeError(
                "a ring slot has h=None (Phase A should capture h for every "
                "slot when ring_capacity>0). Backbone checkpoint mismatch?")
        K = len(ring)
        gold_set = set(q["expected_doc_ids"])
        labels = np.array([1 if str(s.source_id) in gold_set else 0 for s in ring],
                          dtype=np.int64)
        if labels.sum() == 0:
            continue  # lost the gold doc(s) -- skip (no positive to recall)

        # z_mean_last [K,384] -- the mean-pool (reproduce the Phase B z_i).
        z_mean = np.stack([
            ld_head.project(s.h).squeeze(0).detach().to("cpu").to(torch.float32).numpy()
            for s in ring
        ]).astype(np.float32)  # [K,384]

        # z_chan_last [K,16,384] -- the raw last-layer state per channel.
        z_chan = np.stack([
            s.h[-1].detach().to("cpu").to(torch.float32).reshape(D_STATE, D_MODEL).numpy()
            for s in ring
        ]).astype(np.float32)  # [K,16,384]

        # z_flat_last [K,6144] and z_flat_all [K,24576].
        z_flat_last = z_chan.reshape(K, D_STATE * D_MODEL)            # [K,6144]
        z_flat_all = np.stack([
            np.concatenate([
                layer.detach().to("cpu").to(torch.float32).reshape(-1).numpy()
                for layer in s.h
            ])
            for s in ring
        ]).astype(np.float32)  # [K, N_LAYERS*D_STATE*D_MODEL]

        # doc bge input [K,384] -- the wm.step inputs (the doc baseline).
        doc = np.stack([
            v.detach().to("cpu").to(torch.float32).squeeze(0).numpy()
            for d, v in cand_vecs
        ]).astype(np.float32)  # [K,384]

        qv = np.asarray(embedder.encode([q["question"]])[0], dtype=np.float32)

        records.append({
            "query": qv,                 # [384]
            "labels": labels,            # [K]
            "z_mean": z_mean,            # [K,384]
            "z_chan": z_chan,            # [K,16,384]
            "z_flat_last": z_flat_last,  # [K,6144]
            "z_flat_all": z_flat_all,    # [K,24576]
            "doc": doc,                  # [K,384]
        })
        if (qi + 1) % 20 == 0:
            print(f"  captured {qi + 1}/{len(plans)} queries "
                  f"({time.time() - t0:.1f}s, cache={len(doc_cache)})", flush=True)

    if not records:
        print("ERROR: no records captured (all candidates failed to load?)",
              file=sys.stderr)
        return 1
    print(f"Captured {len(records)} queries.", flush=True)

    # ── Analysis ──
    # 1. Across-slot std per representation (std over K slots, mean over D dims),
    #    median across queries. The doc baseline is the normalizer.
    doc_stds = [float(r["doc"].std(axis=0).mean()) for r in records]
    doc_std_med = _median(doc_stds)
    zmean_stds = [float(r["z_mean"].std(axis=0).mean()) for r in records]
    zmean_std_med = _median(zmean_stds)
    zflat_last_stds = [float(r["z_flat_last"].std(axis=0).mean()) for r in records]
    zflat_last_std_med = _median(zflat_last_stds)
    zflat_all_stds = [float(r["z_flat_all"].std(axis=0).mean()) for r in records]
    zflat_all_std_med = _median(zflat_all_stds)

    # Per-channel std (z_chan [K,16,384]): std over K -> [16,384]; mean over 384
    # -> [16] per-channel std; report the MAX and MEDIAN channel (median over
    # queries of the per-query max-channel std).
    per_q_max_chan = []
    per_q_med_chan = []
    for r in records:
        chan_std = r["z_chan"].std(axis=0).mean(axis=1)  # [16]
        per_q_max_chan.append(float(chan_std.max()))
        per_q_med_chan.append(float(np.median(chan_std)))
    max_chan_std_med = _median(per_q_max_chan)
    med_chan_std_med = _median(per_q_med_chan)

    def ratio(x):
        return x / doc_std_med if doc_std_med > 0 else float("nan")

    # 2. Top-3 recall (the actual gate statistic) where dimensionally computable
    #    (rep dim == query dim 384): doc baseline, mean-pool, and EACH of the 16
    #    channels individually. The per-channel top-3 is the DECISIVE test for
    #    "did the mean-pool cancel opposing-sign channel signal".
    doc_top3 = _mean_top3(records, "doc")
    zmean_top3 = _mean_top3(records, "z_mean")
    # per-channel top-3: build a temporary record list with rep = channel c
    chan_top3 = []
    for cc in range(D_STATE):
        chan_records = []
        for r in records:
            chan_records.append({
                "query": r["query"],
                "labels": r["labels"],
                f"chan{cc}": r["z_chan"][:, cc, :],  # [K,384]
            })
        t = _mean_top3(chan_records, f"chan{cc}")
        if t is not None:
            chan_top3.append((cc, t["mean_top3"]))
    chan_top3_sorted = sorted(chan_top3, key=lambda kv: kv[1], reverse=True)
    best_chan_idx, best_chan_top3 = chan_top3_sorted[0] if chan_top3 else (-1, float("nan"))
    median_chan_top3 = _median([t for _, t in chan_top3])

    # ── Report ──
    print("\n" + "=" * 72)
    print("PHASE 0a: state-signal distribution across representations")
    print("=" * 72)
    print(f"  queries captured: {len(records)}   (doc_std baseline median = {doc_std_med:.5f})")
    print()
    print("Across-slot std (median over queries; std over K slots, mean over D dims):")
    print(f"  doc         [K,384]   std = {doc_std_med:.5f}   ratio = {ratio(doc_std_med):.3f}x   (the normalizer)")
    print(f"  z_mean_last [K,384]   std = {zmean_std_med:.5f}   ratio = {ratio(zmean_std_med):.3f}x   "
          f"(Phase B baseline: 0.068x)")
    print(f"  z_chan_last max-chan  std = {max_chan_std_med:.5f}   ratio = {ratio(max_chan_std_med):.3f}x   "
          f"(median channel ratio = {ratio(med_chan_std_med):.3f}x)")
    print(f"  z_flat_last [K,6144]  std = {zflat_last_std_med:.5f}   ratio = {ratio(zflat_last_std_med):.3f}x")
    print(f"  z_flat_all  [K,24576] std = {zflat_all_std_med:.5f}   ratio = {ratio(zflat_all_std_med):.3f}x")
    print()
    print("Top-3 recall (the gate statistic; cosine vs the 384-d query, val split):")
    if doc_top3:
        print(f"  doc baseline  top-3 = {doc_top3['mean_top3']:.3f}  "
              f"(hit {doc_top3['hit_rate']:.2f}, ci [{doc_top3['ci'][0]:.2f},{doc_top3['ci'][1]:.2f}], n={doc_top3['n']})  "
              f"(Phase B baseline: 0.938)")
    if zmean_top3:
        print(f"  z_mean_last   top-3 = {zmean_top3['mean_top3']:.3f}  "
              f"(hit {zmean_top3['hit_rate']:.2f}, ci [{zmean_top3['ci'][0]:.2f},{zmean_top3['ci'][1]:.2f}], n={zmean_top3['n']})  "
              f"(Phase B baseline: 0.196)")
    print(f"  z_chan_last   best channel (c{best_chan_idx}) top-3 = {best_chan_top3:.3f}   "
          f"(median channel top-3 = {median_chan_top3:.3f})")
    if len(chan_top3_sorted) >= 4:
        top4 = ", ".join(f"c{cc}={t:.3f}" for cc, t in chan_top3_sorted[:4])
        print(f"                top-4 channels: {top4}")
    print()

    # ── GATE 0a decision ──
    # TWO distinct signals, do not conflate them:
    #   (1) DOC-IDENTITY VARIANCE -- does the state vary across docs at all?
    #       Measured by the across-slot std ratio (per-channel / flat). The
    #       mean-pool z_mean is near-constant (0.068x) because it averages 16
    #       opposing-sign channels; a per-channel or flat view reveals whether
    #       the state actually moves. If it does (ratio >> 0.068x), the backbone
    #       is NOT collapsed -- the mean-pool readout destroyed the VARIANCE, and
    #       a learned readout (Phase 0b) has signal to work with. -> 0b.
    #   (2) PER-CHANNEL QUERY-RELEVANCE -- does any SINGLE channel's cosine-vs-
    #       query already recover the gold doc (top-3 -> 0.938)? If yes, the
    #       readout is almost trivial (pick that channel). If no (best channel
    #       ~0.196-0.25), NO channel is pre-aligned with the bge query -- expected,
    #       since the backbone was never trained to align state channels with bge
    #       queries. A learned readout must MIX channels to find a query-relevant
    #       direction; whether a linear/MLP readout can do that is exactly what
    #       Phase 0b tests. It is NOT a NO-GO for 0b.
    # NO-GO (-> Phase 1 backbone fine-tune) is ONLY when the state is near-
    #   constant in EVERY representation (every std ratio ~0.068x) -- i.e. the
    #   state truly collapsed under the JEPA objective and no readout has signal.
    GO_STD_RATIO_THRESH = 0.15  # materially above the 0.068x mean-pool

    state_varies = (ratio(max_chan_std_med) >= GO_STD_RATIO_THRESH
                    or ratio(zflat_last_std_med) >= GO_STD_RATIO_THRESH
                    or ratio(zflat_all_std_med) >= GO_STD_RATIO_THRESH)
    channel_prealigned = (not np.isnan(best_chan_top3)) and best_chan_top3 >= 0.5

    print("-" * 72)
    print("GATE 0a DECISION")
    print("-" * 72)
    print(f"  (1) doc-identity variance: max-chan std ratio {ratio(max_chan_std_med):.3f}x, "
          f"flat_last {ratio(zflat_last_std_med):.3f}x, flat_all {ratio(zflat_all_std_med):.3f}x "
          f"(>= {GO_STD_RATIO_THRESH}x means state moves) -> state_varies={state_varies}")
    print(f"  (2) per-channel query-relevance: best channel c{best_chan_idx} top-3 "
          f"{best_chan_top3:.3f} (>= 0.5 means a channel is pre-aligned) -> "
          f"channel_prealigned={channel_prealigned}")
    if state_varies:
        print(f"\n  -> GO to Phase 0b (learned StateReadout). The state varies across docs")
        print(f"     (the backbone did NOT collapse -- the mean-pool readout destroyed the")
        print(f"     variance). A learned readout has signal to work with.")
        if channel_prealigned:
            print(f"     BONUS: channel c{best_chan_idx} is already query-aligned (top-3 "
                  f"{best_chan_top3:.3f}) -- 0b should clear the gate easily.")
        else:
            print(f"     CAVEAT: no single channel is pre-aligned with the bge query (best top-3")
            print(f"     {best_chan_top3:.3f} ~ random) -- 0b's learned projection must MIX channels")
            print(f"     to find a query-relevant direction. The linear-vs-MLP choice is load-bearing.")
        print(f"     Skip the backbone fine-tune (Phase 1) -- the near-constancy was the readout,")
        print(f"     not the backbone.")
    else:
        print(f"\n  -> NO-GO: the state is near-constant in EVERY representation (every std ratio")
        print(f"     ~0.068x). The recurrent state truly collapsed under the JEPA objective -- no")
        print(f"     readout can recover signal that isn't there. Proceed to Phase 1 (warm-started")
        print(f"     backbone doc-identity fine-tune).")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())