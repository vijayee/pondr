"""Phase 1: from-scratch ReferenceSSM backbone, joint L_relevance + L_trajectory.

The STRM state-trajectory vision ([[pondr-strm-transformer-relocator-drift]])
hinged on the SSM recurrent state ``h_t`` carrying query-relevance signal.
Phase B GATE 1 ([[pondr-strm-phaseb-ht-gate-no-go]]) + Phase 0b GATE 0b
([[pondr-strm-phase0b-gate-no-go]]) showed the JEPA-trained 19.5M backbone
encodes doc-identity mostly QUERY-ORTHOGONALLY: a learned readout over the raw
state maxes at top-3 0.564 (sub-gate), and the fixed mean-pool ``z_i`` is
near-constant across docs (0.196 == random). The backbone was trained for a
DIFFERENT purpose (JEPA next-turn-summary prediction on DialogSum), so its
state never saw query/relevance signal. This trainer builds a backbone FOR the
relevance purpose from scratch -- "don't fight the noise of another
architecture with a different purpose" (user).

**From scratch, NOT warm-start.** Fresh ``JGSBackbone(BackboneConfig())`` --
ReferenceSSM, d_model=384, n_layers=4, d_state=16, pred_dim=384, fp32 on the
``reference`` backend (mamba3-cuda FAILS [[mamba3-cuda-build-fails]]). The 19.5M
checkpoint is NEVER loaded; ``DEFAULT_BACKBONE_PATH`` is untouched (binding
constraint: don't break existing functionality). The new backbone is a SEPARATE
artifact at ``data/training/strm_backbone_relevance/backbone_v2.pt``.

**Joint loss** ``L = L_relevance + lambda_traj * L_trajectory``:

  * ``L_relevance`` -- per ERAG query, the candidate set = gold docs +
    ``--neg-per-query`` negatives, SHUFFLED (the same plan
    ``generate_relevance_data.build_records`` builds). Step the backbone over
    the candidate sequence from a ZEROED state -- CUMULATIVE state, so
    ``slot_k.h`` = state after ingesting docs 1..k, matching the gate. The
    per-step ``z_k = mean over d_state of the last layer's state`` [384] IS the
    gate's ``LatentDynamicsHead.project(slot.h)`` (parameter-free). Multi-positive
    InfoNCE: ``L = logsumexp(cos(q, z_all)/T) - logsumexp(cos(q, z_gold)/T)`` --
    trains the MEAN-POOL z_k itself to be query-relevant (not a readout on top
    of a frozen backbone; 0b showed that maxes at 0.564). This directly optimizes
    the P1 probe metric (``bge(query, z_k)`` top-3 recall).

  * ``L_trajectory`` -- over a doc's SECTION sequence (MarkdownParser +
    HierarchicalChunker + bge per-section), ingest sections 0..K-1 from a
    zeroed state and predict section_{k+1}'s bge embedding from the last-layer
    output via ``JEPAPredictor`` (reuses ``jepa_contrastive_loss``, in-batch
    negatives). Section sequences have real structure (a doc's sections are
    topically coherent), so next-section prediction forces the state to carry
    the running prefix -> non-trivial recurrence (avoids passthrough collapse
    g~1 where state = W_B(doc_k), prefix-independent). The SHUFFLED candidate
    sequence is degenerate for this (next is random -> the gradient pushes the
    state toward the marginal mean -> collapses it); NEVER use it for
    L_trajectory.

**Trainer forward path = DIRECT SSM (identity-instance).** The gate
(``generate_relevance_data.py``) drives ``WorkingMemory`` with DEFAULT random
instance params (``input_proj``=random LoRALinear, ``state_lora``~0); the
trainer instead drives ``backbone.layers[i].step`` directly -- equivalent to
identity ``input_proj`` + zero ``state_lora``, which removes the
random-projection confound. The P1 probe (this script) uses the SAME direct-SSM
path, so it is self-consistent. The formal Phase B gate re-run (task #33) needs
a new ``--identity-instance`` flag on ``generate_relevance_data.py`` (default
off -> byte-identical) so it measures the same pure-backbone path; re-baseline
the old 19.5M under identity then. Serve wiring is a FUTURE plan, gated on the
Phase B GO (this plan does NOT wire the new backbone into ``build_ponder`` /
``serve_ponder``).

**P1 cheap probe** (``--probe``, gates the full 511K run): ``--max-queries 80``
(30 train + 50 held-out), ~300 steps, AdamW lr 3e-4 / warmup 200 / wd 0.1 /
grad_accum 2, lambda_traj 1.0, T 0.1. Every ``--eval-every`` steps, measure on
the 50 held-out queries: (1) across-slot std of ``z_k`` (climb from 0.00169
toward the doc-embedding baseline 0.0248); (2) ``bge(query, z_k)`` top-3 recall
(climb from 0.196 toward 0.938 -- the key metric). std~0 + top-3 stuck -> NO-GO
(true collapse); top-3 climbs -> scale to the full 511K-doc ~3000-step run.

Usage (on the RunPod L4 pod, repo at /workspace/Pondr):

    python scripts/train_backbone_relevance.py --probe
    python scripts/train_backbone_relevance.py --probe --eval-every 25 --steps 300

Generate the formal gate traces on the new backbone AFTER a GO (task #33):
    python scripts/generate_relevance_data.py --backbone <NEW_CKPT> --identity-instance --retrace
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import HierarchicalChunker  # noqa: E402
from src.ingestion.parsers import MarkdownParser  # noqa: E402
from src.subconscious.backbone import JGSBackbone  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.training.jepa_loss import jepa_contrastive_loss  # noqa: E402
from src.subconscious.training.routing_training import (  # noqa: E402
    _resolve_device,
    build_embedder,
)

# Reuse the ERAG data helpers + the gate's candidate-plan logic verbatim from
# the generator so the probe's candidate sets match the gate's.
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

DEFAULT_OUTPUT = "data/training/strm_backbone_relevance/backbone_v2.pt"


# ── direct-SSM forward (identity-instance path) ──

def step_sequence(backbone: JGSBackbone, x_seq: torch.Tensor,
                  device: torch.device, dtype: torch.dtype):
    """Step the backbone's SSM layers directly over a sequence from zero state.

    ``x_seq`` is ``[K, d_model]`` (one trajectory; batch=1). Returns
    ``(last_states, outputs)`` where ``last_states`` is ``[K, d_state, d_model]``
    (the last SSM layer's recurrent state per step -- the gate's ``slot.h[-1]``)
    and ``outputs`` is ``[K, d_model]`` (the last layer's output ``y`` per step,
    what ``JEPAPredictor`` reads). This is the ``WorkingMemory.step`` path with
    identity ``input_proj`` + zero ``state_lora`` -- the random-projection
    confound removed. ``z_k = last_states.mean(dim=1)`` reproduces
    ``LatentDynamicsHead.project`` (mean over d_state).
    """
    batch = 1
    states = [layer.init_state(batch, device, dtype) for layer in backbone.layers]
    last_states = []
    outputs = []
    for t in range(x_seq.shape[0]):
        h = x_seq[t].unsqueeze(0)            # [1, d_model]
        new_states = []
        for i, layer in enumerate(backbone.layers):
            h, s = layer.step(h, states[i])
            new_states.append(s)
        states = new_states
        last_states.append(states[-1].squeeze(0))   # [d_state, d_model]
        outputs.append(h.squeeze(0))                # [d_model]
    return torch.stack(last_states), torch.stack(outputs)


def z_from_states(last_states: torch.Tensor) -> torch.Tensor:
    """``z_k = mean over d_state of the last layer state`` -> ``[K, d_model]``.

    Mirrors ``LatentDynamicsHead.project`` (last layer, mean over d_state) so
    the probe's ``z_k`` is the SAME vector the gate measures / the z-head trains
    on.
    """
    return last_states.float().mean(dim=1)        # [K, d_model]


# ── losses ──

def relevance_loss(z: torch.Tensor, q: torch.Tensor, gold_mask: torch.Tensor,
                   temperature: float) -> torch.Tensor:
    """Multi-positive InfoNCE: rank gold z_k above neg z_k for the query.

    ``z`` ``[K, 384]``, ``q`` ``[384]``, ``gold_mask`` ``[K]`` bool. Returns
    ``logsumexp(cos(q, z_all)/T) - logsumexp(cos(q, z_gold)/T)`` (0 if no
    gold). Trains the mean-pool z_k to align with the query for gold docs --
    the P1 probe metric (``bge(query, z_k)`` top-3) directly.
    """
    if gold_mask.sum() == 0:
        return z.new_zeros(())
    z_n = F.normalize(z, dim=-1)
    q_n = F.normalize(q, dim=-1)
    logits = (z_n @ q_n) / temperature            # [K]
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[gold_mask], dim=0)


def trajectory_loss(backbone: JGSBackbone, outputs: torch.Tensor,
                    section_embs: torch.Tensor, temperature: float) -> torch.Tensor:
    """JEPA next-section prediction over a doc's section sequence.

    ``outputs`` ``[K, d_model]`` (last-layer y per step), ``section_embs``
    ``[K, 384]`` (the bge section embeddings that WERE the inputs). Predict
    section_{k+1} from output_k via ``JEPAPredictor`` (reuses
    ``jepa_contrastive_loss`` with in-batch negatives). 0 if the doc has <2
    sections (no next to predict).
    """
    if outputs.shape[0] < 2:
        return outputs.new_zeros(())
    pred = backbone.predictor(outputs[:-1])      # [K-1, pred_dim]
    actual = section_embs[1:]                    # [K-1, 384]
    return jepa_contrastive_loss(pred, actual, section_embs, temperature)


# ── P1 probe metric (matches _probe_relevance_bge_baseline._cos_top3_recall) ──

def cos_top3_recall(qvec: np.ndarray, slot_vecs: np.ndarray,
                    labels: list[int]) -> tuple[float, bool] | None:
    """Per-query top-3 recall + hit (all gold in top-3), bge-baseline metric.

    ``qvec`` ``[384]``, ``slot_vecs`` ``[K, 384]``, ``labels`` ``[K]`` (1=gold).
    Returns ``(n_gold_in_top3 / n_gold, all_gold_in_top3)`` or ``None`` if no
    gold. Identical to ``_probe_relevance_bge_baseline._cos_top3_recall`` so the
    probe's numbers sit on the same 0.196 (random) -> 0.938 (doc baseline) scale.
    """
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    d = slot_vecs / (np.linalg.norm(slot_vecs, axis=1, keepdims=True) + 1e-9)
    sims = d @ q
    gold_idx = [i for i, l in enumerate(labels) if l == 1]
    n_gold = len(gold_idx)
    if n_gold == 0:
        return None
    k_top = min(3, len(sims))
    top = set(np.argsort(-sims)[:k_top].tolist())
    n_in = sum(1 for i in gold_idx if i in top)
    return n_in / n_gold, n_in == n_gold


def evaluate(backbone, val_records, device, dtype) -> dict:
    """P1 probe metrics on held-out queries (no grad).

    ``val_records`` is a list of ``(query_emb[384], doc_embs[K,384], labels[K],
    candidate_ids[K])``. Returns ``{mean_top3_recall, hit_rate, std_z,
    std_doc, std_ratio}``. ``std_z`` = mean over queries of (mean over dims of
    std over the K z_k slots); ``std_doc`` = same on the raw doc embeddings (the
    0.0248 baseline); ``std_ratio`` = std_z / std_doc (the 0.068x Phase B
    baseline).
    """
    backbone.eval()
    recalls = []
    hits = 0
    std_z_list = []
    std_doc_list = []
    with torch.no_grad():
        for q_emb, doc_embs, labels, _cand_ids in val_records:
            x = torch.from_numpy(doc_embs).to(device=device, dtype=dtype)  # [K,384]
            last_states, _out = step_sequence(backbone, x, device, dtype)
            z = z_from_states(last_states).cpu().numpy()                  # [K,384]
            r = cos_top3_recall(q_emb, z, labels)
            if r is not None:
                recalls.append(r[0])
                if r[1]:
                    hits += 1
            std_z_list.append(float(z.std(axis=0).mean()))
            std_doc_list.append(float(doc_embs.std(axis=0).mean()))
    mean_top3 = sum(recalls) / len(recalls) if recalls else 0.0
    hit_rate = hits / len(recalls) if recalls else 0.0
    std_z = sum(std_z_list) / len(std_z_list) if std_z_list else 0.0
    std_doc = sum(std_doc_list) / len(std_doc_list) if std_doc_list else 0.0
    std_ratio = std_z / std_doc if std_doc > 0 else 0.0
    return {"mean_top3_recall": mean_top3, "hit_rate": hit_rate,
            "std_z": std_z, "std_doc": std_doc, "std_ratio": std_ratio}


# ── data prep ──

def _embed_sections(doc_id, title, content, parser, chunker, embedder) -> np.ndarray:
    """Per-section bge embeddings ``[N, 384]`` for a doc (NOT mean-pooled).

    Mirrors ``embed_doc`` up to the mean-pool, returning the per-section matrix
    instead. A doc with no parseable sections falls back to the title (1 section)
    so L_trajectory always has at least 1 step (2 with the cold-start fallback).
    """
    text = (title + "\n" + content) if title else content
    sec_texts: list[str] = []
    if text and text.strip():
        parsed = parser.parse_text(text, source_path=doc_id)
        parsed = chunker.chunk(parsed)
        for s in parsed.sections:
            sec = (s.heading + "\n" + s.content) if s.heading else s.content
            if sec and sec.strip():
                sec_texts.append(sec)
    if not sec_texts:
        sec_texts = [title or doc_id]
    vecs = embedder.encode(sec_texts)
    return np.asarray(vecs, dtype=np.float32)            # [N, 384]


def build_plans(questions, all_doc_ids, neg_per_query, seed):
    """Replicate ``generate_relevance_data.build_records``' candidate plans.

    Per query: gold + ``neg_per_query`` random non-gold, shuffled, with the SAME
    rng (``np.random.default_rng(seed)``) so the probe's candidates match the
    gate's. Returns ``[(query, [candidate_doc_id, ...]), ...]``.
    """
    rng = np.random.default_rng(seed)
    gold_doc_ids = {d for q in questions for d in q["expected_doc_ids"]}
    non_gold = [d for d in all_doc_ids if d not in gold_doc_ids]
    plans = []
    for q in questions:
        gold = q["expected_doc_ids"]
        gold_set = set(gold)
        k_neg = min(neg_per_query, len(non_gold))
        neg_ids = list(rng.choice(non_gold, size=k_neg, replace=False))
        cand = list(gold) + [d for d in neg_ids if d not in gold_set]
        rng.shuffle(cand)
        plans.append((q, cand))
    return plans


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 1 from-scratch backbone: joint L_relevance + L_trajectory")
    p.add_argument("--probe", action="store_true",
                   help="cheap P1 probe (80 queries, ~300 steps) that gates the full run")
    p.add_argument("--questions-parquet", default=DEFAULT_QUESTIONS_PARQUET)
    p.add_argument("--docs-parquet", default=DEFAULT_DOCS_PARQUET)
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="checkpoint path (default: data/training/strm_backbone_relevance/backbone_v2.pt)")
    p.add_argument("--max-queries", type=int, default=80)
    p.add_argument("--n-val-queries", type=int, default=50)
    p.add_argument("--neg-per-query", type=int, default=14)
    p.add_argument("--n-traj-docs", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-queries", type=int, default=8)
    p.add_argument("--batch-traj", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lambda-traj", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--categories", default=",".join(GOLD_CATEGORIES))
    args = p.parse_args()

    if args.probe:
        print("=" * 64, flush=True)
        print("P1 PROBE MODE -- cheap gate for the full 511K run.", flush=True)
        print(f"  {args.max_queries} queries ({args.max_queries - args.n_val_queries} "
              f"train / {args.n_val_queries} held-out), {args.steps} steps.", flush=True)
        print("  GO/NO-GO readout at the end decides whether to scale up.", flush=True)
        print("=" * 64, flush=True)

    dev = _resolve_device(args.device)
    dtype = torch.float32                 # fp32 -- the 19.5M recipe
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── fresh backbone (NO 19.5M load) ──
    # BackboneConfig() defaults match the 19.5M arch exactly (d_model=384,
    # n_layers=4, d_state=16, pred_dim=384, ssm_backend="reference") -- so the
    # new checkpoint is shape-identical and a future head reuse is possible, but
    # NO weights are loaded (the whole point of from-scratch).
    cfg = BackboneConfig()
    backbone = JGSBackbone(cfg).to(device=dev, dtype=dtype)
    backbone.train()
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Fresh backbone: {n_params:,} params, fp32, {dev} (NO 19.5M load)", flush=True)

    # ── load ERAG data ──
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    print(f"Loading questions ({categories}, max {args.max_queries})", flush=True)
    questions = load_questions(args.questions_parquet, categories, args.max_queries)
    if len(questions) < args.n_val_queries + 5:
        print(f"ERROR: only {len(questions)} questions -- need >= n_val+5 train",
              file=sys.stderr)
        return 1
    print(f"Indexing docs parquet (doc_id column)", flush=True)
    doc_idx, all_doc_ids = build_doc_index(args.docs_parquet)
    print(f"  {len(all_doc_ids)} docs indexed; memory-mapping title+content", flush=True)
    docs_tbl = open_docs_table(args.docs_parquet)
    embedder = build_embedder("on-demand")
    parser = MarkdownParser()
    chunker = HierarchicalChunker()

    plans = build_plans(questions, all_doc_ids, args.neg_per_query, args.seed)

    # Split: last n_val_queries held out, rest train (deterministic).
    n_val = args.n_val_queries
    train_plans = plans[:-n_val] if n_val < len(plans) else []
    val_plans = plans[-n_val:] if n_val < len(plans) else plans
    print(f"  {len(train_plans)} train / {len(val_plans)} held-out queries", flush=True)

    # ── embed candidate docs + queries (cache) ──
    print(f"Embedding candidate docs (bge mean-pool) + queries...", flush=True)
    t0 = time.time()

    def doc_vec(doc_id):
        tc = get_doc(docs_tbl, doc_idx, doc_id)
        if tc is None:
            return None
        v = embed_doc(doc_id, tc[0], tc[1], parser, chunker, embedder, "cpu")
        return v.squeeze(0).numpy().astype(np.float32)   # [384]

    doc_cache: dict[str, np.ndarray] = {}

    def cached_doc_vec(doc_id):
        if doc_id not in doc_cache:
            v = doc_vec(doc_id)
            if v is not None:
                doc_cache[doc_id] = v
        return doc_cache.get(doc_id)

    def build_probe_records(plans_subset):
        recs = []
        for q, cand in plans_subset:
            vecs, labels, ids = [], [], []
            for d in cand:
                v = cached_doc_vec(d)
                if v is None:
                    continue
                vecs.append(v)
                labels.append(1 if d in set(q["expected_doc_ids"]) else 0)
                ids.append(d)
            if not vecs or sum(labels) == 0:
                continue
            qv = np.asarray(embedder.encode([q["question"]])[0], dtype=np.float32)
            recs.append((qv, np.stack(vecs), labels, ids))
        return recs

    train_records = build_probe_records(train_plans)
    val_records = build_probe_records(val_plans)
    print(f"  {len(train_records)} train / {len(val_records)} val records, "
          f"{len(doc_cache)} unique docs cached ({time.time()-t0:.1f}s)", flush=True)
    if not train_records:
        print("ERROR: no train records built (every train query's gold docs are "
              "missing from the parquet?) -- cannot train.", file=sys.stderr)
        return 1
    if not val_records:
        print("ERROR: no held-out val records built -- cannot compute the P1 "
              "probe metrics.", file=sys.stderr)
        return 1

    # ── L_trajectory section trajectories ──
    print(f"Embedding sections for {args.n_traj_docs} trajectory docs...", flush=True)
    t0 = time.time()
    traj_pool = list(doc_cache.keys())          # candidate docs already parsed
    if len(traj_pool) < args.n_traj_docs:
        # backfill from the non-gold pool (sample raw ids + embed sections)
        extra = [d for d in all_doc_ids if d not in doc_cache]
        rng.shuffle(extra)
        for d in extra[:args.n_traj_docs - len(traj_pool)]:
            cached_doc_vec(d)
        traj_pool = list(doc_cache.keys())
    rng.shuffle(traj_pool)
    traj_docs = traj_pool[:args.n_traj_docs]
    traj_sections: list[np.ndarray] = []        # each [N, 384]
    for d in traj_docs:
        tc = get_doc(docs_tbl, doc_idx, d)
        if tc is None:
            continue
        secs = _embed_sections(d, tc[0], tc[1], parser, chunker, embedder)
        if secs.shape[0] >= 2:                  # need >=2 sections for next-step pred
            traj_sections.append(secs)
    print(f"  {len(traj_sections)} section trajectories ({time.time()-t0:.1f}s)", flush=True)
    if not traj_sections:
        print("WARNING: no section trajectories -- L_trajectory disabled", file=sys.stderr)

    # ── optimizer ──
    optim = torch.optim.AdamW(
        backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr

    # ── baseline (step 0) eval ──
    base = evaluate(backbone, val_records, dev, dtype)
    print(f"[step 0] top3={base['mean_top3_recall']:.3f} hit={base['hit_rate']:.2f} "
          f"std_z={base['std_z']:.5f} std_doc={base['std_doc']:.5f} "
          f"ratio={base['std_ratio']:.3f}x", flush=True)

    log = [{"step": 0, **base}]
    best_top3 = base["mean_top3_recall"]
    best_step = 0

    # ── training loop ──
    print(f"Training {args.steps} steps (lr {args.lr}, wd {args.weight_decay}, "
          f"lambda_traj {args.lambda_traj}, T {args.temperature}, "
          f"batch q{args.batch_queries}/t{args.batch_traj}, accum {args.grad_accum})",
          flush=True)
    t0 = time.time()
    step = 0
    qi = 0                              # train-query cursor
    ti = 0                              # trajectory cursor
    while step < args.steps:
        # evaluate() flips the backbone to eval mode; restore train mode so the
        # SSM layers (and any future dropout) train correctly. Without this every
        # post-eval training step would run in eval mode.
        backbone.train()
        optim.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            # L_relevance: a batch of train queries.
            rel_loss = torch.zeros((), device=dev, dtype=dtype)
            n_rel = 0
            for _ in range(args.batch_queries):
                if not train_records:
                    break
                qv, doc_embs, labels, _ids = train_records[qi % len(train_records)]
                qi += 1
                x = torch.from_numpy(doc_embs).to(device=dev, dtype=dtype)
                q = torch.from_numpy(qv).to(device=dev, dtype=dtype)
                last_states, _out = step_sequence(backbone, x, dev, dtype)
                z = z_from_states(last_states)
                gold = torch.tensor(labels, dtype=torch.bool, device=dev)
                rel_loss = rel_loss + relevance_loss(z, q, gold, args.temperature)
                n_rel += 1
            if n_rel:
                rel_loss = rel_loss / n_rel

            # L_trajectory: a batch of section trajectories.
            traj_loss = torch.zeros((), device=dev, dtype=dtype)
            n_tr = 0
            if traj_sections:
                for _ in range(args.batch_traj):
                    secs = traj_sections[ti % len(traj_sections)]
                    ti += 1
                    xs = torch.from_numpy(secs).to(device=dev, dtype=dtype)
                    _ls, outs = step_sequence(backbone, xs, dev, dtype)
                    traj_loss = traj_loss + trajectory_loss(
                        backbone, outs, xs, args.temperature)
                    n_tr += 1
                if n_tr:
                    traj_loss = traj_loss / n_tr

            loss = rel_loss + args.lambda_traj * traj_loss
            (loss / args.grad_accum).backward()
        # gradient step
        for g in optim.param_groups:
            g["lr"] = lr_at(step)
        optim.step()
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(backbone, val_records, dev, dtype)
            log.append({"step": step, **m})
            elapsed = time.time() - t0
            print(f"[step {step:4d}] top3={m['mean_top3_recall']:.3f} "
                  f"hit={m['hit_rate']:.2f} std_z={m['std_z']:.5f} "
                  f"ratio={m['std_ratio']:.3f}x "
                  f"({elapsed:.0f}s, {elapsed/max(step,1):.2f}s/step)", flush=True)
            if m["mean_top3_recall"] > best_top3:
                best_top3 = m["mean_top3_recall"]
                best_step = step

    # ── save checkpoint (NEVER overwrite DEFAULT_BACKBONE_PATH) ──
    out_path = Path(args.output)
    if out_path.resolve() == Path(DEFAULT_BACKBONE_PATH).resolve():
        print(f"ERROR: refusing to write to the frozen 19.5M path "
              f"({DEFAULT_BACKBONE_PATH}) -- binding constraint", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone.state_dict(), "step": step,
                "arch": "fromscratch_reltraj_v1",
                "config": {"d_model": cfg.d_model, "n_layers": cfg.n_layers,
                           "d_state": cfg.d_state, "pred_dim": cfg.pred_dim,
                           "ssm_backend": cfg.ssm_backend},
                "best_top3": best_top3, "best_step": best_step,
                "lambda_traj": args.lambda_traj, "temperature": args.temperature,
                "lr": args.lr, "seed": args.seed}, out_path)
    with open(out_path.parent / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_top3": best_top3, "best_step": best_step, "log": log,
                   "args": vars(args)}, f, indent=2)

    final = log[-1]
    print(f"\nDONE. best top3={best_top3:.3f} @ step {best_step} "
          f"(final top3={final['mean_top3_recall']:.3f}, ratio={final['std_ratio']:.3f}x)",
          flush=True)
    print(f"  checkpoint -> {out_path}", flush=True)
    # P1 GATE readout -- only in --probe mode (a scale run's acceptance test is
    # the separate formal Phase B gate re-run, task #33, not this in-script
    # heuristic).
    if args.probe:
        if best_top3 >= 0.5:
            print(f"  P1 GATE: GO -- top3 climbed to {best_top3:.3f} (baseline 0.196, "
                  f"doc 0.938). Scale to the full 511K-doc run.", flush=True)
        elif final["std_ratio"] < 0.1 and best_top3 < 0.3:
            print(f"  P1 GATE: NO-GO -- state near-constant (ratio "
                  f"{final['std_ratio']:.3f}x) and top3 stuck at {best_top3:.3f}. "
                  f"Passthrough/true collapse -- the state arch can't carry "
                  f"doc-identity even under gradient. Document + stop.", flush=True)
        else:
            print(f"  P1 GATE: AMBIGUOUS -- top3 {best_top3:.3f}, ratio "
                  f"{final['std_ratio']:.3f}x. More steps / tune lambda_traj / lr "
                  f"before deciding.", flush=True)
    else:
        print(f"  (scale run -- run the formal Phase B gate re-run, task #33, "
              f"for the acceptance decision)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())