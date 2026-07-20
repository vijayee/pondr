"""Generate the STRM Phase 2a relevance-head training data from ERAG-Bench.

The 2a relevance head scores, for a given query, each WM ring slot's relevance
``r_i in [0,1]``. Its training shape is "query -> relevant slots": a query, a
set of candidate slots, and a per-slot binary label (relevant / not). Upstream
EnterpriseRAG-Bench gives exactly that -- each question carries
``expected_doc_ids`` (the gold passages), so a slot produced from a gold doc is
positive and a slot from a sampled non-gold doc is negative.

Algorithm (one slot per doc):

  1. Load the 480 gold-bearing ERAG questions (8 of 10 categories carry
     ``expected_doc_ids``; ``high_level`` / ``info_not_found`` have none -- excluded).
  2. For each question: candidates = its gold docs + ``--neg-per-query`` docs
     sampled from the non-gold 511K-doc pool, shuffled. Embed each candidate
     doc = mean-pool its section embeddings (MarkdownParser + HierarchicalChunker
     + bge-small, mirroring ``src/ingestion/pipeline.py:134-143``) -> one 384-d
     doc vector.
  3. Step a ``WorkingMemory(ring_capacity=K)`` (ring ON -- the delta from
     ``generate_strm_traces.py`` which used ring_capacity=0) over the candidate
     doc vectors; read ``wm.ring_buffer()`` -> the per-doc step-output slots
     ``y_t`` [K, 256] with provenance ``source_id`` = doc_id.
  4. Embed the query (bge-small, 384-d, same path as ``orchestrator.py:266``).
  5. Emit one record per query: ``{query_id, question, category, query_emb[384],
     slots_y[K,256], slots_doc_emb[K,384], slots_z[K,384] (Phase B: projected SSM
     state per slot), source_ids[K], labels[K] (1 if doc_id in expected_doc_ids)}``.

Output ``data/training/strm_relevance/traces.pt`` (gitignored, regenerable) +
a ``questions_meta.jsonl`` provenance sidecar. The trainer
(``src/subconscious/training/relevance_training.py``) reads this -- no backbone,
no embedder at train time (the y_t slots + query_emb are precomputed here).

Usage:
    python scripts/generate_relevance_data.py --max-queries 80
    python scripts/generate_relevance_data.py --neg-per-query 14 --output data/training/strm_relevance/traces.pt

The ERAG parquet files are already on disk under ``scripts/_scratch/erag/data/``
(pulled once via ``load_dataset("onyx-dot-app/EnterpriseRAG-Bench", ...)``); no
re-download is needed. Pass ``--questions-parquet`` / ``--docs-parquet`` to point
elsewhere.
"""

from __future__ import annotations

import argparse
import json
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
from src.subconscious.training.routing_training import (  # noqa: E402
    _resolve_device,
    build_embedder,
    load_backbone,
)
from src.subconscious.working_memory import WorkingMemory  # noqa: E402

DEFAULT_QUESTIONS_PARQUET = "scripts/_scratch/erag/data/questions/test.parquet"
DEFAULT_DOCS_PARQUET = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
DEFAULT_OUTPUT = "data/training/strm_relevance/traces.pt"

# The 8 ERAG categories that carry expected_doc_ids (ground truth). ``high_level``
# and ``info_not_found`` have empty gold lists (no ground truth) -- excluded.
GOLD_CATEGORIES = (
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "miscellaneous",
)


def load_questions(path: str, categories, max_queries: int) -> list[dict]:
    """Load gold-bearing ERAG questions from the questions parquet.

    Each row: ``question_id, question_type, question, expected_doc_ids,
    gold_answer, answer_facts``. Keeps only rows whose ``question_type`` is in
    ``categories`` AND whose ``expected_doc_ids`` is a non-empty list. Caps at
    ``max_queries``.
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=["question_id", "question_type",
                                       "question", "expected_doc_ids"])
    qids = tbl.column("question_id").to_pylist()
    qtypes = tbl.column("question_type").to_pylist()
    questions = tbl.column("question").to_pylist()
    gold = tbl.column("expected_doc_ids").to_pylist()
    cat_set = set(categories)
    out: list[dict] = []
    for qid, qtype, q, g in zip(qids, qtypes, questions, gold):
        if qtype not in cat_set:
            continue
        if not isinstance(g, list) or not g:
            continue
        out.append({
            "question_id": qid,
            "category": qtype,
            "question": str(q or ""),
            "expected_doc_ids": [d for d in g if isinstance(d, str)],
        })
        if len(out) >= max_queries:
            break
    return out


def build_doc_index(path: str) -> tuple[dict, list[str]]:
    """Read the documents parquet ``doc_id`` column -> ``{doc_id: row_index}``.

    Returns the index and the full ordered ``doc_id`` list (the negative-sampling
    pool). Only the ``doc_id`` column is read (column projection) -- the 1.41 GB
    documents parquet is NOT loaded into memory; ``title``/``content`` for a
    sampled doc are read on demand by slicing the memory-mapped table opened by
    ``open_docs_table``. The gold-vs-missing check is the caller's job (it has
    the gold set already); this builder is a pure index over all docs.
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=["doc_id"])
    ids = tbl.column("doc_id").to_pylist()
    idx = {d: i for i, d in enumerate(ids)}
    return idx, ids


def open_docs_table(path: str):
    """Memory-map the documents parquet ``title`` + ``content`` columns once.

    ``memory_map=True`` keeps the 1.41 GB file lazy (OS pages rows in on demand);
    ``get_doc`` slices ONE row per lookup (``tbl.slice(i, 1)``) so a lookup
    materializes only that row's title+content, not the whole column. Re-reading
    the parquet per lookup (the naive approach) would re-scan 1.41 GB per doc.
    """
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["title", "content"], memory_map=True)


def get_doc(docs_tbl, doc_idx: dict, doc_id: str) -> tuple[str, str] | None:
    """Read a single doc's ``title`` + ``content`` by row index (one-row slice).

    Returns ``None`` if the doc_id is unknown (a stale gold id, defensive).
    """
    i = doc_idx.get(doc_id)
    if i is None:
        return None
    row = docs_tbl.slice(i, 1)
    title = row.column("title").to_pylist()[0]
    content = row.column("content").to_pylist()[0]
    return str(title or ""), str(content or "")


def embed_doc(
    doc_id: str, title: str, content: str, parser, chunker, embedder, device,
) -> torch.Tensor:
    """One 384-d doc vector = mean-pool of the doc's section embeddings.

    Mirrors ``src/ingestion/pipeline.py:134-143``: parse the doc text -> sections
    via the markdown parser, normalize with the HierarchicalChunker, embed each
    section's ``heading\\ncontent`` text with bge-small, mean-pool to one doc
    vector. A doc with no parseable sections falls back to embedding the title
    (so an empty/whitespace doc still yields a vector, not a shape error). Returns
    a ``[1, 384]`` float32 tensor on ``device``.
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
        sec_texts = [title or doc_id]   # cold-start fallback
    vecs = embedder.encode(sec_texts)   # list[list[float]], one 384-d per section
    arr = np.asarray(vecs, dtype=np.float32)            # [N, 384]
    doc_vec = arr.mean(axis=0)                           # [384]
    return torch.from_numpy(doc_vec).to(device).unsqueeze(0)   # [1, 384]


def build_records(
    questions: list[dict],
    docs_tbl,
    doc_idx: dict,
    all_doc_ids: list[str],
    backbone,
    embedder,
    parser,
    chunker,
    neg_per_query: int,
    device,
    seed: int,
    hard_neg_map: dict[str, list[str]] | None = None,
    emit_raw_state: bool = False,
    raw_state_rep: str = "flat_last",
) -> tuple[list[dict], dict]:
    """Step the WM ring over each query's candidate docs; emit labeled records.

    Doc vectors are cached by ``doc_id`` (a gold doc is a candidate for many
    queries; a sampled negative may recur). Returns the records + a small stats
    dict (n_queries, n_pos, n_neg, n_unique_docs, hard_negatives_used).

    Negatives: when ``hard_neg_map`` is None (default), negatives are sampled
    UNIFORM-RANDOM from the non-gold pool (the original Phase 0a path -- random
    negatives are trivially unrelated to the query, so the head never learns
    "close-but-wrong -> low" -> saturates at serve; see
    ``scripts/mine_hard_negatives.py``). When ``hard_neg_map`` is provided (a
    ``{query_id: [doc_id, ...]}`` map of the non-gold docs CLOSEST to the query by
    bge cosine), the per-query negatives are taken from that map instead; any
    shortfall (missing query or short list) is backfilled with random non-gold
    so the record shape stays stable. Labels are gold-membership in BOTH cases
    (unchanged) -- only WHICH non-gold docs are sampled changes.

    Phase 0b: when ``emit_raw_state`` is True, each record ALSO carries
    ``slots_h_raw`` -- the raw flattened SSM recurrent state per slot
    (``flat_last`` [K,6144] or ``flat_all`` [K,24576]) -- for the learned
    StateReadout (``scripts/train_state_readout.py``). Default False -> the
    ``slots_h_raw`` key is ABSENT from every record -> traces.pt byte-identical
    to the pre-Phase-0b generator (no new key, no downstream consumer change).
    """
    rng = np.random.default_rng(seed)
    gold_doc_ids: set[str] = set()
    for q in questions:
        gold_doc_ids.update(q["expected_doc_ids"])
    non_gold = [d for d in all_doc_ids if d not in gold_doc_ids]

    # Pre-sample each query's candidate set (gold + negatives, shuffled) so the
    # rng sequence is deterministic and we know max K up front. One WM instance
    # is then reused across all queries with reset() between them -- mirroring
    # generate_strm_traces.py (a fresh WM per query would re-init the instance
    # LoRA/projection params randomly, making slots incomparable across queries;
    # reset() zeros only the SSM state + clears the ring, leaving the projections
    # fixed).
    plans: list[tuple[dict, list[str]]] = []
    max_k = 1
    for q in questions:
        gold = q["expected_doc_ids"]
        gold_set = set(gold)
        k_neg = min(neg_per_query, len(non_gold))
        if hard_neg_map is not None:
            # Hard negatives: the non-gold docs closest to this query (mined by
            # bge cosine in scripts/mine_hard_negatives.py). Validate each mined
            # id exists in the parquet and is not THIS query's gold (the miner
            # already drops gold, but a stale id or a gold-for-another-query id
            # that the miner kept -- a valid hard neg -- must survive here, so we
            # check existence not non_gold membership).
            mined = [d for d in hard_neg_map.get(q["question_id"], [])
                     if d not in gold_set and d in doc_idx]
            if len(mined) < k_neg:
                pool = [d for d in non_gold if d not in mined]
                need = k_neg - len(mined)
                if pool and need > 0:
                    mined = mined + list(rng.choice(
                        pool, size=min(need, len(pool)), replace=False))
            neg_ids = mined
        else:
            neg_ids = list(rng.choice(non_gold, size=k_neg, replace=False))
        candidate_ids = list(gold) + [d for d in neg_ids if d not in gold_set]
        rng.shuffle(candidate_ids)
        plans.append((q, candidate_ids))
        max_k = max(max_k, len(candidate_ids))

    wm = WorkingMemory(backbone, embedder=embedder, ring_capacity=max_k)

    doc_cache: dict[str, torch.Tensor] = {}
    n_pos = n_neg = 0

    # Untrained is fine: LatentDynamicsHead.project is PARAMETER-FREE (last
    # layer, mean over d_state -> [1, 384]); the trained ``linear`` is only used
    # by predict/surprise, not project. Using the same parameter-free project
    # here AND in the serve probe (probe_strm_selectivity_real.py) guarantees
    # train/serve ``z_i`` match -- the z_i head is trained on EXACTLY the z_i
    # the probe scores it on. Phase A captures slot.h for every ring slot, so
    # every slot is projectable.
    ld_head = LatentDynamicsHead()

    def doc_vec(doc_id: str) -> torch.Tensor | None:
        if doc_id in doc_cache:
            return doc_cache[doc_id]
        tc = get_doc(docs_tbl, doc_idx, doc_id)
        if tc is None:
            return None
        v = embed_doc(doc_id, tc[0], tc[1], parser, chunker, embedder, device)
        doc_cache[doc_id] = v
        return v

    records: list[dict] = []
    t0 = time.time()
    for qi, (q, candidate_ids) in enumerate(plans):
        gold = q["expected_doc_ids"]
        # gather doc vectors, drop any doc that failed to load
        cand_vecs: list[tuple[str, torch.Tensor]] = []
        for d in candidate_ids:
            v = doc_vec(d)
            if v is not None:
                cand_vecs.append((d, v))
        if not cand_vecs:
            continue
        K = len(cand_vecs)
        # ring_capacity == max_k >= K, so all K slots are retained (FIFO deque).
        wm.reset()
        for d, v in cand_vecs:
            wm.step(v, source_id=d, text=d)
        ring = wm.ring_buffer()                       # list[RingSlot], oldest-first
        # ring is oldest-first == the step order == cand_vecs order.
        slots_y = torch.stack([s.y.detach().to("cpu").to(torch.float32)
                               for s in ring]).squeeze(1)   # [K, 256]
        source_ids = [str(s.source_id) for s in ring]
        # The projected SSM recurrent state per slot -- Phase A stores slot.h
        # (the post-step, post-decay per-layer recurrent state, fp16), and
        # ``z_i = LatentDynamicsHead.project(slot.h)`` (last layer, mean over
        # d_state -> [1, 384]) is the state-trajectory rewire's unit of
        # attention. The Phase B ``h_t`` probe trains a ZRelevanceHead on this
        # field (``slot_signal_field="slots_z"``) to test whether ``z_i``
        # carries query-relevance signal the 2a ``y_t`` readout did NOT -- the
        # cheap GATE 1 before any transformer build. Ring ON (max_k>=1) -> every
        # slot has h; a None here would mean Phase A regressed (hard error).
        if any(s.h is None for s in ring):
            raise RuntimeError(
                "build_records: a ring slot has h=None (Phase A should capture h "
                "for every slot when ring_capacity>0). Regenerate the backbone?"
            )
        slots_z = torch.stack([
            ld_head.project(s.h).squeeze(0).detach().to("cpu").to(torch.float32)
            for s in ring
        ])                                                 # [K, 384]
        # Phase 0b: the RAW flattened SSM recurrent state per slot, for the
        # learned StateReadout (scripts/train_state_readout.py -> CompositeZHead).
        # Phase 0a (probe_state_signal_distribution.py) showed the FIXED mean-pool
        # ``project`` (mean over 16 d_state channels) cancels opposing-sign channel
        # signal to near-constant, while the flattened state varies 0.45-0.76x as
        # much as the doc embeddings -- so the state carries doc-identity
        # VARIANCE the mean-pool hides. ``slots_h_raw`` is that flattened state;
        # the composite head's StateReadout learns to MIX the channels into a
        # query-relevant z_i. GATE 0b RESULT: the learned readout does NOT clear
        # the TRAIN gate (flat_last Linear/MLP top3 ~0.27 == mean-pool 0.285;
        # flat_all MLP best 0.564, real but sub-gate, overfits by ~epoch 33) -- the
        # state carries only PARTIAL query-relevance (across all 4 layers,
        # nonlinearly mixed); the JEPA-trained backbone encodes doc-identity in a
        # mostly query-orthogonal subspace. -> Phase 1 (relevance-objective
        # backbone fine-tune).
        # ``flat_last`` [K,6144] = last layer's 16 d_state x 384 d_model;
        # ``flat_all``  [K,24576] = all 4 layers flattened. Only emitted when
        # ``emit_raw_state`` (default off -> records byte-identical to today).
        slots_h_raw = None
        if emit_raw_state:
            if raw_state_rep == "flat_all":
                slots_h_raw = torch.stack([
                    torch.cat([layer.float().reshape(-1) for layer in s.h]).detach()
                    for s in ring
                ])                                         # [K, 24576]
            else:                                          # flat_last (default)
                slots_h_raw = torch.stack([
                    s.h[-1].float().reshape(-1).detach() for s in ring
                ])                                         # [K, 6144]
        # The RAW bge doc embedding per slot -- the step INPUT to wm.step, in the
        # SAME order as the ring (== cand_vecs order). The 2a head reads this
        # alongside y_t: a probe (scripts/_scratch/_probe_relevance_bge_baseline)
        # showed raw bge cosine(query, doc) clears the top-3 gate at 1.000 on this
        # slice, while y_t alone (the frozen routing-trained backbone's readout)
        # carries NO relevance signal -- the backbone was trained for routing, not
        # retrieval-similarity, so its 256-d readout does not preserve query-doc
        # relevance. The head fuses [y_t ; doc_bge ; query_emb]: y_t keeps it a
        # genuine WM-ring-slot head (and lets a future backbone fine-tune activate
        # it); doc_bge is where the signal lives.
        slots_doc_emb = torch.stack([
            v.detach().to("cpu").to(torch.float32).squeeze(0)
            for d, v in cand_vecs
        ])                                                 # [K, 384]
        gold_set = set(gold)
        labels = torch.tensor([1 if sid in gold_set else 0 for sid in source_ids],
                              dtype=torch.long)
        # query embedding (raw bge-small, 384-d) -- same path as orchestrator.py:266
        qv = embedder.encode([q["question"]])[0]
        query_emb = torch.tensor(np.asarray(qv, dtype=np.float32))   # [384]
        n_pos += int(labels.sum().item())
        n_neg += int((1 - labels).sum().item())
        rec = {
            "query_id": q["question_id"],
            "question": q["question"],
            "category": q["category"],
            "expected_doc_ids": list(gold),
            "query_emb": query_emb,            # [384]
            "slots_y": slots_y,                # [K, 256]  (WM recurrent readout)
            "slots_doc_emb": slots_doc_emb,    # [K, 384]  (raw bge doc identity)
            "slots_z": slots_z,                # [K, 384]  (projected SSM state, Phase B)
            "source_ids": source_ids,          # [K]
            "labels": labels,                  # [K]
        }
        # Phase 0b raw state -- only present when --emit-raw-state (default off
        # -> the record dict, and thus traces.pt, is byte-identical to today).
        if emit_raw_state:
            rec["slots_h_raw"] = slots_h_raw   # [K, 6144 or 24576]
        records.append(rec)
        if (qi + 1) % 20 == 0:
            print(f"  built {qi + 1}/{len(questions)} queries "
                  f"({time.time() - t0:.1f}s, cache={len(doc_cache)})", flush=True)
    stats = {"n_queries": len(records), "n_pos": n_pos, "n_neg": n_neg,
             "n_unique_docs": len(doc_cache),
             "hard_negatives_used": hard_neg_map is not None}
    return records, stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate STRM Phase 2a relevance-head training data (ERAG-Bench)")
    p.add_argument("--questions-parquet", default=DEFAULT_QUESTIONS_PARQUET)
    p.add_argument("--docs-parquet", default=DEFAULT_DOCS_PARQUET)
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--meta-output", default=None,
                   help="questions_meta.jsonl sidecar (default: <output dir>/questions_meta.jsonl)")
    p.add_argument("--max-queries", type=int, default=480)
    p.add_argument("--neg-per-query", type=int, default=14)
    p.add_argument("--categories", default=",".join(GOLD_CATEGORIES),
                   help="comma-separated ERAG categories to include (default: all 8 gold-bearing)")
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hard-neg-json", default=None,
                   help="path to a {query_id: [doc_id, ...]} hard-negative map from "
                        "scripts/mine_hard_negatives.py; when set, per-query negatives are "
                        "the non-gold docs CLOSEST to the query (vs the default uniform-random "
                        "sample). Default None -> random negatives (byte-identical to pre-2026-07).")
    p.add_argument("--retrace", action="store_true",
                   help="regenerate even if the output file exists")
    p.add_argument("--emit-raw-state", action="store_true",
                   help="Phase 0b: also emit ``slots_h_raw`` (the raw flattened SSM "
                        "recurrent state per slot) for the learned StateReadout. "
                        "Default off -> records byte-identical to today.")
    p.add_argument("--raw-state-rep", default="flat_last",
                   choices=["flat_last", "flat_all"],
                   help="raw-state representation: ``flat_last`` [6144] = last SSM "
                        "layer flattened (default, apples-to-apples vs the mean-pool); "
                        "``flat_all`` [24576] = all 4 layers flattened. Only used with "
                        "--emit-raw-state.")
    args = p.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.retrace:
        print(f"  traces already exist at {out_path} (use --retrace to regenerate)",
              flush=True)
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_output) if args.meta_output else out_path.parent / "questions_meta.jsonl"

    questions_path = Path(args.questions_parquet)
    docs_path = Path(args.docs_parquet)
    backbone_path = Path(args.backbone)
    for pth, label in ((questions_path, "questions"), (docs_path, "documents"),
                       (backbone_path, "backbone")):
        if not pth.exists():
            print(f"ERROR: {label} not found at {pth}", file=sys.stderr)
            return 1

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    print(f"Loading questions from {questions_path} (categories={categories}, "
          f"max {args.max_queries})", flush=True)
    questions = load_questions(str(questions_path), categories, args.max_queries)
    if not questions:
        print(f"ERROR: no gold-bearing questions loaded from {questions_path}",
              file=sys.stderr)
        return 1
    gold_doc_ids: set[str] = set()
    for q in questions:
        gold_doc_ids.update(q["expected_doc_ids"])
    print(f"  {len(questions)} questions, {len(gold_doc_ids)} unique gold doc_ids",
          flush=True)

    print(f"Indexing documents parquet {docs_path} (doc_id column only)", flush=True)
    doc_idx, all_doc_ids = build_doc_index(str(docs_path))
    missing = [d for d in gold_doc_ids if d not in doc_idx]
    if missing:
        print(f"  WARNING: {len(missing)} gold doc_ids not in documents parquet "
              f"(first: {missing[:3]}) -- those questions lose gold slots", flush=True)
    print(f"  {len(all_doc_ids)} docs indexed", flush=True)
    print(f"Memory-mapping documents parquet title+content (lazy)", flush=True)
    docs_tbl = open_docs_table(str(docs_path))

    print(f"Loading frozen backbone from {backbone_path}", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  backbone: {sum(p.numel() for p in backbone.parameters()):,} params (frozen)",
          flush=True)
    print(f"Loading embedder (bge-small, on-demand)", flush=True)
    embedder = build_embedder("on-demand")

    dev = _resolve_device(args.device)
    parser = MarkdownParser()
    chunker = HierarchicalChunker()

    hard_neg_map: dict[str, list[str]] | None = None
    if args.hard_neg_json:
        hn_path = Path(args.hard_neg_json)
        if not hn_path.exists():
            print(f"ERROR: --hard-neg-json not found at {hn_path}", file=sys.stderr)
            return 1
        with open(hn_path, "r", encoding="utf-8") as f:
            hard_neg_map = json.load(f)
        print(f"Loaded hard-negative map: {len(hard_neg_map)} queries from {hn_path}",
              flush=True)

    print(f"Building records (ring ON, neg_per_query={args.neg_per_query}, "
          f"device={dev}, hard_neg={'ON' if hard_neg_map else 'OFF'}, "
          f"raw_state={'ON(' + args.raw_state_rep + ')' if args.emit_raw_state else 'OFF'}) "
          f"-> {out_path}", flush=True)
    records, stats = build_records(
        questions, docs_tbl, doc_idx, all_doc_ids, backbone, embedder,
        parser, chunker, args.neg_per_query, dev, args.seed, hard_neg_map,
        emit_raw_state=args.emit_raw_state, raw_state_rep=args.raw_state_rep,
    )
    if not records:
        print(f"ERROR: no records built (all candidates failed to load?)",
              file=sys.stderr)
        return 1
    torch.save(records, out_path)
    hard_neg_used = bool(hard_neg_map)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "query_id": r["query_id"], "question": r["question"],
                "category": r["category"],
                "expected_doc_ids": r["expected_doc_ids"],
                "source_ids": r["source_ids"],
                "labels": r["labels"].tolist(),
                "hard_negatives": hard_neg_used,
            }, ensure_ascii=False) + "\n")

    mb = out_path.stat().st_size / 1e6
    print(f"DONE. {stats['n_queries']} queries, {stats['n_pos']} pos / "
          f"{stats['n_neg']} neg slots, {stats['n_unique_docs']} unique docs "
          f"(hard_neg={'ON' if hard_neg_used else 'OFF'}) "
          f"-> {out_path} ({mb:.1f} MB)", flush=True)
    print(f"  meta -> {meta_path}", flush=True)
    print(f"  Next: python scripts/train_relevance_head.py --traces {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())