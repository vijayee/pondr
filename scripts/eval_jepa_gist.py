"""The JEPA-latent gist gate (the Stage-1 verdict, run on a trained JEPA-gist model).

Loads a trained ``JEPAGistModel`` (``scripts/train_jepa_gist.py`` output) via
``load_jepa_gist`` -- the retrain checkpoint carries the retrained ENCODER in
itself, so no separate encoder checkpoint is needed. Then runs the judge-free
latent-space swap gate on fully held-out docs (offset 60_000, beyond the encoder
train slice + the JEPA train slice).

The gate is the latent-space analog of the likelihood-swap gate in
``eval_gist_readout.py``, with the sign flipped (cosine: higher = closer; NLL:
lower = closer). bge-small is the FROZEN referee -- the gold gist for each held-out
doc is ``bge_small.encode(flash_summarize(content))``, and the model's predicted
latent is compared to it by cosine. No token decoder, no LLM judge, no
compression/fluency gates (those were token-decode-specific; the retrieval metric
replaces them as the content-recovery check).

Gate components (all judge-free; bge is the frozen referee):

  1. **Discrimination margin** (PRIMARY) = ``(cos_aa - cos_ab) + (cos_bb - cos_ba)``
     over pairs (A, B), where ``cos_xy = cos(pred|state_x, gist_y)``. Positive = the
     state carries doc-specific gist content (swapping states changes the predicted
     gist latent in the right direction). Bar > 0. This is the analog of the NLL
     discrimination margin -- the decisive number; the summary-CE retrain's was
     -0.003 (unchanged from the frozen probe's 0.000).
  2. **Swap fidelity** = fraction of pairs where ``cos_aa > cos_ab AND cos_bb >
     cos_ba``. Bar 0.8.
  3. **State-vs-zero gap** = ``cos_aa - cos(pred|zero_state, gist_a)``. Positive =
     the state helps over a zero state (the predictor uses the state, not just its
     own prior). Bar > 0. The summary-CE retrain's was -0.037 (decoder IGNORED the
     state).
  4. **State distinctness** ``||sA - sB||`` (mean over layers, L2) -- diagnostic,
     objective-independent: do the encoder states differ per doc?
  5. **Retrieval** (SECONDARY, the content-recovery check): for each held-out doc,
     is ``pred|state`` closest to its OWN gist among ALL held-out gist latents?
     Report top-1 and top-3 accuracy. The predicted latent should retrieve the
     correct gist from the corpus.
  6. **Collapse watchdog** (read from the trainer's ``run_summary.json`` in the same
     output dir): encoder continuation val ppl start -> end. A blow-up (>2-3x
     baseline) means the encoder collapsed (the summary-CE retrain went 8.1x) --
     the LM-prior auxiliary is the fix; this reports whether it held.

PASS = discrimination margin > 0 AND swap fidelity >= 0.8 AND state-vs-zero gap > 0
AND encoder not collapsed. The bge teacher gist text is generated via the trainer's
exact ``flash_summarize`` (so the gold distribution matches the training
distribution) and bge-encoded fresh (held-out docs are not in the training cache).

Usage:
    python scripts/eval_jepa_gist.py \
        --checkpoint data/jepa_gist/jepa_final.pt \
        --tokenizer-cache data/token_lm/tokenizer.json \
        --output-dir data/jepa_gist --max-eval-docs 60 --n-swap-pairs 25
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# scripts/ on the path so we can reuse the trainer's EXACT flash_summarize (the gold
# gist must come from the same distribution the model was trained on) + the bge
# encoder helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.subconscious.jepa_gist import load_jepa_gist  # noqa: E402
from train_gist_readout import flash_summarize as _teacher_summarize  # noqa: E402
from train_jepa_gist import _bge_encode, _build_bge_embedder  # noqa: E402

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"

# Encoder trained on the first 50k ERAG docs (train) + 500 val (see
# train_token_lm.py defaults). The JEPA predictor trains on a SEPARATE slice (the
# first `val_docs` of its own stream + the next `max_train_docs`). To get docs fully
# held-out from BOTH, start the eval slice beyond a safe offset (same as
# eval_gist_readout).
DEFAULT_EVAL_OFFSET = 60_000


# --------------------------------------------------------------------- ERAG
def _iter_erag_docs(path: str, offset: int, max_docs: int):
    """Stream (title, content) from ERAG starting at doc index `offset`."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    seen = 0
    yielded = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["title", "content"])
        titles = tbl.column("title").to_pylist()
        contents = tbl.column("content").to_pylist()
        for title, content in zip(titles, contents):
            seen += 1
            if seen <= offset:
                continue
            if content and str(content).strip() and title and str(title).strip():
                yield str(title), str(content)
                yielded += 1
                if yielded >= max_docs:
                    return


# --------------------------------------------------------------------- eval
@torch.no_grad()
def encode_state(model, tok, content: str, doc_seq_len: int, device) -> list[torch.Tensor]:
    """Encode a doc's content -> final per-layer encoder state."""
    doc_ids = tok.encode_batch([content], max_length=doc_seq_len)
    doc_t = torch.tensor(doc_ids, dtype=torch.long, device=device)
    return model.encode(doc_t, no_grad=True)


@torch.no_grad()
def predict_latent(model, states: list[torch.Tensor]) -> torch.Tensor:
    """Pool state -> predicted gist latent [1, latent_dim] (L2-normed)."""
    return model.predict_latent(states)


def _gold_gist_latents(emb, docs, gist_cache: dict, cache_path: Path
                       ) -> tuple[list[int], np.ndarray, list[str]]:
    """bge-encode the gold gist for each held-out doc.

    The gold gist text comes from the deepseek-flash teacher (the same distribution
    the model was trained on); held-out docs are NOT in the training cache, so the
    text is generated via ``flash_summarize`` (Ollama, cache-miss -> fill + cache)
    then bge-encoded fresh. Returns ``(valid_indices, latents [N, 384] L2-normed,
    gist_texts)`` aligned to ``valid_indices`` -- docs whose gist could not be
    produced are dropped (and absent from ``valid_indices``).
    """
    print(f"[gold] generating + bge-encoding gold gists for {len(docs)} held-out "
          f"docs (cache={cache_path})...", flush=True)
    texts: list[str | None] = []
    t0 = time.time()
    for i, (title, content) in enumerate(docs):
        g = _teacher_summarize(content, gist_cache, cache_path)
        texts.append(g)
        if (i + 1) % 10 == 0:
            print(f"  [gold] {i + 1}/{len(docs)} ({time.time() - t0:.0f}s)", flush=True)
    valid_idx = [i for i, g in enumerate(texts) if g]
    valid_texts = [texts[i] for i in valid_idx]
    print(f"[gold] {len(valid_texts)}/{len(docs)} gold gists ready; bge-encoding...",
          flush=True)
    if not valid_texts:
        return [], np.zeros((0, 0)), []
    vecs = _bge_encode(emb, valid_texts, batch_size=64)  # [N, 384] L2-normed
    return valid_idx, vecs, valid_texts


def run_gate(args) -> int:
    print(f"[load] checkpoint={args.checkpoint}", flush=True)
    model, tok = load_jepa_gist(
        args.checkpoint, args.tokenizer_cache, device=args.device, dtype=args.dtype,
    )
    device = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    enc_frozen = all(not p.requires_grad for p in model.encoder.parameters())
    print(f"[load] encoder FROZEN={enc_frozen}, predictor params="
          f"{model.trainable_parameters():,}, latent_dim="
          f"{model.cfg.latent_dim}", flush=True)
    model.eval()

    # ---- held-out docs (beyond encoder train + JEPA train slices).
    docs = list(_iter_erag_docs(ERAG_PATH, args.eval_offset, args.max_eval_docs))
    print(f"[data] {len(docs)} held-out docs (offset={args.eval_offset})", flush=True)
    if len(docs) < 4:
        print("[gate] not enough held-out docs; abort.", flush=True)
        return 1

    # ---- gold gist latents: bge-encode the deepseek summary of each held-out doc.
    cache_path = (Path(args.gist_cache) if args.gist_cache
                  else Path(args.output_dir) / "gist_teacher_cache_v2.json")
    gist_cache: dict = {}
    if cache_path.exists():
        gist_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    emb = _build_bge_embedder(device)
    valid_idx, gold_latents_np, gold_texts = _gold_gist_latents(
        emb, docs, gist_cache, cache_path)
    # Align the valid-doc list to the latents (docs whose gold gist was produced).
    valid_docs = [docs[i] for i in valid_idx]
    if len(valid_docs) < 4:
        print(f"[gate] only {len(valid_docs)} docs with gold gists; abort.", flush=True)
        return 1
    n_valid = len(valid_docs)
    print(f"[gold] {n_valid} valid docs with gold gist latents "
          f"({gold_latents_np.shape[1]}-d, L2-normed)", flush=True)

    # ---- encode each valid doc's state + predict its gist latent.
    print("[gate] encoding states + predicting latents...", flush=True)
    t0 = time.time()
    pred_latents = np.zeros((n_valid, model.cfg.latent_dim), dtype=np.float32)
    states_list: list[list[torch.Tensor]] = []
    for i, (title, content) in enumerate(valid_docs):
        st = encode_state(model, tok, content, args.doc_seq_len, device)
        states_list.append(st)
        pred = predict_latent(model, st).float().cpu().numpy().squeeze(0)
        pred_latents[i] = pred
        if (i + 1) % 10 == 0:
            print(f"  [pred] {i + 1}/{n_valid} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 1-3: discrimination margin / swap fidelity / state-vs-zero (PRIMARY).
    print("\n[gate 1-3] latent swap (PRIMARY, judge-free; bge referee):", flush=True)
    rng = random.Random(args.seed)
    idxs = list(range(n_valid))
    rng.shuffle(idxs)
    pairs: list[tuple[int, int]] = []
    for k in range(0, min(args.n_swap_pairs * 2, n_valid - 1), 2):
        pairs.append((idxs[k], idxs[k + 1]))
        if len(pairs) >= args.n_swap_pairs:
            break
    pred_t = torch.from_numpy(pred_latents)              # [N, 384] CPU
    gold_t = torch.from_numpy(gold_latents_np)           # [N, 384] CPU

    # Zero state: same shape as a real encoder state, all zeros (does the predictor
    # use the state, or just its own prior?). The predicted latent from a zero state
    # is DETERMINISTIC (the predictor's bias-only output) -- compute it ONCE, not per
    # pair (the prior version recomputed it 25x identically).
    zero_states = [torch.zeros_like(s) for s in states_list[0]]
    with torch.no_grad():
        pred_zero_np = predict_latent(model, zero_states).float().cpu().numpy().squeeze(0)
    pred_zero_t = torch.from_numpy(pred_zero_np).unsqueeze(0)  # [1, 384]

    margins: list[float] = []
    state_gaps: list[float] = []
    zero_gaps: list[float] = []
    swap_correct = 0
    t0 = time.time()
    for k, (ia, ib) in enumerate(pairs):
        pa = pred_t[ia:ia + 1]   # [1, 384]
        pb = pred_t[ib:ib + 1]
        ga = gold_t[ia:ia + 1]
        gb = gold_t[ib:ib + 1]
        cos_aa = F.cosine_similarity(pa, ga, dim=-1).item()
        cos_ab = F.cosine_similarity(pa, gb, dim=-1).item()
        cos_bb = F.cosine_similarity(pb, gb, dim=-1).item()
        cos_ba = F.cosine_similarity(pb, ga, dim=-1).item()
        ok_a = cos_aa > cos_ab
        ok_b = cos_bb > cos_ba
        swap_correct += int(ok_a and ok_b)
        margins.append((cos_aa - cos_ab) + (cos_bb - cos_ba))  # +ve = discriminates
        # State distinctness: mean over layers of L2(state_A - state_B).
        gap = sum((sa - sb).float().pow(2).sum().sqrt().item()
                  for sa, sb in zip(states_list[ia], states_list[ib])) / len(states_list[ia])
        state_gaps.append(gap)
        # State-vs-zero: does predicting from the real state beat predicting from a
        # zero state, for doc A's own gist?
        cos_a_zero = F.cosine_similarity(pred_zero_t, ga, dim=-1).item()
        zero_gaps.append(cos_aa - cos_a_zero)  # +ve = state helps over zero
        if (k + 1) % 5 == 0 or k == 0:
            print(f"  [pair {k + 1}/{len(pairs)}] cosA|A={cos_aa:.3f} cosA|B={cos_ab:.3f} "
                  f"cosB|B={cos_bb:.3f} cosB|A={cos_ba:.3f} A|0={cos_a_zero:.3f} "
                  f"||sA-sB||={gap:.2f} ok={ok_a and ok_b} ({time.time() - t0:.0f}s)",
                  flush=True)
    n_pairs = len(pairs)
    swap_fidelity = swap_correct / n_pairs if n_pairs else float("nan")
    mean_margin = (sum(margins) / len(margins)) if margins else float("nan")
    mean_state_gap = (sum(state_gaps) / len(state_gaps)) if state_gaps else float("nan")
    mean_zero_gap = (sum(zero_gaps) / len(zero_gaps)) if zero_gaps else float("nan")
    print(f"[gate 1] discrimination margin = {mean_margin:.4f}  (bar > 0)", flush=True)
    print(f"[gate 2] swap fidelity = {swap_correct}/{n_pairs} = {swap_fidelity:.3f}  "
          f"(bar {args.swap_bar})", flush=True)
    print(f"[gate 3] state-vs-zero gap = {mean_zero_gap:.4f}  (bar > 0; the summary-CE "
          f"retrain's was -0.037)", flush=True)
    print(f"[gate 4] state distinctness ||sA-sB|| = {mean_state_gap:.3f}  "
          f"(diagnostic; objective-independent)", flush=True)

    # ---- 5: retrieval (SECONDARY, the content-recovery check). For each held-out
    # doc, is pred|state closest to its OWN gist among ALL held-out gist latents?
    print("\n[gate 5] retrieval (SECONDARY): pred latent retrieves own gist?",
          flush=True)
    sim = F.cosine_similarity(pred_t.unsqueeze(1), gold_t.unsqueeze(0), dim=-1)  # [N, N]
    ranks = sim.argsort(dim=1, descending=True)  # [N, N] -- column index of each doc's gist
    self_idx = torch.arange(n_valid)
    rank_of_self = (ranks == self_idx.unsqueeze(1)).nonzero()  # find self's rank
    # rank_of_self[:,1] is the position of self in each row's ranking.
    self_ranks = torch.full((n_valid,), -1, dtype=torch.long)
    if rank_of_self.numel() > 0:
        for r, c in rank_of_self.tolist():
            self_ranks[r] = c
    top1 = int((self_ranks == 0).sum().item())
    top3 = int((self_ranks < 3).sum().item())
    top1_acc = top1 / n_valid
    top3_acc = top3 / n_valid
    print(f"[gate 5] top-1 = {top1}/{n_valid} = {top1_acc:.3f}  "
          f"top-3 = {top3}/{n_valid} = {top3_acc:.3f}", flush=True)

    # ---- nearest-neighbor examples (doc -> predicted latent -> its top-3 gists).
    print("\n[samples] nearest-neighbor (first 5 valid docs):", flush=True)
    nn_samples: list[dict] = []
    for i in range(min(5, n_valid)):
        top3_i = ranks[i, :3].tolist()
        own_rank = int(self_ranks[i].item())
        safe_title = valid_docs[i][0].encode("ascii", "replace").decode()
        print(f"  [{i}] title: {safe_title}", flush=True)
        print(f"      own gist rank among {n_valid}: {own_rank}", flush=True)
        for j, jdx in enumerate(top3_i):
            mark = " (own)" if jdx == i else ""
            g = gold_texts[jdx]
            safe_g = g.encode("ascii", "replace").decode()[:120]
            print(f"      top{j + 1}: {safe_g}{mark}", flush=True)
        nn_samples.append({
            "title": valid_docs[i][0],
            "own_rank": own_rank,
            "top3_indices": top3_i,
            "top3_gists": [gold_texts[j] for j in top3_i],
            "pred_cos_to_own": float(sim[i, i].item()),
        })

    # ---- 6: collapse watchdog (read from the trainer's run_summary.json).
    enc_ppl_start = float("nan")
    enc_ppl_end = float("nan")
    collapsed = False
    summary_path = Path(args.output_dir) / "run_summary.json"
    if summary_path.exists():
        try:
            ts = json.loads(summary_path.read_text(encoding="utf-8"))
            enc_ppl_start = float(ts.get("encoder_val_ppl_start", float("nan")))
            enc_ppl_end = float(ts.get("encoder_val_ppl_end", float("nan")))
            if math.isfinite(enc_ppl_start) and math.isfinite(enc_ppl_end) \
                    and enc_ppl_start > 0:
                ratio = enc_ppl_end / enc_ppl_start
                collapsed = ratio > args.collapse_ratio_bar
                print(f"[gate 6] collapse watchdog: encoder continuation val ppl "
                      f"{enc_ppl_start:.1f} -> {enc_ppl_end:.1f} "
                      f"({ratio:.2f}x; collapse if > {args.collapse_ratio_bar}x) "
                      f"-> {'COLLAPSED' if collapsed else 'OK'}", flush=True)
        except Exception as e:
            print(f"[gate 6] watchdog: could not read {summary_path}: {e}", flush=True)
    else:
        print(f"[gate 6] watchdog: no run_summary.json at {summary_path} "
              f"(collapse not checked)", flush=True)

    # ---- verdict.
    margin_pass = (n_pairs > 0) and (mean_margin > 0)
    swap_pass = (n_pairs > 0) and (swap_fidelity >= args.swap_bar)
    zero_pass = (n_pairs > 0) and (mean_zero_gap > 0)
    not_collapsed = not collapsed
    verdict = "PASS" if (margin_pass and swap_pass and zero_pass and not_collapsed) else "FAIL"
    print("\n==================== VERDICT ====================", flush=True)
    print(f"  discrimination margin : {mean_margin:.4f}  (bar > 0)  -> {'PASS' if margin_pass else 'FAIL'}  [PRIMARY, load-bearing]", flush=True)
    print(f"  swap fidelity         : {swap_fidelity:.3f}  (bar {args.swap_bar})  -> {'PASS' if swap_pass else 'FAIL'}  [PRIMARY]", flush=True)
    print(f"  state-vs-zero gap     : {mean_zero_gap:.4f}  (bar > 0)  -> {'PASS' if zero_pass else 'FAIL'}  [PRIMARY]", flush=True)
    print(f"  state distinctness    : ||sA-sB||={mean_state_gap:.3f}  (diagnostic)", flush=True)
    print(f"  retrieval top-1/top-3 : {top1_acc:.3f} / {top3_acc:.3f}  [SECONDARY, content-recovery]", flush=True)
    print(f"  collapse watchdog     : {enc_ppl_start:.1f} -> {enc_ppl_end:.1f}  "
          f"-> {'COLLAPSED' if collapsed else 'OK'}", flush=True)
    print(f"  VERDICT               : {verdict}", flush=True)
    print("=================================================", flush=True)

    results = {
        "n_eval_docs": n_valid,
        "n_swap_pairs": n_pairs,
        "discrimination_margin": mean_margin,
        "swap_fidelity": swap_fidelity,
        "state_vs_zero_gap": mean_zero_gap,
        "state_distinction_l2": mean_state_gap,
        "retrieval_top1": top1_acc,
        "retrieval_top3": top3_acc,
        "encoder_val_ppl_start": enc_ppl_start,
        "encoder_val_ppl_end": enc_ppl_end,
        "encoder_collapsed": collapsed,
        "swap_bar": args.swap_bar,
        "collapse_ratio_bar": args.collapse_ratio_bar,
        "verdict": verdict,
        "nearest_neighbor_samples": nn_samples,
    }
    out = Path(args.output_dir) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[results] wrote {out}", flush=True)
    return 0 if verdict == "PASS" else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the JEPA-latent gist gate.")
    ap.add_argument("--checkpoint", default="data/jepa_gist/jepa_final.pt")
    ap.add_argument("--tokenizer-cache", default="data/token_lm/tokenizer.json")
    ap.add_argument("--output-dir", default="data/jepa_gist")
    ap.add_argument("--gist-cache", default="",
                    help="path to the teacher gist cache (sha1(content)->summary); "
                         "extended with held-out docs' gists during eval")
    ap.add_argument("--max-eval-docs", type=int, default=60)
    ap.add_argument("--n-swap-pairs", type=int, default=25)
    ap.add_argument("--eval-offset", type=int, default=DEFAULT_EVAL_OFFSET)
    ap.add_argument("--doc-seq-len", type=int, default=128)
    ap.add_argument("--swap-bar", type=float, default=0.8)
    ap.add_argument("--collapse-ratio-bar", type=float, default=3.0,
                    help="encoder val ppl end/start ratio at which the encoder is "
                         "deemed collapsed (the summary-CE retrain went 8.1x)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return run_gate(args)


if __name__ == "__main__":
    sys.exit(main())