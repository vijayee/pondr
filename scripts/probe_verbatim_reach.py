"""Verbatim-reach probe -- the token-content decay curve (FADE architecture, task #31).

Cheap, no-training measurement that gates the fade build. Answers: how long does
recent text CONTENT survive in the token-LM SSM state before it is compressed to
identity-only? This is the §6.1 recoverability recipe
(`docs/JST-architecture-proposal.md` Sec 6.1; `recoverability_training.py`)
applied to the trained token-LM (`SSMLanguageModel`), with the per-token embedding
as the anchor ``u_i`` and the lag ``k = t - i`` as the recency horizon.

The recipe (reuses ``ridge_fit`` from ``strm_traces.py`` -- closed-form, seconds):
  1. Stream real ERAG docs through the token-LM one token at a time via ``step()``,
     capturing the pooled recurrent state ``s_t`` [n_layers*d_model] at each step
     (mean over d_state, concat all layers -- the recoverability head's rep). The
     trained ckpt is d_model=192, 6 layers -> pooled 1152.
  2. For each anchor ``i`` and lag ``k``, pair ``(s_{i+k}, u_i)`` where
     ``u_i = token_emb(token_i)`` [d_model] -- the input the SSM actually ingested.
  3. Decoder ``D = ridge(S_train, U_train, lam)``; reconstruction error
     ``e(i,t) = ||D(s_t) - u_i||^2.mean()`` is the ground-truth forgetting signal.
     Fit D on train docs; evaluate e on held-out docs (no pair leakage -- split by
     DOC, not by pair).
  4. Sweep log-spaced lags ``k``. Report ``e(k)`` (the forgetting curve) vs a
     constant-predictor chance floor (``e`` when the state carries no anchor info)
     AND top-1 token-ID recovery accuracy (``argmax cos(D(s_t), emb_table) == token_i``,
     the interpretable verbatim-reach curve; chance = 1/vocab).

What the curve tells the architecture (the fade router depends on this):
  - The SHAPE of ``e(k)``: a GRACEFUL rise (residual content survives a useful
    window) -> Regime 2 (fill-holes via the Transformer/JEPA readout) is viable and
    the fade is rich. A CLIFF (vanishes in 1-2 steps) -> Regime 2 is thin; the fade
    collapses to ring -> vector-retrieval -> forgotten.
  - The HORIZON where ``e(k)`` hits the chance floor = where verbatim ends and
    gist/forgotten begins (the Regime 1 -> Regime 2/3 boundary for the text leg).

Standalone: reads the trained ckpt + tokenizer + ERAG parquet; writes
``run_summary.json`` + prints the curve. No orchestrator/runtime/serve changes.
CPU-runnable (the model is small; ~50k token steps in fp32 no_grad is seconds-to-
minutes). ERAG public text only.
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

from src.subconscious.token_lm import LMConfig, SSMLanguageModel  # noqa: E402
from src.subconscious.tokenizer_ import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    train_or_load_tokenizer,
)
from src.subconscious.training.strm_traces import ridge_fit  # noqa: E402

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_CKPT = "data/token_lm/token_lm_final.pt"
DEFAULT_TOK = "data/token_lm/tokenizer.json"
DEFAULT_OUTPUT_DIR = "data/probe/verbatim_reach"

# Log-spaced lags -- the falloff is what matters, not every integer k. Covers the
# full seq_len=256 range economically so the curve shape is visible without
# millions of (i,k) pairs.
DEFAULT_LAGS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192]


# --------------------------------------------------------------------- ERAG io
def _iter_erag_content(path: str, max_docs: int | None = None, skip: int = 0):
    """Stream ERAG ``content`` strings from the parquet (row-group by row-group).

    ``skip`` drops the first ``skip`` docs so the probe can read held-out docs
    beyond the training split (the trainer used the first 500 docs as val). The
    probe is about state dynamics, not LM generalization, but staying off the
    training val set is cleaner."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    seen = 0
    yielded = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        for content in tbl.column("content").to_pylist():
            if not content or not str(content).strip():
                continue
            if seen < skip:
                seen += 1
                continue
            yield str(content)
            yielded += 1
            if max_docs is not None and yielded >= max_docs:
                return


# -------------------------------------------------------------- state streaming
@torch.no_grad()
def stream_doc_states(
    model: SSMLanguageModel,
    token_ids: list[int],
    max_seq_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream one doc's tokens through the LM one at a time via ``step()``.

    Returns ``(states [T, n_layers*d_model] fp32, tok_ids [T] int64)`` for the first
    ``min(len(token_ids), max_seq_len)`` tokens. ``states[t]`` is the pooled
    recurrent state AFTER processing token ``t`` (mean over d_state per layer,
    concat all layers -- the recoverability head's "pooled" rep). The state at
    step ``t`` is what the SSM carries forward; pairing it with anchor ``i < t``
    measures how much of ``u_i`` survives ``k = t - i`` steps of recurrence."""
    ids = token_ids[:max_seq_len]
    T = len(ids)
    if T == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    batch = 1
    dtype = model.token_emb.weight.dtype
    # Prime with the first token via a one-step forward (init zero state, process
    # token 0). Then step() the rest. This captures s_t for EVERY t (forward()
    # over the whole sequence would only give the final state).
    states = np.empty((T, model.config.n_layers * model.config.d_model), dtype=np.float32)
    tok = np.empty((T,), dtype=np.int64)

    cur = [layer.init_state(batch, device, dtype) for layer in model.layers]
    for t, tid in enumerate(ids):
        tid_t = torch.tensor([tid], dtype=torch.long, device=device)
        _, cur = model.step(tid_t, cur)
        # Pool: per-layer [1, d_state, d_model] -> mean over d_state -> [1, d_model],
        # concat all layers -> [1, n_layers*d_model]. (Mirrors state_rep_pooled /
        # pool_encoder_states.)
        pooled = torch.cat([st.mean(dim=1).reshape(-1) for st in cur], dim=0)
        states[t] = pooled.to(torch.float32).cpu().numpy()
        tok[t] = tid
    return states, tok


# ----------------------------------------------------------------- pair builder
def build_pairs(
    chains: list[tuple[np.ndarray, np.ndarray]],
    anchor_emb: np.ndarray,
    lags: list[int],
    n_pairs_cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (state_t, anchor_emb_i, lag k) triples across chains.

    ``chains`` is a list of (states [T, n_layers*d_model], tok_ids [T]).
    ``anchor_emb`` is the full token embedding table [vocab, d_model] so
    ``u_i = anchor_emb[tok_i]``. For each chain, for each anchor ``i`` and each lag
    ``k`` in ``lags`` with ``t = i + k < T`` and ``tok_i`` not PAD (PAD is filler
    -- reconstructing it is meaningless), collect the triple. Subsample to
    ``n_pairs_cap`` balanced across lags so the ridge fit stays RAM-bounded (the
    state is n_layers*d_model-d). Returns
    ``(S [N, n_layers*d_model] fp32, U [N, d_model] fp32, K [N] int64, A [N] int64)``
    where ``A`` is the anchor token id (needed for the top-1 token-recovery metric)."""
    per_lag: dict[int, list[tuple[int, int, int]]] = {k: [] for k in lags}
    for ci, (states, tok) in enumerate(chains):
        T = len(tok)
        for i in range(T):
            ti = int(tok[i])
            if ti == PAD_ID:
                continue
            for k in lags:
                t = i + k
                if t >= T:
                    break
                per_lag[k].append((ci, i, t))
    # Subsample balanced across lags so no lag dominates the cap.
    per_lag_quota = max(1, n_pairs_cap // len(lags))
    chosen: list[tuple[int, int, int, int]] = []  # (ci, i, t, k)
    for k in lags:
        lst = per_lag[k]
        if len(lst) > per_lag_quota:
            idx = rng.choice(len(lst), size=per_lag_quota, replace=False)
            lst = [lst[j] for j in idx]
        for (ci, i, t) in lst:
            chosen.append((ci, i, t, k))
    N = len(chosen)
    S = np.empty((N, chains[0][0].shape[1]), dtype=np.float32)
    U = np.empty((N, anchor_emb.shape[1]), dtype=np.float32)
    K = np.empty((N,), dtype=np.int64)
    A = np.empty((N,), dtype=np.int64)  # anchor token id (for top-1 recovery)
    for n, (ci, i, t, k) in enumerate(chosen):
        states, tok = chains[ci]
        S[n] = states[t]
        ti = int(tok[i])
        U[n] = anchor_emb[ti]
        K[n] = k
        A[n] = ti
    return S, U, K, A


# ----------------------------------------------------------------------- driver
def run(args) -> int:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[probe] device={device}", flush=True)

    # ---- load model + tokenizer
    print(f"[load] ckpt={args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = LMConfig(**ckpt["config"])
    model = SSMLanguageModel(cfg).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[load] {model.num_parameters():,} params  d_model={cfg.d_model} "
          f"n_layers={cfg.n_layers} d_state={cfg.d_state} vocab={cfg.vocab}", flush=True)

    tok = train_or_load_tokenizer(iter([]), args.tokenizer, vocab_size=cfg.vocab)
    print(f"[load] tokenizer vocab_size={tok.vocab_size}", flush=True)

    # The anchor embedding table (u_i = token_emb[token_i]). Detach to numpy once.
    anchor_emb = model.token_emb.weight.detach().to(torch.float32).cpu().numpy()  # [vocab, d_model]

    # ---- stream docs -> chains of (states, tok_ids)
    print(f"[data] streaming {args.n_docs} ERAG docs (skip first {args.skip_docs}) "
          f"from {args.erag_path}", flush=True)
    t0 = time.time()
    chains: list[tuple[np.ndarray, np.ndarray]] = []
    for content in _iter_erag_content(args.erag_path, max_docs=args.n_docs,
                                      skip=args.skip_docs):
        ids = tok.encode(content)
        if len(ids) < 8:
            continue  # too short to be a useful chain
        st, tk = stream_doc_states(model, ids, args.max_seq_len, device)
        if len(tk) >= max(args.lags) + 1:
            chains.append((st, tk))
        if len(chains) >= args.n_docs:
            break
    print(f"[data] {len(chains)} chains  ({time.time() - t0:.1f}s)", flush=True)
    if len(chains) < 6:
        raise RuntimeError(f"need >=6 chains for a train/val split; got {len(chains)}")

    # ---- split by DOC (chain) -- no pair leakage
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(chains))
    n_val = max(2, int(round(len(chains) * args.val_fraction)))
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    tr_chains = [chains[i] for i in train_idx]
    va_chains = [chains[i] for i in val_idx]
    print(f"[split] {len(tr_chains)} train chains / {len(va_chains)} val chains",
          flush=True)

    # ---- build pairs (train + val), subsampled balanced across lags
    S_tr, U_tr, K_tr, A_tr = build_pairs(tr_chains, anchor_emb, args.lags,
                                         args.n_pairs, rng)
    S_va, U_va, K_va, A_va = build_pairs(va_chains, anchor_emb, args.lags,
                                         args.n_pairs, rng)
    print(f"[pairs] train={len(S_tr)}  val={len(S_va)}  "
          f"state_dim={S_tr.shape[1]}  anchor_dim={U_tr.shape[1]}", flush=True)

    # ---- ridge decoder D = ridge(state_t, u_i) -> reconstruct the anchor embedding
    print(f"[ridge] fitting decoder D (lam={args.lam})...", flush=True)
    W_d, b_d = ridge_fit(S_tr, U_tr, lam=args.lam)   # W_d [state_dim, d_model], b_d [d_model]
    pred_va = S_va @ W_d + b_d                         # [N, d_model]
    err_va = ((pred_va - U_va) ** 2).mean(axis=1)      # [N] -- the forgetting signal e(i,t)
    err_tr = ((S_tr @ W_d + b_d - U_tr) ** 2).mean(axis=1)

    # ---- chance floor: constant predictor = mean(U_train). e when the state
    # carries NO anchor info = the variance of the anchor embeddings. e(k) hitting
    # this floor = the verbatim-reach horizon (the state no longer distinguishes
    # the anchor from the average token).
    mean_u = U_tr.mean(axis=0)                          # [d_model]
    chance_floor = float(((U_va - mean_u) ** 2).mean())  # scalar

    # ---- decay curve: mean e by lag (val) -- the forgetting signal vs recency
    decay: dict[int, float] = {}
    decay_n: dict[int, int] = {}
    for k in args.lags:
        m = K_va == k
        if m.any():
            decay[k] = float(err_va[m].mean())
            decay_n[k] = int(m.sum())
        else:
            decay[k] = float("nan")
            decay_n[k] = 0

    # ---- top-1 token-ID recovery (interpretable verbatim reach):
    # nearest token in the embedding table to the predicted embedding == true token?
    # argmax cosine(pred, emb_table). chance = 1/vocab. Computed on a subsample of
    # val (cos against [vocab,256] for every pair is the heavy step; cap it).
    n_top1 = min(len(S_va), args.n_top1)
    top1_idx = rng.choice(len(S_va), size=n_top1, replace=False)
    pred_top1 = pred_va[top1_idx]                       # [n, d_model]
    true_tok = A_va[top1_idx]                           # [n] anchor token ids
    lag_top1 = K_va[top1_idx]                           # [n]
    # cosine(pred, emb_table) -> argmax token. Normalize both for cosine.
    emb_n = anchor_emb / (np.linalg.norm(anchor_emb, axis=1, keepdims=True) + 1e-12)
    pred_n = pred_top1 / (np.linalg.norm(pred_top1, axis=1, keepdims=True) + 1e-12)
    sims = pred_n @ emb_n.T                             # [n, vocab]
    pred_tok = sims.argmax(axis=1)                      # [n]
    correct = (pred_tok == true_tok).astype(np.float32)
    top1_acc_by_lag: dict[int, float] = {}
    top1_n_by_lag: dict[int, int] = {}
    for k in args.lags:
        m = lag_top1 == k
        if m.any():
            top1_acc_by_lag[k] = float(correct[m].mean())
            top1_n_by_lag[k] = int(m.sum())
        else:
            top1_acc_by_lag[k] = float("nan")
            top1_n_by_lag[k] = 0
    top1_overall = float(correct.mean())
    top1_chance = 1.0 / cfg.vocab

    # ---- shape verdict: does e(k) rise GRACEFULLY or CLIFF?
    # Compare e at small k (k=1) to e at large k (max lag) and to the chance floor.
    e_k1 = decay.get(1, float("nan"))
    e_kmax = decay.get(max(args.lags), float("nan"))
    # Reach horizon: smallest k where e(k) >= chance_floor (within tolerance).
    horizon = None
    for k in args.lags:
        if not np.isnan(decay[k]) and decay[k] >= chance_floor * 0.9:
            horizon = k
            break
    graceful = (not np.isnan(e_k1) and not np.isnan(e_kmax)
                and e_k1 < e_kmax * 0.6)  # e(1) well below e(kmax) -> spread, not cliff

    print(f"\n[result] chance floor (constant predictor) = {chance_floor:.4f}", flush=True)
    print(f"[result] e(k=1)={e_k1:.4f}  e(k={max(args.lags)})={e_kmax:.4f}  "
          f"ratio e_kmax/chance={e_kmax / chance_floor:.3f}", flush=True)
    print(f"[result] verbatim-reach horizon (e >= 0.9*chance): "
          f"{horizon if horizon is not None else 'not reached in range'}", flush=True)
    print(f"[result] shape: {'GRACEFUL (spread)' if graceful else 'CLIFF / flat'}", flush=True)
    print("\n[decay curve] lag k : mean e(val)  n", flush=True)
    for k in args.lags:
        if not np.isnan(decay[k]):
            frac = decay[k] / chance_floor
            print(f"  k={k:>4} : e={decay[k]:.4f}  ({frac:.2f}x chance)  n={decay_n[k]}",
                  flush=True)

    print(f"\n[top-1 token recovery] overall={top1_overall:.4f}  "
          f"chance(1/vocab)={top1_chance:.5f}", flush=True)
    print("[top-1 curve] lag k : top-1 acc  n", flush=True)
    for k in args.lags:
        if not np.isnan(top1_acc_by_lag[k]):
            x = top1_acc_by_lag[k] / top1_chance
            print(f"  k={k:>4} : acc={top1_acc_by_lag[k]:.4f}  ({x:.1f}x chance)  "
                  f"n={top1_n_by_lag[k]}", flush=True)

    summary = {
        "probe": "verbatim_reach",
        "purpose": "token-content decay curve (FADE Regime 1->2/3 boundary, text leg)",
        "checkpoint": args.checkpoint,
        "n_chains": len(chains),
        "n_train_chains": len(tr_chains),
        "n_val_chains": len(va_chains),
        "n_train_pairs": len(S_tr),
        "n_val_pairs": len(S_va),
        "state_dim": int(S_tr.shape[1]),
        "anchor_dim": int(U_tr.shape[1]),
        "lags": args.lags,
        "lam": args.lam,
        "chance_floor": chance_floor,
        "decay": decay,
        "decay_n": decay_n,
        "e_k1": e_k1,
        "e_kmax": e_kmax,
        "horizon": horizon,
        "shape": "graceful" if graceful else "cliff",
        "top1_overall": top1_overall,
        "top1_chance": top1_chance,
        "top1_by_lag": top1_acc_by_lag,
        "top1_n_by_lag": top1_n_by_lag,
        "model_config": cfg.__dict__,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"\n[summary] wrote {out_dir / 'run_summary.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verbatim-reach probe (token-content decay).")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--erag-path", default=ERAG_PATH)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-docs", type=int, default=64,
                    help="number of ERAG docs to stream as chains")
    ap.add_argument("--skip-docs", type=int, default=500,
                    help="skip the first N docs (stay off the trainer's val split)")
    ap.add_argument("--max-seq-len", type=int, default=256,
                    help="cap tokens per chain (matches the LM training seq_len)")
    ap.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS,
                    help="lag values k = t - i to sweep (log-spaced by default)")
    ap.add_argument("--n-pairs", type=int, default=60_000,
                    help="cap on (s_t, u_i) pairs, balanced across lags (RAM bound)")
    ap.add_argument("--n-top1", type=int, default=8000,
                    help="val pairs for the top-1 token-recovery metric")
    ap.add_argument("--lam", type=float, default=10.0, help="ridge penalty")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="auto=cpu (probe is tiny); or cuda")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cpu"  # the probe is tiny; CPU avoids bf16/dtype fuss
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return run(args)


if __name__ == "__main__":
    sys.exit(main())