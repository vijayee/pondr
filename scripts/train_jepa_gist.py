"""Train the JEPA-latent gist objective (``src/subconscious/jepa_gist.py``).

The Stage-1 pivot from the failed summary-CE retrain (``docs/gist-retrain-result.md``).
That retrain proved the plain summary objective does NOT shape gist-recoverable content
into the SSM state: the decoder SHORTCUTS to the marginal summary distribution and
ignores the state. The fix is to predict the *latent* of the gist (the bge embedding of
the gist text), not the tokens. A lossy latent-space prediction has no generic-token
prior to shortcut to; the contrastive in-batch negatives make doc-specificity the
EXPLICIT objective (the prediction must be closer to the true gist's bge embedding
than to other docs', so the state is the only source of doc-specific signal).

Objective per ERAG doc:
    doc_ids -> encoder -> per-layer states -> pool -> predictor -> pred [b, 384]
    target  = bge_small.encode(flash_summarize(doc_content))   # 384-d, FROZEN, L2-norm
    loss    = jepa_contrastive_loss(pred, target, negatives, temperature)
              + lm_prior_weight * next_token_CE(encoder_logits, doc_ids)

The ``lm_prior_weight * next_token_CE`` is the anti-collapse fix the failed retrain
lacked: it directly penalizes the continuation-prior blowup that went 8.1x last time.
The encoder is the token-LM (d_model=256, 6 layers); the predictor projects the pooled
state (1536-d) to the 384-d bge latent space. ``--train-encoder`` thaws the encoder so
the JEPA-latent loss + LM-prior flow back through the predictor -> encoder recurrence
and reshape the continuation-state into a gist-shaped state.

The gate itself (latent-space swap) runs in ``scripts/eval_jepa_gist.py``; this script
produces the checkpoint + val latent-cosine + the collapse watchdog.

Reuse-first: ``_iter_erag_pairs`` / ``flash_summarize`` / the two-group optim +
warmup-thaw pattern are imported from ``train_gist_readout``; the collapse watchdog
(``eval_continuation_perplexity`` on held-out ERAG chunks) is imported from
``train_token_lm``. New code: the bge latent cache build + the JEPA loss wiring.

Public ERAG only -- no onyx, no private transcripts. Teacher gist text via
deepseek-flash (flash-over-pro, already cached in ``gist_teacher_cache_v2.json``).
bge-small is a frozen open model (the latent target). Checkpoints under
``--output-dir``; upload to HF private after a PASS (a FAIL's ckpt is not uploaded,
per policy).

Usage (the retrain, 5080 bf16):
    python scripts/train_jepa_gist.py --train-encoder --steps 2500 \
        --encoder-warmup-steps 300 --encoder-lr-scale 0.1 --lm-prior-weight 0.1 \
        --num-negatives 16 --gist-seq-len 256 \
        --encoder-checkpoint data/token_lm/token_lm_final.pt \
        --tokenizer-cache data/token_lm/tokenizer.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.subconscious.jepa_gist import (  # noqa: E402
    JEPAGistModel,
    LatentPredictorConfig,
    save_jepa_checkpoint,
)
from src.subconscious.token_lm import LMConfig, SSMLanguageModel  # noqa: E402
from src.subconscious.tokenizer_ import PAD_ID, train_or_load_tokenizer  # noqa: E402
from src.subconscious.training.jepa_loss import jepa_contrastive_loss  # noqa: E402
from train_gist_readout import (  # noqa: E402
    ERAG_PATH,
    _cosine_warmup_lr,
    _iter_erag_pairs,
    flash_summarize,
)
from train_token_lm import (  # noqa: E402
    build_dataset as build_continuation_chunks,
    eval_perplexity as eval_continuation_perplexity,
)

DEFAULT_TOK_CACHE = "data/token_lm/tokenizer.json"
DEFAULT_ENCODER_CKPT = "data/token_lm/token_lm_final.pt"
DEFAULT_OUTPUT_DIR = "data/jepa_gist"


# ----------------------------------------------------------- device/dtype (copied)
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    return torch.float32


# --------------------------------------------------------------- bge latent cache
def _build_bge_embedder(device: torch.device):
    """Load the frozen bge-small-en-v1.5 teacher (384-d) via sentence-transformers.

    Reuses ``src.retrieval.vector_search._sentence_transformers_embedder`` (the same
    model the runtime retriever uses). Returns a SentenceTransformer whose
    ``.encode`` yields 384-d numpy arrays. The target is FROZEN (it cannot collapse --
    the anti-collapse mechanism is the contrastive negatives, not an EMA target).
    """
    from src.retrieval.vector_search import _sentence_transformers_embedder  # noqa: E402

    emb = _sentence_transformers_embedder()
    # sentence-transformers ignores the torch device for CPU; for GPU, move the
    # model explicitly so the one-time encode of ~8k gist texts is fast.
    if device.type == "cuda" and hasattr(emb, "_target_device"):
        try:
            emb._target_device = device
        except Exception:
            pass
    return emb


def _bge_encode(emb, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts -> [N, 384] L2-normalized float32 numpy array."""
    vecs = emb.encode(texts, batch_size=batch_size, show_progress_bar=False,
                      convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def build_latent_cache(gist_cache: dict, out_path: Path, device: torch.device,
                       batch_size: int = 64) -> dict[str, np.ndarray]:
    """bge-encode all cached gist texts -> {sha1(content): 384-d np.ndarray}.

    Persists ``gist_latent_cache_v2.npz`` (keys + vectors stacked) so reruns are
    free. The keys are the sha1(content) hashes from ``gist_teacher_cache_v2.json``;
    the vectors are L2-normalized bge embeddings (the JEPA target space). Sorted by
    key so the npz key order is deterministic across runs.
    """
    if out_path.exists():
        z = np.load(out_path, allow_pickle=False)
        keys = [str(k) for k in z["keys"]]
        vecs = z["vectors"]
        cache = {k: vecs[i] for i, k in enumerate(keys)}
        print(f"[latent] loaded {len(cache)} cached latents from {out_path}",
              flush=True)
        return cache

    keys = sorted(gist_cache.keys())
    texts = [gist_cache[k] for k in keys]
    print(f"[latent] bge-encoding {len(texts)} gist texts (one-time)...",
          flush=True)
    t0 = time.time()
    emb = _build_bge_embedder(device)
    vecs = _bge_encode(emb, texts, batch_size=batch_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed-width unicode dtype (sha1 hex = 40 chars) so the npz loads with
    # allow_pickle=False (an object-array dtype would require allow_pickle=True).
    key_width = max((len(k) for k in keys), default=1)
    np.savez(out_path, keys=np.array(keys, dtype=f"U{key_width}"),
             vectors=vecs.astype(np.float32))
    cache = {k: vecs[i] for i, k in enumerate(keys)}
    print(f"[latent] encoded {len(cache)} latents (384-d, L2-norm) in "
          f"{time.time() - t0:.1f}s -> {out_path}", flush=True)
    return cache


# ----------------------------------------------------------------- val metrics
@torch.no_grad()
def eval_val_latent(model: JEPAGistModel, val_batch, latent_targets: torch.Tensor,
                    negatives: torch.Tensor, temperature: float, device) -> tuple[float, float]:
    """Val latent-cosine (mean cos(pred, target)) + val contrastive loss.

    ``val_batch``: list of doc-id lists. ``latent_targets``: [b, 384] L2-normed bge
    latents for those docs. ``negatives``: [n, 384] sampled from the full pool.
    """
    model.eval()
    doc_ids = torch.tensor(val_batch, dtype=torch.long, device=device)
    enc_states = model.encode(doc_ids, no_grad=True)
    pred = model.predict_latent(enc_states)
    targets = latent_targets.to(device=device, dtype=pred.dtype)
    neg = negatives.to(device=device, dtype=pred.dtype)
    cos = F.cosine_similarity(pred, targets, dim=-1).mean().item()
    loss = jepa_contrastive_loss(pred, targets, neg, temperature).item()
    # restore train mode if the caller is mid-train (the caller re-asserts).
    return cos, loss


# ------------------------------------------------------------------- training
def train(args) -> int:
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"[device] {device}  [dtype] {dtype}", flush=True)

    # ---- tokenizer (shared with the encoder).
    tok = train_or_load_tokenizer(iter([]), args.tokenizer_cache, vocab_size=args.vocab)
    print(f"[tok] vocab_size={tok.vocab_size} (cache={args.tokenizer_cache})", flush=True)

    # ---- encoder: load the token-LM checkpoint strict. Thawed by the model build
    # when --train-encoder; the warmup logic below freezes it for the predictor-only
    # phase, then thaws it so the JEPA-latent loss reshapes the state end-to-end.
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    enc_cfg = LMConfig(**enc_ckpt["config"])
    encoder = SSMLanguageModel(enc_cfg).to(device=device, dtype=dtype)
    enc_sd = enc_ckpt["model"] if "model" in enc_ckpt else enc_ckpt
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder checkpoint mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    if args.train_encoder:
        print(f"[encoder] loaded {args.encoder_checkpoint} "
              f"({encoder.num_parameters():,} params, TRAINABLE -- retrain)", flush=True)
    else:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        print(f"[encoder] loaded {args.encoder_checkpoint} "
              f"({encoder.num_parameters():,} params, FROZEN)", flush=True)

    # ---- JEPA-gist model (predictor + encoder; predictor always trainable).
    latent_cfg = LatentPredictorConfig(
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        n_mlp_layers=args.n_mlp_layers,
    )
    model = JEPAGistModel(encoder, latent_cfg,
                          freeze_encoder=not args.train_encoder).to(device=device, dtype=dtype)
    if not args.train_encoder:
        model._freeze_encoder()  # .to() does not flip requires_grad; re-assert.
    n_train = model.trainable_parameters()
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"[model] predictor {n_pred:,} trainable; encoder {n_enc:,} params "
          f"({'thawed after warmup' if args.train_encoder else 'FROZEN'}); "
          f"latent_dim={latent_cfg.latent_dim}", flush=True)

    # ---- data: stream ERAG, split DOCS train/val (held-out docs, not tokens).
    print(f"[data] streaming ERAG from {ERAG_PATH}", flush=True)
    t0 = time.time()
    val_raw: list[tuple[str, str]] = []
    train_raw: list[tuple[str, str]] = []
    need = args.max_train_docs + args.val_docs
    for title, content in _iter_erag_pairs(ERAG_PATH, max_docs=need):
        if len(val_raw) < args.val_docs:
            val_raw.append((title, content))
        else:
            train_raw.append((title, content))
            if len(train_raw) >= args.max_train_docs:
                break
    print(f"[data] {len(train_raw)} train docs, {len(val_raw)} val docs "
          f"({time.time() - t0:.1f}s)", flush=True)

    # ---- gist text targets: from the deepseek-flash teacher cache (filled by
    # gen_gist_cache.py). --require-cached-gists skips cache misses WITHOUT calling
    # Ollama (for the CPU grad-flow sanity); otherwise flash_summarize fills misses
    # via Ollama (the real run). mkdir the output dir EARLY: flash_summarize writes
    # the cache here, before the optimizer-section mkdir below, so the dir must
    # already exist (a latent ordering bug -- the prior retrain only worked because
    # data/gist_readout pre-existed).
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "gist_teacher_cache_v2.json"
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"[target] gist teacher cache={cache_path} ({len(cache)} entries); "
          f"require_cached={args.require_cached_gists}", flush=True)

    def _to_gist_pairs(raw):
        pairs = []
        for i, (title, content) in enumerate(raw):
            key = hashlib.sha1(content.encode("utf-8")).hexdigest()
            if key in cache and cache[key]:
                pairs.append((content, cache[key]))
                continue
            if args.require_cached_gists:
                continue  # skip misses without Ollama (CPU sanity path)
            g = flash_summarize(content, cache, cache_path)
            if g:
                pairs.append((content, g))
            if (i + 1) % 200 == 0:
                print(f"  [gist] {i + 1}/{len(raw)} ({time.time() - t0:.0f}s)",
                      flush=True)
        return pairs

    t0 = time.time()
    train_pairs_raw = _to_gist_pairs(train_raw)
    val_pairs_raw = _to_gist_pairs(val_raw)
    print(f"[target] {len(train_pairs_raw)} train gists, {len(val_pairs_raw)} val "
          f"gists ({time.time() - t0:.0f}s)", flush=True)
    if len(train_pairs_raw) < args.batch_size:
        raise RuntimeError(
            f"too few cached gist pairs ({len(train_pairs_raw)}); fill the cache "
            f"with gen_gist_cache.py or run with Ollama up (drop --require-cached-gists)"
        )

    # ---- bge latent cache: encode all cached gist texts ONCE -> {sha1: [384]}.
    latent_cache_path = Path(args.output_dir) / "gist_latent_cache_v2.npz"
    latent_cache = build_latent_cache(cache, latent_cache_path, device,
                                      batch_size=args.bge_batch_size)
    # Stack the full pool once for negative sampling (a shared [Npool, 384] tensor).
    pool_keys = sorted(latent_cache.keys())
    pool_vecs = np.stack([latent_cache[k] for k in pool_keys], axis=0).astype(np.float32)
    pool_tensor = torch.from_numpy(pool_vecs)  # [Npool, 384], CPU; moved to device per-step
    print(f"[latent] pool {pool_tensor.shape[0]} latents for negative sampling",
          flush=True)

    def _content_sha1(content: str) -> str:
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    # ---- tokenize doc content -> doc_ids (trunc/pad to doc_seq_len). The gist TEXT
    # is NOT tokenized (the target is its bge latent, not tokens); we only need the
    # sha1 -> latent lookup. Drop pairs whose gist text has no latent (shouldn't
    # happen -- every cached gist was encoded -- but be defensive). ``encode_batch``
    # truncates + pads to ``max_length`` (the tokenizer wrapper has no per-doc
    # ``encode(text, max_length=...)`` -- only the batch variant takes max_length).
    def _tok_and_align(raw):
        kept = [(content, gist) for content, gist in raw
                if _content_sha1(content) in latent_cache]
        if not kept:
            return []
        docs = tok.encode_batch([c for c, _ in kept], max_length=args.doc_seq_len)
        keys = [_content_sha1(c) for c, _ in kept]
        return list(zip(docs, keys))

    train_pairs = _tok_and_align(train_pairs_raw)
    val_pairs = _tok_and_align(val_pairs_raw)
    print(f"[data] {len(train_pairs)} train pairs, {len(val_pairs)} val pairs "
          f"(doc->latent)", flush=True)

    # ---- optimizer. Two-group (retrain): predictor at args.lr (group 0), encoder at
    # 0 during warmup then args.lr * args.encoder_lr_scale (group 1). Probe path:
    # predictor only.
    pred_params = list(model.predictor.parameters())
    if args.train_encoder:
        enc_params = list(model.encoder.parameters())
        optim = torch.optim.AdamW(
            [
                {"params": pred_params, "lr": args.lr},          # group 0: predictor
                {"params": enc_params, "lr": 0.0},               # group 1: encoder (0 in warmup)
            ],
            betas=(0.9, 0.95), weight_decay=args.weight_decay,
        )
    else:
        optim = torch.optim.AdamW(pred_params, lr=args.lr, betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)  # already created above (before gist generation)

    n = len(train_pairs)
    batch_size = args.batch_size
    total_steps = args.steps
    print(f"[train] {n} pairs, batch {batch_size}, {total_steps} total steps, "
          f"num_negatives={args.num_negatives}, temperature={args.temperature}, "
          f"lm_prior_weight={args.lm_prior_weight}", flush=True)

    # ---- collapse watchdog (retrain only): encoder continuation val ppl on held-out
    # ERAG chunks (the val docs -- held-out from gist TRAINING). Reuses
    # train_token_lm.eval_perplexity on model.encoder. A blow-up vs the frozen
    # baseline means the language prior was destroyed (the retrain collapsed) -- the
    # exact failure of the summary-CE retrain (175 -> 1419, 8.1x). The LM-prior
    # auxiliary is the fix; this watches whether it held.
    enc_val_ppl_start = float("nan")
    enc_val_ppl_end = float("nan")
    watchdog_chunks: list[list[int]] | None = None
    if args.train_encoder:
        tc = build_continuation_chunks([c for _, c in val_raw], tok, enc_cfg.seq_len)
        watchdog_chunks = tc.chunks
        enc_val_ppl_start = eval_continuation_perplexity(
            model.encoder, watchdog_chunks, batch_size, device, enc_cfg.vocab)
        print(f"[watchdog] encoder continuation val ppl (start, frozen baseline) = "
              f"{enc_val_ppl_start:.2f}", flush=True)

    # ---- warmup: freeze the encoder for the first --encoder-warmup-steps so the
    # predictor learns to read the (continuation-shaped) frozen state first, THEN
    # thaw so gradient pressure falls on the encoder only where the predictor cannot
    # already solve it.
    encoder_unfrozen = not (args.train_encoder and args.encoder_warmup_steps > 0)
    if args.train_encoder and not encoder_unfrozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
        print(f"[warmup] encoder FROZEN for first {args.encoder_warmup_steps} steps "
              f"(predictor-only), then thaws", flush=True)

    model.predictor.train()
    step = 0
    running_loss = 0.0
    running_lat = 0.0
    running_lm = 0.0
    running_n = 0
    t_start = time.time()
    while step < total_steps:
        perm = torch.randperm(n).tolist()
        for bi in range(0, n, batch_size):
            if step >= total_steps:
                break
            idx = perm[bi:bi + batch_size]
            if not idx:
                continue
            # Thaw the encoder at the end of the predictor-only warmup.
            if args.train_encoder and not encoder_unfrozen and step >= args.encoder_warmup_steps:
                for p in model.encoder.parameters():
                    p.requires_grad = True
                model.train()
                encoder_unfrozen = True
                enc_lr = _cosine_warmup_lr(step, args.warmup, total_steps, args.lr) \
                         * args.encoder_lr_scale
                optim.param_groups[1]["lr"] = enc_lr
                print(f"[warmup] encoder UNFROZEN at step {step}; "
                      f"enc_lr={enc_lr:.2e}", flush=True)

            doc_ids = torch.tensor([train_pairs[i][0] for i in idx],
                                   dtype=torch.long, device=device)
            keys = [train_pairs[i][1] for i in idx]
            targets = torch.from_numpy(
                np.stack([latent_cache[k] for k in keys]).astype(np.float32)
            ).to(device=device, dtype=dtype)  # [b, 384]

            # Sample num_negatives from the full latent pool (shared negatives; the
            # anti-collapse / anti-shortcut mechanism). Random draw each step.
            neg_idx = torch.randint(0, pool_tensor.shape[0], (args.num_negatives,))
            negatives = pool_tensor[neg_idx].to(device=device, dtype=dtype)  # [n, 384]

            # Encode: during the predictor-only warmup the encoder is FROZEN, so we
            # run one no_grad encoder forward (detached states + detached lm_logits)
            # and the predictor STILL trains -- grad flows through the predictor
            # weights on the detached states (a backward through pred reaches the
            # predictor params even when the state input is detached). Once the
            # encoder is thawed, the full grad-flowing forward reshapes the encoder
            # state end-to-end. (The no_grad=True path of model.forward is NOT used
            # here because it detaches pred too, killing the predictor's grad.)
            thawed = args.train_encoder and encoder_unfrozen
            with torch.autocast(device_type=device.type, enabled=(dtype != torch.float32),
                                 dtype=dtype):
                if thawed:
                    pred, lm_logits, _enc_states = model.forward(doc_ids, no_grad=False)
                else:
                    with torch.no_grad():
                        model.encoder.eval()
                        lm_logits, enc_states = model.encoder.forward(doc_ids)
                    lm_logits = lm_logits.detach()
                    enc_states = [s.detach() for s in enc_states]
                    pred = model.predict_latent(enc_states)
                latent_loss = jepa_contrastive_loss(
                    pred, targets, negatives, args.temperature)
                # LM-prior auxiliary: next-token CE on the doc (the encoder's own
                # logits). Anti-collapse: penalizes the continuation-prior blowup.
                if args.lm_prior_weight > 0:
                    lm_logits = lm_logits[:, :-1, :].float().reshape(-1, enc_cfg.vocab)
                    lm_targets = doc_ids[:, 1:].reshape(-1)
                    lm_loss = F.cross_entropy(lm_logits, lm_targets,
                                              ignore_index=PAD_ID)
                    loss = latent_loss + args.lm_prior_weight * lm_loss
                else:
                    lm_loss = latent_loss.new_zeros(())
                    loss = latent_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            clip_params = model.parameters() if (args.train_encoder and encoder_unfrozen) \
                         else model.predictor.parameters()
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            lr = _cosine_warmup_lr(step, args.warmup, total_steps, args.lr)
            optim.param_groups[0]["lr"] = lr
            if args.train_encoder and encoder_unfrozen:
                optim.param_groups[1]["lr"] = lr * args.encoder_lr_scale
            optim.step()

            running_loss += loss.item()
            running_lat += latent_loss.item()
            running_lm += float(lm_loss.item())
            running_n += 1
            step += 1
            if step % args.log_every == 0 or step == 1:
                k = running_n
                mean_loss = running_loss / max(k, 1)
                mean_lat = running_lat / max(k, 1)
                mean_lm = running_lm / max(k, 1)
                rate = step / (time.time() - t_start)
                watchdog_str = ""
                if watchdog_chunks is not None:
                    wd_ppl = eval_continuation_perplexity(
                        model.encoder, watchdog_chunks, batch_size, device, enc_cfg.vocab)
                    watchdog_str = f"  enc_val_ppl {wd_ppl:.1f}"
                    # eval_perplexity set the encoder to eval; restore train mode.
                    if args.train_encoder and encoder_unfrozen:
                        model.train()
                    else:
                        model.predictor.train()
                print(f"[step {step:>5}/{total_steps}] loss {mean_loss:.3f}  "
                      f"lat {mean_lat:.3f}  lm {mean_lm:.3f}  lr {lr:.2e}  "
                      f"{rate:.2f} step/s{watchdog_str}", flush=True)
                running_loss = 0.0
                running_lat = 0.0
                running_lm = 0.0
                running_n = 0
            if step % args.checkpoint_every == 0:
                save_jepa_checkpoint(out_dir / f"jepa_step{step}.pt", model, step,
                                     Path(args.encoder_checkpoint).name,
                                     save_encoder=args.train_encoder)

    # ---- final checkpoint + val metrics + final watchdog.
    final_path = out_dir / "jepa_final.pt"
    save_jepa_checkpoint(final_path, model, step,
                         Path(args.encoder_checkpoint).name,
                         save_encoder=args.train_encoder)
    print(f"[ckpt] saved {final_path}", flush=True)

    # Val latent-cosine + val contrastive loss on held-out docs.
    val_doc_ids = [d for d, _ in val_pairs]
    val_keys = [k for _, k in val_pairs]
    val_latent_cos = float("nan")
    val_contrastive_loss = float("nan")
    if val_doc_ids:
        val_targets = torch.from_numpy(
            np.stack([latent_cache[k] for k in val_keys]).astype(np.float32)
        )
        neg_idx = torch.randint(0, pool_tensor.shape[0], (args.num_negatives,))
        val_neg = pool_tensor[neg_idx]
        # eval_val_latent handles its own batching for the whole val set at once.
        val_latent_cos, val_contrastive_loss = eval_val_latent(
            model, val_doc_ids, val_targets, val_neg, args.temperature, device)
        # restore train mode if retrain mid-train (no-op here -- training is done).
        print(f"[val] held-out-docs latent cosine = {val_latent_cos:.4f}  "
              f"contrastive loss = {val_contrastive_loss:.3f}", flush=True)

    if watchdog_chunks is not None:
        enc_val_ppl_end = eval_continuation_perplexity(
            model.encoder, watchdog_chunks, batch_size, device, enc_cfg.vocab)
        print(f"[watchdog] encoder continuation val ppl (end) = {enc_val_ppl_end:.2f}  "
              f"(start {enc_val_ppl_start:.2f}; collapse if >>x baseline)",
              flush=True)

    summary = {
        "objective": "jepa_latent",
        "latent_dim": latent_cfg.latent_dim,
        "hidden": latent_cfg.hidden,
        "n_mlp_layers": latent_cfg.n_mlp_layers,
        "lm_prior_weight": args.lm_prior_weight,
        "num_negatives": args.num_negatives,
        "temperature": args.temperature,
        "trainable_params": n_train,
        "predictor_params": n_pred,
        "vocab": tok.vocab_size,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "steps": total_steps,
        "val_latent_cos": val_latent_cos,
        "val_contrastive_loss": val_contrastive_loss,
        "uniform_ce": math.log(tok.vocab_size),
        "doc_seq_len": args.doc_seq_len,
        "encoder_ref": Path(args.encoder_checkpoint).name,
        # Retrain-only fields (NaN for the frozen-predictor path):
        "train_encoder": args.train_encoder,
        "encoder_lr_scale": args.encoder_lr_scale,
        "encoder_warmup_steps": args.encoder_warmup_steps,
        "encoder_val_ppl_start": enc_val_ppl_start,
        "encoder_val_ppl_end": enc_val_ppl_end,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"[summary] wrote {out_dir / 'run_summary.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the JEPA-latent gist objective.")
    # data
    ap.add_argument("--max-train-docs", type=int, default=8_000)
    ap.add_argument("--val-docs", type=int, default=200)
    ap.add_argument("--tokenizer-cache", default=DEFAULT_TOK_CACHE)
    ap.add_argument("--encoder-checkpoint", default=DEFAULT_ENCODER_CKPT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    # model
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--latent-dim", type=int, default=384,
                    help="bge latent dimension (384 for bge-small-en-v1.5)")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-mlp-layers", type=int, default=3)
    ap.add_argument("--doc-seq-len", type=int, default=128)
    # bge latent cache
    ap.add_argument("--bge-batch-size", type=int, default=64,
                    help="batch size for the one-time bge encode of cached gist texts")
    # training
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm-prior-weight", type=float, default=0.1,
                    help="weight on the next-token CE auxiliary (anti-collapse; the "
                         "summary-CE retrain collapsed 8.1x without this)")
    ap.add_argument("--num-negatives", type=int, default=16,
                    help="contrastive negatives sampled from the full latent pool per step")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--require-cached-gists", action="store_true",
                    help="skip gist-cache misses WITHOUT calling Ollama (CPU grad-flow "
                         "sanity); off = flash_summarize fills misses via Ollama (real run)")
    # encoder training (the retrain). Default OFF = frozen-encoder predictor-only.
    ap.add_argument("--train-encoder", action="store_true",
                    help="unfreeze the encoder and train end-to-end (the retrain); "
                         "off = frozen-encoder predictor-only path")
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1,
                    help="encoder LR = this x predictor LR (differential LR; the encoder "
                         "reshapes slowly, the predictor adapts to read it)")
    ap.add_argument("--encoder-warmup-steps", type=int, default=0,
                    help="predictor-only warmup: the encoder is frozen for this many "
                         "steps then thaws (0 = thaw immediately, no warmup)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return train(args)


if __name__ == "__main__":
    sys.exit(main())