"""Train the frozen-encoder state->gist readout (``src/subconscious/gist_readout.py``).

The §3.3 content probe done right (post-mortem §6 + §3 item 1): freeze the trained
token-LM as an encoder, train ONLY a small decoder (+ per-layer state projection)
that reads the encoder's final recurrent state and generates a gist, on (doc ->
gist) supervised pairs from public ERAG. Held-out DOCS (§3.9). The gate itself
(faithfulness + swap control) runs in ``scripts/eval_gist_readout.py``; this script
just produces the checkpoint + a perplexity sanity number.

Targets (``--target``):
  * ``title``  -- ERAG ``content -> title`` (free, 512k pairs; weak gist but a valid
    long->short supervised signal, fine for the swap-control test).
  * ``gist``   -- ERAG ``content -> deepseek-flash one-sentence gist`` (cached by
    sha1(content); higher quality; needs local Ollama at :11434).

Self-contained: device/dtype/LR helpers + the flash-gist teacher are copied here
(decoupled from the bge/JEPA pretrainer and from ``scripts/_scratch/``, same
precedent as ``train_token_lm.py``). Public ERAG only -- no onyx, no private
transcripts. Checkpoints under ``--output-dir``; upload to HF private separately.

Usage (local, frozen encoder = the trained token-LM):
    python scripts/train_gist_readout.py --target title --steps 1500 \
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

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.gist_readout import (  # noqa: E402
    GistConfig,
    GistReadoutModel,
    save_gist_checkpoint,
)
from src.subconscious.token_lm import LMConfig, SSMLanguageModel  # noqa: E402
from src.subconscious.tokenizer_ import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    train_or_load_tokenizer,
)

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_TOK_CACHE = "data/token_lm/tokenizer.json"
DEFAULT_ENCODER_CKPT = "data/token_lm/token_lm_final.pt"
DEFAULT_OUTPUT_DIR = "data/gist_readout"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-v4-flash:cloud"   # flash over pro (memory: Oracle labeling)
LLM_TIMEOUT = 60


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


def _cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ----------------------------------------------------------- flash gist teacher (copied)
def flash_summarize(content: str, cache: dict, cache_path: Path) -> str | None:
    """One-sentence gist of ``content`` via local Ollama flash. Cached by sha1.

    Lifted from ``scripts/_scratch/_content_probe_stageB.py`` (the literal §3.3
    Stage-B teacher) and kept self-contained here so the trainer does not import
    from ``_scratch/``. Public ERAG content only.
    """
    key = hashlib.sha1(content.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    prompt = (
        "Summarize the following document in ONE concise sentence. "
        "Capture the key topic, decision, or content. Reply with only the sentence.\n\n"
        + content[:4000]
    )
    import urllib.request
    payload = json.dumps({"model": LLM_MODEL, "prompt": prompt,
                          "stream": False, "options": {"temperature": 0.2,
                                                        "num_predict": 128}}).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            resp = json.loads(r.read().decode("utf-8"))
        text = resp.get("response", "").strip()
        if not text:
            return None
        text = text.split("\n")[0].strip()
        cache[key] = text
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        return text
    except Exception as e:
        print(f"  [llm warn] {e}", flush=True)
        return None


# ----------------------------------------------------------------------- data
def _iter_erag_pairs(path: str, max_docs: int | None = None):
    """Stream (title, content) from the ERAG documents parquet."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["title", "content"])
        titles = tbl.column("title").to_pylist()
        contents = tbl.column("content").to_pylist()
        for title, content in zip(titles, contents):
            if content and str(content).strip() and title and str(title).strip():
                yield str(title), str(content)
                seen += 1
                if max_docs is not None and seen >= max_docs:
                    return


@torch.no_grad()
def eval_perplexity(model: GistReadoutModel, pairs, batch_size, device) -> float:
    """Mean teacher-forced CE over (doc->gist) val pairs -> perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(pairs)
    for start in range(0, n, batch_size):
        batch = pairs[start:start + batch_size]
        doc_ids = torch.tensor([d for d, _ in batch], dtype=torch.long, device=device)
        gist_ids = torch.tensor([g for _, g in batch], dtype=torch.long, device=device)
        enc_states = model.encode(doc_ids)
        logits = model.decoder.forward(gist_ids, enc_states)
        vocab = model.gist_cfg.vocab
        logits = logits[:, :-1, :].reshape(-1, vocab)
        targets = gist_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=PAD_ID, reduction="sum")
        n_tok = (targets != PAD_ID).sum().item()
        total_loss += loss.item()
        total_tokens += n_tok
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ------------------------------------------------------------------- training
def train(args) -> int:
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"[device] {device}  [dtype] {dtype}", flush=True)

    # ---- tokenizer (the token-LM's tokenizer; shared vocab with the encoder).
    tok = train_or_load_tokenizer(iter([]), args.tokenizer_cache, vocab_size=args.vocab)
    print(f"[tok] vocab_size={tok.vocab_size} (cache={args.tokenizer_cache})", flush=True)

    # ---- frozen encoder: load the token-LM checkpoint strict.
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
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"[encoder] loaded {args.encoder_checkpoint} "
          f"({encoder.num_parameters():,} params, FROZEN)", flush=True)

    # ---- readout model (decoder + state projection; trainable).
    gist_cfg = GistConfig(
        vocab=tok.vocab_size,
        d_model_dec=args.d_model_dec,
        n_layers_dec=args.n_layers_dec,
        d_state=enc_cfg.d_state,
        gist_seq_len=args.gist_seq_len,
        tie_head=True,
        dropout=0.0,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
    )
    model = GistReadoutModel(encoder, gist_cfg).to(device=device, dtype=dtype)
    # .to() does not flip requires_grad; re-freeze to be safe.
    model._freeze_encoder()
    n_train = model.trainable_parameters()
    print(f"[decoder] {n_train:,} trainable params (encoder frozen)", flush=True)

    # ---- data: stream ERAG, split DOCS train/val (held-out docs, not tokens).
    print(f"[data] streaming ERAG from {ERAG_PATH} (target={args.target})", flush=True)
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

    # ---- targets: title (free) or flash gist (cached teacher).
    if args.target == "gist":
        cache_path = Path(args.output_dir) / "gist_teacher_cache.json"
        cache: dict = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[target] generating flash gists (cache={cache_path}, "
              f"{len(cache)} cached)...", flush=True)
        t0 = time.time()

        def _to_gist_pairs(raw):
            pairs = []
            for i, (title, content) in enumerate(raw):
                g = flash_summarize(content, cache, cache_path)
                if g:
                    pairs.append((content, g))
                if (i + 1) % 200 == 0:
                    print(f"  [gist] {i + 1}/{len(raw)} ({time.time() - t0:.0f}s)",
                          flush=True)
            return pairs
        train_pairs_raw = _to_gist_pairs(train_raw)
        val_pairs_raw = _to_gist_pairs(val_raw)
        print(f"[target] {len(train_pairs_raw)} train gists, "
              f"{len(val_pairs_raw)} val gists ({time.time() - t0:.0f}s)", flush=True)
    else:
        train_pairs_raw = [(c, t) for t, c in train_raw]
        val_pairs_raw = [(c, t) for t, c in val_raw]

    # ---- tokenize: doc content (trunc/pad to doc_seq_len), gist target (gist_seq_len).
    print("[data] tokenizing doc/gist pairs...", flush=True)
    t0 = time.time()

    def _tok_pairs(raw):
        docs = tok.encode_batch([c for c, _ in raw], max_length=args.doc_seq_len)
        gists = tok.encode_batch([g for _, g in raw], max_length=args.gist_seq_len)
        return list(zip(docs, gists))

    train_pairs = _tok_pairs(train_pairs_raw)
    val_pairs = _tok_pairs(val_pairs_raw)
    print(f"[data] {len(train_pairs)} train pairs, {len(val_pairs)} val pairs "
          f"({time.time() - t0:.1f}s)", flush=True)

    uniform_ce = math.log(tok.vocab_size)
    print(f"[gate] uniform baseline CE = log(vocab) = {uniform_ce:.3f}", flush=True)

    # ---- optimizer (decoder only; encoder frozen out of the optim).
    optim = torch.optim.AdamW(model.decoder.parameters(),
                             lr=args.lr, betas=(0.9, 0.95),
                             weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(train_pairs)
    batch_size = args.batch_size
    steps_per_epoch = max(1, n // batch_size)
    total_steps = args.steps
    print(f"[train] {n} pairs, batch {batch_size} -> {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps", flush=True)

    model.decoder.train()
    step = 0
    running = 0.0
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
            doc_ids = torch.tensor([train_pairs[i][0] for i in idx],
                                   dtype=torch.long, device=device)
            gist_ids = torch.tensor([train_pairs[i][1] for i in idx],
                                    dtype=torch.long, device=device)
            # Encode with the frozen encoder (no grad), decode teacher-forced.
            with torch.autocast(device_type=device.type, enabled=(dtype != torch.float32),
                                 dtype=dtype):
                enc_states = model.encode(doc_ids)
                logits = model.decoder.forward(gist_ids, enc_states)
                logits = logits[:, :-1, :].float().reshape(-1, gist_cfg.vocab)
                targets = gist_ids[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits, targets, ignore_index=PAD_ID)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            lr = _cosine_warmup_lr(step, args.warmup, total_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()

            running += loss.item() * targets.ne(PAD_ID).sum().item()
            running_n += targets.ne(PAD_ID).sum().item()
            step += 1
            if step % args.log_every == 0 or step == 1:
                mean_ce = running / max(running_n, 1)
                ppl = math.exp(min(mean_ce, 20.0))
                rate = step / (time.time() - t_start)
                print(f"[step {step:>5}/{total_steps}] train CE {mean_ce:.3f}  "
                      f"ppl {ppl:.2f}  lr {lr:.2e}  {rate:.2f} step/s", flush=True)
                running = 0.0
                running_n = 0
            if step % args.checkpoint_every == 0:
                save_gist_checkpoint(out_dir / f"gist_step{step}.pt", model, step,
                                     Path(args.encoder_checkpoint).name)

    # ---- final checkpoint + val perplexity + a few generations.
    final_path = out_dir / "gist_final.pt"
    save_gist_checkpoint(final_path, model, step, Path(args.encoder_checkpoint).name)
    print(f"[ckpt] saved {final_path}", flush=True)

    val_ppl = eval_perplexity(model, val_pairs, args.batch_size, device)
    print(f"[val] held-out-docs gist perplexity = {val_ppl:.2f}  "
          f"(CE {math.log(val_ppl):.3f})", flush=True)

    summary = {
        "target": args.target,
        "trainable_params": n_train,
        "vocab": tok.vocab_size,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "steps": total_steps,
        "val_ppl": val_ppl,
        "uniform_ce": uniform_ce,
        "d_model_dec": gist_cfg.d_model_dec,
        "n_layers_dec": gist_cfg.n_layers_dec,
        "d_state": gist_cfg.d_state,
        "doc_seq_len": args.doc_seq_len,
        "gist_seq_len": args.gist_seq_len,
        "encoder_ref": Path(args.encoder_checkpoint).name,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"[summary] wrote {out_dir / 'run_summary.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the frozen-encoder gist readout.")
    # data
    ap.add_argument("--target", choices=["title", "gist"], default="title")
    ap.add_argument("--max-train-docs", type=int, default=8_000)
    ap.add_argument("--val-docs", type=int, default=200)
    ap.add_argument("--tokenizer-cache", default=DEFAULT_TOK_CACHE)
    ap.add_argument("--encoder-checkpoint", default=DEFAULT_ENCODER_CKPT)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    # model
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--d-model-dec", type=int, default=96)
    ap.add_argument("--n-layers-dec", type=int, default=2)
    ap.add_argument("--doc-seq-len", type=int, default=128)
    ap.add_argument("--gist-seq-len", type=int, default=48)
    # training
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
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