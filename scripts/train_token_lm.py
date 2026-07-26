"""Train the token-level LM-SSM (``src/subconscious/token_lm.py``) on ERAG.

The content-objective retrain the §3.3 content probe prescribed (post-mortem §6:
state shape set by the OBJECTIVE). Trains a small token-level LM whose sequence
mixer is the owned ``SelectiveSSM`` on next-token cross-entropy over public ERAG
text, then samples continuations from held-out doc prefixes (the literal
"language in / language out that makes sense" objective).

Self-contained: the device/dtype/LR/checkpoint helpers are copied from
``pretrain.py`` (the bge pretrainer) rather than imported, so the token-LM path
does not couple to the bge/JEPA stack. Trained on public ERAG text only -- no
onyx, no private transcripts. Checkpoints are written under ``--output-dir``;
the user uploads to HF private separately if desired.

Usage (local RTX 5080):
    python scripts/train_token_lm.py --steps 3000 --generate

Run ``--help`` for all knobs. Defaults are a ~14M-param first-working model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.token_lm import LMConfig, SSMLanguageModel  # noqa: E402
from src.subconscious.tokenizer_ import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    train_or_load_tokenizer,
)

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_TOK_CACHE = "data/token_lm/tokenizer.json"
DEFAULT_OUTPUT_DIR = "data/token_lm"


# --------------------------------------------------------------------- helpers
# Copied from src/subconscious/training/pretrain.py (do NOT import -- keep the
# token-LM path decoupled from the bge/JEPA pretrainer).
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    return torch.float32


def _cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ----------------------------------------------------------------------- data
def _iter_erag_content(path: str, max_docs: int | None = None):
    """Stream ERAG ``content`` strings from the parquet (row-group by row-group)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        for content in tbl.column("content").to_pylist():
            if content and str(content).strip():
                yield str(content)
                seen += 1
                if max_docs is not None and seen >= max_docs:
                    return


@dataclass
class TokenChunks:
    """A flat list of fixed-length ``seq_len`` token chunks + the doc boundaries
    they came from (so generation can pick held-out DOC prefixes, not just
    random chunk prefixes)."""

    chunks: list[list[int]]   # each length == seq_len, padded with PAD_ID
    doc_ids: list[int]        # doc index each chunk belongs to


def _chunk_doc(tokens: list[int], seq_len: int) -> list[list[int]]:
    """Split one doc's token list into non-overlapping ``seq_len`` chunks.

    Short docs (< seq_len) become a single padded chunk. The trailing partial
    chunk of a long doc is padded. Padding is ``PAD_ID`` and is masked in the
    loss (targets == PAD_ID are ignored by ``cross_entropy(ignore_index)``)."""
    if not tokens:
        return []
    chunks: list[list[int]] = []
    for i in range(0, len(tokens), seq_len):
        piece = tokens[i:i + seq_len]
        if len(piece) < seq_len:
            piece = piece + [PAD_ID] * (seq_len - len(piece))
        chunks.append(piece)
    return chunks


def build_dataset(
    docs: list[str],
    tok,
    seq_len: int,
) -> TokenChunks:
    """Tokenize each doc and split into ``seq_len`` chunks (padded)."""
    chunks: list[list[int]] = []
    doc_ids: list[int] = []
    for di, content in enumerate(docs):
        ids = tok.encode(content)  # BOS ... EOS
        for ch in _chunk_doc(ids, seq_len):
            chunks.append(ch)
            doc_ids.append(di)
    return TokenChunks(chunks, doc_ids)


def _batch_chunks(chunks: list[list[int]], indices: list[int], device) -> Tensor:
    """Stack a list of chunk-index slices into a ``[batch, seq_len]`` long tensor."""
    return torch.tensor([chunks[i] for i in indices], dtype=torch.long, device=device)


@torch.no_grad()
def eval_perplexity(model, chunks, batch_size, device, vocab) -> float:
    """Mean next-token CE over all val chunks -> perplexity. Obeys PAD masking."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(chunks)
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        ids = _batch_chunks(chunks, idx, device)
        logits, _ = model.forward(ids)  # [b, seq, vocab]
        # next-token: predict ids[:, t+1] from logits[:, t]
        logits = logits[:, :-1, :].reshape(-1, vocab)
        targets = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=PAD_ID, reduction="sum")
        n_tokens = (targets != PAD_ID).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    mean_ce = total_loss / total_tokens
    return math.exp(mean_ce)


# ------------------------------------------------------------------- training
def train(args) -> int:
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"[device] {device}  [dtype] {dtype}", flush=True)

    # ---- data: stream ERAG, split DOCS train/val (held-out docs, not tokens).
    print(f"[data] streaming ERAG from {ERAG_PATH}", flush=True)
    t0 = time.time()
    val_docs: list[str] = []
    train_docs: list[str] = []
    for content in _iter_erag_content(ERAG_PATH, max_docs=args.max_train_docs + args.val_docs):
        if len(val_docs) < args.val_docs:
            val_docs.append(content)
        else:
            train_docs.append(content)
            if len(train_docs) >= args.max_train_docs:
                break
    print(f"[data] {len(train_docs)} train docs, {len(val_docs)} val docs "
          f"({time.time() - t0:.1f}s)", flush=True)

    # ---- tokenizer: train on TRAIN docs only (val docs are fully held-out).
    def _train_corpus():
        for c in train_docs:
            yield c
    tok = train_or_load_tokenizer(
        _train_corpus(), args.tokenizer_cache, vocab_size=args.vocab,
    )
    print(f"[tok] vocab_size={tok.vocab_size} (cache={args.tokenizer_cache})",
          flush=True)

    # ---- model
    cfg = LMConfig(
        vocab=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        seq_len=args.seq_len,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
    )
    model = SSMLanguageModel(cfg).to(device=device, dtype=dtype)
    n_params = model.num_parameters()
    print(f"[model] {n_params:,} params ({n_params / 1e6:.2f}M)", flush=True)

    # ---- datasets
    print("[data] tokenizing train/val...", flush=True)
    t0 = time.time()
    train_ds = build_dataset(train_docs, tok, args.seq_len)
    val_ds = build_dataset(val_docs, tok, args.seq_len)
    print(f"[data] {len(train_ds.chunks)} train chunks, {len(val_ds.chunks)} val chunks "
          f"({time.time() - t0:.1f}s)", flush=True)

    uniform_ce = math.log(tok.vocab_size)
    print(f"[gate] uniform baseline CE = log(vocab) = {uniform_ce:.3f}  "
          f"(val ppl bar = vocab/4 = {tok.vocab_size / 4:.1f})", flush=True)

    # ---- optimizer (AdamW + cosine warmup LR; the bge pretrainer's pattern)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = len(train_ds.chunks)
    batch_size = args.batch_size
    steps_per_epoch = max(1, n_chunks // batch_size)
    total_steps = args.steps
    print(f"[train] {n_chunks} chunks, batch {batch_size} -> {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps", flush=True)

    model.train()
    step = 0
    epoch = 0
    running = 0.0
    running_n = 0
    t_start = time.time()
    while step < total_steps:
        # one epoch: shuffle chunk indices
        perm = torch.randperm(n_chunks).tolist()
        for bi in range(0, n_chunks, batch_size):
            if step >= total_steps:
                break
            idx = perm[bi:bi + batch_size]
            if not idx:
                continue  # range() never yields empty; defensive
            ids = _batch_chunks(train_ds.chunks, idx, device)
            # forward in the model's dtype; CE in fp32 for stability.
            with torch.autocast(device_type=device.type, enabled=(dtype != torch.float32),
                                 dtype=dtype):
                logits, _ = model.forward(ids)
                logits = logits[:, :-1, :].float().reshape(-1, cfg.vocab)
                targets = ids[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits, targets, ignore_index=PAD_ID)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                _save_checkpoint(model, optim, step, out_dir, cfg, final=False)
        epoch += 1
        if step < total_steps:
            print(f"[epoch {epoch}] looped ({step}/{total_steps} steps)", flush=True)

    # ---- final checkpoint + val perplexity
    _save_checkpoint(model, optim, step, out_dir, cfg, final=True)
    val_ppl = eval_perplexity(model, val_ds.chunks, args.batch_size, device, cfg.vocab)
    val_ce = math.log(val_ppl)
    print(f"[val] held-out-docs perplexity = {val_ppl:.2f}  (CE {val_ce:.3f})", flush=True)
    gate_ppl = tok.vocab_size / 4
    passed_gate = val_ppl < gate_ppl
    print(f"[gate] val ppl {val_ppl:.2f} < {gate_ppl:.1f} (vocab/4)? "
          f"{'PASS' if passed_gate else 'FAIL'}", flush=True)

    # ---- run summary (saved BEFORE generation so a generation/print crash
    # can't lose the metrics -- the previous run crashed on a non-ASCII char in
    # a sample print and the summary was never written).
    summary = {
        "params": n_params,
        "vocab": tok.vocab_size,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "train_chunks": len(train_ds.chunks),
        "val_chunks": len(val_ds.chunks),
        "steps": total_steps,
        "val_ppl": val_ppl,
        "val_ce": val_ce,
        "uniform_ce": uniform_ce,
        "gate_ppl": tok.vocab_size / 4,
        "gate_pass": passed_gate,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "d_state": cfg.d_state,
        "seq_len": cfg.seq_len,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"\n[summary] wrote {out_dir / 'run_summary.json'}", flush=True)

    # ---- generation: continuations from held-out doc prefixes (the objective)
    if args.generate:
        _run_generation(model, tok, val_docs, args, device, dtype, out_dir)
    # The perplexity bar is informational (the hard gate is the qualitative
    # coherence of the generated samples, which this script can't auto-grade).
    # Always exit 0 so a high-but-converging ppl doesn't abort the run before a
    # human reads the samples.
    return 0


def _run_generation(model, tok, val_docs, args, device, dtype, out_dir) -> None:
    """Sample continuations from held-out doc prefixes; write to a utf-8 file AND
    print safely. Writing to a file matters because ERAG text contains non-ASCII
    chars (e.g. ``\\u2011`` non-breaking hyphen) that the Windows cp1252 console
    cannot encode -- a raw ``print`` of those crashes the process. The file is
    always written; the print is best-effort (errors='replace')."""
    print("\n[generate] held-out doc-prefix continuations:", flush=True)
    model.eval()
    n_samples = min(args.n_samples, len(val_docs))
    samples_path = out_dir / "generation_samples.txt"
    lines: list[str] = []
    for i in range(n_samples):
        content = val_docs[i]
        full_ids = tok.encode(content)
        prefix_len = min(args.seq_len // 2, max(8, len(full_ids) // 3))
        prefix = full_ids[:prefix_len]
        prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, enabled=(dtype != torch.float32),
                            dtype=dtype):
            out = model.generate(
                prefix_ids,
                max_new_tokens=args.gen_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
            )
        out_ids = out[0].tolist()
        prompt_text = tok.decode(prefix)
        cont_text = tok.decode(out_ids[len(prefix):])
        block = f"\n--- sample {i + 1} ---\nPROMPT: {prompt_text}\nMODEL : {cont_text}\n"
        lines.append(block)
    samples_path.write_text("".join(lines), encoding="utf-8")
    print(f"[generate] wrote {samples_path}", flush=True)
    # Best-effort console echo: replace any char the console encoding can't
    # handle instead of crashing.
    safe = "".join(lines).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8", errors="replace"
    )
    print(safe, flush=True)


def generate_only(args) -> int:
    """Load the trained tokenizer + checkpoint and run generation only (no
    retraining). Lets you re-sample with different temperature/top_k from a
    saved checkpoint without paying the training cost."""
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    out_dir = Path(args.output_dir)
    tok = train_or_load_tokenizer(iter([]), args.tokenizer_cache, vocab_size=args.vocab)
    print(f"[tok] loaded vocab_size={tok.vocab_size} from {args.tokenizer_cache}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = LMConfig(**ckpt["config"])
    print(f"[model] loaded config from {args.checkpoint}: "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers} vocab={cfg.vocab}", flush=True)
    model = SSMLanguageModel(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    print(f"[model] checkpoint loaded ({model.num_parameters():,} params)", flush=True)
    # Held-out docs: re-stream the same val split (first args.val_docs docs of
    # ERAG, matching the training split).
    val_docs: list[str] = []
    for content in _iter_erag_content(ERAG_PATH, max_docs=args.val_docs):
        val_docs.append(content)
    _run_generation(model, tok, val_docs, args, device, dtype, out_dir)
    return 0


def _save_checkpoint(model, optim, step, out_dir: Path, cfg: LMConfig, final: bool) -> None:
    tag = "final" if final else f"step{step}"
    path = out_dir / f"token_lm_{tag}.pt"
    torch.save({
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "step": step,
        "optimizer": optim.state_dict(),
    }, path)
    print(f"[ckpt] saved {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the token-level LM-SSM on ERAG.")
    # data
    ap.add_argument("--max-train-docs", type=int, default=50_000)
    ap.add_argument("--val-docs", type=int, default=500)
    ap.add_argument("--tokenizer-cache", default=DEFAULT_TOK_CACHE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    # model
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    # training
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    # generation
    ap.add_argument("--generate", action="store_true",
                    help="sample continuations from held-out doc prefixes after training")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--gen-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 = off (greedy/full softmax)")
    # generate-only mode (load a saved checkpoint; no training)
    ap.add_argument("--generate-only", action="store_true",
                    help="load --checkpoint + tokenizer and run generation only (no training)")
    ap.add_argument("--checkpoint", default="data/token_lm/token_lm_final.pt",
                    help="checkpoint path for --generate-only")
    args = ap.parse_args()
    # Force utf-8 stdout so non-ASCII ERAG text (e.g. ‑ non-breaking hyphen)
    # does not crash the Windows cp1252 console on print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    if args.generate_only:
        return generate_only(args)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())