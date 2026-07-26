"""The gist-readout gate (the §3.3 content probe, run on a trained readout).

Loads a trained ``GistReadoutModel`` (``scripts/train_gist_readout.py`` output) and
its frozen encoder + tokenizer, then runs the isolated gate the plan specifies --
no orchestrator, no LLM consumer wired in. The load-bearing piece is the §3.3 swap
control, INVERTED: §3.3 failed because ``perm ~= corpus-mean ~= main`` (swapping
states changed nothing -> the state carried no doc-specific content). Here we
decode gists from two docs' states and check the decoded gist FOLLOWS THE STATE.

Gate components (all on fully held-out docs -- unseen in decoder training and
beyond the encoder's training slice):

  1. Faithfulness -- 3-judge DeepSeek-flash consensus (2/3 bar): is the decoded
     gist a faithful summary of the doc? The §3.3 probe measured cosine-sim and
     found nothing; we measure factual faithfulness of GENERATED TEXT.
  2. Compression -- the gist is shorter than the doc (a summary, not a continuation).
  3. Fluency -- the gist decodes to non-empty, non-degenerate real text.
  4. Swap control (LOAD-BEARING) -- for pairs (A, B): decode gist(state_A) and
     gist(state_B); a pairwise judge must assign gist(state_A)->A and
     gist(state_B)->B. PASS = the gist follows the state (swapping states swaps
     the gist); FAIL = swap ~= main (the §3.3 result: state carries no
     doc-specific content).
  5. Negative control (OPTIONAL, off by default; --neg-ctrl-encoder) -- same gate
     on an identity-objective encoder's state: expected swap ~= main (FAIL),
     reproducing §3.3 with a TRAINED decoder. Requires an encoder whose state
     interface matches ``SSMLanguageModel.forward``; the text2x identity backbone
     (Ashes-of-STRM, a JGSBackbone) does NOT match and needs a separate adapter,
     so it is a documented follow-on, not wired here.

Self-contained: the 3-judge panel is written fresh here (same DeepSeek-flash +
2/3-consensus pattern as the Ashes ``erag_judge_harness.py``, but not lifted --
avoids any onyx-path import). Public ERAG only. Verdict printed + written to JSON.

Usage:
    python scripts/eval_gist_readout.py \
        --checkpoint data/gist_readout/gist_final.pt \
        --encoder-checkpoint data/token_lm/token_lm_final.pt \
        --tokenizer-cache data/token_lm/tokenizer.json \
        --max-eval-docs 60 --n-swap-pairs 25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.gist_readout import load_gist_readout  # noqa: E402

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-v4-flash:cloud"
LLM_TIMEOUT = 60

# Encoder trained on the first 50k ERAG docs (train) + 500 val (see
# train_token_lm.py defaults). The decoder trains on a SEPARATE slice (the first
# `val_docs` of its own stream + the next `max_train_docs`). To get docs fully
# held-out from BOTH, start the eval slice beyond a safe offset.
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


# ----------------------------------------------------------------- LLM judge
def _flash(prompt: str, temperature: float = 0.2, num_predict: int = 64) -> str | None:
    """One DeepSeek-flash call (local Ollama). Returns the response text or None.

    ``num_predict`` caps the generation budget; the faithfulness judge (one word)
    is fine at 64, but the swap judge reasons before answering and needs ~128
    (at 64 the model's reasoning eats the budget and it emits an empty string).
    """
    import urllib.request
    payload = json.dumps({"model": LLM_MODEL, "prompt": prompt,
                          "stream": False,
                          "options": {"temperature": temperature,
                                      "num_predict": num_predict}}).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8")).get("response", "").strip()
    except Exception as e:
        print(f"  [judge warn] {e}", flush=True)
        return None


_FAITH_PROMPT = (
    "You are a strict fact-checker. Document:\n```\n{doc}\n```\n"
    "Proposed one-sentence summary:\n```\n{gist}\n```\n"
    "Is the summary a FAITHFUL summary of the document (only claims supported by "
    "the document, no invented facts, not a generic placeholder)? "
    "Reply with exactly one word: yes, no, or partial."
)


def faithfulness(doc: str, gist: str) -> str:
    """3-judge consensus (2/3 bar). Returns 'yes' | 'no' | 'partial'."""
    votes: list[str] = []
    for t in (0.1, 0.3, 0.5):  # 3 independent judges, varied temperature
        r = _flash(_FAITH_PROMPT.format(doc=doc[:3000], gist=gist), temperature=t)
        if r:
            low = r.lower().strip().split()[0]
            for v in ("yes", "no", "partial"):
                if v in low:
                    votes.append(v)
                    break
    if not votes:
        return "no"
    # 2/3 majority; else the strictest (the most negative vote wins ties).
    for v in ("yes", "partial", "no"):
        if votes.count(v) >= 2:
            return v
    return votes[0]


_SWAP_PROMPT = (
    "Two documents:\n"
    "DOC A:\n```\n{a}\n```\n"
    "DOC B:\n```\n{b}\n```\n"
    "Two summaries were produced by decoding two different memory states:\n"
    "SUMMARY 1:\n```\n{s1}\n```\n"
    "SUMMARY 2:\n```\n{s2}\n```\n"
    "Which document does EACH summary describe? Reply on two lines exactly:\n"
    "1: A or B\n2: A or B"
)


def swap_assignment(a: str, b: str, s1: str, s2: str) -> tuple[str, str] | None:
    """Judge which doc each summary matches. Returns ('A'|'B', 'A'|'B') for (s1, s2)."""
    r = _flash(_SWAP_PROMPT.format(a=a[:1500], b=b[:1500], s1=s1, s2=s2),
               temperature=0.1, num_predict=128)
    if not r:
        return None
    out: list[str] = []
    for line in r.lower().splitlines():
        for tok in ("a", "b"):
            if tok in line and ("a" in line or "b" in line):
                out.append(tok)
                break
        if len(out) == 2:
            break
    if len(out) < 2:
        return None
    return out[0].upper(), out[1].upper()


# --------------------------------------------------------------------- eval
@torch.no_grad()
def generate_gist_for(model, tok, content: str, args, device) -> tuple[str, int]:
    """Encode a doc's content -> state -> decode gist. Returns (gist_text, n_tokens)."""
    doc_ids = tok.encode_batch([content], max_length=args.doc_seq_len)
    doc_t = torch.tensor(doc_ids, dtype=torch.long, device=device)
    states = model.encode(doc_t)
    ids = model.decoder.generate(states, max_new_tokens=args.gist_new_tokens,
                                 temperature=args.temperature,
                                 top_k=args.top_k)[0].tolist()
    return tok.decode(ids), len(ids)


@torch.no_grad()
def gist_nll(model, gist_ids: torch.Tensor, enc_states: list[torch.Tensor]) -> float:
    """Mean per-token negative log-likelihood of ``gist_ids`` under ``enc_states``
    (teacher-forced). The clean, judge-free measure of how well a state explains a
    gist: lower = the state favors this gist. PAD tokens are ignored.

    This is the load-bearing diagnostic: it separates "the state carries
    doc-specific content" from "the decoder renders coherent free text." Free
    generation can be degenerate (BPE fragmentation, weak decoder, exposure bias)
    even when the state carries content; a LIKELIHOOD test cuts through that.
    """
    import torch.nn.functional as F
    logits = model.decoder.forward(gist_ids, enc_states)
    vocab = model.gist_cfg.vocab
    logits = logits[:, :-1, :].reshape(-1, vocab).float()
    targets = gist_ids[:, 1:].reshape(-1)
    mask = targets != model.gist_cfg.pad_token_id
    if mask.sum() == 0:
        return float("inf")
    ce = F.cross_entropy(logits, targets, ignore_index=model.gist_cfg.pad_token_id,
                         reduction="sum")
    return (ce / mask.sum()).item()


@torch.no_grad()
def encode_doc(model, tok, content: str, args, device) -> list[torch.Tensor]:
    doc_ids = tok.encode_batch([content], max_length=args.doc_seq_len)
    return model.encode(torch.tensor(doc_ids, dtype=torch.long, device=device))


def _gist_ids(tok, text: str, args) -> torch.Tensor:
    """Tokenize a gold gist to a fixed-length [1, gist_seq_len] tensor (BOS..EOS, PAD)."""
    ids = tok.encode_batch([text], max_length=args.gist_seq_len)
    return torch.tensor(ids, dtype=torch.long)


def run_gate(args) -> int:
    device_arg = args.device
    print(f"[load] checkpoint={args.checkpoint}", flush=True)
    model, tok = load_gist_readout(
        args.checkpoint, args.encoder_checkpoint, args.tokenizer_cache,
        device=device_arg, dtype=args.dtype,
    )
    device = next(model.parameters()).device
    print(f"[load] encoder FROZEN={all(not p.requires_grad for p in model.encoder.parameters())}, "
          f"decoder trainable params={model.trainable_parameters():,}", flush=True)
    model.eval()

    # ---- held-out docs (beyond encoder train + decoder train slices).
    docs = list(_iter_erag_docs(ERAG_PATH, args.eval_offset, args.max_eval_docs))
    print(f"[data] {len(docs)} held-out docs (offset={args.eval_offset})", flush=True)
    if len(docs) < 4:
        print("[gate] not enough held-out docs; abort.", flush=True)
        return 1

    # ---- 1-3: faithfulness / compression / fluency on each doc.
    print("\n[gate 1-3] per-doc faithfulness / compression / fluency:", flush=True)
    faithful = 0
    compressed = 0
    fluent = 0
    samples: list[dict] = []
    t0 = time.time()
    for i, (title, content) in enumerate(docs):
        gist, n_gist = generate_gist_for(model, tok, content, args, device)
        n_doc = len(tok.encode(content))
        verdict = faithfulness(content, gist) if args.use_judge else "skip"
        is_faith = verdict == "yes"
        is_comp = n_gist < n_doc and n_gist <= args.gist_new_tokens
        is_flu = bool(gist.strip()) and len(set(gist.split())) > 2
        faithful += int(is_faith)
        compressed += int(is_comp)
        fluent += int(is_flu)
        samples.append({"title": title, "gist": gist, "n_doc_tok": n_doc,
                        "n_gist_tok": n_gist, "faithful": verdict})
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(docs)}] faithful={verdict} "
                  f"n_doc={n_doc} n_gist={n_gist} ({time.time() - t0:.0f}s)", flush=True)
    n = len(docs)
    main_fidelity = faithful / n if args.use_judge else float("nan")
    compression = compressed / n
    fluency = fluent / n
    print(f"[gate 1] faithfulness (yes/total): {faithful}/{n} = {main_fidelity:.3f}", flush=True)
    print(f"[gate 2] compression: {compressed}/{n} = {compression:.3f}", flush=True)
    print(f"[gate 3] fluency: {fluent}/{n} = {fluency:.3f}", flush=True)

    # ---- 4: LIKELIHOOD swap control (PRIMARY, load-bearing, judge-free).
    # The §3.3 failure was swap ~= main: swapping states changed nothing because
    # the state carried no doc-specific content. The deterministic test: does a
    # state prefer its OWN doc's gist over a different doc's gist? For each pair
    # (A, B), compute NLL(gist_A | state_A) vs NLL(gist_A | state_B). The state
    # carries doc-specific content iff state_A makes gist_A more likely than
    # state_B does (and symmetrically for B). This cuts through free-generation
    # degeneracy (BPE fragmentation / weak decoder / exposure bias) that the
    # faithfulness judge conflates with "no content in state".
    print("\n[gate 4] likelihood swap (PRIMARY, judge-free): state prefers own gist?", flush=True)
    pairs = []
    rng = random.Random(args.seed)
    idxs = list(range(len(docs)))
    rng.shuffle(idxs)
    for k in range(0, min(args.n_swap_pairs * 2, len(idxs) - 1), 2):
        pairs.append((idxs[k], idxs[k + 1]))
    pairs = pairs[:args.n_swap_pairs]
    like_correct = 0
    margins: list[float] = []
    state_gaps: list[float] = []     # ||state_A - state_B|| -- do states differ per doc?
    zero_gaps: list[float] = []      # NLL(gist|zero) - NLL(gist|state) -- does decoder use state?
    t0 = time.time()
    # A zero state of the decoder's seed shape (one per decoder layer).
    zero_states = [
        torch.zeros(1, model.gist_cfg.d_state, model.encoder.config.d_model,
                    device=device, dtype=next(model.parameters()).dtype)
        for _ in range(model.gist_cfg.n_layers_dec)
    ]
    for k, (ia, ib) in enumerate(pairs):
        ta, ca = docs[ia]
        tb, cb = docs[ib]
        st_a = encode_doc(model, tok, ca, args, device)
        st_b = encode_doc(model, tok, cb, args, device)
        ga = _gist_ids(tok, ta, args).to(device)
        gb = _gist_ids(tok, tb, args).to(device)
        nll_a_a = gist_nll(model, ga, st_a)  # own gist, own state
        nll_a_b = gist_nll(model, ga, st_b)  # own gist, OTHER state
        nll_b_b = gist_nll(model, gb, st_b)
        nll_b_a = gist_nll(model, gb, st_a)
        nll_a_z = gist_nll(model, ga, zero_states)  # own gist, ZERO state
        ok_a = nll_a_a < nll_a_b
        ok_b = nll_b_b < nll_b_a
        like_correct += int(ok_a and ok_b)
        margins.append((nll_a_b - nll_a_a) + (nll_b_a - nll_b_b))  # +ve = state discriminates
        # Encoder-state distance between the two docs (mean over layers, L2).
        gap = sum((sa - sb).float().pow(2).sum().sqrt().item()
                  for sa, sb in zip(st_a, st_b)) / len(st_a)
        state_gaps.append(gap)
        zero_gaps.append(nll_a_z - nll_a_a)  # +ve = the state helps vs zero
        if (k + 1) % 5 == 0 or k == 0:
            print(f"  [pair {k + 1}/{len(pairs)}] A|A={nll_a_a:.2f} A|B={nll_a_b:.2f} "
                  f"B|B={nll_b_b:.2f} B|A={nll_b_a:.2f} A|0={nll_a_z:.2f} "
                  f"||sA-sB||={gap:.2f} ok={ok_a and ok_b} ({time.time() - t0:.0f}s)",
                  flush=True)
    n_pairs = len(pairs)
    like_swap_fidelity = like_correct / n_pairs if n_pairs else float("nan")
    mean_margin = (sum(margins) / len(margins)) if margins else float("nan")
    mean_state_gap = (sum(state_gaps) / len(state_gaps)) if state_gaps else float("nan")
    mean_zero_gap = (sum(zero_gaps) / len(zero_gaps)) if zero_gaps else float("nan")
    print(f"[gate 4] likelihood swap: {like_correct}/{n_pairs} = {like_swap_fidelity:.3f}  "
          f"(discrim margin {mean_margin:.3f} nats)", flush=True)
    print(f"[gate 4] state-distinction ||sA-sB||={mean_state_gap:.3f}  "
          f"(>0 = encoder states differ per doc) | "
          f"state-vs-zero NLL gap={mean_zero_gap:.3f}  (>0 = decoder uses the state)",
          flush=True)

    # ---- 5: LLM-judged swap on FREE generation (SECONDARY, confirmatory).
    # The free-generation judge measures the END GOAL (coherent faithful text that
    # follows the state), not the load-bearing property. Off unless --use-judge.
    llm_swap_fidelity = float("nan")
    if args.use_judge:
        print("\n[gate 5] LLM swap judge on free generation (SECONDARY):", flush=True)
        sw_correct = 0
        sw_total = 0
        t0 = time.time()
        for k, (ia, ib) in enumerate(pairs):
            ta, ca = docs[ia]
            tb, cb = docs[ib]
            ga, _ = generate_gist_for(model, tok, ca, args, device)
            gb, _ = generate_gist_for(model, tok, cb, args, device)
            res = swap_assignment(ca, cb, ga, gb)
            if res is not None:
                sw_total += 1
                sw_correct += int(res == ("A", "B"))
            if (k + 1) % 5 == 0 or k == 0:
                print(f"  [pair {k + 1}/{len(pairs)}] assign={res} ({time.time() - t0:.0f}s)",
                      flush=True)
        llm_swap_fidelity = (sw_correct / sw_total) if sw_total else float("nan")
        print(f"[gate 5] LLM swap-follows-state: {sw_correct}/{sw_total} = {llm_swap_fidelity:.3f}",
              flush=True)

    # ---- verdict. PRIMARY = likelihood swap (does the state carry doc-specific
    # content the decoder can read out). SECONDARY = LLM faithfulness (does it
    # render as coherent faithful text -- the ultimate goal). A PASS on the
    # primary with a FAIL on the secondary = "content is in the state; the
    # decoder/objective just isn't rendering it as readable text yet" -> the
    # follow-on is decoder/objective quality, NOT "the state has no content".
    main_bar = args.main_bar
    swap_bar = args.swap_bar
    like_swap_pass = (n_pairs > 0) and (like_swap_fidelity >= swap_bar)
    main_pass = (args.use_judge) and (main_fidelity >= main_bar)
    comp_pass = compression >= 0.8
    flu_pass = fluency >= 0.8
    # The overall verdict is driven by the PRIMARY likelihood swap + compression
    # + fluency. LLM faithfulness is reported but does not veto (a weak decoder
    # can fail faithfulness while the state still carries content).
    verdict = "PASS" if (like_swap_pass and comp_pass and flu_pass) else "FAIL"
    print("\n==================== VERDICT ====================", flush=True)
    print(f"  likelihood swap : {like_swap_fidelity:.3f}  (bar {swap_bar})  -> {'PASS' if like_swap_pass else 'FAIL'}  [PRIMARY, load-bearing]", flush=True)
    print(f"  discrim margin  : {mean_margin:.3f} nats  (>0 = state carries doc-specific content)", flush=True)
    print(f"  state distinction: ||sA-sB||={mean_state_gap:.3f}  state-vs-zero NLL gap={mean_zero_gap:.3f}", flush=True)
    print(f"  LLM faithfulness: {main_fidelity:.3f}  (bar {main_bar})  -> {'PASS' if main_pass else ('skip' if not args.use_judge else 'FAIL')}  [secondary, the end goal]", flush=True)
    print(f"  LLM swap (free) : {llm_swap_fidelity:.3f}                       [secondary]", flush=True)
    print(f"  compression     : {compression:.3f}                       -> {'PASS' if comp_pass else 'FAIL'}", flush=True)
    print(f"  fluency         : {fluency:.3f}                       -> {'PASS' if flu_pass else 'FAIL'}", flush=True)
    print(f"  VERDICT         : {verdict}", flush=True)
    print("=================================================", flush=True)

    results = {
        "n_eval_docs": n,
        "n_swap_pairs": n_pairs,
        "likelihood_swap_fidelity": like_swap_fidelity,
        "mean_discrimination_margin_nats": mean_margin,
        "mean_state_distinction_l2": mean_state_gap,
        "mean_state_vs_zero_nll_gap": mean_zero_gap,
        "main_fidelity": main_fidelity,
        "llm_swap_fidelity": llm_swap_fidelity,
        "compression": compression,
        "fluency": fluency,
        "main_bar": main_bar,
        "swap_bar": swap_bar,
        "verdict": verdict,
        "use_judge": args.use_judge,
        "samples": samples[:20],
    }
    out = Path(args.output_dir) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[results] wrote {out}", flush=True)

    # Print a few samples (ASCII-safe) for a human read.
    print("\n[samples] (first 5):", flush=True)
    for s in samples[:5]:
        safe_title = s["title"].encode("ascii", "replace").decode()
        safe_gist = s["gist"].encode("ascii", "replace").decode()
        print(f"  - title: {safe_title}", flush=True)
        print(f"    gist : {safe_gist}", flush=True)
    return 0 if verdict == "PASS" else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the gist-readout gate.")
    ap.add_argument("--checkpoint", default="data/gist_readout/gist_final.pt")
    ap.add_argument("--encoder-checkpoint", default="data/token_lm/token_lm_final.pt")
    ap.add_argument("--tokenizer-cache", default="data/token_lm/tokenizer.json")
    ap.add_argument("--output-dir", default="data/gist_readout")
    ap.add_argument("--max-eval-docs", type=int, default=60)
    ap.add_argument("--n-swap-pairs", type=int, default=25)
    ap.add_argument("--eval-offset", type=int, default=DEFAULT_EVAL_OFFSET)
    ap.add_argument("--doc-seq-len", type=int, default=128)
    ap.add_argument("--gist-seq-len", type=int, default=48,
                    help="gold-gist tokenization length (must match training --gist-seq-len)")
    ap.add_argument("--gist-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--use-judge", action="store_true",
                    help="call the DeepSeek-flash judge panel (needs Ollama :11434)")
    ap.add_argument("--main-bar", type=float, default=0.6)
    ap.add_argument("--swap-bar", type=float, default=0.8)
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