#!/usr/bin/env python3
"""STRM 1f-7 Stage 3: flat_last backbone FINE-TUNE diagnostic (pre-state replay).

DeepSeek consult #3 verdict (pro): the readout-only path is exhausted because
the routing-trained ``backbone_v2_full.pt``'s ``flat_last [6144]`` is kind-
isolated -- no readout (shared or decomposed) can hold conv/text/code in one
cross-kind logit space. The fix is a backbone retrain that teaches ``flat_last``
a common cross-kind query-doc relevance subspace. This script is the CHEAP
fine-tune diagnostic: lightly fine-tune the existing backbone (5-10 epochs, all
params) with a margin loss on ``flat_last`` over the onyx doc ring, via a
THROWAWAY slim query-conditioned readout. ``backbone_v2_full.pt`` WAS relevance-
trained on mean-pool z_k [384] / ERAG -- NOT flat_last [6144] / onyx -- so this
targets a genuinely-untrained path (not redundant).

Replay strategy (pre-state replay, truncated-BPTT-depth-1). The trace stores
per-kept-slot ``slots_pre_state [K,4,16,384]`` = the cumulative WM state BEFORE
each kept slot's step (overflow + dropped steps included), and
``slots_step_input [K,384]`` = the EXACT step-input embedding captured on the
slot (post-pin, pre-SSM). For each record we seed
``states = slots_pre_state[k]`` (DETACHED, no grad) and re-step ONLY
``slots_step_input[k]`` WITH grad through ``backbone.layers[i].step`` ->
reproduces ``slots_h_raw[k]`` within fp16 epsilon AND backprops into the shared
backbone params (W_A/W_B). The step input is the captured ``u`` (NOT
``embed(slot.text)``): retrieved code docs are injected by MEANING -- the orchestrator steps ``embed(embed_text or summary)`` while ``slot.text = summary``
-- so re-embedding the summary string diverges from the stepped vector for ~20%
of retrieved slots. Capturing the exact ``u`` closes the replay-fidelity gap.
Each slot is grad-independent from a detached seed = no cross-slot BPTT, no
memory blowup, no overflow/drop-gap (the pre-state encodes all prior history).
All K slots of a record are stepped in ONE batched call (``layer.step`` is
batched over the slot dim).

The throwaway readout is a single ``nn.Linear(6144, 384)`` + cosine-similarity
with the query (a "simple shared linear readout", query-conditioned). Low
capacity FORCES the backbone to produce a query-relevant flat_last (a powerful
readout could compensate for a weak flat_last -- DeepSeek's reason for keeping
it simple + discarding it). The readout is NOT saved; only ``backbone.state_dict``
is written (same key namespace as ``backbone_v2_full.pt`` -> ``load_backbone``
reads it unchanged).

Reuses the #5 machinery from ``probe_head_to_head_onyx.py``: ``_drop_self_slot``
(self-slot removal + label re-derivation, slices ``slots_pre_state`` too),
``_gold_doc_kind``/``_build_doc_kind_map_cached`` (gold-kind stamp), the sqrt-
inverse-freq no-replacement ``WeightedRandomSampler`` (AdamW-clean class
balance), and ``margin_ranking_loss`` (m=2.5, hard-negative -- the tight surrogate
for the z_logit gate). The fine-tune's gold definition is IDENTICAL to #5's.

After fine-tuning: regenerate serve traces with ``--backbone <this output>``,
retrain #5 6-seed on them, run the 6-seed gate with ``--backbone <this output>``.
PASS = ret_code >= 2.0 in >= 4/6 AND ret_text >= 2.0 in >= 4/6.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import probe_head_to_head_onyx as phh  # noqa: E402  # _drop_self_slot, margin_ranking_loss, doc-kind helpers
from src.subconscious.training.routing_training import load_backbone  # noqa: E402

DEFAULT_BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DEFAULT_PRESTATE = ("data/training/strm_relevance/"
                    "traces_onyx_doc_ring_summarized_prestate.pt")
DEFAULT_OUT = "data/training/strm_backbone_relevance/backbone_v2_full_finetuned.pt"
DEFAULT_DOC_STORE = ("data/training/strm_relevance/"
                     "doc_corpus_store_summarized")
DIM_IN = 16 * 384  # 6144 = flat_last (last SSM layer [16,384] flattened)
TEMP = 0.05        # fixed cosine-similarity temperature (logit = cos(z,q)/TEMP)


def replay_flatlast(backbone, slots_step_input, slots_pre_state, device):
    """Single-step replay of all K slots in parallel, WITH grad into the backbone.

    ``slots_step_input`` [K,384] (fp32 -- the EXACT step-input embedding per
    slot, captured post-pin/pre-SSM in ``WorkingMemory.step``; NOT
    ``embed(slot.text)``, since code docs are injected by MEANING and re-embedding
    the summary string diverges from the stepped vector), ``slots_pre_state``
    [K,4,16,384] (fp32 -- the detached pre-step seed per slot). Seeds
    ``states[i] = pre_state[:, i]`` (detached -> truncated-BPTT-depth-1, no grad
    through history), steps the backbone's 4 SSM layers once (``layer.step`` is
    batched over K), returns the last layer's new state flattened ->
    ``flat_last [K,6144]``. Mirrors ``step_sequence`` in
    ``train_backbone_relevance.py`` BUT seeded from ``slots_pre_state`` instead
    of zero (the identity-instance path == direct ``layer.step`` since
    input_proj/state_lora/output_proj are ``Identity``)."""
    K = slots_step_input.shape[0]
    h = slots_step_input.to(device)                                # [K,384]
    states = [slots_pre_state[:, i].to(device).detach() for i in range(4)]  # [K,16,384] each
    new_states = []
    for i, layer in enumerate(backbone.layers):
        h, s = layer.step(h, states[i])
        new_states.append(s)
    last = new_states[-1]                                          # [K,16,384]
    return last.reshape(K, -1)                                     # [K,6144]


def fidelity_check(backbone, records, device, n_check=12, atol=0.15):
    """Epoch-0 replay fidelity: re-step from ``slots_pre_state`` (no grad) and
    compare the resulting ``flat_last`` to the stored ``slots_h_raw[:, -1]``.
    High fidelity (within fp16 epsilon) confirms the pre-state seed reproduces
    the trace -> the fine-tune trains on the REAL serve state distribution. Low
    fidelity means the replay is bogus and the fine-tune is worthless -> STOP."""
    backbone.eval()
    max_diff = 0.0
    n_checked = 0
    with torch.no_grad():
        for rec in records[:n_check]:
            if ("slots_pre_state" not in rec or "slots_h_raw" not in rec
                    or "slots_step_input" not in rec):
                continue
            pre = rec["slots_pre_state"].to(torch.float32)
            u = rec["slots_step_input"].to(torch.float32)
            flat = replay_flatlast(backbone, u, pre, device)      # [K,6144]
            h_raw_last = rec["slots_h_raw"][:, -1].to(torch.float32).to(device)
            h_raw_last = h_raw_last.reshape(flat.shape)            # [K,6144]
            d = (flat - h_raw_last).abs().max().item()
            max_diff = max(max_diff, d)
            n_checked += 1
    backbone.train()
    print(f"[fidelity] replay vs slots_h_raw max-abs-diff over {n_checked} "
          f"records: {max_diff:.4f} (atol {atol})", flush=True)
    if n_checked == 0:
        # No records carried the replay keys -> the gate did NOT actually run.
        # Fail loud rather than vacuously passing on 0.0 max_diff.
        print("[fidelity] ABORT: 0 records had slots_pre_state + slots_step_input "
              "+ slots_h_raw -- the gate did not run.", file=sys.stderr)
        return False, max_diff
    return max_diff <= atol, max_diff


def stamp_gold_kind(records, doc_kind_map):
    """Stamp ``gold_doc_kind`` (0/1/2) on each record, matching #5's main."""
    counts = {phh.DOC_KIND_CONV: 0, phh.DOC_KIND_TEXT: 0, phh.DOC_KIND_CODE: 0}
    for rec in records:
        gk_str = phh._gold_doc_kind(rec, doc_kind_map)
        if gk_str == "code":
            gk = phh.DOC_KIND_CODE
        elif gk_str == "text":
            gk = phh.DOC_KIND_TEXT
        else:
            gk = phh.DOC_KIND_CONV
        rec["gold_doc_kind"] = gk
        counts[gk] += 1
    return counts


def build_sampler_weights(records, sqrt_freq=True):
    """sqrt-inverse-freq no-replacement sampler weights (the #5 protocol)."""
    counts = {phh.DOC_KIND_CONV: 0, phh.DOC_KIND_TEXT: 0, phh.DOC_KIND_CODE: 0}
    for r in records:
        counts[int(r["gold_doc_kind"])] += 1
    inv = {k: (1.0 / counts[k] if counts[k] > 0 else 0.0) for k in counts}
    if sqrt_freq:
        inv = {k: inv[k] ** 0.5 for k in counts}
    w = [inv[int(r["gold_doc_kind"])] for r in records]
    return torch.tensor(w, dtype=torch.double), counts, inv


def main() -> int:
    p = argparse.ArgumentParser(
        description="STRM 1f-7 Stage 3: flat_last backbone fine-tune (pre-state replay).")
    p.add_argument("--prestate-trace", default=DEFAULT_PRESTATE,
                   help="trace with slots_pre_state (generated with --capture-pre-state).")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="input backbone to fine-tune (default backbone_v2_full.pt).")
    p.add_argument("--doc-store", default=DEFAULT_DOC_STORE,
                   help="doc corpus store (for the gold-kind map).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="output fine-tuned backbone path.")
    p.add_argument("--epochs", type=int, default=8,
                   help="fine-tune epochs (DeepSeek: 5-10).")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="backbone lr (lightly fine-tune, ~1/10 from-scratch base).")
    p.add_argument("--readout-lr", type=float, default=1e-4,
                   help="throwaway readout lr (from-scratch, faster than backbone).")
    p.add_argument("--margin", type=float, default=2.5,
                   help="margin-ranking hinge (matches #5 / the z_logit gate).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--uniform-sampler", action="store_true",
                   help="disable sqrt-freq balancing (uniform shuffle). Default off = sqrt-freq.")
    p.add_argument("--device", default="auto")
    p.add_argument("--fidelity-atol", type=float, default=0.15,
                   help="replay-fidelity abs-diff threshold; above this the run aborts.")
    p.add_argument("--skip-fidelity", action="store_true",
                   help="skip the epoch-0 replay-fidelity gate (NOT recommended).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # Resolve "auto" -> cuda if available else cpu (load_backbone accepts the
    # raw string, but torch.device() does not; mirror the generator's --device
    # auto semantics so the replay runs on GPU when present).
    dev_str = args.device
    if dev_str == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)

    # ── load + unfreeze backbone ──
    backbone = load_backbone(args.backbone, device=dev_str, map_location="cpu")
    backbone.train()
    for pp in backbone.parameters():
        pp.requires_grad_(True)
    n_params = sum(pp.numel() for pp in backbone.parameters() if pp.requires_grad)
    print(f"backbone: {n_params / 1e6:.1f}M trainable params (unfrozen)", flush=True)

    # ── throwaway slim readout: Linear(6144,384) + cos(z,q)/TEMP ──
    readout = nn.Linear(DIM_IN, 384).to(device)
    n_readout = sum(pp.numel() for pp in readout.parameters())
    print(f"throwaway readout: Linear(6144,384) = {n_readout / 1e3:.0f}K params "
          f"(discarded after fine-tune; cos(z,q)/T={TEMP})", flush=True)

    # ── load + prep records (drop self-slot, stamp gold_doc_kind) ──
    print(f"loading pre-state trace: {args.prestate_trace}", flush=True)
    raw = torch.load(args.prestate_trace, map_location="cpu", weights_only=False)
    records = []
    n_dropped_self = 0       # _drop_self_slot returned None (<3 slots after self-removal)
    n_dropped_keys = 0       # missing slots_pre_state and/or slots_step_input
    for r in raw:
        d = phh._drop_self_slot(r)
        if d is None:
            n_dropped_self += 1
            continue
        if "slots_pre_state" not in d or "slots_step_input" not in d:
            n_dropped_keys += 1
            continue
        records.append(d)
    if not records:
        print("ERROR: no records with slots_pre_state + slots_step_input after "
              "drop-self-slot (was the trace generated with --capture-pre-state "
              "AND the u-capture fix?)", file=sys.stderr)
        return 1
    # Surface silent skips: a non-zero n_dropped_keys means the trace is missing
    # the replay seed on some records (e.g. an old --capture-pre-state trace from
    # before the u-capture fix) -> those records were dropped, not trained on.
    print(f"records: {len(records)}/{len(raw)} kept | dropped {n_dropped_self} "
          f"(<3 slots after self-removal) + {n_dropped_keys} (missing pre-state/"
          f"step-input keys)", flush=True)
    doc_kind_map = phh._build_doc_kind_map_cached(args.doc_store, args.prestate_trace)
    gk_counts = stamp_gold_kind(records, doc_kind_map)
    print(f"gold-kind counts conv={gk_counts[phh.DOC_KIND_CONV]}/"
          f"text={gk_counts[phh.DOC_KIND_TEXT]}/code={gk_counts[phh.DOC_KIND_CODE]}",
          flush=True)

    # ── sampler weights (sqrt-freq no-replacement, the #5 protocol) ──
    sqrt_freq = not args.uniform_sampler
    weights, _, inv = build_sampler_weights(records, sqrt_freq=sqrt_freq)
    flag_desc = "sqrt-inverse-freq" if sqrt_freq else "uniform"
    print(f"sampler: {flag_desc} no-replacement (weighted shuffle, each record "
          f"once/epoch) | weights conv={inv[phh.DOC_KIND_CONV]:.4f}/"
          f"text={inv[phh.DOC_KIND_TEXT]:.4f}/code={inv[phh.DOC_KIND_CODE]:.4f}",
          flush=True)

    # ── epoch-0 replay fidelity gate (before any update) ──
    if not args.skip_fidelity:
        ok, max_diff = fidelity_check(backbone, records, device,
                                      atol=args.fidelity_atol)
        if not ok:
            print(f"ABORT: replay fidelity {max_diff:.4f} > atol "
                  f"{args.fidelity_atol}. The pre-state seed does NOT reproduce "
                  f"the trace -- the fine-tune would train on bogus state. "
                  f"Diagnose before proceeding.", file=sys.stderr)
            return 1
        print("[fidelity] OK -- replay reproduces the trace; proceeding.", flush=True)

    # ── optimizer + cosine schedule ──
    opt = torch.optim.AdamW([
        {"params": list(backbone.parameters()), "lr": args.lr},
        {"params": list(readout.parameters()), "lr": args.readout_lr},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── training loop ──
    for epoch in range(args.epochs):
        backbone.train()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(records), replacement=False)
        ep_loss = 0.0
        n = 0
        t0 = time.time()
        for idx in sampler:
            rec = records[int(idx)]
            pre = rec["slots_pre_state"].to(torch.float32)        # [K,4,16,384]
            u = rec["slots_step_input"].to(torch.float32)         # [K,384] exact step-input
            q = rec["query_emb"].to(torch.float32).to(device)     # [384]
            flat = replay_flatlast(backbone, u, pre, device)      # [K,6144] w/ grad
            z = readout(flat)                                     # [K,384]
            logits = F.cosine_similarity(z, q.unsqueeze(0), dim=-1, eps=1e-8) / TEMP
            gold_mask = rec["labels"].to(torch.bool).to(device)
            loss = phh.margin_ranking_loss(logits, gold_mask, args.margin,
                                           hard_negative=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n += 1
        sched.step()
        print(f"epoch {epoch}: train_loss={ep_loss / max(n, 1):.4f} "
              f"({n} records, lr_backbone={sched.get_last_lr()[0]:.2e}) "
              f"{time.time() - t0:.1f}s", flush=True)

    # ── save (discard the throwaway readout) ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Match load_backbone's checkpoint format: {"backbone": state_dict, "step": n}.
    # Only backbone params are saved -- NO readout keys (it is discarded).
    torch.save({"backbone": backbone.state_dict(), "step": 0}, out_path)
    saved_keys = len(backbone.state_dict())
    print(f"saved fine-tuned backbone -> {out_path} ({saved_keys} tensors, "
          f"readout discarded)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())