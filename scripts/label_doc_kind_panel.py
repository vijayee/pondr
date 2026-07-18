"""Relabel the v2 doc-kind corpus with a 3-teacher majority panel (clean labels).

The audit found DeepSeek-flash over-assigns ``decision_update`` CONFIDENTLY
(0.85-0.90 on support threads / status reports / design questions), so the v2
confidence>=0.7 gate did not catch it -- both the val labels (which capped the
scorecard) AND the train labels (which the head learned from) are contaminated,
and the synthetic generator's blind-verify used the same flash (so v3's 2x dec
data inherited the bias -- that's why tripling didn't help dec).

This relabels all 380 v2 pairs with a 3-teacher panel:
  teacher 1 = flash label ALREADY in pairs_v2.jsonl (no re-call -- it IS flash)
  teacher 2 = glm-5.2:cloud  (independent family)
  teacher 3 = gemma4:31b-cloud (independent family)
using the SAME v2 labeler prompt (isolates the model as the only variable).
Clean label = majority (>=2 of 3 agree); a 3-way split -> ABSTAIN (drop from
train; for val, fall back to the flash label so the 76-doc scorecard denominator
stays comparable, but flag it).

Outputs pairs_clean_train.jsonl (304 real split, majority-only) +
pairs_clean_val.jsonl (76 real split, majority-or-flash-fallback). Same seed-0
split as prep_doc_kind_v3_split.py so the clean val == the same 76 docs.

GPU-free (cloud Oracle only). One-time offline data prep (NOT prod traffic).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reuse the v2 labeler prompt + label set + chunker (isolates model as variable).
_labeler_path = Path(__file__).resolve().parent / "label_doc_kind_corpus.py"
_spec = importlib.util.spec_from_file_location("_label_doc_kind_corpus", _labeler_path)
_labeler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labeler)
_LABELS = _labeler._LABELS
_LABEL_PROMPT = _labeler._PROMPT

SEED = 0
VAL_FRACTION = 0.2


def _majority(labels: list[str | None]) -> str | None:
    """>=2 of 3 agree on a valid label -> that label; else None (abstain)."""
    counts: dict[str, int] = {}
    for l in labels:
        if l in _LABELS:
            counts[l] = counts.get(l, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    if best >= 2:
        # the label with >=2 votes
        return max(counts.items(), key=lambda kv: (kv[1], -_LABELS.index(kv[0])))[0]
    return None  # 3-way split (each teacher a different label) -> abstain


def main() -> int:
    ap = argparse.ArgumentParser(description="3-teacher panel relabel of the v2 corpus.")
    ap.add_argument("--v2-pairs", default="data/training/doc_kind_head/pairs_v2.jsonl")
    ap.add_argument("--glm-cache", default="data/training/doc_kind_head/oracle_cache_audit.json",
                    help="glm cache (reuses the 76-doc audit cache for the val half)")
    ap.add_argument("--gemma-cache", default="data/training/doc_kind_head/oracle_cache_panel_gemma.json")
    ap.add_argument("--glm-model", default="glm-5.2:cloud")
    ap.add_argument("--gemma-model", default="gemma4:31b-cloud")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--out-train", default="data/training/doc_kind_head/pairs_clean_train.jsonl")
    ap.add_argument("--out-val", default="data/training/doc_kind_head/pairs_clean_val.jsonl")
    args = ap.parse_args()

    from src.config import config as _config
    from src.ingestion.doc_kind import join_section_texts
    from src.training.oracle_labeling import OracleClient, OracleConfig

    v2 = []
    with open(args.v2_pairs, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                v2.append(json.loads(line))
    print(f"v2 corpus: {len(v2)} docs", flush=True)

    prompts = [_LABEL_PROMPT + join_section_texts(r["section_texts"]) for r in v2]

    def _client(model: str, cache: Path) -> OracleClient:
        cfg = OracleConfig(
            model=model, endpoint=_config.oracle_endpoint, temperature=0.1,
            max_tokens=512, batch_delay=0.0, cache_path=cache, think=None,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        return OracleClient(cfg)

    # teacher 1 (flash) is already in the v2 file. Call teachers 2 + 3.
    glm = _client(args.glm_model, Path(args.glm_cache))
    print(f"calling {args.glm_model} on {len(v2)} docs (reuses audit cache for the "
          f"76 val docs)...", flush=True)
    glm_res = glm.generate_batch(prompts, response_format="json_object",
                                 max_workers=args.max_workers)

    gemma = _client(args.gemma_model, Path(args.gemma_cache))
    print(f"calling {args.gemma_model} on {len(v2)} docs...", flush=True)
    gemma_res = gemma.generate_batch(prompts, response_format="json_object",
                                     max_workers=args.max_workers)

    def _lab(res) -> str | None:
        if res.error or not isinstance(res.response, dict):
            return None
        l = res.response.get("doc_kind")
        return l if l in _LABELS else None

    # Build per-doc panel + majority.
    rows = []
    n_abstain = 0
    agree_counts = {1: 0, 2: 0, 3: 0}
    for rec, gr, er in zip(v2, glm_res, gemma_res):
        flash = rec["label"]   # teacher 1 (already in v2 file)
        g = _lab(gr)
        e = _lab(er)
        clean = _majority([flash, g, e])
        voters = sum(1 for x in [flash, g, e] if x is not None)
        if clean is None:
            n_abstain += 1
        else:
            n = sum(1 for x in [flash, g, e] if x == clean)
            agree_counts[n] = agree_counts.get(n, 0) + 1
        rows.append({**rec, "flash_label": flash, "glm_label": g, "gemma_label": e,
                     "clean_label": clean, "n_voters": voters})

    print(f"\npanel done: {len(rows)} docs, {n_abstain} abstained (3-way split or "
          f"<2 valid votes).", flush=True)
    print(f"agreement among non-abstained: {agree_counts}", flush=True)

    # Per-class: how often did flash's label survive the panel (flash == clean)?
    flash_survive = 0
    flash_changed = 0
    for r in rows:
        if r["clean_label"] is None:
            continue
        if r["clean_label"] == r["flash_label"]:
            flash_survive += 1
        else:
            flash_changed += 1
    print(f"flash label survived panel: {flash_survive}, changed: {flash_changed} "
          f"(of {len(rows)-n_abstain})", flush=True)

    # Reproduce the seed-0 split (identical to prep_doc_kind_v3_split.py).
    seen, unique = set(), []
    for r in rows:
        did = r.get("doc_id") or "\n".join(r["section_texts"])
        if did in seen:
            continue
        seen.add(did)
        unique.append(r)
    rng = random.Random(SEED)
    idx = list(range(len(unique)))
    rng.shuffle(idx)
    n_val = max(1, int(len(unique) * VAL_FRACTION))
    val = [unique[i] for i in idx[:n_val]]
    train = [unique[i] for i in idx[n_val:]]

    def _dist(rs, key):
        d: dict[str, int] = {}
        for r in rs:
            v = r.get(key) or "(abstain)"
            d[v] = d.get(v, 0) + 1
        return dict(sorted(d.items()))

    # train: majority-only (drop abstentions).
    clean_train = [r for r in train if r["clean_label"] is not None]
    # val: keep all 76 (majority if present, else flash fallback) so the scorecard
    # denominator stays comparable; flag abstentions.
    clean_val = []
    for r in val:
        label = r["clean_label"] if r["clean_label"] is not None else r["flash_label"]
        clean_val.append({
            "doc_id": r.get("doc_id"), "section_texts": r["section_texts"],
            "label": label, "clean_label": r["clean_label"],
            "panel_abstain": r["clean_label"] is None,
            "flash_label": r["flash_label"], "glm_label": r["glm_label"],
            "gemma_label": r["gemma_label"],
            "confidence": r.get("confidence"),
        })
    # train records: use the clean majority label.
    clean_train_out = [{
        "doc_id": r.get("doc_id"), "section_texts": r["section_texts"],
        "label": r["clean_label"], "confidence": r.get("confidence"),
        "flash_label": r["flash_label"], "glm_label": r["glm_label"],
        "gemma_label": r["gemma_label"],
    } for r in clean_train]

    train_abstain = len(train) - len(clean_train)
    val_abstain = sum(1 for r in clean_val if r["panel_abstain"])
    print(f"\nsplit (seed {SEED}): {len(train)} train / {len(val)} val", flush=True)
    print(f"  train: {len(clean_train_out)} clean (dropped {train_abstain} abstentions)",
          flush=True)
    print(f"  train clean label dist: {_dist(clean_train, 'clean_label')}", flush=True)
    print(f"  train original flash dist: {_dist(train, 'label')}", flush=True)
    print(f"  val: {len(clean_val)} docs ({val_abstain} abstentions fell back to flash)",
          flush=True)
    print(f"  val clean label dist: {_dist(clean_val, 'clean_label')}", flush=True)
    print(f"  val final(dist used for scorecard): {_dist(clean_val, 'label')}", flush=True)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in clean_train_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_val, "w", encoding="utf-8") as f:
        for r in clean_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out_train} ({len(clean_train_out)}) and "
          f"{args.out_val} ({len(clean_val)})", flush=True)

    gs = glm.get_stats()
    es = gemma.get_stats()
    print(f"glm: {gs['total_calls']} calls ({gs['cached_calls']} cached), "
          f"{gs['total_tokens']} tokens", flush=True)
    print(f"gemma: {es['total_calls']} calls ({es['cached_calls']} cached), "
          f"{es['total_tokens']} tokens", flush=True)
    print("next: re-score the existing v2 head on pairs_clean_val (CPU, true "
          "scorecard), then retrain on pairs_clean_train (GPU, after game).",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())