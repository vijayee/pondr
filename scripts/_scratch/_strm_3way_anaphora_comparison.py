"""STRM 3-way anaphora comparison table (Step 2 deliverable).

Reads the three candidate result JSONs and prints + writes the 3x2x4 table:
candidate {C1, C2, C3} x set {heldout-17, loo-8-normal} x {top1, in_top3,
median breadth, fires/target_in_ring}. C3's anaphora number is the new data
point (does the doc-ring-trained readout generalize to conv-turn anaphora?).

  C1 = cosine+age + bge-reranker-v2-m3 (rank-then-budget)  -> _c1_reranker_result.json
  C2 = qwen3:8b zero-shot window retrieval                  -> _llm_salience_result.json
  C3 = text2x CompositeZHead readout (seed 0; 6-seed rollup if present)
                                                            -> _c3_anaphora_result.json

The doc-ring C1/C2 comparison is DEFERRED to the chat harness (Step 3): the
doc-ring gold is cosine-top-1 by construction -> circular for C1, fair for C2,
meaningful for C3 (already measured, 1f-7 6-seed PASS). Documented in the JSON.

UNTRACKED scratch. onyx PRIVATE. No uploads. No engine edits.

Run: PYTHONPATH=. python scripts/_scratch/_strm_3way_anaphora_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
C1_PATH = SCRATCH / "_c1_reranker_result.json"
C2_PATH = SCRATCH / "_llm_salience_result.json"
C3_PATH = SCRATCH / "_c3_anaphora_result.json"
OUT_PATH = SCRATCH / "_strm_3way_anaphora_comparison.json"

SETS = ["heldout-17", "loo-8-normal"]


def _c1_numbers(c1: dict) -> dict[str, dict]:
    """C1 best config per set (already picked by highest in_top3)."""
    out = {}
    for s in SETS:
        b = c1.get("best", {}).get(s)
        if b:
            out[s] = {"top1": b.get("target_top1_rate"),
                      "in_top3": b.get("target_in_top3_rate"),
                      "breadth": b.get("median_breadth"),
                      "fires": b.get("fires_on_target_rate"),
                      "config": f"{b.get('model','?').split('/')[-1]} "
                                f"cos_floor={b.get('cos_floor')}"}
    return out


def _c2_numbers(c2: dict) -> dict[str, dict]:
    out = {}
    for s in SETS:
        key = "heldout" if s == "heldout-17" else "loo"
        d = c2.get(key)
        if d:
            out[s] = {"top1": d.get("target_top1_rate"),
                      "in_top3": d.get("target_in_top3_rate"),
                      "breadth": d.get("median_breadth"),
                      "fires": None,  # LLM: fires degenerate (always 3 picks)
                      "config": f"{c2.get('model','?')} zero-shot"}
    return out


def _c3_numbers(c3: dict) -> dict[str, dict]:
    """C3 seed-0 (mixed ring). If a 6-seed rollup is present (key 'seeds'), use
    its mean; else the single-seed 'sets' block. Also surfaces the conv0
    diagnostic so the type-0-only undercount is visible, not hidden."""
    out = {}
    sets = c3.get("sets", {})
    for s in SETS:
        d = sets.get(s)
        if d:
            out[s] = {"top1": d.get("target_top1_rate"),
                      "in_top3": d.get("target_in_top3_rate"),
                      "breadth": d.get("median_breadth"),
                      "fires": d.get("target_in_ring_rate"),
                      "n": d.get("n_in_window"),
                      "config": f"CompositeZHead seed={c3.get('seed','?')} "
                                f"(mixed ring)"}
            diag = sets.get(f"{s}_conv0")
            if diag:
                out[s]["conv0_diag_top1"] = diag.get("target_top1_rate")
                out[s]["conv0_diag_in_top3"] = diag.get("target_in_top3_rate")
    return out


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    return f"{v:.3f}"


def main() -> int:
    c1 = json.loads(C1_PATH.read_text(encoding="utf-8"))
    c2 = json.loads(C2_PATH.read_text(encoding="utf-8")) if C2_PATH.exists() else {}
    c3 = json.loads(C3_PATH.read_text(encoding="utf-8")) if C3_PATH.exists() else {}

    n1 = _c1_numbers(c1)
    n2 = _c2_numbers(c2)
    n3 = _c3_numbers(c3)

    table = {"C1": n1, "C2": n2, "C3": n3}
    out = {"candidates": {
        "C1": {"desc": "cosine+age + bge-reranker-v2-m3 (rank-then-budget, "
                       "no training)", "sets": n1},
        "C2": {"desc": "qwen3:8b zero-shot window retrieval (no training, "
                       "local $0)", "sets": n2},
        "C3": {"desc": "text2x CompositeZHead readout (doc-ring-trained, "
                       "scored on conv anaphora -- the OOD cross-test)",
                       "sets": n3},
    },
        "doc_ring_note": (
        "C3 doc-ring = 1f-7 6-seed PASS (ret_text+ret_code z_logit>=2.0), "
        "already measured -- reused, not re-run. C1/C2 doc-ring quantitative "
        "comparison DEFERRED to the chat harness (Step 3): the doc-ring gold "
        "is cosine-top-1 by construction (generate_onyx_doc_ring_traces.py:215) "
        "-> circular for C1, fair for C2. A fair C1/C2 doc-ring number needs a "
        "hand-authored doc-ring gold (real labeling), not worth paying before "
        "the human-centric verdict."),
        "metric_note": (
        "top1 = gold referent ranked #1 of the ~16-turn window. in_top3 = gold "
        "in the surfaced top-3 (recall@budget=3, the ARTIFICIAL SHORT-TERM "
        "MEMORY goal). fires/target_in_ring = was the target rankable at all "
        "(C1: cos>cos_floor; C3: target turn present in the ring). C2 fires is "
        "degenerate (LLM always returns 3). The ship bar (pinpoint frame) was "
        "2/3 on top1 AND in_top3; the recall@budget frame centers on in_top3."),
    }

    print("=" * 78)
    print("STRM 3-WAY ANAPHORA COMPARISON (recall@budget=3, age>=3, 16-turn window)")
    print("=" * 78)
    for cand, label in (("C1", "C1 cosine+age+CE reranker"),
                        ("C2", "C2 LLM qwen3:8b zero-shot"),
                        ("C3", "C3 CompositeZHead readout")):
        print(f"\n[{label}]")
        print(f"  {'set':<14} {'top1':>7} {'in_top3':>8} {'breadth':>8} "
              f"{'fires/in_ring':>14}  config")
        for s in SETS:
            d = table[cand].get(s, {})
            cfg = d.get("config", "")
            print(f"  {s:<14} {_fmt(d.get('top1')):>7} "
                  f"{_fmt(d.get('in_top3')):>8} {_fmt(d.get('breadth')):>8} "
                  f"{_fmt(d.get('fires')):>14}  {cfg}")
            if d.get("conv0_diag_top1") is not None:
                print(f"    {'(conv0 diag)':<12} "
                      f"{_fmt(d.get('conv0_diag_top1')):>7} "
                      f"{_fmt(d.get('conv0_diag_in_top3')):>8}   "
                      f"--- type-0-only; undercounts target=0 ---")

    print("\n" + "-" * 78)
    print("RECALL@budget=3 (in_top3) -- the artificial short-term memory goal:")
    for s in SETS:
        row = {c: table[c].get(s, {}).get("in_top3") for c in ("C1", "C2", "C3")}
        print(f"  {s:<14}  C1={_fmt(row['C1'])}  C2={_fmt(row['C2'])}  "
              f"C3={_fmt(row['C3'])}")
    print("\nPINPOINT (top1) -- the old flawed gate's bar:")
    for s in SETS:
        row = {c: table[c].get(s, {}).get("top1") for c in ("C1", "C2", "C3")}
        print(f"  {s:<14}  C1={_fmt(row['C1'])}  C2={_fmt(row['C2'])}  "
              f"C3={_fmt(row['C3'])}")

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())