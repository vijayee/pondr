"""Assemble the verified-reversal training corpus (offset-100 harvest).

For each VERIFIED reversal question: all 3 candidates with
  - cosine(question, candidate)     [recomputed from the store's episode texts]
  - kimi judge score                [from the harvest checkpoint]
  - verifier score                  [for the disagreement pair only]
plus the question text and its type.

Output: data/_scratch/icm/icm_head_corpus_o100.json
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"
CKPT = os.path.join("data", "_scratch", "icm", "organic500_ckpt_o100.jsonl")
VERIF = os.path.join("data", "_scratch", "icm", "reversal_verification_o100.json")
DB = os.path.join("data", "_scratch", "icm", "organic_o100", "ledger")
OUT = os.path.join("data", "_scratch", "icm", "icm_head_corpus_o100.json")

_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


def main() -> int:
    ckpt = {json.loads(l)["qi"]: json.loads(l)
            for l in io.open(CKPT, encoding="utf-8") if l.strip()}
    verif = json.loads(io.open(VERIF, encoding="utf-8").read())
    recs = {100 + i: rec for i, rec in
            enumerate(json.load(io.open(FULL500, encoding="utf-8"))[100:300])}

    led = ICMLedger(DB)
    try:
        texts = {i: r["text"] for i in led.ids("episode") if (r := led.get("episode", i))}
    finally:
        led.close()

    corpus = []
    for qi in verif["verified_qids"]:
        cands = ckpt[qi]["cand_scores"]
        q = recs[qi]["question"]
        cos = {}
        mids = [cid for cid, _ in cands if cid in texts]
        cvecs = _embedder([texts[m] for m in mids])
        qvec = _embedder([q])[0]
        for mid, cv in zip(mids, cvecs):
            cos[mid] = round(float(np.dot(cv, qvec)), 4)
        # verifier scores from the screen (rec_v: cand.vb = verifier on cos_top;
        # judge_favored.v = verifier on judge_favored)
        v_scores = {}
        for r in verif["verified"]:
            if r["qi"] == qi:
                v_scores[r["cand"]["cos_top"]] = r["cand"]["vb"]
                v_scores[r["judge_favored"]["id"]] = r["judge_favored"]["v"]
        entries = []
        for cid, js in cands:
            if cid not in cos:
                continue
            entries.append({"id": cid, "cos": cos[cid], "kimi_judge": js,
                            "verifier_judge": v_scores.get(cid), "text": texts[cid]})
        # rank by kimi judge (the label), store the label order
        entries.sort(key=lambda e: -e["kimi_judge"])
        corpus.append({"qi": qi, "question": q, "type": recs[qi]["question_type"],
                       "spearman": ckpt[qi]["spearman"],
                       "judge_favored_id": verif and next(
                           (r["judge_favored"]["id"] for r in verif["verified"] if r["qi"] == qi), None),
                       "candidates": entries})

    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(corpus, indent=2))
    nc = sum(len(c["candidates"]) for c in corpus)
    print(f"[done] {len(corpus)} verified-reversal questions, {nc} labeled candidates -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())