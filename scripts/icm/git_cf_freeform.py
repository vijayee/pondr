"""FREE-FORM PHRASING TEST -- user-originated before/after phrasings, same pairs.

WHY: v3 (addendum 29) passed on a fixed query template. The last authored piece
was the template itself, so the judge's "EARLIEST" reading was guaranteed by
the words "FIRST/before". This run removes the template: the user authored two
blocks of natural phrasings (a before-block and an after-block). The same real
counterfactual pair is queried twice, once with a before-phrasing (target = the
pre-change version) and once with an after-phrasing (target = the post-change
version). The correct answer SWAPS; a pipeline that merely learned "first ->
earliest" fails the after direction.

FIRST ATTEMPT (2026-10-20): the abstract epoch keyer (the store's write-time
prompt) extracted the expected ordinal on 0/8 smoke queries. The judge returned
ABSOLUTE anchors ("today", "last week's", the commit text) because the original
phrasings reference times the isolated trial does not contain, and several are
not version-selection requests. The user authorised minimal legibility edits.
This run therefore uses the user-originated, assistant-edited set
data/_scratch/icm/BeforAndAfter_test.txt: every line is a version-selection
request anchored to {change}; external time anchors and non-selection asks are
removed or re-anchored. The KEYER IS TASK-AWARE (BEFORE/AFTER/NEITHER) because
the abstract time-reference prompt is template-shaped. The original file is
untouched. Author caveat: the wording is user-originated with assistant edits.

PRE-REGISTERED DESIGN (written before this run):
  PAIRS: seeded sample of the 561 v3 trials (default 150; >=120 for power).
  PHRASINGS: cyclic assignment over the edited before/after blocks, shuffled
    with the seed so every phrasing is used across pairs.
  POOLS: before-direction = all v3 records (before/after/removed + decoys);
    after-direction = the same records WITHOUT the post-removal version (the
    "version this change produced" is the post-add version). Decoys are kept;
    the type filter is the exact file path and the key's requirement tokens are
    ignored.
  ORACLE: git blob IDs (parent-of-add for before, add for after).
  ARMS: SEMANTIC-ONLY (type filter -> key re-rank over the full filtered pool)
    is PRIMARY; TWO-STAGE (dense top-5 -> same) and DENSE-RELEVANT are reported.
  KEYER: one task-aware judge call per query: BEFORE -> EARLIEST,
    AFTER -> LATEST, NEITHER -> abstain (counted as an extraction failure).

  PRE-REGISTERED BAR:
    (1) POWER:      pairs >= 120
    (2) EXTRACTION: expected ordinal on >= 80% of all (pair, direction) runs
    (3) BEFORE:     semantic-only accuracy on keyed runs >= 0.85
    (4) AFTER:      semantic-only accuracy on keyed runs >= 0.85
    (5) SWAP:       among pairs where BOTH directions are keyed, both semantic
                    answers correct >= 0.75
    (6) VACUITY:    before-direction dense-relevant <= chance + 0.15

Usage:
  python scripts/icm/git_cf_freeform.py --pairs 150 --time-budget 480
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from icm.intent_keys import KeyResult, ollama_ask  # noqa: E402
from icm.intent_query import TwoStageRetriever  # noqa: E402

POOL = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted",
                    "git_cf_v3", "pool.jsonl")
PHRASES = os.path.join(REPO_ROOT, "data", "_scratch", "icm",
                       "BeforAndAfter_test.txt")
OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted",
                      "git_cf_freeform_v2")
ROWS = os.path.join(OUTDIR, "rows.jsonl")
RESULTS = os.path.join(OUTDIR, "results.json")
CACHE = os.path.join(OUTDIR, "keys_cache.jsonl")

_emb = [None]


def embed(texts):
    if _emb[0] is None:
        from sentence_transformers import SentenceTransformer
        _emb[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _emb[0].encode(list(texts), batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)


def load_phrasings(path):
    """Parse the authored blocks; the blank line separates before from after."""
    blocks, cur = [], []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    if len(blocks) < 2:
        raise SystemExit("phrasing file has no blank-line-separated blocks")
    return blocks[0], blocks[1]


FREEFORM_PROMPT = """The request below refers to a version of a file and to a change made to it.
Does the request ask for the version from BEFORE that change, the version from
AFTER that change, or neither? Answer exactly one word: BEFORE, AFTER, or
NEITHER.

Request: {request}"""


class FreeformKeyer:
    """Task-aware free-form keyer: BEFORE -> earliest, AFTER -> latest.

    This is the free-form counterpart of the store's write-time keyer. The
    abstract time-reference prompt was template-shaped (0/8 on the original
    phrasings); this prompt asks the question the retrieval task actually
    poses. One call per unique query, hash-cached.
    """

    def __init__(self, cache_path=None, ask=None,
                 model="deepseek-v4.1-flash:cloud"):
        self.cache_path = cache_path
        self._ask = ask or (lambda p: ollama_ask(p, model=model))
        self._cache: dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            for line in io.open(cache_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._cache[row["hash"]] = row["val"]
                except Exception:
                    continue

    def _save(self, h, label):
        self._cache[h] = label
        if not self.cache_path:
            return
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with io.open(self.cache_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"hash": h, "val": label}) + "\n")

    @staticmethod
    def _key_for_label(label):
        if label == "BEFORE":
            return {"requirements": [], "qualifier": "",
                    "epoch_ref": {"kind": "ordinal", "value": "EARLIEST"},
                    "abstained": False}
        if label == "AFTER":
            return {"requirements": [], "qualifier": "",
                    "epoch_ref": {"kind": "ordinal", "value": "LATEST"},
                    "abstained": False}
        return {"requirements": [], "qualifier": "",
                "epoch_ref": {"kind": "none", "value": ""}, "abstained": True}

    def key_for(self, text: str, declared_qualifier: str = "") -> KeyResult:
        text = " ".join(str(text).split())
        if not text:
            return KeyResult(None, "error")
        h = hashlib.sha1(("FF|freeform-v1|" + text).encode("utf-8")).hexdigest()
        cached = h in self._cache
        if cached:
            label = self._cache[h]
        else:
            raw = self._ask(FREEFORM_PROMPT.format(request=text[:1200]))
            if not raw.strip():
                return KeyResult(None, "error")
            up = raw.upper()
            if "BEFORE" in up:
                label = "BEFORE"
            elif "AFTER" in up:
                label = "AFTER"
            elif "NEITHER" in up:
                label = "NEITHER"
            else:
                return KeyResult(None, "error")
            self._save(h, label)
        return KeyResult(self._key_for_label(label),
                         "cached" if cached else "judged")


def build_pools(trial):
    before_pool = list(trial["records"])
    after_pool = [r for r in trial["records"]
                  if not (r.get("kind") == "version"
                          and r.get("label") == "removed")]
    return before_pool, after_pool


def first_same_file(order_ids, by_id, file):
    for i in order_ids:
        if by_id.get(i, {}).get("type") == file:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-budget", type=int, default=0)
    args = ap.parse_args()

    before_phr, after_phr = load_phrasings(PHRASES)
    print(f"[phrases] before={len(before_phr)} after={len(after_phr)}", flush=True)
    trials = [json.loads(line) for line in io.open(POOL, encoding="utf-8")
              if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(trials)
    trials = trials[: args.pairs]
    rng.shuffle(before_phr)
    rng.shuffle(after_phr)
    print(f"[pairs] {len(trials)}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)

    keyer = FreeformKeyer(cache_path=CACHE)
    retriever = TwoStageRetriever(embed, keyer=keyer)
    done = {}
    if os.path.exists(ROWS):
        for line in io.open(ROWS, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[f"{r['trial']}|{r['direction']}"] = r
            except Exception:
                continue
    print(f"[checkpoint] {len(done)} runs already scored", flush=True)
    fh = io.open(ROWS, "a", encoding="utf-8", newline="\n")
    t0 = time.time()
    try:
        for pi, t in enumerate(trials):
            before_pool, after_pool = build_pools(t)
            for direction, phrasings, pool, correct, expected in (
                    ("before", before_phr, before_pool, t["correct"], "EARLIEST"),
                    ("after", after_phr, after_pool, t["after"], "LATEST")):
                key_id = f"{t['trial']}|{direction}"
                if key_id in done:
                    continue
                phrase = phrasings[pi % len(phrasings)]
                query = (phrase.replace("{file}", t["file"])
                         .replace("{change}", t["add_msg"]))
                meta = {"type": t["file"], "ignore_key_entities": True, "k": 5}
                full = retriever.query(query, pool, query_meta=meta)
                key = full["key"]
                e = (key or {}).get("epoch_ref", {}).get("value", "none") or "none"
                by_id = {r["id"]: r for r in pool}
                dense_rel = first_same_file(full["stage1_ids"], by_id, t["file"])
                sem_top = full["top"][0]["id"] if full["top"] else None
                cands5 = [by_id[x] for x in full["stage1_ids"][:5] if x in by_id]
                two = retriever.query(query, cands5, query_key=key, query_meta=meta)
                two_top = two["top"][0]["id"] if two["top"] else None
                row = {
                    "trial": t["trial"], "source": t.get("source", "own"),
                    "direction": direction, "phrase_idx": pi % len(phrasings),
                    "query": query, "correct": correct,
                    "expected": expected, "epoch_ref": e,
                    "keyed": 1.0 if e == expected else 0.0,
                    "semantic_top": sem_top, "two_top": two_top,
                    "dense_relevant_top": dense_rel,
                    "sem_correct": 1.0 if sem_top == correct else 0.0,
                    "two_correct": 1.0 if two_top == correct else 0.0,
                    "dense_correct": 1.0 if dense_rel == correct else 0.0,
                    "pool": len(pool),
                    "versions": sum(1 for r in pool if r["kind"] == "version"),
                    "fallback": full["fallback"],
                    "judge_status": full["key_status"],
                }
                done[key_id] = row
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")
                fh.flush()
                if len(done) % 10 == 0:
                    print(f"  [checkpoint {len(done)}] ({time.time()-t0:.0f}s)",
                          flush=True)
                if args.time_budget and (time.time() - t0) >= args.time_budget:
                    print(f"[budget] stopping after {len(done)} runs "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    break
            if args.time_budget and (time.time() - t0) >= args.time_budget:
                break
    finally:
        fh.close()

    need = len(trials) * 2
    rows = [done[f"{t['trial']}|{d}"] for t in trials
            for d in ("before", "after")
            if f"{t['trial']}|{d}" in done]
    complete = len(rows) == need
    by_dir = {}
    for d in ("before", "after"):
        rs = [r for r in rows if r["direction"] == d]
        if not rs:
            continue
        keyed = [r for r in rs if r["keyed"]]
        by_dir[d] = {
            "n": len(rs),
            "judge_expected_rate": round(statistics.mean(r["keyed"] for r in rs), 3),
            "semantic_all": round(statistics.mean(r["sem_correct"] for r in rs), 3),
            "semantic_keyed": round(statistics.mean(r["sem_correct"] for r in keyed), 3)
            if keyed else None,
            "two_all": round(statistics.mean(r["two_correct"] for r in rs), 3),
            "dense_relevant_all": round(statistics.mean(r["dense_correct"] for r in rs), 3),
            "chance_relevant": round(statistics.mean(1.0 / r["versions"] for r in rs), 3),
            "fallbacks": {k: sum(1 for r in rs if (r["fallback"] or "none") == k)
                          for k in {r["fallback"] or "none" for r in rs}},
        }
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["trial"], {})[r["direction"]] = r
    both = [p for p in by_pair.values() if "before" in p and "after" in p]
    both_keyed = [p for p in both if p["before"]["keyed"] and p["after"]["keyed"]]
    swap_sem = round(statistics.mean(
        p["before"]["sem_correct"] and p["after"]["sem_correct"]
        for p in both_keyed), 3) if both_keyed else None
    swap_two = round(statistics.mean(
        p["before"]["two_correct"] and p["after"]["two_correct"]
        for p in both_keyed), 3) if both_keyed else None
    extraction = round(statistics.mean(r["keyed"] for r in rows), 3) if rows else 0.0
    checks = {
        "power_pairs_ge_120": len(trials) >= 120,
        "extraction_ge_0.80": extraction >= 0.80,
        "before_semantic_ge_0.85": (by_dir.get("before", {}).get("semantic_keyed") or 0) >= 0.85,
        "after_semantic_ge_0.85": (by_dir.get("after", {}).get("semantic_keyed") or 0) >= 0.85,
        "swap_ge_0.75": (swap_sem or 0) >= 0.75,
        "before_vacuity": (by_dir.get("before", {}).get("dense_relevant_all") or 1)
        <= (by_dir.get("before", {}).get("chance_relevant") or 0) + 0.15,
    }
    res = {
        "pairs": len(trials), "runs_scored": len(rows), "complete": complete,
        "extraction_rate": extraction, "by_direction": by_dir,
        "swap_semantic": swap_sem, "swap_two_stage": swap_two,
        "pairs_both_keyed": len(both_keyed), "checks": checks,
        "pass": all(checks.values()) and complete,
        "verdict": ("INCOMPLETE -- resume" if not complete else
                    "PASS -- user-authored phrasings discriminate before/after"
                    if all(checks.values()) else
                    "FAIL -- see the failing row"),
    }
    with io.open(RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps({**res, "rows": rows}, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return (0 if (complete and res["pass"]) else (2 if complete else 3))


if __name__ == "__main__":
    sys.exit(main())
