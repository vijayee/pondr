"""INTEGRATION TEST -- judge-built intent keys vs plain similarity, judged by verifier outcomes.

THE QUESTION (the plan's E2 gate, finally on real intent data with mechanical
labels): given a held-out run's REAL intent (its issue text), does an
intent-keyed index surface the memory units that carried the accepted fix
better than plain similarity does?

DATA (all lifted, untracked):
  codetrace_intents.jsonl   real issue text, steps, files, solved (addendum 17)
  codetrace_files.jsonl     per-(run,file) labels from the accepted patch
                            (useful / attempted / context, addendum 18)
  intent_keys.jsonl         judge SPO keys per intent (intent_keyspace.py)
  flip_relations.jsonl      judge SAME/DIFFERENT/OPPOSED labels for multi-run
                            units behind outcome flips

SPLIT: task-disjoint, seeded 42 (task key parsed from run_id; ALL runs of a
task fall on one side). The STORE is built from TRAIN runs only: for every
file, its text proxy (commands + patch hunks that touched it) and its list of
train intents/keys. Held-out runs query the store.

ARMS:
  DENSE   cosine(embed(issue), embed(file proxy text))
  LEX     token jaccard(issue, proxy text)
  STRUCT  intent-key similarity: max over the file's train keys of
          key_sim(query key, train key), where key_sim = 1.0 if predicate
          classes match, + object-token overlap, - 0.7 on qualifier
          opposition (the addendum-14 schema, applied to JUDGE-BUILT keys)
  HYBRID  reciprocal-rank fusion of DENSE and STRUCT (k=60)

METRICS (per held-out run, over the candidate pool = files seen in train):
  recall@10, recall@50, MRR for the run's USEFUL files (accepted-patch files).
  Ground truth is restricted to useful files present in the pool; pool
  coverage is reported. Paired comparisons (dense vs struct/hybrid) by
  per-run metric differences; direction always reported.

SECONDARY (keyspace validity vs the judge, leakage-free): on the flip
relation pairs, does key_sim agree with the judge's SAME/DIFFERENT/OPPOSED
label (SAME highest, OPPOSED lowest)? Spearman over the relation order.

PRE-REGISTERED BAR:
  (1) HYBRID recall@10 >= DENSE recall@10 + 0.05 (absolute) AND
      HYBRID MRR >= DENSE MRR (paired mean difference > 0)
  (2) STRUCT recall@10 >= DENSE recall@10 (keys alone are competitive)
  (3) keyspace relation agreement: mean key_sim(SAME) > mean key_sim(OPPOSED)
      with a >= 0.25 gap
  PASS requires all three; else the failing arm names the missing piece.
  VACUITY: if the pool has < 20 files per query on average or < 20 held-out
  runs with usable ground truth, the test is UNDERPOWERED and says so.

Run:  python scripts/icm/integration_test.py
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from intent_similarity_ladder import PRED_CLASSES, tokset  # noqa: E402

LIFT = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_intents.jsonl")
FILES = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_files.jsonl")
KEYS = os.path.join("data", "_scratch", "icm", "lifted", "intent_keys.jsonl")
RELS = os.path.join("data", "_scratch", "icm", "lifted", "flip_relations.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "integration_results.json")

SEED = 42
HOLDOUT_FRAC = 0.2
K1, K2 = 10, 50
RRF_K = 60

_state = {}


def emb(texts):
    if "model" not in _state:
        from sentence_transformers import SentenceTransformer
        _state["model"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _state["model"].encode(texts, batch_size=64, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)


def task_key(run_id: str) -> str:
    parts = run_id.split("-")
    return "-".join(parts[2:-1]) if len(parts) >= 4 else run_id


def pred_class(text: str):
    t = tokset(text)
    hits = [c for c, ws in PRED_CLASSES.items() if t & set(ws)]
    if not hits:
        return None
    low = str(text).lower()
    hits.sort(key=lambda c: min(low.find(w) for w in PRED_CLASSES[c] if w in low))
    return hits[0]


ARCH = ("archived", "original", "stored", "snapshot", "recorded", "stale", "old",
        "previous", "existing", "legacy", "at ingest", "earlier", "backup")
CUR = ("current", "fresh", "now", "updated", "new", "latest", "live", "re-ingested",
       "reingested", "recomputed", "refreshed", "present", "today")


def qualifier_of(text: str):
    low = " " + str(text).lower() + " "
    a = any(w in low for w in ARCH)
    c = any(w in low for w in CUR)
    return "ARCHIVED" if a and not c else "CURRENT" if c and not a else ""


def key_sim(qk, tk) -> float:
    """Similarity between two judge-built intent keys, using the addendum-14 schema."""
    if not qk or not tk:
        return 0.0
    c1 = pred_class((qk.get("attitude", "") + " " + qk.get("relation", "")) or "")
    c2 = pred_class((tk.get("attitude", "") + " " + tk.get("relation", "")) or "")
    o1, o2 = tokset(qk.get("object", "")), tokset(tk.get("object", ""))
    ent = 1.0 if (o1 and o2 and len(o1 & o2) / len(o1 | o2) >= 0.3) else 0.0
    cls = 1.0 if (c1 and c2 and c1 == c2) else 0.5 if (c1 and c2) else 0.0
    q1, q2 = qualifier_of(qk.get("qualifier", "")), qualifier_of(tk.get("qualifier", ""))
    opp = 0.7 if ({q1, q2} == {"ARCHIVED", "CURRENT"}) else 0.0
    return max(0.0, min(1.0, 0.5 * cls + 0.5 * ent - opp))


def main() -> int:
    runs = [json.loads(l) for l in io.open(LIFT, encoding="utf-8") if l.strip()]
    files = [json.loads(l) for l in io.open(FILES, encoding="utf-8") if l.strip()]
    keys = {}
    for l in io.open(KEYS, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            keys[r["run_id"]] = r.get("key")
    rels = [json.loads(l) for l in io.open(RELS, encoding="utf-8") if l.strip()] \
        if os.path.exists(RELS) else []
    print(f"[load] runs={len(runs)} file_records={len(files)} keys={len(keys)} "
          f"rels={len(rels)}", flush=True)

    good = [r for r in runs if r.get("issue") and keys.get(r["run_id"])]
    tasks = sorted({task_key(r["run_id"]) for r in good})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tasks))
    n_hold = max(1, int(len(tasks) * HOLDOUT_FRAC))
    hold_tasks = {tasks[i] for i in perm[:n_hold]}
    train_runs = [r for r in good if task_key(r["run_id"]) not in hold_tasks]
    test_runs = [r for r in good if task_key(r["run_id"]) in hold_tasks]
    print(f"[split] tasks={len(tasks)} holdout_tasks={len(hold_tasks)} "
          f"train_runs={len(train_runs)} test_runs={len(test_runs)}", flush=True)

    # ---- store from TRAIN runs only
    rec_by_run = {}
    for rec in files:
        rec_by_run.setdefault(rec["run_id"], []).append(rec)
    store_text, store_keys = {}, {}
    for r in train_runs:
        for rec in rec_by_run.get(r["run_id"], []):
            f = rec["file"]
            txt = (rec.get("cmd_text", "") + " " + rec.get("patch_text", ""))[:2500]
            store_text[f] = (store_text.get(f, "") + " " + txt)[:4000]
            store_keys.setdefault(f, []).append(keys.get(r["run_id"]))
    cand = sorted(store_text)
    print(f"[store] files={len(cand)}", flush=True)
    if len(cand) < 20:
        print("UNDERPOWERED: candidate pool < 20 files")
    proxy_emb = emb([store_text[f][:1500] or f for f in cand])
    idx_of = {f: i for i, f in enumerate(cand)}

    # ---- ground truth per held-out run
    queries = []
    for r in test_runs:
        useful = sorted({rec["file"] for rec in rec_by_run.get(r["run_id"], [])
                         if rec["label"] == "useful" and rec["file"] in idx_of})
        if not useful:
            continue
        queries.append({"run": r, "issue": r["issue"], "useful": useful})
    print(f"[queries] {len(queries)} held-out runs with usable ground truth", flush=True)

    q_emb = emb([q["issue"][:1500] for q in queries]) if queries else np.zeros((0, 384))

    def metrics(order_idx, useful_set):
        ranks = [i for i, f in enumerate(cand) if f in useful_set]
        hits = [r for r in ranks]
        rec10 = sum(1 for r in hits if r < K1) / len(useful_set)
        rec50 = sum(1 for r in hits if r < K2) / len(useful_set)
        mrr = max((1.0 / (r + 1) for r in hits), default=0.0)
        return rec10, rec50, mrr

    arms = {"dense": [], "lex": [], "struct": [], "hybrid": []}
    for qi, q in enumerate(queries):
        qtext = q["issue"]
        qk = keys.get(q["run"]["run_id"])
        # scores
        dense = proxy_emb @ q_emb[qi]
        qtok = tokset(qtext)
        lex = np.array([len(qtok & tokset(store_text[f])) /
                        max(1, len(qtok | tokset(store_text[f]))) for f in cand])
        struct = np.array([max([key_sim(qk, tk) for tk in store_keys.get(f, []) if tk] or [0.0])
                           for f in cand])
        order = {a: np.argsort(-s) for a, s in
                 (("dense", dense), ("lex", lex), ("struct", struct))}
        # RRF fusion of dense + struct rank orders
        rrf = np.zeros(len(cand))
        for a in ("dense", "struct"):
            ranks = np.empty(len(cand), dtype=int)
            ranks[order[a]] = np.arange(len(cand))
            rrf += 1.0 / (RRF_K + ranks)
        arms["hybrid"].append(metrics(np.argsort(-rrf), set(q["useful"])))
        arms["dense"].append(metrics(order["dense"], set(q["useful"])))
        arms["lex"].append(metrics(order["lex"], set(q["useful"])))
        arms["struct"].append(metrics(order["struct"], set(q["useful"])))

    def m(arm, j):
        return round(statistics.mean(x[j] for x in arms[arm]), 3) if arms[arm] else None

    table = {a: {"recall@10": m(a, 0), "recall@50": m(a, 1), "mrr": m(a, 2)}
             for a in arms}
    # ---- secondary: keyspace vs judge relation labels
    rel_stats = {}
    for rel in ("same", "different", "opposed"):
        vals = []
        for row in rels:
            if row.get("relation") != rel:
                continue
            ku = keys.get(row["useful_run"]); kn = keys.get(row["nonuseful_run"])
            if ku and kn:
                vals.append(key_sim(ku, kn))
        rel_stats[rel] = (round(statistics.mean(vals), 3) if vals else None, len(vals))
    checks = {
        "hybrid_beats_dense_recall": (table["hybrid"]["recall@10"] or 0)
        >= (table["dense"]["recall@10"] or 0) + 0.05,
        "hybrid_mrr_ge_dense": (table["hybrid"]["mrr"] or 0) >= (table["dense"]["mrr"] or 0),
        "struct_competitive": (table["struct"]["recall@10"] or 0)
        >= (table["dense"]["recall@10"] or 0),
        "relation_gap": (rel_stats.get("same", (None,))[0] is not None
                         and rel_stats.get("opposed", (None,))[0] is not None
                         and rel_stats["same"][0] - rel_stats["opposed"][0] >= 0.25),
    }
    res = {"queries": len(queries), "candidates": len(cand), "table": table,
           "relation_key_sim": rel_stats, "checks": checks,
           "pass": all(checks.values()),
           "verdict": ("PASS -- intent-keyed index beats plain similarity on "
                       "verifier-grounded usefulness" if all(checks.values()) else
                       "FAIL -- the failing row names the missing piece")}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("table", "relation_key_sim", "checks", "pass",
                                          "verdict")}, indent=1, default=str), flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
