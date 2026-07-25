import logging, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers

DB = "data/pod_runs/phase2a/ingest_db_dialogsum_backfilled"
store = HippocampalStore(DB)
pipe = OracleLabelingPipeline(store)
centers = sample_episode_centers(store, n=5)
print(f"5 sampled centers: {centers}", flush=True)
for r in (1, 2, 3):
    sizes = []; node_sets = []
    for c in centers:
        sg = pipe.extract_subgraph(c, radius=r)
        sizes.append(len(sg["nodes"]))
        node_sets.append(frozenset(n["id"] for n in sg["nodes"]))
    avg = sum(sizes)/len(sizes)
    # pairwise Jaccard overlap across the 5 centers
    import itertools
    js = []
    for a, b in itertools.combinations(node_sets, 2):
        u = len(a|b); js.append(len(a&b)/u if u else 1.0)
    mean_j = sum(js)/len(js) if js else 1.0
    print(f"radius={r}: node_counts={sizes} avg={avg:.0f} mean_pairwise_jaccard={mean_j:.3f}", flush=True)
store.close()
