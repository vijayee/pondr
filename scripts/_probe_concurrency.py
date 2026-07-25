"""Concurrency probe: 4 DIFFERENT trimmed salience shards via generate_batch(max_workers=4).
If total wall ~= 50s -> true concurrency; if ~= 200s (4x50) -> ollama.com serializes :cloud.
Also probes max_workers=8 on the same 4 shards (cache miss each since distinct)."""
import logging, sys, time
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleLabelingPipeline, OracleClient, OracleConfig, sample_episode_centers
from src.retrieval.graph_traversal import GraphTraversal
from src.gnn.sharded_labeling import shard_nodes, build_salience_shard_prompt
from src.config import Config as HippoConfig

DB = "data/pod_runs/phase2a/ingest_db_dialogsum_backfilled"
cfg = HippoConfig()
store = HippocampalStore(DB)
try:
    pipe = OracleLabelingPipeline(store); trav = GraphTraversal(store)
    sg = pipe.extract_subgraph(sample_episode_centers(store, n=1)[0], radius=3)
    for node in sg["nodes"]:
        if node.get("type") == "episode":
            hy = trav.hydrate_episode(node["id"])
            node["summary"]=hy.get("summary",""); node["entities"]=hy.get("entities",[])
            node["topics"]=hy.get("topics",[]); node["tones"]=hy.get("tones",[])
            node["decisions"]=hy.get("decisions",[]); node["timestamp"]=hy.get("timestamp","")
    shards = shard_nodes(sg, shard_size=500)[:4]
    prompts = [build_salience_shard_prompt(s) for s in shards]
    print(f"4 distinct shards, prompt_chars={[len(p) for p in prompts]}", flush=True)

    for w in (4, 8):
        oc = OracleConfig(model=cfg.oracle_model, endpoint=cfg.oracle_endpoint,
                          temperature=cfg.oracle_temperature, max_tokens=cfg.oracle_max_tokens,
                          max_retries=1, timeout=600.0)
        c = OracleClient(oc)  # fresh in-mem cache per run (distinct prompts -> no hits)
        t0 = time.time()
        res = c.generate_batch(prompts, max_workers=w)
        tot = time.time() - t0
        lats = [round(r.latency_seconds,1) for r in res]
        errs = [r.error is not None for r in res]
        nscore = [len((r.response or {}).get("node_scores",{})) for r in res]
        print(f"max_workers={w}: total_wall={tot:.1f}s per_call_latency={lats} errors={errs} n_scores={nscore}", flush=True)
finally:
    store.close()
