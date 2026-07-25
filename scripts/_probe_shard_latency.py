"""Measure one real salience-shard Oracle call latency (reasoning model)."""
import logging, sys, time, json
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleLabelingPipeline, OracleClient, OracleConfig
from src.retrieval.graph_traversal import GraphTraversal
from src.gnn.sharded_labeling import shard_nodes, build_salience_shard_prompt
from src.config import Config as HippoConfig

DB = "data/pod_runs/phase2a/ingest_db_dialogsum_backfilled"
cfg = HippoConfig()
oc = OracleConfig(model=cfg.oracle_model, endpoint=cfg.oracle_endpoint,
                  temperature=cfg.oracle_temperature, max_tokens=cfg.oracle_max_tokens,
                  max_retries=1, timeout=600.0)
client = OracleClient(oc)

store = HippocampalStore(DB)
try:
    pipe = OracleLabelingPipeline(store)
    trav = GraphTraversal(store)
    centers = __import__("src.training.oracle_labeling", fromlist=["sample_episode_centers"]).sample_episode_centers(store, n=1)
    sg = pipe.extract_subgraph(centers[0], radius=3)
    # hydrate episodes (mirrors generator)
    for node in sg["nodes"]:
        if node.get("type") == "episode":
            hy = trav.hydrate_episode(node["id"])
            node["summary"] = hy.get("summary","")
            node["entities"] = hy.get("entities",[])
            node["topics"] = hy.get("topics",[])
            node["tones"] = hy.get("tones",[])
            node["decisions"] = hy.get("decisions",[])
            node["timestamp"] = hy.get("timestamp","")
    shards = shard_nodes(sg, shard_size=500)
    print(f"subgraph nodes={len(sg['nodes'])} -> {len(shards)} salience shards", flush=True)
    s0 = shards[0]
    prompt = build_salience_shard_prompt(s0)
    print(f"shard0: nodes_in_shard={len(s0['nodes'])} prompt_chars={len(prompt)}", flush=True)
    t0 = time.time()
    res = client.generate(prompt)
    elapsed = time.time() - t0
    n_scores = 0
    try:
        n_scores = len((res.response or {}).get("node_scores", {}))
    except Exception:
        pass
    print(f"RESULT: elapsed={elapsed:.1f}s cached={res.cached} input_tok={res.input_tokens} output_tok={res.output_tokens} n_scores={n_scores} cost=${res.cost:.4f}", flush=True)
    if res.response is None:
        print("PARSE FAILED; raw content head:", flush=True)
        print(str(res.raw_content)[:500], flush=True)
finally:
    store.close()
