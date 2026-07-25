"""Probe two reasoning-control levers on the SAME salience shard:
  A) cap max_tokens=6000 via /v1 (starve the CoT)
  B) native /api/chat with think:false (disable reasoning)
Report latency + n_scores returned + truncation, vs the 104.6s/17.5K-tok/500-score baseline.
"""
import logging, sys, time, json, requests
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers
from src.retrieval.graph_traversal import GraphTraversal
from src.gnn.sharded_labeling import shard_nodes, build_salience_shard_prompt
from src.config import Config as HippoConfig

DB = "data/pod_runs/phase2a/ingest_db_dialogsum_backfilled"
cfg = HippoConfig()
MODEL = cfg.oracle_model
EP = cfg.oracle_endpoint  # http://localhost:11434/v1
NATIVE = EP.replace("/v1","") + "/api/chat"  # http://localhost:11434/api/chat

def build_shard0():
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
        shards = shard_nodes(sg, shard_size=500)
        return build_salience_shard_prompt(shards[0])
    finally:
        store.close()

def count_scores(content):
    try:
        d = json.loads(content)
        return len((d or {}).get("node_scores", {})), True
    except Exception:
        return 0, False

prompt = build_shard0()
print(f"prompt_chars={len(prompt)}  MODEL={MODEL}", flush=True)

# --- A: /v1 with max_tokens=6000 (cap CoT) ---
print("\n[A] /v1 max_tokens=6000 ...", flush=True)
payloadA = {"model": MODEL, "messages":[{"role":"user","content":prompt}],
            "response_format":{"type":"json_object"}, "temperature":0.1, "max_tokens":6000}
t0=time.time()
try:
    r = requests.post(EP+"/chat/completions", json=payloadA, timeout=600)
    el=time.time()-t0
    if r.status_code != 200:
        print(f"  A: HTTP {r.status_code} body={r.text[:300]}", flush=True)
    else:
        content = r.json()["choices"][0]["message"]["content"]
        usage = r.json().get("usage",{})
        n, ok = count_scores(content)
        print(f"  A: elapsed={el:.1f}s out_tok={usage.get('completion_tokens')} n_scores={n} parsed={ok} content_len={len(content)}", flush=True)
        if not ok: print("  A raw head:", content[:300], flush=True)
except Exception as e:
    print(f"  A: ERROR {type(e).__name__}: {e}", flush=True)

# --- B: /api/chat with think:false ---
print("\n[B] /api/chat think:false num_predict=32768 ...", flush=True)
payloadB = {"model": MODEL, "messages":[{"role":"user","content":prompt}],
            "format":"json", "stream": False, "think": False,
            "options":{"temperature":0.1, "num_predict":32768}}
t0=time.time()
try:
    r = requests.post(NATIVE, json=payloadB, timeout=600)
    el=time.time()-t0
    if r.status_code != 200:
        print(f"  B: HTTP {r.status_code} body={r.text[:300]}", flush=True)
    else:
        body = r.json()
        content = body.get("message",{}).get("content","")
        # native may separate thinking; count eval tokens
        evalc = body.get("eval_count")
        n, ok = count_scores(content)
        print(f"  B: elapsed={el:.1f}s eval_count={evalc} n_scores={n} parsed={ok} content_len={len(content)} think_present={'thinking' in body or body.get('message',{}).get('thinking')}", flush=True)
        if not ok: print("  B raw head:", content[:300], flush=True)
except Exception as e:
    print(f"  B: ERROR {type(e).__name__}: {e}", flush=True)
