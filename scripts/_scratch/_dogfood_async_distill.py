"""Live dogfood: async-distill + 10-pass isolated extraction against the real 8B.

NOT committed (scratch probe). Verifies the three async-distill contracts in the
LIVE system (not offline stubs):

  (a) the response returns FAST -- the ~22.8 s extraction is backgrounded, so
      query() latency is just retrieval + Bonsai synthesis, NOT 22 s.
  (b) the stub is written synchronously -- persisted_episode_id is set + the
      episode is content-retrievable the instant query() returns.
  (c) the worker fills the graph edges in the background -- has_entity edges
      (GLiNER) + (E:entity, state, value) assertion edges (the isolated
      extractor's has_state relations lifted by the deterministic normalizer)
      appear AFTER the response, within the ~22.8 s fill window.

Run: python scripts/_scratch/_dogfood_async_distill.py
(Pre-warm the Bonsai 8B server on :8080 first.)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.runtime import build_ponder

DB = "data/dogfood_async_db"
QUERY = ("We just switched the project's database to Postgres for persistence "
         "and the cache to Redis. Is that a solid choice for the project?")

# Live-validation of the Phase 1c-3c flag flip (commit b16810e): do NOT set
# the flags explicitly -- rely on the DEFAULTS (async_distill_enabled + bonsai
# isolation + gliner_timing all default ON now). If build_ponder + the
# orchestrator pick up the default-ON path against the real 8B, the flag flip
# is live-validated. tests/test_defaults.py pins the defaults offline; this
# pins the ON-behavior live.
print(f"[dogfood] (testing DEFAULTS) async_distill={config.async_distill_enabled} "
      f"bonsai_isolation={config.bonsai_isolation_extraction} "
      f"gliner_timing={config.gliner_timing}", file=sys.stderr)
print(f"[dogfood] db={DB}", file=sys.stderr)

t_build = time.monotonic()
orch = build_ponder(db_path=DB, gliner_device="auto")
print(f"[dogfood] build_ponder took {time.monotonic()-t_build:.1f}s", file=sys.stderr)

store = orch.store

t0 = time.monotonic()
res = orch.query(QUERY)
t1 = time.monotonic()
latency = t1 - t0

eid = res.get("persisted_episode_id")
response = res.get("response") or ""
print(f"\n[dogfood] RESPONSE LATENCY: {latency:.2f}s  (the 22.8s extraction is "
      f"backgrounded -- this must be << 22s)", file=sys.stderr)
print(f"[dogfood] persisted_episode_id: {eid}", file=sys.stderr)
print(f"[dogfood] response: {response[:200]!r}", file=sys.stderr)

# (b) stub is content-retrievable immediately.
ep = store.get_episode(eid) if eid else None
print(f"[dogfood] stub content-retrievable right after query(): {ep is not None}",
      file=sys.stderr)
if ep:
    print(f"[dogfood]   origin={ep.origin} summary_embedding_set="
          f"{ep.summary_embedding is not None} entities_at_stub={ep.entities}",
          file=sys.stderr)

# (c) poll for the background fill: has_entity edges + (E:entity, state, value)
# assertion edges. The fill takes ~22.8 s (10 isolated Bonsai calls + GLiNER).
def _spo_keys():
    out = []
    for k, _ in store.db.create_read_stream(start="memory/spo/", end="memory/spo/\x7f"):
        out.append(k)
    return out

print(f"[dogfood] polling for background fill (has_entity + state edges)...",
      file=sys.stderr)
deadline = time.monotonic() + 50.0
has_entity_found = False
state_edges = []
filled_at = None
while time.monotonic() < deadline:
    keys = _spo_keys()
    he = [k for k in keys if "/has_entity/" in k and f"/{eid}" in k]
    # assertion edges: (E:entity, state, value) -> key memory/spo/E:.../state/...
    se = [k for k in keys if "/state/" in k and k.startswith("memory/spo/E:")]
    if he:
        has_entity_found = True
        state_edges = se
        filled_at = time.monotonic()
        break
    time.sleep(1.0)

if filled_at:
    fill_secs = filled_at - t1
    print(f"[dogfood] FILL detected {fill_secs:.1f}s after response returned "
          f"(~22.8s expected). has_entity edges present.", file=sys.stderr)
    print(f"[dogfood] has_entity sample: "
          f"{[k for k in _spo_keys() if '/has_entity/' in k and f'/{eid}' in k][:3]}",
          file=sys.stderr)
    print(f"[dogfood] state assertion edges (E:entity, state, value): "
          f"{len(state_edges)} found", file=sys.stderr)
    for k in state_edges[:6]:
        print(f"[dogfood]   {k}", file=sys.stderr)
else:
    print(f"[dogfood] !! FILL not detected within 50s -- has_entity_found="
          f"{has_entity_found}", file=sys.stderr)

# Re-read the episode after the fill to confirm the in-memory episode fields
# were populated (the worker mutates the episode object).
ep_after = store.get_episode(eid) if eid else None
if ep_after:
    print(f"[dogfood] episode after fill: entities={ep_after.entities} "
          f"topics={ep_after.topics}", file=sys.stderr)

print(f"[dogfood] draining worker (up to 60s)...", file=sys.stderr)
orch.drain(timeout=60.0)
store.close()
print(f"[dogfood] done.", file=sys.stderr)