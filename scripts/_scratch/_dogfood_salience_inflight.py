"""Live dogfood: STRM Phase 5 IngestionTracker in-flight short-circuit on the
REAL ring (armed --strm-salience + real DistillWorker + real heads + real
salience thresholds). NOT committed (scratch probe).

Confirms in the LIVE system (not offline stubs):

  (a) REAL persist -> REAL in-flight map. The real ``_persist_exchange`` path
      calls the real ``DistillWorker.enqueue`` (the code shipped in aab2cf9),
      which snapshots the stub into the in-flight map. Right after query()
      returns, ``snapshot_if_inflight(E1)`` returns the snapshot WHILE the
      ~22.8 s fill is still pending -- the cheap-read precondition holds on the
      live path. (The offline tests prove the short-circuit fires GIVEN an
      in-flight snapshot; this proves the live persist path actually CREATES
      that snapshot.)
  (b) Armed salience runs live without breakage. ``--strm-salience`` armed +
      ring on + 3 heads + thresholds -> the hook runs, salience_signals /
      salience_retrieval_count / salience_gap_text are present in the result
      (absent when off). The response is byte-identical in shape to flag-off.
  (c) Short-circuit vs vector round-trip, observed. ``retrieve_by_embedding``
      is wrapped to count calls per turn. When a still-distilling episode is a
      salience anchor the hook serves it from the snapshot (zero vector round-
      trip); otherwise it falls through. We report honestly which path each
      turn took -- the short-circuit is a narrow-window optimization (an
      episode retrieved into the ring in a prior turn that is STILL being
      distilled), so it may or may not fire in a 3-turn dogfood.

Run: python scripts/_scratch/_dogfood_salience_inflight.py
(Pre-warm the Bonsai 8B server on :8080 first -- build_ponder connects to it
and the persist path runs after a real response.)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# gliner2 prints a 🧠 emoji at from_pretrained; on Windows cp1252 stdout that
# raises UnicodeEncodeError. Force utf-8 before any import that loads gliner.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.runtime import build_ponder

DB = "data/dogfood_salience_db"  # fresh DB so ring state = only these turns
REC_CKPT = "data/training/strm_recoverability/best.pt"
LD_CKPT = "data/training/strm_latent_dynamics/best.pt"
REL_CKPT = "data/training/strm_relevance/best.pt"
THR = "data/training/strm_salience/thresholds.json"
RING = 16

# Three turns designed to hit the in-flight window: turn 1 persists E1 (a
# Postgres/Redis decision); turn 2 + 3 reference the same topic within E1's
# ~22.8 s fill window so E1 is likely retrieved into the ring at turn 2 and
# re-scored (still in-flight) at turn 3 -> the short-circuit window.
Q1 = ("We just switched the project's database to Postgres for persistence and "
      "the cache to Redis. Is that a solid choice for the project?")
Q2 = ("Given that database and cache choice, what should we watch out for in "
     "the audit log?")
Q3 = ("Remind me again -- what did we decide for persistence and caching, and "
     "is it still holding up?")


def _arm(orch):
    """Instrument retrieve_by_embedding to count calls per turn (the short-
    circuit skips it; the vector fall-through calls it). Returns a mutable
    counter dict the caller resets per turn."""
    real = orch.retriever.retrieve_by_embedding
    state = {"calls": 0}

    def _wrapped(*a, **k):
        state["calls"] += 1
        return real(*a, **k)

    orch.retriever.retrieve_by_embedding = _wrapped
    return state


print(f"[dogfood] async_distill={config.async_distill_enabled} "
      f"inflight_shortcut={config.strm_salience_inflight_shortcut} "
      f"freshness_lag={config.strm_salience_freshness_lag}", file=sys.stderr)
print(f"[dogfood] db={DB} ring={RING} salience=ARMED", file=sys.stderr)

t_build = time.monotonic()
orch = build_ponder(
    db_path=DB,
    gliner_device="auto",
    strm_salience=True,
    salience_thresholds_path=THR,
    relevance_head_path=REL_CKPT,
    recoverability_head_path=REC_CKPT,
    latent_dynamics_head_path=LD_CKPT,
    ring_capacity=RING,
)
print(f"[dogfood] build_ponder took {time.monotonic()-t_build:.1f}s "
      f"(salience armed={getattr(orch, '_salience_armed', False)})", file=sys.stderr)

store = orch.store
rbe_state = _arm(orch)
worker = orch._distill_worker
print(f"[dogfood] distill_worker wired={worker is not None}", file=sys.stderr)


def _run_turn(label, q):
    rbe_state["calls"] = 0
    t0 = time.monotonic()
    res = orch.query(q)
    lat = time.monotonic() - t0
    eid = res.get("persisted_episode_id")
    sigs = res.get("salience_signals") or []
    armed = "salience_signals" in res
    # (a) the real persist path -> real in-flight map, while the fill is pending.
    snap = worker.snapshot_if_inflight(eid) if (worker is not None and eid) else None
    print(f"\n[{label}] latency={lat:.2f}s persisted={eid} armed={armed} "
          f"rbe_calls={rbe_state['calls']} sigs={len(sigs)}", file=sys.stderr)
    for s in sigs:
        print(f"[{label}]   sig anchor={s.get('anchor_source_id')} "
              f"kind={s.get('kind')} r_i={s.get('r_i')} rec_i={s.get('rec_i')} "
              f"age={s.get('age')}", file=sys.stderr)
    print(f"[{label}] in-flight snapshot of JUST-persisted {eid}: "
          f"{'PRESENT (real enqueue -> real map)' if snap else 'absent'}",
          file=sys.stderr)
    if snap:
        print(f"[{label}]   snap keys={sorted(snap.keys())} "
              f"summary={snap.get('summary','')[:60]!r}", file=sys.stderr)
    return res, eid, sigs


res1, e1, sigs1 = _run_turn("T1", Q1)

# While E1's fill is still pending, confirm it is in-flight (the precondition
# for the short-circuit). Then run T2 + T3 quickly, inside the fill window.
if worker is not None and e1:
    snap_e1 = worker.snapshot_if_inflight(e1)
    print(f"\n[pre-T2] E1={e1} in-flight (fill pending)? "
          f"{'YES -- short-circuit precondition holds' if snap_e1 else 'no (fill already done)'}",
          file=sys.stderr)

res2, e2, sigs2 = _run_turn("T2", Q2)
res3, e3, sigs3 = _run_turn("T3", Q3)

# Did the short-circuit fire? Look for a recall signal whose anchor source_id
# is an episode_id (ep_...) that was in-flight at that turn, with no
# retrieve_by_embedding call attributed to it. The cleanest tell: a turn where
# salience fired a recall (salience_retrieval_count > 0) AND rbe_calls stayed
# small / zero for the salience-fired anchors.
print("\n[verdict]", file=sys.stderr)
for label, res, sigs in (("T1", res1, sigs1), ("T2", res2, sigs2), ("T3", res3, sigs3)):
    rc = res.get("salience_retrieval_count")
    kinds = [s.get("kind") for s in sigs]
    print(f"[verdict] {label}: salience_retrieval_count={rc} kinds={kinds}",
          file=sys.stderr)

# (c) post-fill: once the worker drains, the in-flight snapshot is gone ->
# snapshot_if_inflight returns None (queue membership was ground truth).
print(f"\n[drain] waiting for fill(s) to finish + join...", file=sys.stderr)
ok = orch.drain(timeout=60.0)
if worker is not None and e1:
    after = worker.snapshot_if_inflight(e1)
    print(f"[drain] joined={ok} snapshot_if_inflight({e1}) after drain="
          f"{after} (None expected -- episode no longer in flight)",
          file=sys.stderr)
store.close()
print("[dogfood] done.", file=sys.stderr)