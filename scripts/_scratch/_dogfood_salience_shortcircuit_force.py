"""Live dogfood: FORCE the STRM Phase 5 in-flight short-circuit to fire on the
REAL ring + REAL heads + REAL DistillWorker in-flight map. Scratch (NOT committed).

WHY THIS SCRIPT EXISTS
  The natural-run dogfood (_dogfood_salience_inflight.py) confirmed (a) the
  real persist path populates the real in-flight map and (b) armed salience runs
  live without breakage, but (c) the short-circuit window did NOT fire: r_i did
  not clear the live p70 ``phi`` on any fresh-conversation slot, so no anchor
  was salient, so the per-anchor retrieval loop (where the short-circuit lives)
  never ran. The short-circuit is a narrow-window optimization (an episode
  retrieved into the ring in a prior turn that is STILL being distilled), and the
  ~22.8 s fill completes in the gaps between ~44 s turns, so E1 drops out of
  flight before the natural multi-turn chain reaches it.

WHAT THIS SCRIPT DOES (controlled, honest)
  1. Real T1 query -> real persist -> E1 in the REAL in-flight map (fill
     pending). This is the real enqueue path shipped in aab2cf9 -- not a stub.
  2. Hold ``foreground_busy`` so E1's fill stays pending (E1 stays in-flight).
  3. Inject E1's gist into the ring as a retrieved-episode anchor with
     ``source_id = ep_000001`` -- the ONE artificial step, simulating the
     prior-turn vector retrieval that would naturally inject a still-distilling
     episode as a ring anchor. Everything below is the real short-circuit path.
  4. Lower the salience thresholds to permissive so the injected E1 anchor IS
     salient (the live p70 phi is a real gate; lowering it for the dogfood is
     the minimal honest nudge to exercise the short-circuit on the real ring).
  5. Call the REAL ``_run_salience_hook`` directly (no second 8B call, no new
     conversation slot) on the real ring, twice:
       Run A: shortcut ON (default)  -> E1 served from the snapshot, NO
              ``retrieve_by_embedding`` call for E1's doc_emb -> ``recall``.
       Run B: shortcut OFF (rollback) -> E1 goes through ``retrieve_by_embedding``
              (byte-identical to pre-Phase-5) -> ``recall`` via the vector hit.
  6. The attributable proof: wrap ``retrieve_by_embedding`` to record the
     ``id(doc_emb)`` of every call. The E1 anchor's slot embedding id must NOT
     appear in Run A's call list (short-circuit skipped it) and MUST appear in
     Run B's (the vector fall-through). The differential is the live proof that
     the short-circuit fires with a zero vector round-trip on the real ring.

Run: PYTHONPATH=. python scripts/_scratch/_dogfood_salience_shortcircuit_force.py
(Pre-warm the Bonsai 8B server on :8080 first -- build_ponder connects to it and
the T1 persist path runs after a real response.)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# gliner2 prints a brain emoji at from_pretrained; on Windows cp1252 stdout that
# raises UnicodeEncodeError. Force utf-8 before any import that loads gliner.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import config
from src.runtime import build_ponder
from src.subconscious.salience import SalienceThresholds

DB = "data/dogfood_salience_db3"  # fresh DB so ring state = only these turns
REC_CKPT = "data/training/strm_recoverability/best.pt"
LD_CKPT = "data/training/strm_latent_dynamics/best.pt"
REL_CKPT = "data/training/strm_relevance/best.pt"
THR = "data/training/strm_salience/thresholds.json"
RING = 16

# T1 persists E1 (a Postgres/Redis decision). Its gist is then injected as a
# retrieved-episode ring anchor with source_id = E1's episode_id -- the scenario
# the short-circuit is built for (an episode retrieved into the ring that is
# STILL being distilled by Thread 2).
Q1 = ("We just switched the project's database to Postgres for persistence and "
      "the cache to Redis. Is that a solid choice for the project?")
# A prompt semantically close to E1 so the relevance head (r_i) scores the E1
# anchor highly -- with permissive thresholds this makes E1 salient.
Q_PROBE = "What did we decide for persistence and caching?"


def _permissive_thresholds() -> SalienceThresholds:
    """Every scored anchor is salient: rec_i < theta (theta huge), r_i > phi
    (phi very negative), surprise_i < surprise_cap (cap huge). This is the
    minimal honest nudge to exercise the short-circuit on the real ring -- the
    live p70 phi is a real gate that r_i did not clear in the natural run."""
    return SalienceThresholds(
        theta=1e9, phi=-1e9, surprise_cap=1e9,
        theta_percentile=0.0, phi_percentile=0.0, surprise_cap_percentile=0.0,
        basis="dogfood-permissive-forced-shortcircuit",
    )


print(f"[dogfood] async_distill={config.async_distill_enabled} "
      f"inflight_shortcut={config.strm_salience_inflight_shortcut}", file=sys.stderr)
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

worker = orch._distill_worker
store = orch.store
assert worker is not None, "async-distill worker not wired -- cannot dogfood the short-circuit"
assert orch._salience_armed, "salience not armed -- cannot dogfood the short-circuit"

# Lower the live thresholds to permissive so the injected E1 anchor is salient.
orch.salience_thresholds = _permissive_thresholds()
print(f"[dogfood] thresholds -> permissive (forced short-circuit window)",
      file=sys.stderr)

# ── T1: real query -> real persist -> E1 in the real in-flight map ─────────
res1 = orch.query(Q1)
e1 = res1.get("persisted_episode_id")
snap_e1 = worker.snapshot_if_inflight(e1) if e1 else None
print(f"\n[T1] persisted={e1} in-flight snapshot "
      f"={'PRESENT (real enqueue -> real map)' if snap_e1 else 'absent'}",
      file=sys.stderr)
if snap_e1:
    print(f"[T1]   snap keys={sorted(snap_e1.keys())} "
          f"summary={snap_e1.get('summary','')[:60]!r}", file=sys.stderr)
assert snap_e1 is not None, "T1 did not leave E1 in-flight -- precondition failed"

# Hold E1's fill pending so it stays in-flight across the direct hook calls.
worker.foreground_busy.set()
print(f"[dogfood] foreground_busy SET -- E1 fill held pending (stays in-flight)",
      file=sys.stderr)

# Inject E1's gist into the ring as a retrieved-episode anchor (source_id = E1).
# This is the ONE artificial step -- it stands in for the prior-turn vector
# retrieval that would naturally inject a still-distilling episode as a ring
# anchor. The hook + short-circuit + heads below are all real.
gist = snap_e1.get("embed_text") or snap_e1.get("summary") or snap_e1.get("text") or ""
e1_emb = orch.working_memory.embed([gist])[0]
orch.working_memory.inject(e1_emb, source_id=e1, text=snap_e1.get("summary") or snap_e1.get("text") or "", pin=True)
print(f"[dogfood] injected E1 anchor into ring: source_id={e1} "
      f"text={(snap_e1.get('summary') or '')[:50]!r}", file=sys.stderr)

# The attributable handle: the E1 anchor's doc_emb is the re-embedded E1 gist
# (compute_salience re-embeds slot.text -> 384-d, NOT the slot's 256-d ``y``; see
# salience.py:218-221). A deterministic encoder re-embeds identical text to a
# numerically-identical vector, so we identify the E1 anchor's vector round-trip
# by ``torch.allclose(doc_emb, e1_reemb)``. The short-circuit must keep such a
# call OUT of Run A and put it IN Run B.
e1_reemb = orch.working_memory.embed([gist])[0]

# Build the hook inputs exactly as query() does (no new conversation slot added
# because we call the hook directly, not query()).
probe_emb = orch.working_memory.embed([Q_PROBE])[0]
prev_state_tensors = [t.clone() for t in orch.working_memory.state_tensors()]


def _is_e1(doc_emb) -> bool:
    if doc_emb is None:
        return False
    try:
        a = doc_emb.detach().to(torch.float32).cpu().reshape(-1)
        b = e1_reemb.detach().to(torch.float32).cpu().reshape(-1)
        return a.shape == b.shape and bool(torch.allclose(a, b, atol=1e-5))
    except Exception:
        return False


def _wire_rbe():
    """Wrap retrieve_by_embedding to record each call's doc_emb (kept as a
    detached CPU clone so it survives after the real call returns). Returns
    (real, calls) so the caller can restore + inspect."""
    real = orch.retriever.retrieve_by_embedding
    calls: list = []

    def _wrapped(doc_emb, *a, **k):
        calls.append(doc_emb.detach().clone() if doc_emb is not None else None)
        return real(doc_emb, *a, **k)

    orch.retriever.retrieve_by_embedding = _wrapped
    return real, calls


def _run_hook(label):
    """Reset the per-turn stashes, run the real hook, read the signals."""
    orch._salience_anchors = None
    orch._salience_fired_episodes = None
    orch._salience_signals = None
    real, calls = _wire_rbe()
    try:
        orch._run_salience_hook(probe_emb, prev_state_tensors, "routine")
        sigs = orch._salience_signals or []
        fired = orch._salience_fired_episodes or []
    finally:
        orch.retriever.retrieve_by_embedding = real
    return sigs, fired, calls


# ── Run A: shortcut ON (default) ────────────────────────────────────────────
import src.orchestrator as orch_mod  # noqa: E402
orch_mod._runtime_config.strm_salience_inflight_shortcut = True
sigsA, firedA, callsA = _run_hook("A")
e1_sigA = next((s for s in sigsA if s.get("anchor_source_id") == e1), None)
e1_in_callsA = any(_is_e1(c) for c in callsA)
print(f"\n[Run A] shortcut=ON  rbe_calls={len(callsA)} "
      f"E1 gist doc_emb passed to rbe? {e1_in_callsA}", file=sys.stderr)
print(f"[Run A]   signals: {[(s.get('anchor_source_id'), s.get('kind')) for s in sigsA]}",
      file=sys.stderr)
print(f"[Run A]   E1 signal kind = {e1_sigA.get('kind') if e1_sigA else None} "
      f"(expect recall)", file=sys.stderr)

# ── Run B: shortcut OFF (rollback -> byte-identical vector path) ─────────────
orch_mod._runtime_config.strm_salience_inflight_shortcut = False
sigsB, firedB, callsB = _run_hook("B")
e1_sigB = next((s for s in sigsB if s.get("anchor_source_id") == e1), None)
e1_in_callsB = any(_is_e1(c) for c in callsB)
print(f"\n[Run B] shortcut=OFF rbe_calls={len(callsB)} "
      f"E1 gist doc_emb passed to rbe? {e1_in_callsB}", file=sys.stderr)
print(f"[Run B]   signals: {[(s.get('anchor_source_id'), s.get('kind')) for s in sigsB]}",
      file=sys.stderr)
print(f"[Run B]   E1 signal kind = {e1_sigB.get('kind') if e1_sigB else None} "
      f"(expect recall via vector)", file=sys.stderr)

# Restore the default before teardown (leave the process flag as shipped).
orch_mod._runtime_config.strm_salience_inflight_shortcut = True

# ── Verdict ─────────────────────────────────────────────────────────────────
print("\n[verdict]", file=sys.stderr)
short_circuit_fired = (
    e1_sigA is not None and e1_sigA.get("kind") == "recall"
    and not e1_in_callsA           # Run A: E1 served from snapshot, NO vector call
    and e1_in_callsB               # Run B: E1 went through the vector path
    and e1_sigB is not None and e1_sigB.get("kind") == "recall"
    and len(callsB) == len(callsA) + 1  # exactly the E1 vector round-trip added
)
print(f"[verdict] Run A (ON):  E1 short-circuited (no rbe call for E1)? "
      f"{not e1_in_callsA}", file=sys.stderr)
print(f"[verdict] Run B (OFF): E1 went through retrieve_by_embedding?      "
      f"{e1_in_callsB}", file=sys.stderr)
print(f"[verdict] rbe_calls A={len(callsA)} B={len(callsB)} "
      f"(diff={len(callsB)-len(callsA)}; expect +1 = E1 vector round-trip)",
      file=sys.stderr)
print(f"[verdict] E1 kind A={e1_sigA.get('kind') if e1_sigA else None} "
      f"B={e1_sigB.get('kind') if e1_sigB else None} (both recall)",
      file=sys.stderr)
print(f"[verdict] *** SHORT-CIRCUIT FIRED ON THE REAL RING WITH ZERO VECTOR "
      f"ROUND-TRIP? {short_circuit_fired} ***", file=sys.stderr)

# ── Cleanup: release the worker, drain, confirm snapshot cleared ────────────
worker.foreground_busy.clear()
ok = orch.drain(timeout=60.0)
after = worker.snapshot_if_inflight(e1)
print(f"\n[drain] joined={ok} snapshot_if_inflight({e1}) after drain "
      f"={after} (None expected -- episode no longer in flight)", file=sys.stderr)
store.close()
print("[dogfood] done.", file=sys.stderr)

if not short_circuit_fired:
    sys.exit(1)