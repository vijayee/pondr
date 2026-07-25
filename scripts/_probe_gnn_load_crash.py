"""Crash probe for the Phase 3a GPU training segfault (run-1 + run-2 both died
silently at step 105 / ~23.5 min, no traceback -> C-level segfault, likely in a
WaveDB graph walk). Mirrors src/gnn/train.py:_build_inputs for head='all' but
with NO model / NO GPU: for each train center it does the clean load
(loader.load) + the anomaly reproduction path (extract_subgraph -> enrich ->
inject -> data_from_subgraph). Centers are visited in a FIXED sorted order,
repeated, with a per-build log line, capped at CAP builds.

Read /root/probe.log after exit:
- crash at build <= len(centers)  -> data-dependent: the last-logged center
  segfaults in isolation -> skippable -> 20 epochs reachable by excluding it.
- crash at build ~105-106          -> cumulative: heap corruption after ~105
  walks regardless of center -> WaveDB C bug, not skippable without a fix.
- no crash by the cap              -> the load path is clean; the segfault is
  in the model/GPU/loss path (revisit _forward / _losses_from_outputs).

Throwaway diagnostic -- NOT committed. Run on the pod:
  cd /root/hippo && python -u scripts/_probe_gnn_load_crash.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/hippo")  # noqa: E402  -- pod path

from src.gnn.anomaly_injector import inject_anomalies  # noqa: E402
from src.gnn.anomaly_rules import enrich_subgraph  # noqa: E402
from src.gnn.features import training_feature_for  # noqa: E402
from src.gnn.graph_loader import WaveDBGraphLoader, data_from_subgraph  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402
from src.training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers  # noqa: E402

DB = "/root/data/db/ingest_db_dialogsum_backfilled"
LABELS = Path("/root/data/labels")
RADIUS = 3
PRED_VOCAB = 32
CAP = 210  # ~10 cycles of 21 centers; well past the 105 crash point


def _load_jsonl_sids(stem: str) -> set[str]:
    p = LABELS / f"{stem}_labels.jsonl"
    sids: set[str] = set()
    if not p.exists():
        return sids
    for raw in open(p, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("subgraph_id"):
            sids.add(rec["subgraph_id"])
    return sids


def _load_anom() -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = LABELS / "anomaly_labels.jsonl"
    if not p.exists():
        return out
    for raw in open(p, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = rec.get("subgraph_id")
        if sid:
            out[sid] = rec.get("labels") or {}
    return out


def main() -> None:
    store = HippocampalStore(DB)
    try:
        pipe = OracleLabelingPipeline(store)
        loader = WaveDBGraphLoader(store, radius=RADIUS, predicate_vocab_size=PRED_VOCAB)
        feat_for = training_feature_for(store)
        anom = _load_anom()

        # Faithful center set: mirror train_gnn's `all`-mode union of every label
        # file's subgraph_ids, intersected with episodes that actually exist in
        # this store (sample_episode_centers).
        label_sids: set[str] = set()
        for stem in ("salience", "link_prediction", "ontology", "cluster", "anomaly"):
            label_sids |= _load_jsonl_sids(stem)
        valid = set(sample_episode_centers(store))
        centers = sorted(c for c in label_sids if c in valid)
        n_anom = sum(1 for c in centers if c in anom)
        print(f"probe: {len(centers)} centers, radius={RADIUS}, cap={CAP}, "
              f"anomaly-labeled={n_anom}", flush=True)

        t0 = time.time()
        n = 0
        while n < CAP:
            for sid in centers:  # fixed sorted order, repeated each cycle
                n += 1
                if n > CAP:
                    break
                print(f"[build {n:>3}] center={sid} anom={'yes' if sid in anom else 'no '} "
                      f"t={time.time() - t0:.0f}s", flush=True)
                # The common WaveDB walk (clean heads).
                loader.load(sid, radius=RADIUS)
                # The anomaly-reproduction walk (extract_subgraph is a 2nd WaveDB
                # BFS; inject/data_from_subgraph are pure-Python + feature reads).
                if sid in anom:
                    lbl = anom[sid]
                    sub = pipe.extract_subgraph(sid, radius=RADIUS)
                    enriched = enrich_subgraph(store, copy.deepcopy(sub))
                    corrupted, _ = inject_anomalies(
                        enriched, seed=lbl.get("seed", 0), types=lbl.get("types"),
                    )
                    data_from_subgraph(corrupted, feat_for, predicate_vocab_size=PRED_VOCAB)
        print(f"probe: reached cap {CAP} with NO crash in {time.time() - t0:.0f}s", flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()