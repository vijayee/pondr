#!/usr/bin/env python
"""End-to-end dogfood: real store -> real provenance -> real BonsaiDecider (v6
LoRA on :8080) -> real Consolidator._apply dispatcher -> real fact-level
tombstone. Scratch probe (NOT committed).

Proves the three shipped pieces compose on the consolidation path:
  (1) deterministic non-conflict guards (BonsaiDecider.decide_contradiction,
      runs BEFORE HTTP) -- the complementary-temporal guard returns ask_user
      (NON-mutating) so a month-named snapshot pair is NOT false-tombstoned.
  (2) v6 LoRA adapter (served via llama-server --lora bonsai_lora_F32.gguf)
      adjudicates the REAL conflict verbatim -> fix + supersede_assertion.
  (3) provenance enrichment (_gather_entity_context resolves asserted_by doc id
      -> source_path via store.document_source_path) -- what makes the guard
      production-sound (sees the month-named filename, not just the doc id).

Two seeded shapes:
  REAL conflict      E:db  MySQL  (docs/db-pick-v1.md, 2026-07-01)
                      -> Postgres (docs/db-pick-v2.md, 2026-07-05)
                      version-suffixed decision docs, doc_kind=decision_update,
                      NO month prefix -> bypass both guards -> LoRA adjudicates
                      -> fix + supersede -> MySQL tombstoned, Postgres live.
  COMPLEMENTARY      E:dep green (docs/q1-status.md, 2026-07-01)
  temporal            -> red   (docs/q2-status.md, 2026-07-05)
                      NON-month-named paths, doc_kind=point_in_time_snapshot on
                      both -> Sec 7.11 semantic guard fires on doc_kind ALONE
                      (no month prefix -- the production-real case the filename
                      guard was inert on) -> ask_user (no_action) -> NO
                      tombstone, both still live.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

# model reasoning may carry non-ASCII; stdout must not crash on cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# repo root on sys.path so `import src.*` works from scripts/_scratch/
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import ConsolidationConfig  # noqa: E402
from src.gnn import Consolidator  # noqa: E402
from src.gnn.bonsai_decider import BonsaiDecider  # noqa: E402
from src.memory.edge_meta import default_meta, update_edge_meta  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402


def plant_doc(store, doc_id, source_path, doc_kind):
    store.db.put_sync(f"content/doc/{doc_id}/source_type", "text")
    store.db.put_sync(f"content/doc/{doc_id}/source_path", source_path)
    # Phase 3c Sec 7.11: semantic doc-kind tag (zero-shot at ingest).
    store.db.put_sync(f"content/doc/{doc_id}/doc_kind", doc_kind)


def plant_assertion(store, entity_name, value, asserted_by, asserted_at):
    subj = f"E:{entity_name}"
    ops = store.graph.expand_triple(subj, "state", value)
    store.db.batch_sync(ops)
    meta = default_meta()
    meta["state"] = "current"
    meta["asserted_by"] = asserted_by
    meta["asserted_at"] = asserted_at
    update_edge_meta(store, subj, "state", value, meta)


def make_report(nodes):
    rep = {
        "dry_run": False, "trained": True, "subgraphs_scored": 1,
        "abstracts": [], "edges_proposed": [], "edges_accepted": [],
        "edges_unverified": [],
        "anomalies": [{"node": n, "type": "contradictory_state",
                       "score": 0.95, "evidence": "distinct states"}
                      for n in nodes],
        "ontology_proposed": [], "pruned": [],
        "verifier_calls": 0, "verifier_accepted": 0,
        "abstracts_applied": [], "ontology_applied": [],
        "ontology_rejected": [], "identity_drift_decisions": [],
        "contradictions_resolved": [],
        "score_distributions": {"ontology": [0] * 100, "linkpred": [0] * 100,
                                "salience_endpoint": [0] * 100},
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": [], "ontology_deprecated": [],
                       "deep_archived": []},
    }
    return rep


def live_values(cons, entity):
    g = cons.store.graph
    r = g.query().vertex(entity).out("state").execute_sync()
    try:
        out = []
        for v in r.vertices:
            if cons.store.is_edge_current(entity, "state", v):
                out.append(v)
        return sorted(out)
    finally:
        r.close()


def main():
    tmp = tempfile.mkdtemp(prefix="pondr_dogfood_")
    store = HippocampalStore(str(Path(tmp) / "db"))

    # --- seed the two shapes with real doc provenance + doc_kind ---
    plant_doc(store, "doc_000001", "docs/db-pick-v1.md", "decision_update")
    plant_doc(store, "doc_000002", "docs/db-pick-v2.md", "decision_update")
    plant_doc(store, "doc_000003", "docs/q1-status.md", "point_in_time_snapshot")
    plant_doc(store, "doc_000004", "docs/q2-status.md", "point_in_time_snapshot")

    # REAL conflict: MySQL -> Postgres, version-suffixed decision docs.
    plant_assertion(store, "db", "MySQL", "doc_000001_sec_003",
                    "2026-07-01T10:00:00Z")
    plant_assertion(store, "db", "Postgres", "doc_000002_sec_005",
                    "2026-07-05T10:00:00Z")
    # COMPLEMENTARY temporal: green -> red, month-named status docs.
    plant_assertion(store, "dep", "green", "doc_000003_sec_002",
                    "2026-07-01T10:00:00Z")
    plant_assertion(store, "dep", "red", "doc_000004_sec_004",
                    "2026-07-05T10:00:00Z")

    dec = BonsaiDecider()
    print(f"[dogfood] Bonsai decider -> {dec.endpoint} model={dec.model}")
    print(f"[dogfood] server health: {dec.health_check(timeout=5.0)}")

    cons = Consolidator(
        store, dry_run=False, allow_untrained_apply=True, decider=dec,
        config=ConsolidationConfig(contradiction_resolve_threshold=0.0,
                                   bonsai_decider_enabled=True),
    )
    cons._forget_updates = []
    cons._forget_node_salience = {}

    print("\n=== BEFORE: live state values ===")
    print(f"  E:db  live = {live_values(cons, 'E:db')}")
    print(f"  E:dep live = {live_values(cons, 'E:dep')}")

    # show the real gathered context (provenance + resolved source_path)
    for ent in ("E:db", "E:dep"):
        ctx = cons._gather_entity_context(ent)
        print(f"\n[gather] {ent} state_values:")
        for sv in ctx["state_values"]:
            print(f"    value={sv['value']!r} asserted_by={sv['asserted_by']!r} "
                  f"source_path={sv['source_path']!r} "
                  f"doc_kind={sv.get('doc_kind')!r}")

    print("\n=== PASS 1: adjudicate through real _apply (v6 LoRA server) ===")
    rep = make_report(["E:db", "E:dep"])
    cons._apply(rep)

    for rec in rep["contradictions_resolved"]:
        print(f"\n  entity={rec['entity']}")
        print(f"    old={rec['old_value']!r} new={rec['new_value']!r}")
        print(f"    decision={rec['decision']!r} action={rec['action']!r} "
              f"applied={rec['applied']}")
        why = (rec.get("reasoning") or "").replace("\n", " ")
        print(f"    reasoning={why[:240]}")

    print("\n=== AFTER PASS 1: live state values ===")
    print(f"  E:db  live = {live_values(cons, 'E:db')}   "
          f"(expect MySQL GONE, Postgres kept)")
    print(f"  E:dep live = {live_values(cons, 'E:dep')}   "
          f"(expect BOTH kept -- no false tombstone)")

    print("\n=== PASS 2: re-gather (detector quiet check) ===")
    for ent in ("E:db", "E:dep"):
        ctx = cons._gather_entity_context(ent)
        vals = [sv["value"] for sv in ctx["state_values"]]
        distinct = len(set(vals))
        print(f"  {ent}: live values={vals} distinct={distinct} "
              f"-> detector {'QUIET' if distinct < 2 else 'WOULD RE-FIRE'}")

    store.close()
    print("\n[dogfood] done.")


if __name__ == "__main__":
    main()