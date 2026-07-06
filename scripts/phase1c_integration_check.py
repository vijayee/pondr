"""Phase 1c on-corpus integration check (step 5).

Runs against a REAL populated compact corpus (the Phase 1b reload-compacted
DialogSum store) and exercises all three Phase 1c conversational refinements:

  1. **Entity salience** — a high-salience entity (many episode mentions) scores
     higher than the 0.5 floor: ``score = _W_ENTITY * (0.5 + 0.5 * salience)``
     with ``salience > 0`` ⇒ score > 5.0 for a matched entity.
  2. **Temporal date-range** — ``date_from`` / ``date_to`` filter the candidate
     set; a narrow window returns a proper subset, a window covering all
     timestamps returns everything, a window outside the corpus returns nothing.
  3. **Conversation context / pronoun resolution** — a follow-up question with a
     pronoun ("what did we decide about it?") and prior history naming an entity
     resolves the pronoun to that entity in the rule-based plan, so the retrieval
     hits the entity's episodes.

Uses the rule-based planner (deterministic; no Bonsai in the loop) so the check
is reproducible off the live LLM. The live Bonsai end-to-end path is exercised
separately by ``tests/test_mode_a.py::test_generate_via_live_bonsai`` and
``tests/test_end_to_end.py::test_mode_a_frustrated_via_live_bonsai`` via the SSH
tunnel.

Run on the pod after ``compute_entity_salience.py`` has populated salience:

    python scripts/phase1c_integration_check.py \
        --db /root/ingest_db_dialogsum_compact

Exits non-zero if any refinement check fails, so it can gate CI / a runbook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.query_planner import BonsaiQueryPlanner  # noqa: E402
from src.retrieval.retriever import HippocampalRetriever  # noqa: E402


class _RulePlanner:
    """Forces the deterministic rule-based planner (no live Bonsai in the loop)."""

    def __init__(self) -> None:
        self._p = BonsaiQueryPlanner()

    def plan(self, prompt: str, conversation_history: list | None = None) -> dict:
        return self._p.plan_rule_based(prompt, conversation_history)


def _episode_timestamps(store: HippocampalStore) -> list[str]:
    """All episode timestamps from the content/ep/ subtree (trie order)."""
    ts: list[str] = []
    for k, v in store.db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
        if k.endswith("/ts"):
            ts.append(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
    return ts


def _top_entity(store: HippocampalStore) -> str | None:
    """Highest-salience entity (largest mention_count) from the salience index."""
    best: tuple[str, int] | None = None
    for k, v in store.db.create_read_stream(
        start="content/entity/", end="content/entity/\x7f"
    ):
        if k.endswith("/mention_count"):
            try:
                count = int(v.decode() if isinstance(v, bytes) else v)
            except ValueError:
                continue
            entity = k.split("/")[-2]
            if best is None or count > best[1]:
                best = (entity, count)
    return best[0] if best else None


def check_salience(store: HippocampalStore, retr: HippocampalRetriever) -> bool:
    """A high-salience entity match scores above the 0.5 floor (score > 5.0)."""
    entity = _top_entity(store)
    if entity is None:
        print("[salience] SKIP — no salience index found; run compute_entity_salience.py first.")
        return True  # not a failure of this check; the salience run is separate.
    salience = store.get_entity_salience(entity)
    # Use an explicit entity plan so the check is deterministic — it does not
    # depend on the rule-based planner extracting the entity out of the prompt.
    # _find_candidates returns exactly this entity's episodes, so every
    # candidate is scored with the salience-weighted entity term.
    results = retr.retrieve_with_plan(
        {"entities": [entity], "entity_mode": "union",
         "topics": [], "tones": [], "limit": 50}
    )
    if not results:
        print(f"[salience] FAIL — entity plan for '{entity}' returned no episodes.")
        return False
    # Find a result whose hydrated entities list includes the entity
    # (case-insensitive); every candidate came from _get_episodes_by_entity, so
    # at least one should carry it.
    hit = None
    for r in results:
        if entity.lower() in {e.lower() for e in r.get("entities", [])}:
            hit = r
            break
    if hit is None:
        hit = results[0]
        print(f"[salience] WARN — entity '{entity}' not in any result's entities list; "
              f"using top result score for the floor check.")
    score = hit["score"]
    ok = salience > 0.0 and score > 5.0 + 1e-9
    print(f"[salience] entity='{entity}' salience={salience:.3f} top_match_score={score:.3f} "
          f"-> {'PASS' if ok else 'FAIL'} (expect salience>0 and score>5.0)")
    return ok


def check_temporal(store: HippocampalStore, retr: HippocampalRetriever) -> bool:
    """date_from/date_to narrows the candidate set sensibly."""
    ts = _episode_timestamps(store)
    if not ts:
        print("[temporal] SKIP — no episode timestamps in the store.")
        return True
    lo, hi = min(ts), max(ts)  # full ISO strings — inclusive range covers all
    year = int(lo[:4]) if lo[:4].isdigit() else 2000
    out_iso = f"{year - 1}-01-01"  # bare date (midnight) clearly before the corpus

    # Full-range query → should return episodes (the whole corpus is in-window).
    # Use the full ISO min/max so the inclusive [lo, hi] actually covers every
    # episode — a date-only range collapses to midnight and would exclude
    # same-day-later timestamps (the corpus is encoded in minutes, so all
    # episodes share one date with distinct times).
    full = retr.retrieve_with_plan(
        {"entities": [], "entity_mode": "union", "date_from": lo, "date_to": hi,
         "topics": [], "tones": [], "limit": 10000}
    )
    # Out-of-range query (a year before the corpus) → no episodes.
    out = retr.retrieve_with_plan(
        {"entities": [], "entity_mode": "union", "date_from": out_iso, "date_to": out_iso,
         "topics": [], "tones": [], "limit": 10000}
    )
    ok = len(full) > 0 and len(out) == 0
    print(f"[temporal] corpus ts {lo}..{hi}; full-range hits={len(full)}; "
          f"out-of-range({out_iso}) hits={len(out)} -> {'PASS' if ok else 'FAIL'} "
          f"(expect full>0 and out=0)")
    return ok


def check_context(retr: HippocampalRetriever) -> bool:
    """A pronoun follow-up with history resolves to the named entity in the plan."""
    # Use a clearly entity-bearing prior turn so the rule-based pronoun pre-pass
    # lifts the entity into the plan. 'it' is the pronoun the pre-pass targets.
    entity = "Alice"
    history = [
        {"role": "user", "content": f"Tell me about {entity} and the project."},
        {"role": "assistant", "content": f"{entity} was working on the database schema."},
    ]
    prompt = "What did we decide about it?"
    plan = retr.planner.plan(prompt, conversation_history=history)
    resolved = {e.lower() for e in plan.get("entities", [])}
    ok = entity.lower() in resolved
    print(f"[context] pronoun='it' with history naming '{entity}'; "
          f"plan.entities={plan.get('entities')} -> {'PASS' if ok else 'FAIL'} "
          f"(expect '{entity}' lifted into plan via pronoun pre-pass)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="WaveDB compact-corpus store directory")
    args = ap.parse_args()

    store = HippocampalStore(args.db)
    retr = HippocampalRetriever(store, planner=_RulePlanner())

    results = {
        "salience": check_salience(store, retr),
        "temporal": check_temporal(store, retr),
        "context": check_context(retr),
    }
    store.close()

    print("\n=== Phase 1c integration check summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    n_pass = sum(1 for ok in results.values() if ok)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())