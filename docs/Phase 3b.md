# Phase 3b: Forgetting System — Implementation Record

**Ponder Engine · Architecture v2.0 · July 2026**

> **Status: implemented (training-free), pending commit.** This doc records what
> was built, the load-bearing design corrections versus the high-level spec, the
> honest scope (what ships vs. what is deferred), and the next-phase handoff. The
> approved step-by-step plan lived in `mellow-jumping-token.md`; this doc is the
> durable design record that stays in the repo.
>
> **Terminology:** "Phase 3b" is the body of work that `docs/Ponder Engine
> Phases.md` §397-451 calls "Phase 3b: Forgetting System — deploy the complete
> forgetting system with retrieval-weighted persistence."

---

## 0. Alignment Notes — spec vs. the real codebase (the load-bearing corrections)

The `Ponder Engine Phases.md` Phase 3b spec is high-level and per-edge. This
section records every correction needed to ground it in the real codebase, the
way `docs/Phase 3a.md` §0 does. Implementation followed the **Reality** column.

| Spec / draft said | Reality (this doc corrects to) |
|---|---|
| "Per-edge forgetting; each edge carries a utility_score" (edges are stateful) | **Edges are stateless `(s,p,o)` triples.** A per-edge metadata sidecar (`content/edge/{s}/{p}/{o}/...`) is the new storage layer (step 2). Lazy-created: only edges that get retrieved or decayed get a sidecar, bounding write amplification. Defaults: `state="current"`, `utility_decay_rate=0.01`, `utility_score=0.5`, `access_count=0`, `ltp_phase="early"`. |
| "Filter forgotten episodes out of default retrieval" (one filter point) | **Two granularities, two filter points — the load-bearing correction.** (1) **Episode-level** (active-forget, contradiction-supersession) — the whole episode is deprecated; filter in `default_episode_ids` (`store.py`) on `state != "current"` / `validity_end` set. (2) **Edge-level** — one `(ep, has_entity, Alice)` association can be stale while the episode is still live for Bob; filter in `_get_episodes_by_entity` / `_get_episodes_by_topic` (`graph_traversal.py`) on the edge-meta sidecar `state != "current"`. Pushing edge-level state into `default_episode_ids` is a category error (retrieval returns episodes, not edges). |
| "utility_score = 0.4*access_frequency + 0.6*structural_salience" (compose at retrieval) | **Runtime composition reuses the trained 3a heads — no retrain.** `structural_salience` = the trained `SalienceHead` output, **sigmoid'd + clipped to [0,1]** before persisting (raw logits would break the composition math — risk R4). `utility_score` is computed in code at dream-pass time; the author's "salience head ingests retrieval features as model inputs" is a **deferred lever** (salience-head retrain + re-label), documented here, not in 3b scope. 3b ships the "combined, not competing" intent at $0. |
| The decay formula (worked example `0.010 -> 0.0060 -> 0.0018`) | **Fidelity-gated.** Step 1 re-read the exact decay passages (`The_Ponder_Engine_Chat.json` [98]/[100]) and pinned the formula in the pure module `src/memory/forgetting.py` before writing tests: diminishing-returns boost `0.05*signal * 1/(1+0.3*reconsolidation_count)`, saturation (>5/24h), 7-day-half-life boost decay, one-time LTP promotion (x0.3) at `reconsolidation_count>=3` across `>=15` days. The worked example is the unit-test gate (risk R2). |
| "Contradiction -> reconsolidate" (the anomaly head resolves contradictions) | **The anomaly head only flags; it does not resolve.** Its record is `{node, type, score}` — no state values, no source episodes, no old-vs-new ordering. A new **resolver** (`Consolidator._resolve_contradictory_state`) re-derives what it can from the graph (>=2 distinct entity `state` values + source episodes ordered by timestamp) and supersedes the oldest by the latest via an E->E `supersedes` chain. Best-effort, **high-confidence only** (`score >= anomaly_resolve_threshold=0.8`); low-confidence is record-only (the head over-fires on the giant subgraph). The data model carries no value->episode provenance, so the resolver assumes the latest-asserting episode is current truth — honest caveat, risk R6. |
| "Prune the ontology" (decay unused classes) | **Seed classes are never decay-eligible.** The seed writes only `subClassOf` graph triples, no `content/class/` entry, so `scan_classes` never returns them. Decay targets DISCOVERED classes (runtime-invented labels promoted via Bonsai — a deferred path). The mechanism ships so promotion lands into a decay-ready namespace, but it is a **no-op on the seed-only ontology today**; entity->class typing edges don't exist yet, so the reassignment step is a documented no-op skeleton. |
| "Archive >365d -> physically remove + archive JSON" (deep-archive tier) | **Deferred.** The soft tier (`state='archived'`, in-place, excluded from default queries) ships. The deep-archive tier (>365d physical remove + `archive/edge/...` JSON) is a future namespace-growth mitigation (risk R8) with no consumer yet, so no `deep_archive_days` config knob was added (a knob with no consumer is dead config). |
| (Implicit) retrieval is read-only | **Retrieval now writes (the hot-path contract change).** The retrieval-time boost hook (step 5) writes `access_count++` / decay reduction to matched-edge sidecars in one `batch_sync` after results are scored. **Non-blocking:** log+continue on write failure (never break retrieval on a sidecar write). RMW concurrency caveat: `access_count++` is a read-modify-write; under concurrent queries increments can be lost (same single-user assumption 2c makes — acceptable, documented). Risk R1. |
| 3a hard-prune vs. 3b soft-archive collide | **3a hard-prune takes precedence (risk R5).** In `_step_forget`, an edge already in `report["pruned"]` (3a will delete it) is skipped, so 3b never writes a sidecar for a doomed edge. |

### What **already** exists (reused, not rebuilt)

- The 3a trained `SalienceHead` (sigmoid'd for `structural_salience`).
- The 3a anomaly head (`contradictory_state` = `ANOMALY_TYPES[0]`, entity-scoped).
- The materialized `subClassOf` ontology triples (decay substrate).
- `store.get_entity_salience` / `write_entity_salience_batch` (the 1c mention-only
  salience — now the cold-start fallback for the composed salience).
- The Phase 2c `PonderOrchestrator` (`query` gains a `signal` param; new
  `forget` / `reconsolidate` user triggers).

### What is new in 3b (greenfield)

`src/memory/forgetting.py` (pure decay math), `src/memory/edge_meta.py` (sidecar
CRUD + module-level `safe()` hasher extracted from `semantic_memory.py`),
`set_episode_state` / per-edge sidecar / `persist_node_salience` / class-decay
store paths, the two retrieval filter layers, the retrieval-time boost hook, the
`_step_forget` dream pass + apply, the anomaly resolver, ontology decay,
`orchestrator.forget` / `reconsolidate`, the `superseded_by` ontology property,
and the `--forget` / threshold CLI knobs. (No new ADRs: this doc carries the §0
alignment notes + risk register that an ADR would; the 3a ADRs 008/009 cover the
shared GNN + semantic-memory storage 3b builds on.)

---

## 1. Overview

Phase 3b deploys the forgetting system: a memory that **strengthens when used,
fades when ignored, supersedes on contradiction, deprecates on request (never
deletes), and prunes the ontology** — reusing the 3a trained heads with **no new
training** (3b has no row in the cost table; runtime logic like Phase 2c).

Two execution sites + one user trigger:

1. **Retrieval-time (online, hot path):** after `retrieve` scores, the matched
   edges get a boost (signal-weighted) written to their sidecars —
   non-blocking, fire-after-results-in-hand.
2. **Consolidation-time (dream pass):** `_step_forget` in the per-center loop
   applies dream-state decay, recomputes `utility_score`, promotes LTP, and
   soft-archives edges below `utility_prune_below`; `_apply` persists. The
   anomaly->reconsolidation resolver and ontology decay run in the apply phase.
3. **User-triggered:** `orchestrator.forget(episode)` (episode-level deprecate)
   and `orchestrator.reconsolidate(old, new)` (E->E `supersedes` chain).

### 1.1 Prerequisites (status)

- 3a trained heads shipped (salience, anomaly) — reused, not retrained.
- `forgetting_enabled` master gate (default `True`): the filters are a no-op
  until something is actually deprecated, so a fresh corpus is unaffected.

### 1.2 What 3b does NOT do (honest scope)

- **No model retrain.** Structural salience is the trained head output
  sigmoid'd; the "salience head ingests retrieval features as model inputs"
  lever is deferred (would need a retrain + re-label).
- **No deep-archive physical removal** (>365d). Soft-archive (in-place
  `state='archived'`) ships; deep-archive is a future namespace-growth tier.
- **Ontology decay is a no-op on the seed ontology.** Discovered-class promotion
  (Bonsai-gated) is deferred; the mechanism ships ready for it. Entity->class
  reassignment is a documented no-op (no typing edges yet).
- **The anomaly resolver is best-effort.** No value->episode provenance in the
  data model; the resolver assumes latest-asserting = current truth.
  High-confidence flags only; low-confidence is record-only.
- **LTP thresholds are canonical constants**, not config knobs
  (`forgetting.py`: `LTP_RETRIEVAL_COUNT=3`, `LTP_WINDOW_DAYS=15`). Exposing them
  invites breaking the worked-example fidelity gate; they stay pinned.
- **Whether real corpora produce meaningful disuse/contradiction signal is
  empirical for Phase 5**, not a 3b blocker. 3b ships the *mechanisms* of
  graceful forgetting.

---

## 2. File Map

| Path | Role |
|---|---|
| `src/memory/forgetting.py` | Pure decay/composition math (no store). Worked-example unit tests. |
| `src/memory/edge_meta.py` | Sidecar CRUD: `get/update/batch_update_edge_meta`; module-level `safe()` hasher. |
| `src/memory/store.py` | `set_episode_state`, episode/edge filters, `persist_node_salience`, composed `get_entity_salience` (mention x recency x structural), class-decay helpers (`scan_classes`, `persist_class_last_seen`, `set_class_state`, ...). |
| `src/memory/ontology.py` | `supersedes` + new `superseded_by` (Episode->Episode back-pointer). |
| `src/retrieval/graph_traversal.py` | Edge-level filter in `_get_episodes_by_*`; retrieval-time boost hook; `signal` threading. |
| `src/retrieval/retriever.py` | `signal` param threaded to traversal. |
| `src/orchestrator.py` | `query(signal=...)`, `forget(episode)`, `reconsolidate(old, new)`. |
| `src/gnn/consolidate.py` | `_step_forget` + apply (edge-meta, anomaly->reconsolidation, ontology decay); sigmoid+persist node salience; report `forgetting` section. |
| `src/gnn/semantic_memory.py` | `supersede_episode` (E->E chain + state/validity); extracted `safe()`. |
| `src/config.py` | `forgetting_enabled` (master gate); `utility_prune_below`, `anomaly_resolve_threshold`, `ontology_decay_days`. |
| `scripts/run_consolidation.py` | `--forget/--no-forget`, `--utility-prune-below`, `--ontology-decay-days`, `--anomaly-resolve-threshold`. |
| `scripts/compute_entity_salience.py` | Persists `last_mentioned_ts` (max-mention timestamp per entity) for cheap recency. |
| `tests/test_forgetting.py` | Decay math vs. the worked example (`0.010->0.0060->0.0018`). |
| `tests/test_edge_meta.py` | Sidecar CRUD + key hashing (incl. `/`-bearing ids) + batch atomicity. |
| `tests/test_episode_forgetting.py` | Episode-level filter (forget excludes from default, reversible, not deleted). |
| `tests/test_edge_forgetting_filter.py` | Edge-level filter (deprecated Alice edge excludes for Alice, not for Bob). |
| `tests/test_retrieval_boost.py` | Retrieval-time boost + signal threading; write-failure non-fatal. |
| `tests/test_consolidate_forgetting.py` | Dream-pass `_step_forget` + apply; dry-run no-mutation; utility composition. |
| `tests/test_orchestrator_forgetting.py` | `forget` / `reconsolidate` API; signal boosting. |
| `tests/test_anomaly_reconsolidation.py` | Resolver + apply hook (high vs. low confidence). |
| `tests/test_ontology_decay.py` | Class `last_seen` stamping + decay deprecates stale discovered classes. |
| `tests/test_entity_salience_compose.py` | Composed salience (mention x structural x recency) + cold-start fallback. |

---

## 3. The decay math (canonical)

`src/memory/forgetting.py` (pure, no store):

- base `utility_decay_rate = 0.01`/day, floor `0.001`.
- diminishing-returns boost: `0.05 * signal * 1/(1+0.3*reconsolidation_count)`.
- saturation: >5 retrievals in 24h -> skip boost + `decay *= 1.02` + flag.
- 7-day-half-life boost decay: `0.9 ** days_ago`.
- LTP promotion (one-time x0.3): `reconsolidation_count>=3` and `>=15` days since
  `consolidation_window_start` -> `decay *= 0.3`, `ltp_phase="late"`.
- LLM `signal` modifiers: important=1.5, routine=1.0, satisfied=1.2,
  frustration increases decay, correction=0.0 (no boost + reconsolidation).

`compose_utility(access_frequency, structural_salience) = 0.4*af + 0.6*ss`
(the edge utility metric; structural must be sigmoid'd [0,1] or it raises).

Entity salience (step 10, retrieval ranking) is a **different** composition:
`salience = mention_factor * structural_factor * recency_factor`, where
`recency = 0.5 ** (age_days / 30)` and recency is neutral (1.0) when no
`last_mentioned_ts`. Cold-start (no `structural_salience`) -> mention-only,
byte-identical to Phase 1c, so a fresh corpus and the GNN cold-start prior
(`features.py`) are unchanged.

---

## 4. Risk register (resolved / managed)

- **R1 retrieval-now-writes:** bounded to query-matched edges, one `batch_sync`,
  fire-after-results, log+continue on failure, RMW caveat documented. **Managed.**
- **R2 decay-formula fidelity:** pinned from the chat doc, worked-example is the
  unit-test gate. **Resolved.**
- **R3 two-granularity filter:** separate tests for episode vs edge; an episode
  with one deprecated edge and one current edge is excluded for the deprecated
  axis and included for the current one. **Resolved.**
- **R4 raw-logits salience:** sigmoid+clip at persist; composed value tested in
  [0,1]. **Resolved.**
- **R5 3a-prune coexistence:** 3a hard-prune runs first; 3b skips pruned edges. **Resolved.**
- **R6 anomaly resolver:** conservative high-confidence-only apply; low-confidence
  record-only; best-effort, documented. **Managed.**
- **R7 default-query semantics:** gated on `forgetting_enabled` (default True);
  archived excluded from default, available historically. **Managed.**
- **R8 per-edge namespace growth:** lazy-create bounds it; deep-archive tier
  deferred. **Managed (monitor in Phase 5).**

---

## 5. Definition of Done

- [x] Decay math reproduces the worked example `0.010 -> 0.0060 -> 0.0018`.
- [x] Sidecar CRUD + key hashing (incl. `/`-bearing ids) + batch atomicity.
- [x] Episode-level + edge-level filters (two granularities, two test sets).
- [x] Retrieval-time boost (signal-weighted, non-blocking, write-failure safe).
- [x] Dream-pass `_step_forget` + apply (decay, composed utility, LTP,
      soft-archive); sigmoid+persist node salience; dry-run no-mutation.
- [x] `forget` / `reconsolidate` API + E->E `supersedes` chain + `superseded_by`.
- [x] Anomaly->reconsolidation resolver (high-confidence only).
- [x] Ontology decay (discovered-class `last_seen` + deprecate; seed-safe).
- [x] Composed `get_entity_salience` (mention x structural x recency) + fallback.
- [x] CLI knobs (`--forget`, `--utility-prune-below`, `--ontology-decay-days`,
      `--anomaly-resolve-threshold`).
- [x] de-wonk audit (this doc + memory update); commit on user request.
- [ ] **Consolidation dry-run on the 23-center slice** with a trained checkpoint
      (verification §5 of the plan) — empirical, deferred to the next GPU/pod
      opportunity (the trained `all_fixed.pt` is local; the slice is on disk).

---

## 6. Next Phase Handoff

- **Document/record ingestion (RAG-replacement pillar), task #17** — the deferred
  work after 3b. The ingestion layer + a unit model (Episode generalized beyond
  the chat turn) + a Document ontology branch. Media-agnostic pieces already
  exist (extraction/retrieval/GNN); the gap is ingestion. See memory
  `hippo-doc-ingestion-gap`.
- **Empirical forgetting signal (Phase 5):** whether real corpora produce
  meaningful disuse/contradiction is for Phase 5; 3b ships the mechanisms.
- **Deferred levers (honest):** salience-head retrain (retrieval features as model
  inputs); discovered-class promotion (Bonsai-gated); deep-archive physical
  removal (>365d); the process-metadata carve-out (chat [109,110] -> Phase 6
  Procedural Memory).