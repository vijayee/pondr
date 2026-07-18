# Total-Package Capability Assessment

Date: 2026-07-18
Status: Complete. One-pass "what can the total package even do today" view
against the Phase 5 metrics, **reusing existing gold + harnesses and building no
new gold**. Produced by `scripts/assess_capability.py` (a committed, kept
artifact). Decision context (user, 2026-07-18): *assess the whole package
against the roadmap before optimizing any piece.*

The scorecard lives at `data/assessment/scorecard.json` (raw) +
`data/assessment/scorecard.md` (the table); regenerate with
`python scripts/assess_capability.py --stage all`.

## How to read this

Every cell is one of: **measured** (a real number was taken against gold),
**gold_missing** (the metric matters but no labeled right-answer set exists to
score it), **not_run** (the stage was not selected), **dependency_missing**
(a file/model the stage needs is absent), or **error** (the stage ran and
failed). The gold-missing cells are the finding, not a gap in the doc: they say
exactly which measurements we would have to build before piece-optimization is
justifiable.

## Scorecard (run 2026-07-18, Bonsai killed, Ollama/DeepSeek up)

| Metric | Status | Value | Target |
|---|---|---|---|
| Runtime runs end-to-end | measured | ran=true, 3 queries, retrieved_total=10, endpoint_up=true | trained 2a backbone + 2b gate |
| Encoding accuracy | measured | entity 0.875, topic 0.217, tone 0.40 (n=20) | >90% entity, >85% relation |
| Retrieval recall | measured | 2/2 gold queries pass (q1 subset of _FRUSTRATED; q2 == {conv_012,conv_017}) | >80% single-session |
| Retrieval precision | gold_missing | - | >85% |
| Retrieval latency | measured | p50 2.01ms (n=2, in-memory) | <50ms graph traversal |
| Routing accuracy | measured | val 0.826 (training proxy); 3/3 smoke queries -> graph_retrieve | >90% correct routing |
| Consolidation quality | measured | trained=true, 20 subgraphs, 213 edges proposed, 0 accepted, 107 unverified, 532 anomalies, 2908 ontology proposed, 2875 pruned; decider off | >70% GNN edges validated by Bonsai |
| Contradiction detection (det) | measured | recall 1.0 on 4 catchable pairs | >=0.75 (det-normalizer ceiling) |
| Citation resolution | measured | rate 1.0 on 5 pairs | >=0.80 |
| Adjudicator guard soundness | measured | 141/141 non-real guards fire; real 0/59 (falls to dead Bonsai) | non-real false-fix MUST be 0 |
| Forgetting accuracy | gold_missing | - | <5% of pruned edges later needed |
| Context efficiency | gold_missing | - | graph context <=50% of full history |
| Uncertainty calibration | gold_missing | - | >80% precision on "I don't know" |
| Delegation efficiency | gold_missing | - | >80% queries handled by <=8B |

## What this says (interpretation)

**The total package runs, end-to-end, on trained models.** `build_ponder`
constructs a live `PonderOrchestrator` on the frozen 19.5M 2a backbone + the 2b
RetrievalGate, loads the real bge-small embedder, and `query()` returns a
populated metrics envelope: routing fires (`graph_retrieve`, supported=true),
retrieval returns episodes (10 across 3 queries), end-state planning fires
(synthesize for >3 episodes, direct for 0), and generation produces non-empty
answers (150 and 32 chars) via the repointed Ollama/DeepSeek endpoint. This is
not a skeleton -- the runtime is functional today.

**The deterministic layers are the strongest measured area.** Retrieval hits
both hand-coded gold queries exactly. Contradiction detection, citation
resolution, and the deterministic non-conflict guards are all at ceiling on
their gold sets (1.0 / 1.0 / 141-of-141 non-real caught). The guards correctly
leave the 59 `real` pairs for the (dead) LLM -- the soundness story holds
offline, matching `[[pondr-bonsai-contradiction-guards-shipped]]`.

**Encoding is the real bottleneck, and it is worse than the Phase 1a DoD
memory recorded.** Entity recall 0.875 is near target, but topic recall 0.217
sits just above the 0.2 "extraction collapsed" floor and tone recall is 0.40.
The Phase 1a status memory recorded entity 0.93 / topic 0.73 / tone 0.42; the
topic number is materially lower today. The `[[hippo-gliner-threshold-cpu-vs-gpu]]`
memory flags CPU-vs-GPU confidence divergence, and this run was CPU-only -- so
part of the topic gap may be device-dependent. But the run-to-run floor
assertion in `tests/test_extraction_quality.py` is only `>0.2`, so 0.217 is a
genuine near-collapse, not a flake. This is the same signal the Bonsai-
independence plan names: *extraction is the bottleneck, not the decider*
(`[[pondr-erag-bench-judge-harness]]`).

**Consolidation machinery runs, but quality is unmeasurable with the decider
off, and the real corpus does not scale.** On a bounded 20-conv smoke corpus
the trained GNN loads (`trained=true`), scores 20 subgraphs, proposes 213
edges, and flags 532 anomalies / 2908 ontology refinements / 2875 prunes. With
the decider off (Bonsai killed), 0 edges are accepted and
`verifier_validation_rate` is null -- so the Phase 5 "GNN-predicted edges
validated by Bonsai" cell is gold-missing, by construction. The high anomaly /
prune counts on a 20-conv corpus suggest the GNN over-fires on a tiny graph,
but that is machinery-shape, not a quality number. **The scalability finding is
real and load-bearing:** the 4995-episode dialogsum DB did not finish a
dream-pass inside 10 minutes (no report written, no stderr output) -- the GNN
subgraph extraction over the radius-3 giant (`[[hippo-phase3a-head-fixes]]`)
is the bottleneck, not the dream-pass logic. Consolidation on a production
corpus is not a quick operation today.

**Routing is measured only by proxy.** val-accuracy 0.826 is the training
proxy; routing-vs-Oracle gold does not exist (Phase 1d routing pairs not
scaled). All 3 smoke queries routed to `graph_retrieve`, which is consistent
but not a correctness measurement.

**Four Phase 5 metrics are gold-missing because the subsystems do not exist
yet.** Forgetting accuracy and context efficiency have no harness (Phase 3b
unit tests only; graph-context-vs-full-history never measured). Uncertainty
calibration needs the Uncertainty Detector / Self-Model, which is Phase 4 (not
started). Delegation efficiency needs model-sizing/delegation wired at
runtime (it is not). These are honest "not yet built" cells, not failures.

## What was NOT measured (and why)

- **Answer faithfulness / end-to-end RAG quality** -- the Option 2 work
  (ConvoMem/EverMemBench/CloneMem, answer-faithfulness LLM-judge). Deferred per
  the user's "Option 1 for now" scope.
- **Full-corpus consolidation** -- the real DB hangs; only the 20-conv smoke
  dream-pass ran. Pass `--consolidation-db <big DB>` to attempt it (slow).
- **Decider-with-Bonsai soundness** -- Bonsai is killed at user request. The
  deterministic-guards-only slice ran (sound); the full 200-pair 0-false-fix
  measurement needs Bonsai up (`scripts/_scratch/guard_coverage_200.py`).
- **Generation quality** -- the smoke stage confirms non-empty answers are
  produced (150/32 chars) but does not judge their correctness or faithfulness.

## Next-step recommendation (grounded in the observed behavior)

1. **The runtime is real and shippable as a demo.** Everything up through
   routing, retrieval, end-state planning, and generation fires on trained
   models. A roadmap walk can lean on this.

2. **Encoding is the highest-value lever.** Topic recall 0.217 (near collapse)
   is the one measured number that is both below target AND in a subsystem the
   Bonsai-independence plan already names as the real bottleneck. Two
   sub-steps, in order: (a) re-run the encoding stage on GPU to separate the
   CPU-divergence component from a true regression
   (`python scripts/assess_capability.py --stage encoding --gliner-device cuda`;
   the `--gliner-device` flag threads into `GLiNERExtractor(device=...)`);
   (b) if GPU topic recall is still well under 0.73, the extraction regression
   is real and the Bonsai-independence plan's Stage 5 (local extractor head)
   becomes the justifiable next investment -- it is the one piece-optimization
   the data already supports.

3. **Consolidation scalability is the second lever.** The 4995-episode hang is a
   concrete blocker for any consolidation-backed roadmap capability. Before
   optimizing the GNN, profile where the radius-3 subgraph extraction spends
   its time (the `[[hippo-phase3a-head-fixes]]` data-quality root cause -- all
   subgraphs the same 10680-node giant -- is the likely culprit, not the GNN
   itself).

4. **Do not build the four gold-missing harnesses yet.** Forgetting, context
   efficiency, uncertainty, and delegation are gold-missing because the
   subsystems are unbuilt (Phase 4+). Measuring them now would measure nothing.
   Build the subsystem (or not) first; the harness follows.

5. **The Bonsai-independence plan stays deferred.** This assessment did not
   change its priority -- it confirmed the plan's own "extraction is the
   bottleneck" finding (step 2 above) but did not produce a deploy target that
   forces independence. Revisit after step 2(a) lands a GPU encoding number.

## Reproducing

```
python scripts/assess_capability.py --stage all
# offline stages only (no GPU, no endpoint):
python scripts/assess_capability.py --stage encoding,retrieval,routing,contradiction
# encoding on GPU (separate CPU-vs-GPU divergence from a true regression):
python scripts/assess_capability.py --stage encoding --gliner-device cuda
# generation requires Ollama up with deepseek-v4-flash:cloud pulled:
python scripts/assess_capability.py --stage smoke --llm-model deepseek-v4-flash:cloud
# attempt full-corpus consolidation (slow, may hang on the radius-3 giant):
python scripts/assess_capability.py --stage consolidation \
  --consolidation-db data/pod_runs/phase1b_scale/ingest_db_dialogsum_backfilled_full
```

Output: `data/assessment/scorecard.json` + `data/assessment/scorecard.md`
(`data/` is gitignored; this doc is the committed record).

## Pointers

- Script: `scripts/assess_capability.py`.
- Gold reused: `data/sample_conversations.jsonl`,
  `tests/fixtures/enterpriserag/pairs.json`,
  `data/training/bonsai/contradiction_pairs.jsonl`.
- Harness reused: `tests/test_extraction_quality.py`, `tests/test_end_to_end.py`,
  `tests/test_enterpriserag_eval.py`, `scripts/run_consolidation.py`,
  `src/runtime.py::build_ponder`, `src/orchestrator.py::query()`,
  `src/gnn/bonsai_decider.py::_deterministic_non_conflict`.
- Related: `docs/bonsai-independence-plan.md` (the deferred piece-optimization
  plan this assessment informs), `docs/Ponder Engine Phases.md` (Phase 5
  metric targets).
- Memory: `[[hippo-phase-3c-status]]`, `[[hippo-phase3a-head-fixes]]`,
  `[[hippo-gliner-threshold-cpu-vs-gpu]]`, `[[pondr-erag-bench-judge-harness]]`,
  `[[pondr-bonsai-contradiction-guards-shipped]]`, `[[hippo-phase-2b-status]]`.