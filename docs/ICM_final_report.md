# Intent-Conditioned Memory: What Was Measured, What Failed, and What Remains

*Report date:* 20 October 2026
*Scope:* Distilled report of the ICM (Intent-Conditioned Memory) arc, 13-20 October 2026
*Companion record:* docs/ICM_phase_report.md, addenda 1-34; docs/ICM_intent_store_spec.md
*Substrate:* WaveDB (ledger, graph, vector index), git history, prepared sandboxes

---

## 1. Abstract

The program asked whether a memory system can use **what a request is aiming at** - not mere text similarity - to select the right memory, and whether it can learn from its own misses. The answer is a qualified yes, with a clear boundary:

- On **real, mechanically verified counterfactual pairs** from git history (561 trials), a judge-built intent key plus a deterministic type filter and ancestry ordering selected the correct pre-change version **100%** of the time over the full filtered pool, while dense similarity sat **below chance** (24.2% vs 33.7%).
- With **user-originated before/after phrasings** (150 pairs), the system extracted the intended direction on 96.3% of runs, and the same pair yielded the pre-change version under "before" wording and the post-change version under "after" wording - a **perfect swap** on the semantic path.
- The **candidate generator** was the binding constraint, not the intent key: widening similarity's shortlist from top-5 to top-10, or bypassing similarity when the artifact is already known, took two-stage retrieval from **91.4% to 100%**.
- In prepared worlds, the **non-reach -> new intent** loop (layer 3), **outcome-driven adaptation** (E5), and **episodic recall across a context horizon** (E0) all passed their pre-registered checks.
- The **strong thesis remains open**: intent is stated or interpreted, not inferred from unstated traces; organic multi-turn applicability and public-benchmark external validity are not established.

The report also records the failures, because they constrain the claim: a learned relevance "lens" died at its untrained baseline, several payoff benchmarks were retired by their own pre-checks, and the personal corpora carry almost no opposed intentions (1 opposed pair in 280 sampled).

---

## 2. The question and the wall

Memory systems usually retrieve by similarity: the past item nearest the query. Similarity is a **binary** relation (memory x query). Relevance is **ternary** (memory x state x intent): the same memory can be load-bearing under one aim and misleading under another.

The clean test is a **counterfactual pair**: the same memory unit, two aims, where the correct choice swaps. If the memory text states the discriminator ("archived" vs "current"), similarity reads it and the test is vacuous. If the aim is not stated, the system must infer it.

The program spent most of its first week discovering that the corpora it held - agent traces, chat logs, code-task corpora - contain plenty of **different** aims over shared memory but almost no **opposed** ones. That is the wall. Git history turned out to be a real counterfactual producer; the rest of the report is what was built on it and what was learned.

---

## 3. The architecture that survived

The working architecture is deliberately boring:

1. **Write time - judge-built intent keys.** One language-model call per intent turns a real intent statement into a requirement list plus an epoch/qualifier reference. Keys are cached by text hash, partitioned by prompt version and model. A failure returns no key and never blocks the write.
2. **Query time - deterministic.** Dense similarity generates candidates (top-10; or bypassed when the type filter already fixes the artifact). A deterministic type/entity filter narrows them. An epoch/ancestry order re-ranks. No score blending.
3. **Ledger.** A durable record of declared intent, expectation, rule-computed congruence (confirmed/violated), and the memory units used. A non-reach can trigger a new intent, recorded with an enforced `triggered_by` link to the violated outcome.
4. **Episodic store.** WaveDB vector index over episode text, with a rule-based forgetting cascade (violated-linked episodes stay vivid; others blur and drop).

What did **not** survive: a learned projection head (the "lens"), score fusion, and any design that asks dense similarity to be the answer rather than the generator.

---

## 4. Results

### 4.1 Retrieval on real counterfactual pairs (v3)

Corpus: patch-verified inverse pairs from real git history - an add commit and a later descendant that removed its lines - from 19 repos (561 valid trials; 503 upstream, 58 own). Memory units are the real file blobs around the change; the intent is the real add-commit message; the oracle is the git blob. Query: "Restore the FIRST version of <file> (the state before this change): <message>." Pool: before/after/removed versions plus four decoys; the answer is the parent-of-add blob.

| arm | accuracy |
|---|---|
| DENSE@1 (chance 0.143) | 0.160 |
| DENSE first same-file (chance 0.337) | 0.242 |
| SEMANTIC-ONLY (filter -> epoch order) | **1.000 (561/561)** |
| TWO-STAGE (dense top-5 -> filter -> order) | 0.909 |
| Stage-1 recall@5 | 0.909 |
| Judge epoch reference | 561/561 EARLIEST |

All pre-registered checks passed: power, vacuity (dense at/below chance), stage-1, two-stage, judge.

### 4.2 Free-form phrasing

The v3 query used a template. To remove the last authored piece, the user supplied natural before/after phrasings. A first attempt with the store's abstract epoch keyer scored 0/8: the phrasings reference external times ("today", "last week") that an isolated trial cannot resolve. After user-authorized legibility edits and a task-aware BEFORE/AFTER/NEITHER keyer:

- 150 pairs x 2 directions = 300 runs.
- Extraction 0.963; before 1.000, after 0.927 (the 11 misses are change-list questions, correctly answered NEITHER).
- Semantic path 1.000 on keyed runs; two-stage 0.893 / 0.947.
- **Swap:** on 133 both-keyed pairs, the semantic path got both directions right **1.000**; two-stage 0.887.
- Before-direction dense was at/below chance (0.280 vs 0.337); the after direction is the easy one for dense (0.660 vs 0.500).

### 4.3 Stage-1 width

On the 561 v3 trials, with the keys cached:

| arm | accuracy |
|---|---|
| FILTER-ONLY (bypass dense) | **1.000** |
| STAGE-1 recall@5 | 0.914 |
| STAGE-1 recall@10 | **1.000** |
| TWO-STAGE@5 | 0.914 |
| TWO-STAGE@10 | **1.000** |

The intent key was already perfect; the shortlist was the only loss. Store rule: filter-only when the type filter fixes the artifact; dense top-10 when open-ended.

### 4.4 The expectation loop (layer 3)

Prepared sandbox: a real dataset with one hidden verification rule; the verifier reports all defects, naming them. Forty sessions, all first attempts violated.

| arm | reached |
|---|---|
| EXPANSION (reason -> explanation -> new intent -> retry) | **40/40** |
| RETRY (same spec) | 0/40 |
| NO-EVIDENCE (retry without the reason) | 2/40 |

`triggered_by` link conformance 40/40; explanation alignment 40/40. The causal factor is the violation plus its evidence, not the extra attempt.

### 4.5 Outcome-driven adaptation (E5)

Prepared world: 6 users x 24 episodes; each user's hidden convention is stable for 8 episodes, then changes at unknown points.

| arm | post-first reach | recovery | unrecovered changes |
|---|---|---|---|
| ADAPTIVE (decay + reason-identified update) | **0.905** | **2.0 episodes; 1.000 within 2** | **0/12** |
| ACCUMULATION-ONLY (no decay) | 0.333 | none | 12/12 |
| NO-LEDGER (random) | 0.278 | 2.5 avg | 1/12 |

The ablation is the result: the same stored outcomes without recency never recover; the expectation signal drives the improvement.

### 4.6 Memory across a context horizon (E0-lite)

Prepared world: 4 users x 20 tasks; the user's verification convention is stated once in a profile and never restated; the context-only arm sees a 2-message window.

| arm | task success |
|---|---|
| MEMORY (WaveDB episodic recall) | **1.000 (80/80)** |
| CONTEXT (2-message window) | 0.050 (4/80, first task only) |

Recall: 8.7 ms mean, 9.3 ms p95 (budget 250 ms). The profile was outside the window for every task after the first.

---

## 5. Negative results and retractions

These constrain the claim as much as the passes do.

1. **The learned lens is dead.** Thirty-six configurations (contrastive, margin, weight decay, low-rank, InfoNCE) landed at the untrained baseline; the head transferred 0.0 on symbolic twins. The replacement is a judge-built key, not a learned projection.
2. **Similarity as a value ranker is dead.** Dense embeddings added nothing to value prediction (concat degraded it); the dense leg surfaces topics, not usefulness.
3. **Payoff benchmarks v0 and v1 were retired by their own pre-checks.** A benchmark is only valid if the raw similarity baseline cannot solve it. The surviving label source is mechanical: filesystem state, verifier, git blob.
4. **The structural comparator has a ceiling.** It handles paraphrase (0.85) but scores unmarked opposition as similar (0.820); the judge separates the same pairs perfectly (0.000/1.000).
5. **The personal corpora have no flip shape.** Claude sessions and Onyx: 1 opposed pair in 280 sampled (0.36%, 95% CI 0.06-2.0%). They carry different aims, not opposed ones. More sessions of the same kind do not fix a shape problem.
6. **The first git-counterfactual construction failed its vacuity guard.** Whole-file records let dense reach 0.56 on the relevant decision; the construction was discarded. Windowed records fixed it.
7. **The first v2 run was underpowered** at 56 valid trials. Upstream public repos supplied the power.
8. **A world bug was found in E5 before scoring** (the dedupe truth subtracted duplicates twice). The failed run was not reported as a result. World specs need tests.

---

## 6. What this establishes, and what it does not

**Establishes:**
- On real counterfactual pairs, an intent key built from a real statement plus a deterministic filter/order selects the right memory where similarity is at or below chance.
- The mechanism survives ordinary before/after wording (user-originated phrasings), not just a template.
- The candidate generator is the engineering bottleneck; top-10 or the filter-only bypass closes it.
- A non-reach plus its evidence produces a new intent that reaches; blind retry does not.
- Outcome-driven adaptation improves reach and recovers after unannounced changes; accumulation alone does not.
- An episodic store carries an early user fact across a horizon the context window loses, at ~9 ms per recall.

**Does not establish:**
- Detection of the current, **unstated** intent from a trace. The intent operand is stated or interpreted; the lens program that tried to infer it did not work.
- **Organic multi-turn applicability.** Layer 3, E5, and E0 are prepared worlds; the retrieval corpus is organic, but its query template and pair-centred pool are authored.
- **External validity** on a public agent benchmark. No Docker on the development machine; Terminal-Bench / StateM / SWE-bench remain to be run.
- The plan's **full longitudinal E5** (a live user study) or the forgetting schedule's adaptation.
- **HDC relevance quality** (E3); only its latency (E4) is measured.
- OCR compression and chunking research; the store uses verbatim/gist.

---

## 7. Plan status

| Plan phase | Status |
|---|---|
| 0 - instrumentation, ledger, congruence | Shipped; payoff-label version audited and revised |
| 1 - episodic store, forgetting, E0 | Store shipped; E0 mechanism passed in a prepared world; real benchmark open |
| 2 - intent/relevance head (lens), E1/E2 | Lens falsified; judge-key retrieval passes E2's retrieval form; live usefulness open |
| 3 - HDC binding, E3/E4 | E4 latency passes; E3 relevance quality open |
| 4 - self-optimization, E5 | Mechanism passes (adaptive vs accumulation); full longitudinal study open |
| 5 - integration / publication | This report; external benchmark and organic study remain |

---

## 8. Reproducibility

Code and tests are tracked; all lifted corpora, pools, checkpoints, and results live under the gitignored `data/_scratch/icm/`.

```
python scripts/icm/census_git_inverses.py            # census the inverse pairs
python scripts/icm/git_counterfactual_test.py --from-pool   # v3 retrieval
python scripts/icm/git_cf_freeform.py --pairs 150    # free-form phrasing
python scripts/icm/git_cf_stage1.py                  # stage-1 width sweep
python scripts/icm/run_layer3_expansion.py --sessions 40
python scripts/icm/run_e5_longitudinal.py
python scripts/icm/run_e0_lite.py
python -m pytest tests/ -q -k icm                    # 113 ICM tests
```

Key commits: `7b88304` (store build), `eecbaf5` (v3 PASS), `91ce17c` (free-form), `dd008f9` (stage-1), `21eceee` (layer 3), `433972c` (E5), `e9f2fd3` (E0-lite). The full measurement record is `docs/ICM_phase_report.md` addenda 1-34; the build spec is `docs/ICM_intent_store_spec.md`.

---

## 9. Next steps

1. **External validity.** Run the v3 retrieval design and the E0/E5 worlds on Terminal-Bench / StateM / SWE-bench, on a Linux host or RunPod with Docker. This is the main missing claim.
2. **Organic multi-turn.** Replace the prepared worlds with a real task family that produces non-reach and expansion naturally (reverts, rollbacks, failed experiments), with mechanical outcomes.
3. **Unstated intent.** The remaining strong-thesis gap: can the system infer the operative intent before it is stated? The lens program's failure is the baseline to beat.
4. **Optional branches.** HDC relevance quality (E3), OCR compression, chunking, and the forgetting schedule's adaptation.

---

*The honest bottom line: the mechanism - intent as the third operand, built at write time and applied deterministically at query time - is measured and works on real counterfactual data. The plan's larger claim, an unstated intent that is found and a memory that improves itself over a real lifetime, is still ahead.*
