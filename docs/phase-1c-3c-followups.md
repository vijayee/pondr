# Phase 1c–3c deferred follow-ups

Date: 2026-07-19. Status: tracked follow-ups, **not started**.

This is the companion to the Phase 1c–3c hardening pass (commits `d99ac76`
doc-sync, `ff6b8cc` DocumentRetriever, `e5cb4e0` CSR gate, `b16810e` flag flip,
`5b284ec` expand-handler tests). The hardening did the cheap/safe work +
two pieces of real new code; everything below is **expensive and deliberately
deferred** per the user's 2026-07-18 scoping decision:

> Document + defer the expensive runs (training-data at scale, Bonsai
> contradiction LoRA fine-tune, link_prediction re-label/retrain, OGB
> pretrain) as tracked follow-ups — do NOT run them now.

Each item carries the exact run command, prerequisites, rough budget, and the
reason it is deferred. Nothing here is claimed as done.

## Why deferred (common prerequisites)

- **Bonsai status (runtime, not generation).** The local 8B `llama-server`
  (`localhost:8080/v1`) was killed at the user's request (gaming) — see memory
  `hippo-bonsai-local-server` — and restarted 2026-07-19 for the live dogfood.
  Note: **none of the generation follow-ups below actually need Bonsai.** The
  training-data generators use the **Oracle** (DeepSeek) as the teacher; the
  8B Bonsai is the *student* the data trains, served by Bonsai only at runtime
  (live dogfood / `serve_ponder`). Bonsai up is incidental to generation.
  (Earlier draft of this doc mislabeled 1d-2/1d-3 as "Bonsai-dependent" —
  corrected 2026-07-19 after reading `make_oracle` + the 1d-2 cache.)
- **Oracle status + budget.** The Oracle is DeepSeek via Ollama
  (`localhost:11434/v1`, `think=False`); up 2026-07-19. **Use
  `deepseek-v4-flash:cloud`** (memory `deepseek-flash-over-pro`), not the
  `deepseek-v4-pro:cloud` config default — the cost-tracker records $0 for
  `:cloud` models (no price table), so do NOT read $0.0000 as free. Full-scale
  gate/label generation (200k examples) is real spend, not approved for this
  pass; the validate-slices below are cheap on flash.
- **No GPU pod.** Retrains (GNN, linkpred, LoRA) need an L4/A5000 pod with
  `torch_geometric>=2.8` and the `[gnn]` extra; the local box is CPU-dev.
- **`data/` is gitignored** and lives off-disk (HF `vijayee/pondr-datasets`,
  private — see memory `pondr-trained-artifacts-on-hf`). Outputs land under
  `data/training/` and are backed up to HF, not committed.

---

## 1. Training-data at scale (Phase 1d)

The validate-slice datasets shipped (a few records each, enough to prove the
generators run + the trainers load). The production-scale runs are deferred.

### 1d-1 GNN training data — target 4k+ subgraphs
- **Command:** `python scripts/generate_gnn_training_data.py` over the full
  conversation corpus, all 5 heads, `--subgraph-radius 3` (the radius-3
  subgraphs are the realistic unit; radius-1 is the dev slice).
- **Anomaly is Oracle-FREE:** anomaly labels come from
  `anomaly_injector.inject_anomalies` + `anomaly_rules.ANOMALY_TYPES` /
  `enrich_subgraph` / `node_label_vectors` (documented in `docs/Phase 1d.md`
  §3), NOT from an Oracle prompt. So the anomaly head needs no Oracle budget —
  only the corpus + a pod.
- **Prerequisites:** corpus loaded, GPU pod. **Budget:** pod hours (extraction
  is CPU-bound GLiNER; the radius-3 fan-out is the cost).
- **Why deferred:** 23 records is enough to keep the trainer green; 4k needs a
  pod and the data-quality fixes from `hippo-phase3a-head-fixes` (the radius-3
  giant-component bug) verified not to recur at scale.

### 1d-2 Bonsai query + relation pairs — 5k query / 2k relation
- **Command:** `python scripts/generate_bonsai_training_data.py
  --oracle-model deepseek-v4-flash:cloud` (the teacher is the **Oracle**
  via Ollama `localhost:11434`; the 8B Bonsai is the *student* the pairs
  train, NOT the thing queried. "Bonsai" in the generator/prompt names = the
  target, not the teacher).
- **Prerequisites:** Oracle (DeepSeek) up — NOT Bonsai. **Budget:** Oracle
  tokens (use **flash**, not the `deepseek-v4-pro:cloud` default — see memory
  `deepseek-flash-over-pro`; the cost-tracker records $0 for `:cloud` models
  because it has no price table for them, so do NOT read $0.0000 as free).
- **Validate-slice RAN 2026-07-19:** `--num-query-pairs 10
  --num-relation-pairs 10` → 10 query-planning + 10 relation-extraction
  pairs, ~20 new Oracle calls (much served from the persistent
  `.oracle_cache.json`), `quality_report.json` written. Generator works
  end-to-end against the Oracle. (Slice used the pro default; full run uses
  flash.)
- **Why deferred:** the full 5k+2k run is hours of Oracle calls + the
  output is gitignored (HF backup); the slice proves the generator.

### 1d-3 anomaly_decision_pairs
- **Command:** `python scripts/generate_gnn_training_data.py
  --oracle-model deepseek-v4-flash:cloud --heads anomaly --num-subgraphs 3
  --subgraph-radius 1 --anomaly-radius 1 --anomaly-fanout-cap 16
  --max-decision-pairs-per-subgraph 5` WITHOUT
  `--skip-anomaly-decision-pairs` — the decision pairs are a sub-task of the
  GNN generator (`_run_anomaly_decision` at line 379), produced from the SAME
  injected anomalies as the anomaly head. Use `--subgraph-radius 1` for a
  slice; radius-3 hits the giant-component landmine (memory
  `hippo-phase3a-head-fixes`).
- **Prerequisites:** Oracle (DeepSeek-flash via Ollama `localhost:11434`) up.
  The *teacher* is the **Oracle**, NOT the 8B — `run_batches(oracle, ...)` in
  `_run_anomaly_decision`. The 8B `decide_anomaly`
  (`src/gnn/bonsai_decider.py:190`) is the *student* the pairs distill INTO.
  (Earlier draft of this doc mis-stated the prereq as Bonsai — corrected
  2026-07-19 after reading the generator.)
- **Budget:** Oracle tokens (cheap on flash; ~$0 if a local Ollama model is
  used). Coupled to the GNN radius-fanout, so the pod/CPU cost of 1d-1 applies
  too.
- **Validate-slice RAN 2026-07-19** (`--heads anomaly --num-subgraphs 3
  --subgraph-radius 1 --anomaly-radius 1 --anomaly-fanout-cap 16
  --max-decision-pairs-per-subgraph 5 --oracle-model deepseek-v4-flash:cloud`):
  3 radius-1 subgraphs (max 10 nodes — NO giant-component), 3 anomaly records
  (Oracle-FREE injection) + 15 decision candidates → **15 anomaly_decision_pairs**
  distilled via the Oracle flash teacher (15 calls, 24084 tokens). Record
  shape verified: `{flagged_entity, retrieved_context, anomaly_type, decision,
  action, reasoning}`; sample decision `ask_user` on a `contradictory_state`
  anomaly (consistent with the shipped 877d29e guards). Generator works
  end-to-end.
- **Why deferred:** the file is empty (cold-start). The anomaly *labels* are
  Oracle-FREE (1d-1 injection), but the *decision pairs* use the Oracle
  teacher.

### 1d-4 gates — 50k each × 4 gates (incl. the new CSR)
- **Command:** `python scripts/generate_gate_training_data.py --num-examples 200000`
  (40 default across all 4 gates is the dev slice; 200k = 50k each).
- **Prerequisites:** Oracle (DeepSeek-flash) up for the labeling calls. The
  four gates: `uncertainty_detector`, `aspirational_model`, `self_model`, and
  `common_sense_resolver` (added `e5cb4e0`). **Budget:** ~$20 Oracle tokens.
- **Why deferred:** no Oracle budget approved; 3-each is enough to keep the
  generator + prompt tests green.

### 1d-6 code_aware — target 2k
- **Command:** `python scripts/generate_code_aware_data.py` -> 2k.
- **Prerequisites:** `tree-sitter` (the code parser skips without it). **Budget:**
  CPU + tree-sitter.
- **Why deferred:** 5 records shipped; 2k needs tree-sitter installed + the
  corpus.

### 1d-8 aggregated `data/training/reports/` roll-up
- **Status:** the `reports/` dir is absent; each dataset writes its own
  `quality_report.json`. Add a small script that consolidates the per-dataset
  reports into `reports/{gnn,bonsai,jepa,gates}_quality.json`.
- **Budget:** trivial to add anytime, but only meaningful after the runs above
  produce real reports. Defer with them.

---

## 2. Bonsai contradiction LoRA fine-tune (3c-P1)

- **What:** generate contradiction decision pairs with the 27B / DeepSeek
  teacher, PEFT-LoRA on **dense** Qwen3-8B-Instruct (the ternary base per
  Sec 7.8), eval against the 16-pair harness, serve as a **runtime LoRA
  adapter** — NEVER merge the LoRA into the ternary base (rounds deltas to 0;
  see memory `pondr-bonsai-27b-zeroshot-probe`).
- **Stage B plan:** `docs/Phase 3c.md` Sec 7.5; repro scripts shipped in
  commit `9b71ab8`. Stage A (the zero-shot probe decision) is DONE — LoRA
  fine-tune WARRANTED (memory `pondr-bonsai-zeroshot-eval-finetune-warranted`).
- **Prerequisites:** a pod with the dense Qwen3-8B-Instruct + PEFT; the 16-pair
  eval harness (uncommitted probe under `scripts/_scratch/`). **Budget:** pod
  hours for the LoRA train + eval.
- **Why deferred:** own evidence-based decision (the zero-shot probes already
  cleared the capacity-bound question); needs a pod + Bonsai-class GPU. The
  deterministic guards shipped in `877d29e` (memory
  `pondr-bonsai-contradiction-guards-shipped`) hold the production line until
  the LoRA lands.

---

## 3. link_prediction re-label + retrain (3a-P5)

- **What:** 4200 skipped endpoints in the link-pred labels; val AUC=1.0 is
  suspicious (a label-leak / node-id misalignment). Regenerate the link-pred
  labels with node-id alignment verified, retrain the linkpred head, re-assemble
  `all_fixed.pt` via `scripts/assemble_gnn_checkpoint.py`.
- **Prerequisites:** GPU pod. **Budget:** pod hours (label regen + retrain +
  reassemble).
- **Why deferred:** needs a pod; the current `all.pt` is the production
  checkpoint and the rest of the GNN stack does not depend on the linkpred
  head being retrained (it is one of five heads). See memory
  `hippo-phase-3a-status`.

---

## 4. OGB pretrain-then-transfer (3a-P1)

- **What:** implement the `ogbn-arxiv` pretrain loop + per-layer
  `load_state_dict(strict=False)` transfer. The stub is
  `src/gnn/train.py:219` (`_ogb_pretrain`) — a loud-fail that prints the
  implementation plan (the docstring there is the spec). A/B it vs the
  direct-train `all.pt`, keep the better, record the decision in ADR 008.
- **Command (after wiring):** `python -m src.gnn.train --ogb-pretrain ...`
  (currently the flag hits the stub and exits with the plan message).
- **Prerequisites:** pod + `pip install '.[gnn]'` (brings `ogb`). **Budget:**
  pod hours (arxiv pretrain + transfer + A/B).
- **Why deferred:** pod-only; direct-train is the cold-start fallback and is
  what the production `all.pt` currently uses. See memory
  `hippo-phase-3a-status`.

---

## 5. Deferred-by-design (explicit future work — NOT partials)

These are **not** incomplete Phase 1c–3c work; they are later-phase items
called out here so they are not mistaken for partials. Leave them.

- **Learned Presentation Gate (2c-1):** needs outcome signals (the
  `ReplayBuffer` is the seed; the heuristic gate is the production default).
- **Gate context features (2b-1, Phase 2.5):** richer gate input features.
- **process / tool / ssm_direct executors (2b-2):** later-phase executors;
  the current executor set is the intended 2c scope.
- **Ontology decay discovered-class promotion (3b-P2):** Bonsai-gated;
  deferred with the Bonsai-dependence decision.
- **Salience-head retrain (3b-P3):** training-free forgetting shipped (Phase
  3b A1/A3/A4); the salience head retrain is a later quality lever.
- **Full ERAG LLM-judge harness (3c-P2):** the 200/200 scale guard ran
  (memory `pondr-erag-bench-judge-harness`); the full bench harness is a
  later eval investment.

---

## 6. Doc-only nits (folded here, not worth a separate change)

- The DocKindHead "Sec 7.10 -> 7.11" drift the hardening plan expected was
  already fixed (the docs read "Sec 7.11" — see
  `docs/doc-kind-head-architectural-learnings.md:7`). No remaining nits found
  in the Phase 1c–3c doc set; this section is kept as a placeholder for the
  next cosmetic pass.

---

## De-wonk note

This doc is a deferred-work register, not a TODO list disguised as done. Every
item lists prerequisites + budget + the reason it is deferred, so the
"measured 0 / false-green" failure mode (a follow-up claimed as shipped) does
not apply — nothing here is claimed to run.