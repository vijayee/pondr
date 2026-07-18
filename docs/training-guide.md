# Pondr Training Guide — Reproducing Every Trained Part of the Architecture

Date: 2026-07-18
Scope: a complete, runnable recipe for every component in Pondr that is
**trained**, plus an explicit list of the parts that are deliberately
**training-free**. The guide is written so that a future swap of the SSM
backbone (e.g. `reference` → `mamba3-cuda`) can be reproduced **fairly and
cleanly** — same data, same seeds, same flags, only the backend changed — and so
that every trained artifact can be regenerated from scratch on a fresh clone.

All trained artifacts are **gitignored** (`checkpoints/`, `data/training/`,
`data/pod_runs/`, `models/*`, `.oracle_cache.json`). They are regenerable from
the scripts in this repo plus a populated WaveDB corpus. Nothing trained is
committed; the source of truth is the code, the seeds, and the corpus.

---

## 0. Global setup (applies to every pipeline)

### 0.1 The Oracle / teacher (training-data labels only)

Every "label" in Pondr is produced by an **Oracle**: a strong LLM (DeepSeek)
served **locally by Ollama** over its OpenAI-compatible endpoint. The Oracle is
a *teacher* used only during data generation — it is never in the runtime path.

- Module: `src/training/oracle_labeling.py` (`OracleClient`, `OracleConfig`).
- Shared CLI helpers: `src/training/generator_common.py` (`add_oracle_args`,
  `make_oracle`). Every `generate_*.py` script pulls these in.
- Default model: `deepseek-v4-pro:cloud` (env `ORACLE_MODEL`); the
  `deepseek-v4-flash:cloud` variant is preferred for cheaper labeling runs
  (see memory `deepseek-flash-over-pro`). Set per-run with `--oracle-model`.
- Default endpoint: `http://localhost:11434/v1` (env `ORACLE_ENDPOINT`).
- Other defaults: `oracle_max_tokens=32768`, `oracle_temperature=0.1`,
  `oracle_timeout=120s`, `oracle_max_retries=3`, `oracle_batch_size=10`.
- **`think` flag**: `--oracle-think {none,true,false}` (default `none`). Use
  `none` for DeepSeek (OpenAI `/v1/chat/completions` path with
  `response_format=json_object`). Use `false` for qwen3-class models, whose
  thinking is on by default and cannot be disabled on `/v1` — `false` routes
  through Ollama's native `/api/chat` with `think=false`. This setting is
  load-bearing for cost and determinism.
- On-disk prompt-hash cache `.oracle_cache.json` per output dir (gitignored).
  Resuming is free; re-running with the same prompts costs nothing.

**To run any Oracle-dependent generator you must have Ollama up locally with a
DeepSeek model pulled**, e.g. `ollama serve` and the model served at
`localhost:11434/v1`. No GPU is needed for the Oracle itself.

### 0.2 The WaveDB corpus (ingest prerequisite)

Most generators read an **ingested WaveDB corpus** (the `--db` argument,
default `data/pod_runs/phase1b_scale/ingest_db_dialogsum`). That store must be
populated first by ingestion (`scripts/process_corpus.py`,
`scripts/ingest_document.py`). The corpus store itself is gitignored and
regenerable. Reproducibility of training therefore depends on reproducibility
of the *ingested corpus* — re-ingest the same source corpora with the same
encoder to get a byte-equivalent store.

### 0.3 The embedder

The backbone and instance trainers embed text with `BAAI/bge-small-en-v1.5`
(sentence-transformers), producing **384-dim** vectors. The backbone operates
directly in this 384-dim space (`d_model=384`). Trainers take
`--embed-source {on-demand,stub}` (default `on-demand` — real bge-small,
downloaded on first run). Use `--embed-source stub` for offline smoke tests
(hash vectors; byte-identical shape, no semantic content).

### 0.4 The SSM backend selector (the Mamba3 pivot point)

`BackboneConfig.ssm_backend` (`src/subconscious/configs.py:37`) selects the
recurrent kernel: `"reference"` (pure PyTorch, CPU-runnable, **default**),
`"mamba3-pytorch"` (community faithful-Mamba3, CPU), or `"mamba3-cuda"`
(official `mamba_ssm.Mamba3`, CUDA pod only). Factory: `make_ssm` in
`src/subconscious/ssm.py:125`.

**Only `scripts/train_backbone.py` exposes `--backend`.** The instance
trainers (`train_retrieval_gate.py`, `train_doc_kind_head.py`) load the frozen
backbone with `load_backbone(path, BackboneConfig(), device=...)`, which uses
the default `"reference"`. This is the one friction point for a Mamba3 swap —
see §7.

### 0.5 Reproducibility hygiene (apply to every run)

- Pin a `--seed` (every script accepts one; default 0). Record it.
- Record the exact `generate_*.py` flags that produced each dataset (they
  determine the labels; the trainer only learns from labels).
- Keep the ingested corpus fixed; re-ingest only deliberately and record it.
- Oracle labels are cached — keep `.oracle_cache.json` to make re-runs free
  and identical. Delete it only if you intend to re-label.
- All outputs are gitignored; persist worthwhile artifacts to the private HF
  repos (`vijayee/pondr-models`, `vijayee/pondr-datasets`) for backup.

---

## 1. What is trained vs. training-free (the master table)

| Component | Trained? | Trainer | Data generator | Checkpoint (gitignored) |
|---|---|---|---|---|
| **JGS backbone** (SSM + JEPA) | YES | `scripts/train_backbone.py` | `scripts/extract_backbone_sequences.py` | `checkpoints/backbone/backbone_final.pt` |
| **RetrievalGate** JGS instance | YES (supervised) | `scripts/train_retrieval_gate.py` | `scripts/generate_jepa_training_data.py` | `data/training/routing_gate/{best,final}.pt` |
| **DocKindHead** JGS instance (shipped as 2-head ensemble) | YES | `scripts/train_doc_kind_head.py` | `scripts/generate_doc_kind_synthetic.py` + `label_doc_kind_*` + `export_doc_kind_pairs` | `data/training/doc_kind_head_attn_ce{0,2}/best.pt` |
| **5-head GNN** (salience / link / ontology / cluster / anomaly) | YES | `scripts/train_gnn.py` | `scripts/generate_gnn_training_data.py` | `data/pod_runs/phase3a/{head}.pt` |
| **Bonsai QLoRA adapter** (Qwen3-8B, contradiction path) | YES | `scripts/train_bonsai.py` | `scripts/generate_contradiction_training_data.py` (+ `generate_bonsai_training_data.py`) | `data/training/bonsai/lora_adapter/` |
| **WorkingMemory** JGS instance | **NO** — training-free | none | none | none (runtime state only) |
| **Presentation Gate** | **NO** — heuristic; learned gate deferred | none | none | none |
| **Uncertainty / Aspirational / Self-Model gates** | Data generated, **not yet trained** by a committed trainer | (none committed) | `scripts/generate_gate_training_data.py` | `data/training/gates/*.jsonl` |
| Oracle (DeepSeek via Ollama) | N/A — teacher only | — | — | `.oracle_cache.json` |
| Bonsai (Ternary-8B) deploy-time decider | N/A — served, not trained here | — | — | `models/*` |

The three **trained JGS instances on the shared backbone** are: RetrievalGate,
DocKindHead, and (training-free) WorkingMemory. The GNN and the Bonsai QLoRA
adapter are trained but are **independent of the SSM backbone** — a backbone
swap does not touch them.

---

## 2. Backbone pre-training (Phase 2a) — the shared understanding

This is the only training that touches the SSM dynamics. Done **once**, then
**frozen**; every instance inherits it.

### 2.1 Generate the backbone training data

`scripts/extract_backbone_sequences.py` walks each conversation's `follows`
turn-chain in the WaveDB corpus and emits forward + reverse
`(state_t, state_{t+1})` 384-dim bge-small embedding pairs. **No Oracle, no
spend.**

```bash
python scripts/extract_backbone_sequences.py \
  --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \
  --output data/training/backbone/sequences.jsonl \
  --embed-source on-demand
```

Flags: `--db`, `--output` (default
`data/training/backbone/sequences.jsonl`), `--min-chain-length` (default 2),
`--embed-source {persisted,on-demand,stub}` (default `on-demand`), `--limit`
(0 = all), `--scan-limit` (0 = all). Expect ~4,000–8,000 pairs from the
DialogSum-scale corpus.

### 2.2 Train the backbone

```bash
python scripts/train_backbone.py \
  --pairs data/training/backbone/sequences.jsonl \
  --backend reference --device cuda --dtype float32 \
  --checkpoint-dir checkpoints/backbone \
  --total-steps 3000 --batch-size 32 --seed 0
```

- `--backend` (default `reference`): the pivot for the Mamba3 swap (§7).
- `--device` (default `auto`): `cpu` | `cuda` | `auto`.
- `--dtype` (default `bfloat16`): **use `float32` on the reference backend**
  — the bf16/autocast path is unfixed in the 2a code. Mamba3-CUDA on a pod
  uses `bfloat16`.
- `--total-steps` 3000, `--batch-size` 32, `--val-fraction` 0.1, `--seed` 0.
- Config defaults (`BackboneTrainingConfig`): `learning_rate=3e-4`,
  `warmup_steps=200`, `gradient_accumulation=2`, `temperature=0.1`,
  `num_negative_samples=16`, `target_ema_decay=0.996`, `weight_decay=0.1`,
  `checkpoint_every=500`.
- Architecture (`BackboneConfig`): `d_model=384`, `n_layers=4`, `d_state=16`,
  `pred_layers=3`, `pred_dim=384` → **~19.5M params**.
- Output: `checkpoints/backbone/backbone_final.pt`.

> **Path note.** The backbone trainer writes to `checkpoints/backbone/` by
> default, but the instance trainers look for the backbone at
> `data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt` by
> default. Either train to that path (`--checkpoint-dir
> data/pod_runs/phase2a_full/checkpoints/backbone`) or pass `--backbone
> <path>` to each instance trainer. Record which you did.

---

## 3. RetrievalGate instance (Phase 2b)

### 3.1 Generate the routing pairs (Oracle)

```bash
python scripts/generate_jepa_training_data.py \
  --output data/training/jepa/ \
  --num-pairs 200 --seed 0 \
  --oracle-model deepseek-v4-flash:cloud \
  --oracle-endpoint http://localhost:11434/v1 \
  --oracle-think none
```

Output: `data/training/jepa/routing_pairs.jsonl` + `quality_report.json`.
Synthetic templated queries (no corpus read). Requires ≥10 unique pairs to
train (dedup is enforced). Note: the Oracle's `pathway` label is overwritten
with the template's `expected_pathways` (the Oracle collapses pathways to
`graph_retrieve`); this is intentional.

### 3.2 Train the gate

```bash
python scripts/train_retrieval_gate.py \
  --pairs data/training/jepa/routing_pairs.jsonl \
  --backbone data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt \
  --output data/training/routing_gate \
  --embed-source on-demand \
  --epochs 20 --batch-size 32 --lr 3e-4 --val-fraction 0.2 --seed 0 \
  --device auto --dtype float32
```

- Trains the `retrieval_gate` JGS instance (LoRA rank 4, 3 context features:
  entity_recency, topic_recency, query_complexity). Backbone frozen.
- `--dtype float32` (bf16 path unfixed). Shipped val accuracy: **0.826**.
- Output: `data/training/routing_gate/best.pt`, `final.pt`, `train_log.json`.
- A REINFORCE online path exists in config (`online_lr=1e-5`,
  `replay_buffer_capacity=1000`) but is driven by live pipeline signals, not
  this script.

---

## 4. DocKindHead instance (Phase 3c) — shipped as a 2-head ensemble

This is the instance with the strict ship gate and the multi-gate ensemble.
Reproducing the **shipped** artifact means training two heads (`pen0`, `pen2`)
and averaging their logits at serve.

### 4.1 Generate the doc-kind pairs

Multiple feeders; the canonical shipped path uses real docs labeled by a
3-teacher panel (the clean 261-train / 76-val split):

- `scripts/generate_doc_kind_synthetic.py` — Oracle (DeepSeek) synthetic pairs.
  ```bash
  python scripts/generate_doc_kind_synthetic.py \
    --out data/training/doc_kind_head/pairs_synth.jsonl \
    --model deepseek-v4-flash:cloud --min-confidence 0.7
  ```
  (This script has its own `--model`/`--endpoint` args rather than the shared
  `add_oracle_args`, so it does not take `--seed` or `--oracle-think`.)
- `scripts/label_doc_kind_corpus.py`, `scripts/label_doc_kind_panel.py` —
  zero-shot Bonsai labeling + 3-teacher panel relabel (the clean labels).
- `scripts/prep_doc_kind_v3_split.py` — reproduces the seed-0 train/val split
  (76 real val docs) for `--train`/`--val` mode.
- `export_doc_kind_pairs` (in `src/subconscious/training/doc_kind_training.py`)
  — exports from a live HippocampalStore via `--export-from-db ./data/memory_db
  --db <store>`.

> **Label-discipline warning.** A teacher's confidence does not guarantee
> label quality (see `docs/doc-kind-head-architectural-learnings.md` §3.9).
> The clean labels are a 3-teacher panel majority; reproducing the shipped
> numbers requires the panel-relabeled split, not raw single-teacher labels.

### 4.2 Train the two ensemble heads

Both heads are the same architecture with `--attention --temporal-feature` on,
differing only in `--unsafe-penalty` (0 vs 2). Same data, same seed.

```bash
# pen0 (pure CE, snap-strong)
python scripts/train_doc_kind_head.py \
  --train data/training/doc_kind_head/pairs_clean_train.jsonl \
  --val   data/training/doc_kind_head/pairs_clean_val.jsonl \
  --backbone data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt \
  --output data/training/doc_kind_head_attn_ce0 \
  --attention --temporal-feature --unsafe-penalty 0.0 \
  --epochs 80 --lr 3e-4 --seed 0 --device auto --dtype float32

# pen2 (dec-strong)
python scripts/train_doc_kind_head.py \
  --train data/training/doc_kind_head/pairs_clean_train.jsonl \
  --val   data/training/doc_kind_head/pairs_clean_val.jsonl \
  --backbone data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt \
  --output data/training/doc_kind_head_attn_ce2 \
  --attention --temporal-feature --unsafe-penalty 2.0 \
  --epochs 80 --lr 3e-4 --seed 0 --device auto --dtype float32
```

- Output: `data/training/doc_kind_head_attn_ce{0,2}/best.pt`.
- The served tagger is `EnsembleBackboneDocKindTagger`, wired via
  `build_doc_kind_tagger(ensemble_paths=[...ce0/best.pt, ...ce2/best.pt])` and
  CLI `--doc-kind-ensemble` in `scripts/ingest_document.py` (default on).
- **Ship gate (strict, verify before trusting):** `unsafe_cell<=1` AND
  `snapshot_recall>=0.70` AND `decision_update_recall>=0.70` AND
  `val_acc>=0.55` AND snapshot_recall Wilson-CI95 lower bound >=0.50. Shipped
  scorecard: both guard classes 13/17=0.765, unsafe=0, acc 0.632, CI_lo 0.527.
- Verify end-to-end through the real serve entrypoint (the probe
  `scripts/_scratch/ensemble_serve_gate.py` is the template — do not commit
  scratch).

---

## 5. 5-head GNN (Phase 3a) — independent of the SSM backbone

A GAT backbone with five heads: `salience`, `link_prediction`, `ontology`,
`cluster` (DiffPool, self-supervised), `anomaly` (Oracle-free injection).
Trained on subgraphs of the WaveDB corpus. **No SSM, no Mamba3 dependency.**

### 5.1 Generate the GNN labels

```bash
python scripts/generate_gnn_training_data.py \
  --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \
  --output data/training/gnn/ \
  --bonsai-output data/training/bonsai/ \
  --num-subgraphs 10 --subgraph-radius 3 --seed 0 \
  --oracle-model deepseek-v4-flash:cloud \
  --oracle-endpoint http://localhost:11434/v1 --oracle-think none
```

- Produces per-head `*_labels.jsonl` + `quality_report.json`, plus
  `data/training/bonsai/anomaly_decision_pairs.jsonl` (feeds Bonsai distillation).
- `salience`/`link`/`ontology` are Oracle-labeled (sharded at radius≥2);
  `cluster` is self-supervised (add `--oracle-cluster-supervision` to supervise);
  `anomaly` is Oracle-free injection.

### 5.2 Train the GNN

```bash
python scripts/train_gnn.py \
  --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \
  --labels data/training/gnn/ \
  --head all \
  --checkpoint-dir data/pod_runs/phase3a/ \
  --epochs 20 --lr 1e-3 --val-fraction 0.1 --seed 0 \
  --device cuda --dtype float32
```

- `--db` is **required** (compact corpus, opened read-only during training; the
  loader's BFS is the same walk the label generator used — zero skew, ADR 010).
- `--head all` trains the joint multi-task model and saves `all.pt` + per-head
  `.pt`; `--head <one>` refines a single head on a frozen GAT backbone (pass
  `--backbone-checkpoint`).
- Config: `hidden_dim=128`, `num_heads=4`, `num_layers=3`, `dropout=0.1`,
  `num_clusters=16`. Pod: RTX 4090/Ampere, `--device cuda --epochs 50`; dev
  smoke: `--device cpu --epochs 2`.
- `--ogb-pretrain` is **deferred** (raises a clear error; direct-train is the
  cold-start fallback).
- Output: `data/pod_runs/phase3a/{head}.pt` + sidecar `.meta.json`.

---

## 6. Bonsai QLoRA adapter (Stage B) — independent of the SSM backbone

A PEFT QLoRA fine-tune of **Qwen3-8B** for the contradiction path (extraction +
adjudication). Separate from the SSM backbone entirely; a Mamba3 swap does
not touch it.

### 6.1 Generate the contradiction pairs

```bash
python scripts/generate_contradiction_training_data.py \
  --output data/training/bonsai/ \
  --num-extraction 200 --num-adjudication 200 --seed 0 \
  --oracle-model deepseek-v4-flash:cloud \
  --oracle-endpoint http://localhost:11434/v1 --oracle-think none --report
```

- Labels are **planted/structural** — no model judges any pair (the eval
  proved model judges rubber-stamp). The Oracle is used **only as a
  paraphraser** for extraction-pair input docs; adjudication pairs are fully
  structural, no generator calls.
- Output: `data/training/bonsai/contradiction_pairs.jsonl` — chat-message
  pairs `{"messages":[user, assistant]}` where the user turn is the exact
  deploy-time prompt and the assistant turn is gold JSON. Loss is masked to
  the assistant turn.
- Optionally also generate `generate_bonsai_training_data.py` (query-planning
  + relation-extraction pairs) from the WaveDB corpus.

### 6.2 Train the QLoRA adapter

```bash
python scripts/train_bonsai.py \
  --data data/training/bonsai/contradiction_pairs.jsonl \
  --model Qwen/Qwen3-8B \
  --output data/training/bonsai/lora_adapter \
  --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 \
  --batch-size 2 --grad-accum 8 --max-len 1024 --seed 0 \
  --save
```

- 4-bit nf4 base (`BitsAndBytesConfig`, `bnb_4bit_compute_dtype=bfloat16`,
  double quant); LoRA targets
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`; `bf16=True`,
  gradient checkpointing on. `device_map="cuda"` (hardcoded).
- Hardware: local Blackwell RTX 5080 16GB, sm_120 (4-bit fwd+bwd verified viable
  2026-07-16). Base Qwen3-8B in 4-bit ≈ 5–6 GB.
- Output: `data/training/bonsai/lora_adapter/` (saved only with `--save`).
- **Never merge the adapter into the ternary Bonsai base** — merging into
  ternary rounds deltas to zero (`docs/Phase 3c.md` §7.5). Serve at runtime
  via `llama-server --lora` applied at F32.

---

## 7. The Mamba3 swap — a fair, clean reproduction

The SSM backend is the only thing that changes in a Mamba3 swap. The GNN and
Bonsai QLoRA pipelines are unaffected. The work is: rebuild the backbone with a
different backend, then retrain every JGS instance on the new backbone with
**identical data, seeds, and flags**, and re-validate every gate.

### 7.1 What changes vs. what doesn't

- **Changes:** the backbone weights (different SSM dynamics → not
  load-compatible; `ReferenceSSM` params `W_A/W_B/W_C/D` do not map onto
  Mamba3's `A_log/in_proj/out_proj/conv/norm`). So the backbone must be
  **re-pre-trained from scratch**, and every instance retrained on top.
- **Does not change:** all data-generation scripts (they produce labels, not
  SSM weights); the GNN; the Bonsai QLoRA; all instance architectures, LoRA
  ranks, gate configs, ship gates, and serve code.

### 7.2 Build Mamba3 (on a CUDA pod; it does not build on the Windows dev box)

```bash
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation \
  git+https://github.com/state-spaces/mamba.git
```

Caveats (from `src/subconscious/ssm.py:191-208` and `docs/Phase 2a.md` §0.1):
Mamba3 is **CUDA-only**; the per-step `step()` decode path is "only tested on
H100," but **backbone pre-training uses the bulk-sequence Triton prefill
path**, which runs on Ampere/Ada — **no H100 required for training**. The
`mamba3-cuda` import currently fails on this dev box (`tilelang`/`tvm_ffi`
py3.11 incompat); the `mamba3-pytorch` backend is a CPU-runnable
faithful-Mamba3 fallback for a low-risk first look.

### 7.3 Re-train the backbone on Mamba3

```bash
python scripts/train_backbone.py \
  --pairs data/training/backbone/sequences.jsonl \
  --backend mamba3-cuda --device cuda --dtype bfloat16 \
  --checkpoint-dir data/pod_runs/phase2a_mamba3/checkpoints/backbone \
  --total-steps 3000 --batch-size 32 --seed 0
```

Use a distinct `--checkpoint-dir` so the `reference` backbone is preserved for
the A/B. **Keep `--seed 0`, `--total-steps 3000`, `--batch-size 32` identical**
to the reference run — only `--backend` and `--dtype` change. (Mamba3 supports
`bfloat16`; the reference path needed `float32`.)

### 7.4 Re-train every instance on the new backbone — with one code tweak

The instance trainers currently load the backbone with
`load_backbone(path, BackboneConfig(), device=...)`, and `BackboneConfig()`
defaults to `ssm_backend="reference"`. **To load a Mamba3 backbone you must
pass `BackboneConfig(ssm_backend="mamba3-cuda")` instead.** Until those two
scripts expose a `--backend` flag (a small, recommended change: mirror
`train_backbone.py`'s `--backend`), you must either:

1. **Preferred:** add a `--backend` arg to `train_retrieval_gate.py` and
   `train_doc_kind_head.py` and thread it into `BackboneConfig(sssm_backend=...)`
   on the `load_backbone` call. This is the clean fix that makes the swap a
   pure flag change.
2. **Quick-and-dirty:** temporarily set the default in `configs.py` to
   `"mamba3-cuda"` for the run (do not commit this).

Then retrain, identical to §3.2 and §4.2 but pointing `--backbone` at the
Mamba3 checkpoint:

```bash
BB=data/pod_runs/phase2a_mamba3/checkpoints/backbone/backbone_final.pt

# RetrievalGate — same data, seed, flags
python scripts/train_retrieval_gate.py \
  --pairs data/training/jepa/routing_pairs.jsonl --backbone $BB \
  --output data/training/routing_gate_mamba3 \
  --epochs 20 --batch-size 32 --lr 3e-4 --val-fraction 0.2 --seed 0 \
  --device cuda --dtype float32 --backend mamba3-cuda   # after the tweak

# DocKindHead pen0 + pen2 — same data, seed, flags, penalties
for P in 0.0 2.0; do
  python scripts/train_doc_kind_head.py \
    --train data/training/doc_kind_head/pairs_clean_train.jsonl \
    --val   data/training/doc_kind_head/pairs_clean_val.jsonl \
    --backbone $BB --output data/training/doc_kind_head_mamba3_ce${P} \
    --attention --temporal-feature --unsafe-penalty $P \
    --epochs 80 --lr 3e-4 --seed 0 --device cuda --dtype float32 --backend mamba3-cuda
done
```

### 7.5 Fair comparison protocol

For the comparison to be fair and clean:

- **Same data.** Reuse the exact `routing_pairs.jsonl`,
  `pairs_clean_{train,val}.jsonl`, and `sequences.jsonl`. Do not re-generate
  labels (they would re-introduce Oracle nondeterminism).
- **Same seeds.** `--seed 0` everywhere, both backends.
- **Same hyperparameters.** Identical LR, epochs, batch size, LoRA ranks,
  penalties, val fractions. Only the backend differs.
- **Same gate criteria.** Re-run the strict DocKindHead ship gate and the
  RetrievalGate val accuracy against the **same** val sets.
- **Record four numbers per instance:** reference metric, Mamba3 metric,
  delta, and the seed/flags used. If Mamba3 does not clear the DocKindHead
  gate where the reference ensemble did, that is a real finding — the ensemble
  mechanism is backend-agnostic, so a failure would mean the Mamba3
  representation is worse for *this* task, not that the method is wrong.

### 7.6 When the swap is worth it (decision guidance)

- **Speed:** Mamba3-CUDA's fused scan beats the `ReferenceSSM` Python loop on
  long sequences and removes per-step overhead. Matters at scale; not yet
  felt on our short-sequence, few-thousand-pair workload.
- **Quality:** potentially richer per-channel dynamics, but unproven here.
  The DocKindHead ceiling we hit was **not** a backbone-capacity problem
  (it was the pooling readout and label noise — see
  `doc-kind-head-architectural-learnings.md`), so a backend swap alone is
  unlikely to move the metrics that mattered.
- **Cost:** a full backbone retrain + two instance retrains + re-gating. Not
  justified until either (a) the backbone corpus scales, or (b) an instance
  provably hits a representation-bound ceiling.
- **Low-risk first step:** flip `--backend mamba3-pytorch` (CPU, no build) and
  check whether the canonical Mamba3 math changes a held-out metric before
  committing to the CUDA build and the full retrain.

---

## 8. The training-free parts (documented for completeness)

- **WorkingMemory** (Phase 2c): a `JGSInstance` that does **not** reset state
  between queries. Its state *is* the trained backbone's recurrent state; it
  adds no parameters and no training (ADR 006). The `working_memory` LoRA
  adapter exists in the instance framework but is not trained by any script; a
  future REINFORCE path is deferred and would never touch the 2a backbone. So:
  nothing to reproduce; it inherits the backbone.
- **Presentation Gate** (Phase 2c): heuristic; the learned gate is deferred
  until outcome signals are wired live. Nothing to train.
- **Uncertainty / Aspirational / Self-Model gates:** labels are generated by
  `scripts/generate_gate_training_data.py` (to `data/training/gates/*.jsonl`),
  but no committed trainer consumes them yet. When trainers are added, follow
  the RetrievalGate pattern (frozen backbone + per-instance LoRA + gate head).

---

## 9. Reproducibility checklist

For a from-scratch reproduction on a fresh clone:

1. Ingest the source corpora into a WaveDB store
   (`scripts/process_corpus.py` / `scripts/ingest_document.py`) and record the
   `--db` path.
2. Start Ollama locally with a DeepSeek model; confirm
   `http://localhost:11434/v1`.
3. `scripts/extract_backbone_sequences.py` → `sequences.jsonl`.
4. `scripts/train_backbone.py --backend reference …` → `backbone_final.pt`
   (record the path; point instance trainers at it with `--backbone`).
5. `scripts/generate_jepa_training_data.py` → `routing_pairs.jsonl`;
   `scripts/train_retrieval_gate.py` → `routing_gate/best.pt`.
6. Doc-kind: generate + panel-relabel + split → `pairs_clean_{train,val}.jsonl`;
   train `pen0` and `pen2` → `doc_kind_head_attn_ce{0,2}/best.pt`; verify the
   strict gate through the real serve entrypoint.
7. GNN: `scripts/generate_gnn_training_data.py` → labels;
   `scripts/train_gnn.py --head all` → `data/pod_runs/phase3a/*.pt`.
8. Bonsai: `scripts/generate_contradiction_training_data.py` → pairs;
   `scripts/train_bonsai.py --save` → `bonsai/lora_adapter/`. Serve via
   `llama-server --lora` (never merge).
9. `scripts/validate_training_data.py --data-dir data/training/` to confirm
   every generated JSONL is well-formed.
10. Persist canonical artifacts to the private HF repos
    (`vijayee/pondr-models`, `vijayee/pondr-datasets`) for backup.

For a **Mamba3 swap**, run the same checklist but substitute §7.3–§7.4 for
steps 3–4 and 6, keep all data/seeds/flags identical, and compare gates per
§7.5.

---

## 10. Pointers

- Architecture: `src/subconscious/{configs.py, backbone.py, instance.py,
  gate.py, ssm.py}`, `docs/adr/006-working-memory-design.md`,
  `docs/adr/007-ssm-chunking-strategy.md`, `docs/jgs-the-new-primitive.md`.
- DocKindHead best-practices: `docs/doc-kind-head-architectural-learnings.md`.
- Phase plans: `docs/Phase 2a.md` (backbone, with §0 correction notes),
  `docs/Phase 2b.md` (retrieval gate), `docs/Phase 2c.md` (working memory),
  `docs/Phase 3a.md` (GNN), `docs/Phase 3c.md` (contradiction / doc-kind).
- Oracle: `src/training/oracle_labeling.py`, `src/training/generator_common.py`.
- Memory: `pondr-trained-artifacts-on-hf`, `deepseek-flash-over-pro`,
  `mamba3-cuda-build-fails`, `hippo-runtime-entrypoint-served`,
  `pondr-doc-kind-backbone-head-shipped`.