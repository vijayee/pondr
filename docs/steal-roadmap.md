# Steal roadmap

The committed counterpart of the running steal-roadmap notes. Each item is a
mechanism steal from an external system survey: falsifiable or cheap, gated on
explicit triggers. Status codes: DONE / PROBE (smoke-level probe ran, no
runtime code) / PLANNED (conditional trigger stated) / NOT-ADOPTED.

## Tier A - done

- **R1. Supersede blast-radius propagation** (Graft survey). DONE `1924639`.
  `--stale-propagation` default OFF; write-side in-edge mark (`stale_since` +
  `stale_of`) + query-time tip-recheck in `_filter_stale_derived`; vector legs
  share the recheck.
- **R2. TTT-as-STM probe** (TTT-E2E, arXiv 2512.23675). PROBE, smoke-level
  CONFIRMED: per-context TTA on last-quarter MLP drops loss 2.61 -> 2.39 but
  walled recall stays 0.000 at every grid cell; the bottleneck reading holds,
  no test-time memory model built. Full run blocked by machine contention.
- **R3. Session-resume state persistence** (TMT survey). DONE `3036ba6`.
  `--fade-memory-session-resume` default OFF; snapshot_carry/restore_carry on
  Mamba3Voice + orchestrator save/load seam + serve exit save trigger.
- **R4. Per-query freshness check on derived context** (Graft survey).
  DONE `7020d51`. `--query-freshness` default OFF; read-only
  `GraphTraversal.check_freshness` stat + assembly-time STALE stamp + loop-path
  tool. The pending recall A/B closed 2026-10-07 as N/A-by-construction on the
  bench corpus (zero staleness; confirmed empirically with a 50-q dual-leg
  byte-identity run). A meaningful A/B needs a staleness-bearing corpus (serve
  ingest with dedup ON, or supersession injection).
- **R5. Notes-preserve invariant for scene regeneration** (Graft survey).
  DONE `97cef31`. Invariant (no flag): encode_scene put-only codified +
  `store.scene_annotations` / `transfer_scene_annotations` +
  SceneAuthoringWorker transfer-before-delete.
- **R8. Decision-model backend (Clef-Flash / Nimble / Tev1)** (Ollama
  decision-model survey). DONE `4a90f13`. `--decision-backend` default
  OFF/bonsai; JevDecider subclass adapter over `/v1/systemone` for the five
  closed-vocab decision methods only; generation stays Bonsai. clef-vs-nimble
  A/B deferred (CPU-only).

## Tier B - done

- (R3/R4/R5/R8 listed above were Tier B at survey time; all shipped.)

## Tier C - planned, conditional (trigger per item)

- **R6. TTA recipe for per-user personalization** (TTT-E2E). Trigger: dynamic
  evaluation of the local 8B is ever wanted. Recipe fixed by their ablations:
  MLP layers only, last quarter of blocks, static-twin MLP preserving
  pretrained knowledge, chunked updates (~1 step per 1K tokens).

- **R7. Multimodal multi-vector retrieval (ColPali/ColQwen) for media memory**.
  Trigger: media ingestion is built. Replaces transcribe-then-dense with direct
  image/audio retrieval for media memories.

- **R9. OCR-backed image/scanned-PDF ingest leg** (DeepSeek-OCR survey,
  2026-10-11). Trigger: media ingestion is built AND GPU access is available
  (~7GB bf16; not CPU-feasible). Use DeepSeek-OCR-2 (91.09 OmniDocBench v1.5 at
  256-1120 visual tokens; supported in HF transformers and vLLM) as the
  image/scanned-PDF slot of the planned media-ingestion architecture: its
  markdown output drops into the existing Document/DocumentSection pipeline
  with the same parser-registry seam as pdf/code/email/web. Fidelity context:
  <10x compression -> 97% OCR precision, 20x -> ~60%. See
  arXiv 2510.18234 (V1) and arXiv 2601.20552 (V2, causal-flow query
  reordering).

- **R10. Optical-context-compression probe (third-way recall control)**
  (DeepSeek-OCR survey). Trigger: GPU access (single probe, no ingest infra).
  Render faded episode text to an image, VLM-decode at query time, measure
  fact_recall vs the fade-blurb R3 baseline at matched query cost. Prediction
  from their fidelity curve: at <=10x recall stays high -> the fade bottleneck
  is a TEXT-tokenizer bottleneck, not a compression-rate law; at >=20x it
  should collapse (corroborating the gist-bottleneck + TTT-E2E verdicts).
  This is the cheapest GPU-conditional probe on the board.

## Deferred / explicitly NOT adopted

- Deeper items parked under CPU-only constraint: clef-flash vs nimble A/B on
  ERAG-Bench 200/200, R2 full 100-ctx TTT run, 500-q / LoCoMo full runs, a
  staleness-bearing R1+R4 A/B corpus.
- NOT adopted (do not revisit without new evidence):
  - Query-time instanceOf sibling expansion (falsified on the 50-q set).
  - Multi-vector (ColBERT) as a wholesale WaveDB dense-leg swap.
  - TTT-E2E wholesale, TMT model itself, Graft as a dev tool.
  - DeepSeek-OCR / -OCR2 as a runtime default (nothing serves it until media
    ingestion exists); DeepEncoder V2 internals and visual-token reordering
    (the PresentationGate already owns select-then-present).