# Model manifest

All models used by the Ponder Engine (née Hippo) encoding + retrieval pipeline are
**pretrained, public HuggingFace models**. We did not train any model in Phase 1a/1b/1c —
the first trained model is the GNN scheduled for Phase 1d. This manifest records the exact
model ids + versions so the pipeline is reproducible without re-hosting the weights: every
entry is downloadable from HuggingFace by the id below.

The RunPod pod's container disk wipes on restart, so the Bonsai GGUF is also mirrored locally
under `models/bonsai/` (gitignored — see `.gitignore` `*.gguf` and `models/`). The other models
are small enough that the pipeline re-downloads them on first use via the respective libraries
(`gliner2`, `sentence-transformers`), caching to the HF cache.

## Encoding pipeline (Phase 1a)

| Role | Model id | Source | Notes |
|---|---|---|---|
| Entity / topic / decision span extraction + tone classification | `fastino/gliner2-base-v1` | HF | GLiNER2. `multi_label:True` classifications return lists; `multi_label:False` returns a bare string (handled by `_as_list`). |
| Open-discovery relations (secondary) | `knowledgator/gliner-decoder-base-v1.0` | HF | GLiNER-Decoder. |
| Relation extraction LLM | `prism-ml/Ternary-Bonsai-8B-gguf` | HF | Ternary-Bonsai-8B, **Q2_0** quant (`Ternary-Bonsai-8B-Q2_0.gguf`, 2.18 GB). Served via the Prism fork of llama.cpp's `llama-server` (build `llama-prism-b8846-d104cf1`), OpenAI-compatible endpoint. |

These are referenced from `src/config.py` (`gliner2_model`, `gliner_decoder_model`,
`bonsai_model`). The Bonsai server binary lives on the pod at `/root/bonsai/`; the GGUF is
mirrored locally at `models/bonsai/Ternary-Bonsai-8B-Q2_0.gguf`.

## Retrieval pipeline (Phase 1b/1c)

| Role | Model id | Source | Notes |
|---|---|---|---|
| Query planning + Mode A generation | `prism-ml/Ternary-Bonsai-8B-gguf` | HF | Same Bonsai server; OpenAI-compatible client pointed at `config.bonsai_endpoint`. |
| Vector-search embeddings | `BAAI/bge-small-en-v1.5` | HF | sentence-transformers, 384-dim, L2-normalized, `IndexFlatIP`. |

Referenced from `src/config.py` (`generation_model`, `embedding_model`).

## Reproducing the Bonsai server

```
# On a CUDA 12.8 box, no build — use the prebuilt Prism llama-server binary:
cd /root/bonsai
setsid ./llama-prism-b8846-d104cf1/llama-server \
  -m Ternary-Bonsai-8B-Q2_0.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -fa 1 -c 4096 \
  < /dev/null > server.log 2>&1 &
curl -s http://localhost:8080/v1/models   # verify
```

## Versioning this manifest

Bump the date below whenever a model id changes. Do **not** commit model weights — they are
public on HF and gitignored here.

_Last updated: 2026-07-06 (Phase 1b scale run complete)._