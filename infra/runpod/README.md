# RunPod provisioning + run flow

Phase 1a's heavy compute — GLiNER2, GLiNER-Decoder, and the local 8B ternary
Bonsai model — runs on a RunPod GPU pod, not locally. The local machine only
edits code and reads results.

## GPU target

**RTX 3090 24 GB, community cloud, pool `AMPERE_24`, $0.22/hr.**

Chosen because the Bonsai ternary model is only ~2.03 GiB packed, so 24 GB is
generous headroom for Bonsai + both GLiNER models on GPU + a 4K context window,
and the 3090 has the best community-cloud availability. Estimated total Phase 1a
spend: under $0.50 (one sub-hour pod session). An A100 (~$1.10–1.50/hr) would be
~5–12× the cost for zero benefit on a 2 GB model.

## Bonsai serving

The prism-ml Ternary-Bonsai GGUFs use Q2_0 ternary quantization with **group
size 128**, whose kernels live only in the **Prism fork of llama.cpp**
(https://github.com/PrismML-Eng/llama.cpp, `prism` branch) — NOT mainline
llama.cpp / llama-cpp-python. The fork ships prebuilt Linux CUDA binaries
including an OpenAI-compatible `llama-server`.

So Bonsai is served as a long-running process inside the pod:

    ./llama-server \
        -m Ternary-Bonsai-8B-Q2_0.gguf \
        --host 0.0.0.0 --port 8080 \
        -ngl 99 -fa 1 -c 4096

and `src/encoding/bonsai_relations.py` POSTs relation-extraction requests to
`http://localhost:8080/v1/chat/completions` (config: `bonsai_endpoint`). No
llama-cpp-python is built on the Python side.

## Files (to author when we launch)

- `Dockerfile` — base CUDA image + build toolchain (CMake, build-essential) for
  the WaveDB binding; install torch/transformers/gliner/gliner2; fetch the Prism
  fork prebuilt CUDA binary + the Bonsai GGUF; fetch the GLiNER checkpoints.
- `launch.sh` — boots a RunPod pod (RTX 3090, community) from the image, mounts
  the repo + a persistent volume for downloaded models, starts `llama-server`.
- `run_phase1a.sh` — inside the pod: `pip install -e .`, run the test suite and
  `scripts/process_corpus.py`, then sync the WaveDB dir + test results to the
  repo volume for pickup.

## Access / spend

Provisioning is gated on explicit approval — nothing here rents hardware or
spends money without a confirmed go-ahead. The RunPod MCP server is connected to
this session, so a pod can be created/stopped via the runpod tools when you say
go.