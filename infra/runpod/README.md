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

### Provisioning (done 2026-07-05, pod `1dnqbnup9mem3l`, RTX 3090, CUDA 12.8)

No build needed — the fork ships prebuilt Linux CUDA binaries. From an SSH
session on the pod (`ssh -i ~/.ssh/github_rsa -p <public-22-port> root@<ip>`):

    mkdir -p /root/bonsai && cd /root/bonsai
    # Prebuilt llama-server (CUDA 12.8) — pick the asset matching the pod's CUDA.
    curl -sL -o llama.tar.gz \
      "https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b8846-d104cf1/llama-prism-b8846-d104cf1-bin-linux-cuda-12.8-x64.tar.gz"
    tar -xzf llama.tar.gz   # -> llama-prism-b8846-d104cf1/llama-server
    # Bonsai GGUF (~2.1 GB).
    curl -sL -o Ternary-Bonsai-8B-Q2_0.gguf \
      "https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf"
    # Serve detached (redirect stdin too, or SSH lingers on the bg process).
    setsid ./llama-prism-b8846-d104cf1/llama-server \
        -m Ternary-Bonsai-8B-Q2_0.gguf --host 0.0.0.0 --port 8080 \
        -ngl 99 -fa 1 -c 4096 < /dev/null > /root/bonsai/server.log 2>&1 &

Verify on the pod: `curl -s http://localhost:8080/v1/models` (returns
`Ternary-Bonsai-8B-Q2_0.gguf`, 8.18B params). The server accepts ANY model
string in `/v1/chat/completions` (uses the single loaded model), so
`config.bonsai_model = "prism-ml/Ternary-Bonsai-8B-gguf"` works without
matching the server's filename. ~258 prompt tok/s, ~202 predicted tok/s on
the 3090.

Port 8080 is NOT publicly exposed (only 22/tcp + 8888/http). The on-pod
pipeline reaches it at `http://localhost:8080/v1`. To run the live planner
tests from the local machine, tunnel: `ssh -i ~/.ssh/github_rsa -p <22-port>
-L 8080:localhost:8080 -N root@<ip>`, then `localhost:8080` locally maps to
the pod.

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