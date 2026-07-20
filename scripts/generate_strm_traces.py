"""Generate the STRM Phase 0a traces (the training data for the 2b + 2c heads).

Steps the Phase 2a backbone's WorkingMemory through each forward chain of
384-dim episode embeddings and records, per chain, the input stream
``u_0..u_T`` and the per-step recurrent state ``state_t`` [4, 16, 384]. The
2b recoverability head trains on (state_t, anchor u_i, decoder e(i,t)); the
2c latent-dynamics head trains on consecutive (z_t, z_{t+1}). Both consume
this one artifact.

This is the committed, reproducible equivalent of the (uncommitted, probe-only)
``scripts/_probe_recoverability.py`` trace builder. The trace builder is
promoted to a committed script so the heads' training data is regenerable
from committed code + the backbone checkpoint + the forward-chain embeddings
(all three gitignored but regenerable) -- without it the trainers would
depend on a throwaway probe artifact.

No ring buffer (``ring_capacity=0``); the generator reads ``wm.state``
directly after each step (the ring is a Phase 1 serve-side concern). States
are stored fp16 (halves disk; upcast at fit time).

Usage:
    python scripts/generate_strm_traces.py --max-chains 400 \
        --output data/probe/recoverability/traces.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.training.routing_training import load_backbone  # noqa: E402
from src.subconscious.working_memory import WorkingMemory  # noqa: E402

DEFAULT_SEQ_PATH = "data/training/backbone/_after_fix2_full.jsonl"
DEFAULT_BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
DEFAULT_OUTPUT = "data/probe/recoverability/traces.pt"


def load_chains(path: str, max_chains: int) -> list[list[np.ndarray]]:
    """Forward chains as ordered lists of 384-dim input embeddings u_0..u_T.

    Reads the JEPA pre-train transition JSONL (each record has ``type``,
    ``chain_id``, ``position``, ``state_t`` -- the 384-dim input embedding,
    confusingly named ``state_t``). Groups by ``chain_id``, sorts by
    ``position``, drops chains with <2 steps. Caps at ``max_chains``.
    Mirrors the Phase 0a probe's loader.
    """
    chains: dict[str, list[tuple[int, list[float]]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "forward":
                continue
            chains.setdefault(rec["chain_id"], []).append(
                (rec["position"], rec["state_t"]))
    out: list[list[np.ndarray]] = []
    for cid in sorted(chains):
        steps = sorted(chains[cid], key=lambda p: p[0])
        stream = [np.asarray(s, dtype=np.float32) for _, s in steps]
        if len(stream) >= 2:
            out.append(stream)
        if len(out) >= max_chains:
            break
    return out


def build_traces(chains, backbone, device) -> list[dict]:
    """Step WorkingMemory through each chain; record (inputs, states) per chain.

    The ring is OFF (``ring_capacity=0``) -- the generator reads ``wm.state``
    directly. Per-step state stacked ``[4, 16, 384]`` then all steps stacked
    ``[T, 4, 16, 384]`` and cast to fp16 to halve disk. Inputs kept fp32.
    """
    wm = WorkingMemory(backbone, ring_capacity=0)
    traces: list[dict] = []
    t0 = time.time()
    for ci, stream in enumerate(chains):
        wm.reset()
        inputs = torch.from_numpy(np.stack(stream)).unsqueeze(1).to(device)  # [T,1,384]
        states = []
        for t in range(len(stream)):
            wm.step(inputs[t])
            st = torch.stack([s.detach().to("cpu") for s in wm.state])  # [4,1,16,384]
            states.append(st.squeeze(1))                                 # [4,16,384]
        states_t = torch.stack(states).to(torch.float16)                # [T,4,16,384]
        traces.append({"inputs": inputs.squeeze(1).to(torch.float32),   # [T,384]
                       "states": states_t})
        if (ci + 1) % 50 == 0:
            print(f"  traced {ci + 1}/{len(chains)} chains ({time.time() - t0:.1f}s)",
                  flush=True)
    return traces


def main() -> int:
    p = argparse.ArgumentParser(description="Generate STRM Phase 0a traces")
    p.add_argument("--seq", default=DEFAULT_SEQ_PATH,
                   help="JEPA pre-train forward-chain embeddings JSONL")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH,
                   help="Phase 2a backbone checkpoint (backbone_final.pt)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="output trace file (per-chain {inputs, states})")
    p.add_argument("--max-chains", type=int, default=400,
                   help="cap on the number of forward chains to trace")
    p.add_argument("--device", default="cpu", help="cpu|cuda|auto")
    p.add_argument("--retrace", action="store_true",
                   help="regenerate even if the output file exists")
    args = p.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.retrace:
        print(f"  traces already exist at {out_path} (use --retrace to regenerate)",
              flush=True)
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seq_path = Path(args.seq)
    if not seq_path.exists():
        print(f"ERROR: forward-chain embeddings not found at {seq_path}. "
              f"Regenerate the JEPA pre-train data first.", file=sys.stderr)
        return 1
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone checkpoint not found at {backbone_path}",
              file=sys.stderr)
        return 1

    print(f"Loading forward chains from {seq_path} (max {args.max_chains})",
          flush=True)
    chains = load_chains(str(seq_path), args.max_chains)
    if not chains:
        print(f"ERROR: no forward chains with >=2 steps loaded from {seq_path}",
              file=sys.stderr)
        return 1
    lens = sorted(len(c) for c in chains)
    print(f"  {len(chains)} chains (len min/med/max="
          f"{lens[0]}/{lens[len(chains) // 2]}/{lens[-1]})", flush=True)

    print(f"Loading frozen backbone from {backbone_path}", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  backbone: {sum(p.numel() for p in backbone.parameters()):,} params (frozen)",
          flush=True)

    print(f"Tracing (ring OFF, fp16 states) -> {out_path}", flush=True)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else args.device if args.device != "auto" else "cpu")
    traces = build_traces(chains, backbone, dev)
    torch.save(traces, out_path)
    mb = out_path.stat().st_size / 1e6
    print(f"DONE. saved {len(traces)} chain traces -> {out_path} ({mb:.1f} MB)",
          flush=True)
    print(f"  Next: train the heads --", flush=True)
    print(f"    python scripts/train_latent_dynamics_head.py --traces {out_path}",
          flush=True)
    print(f"    python scripts/train_recoverability_head.py  --traces {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())