"""Serve the Ponder Engine on the TRAINED SSM/JEPA backbone + RetrievalGate.

Thin CLI over ``src.runtime.build_ponder`` -- the runtime entrypoint the
SSM/JEPA consistency audit (2026-07-15) found missing. Loads the trained Phase
2a backbone + trained Phase 2b gate, wires the real bge embedder + the local
8B Bonsai endpoint, and -- by default -- live-encodes each exchange as a
learnable episode (GLiNER on CUDA with an OOM-safe CPU fallback).

Usage (one-shot, against the running 8B Bonsai server):
    python scripts/serve_ponder.py --query "Why did we choose Postgres?"

Usage (interactive REPL; blank line or Ctrl-C to exit):
    python scripts/serve_ponder.py

Flags:
    --backbone PATH        Phase 2a backbone checkpoint
    --gate PATH            Phase 2b RetrievalGate checkpoint (best.pt)
    --db PATH              WaveDB memory DB path
    --embed-source NAME    on-demand (real bge) | stub (shape-only smoke)
    --device NAME          backbone+gate device (auto|cpu|cuda)
    --gliner-device NAME   GLiNER device (auto|cpu|cuda); OOM-safe fallback
    --gliner-timing        log per-stage GLiNER extraction timing
    --no-live-encode       do not persist exchanges (skip the encoder + GLiNER)
    --user-id ID           user the encoder attributes episodes to
    --query TEXT           one-shot query; omit for the interactive REPL
    --bonsai-endpoint URL  override the Bonsai LLM endpoint

Pre-warm the Bonsai server first (PTX-JIT cold-start ~18s/shape; see memory
``hippo-bonsai-local-server``). Live-encode (default on) loads GLiNER; the
first exchange also pays the GLiNER model-download/warm-up cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import DEFAULT_BACKBONE_PATH, DEFAULT_GATE_PATH, build_ponder  # noqa: E402


def _print_result(res: dict) -> None:
    """Print the query result + the self-chat loop transcript + persistence."""
    end_state = res.get("end_state_plan")
    end_state_name = getattr(end_state, "end_state", "?") if end_state else "?"
    print(f"\n[end-state] {end_state_name}")
    response = res.get("response") or ""
    # Model output may be UTF-8; stdout is reconfigured to utf-8 in main().
    print(f"\n{response}")
    fb = res.get("feedback_collected")
    if fb:
        print(f"\n[feedback_collected] {fb}", file=sys.stderr)
    # Self-chat tool-loop transcript (surfaced only when the loop ran).
    if "loop_tool_messages" in res:
        names = [c.get("name") for c in res.get("loop_collected", [])]
        print(f"[loop] exhausted={res.get('loop_exhausted')} tools={names}",
              file=sys.stderr)
    pid = res.get("persisted_episode_id")
    if pid:
        print(f"[live-encode] persisted episode {pid}", file=sys.stderr)
    # Per-stage GLiNER timing is already logged to stderr by the extractor when
    # --gliner-timing is on (the flag is passed through to build_ponder), so
    # nothing extra is printed here.


def main() -> int:
    # Model output is UTF-8; the Windows console default (cp1252) would crash
    # on non-ASCII. Reconfigure stdout/stderr to UTF-8 with replacement so a
    # live Bonsai response never kills the REPL. Our own strings stay ASCII.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover - non-CPython / fixed stream
        pass

    p = argparse.ArgumentParser(description="Serve the Ponder Engine on the trained models")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH,
                   help="Phase 2a backbone checkpoint (backbone_final.pt)")
    p.add_argument("--gate", default=DEFAULT_GATE_PATH,
                   help="Phase 2b RetrievalGate checkpoint (best.pt)")
    p.add_argument("--db", default=None,
                   help="WaveDB memory DB path (default: config.db_path)")
    p.add_argument("--embed-source", default="on-demand", choices=["on-demand", "stub"],
                   help="on-demand = real bge-small; stub = shape-only smoke")
    p.add_argument("--device", default="auto", help="backbone+gate device: auto|cpu|cuda")
    p.add_argument("--gliner-device", default="auto",
                   help="GLiNER device: auto|cpu|cuda (OOM-safe CPU fallback)")
    p.add_argument("--gliner-timing", action="store_true",
                   help="log per-stage GLiNER extraction timing to stderr")
    p.add_argument("--no-live-encode", action="store_true",
                   help="do not persist exchanges (skip the encoder + GLiNER)")
    p.add_argument("--user-id", default="ponder", help="user the encoder attributes episodes to")
    p.add_argument("--query", default=None, help="one-shot query; omit for the interactive REPL")
    p.add_argument("--bonsai-endpoint", default=None, help="override the Bonsai LLM endpoint")
    args = p.parse_args()

    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone checkpoint not found at {backbone_path}", file=sys.stderr)
        return 1
    gate_path = Path(args.gate)
    if not gate_path.exists():
        print(f"ERROR: gate checkpoint not found at {gate_path}", file=sys.stderr)
        return 1

    print(f"[load] backbone={backbone_path}", file=sys.stderr)
    print(f"[load] gate={gate_path}", file=sys.stderr)
    print(f"[load] live_encode={not args.no_live_encode} "
          f"gliner_device={args.gliner_device}", file=sys.stderr)

    orch = build_ponder(
        args.db,
        backbone_path=str(backbone_path),
        gate_path=str(gate_path),
        embedder_source=args.embed_source,
        bonsai_endpoint=args.bonsai_endpoint,
        device=args.device,
        gliner_device=args.gliner_device,
        gliner_timing=args.gliner_timing,
        live_encode=not args.no_live_encode,
        user_id=args.user_id,
    )

    try:
        if args.query is not None:
            res = orch.query(args.query)
            _print_result(res)
            return 0

        # Interactive REPL. The orchestrator's Working Memory carries cross-query
        # state; we also thread the raw turn pairs as conversation_history so the
        # planner can resolve pronouns across turns.
        print("Ponder REPL. Blank line to exit.", file=sys.stderr)
        history: list[dict] = []
        while True:
            try:
                line = input("you> ")
            except EOFError:
                break
            if not line.strip():
                break
            res = orch.query(line, conversation_history=list(history))
            _print_result(res)
            response = res.get("response")
            if isinstance(response, str) and response.strip():
                history.append({"role": "user", "content": line})
                history.append({"role": "assistant", "content": response})
        return 0
    finally:
        try:
            orch.store.close()
        except Exception as e:  # noqa: BLE001 - never crash on cleanup
            print(f"[close-fail] {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())