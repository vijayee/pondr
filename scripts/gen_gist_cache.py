"""Parallel pre-fill of the gist teacher cache (the retrain's long pole).

``scripts/train_gist_readout.py --target gist`` calls ``flash_summarize`` inline
and SEQUENTIALLY -- fine for a handful of docs, but ~3.6s/doc over 8k+ docs is
~8h. Teacher generation is embarrassingly parallel (one independent Ollama
request per doc, cached by sha1(content)), so this script fans it out across
``--workers`` threads into the SAME cache file the trainer reads
(``<output-dir>/gist_teacher_cache_v2.json``). The trainer then sees cache hits
and skips straight to training.

Resumable: reads the existing cache first and only requests uncached docs, so an
interrupted run is continued, not restarted. Failures (transient Ollama hiccups)
are retried once; docs that still fail are skipped (the trainer skips docs with
no gist -- a small skip rate is fine over thousands of docs).

Public ERAG content only; teacher = deepseek-flash over pro (memory). No onyx,
no private transcripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_gist_readout import (  # noqa: E402
    ERAG_PATH,
    LLM_MODEL,
    LLM_TIMEOUT,
    OLLAMA_URL,
    _iter_erag_pairs,
)

PROMPT = (
    "Summarize the following document. Capture the key topic, decisions, and "
    "content at a length appropriate to the document -- a short doc gets a short "
    "summary, a dense doc gets a longer one. Reply with only the summary.\n\n"
)


def _one(content: str) -> str | None:
    payload = json.dumps(
        {"model": LLM_MODEL, "prompt": PROMPT + content[:4000],
         "stream": False, "options": {"temperature": 0.2, "num_predict": 256}}
    ).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            resp = json.loads(r.read().decode("utf-8"))
        text = resp.get("response", "").strip()
        return text or None
    except Exception:
        return None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="Parallel pre-fill of the gist teacher cache.")
    ap.add_argument("--max-docs", type=int, default=8200,
                    help="number of ERAG docs to pull (train+val; matches the trainer's need)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output-dir", default="data/gist_retrain")
    ap.add_argument("--erag-path", default=ERAG_PATH)
    ap.add_argument("--retry", type=int, default=1,
                    help="retry transient failures this many times")
    args = ap.parse_args()

    cache_path = Path(args.output_dir) / "gist_teacher_cache_v2.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"[cache] {len(cache)} entries at {cache_path}", flush=True)

    # Pull the first --max-docs ERAG docs (same order the trainer sees).
    docs = [(t, c) for t, c in _iter_erag_pairs(args.erag_path, max_docs=args.max_docs)
            if c and str(c).strip()]
    print(f"[data] {len(docs)} docs pulled from {args.erag_path}", flush=True)

    # Only request docs whose sha1 is not already cached.
    todo = [(t, c) for t, c in docs
            if hashlib.sha1(c.encode("utf-8")).hexdigest() not in cache]
    print(f"[plan] {len(todo)} uncached -> {args.workers} workers "
          f"(skipping {len(docs) - len(todo)} cached)", flush=True)
    if not todo:
        print("[done] cache already covers all pulled docs", flush=True)
        return 0

    lock = threading.Lock()
    done = 0
    ok = 0
    failed: list[str] = []
    t_start = time.time()

    def work(item):
        nonlocal done, ok
        title, content = item
        key = hashlib.sha1(content.encode("utf-8")).hexdigest()
        text = None
        for attempt in range(args.retry + 1):
            text = _one(content)
            if text:
                break
            time.sleep(0.5)
        with lock:
            if text:
                cache[key] = text
                ok += 1
                # Persist incrementally so an interrupt keeps progress.
                cache_path.write_text(json.dumps(cache, ensure_ascii=False),
                                      encoding="utf-8")
            else:
                failed.append(title)
            done += 1
            if done % 100 == 0 or done == len(todo):
                rate = done / (time.time() - t_start)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  [gist] {done}/{len(todo)} ok={ok} fail={done - ok} "
                      f"({rate:.2f} doc/s, eta {eta:.0f}s)", flush=True)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    print(f"[done] {ok}/{len(todo)} summaries cached in {time.time() - t_start:.0f}s; "
          f"{len(failed)} failed (skipped)", flush=True)
    if failed[:5]:
        print(f"  failed titles (first 5): {[t[:40] for t in failed[:5]]}", flush=True)
    print(f"[cache] now {len(cache)} entries at {cache_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())