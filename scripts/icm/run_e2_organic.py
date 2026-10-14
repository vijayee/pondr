"""E2-organic -- Phase 2 gate slice on the loop-closed correlated world.

1. HARVEST: the correlated-value world (rows -> latency/disk/pruned by law;
   backup-time symbolic) captured into an EpisodicStore as real episodes.
2. FACT-PARSE INTEGRITY: all facts parsed back from episode TEXT by regex
   (the store must be self-sufficient -- no Session objects reach the eval).
3. TAXON LENS on stored records: semantic head (question embedding + numeric
   channel -> latency, band-graded +-2) and symbolic control (-> backup
   minutes; no law, expects chance).
4. DUAL-CONSULT E2 on held-out sessions (28 semantic + 28 symbolic):
     dense-only: consult(question, k=3) -> answer
     dual:       the question's unique anchor token routes to its episode,
                 which answers exactly
   Success = exact gold-token containment; latency per arm recorded.

Run:  python scripts/icm/run_e2_organic.py  (~112 relay calls, ~15 min)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(1)

from e0_v3_corpus import N_SESSIONS, build, episode_text, question_semantic, question_symbolic  # noqa: E402
from icm.consult import consult, consult_block  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

EP_REGEX = re.compile(
    r"\[run (\d{4})\] Ingested a dataset of (\d+) rows\. "
    r"Checkpoint \[deploy\]: probe returned (\d+)ms\. "
    r"Checkpoint \[storage\]: disk at (\d+) percent\. "
    r"Checkpoint \[ci\]: pruned (\d+) stale images\. "
    r"Checkpoint \[backup\]: completed at (\d{2}:\d{2})\.")


def parse_episode(text: str) -> dict | None:
    m = EP_REGEX.search(text)
    if not m:
        return None
    return {"run": int(m.group(1)), "rows": int(m.group(2)), "lat": int(m.group(3)),
            "pct": int(m.group(4)), "pruned": int(m.group(5)), "ts": m.group(6)}


_model = None


def _embedder(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


def ask_retry(policy, question: str, memory_block: str | None) -> tuple[str, float]:
    prompt = (question +
              ' Reply ONLY as JSON {"answer": "<a short phrase>"} containing the specific value.')
    if memory_block:
        prompt += "\n\nEvidence record:\n" + memory_block
    body = {"model": policy.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]},
            "options": {"temperature": 0.2, "num_ctx": 8192}}
    backoff = 10.0
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
            rec = json.loads(resp["message"]["content"])
            return str(rec.get("answer", "")).strip(), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60))
                backoff *= 1.8
                continue
            if attempt >= 2:
                return "", (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
        except Exception:
            if attempt >= 2:
                return "", (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
    return "", 0.0


class BandHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_band_head(X: torch.tensor, y: torch.tensor, tr: torch.tensor,
                    epochs: int = 300) -> BandHead:
    model = BandHead(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        idx = torch.randperm(len(tr))[:96]
        loss = nn.functional.smooth_l1_loss(model(X[tr][idx]), y[tr][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "e2organic"))
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)

    # ---- 1. harvest ----
    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    data = build(7, N_SESSIONS)
    try:
        t0 = time.time()
        for s in data:
            store.capture(session_id=f"run-{s.si:04d}", intent_id=None,
                          text=episode_text(s), episode_id=f"ep-{s.si:04d}",
                          created_ts=t0 + s.si)
        print(f"[harvest] {N_SESSIONS} episodes in {time.time()-t0:.1f}s", flush=True)

        # ---- 2. parse integrity + features ----
        facts_by_id: dict[str, dict] = {}
        for eid in led.ids("episode"):
            rec = led.get("episode", eid)
            f = parse_episode(rec["text"])
            assert f is not None, f"fact-parse failed for {eid}"
            assert f["run"] == int(eid.split("-")[1]), f"run-id mismatch in {eid}"
            facts_by_id[eid] = f
        print(f"[parse] {len(facts_by_id)}/{N_SESSIONS} episodes parsed cleanly", flush=True)

        ids = led.ids("episode")
        rows_np = np.asarray([facts_by_id[i]["rows"] for i in ids], dtype=np.float32)
        tsmin_np = np.asarray([int(facts_by_id[i]["ts"].split(":")[0]) * 60 +
                               int(facts_by_id[i]["ts"].split(":")[1]) for i in ids],
                              dtype=np.float32)
        class _S:
            pass

        def _as_sess(f):
            o = _S(); o.rows = f["rows"]; o.lat = f["lat"]; o.pct = f["pct"]
            o.pruned = f["pruned"]; o.ts = f["ts"]
            return o
        q_texts_sem = [question_semantic(_as_sess(facts_by_id[i])) for i in ids]
        embs_q = np.asarray(_embedder(q_texts_sem), dtype=np.float32)
        num_ch = np.stack([rows_np / 3200.0,
                           np.asarray([int(facts_by_id[i]["ts"].split(":")[0]) for i in ids]) / 24.0,
                           np.asarray([int(facts_by_id[i]["ts"].split(":")[1]) for i in ids]) / 60.0],
                          axis=1).astype(np.float32)
        X = torch.tensor(np.concatenate([embs_q, num_ch], axis=1))
        y_lat = torch.tensor([facts_by_id[i]["lat"] for i in ids], dtype=torch.float32)
        y_tsmin = torch.tensor(tsmin_np)

        rng = np.random.default_rng(23)
        perm = rng.permutation(len(ids))
        tr_idx, ev_idx = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
        ev_ids = [ids[i] for i in ev_idx]
        tr_t = torch.tensor(tr_idx)

        torch.manual_seed(191)  # deterministic heads for reproducible metrics
        sem_head = train_band_head(X, y_lat, tr_t)
        sym_head = train_band_head(X, y_tsmin, tr_t)
        with torch.no_grad():
            pred_lat = sem_head(X).numpy()
            pred_ts = sym_head(X).numpy()

        # lens band metrics on STORED records (held-out). NOTE: pred_* and
        # truth arrays index POSITIONS IN ids (like pred_lat[i]); ev_pos is a
        # rank within the eval subset and must not index them.
        lens_sem_band = float(np.mean([abs(pred_lat[ev_idx[p]] -
                                            facts_by_id[ev_ids[p]]["lat"]) <= 2
                                       for p in range(len(ev_ids))]))
        lens_sym_band = float(np.mean([abs(pred_ts[ev_idx[p]] - y_tsmin[ev_idx[p]].item()) <= 20
                                       for p in range(len(ev_ids))]))
        med_lat = float(np.median(y_lat.numpy()[tr_idx]))
        med_sym = float(np.median(y_tsmin.numpy()[tr_idx]))
        base_sem = float(np.mean([abs(med_lat - facts_by_id[ev_ids[p]]["lat"]) <= 2
                                  for p in range(len(ev_ids))]))
        base_sym = float(np.mean([abs(med_sym - y_tsmin[ev_idx[p]].item()) <= 20
                                  for p in range(len(ev_ids))]))

        # ---- 4. dual-consult E2 on held-out sessions ----
        rows_by_ep = {f"ep-{facts_by_id[i]['run']:04d}": facts_by_id[i]["rows"] for i in ev_ids}
        ts_by_ep = {f"ep-{facts_by_id[i]['run']:04d}": facts_by_id[i]["ts"] for i in ev_ids}
        by_rows = {f["rows"]: f"ep-{f['run']:04d}" for f in facts_by_id.values()}
        by_ts = {f["ts"]: f"ep-{f['run']:04d}" for f in facts_by_id.values()}

        arms = {"dense_sem": [0, 0, []], "dual_sem": [0, 0, []],
                "dense_sym": [0, 0, []], "dual_sym": [0, 0, []]}
        consult_lat = []
        half = len(ev_ids) // 2
        for p, eid in enumerate(ev_ids):
            use_sem = p < half
            f = facts_by_id[eid]
            sess = _as_sess(f)
            if use_sem:
                q = question_semantic(sess)
                gold = str(f["lat"])
                anchor = f["rows"]
                anchor_map = by_rows
            else:
                q = question_symbolic(sess)
                gold = f["ts"]
                anchor = f["ts"]
                anchor_map = by_ts
            payload = consult(store, q, k=3)
            consult_lat.append(payload["latency_ms"])
            d_ans, _ = ask_retry(policy, q, consult_block(payload))
            routed_eid = anchor_map.get(anchor)
            r_block = None
            if routed_eid:
                r_rec = led.get("episode", routed_eid)
                r_block = f"[{r_rec['episode_id']}] {r_rec['text']}"
            r_ans, _ = ask_retry(policy, q, r_block)
            graded = (gold.lower() in d_ans.lower(), gold.lower() in r_ans.lower())
            key = "sem" if use_sem else "sym"
            arms[f"dense_{key}"][0] += int(graded[0])
            arms[f"dense_{key}"][1] += 1
            arms[f"dense_{key}"][2].append(graded[0])
            arms[f"dual_{key}"][0] += int(graded[1])
            arms[f"dual_{key}"][1] += 1
            arms[f"dual_{key}"][2].append(graded[1])
            time.sleep(3.0)
            if p % 8 == 0:
                print(f"[q {p}/{len(ev_ids)}] key={key} dense={graded[0]} dual={graded[1]}",
                      flush=True)

        def rate(arm):
            hit, n, _ = arms[arm]
            return {"success": hit, "n": n, "rate": round(hit / max(1, n), 4)}

        out = {
            "harvest": {"episodes": N_SESSIONS, "parse_integrity": len(facts_by_id) == N_SESSIONS},
            "lens_band_stored": {
                "semantic": {"lens": round(lens_sem_band, 4), "majority_baseline": round(base_sem, 4)},
                "symbolic": {"lens": round(lens_sym_band, 4), "majority_baseline": round(base_sym, 4)}},
            "consult_e2": {"dense_semantic": rate("dense_sem"), "dual_semantic": rate("dual_sem"),
                           "dense_symbolic": rate("dense_sym"), "dual_symbolic": rate("dual_sym")},
            "consult_latency_ms_avg": round(sum(consult_lat) / max(1, len(consult_lat)), 2),
        }
    finally:
        led.close()
    path = os.path.join("data", "_scratch", "icm", "e2_organic_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())