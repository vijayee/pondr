"""ICM Phase 2 smoke -- E2 probe: can a small trained lens discriminate
value-confuser twins where dense embeddings cannot?

Variants (the decision input for the lens input-architecture):
    A: query/episode encoding = bge 384-d
    B: bge + numeric feature channel (anchor hour/minute/half-min bits,
       parsed from the QUERY's ts, and from each episode's own text)

Model: 2-layer MLP projection to a lens space; InfoNCE with same-template
twins as in-batch hard negatives; train/eval split BY SESSION (held-out).
Gate (E2-smoke): rank of the correct session twin among ALL eval sessions
in lens space vs the raw-bge baseline; top-1/top-3 on the eval split.

Deterministic; numpy/torch CPU; no relay calls.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(1)

from e0_v2_corpus import N_SESSIONS, corpus_seed, episode_text, question_for, session_texts  # noqa: E402

DAY_TOKEN_RE = re.compile(r"(\d{2}):(\d{2})")


def num_features(ts: str) -> np.ndarray:
    h, m = ts.split(":")
    return np.array([int(h) / 24.0, int(m) / 60.0], dtype=np.float32)


def ts_of(text: str) -> str | None:
    m = DAY_TOKEN_RE.search(text)
    return m.group(0) if m and ":0" in f"{m.group(0)}" else (m.group(0) if m else None)


def first_ts(text: str) -> str | None:
    m = DAY_TOKEN_RE.search(text)
    return m.group(0) if m else None


def main() -> int:
    torch.manual_seed(11)
    np.random.seed(11)
    values = corpus_seed(7)
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("BAAI/bge-small-en-v1.5")

    questions, q_feats, episode_texts, ep_feats, owners = [], [], [], [], []
    for si in range(N_SESSIONS):
        q, _gold = question_for(si, values)
        ts = first_ts(q)
        assert ts == values[si]["ts"]
        questions.append(q)
        q_feats.append(num_features(ts))
        ep_text = episode_text(si, values, 0) + "\n" + "\n".join(session_texts(si, values))
        episode_texts.append(ep_text)
        ep_feats.append(num_features(values[si]["ts"]))
        owners.append(si)

    embs_q = np.asarray(st.encode(questions, batch_size=64, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    embs_e = np.asarray(st.encode(episode_texts, batch_size=64, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    print(f"[embed] {len(questions)} questions, {len(episode_texts)} episodes", flush=True)

    rng = np.random.default_rng(11)
    perm = rng.permutation(N_SESSIONS)
    train_idx, eval_idx = perm[:240], perm[240:]
    print(f"[split] train={len(train_idx)} eval={len(eval_idx)}", flush=True)

    Xq = torch.tensor(embs_q)
    Xe = torch.tensor(embs_e)
    Fq = torch.tensor(np.stack(q_feats))
    Fe = torch.tensor(np.stack(ep_feats))

    owners_np = np.asarray(owners)
    baseline = raw_baseline(Xq[eval_idx], Xe, owners_np[eval_idx])
    print(f"[raw-bge baseline] eval top1={baseline['top1']:.3f} top3={baseline['top3']:.3f} "
          f"mean_rank={baseline['mean_rank']:.1f} (chance ~{1/N_SESSIONS:.4f})", flush=True)

    report = {}
    for variant, in_dim, fq, fe in (("A", 384, Xq, Xe), ("B", 386, torch.cat([Xq, Fq], 1), torch.cat([Xe, Fe], 1))):
        res = train_and_eval(in_dim, fq, fe, train_idx, eval_idx, owners_np)
        report[f"lens_{variant}"] = res
        print(f"[lens-{variant}] eval top1={res['top1']:.3f} top3={res['top3']:.3f} "
              f"mean_rank={res['mean_rank']:.1f} train_loss={res['final_loss']:.3f}", flush=True)

    out = {"baseline_raw_bge": baseline, **report}
    path = os.path.join("data", "_scratch", "icm", "lens_e2_smoke.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")
    return 0


class Lens(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, out_dim))

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-9)


def train_and_eval(in_dim, Xq, Xe, train_idx, eval_idx, owners,
                   epochs: int = 300, lr: float = 3e-3, out_dim: int = 128) -> dict:
    lens = Lens(max(in_dim, 4), out_dim)
    opt = torch.optim.AdamW(lens.parameters(), lr=lr)
    tq, te = torch.tensor(np.asarray(train_idx)), torch.tensor(np.asarray(eval_idx))
    final_loss = 0.0
    for epoch in range(epochs):
        idx = torch.randperm(len(train_idx))[:64]
        batch = train_idx[idx]
        zq = lens(Xq[batch])
        ze = lens(Xe[batch])
        logits = zq @ ze.T / 0.07  # InfoNCE; twins in batch = hard negatives
        labels = torch.arange(len(batch) if len(batch) > 1 else 2)
        labels = labels[: len(batch)]
        loss = nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss)
    lens.eval()
    with torch.no_grad():
        Ze = lens(Xe).numpy()
        Zq = lens(Xq).numpy()
    sims = (torch.tensor(Zq) @ torch.tensor(Ze).T).numpy()
    top1 = top3 = 0
    ranks = []
    for pos, si in enumerate(eval_idx):
        row = sims[pos]
        order = np.argsort(-row)
        rank = int(list(order).index(si)) + 1
        ranks.append(rank)
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
    return {"top1": top1 / len(eval_idx), "top3": top3 / len(eval_idx),
            "mean_rank": float(np.mean(ranks)), "final_loss": final_loss,
            "epochs": epochs}


def raw_baseline(Xq_eval, Xe, eval_owners) -> dict:
    sims = (Xq_eval @ Xe.T).numpy()
    top1 = top3 = 0
    ranks = []
    for pos, si in enumerate(eval_owners):
        order = np.argsort(-sims[pos])
        rank = int(list(order).index(si)) + 1
        ranks.append(rank)
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
    return {"top1": top1 / len(eval_owners), "top3": top3 / len(eval_owners),
            "mean_rank": float(np.mean(ranks))}

if __name__ == "__main__":
    sys.exit(main())