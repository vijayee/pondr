"""Probe: does a SHARED-projection bilinear head clear the synthetic gate?

The dual-tower (separate doc_proj/query_proj) plateaus at top3=0.583 on the
similarity synthetic. Both doc and query are bge-small vectors in the SAME
space, so a single shared projection W is better-conditioned (no rotation
gauge-freedom) and represents cosine exactly (W=I). Sweeps lr/epochs/cap to
find a config that clears the gate, so the test + real-data configs can encode
working hyperparameters rather than guesses.

Untracked probe -- NOT committed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DOC_DIM = 384
QUERY_DIM = 384
SLOT_DIM = 256
PROJ_DIM = 128


class SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DOC_DIM, PROJ_DIM)
        self.yt_sidepath = nn.Sequential(nn.Linear(SLOT_DIM, 64), nn.GELU(),
                                        nn.Linear(64, 1))
        self.scale = 1.0 / math.sqrt(PROJ_DIM)
        self.bias = nn.Parameter(torch.zeros(1))

    def logits(self, slot_y, doc_emb, query_emb):
        # slot_y [K,256], doc_emb [K,384], query_emb [384]
        dp = self.proj(doc_emb)
        qp = self.proj(query_emb).unsqueeze(0).expand_as(dp)
        sim = (dp * qp).sum(-1, keepdim=True) * self.scale
        yt = self.yt_sidepath(slot_y)
        return (sim + yt + self.bias).squeeze(-1)


def synth(n=60, k=15, seed=7):
    rng = np.random.default_rng(seed)
    traces = []
    for qi in range(n):
        q = rng.standard_normal(QUERY_DIM).astype(np.float32)
        doc = [q + 0.3 * rng.standard_normal(DOC_DIM).astype(np.float32)]
        ys = [0.3 * rng.standard_normal(SLOT_DIM).astype(np.float32)]
        lab = [1]
        for _ in range(k - 1):
            doc.append(0.3 * rng.standard_normal(DOC_DIM).astype(np.float32))
            ys.append(0.3 * rng.standard_normal(SLOT_DIM).astype(np.float32))
            lab.append(0)
        traces.append({
            "query_emb": torch.from_numpy(q),
            "slots_y": torch.from_numpy(np.stack(ys)).float(),
            "slots_doc_emb": torch.from_numpy(np.stack(doc)).float(),
            "labels": torch.tensor(lab, dtype=torch.long),
        })
    return traces


def wilson(p, n):
    z = 1.96
    if n == 0:
        return [0.0, 1.0]
    num = p + z * z / (2 * n)
    den = 1 + z * z / n
    cent = num / den
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [max(0.0, cent - adj), min(1.0, cent + adj)]


def eval_head(head, val):
    recalls, hits = [], 0
    for r in val:
        with torch.no_grad():
            logits = head.logits(r["slots_y"], r["slots_doc_emb"], r["query_emb"])
        gold = (r["labels"] > 0).nonzero(as_tuple=True)[0].tolist()
        top = set(logits.topk(min(3, logits.shape[0])).indices.tolist())
        ng = len(gold)
        ngin = sum(1 for i in gold if i in top)
        recalls.append(ngin / ng)
        if ngin == ng:
            hits += 1
    mean = sum(recalls) / len(recalls)
    hr = hits / len(recalls)
    ci = wilson(hr, len(recalls))
    return mean, hr, ci


def run(epochs, lr, cap, seed=0):
    torch.manual_seed(seed)
    tr = synth(60, 15, 7)
    n_val = 12
    val = tr[:n_val]
    train = tr[n_val:]
    head = SharedHead()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    npos = sum(int(r["labels"].sum()) for r in train)
    nneg = sum(int((1 - r["labels"]).sum()) for r in train)
    pw = min(nneg / max(npos, 1), cap)
    pw_t = torch.tensor([pw])
    rng = np.random.default_rng(seed)
    best = (0, 0, [0, 1])
    for ep in range(epochs):
        head.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        opt.zero_grad()
        for k, qi in enumerate(order):
            r = train[qi]
            logits = head.logits(r["slots_y"], r["slots_doc_emb"], r["query_emb"])
            loss = F.binary_cross_entropy_with_logits(
                logits, r["labels"].float(), pos_weight=pw_t) / 4
            loss.backward()
            if (k + 1) % 4 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        head.eval()
        m, hr, ci = eval_head(head, val)
        if (m, hr) > (best[0], best[1]):
            best = (m, hr, ci)
    go = best[0] >= 0.6 and best[2][0] > 0.5
    print(f"  epochs={epochs} lr={lr} cap={cap}: best top3={best[0]:.3f} "
          f"hit={best[1]:.2f} ci=[{best[2][0]:.2f},{best[2][1]:.2f}] "
          f"{'GO' if go else 'no-go'}")


if __name__ == "__main__":
    for cap in (3.0, 14.0):
        for lr in (3e-4, 1e-3):
            for ep in (20, 40, 60):
                run(ep, lr, cap)