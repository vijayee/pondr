"""E0-v3 lens run -- the taxon-split E2 (value-kind decides the leg).

SEMANTIC arm: predict probe latency from the question (rows anchor given in
    text; bge embedding + numeric feature channel). Success = |pred-real| <= 2
    (the noise band). Baselines: majority-median band-hit, and the TRUE LAW's
    band-hit (the ceiling any head approaches).
SYMBOLIC arm (control): predict backup time (no law exists) the same way.
    Expected: chance; taxonomy confirmation.

Grading is BAND-based on the semantic arm because +-2 noise makes the exact
token under-determined by the law -- exact tokens remain the SYMBOLIC leg's
property. This is the honest metric split the taxonomy implies.
"""

from __future__ import annotations

import io
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from e0_v3_corpus import (N_SESSIONS, build, episode_text,  # noqa: E402
                          question_semantic, question_symbolic)


def main() -> int:
    torch.manual_seed(23)
    data = build(23, N_SESSIONS)
    band = 2.0

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    q_sem = [question_semantic(s) for s in data]
    q_sym = [question_symbolic(s) for s in data]
    embs_sem = np.asarray(st.encode(q_sem, batch_size=64, show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    embs_sym = np.asarray(st.encode(q_sym, batch_size=64, show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    print(f"[embed] {len(data)} questions x 2 arms", flush=True)

    def feats(emb: np.ndarray, s) -> np.ndarray:
        h, m = s.ts.split(":")
        rows_f = s.rows / 3200.0
        return np.concatenate([emb, [rows_f, int(h) / 24.0, int(m) / 60.0]]).astype(np.float32)

    X_sem = np.stack([feats(embs_sem[i], s) for i, s in enumerate(data)])
    X_sym = np.stack([feats(embs_sym[i], s) for i, s in enumerate(data)])
    y_lat = np.array([s.lat for s in data], dtype=np.float32)
    y_ts_min = np.array([int(s.ts.split(":")[0]) * 60 + int(s.ts.split(":")[1])
                         for s in data], dtype=np.float32)
    y_sym_band = 20.0  # +-20 min band for the symbolic control

    rng = np.random.default_rng(23)
    perm = rng.permutation(N_SESSIONS)
    tr, ev = perm[:240], perm[240:]

    def head(X, y, epochs: int = 400):
        model = nn.Sequential(nn.Linear(X.shape[1], 64), nn.GELU(), nn.Linear(64, 1))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr]).unsqueeze(1)
        for _ in range(epochs):
            idx = torch.randperm(len(tr))[:96]
            loss = nn.functional.smooth_l1_loss(model(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X)).squeeze(1).numpy()
        return pred

    pred_lat = head(X_sem, y_lat)
    pred_sym = head(X_sym, y_ts_min)

    def band_hit(pred: np.ndarray, true: np.ndarray, tol: float, idx: list) -> float:
        return float(np.mean(np.abs(pred[idx] - true[idx]) <= tol))

    # baselines
    med_lat = float(np.median(y_lat[tr]))
    med_sym = float(np.median(y_ts_min[tr]))
    law_pred = np.array([8 + s.rows / 40 for s in data])
    out = {
        "split": {"train": len(tr), "eval": len(ev)},
        "semantic_lat": {
            "band_hits_within_2": round(band_hit(pred_lat, y_lat, band, ev), 4),
            "majority_baseline": round(band_hit(np.full(N_SESSIONS, med_lat), y_lat, band, ev), 4),
            "true_law_ideal": round(band_hit(law_pred, y_lat, band, ev), 4),
        },
        "symbolic_ts_control": {
            "band_hits_within_20min": round(band_hit(pred_sym, y_ts_min, y_sym_band, ev), 4),
            "majority_baseline": round(band_hit(np.full(N_SESSIONS, med_sym), y_ts_min, y_sym_band, ev), 4),
            "note": "no law exists; chance = majority-band share; a trained head cannot exceed it",
        },
    }
    path = "data/_scratch/icm/lens_v3_taxon.json"
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())