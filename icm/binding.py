"""Ternary HDC binding module -- ICM Phase 3 (plan §6.3, the plan's Phase-3 build).

Operations (Gayler/Kanerva VSA, bipolar {-1,+1} for self-inverse bind;
ternary {-1,0,+1} for value-carrying slots):

    project(x)     -- random projection (Probe A: RP-to-10k + sign == full
                      precision gold recall) to the HDC dimensionality
    bind(a, b)     -- elementwise multiply (SELF-INVERSE for bipolar)
    unbind(a, b)   -- bind again (bipolar self-inverse)
    bundle(vectors)-- elementwise majority/mode (superposition)
    similarity(a,b)-- cosine

Probe A findings baked in:
    RP-to-d=10k + sign == full-precision gold recall (0.80/0.88)
    d=384 sign degraded-not-collapsed -> default dim = 10,000
    band sweet spot eps=0.05 (38% zeros) -> projection uses sign, not band

Usage:
    hv = HDCSpace(dim=10_000)
    q_hv = hv.project(query_embedding)       # 384-d float -> 10k-d bipolar
    m_hv = hv.project(memory_embedding)
    bound = hv.bind(q_hv, m_hv)              # role-filler composition
    freed = hv.unbind(bound, q_hv)           # recovers m_hv (bipolar)
    score = hv.similarity(freed, m_hv)       # cos ~ 1.0 (bipolar roundtrip)
"""

from __future__ import annotations

import os
import time

import numpy as np


class HDCSpace:
    """Bipolar HDC space with random projection and self-inverse bind."""

    def __init__(self, dim: int = 10_000, seed: int = 42):
        self.dim = dim
        rng = np.random.default_rng(seed)
        # random projection matrix: (input_dim, hdc_dim), ±1/sqrt(input_dim)
        # the actual input_dim is set lazily on first project call (384-d bge)
        self._Rp = None
        self._seed = seed

    def _ensure_Rp(self, input_dim: int):
        if self._Rp is None:
            rng = np.random.default_rng(self._seed)
            self._Rp = rng.choice([-1.0, 1.0], size=(input_dim, self.dim)) / np.sqrt(input_dim)
            self._Rp = self._Rp.astype(np.float32)

    def project(self, emb: np.ndarray) -> np.ndarray:
        """Float embedding -> bipolar HDC vector (sign of random projection)."""
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self._ensure_Rp(emb.shape[1])
        z = emb @ self._Rp
        return np.where(z >= 0, 1.0, -1.0).astype(np.float32)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Elementwise multiply (bipolar self-inverse)."""
        return a * b

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Recover the other operand (bipolar: bind again)."""
        return bound * key

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Elementwise majority (bipolar superposition)."""
        stack = np.stack(vectors)
        sums = stack.sum(axis=0)
        return np.where(sums >= 0, 1.0, -1.0).astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (bipolar: dot / dim)."""
        return float(np.dot(a, b) / self.dim)

    def bundle_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine for non-bipolar (e.g. bundled) vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class TernarySpace(HDCSpace):
    """Ternary {-1,0,+1} variant with 0-propagation bind and consensus unbind.

    NOT self-inverse: 0 slots destroy information under multiply. Unbind uses
    the key's nonzero entries only (partial recovery). For full self-inverse
    use bipolar. The ternary variant is for value-carrying slots where 0 =
    "no information" is semantically meaningful (the plan's "absent role")."""

    def project(self, emb: np.ndarray, band_eps: float = 0.05) -> np.ndarray:
        """Float embedding -> ternary via sign + band (Probe A: eps=0.05 sweet spot)."""
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self._ensure_Rp(emb.shape[1])
        z = emb @ self._Rp
        out = np.zeros_like(z, dtype=np.float32)
        out[z >= band_eps] = 1.0
        out[z <= -band_eps] = -1.0
        return out

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ternary bind: 0 in either operand -> 0 (absent role = absent binding)."""
        return a * b

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Partial unbind: divide where key != 0, else 0 (data destroyed)."""
        out = np.zeros_like(bound)
        mask = key != 0
        out[mask] = bound[mask] / key[mask]
        return out