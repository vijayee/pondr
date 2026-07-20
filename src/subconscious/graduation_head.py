"""Graduation heads: which compressed-out WM slots to promote to LTM.

STRM Phase 2d -- the graduation read-out. When a WM ring slot is about to be
evicted (compressed out), graduation decides whether its content is worth
promoting to long-term memory. Two heads live here:

1. ``GraduationProxyV1`` -- a PARAMETER-FREE heuristic baseline: the time-
   integral of the 2a relevance head's per-slot ``r_i`` stream over the slot's
   lifetime in the ring,

       graduation_score = integral(r_i dt) = sum(r_i * dt)

   A slot whose content was consistently relevant to recent queries accumulates
   a high score and graduates; a slot never touched by a relevant query does
   not. This is the baseline the v2 head must beat (shipped now, no data, no
   training -- the 2a ``r_i`` stream is its only input).

2. ``GraduationHeadV2`` -- a LEARNED classifier (small MLP) over
   ``[state_t_pooled (1536) ; slot_y_t (256) ; llm_signal_onehot (5)]`` that
   predicts ``later_needed``: was this slot referenced by a later
   salience-recall / consumer search AFTER it was compressed out (the
   "would-have-been-needed" signal, labeled by
   ``scripts/generate_graduation_labels.py`` from the replay log). The v2
   head's CODE lands now (module + loader + trainer + CLI + synthetic tests);
   its TRAINING RUN is deferred until ``replay_labeled.jsonl`` has enough
   labeled slots (the replay logger that populates it ships in Step 5).

Both are plain ``nn.Module`` (NOT ``JGSInstance``) -- like the shipped 2b/2c
read-out heads, they READ the WM state + a slot and do NOT step the SSM
(documented deviation from a JGSInstance head; same rationale as
``relevance_head.py``). The v2 head reuses the ``llm_signal`` vocabulary from
``src/memory/forgetting.py`` (``LLM_SIGNAL_MODIFIERS`` -- important, routine,
satisfied, frustration, correction) -- it does NOT invent a new signal
taxonomy. The state projection is the 0a-validated "pooled" rep (per-layer
mean over d_state, 4 layers concatenated -> [1536]), shared with
``RecoverabilityHead``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..memory.forgetting import LLM_SIGNAL_MODIFIERS
from .recoverability_head import STATE_DIM_POOLED

# The ring slot y_t = the step OUTPUT (output_proj readout), 256-d. Matches
# INSTANCE_CONFIGS["working_memory"].output_dim (configs.py); same as
# RelevanceHead.SLOT_DIM.
SLOT_DIM = 256
# The llm_signal one-hot width -- the size of LLM_SIGNAL_MODIFIERS (5:
# important, routine, satisfied, frustration, correction). Reusing the
# forgetting vocabulary keeps the v2 head's signal input consistent with the
# rest of the engine's importance taxonomy.
LLM_SIGNAL_VOCAB: tuple[str, ...] = tuple(LLM_SIGNAL_MODIFIERS.keys())
LLM_SIGNAL_DIM = len(LLM_SIGNAL_VOCAB)
# v2 head input = [pooled state (1536) ; slot y_t (256) ; llm_signal one-hot (5)].
INPUT_DIM = STATE_DIM_POOLED + SLOT_DIM + LLM_SIGNAL_DIM

# v1 proxy graduation threshold. UNCALIBRATED -- the v1 proxy is a heuristic
# baseline the v2 head must beat, not a tuned ship decision. The integral of
# r_i (in [0,1]) over a slot's lifetime: a slot relevant for ~2 turns at r~0.5
# clears 1.0; a slot never relevant stays near 0. Held as a named constant so
# the calibration lever is one obvious site (Phase 4 / a later sweep sets it
# against the v2 head's recall).
DEFAULT_GRADUATION_THRESHOLD = 1.0


# ── v1: parameter-free ∫r_i dt proxy ──

class GraduationProxyV1(nn.Module):
    """Parameter-free graduation baseline: ``graduation_score = integral(r_i dt)``.

    The time-integral of the 2a relevance head's per-slot ``r_i`` stream over
    the slot's lifetime in the ring. A slot graduates (promotes to LTM) when
    its accumulated relevance clears ``threshold``. No parameters, no
    checkpoint, no training -- the 2a ``r_i`` stream is the only input. This is
    the heuristic baseline the v2 head must beat on held-out replay recall.
    """

    def __init__(self, threshold: float = DEFAULT_GRADUATION_THRESHOLD,
                 dt: float = 1.0) -> None:
        super().__init__()
        if dt <= 0:
            raise ValueError(f"GraduationProxyV1: dt must be > 0 (got {dt})")
        self.threshold = float(threshold)
        self.dt = float(dt)

    def integrate_relevance(self, r_stream: list[float] | Tensor) -> float:
        """``graduation_score = sum(r_i * dt)`` over the slot's ``r_i`` stream.

        ``r_stream`` is the per-turn relevance ``r_i in [0,1]`` the 2a head
        emitted for this slot over its lifetime in the ring (oldest-first or
        newest-first -- the integral is order-independent). Empty stream -> 0.0
        (a slot never scored is not graduated).
        """
        if isinstance(r_stream, Tensor):
            rs = r_stream.to(torch.float32).reshape(-1).tolist()
        else:
            rs = [float(r) for r in r_stream]
        return sum(r * self.dt for r in rs)

    def graduation_score(self, r_stream: list[float] | Tensor) -> float:
        """Alias for ``integrate_relevance`` (the score the threshold acts on)."""
        return self.integrate_relevance(r_stream)

    def graduate(self, r_stream: list[float] | Tensor) -> bool:
        """``graduation_score(r_stream) >= threshold`` -> promote to LTM."""
        return self.integrate_relevance(r_stream) >= self.threshold

    def forward(self, r_stream: list[float] | Tensor) -> float:
        """Return the graduation score (not the boolean -- the score is the
        v2-beat metric; callers compare to ``threshold`` for the decision)."""
        return self.integrate_relevance(r_stream)


# ── llm_signal one-hot encoding (shared by the v2 head + its trainer) ──

def encode_llm_signal(signal: str | None) -> Tensor:
    """One-hot encode an ``llm_signal`` string -> ``[LLM_SIGNAL_DIM]`` float.

    ``signal`` is one of ``LLM_SIGNAL_VOCAB`` (the ``LLM_SIGNAL_MODIFIERS``
    keys: important, routine, satisfied, frustration, correction). ``None`` /
    unknown -> the all-zeros vector (a missing signal, not a silent mis-wire:
    the v2 head learns to treat absence as its own evidence). Used by the v2
    head's ``predict`` and by the trainer when building the input feature.
    """
    v = torch.zeros(LLM_SIGNAL_DIM, dtype=torch.float32)
    if signal is None:
        return v
    if not isinstance(signal, str):
        return v
    idx = LLM_SIGNAL_VOCAB.index(signal) if signal in LLM_SIGNAL_VOCAB else -1
    if idx >= 0:
        v[idx] = 1.0
    return v


# ── v2: learned classifier (training deferred) ──

class GraduationHeadV2(nn.Module):
    """Learned ``later_needed`` classifier over ``[state_t ; y_t ; llm_signal]``.

    A small MLP ``Linear(INPUT_DIM, 128) -> GELU -> Linear(128, 1)`` whose
    sigmoid output is ``P(later_needed | state_t, slot_y_t, llm_signal)`` --
    the probability this slot would be referenced by a later salience-recall /
    consumer search after it is compressed out (the "would-have-been-needed"
    signal). Inputs per slot: the pooled WM state ``state_t`` (1536-d, the
    0a-validated rep shared with ``RecoverabilityHead``), the slot's recurrent
    readout ``y_t`` (256-d), and the ``llm_signal`` one-hot (5-d, the
    ``forgetting.LLM_SIGNAL_MODIFIERS`` vocabulary). The CODE lands now; the
    TRAINING RUN is deferred until ``replay_labeled.jsonl`` has enough labeled
    slots (the replay logger ships in Step 5).
    """

    def __init__(
        self,
        state_dim_pooled: int = STATE_DIM_POOLED,
        slot_dim: int = SLOT_DIM,
        llm_signal_dim: int = LLM_SIGNAL_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim_pooled = int(state_dim_pooled)
        self.slot_dim = int(slot_dim)
        self.llm_signal_dim = int(llm_signal_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = self.state_dim_pooled + self.slot_dim + self.llm_signal_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    # ── prediction ──

    def predict(self, state_pooled: Tensor, slot_y: Tensor,
                llm_signal_onehot: Tensor) -> Tensor:
        """``P(later_needed)`` -> ``[batch, 1]`` in ``[0, 1]``.

        ``state_pooled`` ``[batch, 1536]`` (from ``RecoverabilityHead``'s
        ``project_state``), ``slot_y`` ``[batch, 256]``, ``llm_signal_onehot``
        ``[batch, 5]`` (from ``encode_llm_signal``). All three broadcast from
        1-D. Concatenated along the feature dim and pushed through the MLP +
        sigmoid.
        """
        s = state_pooled.to(torch.float32)
        y = slot_y.to(torch.float32)
        sig = llm_signal_onehot.to(torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)
        bsz = max(s.shape[0], y.shape[0], sig.shape[0])
        for name, t in (("state_pooled", s), ("slot_y", y),
                        ("llm_signal_onehot", sig)):
            if t.shape[0] not in (1, bsz):
                raise ValueError(
                    f"GraduationHeadV2: {name} batch {t.shape[0]} is "
                    f"incompatible with the others (max {bsz}) -- pass one "
                    f"row to broadcast"
                )
        s = s.expand(bsz, -1)
        y = y.expand(bsz, -1)
        sig = sig.expand(bsz, -1)
        if s.shape[1] != self.state_dim_pooled:
            raise ValueError(
                f"GraduationHeadV2: state_pooled dim {s.shape[1]} != "
                f"self.state_dim_pooled {self.state_dim_pooled}"
            )
        if y.shape[1] != self.slot_dim:
            raise ValueError(
                f"GraduationHeadV2: slot_y dim {y.shape[1]} != "
                f"self.slot_dim {self.slot_dim}"
            )
        if sig.shape[1] != self.llm_signal_dim:
            raise ValueError(
                f"GraduationHeadV2: llm_signal_onehot dim {sig.shape[1]} != "
                f"self.llm_signal_dim {self.llm_signal_dim}"
            )
        x = torch.cat([s, y, sig], dim=1)
        return torch.sigmoid(self.mlp(x))

    forward = predict

    # ── load ──

    @classmethod
    def from_state_dict(
        cls,
        sd: dict,
        state_dim_pooled: int = STATE_DIM_POOLED,
        slot_dim: int = SLOT_DIM,
        llm_signal_dim: int = LLM_SIGNAL_DIM,
        hidden_dim: int = 128,
    ) -> "GraduationHeadV2":
        """Build a head and load a raw ``state_dict`` into it.

        Strict: a shape mismatch from a different state_dim_pooled/slot_dim/
        llm_signal_dim/hidden_dim is a hard error, not a silent mis-wire.
        """
        head = cls(state_dim_pooled=state_dim_pooled, slot_dim=slot_dim,
                   llm_signal_dim=llm_signal_dim, hidden_dim=hidden_dim)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"graduation head v2 state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        head.eval()
        return head


def load_graduation_head(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> GraduationHeadV2:
    """Load a trained GraduationHeadV2 checkpoint -> ready-to-serve module.

    The checkpoint is ``{"head": state_dict, "state_dim_pooled": int,
    "slot_dim": int, "llm_signal_dim": int, "hidden_dim": int, "go": bool,
    ...}`` (see ``graduation_training.fit_graduation``). Like the 2b/2c/2a
    loaders this takes NO backbone -- the head reads WM state + a slot at
    serve. The dims are read before construction so a checkpoint fit on
    different dims loads into a matching head. Moves to the resolved device,
    eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        state_dim_pooled = int(ckpt.get("state_dim_pooled", STATE_DIM_POOLED))
        slot_dim = int(ckpt.get("slot_dim", SLOT_DIM))
        llm_signal_dim = int(ckpt.get("llm_signal_dim", LLM_SIGNAL_DIM))
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        sd = ckpt["head"] if "head" in ckpt else ckpt
    else:
        state_dim_pooled, slot_dim, llm_signal_dim, hidden_dim = (
            STATE_DIM_POOLED, SLOT_DIM, LLM_SIGNAL_DIM, 128)
        sd = ckpt
    head = GraduationHeadV2.from_state_dict(
        sd, state_dim_pooled=state_dim_pooled, slot_dim=slot_dim,
        llm_signal_dim=llm_signal_dim, hidden_dim=hidden_dim,
    )
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()