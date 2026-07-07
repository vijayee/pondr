"""JEPA contrastive loss.

Predict the next state's embedding. Pull the prediction toward the actual next
embedding (positive) and push it away from other embeddings in the batch
(negatives) — the negatives are the anti-collapse mechanism, so we do NOT need
a separate EMA target encoder for the loss itself. (The pre-training loop still
maintains an EMA copy of the backbone as the *target encoder* whose predictions
serve as the positive target, per the doc; this avoids the online predictor
chasing a moving target it itself produces.)

Shapes:
- ``predicted``: ``[batch, pred_dim]`` — the JEPA predictor's output for one step.
- ``actual``:    ``[batch, pred_dim]`` — the real next embedding.
- ``negatives``: ``[num_neg, pred_dim]`` — embeddings to push away from.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def jepa_contrastive_loss(predicted: Tensor, actual: Tensor, negatives: Tensor,
                          temperature: float = 0.1) -> Tensor:
    """JEPA contrastive loss over one step's predictions.

    Positive term: maximize cosine similarity between ``predicted`` and ``actual``
    (so minimize its negative). Negative term: a logsumexp over similarities to
    ``negatives`` scaled by ``temperature`` — pushes ``predicted`` away from the
    negatives. A small MSE on the positive pair is added for faster convergence.
    """
    pos_cos = F.cosine_similarity(predicted, actual, dim=-1)     # [batch]
    pos_loss = -pos_cos.mean()

    neg_cos = F.cosine_similarity(
        predicted.unsqueeze(1),        # [batch, 1, dim]
        negatives.unsqueeze(0),        # [1, num_neg, dim]
        dim=-1,
    ) / temperature                    # [batch, num_neg]
    neg_loss = torch.logsumexp(neg_cos, dim=-1).mean()

    mse = F.mse_loss(predicted, actual)

    return pos_loss + neg_loss + 0.1 * mse


def step_loss(predictions: Tensor, targets: Tensor, mask: Tensor, negatives: Tensor,
              temperature: float = 0.1) -> Tensor:
    """JEPA loss summed over the valid (non-pad) timesteps of a sequence batch.

    Args:
        predictions: ``[batch, seq, pred_dim]`` — predictor output at each step.
        targets:     ``[batch, seq, pred_dim]`` — actual next embedding at each step
                     (already shifted by the caller so ``targets[:, t]`` is the
                     embedding the model should predict from ``inputs[:, :t+1]``).
        mask:        ``[batch, seq]`` bool — True where the timestep is valid.
        negatives:   ``[num_neg, pred_dim]`` — pooled negatives for contrastive.
    Returns:
        scalar mean loss over valid timesteps.
    """
    valid = mask.bool()
    if valid.sum() == 0:
        return predictions.new_zeros(())
    # Flatten valid steps to a single batch for the contrastive loss.
    pred = predictions[valid]                  # [N, dim]
    tgt = targets[valid]                        # [N, dim]
    return jepa_contrastive_loss(pred, tgt, negatives, temperature)