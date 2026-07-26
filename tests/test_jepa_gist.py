"""Unit tests for the JEPA-latent gist objective (``jepa_gist.py``).

CPU-runnable, self-contained (no skip, no external checkpoint, no bge, no teacher
LLM). Exercises the four things the JEPA pivot depends on before any real training
runs:

1. Shape: ``encode`` -> per-layer encoder states; ``predict_latent`` -> a
   ``[b, latent_dim]`` L2-normalized vector (the gist latent in bge space). The
   encoder is frozen for real on the probe/eval path and the predictor is the only
   trainable surface; the predictor reads the pooled encoder STATE, not the
   encoder block output ``x``.
2. Grad isolation (frozen path): a backward through the latent loss produces NO
   encoder grads and DOES produce predictor grads -- the mirror of the probe's
   isolation guarantee.
3. Grad flow (retrain path): with ``freeze_encoder=False`` and the grad-flowing
   encode, a backward through the latent loss reaches BOTH the predictor AND the
   encoder -- the load-bearing property the retrain depends on (gradient pressure
   from the JEPA-latent loss flows back through the predictor -> encoder recurrence
   to reshape the continuation-state into a gist-shaped state).
4. Save/load roundtrip: a retrain checkpoint (``save_encoder=True``) carries the
   retrained encoder in the JEPA ckpt, and ``load_jepa_gist`` restores it WITHOUT a
   separate encoder checkpoint; the loaded encoder matches exactly and is re-frozen
   for eval.
5. **Swap-follows-state (the load-bearing gate, deterministic):** on a toy
   (doc -> doc-specific target latent) corpus, the predicted latent follows the
   STATE -- swapping states swaps the predicted latent. This is the JEPA equivalent
   of ``test_swap_control_follows_state``: a random frozen encoder's state differs
   between two distinct docs, so the predictor can learn to map each state to its
   doc-specific target latent, and swapping the state swaps the prediction.
6. The reused ``jepa_contrastive_loss`` accepts the predictor's output shape and
   returns a finite scalar (smoke).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.subconscious.jepa_gist import (
    JEPAGistModel,
    LatentPredictorConfig,
    pool_encoder_states,
)
from src.subconscious.token_lm import LMConfig, SSMLanguageModel


# Special-token ids are fixed by the tokenizer wrapper (PAD=0, BOS=1, EOS=2).
PAD, BOS, EOS = 0, 1, 2


def _enc_cfg(**kw) -> LMConfig:
    base = dict(vocab=32, d_model=16, n_layers=2, d_state=4, seq_len=8)
    base.update(kw)
    return LMConfig(**base)


def _latent_cfg(**kw) -> LatentPredictorConfig:
    base = dict(latent_dim=8, hidden=16, n_mlp_layers=2, pool="mean_d_state_concat_layers")
    base.update(kw)
    return LatentPredictorConfig(**base)


def _build(**kw) -> JEPAGistModel:
    enc = SSMLanguageModel(_enc_cfg())
    return JEPAGistModel(enc, _latent_cfg(**kw))


# --------------------------------------------------------------------- shapes
def test_encode_predict_latent_shape():
    m = _build()
    doc = torch.randint(3, 32, (1, 6))
    states = m.encode(doc)
    # encode returns one state per encoder layer, shape [b, d_state, d_model_enc].
    assert len(states) == m.encoder.config.n_layers
    assert states[0].shape == (1, 4, 16)

    pred = m.predict_latent(states)
    # predict_latent returns a [b, latent_dim] L2-normalized vector.
    assert pred.shape == (1, 8)
    assert torch.allclose(pred.norm(dim=-1), torch.ones(1), atol=1e-5), \
        "predict_latent output must be L2-normalized"


def test_pool_encoder_states_shape():
    """pool_encoder_states: mean over d_state per layer, concat all layers."""
    # 2 layers, each [b=2, d_state=4, d_model_enc=16] -> [2, 2*16=32]
    states = [torch.randn(2, 4, 16) for _ in range(2)]
    pooled = pool_encoder_states(states)
    assert pooled.shape == (2, 32)


def test_encoder_frozen_predictor_trainable():
    m = _build()
    assert all(not p.requires_grad for p in m.encoder.parameters()), \
        "encoder must be frozen on the probe/eval path"
    assert any(p.requires_grad for p in m.predictor.parameters()), \
        "predictor must be trainable"
    # The trainable surface is ONLY the predictor (the encoder is frozen): every
    # trainable param belongs to the predictor, and its count matches
    # ``trainable_parameters``.
    pred_n = sum(p.numel() for p in m.predictor.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable == pred_n, "the only trainable params should be the predictor's"
    assert m.trainable_parameters() == sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )


def test_predictor_reads_states_not_x():
    """The predictor reads the pooled encoder STATE, not the encoder block output
    ``x``. Verify by shape: pool_encoder_states consumes the [b, d_state, d_model]
    recurrent state (NOT the [b, seq, d_model] block output) and the predictor
    forward accepts it.
    """
    m = _build()
    doc = torch.randint(3, 32, (2, 5))
    enc_states = m.encode(doc)
    pooled = pool_encoder_states(enc_states)
    # Pooled state is [b, n_layers * d_model_enc] = [2, 2*16=32] -- NOT [b, seq, d].
    assert pooled.shape == (2, 32)
    pred = m.predictor(enc_states)
    assert pred.shape == (2, 8)


# ------------------------------------------------------- grad isolation / flow
def test_predictor_grad_does_not_reach_encoder():
    """Frozen path: a backward through the latent loss must not produce grads in
    the frozen encoder (the encode path is detached + requires_grad=False)."""
    m = _build()
    doc = torch.randint(3, 32, (2, 5))
    enc_states = m.encode(doc)  # no_grad=True (default) -> detached
    pred = m.predict_latent(enc_states)
    target = F.normalize(torch.randn(2, 8), p=2, dim=-1)
    loss = F.mse_loss(pred, target)
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in m.encoder.parameters()), \
        "encoder params must receive no grad on the frozen path"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.predictor.parameters()), \
        "predictor params must receive grad on the frozen path"


def test_encoder_receives_grad_when_unfrozen():
    """Retrain path: with ``freeze_encoder=False`` and the grad-flowing encode
    (``no_grad=False``), a backward through the latent loss must reach BOTH the
    predictor AND the encoder. This is the inverted mirror of
    ``test_predictor_grad_does_not_reach_encoder`` and is the load-bearing property
    the retrain depends on: gradient pressure from the JEPA-latent loss flows back
    through the predictor -> encoder recurrence to reshape the continuation-state
    into a gist-shaped state.
    """
    enc = SSMLanguageModel(_enc_cfg())
    m = JEPAGistModel(enc, _latent_cfg(), freeze_encoder=False)
    assert any(p.requires_grad for p in m.encoder.parameters()), \
        "encoder must be trainable when freeze_encoder=False"

    doc = torch.randint(3, 32, (2, 5))
    # Grad-flowing forward: latent + lm logits + states all live in the graph.
    pred, lm_logits, enc_states = m.forward(doc, no_grad=False)
    target = F.normalize(torch.randn(2, 8), p=2, dim=-1)
    # JEPA-latent loss + a small LM-prior auxiliary (the anti-collapse term).
    latent_loss = F.mse_loss(pred, target)
    lm_loss = F.cross_entropy(
        lm_logits[:, :-1, :].reshape(-1, 32),
        doc[:, 1:].reshape(-1),
    )
    loss = latent_loss + 0.1 * lm_loss
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.parameters()), \
        "encoder params must receive grad on the retrain path"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.predictor.parameters()), \
        "predictor params must receive grad on the retrain path"


# --------------------------------------------------- save/load roundtrip (retrain)
def test_save_load_roundtrips_retrained_encoder(tmp_path):
    """A retrain checkpoint (``save_encoder=True``) carries the retrained encoder
    in the JEPA ckpt, and ``load_jepa_gist`` restores it WITHOUT a separate encoder
    checkpoint (the retrain path). The loaded encoder's state_dict must match what
    was saved, and the loader must re-freeze the encoder for eval.
    """
    from src.subconscious.jepa_gist import load_jepa_gist, save_jepa_checkpoint

    enc = SSMLanguageModel(_enc_cfg())
    m = JEPAGistModel(enc, _latent_cfg(), freeze_encoder=False)
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(0.01 * torch.randn_like(p))

    ckpt_path = tmp_path / "jepa_gist.pt"
    tok_path = tmp_path / "tok.json"
    save_jepa_checkpoint(ckpt_path, m, step=42, encoder_ref="retrained",
                        save_encoder=True)
    assert ckpt_path.exists(), "checkpoint must be written"
    # The retrain checkpoint must carry the encoder key (the retrain signature).
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "encoder" in raw, "retrain checkpoint must contain an 'encoder' key"
    assert "predictor" in raw, "checkpoint must contain a 'predictor' key"

    # Load WITHOUT a separate encoder checkpoint (retrain path: the encoder lives
    # in the JEPA ckpt). device=cpu, dtype=float32 keeps the test CPU/self-contained.
    loaded, tok = load_jepa_gist(
        ckpt_path, tokenizer_path=str(tok_path), device="cpu", dtype="float32",
    )
    # The loaded encoder must match the saved encoder (the roundtrip).
    saved_sd = {k: v.clone() for k, v in m.encoder.state_dict().items()}
    loaded_sd = loaded.encoder.state_dict()
    assert set(saved_sd.keys()) == set(loaded_sd.keys()), \
        "encoder state_dict keys must match after roundtrip"
    for k in saved_sd:
        assert torch.equal(saved_sd[k], loaded_sd[k]), \
            f"encoder param '{k}' must roundtrip exactly"
    # The loaded predictor must match too.
    saved_pred = {k: v.clone() for k, v in m.predictor.state_dict().items()}
    loaded_pred = loaded.predictor.state_dict()
    for k in saved_pred:
        assert torch.equal(saved_pred[k], loaded_pred[k]), \
            f"predictor param '{k}' must roundtrip exactly"
    # The loader re-freezes the encoder for eval (eval never trains).
    assert all(not p.requires_grad for p in loaded.encoder.parameters()), \
        "loaded encoder must be frozen for eval"
    assert tok is not None


# ------------------------------------------------------- swap-control mechanics
def _train_toy_predictor(m: JEPAGistModel, pairs, targets, steps=600, lr=5e-3):
    """Train ONLY the predictor on (doc_ids, target_latent) pairs until it maps each
    doc's state to its doc-specific target latent. Tiny CPU problem; converges in a
    few hundred steps. The encoder stays frozen (the probe-style path)."""
    optim = torch.optim.AdamW(m.predictor.parameters(), lr=lr)
    m.encoder.eval()
    for _ in range(steps):
        for doc_ids, target in zip(pairs, targets):
            enc_states = m.encode(doc_ids)
            pred = m.predict_latent(enc_states)
            # MSE to the L2-normalized target: pushes pred -> target, which both
            # minimizes MSE and maximizes cosine (pred is L2-normalized).
            loss = F.mse_loss(pred, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()


def test_latent_swap_follows_state():
    """Toy corpus: doc A -> target latent A, doc B -> target latent B (two DISTINCT
    L2-normalized vectors, standing in for the bge gist latents). After training
    the predictor, predicting from state_A yields a latent closer to target_A than
    to target_B, and vice versa. Swapping states (predicting from state_B) yields a
    latent closer to target_B -- the prediction follows the STATE, the load-bearing
    JEPA gate property. A random frozen encoder suffices: distinct docs -> distinct
    states -> a learnable predictor.
    """
    torch.manual_seed(0)
    m = _build(latent_dim=8, hidden=16, n_mlp_layers=2)
    # Doc A and B are distinct token streams (content tokens >= 3, avoid PAD/BOS/EOS).
    doc_a = torch.tensor([[BOS, 3, 3, 3, EOS]])
    doc_b = torch.tensor([[BOS, 4, 4, 4, EOS]])
    # Two DISTINCT target latents (L2-normalized), standing in for the bge gist
    # latents of two distinct docs.
    torch.manual_seed(1)
    target_a = F.normalize(torch.randn(1, 8), p=2, dim=-1)
    target_b = F.normalize(torch.randn(1, 8), p=2, dim=-1)
    # Sanity: the two targets are distinct (not the same point in latent space).
    assert 1.0 - F.cosine_similarity(target_a, target_b).item() > 1e-3, \
        "target latents must be distinct for the swap test to be meaningful"
    _train_toy_predictor(m, [doc_a, doc_b], [target_a, target_b], steps=800)

    state_a = m.encode(doc_a)
    state_b = m.encode(doc_b)
    pred_a = m.predict_latent(state_a)
    pred_b = m.predict_latent(state_b)

    cos_aa = F.cosine_similarity(pred_a, target_a).item()
    cos_ab = F.cosine_similarity(pred_a, target_b).item()
    cos_bb = F.cosine_similarity(pred_b, target_b).item()
    cos_ba = F.cosine_similarity(pred_b, target_a).item()

    # Main fidelity: each state predicts a latent closer to its OWN target than to
    # the other doc's target.
    assert cos_aa > cos_ab, \
        f"state_A should predict a latent closer to target_A than target_B " \
        f"(cos_aa={cos_aa:.3f} <= cos_ab={cos_ab:.3f})"
    assert cos_bb > cos_ba, \
        f"state_B should predict a latent closer to target_B than target_A " \
        f"(cos_bb={cos_bb:.3f} <= cos_ba={cos_ba:.3f})"

    # Swap control: swapping states swaps the predicted latent (distinct states
    # produce distinct predictions -- the literal anti-§3.3 property).
    assert not torch.allclose(pred_a, pred_b, atol=1e-5), \
        "swapping states must change the predicted latent (§3.3 failed here)"


# ----------------------------------------------------------- jepa_loss smoke
def test_jepa_contrastive_loss_smoke():
    """The reused ``jepa_contrastive_loss`` accepts the predictor's
    ``[b, latent_dim]`` output + ``[b, latent_dim]`` target + ``[n, latent_dim]``
    negatives and returns a finite scalar."""
    from src.subconscious.training.jepa_loss import jepa_contrastive_loss

    b, dim, n = 4, 8, 16
    predicted = F.normalize(torch.randn(b, dim), p=2, dim=-1)
    actual = F.normalize(torch.randn(b, dim), p=2, dim=-1)
    negatives = F.normalize(torch.randn(n, dim), p=2, dim=-1)
    loss = jepa_contrastive_loss(predicted, actual, negatives, temperature=0.1)
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), f"loss must be finite, got {loss}"


# ----------------------------------------------------- infonce loss (the fix)
def test_jepa_infonce_loss_smoke():
    """``jepa_infonce_loss`` accepts the predictor's ``[b, dim]`` output, the
    ``[b, dim]`` target, and ``[n, dim]`` negatives, and returns a finite scalar."""
    from src.subconscious.jepa_gist import jepa_infonce_loss

    b, dim, n = 4, 8, 16
    predicted = F.normalize(torch.randn(b, dim), p=2, dim=-1)
    actual = F.normalize(torch.randn(b, dim), p=2, dim=-1)
    negatives = F.normalize(torch.randn(n, dim), p=2, dim=-1)
    loss = jepa_infonce_loss(predicted, actual, negatives, temperature=0.1)
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), f"loss must be finite, got {loss}"


def test_infonce_prefers_actual_over_anticorrelation():
    """The load-bearing test that would have caught the degeneracy BEFORE the
    42-min retrain. On a TIGHT target cluster (bge gist latents of similar docs are
    all cos ~0.7 -- they cluster), the OLD ``jepa_contrastive_loss`` is minimized by
    ANTI-correlating with the cluster (``pred = -actual``), not by ``pred = actual``
    -- its unbounded negative term overwhelms the bounded positive term. The
    ``jepa_infonce_loss`` fix puts the positive in the same denominator as the
    negatives, so its optimum is ``pred = actual``: the loss ordering is
    actual < mean < anti. This test pins both facts (regression guard for the fix).
    """
    from src.subconscious.jepa_gist import jepa_infonce_loss
    from src.subconscious.training.jepa_loss import jepa_contrastive_loss

    torch.manual_seed(42)
    dim = 64
    # A tight cluster: one "mean direction" + small random perturbations, all
    # L2-normalized -- mirrors bge gist latents of semantically-similar docs
    # (mean-vs-member cos ~0.7, the regime where the real retrain collapsed to
    # val_latent_cos = -0.84). The noise scale MUST be << 1/sqrt(dim): a flat
    # 0.5 in dim=64 produces a near-orthogonal (cos~0), unclustered set in which
    # the loss is NOT degenerate and the assertion below does not hold. We
    # parametrize as noise = c/sqrt(dim) so the cluster tightness
    # (cos ~ 1/sqrt(1+c^2)) is dimension-independent: c=1.0 -> cos~0.707.
    mean_dir = F.normalize(torch.randn(1, dim), p=2, dim=-1)
    cluster_noise = 1.0 / math.sqrt(dim)
    def _cluster(n):
        return F.normalize(mean_dir + cluster_noise * torch.randn(n, dim), p=2,
                           dim=-1)
    actual = _cluster(8)                       # [8, dim] -- the per-doc targets
    negatives = _cluster(32)                  # [32, dim] -- the negative pool
    temp = 0.1
    # Three candidate predictions of the same shape as `actual`:
    pred_actual = actual.clone()              # the correct optimum
    pred_mean = mean_dir.expand_as(actual)    # cluster-mean collapse (anti-shortcut target)
    pred_anti = -actual                       # the degenerate anti-correlation optimum

    # InfoNCE (the fix): optimum is pred=actual. Ordering actual < mean < anti.
    infonce_actual = jepa_infonce_loss(pred_actual, actual, negatives, temp).item()
    infonce_mean = jepa_infonce_loss(pred_mean, actual, negatives, temp).item()
    infonce_anti = jepa_infonce_loss(pred_anti, actual, negatives, temp).item()
    assert infonce_actual < infonce_mean, \
        f"InfoNCE should prefer pred=actual ({infonce_actual:.3f}) over mean " \
        f"({infonce_mean:.3f})"
    assert infonce_mean < infonce_anti, \
        f"InfoNCE should prefer mean ({infonce_mean:.3f}) over anti-correlation " \
        f"({infonce_anti:.3f})"

    # The OLD contrastive loss is DEGENERATE here: anti < actual (anti-correlation
    # is the LOWER loss). This is the regression guard documenting the bug we fixed.
    con_actual = jepa_contrastive_loss(pred_actual, actual, negatives, temp).item()
    con_anti = jepa_contrastive_loss(pred_anti, actual, negatives, temp).item()
    assert con_anti < con_actual, (
        f"the old contrastive loss should be degenerate (anti {con_anti:.3f} < "
        f"actual {con_actual:.3f}) -- this assertion documents the bug; if it "
        f"fails the loss is no longer degenerate and the fix can be reconsidered"
    )