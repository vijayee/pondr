"""Phase 1f-7 Stage 1 unit tests: the per-doc-kind ``StateReadout``.

Covers the binding contract in ``docs/`` (plan reflective-greeting-beacon):
  (a) ``n_doc_kinds=0`` builds the byte-identical shared readout (keys
      ``net.0.*`` / ``net.2.*``) so an old ``final.pt`` strict-loads.
  (b) ``n_doc_kinds=3`` builds a shared ``body`` + per-kind ``kind_heads`` and
      routes a ``doc_kinds`` tensor to the right head.
  (c) ``CompositeZHead.logits(..., slot_doc_kinds=)`` matches the per-kind
      routing and is byte-identical when ``n_doc_kinds=0``.
  (d) the per-kind arch round-trips through ``from_state_dict`` (n_doc_kinds is
      INFERRED from the keys, not a stored field).
"""
from __future__ import annotations

import glob

import torch

from src.subconscious.state_readout import (
    DEFAULT_DIM_IN,
    CompositeZHead,
    StateReadout,
    load_composite_z_head,
)


def _prefixes(module: torch.nn.Module) -> list[str]:
    return sorted({k.rsplit(".", 1)[0] for k, _ in module.named_parameters()})


def test_n0_byte_identical_keys():
    """n_doc_kinds=0 mlp128 keeps the ORIGINAL net.0/net.2 key namespace."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=0)
    assert _prefixes(ro) == ["net.0", "net.2"]
    assert ro.n_doc_kinds == 0


def test_n3_per_kind_keys():
    """n_doc_kinds=3 builds shared body.0 + per-kind kind_heads.{0,1,2}."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3)
    assert _prefixes(ro) == ["body.0", "kind_heads.0", "kind_heads.1", "kind_heads.2"]
    assert ro.n_doc_kinds == 3


def test_n0_requires_hidden_for_n3():
    """n_doc_kinds>0 without a shared body is degenerate -> ValueError."""
    try:
        StateReadout(dim_in=DEFAULT_DIM_IN, hidden=None, n_doc_kinds=3)
    except ValueError:
        return
    raise AssertionError("n_doc_kinds>0 with hidden=None should raise")


def test_routing_to_correct_kind_head():
    """A doc_kinds tensor routes each row to its kind head's output."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3)
    x = torch.randn(6, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = ro(x, doc_kinds=dk)
    assert out.shape == (6, 384)
    h = ro.body(x)
    # Each kind's rows equal that kind head applied to the shared body output.
    assert torch.allclose(ro.kind_heads[0](h[:2]), out[:2])
    assert torch.allclose(ro.kind_heads[1](h[2:4]), out[2:4])
    assert torch.allclose(ro.kind_heads[2](h[4:]), out[4:])


def test_n0_forward_ignores_doc_kinds():
    """n_doc_kinds=0 forward is byte-identical whether or not doc_kinds is passed."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=0)
    x = torch.randn(5, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 1, 2, 0, 1])
    assert torch.allclose(ro(x), ro(x, doc_kinds=dk))


def test_n3_requires_doc_kinds():
    """n_doc_kinds>0 forward without a doc_kinds tensor is a hard error."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3)
    x = torch.randn(4, DEFAULT_DIM_IN)
    try:
        ro(x)
    except RuntimeError:
        return
    raise AssertionError("n_doc_kinds>0 forward without doc_kinds should raise")


def test_round_trip_infers_n_doc_kinds():
    """from_state_dict infers n_doc_kinds from the keys (not a stored field)."""
    ro3 = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3)
    x = torch.randn(6, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = ro3(x, doc_kinds=dk)
    rebuilt = StateReadout.from_state_dict(ro3.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert rebuilt.n_doc_kinds == 3
    assert torch.allclose(rebuilt(x, doc_kinds=dk), out, atol=1e-6)


def test_composite_n0_byte_identical_with_slot_doc_kinds():
    """CompositeZHead n_doc_kinds=0 ignores slot_doc_kinds (byte-identical)."""
    c = CompositeZHead(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=0)
    sy = torch.zeros(6, c.slot_dim)
    sig = torch.randn(6, DEFAULT_DIM_IN)
    q = torch.randn(384)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    assert torch.allclose(c.logits(sy, sig, q), c.logits(sy, sig, q, slot_doc_kinds=dk))


def test_composite_n3_routing_and_round_trip():
    """CompositeZHead n_doc_kinds=3 routes slot_doc_kinds + round-trips."""
    c = CompositeZHead(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3)
    sy = torch.zeros(6, c.slot_dim)
    sig = torch.randn(6, DEFAULT_DIM_IN)
    q = torch.randn(384)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = c.logits(sy, sig, q, slot_doc_kinds=dk)
    # Matches the per-kind readout routed manually.
    z_i = c.readout(sig, doc_kinds=dk)
    assert torch.allclose(out, c.z_head.logits(sy, z_i, q))
    # Round-trip via from_state_dict (infers n_doc_kinds=3 from keys).
    c2 = CompositeZHead.from_state_dict(c.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert c2.n_doc_kinds == 3
    assert torch.allclose(c2.logits(sy, sig, q, slot_doc_kinds=dk), out, atol=1e-6)


def test_old_n0_checkpoint_strict_loads():
    """Binding check: an existing 1f-6 n_doc_kinds=0 final.pt strict-loads."""
    paths = glob.glob(
        "data/training/strm_state_readout/phase1f6_margin_doc_ring/"
        "bilinear_s*/final.pt"
    )
    if not paths:
        return  # not present in every checkout; skip rather than fail
    head = load_composite_z_head(paths[0], device="cpu", map_location="cpu")
    assert head.n_doc_kinds == 0, "old 1f-6 ckpt must load as n_doc_kinds=0"
    # It must score without a slot_doc_kinds arg (byte-identical serve path).
    sy = torch.zeros(4, head.slot_dim)
    sig = torch.randn(4, head.dim_in)
    q = torch.randn(384)
    out = head.logits(sy, sig, q)
    assert out.shape[0] == 4


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1f-7 Stage 1 REDESIGN: per_kind_full MoE arch (no shared body).
# ─────────────────────────────────────────────────────────────────────────────


def test_per_kind_full_keys():
    """per_kind_full builds N INDEPENDENT full readouts; no body/kind_heads/net."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_full=True)
    assert _prefixes(ro) == [
        "kind_readouts.0.0", "kind_readouts.0.2",
        "kind_readouts.1.0", "kind_readouts.1.2",
        "kind_readouts.2.0", "kind_readouts.2.2",
    ]
    assert ro.n_doc_kinds == 3
    assert ro.per_kind_full is True
    # No shared-body / kind_heads / net params -- the MoE has NO shared body.
    assert not hasattr(ro, "body")
    assert not hasattr(ro, "kind_heads")
    assert not hasattr(ro, "net")


def test_per_kind_full_routing():
    """A doc_kinds tensor routes each row to its kind's FULL readout (no body)."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_full=True)
    x = torch.randn(6, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = ro(x, doc_kinds=dk)
    assert out.shape == (6, 384)
    # Each kind's rows equal that kind's FULL readout applied to its rows (no
    # shared body -- the readout is Linear->ReLU->Linear on the raw state).
    assert torch.allclose(ro.kind_readouts[0](x[:2]), out[:2], atol=1e-6)
    assert torch.allclose(ro.kind_readouts[1](x[2:4]), out[2:4], atol=1e-6)
    assert torch.allclose(ro.kind_readouts[2](x[4:]), out[4:], atol=1e-6)


def test_per_kind_full_round_trip_infers():
    """from_state_dict infers per_kind_full + n_doc_kinds from the keys."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_full=True)
    x = torch.randn(6, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = ro(x, doc_kinds=dk)
    rebuilt = StateReadout.from_state_dict(ro.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert rebuilt.per_kind_full is True
    assert rebuilt.n_doc_kinds == 3
    assert torch.allclose(rebuilt(x, doc_kinds=dk), out, atol=1e-6)


def test_per_kind_full_requires_n_doc_kinds_and_hidden():
    """per_kind_full=True with n_doc_kinds=0 or hidden=None raises ValueError."""
    try:
        StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=0,
                     per_kind_full=True)
    except ValueError:
        pass
    else:
        raise AssertionError("per_kind_full with n_doc_kinds=0 should raise")
    try:
        StateReadout(dim_in=DEFAULT_DIM_IN, hidden=None, n_doc_kinds=3,
                     per_kind_full=True)
    except ValueError:
        pass
    else:
        raise AssertionError("per_kind_full with hidden=None should raise")


def test_composite_per_kind_full_round_trip():
    """CompositeZHead per_kind_full routes + round-trips through from_state_dict."""
    c = CompositeZHead(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                      per_kind_full=True)
    assert c.per_kind_full is True
    sy = torch.zeros(6, c.slot_dim)
    sig = torch.randn(6, DEFAULT_DIM_IN)
    q = torch.randn(384)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = c.logits(sy, sig, q, slot_doc_kinds=dk)
    # Matches the per-kind readout routed manually (no shared body).
    z_i = c.readout(sig, doc_kinds=dk)
    assert torch.allclose(out, c.z_head.logits(sy, z_i, q), atol=1e-6)
    # Round-trip via from_state_dict (infers per_kind_full=True from keys).
    c2 = CompositeZHead.from_state_dict(c.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert c2.per_kind_full is True
    assert c2.n_doc_kinds == 3
    assert torch.allclose(c2.logits(sy, sig, q, slot_doc_kinds=dk), out, atol=1e-6)


def test_per_kind_full_zero_init_unmatched_kind():
    """An out-of-[0,n) doc_kind yields ZERO rows (new_zeros guard), not garbage."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_full=True)
    x = torch.randn(4, DEFAULT_DIM_IN)
    # kind=5 is out of [0,3) -> those rows stay zero (no readout owns them).
    dk = torch.tensor([0, 5, 1, 99])
    out = ro(x, doc_kinds=dk)
    assert torch.allclose(out[1], torch.zeros(384))
    assert torch.allclose(out[3], torch.zeros(384))
    # The in-range rows still route correctly.
    assert torch.allclose(ro.kind_readouts[0](x[:1]), out[:1], atol=1e-6)
    assert torch.allclose(ro.kind_readouts[1](x[2:3]), out[2:3], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1f-7 Stage 2 #6: per_kind_bodies arch (per-kind bodies + shared head).
# ─────────────────────────────────────────────────────────────────────────────


def test_per_kind_bodies_keys():
    """per_kind_bodies builds N bodies + ONE shared head; no body/kind_heads/net."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_bodies=True)
    assert _prefixes(ro) == [
        "head",
        "kind_bodies.0.0",
        "kind_bodies.1.0",
        "kind_bodies.2.0",
    ]
    assert ro.n_doc_kinds == 3
    assert ro.per_kind_bodies is True
    assert ro.per_kind_full is False
    # No shared-body / kind_heads / kind_readouts / net params.
    assert not hasattr(ro, "body")
    assert not hasattr(ro, "kind_heads")
    assert not hasattr(ro, "kind_readouts")
    assert not hasattr(ro, "net")


def test_per_kind_bodies_routing():
    """A doc_kinds tensor routes each row through its kind body then the shared head."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_bodies=True)
    x = torch.randn(6, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = ro(x, doc_kinds=dk)
    assert out.shape == (6, 384)
    # Each kind's rows == its body -> the SHARED head (the head is shared/linear,
    # so per-slice application == routing all rows through body+head).
    assert torch.allclose(ro.head(ro.kind_bodies[0](x[:2])), out[:2], atol=1e-6)
    assert torch.allclose(ro.head(ro.kind_bodies[1](x[2:4])), out[2:4], atol=1e-6)
    assert torch.allclose(ro.head(ro.kind_bodies[2](x[4:])), out[4:], atol=1e-6)


def test_per_kind_bodies_round_trip_infers():
    """from_state_dict infers per_kind_bodies + n_doc_kinds from the keys."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_bodies=True)
    x = torch.randn(6, DEFAULT_DIM_IN)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = ro(x, doc_kinds=dk)
    rebuilt = StateReadout.from_state_dict(ro.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert rebuilt.per_kind_bodies is True
    assert rebuilt.per_kind_full is False
    assert rebuilt.n_doc_kinds == 3
    assert torch.allclose(rebuilt(x, doc_kinds=dk), out, atol=1e-6)


def test_per_kind_bodies_requires_n_doc_kinds_and_hidden():
    """per_kind_bodies=True with n_doc_kinds=0 or hidden=None raises ValueError."""
    try:
        StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=0,
                     per_kind_bodies=True)
    except ValueError:
        pass
    else:
        raise AssertionError("per_kind_bodies with n_doc_kinds=0 should raise")
    try:
        StateReadout(dim_in=DEFAULT_DIM_IN, hidden=None, n_doc_kinds=3,
                     per_kind_bodies=True)
    except ValueError:
        pass
    else:
        raise AssertionError("per_kind_bodies with hidden=None should raise")


def test_composite_per_kind_bodies_round_trip():
    """CompositeZHead per_kind_bodies routes + round-trips through from_state_dict."""
    c = CompositeZHead(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                      per_kind_bodies=True)
    assert c.per_kind_bodies is True
    assert c.per_kind_full is False
    sy = torch.zeros(6, c.slot_dim)
    sig = torch.randn(6, DEFAULT_DIM_IN)
    q = torch.randn(384)
    dk = torch.tensor([0, 0, 1, 1, 2, 2])
    out = c.logits(sy, sig, q, slot_doc_kinds=dk)
    # Matches the per-kind readout routed manually (bodies -> shared head).
    z_i = c.readout(sig, doc_kinds=dk)
    assert torch.allclose(out, c.z_head.logits(sy, z_i, q), atol=1e-6)
    # Round-trip via from_state_dict (infers per_kind_bodies=True from keys).
    c2 = CompositeZHead.from_state_dict(c.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert c2.per_kind_bodies is True
    assert c2.per_kind_full is False
    assert c2.n_doc_kinds == 3
    assert torch.allclose(c2.logits(sy, sig, q, slot_doc_kinds=dk), out, atol=1e-6)


def test_per_kind_bodies_zero_init_unmatched_kind():
    """An out-of-[0,n) doc_kind yields ZERO rows (new_zeros guard), not garbage."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_bodies=True)
    x = torch.randn(4, DEFAULT_DIM_IN)
    # kind=5 / kind=99 are out of [0,3) -> those rows stay zero (no body owns them).
    dk = torch.tensor([0, 5, 1, 99])
    out = ro(x, doc_kinds=dk)
    assert torch.allclose(out[1], torch.zeros(384))
    assert torch.allclose(out[3], torch.zeros(384))
    # The in-range rows still route through their kind body -> shared head.
    assert torch.allclose(ro.head(ro.kind_bodies[0](x[:1])), out[:1], atol=1e-6)
    assert torch.allclose(ro.head(ro.kind_bodies[1](x[2:3])), out[2:3], atol=1e-6)


def test_per_kind_bodies_distinct_from_moe_and_shared_body():
    """A per_kind_bodies state_dict is NOT mis-detected as per_kind_full/shared-body."""
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                     per_kind_bodies=True)
    sd = ro.state_dict()
    # The per_kind_bodies namespace is disjoint from the other two per-kind
    # namespaces -- no kind_readouts.* / kind_heads.* / body.0.* keys present.
    assert not any(k.startswith("kind_readouts.") for k in sd)
    assert not any(k.startswith("kind_heads.") for k in sd)
    assert not any(k.startswith("body.") for k in sd)
    rebuilt = StateReadout.from_state_dict(sd, dim_in=DEFAULT_DIM_IN)
    assert rebuilt.per_kind_bodies is True
    assert rebuilt.per_kind_full is False
    assert rebuilt.n_doc_kinds == 3
    # Composite-level inference is likewise distinct.
    c = CompositeZHead(dim_in=DEFAULT_DIM_IN, hidden=128, n_doc_kinds=3,
                      per_kind_bodies=True)
    c2 = CompositeZHead.from_state_dict(c.state_dict(), dim_in=DEFAULT_DIM_IN)
    assert c2.per_kind_bodies is True
    assert c2.per_kind_full is False