"""Tests for the Phase 3a node-feature pipeline (``src/gnn/features.py``)."""

from __future__ import annotations

import json

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.gnn.features import (
    NodeFeatureBuilder, NODE_KINDS, NODE_KIND_INDEX, FEATURE_DIM, infer_kind,
)


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def test_infer_kind_covers_every_prefix():
    assert infer_kind("ep_000001") == "episode"
    assert infer_kind("E:Alice") == "entity"
    assert infer_kind("T:db") == "topic"
    assert infer_kind("A:curious") == "tone"
    assert infer_kind("D:pick") == "decision"
    assert infer_kind("S:0001") == "session"
    assert infer_kind("U:alice") == "user"
    assert infer_kind("???") == "unknown"


def test_feature_dim_matches_embedding_width_and_onehot_fits():
    # 384 = bge-small embedding width; the 8-wide onehot + 1 salience slot fit
    # in the leading bytes, leaving 375 for the embedding slice.
    assert FEATURE_DIM == 384
    assert len(NODE_KINDS) == 8
    assert NODE_KIND_INDEX["episode"] == 0


def test_episode_feature_uses_persisted_embedding(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(Episode(id="ep_000001", timestamp="t", summary="s",
                                  full_text="f", entities=["Alice"]))
    store.set_summary_embedding("ep_000001", [0.1] * 384)
    fb = NodeFeatureBuilder(store)
    k, vec = fb.feature_for("ep_000001")
    assert k == NODE_KIND_INDEX["episode"]
    assert vec.shape[0] == FEATURE_DIM
    assert vec[k] == 1.0  # onehot stamped
    # The embedding slice (after the 9 leading slots) carries the persisted vec.
    assert abs(float(vec[9]) - 0.1) < 1e-6
    store.close()


def test_episode_feature_falls_back_to_hash_stub_when_no_embedding(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(Episode(id="ep_000001", timestamp="t", summary="s",
                                  full_text="f"))
    fb = NodeFeatureBuilder(store)
    k, vec = fb.feature_for("ep_000001")
    # Stub: the salience slot is tagged -1.0 so callers can tell it apart.
    assert vec[8] == -1.0
    assert vec[k] == 1.0
    # Non-zero (a hash, not zeros).
    assert float(vec[10:].abs().sum()) > 0.0
    store.close()


def test_entity_feature_carries_heuristic_salience(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(Episode(id="ep_000001", timestamp="t", summary="s",
                                  full_text="f", entities=["Alice"]))
    store.write_entity_salience_batch({"Alice": 25}, {"Alice": "ep_000001"})
    fb = NodeFeatureBuilder(store)
    k, vec = fb.feature_for("E:Alice")
    assert k == NODE_KIND_INDEX["entity"]
    assert vec[k] == 1.0
    # Salience in slot 8 (after the 8-wide onehot).
    assert vec[8] > 0.0
    store.close()


def test_non_episode_non_entity_feature_is_onehot_only(tmp_path):
    store = _store(tmp_path)
    fb = NodeFeatureBuilder(store)
    k, vec = fb.feature_for("T:db")
    assert k == NODE_KIND_INDEX["topic"]
    assert vec[k] == 1.0
    # No embedding, no salience → those slots are zero.
    assert vec[8] == 0.0
    assert float(vec[9:].abs().sum()) == 0.0
    store.close()