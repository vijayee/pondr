"""Tests for the NOTES-PRESERVE invariant (steal R5, from Graft).

The invariant (not a feature): extra content fields on a scene -- today R1's
``stale_of``/``stale_since`` marks, tomorrow user notes / system annotations --
survive SCENE REGENERATION. Three rules pinned here:

1. ``encode_scene`` is PUT-ONLY: re-encoding a scene (UPDATE) never touches
   keys it doesn't name, so extra fields survive bit-identical.
2. ``transfer_scene_annotations`` moves the extra set src -> dst in one atomic
   batch: dst-absent -> verbatim copy, both JSON lists -> union (dst members
   first), conflicting scalars -> dst wins.
3. ``SceneAuthoringWorker._merge`` transfers BEFORE its source
   ``delete_scene`` (regeneration preserves); plain ``delete_scene`` (eviction)
   still deletes everything as INTENDED macro-forgetting.

Offline throughout: the merge path runs against the same ``_StubDecider``
harness ``tests/test_scene_blocks.py`` uses (no Bonsai HTTP).
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)",
                allow_module_level=True)

from src.memory.store import HippocampalStore, _SCENE_STANDARD_FIELDS, _merge_annotation_values
from src.subconscious.scene_worker import SceneAuthoringWorker


# ── stubs (mirrors test_scene_blocks.py harness) ──

class _StubEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            buf = bytearray()
            counter = 0
            while len(buf) < 384:
                buf += hashlib.sha256(digest + counter.to_bytes(4, "little")).digest()
                counter += 1
            out.append([(b / 127.5 - 1.0) for b in buf[:384]])
        return out


class _StubDecider:
    def __init__(self, queue: list | None = None) -> None:
        self._queue = list(queue or [])
        self.calls: list[dict] = []

    def author_scene(self, topic, existing_body, candidate_summaries, user_id,
                     heat_budget, merge_candidate=None):
        self.calls.append({
            "topic": topic, "existing_body": existing_body,
            "candidate_summaries": list(candidate_summaries),
            "user_id": user_id, "heat_budget": heat_budget,
            "merge_candidate": (dict(merge_candidate) if merge_candidate else None),
        })
        if not self._queue:
            return None
        return self._queue.pop(0)


# ── fixtures ──

def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _encode_scene(store, sid, *, body, topic, heat, user_id, source_eps):
    store.encode_scene(sid, body=body, topic=topic, heat=heat,
                       updated_ts="2026-08-01T10:00:00", user_id=user_id,
                       source_eps=source_eps)


def _annotate(store, sid, **fields):
    """Write extra annotation keys DIRECTLY -- the notes-preserve reader must
    see them after any regeneration path (a note the user/system wrote on the
    scene, independent of encode_scene's signature)."""
    store.db.batch_sync([{"type": "put", "key": f"content/scene/{sid}/{field}",
                          "value": (json.dumps(val) if isinstance(val, list) else val)}
                         for field, val in fields.items()])


def _worker(store, decider):
    w = SceneAuthoringWorker(store, decider, _StubEmbedder())
    return w


def _drain(w):
    w.drain(timeout=5.0)


def _wait_queue():
    time.sleep(0.1)


def _scan_scene_keys(store, sid) -> set:
    out: set = set()
    for k, _ in store.db.create_read_stream(start=f"content/scene/{sid}/",
                                            end=f"content/scene/{sid}/\x7f"):
        out.add(k)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 1. Store layer: put-only re-encode / transfer merge rules / eviction
# ──────────────────────────────────────────────────────────────────────────

def test_update_reencode_preserves_annotations_bit_identical(tmp_path):
    """Rule 1: encode_scene is PUT-ONLY -- a re-encode (the UPDATE path) never
    deletes or reads unknown keys, so extra annotations survive bit-identical."""
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="v1 body", topic="storage", heat=0.5,
                  user_id="alice", source_eps=["ep_001"])
    _annotate(store, sid, stale_of=json.dumps(["ep_000001"]),
              stale_since="2026-09-01T00:00:00",
              user_note="double-check the shard numbers")
    before = dict(store.scene_annotations(sid))
    assert set(before) == {"stale_of", "stale_since", "user_note"}
    # Re-encode (UPDATE in the worker calls exactly this).
    _encode_scene(store, sid, body="v2 body", topic="storage", heat=0.7,
                  user_id="alice", source_eps=["ep_001", "ep_002"])
    assert store.scene_annotations(sid) == before
    assert store.get_scene(sid)["body"] == "v2 body"
    store.close()


def test_scene_annotations_excludes_standard_fields(tmp_path):
    """The reader's exclusion: exactly the standard fields set, nothing else
    counted as an annotation."""
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="b", topic="t", heat=0.5, user_id="u",
                  source_eps=["ep_001"])
    assert store.scene_annotations(sid) == {}
    assert _SCENE_STANDARD_FIELDS == {
        "body", "topic", "heat", "updated_ts", "user_id", "source_eps",
        "embedding"}
    store.close()


def test_scene_annotations_absent_scene_empty(tmp_path):
    store = _store(tmp_path)
    assert store.scene_annotations("scene_999999") == {}
    store.close()


def test_transfer_copies_verbatim_when_dst_absent(tmp_path):
    store = _store(tmp_path)
    src = store.next_scene_id()
    _annotate(store, src, user_note="keep me", stale_since="T1",
              stale_of=json.dumps(["ep_000001"]))
    moved = store.transfer_scene_annotations(src, "scene_000099")
    assert moved == ["stale_of", "stale_since", "user_note"]  # sorted
    dst_ann = store.scene_annotations("scene_000099")
    assert dst_ann["user_note"] == "keep me"
    assert dst_ann["stale_since"] == "T1"
    assert json.loads(dst_ann["stale_of"]) == ["ep_000001"]
    # COPY (not move) semantics: the transfer never deletes the source's keys
    # -- delete_scene does that next; a failed/aborted transfer must not leave
    # the source annotated-but-broken either way.
    assert store.scene_annotations(src)["user_note"] == "keep me"
    store.close()


def test_transfer_unions_json_lists_dst_members_first(tmp_path):
    store = _store(tmp_path)
    src = store.next_scene_id()
    dst = store.next_scene_id()
    _annotate(store, src, stale_of=json.dumps(["ep_000003", "ep_000001"]))
    _annotate(store, dst, stale_of=json.dumps(["ep_000001", "ep_000002"]))
    moved = store.transfer_scene_annotations(src, dst)
    assert moved == ["stale_of"]
    assert json.loads(store.scene_annotations(dst)["stale_of"]) == \
        ["ep_000001", "ep_000002", "ep_000003"]
    store.close()


def test_transfer_conflicting_scalar_dst_wins(tmp_path):
    """Conflict policy: the surviving node's own annotation value stands."""
    store = _store(tmp_path)
    src = store.next_scene_id()
    dst = store.next_scene_id()
    _annotate(store, src, stale_since="T1", revision_note="src view")
    _annotate(store, dst, stale_since="T5", revision_note="dst view")
    assert store.transfer_scene_annotations(src, dst) == []
    ann = store.scene_annotations(dst)
    assert ann["stale_since"] == "T5"
    assert ann["revision_note"] == "dst view"
    store.close()


def test_transfer_no_annotations_returns_empty(tmp_path):
    store = _store(tmp_path)
    src = store.next_scene_id()
    dst = store.next_scene_id()
    _encode_scene(store, src, body="b", topic="t", heat=0.5, user_id="u",
                  source_eps=["ep_001"])
    assert store.transfer_scene_annotations(src, dst) == []
    store.close()


def test_merge_annotation_values_unit_rules():
    """The module-level merge rule, unit-level (dst, src) -> value | None."""
    assert _merge_annotation_values("[1]", "[1]") is None  # equal -> no-op
    assert _merge_annotation_values("[1]", "[2]") == "[1, 2]"  # union dst first
    assert _merge_annotation_values('"x"', '"y"') is None  # scalar -> dst wins
    assert _merge_annotation_values('["1"]', '"y"') is None  # mixed shape -> dst
    assert _merge_annotation_values("not json", "[1]") is None  # bad -> dst wins
    assert _merge_annotation_values("", "[1]") is None  # empty dst = present ""


def test_delete_scene_still_clears_everything(tmp_path):
    """Eviction is INTENDED macro-forgetting: delete_scene (NOT the transfer
    path) removes annotations with everything else."""
    store = _store(tmp_path)
    src = store.next_scene_id()
    dst = store.next_scene_id()
    _encode_scene(store, src, body="evict me", topic="t", heat=0.9,
                  user_id="alice", source_eps=["ep_001"])
    _annotate(store, src, user_note="will vanish with the scene")
    moved = store.transfer_scene_annotations(src, dst)
    assert moved == ["user_note"]
    store.delete_scene(src)
    assert store.get_scene(src) is None
    assert _scan_scene_keys(store, src) == set()
    # dst received the note (transferred BEFORE the delete here, per the merge
    # recipe; eviction alone transfers nothing).
    assert store.scene_annotations(dst)["user_note"] == \
        "will vanish with the scene"
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# 2. Worker layer: MERGE transfers before the source delete
# ──────────────────────────────────────────────────────────────────────────

def test_worker_merge_transfers_annotations_before_source_delete(tmp_path):
    """THE invariant end-to-end: a MERGE verdict folds the source onto the
    target and the target inherits the source's annotations (R1 stale marks +
    a synthetic user note), all BEFORE delete_scene clears the source."""
    store = _store(tmp_path)
    src = store.next_scene_id()
    tgt = store.next_scene_id()
    _encode_scene(store, src, body="storage notes", topic="storage cluster",
                  heat=0.5, user_id="alice", source_eps=["ep_001"])
    _annotate(store, src, stale_of=json.dumps(["ep_000001"]),
              stale_since="2026-09-01T00:00:00",
              user_note="user underlined the throughput number")
    _encode_scene(store, tgt, body="cluster architecture",
                  topic="storage cluster design", heat=0.6, user_id="alice",
                  source_eps=["ep_002"])
    _annotate(store, tgt, stale_of=json.dumps(["ep_000002"]))
    decider = _StubDecider([{
        "action": "MERGE", "topic": "storage cluster", "body": "merged body",
        "merge_with": tgt, "reason": "fold",
    }])
    w = _worker(store, decider)
    w.tick("alice", ["ep_001", "ep_002", "ep_003"], "storage cluster")
    _wait_queue(); _drain(w)
    # Source gone (D5 unchanged).
    assert store.get_scene(src) is None
    assert _scan_scene_keys(store, src) == set()
    # Target rewritten (the standard merge semantics still hold).
    t = store.get_scene(tgt)
    assert t["body"] == "merged body"
    assert set(t["source_eps"]) == {"ep_001", "ep_002", "ep_003"}
    # ...and it INHERITED the source's annotations: verbatim copies (dst
    # absent) plus the union of stale_of.
    ann = store.scene_annotations(tgt)
    assert ann["stale_since"] == "2026-09-01T00:00:00"  # dst absent -> copy
    assert ann["user_note"] == "user underlined the throughput number"
    assert json.loads(ann["stale_of"]) == ["ep_000002", "ep_000001"]  # dst first
    store.close()


def test_worker_merge_no_annotations_is_plain_merge(tmp_path):
    """The transfer is a no-op when the source carries no extras -- the merge
    result is bit-identical to pre-R5 (no spurious annotation keys appear)."""
    store = _store(tmp_path)
    src = store.next_scene_id()
    tgt = store.next_scene_id()
    _encode_scene(store, src, body="storage notes", topic="storage cluster",
                  heat=0.5, user_id="alice", source_eps=["ep_001"])
    _encode_scene(store, tgt, body="cluster architecture",
                  topic="storage cluster design", heat=0.6, user_id="alice",
                  source_eps=["ep_002"])
    decider = _StubDecider([{
        "action": "MERGE", "topic": "storage cluster", "body": "merged body",
        "merge_with": tgt, "reason": "fold",
    }])
    w = _worker(store, decider)
    w.tick("alice", ["ep_001", "ep_002"], "storage cluster")
    _wait_queue(); _drain(w)
    assert store.get_scene(src) is None
    assert store.scene_annotations(tgt) == {}
    assert _SCENE_STANDARD_FIELDS.issuperset(
        {k.rsplit("/", 1)[-1] for k in _scan_scene_keys(store, tgt)})
    store.close()


def test_worker_update_preserves_existing_annotations(tmp_path):
    """UPDATE on an annotated scene: the re-encode keeps the notes (the put-only
    rule exercised through the worker, not just the store)."""
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="old body", topic="storage", heat=0.5,
                  user_id="alice", source_eps=["ep_001"])
    _annotate(store, sid, user_note="verify this")
    decider = _StubDecider([{
        "action": "UPDATE", "topic": "storage", "body": "new body",
        "reason": "refresh",
    }])
    w = _worker(store, decider)
    w.tick("alice", ["ep_002"], "storage")
    _wait_queue(); _drain(w)
    sc = store.get_scene(sid)
    assert sc["body"] == "new body"
    assert store.scene_annotations(sid)["user_note"] == "verify this"
    store.close()