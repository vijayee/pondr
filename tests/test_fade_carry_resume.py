"""Session-resume of the carried Mamba3 state (R3, TMT steal).

``Mamba3Voice.snapshot_carry`` / ``restore_carry`` serialize the carried
``InferenceParams`` (per-layer SISO state tensors + seqlen/max/offset) so a
resumed conversation keeps its within-window memory instead of starting cold.
The carried state is the state -- restore replays NOTHING (a state-tensor
copy), so the lossless <=2048-token carry probe transfers to restoration.

These tests pin:

- snapshot with no carry -> the ``{"carry": None}`` clear marker (a save over
  a previously-persisted carry DROPS the stale blob instead of leaving it
  keyed by user for a later load to wrongly restore).
- full round trip: ingest -> snapshot -> reset -> restore -> cache tensors
  element-equal (``torch.equal``), seqlen/max/offset equal, and the snapshot
  of the restored voice is byte-identical to the original (JSON round-tries).
- the restore contract on a REAL ``Mamba3Voice`` with a stub model whose cache
  is a real tensor dict (same shape as the ``InferenceParams`` cache the real
  443M model allocates); no mamba_ssm model load, no CUDA.
- bf16 caches round-trip bit-exact (widened to f32 for storage, narrowed back).
- every mismatch (version, model_id, layer structure, shape, dtype, malformed
  entry) -> False + reset (never a half-restored state, never a raise).
- the orchestrator seam: ``save_session`` writes the carry blob (store scope
  ``"voice_carry"`` + ``<sid>_carry.json``) and ``load_session`` restores it
  (store-first, file fallback, clear-marker drop).
- the flag-off contract: ``voice_carry_resume=False`` -> the seam stays silent
  (no carry file, load touches no carry); byte-identical WM-only persistence.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.fade import FadeConfig, FadeMemory, Mamba3Voice
from tests.test_fade import _StubHFTokenizer
from tests.test_fade_voice_carry import (
    _StubCarryModel,
    _StubCarryVoice,
    _StubEmbedder,
)
from tests.test_orchestrator import (
    _StubModeA,
    _StubPlanner,
    _ep,
    _orchestrator,
)


# -- helpers -------------------------------------------------------------------

def _voice(max_tokens: int = 2048, dtype: torch.dtype | None = None,
           model_id: str = "stub/mamba3-siso-40m") -> Mamba3Voice:
    model = _StubCarryModel()
    if dtype is not None:
        orig = model.allocate_inference_cache

        def alloc(batch_size, max_seqlen, dtype_=None, **kw):
            orig(batch_size, max_seqlen)
            return {0: [torch.zeros(2, 2, dtype=dtype) for _ in range(4)]}

        model.allocate_inference_cache = alloc  # type: ignore[method-assign]
    return Mamba3Voice(model, _StubHFTokenizer(), device="cpu",
                       temperature=0.0, model_id=model_id)


def _stuffed_state(voice: Mamba3Voice) -> tuple[list, int, int]:
    """Ingest a turn (via the real writer, so seqlen/offset/cache all state),
    then leave distinctive values in the cache (the stub model stamped each
    tensor with its call count -- equal per tensor)."""
    n = voice.ingest_turn("User: the badge is FENNEC-07\nAssistant: ok",
                          max_tokens=2048)
    assert n > 0
    vals = [t.clone() for t in (voice._carry_inf.key_value_memory_dict[0])]
    return vals, voice._carry_seqlen, voice._carry_max


def _assert_same_cache(a: Mamba3Voice, b: Mamba3Voice) -> None:
    ac = a._carry_inf.key_value_memory_dict
    bc = b._carry_inf.key_value_memory_dict
    assert sorted(ac.keys(), key=str) == sorted(bc.keys(), key=str)
    for ak in ac:
        for ta, tb in zip(ac[ak], bc[ak]):
            assert ta.dtype == tb.dtype
            assert torch.equal(ta, tb)
    assert a._carry_seqlen == b._carry_seqlen
    assert a._carry_max == b._carry_max
    assert a._carry_inf.seqlen_offset == b._carry_inf.seqlen_offset


# ---------------------------------------------------------------------------
# snapshot: shape / clear marker
# ---------------------------------------------------------------------------

def test_snapshot_no_carry_returns_clear_marker():
    """No carried state -> ``{"v": 1, "carry": None}`` (JSON-safe, and the
    marker a later save uses to drop a stale persisted blob)."""
    voice = _voice()
    assert voice.snapshot_carry() == {"v": 1, "carry": None}


def test_snapshot_carries_state_and_bookkeeping():
    """A carried state snapshots to the per-layer 4-tensor dict + seqlen/
    offset/max/model_id; JSON round-trips it unchanged."""
    voice = _voice()
    vals, seqlen, _cmax = _stuffed_state(voice)
    snap = json.loads(json.dumps(voice.snapshot_carry()))
    assert snap["v"] == 1
    assert snap["model_id"] == "stub/mamba3-siso-40m"
    assert snap["seqlen"] == seqlen
    assert snap["offset"] == voice._carry_inf.seqlen_offset == seqlen
    assert snap["max"] == voice._carry_max
    layers = snap["carry"]
    assert sorted(layers.keys(), key=str) == ["0"]
    assert len(layers["0"]) == 4
    # every tensor entry names its dtype/shape and carries base64 bytes.
    for orig, entry in zip(vals, layers["0"]):
        assert entry["dtype"] == str(orig.dtype)
        assert entry["shape"] == list(orig.shape)
        assert isinstance(entry["data"], str) and entry["data"]


# ---------------------------------------------------------------------------
# restore: the round trip
# ---------------------------------------------------------------------------

def test_restore_round_trip_element_equal():
    """snapshot -> reset -> restore: the carried state is back element-equal
    (``torch.equal``), with seqlen/max/offset restored. Restore is a state-
    tensor copy, not a re-ingest (the model is never called)."""
    voice = _voice()
    vals, seqlen, cmax = _stuffed_state(voice)
    snap = json.loads(json.dumps(voice.snapshot_carry()))
    voice.reset_carry()
    assert voice._carry_inf is None

    calls_before = voice.model.calls
    assert voice.restore_carry(snap) is True
    assert voice.model.calls == calls_before   # restore replayed NOTHING
    assert voice._carry_seqlen == seqlen
    assert voice._carry_max == cmax
    assert voice._carry_inf.seqlen_offset == seqlen
    for orig, restored in zip(
            vals, voice._carry_inf.key_value_memory_dict[0]):
        assert torch.equal(orig, restored)


def test_restore_into_fresh_voice_is_idempotent_snapshot():
    """A fresh voice restored from a snapshot snapshots back to the SAME blob
    (save -> load -> save is stable, no drift)."""
    v1 = _voice()
    _stuffed_state(v1)
    snap = json.loads(json.dumps(v1.snapshot_carry()))
    v2 = _voice()
    assert v2.restore_carry(snap) is True
    assert json.loads(json.dumps(v2.snapshot_carry())) == snap


def test_restore_clear_marker_resets():
    """The ``{"carry": None}`` marker -> True + cleared (a no-carry save
    consumed over a previously-restored state)."""
    voice = _voice()
    _stuffed_state(voice)
    assert voice.restore_carry({"v": 1, "carry": None}) is True
    assert voice._carry_inf is None
    assert voice._carry_seqlen == 0
    assert voice._carry_max == 0


def test_restore_bf16_cache_bit_exact():
    """A bf16 cache (the real 443M allocates bf16 state with a bf16 model)
    round-trips bit-exact: stored widened to f32, restored back to bf16 -- the
    f32 widening carries the bf16 mantissa fully, so the narrow-back is an
    identity. A float32 f32->f32->f32 path is trivially exact too."""
    for dtype in (torch.bfloat16, torch.float32):
        voice = _voice(dtype=dtype)
        _stuffed_state(voice)
        pre = voice._carry_inf.key_value_memory_dict[0][0].clone()
        assert pre.dtype == dtype
        snap = json.loads(json.dumps(voice.snapshot_carry()))
        assert json.loads(json.dumps(snap))["carry"]["0"][0]["dtype"] == str(
            dtype)
        voice.reset_carry()
        assert voice.restore_carry(snap) is True
        assert torch.equal(voice._carry_inf.key_value_memory_dict[0][0], pre)
        assert voice._carry_inf.key_value_memory_dict[0][0].dtype == dtype


def test_restored_carry_recalls_without_extra_ingest():
    """After restore, ``recall_from_carry`` fires from the RESTORED state (the
    resume end-state: the resumed conversation's recall works without
    re-ingesting the prior turns). The recall stays read-only."""
    voice = _voice()
    _stuffed_state(voice)
    snap = json.loads(json.dumps(voice.snapshot_carry()))
    voice.reset_carry()
    voice.restore_carry(snap)
    seqlen = voice._carry_seqlen
    out = voice.recall_from_carry("what was the badge", max_new_tokens=4)
    assert out                                 # decoded from the restored state
    assert voice._carry_seqlen == seqlen       # recall did not advance it


# ---------------------------------------------------------------------------
# restore: mismatch guards (False + reset, never a half-restored state)
# ---------------------------------------------------------------------------

def _snap_of(voice: Mamba3Voice) -> dict:
    _stuffed_state(voice)
    return json.loads(json.dumps(voice.snapshot_carry()))


def test_restore_model_id_mismatch_is_false_and_reset():
    """A blob from a DIFFERENT checkpoint must not restore into this model."""
    blob = _snap_of(_voice())
    voice = _voice(model_id="stub/mamba3-siso-42b")
    assert voice.restore_carry(blob) is False
    assert voice._carry_inf is None
    assert voice._carry_seqlen == 0


def test_restore_version_mismatch_is_false():
    blob = _snap_of(_voice())
    blob["v"] = 99
    voice = _voice()
    assert voice.restore_carry(blob) is False
    assert voice._carry_inf is None


def test_restore_structure_mismatch_is_false_and_reset():
    """A different layer count (blob saved from a deeper model) -> False."""
    blob = _snap_of(_voice())
    blob["carry"]["1"] = json.loads(json.dumps(blob["carry"]["0"]))
    voice = _voice()
    assert voice.restore_carry(blob) is False
    assert voice._carry_inf is None


def test_restore_shape_mismatch_is_false_and_reset():
    """A per-tensor shape mismatch (e.g. a different d_state) -> False; the
    fresh cache is NOT committed (no half-restored state)."""
    blob = _snap_of(_voice())
    changed = json.loads(json.dumps(blob))
    changed["carry"]["0"][0]["shape"] = [3, 3]
    voice = _voice()
    assert voice.restore_carry(changed) is False
    assert voice._carry_inf is None


def test_restore_dtype_mismatch_is_false():
    blob = _snap_of(_voice())
    blob["carry"]["0"][0]["dtype"] = "torch.float16"
    assert _voice().restore_carry(blob) is False


def test_restore_truncated_bytes_is_false():
    """Corrupt base64 payload (too few bytes to reshape) -> False, no raise."""
    blob = _snap_of(_voice())
    blob["carry"]["0"][0]["data"] = "AAAA"
    assert _voice().restore_carry(blob) is False


def test_restore_bad_offset_is_false_and_reset():
    """A corrupt ``offset`` (outside 0..max) -> False + reset: the restored
    cache's write cursor must never sit outside the cache it belongs to."""
    blob = _snap_of(_voice())
    blob["offset"] = 10 ** 9
    voice = _voice()
    assert voice.restore_carry(blob) is False
    assert voice._carry_inf is None


def test_restore_garbage_types_are_false():
    """Malformed scalar fields -> False (caught, reset), not an exception."""
    for blob in ({"v": 1, "carry": {}},                       # missing seqlen
                 {"v": 1, "carry": {}, "seqlen": "abc",
                  "max": 100},                          # non-int seqlen
                 {"v": 1, "carry": {}, "seqlen": 0,
                  "max": -1},                           # invalid max
                 {"v": 1, "carry": {}, "seqlen": 999,
                  "max": 100},                          # seqlen > max
                 "not a dict"):                         # not a dict at all
        voice = _voice()
        assert voice.restore_carry(blob) is False
        assert voice._carry_inf is None


# ---------------------------------------------------------------------------
# the orchestrator persistence seam
# ---------------------------------------------------------------------------

def _init_wm(orch) -> None:
    """Initialize the WM state so ``snapshot_from_instance`` has shapes (the
    orchestrator's save precondition -- same step() the pre-existing session
    tests do before snapshotting)."""
    orch.working_memory.reset()
    orch.working_memory.step(torch.zeros(1, 384))


def test_orchestrator_save_load_carry_round_trip(tmp_path):
    """save_session persists the carry (store scope 'voice_carry' +
    ``<sid>_carry.json``); load_session restores it store-first; with the
    store key emptied, the FILE fallback restores it."""
    orch, store = _orchestrator(
        tmp_path, {"entities": ["Alice"], "entity_mode": "union"},
        [_ep("ep_001", entities=["Alice"])], user_id="victor")
    voice = _voice()
    voice.ingest_turn("User: the badge is FENNEC-07\nAssistant: ok",
                      max_tokens=2048)
    seqlen = voice._carry_seqlen
    pre = voice._carry_inf.key_value_memory_dict[0][0].clone()
    orch._fade = FadeMemory(
        FadeConfig(voice_carry=True, voice_carry_resume=True),
        _StubEmbedder(), voice)
    _init_wm(orch)

    orch.save_session("victor")
    cpath = tmp_path / "sessions" / "victor_carry.json"
    assert cpath.exists()
    blob = store.load_jgs_state("victor", scope="voice_carry")
    assert blob and json.loads(blob)["v"] == 1

    voice.reset_carry()
    assert orch.load_session("victor") is True       # store-first restore
    assert voice._carry_seqlen == seqlen
    assert torch.equal(voice._carry_inf.key_value_memory_dict[0][0], pre)

    # File fallback: empty the store key (load_jgs_state returns None on an
    # empty value), then the sibling file restores.
    store.db.put_sync("content/system/user/victor/voice_carry/state", "")
    voice.reset_carry()
    assert orch.load_session("victor") is True
    assert voice._carry_seqlen == seqlen
    store.close()


def test_orchestrator_snapshot_failure_degrades_to_clear_marker(tmp_path):
    """A voice whose snapshot raises NEVER fails save_session (the WM state is
    already persisted by then): the seam writes the CLEAR marker instead, so a
    stale persisted carry is dropped, not left to leak into a later load."""
    orch, store = _orchestrator(
        tmp_path, {"entities": ["Alice"], "entity_mode": "union"},
        [_ep("ep_001", entities=["Alice"])], user_id="victor")

    class _ExplodingSnapshotVoice(_StubCarryVoice):
        def snapshot_carry(self):
            raise ValueError("non-serializable carry")

    voice = _ExplodingSnapshotVoice()
    voice.ingest_turn("User: hi\nAssistant: hello", max_tokens=2048)
    orch._fade = FadeMemory(
        FadeConfig(voice_carry=True, voice_carry_resume=True),
        _StubEmbedder(), voice)
    _init_wm(orch)
    orch.save_session("victor")
    blob = json.loads(store.load_jgs_state("victor", scope="voice_carry"))
    assert blob == {"v": 1, "carry": None}
    store.close()


def test_orchestrator_no_carry_save_drops_stale_persisted_carry(tmp_path):
    """Save with NO carried state (a fresh/reset session) -> the clear marker
    is persisted; the subsequent load consumes it and leaves the carry empty
    (a stale blob never bleeds into a carry-less session)."""
    orch, store = _orchestrator(
        tmp_path, {"entities": ["Alice"], "entity_mode": "union"},
        [_ep("ep_001", entities=["Alice"])], user_id="victor")
    voice = _voice()
    voice.ingest_turn("User: hi\nAssistant: hello", max_tokens=2048)
    orch._fade = FadeMemory(
        FadeConfig(voice_carry=True, voice_carry_resume=True),
        _StubEmbedder(), voice)
    _init_wm(orch)
    orch.save_session("victor")                       # a carry IS persisted

    voice.reset_carry()                               # the session went carry-less
    orch.save_session("victor")                       # the CLEAR marker overwrote it
    blob = json.loads(store.load_jgs_state("victor", scope="voice_carry"))
    assert blob == {"v": 1, "carry": None}

    assert orch.load_session("victor") is True        # the marker consumed
    assert voice._carry_inf is None                   # still no carry
    assert voice._carry_seqlen == 0
    store.close()


def test_orchestrator_flag_off_seam_silent(tmp_path):
    """``voice_carry_resume=False`` (the default) -> save_session writes NO
    carry file and load_session touches no carry: the WM-only persistence path
    is byte-identical to pre-R3."""
    orch, store = _orchestrator(
        tmp_path, {"entities": ["Alice"], "entity_mode": "union"},
        [_ep("ep_001", entities=["Alice"])], user_id="victor")
    voice = _StubCarryVoice()
    voice.ingest_turn("turn one", max_tokens=2048)
    orch._fade = FadeMemory(FadeConfig(), _StubEmbedder(), voice)  # resume OFF
    _init_wm(orch)
    orch.save_session("victor")
    assert not (tmp_path / "sessions" / "victor_carry.json").exists()
    assert store.load_jgs_state("victor", scope="voice_carry") is None
    # load (nothing saved) -> no crash, no restore attempt.
    assert orch.load_session("victor_v2") is False
    store.close()


def test_orchestrator_corrupt_carry_blob_resets_not_fails(tmp_path):
    """A corrupt carry blob (not even JSON) NEVER fails the session load: the
    WM state still restores; the carry is reset (cold start, not broken)."""
    orch, store = _orchestrator(
        tmp_path, {"entities": ["Alice"], "entity_mode": "union"},
        [_ep("ep_001", entities=["Alice"])], user_id="victor")
    voice = _voice()
    voice.ingest_turn("User: hi\nAssistant: hello", max_tokens=2048)
    orch._fade = FadeMemory(
        FadeConfig(voice_carry=True, voice_carry_resume=True),
        _StubEmbedder(), voice)
    _init_wm(orch)
    orch.save_session("victor")
    store.close()

    # Corrupt BOTH sources (store value + file) to garbage.
    store2 = HippocampalStore(str(tmp_path / "db"))
    store2.db.put_sync("content/system/user/victor/voice_carry/state",
                       "not json at all")
    (tmp_path / "sessions" / "victor_carry.json").write_text("{{{{",
                                                             encoding="utf-8")
    retriever2 = HippocampalRetriever(store2, planner=_StubPlanner(
        {"entities": ["Alice"], "entity_mode": "union"}),
        embedder=_StubEmbedder())
    from src.config import Phase2cConfig, config as _config
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch2 = PonderOrchestrator(
        store=store2, retriever=retriever2, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_StubModeA(), config=cfg,
        user_id="victor",
    )
    orch2._fade = FadeMemory(
        FadeConfig(voice_carry=True, voice_carry_resume=True),
        _StubEmbedder(), voice)
    assert orch2.load_session("victor") is True       # the load did not fail
    assert voice._carry_inf is None                   # reset, not failed
    assert voice._carry_seqlen == 0
    store2.close()


def test_orchestrator_non_snapshot_voice_is_noop(tmp_path):
    """A ``CarryVoice`` WITHOUT the snapshot contract (the pre-R3 voice
    double) + resume on -> the seam stays silent (the ``hasattr`` guard)."""
    orch, store = _orchestrator(
        tmp_path, {"entities": ["Alice"], "entity_mode": "union"},
        [_ep("ep_001", entities=["Alice"])], user_id="victor")
    orch._fade = FadeMemory(
        FadeConfig(voice_carry=True, voice_carry_resume=True),
        _StubEmbedder(), _StubCarryVoice())
    _init_wm(orch)
    orch.save_session("victor")
    assert not (tmp_path / "sessions" / "victor_carry.json").exists()
    store.close()