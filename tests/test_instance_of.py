"""Phase 3b A3: entity->class typing edges (``instanceOf``) + live reassignment.

A3 emits ``(E:entity, instanceOf, Class)`` typing edges on encode (GLiNER2's
person/project/technology categories -> the Person/Project/Technology seed
classes) and wires the ontology-decay reassignment to rewrite a deprecated
class's ``instanceOf`` entities to the ``subClassOf`` parent.

Honest scope (documented in docs/Phase 3b.md): the reassignment is
**live-but-dormant** -- seed classes are never decay-deprecated (decay targets
DISCOVERED classes only, a deferred Bonsai-gated path), so this rarely fires.
GLiNER types to 3 broad seed subclasses; the real partition-bridging payoff
needs discovered-class typing (A5/Bonsai, excluded). These tests plant
``instanceOf`` edges + discovered classes directly to exercise the mechanism.
``instanceOf`` is NOT in ``_NODE_PREDICATES`` (the BFS traversal set), so it is
invisible to GNN subgraphs today; the partition-bridging benefit is a
future-retrain step, not a 3b deliverable.
"""

from __future__ import annotations

from src.config import ConsolidationConfig
from src.gnn import Consolidator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities, entity_classes):
    store.encode_episode(Episode(
        id=eid, timestamp="2026-07-01T10:00:00Z", summary=f"s {eid}",
        full_text=f"f {eid}", entities=entities, entity_classes=entity_classes,
    ))


def _out_vertices(store, subject, pred):
    r = store.graph.query().vertex(subject).out(pred).execute_sync()
    try:
        return list(r.vertices)
    finally:
        r.close()


def _in_vertices(store, obj, pred):
    r = store.graph.query().vertex(obj).in_(pred).execute_sync()
    try:
        return list(r.vertices)
    finally:
        r.close()


# ── encode emits instanceOf from entity_classes ──

def test_encode_emits_instance_of_from_entity_classes(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_001", ["Alice", "WaveDB", "PyTorch"],
            {"Alice": "Person", "WaveDB": "Project", "PyTorch": "Technology"})
    assert _out_vertices(store, "E:Alice", "instanceOf") == ["Person"]
    assert _out_vertices(store, "E:WaveDB", "instanceOf") == ["Project"]
    assert _out_vertices(store, "E:PyTorch", "instanceOf") == ["Technology"]
    # The has_entity / in_episode edges are unchanged (A3 adds, not replaces).
    assert "E:Alice" in _out_vertices(store, "ep_001", "has_entity")
    store.close()


def test_encode_without_entity_classes_emits_no_instance_of(tmp_path):
    """Back-compat: an episode with no entity_classes (pre-A3 / open-discovery)
    emits no instanceOf edges -- the has_entity edge still ships."""
    store = _store(tmp_path)
    _encode(store, "ep_001", ["Alice"], {})
    assert _out_vertices(store, "E:Alice", "instanceOf") == []
    assert "E:Alice" in _out_vertices(store, "ep_001", "has_entity")
    store.close()


def test_encode_skips_entity_without_a_class(tmp_path):
    """An entity in ``entities`` but not in ``entity_classes`` (open-discovery
    merge) gets has_entity but NO instanceOf -- only typed entities are edged."""
    store = _store(tmp_path)
    _encode(store, "ep_001", ["Alice", "Mystery"],
            {"Alice": "Person"})  # Mystery untyped
    assert _out_vertices(store, "E:Alice", "instanceOf") == ["Person"]
    assert _out_vertices(store, "E:Mystery", "instanceOf") == []
    store.close()


# ── reassignment: deprecated class -> subClassOf parent ──

def _plant_class(store, child, parent):
    """Seed a subClassOf edge child->parent (the taxonomy the parent comes from)."""
    store.db.batch_sync(store.graph.expand_triple(child, "subClassOf", parent))


def _plant_typing(store, entity, cls):
    store.db.batch_sync(store.graph.expand_triple(entity, "instanceOf", cls))


def test_reassignment_rewrites_instance_of_to_parent(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_001", ["Alice"], {"Alice": "Person"})  # E:Alice instanceOf Person
    # Plant a deprecated class OldFeature subClassOf Feature + an entity typed to it.
    _plant_class(store, "OldFeature", "Feature")
    _plant_typing(store, "E:Bob", "OldFeature")
    assert "OldFeature" in _out_vertices(store, "E:Bob", "instanceOf")

    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        config=ConsolidationConfig())
    n = cons._reassign_entities_from_deprecated_class("OldFeature", "Feature")
    assert n == 1
    # The old typing is gone; the new typing to the parent is present.
    assert _out_vertices(store, "E:Bob", "instanceOf") == ["Feature"]
    assert "OldFeature" not in _out_vertices(store, "E:Bob", "instanceOf")
    # The class's in_("instanceOf") no longer lists E:Bob under OldFeature.
    assert "E:Bob" not in _in_vertices(store, "OldFeature", "instanceOf")
    store.close()


def test_reassignment_no_typing_edges_returns_zero(tmp_path):
    """A deprecated class with no instanceOf entities -> 0 (dormant-but-live)."""
    store = _store(tmp_path)
    _plant_class(store, "OldFeature", "Feature")
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        config=ConsolidationConfig())
    n = cons._reassign_entities_from_deprecated_class("OldFeature", "Feature")
    assert n == 0
    store.close()


def test_reassignment_no_parent_returns_zero(tmp_path):
    """No subClassOf parent -> don't orphan the entities; return 0."""
    store = _store(tmp_path)
    _plant_typing(store, "E:Bob", "OldFeature")  # no subClassOf parent
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        config=ConsolidationConfig())
    n = cons._reassign_entities_from_deprecated_class("OldFeature", None)
    assert n == 0
    # The entity keeps its typing (not orphaned).
    assert _out_vertices(store, "E:Bob", "instanceOf") == ["OldFeature"]
    store.close()


def test_reassignment_moves_multiple_entities(tmp_path):
    """All entities typed to the deprecated class move to the parent in one batch."""
    store = _store(tmp_path)
    _plant_class(store, "OldFeature", "Feature")
    _plant_typing(store, "E:Bob", "OldFeature")
    _plant_typing(store, "E:Carol", "OldFeature")
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        config=ConsolidationConfig())
    n = cons._reassign_entities_from_deprecated_class("OldFeature", "Feature")
    assert n == 2
    assert _out_vertices(store, "E:Bob", "instanceOf") == ["Feature"]
    assert _out_vertices(store, "E:Carol", "instanceOf") == ["Feature"]
    store.close()