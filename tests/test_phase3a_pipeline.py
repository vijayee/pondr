"""End-to-end Phase 3a pipeline test (no GPU, no Bonsai, untrained model).

Exercises the full CPU-dev slice in one pass: encode episodes → load a PyG
subgraph → run the GNN → run a dry-run consolidation pass → verify the report.
This is the integration counterpart to the per-module tests in
``test_gnn_*``, ``test_semantic_memory``, ``test_consolidate``.
"""

from __future__ import annotations

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.gnn import Consolidator, WaveDBGraphLoader, GNNModel


def _populate(store, n=5):
    for i in range(1, n + 1):
        store.encode_episode(Episode(
            id=f"ep_00000{i}", timestamp=f"2026-07-0{i}", summary=f"summary {i}",
            full_text=f"User: u{i}\nAssistant: a{i}",
            entities=["Alice", "Bob"], topics=["db", "planning"],
            tones=["curious"],
        ))


def test_pipeline_encode_load_score_consolidate(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    _populate(store)

    # 1. loader produces a well-shaped PyG Data over the real graph.
    loader = WaveDBGraphLoader(store, radius=2)
    data = loader.load("ep_000001")
    assert data.x.shape[1] == 384
    assert data.edge_index.shape[1] >= 1
    assert int(data.center_idx) == 0

    # 2. model forward over the loaded subgraph yields all 5 head outputs.
    model = GNNModel(hidden_dim=64, num_heads=2, num_layers=2,
                     predicate_vocab_size=32, num_clusters=8)
    model.eval()
    out = model(data)
    n = data.x.shape[0]
    assert out["salience"].shape == (n,)
    assert out["anomaly"].shape == (n, 6)
    assert out["diffpool"].shape == (n, 8)

    # 3. a dry-run consolidation pass over the corpus produces a report and
    #    mutates nothing.
    cons = Consolidator(store, dry_run=True)
    rep = cons.run(limit=3)
    assert rep["dry_run"] is True
    assert rep["subgraphs_scored"] == 3
    assert not any(store.is_abstracted(f"ep_00000{i}") for i in range(1, 6))

    # 4. an apply pass (forced, untrained) writes semantic memories + marks
    #    sources abstracted, and default queries then exclude them.
    cons_apply = Consolidator(store, dry_run=False, allow_untrained_apply=True)
    rep_apply = cons_apply.run(limit=3)
    assert "apply_skipped" not in rep_apply
    mems = sorted({k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")})
    assert len(mems) >= 1
    # At least one source episode was abstracted.
    assert any(store.is_abstracted(f"ep_00000{i}") for i in range(1, 6))
    store.close()