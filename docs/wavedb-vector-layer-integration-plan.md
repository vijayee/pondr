# Use the WaveDB VectorLayer as Hippo's vector index

Date: 2026-07-14
Status: Plan (pending approval) — saved per user directive "we should have a
saved plan for how to used the db". **Wheel build slice DONE** (0.2.0 Windows
wheel built + installed into Hippo's env; VectorLayer imports + round-trips;
see §"Windows wheel build" + §"Verification"). Hippo integration code is the
remaining, approval-pending work.

## Context / why now

WaveDB gained a native **VectorLayer** (commits `4526442`..`84125e4`, after
the published `0.1.16`). It is a real vector index living *inside* the same
WaveDB database Hippo already uses — FLAT (exact), IVF (k-means), and SLSH
(sortable LSH, bidirectional scan) backends, sync Python API, exposed as
`from wavedb import VectorLayer, Format, Runtime, IndexType, Distance`. The
C layer is built by default; `src/wavedb.def` already exports all 17
`vector_layer_*` symbols (+ the engine reverse-scan symbols), so the R7 MSVC
export risk is already handled — a Windows rebuild will expose them.

Hippo today does vector search with a **separate FAISS sidecar**
(`src/retrieval/vector_search.py` + `scripts/build_vector_index.py`): an
offline job reads `content/ep/{eid}/embedding`, builds `vector_index.faiss`,
and the retriever loads it as a semantic fallback. Two pain points the
runtime-gap work (commit `8604678`) made sharper:

1. **The FAISS sidecar is not updated live.** `_persist_exchange` now
   backfills `summary_embedding` into WaveDB on every query, and the graph
   path surfaces new episodes immediately — but the semantic-fallback
   (FAISS) won't see a new episode until an offline `build_vector_index.py`
   rebuild. Documented as a caveat in the runtime-gap commit.
2. **FAISS can't delete.** `forget` (active-forget) / `reconsolidate`
   (supersede) take an episode out of the default graph/axis queries, but it
   stays in the FAISS index until a full rebuild — a deprecated episode can
   still be returned by the semantic fallback.

The WaveDB VectorLayer fixes both: insert is a single in-process
`insert_sync(eid, vec)` (live, atomic with the episode), and `delete_sync(eid)`
removes a forgotten/superseded episode from the vector index immediately. No
separate `.faiss` file, no offline rebuild for the live path, no numpy/faiss
dependency for the common path. This plan wires Hippo to use it.

## VectorLayer API surface (verified from `bindings/python/src/wavedb/vector_layer.py`)

Import: `from wavedb import VectorLayer, Format, Runtime, IndexType, Distance`

- `IndexType`: `FLAT=0`, `IVF=1`, `SLSH=2`. `Distance`: `L2=0`, `COSINE=1`, `DOT=2`.
- `Format(index_type, dim, delimiter="/", distance=COSINE, ivf_n_clusters=0,
  slsh_lsh_tables=0, slsh_hash_bits=0, slsh_bucket_width=0.0)` — **immutable
  after create**.
- `Runtime(top_k=10, sync_only=1, ivf_nprobe=8, ivf_flat_until=1000,
  slsh_scan_radius=200)` — **mutable via `reconfigure(rt)`**.
- `VectorLayer.open(index_name, db, fmt, rt, subtree=None)` — shares the
  Hippo `WaveDB`'s underlying `database_t`; keys land under `{index_name}/`.
  Registers with `db._open_subtrees` so the parent `WaveDB` refuses to close
  while the layer is open. **This is the constructor we want** — no second DB
  file, index lives in the Hippo store's own WaveDB.
- `VectorLayer.open_separate(db_location, index_name, fmt, rt)` — dedicated
  DB (not used here).
- `insert_sync(id: str, vec, metadata: bytes = b"") -> int` (raises on rc<0;
  `vec` any iterable of floats; `id` must be `str`, `metadata` bytes).
- `search_sync(query, k: int) -> list[VectorResult]` — `VectorResult` has
  `.id_str`, `.distance` (float), `.metadata_str` (utf-8, errors=replace).
- `delete_sync(id: str) -> int`, `train() -> int`, `rebuild() -> int`,
  `count() -> int`, `reconfigure(rt) -> int`, `close()`.
- Context manager + `__del__` call `close()`. `_check_open()` raises if closed.
- **Sync API only.** The async (`promise_t`) API is declared in cdef but not
  exposed by the Python wrapper (trampoline deferred). Fine for Hippo —
  retrieval is single-threaded.

## Index choice: FLAT first, IVF graduation path

Hippo embeds episode summaries with `bge-small-en-v1.5` (384-dim, cosine).
Episode-summary embeddings are **clustered** (semantic clusters), and the
corpus is small-to-medium with **incremental live growth** (one episode per
query) plus occasional bulk corpus builds.

- **Ship FLAT first.** Exact (recall@10 = 1.0 by definition), no `train()`
  needed, zero-config, and it is also IVF's cold-start fallback
  (`ivf_flat_until=1000`). For Hippo's N (hundreds to low thousands) FLAT is
  fast enough (bench: ~60 ms p50 @ 10k) and removes the "did you train?" footgun.
- **Graduate to IVF** when N exceeds ~1000 and search latency matters: set
  `index_type=IVF`, `ivf_n_clusters ~ sqrt(N)`, call `train()` after the bulk
  build, `ivf_nprobe=8` (raise to 16-32 if recall dips). IVF gives 0.96-0.99
  recall on clustered embeddings at ~3x FLAT speed. This is a config flip +
  one `train()` call — documented as the path, NOT wired in the first slice.
- **SLSH** only if embeddings turn out uniform/gaussian (IVF degenerates there,
  ~0.36 recall). Unlikely for semantic episode embeddings; not wired now.

Recommendation: first slice ships `Format(index_type=FLAT, dim=384,
distance=COSINE)` + `Runtime(top_k=10, sync_only=1)`. The FLAT-first choice
also sidesteps the unmeasured-real-arm caveat (synthetic-only bench) — FLAT is
exact, so recall is not in question.

## Integration design

### 1. `HippocampalStore` owns the `VectorLayer` (gated, lifecycle-safe)

`src/memory/store.py`:

- `__init__`: after `self.db = WaveDB(...)`, if `VectorLayer` is importable AND
  a new `config.vector_index_enabled` (default **True**) is set, open
  `self.vector_layer = VectorLayer.open("episodes", self.db, fmt=Format(FLAT,
  384, COSINE), rt=Runtime(top_k=10, sync_only=1))`. Else `self.vector_layer =
  None`. The gate on `hasattr(wavedb, "VectorLayer")` keeps Hippo working on
  old WaveDB (<0.2.0) and in the offline test suite (no vector build needed).
- `close()`: if `self.vector_layer is not None: self.vector_layer.close()`
  BEFORE `self.db.close()`. The binding already refuses to close the parent
  while a layer is open; closing in this order is belt-and-suspenders.
- `dim` comes from `config.embedding_dim` (384) — do not hardcode.

### 2. One embedding-write chokepoint -> upsert into the VectorLayer

Today embeddings are written in two places: `encode_episode` (inline, when
`episode.summary_embedding` is set — the live path) and `set_summary_embedding`
(the backfill path used by `build_vector_index.py`). Route both through one
private helper:

```
def _index_embedding(self, eid: str, vec: list[float], summary: str) -> None:
    if self.vector_layer is None:
        return
    try:
        self.vector_layer.insert_sync(eid, vec, metadata=summary.encode("utf-8"))
    except Exception as e:  # noqa: BLE001 - vector index is best-effort
        print(f"[vector-index-fail] {eid}: {e}", file=sys.stderr)
```

- `set_summary_embedding(eid, vec)`: after writing `content/ep/{eid}/embedding`,
  call `self._index_embedding(eid, vec, summary=self.get_episode(eid).summary or "")`.
- `encode_episode`: the inline `if episode.summary_embedding: ops.append(... /
  embedding ...)` block writes the JSON in the batch; AFTER `batch_sync(ops)`,
  if `episode.summary_embedding`, call `self._index_embedding(eid,
  episode.summary_embedding, episode.summary)`. (The VectorLayer insert is a
  separate `insert_sync` — it cannot ride in the content `batch_sync`; it is
  its own atomic op on the same database_t.)
- Best-effort, logged, never raises — a vector-index hiccup must never fail an
  episode encode (mirrors `_persist_exchange`'s philosophy). An episode with a
  missing vector entry is still fully retrievable via the graph path.

This makes the vector index **always match the content index**, for BOTH the
live path and the offline bulk build, with zero extra wiring in callers.

### 3. `forget` / `reconsolidate` -> `delete_sync` (the FAISS-can't-do win)

- `set_episode_state(episode_id, state, ...)`: when `state` in
  (`"deprecated"`, `"superseded"`) and `self.vector_layer is not None`, call
  `self.vector_layer.delete_sync(episode_id)` (best-effort, logged). A
  forgotten/superseded episode leaves the vector index immediately — no full
  rebuild needed. (A later `set_episode_state(..., "current")` revival re-inserts
  on its next encode; revivals are rare. Documented, not auto-reinserted.)

### 4. `WavedbVectorStore` adapter for the retriever (same `search` interface)

`src/retrieval/wavedb_vector_store.py` (NEW) — a thin adapter so the retriever's
existing `self.vector_search` slot works unchanged:

```
class WavedbVectorStore:
    def __init__(self, store, embedder): ...
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if store.vector_layer is None: return []
        vec = embedder.encode([query])[0]
        results = store.vector_layer.search_sync(vec, k)
        return [(r.id_str, r.distance) for r in results]
```

`HippocampalRetriever`: in `_attach_vector_index` / `__init__`, prefer the
`WavedbVectorStore` when `store.vector_layer is not None`; fall back to the
existing `VectorSearch` (FAISS / pure-Python) when the layer is absent or
disabled. The retriever's `search(prompt, k)` call site (retriever.py:249) is
unchanged — the adapter matches `VectorSearch.search`'s signature and return
shape `[(eid, score)]`. `build_index` is a no-op for the WaveDB path (the
index is maintained live by the store chokepoint); keep the FAISS
`build_vector_index.py` script for the one-time bulk backfill of pre-existing
episodes (it calls `set_summary_embedding`, which now also upserts into the
VectorLayer — so the bulk build populates BOTH indexes).

### 5. Live insert closes the runtime-gap caveat

Because the store chokepoint upserts on every `encode_episode` with an
embedding, a newly live-encoded episode (from `PonderOrchestrator.query` ->
`_persist_exchange` -> `encode_messages` -> `encode_episode`) is
**immediately searchable** via `vector_layer.search_sync` — no FAISS rebuild.
The runtime-gap commit's "FAISS sidecar not updated live" caveat is closed.

## Windows wheel build + install into Hippo

The prebuilt `build-msvc12/Release/wavedb.dll` is **stale** (mtime 2026-07-04;
no `vector_layer_*` exports — confirmed). Rebuild from current source:

1. **Rebuild `libwavedb.dll`** (WaveDB repo, current source — vector layer
   included by default; `src/wavedb.def` already exports the 17 vector
   symbols + reverse-scan, so MSVC will expose them):
   `cmake -S . -B build-msvc12 -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=OFF
   -DCMAKE_BUILD_TYPE=Release` then `cmake --build build-msvc12 --config
   Release --target wavedb_shared`. Verify exports:
   `dumpbin /exports build-msvc12/Release/wavedb.dll | findstr vector_layer`
   (expect 17).
2. **Bump version** in TWO places (a wheel reporting the wrong runtime
   version is a publish defect):
   - `bindings/python/pyproject.toml`: `0.1.16` -> `0.2.0` (pip metadata).
   - `bindings/python/src/wavedb/__init__.py`: `__version__ = "0.1.16"` ->
     `"0.2.0"` (the runtime `wavedb.__version__` attribute — hardcoded, NOT
     derived from importlib.metadata; pip reports 0.2.0 but `wavedb.__version__`
     would still say 0.1.16 if this isn't bumped). Caught + fixed during the
     build: the first installed wheel passed the import check but failed the
     version check on this string.
3. **Build the wheel**: from `bindings/python/`,
   `python -m build --wheel` (or `pip wheel . -w dist/`). `setup.py`'s
   `_find_prebuilt_lib()` will pick up the freshly rebuilt
   `build-msvc12/Release/wavedb.dll` and bundle it into `wavedb/_lib/wavedb.dll`.
4. **Install into Hippo**: `pip install --force-reinstall --no-deps
   dist/wavedb-0.2.0-cp3xx-win_amd64.whl`. Verify:
   `python -c "from wavedb import VectorLayer; print('ok')"`.
5. **Bump Hippo's pin**: `pyproject.toml` `wavedb>=0.1.4` -> `wavedb>=0.2.0`
   (comment: `>=0.2.0: VectorLayer native vector index`). Old WaveDB falls back
   to the FAISS `VectorSearch` path via the `hasattr` gate, so the pin bump is
   safe for anyone who hasn't upgraded.
6. **PyPI upload**: optional / separate step (`twine upload dist/*`). The user
   said "publish a python wheel for windows" — building + installing locally
   into Hippo satisfies "start using it"; PyPI upload is a one-line follow-up
   when the user wants it public.

## Files to modify

- `src/memory/store.py` — open/close the `VectorLayer`; `_index_embedding`
  helper called from `encode_episode` + `set_summary_embedding`;
  `set_episode_state` -> `delete_sync` on deprecate/supersede.
- `src/retrieval/wavedb_vector_store.py` (NEW) — `WavedbVectorStore` adapter
  (`search(prompt, k) -> [(eid, score)]`).
- `src/retrieval/retriever.py` — prefer `WavedbVectorStore` when
  `store.vector_layer` is set; fall back to `VectorSearch`.
- `src/config.py` — `vector_index_enabled: bool = True`, `embedding_dim: int = 384`.
- `pyproject.toml` — `wavedb>=0.2.0`.
- (WaveDB repo) `bindings/python/pyproject.toml` version bump 0.1.16 -> 0.2.0;
  rebuild DLL; build wheel.

No change to `src/retrieval/vector_search.py` — kept as the fallback.
`scripts/build_vector_index.py` unchanged (it calls `set_summary_embedding`,
which now also feeds the VectorLayer — one bulk build populates both).

## Tests

`tests/test_wavedb_vector_store.py` (NEW) — requires `wavedb>=0.2.0`
(VectorLayer importable); skip cleanly via `pytest.importorskip("wavedb",
minversion="0.2.0")` or `hasattr` gate if not:

1. Open a tmp `HippocampalStore`; `store.vector_layer` is not None; `count()==0`.
2. `set_summary_embedding(eid, vec)` -> `count()==1` and `search_sync(vec, 1)`
   returns `eid`.
3. `encode_episode` of an episode WITH `summary_embedding` -> immediately
   searchable via `search_sync` (no `build_index` call) — the live-insert win.
4. `WavedbVectorStore.search(prompt, k)` returns `[(eid, score)]` using a stub
   embedder (matches `VectorSearch.search`'s signature/shape).
5. `forget(eid)` (set_episode_state "deprecated") -> `count()` drops by 1 and
   `search_sync` no longer returns `eid` — the delete-on-forget win.
6. `reconsolidate(old, new)` -> `old` leaves the vector index.
7. Best-effort: force `insert_sync` to raise (monkeypatch) -> logged, episode
   still encoded, no exception propagates.
8. Lifecycle: `store.close()` closes the layer; a second open after close works.
9. Gating: with `vector_index_enabled=False` (or old wavedb mocked), the store
   has `vector_layer is None` and the retriever falls back to `VectorSearch`
   (existing `tests/test_retriever.py` / `test_vector_search.py` stay green).

Regression: `tests/test_retriever*.py`, `tests/test_vector_search.py`,
`tests/test_orchestrator*.py` (the live-encode path now also inserts into the
vector index — assert it doesn't break the existing orchestrator tests; the
stub embedder's vectors are deterministic so the insert is a no-risk add).
Full `pytest -q`.

## De-wonk (CLAUDE.md, at completion gate)

- **Two vector backends?** `VectorSearch` (FAISS/pure-Python) is the FALLBACK,
  not a parallel path — the retriever uses one OR the other based on
  `store.vector_layer`. Verify no code path queries both.
- **One embedding-write chokepoint** — verify BOTH `encode_episode` and
  `set_summary_embedding` route through `_index_embedding` (no third write
  site). `build_vector_index.py` already uses `set_summary_embedding`, so it
  feeds the VectorLayer for free.
- **Best-effort symmetry** — `_index_embedding` and `delete_sync` mirror
  `_persist_exchange`'s "never lose the response/episode" philosophy; verify
  both are try/except + stderr-logged + non-raising.
- **Lifecycle** — layer closed in `store.close()` BEFORE `db.close()`; verify
  no use-after-close (re-open path in test 8).
- **No `train()` left hanging** — FLAT needs no train; if a future slice flips
  to IVF, the train call is documented in the graduation path, not TODO'd here.
- **`dim` not hardcoded** — read from `config.embedding_dim`; verify the
  `Format` uses it.
- ASCII-only in print()/argparse/help.

## Verification

1. **DONE (2026-07-14):** Wheel built + installed into Hippo's env
   (`wavedb-0.2.0-cp314-cp314-win_amd64.whl`, 443940 B). `from wavedb import
   VectorLayer` succeeds; `wavedb.__version__ == "0.2.0"`. Functional
   round-trip on a tmp WaveDB: `VectorLayer.open('episodes', db, Format(FLAT,
   4, COSINE), Runtime(top_k=3, sync_only=1))` -> `insert_sync` 3 vecs ->
   `search_sync` returns the correct top-2 (cosine distances 0.0 identical,
   0.0061 near-neighbor) -> `delete_sync('ep1')` -> re-search no longer
   returns `ep1`. The integration code below is the remaining (un-started,
   approval-pending) work.
2. `pytest tests/test_wavedb_vector_store.py -q` (new).
3. `pytest tests/test_retriever*.py tests/test_vector_search.py
   tests/test_orchestrator*.py -q` (regression).
4. `pytest -q` (full suite green).
5. Manual: with a real (stub-embedder) tmp store, encode 3 episodes with
   embeddings, `WavedbVectorStore.search("<summary text>", 2)` returns the 2
   nearest; `forget(eid)` removes one; re-search no longer returns it; the
   graph path still returns it via `include_inactive=True`.
6. de-wonk clean; commit at will ([[commit-at-will]]); update memory
   ([[hippo-phase-2c-status]] runtime-gap caveats — the FAISS-sidecar-live
   caveat is now closed by the in-DB VectorLayer).

## Honest caveats

- **FLAT scales linearly.** ~60 ms p50 @ 10k 384-dim (synthetic bench). Fine
  for Hippo's N today; the IVF graduation path (above) is the lever when N
  grows. IVF/SLSH need `train()` after a bulk load and degenerate on uniform
  data — not wired in the first slice.
- **Real-arm recall unmeasured.** The WaveDB bench is synthetic only
  (sentence-transformers unavailable on the bench box). FLAT is exact so
  recall is not in question for the shipped path; IVF/SLSH recall on real
  bge-small embeddings is unmeasured — another reason to ship FLAT first.
- **VectorLayer is sync-only** in the Python wrapper (async deferred). Hippo
  retrieval is single-threaded, so this is fine; do not plan on concurrent
  vector inserts from multiple threads.
- **The VectorLayer shares the Hippo WaveDB database_t.** It must stay open
  for the store's lifetime; closing the parent `WaveDB` while the layer is
  open is refused by the binding. The store closes the layer first in
  `close()`.
- **Revival not auto-reinserted.** A deprecated-then-revived episode is gone
  from the vector index until re-encoded. Revivals are rare; documented, not
  auto-handled (avoids a re-embed cost on a rare path).
- **Metadata is bytes.** `insert_sync` metadata is `bytes`; the adapter
  carries the summary for cheap hydration, but the retriever hydrates via
  `get_episode(eid)` anyway (matching `VectorSearch`), so metadata is a
  convenience, not a dependency.
- **PyPI upload is optional.** This plan builds the wheel + installs into
  Hippo; publishing to PyPI is a separate one-line step the user can request.

## Phasing

- **Slice A (this plan):** FLAT-first in-DB vector index; store-owned layer;
  live insert closes the runtime-gap FAISS caveat; delete-on-forget; retriever
  adapter + FAISS fallback. Wheel built + Hippo bumped to 0.2.0.
- **Slice B (later):** IVF graduation when N > 1000 (config flip + `train()`
  after bulk build) if latency warrants; measure real-arm recall.
- **Out of scope:** SLSH (only if embeddings prove uniform), the async
  VectorLayer API, FAISS removal (kept as fallback indefinitely).