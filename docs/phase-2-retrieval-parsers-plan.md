# Plan: Phase 2 -- Retrieval integration + cheap parsers (task #17 cont.)

> STATUS: ON HOLD (2026-07-12) -- new possibilities arose after this plan was
> drafted; the user paused before approval. This document is the saved plan so
> we can resume. The implementation has NOT been started.

## Context

Phase 1 shipped as `99eccd3` (document/record ingestion: `Document`/
`DocumentSection` unit model, hot/cold split, markdown+text parsers, forgetting
exemption, CLI). But Phase 1 delivered only **store-level findability** --
`documents_by_entity` / `documents_by_topic` / `default_document_ids`. A
document ingested today does NOT surface in `GraphTraversal.retrieve()`, the
actual RAG path, so the "replacement for RAG" promise (chat IDX 87-88) is still
unmet.

Verified during planning, the gaps are concrete:

- **Entity axis is blind to docs.** `_get_episodes_by_entity`
  (graph_traversal.py:129) walks `out("in_episode")`; docs emit
  `(E:x, appears_in_doc, doc)` NOT `in_episode` -> docs invisible to entity
  queries.
- **Topic axis already finds docs** (`in_("has_topic")`, :134; doc edges are
  "current" so `_filter_current_edges` keeps them) -- no change needed there.
- **No-axis fallback** `_get_all_episode_ids` (:156) enumerates episodes only.
- **`_hydrate`** (:385) calls `get_episode(eid)` -> `None` for docs -> empty
  result dict, so even when a doc IS a candidate it renders as nothing.
- **Temporal filters** `_filter_temporal` (:294) / `_filter_date_range` (:324)
  call `get_episode` -> drop docs.
- **`_filter_active_episodes`** (:216) calls `store.is_episode_active` on every
  candidate; a `doc_` id is not an episode.
- **Renderers** `retriever.build_context_string` (:307) + `chunked_context.
  _format_episode` (:104) emit episode-shaped blocks; docs need a doc block
  that cites `source_path` and pulls the matched section body from the cold
  blob (the cold pull, kept out of the hot LRU).
- **`end_state._build_graph`** (:191) hardcodes `kind="episode"`; **`ssm_chunker.
  expand`** (:218) calls `get_episode` -> raises `EpisodeNotFound` for docs.

The engine (scoring, presentation gate, end-state dispatch, Mode A) is
**type-agnostic** once the result dict is well-formed -- scoring reads
`r["timestamp"]` (works if we fill `ingested_at`), the boost hook already skips
`doc_` (:611). So the integration is a **prefix-gated branch at each
load-bearing site**, keeping the `episode_id` field name (the ~225-touchpoint
rename is blast radius; defer, same call as Phase 1) and adding a `kind`
discriminator.

User-chosen slice: **"Retrieval + cheap parsers"** = the retrieval integration
above + PDFParser (pypdf, **0 new deps** -- already installed 6.14.2) +
CodeParser (tree-sitter-languages, **2 new deps**, user-approved; high dogfood
value on this Python repo). Plus the pre-existing `pyproject.toml` `src.
ingestion` packaging fix.

## A. Retrieval integration

All in `src/retrieval/graph_traversal.py` unless noted. Branch on the `doc_`
prefix; episode paths unchanged (regression-gated).

1. **`_get_episodes_by_entity`** (:120) -- after the `in_episode` walk, ALSO
   walk `appears_in_doc`: `q2 = self.graph.query().vertex(f"E:{entity}").out(
   "appears_in_doc")`, union the vertices, run the combined list through
   `_filter_current_edges(..., "has_entity", f"E:{entity}")` (doc `has_entity`
   edges are current, so they pass). Update the docstring; keep the method name
   (minimizes touchpoints -- it now returns episode + doc ids).
2. **`_get_all_episode_ids`** (:143) -- `return set(self.store.default_episode_ids(
   include_abstracted=False)) | set(self.store.default_document_ids())`.
3. **`_filter_active_episodes`** (:216) -- `if eid.startswith("doc_"): continue`
   into the kept set (docs are always active -- not forgotten). Gated on
   `forgetting_enabled` like the episode branch.
4. **`_filter_temporal`** (:279) / **`_filter_date_range`** (:305) -- for `doc_`
   ids, `doc = self.store.get_document(eid, load_bodies=False)` and use
   `doc.ingested_at` as the timestamp (mirrors the salience-compute type-dispatch
   shipped in Phase 1). `get_document(load_bodies=False)` does NO cold pull.
5. **`_hydrate`** (:377) -- `if eid.startswith("doc_"): return self._hydrate_document(eid)`;
   else the existing episode path (now also sets `kind="episode"`).
6. **NEW `_hydrate_document(eid)`** -- `get_document(eid, load_bodies=False)`
   (NO cold pull here; bodies pulled on demand at render). Return the SAME result
   dict shape as an episode so the type-agnostic engine reads it:
   `{episode_id: doc_id, kind: "document", summary: doc.title, text: "",
   timestamp: doc.ingested_at, entities: doc.entities, topics: doc.topics,
   tones: [], decisions: [], session_id: None, user_id: None, follows: None,
   score: 0.0, source_path: doc.source_path, sections: [{"id","heading",
   "level","blob_hash"} for s in doc.sections]}`. The `sections` list carries
   metadata ONLY (no bodies) -- the hot/cold split gate.
7. **Public aliases** `episodes_by_entity` / `episodes_by_topic` (:368/:373) --
   already correct after (1); no change.

`src/memory/store.py`:

8. **NEW `get_section_body(doc_id, section_id) -> Optional[str]`** -- read hot
   `content/doc/{doc_id}/sec/{sid}/blob_hash`, then `self._blob_store().get_blob(
   hash)`. The single-section cold pull the renderers use; respects the split
   (cold read, never written into the hot store).

`src/retrieval/retriever.py:build_context_string` (:286) and
`src/retrieval/chunked_context.py:_format_episode` (:104):

9. Doc block (branch on `r.get("kind") == "document"` or `eid.startswith("doc_")`):
   `--- Document {eid} ({ts}) ---` + `Source: {source_path}` + `Title: {title}`
   + `Entities:` / `Topics:` + the **matched section body**: the first section
   whose `entities`/`topics` intersect the query axes, cold-pulled via
   `store.get_section_body(doc_id, sec_id)`; if none match, the first section.
   Episode block unchanged (`kind="episode"` default).

`src/retrieval/end_state.py:_build_graph` (:183):

10. `kind = "document" if eid.startswith("doc_") else "episode"`; doc edges
    labeled `appears_in` (keep `appears_in` for UI symmetry with entities).

`src/subconscious/ssm_chunker.py:expand` (:218):

11. `if episode_id.startswith("doc_"): doc = store.get_document(episode_id,
    load_bodies=True)` -> fill the same dict shape. Docs aren't compressed-gist
    chunks in practice, but `expand` must not crash on a doc id.

**Deferred (honest):** semantic-fallback / VectorSearch doc integration. Docs
surface via the graph path (entity/topic/no-axis) with NO embedding work; the
FAISS fallback only fires when graph returns <3 results, and wiring doc section
embeddings there is the separate embedding-backfill slice. Pure-semantic
queries ("a doc about X with no named entities") need that slice -- note as
deferred, not a blocker for the unified graph-path RAG win.

## B. Cheap parsers

`src/ingestion/pdf_parser.py` (NEW) -- `PDFParser` via `pypdf` (0 new deps):

- `parse(source_path)` / `parse_text` mirror: `pypdf.PdfReader(path)`,
  `page.extract_text()` per page. Section structure: use `reader.outline` when
  present (a nested list of `Destination` objects) -> one `RawSection` per
  outline entry, `level` from nesting depth, `content` = the page text under
  that heading; **else one section per page** (heading `Page {n}`, level 1) --
  coarse but honest for TOC-less PDFs. `title` = first top-level outline entry
  or the filename stem; `created_at` from `reader.metadata.creation_date` when
  parseable. Register `.pdf` -> `"pdf"` in `_TYPE_BY_EXT`, `"pdf": PDFParser`
  in `_PARSERS` (parsers.py).

`src/ingestion/code_parser.py` (NEW) -- `CodeParser` via `tree_sitter_languages`
(2 new deps, user-approved):

- Language by extension (`.py`->python, `.js`/`.mjs`->javascript, `.ts`->
  typescript, `.c`/`.h`->c, `.cpp`->cpp, `.go`->go, `.rs`->rust, `.java`->java).
  **Lazy import** `from tree_sitter_languages import get_parser` inside
  `parse()` (GLiNER pattern, `ingest_document.py:_maybe_extractors`); on
  ImportError raise a clear `RuntimeError("CodeParser needs tree-sitter-
  languages: pip install tree-sitter tree-sitter-languages")`.
- Walk the AST; one `RawSection` per `function_definition` /
  `class_definition` / `method_definition`: `heading` = name + signature
  (the `name` child + parameter text), `level` = nesting depth (top-level defs
  at level 2 under a **module-root section at level 1** holding the module
  docstring + imports + top-level statements so they are not lost), `content` =
  the full source span `src[node.start_byte:node.end_byte]`.
- Register `.py`/`.js`/`.ts`/... -> `"code"` in `_TYPE_BY_EXT`, `"code":
  CodeParser` in `_PARSERS`. Add a `parse_text` mirror so the test can run
  without a temp file.
- **Honest caveat:** an oversized Python function (one node, no blank-line
  paragraphs inside) will NOT sub-split -- the chunker's paragraph-boundary
  sub-split won't fire inside a single function node. A code-aware sub-splitter
  (on statement blocks) is a Phase-2 refinement; note it, don't fix it here.
- **Dep risk:** `tree-sitter-languages` may have a version-compat issue in this
  env (cf. the mamba3-cuda lesson). Mitigation: tests
  `pytest.importorskip("tree_sitter_languages")` so the suite stays green; the
  parser raises a clear error if unavailable. IF install proves broken during
  impl, fall back to a stdlib `ast`-based **Python-only** CodeParser (zero deps)
  -- still delivers the repo-dogfood win -- and raise the tree-sitter deferral
  with the user. Flagged as a risk.

## C. Packaging fix + deps

`pyproject.toml`:

- `packages` (:57) `["src", "src.memory", "src.encoding"]` -> add `"src.
  ingestion"` (pre-existing packaging bug; `src.ingestion` exists on disk but
  isn't declared).
- Declare parser deps: add `pypdf>=4.0` to core `dependencies` (already
  installed, PDFParser needs it). Put `tree-sitter` + `tree-sitter-languages`
  in a NEW `[project.optional-dependencies] ingestion` extra (mirrors the `gnn`
  extra pattern) so the offline test suite stays green-by-default; the
  CodeParser test uses `importorskip`. `pip install -e .[ingestion]` opts in.
  (User said "2 new deps" as part of the slice; the extra keeps them opt-in
  without blocking offline dev -- flagged as a minor decision.)

## D. Tests

`tests/test_doc_retrieval.py` (NEW) -- ingest a markdown doc (entities=`["Alice"]`,
topics=`["Storage"]`) into a tmp store via `UnifiedIngestionPipeline`
(structure-only), then:

- `GraphTraversal.retrieve(entities=["Alice"])` -> doc surfaces (entity axis
  now walks `appears_in_doc`); hydrated result has `kind="document"`,
  `summary`=title, `entities`/`topics` populated, `sections` metadata, NO
  bodies in the dict.
- `retrieve(topics=["Storage"])` -> doc surfaces (topic axis, already worked).
- No-axis `retrieve()` -> doc in candidates (union with `default_document_ids`).
- `retrieve(entities=["Nonexistent"])` -> doc NOT in results (entity axis
  honest).
- Temporal: doc with an old `ingested_at` + `retrieve(temporal="last_week")`
  -> filtered correctly; `temporal="all_time"` keeps it.
- `build_context_string` + `ChunkedContextFormatter._format_episode` -> doc
  block cites `source_path` + the matched section body pulled from the cold
  blob (assert the body text appears, proving the cold pull works).
- `end_state._build_graph` -> doc node has `kind="document"`.
- `_apply_retrieval_boost` -> still writes NO sidecar for the doc result
  (regression for the Phase-1 exemption).
- `ssm_chunker.expand(doc_id, store=store)` -> returns a dict, no
  `EpisodeNotFound`.

`tests/test_pdf_parser.py` (NEW) -- commit a tiny `tests/fixtures/sample.pdf`
(binary, deterministic) OR build one in-memory with `pypdf.PdfWriter`;
`PDFParser().parse(fix)` -> sections present, title set, page text extractable.

`tests/test_code_parser.py` (NEW) -- `pytest.importorskip("tree_sitter_
languages")`; `CodeParser().parse_text(<small .py snippet>)` -> function +
class sections with correct headings (name + signature) and full source spans;
module-root section holds the imports.

Regression: `pytest tests/test_graph_traversal.py tests/test_retriever.py
tests/test_chunked_context.py tests/test_end_state.py tests/test_ssm_chunker.py
tests/test_ingestion.py tests/test_document.py -q` -- doc branches are
prefix-gated; episode dicts now carry `kind="episode"` (consumers use `.get`,
so safe).

## E. De-wonk + commit

Run the de-wonk skill (CLAUDE.md) across the changed surface. Watch for:

- **Doc dict key gap** -> engine KeyError. Gate: every doc result dict has the
  SAME keys as an episode dict (fill defaults) PLUS doc extras. `test_doc_
  retrieval` gates this.
- **`_hydrate_document` must NOT pull bodies** -- verify `load_bodies=False`;
  bodies only via `get_section_body` at render (cold read, never written hot).
- **`_filter_active_episodes` doc-skip must NOT bypass the forgetting filter
  for episodes** -- prefix-gated, episode path byte-identical.
- **Entity-axis `appears_in_doc` union must NOT pollute episode results** --
  additive union; episode ids unchanged; regression gate.
- **`get_section_body` cold pull must NOT cache in the hot store** -- uses
  `_blob_store().get_blob` (cold).
- **CodeParser lazy import** raises a clear error, not a bare ImportError.
- **`kind` discriminator symmetry** -- `_hydrate` sets `kind="episode"`;
  verify consumers ignore unknown keys (`.get`).
- No TODO/stub left; ASCII-only in print()/argparse.

Commit at will ([[commit-at-will]]) after de-wonk + tests green. Update memory
`hippo-doc-ingestion-gap`: Phase 2 shipped (retrieval integration + PDF/code
parsers + `src.ingestion` packaging fix); remaining = Phase 3 media (Whisper/
vision), Phase 4 citation/contradiction/eval, semantic-fallback vector doc
integration.

## F. Verification

1. `pip install tree-sitter tree-sitter-languages` (CodeParser; PDF needs
   nothing new).
2. `pytest tests/test_doc_retrieval.py tests/test_pdf_parser.py tests/test_
   code_parser.py -q` -- new tests.
3. `pytest tests/test_graph_traversal.py tests/test_retriever.py tests/test_
   chunked_context.py tests/test_end_state.py tests/test_ssm_chunker.py tests/
   test_ingestion.py tests/test_document.py -q` -- regression.
4. `pytest -q` -- full suite green.
5. Manual dogfood: `python scripts/ingest_document.py --source src/retrieval/
   graph_traversal.py --type code` (ingest this repo), then a retrieval query
   hitting an entity/topic in it -> doc surfaces with the matched function body
   in context. `--source README.md --type markdown` -> same. `--source
   tests/fixtures/sample.pdf` -> PDF parsed and retrievable.
6. de-wonk clean; commit; update memory.

## Risks + honest caveats

- **R-tree-sitter-deps:** tree-sitter-languages may be broken in this env
  (cf. mamba3-cuda). Mitigation: `importorskip` tests, lazy import + clear
  error, `ast`-fallback ready.
- **R-engine-key-gap:** a doc result dict missing a key the engine reads ->
  crash. Mitigation: same-key contract + test gate.
- **R-cold-pull-leak:** `_hydrate_document` accidentally pulling bodies into
  the hot LRU. Mitigation: `load_bodies=False` + test gate (assert no body
  text in the hydrated dict, only `blob_hash` refs).
- **R-entity-pollution:** `appears_in_doc` walk polluting episode entity
  results. Mitigation: additive union, episode ids unchanged, regression gate.
- **R-retrieval-blend:** mixing docs + episodes in one ranked list may surface
  a doc above a more-relevant episode. Acceptable for this slice (the unified
  RAG win); a `kind`-aware rerank is a later refinement. Note it.
- **R-pdf-structure:** TOC-less PDFs fall back to one-section-per-page (coarse
  but honest); `extract_text` quality varies on columns/tables. Note; not a
  blocker.
- **R-semantic-fallback-deferred:** docs do NOT surface in the VectorSearch
  fallback until section embeddings are backfilled. Honest -- the graph path
  covers entity/topic/no-axis queries; pure-semantic needs the embedding slice.
- **R-code-oversize:** oversized single functions won't sub-split (noted
  caveat; code-aware sub-splitter is a refinement, not this slice).