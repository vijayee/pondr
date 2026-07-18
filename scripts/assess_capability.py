#!/usr/bin/env python3
"""Total-package capability assessment (Option 1: smoke + reuse scorecard).

Runs the REAL shipped runtime end-to-end and collects, in one place, every
measurement we already have gold/harness for -- so "what can the total package
even do" gets a single answer as a scorecard mapping each Phase 5 metric to
{measured value | gold-missing reason | target}. Builds NO new gold; many
cells read "gold-missing" and that IS the finding (tells us which measurements
we'd need before piece-optimization is justifiable).

Reuses (does not reinvent):
  - tests/test_extraction_quality.py gold + recall helpers (Stage 1)
  - tests/test_end_to_end.py gold-label retrieval queries + corpus builders (Stage 2)
  - tests/test_enterpriserag_eval.py scoring + fixture (Stage 5a)
  - src/gnn/bonsai_decider.py::_deterministic_non_conflict + the 200-pair
    reconstruction from scripts/_scratch/guard_coverage_200.py (Stage 5b)
  - src/runtime.py::build_ponder + scripts/run_consolidation.py (Stages 0, 4)
  - src/orchestrator.py::query() per-query metrics envelope (Stage 0/3)

Every stage degrades to a documented cell on failure/missing-dep -- the run
never crashes, and a blank cell is a bug (de-wonk: no silent caps).

Usage:
  python scripts/assess_capability.py [--db <waveDB>] [--llm-endpoint URL]
      [--llm-model MODEL] [--gnn-ckpt PATH] [--consolidation-db <waveDB>]
      [--stage {all,smoke,encoding,retrieval,routing,consolidation,contradiction}]
      [--out data/assessment/scorecard]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Defaults (verified on disk 2026-07-18)
# ---------------------------------------------------------------------------
DEFAULT_DB = "data/pod_runs/phase1b_scale/ingest_db_dialogsum_backfilled_full"
DEFAULT_LLM_ENDPOINT = "http://localhost:11434/v1"  # Oracle/Ollama (Bonsai killed)
DEFAULT_LLM_MODEL = "deepseek-v4-flash:cloud"  # user prefers flash over pro
DEFAULT_GNN_CKPT = "data/pod_runs/phase3a/all_fixed_bounded.pt"
DEFAULT_BACKBONE = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
DEFAULT_GATE = "data/pod_runs/phase2b/best.pt"
DEFAULT_OUT = "data/assessment/scorecard"

# Phase 5 metric targets (docs/Ponder Engine Phases.md:655-665)
TARGETS = {
    "runtime_runs": "build_ponder + query() on trained 2a backbone + 2b gate",
    "encoding": ">90% entity F1, >85% relation F1",
    "retrieval_recall": ">80% single-session, >60% cross-session",
    "retrieval_precision": ">85%",
    "retrieval_latency": "<50ms graph traversal + HBTrie load",
    "routing": ">90% correct domain routing",
    "consolidation": ">70% GNN-predicted edges validated by Bonsai",
    "forgetting": "<5% of pruned edges later needed",
    "context_efficiency": "graph context <=50% size of full-history for equal recall",
    "uncertainty": ">80% precision on 'I don't know'",
    "delegation": ">80% of queries handled by <=8B model",
}

STAGES = ("smoke", "encoding", "retrieval", "routing", "consolidation", "contradiction")


def _cell(status, value=None, target=None, notes=""):
    """One scorecard cell. status in {measured, gold_missing, endpoint_down,
    dependency_missing, not_run, error}. Keeping status explicit so the
    scorecard never reads a measured number where none was taken (de-wonk:
    gold-missing/not_run must be visually distinct from measured)."""
    return {"status": status, "value": value, "target": target, "notes": notes}


def _endpoint_up(endpoint: str, timeout: float = 2.0) -> bool:
    import requests
    try:
        r = requests.get(f"{endpoint.rstrip('/')}/models", timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


# ===========================================================================
# Stage 0 -- Runtime smoke: does the total package run today?
# ===========================================================================
def stage_smoke(db, llm_endpoint, llm_model, backbone, gate):
    """Build a live PonderOrchestrator on trained checkpoints and run queries.

    Copies the DB to a temp dir first so the assessment never mutates the
    precious ingested corpus (query() records presentation outcomes -> writes).
    """
    out = {"headline": None, "endpoint_up": _endpoint_up(llm_endpoint),
           "queries": []}
    if not Path(backbone).is_file():
        out["headline"] = _cell("dependency_missing", None,
                                TARGETS["runtime_runs"],
                                f"backbone checkpoint missing: {backbone}")
        return out
    if not Path(gate).is_file():
        out["headline"] = _cell("dependency_missing", None, TARGETS["runtime_runs"],
                                f"gate checkpoint missing: {gate}")
        return out
    if not Path(db).is_dir():
        out["headline"] = _cell("dependency_missing", None, TARGETS["runtime_runs"],
                                f"WaveDB dir missing: {db}")
        return out

    from src.runtime import build_ponder  # noqa: E402

    # Copy the DB so query() outcome-writes don't mutate the source corpus.
    tmp = tempfile.mkdtemp(prefix="pondr_assess_")
    db_copy = Path(tmp) / "db"
    orch = None
    try:
        try:
            shutil.copytree(db, db_copy)
        except Exception as e:
            out["headline"] = _cell("error", None, TARGETS["runtime_runs"],
                                    f"DB copy failed: {e}")
            return out

        queries = [
            "What did the customer want?",
            "What was discussed about the database?",
            "hello",  # trivial -> likely ssm_direct / unsupported
        ]
        try:
            # ModeA reads config.generation_model at construction; the env var
            # only applies at first import (config is a module singleton already
            # imported by earlier stages), so mutate the singleton instance
            # directly to make --llm-model actually take effect.
            # (`from ..config import config` in mode_a.py binds the Config()
            # instance, so mutate THAT, not the module.)
            os.environ["GENERATION_MODEL"] = llm_model
            try:
                from src.config import config as _cfg  # noqa: E402
                _cfg.generation_model = llm_model
            except Exception:
                pass
            orch = build_ponder(
                str(db_copy),
                backbone_path=backbone,
                gate_path=gate,
                embedder_source="on-demand",
                bonsai_endpoint=llm_endpoint,
                device="auto",
                live_encode=False,  # no GLiNER, no live-encode writes
            )
        except Exception as e:
            out["headline"] = _cell("error", None, TARGETS["runtime_runs"],
                                    f"build_ponder failed: {e}")
            return out

        try:
            for q in queries:
                t0 = time.perf_counter()
                try:
                    result = orch.query(q)
                except Exception as e:
                    out["queries"].append({"query": q, "error": str(e)})
                    continue
                lat_ms = (time.perf_counter() - t0) * 1000.0
                route = result.get("route")
                pathway = getattr(route, "pathway", None) if route else None
                supported = result.get("supported")
                n_ret = len(result.get("retrieved_episodes") or [])
                end_state = result.get("end_state_plan")
                resp = result.get("response")
                resp_len = len(resp) if isinstance(resp, str) else 0
                out["queries"].append({
                    "query": q, "pathway": pathway, "supported": supported,
                    "retrieved": n_ret, "end_state": end_state,
                    "response_len": resp_len, "latency_ms": round(lat_ms, 1),
                    "measured_expand_count": result.get("measured_expand_count"),
                })
            runs = bool(out["queries"]) and all("error" not in q for q in out["queries"])
            n_ret_total = sum(q.get("retrieved", 0) for q in out["queries"])
            notes = (f"build_ponder constructed on trained backbone+gate; "
                     f"{len(out['queries'])} queries ran; endpoint_up={out['endpoint_up']}; "
                     f"response non-empty requires endpoint_up + synthesize end-state.")
            out["headline"] = _cell(
                "measured" if runs else "error",
                {"ran": runs, "queries_run": len(out["queries"]),
                 "retrieved_total": n_ret_total, "endpoint_up": out["endpoint_up"]},
                TARGETS["runtime_runs"], notes)
        finally:
            if orch is not None:
                try:
                    orch.store.close()
                except Exception:
                    pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ===========================================================================
# Stage 1 -- Encoding recall (reuse test_extraction_quality gold + helpers)
# ===========================================================================
def _recall(expected, extracted):
    if not expected:
        return 1.0
    return len(set(expected) & set(extracted)) / len(expected)


def _topic_recall(expected_labels, extracted_spans):
    if not expected_labels:
        return 1.0
    spans_low = [s.lower() for s in extracted_spans if isinstance(s, str)]
    hits = 0
    for label in expected_labels:
        keywords = [k for k in label.lower().split("_") if k]
        if not keywords:
            continue
        if any(kw in span for kw in keywords for span in spans_low):
            hits += 1
    return hits / len(expected_labels)


def stage_encoding(gliner_device="cpu"):
    """Entity/topic/tone recall on the 20 hand-labeled sample conversations.

    NOTE: this is recall on 20 hand-crafted convs (a Phase 1a DoD proxy), NOT
    F1 on a held-out bench. Replicates the helpers from test_extraction_quality
    verbatim (that module's module-level pytest.importorskip makes importing it
    outside pytest fragile); the GOLD (data/sample_conversations.jsonl) is
    reused as-is. ``gliner_device`` defaults to CPU; pass 'cuda'/'auto' to
    separate the CPU-vs-GPU confidence divergence ([[hippo-gliner-threshold-cpu-vs-gpu]])
    from a true extraction regression.
    """
    corpus = ROOT / "data" / "sample_conversations.jsonl"
    if not corpus.is_file():
        return _cell("dependency_missing", None, TARGETS["encoding"],
                     f"corpus missing: {corpus}")
    try:
        from src.encoding.gliner_extractor import GLiNERExtractor  # noqa: E402
    except Exception as e:
        return _cell("dependency_missing", None, TARGETS["encoding"],
                     f"GLiNERExtractor import failed: {e}")

    try:
        extractor = GLiNERExtractor(device=gliner_device)
    except Exception as e:
        return _cell("dependency_missing", None, TARGETS["encoding"],
                     f"GLiNER model load failed (device={gliner_device}): {e}")

    convs = [json.loads(l) for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    er, tr, nr = [], [], []
    for conv in convs:
        full = " ".join(f"User: {u} Assistant: {a}" for u, a in conv["turns"])
        try:
            result = extractor.extract(full)
        except Exception as e:
            return _cell("error", None, TARGETS["encoding"],
                         f"extraction failed on {conv['id']}: {e}")
        er.append(_recall(set(conv.get("expected_entities", [])), set(result["entities"])))
        tr.append(_topic_recall(conv.get("expected_topics", []), result["topics"]))
        nr.append(_recall(set(conv.get("expected_tones", [])), set(result["tones"])))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return _cell("measured",
                 {"entity_recall": round(mean(er), 3),
                  "topic_recall": round(mean(tr), 3),
                  "tone_recall": round(mean(nr), 3),
                  "n": len(convs),
                  "device": gliner_device},
                 TARGETS["encoding"],
                 f"recall on 20 hand-crafted convs (Phase 1a DoD proxy; not F1 "
                 f"on held-out bench); device={gliner_device}")


# ===========================================================================
# Stage 2 -- Retrieval recall on labeled corpus (reuse test_end_to_end gold)
# ===========================================================================
def stage_retrieval():
    """The 2 hand-coded gold retrieval queries against the 20-conv corpus.

    Reuses tests.test_end_to_end helpers + gold (_FRUSTRATED, the Alice-decide
    set). Flag: n=2, illustrative not benchmark. Also reports retrieval latency.
    """
    try:
        from tests.test_end_to_end import (  # noqa: E402
            _load_corpus_episodes, _store_with_corpus, _RulePlanner, _FRUSTRATED,
        )
        from src.retrieval.retriever import HippocampalRetriever  # noqa: E402
    except Exception as e:
        return {"recall": _cell("dependency_missing", None,
                                 TARGETS["retrieval_recall"],
                                 f"test_end_to_end import failed: {e}"),
                "precision": _cell("gold_missing", None,
                                   TARGETS["retrieval_precision"],
                                   "no precision gold on the 20-conv corpus"),
                "latency": _cell("gold_missing", None,
                                 TARGETS["retrieval_latency"],
                                 "latency not measured at scale here")}

    tmp = tempfile.mkdtemp(prefix="pondr_retrieval_")
    try:
        store = _store_with_corpus(Path(tmp))
        retr = HippocampalRetriever(store, planner=_RulePlanner())

        # Query 1: frustrated -> ids subset of _FRUSTRATED
        t0 = time.perf_counter()
        r1 = retr.retrieve("What was I frustrated about?")
        lat1 = (time.perf_counter() - t0) * 1000.0
        ids1 = {x["episode_id"] for x in r1}
        q1_pass = bool(ids1) and ids1 <= _FRUSTRATED

        # Query 2: Alice + decide -> {conv_012, conv_017}
        t0 = time.perf_counter()
        r2 = retr.retrieve("What did Alice and I decide?")
        lat2 = (time.perf_counter() - t0) * 1000.0
        ids2 = {x["episode_id"] for x in r2}
        q2_pass = ids2 == {"conv_012", "conv_017"}

        recall_value = {"n_queries": 2, "pass": int(q1_pass) + int(q2_pass),
                        "q1_frustrated_pass": q1_pass, "q1_ids": sorted(ids1),
                        "q2_alice_decide_pass": q2_pass, "q2_ids": sorted(ids2)}
        lat_value = {"p50_ms": round(statistics.median([lat1, lat2]), 2),
                     "per_query_ms": [round(lat1, 2), round(lat2, 2)]}
        store.close()
    except Exception as e:
        return {"recall": _cell("error", None, TARGETS["retrieval_recall"],
                                f"retrieval run failed: {e}"),
                "precision": _cell("gold_missing", None,
                                   TARGETS["retrieval_precision"],
                                   "no precision gold on the 20-conv corpus"),
                "latency": _cell("gold_missing", None,
                                 TARGETS["retrieval_latency"],
                                 "latency not measured at scale here")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "recall": _cell("measured", recall_value, TARGETS["retrieval_recall"],
                        "n=2 hand-coded gold queries on 20-conv corpus -- illustrative, not a benchmark"),
        "precision": _cell("gold_missing", None, TARGETS["retrieval_precision"],
                          "no precision gold on the 20-conv corpus"),
        "latency": _cell("measured", lat_value, TARGETS["retrieval_latency"],
                         "n=2 in-memory corpus queries; not the graph-traversal+HBTrie p50 at scale"),
    }


# ===========================================================================
# Stage 3 -- Routing (training-val proxy; routing-vs-Oracle gold missing)
# ===========================================================================
def stage_routing(smoke_result):
    """RetrievalGate training val accuracy (proxy) + observed pathways from
    the Stage 0 smoke queries. Routing-vs-Oracle gold does not exist."""
    val = None
    log_path = ROOT / "data" / "pod_runs" / "phase2b" / "train_log.json"
    if log_path.is_file():
        try:
            val = json.loads(log_path.read_text(encoding="utf-8")).get("best_val")
        except Exception:
            val = None

    pathways = [q.get("pathway") for q in smoke_result.get("queries", [])
                if "pathway" in q]
    notes = ("training val-accuracy proxy (routing-vs-Oracle gold missing -- "
             "Phase 1d routing pairs not scaled); observed pathways from Stage 0 smoke")
    value = {"val_accuracy": round(val, 3) if isinstance(val, (int, float)) else None,
             "observed_pathways": pathways}
    status = "measured" if val is not None else "gold_missing"
    return _cell(status, value, TARGETS["routing"], notes)


# ===========================================================================
# Stage 4 -- Consolidation dream-pass (reuse run_consolidation.py)
# ===========================================================================
def stage_consolidation(db, gnn_ckpt, limit=0):
    """Run run_consolidation.py in DRY-RUN (--no-bonsai, decider off since
    Bonsai is killed). Captures the JSON report. Tries --checkpoint first;
    falls back to --force-untrained if the GNN ckpt won't load (itself a
    finding: 'consolidation machinery runs, quality unmeasured').

    By default builds a bounded 20-conv corpus store and runs the dream-pass
    on THAT -- the GNN subgraph extraction over the 4995-episode dialogsum DB
    is the radius-3 giant (see [[hippo-phase3a-head-fixes]]) and takes >10min,
    which is too slow for an assessment smoke. The smoke stage already proves
    the real big DB loads + retrieves; this stage proves the consolidation
    MACHINERY runs end-to-end. Pass a real ``--consolidation-db`` to score a
    production DB (slow; quality still unmeasured without the decider).
    """
    script = ROOT / "scripts" / "run_consolidation.py"
    if not script.is_file():
        return _cell("dependency_missing", None, TARGETS["consolidation"],
                     "run_consolidation.py missing")

    tmpdirs = []
    try:
        report_path = Path(tempfile.mkdtemp(prefix="pondr_consol_rep_")) / "report.json"
        tmpdirs.append(report_path.parent)
        limit_args = ["--limit", str(limit)] if limit and limit > 0 else []

        # Resolve the DB to score: explicit --consolidation-db wins; otherwise
        # build a 20-conv smoke store so the run is bounded.
        if db and Path(db).is_dir():
            score_db = str(db)
            db_note = f"scored DB: {db} ({'limit=' + str(limit) if limit else 'full'})"
        else:
            try:
                from tests.test_end_to_end import _store_with_corpus  # noqa: E402
            except Exception as e:
                return _cell("dependency_missing", None, TARGETS["consolidation"],
                             f"could not build 20-conv smoke store (test_end_to_end "
                             f"import failed: {e}); pass --consolidation-db")
            smoke_dir = Path(tempfile.mkdtemp(prefix="pondr_consol_smoke_"))
            tmpdirs.append(smoke_dir)
            store = _store_with_corpus(smoke_dir)
            store.close()
            score_db = str(smoke_dir / "db")
            db_note = ("scored DB: 20-conv smoke corpus (subgraphs_scored in "
                       "the report confirms the size); full-corpus timing not measured")

        def _run(extra):
            cmd = ([sys.executable, str(script), "--db", score_db, "--no-bonsai",
                    "--report", str(report_path)] + limit_args + extra)
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=600, cwd=str(ROOT))

        report = None
        notes_extra = ""
        # Try with the trained checkpoint first; fall back to --force-untrained
        # (untrained model -> random salience prunes ~every edge, so the report
        # is machinery-shape only -- a finding, not a quality number).
        if gnn_ckpt and Path(gnn_ckpt).is_file():
            proc = _run(["--checkpoint", str(gnn_ckpt)])
            if proc.returncode != 0:
                notes_extra = (f"--checkpoint {gnn_ckpt} failed (rc={proc.returncode}); "
                               f"fell back to --force-untrained. stderr tail: "
                               f"{(proc.stderr or '')[-300:]}")
                report_path.unlink(missing_ok=True)
                proc = _run(["--force-untrained"])
        else:
            notes_extra = "no --gnn-ckpt supplied; ran --force-untrained"
            proc = _run(["--force-untrained"])

        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                report = None

        if report is None:
            # Either run_consolidation crashed (rc!=0) OR it exited 0 without
            # writing the report path -- both mean no measurement was taken.
            return _cell("error", None, TARGETS["consolidation"],
                         f"run_consolidation produced no report rc={proc.returncode}; "
                         f"stderr tail: {(proc.stderr or '')[-400:]}")

        # Pull headline counts out of the report. run_consolidation's report
        # shape (verified): trained(bool), subgraphs_scored(int), and the rest
        # are LISTS whose length is the count (edges_proposed/accepted/unverified,
        # anomalies, ontology_proposed, pruned, contradictions_resolved).
        def _n(key):
            v = report.get(key)
            return len(v) if isinstance(v, (list, dict)) else v

        value = {
            "ran": True,
            "trained": report.get("trained"),
            "subgraphs_scored": report.get("subgraphs_scored"),
            "edges_proposed": _n("edges_proposed"),
            "edges_accepted": _n("edges_accepted"),
            "edges_unverified": _n("edges_unverified"),
            "anomalies": _n("anomalies"),
            "ontology_proposed": _n("ontology_proposed"),
            "pruned": _n("pruned"),
            "contradictions_resolved": _n("contradictions_resolved"),
            "verifier_validation_rate": report.get("verifier_validation_rate"),
            "dry_run": report.get("dry_run"),
            "decider": "off (--no-bonsai; Bonsai killed)",
            "subgraph_cap": limit if limit and limit > 0 else "none (full smoke corpus)",
            "db": db_note,
        }
        notes = ("dry-run dream-pass; decider off (Bonsai killed) so "
                 "edges_accepted=0 and verifier_validation_rate=null -- the "
                 "Phase 5 'GNN-predicted edges validated by Bonsai' cell is "
                 "gold-missing. " + notes_extra)
        return _cell("measured", value, TARGETS["consolidation"], notes)
    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# Stage 5 -- Contradiction + citation (reuse deterministic gold)
# ===========================================================================
def stage_contradiction():
    """5a: deterministic contradiction recall + citation resolve-rate on the 5
    vendored erag pairs (reuse tests.test_enterpriserag_eval scoring). 5b:
    deterministic-guard fire rate on the 200 adjudication pairs (reuse the
    state_values_from_spec reconstruction + _deterministic_non_conflict)."""
    out = {"det_contradiction_recall": None, "citation_resolve": None,
           "guard_soundness": None}

    # 5a -- deterministic contradiction + citation (offline, no Bonsai).
    tmp_path = None
    try:
        from tests.test_enterpriserag_eval import (  # noqa: E402
            _encode_pair, _entity_state_values, FIXTURE,
            CATCHABLE_RECALL_THRESHOLD, CITATION_RESOLVE_THRESHOLD,
        )
        pairs = json.loads(FIXTURE.read_text(encoding="utf-8"))["pairs"]
        catchable = [p for p in pairs if p["catchable"]]
        tmp_path = Path(tempfile.mkdtemp(prefix="pondr_erag_"))
        c_hits = 0
        for pair in catchable:
            store = _encode_pair(tmp_path, pair)
            try:
                vals = _entity_state_values(store, pair["conflicting_entity"])
                if len(vals) >= 2:
                    c_hits += 1
            finally:
                store.close()
        recall = c_hits / len(catchable) if catchable else 0.0

        cit_hits = 0
        for pair in pairs:
            store = _encode_pair(tmp_path, pair)
            try:
                if store.find_document_by_title_or_url(
                        pair["new_doc"]["title"]) == pair["expected_doc_id"]:
                    cit_hits += 1
            finally:
                store.close()
        out["det_contradiction_recall"] = _cell(
            "measured",
            {"recall": round(recall, 3), "n_catchable": len(catchable),
             "threshold": CATCHABLE_RECALL_THRESHOLD},
            ">=0.75 (deterministic-normalizer ceiling)",
            "deterministic-path recall on catchable pairs; paraphrased-only pairs are honest misses (Bonsai's job)")
        out["citation_resolve"] = _cell(
            "measured",
            {"rate": round(cit_hits / len(pairs), 3), "n": len(pairs),
             "threshold": CITATION_RESOLVE_THRESHOLD},
            ">=0.80",
            "find_document_by_title_or_url on the 5 vendored erag pairs")
    except Exception as e:
        out["det_contradiction_recall"] = _cell(
            "error", None, ">=0.75", f"erag scoring failed: {e}")
        out["citation_resolve"] = _cell(
            "error", None, ">=0.80", f"erag scoring failed: {e}")
    finally:
        if tmp_path is not None:
            shutil.rmtree(tmp_path, ignore_errors=True)

    # 5b -- deterministic-guard fire rate on the 200 adjudication pairs.
    try:
        from src.gnn.bonsai_decider import _deterministic_non_conflict  # noqa: E402

        # state_values_from_spec -- verbatim from scripts/_scratch/guard_coverage_200.py
        # (that probe is not a committed importable module; replicate the 17-line
        # reconstruction rather than depend on _scratch/).
        def state_values_from_spec(spec):
            ct = spec.get("conflict_type")
            old_path = spec.get("old_path") or ""
            new_path = spec.get("new_path") or ""
            if ct == "different_entity":
                v = spec.get("value", "")
                return [{"value": v, "asserted_by": old_path, "asserted_at": "2026-07-01",
                         "source_path": old_path},
                        {"value": v, "asserted_by": new_path, "asserted_at": "2026-07-05",
                         "source_path": new_path}]
            ov = spec.get("old_value", "")
            nv = spec.get("new_value", "")
            return [{"value": ov, "asserted_by": old_path, "asserted_at": "2026-07-01",
                     "source_path": old_path},
                    {"value": nv, "asserted_by": new_path, "asserted_at": "2026-07-05",
                     "source_path": new_path}]

        pairs_path = ROOT / "data" / "training" / "bonsai" / "contradiction_pairs.jsonl"
        rows = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        adj = [r for r in rows if r.get("task") == "adjudication"]
        by_type = {}
        for r in adj:
            ct = r["spec"]["conflict_type"]
            svs = state_values_from_spec(r["spec"])
            guard = _deterministic_non_conflict(svs)
            guard_fired = guard is not None
            # Without Bonsai we can only check the GUARD path. A guard fire is
            # "correct-by-construction" for non-real types; for `real` the guard
            # never fires (falls to the dead LLM). So guard-correctness is only
            # measurable on the non-real subset the guard actually catches.
            t = by_type.setdefault(ct, {"n": 0, "guard_fired": 0})
            t["n"] += 1
            t["guard_fired"] += int(guard_fired)
        nonreal = [c for c in by_type if c != "real"]
        nonreal_n = sum(by_type[c]["n"] for c in nonreal)
        nonreal_guards = sum(by_type[c]["guard_fired"] for c in nonreal)
        out["guard_soundness"] = _cell(
            "measured",
            {"n_adjudication": len(adj), "by_type": by_type,
             "nonreal_guard_fires": nonreal_guards,
             "nonreal_n": nonreal_n,
             "guard_fire_rate": round(nonreal_guards / nonreal_n, 3) if nonreal_n else 0.0},
            "non-real false-fix MUST be 0",
            "deterministic-guards-only slice (Bonsai killed). The 0-false-fix "
            "decider soundness needs Bonsai up -- run scripts/_scratch/guard_coverage_200.py to measure.")
    except Exception as e:
        out["guard_soundness"] = _cell(
            "error", None, "non-real false-fix MUST be 0",
            f"guard slice failed: {e}")
    return out


# ===========================================================================
# Stage 6 -- Scorecard assembly
# ===========================================================================
def _nr(target_key, why):
    """A 'not_run' cell -- the stage was not selected, so there is no value.
    Distinct from 'error' (a selected stage that failed) so a subset run does
    not read as a failure and does not trip the non-zero exit."""
    return _cell("not_run", None, TARGETS[target_key], why)


def assemble_scorecard(stages, out_path):
    """Emit scorecard.json (raw stages) + scorecard.md (the table)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map each capability to a cell. Selected stages that fail -> 'error';
    # unselected stages -> 'not_run' (a choice, not a failure); stages with no
    # gold -> 'gold_missing'.
    smoke = stages.get("smoke", {})
    retr = stages.get("retrieval", {})
    contra = stages.get("contradiction", {})
    rows = [
        ("Runtime runs end-to-end (trained models)", smoke.get("headline",
            _nr("runtime_runs", "smoke stage not selected"))),
        ("Encoding accuracy", stages.get("encoding",
            _nr("encoding", "encoding stage not selected"))),
        ("Retrieval recall", retr.get("recall",
            _nr("retrieval_recall", "retrieval stage not selected"))),
        ("Retrieval precision", retr.get("precision",
            _nr("retrieval_precision", "retrieval stage not selected"))),
        ("Retrieval latency", retr.get("latency",
            _nr("retrieval_latency", "retrieval stage not selected"))),
        ("Routing accuracy", stages.get("routing",
            _nr("routing", "routing stage not selected"))),
        ("Consolidation quality", stages.get("consolidation",
            _nr("consolidation", "consolidation stage not selected"))),
        ("Contradiction detection (deterministic recall)",
            contra.get("det_contradiction_recall",
                _nr("routing", "contradiction stage not selected"))),
        ("Citation resolution", contra.get("citation_resolve",
            _nr("routing", "contradiction stage not selected"))),
        ("Adjudicator guard soundness (non-real false-fix)",
            contra.get("guard_soundness",
                _nr("routing", "contradiction stage not selected"))),
        ("Forgetting accuracy", _cell("gold_missing", None, TARGETS["forgetting"],
            "no harness; Phase 3b unit tests only")),
        ("Context efficiency", _cell("gold_missing", None, TARGETS["context_efficiency"],
            "no harness; graph-context vs full-history not measured")),
        ("Uncertainty calibration", _cell("gold_missing", None, TARGETS["uncertainty"],
            "Uncertainty Detector / Self-Model not trained (Phase 4 not started)")),
        ("Delegation efficiency", _cell("gold_missing", None, TARGETS["delegation"],
            "no harness; model-sizing/delegation not wired at runtime")),
    ]

    # If a selected stage RAISED (caught by the _safe backstop), its dict carries
    # _stage_error. Surface that on EVERY row the stage owns so the failure is
    # not masked as not_run by the .get() defaults above (de-wonk: no silent
    # masking of a real failure).
    stage_rows = {
        "smoke": ["Runtime runs end-to-end (trained models)"],
        "encoding": ["Encoding accuracy"],
        "retrieval": ["Retrieval recall", "Retrieval precision", "Retrieval latency"],
        "routing": ["Routing accuracy"],
        "consolidation": ["Consolidation quality"],
        "contradiction": ["Contradiction detection (deterministic recall)",
                         "Citation resolution",
                         "Adjudicator guard soundness (non-real false-fix)"],
    }
    rows_dict = {label: cell for label, cell in rows}
    for stage_name, labels in stage_rows.items():
        stage = stages.get(stage_name)
        if isinstance(stage, dict) and stage.get("_stage_error"):
            err = stage["_stage_error"]
            for label in labels:
                if label in rows_dict:
                    old = rows_dict[label]
                    rows_dict[label] = _cell("error", None, old["target"],
                                             f"{stage_name} stage: {err}")
    rows = [(label, rows_dict[label]) for label, _ in rows]

    # JSON (raw stages + the row map).
    json_payload = {"stages": stages, "scorecard": {label: cell for label, cell in rows}}
    out_path.with_suffix(".json").write_text(
        json.dumps(json_payload, indent=2, default=str), encoding="utf-8")

    # Markdown table -- status in its own column so gold-missing/not_run is
    # visually distinct from measured (de-wonk: never read as "covered" when
    # it isn't).
    lines = ["# Total-Package Capability Scorecard", "",
             "| Metric | Status | Value | Target | Notes |",
             "|---|---|---|---|---|"]
    for label, cell in rows:
        status = cell["status"]
        val = cell["value"]
        if isinstance(val, dict):
            val = json.dumps(val, default=str)
        lines.append(f"| {label} | {status} | {val if val is not None else ''} "
                     f"| {cell['target']} | {cell['notes']} |")
    md = "\n".join(lines) + "\n"
    out_path.with_suffix(".md").write_text(md, encoding="utf-8")
    return json_payload


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Total-package capability assessment (Option 1: smoke + reuse scorecard).",
        epilog="Reuses existing gold + harnesses; builds no new gold. Many cells read gold-missing.")
    p.add_argument("--db", default=DEFAULT_DB, help="WaveDB dir for the smoke + consolidation stages")
    p.add_argument("--llm-endpoint", default=DEFAULT_LLM_ENDPOINT, help="ModeA LLM endpoint (default Ollama :11434)")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="GENERATION_MODEL for ModeA")
    p.add_argument("--gnn-ckpt", default=DEFAULT_GNN_CKPT, help="GNN checkpoint for the consolidation stage")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Phase 2a backbone checkpoint")
    p.add_argument("--gate", default=DEFAULT_GATE, help="Phase 2b gate checkpoint")
    p.add_argument("--gliner-device", default="cpu",
                   help="Device for the encoding stage's GLiNER (cpu|cuda|auto). "
                        "Default cpu; pass cuda/auto to separate CPU-vs-GPU "
                        "confidence divergence from a true extraction regression.")
    p.add_argument("--consolidation-db", default=None,
                   help="WaveDB dir for the consolidation dream-pass. If unset, "
                        "builds a bounded 20-conv smoke corpus (the 4995-episode "
                        "dialogsum DB's GNN subgraph extraction takes >10min).")
    p.add_argument("--consolidation-limit", type=int, default=0,
                   help="Cap on subgraphs scored in the consolidation dream-pass "
                        "(0 = no cap; only meaningful with --consolidation-db)")
    p.add_argument("--stage", default="all", help=f"comma-separated subset of {STAGES} or 'all'")
    p.add_argument("--out", default=DEFAULT_OUT, help="scorecard output stem (.json + .md)")
    args = p.parse_args()

    wanted = STAGES if args.stage == "all" else tuple(
        s.strip() for s in args.stage.split(",") if s.strip() in STAGES)

    # Wrap every stage so a raise degrades to an error cell -- the run never
    # crashes (de-wonk: a stage that throws must still produce a documented cell,
    # and the error must surface on EVERY row that stage owns, not be masked as
    # not_run by assemble_scorecard's .get() defaults).
    def _safe(name, fn):
        try:
            return fn()
        except Exception as e:
            import traceback
            tb = traceback.format_exc().splitlines()
            tail = tb[-1] if tb else str(e)
            print(f"[{name}] stage raised -> error cell: {tail}", file=sys.stderr)
            return {"_stage_error": f"stage raised: {tail}"}

    stages = {}
    if "smoke" in wanted:
        print("[smoke] building PonderOrchestrator on trained checkpoints...")
        stages["smoke"] = _safe("smoke", lambda: stage_smoke(
            args.db, args.llm_endpoint, args.llm_model, args.backbone, args.gate))
    if "encoding" in wanted:
        print("[encoding] running 20-conv extraction recall...")
        stages["encoding"] = _safe("encoding", lambda: stage_encoding(args.gliner_device))
    if "retrieval" in wanted:
        print("[retrieval] running 2 gold retrieval queries...")
        stages["retrieval"] = _safe("retrieval", stage_retrieval)
    if "routing" in wanted:
        print("[routing] reading gate val accuracy + observed pathways...")
        stages["routing"] = _safe("routing", lambda: stage_routing(stages.get("smoke", {})))
    if "consolidation" in wanted:
        print("[consolidation] running dream-pass dry-run...")
        stages["consolidation"] = _safe("consolidation", lambda: stage_consolidation(
            args.consolidation_db, args.gnn_ckpt, args.consolidation_limit))
    if "contradiction" in wanted:
        print("[contradiction] running deterministic erag + guard slice...")
        stages["contradiction"] = _safe("contradiction", stage_contradiction)

    payload = assemble_scorecard(stages, args.out)
    print(f"\nScorecard written to {args.out}.json + {args.out}.md")
    # Print a short summary table to stdout.
    print("\n=== Capability scorecard ===")
    for label, cell in payload["scorecard"].items():
        v = cell["value"]
        vs = json.dumps(v, default=str) if isinstance(v, dict) else (v if v is not None else "")
        print(f"  [{cell['status']:16s}] {label}: {vs}")

    # Exit non-zero if any stage ERRORED (so the run flags a real failure, not
    # a silent gold-missing). gold_missing/endpoint_down/dependency_missing are
    # clean exits (documented cells, not failures).
    errored = any(c["status"] == "error" for c in payload["scorecard"].values())
    return 1 if errored else 0


if __name__ == "__main__":
    sys.exit(main())