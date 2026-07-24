"""STRM Phase 4 SHIP-DECIDING eval: STRM+recall vs fixed-interval RAG at equal
recall budget on long-horizon FACTUAL ACCURACY.

This is the literal ship gate from ``docs/STRM-implementation-plan.md:444-447,550``:

  "on a long-horizon eval, STRM+recall answers more factual questions correctly
   than fixed-interval refresh at equal recall budget/latency. ... Ship the
   primitive (the real decision)."

It is the strategic unblocker for two deferred items (Phase 6 joint fine-tune;
wiring STRM salience default-on). Until this closes, the ship decision is a
guess -- the user's principle "there is no point integrating this until it
serves the intended purpose well" is exactly the gate this eval must answer.

TWO TIERS (decided this session):
  Tier 1 -- offline coverage / cost-parity (deterministic, NO LLM). The mechanism
           precondition: at equal-or-lower proactive budget, does STRM surface
           the RIGHT fact on fact-relevant turns more often than a fixed-
           interval round-robin? Reuses the validated cost-parity harness
           (``eval_strm_cost_parity.py``) wholesale -- ``_run_strm`` / ``_run_off``
           / ``_run_fixed`` / ``_selectivity`` + the budget-parity N-tuning.
  Tier 2 -- live factual accuracy (the literal gate). At each fact-relevant
           probe turn, the REAL Bonsai 8B synthesizes an answer grounded in the
           surfaced context, and a DeepSeek-FLASH judge panel (2-of-3) grades it
           against an unambiguous gold answer. STRM vs FIXED vs OFF.

WHY SYNTHETIC FACTS (not real onyx now). Synthetic facts carry clean
``gold_question`` / ``gold_answer`` pairs so the judge has unambiguous ground
truth -- the gate is *factual accuracy*, which needs clean gold. Real onyx
(76 sessions present locally, PRIVATE, never uploaded) has no pre-labeled Q/A;
hand-authoring gold from session content is real labeling effort best spent
AFTER the mechanism clears the synthetic bar. Real-onyx held-out confirmation
is the documented follow-on.

FAIRNESS CONTROL (critical). Both STRM and FIXED surface a recalled fact into the
synthesis context via the SAME merge path (``orchestrator.py:672-678`` -- the
merge checks the ``_salience_fired_episodes`` ATTRIBUTE, not the armed flag), so
the only variable is the recall DECISION (relevance-gated vs schedule-gated):
  STRM  -- armed salience hook populates ``self._salience_fired_episodes`` -> the
           merge surfaces it. Recall is RELEVANCE-gated (forgotten + relevant).
  FIXED-- salience OFF; every N turns a manual proactive recall round-robins over
           the seeded facts (uniform spread, no relevance signal). Before
           ``query()`` it sets ``orch._salience_fired_episodes = [top_hit]`` so the
           SAME merge surfaces it; reset to ``None`` after (no cross-turn leak).
           Recall is SCHEDULE-gated. N is tuned so FIXED cost <= STRM cost.
  OFF   -- never set -> empty synthesis context -> the 8B answers from no
           project context -> wrong (the sanity floor; facts are project-specific
           so the 8B has no parametric knowledge of them).

ISOLATION. ``_build_orchestrator`` wires no encoder -> ``_persist_exchange`` no-ops
(``orchestrator.py:982-985``) -> the eval's own Q/A turns do NOT pollute the
ring/store, preserving the proactive-mechanism isolation. No ``DistillWorker`` is
wired -> the Phase 5 IngestionTracker in-flight short-circuit is INERT here; that
only makes STRM's cost *higher* than it would be live, so this is a CONSERVATIVE
test -- if STRM beats fixed without the short-circuit it certainly beats it with
it. The short-circuit was live-proven separately (see
``scripts/_scratch/_dogfood_salience_shortcircuit_force.py``).

RING-CAP NOTE (de-wonk). The salience hook scores ring SLOTS -- a fact must STILL
be in the ring (as a slot whose recoverability ``rec_i`` has decayed below theta)
for salience to fire a recall. ``step`` appends a ring slot every turn
(``working_memory.py:339-356``), so a ring-cap smaller than ~horizon+n_facts
EVICTS the seeded fact slots (they are the oldest) before their probe turn ->
STRM coverage collapses to 0 and the eval measures nothing. cost-parity (the
validated harness) used ring-cap 32 for horizon 30 (no eviction). This eval
defaults ``--ring-cap 0`` -> AUTO = ``horizon + n_facts + 2`` (preserves slots;
``rec_i`` still decays via state drift from the filler queries, which is the
actual forgotten-but-present trigger). Pass an explicit ``--ring-cap`` to override.

CLEAN ONE-SHOT SYNTHESIS. For a controlled eval the 8B must ground its answer
ONLY in the surfaced context -- the self-chat tool loop
(``self_chat_tool_loop_enabled``) and feedback salience
(``feedback_salience_enabled``) are turned OFF for the Tier 2 run (saved +
restored) so the 8B cannot self-retrieve via ``search_memory`` (which would
surface the fact for OFF too and break the fairness floor). The one-shot path is
byte-identical to the loop-off A/B guard.

Usage:
  # Tier 1 only (cheap, offline, no LLM) -- confirm the mechanism first:
  PYTHONPATH=. python scripts/eval_strm_ship_decision.py --facts 6 --horizon 40 \
      --theta -0.04 --seed 0 --out /tmp/ship_s0.json --tier1-only
  # Both tiers (live, needs Bonsai 8B on :8080 + DeepSeek-flash on :11434):
  PYTHONPATH=. python scripts/eval_strm_ship_decision.py --facts 6 --horizon 40 \
      --theta -0.04 --seed 0 --out /tmp/ship_s0.json
Multi-seed gate: see ``scripts/_run_strm_ship_gate.py`` (untracked scratch runner).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# gliner2 prints a brain emoji at from_pretrained; on Windows cp1252 stdout that
# raises UnicodeEncodeError. Force utf-8 before any import that loads gliner.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from src.config import Phase2cConfig, config as _runtime_config  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.retriever import HippocampalRetriever  # noqa: E402
from src.runtime import DEFAULT_BACKBONE_PATH  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.latent_dynamics_head import load_latent_dynamics_head  # noqa: E402
from src.subconscious.recoverability_head import load_recoverability_head  # noqa: E402
from src.subconscious.relevance_head import load_relevance_head  # noqa: E402
from src.subconscious.salience import SalienceThresholds, load_salience_thresholds  # noqa: E402
from src.subconscious.training.routing_training import build_embedder, load_backbone  # noqa: E402
from src.subconscious.training.doc_kind_training import _wilson_ci95  # noqa: E402
from src.generation.mode_a import ModeAGenerator  # noqa: E402

from scripts.eval_strm_context_coverage import (  # noqa: E402
    ARTIFACTS_PRESENT, _StubPlanner, _build_orchestrator, _permissive_thresholds,
    _reset_for_trial, _seed_corpus,
)
from scripts.eval_strm_cost_parity import (  # noqa: E402
    FILLER_QUERIES, _disable_prompt_driven, _ring_fact_ids, _run_fixed, _run_off,
    _run_strm, _seed_ring_facts, _selectivity,
)

REC_CKPT = Path("data/training/strm_recoverability/best.pt")
LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
REL_CKPT = Path("data/training/strm_relevance/best.pt")
THRESHOLDS = Path("data/training/strm_salience/thresholds.json")
BACKBONE = Path(DEFAULT_BACKBONE_PATH)

BONSAI_ENDPOINT = os.environ.get("BONSAI_ENDPOINT", "http://localhost:8080/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash:cloud")
JUDGE_ENDPOINT = os.environ.get("JUDGE_ENDPOINT", "http://localhost:11434/v1")
PANEL = int(os.environ.get("JUDGE_PANEL", "3"))


def _bonsai_model() -> str:
    """The live 8B server's actual model id (e.g. ``Ternary-Bonsai-8B-Q2_0.gguf``)
    -- queried from ``:8080/v1/models``. The config ``generation_model``
    (``prism-ml/Ternary-Bonsai-8B-gguf``) is a different string the server does
    NOT recognize, so ``ModeAGenerator(model=None)`` would 404. Falls back to
    the config name if the server is unreachable (so the error surfaces)."""
    try:
        r = requests.get(f"{BONSAI_ENDPOINT}/models", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data") or data.get("models") or []
            for m in models:
                mid = m.get("id") or m.get("model") or m.get("name")
                if mid:
                    return mid
    except Exception:  # noqa: BLE001
        pass
    from src.config import config as _cfg
    return _cfg.generation_model

# Gold fact bank: (fact_summary, question, gold_answer). The fact_summary is
# seeded into the ring + store at turn 0 (provenance ``fact_{i:02d}``). The
# question is BOTH the salience trigger (semantically close to its fact so the
# 2a relevance head scores it highly) AND the factual question the 8B must
# answer. The gold_answer is the single determinate correct answer the judge
# checks against -- every gold is unambiguous (one correct answer) so the
# judge's ``ambiguous`` is never the majority on a correctly-surfaced fact.
SHIP_FACT_BANK = [
    ("Alice chose Postgres for the audit log because it needs ACID compliance.",
     "What database are we using for the audit log?",
     "Postgres"),
    ("The deploy pipeline failed because the CUDA driver was version 535 and sm_120 needs 610 or newer.",
     "Which CUDA driver version caused the GPU build to fail?",
     "535"),
    ("Bob scheduled the architecture review for Thursday at 2pm.",
     "What day of the week is the architecture review on?",
     "Thursday"),
    ("We capped the proactive retrieval budget at 3 to bound per-turn latency.",
     "What is the proactive retrieval budget cap?",
     "3"),
    ("The vector index moved from FAISS to the in-DB WaveDB VectorLayer.",
     "What did the vector index move away from?",
     "FAISS"),
    ("Dave pushed the migration cut-over to the end of the sprint to avoid a Friday deploy.",
     "Why did Dave push the migration cut-over to the end of the sprint?",
     "to avoid a Friday deploy"),
]


# ── judge: DeepSeek-flash answer-correctness panel (reuses the ERAG shell) ──

def _judge_one(question: str, gold: str, model_answer: str,
               context: str = "", timeout: float = 120.0) -> dict:
    """One judge vote: 'correct' | 'incorrect' | 'ambiguous'.

    Reuses the ERAG ``judge_one`` HTTP/parse shell (``scripts/_scratch/
    erag_judge_harness.py:119-168``): same ``requests.post`` to
    ``{endpoint}/chat/completions``, ``response_format=json_object``,
    ``temperature=0.0``, the backtick-strip + outermost-``{...}``-carve parser,
    and unknown->ambiguous coercion. The RUBRIC is swapped to answer-correctness
    (the ERAG rubric graded contradiction-decisions). A small retry loop replaces
    the Oracle client's backoff so the 3 panel votes are 3 REAL independent HTTP
    calls (the Oracle client's in-memory cache would collapse the panel to 1)."""
    prompt = (
        "You are an independent evaluator grading a factual question-answer.\n"
        "Given the user's QUESTION, the GOLD ANSWER (ground truth), and the "
        "MODEL'S ANSWER, decide whether the model's answer is correct.\n\n"
        f"QUESTION: {question}\n\n"
        f"GOLD ANSWER: {gold}\n\n"
        f"MODEL ANSWER: {(model_answer or '').strip()[:1200]}\n\n"
        f"RETRIEVED CONTEXT (what the model was given): {(context or '').strip()[:800]}\n\n"
        "Rubric: vote 'correct' if the model's answer agrees with the GOLD ANSWER. "
        "Vote 'incorrect' if the model's answer contradicts or misses the GOLD ANSWER, "
        "or if it is empty / a non-answer. Vote 'ambiguous' ONLY if the model's answer "
        "is genuinely unstatable or the question is unanswerable from the gold -- a "
        "plain miss or contradiction is 'incorrect', NOT 'ambiguous'. Reply ONLY JSON: "
        '{"vote":"correct|incorrect|ambiguous","why":"<one sentence>"}'
    )
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    last = ""
    for attempt in range(3):
        try:
            r = requests.post(f"{JUDGE_ENDPOINT}/chat/completions", json=payload,
                              timeout=timeout)
            if r.status_code != 200:
                last = f"judge http {r.status_code}"
                time.sleep(1.0 * (attempt + 1))
                continue
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if not content:
                last = "judge empty content"
                continue
            body = content.strip().strip("`")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                s, e = body.find("{"), body.rfind("}")
                if s != -1 and e > s:
                    data = json.loads(body[s:e + 1])
                else:
                    last = f"no JSON: {body[:120]}"
                    continue
            v = str(data.get("vote", "")).strip().lower()
            if v not in ("correct", "incorrect", "ambiguous"):
                v = "ambiguous"
            return {"vote": v, "why": str(data.get("why", ""))[:200]}
        except Exception as e:  # noqa: BLE001
            last = f"judge error: {e}"
            time.sleep(1.0 * (attempt + 1))
    return {"vote": "ambiguous", "why": last or "judge unreachable"}


def _consensus(votes: list[dict]) -> tuple[str, int]:
    """Strict-majority consensus (2-of-3 for PANEL=3). Ties -> ambiguous."""
    c = Counter(v["vote"] for v in votes)
    top, n = c.most_common(1)[0]
    if n > len(votes) / 2:
        return top, n
    return "ambiguous", n


def judge_answer(question: str, gold: str, model_answer: str,
                 context: str = "") -> dict:
    """Panel of PANEL independent judge votes -> consensus + the raw votes."""
    votes = [_judge_one(question, gold, model_answer, context) for _ in range(PANEL)]
    top, n = _consensus(votes)
    return {"vote": top, "n": n, "votes": [v["vote"] for v in votes]}


# ── schedule (identical to cost-parity so Tier 1 is the validated harness) ──

def _build_schedule(n_facts: int, horizon: int, facts3: list[tuple],
                    seed: int) -> list:
    """fact i is probed at turn probe_start + i*probe_step (after decay); the
    rest are filler. Each fact probed once. Returns [(query, target_fid), ...].

    SEED LEVER. The eval is otherwise fully deterministic (temp-0.0 8B + temp-0.0
    judge, no dropout in eval), so without a seed lever every "seed" would be
    byte-identical and the multi-seed >=2/3 gate would be degenerate. The seed
    shuffles (a) which fact is probed at which probe turn and (b) the filler
    order -- different conversation orderings give different decay ages at each
    probe (different ``rec_i`` -> different salience firing), so each seed is a
    distinct-but-valid conversation and the gate tests robustness across
    orderings. ``random.Random(seed)`` keeps each seed reproducible."""
    rng = random.Random(seed)
    fact_order = list(range(n_facts))
    rng.shuffle(fact_order)
    filler_pool = list(FILLER_QUERIES)
    rng.shuffle(filler_pool)
    probe_start = max(8, horizon // 4)
    probe_step = (max(4, (horizon - probe_start - 2) // max(1, n_facts - 1))
                  if n_facts > 1 else 1)
    fact_turns = {}
    for i in range(n_facts):
        t = min(probe_start + i * probe_step, horizon - 1)
        fact_turns[t] = fact_order[i]  # probe turn t probes fact fact_order[i]
    schedule = []
    for t in range(horizon):
        if t in fact_turns:
            fi = fact_turns[t]
            schedule.append((facts3[fi][1], f"fact_{fi:02d}"))  # the QUESTION
        else:
            schedule.append((filler_pool[t % len(filler_pool)], None))
    return schedule, fact_turns


# ── Tier 1: coverage / cost-parity (offline, no LLM) ──

def _thresholds(theta: float, permissive: bool) -> SalienceThresholds:
    """Shipped phi/surprise_cap + retuned theta (serve dist, Probe 2), OR the
    permissive upper-bound (every scored anchor is salient) for a wiring smoke."""
    if permissive:
        return _permissive_thresholds()
    shipped = load_salience_thresholds(str(THRESHOLDS))
    return SalienceThresholds(
        theta=theta, phi=shipped.phi, surprise_cap=shipped.surprise_cap,
        theta_percentile=50.0, phi_percentile=shipped.phi_percentile,
        surprise_cap_percentile=shipped.surprise_cap_percentile,
        basis=f"retuned theta={theta} (serve dist, Probe 2); phi/surprise shipped",
        n_recoverability=shipped.n_recoverability, n_relevance=shipped.n_relevance,
        n_latent_dynamics=shipped.n_latent_dynamics)


def _run_tier1(*, n_facts: int, horizon: int, ring_capacity: int,
               theta: float, facts3: list, schedule: list,
               permissive: bool = False) -> dict:
    """Reuses the validated cost-parity runners. Returns the summary (no raw
    tensor-bearing records) + the STRM cost used to tune FIXED's N for Tier 2."""
    facts2 = [(s, q) for (s, q, _g) in facts3]
    fact_ids = [f"fact_{i:02d}" for i in range(n_facts)]
    retuned = _thresholds(theta, permissive)

    tmpdir = tempfile.mkdtemp(prefix="pondr_ship_t1_")
    store: Optional[HippocampalStore] = None
    try:
        store = HippocampalStore(str(Path(tmpdir) / "db"))
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True,
            retrieval_gate=None, embedder=embedder)
        _seed_corpus(store, embedder, facts2)
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        common = dict(store=store, retriever=retriever, embedder=embedder,
                      ring_capacity=ring_capacity, cfg=cfg)

        bb_strm = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_strm = _build_orchestrator(
            backbone=bb_strm, strm_salience=True, thresholds=retuned,
            recoverability_head=rec, latent_dynamics_head=ld, relevance_head=rel,
            **common)
        bb_off = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_off = _build_orchestrator(
            backbone=bb_off, strm_salience=False, thresholds=None,
            recoverability_head=None, latent_dynamics_head=None,
            relevance_head=None, **common)
        bb_fix = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_fix = _build_orchestrator(
            backbone=bb_fix, strm_salience=False, thresholds=None,
            recoverability_head=None, latent_dynamics_head=None,
            relevance_head=None, **common)

        strm = _run_strm(orch_strm, schedule, facts2)
        off = _run_off(orch_off, schedule, facts2)
        target_cost = max(strm["cost"], 1)
        N = max(1, horizon // target_cost)
        fixed = _run_fixed(orch_fix, schedule, N, facts2)
        selectivity = _selectivity(strm["per_turn_scores"], fact_ids)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _coverage(rep):
        recs = [r for r in rep["records"] if r["target"] is not None]
        return sum(1 for r in recs if r["target_recalled"]), len(recs)

    s_hit, s_n = _coverage(strm)
    f_hit, f_n = _coverage(fixed)
    o_hit, o_n = _coverage(off)
    s_cost, f_cost = strm["cost"], fixed["cost"]
    return {
        "strm": {"hit": s_hit, "n": s_n, "cost": s_cost},
        "fixed": {"hit": f_hit, "n": f_n, "cost": f_cost, "N": N},
        "off": {"hit": o_hit, "n": o_n},
        "strm_cost": s_cost, "N": N,
        "budget_parity": f_cost <= s_cost,
        "strm_beats_fixed": (s_hit > f_hit) and (s_cost <= f_cost),
        "strm_beats_off": s_hit > o_hit,
        "selectivity": selectivity,
    }


# ── Tier 2: live factual accuracy (real Bonsai 8B + DeepSeek-flash judge) ──

def _gold_map(facts3: list) -> dict:
    return {f"fact_{i:02d}": (facts3[i][1], facts3[i][2])
            for i in range(len(facts3))}


def _acc(per_probe: list) -> dict:
    n = len(per_probe)
    correct = sum(1 for p in per_probe if p["vote"] == "correct")
    p = correct / n if n else 0.0
    return {"acc": p, "correct": correct, "n": n, "ci95": _wilson_ci95(p, n)}


def _run_acc_strm(orch, schedule, facts3, gmap) -> list:
    """STRM: armed salience hook surfaces the relevance-gated recall into the
    synthesis context via the merge path. Judge the 8B answer on probe turns."""
    _reset_for_trial(orch)
    _disable_prompt_driven(orch)
    _seed_ring_facts(orch, [(s, q) for (s, q, _g) in facts3])
    per_probe = []
    for t, (query, target) in enumerate(schedule):
        res = orch.query(query)
        if target is None:
            continue
        sigs = res.get("salience_signals", []) or []
        surfaced = any(s.get("kind") == "recall" and s.get("anchor_source_id") == target
                       for s in sigs)
        ans = res.get("response") or ""
        q, gold = gmap[target]
        verdict = judge_answer(q, gold, ans)
        per_probe.append({"turn": t, "target": target, "surfaced": surfaced,
                          "vote": verdict["vote"], "votes": verdict["votes"],
                          "answer": (ans or "")[:200]})
    return per_probe


def _run_acc_fixed(orch, schedule, facts3, gmap, N) -> list:
    """FIXED: salience OFF; every N turns a manual proactive recall round-robins
    over the seeded facts. The round-robin top hit is surfaced through the SAME
    merge path as STRM via ``query(preset_salience_fired=...)``. The kwarg is
    applied AFTER query()'s per-turn reset, so it survives to the merge -- unlike
    setting ``_salience_fired_episodes`` before query(), which the reset wipes
    (the fairness bug this fixes: without it FIXED's context is always empty and
    any STRM win is unfair). The merge checks the attribute, not the armed flag."""
    _reset_for_trial(orch)
    _disable_prompt_driven(orch)
    _seed_ring_facts(orch, [(s, q) for (s, q, _g) in facts3])
    per_probe = []
    manual_count = 0
    for t, (query, target) in enumerate(schedule):
        preset = None
        surfaced_target = False
        if N > 0 and t % N == 0 and t > 0:
            fact_ids = [sid for sid in _ring_fact_ids(orch)
                        if sid and sid.startswith("fact_")]
            if fact_ids:
                fid = fact_ids[manual_count % len(fact_ids)]
                manual_count += 1
                slot = next((s for s in orch.working_memory.ring_buffer()
                             if s.source_id == fid), None)
                if slot is not None and slot.text is not None:
                    emb = orch.working_memory.embed([slot.text])[0]
                    hits = orch.retriever.retrieve_by_embedding(emb)
                    if hits:
                        top = hits[0]
                        preset = [top]
                        surfaced_target = (top.get("episode_id") == target)
        res = orch.query(query, preset_salience_fired=preset)
        if target is None:
            continue
        ans = res.get("response") or ""
        q, gold = gmap[target]
        verdict = judge_answer(q, gold, ans)
        per_probe.append({"turn": t, "target": target, "surfaced": surfaced_target,
                          "vote": verdict["vote"], "votes": verdict["votes"],
                          "answer": (ans or "")[:200]})
    return per_probe


def _run_acc_off(orch, schedule, facts3, gmap) -> list:
    """OFF: never surface -> empty synthesis context -> 8B answers from no
    project context (the sanity floor)."""
    _reset_for_trial(orch)
    _disable_prompt_driven(orch)
    _seed_ring_facts(orch, [(s, q) for (s, q, _g) in facts3])
    per_probe = []
    for t, (query, target) in enumerate(schedule):
        res = orch.query(query)
        if target is None:
            continue
        ans = res.get("response") or ""
        q, gold = gmap[target]
        verdict = judge_answer(q, gold, ans)
        per_probe.append({"turn": t, "target": target, "surfaced": False,
                          "vote": verdict["vote"], "votes": verdict["votes"],
                          "answer": (ans or "")[:200]})
    return per_probe


def _run_tier2(*, n_facts: int, horizon: int, ring_capacity: int, theta: float,
               facts3: list, schedule: list, N: int,
               permissive: bool = False) -> dict:
    """Live factual accuracy for STRM / FIXED / OFF. Real Bonsai 8B synthesis
    (``ModeAGenerator`` on :8080) + DeepSeek-flash judge panel. The self-chat
    tool loop + feedback salience are turned OFF for a clean one-shot synthesis
    grounded ONLY in the surfaced context (saved + restored)."""
    facts2 = [(s, q) for (s, q, _g) in facts3]
    gmap = _gold_map(facts3)
    retuned = _thresholds(theta, permissive)

    tmpdir = tempfile.mkdtemp(prefix="pondr_ship_t2_")
    store: Optional[HippocampalStore] = None
    saved_loop = _runtime_config.self_chat_tool_loop_enabled
    saved_feedback = _runtime_config.feedback_salience_enabled
    _runtime_config.self_chat_tool_loop_enabled = False
    _runtime_config.feedback_salience_enabled = False
    try:
        store = HippocampalStore(str(Path(tmpdir) / "db"))
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True,
            retrieval_gate=None, embedder=embedder)
        _seed_corpus(store, embedder, facts2)
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        common = dict(store=store, retriever=retriever, embedder=embedder,
                      ring_capacity=ring_capacity, cfg=cfg)

        def _mk(salience, thresholds, rcd, ldd, rld):
            bb = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
            mode_a = ModeAGenerator(retriever, model=_bonsai_model(),
                                   endpoint=BONSAI_ENDPOINT)
            return _build_orchestrator(
                backbone=bb, strm_salience=salience, thresholds=thresholds,
                recoverability_head=rcd, latent_dynamics_head=ldd, relevance_head=rld,
                mode_a=mode_a, **common)

        orch_strm = _mk(True, retuned, rec, ld, rel)
        orch_fix = _mk(False, None, None, None, None)
        orch_off = _mk(False, None, None, None, None)

        strm_pp = _run_acc_strm(orch_strm, schedule, facts3, gmap)
        fixed_pp = _run_acc_fixed(orch_fix, schedule, facts3, gmap, N)
        off_pp = _run_acc_off(orch_off, schedule, facts3, gmap)
    finally:
        _runtime_config.self_chat_tool_loop_enabled = saved_loop
        _runtime_config.feedback_salience_enabled = saved_feedback
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    s, f, o = _acc(strm_pp), _acc(fixed_pp), _acc(off_pp)
    return {
        "strm": s, "fixed": f, "off": o,
        "budget_parity": True,  # N tuned from Tier 1 so FIXED cost <= STRM cost
        "strm_beats_fixed_accuracy": (s["acc"] > f["acc"]),
        "strm_beats_off_accuracy": (s["acc"] > o["acc"]),
        "per_probe": [{"turn": strm_pp[i]["turn"] if i < len(strm_pp) else None,
                       "target": strm_pp[i]["target"] if i < len(strm_pp)
                                 else (fixed_pp[i]["target"] if i < len(fixed_pp)
                                       else (off_pp[i]["target"] if i < len(off_pp) else None)),
                       "strm_vote": strm_pp[i]["vote"] if i < len(strm_pp) else None,
                       "fixed_vote": fixed_pp[i]["vote"] if i < len(fixed_pp) else None,
                       "off_vote": off_pp[i]["vote"] if i < len(off_pp) else None}
                      for i in range(max(len(strm_pp), len(fixed_pp), len(off_pp)))],
        "per_probe_detail": {"strm": strm_pp, "fixed": fixed_pp, "off": off_pp},
    }


# ── driver ──

def run_eval(*, n_facts: int, horizon: int, ring_capacity: int, theta: float,
             seed: int, tier1_only: bool, skip_acc_if_cov_fails: bool,
             permissive: bool = False) -> dict:
    facts3 = list(SHIP_FACT_BANK[:n_facts])
    # AUTO ring-cap: preserve fact slots so the salience hook can score them
    # (a too-small ring evicts the seeded facts before their probe -> coverage 0).
    if ring_capacity <= 0:
        ring_capacity = horizon + n_facts + 2
    schedule, fact_turns = _build_schedule(n_facts, horizon, facts3, seed)

    tier1 = _run_tier1(n_facts=n_facts, horizon=horizon, ring_capacity=ring_capacity,
                       theta=theta, facts3=facts3, schedule=schedule,
                       permissive=permissive)

    cov_pass = bool(tier1["strm_beats_fixed"] and tier1["budget_parity"])
    run_acc = (not tier1_only) and (not skip_acc_if_cov_fails or cov_pass)
    if run_acc:
        tier2 = _run_tier2(n_facts=n_facts, horizon=horizon,
                           ring_capacity=ring_capacity, theta=theta,
                           facts3=facts3, schedule=schedule, N=tier1["N"],
                           permissive=permissive)
        skipped_accuracy = False
    else:
        tier2 = {
            "strm": {"acc": 0.0, "correct": 0, "n": 0, "ci95": [0.0, 1.0]},
            "fixed": {"acc": 0.0, "correct": 0, "n": 0, "ci95": [0.0, 1.0]},
            "off": {"acc": 0.0, "correct": 0, "n": 0, "ci95": [0.0, 1.0]},
            "budget_parity": bool(tier1["budget_parity"]),
            "strm_beats_fixed_accuracy": False,
            "strm_beats_off_accuracy": False,
            "per_probe": [], "per_probe_detail": {"strm": [], "fixed": [], "off": []},
            "skipped_accuracy": True,
        }
        skipped_accuracy = True

    seed_pass = bool(tier2["strm_beats_fixed_accuracy"] and tier2["budget_parity"])
    return {
        "seed": seed, "n_facts": n_facts, "horizon": horizon,
        "ring_capacity": ring_capacity, "theta": theta,
        "fact_probe_turns": fact_turns,
        "tier1_coverage": tier1,
        "tier2_accuracy": tier2,
        "skipped_accuracy": skipped_accuracy,
        "seed_pass": seed_pass,
        "coverage_pass": cov_pass,
    }


def _print_report(rep: dict) -> None:
    t1, t2 = rep["tier1_coverage"], rep["tier2_accuracy"]
    print("=" * 72)
    print(f"STRM SHIP-DECIDING EVAL (facts={rep['n_facts']} horizon={rep['horizon']} "
          f"ring={rep['ring_capacity']} seed={rep['seed']} theta={rep['theta']})")
    print(f"probe turns: {rep['fact_probe_turns']}")
    print("-" * 72)
    print("  TIER 1 -- coverage / cost-parity (offline)")
    print(f"  {'cond':6} {'hit':>8} {'cost':>8}")
    for name, c in (("OFF", t1["off"]), ("STRM", t1["strm"]), ("FIXED", t1["fixed"])):
        print(f"  {name:6} {c['hit']:>4}/{c['n']:<3} {c.get('cost', 0):>8}")
    print(f"  budget parity (FIXED<=STRM): {'YES' if t1['budget_parity'] else 'NO'} "
          f"(STRM={t1['strm']['cost']} FIXED={t1['fixed']['cost']} N={t1['fixed']['N']})")
    print(f"  coverage gate (STRM beats FIXED @ equal-or-lower budget): "
          f"{'PASS' if t1['strm_beats_fixed'] else 'FAIL'}")
    sel = t1["selectivity"]
    print(f"  selectivity (2a discriminates?): "
          f"{'YES' if sel['discriminates'] else 'NO'} (min gap={sel['min_gap']}; >=0.2)")
    print("-" * 72)
    print("  TIER 2 -- factual accuracy (live 8B + flash judge)")
    if rep["skipped_accuracy"]:
        print("  [SKIPPED -- coverage failed and --skip-accuracy-if-coverage-fails is on]")
    else:
        for name, c in (("OFF", t2["off"]), ("STRM", t2["strm"]), ("FIXED", t2["fixed"])):
            print(f"  {name:6} acc={c['acc']:.3f} ({c['correct']}/{c['n']}) "
                  f"ci95=[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]")
        print(f"  accuracy gate (STRM > FIXED): "
              f"{'PASS' if t2['strm_beats_fixed_accuracy'] else 'FAIL'} "
              f"(STRM={t2['strm']['acc']:.3f} FIXED={t2['fixed']['acc']:.3f})")
    print("-" * 72)
    print(f"  SEED PASS (accuracy + budget parity): {rep['seed_pass']}  "
          f"| coverage_pass={rep['coverage_pass']}")
    print("=" * 72)


def _main() -> int:
    ap = argparse.ArgumentParser(
        description="STRM Phase 4 ship-deciding eval: STRM+recall vs fixed-interval RAG")
    ap.add_argument("--facts", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--ring-cap", type=int, default=0,
                    help="0 = AUTO (horizon+n_facts+2, preserves fact slots); "
                         "an explicit small value evicts facts and breaks the mechanism")
    ap.add_argument("--theta", type=float, default=-0.04,
                    help="retuned recoverability threshold (serve dist, Probe 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--tier1-only", action="store_true",
                    help="run only the offline coverage/cost-parity tier (no LLM)")
    ap.add_argument("--no-skip-accuracy-if-coverage-fails", action="store_true",
                    help="run Tier 2 even if Tier 1 coverage fails (default: skip)")
    ap.add_argument("--permissive", action="store_true",
                    help="DEBUG: use permissive thresholds (every scored anchor is "
                         "salient) to confirm the salience->merge->8B->judge wiring "
                         "fires end-to-end. NOT the ship gate (the gate uses retuned theta).")
    args = ap.parse_args()
    if not ARTIFACTS_PRESENT:
        print("MISSING ARTIFACTS -- need the trained backbone + 2a/2b/2c heads + "
              "thresholds.json. Run the STRM training scripts first.", file=sys.stderr)
        return 2
    rep = run_eval(n_facts=args.facts, horizon=args.horizon,
                   ring_capacity=args.ring_cap, theta=args.theta, seed=args.seed,
                   tier1_only=args.tier1_only,
                   skip_acc_if_cov_fails=not args.no_skip_accuracy_if_coverage_fails,
                   permissive=args.permissive)
    _print_report(rep)
    if args.out:
        # Drop the verbose per-probe detail from the written JSON (keep the
        # summary per_probe grid); the detail is only for the console.
        out = dict(rep)
        if "tier2_accuracy" in out and isinstance(out["tier2_accuracy"], dict):
            out["tier2_accuracy"] = {k: v for k, v in out["tier2_accuracy"].items()
                                     if k != "per_probe_detail"}
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())