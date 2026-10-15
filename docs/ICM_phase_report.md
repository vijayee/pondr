# ICM Phase Report — Intent-Conditioned Memory, status through 2026-10-15

> **ARC STATUS: CLOSED 2026-10-15** at 48 verified reversals (corpus-gated;
> see Addendum 2). The head-vs-strongest-cheap-leg / segmenter question is
> parked as a power-starved open thread — resumable via the LoCoMo path in §5
> whenever a larger organic corpus is wanted. Everything measured is committed
> (69d8839 → 48e4af5); nothing is in-flight.

**Purpose:** handoff summary for a second agent. Covers what happened this
phase (2026-10-14 → 15), the measured results, where everything lives, and what
is honestly open. The fuller evidence write-up is
`docs/ICM_retrieval_taxonomy.md` (committed b14a3eb); this doc is the "where we
are" map.

## 0. Context — what ICM is

A separate research experiment in this repo (do NOT fold into the Pondr/fade
architecture): the plan is `docs/Intent_Conditioned_Memory_Research_Plan.md`
(authored from an LLM-design conversation preserved in `docs/research-chat.json`).
Central thesis under test: **relevance is irreducible to cheap signals** — i.e.,
a learned/intent-conditioned lens should beat raw dense cosine where it matters.
Nine pre-registered demonstrations have run against it; none has found a case
where learning was required. `icm*/` + `scripts/icm/` are the experiment's code;
`data/_scratch/icm/` (gitignored) holds the run artifacts.

## 1. What happened this phase (chronological, with the numbers that stick)

**a. The 500-question LongMemEval file was fully harvested** (segments offset
50 / 100 / 300; 4-arm counterfactual protocol, cross-model judge, per-question
labels-recovery spearman). Combined: ~50 raw reversals / 368 measurable questions
(13.6%). Segment tail rates swing 6–18% — the file is heterogeneous; tail
concentrates in temporal-reasoning / multi-session / knowledge-update types and
is **absent (0/25) from single-session-assistant** questions.

- Judge lineage: kimi-k3:cloud → **kimi-k2.7-code:cloud** (user cost decision;
  probe-verified equal on the discriminating pair; labels coarser → more ties).
- All rows are checkpointed JSONL with resume; artifacts are run-scoped
  (`*_o{offset}.json`); rows carry `relay_fail` + `judge_suspect` visibility
  markers (added after the user raised ollama rate-limit contamination — audit
  found zero empty answers in the headline measurements; the unauditable runs
  are harvest payoff means only, interpreted for structure).

**b. Two-judge verification screens** (kimi family + deepseek as independent
second judge, margin 0.1) — offset-100: 25/34 verified (74%); offset-300
reversals: 10/13; offset-300 *near-zero spearmans*: 10/20 (screening the
"cosine fails to discriminate" bucket doubled the graded pool). Combined
verified tail: **35 questions** across two segments/judges.

**c. The k=1 selection A/B on the offset-100 verified tail** (judge-free: graded
answers, no oracle): cosine-top1 **0.160** vs time-token-rule-top1 **0.400** vs
gold ceiling **0.680**; discordant **6/0, binomial p = 0.03125** — cosine never
won. Recorded per the second agent's critique as **fixed-point demo #8 in a
judge-free robe**: the oracle finally mechanical, the discriminator still a
cheap rule — strong thesis *further falsified*, not demonstrated.

**d. The structured temporal index** (generic operators before/after/total/
earliest/latest + lexical anchor + dataset `haystack_dates` arithmetic over the
FULL episode store; no learning): scored **0.480** — a statistical TIE with the
trivial rule (3/1, p=0.625). The critique's pre-registered "the query engine
wins" was REJECTED on o100.

**e. Out-of-sample router test** — the o100-fit dispatch (`earliest→time, else
→index`) was frozen and run on unseen o300 verified reversals: direction-positive
(5/0 F1 vs time-rule, p=0.0625 at n=9). After the near-zero expansion, **pooled
o300 (n=17, relay-clean) made both decisive: index vs time-rule 10/1 (p=0.0117);
router vs time-rule 8/1 (p=0.039)**. Out-of-sample ordering: **index (0.304) >
gold-context (0.271) > router (0.227) > cos (0.158) > time (0.121)** — the cheap
structured engine is the strongest temporal leg OOS, and selection arms beat the
full-gold-context arm a third time (focused evidence beats diluted context; a
"gold ceiling" is not a ceiling under F1).

**f. Grader-bias finding (must-read before any grading run):** the 6-gram
containment grader floor-collapses on paraphrase-form golds segment-wide
(o300: correct "Twice." vs gold "We've met up twice." = 0.0; "2" vs "two" = 0.0;
the o300 gold "ceiling" of 1/10 was a grader artifact). All comparisons now
report **token-F1 alongside 6-gram**. Graders are biased instruments; never
grade single-metric.

**g. Answerer-swap null:** deepseek-v4-pro vs v4.1-flash on 17 paired identical
cells moved nothing (discordants 4/3, 4/3, 3/3 — all p=1.0). The "stronger
answerer recovers the ceiling" hypothesis is dead for this tail. The o300
residue decomposes into: grader partial-credit, **2/20 abstain-type golds**
(LongMemEval questions whose gold is a refusal — "information not enough" —
which neither answerer produces; an abstain instruction would fix that slice for
every arm), and temporal bookkeeping beyond the evidence text.

## 2. Where we are (the end-state)

- **The taxonomy is the deliverable** (four measured axes, each with its cheap
  leg): values/identity → lexical anchor (43/43); goal-conditioning → trained
  dense head (12/12 vs 0/12 — the dense lens's one measured job); temporal/
  relational → structured index + routing among cheap legs (now OOS-decisive:
  10/1, p=0.0117); storage/decay → forgetting ledger (7/12 harm-path caught).
- **No measurement anywhere in the arc argues a learned intent-conditioned lens
  adds payoff** over cheap specialized legs plus routing. The weak thesis holds.
  The strong thesis is *either false or untestable, and for engineering purposes
  the two are indistinguishable* (learning never required in any of the 9
  worlds; see Addendum for the precise form).
- **The one open question is power-starved, not falsified:** whether a learned
  head beats the 2-leg router (o100 OR-ceiling 0.640/0.680) on a real train/test
  split. The verified tail = 35 questions — too few. Needs more organic
  segments harvested + screened before any head run counts (LOO on 24 samples
  was rejected as a variance machine).
- Oracle-router over index|time on o300 = 0.305 ≈ index alone (index dominates
  there; on o100 the OR over the two legs was the big number, 0.640).

## 3. Standing protocol laws (the arc's constraints; a follow-up agent should keep them)

1. Pre-check before any training counts: raw cosine must fail to separate the
   payoff winner; the none-arm must be measured; the goal string must not name
   the memory.
2. Report spearman as a DISTRIBUTION (the tail is the finding), never a mean.
3. Judge labels are cosine-recoverable by construction (oracle theorem);
   judge-free graded answers are the decisive instrument — but grade with BOTH
   6-gram and token-F1 (grader bias finding above).
4. Ranks pay through SELECTION (k=1 under budget), never permutation of the
   same set (the k=3 A/B tied 25/25 for exactly this reason).
5. Authored/synthetic worlds are closed (fixed-point theorem): only organic
   corpora test irreducibility.
6. Harvest-first before any learned-head test; LOO on <25 samples is noise.
7. Judge = kimi-k2.7-code:cloud (never kimi-k3:cloud — token cost), second judge
   = deepseek (cross-family agreement required; margin 0.1).
8. Every run row carries `relay_fail`/`judge_suspect` markers; audit empty
   answers before believing zeros.
9. Runner artifacts get run-scoped names; smoke runs never write to a real
   run's output path (a smoke once clobbered the committed offset-50 results).

## 4. File map

| artifact | path |
|---|---|
| plan / provenance | `docs/Intent_Conditioned_Memory_Research_Plan.md`, `docs/research-chat.json` |
| evidence write-up | `docs/ICM_retrieval_taxonomy.md` |
| this report | `docs/ICM_phase_report.md` |
| harvest runner (offset+resume) | `scripts/icm/organic_scale_500q.py` |
| verification screen | `scripts/icm/verify_reversal_tail.py` (--ckpt/--ledger/--out/--sp-min/--sp-max) |
| verified corpus (25 q, o100) | `data/_scratch/icm/icm_head_corpus_o100.json` |
| payoff A/B (k=1/k=3) | `scripts/icm/payoff_tail_ab.py` |
| temporal index engine | `scripts/icm/temporal_index_leg.py` (`engine_pick`) |
| OOS router test | `scripts/icm/router_oos_test.py` (--model for answerer swap) |
| result JSONs / checkpoints | `data/_scratch/icm/*_o{50,100,300}.json`, `*_ckpt*.jsonl`, results tables in `docs` |

**Model config:** answerer deepseek-v4.1-flash:cloud (pro=swap-null); judge
kimi-k2.7-code:cloud; verifier deepseek-v4.1-flash:cloud; embedder
BAAI/bge-small-en-v1.5 local; all via local Ollama relay :11434.

## 5. Concrete next actions (in value order)

1. **Abstain-instruction quick win:** add "if the evidence is insufficient say
   'cannot determine'" to the answerer prompt; re-run the 2 abstain-gold cells
   (q327, q325) — cheap, likely lifts every arm on those cells.
2. **Grow the verified tail:** harvest additional organic segments (or other
   LongMemEval files / LoCoMo) with the same runner, screen near-zero + reversal
   buckets, aim for 70–80 verified with F1-gradeable golds → then the
   head-vs-2-leg-router split becomes answerable.
3. **Router v2:** the dispatch is currently `earliest→time else→index`; the o300
   data suggests near-universal index dispatch might beat it — test out-of-sample
   before adopting.

---
*All numbers in this doc trace to committed scripts + result JSONs. Commits run
69d8839 → b14a3eb on main, 2026-10-14/15.*

## 6. Addendum — the second agent's corrections, accepted (2026-10-15)

The second agent's read of v1 won twice against its own priors and sharpened
four things:

1. **"Falsified 9/9" over-claims; the precise form is now:** *the strong thesis
   is either false or untestable, and for engineering purposes the two are
   indistinguishable* — what is proven is that learning was never REQUIRED in
   any of the 9 worlds (and provably cannot be in authored worlds, fixed-point
   theorem); what is not proven is the negation ("no world requires learning"),
   which enumeration cannot establish. The engineering consequence is identical
   either way: build the cheap legs, don't wait for the lens.

2. **Gold-context is not a ceiling — PROMOTED to a headline finding.** Three
   independent replications now show selection arms beating the
   full-gold-context arm (o300 F1: index 0.304 > gold 0.271; o100 same
   direction; near-zero bucket same). This is a GENERAL LLM finding, not an ICM
   one: for this answerer on this task class, diluted-but-complete context
   underperforms focused-but-partial context. Flagged for Ponder (NOT folded
   into Pondr per the separation rule): Ponder's tier-1 context strategy
   ("top-k episodes, full text") is downstream of this result — a context-focus
   experiment there is warranted before treating top-k-full-text as a floor.

3. **The router is segment-dependent, not cleanly retired:** on o300 it adds
   nothing over index alone (0.305 vs 0.304 — the "next costume" reading); on
   o100 the OR over the two legs reached 0.640/0.680. Both readings are honest;
   the segmentation is the finding (tail rates 6-18% by segment) and
   "segment, then dispatch" is the architecture shape it implies — making the
   SEGMENTER the one remaining candidate that might need to be learned.

4. **The head-vs-router open question is now DOUBLY gated:** before any head
   run, (gate i) the corpus must reach 70-80 verified reversals with
   F1-gradeable golds, AND (gate ii) the opponent on the TARGET segment must be
   measured to be the strongest cheap leg there — if the router adds nothing
   over a single leg on that segment, the head's opponent is that leg, and the
   test must be named accordingly (law 1's pre-check, applied to the opponent).

## 7. Addendum 2 — corpus-growth phase results (2026-10-15, closing)

1. **Abstain-instruction null:** paired on 17 cells, nothing moved (all p >=
   0.625); explicit "cannot determine" appeared in only 6/85 cells despite the
   instruction. Abstention is not instruction-fixable at this scale; not
   pursued further.
2. **Tie re-judge sweep:** 82 judge-tie rows re-judged under deepseek; only
   21/82 became measurable — BOTH judge families tie coarsely, so the tie
   ceiling is judge-population-wide, not kimi-specific. 6 new reversals surfaced,
   3 survived the inverted-pair screen (deepseek primary + kimi verifier).
3. **Verified pool = 48** (25 o100 + 20 o300 + 3 newcomers); arm runs cover all
   of them. Still ~25-30 short of the ~70-80 head-test gate; growth options are
   LoCoMo (harness ships none installed) or formal closure at 48.
4. **The consolidated cross-segment tables moved the end-state one notch:** on
   o100, freshly F1-graded, the structured index leads decisively over cosine
   (14/3, p=0.0127) and suggestively over the token rule (10/3, p=0.092) —
   the 6-gram read had called that pair a tie; under F1 the index dominates
   BOTH other legs on BOTH segments. On o300 (n=19) index vs time-rule is
   decisive (10/2, p=0.0386), router-vs-time-rule decisive (9/1, p=0.0215),
   and OR(index|time) = 0.28 ≈ index alone (index dominates broadly there;
   the OR only earned its keep on o100, where no single leg dominated).
   Final leg story: the structured temporal engine is the strongest retrieval
   leg on the verified tail everywhere it was measured; the dense lens's real
   job remains goal-conditioned ranking only; the open question stays the
   segmenter (and after more corpus, the head-vs-strongest-cheap-leg split).

## 8. Addendum 3 — the power-flagged head test ran (2026-10-15, user override)

Claim #5's test was executed despite the unmet corpus gate (48 of ~70-80;
declared exploratory in the script header, verdict line says UNDERPOWERED, not
decisive). Design: head = logistic on [cos, time-jaccard, log-len, op
one-hots, n_sessions], z-normalized by TRAIN stats (the o100 scaling bug's
fix), trained ONLY on o100's 75 kimi-labeled pairs, given the same full-store
access as the index; opponent = o300's measured strongest cheap leg (index,
F1 0.272-0.287); test = o300's 19 verified cells graded judge-free.

**Result: head 0.223 vs index 0.287 — paired 1/7, p=0.0703, suggestive AGAINST
the lens** (a real relay-fail on one cell was caught by the visibility markers
and excluded, n=18). The head did beat the token rule (7/3, ns), landing
between the two cheap legs. Reading: in an organic world, the cheap structured
leg stayed ahead of a label-trained learned head — the segmenter/lens open
question moves from open-neutral to **open-leaning-no**. The gate still applies
to a *decisive* verdict; the direction is now measured. scripts/icm/
head_vs_index_oos.py (96d2263).

## 9. Addendum 4 — demo #9 ran (2026-10-15): the rescue test says no, airtight

The second agent's rescue test (semantic match under zero lexical overlap) was
pre-registered in `scripts/icm/demo9_zero_overlap.py` and measured same-day on
data already on disk (all 450 harvested questions, own-haystack gold pairs,
zero relay): on the 110/298 pairs with content-token Jaccard <= 0.05, dense
(bge) ranked gold at **41.7 mean percentile — significantly below chance**
(p = 0.045), top-1 0.30. Dense has no semantic second job; the negative result
is airtight in exactly the regime where its theoretical advantage should live.
Correction (same-day): the 0.90 time-channel figure was a tie-break artifact
(99/110 rows all-tie; on separated rows it is 0.364) — there is no fifth leg;
that regime is genuinely sparse under every measured leg. The taxonomy doc carries the full elevations
(dilution-not-ceiling, flat-scale null, grader-collapse, ties-as-information),
two proposal axes (negation/absence, contradiction/versioning), the
inverse-bitter-lesson framing, and the external-validity note.

## 10. Addendum 5 — the intent-variation adjudication (2026-10-15)

The second agent's final round produced one overreach and one confirmed claim,
and the confirmed claim closes the "why did intent never show organically"
question with a measurement:

- **Overreach: "we measured hygiene, not relevance."** The A/Bs measured
  answer utility (the mechanical definition of relevance) throughout, and the
  temporal/values legs are query-memory relations, not store hygiene. What was
  never measured is NARROWER: organic intent-VARIATION.
- **Confirmed + measured: the corpus cannot carry it.** Demo-10 gate-0 census
  (demo10_valueflip_census.py, cb32407): in the full 500-question LongMemEval
  file, only 8/940 sessions appear in more than one question's haystack and
  ZERO have mixed gold/non-gold utility — same memory, different value under
  different goals NEVER occurs naturally. The haystacks are woven per-record;
  organic intent-variation is designed out. This is why intent only ever
  appeared in the authored 2x2, which the fixed-point theorem flags as closed
  — "false or untestable" now resolves further toward "UNTESTABLE ON THIS
  CORPUS."
- **The corrected next action:** LoCoMo is no longer just "more volume" — its
  one-shared-conversation/many-questions shape is the INVERSE of LongMemEval's
  per-question weaving and is the first candidate corpus that may naturally
  carry value-flips. Next rung: run the SAME census on LoCoMo FIRST (gate-0);
  a non-zero census unlocks organic demo #10 (for each shared session: can any
  cheap leg separate its gold-questions from its non-gold-questions? the
  residue = the intent-shaped signal). A zero census closes intent-variation
  as corpus-absent in this dataset family too.
- The Ponder live-instrumentation proposal (log intent + outcome alongside
  retrieval; the user as oracle; harvest organic value-flips from real usage)
  is parked as an option requiring the user's decision — it touches the main
  repo.

## 11. Addendum 6 — demo #10 ran (2026-10-15): the final answer to the intent question

LoCoMo was acquired from the official repo (data/_scratch/locomo/locomo10.json)
and gate-0 PASSED decisively: **272/272 sessions in value-flip shape**
(LongMemEval: 0/940) — one shared conversation grounds ~199 questions each,
and the evidence pointers ("D{n}:turn") give per-question gold labels FREE,
so the whole demo ran with no relay and no oracle.

**Results (local, scripts/icm/demo10_locomo_legs.py, 2b47402):**
cheap legs separate 270/272 organic value-flips (99.3%); lexical overlap
dominates (mean AUC 0.835, 82% of sessions' winning leg; captured>=0.8 on 65%);
cosine second (0.697); time-tokens fail (0.515 — chat sessions carry no time
words). **The segmenter's first organic verdict: NO — the winning leg does not
vary enough (one leg suffices); beyond-cheap residue = 2/272 sessions.**

Caveat in the record: pointer-labels are gold-by-construction (the pointed
session contains the answer), which partially flatters the lexical leg; cosine
at 0.697 is the non-circular cheap reference. The headline's shape — cheap
legs capture ~99% — survives the caveat.

**The user's driving question ("is it worthwhile to index memories by
intent?") is now answered by measurement: NO** — organic intent-variation
exists (unlike LongMemEval) and cheap legs capture it; intent's only
measured job remains the goal-conditioned ranker. The arc closes at ten
demonstrations with both rescue paths and the segmenter measured dead.

## 12. Addendum 7 — the agent-intent corpus protocol takes shape (2026-10-15)

The user's thesis (training data must be AGENT data — goal intent, step intent,
process, traces) was accepted as the identification of the whole arc's unit
error. Design response:

- **Where agent data exists** (by honesty-per-cost): this repo's own sessions
  and memory files (real user goals, small, dodges the authored-world
  objection because goals are the user's, not a benchmark author's); public
  agent trajectories (SWE-bench-class traces); live Ponder instrumentation
  (parked, user decision).
- **The extraction-precision probe ran** (probe_extraction_precision.py,
  3ed2acc): 24 ledger records reconstructed from this arc's explicit artifacts
  — structure (intent / branches / observed) recovered 24/24; commit messages
  alone carry verdicts for only 12/56 rounds; **58% of miner-STATED labels were
  valence-flipped** (adverse outcome labeled "violated" when the rule's
  semantics are prediction-match — a predicted feared branch is CONFIRMED).
  Protocol consequence: **labels are always COMPUTED by the congruence rule;
  never stated by a miner.** The retroactive layer is viable for structure,
  not labels.
- **The granularity question ("root intent + hierarchy of step intents — what
  granularity do we need?") gets a designed test, not an assumption:** capture
  events with declarations at ALL levels (goal / batch / step), then ablate:
  goal-only vs step-only vs full-hierarchy prediction of retrieval utility.
  Each grain's marginal contribution = its measured value. "Finest wins" is a
  registered hypothesis; Phase-0's hedging finding (heavy declarations shrink
  the useful stream 5->2/120) bounds the design: one-line step declarations,
  retrieval-bearing steps only.
- The ICM ledger (Phase 0-1) is the recording schema for all of this — built
  and gated; the new work is pointing real agent data through it.

