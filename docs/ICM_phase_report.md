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

## 13. Addendum 8 — the capture layer + ablation SHIPPED (869f5ac, 2026-10-15)

The agent-intent corpus's recording layer is live:

- **icm/capture.py** — hierarchical capture: goal (structured branches,
  at least one mandatory) -> steps (one-liners, retrieval-bearing only,
  branchless-step inheritance from the goal's prediction) -> retrieval events
  -> outcomes. Laws wired in: **labels are computed by the congruence rule at
  observe() time and may never be stated by the caller**; unpredicted named
  branches are VIOLATED (not errors); "other" = max surprise; records persist
  via the ledger (kind "icapture" registered in RECORD_KINDS/ID_KEYS) + a
  canonical append-only JSONL mirror. 6 tests green; all 34 ICM tests green;
  the main-repo suite's 18 failures are the known pollution family
  (fade/orchestrator files, flaps run-to-run), unrelated.
- **granularity_ablation.py** — the grain-value instrument: per retrieval
  event, utility (rule-computed) predicted by [goal+q | step+q | full |
  query-only] cosine features; each grain's marginal = its measured value.
  Synthetic smoke (instrument test, authored-world stamp): query-only 0.565
  and goal+q 0.526 when the goal carries no discriminating words, **step+q
  0.998 (+0.433 marginal)** — the machinery detects step-level signal when
  the query carries none, and can also ABSENT a grain's signal. Real mode
  consumes capture mirrors (requires >=3 events; events without outcomes are
  not scored — no fabrication; the mirror stores retrieval ids, so real runs
  join texts via the ledger).

## 14. Addendum 9 — the first training run on honest labels + its decomposition (2026-10-15)

The program's first pre-registered training run ran and its composition was
audited (both laws enforced):

- **train_faulty_head.py** — faultiness prediction on the CodeTraceBench-SWE
  mapped slice (11,963 steps / 259 tasks / 146 human-verified faulty), scored
  LEAVE-ONE-TASK-OUT pooled. Head 0.814 vs opponents length 0.754 / lexical
  0.64 / best-single-cosine 0.593 — **formal PASS by the pre-registered bar**.
- **audit_faulty_head.py (mandatory audit of any PASS):** loglen is the
  workhorse (0.754 alone); the learned context-only model scored 0.482 —
  WORSE than its own singles (tiny per-fold overfit on correlated weak
  features); the head's +0.06 over the best trivial rule = a STACK of cheap
  signals (len + lexical + step-position + file-access + weak context).
  **The win is the cheap-signal ensemble (the router) wearing a learned hat
  = demo #11.**
- **Round-1 correction owned:** the round-1 ablation's "context grain carries
  the signal (0.73)" was self-similarity contamination (the context text
  contained the compared action); clean context cosines = 0.55. Standing
  rule: never embed the compared item inside its context feature.
- Net: even on honest human labels + fair training + real power, no learned
  intent lens emerges; the router/ensemble is the material — consistent with
  all prior demonstrations. The arc's full question chain is now measured at
  every grain it ever posed.


## 15. Addendum 10 — the standard ML remediation battery RUN (2026-10-16)

The lens's two failure modes were never a closed verdict — the standard ML
toolbox had not run. It has now run, pre-registered before fitting (three
commits of pre-registration: 15173df, smalln, + fine/F4), seeded task-disjoint
splits, direction-always reporting.

**MODE 1 (small-n, CodeTraceBench head — battery_smalln_remediation.py):**
ridge-λ {0.01..10} × early-stop-iters {100/500/2500} leave-one-task-out grid,
both heads. Result: the full stack survives regularization (best cell 0.799 =
λ0.01/500 iters, beats all three opponents incl. loglen 0.754) and is ~at the
original unregularized head (0.814) — regularization neither rescues nor
destroys the stack. The ctx-only arm improved 0.482→0.558 under early stop
(mild per-fold overfit confirmed) yet remains far below the trivial length
rule. **Composition reading unchanged: demo #11 stands through remediation.**
(λ=10 diverges mechanically at lr=0.3: |1−lr·λ|≈3 oscillation — reported as
diverged cells, never scored as silent zeros.)

**MODE 2 (the E2 lens on the composed mixture — battery_mode2_remediation.py
+ fine addendum + F4):** the original readouts had a bug (val/holdout bound
scored against WRONG question vectors via subset-position indexing) — found,
fixed, rerun. Corrected battery = 26 configs across 4 families:

- *Anchor + early-stop/WD (from-scratch, proj-only hinge):* final-epoch
  fidelity on the battery's own 320-question split = 0.5531 (reproduces the
  baseline's 0.565); early stopping rescues 0.5531→0.774 (val-best ep150,
  SAME split); still −0.05 below cosine (0.8235); WD inert across 2 decades.
  Learning = slow decay from a bad init. (Direction note: the collapsed
  final-epoch lens recovers 5/15 tail while dropping mean-arm to 0.55 — it
  buys tail fixes with global accuracy; useless but measured.)
- *Identity-init / residual deltas (item 1, the mechanism target):* the ONLY
  family whose bound holds/Beats cosine OOS — 8/8 cells in 0.827–0.831
  (cosine 0.8235), deltas tiny by design, tail recovery 2/15 FROZEN across
  every lr/L2/epoch cell. A hair of extra agreement; not the failure cases.
  The geometry is knife-edge: the first hinge step out of identity drops val
  0.84→0.57 with lr 1.5, while lr 0.05–0.15 rides a slow, convergent
  improvement plateau (val flat 0.849 from ~ep300).
- *Residual-hinge at margins {0.02..0.3}+{0,0.005}:* SILENT — delta ≈ its
  σ-init after 1200 epochs, hinge constant, tail 0/15. Mechanism measured:
  the raw cosine already satisfies these margins in train, so the violation
  gradient is nearly empty; the targeted-loss framing has nothing to learn.
- *InfoNCE in-batch negatives (τ {0.05,0.1,0.3}):* same silence — delta ≈
  init, val at raw-cosine's own level, tail 0/15. The real-distribution
  negatives change nothing because the residual framing's softmax is dominated
  by the raw term.

**Verdict by the pre-registered bar (bound ≥ cosine−0.01 AND tail ≥ 0.5 AND
non-noop): NO SURVIVOR in 26 configs.** But the battery UPGRADES the negative
into structure, not silence: (a) perturb-don't-destroy WORKS in direction —
identity-init is the first lens parametrization that ends OOS *above* the
cosine it starts from, consistently; (b) it recovers only 2/15 of cosine's
tail failures regardless of every swept hyperparameter — the missing piece is
not training hygiene but violations to learn from; (c) on the composed
mixture the residual/InfoNCE objectives are empirically empty (margins
satisfied by cosine). **The measured bottleneck has moved from the optimizer
to the objective+data: the lens needs violations (real payoff contrasts), and
this corpus supplies almost none beyond what cosine already resolves.**

Still OPEN (none dismissed): more organic payoff data (LoCoMo harvest — the
one-shared-conversation corpus with value-flip shape), the two-stage residual
re-ranker (a loss defined ONLY on cosine's failures), curriculum, and identity
+ violation-enriched data (the battery cannot distinguish "identity helps a
little" from "identity + real violations helps a lot"). No E2 relay was run:
no config survived the discrimination gate, so by the pre-registration there
is nothing to relay yet.

Instruments (committed): battery_mode2_remediation.py (13 configs, exact
unit-checked collapsed gradients), battery_mode2_fine.py (fine checkpoints +
engaging margins), battery_mode2_f4.py (identity extension grid),
battery_smalln_remediation.py (mode-1 ridge/early-stop grid). Results JSONs +
trained W matrices under data/_scratch/icm/ (untracked).

## 16. Addendum 11 — Phase-2 E2 START-OVER on intent-bearing data (2026-10-16)

The corpus-drift guard (written this session after the mode-2 battery ran on
the rejected LongMemEval-composed mixture) forced the run the plan actually
names: the FIRST training with the intent operand in the loss. Corpus: the
agent-intent capture slice (CodeTraceBench-SWE mapped; 259 tasks / 11,963
steps / 146 human-verified faulty). Arms named in ternary terms --
bind(state,intent) = parameter-free BUNDLE(normalize(e(task)+e(plan))) with a
learned residual tanh-delta (battery-graded trainer); embed(prompt) = the
1-in-5-strided val carve kept on the TRAIN side, every scored task fully held
out; 8-fold task-disjoint CV; same pre-registered survival bar (beat all 4
opponents +0.02).

**Results: NO SURVIVOR, and the negative now has its intent-axis twin:**
- o1 embed(prompt) leg 0.6114 > o2 the untrained BUNDLE 0.5901 > lexical
  0.4482 / loglen 0.2459 (sanity: 1-0.754 loglen-inverted = 0.246 exact).
  **The parameter-free bind(state,intent) composition LOSES to plain prompt
  cosine before any learning.**
- The learned delta: hinge fully engaged (7,896 violating pairs at init,
  96.6% of all within-task useful/faulty pairs; gradient norm ~1.76) yet val
  monotonically declines; val-best = ep25 = the least-moved checkpoint for
  BOTH lrs; pooled = o2 exactly (delta sub-threshold at selection). Same
  geometry-fragility signature as the battery's proj-only family, now with a
  real intent operand. Not a cap-cut (600 ep run out).
- Cross-check flag: task/plan cosine direction here says high cosine =
  USEFUL (0.611), the faulty-head's reported 0.593 for faultiness used
  max-over-THREE (incl. stage text) -- a small recorded discrepancy, unresolved.

**THE GAP REPORT (first-class, numbers not adjectives):**
- Label famine quantified: 146 verified-faulty = 1.2% of steps; contrast
  exists in only 51/259 tasks (20%); ~208 tasks contribute NO usable
  per-task signal. The usefulness oracle is faulty-labels-INVERTED.
- Intent operand present but thin: declared agent plan on 53% of tasks (the
  bundle collapses to task text elsewhere); plan-echo steps (n measured) are
  a topically-trivial class the PASS audit must watch.
- State operand ABSENT (trajectory position is all) -- run trains
  bind(intent) only.
- Memory side = the agent's own actions, not retrieved records; pay-off
  labels for retrieval proper remain the scarce commodity (11 organic likes).
- Flip shape present at file grain (154 shared units) but utility labels too
  thin (16 negatives) to train on.
- ~58% of tars unparsed; SWE-TaskFlow (25,200 verifier-labeled) untouched.

**Verdict (per the standing no-dismissal rule): OPEN.** The ternary gate has
now been run twice with the intent operand present (composed corpus: axis
absent; this corpus: axis present, label power absent) and both times the
learned delta loses to the geometry it starts from. What changed: the
bottleneck is no longer attributable to the corpus lacking intent -- THIS
corpus has it and the result replicated. The live doors are (1) label
acquisition at utility-label power (SWE-TaskFlow verifier runs; user thumbs;
captured sessions) and (2) the two-stage re-ranker over the geometry's
failures. Instrument: scripts/icm/phase2_e2_intent_lens.py (gap report built
in). Purpose guard: memory icm-lens-corpus-guard.

## 17. Addendum 12 — the remaining levers on the intent corpus (2026-10-16)

The user's question -- "did we try all the ML levers on the data that HAS the
signal?" -- was answered by measurement: the addendum-11 run had used only
part of the battery. The rest ran (pre-registered, commit 1b0.., script
phase2_e2_intent_battery.py; 8 configs x 8-fold task-disjoint CV; same
opponents o1 0.6114 / o2 0.5901 / o3 0.4482 / o4 0.2459; same bar).

| config | pooled | best epochs (per fold) |
|---|---|---|
| G1 margin 0.0 | 0.5901 | scattered 25..600 |
| G1 margin 0.01 | 0.5902 | scattered |
| G1 margin 0.05 | 0.5901 | scattered |
| G3 WD 1e-3 | 0.5911 | clustered LATE 550..600 |
| G3 WD 1e-2 | 0.5909 | clustered LATE 550..600 |
| G4 low-rank rank16 | 0.5901 | mid |
| G2 InfoNCE t0.1 | 0.5901 | earliest 25 (7/8 folds) |
| G2 InfoNCE t0.3 | 0.5901 | earliest 25 (6/8 folds) |

**NO SURVIVOR; every lever lands on the init bundle (0.5901) within +-0.001.**
Readings: (1) margin is irrelevant to the outcome (0 through 0.1 identical);
(2) rank-16 delta == full-rank delta == no delta; (3) InfoNCE also degrades
from its first checkpoint and never returns (5x slower, same number);
(4) the ONLY directional lever is weight decay: stronger L2 pushes the
val-best checkpoint to the END of training and buys +0.001 -- the same
"smaller perturbations hurt less" signature, with no traction; (5) the
per-fold spread is IDENTICAL across all 8 configs (min 0.3089, max 0.7574),
i.e. the pooled variance belongs to the corpus (51/259 tasks carry labels),
not to any training choice. No early exits fired -- the objectives never ran
out of violations; the trained model simply never beats its own init on
held-out tasks.

**CORRECTION recorded with this run:** the addendum-11 instrument computed
o1 as task-cosine alone (a patch had silently no-op'd; the offset line was
accidentally correct because task_ids is sorted). Both are fixed in the
refactored script, which reproduces o2/o3/o4 EXACTLY (machine precision);
the corrected o1 is IDENTICAL (0.6114) -- the plan cosine never exceeds the
task cosine on this corpus, so the mislabel was cosmetic.

**Standing verdict, now fully lever-exhausted on intent-bearing data:**
the lens's failure is not optimizer-, schedule-, objective-, or
subspace-shaped; it is the ternary gate itself failing on this corpus, whose
label power (1.2 percent coverage, 51/259 contrast-bearing tasks) cannot
supply what the geometry lacks. Open doors unchanged: utility-label
acquisition at scale (SWE-TaskFlow 25,200 verifier runs; user thumbs;
captured sessions) and the two-stage re-ranker over the geometry's failures.

**Addendum-12 note (de-wonk, same session):** two write-path bugs of the
same class were found and fixed in the lens instrument -- `--opponents-only`
and `--smoke` both wrote to the real results file (the smoke path was caught
only on the second audit round, after the first fix). Both now write side
files; the lens results JSON was verified intact after each. The InfoNCE
trainer's coverage is recorded precisely: with positives = own non-faulty
steps it ALSO trains on the 208 no-faulty tasks, i.e. it saw all 259 tasks
(more data than the hinge family's 51 contrast tasks) and still landed on
the init bundle. The two smoke fixes were verified behaviour-preserving
(both trainers reproduce their pre-fix smoke values bit-for-bit).

## 18. Addendum 13 — the intent-detection smoke gate ran, and caught its own construction error (2026-10-16)

Script: scripts/icm/intent_detection_smoke.py (pre-registered, commit 5e38b8d;
public data only -- the personal corpora are excluded because the LLM
instruments are cloud calls). Gate: before any real-data run, an instrument
must move when intent changes and hold when only wording changes.

Numbers (96 pairs: 32 SENS / 64 INV, CodeTraceBench mapped):

| instrument | SENS (want DIFFERENT) | INV (want SAME) | LEX-HARD residual |
|---|---|---|---|
| LEX jaccard >= 0.25 | 0.00 | 0.00 | 0.00 |
| const-SAME / const-DIFFER | 0.00 / 1.00 | 1.00 / 0.00 | -- |
| SPO-STRUCT (extract + rule) | 0.75 | 0.00 | 0.25 |
| LLM-JUDGE (oracle, separate) | 0.00 | 0.719 | 0.479 |

Verdict: FAIL by the bar; vacuity guard CLEARED (the pair sets are not
lexically solvable). But the failures are not what the labels claimed:

1. **THE SENS CONSTRUCTION WAS MINE AND WRONG.** On inspection, the 32
   "changed intent" pairs are plan RE-DECLARATIONS whose texts differ by a
   refinement ("fix SQL syntax error" -> "fix SQL syntax error in Count with
   distinct and Case"). Both the triple extraction and the judge say SAME --
   correctly: these are sub-intent REFINEMENTS of one intent (hierarchy
   expansion), not different intents. I had labeled refinements as intent
   changes; the gate surfaced the mislabel before it became a conclusion.
   Consequence: the capability is NOT shown absent; the instrument's SENS
   number is uninterpretable until the construction is rebuilt.
2. **SPO-STRUCT's INV=0 is a REAL failure and it is diagnostic.** Triple
   extraction is good (the triples are sensible) but the MATCH RULE
   (attitude overlap >= 0.5 AND object jaccard >= 0.3) is string-overlap on
   the object field -- the instrument as implemented is a LEXICAL instrument
   wearing a structured hat, so it inherits the lexical ceiling (LEX-HARD
   0.25). Lesson with a name: extraction is not detection; the comparison
   OPERATOR is where the instrument's power lives.
3. **BINARY same/different is the wrong vocabulary for this domain.**
   Refinements are neither "same" nor "different": they are near-relations
   (sub-intent). This is precisely the user's follow-up question -- how
   SIMILAR is one intent to another -- and the honest next instrument is
   graded, not binary.

Next instrument (pre-registered when run): a graded ladder -- identical >
paraphrase > sub-intent (goal<->plan, refinements) > sibling (same repo) >
unrelated (different repos) -- plus REVERSAL pairs (same file, opposite
intent), with the decisive readout LOW-where-surface-is-HIGH and rank
correlation across the ladder. Nothing above may be quoted as an instrument
verdict until that runs.

## 19. Addendum 14 — the comparator's capacity, measured (2026-10-17)

The comparator was the bottleneck, not the data (measured: all instruments
called OPPOSITE intents similar -- lex 0.471 / emb 0.906 / struct-v1 0.667).
So the comparator was fixed first, against a constructed capacity test with
known answers (42 pairs, 7 rungs; script comparator_capacity_test.py,
pre-registered 86c7c78; NO mined corpora, no personal text).

THE FIX (schema extension, not more lexicon): the object slot carries a
STATE QUALIFIER (ARCHIVED vs CURRENT, read by fixed rules), plus opposition
rules for state / antonym relations / negation mismatch, plus a SEMANTIC
entity fallback (object-phrase embedding cosine >= 0.60) because paraphrases
share no tokens, plus a completed predicate-class table (the earlier antonym
list had named classes that were never defined -- a bug).

| rung (want) | lexical | embeddings | v1 | v4 |
|---|---|---|---|---|
| same (>=0.90) | 1.00 | 1.00 | 1.00 | **1.00** |
| paraphrase (>=0.60) | 0.02 | 0.77 | 0.33 | **0.75** |
| refinement (>=0.60) | 0.51 | 0.83 | 0.70 | **0.96** |
| unrelated (<=0.35) | 0.00 | 0.54 | 0.08 | **0.375 FAIL** |
| reversal-antonym (<=0.35) | 0.61 | 0.91 | 0.33 | **0.125** |
| reversal-state (<=0.35) | 0.47 | 0.91 | 0.67 | **0.000** |
| negation (<=0.40) | 0.80 | 0.94 | 0.75 | **0.000** |

9 of 10 checks pass. The decisive reversal (same memory, archived-vs-current)
goes 0.667 -> 0.000 while paraphrase recognition RISES 0.333 -> 0.75 -- the
first time a structural comparator separates "same intent reworded" from
"opposite aim, same words". The single failure is the recorded COST of the
semantic entity fallback: some unrelated pairs (0.375 vs the 0.35 bar) now
count as entity-matched. No per-case tuning was done; raising tau would be
tuning on these fixtures, so it is left for the arbiter.

ARBITER AND CAVEAT (state it plainly): v4 was designed after seeing these
fixtures, so this test can no longer decide the comparator's fate. The
arbiter is the INTEGRATION test on LIFTED data -- real intent text with
mechanical outcomes: CodeTraceBench's trajectories carry the real issue/PR
description in messages[1] (the capture mapper had discarded it in favour of
the template goal) plus plans, per-step THOUGHTS, verifier outcomes and the
154-file shared-unit structure. If v4 passes the fixtures but fails the
lifted data, the fixture work was the fixed-point trap and the record will
say so.

## 20. Addendum 15 — the non-circular test: the structural comparator's ceiling, and the judge's clean win (2026-10-18)

The capacity test in addendum 14 was authored by the same hand as the
comparator and iterated until it passed — weak evidence, and its fixtures
STATED their opposition in the marker words the rules detect. The
non-circular version: 10 opposed + 5 same-intent pairs authored by the USER
(untracked file; content never committed), comparator FROZEN before seeing
them (sha256 recorded in the results), one run, no iteration.

| instrument | opposed pairs (want low) | same-intent pairs (want high) |
|---|---|---|
| lexical | 0.577 (fooled -- set has teeth) | 0.241 |
| embeddings | 0.911 (maximally fooled) | 0.821 |
| structural v4 (frozen) | **0.820 FAIL** | 0.850 |
| **LLM judge (separate diagnostic)** | **0.000** | **1.000** |

Two findings, both durable:

1. **The structural comparator's ceiling is now measured**: it handles
   paraphrase (0.85) and catches opposition only when the opposition is
   stated in words or verb classes -- it scores UNMARKED opposition at 0.82,
   i.e. as similar. Nine of the ten user-authored opposed pairs scored
   0.75-1.0. The pre-registered prediction was correct, and no iteration is
   permitted on this result.
2. **The semantic judge separated the same pairs perfectly** (0.0 on all ten
   opposed, 1.0 on all five same-intent). Note the judge's earlier "failure"
   in the detection smoke was a MISLABEL (the SENS pairs were refiments it
   correctly called same); on the first clean reversal test it wins outright.

CAVEATS, stated: n=15, single author, single model, single prompt; pairs are
correlated by authorship; the judge is a similarity engine by construction
(icm-reality-oracle-reframe) and costs a cloud call per comparison.

**THE ARCHITECTURAL CONSEQUENCE** (this is what the result buys): intent
COMPARISON can run offline at index-build time -- the judge canonicalises
and detects oppositions between intents when writing the store -- while
QUERY time stays deterministic, local and cheap over the resulting intent
key space. That is the plan's "intelligence lives in the front-end, the
store stays dumb", implemented as a judge-built key space rather than a
learned lens; and it is the first such implementation our non-circular test
supports.

NEXT (in order): (a) scale the unmarked-opposition test (more pairs,
second author/model) before building on it; (b) integration test on LIFTED
data -- real intent texts (messages[1] issue descriptions, plans, step
THOUGHTS) + verifier outcomes + the 154-file shared-unit structure -- with
the judge building the key space and the structural index answering.

## 21. Addendum 16 — the judge route replicated across authors (2026-10-18)

Scale-up of addendum 15 with author != comparator != judge: three generator
models (glm-5.3-flash:cloud, minimax-m3:cloud, qwen3:8b local) each wrote
10 opposed + 5 same-intent pairs under a fixed authoring prompt (same
artifact, shared wording, no marker words, explicit shared_memory and
why_opposite); plus the user's 15. One frozen scoring pass, 60 pairs,
comparator and judge unchanged.

| author | n | unmarked | judge_opp (want low) | judge_same (want high) | v4_opp | lex_opp (fooled) |
|---|---|---|---|---|---|---|
| user | 15 | 13 | 0.10 | 1.00 | 0.82 | 0.58 |
| glm-5.3-flash:cloud | 15 | 15 | 0.00 | 1.00 | 0.83 | 0.68 |
| minimax-m3:cloud | 15 | 13 | 0.00 | 0.80 | 0.82 | 0.81 |
| qwen3:8b (local) | 15 | 12 | 0.20 | 0.80 | 0.72 | 0.52 |
| POOLED | 60 | | **0.075** | **0.90** | **0.796** | 0.646 |

Checks: judge_opposite_low PASS, judge_same_high PASS, two_of_three_authors
PASS (all three), v4_failure_replicates PASS. Embeddings (reported): 0.922
opposed / 0.842 same -- fooled on everything, as always.

READING: (1) the judge generalises across authors (0.00-0.20 on opposed,
0.80-1.00 on same-intent); (2) the structural comparator's ceiling
replicates on every author's unmarked pairs (0.72-0.83 on opposed) while it
remains adequate on sameness (0.81); (3) the pair sets have teeth (lexical
0.52-0.81 on opposed, 0.22 on same -- no constant answer passes).

CAVEATS: judge is a cloud model (no local equivalent measured yet); 45/60
pairs are model-authored, so "matches real work distribution" is not
established; same-intent rungs are small (n=5/author) with two authors at
0.80 (1 of 5 wrong each). These bound the claim to: OPPOSITION COMPARISON
under the stated authoring conditions.

DIVISION OF LABOR this establishes: semantics (judge) is needed for
opposition/canonicalisation; structure remains adequate for sameness,
refinement and the query-time index. That is the architecture: judge-built
intent key space at write time + deterministic local index at query time.

NEXT: the lifted-data integration test (CodeTraceBench messages[1] issue
texts + plans + step THOUGHTS + verifier outcomes + the 154 shared file
units) -- judge canonicalises intents and flags oppositions on the SAME
shared files; the structural index answers; outcomes judge the result.

## 22. Addendum 17 — the LIFT: real intents from raw trajectories (2026-10-18)

The capture mapper had been recording the TEMPLATE goal
(`resolve {task} ({agent})`) and discarding the real intent, which lives in
the trajectory: the benchmark's own issue description. The lift now parses
ALL THREE agent layouts in trajs3/ and joins verifier outcomes from the
manifest by artifact basename (script lift_codetrace_intents.py, 5s for 489
tars; corpus untracked).

| agent | runs | real issue text | declared plan | step THOUGHTS | outcome |
|---|---|---|---|---|---|
| miniswe | 68 | 68 | 0 (none declared) | 68 | 68 |
| openhands | 259 | 259 | **256** | 0 (none emitted) | 259 |
| sweagent | 108 | 90 | 0 (none declared) | 107 | 108 |
| **total** | **435** | **417 (96%)** | 256 | 175 | **435 (100%)** |

Corpus: 16,830 steps (vs 11,963 in the template-goal capture),
**3,759 distinct files, 543 shared-file units** (vs 154), outcomes
319 solved / 116 unsolved (mechanical), issue texts median 1,296 chars.
54 tars (11%) failed to parse -- counted, not hidden.

GAPS SURFACED: openhands emits no THOUGHT blocks (reasoning is in tool-call
arguments); miniswe/sweagent declare no plan; sweagent issue text missing for
18 runs; 54 parse failures (mixed/odd layouts). These bound what each grain
can be trained or tested on.

WHAT THIS ENABLES: the integration test the program has been unable to run --
real intent statements (issue texts) against mechanical outcomes over a
shared-file structure 3.5x larger than the flip table. Next: the judge builds
the intent keyspace over the 417 real intents (canonicalise, flag
oppositions among the 543 shared-file units); the structural index answers;
the verifier outcomes judge the result.

## 23. Addendum 18 — per-file usefulness labels from the accepted patch (2026-10-18)

The integration test needs a usefulness signal at the memory-unit grain. This
lift derives it mechanically: a file is USEFUL for a run when it appears in
the run's final patch AND the verifier accepted the run; ATTEMPTED when it
appears in the patch but the run failed; CONTEXT when touched without being
changed. Patches come from the tars (patch.diff / .patch / .pred /
SWE-agent info.submission); paths are normalised on both sides.

| label | count |
|---|---|
| useful | 2,380 |
| attempted | 239 |
| context | 3,734 |
| **records** | **6,353 (run,file)** |

Coverage: 295/435 runs carry a patch (openhands 198/259, sweagent 83/108,
miniswe 14/68 -- a counted GAP, likely a member-naming difference). 5,182
distinct files; 562 multi-run units; **98 units carry BOTH useful and
non-useful labels** -- the outcome-flip evaluation set. A file-path regex bug
was fixed first (".c" matching the start of ".contrib" had produced junk
units; shared-file count corrected 543 -> 430 pre-normalisation).

CAVEATS recorded: diffs sometimes include environment files (package.json,
pyproject.toml appear as "useful" patch files in some runs); miniswe's low
patch coverage limits that agent's weight; 98 flip units is a small first
pass, adequate for a paired readout but not for a confident single number.

WHAT REMAINS for the integration test: (1) the judge keyspace pass over the
417 real intents (~600 calls: SPO extraction + canonicalisation + relation
labels among the flip units); (2) the harness -- task-disjoint split, arms =
dense (issue -> file text proxy) vs structural keys vs hybrid, primary
metric recall@k / MRR for the patch files of held-out runs, secondary the
outcome-flip readout on the 98 units.

## 24. Addendum 19 — the INTEGRATION TEST ran: the single-key keyspace does not transfer (2026-10-18)

Pre-registered harness (integration_test.py) on the lifted corpus: task-disjoint
split (344 tasks -> 68 held out; 338 train runs / 79 test runs); store built
from train runs only (4,562 candidate files); 28 held-out runs with usable
ground truth (their accepted-patch files present in the pool); query = the
run's REAL issue text; ground truth = its USEFUL (accepted-patch) files.

| arm | recall@10 | recall@50 | MRR |
|---|---|---|---|
| DENSE (issue -> file text proxy) | **0.179** | **0.339** | **0.092** |
| LEXICAL | 0.179 | 0.321 | 0.061 |
| STRUCT (judge SPO keys) | **0.000** | **0.000** | 0.001 |
| HYBRID (RRF) | 0.071 | 0.286 | 0.036 |

ALL pre-registered checks FAIL. Chance recall@10 for ~2 useful files among
4,562 is about 0.004 -- so DENSE at 0.179 is ~45x chance and clearly the only
working signal; the judge-built single-key keyspace is indistinguishable from
noise, and fusing it into dense DEGRADES dense (0.179 -> 0.071).

DIAGNOSIS (named, not hand-waved):
1. ONE key per intent is too coarse for real issue texts. The addendum-15/16
   tests used 6-20-word sentences where a single SPO triple captures the
   intent; real issues are ~1,300 chars carrying several requirements, and
   one triple throws that away -- key_sim barely varies across files.
2. The relation-labelling prompt collapsed on long texts: 94/96 pairs called
   DIFFERENT, 0 OPPOSED (vs 2 SAME), so the secondary keyspace-vs-judge check
   could not run. A file-centric question ("would the changes these two tasks
   require to THIS file conflict?") is the obvious reformulation.
3. The keyspace bar in intent_keyspace.py used the wrong denominator
   (562 multi-run units instead of the ~98 outcome-flip units): the honest
   reading is 96/98 flip units labelled = 98 percent, but the pre-registered
   number (17 percent) is what the record shows; the bar was mis-specified.
   Also fixed en route: a harness ranking bug that scored positions over the
   alphabetical file list instead of the ranked order (made every arm look
   like chance at recall 0.0).

WHAT SURVIVES / WHAT DOES NOT: dense text similarity is the working retrieval
signal on real intents (45x chance); the judge-built keyspace as ONE triple
per intent does not transfer. The addenda 15/16 result stands only for short
intent sentences under those authoring conditions. The next honest iteration,
if attempted, is multi-requirement keys (3-5 SPO triples per issue) + a
file-centric relation prompt -- a NEW pre-registered test, not a re-score.

## 25. Addendum 20 — v2 keyspace (multi-requirement) still loses to dense; the opposition test has no bed (2026-10-18)

Keyspace v2: 417/417 intents keyed, mean **4.63 requirements per issue**
(median 5, range 3-6), keys visibly decomposing real issues (e.g. four
separate requirements from one FITS-card bug). Harness unchanged in every
other respect (same split, pool, ground truth).

| arm | recall@10 | recall@50 | MRR |
|---|---|---|---|
| DENSE | **0.179** | 0.339 | **0.092** |
| LEXICAL | 0.179 | 0.321 | 0.061 |
| STRUCT (multi-keys) | 0.000 | 0.089 | 0.009 |
| HYBRID | 0.107 | **0.429** | 0.053 |

ALL pre-registered checks FAIL again. READINGS:
1. Multi-requirement keys clearly help the TAIL: struct goes 0.000 -> 0.089
   @50; hybrid @50 0.286 -> 0.429, now ABOVE dense (0.339). But at the HEAD
   that matters (recall@10, MRR) dense still wins. The keyspace carries real
   but weak signal relative to dense at this grain.
2. The file-centric relation prompt did NOT "collapse" -- it agreed with the
   corpus: 95/96 shared-file task pairs are DIFFERENT purposes, 1 SAME, 0
   OPPOSED. That is almost certainly TRUE, not an instrument failure: two
   different tasks sharing a file in a repo have different purposes. **The
   corpus contains no natural opposed-intent pairs at all** (consistent with
   the Onyx census 0/6336 and the absence of same-task-two-intent runs), so
   the keyspace's opposition capability from addenda 15/16 has NO test bed
   here. The relation-gap check was mis-specified against this corpus.

CUMULATIVE STATE OF THE RETRIEVAL QUESTION: dense text similarity is the
working retrieval mechanism (45x chance on real intents); a judge-built
intent keyspace -- one triple or five -- does not beat it at the head of
ranking, though multi-keys add tail signal. The judge's intent COMPARISON
capability stands only on constructed short-sentence pairs and is untested
where it matters (natural opposed intents on a shared memory).

WHAT WOULD CHANGE THIS (named, not promised): a corpus with the same memory
under genuinely opposed intents with mechanical outcomes -- i.e. production
usage of the store, or a purpose-built harness in the 2x2-fixture family at
scale. Nothing in the current public corpora supplies it.

## 26. Addendum 21 — the scaled 2x2 harness: the construction failed its own vacuity guard (2026-10-18)

40 filesystem-verified trials (4 artifact types x 5 scenarios x 2 intents; the
answer is READ FROM THE CHOSEN FILE and compared to the computed value).

| arm | accuracy |
|---|---|
| DENSE | **0.850** |
| LEXICAL | 0.875 |
| KEYSPACE | 0.500 |
| HYBRID | 0.500 (artifact -- see below) |

ALL checks fail, and the FIRST failure is the one that matters: **the vacuity
guard** (dense must sit near chance for the construction to prove anything).
At 0.85 dense simply reads the state words ("Recorded at the original
ingest" vs "Appended one row") -- the fixed-point theorem again: the authored
world carried a discriminator I did not intend. This harness therefore says
NOTHING about intent conditioning; it must be redesigned with a harder
construction (state NOT stated in words: inferable only from chronology or an
external log).

The keyspace arm (0.500 = chance) is also not a fair test yet -- three named
implementation gaps found by inspecting the cached keys:
  1. the memory-side "object" is the VALUE (checksum digits), so object
     overlap with the intent's object ("dataset alpha") is zero BY
     CONSTRUCTION; memory matching must run on relation/class + epoch, not
     object;
  2. "report" is not in the predicate-class table, so class match is 0 for
     every trial;
  3. qualifier opposition is decided by marker words, and the intent's
     qualifier ("saved at the first ingest") contains none of them -- the
     OLD/NEW comparison needs a SEMANTIC test (the addendum-15 finding:
     the judge separates epochs, rules do not).
The hybrid's 0.500 is a tie artifact: RRF with two candidates produces ties
when the arms disagree and argmax then always picks index 0 -- needs a
tie-break, not a conclusion.

STATE: the retrieval question now has a measured answer (dense works, 45x
chance on real intents), an untested opposition claim (judge separates short
constructed pairs; no natural corpus carries opposed intents), and a
2x2 construction that is too easy for its own guard. The next honest move is
a HARDER construction, not another score on this one.

## 27. Addendum 22 — the HARD construction: intent interpretation beats chance similarity 40/40 (2026-10-18)

The first 2x2 failed its vacuity guard because memory records stated their own
state. The hard version removes the tell: records differ ONLY in value, run
number and date; intents refer to the epoch relatively ("at the first load",
"at the most recent load") using none of the records' date words; ordering is
mechanical by date. 40 trials, answers read from the chosen file.

| arm | accuracy |
|---|---|
| DENSE | 0.575 (vacuity guard PASSED: near chance) |
| LEXICAL | 0.500 |
| **SEMANTIC (judge EARLIEST/LATEST -> date ordering)** | **1.000 (40/40)** |
| HYBRID | 0.575 |

Checks: vacuity PASS, semantic >= 0.85 PASS, semantic >= dense + 0.25 PASS,
hybrid >= dense + 0.25 FAIL. Epoch judgments: 20 EARLIEST / 20 LATEST, 0
abstentions.

WHAT THIS ESTABLISHES (first fair construction of the arc): when the epoch is
NOT written on the surface, plain similarity is at chance (0.575 / 0.500)
while a mechanism that INTERPRETS the intent's temporal reference and applies
mechanical ordering decides every trial correctly. That is intent
conditioning doing real work -- and it required exactly the architecture
addenda 15/16 pointed at: semantics at build time + deterministic structure at
match time.

THE HYBRID FAILURE IS A DESIGN FLAW, RECORDED AS SUCH: equal weights
(0.5 dense + 0.5 semantic) with a dense tie-break reproduce DENSE wherever the
arms disagree, so the fused arm inherits 0.575. Re-weighting is a NEW
pre-registered variant, not a rescore of this run.

CAVEATS: n=40; constructed world (the date is a surface feature, though not
one the intent's words carry); one judge model; the epoch question is a 2-way
classification, which language models do easily. The claim is bounded:
"the judge can read an intent's temporal reference and mechanical ordering
turns that into correct memory selection where text similarity is at chance."
Scaling the construction (more artifact types, more than two epochs,
non-temporal oppositions) is the next test; the immediate candidate for
retrieval is a two-stage design -- dense candidate generation, then
epoch/opposition re-ranking -- which is exactly what the naive hybrid failed
to approximate.

## 28. Addendum 23 — the scaled two-stage retriever PASSES (2026-10-18)

60 filesystem-verified trials (4 artifact types x 5 datasets x 3 epochs;
chance = 1/3); pools = the 15 same-type records + 10 seeded decoys (25
candidates). Records differ only in value/run/date -- no state words; intents
refer to the epoch relatively; the judge classifies each intent's epoch
(12 template calls, all correct) and DATE ORDER is mechanical.

| arm | accuracy |
|---|---|
| DENSE@1 (top-1 over 25 candidates) | 0.433 (chance 0.333 -- vacuity PASS) |
| SEMANTIC-ONLY (judge epoch + entity/type filter + date rank) | **1.000** |
| **TWO-STAGE (dense top-5 -> filter -> epoch rank)** | **1.000** |
| HYBRID (semantic primary, dense fallback) | **1.000** |
| STAGE-1 recall@5 | **1.000** |

ALL checks PASS. The architecture measured across addenda 15/16/22/23 is
coherent and now demonstrated at this scale:
  build time   : judge reads the intent (epoch/opposition);
  match time   : deterministic entity/type filtering + date ordering;
  stage 1      : dense generates candidates (recall@5 = 1.0 here);
  stage 2      : intent re-ranks -- 1.000 where dense@1 is at chance.
The addendum-22 failure (score blending) is explained by this design: fusing
scores at one level cancels; filtering and re-ranking do not.

CAVEATS (unchanged): constructed world; 60 trials; the epoch axis is temporal
and the judge's 3-way classification is easy for language models; the date is
a surface feature not carried by the intent's words. What has NOT transferred
is the real corpus -- CodeTraceBench has no epochs or opposed intents per
memory (addendum 20), so the real-data side of this architecture still has no
test bed. The honest status: the MECHANISM is measured; its REAL-WORLD
APPLICABILITY awaits a corpus with per-memory intent variation (production
usage of the store, or a multi-run-per-task corpus).

## 29. Addendum 24 — the store build: schema, write-time keys, two-stage query, live seam (2026-10-18)

The spec's build items 1-4 are implemented and item 5 is seamed. No new
research claim is made here; the only measured number is the regression.

WHAT WAS BUILT (all default OFF; no runtime behaviour changed):
1. **Capture schema v2** (`icm/capture.py`). Every goal/step now records
   `qualifier`, `requirements` (an S-P-O LIST -- measured 4.63 per real issue;
   a single triple is accepted and never padded), `epoch` (declared run
   number/timestamp, else capture time with `epoch_source` saying so), and an
   expansion link `triggered_by`. The link is enforced: it must reference an
   outcome whose rule-computed label is `violated` -- a confirmed outcome
   cannot spawn a new intent (the causal-direction law). Retrieval events now
   carry the operative intent id, its key, epoch and qualifier beside the
   memory-unit ids. A prediction-free `intent()` record exists for live work;
   it can never produce a congruence label (`outcome()` refuses it), so the
   label law is unchanged. Outcome ids gained a uuid suffix (the old
   len()-only id collided across captures sharing a ledger).
2. **Write-time judge keying** (`icm/intent_keys.py`). One call per unique
   intent: requirement LIST + qualifier + `epoch_ref` {kind: ordinal/absolute/
   none, value}. Cache key = prompt version + model + declared qualifier +
   normalised text, so a model switch does not serve the old model's keys.
   Four explicit statuses (judged / cached / abstained / error); failure
   returns no key and never blocks the write. Pairwise SAME/DIFFERENT/OPPOSED
   is available as `relation_between()` and is deliberately NOT run on every
   write -- the cost-per-write / epochs-per-memory question is still open.
3. **Two-stage query path** (`icm/intent_query.py`). Dense cosine generates;
   a type/entity filter narrows; the key's epoch reference re-ranks (ordinal
   EARLIEST/MIDDLE/LATEST, or absolute date/run by distance). NO score
   blending. Every fallback is explicit and returned: `no_key`,
   `empty_filter`, `no_epoch` (including an epoch-domain mismatch, e.g. a date
   key against a pool of run numbers). Missing epochs never rank first.
4. **Live seam** (`icm/runtime_capture.py`). `SessionCapture` wraps an
   orchestrator's `query()` and records the user prompt (declared intent), the
   episode ids actually retrieved, the epoch and the write-time key. A chat
   answer is NOT a mechanical outcome, so turns are recorded as
   `outcome: unmeasured` and no congruence label is computed. Capture failure
   never loses the answer. `variation_index()` is the census: memory id ->
   the distinct intents it was retrieved under. Wiring is an opt-in
   `--icm-capture` flag on `scripts/serve_ponder.py` (default OFF; writes to
   the untracked `data/_scratch/icm/live_ledger`; `--icm-capture-no-keyer`
   records intent + memory links without a judge call).

REGRESSION (the harness is unchanged; `git diff` on twobytwo_stage2.py is
empty). Re-run 2026-10-18: 60 trials, chance 1/3.

| arm | accuracy |
|---|---|
| DENSE@1 | 0.433 (vacuity PASS) |
| SEMANTIC-ONLY | 1.000 |
| **TWO-STAGE** | **1.000** |
| HYBRID | 1.000 |
| STAGE-1 recall@5 | 1.000 |

All four pre-registered checks PASS -- byte-for-byte the addendum-23 result.
The build therefore did not disturb the measured mechanism.

TESTS: `python -m pytest tests/ -q -k icm` -> 84 passed (35 new tests across
the four modules; no network in tests -- transports and keyers are injected).

WHAT THIS DOES AND DOES NOT CHANGE: the store can now RECORD the six-field
shape and ANSWER with the measured two-stage mechanism. It has not yet been
run behind real work, so `variation_index()` has no real-session data and the
wall stands. The next honest measurement is the first real session capture;
no authored world substitutes for it. Open questions unchanged: judge cost per
write vs epochs per memory; entity filtering on messy real entities; the
abstention fallback (now visible as `fallback` rather than assumed).

## 30. Addendum 25 — the personal-corpus census: the flip shape is not there (2026-10-18)

User directive: re-check the Claude sessions and Onyx, extracting intent rather
than assuming the old mining's goals. Two lifts were built (pre-registered
docstrings; census only, no training, no retrieval claim):

**Claude sessions** (`scripts/icm/lift_claude_sessions.py`). 44 transcripts ->
3,716 turns (3,407 substantive), 33,632 tool events. Intent = the user's
directive at the turn start (271 terse directives lifted with the agent's own
first plan sentence, marked `user+agent-plan`); memory unit = the file paths in
Read/Edit/Write; outcome = the tool result (`is_error`, stdout/stderr) and the
structured patch, mechanical. Result: 1,991 distinct edited files, **407
shared edited files, 1,141 turn pairs**; per-(turn,file) labels useful 3,120 /
context 3,081 / attempted 227. Only **33 turns ran a verification command**
(all 33 passed), so external verification is thin.

**Turn-level judge sample** (120 seeded pairs, addendum-16 prompt):

| relation | n |
|---|---|
| same | 64 |
| different | 55 |
| **opposed** | **1** (0.8 %, Wilson 95 % CI 0.15-4.6 %) |

Stratified, the single opposed pair is a project-relative verdict file; memory
notes 0/23, absolute paths 0/31.

**Session-goal grain** (`scripts/icm/census_session_goals.py`; first
substantive edit turn per session/file, sessions paired over shared files):
66 shared units, 254 pairs, sample 80 -> different 66 / same 14 / **opposed 0**.

**Onyx shared documents** (`scripts/icm/census_onyx_docs.py`; two queries that
retrieved the same document): 510 shared docs, 2,457 pairs, sample 80 ->
different 76 / same 4 / **opposed 0**.

**POOLED: 1 opposed in 280 sampled pairs = 0.36 % (95 % CI 0.06-2.0 %).**

READING: these corpora are task work. The same file or document recurs under
DIFFERENT aims -- plenty of variation -- but essentially never under OPPOSED
aims. The 2x2 mechanism needs the swap (one memory right under A, wrong under
B); "different" cannot supply it, and a retrieval run on different-aim pairs
would only re-measure dense candidate generation, which addenda 17-20 already
settled. This replicates CodeTraceBench (0/96) and the prior Onyx census
(0/6336) with the new judge and two new grains.

CAVEATS: the intent is the user's directive, which may under-specify the aim
(the agent-plan lift covers terse directives only); the judge is the measured
instrument but reads long, context-dependent directives; the samples are seeded
and not exhaustive; and the mechanical verification is thin (33 test turns),
so even a discovered flip unit would usually lack a per-memory outcome.

WHAT THIS DOES NOT CHANGE: the mechanism (addenda 22-23) stands, and the data
wall stands. More sessions of the same kind do not fix a shape problem -- the
next honest source is a task family that actually produces counterfactual
pairs (revert/rollback, flag on/off, archive-vs-refresh, hypothesis-vs-refute)
with a mechanical check, captured as it happens.

## 31. Addendum 26 — git history IS a counterfactual producer: ~159 patch-verified inverse pairs (2026-10-19)

The personal-corpus census (addendum 25) found no opposed aims. The next place
to look was the one we already own: the git history of the local repos. A
commit pair over the SAME file whose patches invert each other is a real
counterfactual pair -- real memory units (file snapshots), real oracle (the
blob content), and real intent text (the commit messages).

CENSUS (`scripts/icm/census_git_inverses.py`, pre-registered; message-level
candidates capped at 8/file, sampled 120 seeded pairs, patches compared with
`git show --unified=0`; inverse = later commit removes >= max(2, 50 %) of the
lines the earlier added, or the reverse):

| repo | files | msg-level candidates |
|---|---|---|
| Pondr | 904 | 5 |
| HPMT | 1,234 | 31 |
| liboffs | 695 | 418 |
| WaveDB | 621 | 453 |
| **total (capped)** | | **907** |

Patch verification on the 120-pair sample: **21 true inverses = 17.5 %**
(Wilson 95 % CI 11.7-25.3 %), 9 in liboffs and 12 in WaveDB. Estimated
**~159 inverse pairs** in the capped pool (uncapped message-level opposite
pairs were ~3,044 before the per-file cap). That is the order needed for a
first powered run (~159 memories, one adversarial trial each).

WHY THIS MATTERS: this is not an authored world. The memory units are real
file versions, the opposition is verified against the actual patches, and the
label is the file content at the expected commit -- no judge. The only authored
part is the QUERY PHRASING that turns a commit message into a retrieval
request ("restore the state before this change" / "select the state this
change produced"), which must be pre-registered and must pass the vacuity
guard: if dense similarity reads the feature words and wins, the construction
is vacuous and is discarded (the addendum-21 failure mode).

FAIRNESS NOTE: commit messages STATE their change ("add tracing", "remove
tracing"). That is a surface tell in the INTENT, which is allowed -- the 2x2
also states "the first load"; the test is whether the MEMORY records state
their state. File snapshots do not. The adversarial direction is the removal /
revert: the message mentions the feature, the wrong snapshot contains it, and
the intent key must select the version without it.

CAVEATS: the inverse rule is a heuristic on line overlap (reformatting and
moves can produce false positives); pairs are concentrated in two repos; the
query layer is authored; and a first run must report the adversarial direction
separately from any easy direction. This does not settle the strong thesis by
itself, but it is the first organic counterfactual corpus large enough to try.

PUBLIC-REPO PILOT (2026-10-19, same rule): eight public non-fork repos
(CRABS, SecretAgent, poll-dancer, EAuth, poseidon, poseidon-crdt,
liboffs_pony, Tangerine-Id-Migrator) were cloned and censused with the same
script. 178 message-level candidates; seeded 160-pair sample -> **11 verified
= 6.9 % (95 % CI 3.9-11.9 %)**, estimated **~12 pairs** (CRABS 4, EAuth 4,
poseidon 3). The public repos are mostly smaller histories with fewer
add/remove cycles; they add domain diversity but not much volume. POOLED
available corpus: **~171 verified pairs** (local liboffs + WaveDB ~159, public
pilot ~12) -- enough for a first powered run. The corresponding Claude
sessions are NOT needed for this test: they were censused at 1/280 opposed
(addendum 25) and add intent text, not counterfactual pairs.
