# A Taxonomy of Retrieval Failure Axes — and the Cheap Legs That Fill Them

**The negative-metascience result of the Intent-Conditioned Memory (ICM) arc.**
2026-10-14. Eight pre-registered demonstrations; one 4-axis taxonomy; the strong
thesis retired by its own protocol.

## The strong thesis, and how it died

Thesis posed at the start: *relevance is irreducible to cheap signals — a
learned, intent-conditioned lens is required to beat raw dense similarity where
it matters.* Eight pre-registered demonstrations each ended the same way:
whenever a measurable gap separated dense similarity from the payoff-optimal
ranking, a **compact, cheaply-expressible rule** closed it. Not once did the
gap require learning.

The demonstrations, in order:

| # | World | Compact discriminator that closed the gap |
|---|-------|-------------------------------------------|
| 1 | B0 label-fork audit | payoff records did not exist; none needed |
| 2 | B1 v1 | untrained cosine beat head on goal-fit/sibling 20/20 |
| 3 | B2 adversarial fixtures | lexical keyword scorer tied the trained head |
| 4 | B3 weak-thesis run | fixed-point theorem: authored worlds carry compact-rule discriminators |
| 5 | 2×2 reality oracle | the pre-check cheap legs failed goal-A; mechanism split = template-local artifact |
| 6 | labels-recovery | judge labels cosine-recoverable (Spearman 0.8125 at 40-q) |
| 7 | 2×2 counterfire | token collision + template geometry; head training canceled |
| 8 | **organic tail payoff A/B** | trivial time-token overlap beat cosine **6/0/19, p=0.03125**, judge-free |

Demo #8 deserves its own note because the oracle finally was mechanical (graded
answers, no LLM judge), and cosine **still** never won (cos-top-1 payoff 0.160
≈ guess-floor vs time-rule-top-1 0.400, gold ceiling 0.680). The robe changed;
the conclusion did not.

## The taxonomy (the actual deliverable)

Dense cosine fails along identifiable axes. Each axis, measured, with its
filling leg:

| Axis | Failure mode | Measured evidence | Correct leg |
|------|--------------|-------------------|-------------|
| **Values / identity** | content-twins and value-confusers score identically under cosine | dense ~0 on value twins; dense+filter 34%; lexical-first 43/43 | lexical anchor (exact keys never into dense MLPs) |
| **Goal-conditioning** | cosine ranks by surface match, points the wrong way under an adversarial goal | trained dense head 12/12 vs cosine 0/12 on the 2×2's goal-A | trained dense head (the lens's one real job) |
| **Temporal / relational** | questions like "first", "day before", "total spent" name a relational query, not a similarity target | time-rule-top1 0.400 vs cos-top1 0.160 (p=0.031), gold 0.680 | temporal index + reasoner (timestamp arithmetic, first/sum/day-before joins) |
| **Storage / decay** | stale memory outranks the current state and induces harm | 2×2 goal-B: 7/12 stale-trust = the harm path | forgetting ledger / freshness marks |

**The hybrid system this implies:** a retrieval router that (a) surfaces by
dense cosine, (b) anchors symbolic values lexically, (c) routes temporal/
relational questions to a structured index, (d) routes goal-bearing queries to
a trained head, and (e) keeps a freshness/forgetting ledger on storage. The
dense head has a real, bounded, measured job in it — goal-conditioned ranking,
nothing more.

## Protocol laws discovered along the way

The arc's second product. Each one caught a real failure mode:

1. **The oracle theorem** — every available oracle (n-gram, keyword, LLM judge)
   is a similarity engine by construction. Judge labels were cosine-recoverable
   at 0.81; report the *distribution*, never a single mean; the tail is the
   finding.
2. **Judge-free payoff** is achievable: grade answers against ground truth. But
   the *grader* becomes part of the measurement (n-gram strictness on long
   golds), and the **answerer is a co-bottleneck**: a perfect memory scored
   only 0.680/1.0 — ceiling caps every downstream claim.
3. **The none-arm check** — always measure the model's default answer when it
   cannot know (guess-floor ≠ 0.5; the default is often "mismatch"/"don't
   know").
4. **Pre-check before any training counts**: raw cosine must fail to separate
   the payoff winner first; the goal string must not name the memory.
5. **Ranks pay through selection, not permutation** — an A/B that shows both
   arms the same candidate set in different order ties 25/25. Vary the
   *selected set* (k=1 or a contested budget).
6. **The reverse-order lesson**: train/test capacity gates the test — LOO on 24
   samples is a variance machine, not a verdict. Harvest to n≥50–75 with a
   real split before any head verdict.
7. **Authored worlds are closed**: synthetic benchmarks always carry compact
   discriminators; only organic corpora test irreducibility seriously.
8. **Artifacts get run-scoped names**; smoke runs never write to a real run's
   output path.

## What is honestly left open

One experiment: **temporal index + reasoner vs time-rule** on the harvested
tail — how much of the 0.400→0.680 residue is indexable structure versus
requires an actual reasoner? The dataset carries session timestamps
(`haystack_dates`) + `question_date`, so the index is constructible. The
critique's pre-registration: the query engine wins.

And one re-gated thesis: whether a *learned* head can beat a *proper* structured
index (not a trivial rule) on a real train/test split of 50–75 verified
reversals. If it cannot — or ties — the strong thesis closes as falsified
organic-wide, and the taxonomy above is the whole answer.

## The provenance

Runs `scripts/icm/` in this repo; LongMemEval organic data (500-record file);
cross-model judge (kimi-k3 judging deepseek-v4.1-flash); WaveDB substrate.
Memory index: `icm-*` files; every number above has a committed script and a
result JSON behind it.