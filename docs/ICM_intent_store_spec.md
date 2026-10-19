# Intent-Conditioned Memory — Store Spec and Next Build

*Date:* 18 October 2026
*Status:* Derived from measured results (ICM_phase_report.md addenda 14–23)
*Audience:* whoever builds the next iteration

---

# 1. What the measurements established

| claim | status | evidence |
|---|---|---|
| Dense text similarity is a good *candidate generator* on real data | measured | CodeTraceBench lift: recall@10 0.179 / MRR 0.092 ≈ 45× chance (addenda 17–19) |
| A learned projection ("lens") does not add relevance signal | measured, repeatedly | 26 battery configs + 10 intent-corpus configs at the untrained baseline (addenda 10–12) |
| Structural comparison of intent catches only *marked* opposition | measured | unmarked opposition 0.82 vs judge 0.00; paraphrase 0.85 kept (addenda 14–15) |
| The judge separates SAME/OPPOSED intent across authors | measured | pooled 0.075 opposed / 0.90 same, 60 pairs, author ≠ comparator ≠ judge (addendum 16) |
| One SPO triple per intent does not transfer to real 1.3k-char issues | measured | struct recall@10 0.000 vs dense 0.179 (addendum 19) |
| Multi-requirement keys help the tail, not the head | measured | hybrid recall@50 0.429 > dense 0.339; recall@10 still below dense (addendum 20) |
| Epoch interpretation + mechanical ordering decides correctly where similarity is at chance | measured on a fair construction | 40/40 (addendum 22); 60/60 two-stage at 25 candidates, chance 1/3 (addendum 23) |
| Score blending cancels; filter + re-rank does not | measured | hybrid 0.575 vs two-stage 1.000 (addenda 22–23) |
| The real corpora we hold have NO per-memory intent variation | measured | CodeTraceBench 95/96 shared-file task pairs merely DIFFERENT, 0 OPPOSED (addendum 20); Onyx 0/6336 |

**The architecture implied by all of it:** dense generates candidates; a
deterministic entity/type filter narrows them; a judge-built intent key
(epoch / requirement / opposition) re-ranks the head. Semantics at build time,
deterministic structure at match time, no learned lens, no score blending.

**The wall:** that architecture can only be exercised where the SAME memory
appears under DIFFERENT intents. No corpus we hold supplies that shape with
mechanical labels. Everything below exists to produce such a corpus and to
make the architecture usable when it exists.

---

# 2. What the store must record (the six-field intent record)

Every memory write carries a record with all six fields; labels are computed
from them, never stated by a caller.

1. **intent** — declared at capture in SPO form by the agent holding it:
   `{subject, predicate_class, relation, object, qualifier}`.
   `qualifier` carries the state/version/time marker (earliest/latest, before/
   after, archived/current, on/off). Multiple requirements per intent are
   allowed and expected (measured: 4.6 per real issue).
2. **outcome** — mechanical and linked: verifier result, filesystem state, test
   result. Never a judgement call.
3. **memory units** — stable ids of what was retrieved or used (file, record,
   episode), recorded at the moment of use.
4. **intent variation** — when a non-reach event occurs, the record of the
   NEW intent that followed (the expansion loop). This is what makes the same
   memory appear under different intents over time.
5. **epoch / ordering** — the timestamp or run number that orders memories
   against each other. The two-stage mechanism ranks by this, so it must be
   recorded, not inferred.
6. **labels** — rule-computed from 1–5 (e.g. useful = in the accepted patch AND
   run solved, as in addendum 18). No miner-stated labels (58 % valence-flip
   audit).

# 3. What NOT to rebuild

- No learned projection / lens (measured dead across 36 configs).
- No score blending of semantic and dense signals (measured to cancel).
- No one-triple-per-intent keys for long texts (measured at noise).
- No reliance on corpora that lack intent variation for retrieval claims
  (LongMemEval-composed is closed; CodeTraceBench serves as the dense-generator
  benchmark only).

# 4. Next build items, in order

1. **Capture layer extension** (small): add `qualifier` and `epoch` to the
   existing capture record; record the *requirement list* per intent
   (3–5 triples) rather than one. `icm/capture.py` already computes labels at
   observe() time; this is a schema addition.
2. **Judge keying pass** at write time: for each new intent, the judge extracts
   requirements + epoch/opposition; store the keys beside the memory unit.
   Cost is one call per intent at write time, not per query.
3. **Query path**: dense top-k → filter by entity/type → re-rank by key match
   (epoch/requirement), per addenda 22–23. No blending.
4. **Evaluation harness**: reuse the filesystem-verified 2×2 design
   (twobytwo_stage2.py) as the regression test for every change; it is fair
   (vacuity guard) and mechanical.
5. **Real-data test bed** (the wall): record real usage of the store so the
   same memory appears under different intents with outcomes — i.e. run the
   store behind actual work (Pondr's own sessions are the first candidate),
   with the capture layer writing fields 1–6. Until then, real-corpus claims
   stop at "dense generates candidates".

**BUILD STATUS (2026-10-18, addendum 24).** Items 1-4 are implemented, default
OFF, with 35 new tests: `icm/capture.py` (schema v2: qualifier, requirement
list, epoch, expansion link, prediction-free intents), `icm/intent_keys.py`
(write-time judge keys with a model-partitioned hash cache and explicit
judged/cached/abstained/error statuses), `icm/intent_query.py` (dense ->
type/entity filter -> epoch re-rank, no blending, explicit fallbacks). The
unchanged 60-trial harness re-ran to the addendum-23 numbers (dense@1 0.433,
two-stage 1.000, stage-1 recall@5 1.000). Item 5 is seamed, not measured:
`icm/runtime_capture.py` wraps a live orchestrator and records declared
intents + retrieved episode ids + epochs + keys, marking every turn
`outcome: unmeasured` because a chat answer is not a mechanical outcome. No
real session has been recorded yet, so the wall stands.

**PERSONAL-CORPUS CENSUS (2026-10-18, addendum 25).** The real-data test bed
was attempted on the corpora we actually hold. Claude sessions: 3,716 turns,
407 shared edited files, 1,141 turn pairs; a seeded judge sample found 1
opposed pair in 120 (0.8 %). Session-goal grain: 0/80. Onyx shared documents:
0/80. Pooled 1/280 = 0.36 % (95 % CI 0.06-2.0 %), replicating CodeTraceBench
0/96 and the prior Onyx census. The corpora carry plenty of DIFFERENT aims over
the same memory and almost no OPPOSED aims; more sessions of the same kind will
not change the shape. The wall is a data-shape problem, not a data-volume one.

# 5. Open questions the spec does not answer

- How many epochs/intents per memory unit are needed before the re-rank pays
  for itself (the judge cost per write)?
- Does the entity/type filter survive messy real entities (aliases, path
  variants)? The 2×2 world's entities are clean by construction.
- What is the failure mode when the judge abstains (UNSPECIFIED)? The current
  answer is "fall back to dense@1" — untested at scale.

---

*All numbers in this spec are traceable to docs/ICM_phase_report.md addenda
14–23 and the untracked artifacts under data/_scratch/icm/ (lifted corpus,
keyspaces, harness results).*
