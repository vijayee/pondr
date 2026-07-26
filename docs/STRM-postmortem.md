# STRM Post-Mortem -- the JST salience path, burned 2026-07-25

> Written for a FRESH session. The conversation that produced this is gone. This
> file is the handoff: what we tried, what failed, what NOT to do again, what to
> do instead, what code is kept, and where everything is preserved.

## 0. How to use this file

- **master** was reset to `61e0e9c` ("docs: consolidation pipeline map"). The four
  STRM-salience commits after it (`47b4899`, `aab2cf9`, `7f6a5cb`, `d537f5c`) were
  reverted locally. Origin was left untouched (see 5.4).
- **Ashes-of-STRM** is a local-only branch (`39f1b04`) holding the full snapshot:
  every experiment script, the training/eval data, the trained checkpoints, and the
  private onyx reference chats used as test/reference data. **Never push it** --
  it contains unsanitized onyx.
- To restore the preserved scratch into a working tree:
  `git checkout Ashes-of-STRM -- scripts/_scratch/`
  To read one file without restoring: `git show Ashes-of-STRM:<path>`

## 1. The short version

**The faith under test.** "The SSM state carries query-relevance -- the backbone
encodes all we need to know about the text. Salience = which states are relevant
right now; forgetting = evicting the rest."

**The decisive test.** Distill DENSE qwen3.5:397b anaphora teacher labels ("which
prior turn does THIS anaphoric query refer to") into the
`CrossSlotTransformerZHead` (the right primitive -- cross-slot attention, beat
bilinear 2.614 vs 0.200 on task #45) trained over the RAW SSM ring states
(`slots_h_raw[k,-1]`, 6144 = last of 4 layers).

**The result: FAIL.** heldout-17 in_top3 = 0.235 vs the 0.647 bar -- below ALL
three baselines (C1 cosine+age+reranker 0.235/0.647; C2 qwen3:8b zero-shot
0.294/0.647, retrieval 0.726; C3 text2x bilinear 0.353/0.355). It cannot fit even
the TRAIN set (in-distribution loo-8 in_top3 = 0.371; best-checkpoint proxy top1
= 0.000). z_logit is non-degenerate at serve (per-record std ~0.295 heldout-17,
~0.362 loo-8; 0/121 degenerate on heldout-17 and 0/470 on loo-8) -- so this is
real signal-absence, NOT a head-collapse bug. An expressive transformer that beat bilinear elsewhere cannot
learn the teacher's labels from these states even on data it trained on.

**The conclusion.** Re-confirms Phase 0b at the retrieval objective: the SSM
state encodes doc-IDENTITY, query-ORTHOGONAL -- not query-relevance-anaphora.
Dense qwen-397b supervision over the RIGHT primitive could not extract a signal
that isn't there. The transferable signal lives in the TEXT (C1 0.647, C2 0.726),
not the state. The state-as-salience-signal path is now falsified twice (bilinear
+ cross-slot transformer, both supervised, both OOD/weak).

**What was text2x (you asked).** text2x was a fine-tune of the 2a backbone (the
19.5M ReferenceSSM) trained with a 2x text-gold oversample, then wired as the
LIVE ENGINE DEFAULT (serve + ingest) in commit `47b4899`, with a single
`ce0_text2x` DocKindHead as the readout. It was the "C3" baseline in the 3-way
anaphora test -- the WORST (held-out 0.353/0.355). It is a state-side readout
(reads the SSM state), which is exactly the path that failed. The revert
un-ships it as the default; the text2x TRAINING commit (`6756e94`) stays in
history but is no longer the served default.

## 2. What NOT to do (anti-patterns in pursuit of the original JST plan)

1. **Do not read the SSM state for a property its training objective never put
   there.** The backbone was trained on next-embedding identity (DialogSum) ->
   the state encodes doc-identity, query-orthogonal. Reading RELEVANCE out of an
   identity-shaped state is a category error. No head architecture fixes it --
   bilinear -> cross-slot transformer -> dense distill all failed for the same
   reason. The shape is set by the OBJECTIVE, not the backbone (see 6).

2. **Do not use a passing probe for the WRONG property as a green-light.** The
   6.1 probe tested RECOVERABILITY (forgetting prediction) -- it passed (AUC
   ~0.810) -- and was used as a green-light to build RELEVANCE (anaphora), which
   it never tested. We skipped the probed-viable weaker proxy (recoverability
   drop on recently-high-relevance slots) and went straight to the unprobed hard
   half (query-conditioned relevance) -- exactly where 6.5 warned "the dependency
   half of salience is the hard part." Probe the EXACT property you will serve. A
   passing recoverability probe says nothing about relevance.

3. **Do not treat Mamba3 as the lever for a relevance/objective failure.** Mamba3
   changes dynamics fidelity and capacity, NOT what the state encodes. A Mamba3
   trained on next-embedding identity is still identity-shaped. The probe
   confirmed ReferenceSSM's selectivity suffices for recoverability. Mamba3 is a
   red herring for the salience/relevance job. (Relevant only later: as an LM
   decoder backbone for text-out, or for higher-fidelity JGS cognitive
   primitives down the roadmap.)

4. **Do not conflate the three jobs and hand all of them to the SSM state.** The
   three jobs are: (a) selective buffer / gist / recoverability, (b) relevance
   ranking, (c) text / compressed-meaning generation. The SSM state is the right
   tool for (a) -- probed, works. Relevance ranking (b) is a text-ranker job --
   works (C1 0.647, C2 0.726). Text generation (c) is a decoder-LM job -- never
   built; needs a vocab head + a token objective. Match the tool to the job.

5. **Do not read the RAW identity state (`slots_h_raw[k,-1]`) with bolt-on heads
   instead of using the JGSInstance framework.** `JGSInstance` (own recurrent
   state + LoRA projections + decomposed gate over a shared frozen backbone) was
   built specifically to give each cognitive function its OWN state subspace --
   the 2.3 anti-collapse mechanism. STRM bypassed it: bolted heads on the
   identity instance's raw state. That is the exact collapse 2.3 predicted, via
   the exact shortcut 1.2 warned against ("separate models with separate
   purposes"). If you want a function's state, train a JGSInstance for it; do
   not read the identity instance's state.

6. **Do not expect a built relevance model to generalize from tiny hard-pair
   gold.** The CE fine-tune (123 hand-picked hard pairs) fit train (0.82 pair-win)
   but transferred ZERO to held-out (0.294/0.588, IDENTICAL to zero-shot) --
   session-style-bound. 123 pairs overfit to session style. The consumed zero-shot
   models generalize because of scale. A built relevance model needs either
   distillation from a working teacher at scale (dense qwen labels) or enough
   data to generalize -- not 123 hand-picked pairs.

7. **Do not conflate "content" with "relevance" in the state.** Phase 0b
   established the state is query-orthogonal (relevance isn't in it). But it
   never cleanly tested whether the state carries CONTENT. Every test (bilinear,
   cross-slot transformer, distill) tested relevance, not content. "Compressed
   meaning out of the SSM" (a decoder reading content) is a DIFFERENT, untested
   property -- possibly viable even though relevance failed. Do not conclude the
   state is useless from a relevance test alone (see 3.3).

8. **Do not skip the de-risking probe's decision tree.** 6.1 said: poor AUC +
   discretization suspected -> Mamba2 swap; poor AUC + no fix -> simplify to
   fixed-interval refresh and STOP. The probe passed (recoverability), so we
   built -- but we built the RELEVANCE head, which the probe never covered. The
   discipline (probe first, gate, stop on failure) was right; the SCOPE of the
   probe was too narrow.

9. **Do not retire JEPA and then assume JEPA-shaped jobs are dead -- but do not
   revive JEPA cargo-cult either.** The 0b probe correctly found linear > JEPA on
   the SURPRISE signal (0.7625 vs 0.565) because: surprise is L2-magnitude but
   JEPA optimizes cosine-direction (objective mismatch); anti-collapse fights the
   prediction objective on temporal data; a frozen backbone has no encoder to
   collapse (JEPA's reason-for-existing is moot). That retirement was correct for
   the surprise/dynamics head. It does NOT mean "JEPA useless everywhere" -- the
   sufficiency / retrieval-need predictor (the ORIGINAL JEPA intent: "is the state
   ready to answer, or do we need to query the graph?") was never built or tested.

10. **Do not conflate JGS and STRM.** JGS = the full cognitive-primitive amendment
    (instances, curiosity cascade, learned self-model -- much down the roadmap).
    STRM = a lighter opportunity to improve short-term memory in the existing
    codebase. We ran STRM as if it were JGS-scale (reading raw state with bolt-on
    heads) when it should have been a focused STM read-out. Keep them separate in
    scope.

## 3. Testing / training procedures for future JST versions

1. **Probe the EXACT property you will serve -- not a proxy.** Before building
   any read-out head, run a probe (frozen backbone, no retrain) for the SPECIFIC
   property: recoverability, relevance, OR content-reconstruction. Each is a
   different probe. A passing recoverability probe does NOT green-light
   relevance. Gate each head on its own probe.

2. **Recoverability probe (passed -- reuse the method).** Log state_t
   trajectories, train a recovery decoder D(state_t) -> u_i, train a probe
   P(state_t, anchor) -> error, measure AUC. This works (0.810). Keep it. It is
   the one validated state read-out.

3. **Content probe (NEW, untested -- the prerequisite for any text-out path).**
   Train a small state -> summary decoder D(state_t) -> gist_text and measure
   reconstruction. If the state carries content, the decoder recovers it (unblocks
   "compressed meaning out of the SSM"). If it cannot, the state is identity-only
   and you must retrain on a content objective. Run this BEFORE building any
   text-out path. This is the cheapest open question and the one most likely to
   rehabilitate the SSM.

   > **STATUS 2026-07-25: PROBED -- FAIL.** The content probe ran (scripts/_scratch/
   > _content_probe_stageA.py + _content_probe_traces.py + _content_probe_stageB.py;
   > full result in memory: pondr-jst-content-probe-result). Two stages, both with
   > the de-wonk guards the §6.1 probe lacked.
   >
   > - **Stage A (per-input identity fidelity + retrieval) -- PASS -> escalate.**
   >   The state carries per-input identity at low lag: retrieval MRR 0.567 intra /
   >   0.468 cross, both beat echo-last-input (0.295 / 0.066). Random-anchor
   >   discrimination 0.69 (above chance), permutation regresses to ~0. Not
   >   identity-only at recency -- so NOT a blanket "state is useless" result.
   > - **Stage B (aggregated gist -- the real §3.3 test) -- FAIL.** On ERAG 400
   >   doc-streams (genuinely diverse targets, cross-chain cosine 0.70). Concat
   >   target: cos_main 0.848 loses to max-single-turn 0.930, mean-pool 0.962,
   >   chain-mean 0.922; lift over the permutation null 0.013 (real_signal=False);
   >   lift over chain-mean -0.073 (negative). LLM-summary target (the primary
   >   gist, deepseek-flash rewrite): cos_main 0.769 beats max-single 0.742 -- but
   >   loses to mean-pool 0.774 and chain-mean 0.870, and the permutation guard is
   >   decisive: perm 0.765 ≈ corpus-mean 0.766 ≈ main 0.769 -> the state->gist map
   >   is null-level (lift 0.004). The "beat max-single" is because a single chunk
   >   is a poor rewrite baseline, NOT because the state aggregates content; a
   >   simple window mean-pool beats the state decoder.
   >
   > **Verdict: the state does NOT carry aggregated content (compressed meaning).**
   > Content -- like relevance 3x before it -- is NOT in the identity-shaped state.
   > Both halves of "the state carries X" (relevance, content) are now falsified;
   > the state's viable job is selective buffer + recoverability (probed, AUC 0.81).
   > "Compressed meaning out of the SSM" requires training a content/token objective
   > into a NEW backbone (vocab head + token CE) -- a retrain, not a bolt-on
   > decoder -- consistent with §6 (state shape set by the OBJECTIVE). The shipped
   > `ssm_chunker` ID-pointer (re-hydrate by ID, not by decoding state) remains the
   > live text-out mechanism. The JST proposal's text-out / gist heads (gated on
   > this probe) do NOT proceed; the buffer/recoverability heads are the state's
   > viable job.


4. **Relevance probe (NEW, never run -- should have been).** Before any relevance
   head, probe whether a query-conditioned reader can extract relevance from the
   state at all. The distill WAS this probe retroactively, and it returned 0.235
   (cannot fit train) -- so the retroactive answer is: relevance is NOT in the
   identity state. On any NEW backbone, run this probe FIRST, before building on
   it.

5. **Linear baseline first, always.** The 0b lesson: a closed-form linear map
   beat a tuned JEPA predictor on the surprise signal. For any new state
   read-out, fit a linear / ridge baseline first. Only build the complex version
   if it beats linear. (The cross-slot transformer was the right primitive but
   could not beat "the state has no signal" -- linear would not have either, and
   would have been the cheaper signal.)

6. **Distillation over hand-picked gold.** For a built relevance transformer,
   supervise by distilling the qwen teacher (dense labels over every window
   turn) -- not 123 hand-picked hard pairs. The teacher is the thing that works
   (0.726); distill it into a small student. Consume the teacher OFFLINE for
   labels; build the cheap serve-time student. This is "build not consume" done
   right.

7. **The cross-slot transformer is the right primitive for anaphora -- keep it,
   feed it right.** Inter-candidate comparison (each slot scored as a function
   of the query AND every other slot) is the correct inductive bias for "which
   prior turn does THIS anaphoric query refer to." It beat bilinear 2.614 vs
   0.200. Keep the architecture. Feed it TEXT (or a content/relevance-carrying
   state), NOT identity state. Bar: clear C1's 0.647, approach C2's 0.726.

8. **If training relevance INTO the state: use L_relevance (the backbone_v2
   path).** `scripts/train_backbone_relevance.py` has L_relevance (multi-positive
   InfoNCE: query embedding vs mean-pooled recurrent state z_k) + L_trajectory
   (next-section JEPA). It directly optimizes the STATE to carry retrieval
   relevance. It showed a glimmer (top3 0.693 @step300, overfit; best-val
   backbone_v2) but was NEVER brought to a serve-clearing anaphora gate. This is
   the "reshape the objective" path (vs "swap the backbone") and the only
   unexplored state-side path. ReferenceSSM can hold it; Mamba3 not required.

9. **Gate on held-out LIVE SERVE, not in-distribution.** The distill's
   in-distribution loo-8 was 0.371 -- weak even on train. A real gate is held-out
   sessions (heldout-17), bar set by the best baseline (0.647). Do not declare
   PASS on in-distribution fit.

10. **De-wonk at completion (per CLAUDE.md).** Audit for
    unimplemented/stubbed/disabled/broken/weird before declaring done. The
    distill's trustworthiness hinged on verifying z_logit was non-degenerate
    (real signal-absence, not a collapse bug) -- that de-wonk check is what made
    the FAIL trustworthy. Keep that discipline.

## 4. The transformer -- relevant, kept

`CrossSlotTransformerZHead` (`src/subconscious/cross_slot_transformer.py`) IS
relevant and is KEPT. It already lives in master at `61e0e9c`, so it survives the
revert automatically -- no special action was needed.

**How / why it is relevant:**
- It is the JST 4.3 context-builder primitive -- a transformer we BUILD, not
  consume. Cross-slot attention: each slot's logit is a function of the query AND
  every other slot (relative scoring), not an independent absolute score.
- Inter-candidate comparison is the right inductive bias for anaphora ("pick
  turn A over the more-recent E by COMPARING candidates").
- The primitive is proven: it beat bilinear 2.614 vs 0.200 on the serve gate
  (task #45).
- It FAILED in the distill ONLY because it was fed the identity state
  (`slots_h_raw[k,-1]`) instead of text/content. The primitive is fine; the input
  was wrong. (It could not fit TRAIN -- the state has no relevance signal -- not
  a capacity problem; proven by non-degenerate z_logit.)
- **Corrected use:** feed it text (or a content/relevance-carrying state),
  distill the qwen teacher, bar 0.647. This is the cheapest next-iteration
  experiment, and the primitive is already built.

**Caveat (do not skip):** the CE fine-tune was ALSO built text-side (123 hard
pairs) and ALSO failed to generalize -- held-out identical to zero-shot,
session-style-bound. So "built text-side" is not automatically winning. Any
distillation must clear the 0.647 bar empirically, not by assertion.

## 5. Manifest: what is where

### 5.1 In master (61e0e9c, survives the reset)
- `src/subconscious/cross_slot_transformer.py` -- the keepable transformer
  (`CrossSlotTransformerZHead`).
- `src/subconscious/{instance,gate,lora,backbone,configs}.py` -- the JGS
  primitive framework: shared backbone held via `object.__setattr__`, separate
  per-instance recurrent state, LoRA projections, decomposed gate.
- `src/subconscious/{retrieval_gate,relevance_head,z_relevance_head,
  recoverability_head,state_readout,latent_dynamics_head,doc_kind_head,
  graduation_head,presentation_gate,context_builder}.py` -- the head zoo.
- `src/subconscious/{ssm,ssm_chunker}.py` -- ReferenceSSM (the actually-running
  SSM, NOT Mamba) and the gist/compression chunker. The chunker recovers text by
  ID-pointer (retained dicts / store / EXPAND), NOT by decoding state.
- `scripts/train_backbone_relevance.py` -- the L_relevance / backbone_v2 trainer
  (the unexplored "train relevance into the state" path, 3.8).
- `src/memory/edge_meta.py` -- reconsolidation counting (the 4.3 two-phase LTP
  fields: `reconsolidation_count`, `ltp_phase`, `consolidation_window_start`,
  `retrieval_timestamps`). Built and tested.
- `src/subconscious/salience.py` -- the EARLIER opt-in version (291 lines). The
  101-line cosine+age / Phase-4 extension from `d537f5c` was reverted; salience
  stays opt-in.
- The text2x TRAINING commit (`6756e94`) is in history; the text2x-as-DEFAULT
  wire commit (`47b4899`) is NOT -- reverted.

### 5.2 In Ashes-of-STRM (39f1b04) -- local-only, NEVER push
- All experiment scripts under `scripts/_scratch/_*.py`: the distill pipeline
  (`_dense_teacher_labels.py`, `_capture_states.py`, `_train_xslot_distill.py`,
  `_xslot_distill_gate.py`), the 3-way anaphora probes (`_c1_reranker.py`,
  `_c3_anaphora_probe.py`, `_strm_3way_anaphora_comparison.py`), the
  CE/cosine/llm-salience probes (`_probe_heldout_crossencoder.py`,
  `_finetune_ce.py`, `_probe_cosine_discrimination.py`, `_llm_salience_probe*.py`,
  `_loo_generalization_probe.py`), the deepseek consults, and the rest.
- Training/eval data: gold pairs (`_heldout_gold.json`, `_trained_gold.json`),
  dense qwen-397b anaphora teacher labels (`_dense_teacher_labels_loo8.json`),
  the trained `CrossSlotTransformerZHead` checkpoint (`_xslot_distill_s0/`), the
  CE fine-tune (`_ce_anaphora_finetuned/`), and the gate result JSONs.
- Private onyx reference chats (the test/reference data sets to KEEP):
  `_chat_intent_ssm.md`, `_chat_intent_jepa.md`, `_heldout_episodes.json`,
  `_trained_episodes_for_labeling.json`, `_xslot_distill_gate_raw_turns.json`,
  `_c3_anaphora_raw_turns.json`, `_heldout_sessions.json`, the ponder
  transcripts.
- Retrieval: `git checkout Ashes-of-STRM -- scripts/_scratch/` restores the small
  scratch into the working tree. `git show Ashes-of-STRM:<path>` reads one file.

### 5.3 Left in the working tree, NOT deleted, NOT in Ashes (too big or nested)
These remain because you said "don't delete anything." They can be deleted at
your discretion:
- `scripts/_scratch/erag/` (1.4 GB) -- ERAG public benchmark parquet. Not onyx;
  regenerable.
- `scripts/_scratch/_states_loo8.pt` (378 MB) -- the raw identity SSM states used
  as the failed distill's input. Useless for the text-side next iteration.
- `scripts/_scratch/llama.cpp/` (198 MB) -- a third-party clone with its own
  `.git` (would have created a submodule mess if committed).
- the Windows `nul` device artifact, and various `*.log` files (gitignored).

### 5.4 Origin
Local `main` is 3 commits BEHIND `origin/main` (origin still has `47b4899`,
`aab2cf9`, `7f6a5cb`). Per your instruction, origin was left untouched. Reconcile
later, when you are ready:
- `git push --force-with-lease origin main` -- rewrites public history. Only if no
  collaborator/CI depends on those three commits.
- `git revert 47b4899 aab2cf9 7f6a5cb && git push` -- safe, additive; keeps both
  the original commits and their inverses in history.
- The unpushed `d537f5c` is now orphaned from main; it is preserved in Ashes.

## 6. First-principles reminder (the one thing to carry forward)

**The state's shape is set by the OBJECTIVE, not the backbone.**
- Identity objective (next-embedding over DialogSum) -> identity state (Phase 0b:
  doc-identity, query-orthogonal).
- To get a relevance-shaped state, train a relevance objective (L_relevance,
  3.8). To get a content-shaped state, train a content/reconstruction objective
  (3.3). To get text out of the state, add a vocab head + a token objective (a
  decoder-LM; Mamba3 is one option for that backbone, not a prerequisite).
- Swapping the backbone (ReferenceSSM -> Mamba3) does NOT change what the state
  encodes. Mamba3 is a red herring for the relevance job. It is only relevant
  later: as an LM decoder backbone for text-out, or for higher-fidelity dynamics
  in the JGS cognitive primitives down the roadmap.

The state is the right tool for buffering / gist / recoverability (probed,
works). Relevance is a text-ranker job (works). The honest next step is either
(a) ship the text-side salience gate (C1 cosine+age+reranker / C2 qwen retrieval)
as the short-term-memory salience signal, or (b) rehabilitate the state by
training a NEW objective into it (L_relevance for relevance, a content
reconstruction objective for compressed-meaning-out) -- but ONLY after the
content probe (3.3) says the state can carry content at all.