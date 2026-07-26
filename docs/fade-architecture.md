# The fade architecture -- dual-SSM, routed by the forgetting signal

> The design record for the Stage-1 foundation, written for a FRESH session.
> Replaces gist-derivation (summary-CE, JEPA-latent) as the foundation; Stage 2
> (the fade itself) is now UNGATED from the gist wall and rebuilt on this.
> Result memories: `pondr-fade-architecture-router` (the decision),
> `pondr-verbatim-reach-result` (probe #31), `pondr-vector-carry-result` (probe #32).
> Probe scripts: `scripts/probe_verbatim_reach.py`, `scripts/probe_vector_carry.py`.

## 1. The vision (load-bearing, verbatim, still in force)

"Exact recall that fades into gists as the compression happens... Human beings
summarize and paraphrase as memories get older. Mirroring this is ideal."

An SSM memory that recalls recent content verbatim and degrades to the gist of
older content as the state compresses over a streamed session. The fade must
EMERGE from actual state degradation, not be a policy switch on wall-clock age.

Also verbatim from the user: "text as state is a solid basis" and "one sentence
summaries are probably not enough... the summary is qualitative rather than a
fixed budget of size."

## 2. Why a new foundation (the wall this clears)

Three Stage-1 attempts to derive gist from the SSM state all hit the same wall --
the state is doc-DISTINCT but doc-IDENTITY-shaped, not gist-shaped:

- **summary-CE retrain** (`docs/gist-retrain-result.md`): end-to-end on
  doc -> deepseek-flash summary. Discrimination margin -0.003 nats (bar >0),
  literally unchanged from the frozen probe's 0.000 -- swapping states still
  changes the gist likelihood by NOTHING. The decoder learned to IGNORE the state
  (state-vs-zero gap -0.037) and the encoder COLLAPSED (continuation val ppl
  175 -> 1419, 8.1x). The summary objective lets the decoder SHORTCUT to the
  marginal summary distribution (most summary tokens are generic structure) and
  bypass the state.
- **JEPA-latent contrastive** then **InfoNCE** (`docs/jepa-gist-result.md`):
  predict the bge latent of the gist from the state. The contrastive variant was
  DEGENERATE for the tight bge cluster (val cos -0.84, anti-correlation); the
  InfoNCE fix bounded the loss but val_latent_cos = 0.067 = RANDOM (noise floor
  0.051), retrieval top-1 0/60. The predictor OVERFIT train and fell back to
  cluster-mean on held-out -- the state carries doc-IDENTITY that does not
  generalize to gist.

Same wall as Phase 0b (`pondr-strm-phase0b-gate-no-go`: "state encodes
doc-identity, query-orthogonal") and the gist-readout probe
(`docs/gist-readout-probe.md`: "state = continuation + recoverability, not
content"). The compression that hurt us there (state keeps identity, loses
content) IS the fade mechanism for the text leg -- if we stop asking the state
for content it does not have and route on what it DOES carry.

## 3. The architecture -- dual-SSM, routed by the forgetting signal

- **SSM-A (identity / fade leg):** carries `bge(chunk)` -- the doc's OWN bge
  vector, injected when the chunk is current (frozen bge, NO teacher LLM). Fades
  slowly over the stream via the SSM recurrence. Read out as a (faded) retrieval
  query. THE FADE LIVES HERE -- in the vector's graceful decay (vectors degrade
  gracefully; doc-IDs do not -- the whole point of carry-the-vector over
  carry-the-docID). NO text is decoded from the state.
- **SSM-B (recall / voice leg):** the token-LM `SSMLanguageModel`
  (`pondr-token-lm-ssm-result`). Given a blurb (a short text excerpt, the content
  seed), EXPANDS it into fuller recall via continuation -- the one thing the
  lm-ssm can do natively (continuation, NOT latent->text decode). Research
  substrate; production swaps in the serving LLM as the voice.
- **The blurb (the bridge):** a short, gist-sized excerpt stored EXTERNALLY at
  ingestion, keyed by the chunk's bge vector. Retrieved by SSM-A's faded vector
  (NOT decoded from the state -- decoding is the 3x-disproven path). Fresh vector
  -> exact blurb -> tight expansion (near-verbatim); faded vector -> approximate
  blurb -> loose expansion (gist).

### The router -- the existing recoverability/forgetting signal (the keystone)

`forget(t) = D(g(z_{t+k}), z_t)` (from `docs/JST-architecture-proposal.md`
Sec 6.1): a recovery decoder's reconstruction error `e(i,t)` IS the ground-truth
forgetting signal; the **recoverability head** `P(state_t, anchor_i) -> e_hat(i,t)`
predicts it. This is the ONE thing the SSM state reliably carries -- AUC 0.81,
Phase 2b shipped (`pondr-strm-phase4-step1-2-heads-wired`). It tracks
lag-independent WHICH-anchor-was-forgotten info (content fate), not just
older=more-forgotten (wall-clock age). The forgetting signal ROUTES recall to
the right regime -- it is not just a 4th regime, it is the regime selector.

### The four regimes (selected per-anchor by e(i,t))

1. **low e** -- still in the state -> **verbatim** from the ring
   (`WorkingMemory._ring`). Exact recall.
2. **medium e** -- degraded but residual -> **fill the holes**: the Transformer
   cross-attention readout (`CrossSlotTransformerZHead`, KEPT from the STRM burn)
   reads the degraded state and reconstructs a fuzzy/partial blurb. Grounded in
   Phase 0a (`pondr-strm-phase0a-state-signal-readout`: signal in READOUT not
   backbone; readout must MIX CHANNELS -- a linear/MLP predictor is too weak,
   cross-attention Transformer is the right shape).
3. **high e, vector still retrievable** -- gone from state, address survives ->
   SSM-A's faded vector retrieves the stored blurb (fuzzily -> a related blurb)
   -> SSM-B expands it. Fuzzy gist.
4. **high e, vector too faded to retrieve** -> **truly forgotten** -> the
   forgetting signal fires; the system SAYS "I've forgotten this" rather than
   confabulating. The graceful FLOOR of the fade -- metacognition (knowing what
   you don't know), mirrors human tip-of-tongue.

## 4. The two no-training gating probes (run BEFORE the build)

Both ran 2026-07-26. Both PASS. The dual-SSM split is fully validated; the build
(#34) is unblocked.

### Probe #31 -- verbatim-reach (the token-content decay curve)

The Sec-6.1 recoverability recipe on the trained token-LM: stream ERAG docs one
token at a time via `SSMLanguageModel.step()`, capture the pooled state
`[n_layers*d_model = 6*192 = 1152]` each step, pair `(s_{i+k}, u_i)` with
`u_i = token_emb(token_i)` `[192]`, ridge decoder, sweep log-spaced lags 1..192.
64 docs, 51/13 train/val chains, 58k/32k pairs. CPU, seconds.

- **top-1 token recovery (the real signal):** k=1 **41%** (1681x chance),
  k=2 7.1%, k=4 2.1%, k=8 0.6%, k>=16 ~0.5-1% -- a thin prior-dominated residual
  tail to k=192. Verbatim reach is ~1-2 TOKENS strong, then prior-only.
- **e(k) MSE = FLAT** (~chance from k=1; horizon 4). Uninformative here: the
  token embedding table is a TIGHT cluster (chance floor 0.0007), so MSE is
  dominated by the cluster mean. This operationalization is the WRONG metric for
  the token-LM substrate. The recoverability head that WORKS (AUC 0.81) predicts
  recovery of a SPREAD-OUT bge CHUNK embedding from the bge-backbone -- a
  different, separately-validated target. So this is NOT a finding about the
  router.

**Verdict: Regime 2 (fill-holes from residual token-LM state content) is THIN.**
A Transformer readout on the token-LM state would mostly recover the language
PRIOR, not doc-specific content. The token-LM is the VOICE (SSM-B), not a fade
substrate. This validates the fade split from a new angle and confirms the 3x
gist fails + Phase 0b.

NOTE: the trained token-LM ckpt is d_model=192 (the gate-run config; the
LMConfig default is 256). 7,920,960 params; pooled state = 6*192 = 1152.

### Probe #32 -- vector-carry (the vector decay curve)

SSM-A as a controlled-decay 384-d channel (the no-training stand-in: a parallel
vector channel with its own decay, no 388<->256 projection for the probe):

    state_p = decay * state_{p-1} + write_gate * bge(chunk_p),  state_0 = 0

so the anchor at stream position i contributes `decay**N * bge(chunk_i)` to
`state_{i+N}`. Read state as a retrieval query, retrieve from the corpus by
cosine, classify the hit: exact (== anchor), sibling (same doc, != anchor =
fuzzy gist / Regime 3), unrelated (other doc = forgotten / Regime 4). Corpus =
537 chunks (48 ERAG docs x ~120-word chunks, bge-small-en-v1.5 384-d frozen),
60 anchors, lags 0..128, decays {0.99,0.97,0.95,0.9,0.8,0.5}. CPU.

**The fade is GRACEFUL and TUNABLE.** For every decay: N=0 100% exact (verbatim);
N=1 exact -> 0 (the next chunk's write at weight 1 beats the anchor), sibling
~98%; N=2..~8-16 a GRADUAL sibling -> unrelated transition whose WIDTH is set by
decay (0.99 ~8-16-step gist window, 0.95 ~8, 0.9 ~4-8, 0.5 ~4); N>~16 unrelated
dominates (forgotten). cos(q,anchor) PLATEAUS at ~0.78-0.85 -- a residual
directional alignment that no longer retrieves the anchor OR a sibling: the
"tip-of-tongue" floor (the topic lingers; the specific chunk is gone). = Regime
4 WITH a residual address the recoverability router can read.

**CONTROL (read-only slot, scalar decay only, no subsequent writes): 100% exact
for ALL N -- argmax stays the anchor forever (pure scalar decay is scale-invariant,
so it does NOT fade cosine retrieval). The cos_to_anchor diagnostic reads 1.0
everywhere after a float64-norm fix; an earlier float32-norm build underflowed to
0.0 at extreme decay**N (decay=0.5, N>=96) where the state is ~1e-29*unit_vec and a
float32 norm squares the components into 0 -- cosmetic, argmax was always the
anchor. Confirms the mechanism: the fade REQUIRES interference -- the recurrence
OVERWRITING the slot with newer chunk vectors. The fade is the SSM doing its job
(compressing the stream), not the vector "shrinking." This is why carry-the-vector
works where carry-the-docID cannot.

**Honesty nuance:** at N=1-2 the "sibling" retrieved is largely the MOST-RECENT
chunk (weight 1 on itself), not yet a genuine gist BLEND of the anchor. The
genuine gist regime (a fuzzy blend that retrieves a related-but-not-just-the-
neighbor chunk) is N=4-8. Both are useful; the interpretation differs by N.

**Verdict: Regime 3 (fuzzy gist via faded-vector retrieval) is VIABLE and rich.**
The fade is graceful, tunable (the trained SSM-A's learnable decay sets the
timescale), with a real gist window + a tip-of-tongue floor. The architecture
holds. The "exact only at N=0" is the simple EWMA's conservative lower bound; a
TRAINED SelectiveSSM with selective gating (write only important chunks, preserve
others) should EXTEND the exact window -- and the ring gives true verbatim for
the ring window regardless.

## 5. What this replaces and reuses

**Replaces:** gist-derivation (summary-CE, JEPA-latent) as the Stage-1
foundation. The foundation is now carry-the-vector + forgetting-router, NOT
derive-gist-from-tokens. We stopped asking the state for content it does not
have.

**Reuses (no rebuild):**
- recoverability head (Phase 2b, AUC 0.81) -- the router.
- `CrossSlotTransformerZHead` (kept from the STRM burn) -- the Regime-2 readout
  (deprioritized -- thin per probe #31).
- token-LM `SSMLanguageModel` -- SSM-B, the voice.
- bge + the vector store -- SSM-A's vector + the blurb store.
- `WorkingMemory._ring` -- Regime 1 verbatim.
- The JEPA-fade objective (if Regime 2 is later pursued) reuses the validated
  InfoNCE loss + LM-prior anti-collapse from `pondr-jepa-gist-result` (both
  worked), but in a LEARNED target space with a Transformer cross-attention
  predictor (not the failed MLP-into-frozen-bge).

## 6. Build priority (#34, now unblocked)

1. ring (Regime 1, `WorkingMemory._ring` -- true verbatim).
2. SSM-A vector-carry + blurb store (external, keyed by bge) + retrieval
   (Regime 3).
3. recoverability router (wire the existing AUC-0.81 head as the regime
   selector).
4. SSM-B voice (token-LM blurb expansion).

Regime 2 (Transformer/JEPA-fade fill-holes) is the THIN optional layer (probe
#31 showed the token-LM state residual is thin) -- deprioritized. Stage 2 (#35)
adds the Transformer T readout only if a richer Regime-2 substrate emerges.

Isolated module, no orchestrator/runtime/serve changes for #34.

## 7. Constraints (unchanged)

ERAG public only; no onyx, no private transcripts. Bible OEB-US via
bible-api.com for Stage-2 eval (NOT copyrighted CJB). deepseek-flash over pro
for any teacher LABELING (this architecture drops the teacher for SSM-A -- bge
is frozen, no LLM -- but SSM-B expansion may use the serving LLM in production).
commit-at-will (no Co-Author, ASCII, no push unless asked, never commit
untracked data/scratch). HF private for PASS ckpts only; a FAIL's ckpt is not
uploaded. De-wonk before completing an implementation (CLAUDE.md).