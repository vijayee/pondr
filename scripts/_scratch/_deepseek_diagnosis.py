import requests, json, sys

url = "http://localhost:11434/v1/chat/completions"

briefing = """PROJECT: "Pondr" / "The Ponder Engine" - a personal-memory engine. An SSM (state-space model, ReferenceSSM: new_state = g*b + (1-g)*state, g=sigmoid(W_A(x)), b=W_B(x); one recurrent step per ingested document from a zeroed state) backbone ingests documents into a read-only short-term-memory ring (STRM): each slot holds (y_t readout, source_id, text, pinned, h = the 4-layer recurrent state [4,16,384]).

THE VISION: use the SSM's recurrent state trajectory h_t across the ring + a small Transformer attending over those states to LOCATE which ingested memories are relevant to a query - as a learned alternative/complement to frozen bge cosine retrieval. The thesis: the recurrent state accumulates context that a per-doc embedding cannot, so a state-trajectory locator could beat bge for "which prior memory does this query refer to."

THE WORKING BASELINE THAT ALREADY EXISTS: a frozen bge relevance head ("2a") trained on ERAG-Bench reaches top-3 recall 0.889. The shipped engine uses bge. The state-trajectory work is the attempt to do better than this.

THE JOURNEY (chronological, what's been tried):

1. Phase B GATE 1 (old 19.5M backbone, JEPA-trained on DialogSum next-turn-summary prediction): z_i = a fixed parameter-free projection (mean over the 16 d_state channels of the last layer) of slot.h gave top-3 recall 0.285 == random (0.20), and across-slot std of z_i was 0.068x the doc-embedding baseline - the projected recurrent state was near-constant across 15 different doc inputs. NO-GO. Diagnosis at the time: near-constancy looked like architectural collapse.

2. Re-diagnosis (codebase insight): the near-constancy was likely a TRAINING artifact (the JEPA next-summary objective has no doc-identity signal) PLUS the fixed mean-pool over 16 channels could cancel opposing-sign channel signal. Not necessarily arch collapse. Two levers: (a) a LEARNED readout that mixes channels, (b) a new backbone trained FOR relevance.

3. Phase 0a/0b (learned readout on the FROZEN 19.5M backbone): a learned StateReadout (flat last layer [6144] -> 384) + ZRelevanceHead maxed at top-3 0.564, sub the 0.6 gate. The frozen JEPA state encodes doc-identity mostly query-orthogonally. NO-GO -> concluded the backbone itself must change.

4. Phase 1: built a NEW from-scratch ReferenceSSM backbone (19.5M params, shape-identical, NO warm-start - "don't fight the noise of a different-purpose arch") trained with a JOINT loss: L_relevance (InfoNCE on the mean-pool z_k: logsumexp(cos(q,z_all)/T) - logsumexp(cos(q,z_gold)/T)) + L_trajectory (JEPA predict next section's bge embedding from the last-layer state, in-batch negatives). Direct-SSM identity-instance path (drive backbone.layers[i].step directly from zeroed state, cumulative over the candidate sequence). P1 probe: top-3 0.693 peak (clears 0.6 gate), across-slot std_ratio climbed to 3.8x the collapse floor, monotonic. The from-scratch state is NO LONGER near-constant under a relevance objective. SOFT GO (overfit signature: peaks early at step 300, regresses to 0.533 by step 3000 on a small 30-query train set). Best checkpoint = backbone_v2_full.pt.

5. The hard problem is the SERVE distribution. Training used ERAG candidate-sequences (gold + shuffled negs per query). SERVE is real Onyx chat-replay: the WM ring persists across a session, slots are prior turns, and at each user turn the query is scored against the accumulated ring. This is train/serve OOD. A z-head trained on ERAG traces (task #33): TRAIN top-3 0.935 GO, but SERVE z_r selectivity gap median -0.0066 FAIL.

6. The scoring head = CompositeZHead = StateReadout (flat_last [6144] -> 384, learned MLP-128) + ZRelevanceHead. ZRelevanceHead: per-slot score = (proj_z(z_i) . proj_q(query))/sqrt(P) + bias, where z_i is the readout output, proj_z/proj_q are learned [384->128], and CRUCIALLY bias is a SINGLE nn.Parameter scalar broadcast to ALL slots (one number, shared across every candidate in the ring).

GATES: TRAIN gate = mean top-3 recall >= 0.6 AND Wilson CI lower > 0.5. SERVE gate = per-source selectivity gap median >= 0.2 (z_r, post-sigmoid) OR >= 2.0 (z_logit, pre-sigmoid). The per-source gap groups a source's recurrences: probe = the max-cos occurrence, fillers = the rest; gap = probe_score - mean(filler_scores). Needs sources with >=3 occurrences.

7. Task #41 (CompositeZHead on 114 real Onyx serve traces, trained via fit_relevance = per-slot BCE with pos_weight, the standard trainer): z_r gate FAILS (held-out median ~0 across 3 seeds). z_logit PASSES IN-SAMPLE (ceiling 2.27-3.67) but held-out is noisy sub-2.0 (linear 0.04-1.64, MLP 0.20-1.27). Diagnosed SATURATION: serve fillers are topically close to the gold, so the bilinear scores all candidates high, the sigmoid compresses the small real margin to ~0. Suspicion: 934K-2.5M params on ~91 train turns overfits; "needs more Onyx transcripts + regularization."

8. Task #43 (the overfit-hypothesis test): built 5045 lmsys-chat-1m serve-like traces (55x the Onyx train set, English multi-turn convs; each prior message ingested as a ring slot, query = bge(user turn) vs the ring, gold = top-1-cos). Trained the SAME CompositeZHead (mlp128) on lmsys via fit_relevance (per-slot BCE). RESULT: lmsys held-out z_logit 0.326 (0.25/0.33/0.41), Onyx z_logit 0.258 (0.03/0.26/0.31) - 0/3 pass the 2.0 gate. 55x more data did NOT raise the margin (stayed ~0.3). REFUTED the overfit hypothesis: the in-sample ceiling was memorization of 114 turns; the genuine held-out margin is ~0.3 and does not grow with data. Diagnosed: saturation is INTRINSIC to the per-slot z_i bilinear on topically-close serve-like data - lmsys (conversational context retrieval) reproduces Onyx (ingested-document recall) saturation. Recommended fix: a CONTRASTIVE InfoNCE margin loss (fit_relevance's independent per-slot BCE has no inter-slot contrast -> small margins on topically-close data).

9. Task #44 (just completed - the contrastive fix): new scripts/probe_contrastive_zlogit.py trains the SAME CompositeZHead (mlp128) on the SAME lmsys traces with L = logsumexp(logits/T) - logsumexp(logits_gold/T) instead of per-slot BCE. This loss is BIAS-INVARIANT: the bias is a single scalar added to every slot's logit, so it cancels in (logsumexp-all - logsumexp-gold). The head CANNOT cheat via the bias the way BCE can (BCE can push all sigmoids toward 0 - minimizing the 14-filler loss at weight 1 each - while dragging the pos_weight-upweighted gold down too, yielding "low relevance everywhere" with a small gold-filler margin). Eval is byte-identical to task #43. RESULT (3 seeds, 120 ep, T=1.0):
   - lmsys held-out z_logit 0.735/1.545/0.931 (median 0.931, ~3x BCE's 0.326; 37-48% of sources clear the 2.0 gate) -> PARTIAL DE-SATURATION in-distribution. The bias-collapse mechanism WAS real; this refutes task #43's "fully intrinsic" -> it is PARTLY the loss.
   - BUT the median stays sub-2.0 (even the best seed 1.545 is sub-gate) -> the arch margin is bounded ~1.0. So it is PARTLY the arch too. Truth: partly loss, partly arch.
   - Onyx z_logit 0.120/-2.103/0.048 (median 0.048, WORSE than BCE's 0.258; seed1 strongly negative) -> TRANSFER FAILS and is worse than BCE. Onyx z_r still ~0.
   - NEW FINDING: bias-invariance de-saturates in-distribution but HURTS transfer. Mechanism: removing the bias as a learnable distribution-shift absorber breaks distribution robustness. BCE's learnable threshold transfers weakly-but-positively (0.258); the contrastive's distribution-specific relative margin anti-correlates on the different Onyx distribution (seed1 -2.10). This confirms lmsys (conversational context retrieval) and Onyx (ingested-document recall) are genuinely different distributions.

SUMMARY OF THE STATE-PATH EVIDENCE: the flat-readout per-slot z_i bilinear has now been tested FOUR ways - (a) mean-pool readout, (b) flat-readout BCE, (c) flat-readout BCE + 55x lmsys data, (d) flat-readout CONTRASTIVE - and the contrastive is the best in-distribution number yet (0.931) but is still sub the 2.0 gate and transfers WORSE than BCE. The per-slot z_i bilinear saturates on serve-like data.

THE THREE OPTIONS ON THE TABLE:
(A) Capture more real Onyx serve transcripts + train the contrastive composite ON Onyx in-distribution (kills the transfer confound; tests whether the 0.931 in-distribution margin can clear the 2.0 gate with more Onyx data). Cost: more Onyx serve sessions through the replay harness.
(B) Build the CROSS-SLOT state-trajectory Transformer (the original STRM vision, NEVER tested). Instead of scoring each slot independently with a bilinear, a small Transformer attends ACROSS the ring's state trajectory (the sequence of h_t's) conditioned on the query, and locates the relevant slot. Hypothesis: attention could produce sharper margins than the per-slot bilinear; the cross-slot comparison is what the per-slot bilinear lacks. Risk: it may face the same arch margin bound / saturation, and it's the most engineering effort. (Note: a PER-SLOT channel-Transformer - query cross-attending the 16 channels within ONE slot - was already tested and was WORSE than mean-pool; the cross-slot trajectory Transformer is a different, untested thing.)
(C) Step back: the state path has been tested 4 ways and saturates; the bge 2a head (0.889 train) already works and the engine uses it. Accept that the SSM recurrent state does not beat frozen bge for relevance location, and stop investing in the state-trajectory lever. Redirect effort elsewhere (e.g. the ingestion/tracker work, or the bge head's own serve-distribution gap).

THE QUESTION FOR YOU:
Diagnose what is actually going on mechanistically, and tell me which option to pursue (A, B, C, or something else). Specifically:
1. Is the "partly loss, partly arch" saturation diagnosis correct, or is there a deeper mechanism (e.g. the z_i representation itself is the wrong object, or the per-source gap metric is flawed, or the train/serve gap is the real blocker and no in-distribution fix matters)?
2. The transfer result (contrastive worse than BCE on Onyx) - is the "bias as distribution-shift absorber" interpretation sound, or is there a better explanation? Does it change the recommendation?
3. Given the from-scratch backbone DID uncollapse the state (std_ratio 3.8x, top-3 0.693 on ERAG) but the per-slot bilinear STILL saturates on serve, is the lever the SCORING (-> B, cross-slot attention) or the DATA/DISTRIBUTION (-> A, more Onyx) or the whole premise (-> C, abandon)?
4. If you recommend B, what is the concrete reason attention across the state trajectory would escape a margin bound that the per-slot bilinear hits? If you cannot give that reason, say so.
5. Be decisive. What is the single highest-value next step, and what is the cheap decisive test that would confirm or kill it before committing to the full build?

Answer in full, with mechanistic reasoning. Do not hedge into "it depends" - commit to a recommendation and justify it, then note the key uncertainty."""

payload = {
    "model": "deepseek-v4-pro:cloud",
    "messages": [
        {"role": "system", "content": "You are a senior ML research scientist giving a peer a frank, mechanistic second opinion on a research direction. Be concrete and decisive. Reason from the mechanism, not vibes. If the evidence says stop, say stop."},
        {"role": "user", "content": briefing}
    ],
    "temperature": 0.3,
    "max_tokens": 32768,
    "response_format": {"type": "text"},
}

print("SENDING...", file=sys.stderr)
r = requests.post(url, json=payload, timeout=900)
print("HTTP", r.status_code, file=sys.stderr)
print("RESPONSE_HEADERS", dict(r.headers), file=sys.stderr)
print("=" * 80, file=sys.stderr)

try:
    j = r.json()
except Exception as e:
    print("JSON_DECODE_FAIL", e, file=sys.stderr)
    print(r.text)
    sys.exit(0)

# Save full raw response
with open(r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_response_raw.json", "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False, indent=2)

msg = j.get("choices", [{}])[0].get("message", {})
print("KEYS_IN_MESSAGE:", list(msg.keys()), file=sys.stderr)
print("=" * 80, file=sys.stderr)

reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
content = msg.get("content") or ""

print("REASONING_LEN:", len(reasoning), file=sys.stderr)
print("CONTENT_LEN:", len(content), file=sys.stderr)
print("=" * 80, file=sys.stderr)

print("===REASONING_START===")
print(reasoning)
print("===REASONING_END===")
print("===CONTENT_START===")
print(content)
print("===CONTENT_END===")

# Also print usage
usage = j.get("usage", {})
print("===USAGE===", file=sys.stderr)
print(json.dumps(usage), file=sys.stderr)