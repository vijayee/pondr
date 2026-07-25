import json, time, sys, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. Be concrete, rank actions by expected leverage, and "
    "give a crisp ship-vs-continue verdict. Do not hedge with disclaimers."
)

user_msg = """Advising on the STRM (State-Trajectory Ring Memory) project: a frozen 19.5M ReferenceSSM backbone (d_model=384, 4 layers, d_state=16) trained ONLY on a routing objective, whose `flat_last` readout [6144-dim = last layer 16 channels x 384] encodes PROCEDURAL features (turn position, recency, slot type, retrieval scores) NOT deep query-document semantics. Two relevance heads are trained on top (frozen backbone) to LOCATE query-relevant memory in a live ring of (conversation-message slots type 0 + retrieved-document slots type 1):

- Head A "bilinear" (CompositeZHead): StateReadout mlp128 [6144->384] + ZRelevanceHead bilinear. ~the per-slot projection path.
- Head B "transformer" (CrossSlotTransformerZHead, ~2.98M params): cross-slot attention over the ring's SSM states, with a slot-type embedding + learnable logit_temp. ~the STRM thesis (cross-slot relational reasoning).

Training data: 934 records from 50 Onyx chat sessions replayed against a PERSISTED 72-document corpus (this repo's .md/.py/.txt, ingested via the production pipeline). 58/72 docs are CODE (.py, tree-sitter parsed into one section per def/class, heading=signature, content=source bytes), 14/72 are TEXT (.md). Two live chat sessions are held OUT entirely. The live gate = per-source z_logit selectivity gap (top-relevant-slot logit minus mean filler logit, per source_id, median over eligible turns) >= 2.0 in >= 2/3 seeds.

PRIOR STATE (your earlier diagnosis, for continuity): after the InfoNCE-trained heads FAILED the live gate (bilinear 0.0, transformer 0.994, both 0/3), you diagnosed an overfit-vs-underfit inversion and ranked experiment #1 = train the TRANSFORMER with a pairwise hinge MARGIN loss (margin=2.5) + hard-negative mining + L2 wd 1e-3 + dropout 0.3 on the existing within-corpus data. You predicted: conv -> >=2.0, doc -> 0.8-1.2. Decisive rule: if it lifts live to >=2.0 -> clear path; if it FAILS -> strong evidence the backbone states lack relevance info -> escalate to joint backbone fine-tune.

NEW RESULT (margin loss, m=2.5, hard-neg, wd1e-3, dropout0.3, 120ep, cosine, final-ckpt, drop-self, 3 seeds):

WITHIN-CORPUS held-out (22 unseen sessions):
- bilinear: s0=3.616 P, s1=1.344 f, s2=5.575 P -> median 3.616, 2/3 ROBUST (up from InfoNCE 2.464; margin loss lifted the memorization)
- transformer: s0=0.014 f, s1=0.105 f, s2=2.666 P -> median 0.105, 1/3 (still flat in 2/3; one seed latched)

LIVE (2 held-out transcripts, doc-kind split -- this is the gate):
BILINEAR:
  seed | full | conv | retrieved | retrieved_text | retrieved_code
   s0  | 0.395| 1.258| 0.345    | 4.776          | -0.578
   s1  | 0.000| 0.044| -4.415   | 3.969          | -5.711
   s2  | 0.000| 0.000| -0.801   | -0.542         | -0.801
  -> full median 0.000 (0/3); retrieved_TEXT median 3.969 (2/3 ROBUST PASS); retrieved_CODE median -0.801 (0/3, MIS-RANKS code docs below fillers); conv median 0.044 (0/3, flat)
TRANSFORMER:
  seed | full | conv | retrieved | retrieved_text | retrieved_code
   s0  | 0.037| 0.140| 0.008    | 0.001          | 0.008
   s1  | -0.012| -0.035| 0.041  | 0.214          | -0.047
   s2  | 0.344| 0.106| 0.504    | 0.564          | 0.427
  -> full median 0.037 (0/3); retrieved_TEXT median 0.214 (0/3, FLAT); retrieved_CODE median 0.008 (0/3, FLAT); conv median 0.106 (0/3, FLAT). Note: transformer s2 latched within-corpus at 2.666 but collapsed to 0.344 live (the within-corpus latching was spurious memorization, not generalization).

KEY NEW FINDING (the doc-kind split, which was not in your prior diagnosis): bilinear's apparent "live collapse" (full-ring 0.0) is ENTIRELY a code-doc artifact. On TEXT docs alone, bilinear margin-loss CLEARS the live 2.0 gate ROBUSTLY (2/3, median 3.969). The 58/72 code docs (tree-sitter signatures/source next to prose queries) mis-rank (median -0.801) and drag the full retrieved bucket to 0.0. So bilinear CAN locate text-doc relevance on the live ring; the code-doc REPRESENTATION is the drag, not the bilinear head. The transformer, by contrast, is flat on BOTH text and code (0.214 / 0.008) -- it cannot latch even the text-doc signal that bilinear finds.

The bge embedder + the frozen backbone are prose-trained; code sections (signatures, source bytes) are semantically malformed next to prose queries. retrieval_coverage=1.0 on all runs (docs surface; the gate is well-posed).

QUESTIONS:
1. Given margin loss FAILED for the transformer (flat on both slot types live, including the within-corpus-latched seed collapsing) but SUCCEEDED for bilinear on text-docs (2/3 robust live pass): does this split the two heads' failure modes the way I describe -- bilinear = code-representation drag (fixable at the data/representation layer), transformer = backbone lacks the cross-slot signal (not fixable by loss/head changes)? Or is there a transformer-specific fix (e.g. your prior #2 query-conditioned low-rank projection of h_raw, or #4 wider transformer) still worth trying before concluding Head B architecturally blocked on the live ring?
2. Ranked next experiments to reach a SHIPPABLE state, given the doc-kind revelation. Specifically evaluate: (a) fix the code-doc representation (summarize code into prose before embedding, or a code-specific encoder, or exclude code from the production doc ring) so bilinear's text-doc success extends to code; (b) the conv side is flat for bilinear (0.044) -- why would bilinear pass text-docs but fail conv on the live ring, and is that fixable; (c) joint backbone fine-tune (#7) -- is it now clearly necessary for the transformer, or only if we insist on Head B; (d) a slot-type/doc-kind-SPECIFIC gate as a partial-ship (require >=2.0 only for text-doc retrieved slots, lower/none for code-doc + conv) -- is that product-viable, and what's the risk.
3. Ship-vs-continue verdict: is the bilinear-on-text-docs 2/3 robust live pass enough to ship ANYTHING (partial), or is the conv-flat + code-mis-rank a blocker? Rank the paths by expected effort vs probability of reaching a full live 2.0 robust pass on the production mixed ring.
4. Single highest-leverage next step.
"""

payload = {
    "model": "deepseek-v4-pro:cloud",
    "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    "stream": False,
    "options": {"temperature": 0.3},
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    URL, data=data, headers={"Content-Type": "application/json"}
)

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\ollama_response.txt"

start = time.time()
last_err = None
for attempt in range(1, 4):
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - start
            obj = json.loads(body)
            content = obj.get("message", {}).get("content", "")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("=== OLLAMA_CALL_OK ===\n")
                f.write(f"elapsed_seconds={elapsed:.2f}\n")
                f.write(f"attempt={attempt}\n")
                f.write(f"model={obj.get('model','?')}\n")
                f.write(f"eval_count={obj.get('eval_count','?')}\n")
                f.write("=== CONTENT ===\n")
                f.write(content)
            # also print ascii-only banner so stdout shows progress
            sys.stdout.buffer.write(b"OK\n")
            sys.stdout.buffer.write(f"elapsed_seconds={elapsed:.2f}\n".encode())
            sys.stdout.buffer.write(f"attempt={attempt}\n".encode())
            sys.stdout.buffer.write(f"eval_count={obj.get('eval_count','?')}\n".encode())
            sys.stdout.buffer.write(f"response_file={out_path}\n".encode())
            sys.exit(0)
    except urllib.error.URLError as e:
        last_err = f"URLError: {e}"
        sys.stdout.buffer.write(f"attempt {attempt} failed: {last_err}\n".encode())
        time.sleep(2)
    except Exception as e:
        last_err = f"{type(e).__name__}: {e}"
        sys.stdout.buffer.write(f"attempt {attempt} failed: {last_err}\n".encode())
        time.sleep(2)

sys.stdout.buffer.write(f"=== OLLAMA_CALL_FAILED ===\nlast_error={last_err}\n".encode())
sys.exit(1)