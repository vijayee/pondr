# JGS — JEPA-Gated SSM: A New Primitive for Behavior Over Data

Date: 2026-07-18
Author: Victor Morrow (architecture) with Claude (build)
Audience: a curious builder who is **not** a data scientist or ML engineer. Every
ML term is explained the first time it appears.

---

## 1. The problem this solves: behavior over data

Most software is **stateless over data**. A search index takes a query, walks a
data structure, returns rows, and forgets you asked. A database stores facts
but has no opinion about which facts matter *now*. An LLM takes a prompt,
computes a giant one-shot answer, and retains nothing between calls unless you
hand it the whole history again. The data is the product; the behavior around
it — deciding what to attend to, what to keep, what to discard, what to act on
— is bolted on with imperative code (`if` statements, hand-written rules, a
prompt template).

Pondr is the opposite kind of system. It is a long-running "pondering" agent
that ingests conversations and documents over time and must **behave** over that
growing stream: notice that a decision today contradicts a snapshot from last
month; remember that a topic is still open; judge whether a retrieved passage is
worth surfacing *given what it already knows*; decide that an incoming document
is a point-in-time snapshot (do not overwrite a stored fact) versus a decision
update (overwrite it). None of that is a lookup. All of it is behavior — small,
fast, stateful judgments made continuously, in the background, on-device, with
no network call.

We needed a building block that does *that*, not data retrieval. JGS is what we
built, and it is new enough that we are still writing the best-practices for it
(see `docs/doc-kind-head-architectural-learnings.md`). This article explains
what it is, why it is a primitive rather than an application, and how it is
trained and combined.

---

## 2. What JGS is, in one sentence

**JGS (JEPA-Gated SSM) is a single trained "backbone" model that many small,
independent "instances" hang off of — each instance being one cognitive
function with its own memory, its own decision policy, and its own tiny set of
learnable knobs, all sharing one common understanding of meaning.**

The name unpacks the three pieces:

- **SSM** — a State-Space Model: a network with a *memory* that evolves
  step-by-step as it reads a sequence (a conversation turn, a document
  section). This is the part that gives behavior-over-time.
- **JEPA** — Joint-Embedding Predictive Architecture: the *way* the shared
  backbone was pre-trained (see §5). It is what makes the backbone a general
  "understanding of meaning" rather than a thing trained for one task.
- **Gated** — each instance carries a *decomposed gate*: a small internal
  decision policy that weighs *is this worth pursuing, and what does it cost?*
  before committing. This is the part that gives selective behavior rather than
  reflexive reaction.

The "new primitive" claim is this: instead of one model per task (the normal
way), or one giant model that does everything via prompting (the LLM way), you
get **one shared understanding + many cheap specialists**, where adding a new
cognitive function costs you a tiny adapter, not a new network and not a new
billion-parameter model.

---

## 3. The anatomy, explained for a non-ML reader

### 3.1 Embeddings: meaning as coordinates

Before anything learns, text has to become numbers a model can do math on. An
**embedder** (we use `bge-small`, a small open model) turns a sentence into a
list of 384 numbers — a point in a 384-dimensional "meaning space" where
sentences about similar ideas land close together. These numbers are called an
**embedding**. Everything in JGS lives in this 384-dimensional space; the
backbone never sees raw words, only embeddings.

### 3.2 The SSM backbone: a memory that updates as it reads

An **SSM (State-Space Model)** is a neural network with a hidden *state* — a
little notebook of numbers — that it updates as it consumes a sequence one step
at a time. Read turn 1, update the notebook. Read turn 2, update the notebook
*using what was already in it*. The notebook is the model's memory of
everything so far. This is fundamentally different from a transformer/LLM, which
re-reads its entire context every time.

Our shipped backbone uses a *selective* SSM. "Selective" means the model decides,
per step, *how much of the new input to write into memory and how much of the old
memory to keep*. Concretely, at each step it computes a "gate" value `g` between
0 and 1 from the current input, then updates memory as:

> new memory = `g` * (write-in from input) + (1 - `g`) * (old memory)

When `g` is high, the new input overwrites memory (the model "pays attention");
when `g` is low, memory mostly persists (the model "remembers"). The model
*learns* the weights that produce `g`, so it learns when to attend and when to
remember. The memory itself is a grid of 16 x 384 numbers (16 "tracks" of
memory, each 384-wide to match the embedding space).

The shipped backbone is **19.5 million parameters**, four SSM layers deep. We
planned for ~480M parameters originally, then right-sized it down because our
real dataset is only a few thousand training pairs — a 480M model would memorize
them instead of learning. This is a recurring theme: JGS is deliberately *small*,
because the design does its work through structure, not scale.

### 3.3 JEPA: how the shared backbone learned "meaning"

This is the part that makes the backbone *shared* and *general*. **JEPA
(Joint-Embedding Predictive Architecture)** is a pre-training objective invented
by Yann LeCun's group. The idea: don't teach a model to predict the next *word*
or *pixel* (that bakes in surface detail). Teach it to predict the next
**embedding** — the next chunk of meaning — directly in meaning space.

We train the backbone by feeding it a sequence of episode embeddings and asking
its **predictor** (a small 3-layer network bolted on top) to guess the *next*
embedding. Crucially, the "correct answer" it predicts against is produced by a
slow-moving copy of the backbone called the **target encoder** (an
**EMA** — exponentially-weighted moving average — of the main backbone's
weights, decaying gently at 0.996 per step). The target encoder is not updated
by gradients; it is just a slowly-drifting "teacher." Predictions are compared
against 16 *negative* examples (wrong embeddings) so the model learns to push
the predicted-next-meaning *away from* irrelevant meanings, not just toward the
right one.

Why this matters: because the backbone learns "what comes next in meaning
space" on general sequences, it ends up with a general representation of
semantic continuity — a shared understanding. Every cognitive function we then
build on top starts from that same understanding instead of from scratch. The
backbone is trained **once**, then **frozen** at serve time.

### 3.4 The JGSInstance: one cognitive function with its own memory and policy

Here is where the primitive becomes a primitive. A **JGSInstance** is a small
bundle that wraps the shared frozen backbone and adds, per cognitive function:

- **Its own recurrent state.** Each instance keeps its own copy of the 16x384
  memory. The RetrievalGate has one memory; the WorkingMemory has a *different*
  memory. They evolve independently. After each step the state is *detached*
  (explained in §5.3) — the instance does not try to backpropagate gradients
  across time steps at serve time; it just carries memory forward.
- **An input projection and an output projection** — small matrices that adapt
  the shared backbone's 384-dim space to this function's needs. These are
  **LoRA** (see §5.2): very low-rank adapters, only a few thousand parameters.
- **A StateLoRA** — a tiny adapter that *modulates the SSM dynamics themselves*
  for this function. The shared SSM has fixed dynamics; the StateLoRA lets each
  function bias how the shared memory updates without changing the shared
  weights. This is how one frozen backbone can behave differently per function.
- **A decomposed gate** — the instance's decision policy (§3.5).

The clever engineering trick: the instance holds the backbone by a back-channel
(`object.__setattr__`) so that when you ask "what are this instance's
parameters?", the 19.5M backbone is **excluded**. The instance's own parameter
count is tiny — a few hundred thousand. That means checkpoints are lean (a few
hundred KB, not 19.5MB), and a standard optimizer touching `instance.parameters()`
naturally leaves the frozen backbone alone. The shared backbone is loaded
*once* and shared across all instances in a process.

### 3.5 The decomposed gate: deciding whether to act

The "Gated" in JGS. Each instance has a **DecomposedGate** with three small
heads:

- a **value head** — how valuable is pursuing this?
- a **cost head** — how expensive / risky is it?
- a **decision head** — combining value and cost (and a per-instance *context*
  vector, e.g. "how recent is this entity? how novel is this input?") into a
  single `pursue: yes/no` plus a confidence.

Decomposing value, cost, and decision is the whole point: instead of a single
opaque "should I act?" score, the system can reason about *why* — and you can
weight one factor differently per function (the retrieval gate weights recency;
the disturbance detector weights novelty). The gate is the mechanism that makes
behavior *selective* rather than reflexive: most inputs are looked at and
discarded; only some are pursued.

### 3.6 Three cognitive functions shipped today

- **RetrievalGate (Phase 2b).** A JGSInstance that decides, per query, what to
  retrieve and via which pathway. It **resets its state per query** — each
  query is an independent judgment. Validation accuracy 0.826.
- **WorkingMemory (Phase 2c).** A JGSInstance that does **not** reset its state
  between queries — its defining property is that it *carries awareness
  forward* across the conversation, so the system's sense of "what's currently
  salient" persists. This is the difference between a stateless router and a
  stateful consciousness layer.
- **DocKindHead (Phase 3c).** A JGSInstance that classifies an incoming
  document into one of five kinds (snapshot / decision_update / plan /
  reference / other). It **resets its state per document**. This classification
  is consumed by the contradiction guard: a snapshot must not overwrite a stored
  fact; a decision update may. Shipped as a 2-head ensemble (see §6).

Seven more instances are declared in config but not yet trained
(`presentation_gate`, `uncertainty_detector`, `aspirational_model`,
`self_model`, `common_sense_resolver`, `disturbance_detector`,
`intuition_module`) — ten cognitive functions in total on one backbone, three
shipped today.

---

## 4. Behavior over data: the functionality it enables

Because each instance has its *own evolving memory* and its *own decision
gate*, JGS gives you things a data-retrieval stack fundamentally cannot:

- **Stateful judgment.** WorkingMemory's non-resetting state means the system's
  view of "what matters right now" is a continuous carry, not a recomputation.
  A retrieval system can re-fetch; it cannot *remember that it already
  considered this five minutes ago and decided it was stale*.
- **Selective attention.** The decomposed gate means most incoming events are
  noted and dropped; the system spends its computation on the few that clear
  the value/cost bar. This is how you get an agent that *notices* the one
  contradiction in a month of otherwise-boring logs.
- **Per-function specialization on one understanding.** You get ten different
  behaviors — routing, working memory, kind classification, presentation,
  uncertainty, aspiration, self-modeling, common-sense, disturbance,
  intuition — all running as cheap local forward passes off one shared 19.5M
  "meaning engine." Adding a behavior is a new ~hundred-thousand-parameter
  adapter and a short training run, not a new model and not a new API bill.
- **On-device, offline, deterministic.** Because the whole thing is a 19.5M
  model plus tiny adapters, it runs as a local forward pass — no HTTP, no
  rate limits, no nondeterminism, no data leaving the machine. The
  contradiction guard that consumes DocKindHead fires per-ingest, on your
  hardware, identically every time.

The impact on "software that needs behavior over data": you stop building
agents as "prompt + giant model + re-read everything each turn," and start
building them as "one shared understanding + a roster of small stateful
specialists." Behavior becomes a first-class, trainable, composable part of the
system rather than hand-coded glue around a data store.

---

## 5. How we train and modify JGS instances (ML concepts explained)

### 5.1 The two-level training story

There are two kinds of training, and they are completely separate:

1. **Backbone pre-training (done once).** We train the 19.5M SSM + its JEPA
   predictor on a few thousand general "episode follows episode" sequences so
   it learns semantic continuity. Then we **freeze** it. We never touch these
   weights again. This is the shared understanding every instance inherits.
2. **Instance training (done per function).** For each cognitive function we
   train *only* that instance's tiny adapters (input/output projections,
   StateLoRA) and its gate/head, on *task-specific labels*. The backbone is
   frozen throughout; gradients flow through it but do not update it.

This separation is the whole economic argument: the expensive, general part is
paid once; each new behavior is cheap.

### 5.2 LoRA: changing a frozen model with a few numbers

**LoRA (Low-Rank Adaptation)** is the trick that makes instance training cheap.
Normally, changing a neural network layer means learning a full matrix of
weights — millions of numbers. LoRA says: don't change the big matrix; add a
pair of *very small* matrices (rank 4, 6, or 8 in our configs) whose product
approximates the change you'd want. A rank-4 adapter on a 384-dim layer is
~3,000 numbers instead of ~150,000. You keep the frozen backbone's big matrix
unchanged and *add* the LoRA delta at serve time. Each instance's LoRA is its
own — that is literally how one frozen backbone behaves nine different ways.

### 5.3 "Detached state": memory without backprop-through-time

A subtle but load-bearing choice. At serve time, each step produces a new
state, and the instance stores it *detached* — meaning "this number is a value,
not something to differentiate through later." In ML terms we do **not** do
BPTT (backpropagation-through-time) across steps at serve. The reason: we want a
long-running agent that remembers indefinitely, and BPTT across thousands of
steps is intractable and unstable. So the memory is carried forward as state,
not as a gradient graph. Training updates the *policy* that produces the state
(via LoRA), not by unrolling the whole history each step.

### 5.4 Labels: the teacher and the discipline problem

Instance training needs labeled examples (e.g. "this document is a
decision_update"). We did not hand-label thousands; we used a strong LLM as a
**teacher** (DeepSeek, via a local Ollama endpoint, set to not show its
reasoning — that setting is load-bearing for token cost and determinism). The
teacher emits a confidence; we keep labels only above a confidence bar and let
it abstain otherwise.

We learned a hard lesson worth flagging: **a teacher's confidence does not
guarantee label quality.** Our teacher confidently over-assigned the
"decision_update" label to support-thread content. A later three-teacher panel
audit (majority vote of three different models) overruled ~17.5% of the labels,
and those bad labels had been contaminating both training *and* validation. We
relabelled with the panel before trusting any architecture conclusion. The
general lesson, stated for the lay reader: the model can only ever be as good as
the labels it learned from; if your labels are wrong, you will misread a good
architecture as a bad one.

### 5.5 The ship gate: deciding "good enough" honestly

For DocKindHead we wrote a strict, up-front **ship gate** — a conjunction of
thresholds the head had to clear before it could go to production: unsafe
confusions at most 1; recall on both date-sensitive classes at least 0.70;
overall accuracy at least 0.55; and — importantly — the *lower bound of a
confidence interval* on the harder class's recall above 0.50. That last one
matters because our validation set was small (17 examples for one class). On 17
examples, "12/17 = 70.6%" is statistically indistinguishable from a coin flip's
worth of noise; the Wilson confidence interval's *lower edge* is the honest
number to gate on. One example swung that lower edge by 0.06. Gating on the
point estimate would have been self-deception; gating on the CI lower bound
kept us honest.

### 5.6 Modifying an instance after the fact

Because instances are tiny and the backbone is frozen, "modifying" a function
is cheap and non-destructive: retrain the LoRA + gate/head with a different
recipe, keep the checkpoint as a small file, and swap it in. You can keep
several variants of one function (trained with different tradeoffs) and combine
them — which is exactly what we did to ship DocKindHead.

---

## 6. Combining JGS and their parts

### 6.1 Instances compose into a system

The shipped Pondr runtime is *several instances on one backbone*:
WorkingMemory carries state forward; RetrievalGate routes per query;
DocKindHead tags per ingest; the contradiction guard consumes the tag. Each is
a JGSInstance; together they form a layered "subconscious" — memory, routing,
perception, guarding — all running as local forward passes off the shared
frozen backbone. Composition is by *wiring instances' outputs into each
others' inputs*, not by retraining anything.

### 6.2 The multi-gate ensemble: combining *heads* of one instance

This is the result that confirmed the primitive's best-practice. DocKindHead
is a single 5-way classifier: a document is exactly one of five kinds. That
mutual exclusivity is the normal case for a single softmax head. But we found a
structural problem: the two date-sensitive classes (snapshot vs decision_update)
were **coupled** — a single head trained at any penalty setting could get
snapshot recall high *or* decision recall high at a safe operating point, but
**never both**. It was a forced tradeoff baked into the single softmax.

The fix was not a new architecture. We trained two heads at opposite ends of the
tradeoff frontier — one (`pen0`) strong on snapshots, one (`pen2`) strong on
decision-updates — and **averaged their output logits** before deciding. The
ensemble cleared the full strict gate with margin (both guard classes 13/17,
zero unsafe confusions, accuracy 0.632, CI lower bound 0.527), where no single
head could clear it. The mechanism: each head spends its "separability budget"
on a different class; averaging relaxes the softmax competition because the two
heads *disagree productively*.

The general best-practice, learned from data: **when a single head on this
primitive forces a costly tradeoff between two criteria, train heads at
opposite ends of the tradeoff and average — before reaching for a more complex
architecture.** It costs N forward passes at serve (acceptable when the
function is per-ingest, not per-query) and no retrain. This is now documented as
the primitive's first established pattern.

### 6.3 Parts you can mix

- **Backbone** — one frozen, shared. Swappable SSM backend (we ship the pure
  PyTorch `ReferenceSSM`; a faithful Mamba3 backend is a drop-in when it builds
  on the target platform).
- **Adapters** — per-instance LoRA (input/output projections, StateLoRA). Mix
  and match per function; rank tunable per function (fast functions rank 4,
  rich-state functions rank 8).
- **Gate** — decomposed value/cost/decision, with a per-function context vector.
- **Head** — the readout above the instance (attention-over-sections for
  DocKindHead; a routing head for RetrievalGate). Heads can be ensembled.
- **Feature re-injection** — for DocKindHead, a small hand-engineered
  doc-level regex feature (does it contain an explicit date? an "as of"
  phrase?) is concatenated with the pooled embedding. The lesson here
  generalizes: a cheap engineered feature can break a class-specific ceiling
  the learned pool can't see; combine it with the learned readout, don't
  substitute.

---

## 7. What JGS gave us vs the LLM alternative over this problem space

The obvious alternative to all of this is "just call an LLM." For each judgment
— is this doc a snapshot? does this contradict what we stored? what should we
retrieve? — send the relevant text to a hosted LLM and take its answer. That
is the default architecture for agentic software today. Here is why we did not
build that, and what JGS gave us instead:

- **Latency and cost at the call site.** The contradiction guard fires *per
  ingested document*. An LLM call per ingest means a network round-trip, a
  queue, a token bill, and rate limits — for a judgment that must happen
  synchronously during ingestion. JGS does it as a local 19.5M forward pass,
  milliseconds, no network.
- **Determinism.** Hosted LLMs are nondeterministic and drift with provider
  updates. A contradiction guard that occasionally lets a snapshot overwrite a
  stored fact because the model had a different mood today is unacceptable for
  a memory system. A frozen local model is bit-stable.
- **Data sovereignty.** Conversations and documents staying on-device, never
  shipped to a third party, is a feature, not a limitation. JGS runs entirely
  locally.
- **Composability without re-billing.** With the LLM approach, every new
  cognitive function is another prompt and another bill, and they do not share
  anything. With JGS, every new function is a tiny adapter on the *same* frozen
  understanding; adding the ninth function does not re-pay for the first eight.
- **Right-sizing.** An 8B-parameter LLM is overkill for "is this a snapshot or
  a decision_update." We do not need a general world model for that; we need a
  specialist that has learned the specific boundary. JGS gives us the
  specialist at a few hundred thousand parameters, and lets the general
  understanding be paid for once and shared.

The honest tradeoff: an LLM is more *flexible* and handles open-ended tasks we
have not trained a specialist for. JGS specialists are *better* at the narrow
jobs we trained them for — faster, cheaper, deterministic, private — and
composable into a system. The right architecture uses each for what it is good
at: the LLM as a *teacher* during training (and as a fallback for the
open-ended), the JGS system as the *runtime* that does the repetitive stateful
judgments continuously and locally. We already use the LLM exactly that way —
as the labeler/teacher — not as the runtime.

---

## 8. Future work and exploration

- **More instances on the backbone.** Seven declared functions are not yet
  trained: presentation gate, uncertainty detector, aspirational model,
  self-model, common-sense resolver, disturbance detector, intuition module.
  Each is a JGSInstance waiting for a label source and a training run. The
  architecture is ready; the work is labeling and gating.
- **A faithful Mamba3 backend.** We ship the pure-PyTorch `ReferenceSSM`
  because the optimized Mamba3-CUDA build does not work on this dev box. A
  faithful Mamba3 backend is a drop-in (`sssm_backend="mamba3-cuda"`); swapping
  it in on a CUDA machine is a config change, and would give us the
  well-validated Mamba3 dynamics for free.
- **Larger / cleaner backbone corpus.** The backbone was right-sized to a few
  thousand pairs. A larger, curated pre-training set would deepen the shared
  understanding and lift every instance that inherits it — but only after we
  are sure the per-instance labels are clean (the teacher-confidence lesson).
- **Distillation of the ensemble.** The shipped DocKindHead is a 2-head
  ensemble (2x forward at serve). We could distill it back to a single head
  for 1x serve. We have deliberately *not* done this: the ensemble's power is
  the productive disagreement of the two heads, and a distilled single head
  might collapse back to the coupling. Distillation is a deferred optimization,
  to be attempted only if the 2x cost ever matters (doc-kind tagging is
  per-ingest, not per-query, so it does not today).
- **Outcome signals wired back into the gates.** The decomposed gates are
  trainable in principle, but several ship in a heuristic/placeholder form
  because live *outcome* signals (did pursuing this help?) are not yet wired.
  Closing that loop — letting the gate learn from downstream outcomes — is the
  next big capability unlock for the primitive.
- **Generalizing the multi-gate best-practice.** The ensemble lesson
  (§6.2) is the primitive's first established pattern. As more instances ship,
  we expect more patterns: when to use a binary gate vs a softmax, when to
  cascade a coarse head with a fine resolver, when to re-inject engineered
  features. The `doc-kind-head-architectural-learnings.md` document is the
  accumulating record.
- **Unfreezing the backbone, carefully.** Not done, because it risks the
  shared backbone that RetrievalGate (val 0.826) depends on. Reserved for a
  moment when a specific instance provably needs a richer shared representation
  and we can re-validate every existing instance after.

---

## 9. Closing

JGS is a bet that the right unit of "agent" is not one giant model prompted per
turn, and not one bespoke model per task, but **one shared trained
understanding plus a roster of small, stateful, gated specialists** that
*behave* continuously over a growing stream of experience. It is small on
purpose, local on purpose, deterministic on purpose, and composable by design.
We are still writing the manual for it — the DocKindHead ensemble was the first
confirmed best-practice, and it came from data, not from theory. The primitive
is new; the patterns are being discovered as we ship. The point of this article
is that you do not need to be a machine-learning professional to reason about
it: the core ideas are memory that updates selectively, a shared understanding
learned once, small specialists trained cheaply, and gates that decide when to
act. Everything else is implementation.