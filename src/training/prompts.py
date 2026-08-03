"""Oracle prompts for Phase 1d training-data generation.

Each prompt is a pure function that takes structured input and returns a
formatted prompt string. Keeping prompts as functions (not inline literals in
the generators) makes them versionable, unit-testable, and independent of the
Oracle API-calling code in ``src/training/oracle_labeling.py: OracleClient``.

Ported from ``docs/Phase 1d.md`` §3. The prompt *bodies* are verbatim from the
doc; only the module scaffolding is new. Every prompt instructs the Oracle to
``Return ONLY valid JSON`` so the client's ``response_format: json_object`` +
salvage parser can recover the payload.
"""

from __future__ import annotations

import json

# ═══════════════════════════════════════════════════════════════
# GNN TRAINING DATA
# ═══════════════════════════════════════════════════════════════


def gnn_salience_prompt(subgraph_json: str) -> str:
    """Prompt for GNN salience scoring labels."""
    return f"""You are labeling a memory graph for GNN training.
Score each node and edge by structural importance (0.0-1.0).

HIGH salience (>0.7):
- Bridge nodes connecting otherwise-separate clusters
- Episodes containing major decisions
- Temporal chain anchors (first episode in a sequence)
- Nodes with unique information not available through other paths

MEDIUM salience (0.3-0.7):
- Episodes with moderate entity/topic overlap with other episodes
- Nodes that are part of active temporal chains but not anchors

LOW salience (<0.3):
- Routine conversations with no decisions
- Nodes with redundant information available through other paths
- Peripheral entities mentioned only once

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"node_scores": {{"ep_001": {{"salience": 0.92, "reason": "..."}}, ...}},
 "edge_scores": {{"edge_001": {{"salience": 0.80, "reason": "..."}}, ...}}}}"""


def gnn_cluster_prompt(subgraph_json: str) -> str:
    """Prompt for GNN cluster detection labels."""
    return f"""You are labeling a memory graph for GNN training.
Identify groups of episodes that should be abstracted into semantic memories.

A valid cluster has:
- Shared entities (at least 2 entities in common)
- Shared topics (at least 1 topic in common)
- Temporal proximity (within 7 days of each other)
- Coherent theme (the episodes tell a connected story)

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"clusters": [
    {{"name": "WaveDB initial development (June 20-24)",
     "episodes": ["ep_001", "ep_002", "ep_003", "ep_004"],
     "abstracted_summary": "Decided on HBTrie architecture...",
     "coherence_score": 0.89}}
]}}"""


def gnn_link_prediction_prompt(subgraph_json: str) -> str:
    """Prompt for GNN link prediction labels (positive + negative edges).

    SEAL/GAE need explicit negative edges, not just positives — a link-prediction
    head trained on positive-only labels collapses to "predict 1 everywhere."
    So the Oracle emits both ``predicted_edges`` (edges that SHOULD exist but
    don't) and ``negative_edges`` (node pairs that plausibly COULD share an edge
    but should NOT — unrelated entities/topics, non-adjacent episodes with no
    shared context). Phase 3a Task 3 added the negative-edges field.
    """
    return f"""You are labeling a memory graph for GNN training.
Identify edges that SHOULD exist but are not explicitly in the graph, AND node
pairs that should NOT be linked (negative examples for link prediction).

Look for POSITIVE edges (predicted_edges):
- Entities that co-occur in similar contexts but have no direct edge
- Episodes that share topics/entities but aren't linked
- Hierarchical relationships implied by usage patterns
- Causal relationships implied by temporal order
- Contradictions between statements in different episodes

Look for NEGATIVE edges (negative_edges) — pairs that plausibly COULD share an
edge given their types/proximity but should NOT:
- Entities from unrelated domains that happen to appear in the subgraph
- Episodes far apart in time with no shared entities/topics
- Topic/entity pairs with no semantic relationship

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"predicted_edges": [
    {{"subject": "Postgres", "predicate": "related_to", "object": "performance",
     "confidence": 0.82, "evidence": "Both appear in episodes about..."}}
],
 "negative_edges": [
    {{"subject": "Postgres", "predicate": "related_to", "object": "WaveDB",
     "confidence": 0.10, "evidence": "No shared context; unrelated domains"}}
]}}"""


def gnn_ontology_prompt(subgraph_json: str, current_ontology: str) -> str:
    """Prompt for GNN ontology refinement labels."""
    return f"""You are labeling a memory graph for GNN training.
Suggest missing subClassOf edges and misclassified entities.

CURRENT ONTOLOGY:
{current_ontology}

SUBGRAPH:
{subgraph_json}

Return ONLY valid JSON:
{{"suggested_edges": [
    {{"child": "DEBOUNCED", "parent": "WALSyncMode", "confidence": 0.90,
     "evidence": "Discussed alongside IMMEDIATE and ASYNC as alternatives"}}
],
 "misclassified": [
    {{"entity": "WaveDB", "current_class": "Application",
     "suggested_class": "Database", "confidence": 0.85}}
]}}"""


# ═══════════════════════════════════════════════════════════════
# BONSAI TRAINING DATA
# ═══════════════════════════════════════════════════════════════


def bonsai_anomaly_decision_prompt(
    flagged_entity: str, retrieved_context: dict, anomaly_type: str
) -> str:
    """Prompt for generating Bonsai anomaly-DECISION training pairs (spec §2.5).

    This is **retrieve-then-prompt**: the Oracle demonstrates the fix/ask_user/dismiss
    decision from the SAME context Bonsai will have at deploy (the radius-1 graph
    neighborhood of the flagged node, retrieved from the store before the call) — so
    Bonsai's fine-tuned decision is reproducible, not based on context only the teacher
    sees. The Oracle is the teacher (we planted the drift, so the "correct" decision is
    checkable against ``anomaly_type``); Bonsai is the student; the Oracle is NOT in the
    deploy loop. Returns ONLY the three fields Bonsai must predict; the caller echoes
    ``flagged_entity``/``retrieved_context``/``anomaly_type`` back into the pair record.
    """
    ctx_json = json.dumps(retrieved_context, ensure_ascii=False)
    return f"""You are generating training data for a small local model (Bonsai) that decides
what the memory system should DO about a flagged anomaly, given the same retrieved
context the model will have at deploy.

Anomaly types (the 9 structural head labels + the identity_drift review-flag):
- contradictory_state: one entity carries two distinct live state values
- duplicate_episode: two near-identical episode summaries (re-import / sync)
- duplicate_decision: two near-identical decision records
- orphan_decision: a decision node with no link edges (partial ingest)
- detached_episode: an episode node with no link edges (partial ingest)
- broken_follows: a follows edge pointing at a missing target
- type_violation: an edge whose predicate domain/range doesn't match the endpoints
- isolated_cluster: a connected component with no path to the query focus
- stale_abstraction: a semantic-memory M: node abstracts a dead/missing episode
- identity_drift: one entity name appears to refer to two different referents
  (disjoint topic neighborhoods — often legitimate; genuinely semantic)

Decisions (pick exactly one):
- fix: the system can safely resolve this without bothering the user
- ask_user: a human must clarify (ambiguous, costly, or genuinely semantic)
- dismiss: this is legitimate structure, not an error worth touching

Hippo actions (the concrete operation behind the decision; one short phrase):
- fix duplicate_episode/duplicate_decision → "merge the duplicate into the original"
- fix orphan_decision/detached_episode → "re-link the orphaned node to its episode/entities"
- fix broken_follows → "re-link or delete the dangling follows edge"
- fix type_violation → "rewire the edge to a kind that satisfies the ontology"
- fix stale_abstraction → "repoint the abstraction at a live episode or retire the M: node"
- fix contradictory_state → "supersede the older state, keep the latest"
- ask_user identity_drift → "ask a clarifying question to split or confirm the entity"
- dismiss isolated_cluster → "leave the legitimate separate-domain cluster alone"
- ask_user (anything costly/ambiguous) → "ask the user before mutating the graph"

FLAGGED ENTITY: {flagged_entity}
ANOMALY TYPE: {anomaly_type}

RETRIEVED CONTEXT (radius-1 neighborhood of the flagged node, as Bonsai sees it):
{ctx_json}

Decide what the system should do, then explain it. Default to "ask_user" when the
fix could lose information and you are not sure.

Return ONLY valid JSON:
{{"decision": "fix|ask_user|dismiss", "action": "...", "reasoning": "..."}}"""


def bonsai_contradiction_decision_prompt(
    flagged_entity: str, retrieved_context: dict
) -> str:
    """Prompt for the deploy-time Bonsai contradiction adjudicator (Phase 3c D3).

    Mirror of ``bonsai_anomaly_decision_prompt`` specialized for the
    fact-level contradiction: one entity carries two (or more) distinct live
    ``state`` values, each asserted by a different source (episode/document/
    section) at a different time. The disturbance record -- the conflicting
    values WITH their provenance (``asserted_by`` / ``asserted_at``) -- is
    carried in ``retrieved_context["state_values"]`` (gathered by
    ``_gather_entity_context``), so Bonsai adjudicates from the same evidence
    the chat's disturbance record specifies, not just the generic graph
    neighborhood.

    The conservative dispatcher (consolidate ``_apply``) auto-applies ONLY a
    ``fix`` whose ``action`` contains ``supersede_assertion`` (the fact-level
    tombstone); any other ``fix`` routes to ``ask_user`` (record-only). So the
    prompt steers a safe fix toward the literal action ``supersede_assertion``.
    Returns ONLY the three fields Bonsai must predict.
    """
    values = retrieved_context.get("state_values") if isinstance(
        retrieved_context, dict) else None
    values_json = json.dumps(values, ensure_ascii=False) if values else "[]"
    # The rest of the context (states, episodes, topics, instance_of) -- the
    # same neighborhood shape the anomaly prompt receives.
    ctx = {k: v for k, v in (retrieved_context or {}).items()
           if k != "state_values"} if isinstance(retrieved_context, dict) else {}
    ctx_json = json.dumps(ctx, ensure_ascii=False)
    return f"""You are the deploy-time decider for a small local model (Bonsai) that decides
what a memory system should DO about a CONTRADICTION: one entity carries two
(or more) distinct live state values, each asserted by a different source
(episode / document / section) at a different time. This is the conflict-aware
cognitive mode -- detect, adjudicate, tombstone the superseded fact at the FACT
level (not the whole episode), and keep it retrievable.

Decide by running these checks IN ORDER against the CONTRADICTING VALUES +
PROVENANCE below. The FIRST check that matches gives the decision; stop there.
- Check 1 -- EQUAL VALUES: if the two state values are the same string (e.g.
  both "Postgres"), there is no collision -> dismiss (action "no_action"). A
  value repeated across two sources is a duplicate, not a contradiction.
- Check 2 -- DIFFERENT ENTITIES: if the two source documents name DIFFERENT
  subjects that merely share a value (e.g. one says "the frontend framework is
  React", the other says "the mobile framework is React"), there is no shared
  entity -> dismiss. Compare the asserted_by source-doc names: if they name two
  different things (frontend vs mobile, two different subsystems), dismiss.
- Check 3 -- COMPLEMENTARY TEMPORAL: if the two source docs are month- or
  year-named point-in-time records (e.g. docs/jan-status.md, docs/jul-status.md
  -- the filename starts with a month abbreviation jan/feb/mar/apr/may/jun/
  jul/aug/sep/oct/nov/dec), the two values are states at different dates and
  both can be true at their respective times -> dismiss.
- Check 4 -- GENUINE CONFLICT: none of checks 1-3 matched -- same entity, two
  DIFFERENT values, and the sources are NOT month-/year-named point-in-time
  records -> fix, action "supersede_assertion" (tombstone the older value, keep
  the newer retrievable).

A value being NEWER is never, by itself, a reason to supersede. Many newer
assertions just repeat an existing value (check 1 -> dismiss), report a
different entity (check 2 -> dismiss), or record a point-in-time state (check 3
-> dismiss). ONLY check 4 (a genuine same-entity value change) is a fix -- and
only after checks 1, 2, and 3 have all been ruled out.

Decisions (pick exactly one):
- fix: safely resolve without the user -> action "supersede_assertion"
  (ONLY from check 4; tombstone the older value, keep the newer)
- ask_user: a human must clarify -- use this when you cannot confidently place
  the pair into any check above; do NOT auto-supersede when unsure
- dismiss: not a real contradiction (check 1 / check 2 / check 3 / stale noise)

FLAGGED ENTITY: {flagged_entity}

CONTRADICTING VALUES + PROVENANCE (value, who asserted it, when):
{values_json}

SURROUNDING CONTEXT:
{ctx_json}

Decide what the system should do, then explain it. Default to "ask_user" only
when you cannot confidently place the pair into any of the cases above.

Return ONLY valid JSON:
{{"decision": "fix|ask_user|dismiss", "action": "...", "reasoning": "..."}}"""


def bonsai_doc_kind_prompt(doc_text: str) -> str:
    """Prompt for the zero-shot Bonsai doc-kind tagger (Phase 3c Sec 7.11).

    A SINGLE document's text -> one of five content-derived kinds. The tag is
    written at ingest (``Document.doc_kind``) and consumed by the
    complementary-temporal guard (``_deterministic_non_conflict``) so the guard
    fires on a SEMANTIC signal -- ``both sources are point_in_time_snapshot``
    -- instead of a filename month-prefix, which is inert on real enterprise
    docs that carry no month in their names (the EnterpriseRAG-Bench finding).

    The taxonomy (see the plan, ``mellow-jumping-token.md``):
      - ``point_in_time_snapshot`` -- a status / reading / telemetry at a
        moment in time ("as of <date>", a monthly status, a snapshot). Two of
        these with different values are COMPLEMENTARY, not a supersession.
      - ``decision_update`` -- a decision that supersedes an earlier one ("we
        switched to X", "the updated target is Y", a newer runbook). A REAL
        conflict; the guard must NOT fire.
      - ``plan`` -- a forward-looking plan / roadmap / intent.
      - ``reference`` -- a stable reference / spec / manual (timeless).
      - ``other`` -- anything else, or unclear.

    The distinguishing rule the prompt drives: is this a READING of the world
    at a moment (snapshot) or a DECISION that changes a prior state (update)?
    The model returns ONLY the two fields the tagger validates (``doc_kind`` +
    a one-sentence ``why``); ``classify_doc_kind`` returns ``None`` on a
    missing / out-of-vocabulary label so the caller writes the cold-start
    ``"other"`` default (no fabricated label).
    """
    return f"""You are classifying ONE document for a memory system. Read the text and
return its KIND -- what role the document plays, derived from its CONTENT (not
its filename). The kind is used by a contradiction guard: two documents that
are both point-in-time snapshots carrying different values are COMPLEMENTARY
(both true at their respective times), NOT a supersession.

The five kinds (pick exactly one):
- point_in_time_snapshot: a STATUS / READING / TELEMETRY at a moment in time.
  Markers: "as of <date>", a monthly/weekly status report, a metrics snapshot,
  a reading taken on a date. It describes the world AT a time, not a change.
- decision_update: a DECISION that supersedes an earlier one. Markers: "we
  switched to X", "the updated target is Y", "replacing the old policy with",
  a newer runbook that overrides the prior. It CHANGES a prior state.
- plan: a forward-looking plan / roadmap / intent (what we WILL do).
- reference: a stable reference / spec / manual / documentation (timeless, not
  dated).
- other: anything else, or unclear.

Distinguishing rule: is this a READING of the world at a moment
(point_in_time_snapshot) or a DECISION that changes a prior state
(decision_update)? A status report IS a snapshot even if it mentions a
decision; a decision memo IS a decision_update. When unsure, prefer
point_in_time_snapshot for dated status-like text and decision_update for
explicit "we changed / switched / updated" text; otherwise other.

DOCUMENT TEXT:
{doc_text}

Return ONLY valid JSON:
{{"doc_kind": "point_in_time_snapshot|decision_update|plan|reference|other",
  "why": "<one sentence>"}}"""


def bonsai_gist_prompt(source_episodes: list[dict]) -> str:
    """Prompt for the deploy-time Bonsai gist decider (spec §2.5 deploy step).

    A gist is a summary-of-summaries: one paragraph that abstracts a DiffPool
    cluster of episodes into a single semantic memory (``M:NNNN``). The caller
    pre-caps the source list (``abstract_gist_max_episodes``) so the 8B's 4096
    ctx is never blown. Each source carries ``id`` + ``summary`` + (optional)
    ``text`` (a truncated full-text excerpt). Returns ONLY the gist string
    envelope so ``BonsaiDecider._parse_json_object`` can carve it out.
    """
    lines = []
    for ep in source_episodes:
        eid = ep.get("id", "?")
        sm = ep.get("summary", "")
        tx = ep.get("text", "")
        if tx:
            lines.append(f"- [{eid}] {sm}\n  excerpt: {tx}")
        else:
            lines.append(f"- [{eid}] {sm}")
    block = "\n".join(lines)
    return f"""You are the subconscious of a memory system. A clustering pass grouped the
episodes below because they are about the same underlying thing. Write ONE tight
paragraph (3-6 sentences) that captures what the group is ABOUT -- the shared
subject, the through-line, what was decided or learned -- NOT a list. This
paragraph becomes a semantic memory that other episodes will be abstracted into,
so it must stand alone: a reader who never saw the sources should understand the
gist. Do not invent facts not supported by the sources; synthesize only.

SOURCE EPISODES:
{block}

Return ONLY valid JSON:
{{"gist": "..."}}"""


def bonsai_consolidate_gist_prompt(blurb: str, prior_gist: Optional[str],
                                   count: int) -> str:
    """Prompt for the fade consolidation gist (the gist-on-forgetting loop).

    Distinct from ``bonsai_gist_prompt`` (a summary-of-summaries over a DiffPool
    cluster): this compresses a SINGLE fading blurb in place, and -- on the
    second+ pass -- feeds the PRIOR gist back in so the compression is
    fidelity-preserving (the headroom ``learn/analyzer.py`` prior-baseline-merge
    pattern: emit a COMPLETE replacement, not a delta; preserve accurate facts;
    drop only when contradicted). ``count`` is the consolidation depth (1 = first
    gist of verbatim, 2 = gist-of-gist, ...) so the model knows how aggressively
    to compress. Returns the SAME ``{{"gist": "..."}}`` envelope as
    ``bonsai_gist_prompt`` so ``BonsaiDecider._parse_json_object`` carves it out.
    """
    if prior_gist is None:
        return f"""You are the subconscious of a memory system. A single memory is
fading from exact recall and must be compressed into a gist before it is lost.
Write ONE tight paragraph (3-6 sentences) that captures what this memory is
ABOUT -- the subject, what was decided or learned -- so a reader who never saw
the original understands the gist. Do not invent facts not supported by the
source; synthesize only. This paragraph replaces the verbatim as the memory's
recalled form, so it must stand alone.

SOURCE MEMORY:
{blurb}

Return ONLY valid JSON:
{{"gist": "..."}}"""
    return f"""You are the subconscious of a memory system. A memory has already been
compressed once into the gist below, and is fading again -- compress it FURTHER
(compression level {count}). Emit a COMPLETE new gist (NOT a delta): the full
paragraph that should replace the current one. Preserve every accurate fact;
drop a fact ONLY when the source contradicts it; shorten phrasing. Do not invent
facts. A reader who never saw the original should still understand the memory.

CURRENT GIST (compress this further):
{prior_gist}

Return ONLY valid JSON:
{{"gist": "..."}}"""


def bonsai_verify_fidelity_prompt(blurb: str, narrative: str) -> str:
    """Prompt for the fade consolidation FIDELITY JUDGE (validated compaction).

    Given the original fading blurb + a gist that the consolidator produced from
    it, decide whether the gist CORRUPTED the memory -- changed a fact's
    MEANING (inverted, swapped, or fabricated a fact) -- vs only DROPPED detail
    (paraphrase / compression, which is the intended behavior and NOT
    corruption). This is the human-in-the-loop gate: a ``corruption=true``
    verdict DEFERS the consolidation and escalates the gist to the user for
    review; ``corruption=false`` accepts it silently. ``reason`` is a short
    phrase the user sees next to the held review, so keep it concrete and
    brief. Returns the ``{{"corruption": bool, "reason": "..."}}`` envelope so
    ``BonsaiDecider._parse_json_object`` carves it out (same path as the other
    deciders).
    """
    return f"""You are the fidelity judge of a memory system. A fading memory was compressed
into the gist below. Decide whether the gist CORRUPTED the memory or only
compressed it.

Corruption = the gist changes a fact's MEANING. Check each of these, because
they are corruption even when the gist otherwise reads similarly:
  - INVERSION: a fact says the OPPOSITE of the source (e.g. source "X is
    larger" -> gist "X is smaller"; source "Alice approved" -> gist "Alice
    rejected").
  - SWAP: an attribute that belonged to one entity is reassigned to a
    DIFFERENT entity. Example -- source "Alice owns the Subaru; Bob owns the
    Toyota" -> gist "Bob owns the Subaru; Alice owns the Toyota" is
    CORRUPTION, not compression, even though the same two entities and two
    cars appear. Compare EACH named entity's attributes one by one: for every
    entity the source mentions, does the gist give it the SAME attributes, or
    did an attribute migrate to another entity?
  - FABRICATION: the gist states a specific fact (a number, name, date,
    relation) the source does not support.

Dropping detail, shortening, or paraphrasing the SAME facts is NOT corruption
-- that is the intended compression, answer corruption=false. A gist that
keeps every named fact's meaning intact, even if terser, is clean.

SOURCE MEMORY (the fading original):
{blurb}

PROPOSED GIST (the compression to judge):
{narrative}

Return ONLY valid JSON:
{{"corruption": true|false, "reason": "one short phrase naming the changed fact and how it changed, or empty when clean"}}"""


def bonsai_author_scene_prompt(
    topic: str,
    existing_body: Optional[str],
    candidate_summaries: list[str],
    user_id: str,
    heat_budget: int,
    merge_candidate: Optional[dict],
) -> str:
    """Prompt for the Bonsai scene-block AUTHORING decider (B1).

    The scene worker (``scene_worker.py``) calls this ONCE per ingest batch with
    the batch's candidate episode summaries + the topic's existing scene body
    (if any) + ONE pre-filtered merge candidate (D7). The 8B picks one of four
    actions:

    * ``CREATE`` -- no existing scene; write a fresh Markdown scene from the
      candidate episodes.
    * ``UPDATE`` -- an existing scene exists; emit a COMPLETE replacement body
      that folds in the new episodes (NOT a delta). ``topic`` MUST stay stable on
      UPDATE (the worker relies on it -- ``has_topic``/``owned_by`` edges don't
      shift, only ``cites`` churns).
    * ``MERGE`` -- the pre-filtered ``merge_candidate`` (a DIFFERENT topic's
      scene) overlaps enough to fold together; emit ONE merged body whose
      ``topic`` is the merged topic. The worker deletes the source scene (D5).
    * ``skip`` -- the candidates add nothing new; write nothing.

    Returns the ``{{"action", "topic", "body", "merge_with", "reason"}}``
    envelope so ``BonsaiDecider._parse_json_object`` carves it out (same path as
    the other deciders). Wording is OPTIONAL guidance ("you may", "if useful"),
    never imperative (per [[llm-tool-use-prompts-optional]] -- imperative wording
    suppresses small-model tool/decision chains).
    """
    import json as _json
    cand_block = "\n".join(
        f"- ({i + 1}) {s}" for i, s in enumerate(candidate_summaries)
    ) or "(none)"
    if existing_body is None:
        existing_block = "(no scene exists yet for this topic)"
    else:
        existing_block = existing_body
    if merge_candidate is None:
        merge_block = "(no merge candidate offered this round)"
    else:
        merge_block = (
            f"scene_id={merge_candidate.get('scene_id', '')}  "
            f"topic={merge_candidate.get('topic', '')}\n"
            f"{merge_candidate.get('body', '')}"
        )
    return f"""You are the macro-memory author of a personal memory system. For one
user ({user_id}) you maintain a "scene block" per topic -- a short Markdown
summary of the system's synthesized understanding of that topic, cited to the
episodes it draws from. The scene is the user's macro picture; episodes are the
raw events underneath it.

New candidate episodes have just been ingested for the topic "{topic}". You may
author or revise the scene so it reflects them. You may also fold the scene into
a pre-filtered MERGE candidate if the two topics overlap enough to be one
scene. If the candidates add nothing new, you may skip.

Choose ONE action:
  CREATE  -- no scene exists yet; write a fresh Markdown scene (4-10 lines)
           synthesizing the candidate episodes for this topic.
  UPDATE  -- a scene exists; emit a COMPLETE replacement body folding in the new
           candidates (NOT a delta). Keep the SAME topic "{topic}".
  MERGE   -- fold this topic's scene into the merge candidate below; emit ONE
           merged body whose topic names the combined topic, and set
           "merge_with" to that candidate's scene_id.
  skip    -- the candidates add nothing new; write nothing (body empty).

Rules: synthesize only from the episodes shown; do not invent facts. Keep it
tight and useful -- a later retrieval surfaces this to the user as context.
There is room for roughly {heat_budget} scene(s) of heat budget for this user
near this topic; if you are at the cap, prefer MERGE or skip over CREATE.

TOPIC: {topic}

EXISTING SCENE BODY:
{existing_block}

NEW CANDIDATE EPISODES:
{cand_block}

MERGE CANDIDATE (consider for MERGE only; ignore for CREATE/UPDATE):
{merge_block}

Return ONLY valid JSON:
{{"action": "CREATE"|"UPDATE"|"MERGE"|"skip",
  "topic": "the scene's topic (stable on UPDATE; the merged topic on MERGE)",
  "body": "the Markdown scene body, or empty when skip",
  "merge_with": "the merge candidate's scene_id, or empty",
  "reason": "one short phrase"}}"""


def bonsai_dedup_prompt(
    new_summary: str,
    new_entities: list[str],
    new_topics: list[str],
    candidates: list[dict],
) -> str:
    """Prompt for the Bonsai DEDUP judge (A1: the post-commit 4-action dedup gate).

    The dedup reconcile (``src/encoding/dedup.py``) calls this ONCE per encoded
    episode with the vector-recalled candidate pool (the user's active episodes
    nearest the new one's summary embedding). The 8B judges the new episode vs
    EACH existing one and picks one of four actions:

    * ``store`` -- the new is a genuinely NEW fact; keep both, change nothing.
    * ``update`` -- the new is a NEWER/BETTER version of the SAME fact as this
      existing episode; the new should replace the old (``supersede(new, old)``).
    * ``merge`` -- the new is COMPLEMENTARY to this existing episode (together a
      fuller picture of one fact); the new stands for the merged fact
      (``supersede(new, old)``). v1 does NOT text-fold the old's content into the
      new -- that is a deferred refinement; the old is superseded (content
      preserved, recoverable, never deleted).
    * ``skip`` -- the new is a DUPLICATE of this existing episode; discard the
      new (``supersede(existing, new)``).

    Returns the ``{{"verdicts": [...]}}`` envelope so
    ``BonsaiDecider._parse_json_object`` carves it out (it returns dict-only; a
    top-level array would return None). Each verdict carries ``pair_id`` (the
    0-based index into ``candidates``) which the caller resolves to an eid.
    Wording is OPTIONAL guidance ("you may", "if unsure prefer store"), never
    imperative (per [[llm-tool-use-prompts-optional]] -- imperative wording
    suppresses small-model decision chains). The prompt biases to ``store`` when
    unsure: a wrong ``skip`` discards the new (recoverable via MVCC, but a
    graph-thin episode); a wrong ``store`` only leaves a harmless near-dup.
    """
    import json as _json
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"- (pair_id={i}) summary: {c.get('summary', '')}\n"
            f"  entities: {_json.dumps(c.get('entities', []), ensure_ascii=False)}\n"
            f"  topics: {_json.dumps(c.get('topics', []), ensure_ascii=False)}"
        )
    cand_block = "\n".join(lines) if lines else "(none)"
    return f"""You are the dedup judge of a memory system. A new episode was just stored.
For each EXISTING episode below, decide whether the new one is a duplicate, a
newer version of the same fact, a complementary part of the same fact, or
unrelated.

For each pair, choose ONE action:
  store  -- the new episode is a genuinely NEW fact (no existing episode it
            duplicates or complements). Keep both; change nothing.
  update -- the new is a NEWER or BETTER version of the SAME fact as this
            existing episode. The new should replace the old.
  merge  -- the new is COMPLEMENTARY to this existing episode (together they
            describe one fact more fully). The new stands for the merged fact.
  skip   -- the new is a DUPLICATE of this existing episode (same fact, same
            vintage). Discard the new.

Rules: judge on FACT identity, not wording -- a paraphrase of the same fact is
skip/update, not store. Two facts about the same entity that differ in detail
are merge, not skip. If unsure, prefer store (keep both) -- a wrong skip loses
data. You may judge multiple pairs; give one verdict per pair.

NEW EPISODE:
  summary: {new_summary}
  entities: {_json.dumps(new_entities, ensure_ascii=False)}
  topics: {_json.dumps(new_topics, ensure_ascii=False)}

EXISTING EPISODES (one verdict each, by pair_id):
{cand_block}

Return ONLY valid JSON:
{{"verdicts": [{{"pair_id": 0, "action": "store"|"update"|"merge"|"skip",
   "reason": "one short phrase"}}, ...]}}"""


def bonsai_typing_prompt(entity: str, candidate_class: str,
                         retrieved_context: dict) -> str:
    """Prompt for the deploy-time Bonsai ontology-typing decider.

    Given an entity the ontology head proposes to type as ``candidate_class``
    (above ``accept_threshold``) plus the entity's retrieved neighborhood, the
    8B decides: is the typing right (``accept``), or is a NEW narrower class
    under a named parent warranted? The caller verifies the proposed ``parent``
    EXISTS in the seed ontology before creating the class (never orphans).
    """
    ctx_json = json.dumps(retrieved_context, ensure_ascii=False)
    return f"""You are the ontology judge of a memory system. An entity-typing head proposed
typing the entity below as a class. Decide whether that typing is correct, or
whether the entity is better described by a NEW narrower class under an EXISTING
parent class. The parent must already exist in the seed ontology (Person,
Project, Technology, Concept, and their subclasses); if you propose a parent
that does not exist, the system will reject the new class and record nothing.

ENTITY: {entity}
PROPOSED CLASS: {candidate_class}

RETRIEVED CONTEXT (entity's neighborhood):
{ctx_json}

Decisions:
- accept=true, new_class=null: the entity genuinely is an instance of the
  proposed class. The system will write an instanceOf edge.
- accept=true, new_class="<name>", parent="<existing class>": the entity is an
  instance of a NEW narrower class under an existing parent. The system creates
  the class then writes the instanceOf edge.
- accept=false: the typing is wrong. The system records nothing.

Return ONLY valid JSON:
{{"accept": true|false, "new_class": "... or null", "parent": "... or null", "reasoning": "..."}}"""


def bonsai_query_planning_prompt(conversation_text: str, question: str) -> str:
    """Prompt for generating Bonsai query planning training pairs."""
    return f"""You are generating training data for a query planner that converts
natural language questions into structured memory queries.

CONVERSATION:
{conversation_text}

HYPOTHETICAL QUESTION: {question}

The memory graph stores episodes with these attributes:
- entities: [Person, Project, Technology, Concept]
- topics: [database_design, configuration, graph_database, performance,
           decision_making, ai_architecture, api_design, security]
- tones: [frustrated, excited, curious, neutral]
- decisions: specific choices made

Query parameters:
- entities: list of entities to search for
- topics: list of topics to filter by
- tones: list of emotional tones to filter by
- entity_mode: "intersection" (ALL entities) or "union" (ANY entity)
- temporal_after: keyword to find anchor and follow chain forward, or null
- temporal_before: keyword to find anchor and follow chain backward, or null
- temporal_filter: "today", "this_week", "last_week", "this_month", or null
- date_from: ISO date for start of range, or null
- date_to: ISO date for end of range, or null
- limit: max episodes to return (default 5)

IMPORTANT RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union"
- "What did Alice and Bob disagree about?" → entities=["Alice", "Bob"], entity_mode="intersection"
- "What happened after morphisms?" → temporal_after="morphism"
- "Why did we choose X over Y?" → topics=["decision_making"], entities=["X", "Y"], entity_mode="union"

Return ONLY valid JSON:
{{"question": "{question}",
 "query": {{"entities": [], "topics": [], "tones": [], "entity_mode": "union",
           "temporal_after": null, "temporal_before": null,
           "temporal_filter": null, "date_from": null, "date_to": null,
           "limit": 5}},
 "reasoning": "Brief explanation of why these parameters were chosen"}}"""


def bonsai_relation_extraction_prompt(conversation_text: str) -> str:
    """Prompt for generating Bonsai relation extraction training pairs."""
    return f"""Extract relationships from this conversation. Return ONLY valid JSON.

Relation types:
- explains(Person, Concept): someone explains something
- decides(Person, Decision): someone makes a decision
- expresses(Person, Tone): someone expresses an emotion
- questions(Person, Concept): someone asks about something
- suggests(Person, Concept): someone proposes an idea
- concerns(Episode, Topic): the conversation is about a topic
- involves(Episode, Entity): an entity participates
- contradicts(Statement, Statement): one statement contradicts another
- follows_up_on(Episode, Episode): this conversation continues from another

CONVERSATION:
{conversation_text}

Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""


# ═══════════════════════════════════════════════════════════════
# JEPA ROUTING DATA
# ═══════════════════════════════════════════════════════════════


def jepa_routing_prompt(prompt: str, available_domains: str,
                        available_pathways: str) -> str:
    """Prompt for generating JEPA routing training pairs."""
    return f"""You are generating training data for a subconscious router that
decides how to handle a user's query before any retrieval or generation.

USER QUERY: {prompt}

AVAILABLE DOMAINS:
{available_domains}

AVAILABLE PATHWAYS:
{available_pathways}

MODEL SIZES: 1B, 3B, 8B, 70B, 175B

META-SKILLS: factual_recall, basic_synthesis, pattern_recognition,
             decomposition, process_selection, creative_synthesis,
             security_analysis, tradeoff_analysis

Decide:
1. Which domain(s) to query?
2. Which pathway to use?
3. What meta-skills are required?
4. What model size is needed?
5. Is conscious deliberation needed?

Return ONLY valid JSON:
{{"domains": ["database"],
 "pathway": "graph_retrieve",
 "meta_skills": ["factual_recall", "basic_synthesis"],
 "model_size": "3B",
 "needs_deliberation": false,
 "confidence": 0.89,
 "reasoning": "Brief explanation"}}"""


# ═══════════════════════════════════════════════════════════════
# GATE TRAINING DATA
# ═══════════════════════════════════════════════════════════════


def uncertainty_detector_prompt(context: str, query: str,
                                retrieval_results: str) -> str:
    """Prompt for Uncertainty Detector gate training."""
    return f"""You are generating training data for an uncertainty detector.
Given a query, the retrieved context, and what the system knows, determine
whether the system should flag uncertainty.

CONTEXT (what the system knows):
{context}

USER QUERY: {query}

RETRIEVAL RESULTS:
{retrieval_results}

Should the system flag uncertainty? Consider:
- Is the retrieved context sufficient to answer the query?
- Are there novel entities not in the ontology?
- Are there contradictions in the retrieved results?
- Is the routing confidence low?

Return ONLY valid JSON:
{{"should_flag": true/false,
 "uncertainty_type": "routing_uncertainty|novel_entity|unresolved_contradiction|none",
 "confidence": 0.0-1.0,
 "reasoning": "Brief explanation"}}"""


def aspirational_model_prompt(goal_context: str, candidate_action: str) -> str:
    """Prompt for Aspirational Model gate training."""
    return f"""You are generating training data for an aspirational model.
Given the agent's current goals and a candidate action, determine whether
the agent should commit to this action.

CURRENT GOALS AND CONTEXT:
{goal_context}

CANDIDATE ACTION: {candidate_action}

Consider:
- Does this align with known goals?
- Is the expected value worth the effort?
- Is this a novel opportunity or a routine action?
- Should a prospective trigger be set?

Return ONLY valid JSON:
{{"should_commit": true/false,
 "encoding_strength": 0.0-1.0,
 "set_prospective_trigger": true/false,
 "trigger_condition": "description or null",
 "reasoning": "Brief explanation"}}"""


def self_model_prompt(knowledge_state: str, query: str) -> str:
    """Prompt for Self-Model gate training."""
    return f"""You are generating training data for a self-model that estimates
its own knowledge boundaries.

KNOWLEDGE STATE:
{knowledge_state}

USER QUERY: {query}

Should the system say "I don't know" or attempt to answer?
Consider:
- Is the knowledge in this domain dense or sparse?
- Is the specific fact likely to be known?
- Would answering risk hallucination?

Return ONLY valid JSON:
{{"should_say_dont_know": true/false,
 "confidence_in_answer": 0.0-1.0,
 "knowledge_boundary_hit": true/false,
 "reasoning": "Brief explanation"}}"""


def common_sense_resolver_prompt(
    input_text: str,
    retrieved_context: str,
    candidate_interpretations: str,
) -> str:
    """Prompt for Common Sense Resolver (CSR) gate training (Phase 4d).

    The CSR resolves AMBIGUITY before the system commits to an action: it
    detects ambiguity in the input / query / retrieved results, evaluates the
    candidate interpretations the world-model graph surfaced, and either picks
    the best-supported interpretation or asks the user for clarification (the
    "I need to get to the bank" -> financial institution vs. riverbank case,
    resolved by context). This is the 4th gate alongside Uncertainty Detector,
    Aspirational Model, and Self-Model (master roadmap sec 198: 50k each).

    Input shape mirrors the deploy-time CSR: ``input_text`` is the user turn
    or retrieved passage that may be ambiguous, ``retrieved_context`` is the
    surrounding memory (recent turns + retrieved episodes) that disambiguates,
    and ``candidate_interpretations`` is the list of readings the world-model
    graph proposed (e.g. ["financial institution", "riverbank"] for "bank").
    Returns the formatted prompt string (the body is built by the f-string
    below; the checks run IN ORDER and default to ask_clarification when the
    context does not disambiguate).
    """
    return f"""You are generating training data for a Common Sense Resolver: a gate that
detects AMBIGUITY in an input, query, or retrieved result and resolves it BEFORE
the system commits to an action. It evaluates candidate interpretations the
world-model graph surfaced and either selects the best-supported reading or asks
the user for clarification (the "I need to get to the bank" -> financial
institution vs. riverbank case, resolved by context).

Decide by running these checks IN ORDER. The FIRST check that matches gives the
decision; stop there. Default to should_ask_clarification=true when no
interpretation is clearly supported -- the CSR must NOT silently guess when the
context does not disambiguate.
- Check 1 -- NOT AMBIGUOUS: the input has only one sensible reading (the
  candidates list is empty or all candidates collapse to the same meaning)
  -> is_ambiguous=false, interpretation = that reading,
  should_ask_clarification=false.
- Check 2 -- CONTEXT RESOLVES: the retrieved context clearly supports one
  candidate over the others (a recent turn / retrieved episode mentions
  money, so "bank" = financial institution) -> is_ambiguous=true,
  interpretation = the supported candidate, should_ask_clarification=false.
- Check 3 -- GENUINELY AMBIGUOUS: multiple candidates remain plausible and
  the context does not favor one -> is_ambiguous=true, interpretation=null,
  should_ask_clarification=true (ask the user; do not guess).

INPUT (the user turn / retrieved passage that may be ambiguous):
{input_text}

RETRIEVED CONTEXT (recent turns + retrieved episodes that disambiguate):
{retrieved_context}

CANDIDATE INTERPRETATIONS (readings the world-model graph proposed):
{candidate_interpretations}

Decide whether the input is ambiguous and which interpretation (if any) the
context supports, then explain it. Default to should_ask_clarification=true
when the context does not clearly favor one candidate.

Return ONLY valid JSON:
{{"is_ambiguous": true|false,
  "interpretation": "<the supported reading, or null if genuinely ambiguous>",
  "candidates": ["<candidate 1>", "<candidate 2>", "..."],
  "should_ask_clarification": true|false,
  "confidence": 0.0,
  "reasoning": "Brief explanation"}}"""


# ═══════════════════════════════════════════════════════════════
# CODE-AWARE SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════


def code_aware_synthetic_prompt(domain: str, code_ontology_fragment: str) -> str:
    """Prompt for generating synthetic code-aware training examples."""
    return f"""You are generating synthetic training data for a memory system
that needs to learn about code structure before any real code is parsed.

DOMAIN: {domain}

CODE ONTOLOGY (available types):
{code_ontology_fragment}

Generate a realistic conversation about software development that includes
code artifacts. Then extract structured triples using the code ontology types.

Return ONLY valid JSON:
{{"conversation": "User: ... Assistant: ...",
 "extracted_entities": ["auth.py", "authenticate_user", "JWT", ...],
 "extracted_topics": ["security", "api_design"],
 "extracted_relations": [
    {{"subject": "auth.py", "predicate": "contains", "object": "authenticate_user"}},
    {{"subject": "authenticate_user", "predicate": "calls", "object": "validate_token"}}
 ],
 "code_artifacts": [
    {{"type": "File", "name": "auth.py"}},
    {{"type": "Function", "name": "authenticate_user", "defined_in": "auth.py"}}
 ]}}"""


# ═══════════════════════════════════════════════════════════════
# STRM 1f-6: CODE-SECTION PROSE SUMMARY (embedding handle for code docs)
# ═══════════════════════════════════════════════════════════════


def code_section_summary_prompt(heading: str, content: str) -> str:
    """Prompt for the STRM 1f-6 code-section prose summary.

    A code section's *embedding handle* (what the document retriever + the STRM
    backbone state locate it by) is computed from the raw source today, which
    mis-ranks against prose queries -- the 1f-5 doc-kind split showed code-doc
    z_logit median -0.801 vs text-doc 3.969 ([[pondr-strm-phase1f5-margin-dockind-
    result]]). This prompt asks the Oracle for a 1-2 sentence plain-prose
    description of WHAT the function/class does and its role, so the embedding
    can be computed from prose instead. The summary is an *embedding handle*
    ONLY -- the actual source stays as the recalled content (the ``text`` field
    is untouched; see the 1f-6 plan). Returns ONLY the JSON envelope so
    ``OracleClient``'s ``response_format: json_object`` + salvage parser can
    recover it. ``content`` is capped by the caller to keep the prompt bounded.
    """
    body = content[:4000]
    return f"""You are describing one section of a source file for a retrieval system that
finds code by MEANING, not by symbol name. Read the signature and source below
and write 1-2 plain sentences describing what this {heading or "code section"}
does and its role or purpose -- its behavior, what it takes in, what it returns
or changes, and why a program might call it. Write as if explaining to an
engineer who is searching by intent ("how do I ..."). Do NOT include code, do
NOT restate the signature, do NOT use bullet points. Be concrete, not generic.

SOURCE:
{body}

Return ONLY valid JSON:
{{"summary": "..."}}"""


def bonsai_judge_task_lifecycle_prompt(
    user_id: str,
    recent_messages: list[dict],
    active_canvas: Optional[dict],
    historical_canvases: list[dict],
) -> str:
    """Prompt for the Bonsai L1.5 task-canvas LIFECYCLE gate (B4).

    The orchestrator calls this ONCE per turn (synchronous, before synthesis)
    to decide create/switch/clear/keep for the active task canvas. The 8B picks
    one of five lifecycle judgments, emitted as a FLAT JSON object (single
    object, like ``bonsai_author_scene_prompt`` -- NOT a list, so no
    ``{{"tasks":[...]}}`` envelope trick is needed; ``_parse_json_object``'s
    dict-only guard accepts it):

    * ``taskCompleted`` -- the active task is done; clear it (flip to historical).
    * ``isLongTask`` -- this turn starts/resumes a long-horizon task that
      warrants a canvas (create one if none is active).
    * ``isContinuation`` + ``continuationCanvasId`` -- this turn resumes a
      PRIOR task; switch the active canvas to that historical id.
    * ``newTaskLabel`` -- a kebab-case label <=30 chars for a newly created
      canvas (empty when not creating).

    Inputs: the last ~6 ``conversation_history`` messages, the active canvas
    (id+label+progress or ``None``), the most recent ~10 historical canvases by
    ``updated_ts`` (id+label+progress, for the resume decision). Wording is
    OPTIONAL guidance ("you may"), never imperative (per
    [[llm-tool-use-prompts-optional]]). The orchestrator's deterministic
    ``apply_lifecycle`` maps the 5-field judgment to store transitions.
    """
    import json as _json
    # Recent messages: last 6, rendered as "role: text" (text truncated).
    msg_lines = []
    for m in (recent_messages or [])[-6:]:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        text = (m.get("content", "") if isinstance(m, dict) else str(m)) or ""
        msg_lines.append(f"- {role}: {text[:300]}")
    msg_block = "\n".join(msg_lines) or "(no prior messages)"
    if active_canvas is None:
        active_block = "(no active canvas)"
    else:
        active_block = (
            f"canvas_id={active_canvas.get('canvas_id', '')}  "
            f"label={active_canvas.get('task_label', '')}  "
            f"progress={active_canvas.get('progress', 0)}%")
    if not historical_canvases:
        hist_block = "(no historical canvases to resume)"
    else:
        hist_lines = [
            f"- {c.get('canvas_id', '')} (label={c.get('task_label', '')}, "
            f"progress={c.get('progress', 0)}%)"
            for c in historical_canvases[:10]
        ]
        hist_block = "\n".join(hist_lines)
    return f"""You are the task-lifecycle gate of a personal memory system. For one
user ({user_id}) the system maintains an OPTIONAL "task canvas" -- a per-task
Mermaid flowchart of the phases of a long-horizon task (done|doing|paused|
blocked). At most ONE canvas is active at a time. Your job: decide THIS turn's
lifecycle action from the recent conversation + the canvas state.

You may decide the task is complete, that this turn is casual (no long task),
that a long task is starting, or that the user is resuming a prior task. If you
are unsure whether a long task is underway, prefer keeping the current state
(no change) over creating or clearing.

Decide:
  taskCompleted (bool)    -- the active task is finished; the system will clear
                             it (flip to historical). True only if an active
                             canvas exists AND the user clearly wrapped it up.
  isLongTask (bool)       -- this turn is part of a long-horizon task that
                             warrants a canvas. False for casual chat / one-shot
                             questions. If True and no canvas is active, the
                             system creates one with newTaskLabel.
  isContinuation (bool)   -- the user is resuming a PRIOR task listed in the
                             historical canvases below. Set
                             continuationCanvasId to that canvas's id.
  continuationCanvasId    -- the historical canvas_id to resume (empty unless
                             isContinuation is True).
  newTaskLabel            -- a kebab-case label <=30 chars for a newly created
                             canvas (lowercase, hyphens, no spaces). Empty when
                             not creating.

Precedence: taskCompleted > isContinuation > isLongTask(create) > keep.

RECENT CONVERSATION:
{msg_block}

ACTIVE CANVAS:
{active_block}

HISTORICAL CANVASES (resumable):
{hist_block}

Return ONLY valid JSON:
{{"taskCompleted": false,
  "isLongTask": false,
  "isContinuation": false,
  "continuationCanvasId": "",
  "newTaskLabel": ""}}"""