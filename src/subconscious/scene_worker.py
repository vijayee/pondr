"""Background authoring worker for scene blocks (B1).

A scene block is the system's synthesized understanding of one TOPIC for a user
-- an LLM-authored Markdown summary stored IN WaveDB (``content/scene/{id}``,
NOT a file on disk). It is the deployable, no-training symbolic macro layer
tier 1 lacked AND the substrate the trained-but-unwired GNN scene-ontology head
will later write onto ([[pondr-gnn-already-does-ontology]]). Authored on ingest
by the Bonsai LLM with four actions (CREATE/UPDATE/MERGE/skip); heat decays per
tick and cold scenes are evicted = macro-forgetting (the scene-level analog of
fade R4).

Modeled line-for-line on ``ConsolidationWorker`` (``consolidation_worker.py``):
a single daemon ``threading.Thread`` + ``queue.Queue`` + a SHARED
``foreground_busy`` ``threading.Event`` priority gate (D3 -- the orchestrator
assigns its OWN event at construction so all background workers yield to the
foreground together). The orchestrator calls ``tick()`` at the tail of each
query (read-only -- it builds ``(user_id, topic_hint, [ep_ids])`` from the
just-persisted batch and enqueues); the worker thread processes the queue
BETWEEN turns (the gate blocks while ``foreground_busy`` is set). One LLM call
per ingest BATCH (not per turn, not per episode) -- the macro authoring cost is
amortized over the batch.

Four-action gate (mirrors the Phase C three-state gate's "never a silent auto-
write"): the decider returns ``None`` (Bonsai down / parse fail / unrecognized
action) OR ``skip`` -> the worker DEFERS, never writes a scene. CREATE/UPDATE/
MERGE each apply atomically (one ``encode_scene``/``delete_scene`` batch each).
MERGE deletes the source scene (D5) and is offered ONE pre-filtered target (D7).

Failure semantics: a per-job exception is logged and the batch is skipped --
the queue survives, the next turn still authors. Cold-start honest: ``None``
is a skip, not a fabricated scene.

Flag-gated (``--scene-blocks``) default OFF: when the flag is off, no
``SceneAuthoringWorker`` is constructed (the ``build_ponder`` gate) -> no
``tick()`` -> no scene writes -> ``default_scene_ids()`` is empty -> byte-
identical to pre-B1.
"""

from __future__ import annotations

import sys
import threading
import queue
from datetime import datetime
from typing import Optional, Any

from ..memory.store import HippocampalStore


_SCENE_SENTINEL = object()  # signals the worker thread to exit (drain)


def _now_iso() -> str:
    """Naive-local ISO timestamp for the ``updated_ts`` field, MATCHING the
    episode ``timestamp`` format (``datetime.now().isoformat()`` in
    ``episode.py:178``/``:234`` -- naive, no Z-suffix). The recency sort in
    ``GraphTraversal._score_candidates`` (``graph_traversal.py:949``) parses
    every result's timestamp into ONE list and sorts it; mixing a naive
    (episode) with an aware (Z-suffix scene) datetime raises ``TypeError``.
    Scenes MUST share the episode format so a scene + episode in the same
    result set sort cleanly."""
    return datetime.now().isoformat()


class SceneAuthoringWorker:
    """A single-worker background FIFO that authors + maintains scene blocks.

    Constructed by ``build_ponder`` when ``scene_blocks`` is on. The
    orchestrator assigns the SHARED ``foreground_busy`` event (D3) after
    construction (mirrors the ConsolidationWorker wiring). The worker reads it
    via ``_wait_foreground`` before each authoring step (the store mutation).

    Args:
        store: the HippocampalStore (scene CRUD lives here).
        decider: a ``BonsaiDecider`` (or compatible) exposing
            ``author_scene(...)``. ``default_scene_author().decider`` is the
            canonical source (``gister.py``).
        embedder: the WM bge-small embedder (384-d) used to embed the LLM-
            authored body before ``encode_scene`` so scenes are retrievable via
            the vector index (D1). ``None`` -> scenes index on the graph topics
            axis only (no vector path); still functional, just less recall.
        max_scenes_per_user: the per-user ``maxScenes`` cap. At cap, a CREATE
            evicts the coldest scene first.
        heat_floor: below this heat a scene is evicted on the decay sub-pass.
        touch_bump: heat added on UPDATE / retrieval (clamped to [0, 1]).
        heat_decay: multiplicative factor applied per tick (macro-forgetting).
        max_episodes_per_batch: cap on candidate episodes fed to one author
            call (caps the prompt size).
        max_evict_per_tick: bound on evictions per decay sub-pass so a large
            scene store can't stall the gap between turns.
    """

    def __init__(
        self,
        store: HippocampalStore,
        decider: Any,
        embedder: Optional[Any] = None,
        max_scenes_per_user: int = 24,
        heat_floor: float = 0.05,
        touch_bump: float = 0.2,
        heat_decay: float = 0.95,
        max_episodes_per_batch: int = 8,
        max_evict_per_tick: int = 16,
    ) -> None:
        self._store = store
        self._decider = decider
        self._embedder = embedder
        self.max_scenes_per_user = int(max_scenes_per_user)
        self.heat_floor = float(heat_floor)
        self.touch_bump = float(touch_bump)
        self.heat_decay = float(heat_decay)
        self.max_episodes_per_batch = int(max_episodes_per_batch)
        self.max_evict_per_tick = int(max_evict_per_tick)
        # Queue entries: ``(user_id, topic_hint, [ep_ids])``.
        self._q: "queue.Queue[tuple]" = queue.Queue()
        # Set by the orchestrator while a foreground query() is busy. The worker
        # assigns the orchestrator's SAME event (D3) so it yields together with
        # the consolidation + distill workers. Until assigned, a fresh event
        # (never set) keeps the worker running.
        self.foreground_busy = threading.Event()  # orchestrator reassigns (D3)
        # Dedup of in-flight + recently-enqueued (user_id, topic_hint) so a fast
        # stream of batches for the same topic doesn't queue N author calls that
        # each re-read the same pre-merge state. Mirrors ConsolidationWorker's
        # ``pending_ids`` skip (``consolidation_worker.py:125-129``).
        self._pending: set[tuple] = set()
        self._thread = threading.Thread(
            target=self._run, name="ponder-scene-worker", daemon=True
        )
        self._stopped = False
        self._thread.start()

    # -- foreground-priority yielding (mirrors ConsolidationWorker._wait_foreground) --

    def _wait_foreground(self) -> None:
        """Block while a foreground query() is busy. Called before each
        authoring step so the store mutation never races retrieval."""
        while self.foreground_busy.is_set() and not self._stopped:
            self.foreground_busy.wait(timeout=0.5)

    # -- the sweep (read-only; safe during foreground) --

    def tick(self, user_id: Optional[str], batch_episode_ids: list[str],
             topic_hint: Optional[str] = None) -> int:
        """Enqueue one authoring job for the just-persisted batch. Called by the
        orchestrator at the tail of ``query()`` (after the persist). Read-only
        on the store (the orchestrator passes the batch's episode ids + the
        union topic hint), so it is safe to run while ``foreground_busy`` is set.

        ``topic_hint`` is the union of the batch episodes' ``topics`` (the
        orchestrator computes it). A ``None`` user_id OR empty batch OR empty
        topic hint is a no-op (scenes are user-owned topic macro-memories; no
        topic -> no scene). Dedup on ``(user_id, topic_hint)`` so a rapid stream
        of same-topic batches coalesces (mirrors the consolidation ``pending_ids``
        skip). Returns the number of jobs enqueued this tick (0 or 1).
        """
        if not user_id or not batch_episode_ids or not topic_hint:
            return 0
        key = (user_id, topic_hint)
        if key in self._pending:
            return 0
        self._pending.add(key)
        self._q.put((user_id, topic_hint, list(batch_episode_ids)))
        return 1

    # -- the worker loop --

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is _SCENE_SENTINEL:
                self._q.task_done()
                return
            user_id, topic_hint, ep_ids = job
            try:
                # Gate the mutation: never author while a foreground query is
                # reading the store / graph / vector index. Blocks until the gap
                # between turns.
                self._wait_foreground()
                self._author_one(user_id, topic_hint, ep_ids)
            except Exception as e:  # noqa: BLE001 - never kill the queue
                print(f"[scene-worker-fail] {user_id}/{topic_hint}: {e}",
                      file=sys.stderr)
            finally:
                self._pending.discard((user_id, topic_hint))
                self._q.task_done()
        # NB: the decay/eviction sub-pass runs in ``tick``'s foreground-clear
        # window (see ``decay_tick``), NOT here, so a long authoring queue can't
        # stall eviction behind N author calls.

    def _author_one(self, user_id: str, topic_hint: str,
                    ep_ids: list[str]) -> None:
        """Author/revise ONE scene for ``(user_id, topic_hint)`` from the batch.

        Mirrors the ConsolidationWorker four-action gate: ``None``/``skip`` ->
        defer (never a silent auto-write). The decider is offered the existing
        scene body (if any), the candidate episode summaries (capped), the
        per-user heat budget, and ONE pre-filtered merge candidate (D7).
        """
        existing = self._find_existing_scene(user_id, topic_hint)
        existing_body = existing["body"] if existing else None
        candidate_summaries = self._load_candidate_summaries(ep_ids)
        if not candidate_summaries and existing is None:
            return  # nothing to author from and no scene to revise
        user_scenes = self._store.scene_ids_for_user(user_id)
        heat_budget = max(0, self.max_scenes_per_user - len(user_scenes))
        merge_candidate = self._pick_merge_candidate(user_id, topic_hint,
                                                     user_scenes, existing)
        verdict = self._decider.author_scene(
            topic_hint, existing_body, candidate_summaries, user_id,
            heat_budget, merge_candidate)
        if verdict is None:
            return  # cold-start / unrecognized action -> defer, no write
        action = verdict.get("action")
        if action == "skip" or not verdict.get("body"):
            return  # skip -> defer, no write
        if action == "CREATE":
            self._create(user_id, verdict, ep_ids, user_scenes)
        elif action == "UPDATE":
            self._update(user_id, verdict, existing, ep_ids)
        elif action == "MERGE":
            self._merge(user_id, verdict, existing, ep_ids, merge_candidate)
        # unknown action -> defer (never auto-write)

    # -- the four actions --

    def _create(self, user_id: str, verdict: dict, ep_ids: list[str],
                user_scenes: set) -> None:
        """CREATE a fresh scene. Enforce ``maxScenes``: at the cap, evict the
        coldest scene first (lowest heat). Heat starts at 1.0 (warmest)."""
        if len(user_scenes) >= self.max_scenes_per_user:
            self._evict_coldest(user_id, user_scenes, n=1)
        scene_id = self._store.next_scene_id()
        body = verdict.get("body") or ""
        emb = self._embed_body(body)
        self._store.encode_scene(
            scene_id, body=body, topic=verdict.get("topic") or "",
            heat=1.0, updated_ts=_now_iso(), user_id=user_id,
            source_eps=ep_ids, body_embedding=emb)

    def _update(self, user_id: str, verdict: dict, existing: Optional[dict],
                ep_ids: list[str]) -> None:
        """UPDATE an existing scene in place (idempotent overwrite). Topic stays
        STABLE (use the existing scene's topic -- the prompt constrains this, and
        the worker enforces it so ``has_topic``/``owned_by`` edges never shift;
        only ``cites`` grows). ``source_eps`` is the UNION of old + new (cites
        only ever grows, so no orphan edges from a put). Heat bumps + bounds at 1.0."""
        if existing is None:
            # No existing scene to UPDATE -- treat as a CREATE so a mislabeled
            # verdict still lands the content (never silently drop).
            self._create(user_id, verdict, ep_ids,
                         self._store.scene_ids_for_user(user_id))
            return
        scene_id = existing["scene_id"]
        body = verdict.get("body") or ""
        emb = self._embed_body(body)
        # Union the cited episodes (old + new); cites only grows -> no orphan.
        src = list(dict.fromkeys(
            list(existing.get("source_eps") or []) + list(ep_ids)))
        heat = min(1.0, float(existing.get("heat") or 0.0) + self.touch_bump)
        # Topic held stable on UPDATE (existing topic) so has_topic/owned_by
        # edges don't shift -- the prompt asks for this, the worker enforces it.
        self._store.encode_scene(
            scene_id, body=body, topic=existing.get("topic") or "",
            heat=heat, updated_ts=_now_iso(), user_id=user_id,
            source_eps=src, body_embedding=emb)

    def _merge(self, user_id: str, verdict: dict, existing: Optional[dict],
               ep_ids: list[str], merge_candidate: Optional[dict]) -> None:
        """MERGE the topic's scene into the pre-filtered target (D5 + D7). The
        LLM's merged body folds onto the TARGET scene; ``source_eps`` is the
        union of target + source (the topic's existing scene, if any) + new;
        the SOURCE scene (the topic's existing scene) is DELETED (D5 -- the
        merged body supersedes both). If the verdict's ``merge_with`` doesn't
        match the offered candidate, defer (never write to an unoffered target)."""
        target_id = verdict.get("merge_with") or ""
        # D7: MERGE is allowed ONLY into the ONE pre-filtered candidate the
        # worker offered. If no candidate was offered (``merge_candidate is
        # None``) OR the verdict's ``merge_with`` is empty OR doesn't match the
        # offered candidate, defer -- never write to an unoffered/invented target.
        if (merge_candidate is None or not target_id
                or target_id != merge_candidate.get("scene_id")):
            return  # defer (no write)
        target = self._store.get_scene(target_id)
        if target is None:
            return  # target vanished (evicted between offer and apply) -> defer
        body = verdict.get("body") or ""
        emb = self._embed_body(body)
        src = list(dict.fromkeys(
            list(target.get("source_eps") or [])
            + list((existing or {}).get("source_eps") or [])
            + list(ep_ids)))
        heat = min(1.0, float(target.get("heat") or 0.0) + self.touch_bump)
        self._store.encode_scene(
            target_id, body=body, topic=verdict.get("topic") or "",
            heat=heat, updated_ts=_now_iso(), user_id=user_id,
            source_eps=src, body_embedding=emb)
        # D5: delete the SOURCE scene (the topic's existing scene). The merged
        # body supersedes it; symmetric to fade consolidation's in-place replace.
        # No orphan edges (delete_scene is the symmetric-reversal chokepoint).
        if existing is not None and existing.get("scene_id") != target_id:
            self._store.delete_scene(existing["scene_id"])

    # -- helpers --

    def _find_existing_scene(self, user_id: str, topic_hint: str) -> Optional[dict]:
        """The user's scene for ``topic_hint`` (case-insensitive topic match), or
        ``None``. Scans the user's owned scenes via the ``owns_scene`` SPO index."""
        hint = (topic_hint or "").strip().lower()
        if not hint:
            return None
        for sid in self._store.scene_ids_for_user(user_id):
            sc = self._store.get_scene(sid)
            if sc is None:
                continue
            if (sc.get("topic") or "").strip().lower() == hint:
                return sc
        return None

    def _load_candidate_summaries(self, ep_ids: list[str]) -> list[str]:
        """Episode summaries for the batch (capped at ``max_episodes_per_batch``
        to bound the prompt size). Skips ids that aren't episodes / are missing."""
        out: list[str] = []
        for eid in ep_ids:
            if not isinstance(eid, str) or not eid.startswith("ep_"):
                continue
            ep = self._store.get_episode(eid)
            if ep is None or not (ep.summary or "").strip():
                continue
            out.append(ep.summary)
            if len(out) >= self.max_episodes_per_batch:
                break
        return out

    def _pick_merge_candidate(self, user_id: str, topic_hint: str,
                              user_scenes: set, existing: Optional[dict]) -> Optional[dict]:
        """Pre-filter ONE merge candidate (D7): the user's OTHER scene whose
        topic overlaps ``topic_hint`` most (lexical token overlap; ties broken by
        higher heat). Returns ``None`` when the user has no other overlapping
        scene (no MERGE to offer -> the model picks CREATE/UPDATE/skip). The
        offered candidate is the ONLY merge target; ``_merge`` rejects a verdict
        that merges into an unoffered scene."""
        hint_tokens = _tokens(topic_hint)
        if not hint_tokens:
            return None
        existing_id = existing.get("scene_id") if existing else None
        best: Optional[dict] = None
        best_score = 0.0
        for sid in user_scenes:
            if sid == existing_id:
                continue
            sc = self._store.get_scene(sid)
            if sc is None:
                continue
            toks = _tokens(sc.get("topic") or "")
            if not toks:
                continue
            overlap = len(hint_tokens & toks)
            if overlap == 0:
                continue
            score = overlap + float(sc.get("heat") or 0.0)  # ties -> hotter
            if score > best_score:
                best_score = score
                best = sc
        return best

    def _embed_body(self, body: str) -> Optional[list]:
        """Embed the LLM-authored body for the vector index (D1). ``None`` when
        no embedder is configured (scenes index on the graph topics axis only)."""
        if self._embedder is None or not (body or "").strip():
            return None
        try:
            return list(self._embedder.encode([body])[0])
        except Exception as e:  # noqa: BLE001 - vector index is best-effort
            print(f"[scene-embed-fail] {e}", file=sys.stderr)
            return None

    def _evict_coldest(self, user_id: str, user_scenes: set, n: int = 1) -> None:
        """Evict the ``n`` coldest (lowest-heat) of the user's scenes to make
        room at the ``maxScenes`` cap. Used by ``_create``. Each eviction goes
        through ``delete_scene`` (symmetric-reversal + unindex -- no orphans)."""
        if n <= 0 or not user_scenes:
            return
        scored: list[tuple[float, str]] = []
        for sid in user_scenes:
            sc = self._store.get_scene(sid)
            if sc is None:
                continue
            scored.append((float(sc.get("heat") or 0.0), sid))
        scored.sort(key=lambda t: t[0])  # coldest first
        for _, sid in scored[:n]:
            self._store.delete_scene(sid)

    # -- heat decay + eviction sub-pass (macro-forgetting) --

    def decay_tick(self) -> int:
        """Multiply every scene's heat by ``heat_decay`` and evict any below
        ``heat_floor``. THE macro-forgetting analog of fade R4. Call this in a
        foreground-clear window (NOT on the hot authoring path) -- e.g. the
        orchestrator may call it alongside ``tick`` at the persist tail. Evictions
        are bounded by ``max_evict_per_tick`` so a large scene store can't stall
        the gap between turns. Returns the number of scenes evicted.

        Decay is a multiplicative RMW via ``touch_scene`` (``delta = heat *
        (decay - 1)``); the clamp in ``touch_scene`` keeps heat in [0, 1]."""
        if self._stopped:
            return 0
        evicted = 0
        for sid in self._store.default_scene_ids():
            sc = self._store.get_scene(sid)
            if sc is None:
                continue
            heat = float(sc.get("heat") or 0.0)
            if heat > 0.0:
                # Multiplicative decay as an additive delta (clamped by touch_scene).
                self._store.touch_scene(sid, delta=heat * (self.heat_decay - 1.0))
            if heat * self.heat_decay < self.heat_floor:
                if evicted >= self.max_evict_per_tick:
                    continue  # bound: finish remaining next tick
                self._store.delete_scene(sid)
                evicted += 1
        return evicted

    # -- teardown --

    def drain(self, timeout: Optional[float] = 5.0) -> bool:
        """Stop accepting new work, finish in-flight + queued authorings, join
        the worker thread. Returns True if the thread joined within ``timeout``.
        Called from the orchestrator's teardown hook (mirrors
        ``ConsolidationWorker.drain``)."""
        if self._stopped:
            return True
        self._stopped = True
        # Wake a blocked _wait_foreground so the worker can exit.
        self.foreground_busy.clear()
        self._q.put(_SCENE_SENTINEL)
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()


def _tokens(s: str) -> set:
    """Lowercased whitespace/punctuation token set for the merge-candidate
    overlap heuristic. Kept tiny + dependency-free (the merge pre-filter is a
    cheap lexical gate, not a semantic similarity -- the LLM makes the real
    merge decision)."""
    out: set = set()
    for tok in (s or "").lower().replace("-", " ").replace("_", " ").split():
        if len(tok) >= 2:
            out.add(tok)
    return out