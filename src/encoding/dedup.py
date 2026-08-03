"""A1: LLM-judged 4-action dedup reconcile (Tencent-survey Phase 1 item 4).

Pondr had **no cross-episode dedup** anywhere in the encode path -- every
extracted episode was stored as-is, so the corpus accumulated near-duplicate
facts (the user repeats themselves, the same fact is re-extracted phrased
differently, a correction supersedes an old claim). This module closes that gap
with a **post-commit reconcile**: after a new episode is fully encoded, vector-
recall the user's active corpus for near-duplicates, ONE BATCHED Bonsai LLM call
judges every new-vs-existing pair (``store``/``update``/``merge``/``skip``), and
the verdict is applied via ``SemanticMemoryWriter.supersede_episode`` (MVCC --
old content preserved, recoverable, never deleted).

Design (see ``docs/JST-architecture-proposal.md`` + the Tencent survey,
[[pondr-tencent-agent-memory-survey]]):

* **Post-commit, not pre-commit.** The new episode is always fully encoded +
  stored first; dedup decides what to supersede afterward. Avoids folding
  verdicts into the encode batch (which would suppress edge commits) and keeps
  the encode path byte-identical when the flag is off.
* **Judge + apply separation.** ``DedupJudge.judge`` is the LLM-call seam (tests
  subclass + override to return queued verdicts, exercising the REAL ``apply``).
  ``DedupJudge.apply`` is pure + deterministic (unit-tested directly with
  hand-built verdicts + a real store). Auto-apply is safe because supersession is
  NON-DESTRUCTIVE (state="superseded", not deleted; recoverable via
  ``default_episode_ids(include_inactive=True)``) -- this differs from Phase C
  compaction, which DEFERS corruption because overwriting a verbatim with a
  corrupted gist is destructive.
* **Cold-start-safe.** ``judge`` returns None on Bonsai-down / no-candidates /
  no embedding -> ``apply`` is not called -> the episode stays as-encoded. Never
  raises. The encoder's ``_maybe_dedup`` wraps the whole thing in try/except +
  logs, so a dedup hiccup never loses the episode.
* **Precedence: skip > update/merge > store.** If the 8B emits ``skip`` for any
  pair, the new is a duplicate -> supersede the new (discard it) and stop (don't
  mutate the olds). Else apply each ``update``/``merge`` verdict (supersede that
  old by the new). ``store`` is a no-op.

v1 defers (see Scope in the plan): the explicit ``priority`` field + salience
bump on merge (-> C4); literal text-folding on merge (the new stands as the
consolidated rep; the old is superseded); the defer/review surface. Episodes-
only. ``config.dedup_enabled`` flag, default OFF, byte-identical when off.
"""

from __future__ import annotations

from typing import Optional

from ..gnn.semantic_memory import SemanticMemoryWriter
from ..memory.episode import Episode

__all__ = ["DedupJudge"]


class DedupJudge:
    """The post-commit 4-action dedup reconcile engine.

    Constructed by ``runtime.build_ponder`` when ``dedup`` is on and injected
    onto the encoder as ``encoder._dedup_judge`` (DI so tests inject a subclass
    overriding ``judge`` while inheriting the real ``apply``). The encoder's
    ``_maybe_dedup`` is the only caller, and it gates on
    ``config.dedup_enabled`` at call time (the master-config convention, same
    pattern as ``hybrid_retrieval``) PLUS ``self._dedup_judge is not None`` -- both
    must be true to run, so flag-off and no-judge are both byte-identical no-ops.

    The candidate pool is ``vector_search.search_by_vector(episode
    .summary_embedding, k)`` -- the new episode's own embedding, no re-embed.
    The hit list is the active set (``search_by_vector`` already excludes
    abstracted/deprecated/superseded). ``judge`` excludes the new's own eid
    (self-match guard, relevant on the async path where the new is committed
    before the fill) and intersects ``store.episode_ids_for_user(episode.user_id)``
    so dedup never merges across users.
    """

    def __init__(
        self,
        decider,
        vector_search,
        store,
        *,
        candidate_k: int = 10,
        max_pairs: int = 8,
    ) -> None:
        self._decider = decider
        self._vs = vector_search  # retriever.vector_search (or None)
        self._store = store
        self._writer = SemanticMemoryWriter(store)
        self._candidate_k = candidate_k
        self._max_pairs = max_pairs

    # ── the LLM-call seam (override in tests) ──

    def judge(self, episode: Episode) -> Optional[list[dict]]:
        """Find candidates + ONE batched Bonsai call.

        Returns a list of ``{"eid": str, "action": "store"|"update"|"merge"|
        "skip", "reason": str}`` (the decider resolves ``pair_id`` -> ``eid``),
        or ``None`` to defer (cold-start: no embedding / no vector index / no
        candidates / Bonsai down / parse fail). ``apply`` is only called when this
        returns a non-None list, so ``None`` keeps the episode as-encoded. Never
        raises -- the encoder wraps the call in try/except too, but the decider's
        ``_post_json`` already never raises (returns None on any HTTP/parse fail).
        """
        vec = episode.summary_embedding
        if not vec or self._vs is None:
            return None
        try:
            hits = self._vs.search_by_vector(vec, k=self._candidate_k)
        except Exception:  # noqa: BLE001 - cold-start: index not loaded, etc.
            return None
        if not hits:
            return None
        # User-scope: only dedup against the new episode's own user's episodes
        # (don't merge alice's fact into bob's). user_id=None -> no filter (the
        # global path, byte-identical to pre-user-scope).
        allowed = None
        if episode.user_id:
            try:
                allowed = self._store.episode_ids_for_user(episode.user_id)
            except Exception:  # noqa: BLE001 - best-effort scope; empty -> no-op
                return None
        cands: list[tuple[str, float]] = []
        for eid, score in hits:
            if eid == episode.id:
                continue  # self-match guard (async: new is committed pre-fill)
            if allowed is not None and eid not in allowed:
                continue  # cross-user guard
            cands.append((eid, score))
            if len(cands) >= self._max_pairs:
                break
        if not cands:
            return None
        # Hydrate candidate content (summary/entities/topics) so the judge has
        # enough to decide dup vs complement. get_episode returns None for a
        # missing/orphan eid -> skip it (defensive; the vector index should not
        # hold missing eids, but never trust an index).
        cand_content: list[dict] = []
        for eid, _score in cands:
            ep = self._store.get_episode(eid)
            if ep is None:
                continue
            cand_content.append({
                "eid": eid,
                "summary": ep.summary,
                "entities": list(ep.entities or []),
                "topics": list(ep.topics or []),
            })
        if not cand_content:
            return None
        return self._decider.judge_dedup_pairs(
            episode.summary, list(episode.entities or []),
            list(episode.topics or []), cand_content)

    # ── the deterministic applier (unit-tested directly) ──

    def apply(self, episode: Episode, verdicts: list[dict]) -> None:
        """Apply the 4-action verdicts via MVCC supersession.

        Precedence ``skip`` > ``update``/``merge`` > ``store``:

        * If ANY verdict is ``skip``, the new is a duplicate -> supersede the new
          (``supersede_episode(existing_eid, episode.id)``: existing survives, new
          marked superseded + unindexed) and return. Don't mutate the olds -- the
          new is the redundant one, discard it once, not N times.
        * Else for each ``update``/``merge`` verdict, supersede that old by the
          new (``supersede_episode(episode.id, verdict_eid)``: new survives, old
          superseded + unindexed). ``merge`` and ``update`` are the SAME apply
          for v1 (the new stands as the consolidated rep; the salience bump +
          text-folding are deferred refinements). ``store`` is a no-op.

        Each ``supersede_episode`` is its own atomic ``batch_sync`` (it cannot
        ride the new episode's encode batch -- that already committed). The
        window where both new + old are live is between-turns (async, gated on
        ``foreground_busy``) or post-encode (sync) -- no retrieval races it (a
        single query doesn't interleave with the worker's single-thread fill).
        Supersession is non-destructive: the old's content is preserved
        (state="superseded", not deleted), recoverable via
        ``default_episode_ids(include_inactive=True)``.
        """
        if not verdicts:
            return
        # skip short-circuits: the new is a dup of an existing episode. A skip
        # verdict WITHOUT a usable eid is malformed (the decider always sets
        # eid, but apply is defensive) -> drop it and fall through to
        # update/merge, mirroring the update/merge eid guard below.
        for v in verdicts:
            if v.get("action") == "skip":
                existing_eid = v.get("eid")
                if not existing_eid or existing_eid == episode.id:
                    continue
                self._writer.supersede_episode(existing_eid, episode.id)
                return
        # No skip: apply update/merge (supersede each old by the new). store = noop.
        for v in verdicts:
            action = v.get("action")
            if action not in ("update", "merge"):
                continue
            old_eid = v.get("eid")
            if not old_eid or old_eid == episode.id:
                continue
            self._writer.supersede_episode(episode.id, old_eid)