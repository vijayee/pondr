"""Offline chat-replay harness -> STRM 2d v2 graduation training data.

Replays the real local chat transcripts on disk (``docs/*.json``, Onyx
``ChatSessionDetailResponse`` shape) through the TRAINED backbone with the WM
ring ON + graduation logging ON -- no Bonsai, no GLiNER, no Onyx, no secrets --
to produce a real ``replay.jsonl``, then labels it + trains the v2 graduation
head. This is the "ok do 2d v2" path: instead of waiting for live Onyx sessions
to accumulate replay labels, replay the transcripts we already have.

Pipeline:

  1. Load one or more Onyx-format chat transcripts (``--transcripts``).
  2. Build a ``HippocampalStore`` on a fresh temp DB; encode each user turn as an
     ``Episode`` with rule-based entities (``_extract_entities``) + topics
     (``_TOPIC_MAP``) + a bge-small ``summary_embedding`` (GLiNER-free,
     Bonsai-free). The rule-based planner is the SAME offline fallback the live
     retriever uses when Bonsai is unreachable, so this is a faithful replay.
  3. Construct a ``PonderOrchestrator`` manually (no encoder -> no live-encode;
     no gate -> plain ``retrieve`` so EVERY turn reaches the replay logger;
     ring ON K=``--ring-capacity``; the trained 2a relevance head loaded so
     ``r_i`` is populated; a stub ``mode_a`` so ``synthesize`` never calls
     Bonsai).
  4. Per session: reset the WM, set the session id, then for each user turn
     (after the first) run ``orch.query(text, conversation_history=history,
     auto_persist=False)`` then encode the turn. The orchestrator's
     ``_write_graduation_replay`` appends one record per ring slot per turn to
     ``replay.jsonl`` (``source_id`` = the recalled episode id).
  5. Close the store. Run ``label_later_needed`` -> ``replay_labeled.jsonl``,
     then ``fit_graduation`` -> ``best.pt``. Report the v2-vs-v1 gate honestly:
     if the gate fails (insufficient re-appearance labels, or v2 does not beat
     the v1 proxy), say so -- do NOT persist a non-gate-passing head.

This script uses NO secrets (the Onyx API key is not needed -- the transcripts
are already on disk). The relevance-head checkpoint is a trained artifact, not
a secret.

Usage:
    python scripts/replay_chat_to_graduation.py
    python scripts/replay_chat_to_graduation.py --transcripts docs/a.json docs/b.json \\
        --ring-capacity 16 --max-turns 80

Outputs (all under ``--out-dir``, default ``data/training/strm_graduation``):
  - ``replay.jsonl``            (the orchestrator's graduation logger output)
  - ``replay_labeled.jsonl``    (label_later_needed output, v2 training input)
  - ``best.pt`` / ``final.pt`` / ``train_log.json``  (fit_graduation output, only
    when the gate passes; otherwise best.pt is still written by fit_graduation
    but the script reports NO-GO and does not claim it shipped)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import config as _config  # noqa: E402
from src.memory.episode import Episode  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402
from src.orchestrator import PonderOrchestrator  # noqa: E402
from src.retrieval.query_planner import (  # noqa: E402
    BonsaiQueryPlanner,
    _TOPIC_MAP,
    _extract_entities,
)
from src.retrieval.retriever import HippocampalRetriever  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.relevance_head import load_relevance_head  # noqa: E402
from src.subconscious.training.graduation_training import (  # noqa: E402
    GraduationTrainingConfig,
    fit_graduation,
)
from src.subconscious.training.routing_training import build_embedder, load_backbone  # noqa: E402

# Reuse the committed label generator (same module the live path will use).
from scripts.generate_graduation_labels import (  # noqa: E402
    label_later_needed,
    load_replay,
    write_labeled,
)

DEFAULT_BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
# Cap on entities per encoded episode. The rule-based extractor was built for
# short QUERIES (a handful of Capitalized tokens); a long chat message yields
# 100+ "entities" (every sentence-initial "This"/"You"/"If" survives the
# stoplist), which explodes the per-encode graph-op batch past WaveDB's
# memory-pool comfort and corrupts the batch (double-free + batch_sync fail).
# The live encoder uses selective GLiNER, so capping here also aligns the
# harness's entity density with live behavior. 40 keeps enough named entities
# for re-appearance matching (a later short query overlapping on any of them
# recalls the episode) while bounding the batch.
MAX_ENTITIES_PER_EPISODE = 40
DEFAULT_RELEVANCE_HEAD = "data/training/strm_relevance/best.pt"
DEFAULT_TRANSCRIPTS = (
    "docs/The_Ponder_Engine_Chat.json",
    "docs/The _Ponder_Engine_Coding_Chat.json",
)
DEFAULT_OUT_DIR = "data/training/strm_graduation"


class _StubModeA:
    """Stand-in for ``ModeAGenerator`` so ``synthesize`` never calls Bonsai.

    ``query()``'s one-shot synthesize path calls ``mode_a._complete(messages,
    tools=tools)`` and unpacks ``(content, tool_calls)``. We return a constant
    string + ``None`` tool_calls. The response text is irrelevant to the replay
    logger (which fires BEFORE synthesize); it only has to be a non-empty
    string so downstream paths that check ``response`` stay on the happy path.
    """

    def _complete(self, messages, tools=None, tool_choice=None):
        return ("[replay-stub-response]", None)


def _extract_signals(text: str) -> tuple[list[str], list[str]]:
    """Rule-based entity + topic extraction (the offline planner fallback).

    Mirrors ``BonsaiQueryPlanner.plan_rule_based``'s entity/topic extraction
    (``_extract_entities`` + the ``_TOPIC_MAP`` keyword scan) so the episodes
    we encode carry the SAME signal the live rule-based planner would extract
    at query time -- i.e. a later query that mentions the same Capitalized
    entity / topic keyword recalls the same episode via graph traversal. This
    is the re-appearance dynamic the v2 ``later_needed`` label keys on.
    """
    entities = _extract_entities(text)
    lower = text.lower()
    topics: list[str] = []
    for word, topic in _TOPIC_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", lower) and topic not in topics:
            topics.append(topic)
    return entities, topics


def load_transcript_threads(path: str) -> tuple[str, list[dict]]:
    """Load an Onyx transcript -> (session_id, ordered user/assistant turns).

    Returns the session id + an ordered list of ``{"role", "content"}``
    pairs (only ``user`` + ``assistant``, non-empty, in file order). Each
    user message is followed by its assistant reply; the query loop pairs them
    so ``conversation_history`` (the planner's pronoun-resolution window) and
    the encoded episode's ``full_text`` both reflect the real exchange.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    session_id = doc.get("chat_session_id") or Path(path).stem
    msgs = doc.get("messages", [])
    turns: list[dict] = []
    for m in msgs:
        role = m.get("message_type")
        if role not in ("user", "assistant"):
            continue
        text = m.get("message")
        if not isinstance(text, str) or not text.strip():
            continue
        turns.append({"role": role, "content": text})
    return str(session_id), turns


def build_episode(
    ep_id: str,
    user_text: str,
    assistant_text: str,
    timestamp: str,
    user_id: str,
    session_id: str,
    embedder,
) -> Episode:
    """Build a GLiNER-free Episode from one user/assistant exchange.

    Entities + topics come from the rule-based extractor (``_extract_signals``);
    ``summary_embedding`` is the bge-small embedding of the USER text (the
    query) so the semantic-fallback path can match later similar queries to
    this episode. The assistant text is the readable ``summary`` (first 200
    chars); ``full_text`` is the role-tagged join (matches the live encoder).
    """
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    full_text = Episode._join_messages(messages)
    summary = Episode._assistant_summary(messages)
    entities, topics = _extract_signals(user_text)
    # Bound the per-episode entity count (see MAX_ENTITIES_PER_EPISODE).
    entities = entities[:MAX_ENTITIES_PER_EPISODE]
    summary_embedding = None
    if embedder is not None:
        vecs = embedder.encode([user_text])
        if vecs:
            summary_embedding = [float(x) for x in vecs[0]]
    return Episode(
        id=ep_id,
        timestamp=timestamp,
        summary=summary,
        full_text=full_text,
        entities=entities,
        topics=topics,
        user_id=user_id,
        session_id=session_id,
        origin="corpus",
        messages=messages,
        summary_embedding=summary_embedding,
    )


def _pair_turns(turns: list[dict]) -> list[tuple[str, str]]:
    """Pair each user turn with the assistant turn that follows it.

    Onyx exports interleave user/assistant chronologically; a user turn is
    followed by its assistant reply. A user turn with no following assistant
    turn is paired with an empty assistant string (the episode is still
    readable -- ``_assistant_summary`` falls back to the user content).
    """
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns):
        if turns[i]["role"] != "user":
            i += 1
            continue
        user_text = turns[i]["content"]
        assistant_text = ""
        if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
            assistant_text = turns[i + 1]["content"]
            i += 2
        else:
            i += 1
        pairs.append((user_text, assistant_text))
    return pairs


def _encode_best_effort(store, ep, session_id, turn_index) -> bool:
    """Encode an episode; on WaveDB failure, log + skip (return False).

    The local WaveDB dev build's C memory pool intermittently double-frees
    after enough writes, after which ``batch_sync`` raises a generic
    ``WaveDBError("batch_sync failed")`` on otherwise-valid episodes (small
    op counts, short entities -- NOT a batch-size issue). That is a WaveDB
    C-extension bug, out of scope for this harness to fix.

    The replay records come from the QUERY path (retrieve + inject + the
    replay logger), NOT from encode -- encode only populates the recallable
    memory. A skipped episode simply never becomes recallable (it never
    appears as a ``source_id`` in any replay record, so it is neither a
    positive nor a negative ``later_needed`` -- just absent). Skipping keeps
    the run going so real replay data still accumulates from the turns that
    did encode. The caller reports the skip count so the dataset's coverage
    is honest, not silent.
    """
    try:
        store.encode_episode(ep)
        return True
    except Exception as e:  # noqa: BLE001 - skip corrupted episode, keep going
        ops = store._content_ops(ep.id, ep) + store._edge_ops(ep.id, ep)
        print(f"[encode-skip] session={session_id} turn={turn_index} ep={ep.id} "
              f"err={e!r} ops={len(ops)} entities={ep.entities[:8]}",
              file=sys.stderr)
        return False


def replay_session(
    orch: PonderOrchestrator,
    store: HippocampalStore,
    embedder,
    session_id: str,
    user_id: str,
    pairs: list[tuple[str, str]],
    max_turns: int,
    epoch_base: float,
) -> tuple[int, int, int]:
    """Replay one session's user turns through the orchestrator; encode each.

    Encodes turn 0 first (seeds the memory), then for each subsequent user turn
    runs ``orch.query`` (recalling episodes 0..i-1) and encodes turn i after.
    Returns ``(n_queries, n_encoded, n_encode_skipped)``. Resets the WM at the
    session boundary (called by the caller before this); sets ``orch.user_id``
    so the replay records carry this session's id.
    """
    orch.user_id = session_id
    orch.working_memory.reset()
    history: list[dict] = []
    n_queries = 0
    n_encoded = 0
    n_skipped = 0
    # Cap the pairs (dev speed); the cap is applied BEFORE encoding so the
    # encoded memory + the replayed turns are consistent.
    if max_turns > 0:
        pairs = pairs[:max_turns]
    if not pairs:
        return 0, 0, 0
    # Seed: encode turn 0 so query 1 has memory to recall.
    u0, a0 = pairs[0]
    ep0 = build_episode(
        f"{session_id}__ep0000", u0, a0,
        timestamp=_iso(epoch_base, 0), user_id=user_id, session_id=session_id,
        embedder=embedder,
    )
    if _encode_best_effort(store, ep0, session_id, 0):
        n_encoded += 1
    else:
        n_skipped += 1
    history.append({"role": "user", "content": u0})
    history.append({"role": "assistant", "content": a0})
    for i in range(1, len(pairs)):
        u, a = pairs[i]
        try:
            orch.query(
                u,
                conversation_history=list(history),
                auto_persist=False,
                signal="routine",
            )
        except Exception as e:  # noqa: BLE001 - one bad turn must not kill the run
            print(f"  [query-fail] session={session_id} turn={i}: {e}",
                  file=sys.stderr)
        # Encode this turn AFTER its query so the memory grows realistically
        # (query i recalls episodes 0..i-1, never itself).
        ep = build_episode(
            f"{session_id}__ep{i:04d}", u, a,
            timestamp=_iso(epoch_base, i), user_id=user_id, session_id=session_id,
            embedder=embedder,
        )
        if _encode_best_effort(store, ep, session_id, i):
            n_encoded += 1
        else:
            n_skipped += 1
        history.append({"role": "user", "content": u})
        history.append({"role": "assistant", "content": a})
        n_queries += 1
        if (i + 1) % 20 == 0:
            print(f"  session={session_id} replayed {i + 1}/{len(pairs)} turns "
                  f"(encoded={n_encoded} skipped={n_skipped})", flush=True)
    return n_queries, n_encoded, n_skipped


def _iso(epoch_base: float, turn_index: int) -> str:
    """Deterministic ISO timestamp for a turn (no real clock at replay time).

    ``datetime.now`` would be non-deterministic across runs; pin each turn to
    ``epoch_base + turn_index`` seconds so episode timestamps are stable and
    ordered within a session. The base is passed in (a fixed reference).
    """
    import datetime as _dt

    return (_dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
            + _dt.timedelta(seconds=epoch_base + turn_index)).isoformat()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Replay local chat transcripts -> STRM 2d v2 graduation data")
    p.add_argument("--transcripts", nargs="+", default=list(DEFAULT_TRANSCRIPTS),
                   help="Onyx-format chat transcript JSON files to replay")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH,
                   help="trained Phase 2a backbone checkpoint")
    p.add_argument("--relevance-head", default=DEFAULT_RELEVANCE_HEAD,
                   help="trained STRM 2a relevance-head checkpoint (populates r_i)")
    p.add_argument("--db", default=None,
                   help="WaveDB path for the replay store (default: <out-dir>/replay_store)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="output dir for replay.jsonl / replay_labeled.jsonl / best.pt")
    p.add_argument("--ring-capacity", type=int, default=16,
                   help="WM ring buffer capacity K (the ring must be ON for replay)")
    p.add_argument("--device", default="auto", help="backbone+head device: auto|cpu|cuda")
    p.add_argument("--max-turns", type=int, default=0,
                   help="cap user turns per session (0 = all; dev speed knob)")
    p.add_argument("--seed", type=int, default=0, help="fit_graduation seed")
    p.add_argument("--epochs", type=int, default=40, help="fit_graduation epochs")
    p.add_argument("--user-id", default="pondr", help="user_id for encoded episodes")
    p.add_argument("--retrace", action="store_true",
                   help="regenerate even if replay.jsonl already exists")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = out_dir / "replay.jsonl"
    labeled_path = out_dir / "replay_labeled.jsonl"
    db_path = args.db or str(out_dir / "replay_store")

    # Validate inputs up front.
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone not found at {backbone_path}", file=sys.stderr)
        return 1
    rel_head_path = Path(args.relevance_head)
    if not rel_head_path.exists():
        print(f"ERROR: relevance-head checkpoint not found at {rel_head_path} "
              f"(needed to populate r_i for the v1 proxy AUC)", file=sys.stderr)
        return 1
    for t in args.transcripts:
        if not Path(t).exists():
            print(f"ERROR: transcript not found at {t}", file=sys.stderr)
            return 1

    # ---- Phase A: replay transcripts -> replay.jsonl ----
    if replay_path.exists() and not args.retrace:
        print(f"  replay.jsonl already exists at {replay_path} "
              f"(use --retrace to regenerate) -- skipping to labeling+train",
              flush=True)
    else:
        # Fresh store: wipe any prior replay DB so retrieval sees only this run.
        db_p = Path(db_path)
        if db_p.exists():
            if db_p.is_dir():
                shutil.rmtree(db_p)
            else:
                db_p.unlink()
        # Remove any prior replay log (the orchestrator appends).
        if replay_path.exists():
            replay_path.unlink()

        # Flip the config flags the orchestrator reads at query time. The loop
        # is OFF so the one-shot synthesize path runs (one stub _complete call);
        # feedback salience is OFF so the one-shot path passes tools=None.
        # graduation logging is ON so _write_graduation_replay fires. These are
        # set on the SAME singleton the orchestrator imports (_runtime_config).
        _config.self_chat_tool_loop_enabled = False
        _config.feedback_salience_enabled = False
        _config.strm_graduation_logging = True
        # Redirect the orchestrator's replay path to our out-dir (class attr).
        PonderOrchestrator._REPLAY_PATH = replay_path

        print(f"[load] backbone={backbone_path} device={args.device}", flush=True)
        backbone = load_backbone(str(backbone_path), BackboneConfig(),
                                  device=args.device)
        embedder = build_embedder("on-demand")
        print(f"[load] relevance_head={rel_head_path}", flush=True)
        relevance_head = load_relevance_head(str(rel_head_path), device=args.device)

        store = HippocampalStore(db_path)
        planner = BonsaiQueryPlanner(endpoint=None, force_rule_based=True)  # offline: deterministic rule-based plans, no (flapping) server dependency
        retriever = HippocampalRetriever(
            store, planner=planner, auto_load_index=True,
            retrieval_gate=None, embedder=embedder,
        )
        from src.config import Phase2cConfig  # noqa: E402
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(out_dir / "sessions")
        orch = PonderOrchestrator(
            store=store,
            retriever=retriever,
            backbone=backbone,
            embedder=embedder,
            mode_a=_StubModeA(),
            config=cfg,
            user_id=args.user_id,
            encoder=None,             # no live-encode (we encode manually)
            relevance_head=relevance_head,
            graduation_proxy=None,    # v1 is scored offline by fit_graduation
            ring_capacity=args.ring_capacity,
        )

        total_queries = 0
        total_encoded = 0
        total_skipped = 0
        t0 = time.time()
        # A fixed epoch base per session keeps timestamps deterministic + ordered
        # across sessions (session B's turns come after session A's).
        epoch_base = 0.0
        for tpath in args.transcripts:
            session_id, turns = load_transcript_threads(tpath)
            pairs = _pair_turns(turns)
            print(f"[replay] {tpath} session={session_id} -> {len(pairs)} user turns",
                  flush=True)
            nq, ne, ns = replay_session(orch, store, embedder, session_id,
                                        args.user_id, pairs, args.max_turns,
                                        epoch_base)
            total_queries += nq
            total_encoded += ne
            total_skipped += ns
            epoch_base += 1e6  # space sessions apart so timestamps never collide
        print(f"[replay] {total_queries} queries in {time.time() - t0:.1f}s "
              f"(encoded {total_encoded} episodes, skipped {total_skipped} on "
              f"WaveDB batch_sync errors)", flush=True)

        # Close the store so all writes flush before labeling reads the file.
        try:
            store.close()
        except Exception as e:  # noqa: BLE001
            print(f"[close-fail] {e}", file=sys.stderr)

    # ---- Phase B: label replay -> replay_labeled.jsonl ----
    if not replay_path.exists():
        print(f"ERROR: no replay.jsonl at {replay_path}", file=sys.stderr)
        return 1
    records = load_replay(str(replay_path))
    if not records:
        print(f"ERROR: no replay records in {replay_path}", file=sys.stderr)
        return 1
    sessions = sorted({r.get("session_id") for r in records})
    print(f"[label] {len(records)} replay records across {len(sessions)} session(s): "
          f"{sessions}", flush=True)
    labeled = label_later_needed(records)
    stats = write_labeled(labeled, str(labeled_path))
    pos = stats["positive"]
    neg = stats["negative"]
    print(f"[label] wrote {stats['total']} records -> {labeled_path}", flush=True)
    print(f"[label] later_needed: positive={pos} negative={neg} null={stats['null']}",
          flush=True)
    if pos == 0:
        print("  NO-GO: zero positive labels -- no source_id re-appeared after a "
              "ring gap in any session. Either run more/longer sessions, raise "
              "--ring-capacity, or lower --max-turns to widen the ring gap. The v2 "
              "head has nothing to learn; not training.", file=sys.stderr)
        return 2
    if pos < 8:
        print(f"  WARNING: only {pos} positive labels -- the v2 gate AUC will be "
              f"noisy (chance-level prone). Consider more replay data.", file=sys.stderr)

    # ---- Phase C: train v2 -> best.pt + report gate ----
    train_records = [r for r in labeled if r.get("later_needed") is not None]
    if len(train_records) < 16:
        print(f"  NO-GO: only {len(train_records)} labeled (non-null) records -- "
              f"need >= 16 for a train/val split. Not training.", file=sys.stderr)
        return 2
    tcfg = GraduationTrainingConfig(
        checkpoint_dir=str(out_dir),
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    print(f"[train] fit_graduation on {len(train_records)} labeled records "
          f"(epochs={args.epochs}, device={args.device})", flush=True)
    t0 = time.time()
    result = fit_graduation(train_records, config=tcfg)
    print(f"[train] done in {time.time() - t0:.1f}s", flush=True)
    print(f"[train] best_v2_auc={result.get('best_v2_auc'):.4f} "
          f"best_v1_auc={result.get('best_v1_auc'):.4f} "
          f"go={result.get('go')}", flush=True)
    if result.get("go"):
        print(f"[train] GATE PASSED -- v2 beats v1 + clears the chance floor. "
              f"best.pt -> {out_dir / 'best.pt'}", flush=True)
        return 0
    print(f"[train] GATE NOT PASSED -- v2 did not beat the v1 proxy (or did not "
          f"clear the chance floor). best.pt was written by fit_graduation but is "
          f"NOT a gate-passing head; do not ship it. Inspect "
          f"{out_dir / 'train_log.json'} for per-epoch AUCs.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())