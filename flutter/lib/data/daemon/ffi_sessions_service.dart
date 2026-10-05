/// The FFI-backed [SessionsService] (the FFI-binding plan's Task 5): the
/// daemon's session store, mirrored app-side.
///
/// THE HONEST SHAPES this impl pins:
/// - `list()` answers the LOCAL mirror's rows; `refresh()` (kicked at
///   construction) folds CA_SESSIONS's rows in — sorted NEWEST-FIRST by the
///   rows' created epoch-seconds (the wire carries u64 seconds — the C's
///   `_ca_created_seconds` ISO→epoch conversion; the app's display uses it
///   as UTC instants). The rows never carry messages: their fold is the
///   events replay's job ([FfiChatService]'s record fold); a refresh
///   PRESERVES a known session's messages and its app-local name (the
///   daemon's `goal` meta is the empty sentinel today — `frame.c`'s listing
///   comment; no composer writes a goal) and adopts a non-empty goal only
///   for unknown sessions.
/// - `create()` RETURNS A LOCAL EMPTY DRAFT — the daemon owns session birth
///   (a frame starts at the FIRST prompt): the draft is id-empty
///   ("New conversation", no messages, front-inserted, immediately active).
///   Its FIRST send (the chat service's `prompt(null)`) moves the daemon's
///   sid into it — [adoptDraftSid], THE REPLACE RULE, pinned in the tests.
///   A draft is SINGULAR: a second `create()` while one stands returns it
///   (the daemon's store has no second empty session either).
/// - `delete()` is APP-LOCAL HIDING (the spec's recorded out-of-scope: the
///   daemon's frame stays; the re-subscribe's replay would re-meet it — the
///   hidden id keeps its row-less record in `hiddenSids`).
/// - `active()`/`select()` follow the interface's contracts (the export's
///   reselect rules).
///
/// The record fold lives in [FfiChatService] (the events channel's
/// per-session subscription is the C's one-at-a-time contract) — it writes
/// through this store's mutators, so every fold fires [changes].
library;

import 'dart:async';

import 'package:pondr/ffi/sa_client_binding.dart';

import '../models.dart';
import '../services.dart';

final class FfiSessionsService implements SessionsService {
  /// [clientFuture] is the ensured `SaClientNative` (the runtime's one
  /// connection; the bindings hand the supervisor's `ensure()` future).
  FfiSessionsService({required Future<SaClientNative> clientFuture})
      : _controller = StreamController<ChatsChanged>.broadcast() {
    _clientFuture = clientFuture;
    // The boot's first fold: the CA_SESSIONS listing (a daemon refusal
    // leaves the empty mirror standing — the stores re-read on the next
    // fold; the failure is the supervisor's channels' business).
    unawaited(Future<void>(() async {
      try {
        await refresh();
      } on StateError {
        // The client is gone (or never was): the empty mirror stands.
      }
    }));
  }

  /// The ensured client's future (the runtime's one connection).
  late final Future<SaClientNative> _clientFuture;
  final StreamController<ChatsChanged> _controller;

  /// The mirror: newest-first by construction (the draft or the record
  /// folds front-insert; the refresh sorts by created descending).
  final List<ChatSession> _sessions = <ChatSession>[];

  final Map<String, int> _createdSeconds = <String, int>{};
  String? _activeId;

  /// The locally hidden rows (the delete's app-local rule; the refresh
  /// keeps them hidden).
  final Set<String> hiddenSids = <String>{};

  /// The sids a listing has confirmed once (an un-confirmed fold row rides
  /// a race; a confirmed row's later absence means the daemon dropped it).
  final Set<String> _confirmed = <String>{};

  // ── the store's fold mutators (FfiChatService's write surface) ──────────

  /// The record fold's write: the message lands at the (sid, seq)'s
  /// deterministic id — a session that already carries the id is UNTOUCHED
  /// (the fold's store-side dedupe guard: a replay + a live duplicate can
  /// never double-message, the pinned record-dedupe's backstop). An unknown
  /// sid ADOPTS a standing empty draft row (the replace rule — most often
  /// the send's own frame) or front-inserts a fresh row (a record that
  /// arrived before its row's listing).
  void storeMessage(String sid, Message message) {
    if (hiddenSids.contains(sid)) {
      return; // the locally hidden rows never re-materialize
    }
    var i = _indexOf(sid, adoptDraftInto: true);
    if (i < 0) {
      // THE FOLD-BEFORE-LISTING shape: the daemon's record arrived before
      // its row's listing — the row front-inserts fresh (newest-first by
      // construction).
      _sessions.insert(
        0,
        ChatSession(
          id: sid,
          name: 'New conversation',
          updatedAt: createdAtOf(sid),
          messages: const <Message>[],
        ),
      );
      i = 0;
    }
    final session = _sessions[i];
    for (final m in session.messages) {
      if (m.id == message.id) {
        return; // the (sid, seq) message id — already folded
      }
    }
    _sessions[i] = session.copyWith(
      messages: <Message>[...session.messages, message],
      updatedAt: DateTime.now(),
    );
    _fire();
  }

  /// THE DRAFT REPLACE (the create()'s defer rule): the standing empty
  /// draft takes the daemon's sid in place — position, name and messages
  /// ride; the active id follows when it pointed at the draft.
  void adoptDraftSid(String sid) {
    _indexOf(sid, adoptDraftInto: true);
    _fire();
  }

  /// The chat's send-time rename (the export's first-send rule — a name
  /// bookkeeping, the daemon's store carries no goal text).
  void rename(String id, String name) {
    final i = _indexOf(id);
    if (i < 0) return;
    _sessions[i] = _sessions[i].copyWith(name: name);
    _fire();
  }

  /// The store's id lookup (the chat's fold target reads).
  ChatSession? byId(String id) {
    for (final s in _sessions) {
      if (s.id == id) return s;
    }
    return null;
  }

  // ── the CA_SESSIONS fold ─────────────────────────────────────────────────

  /// The listing fold: the rows merge over the mirror (known ids keep their
  /// folded messages + name), sorted newest-first by the created
  /// epoch-seconds; locally hidden sids stay hidden. The active id: a
  /// vanished active re-heads the list (the interface's delete rule), or
  /// reads null when nothing remains.
  Future<SaSessionsResult> refresh() async {
    final native = await _clientFuture;
    final result = await native.listSessions();
    if (!result.ok) {
      return result; // the listing failed: the mirror stands unchanged
    }
    final merged = <ChatSession>[];
    for (final row in result.rows) {
      if (hiddenSids.contains(row.sid)) continue;
      final existing = byId(row.sid);
      final goal = row.goal;
      _createdSeconds[row.sid] = row.created;
      _confirmed.add(row.sid);
      merged.add(existing != null
          ? existing.copyWith(
              name: goal != null && goal.isNotEmpty ? goal : existing.name,
            )
          : ChatSession(
              id: row.sid,
              name: goal != null && goal.isNotEmpty ? goal : 'New conversation',
              updatedAt: createdAtOf(row.sid),
              messages: const <Message>[],
            ));
    }
    // Newest-first by the rows' created seconds (descending); equal or
    // absent stamps keep the listing's row order (the index tiebreak).
    merged.sort((a, b) {
      final sa = _createdEpoch(a.id);
      final sb = _createdEpoch(b.id);
      if (sa != sb) return sb.compareTo(sa);
      return result.rows.indexWhere((r) => r.sid == a.id)
          .compareTo(result.rows.indexWhere((r) => r.sid == b.id));
    });
    // The standing DRAFT (an id-empty row — the daemon's listing never
    // carries it) rides the front; the REFOLD-ONLY rows (a session whose
    // records folded before its listing's row existed — the record fold's
    // front-insert shape) ride their own order — the listing never erases
    // local knowledge.
    final localRows = <ChatSession>[];
    for (final s in _sessions) {
      if (s.id.isEmpty) continue; // the draft's own front ride below
      if (hiddenSids.contains(s.id)) continue; // a hidden row stays hidden
      final inListing = result.rows.any((row) => row.sid == s.id);
      if (inListing) {
        _confirmed.add(s.id);
        continue; // the merged row carries it (the fold's messages ride)
      }
      if (!_confirmed.contains(s.id)) {
        // UN-CONFIRMED (its records folded before its first listing — the
        // fold's front-insert racing the listing): local knowledge, kept.
        localRows.add(s);
      }
      // Confirmed-then-absent = deleted server-side (or the frame is gone):
      // it drops here — this listing is the daemon's truth of what exists.
    }
    final drafts = [for (final s in _sessions) if (s.id.isEmpty) s];
    _sessions
      ..clear()
      ..addAll(<ChatSession>[...drafts, ...merged, ...localRows]);
    if (_activeId == null || byId(_activeId!) == null) {
      _activeId = _sessions.isNotEmpty ? _sessions.first.id : _activeId;
    }
    _fire();
    return result;
  }

  /// The row's created instant (the wire's u64 UTC epoch seconds), the fold
  /// time when absent (0 on the wire = the listing's absent sentinel).
  DateTime createdAtOf(String sid) {
    final seconds = _createdEpoch(sid);
    if (seconds == 0) return DateTime.now();
    return DateTime.fromMillisecondsSinceEpoch(seconds * 1000, isUtc: true)
        .toLocal();
  }

  int _createdEpoch(String sid) => _createdSeconds[sid] ?? 0;

  // ── the interface ────────────────────────────────────────────────────────

  @override
  List<ChatSession> list() => List.unmodifiable(_sessions);

  @override
  ChatSession? active() {
    for (final s in _sessions) {
      if (s.id == _activeId) return s;
    }
    return null;
  }

  @override
  void select(String id) {
    _activeId = id;
    _fire();
  }

  @override
  ChatSession create() {
    for (final s in _sessions) {
      if (s.id.isEmpty) {
        // THE SINGULAR DRAFT: the standing empty draft is the answer (the
        // daemon's store has no second empty session either).
        _activeId = '';
        _fire();
        return s;
      }
    }
    final draft = ChatSession(
      id: '', // THE DRAFT SENTINEL: no daemon sid exists before the first send
      name: 'New conversation',
      updatedAt: DateTime.now(),
      messages: const <Message>[],
    );
    _sessions.insert(0, draft);
    _activeId = '';
    _fire();
    return draft;
  }

  @override
  void delete(String id) {
    // The app-local hide (the recorded out-of-scope: the daemon's frame
    // stays — the delete hides the row, never the frame's records).
    if (id.isEmpty) return;
    _sessions.removeWhere((s) => s.id == id);
    hiddenSids.add(id);
    if (_activeId == id) {
      _activeId = _sessions.isNotEmpty ? _sessions.first.id : id;
    }
    _fire();
  }

  @override
  Stream<ChatsChanged> changes() => _controller.stream;

  // ── the pieces ───────────────────────────────────────────────────────────

  /// The index of [id]'s row; -1 when absent. [adoptDraftInto] = the
  /// unknown-sid call may ADOPT a standing empty draft (the replace rule:
  /// the id changes in place, position/name/messages ride, the active id
  /// follows) — the fold's and the chat's send both land their daemon sid
  /// through the same shape.
  int _indexOf(String id, {bool adoptDraftInto = false}) {
    for (var i = 0; i < _sessions.length; i++) {
      if (_sessions[i].id == id) return i;
    }
    if (!adoptDraftInto) return -1;
    for (var i = 0; i < _sessions.length; i++) {
      if (_sessions[i].id.isEmpty) {
        final draft = _sessions[i];
        _sessions[i] = ChatSession(
          id: id,
          name: draft.name,
          updatedAt: draft.updatedAt,
          messages: draft.messages,
        );
        // The active id follows ONLY when it pointed at the draft (a
        // dangling or a foreign active id is never hijacked).
        if (_activeId != null && _activeId!.isEmpty) _activeId = id;
        return i;
      }
    }
    return -1;
  }

  void _fire() {
    if (!_controller.isClosed) _controller.add(const ChatsChanged());
  }
}