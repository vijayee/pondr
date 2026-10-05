/// The FFI-backed [ChatService] (the FFI-binding plan's Task 5): the send +
/// the record fold, against the supervisor's client and its events channel.
///
/// THE RECORDS ARE THE MESSAGE SOURCE OF TRUTH (the spec's chat mapping;
/// the send does NOT locally append — THE NO-DOUBLE-APPEND PIN, tested): a
/// send emits [ChatTyping], runs the prompt (a steer, or the create when the
/// captured session is the local empty draft — the replace rule lands the
/// daemon's sid into the draft), then WAITS for the assistant's `msg.append`
/// record to fold; its content is the [ChatDone]. The user's bubble is the
/// USER record's arrival too.
///
/// THE RECORD FOLD: every events record folds by (sid, seq) — the seq set
/// here is the fast-path dedupe and the deterministic `m:sid:seq` message id
/// is the store-side backstop ([FfiSessionsService.storeMessage]); a
/// RE-SUBSCRIBED session's replay-then-live from seq 0 therefore re-delivers
/// everything it already folded — the seen set + the id guard swallow the
/// whole replay (ONE message per (sid, seq), pinned in the tests), and the
/// gap a focus change missed replays in on the way back. Non-`msg.append`
/// records (cell/lifecycle/control) fold nothing — the chat view never sees
/// them (the audit's view = a recorded follow-on; the ChatEvent tree is the
/// interface's sealed typing/delta/done trio).
///
/// THE SUBSCRIPTION POLICY: the C contract is ONE active subscription per
/// client, so the service follows the ACTIVE session ([_focus]; the
/// sessions store's changes re-focus; a send refocuses to its captured
/// session). Unfocused sessions' records arrive only through their next
/// replay — the honest v1 shape (the record buffer IS the daemon's log,
/// replayed from seq 0).
///
/// THE ATTACHMENTS: the mock's AttachedFile chips are app-local — the
/// `msg.append` record carries content only, no wire verb uploads files
/// (RECORDED follow-on). A daemon send carries TEXT ONLY; a blank-text send
/// is the interface's no-op guard (nothing to send, the composer keeps its
/// draft), with the files listed and nothing riding.
library;

import 'dart:async';
import 'dart:collection';
import 'dart:convert';

import 'package:pondr/ffi/sa_client_binding.dart';
import 'package:pondr/ffi/sa_ffi.dart';

import '../models.dart';
import '../services.dart';
import 'ffi_sessions_service.dart';

final class FfiChatService implements ChatService {
  FfiChatService({
    required Future<SaClientNative> client,
    required FfiSessionsService sessions,
    required Stream<SaEventRecord> events,
    required Stream<SaFailure> failures,
    /// The typing indicator's pacing (the view's flag closes on the
    /// [ChatDone]; the mock's drawn 1100-1799 ms band's floor is the
    /// nominal pace here).
    this.typingPace = const Duration(milliseconds: 1100),
  }) {
    _client = client;
    _sessions = sessions;
    _eventSub = events.listen(_onRecord);
    _failureSub = failures.listen(_onFailure);
    // The boot's focus: the ACTIVE session's channel (an await of the
    // connect; no-op when nothing is active yet).
    unawaited(Future<void>(() async {
      _native = await _client;
      await _focus(_sessions.active()?.id);
    }).catchError(_bootFocusError));
  }

  static void _bootFocusError(Object e) {
    // The boot's focus failing (a gone daemon before the first ensure) is
    // not a chat surface bug: the next focus or send re-ensured. Swallowed
    // HERE only — the sends' own failures ride their streams.
  }

  late final Future<SaClientNative> _client;
  late final FfiSessionsService _sessions;
  final Duration typingPace;

  SaClientNative? _native;
  StreamSubscription<SaEventRecord>? _eventSub;
  StreamSubscription<SaFailure>? _failureSub;
  StreamSubscription<ChatsChanged>? _changeSub;

  /// The focused session's sid (the C's one-subscription contract's local
  /// bookkeeping; null = nothing subscribed).
  String? _focusedSid;

  /// The per-(sid) seq set — the record fold's dedupe fast path.
  final Map<String, Set<int>> _seen = <String, Set<int>>{};

  /// The fold's per-sid max seq.
  final Map<String, int> _maxSeq = <String, int>{};

  /// The pending sends' completers, FIFO per sid — each assistant
  /// `msg.append` fold completes the HEAD; the parallel-sends' pairing is
  /// the record order's.
  final Map<String, Queue<Completer<String>>> _waiters =
      <String, Queue<Completer<String>>>{};

  /// The create-race's fast-reply cache: an assistant record that folded
  /// while a pending create had no sid yet (the fold landed between the
  /// prompt's response and the waiter's registration) waits here for its
  /// send; entries die at 60 s (a reply nobody claimed was history's).
  final Map<String, List<(String, DateTime)>> _unclaimed =
      <String, List<(String, DateTime)>>{};

  /// Sends in the prompt-awaiting-its-create-sid window.
  int _pendingCreates = 0;

  /// Focus changes serialize (the unsubscribe's marker has to land before
  /// the next subscribe's request — the C's one-subscription contract). The
  /// chain's tail CLEARS at its own completion (an equal target that
  /// arrives afterwards is the honest no-op, not a growing chain).
  Future<void>? _focusJob;

  /// The service's teardown (the bindings' dispose hook): the folds'
  /// subscriptions end; the client itself is the supervisor's. The pending
  /// sends' futures error — their listeners ride the (awaiting) send
  /// generators; a cancelled generator's own error is swallowed HERE (a
  /// consumed-quietly listener), never an unhandled zone error.
  void dispose() {
    _eventSub?.cancel();
    _failureSub?.cancel();
    _changeSub?.cancel();
    for (final queue in _waiters.values) {
      while (queue.isNotEmpty) {
        final waiter = queue.removeFirst();
        if (!waiter.isCompleted) {
          waiter.completeError(
              StateError('the chat service was disposed before the reply'));
          waiter.future.then((_) {}, onError: (Object _) {});
        }
      }
    }
    _eventSub = null;
    _failureSub = null;
    _changeSub = null;
    _focusJob = null;
  }

  // ── the send ─────────────────────────────────────────────────────────────

  @override
  Stream<ChatEvent> send(String text, List<AttachedFile> files) async* {
    final trimmed = text.trim();
    // The interface's guard, honestly narrowed for the daemon: content is
    // the text (no upload verb — the files are recorded out of scope), so a
    // blank draft is a no-op EVEN when files ride.
    if (trimmed.isEmpty) return;
    final captured = _sessions.active();
    final capturedId = captured?.id;
    if (captured == null || capturedId == null) {
      return; // no captured session — nothing to talk into
    }
    // The export's rename rule, app-local: only a still-"New conversation"
    // session takes the first 45 characters of the trimmed text ('…' when
    // the RAW input ran longer).
    if (captured.name == 'New conversation') {
      final cut = trimmed.substring(0, trimmed.length.clamp(0, 45));
      _sessions.rename(capturedId, text.length > 45 ? '$cut…' : cut);
    }
    yield ChatTyping(duration: typingPace);

    final waiter = Completer<String>();
    var sid = capturedId;
    final creates = capturedId.isEmpty;
    if (creates) _pendingCreates++;
    try {
      final native = _native ?? await _client;
      _native = native;
      if (native.isDisposed) {
        throw StateError('the daemon client is disposed (the app closed)');
      }
      if (!creates) {
        await _focus(capturedId);
      }
      final prompt = await native.prompt(creates ? null : capturedId, trimmed);
      if (prompt.started) {
        sid = prompt.sid;
        if (creates) {
          // THE REPLACE: the draft takes its daemon sid (the tests pin it).
          _sessions.adoptDraftSid(prompt.sid);
          _seen.putIfAbsent(prompt.sid, () => <int>{});
        }
        await _focus(sid);
        final queue = _waiters.putIfAbsent(sid, () => Queue<Completer<String>>());
        queue.add(waiter);
        _drainUnclaimed(sid, waiter);
      } else if (prompt.queued) {
        final queue =
            _waiters.putIfAbsent(capturedId, () => Queue<Completer<String>>());
        queue.add(waiter);
      } else {
        // The prompt failed loud: the send's stream errors into the view's
        // onError (the typing flag closes — the composer keeps its draft).
        throw StateError(
            'the daemon refused the prompt '
            '(status ${prompt.status})');
      }
      yield ChatDone(await waiter.future);
    } catch (e) {
      // The send's registration (if any) dies with it: the FIFO keeps
      // pairing later assistant folds with later sends.
      _removeWaiter(sid, waiter);
      rethrow;
    } finally {
      if (creates) _pendingCreates--;
    }
  }

  /// The dead send's queue entry leaves (the fold's pairing of a stale
  /// entry would mis-route a later send's reply).
  void _removeWaiter(String sid, Completer<String> waiter) {
    final queue = _waiters[sid];
    if (queue == null) return;
    while (queue.remove(waiter)) {
      // every copy gone (the queue is a remove-while-present drain)
    }
    if (queue.isEmpty) _waiters.remove(sid);
  }

  // ── the subscription policy ──────────────────────────────────────────────

  /// The focus serialization: one subscription move at a time (each move
  /// awaits the previous's completion; an equal target with no move in
  /// flight is the plain no-op — the chain never grows).
  Future<void> _focus(String? sid) {
    _changeSub ??= _sessions.changes().listen((_) {
      // The store's own reselects (create/select/delete) re-focus here; a
      // fold's change lands the same active id — the below no-ops.
      unawaited(_focus(_sessions.active()?.id).catchError(_bootFocusError));
    });
    if (_focusJob == null && (sid == _focusedSid || _isFocusNoOp(sid))) {
      return Future<void>.value();
    }
    final previous = _focusJob ?? Future<void>.value();
    final job = previous.then((_) => _focusNow(sid));
    _focusJob = job;
    job.whenComplete(() {
      if (identical(_focusJob, job)) _focusJob = null;
    });
    return job;
  }

  bool _isFocusNoOp(String? sid) =>
      (sid == null || sid.isEmpty) && _focusedSid == null;

  Future<void> _focusNow(String? sidOrNull) async {
    // THE DRAFT IS NOT A SUBSCRIBABLE SID: the empty id ('') normalizes to
    // null — the daemon has no session there yet.
    final sid = sidOrNull == null || sidOrNull.isEmpty ? null : sidOrNull;
    final native = _native;
    if (native == null || native.isDisposed) {
      _focusedSid = null;
      return;
    }
    if (sid == null) {
      if (_focusedSid != null && native.isSubscribed) {
        await native.unsubscribeEvents();
      }
      _focusedSid = null;
      return;
    }
    if (sid == _focusedSid) return;
    if (native.isSubscribed) {
      await native.unsubscribeEvents();
    }
    final result = await native.subscribeEvents(sid);
    if (result is SaOk) {
      _focusedSid = sid;
    } else {
      _focusedSid = null;
      throw StateError(
          'the events subscription was refused (sid: $sid, '
          'status: ${(result as SaErr).status})');
    }
  }

  // ── the record fold ──────────────────────────────────────────────────────

  void _onRecord(SaEventRecord record) {
    if (record.isMarker) {
      if (record.isUnsubscribeAck &&
          (record.sid == _focusedSid || record.sid.isEmpty)) {
        _focusedSid = null;
      }
      return;
    }
    final sid = record.sid;
    final seen = _seen.putIfAbsent(sid, () => <int>{});
    if (!seen.add(record.seq)) {
      return; // THE (sid, seq) DEDUPE: a replay + a live duplicate fold once
    }
    final previous = _maxSeq[sid] ?? 0;
    if (record.seq > previous) _maxSeq[sid] = record.seq;
    final json = record.recordJson;
    if (json == null) return;
    Map<String, dynamic> decoded;
    try {
      decoded = (jsonDecode(json) as Map<String, dynamic>).cast<String, dynamic>();
    } on FormatException {
      return; // a record the fold cannot read is not a chat message
    }
    if (decoded['type'] != 'msg.append') {
      return; // the cell/lifecycle/control records: the chat ignores them
    }
    final payload = decoded['payload'];
    if (payload is! Map<String, dynamic>) return;
    final role = payload['role'];
    final content = payload['content'];
    if (role is! String || content is! String) return;
    if (role != 'user' && role != 'assistant') {
      return; // some other role rides the same record type — fold nothing
    }
    final message = Message(
      // The deterministic (sid, seq) id — the fold's store-side dedupe
      // backstop (the same record's re-delivery never re-appends).
      id: 'm:$sid:${record.seq}',
      role: role == 'user' ? MessageRole.user : MessageRole.assistant,
      content: content,
      timestamp: DateTime.now(),
    );
    _sessions.storeMessage(sid, message);
    if (role == 'assistant') {
      _completeWaiters(sid, content);
    }
  }

  void _onFailure(SaFailure failure) {
    if (failure.status != saClientStatusDisconnected) return;
    // The channel's death: every pending send's stream errors (the view's
    // onError closes its typing flag); the next ensure reattaches.
    for (final queue in _waiters.values) {
      while (queue.isNotEmpty) {
        final waiter = queue.removeFirst();
        if (!waiter.isCompleted) {
          waiter.completeError(
              StateError('the daemon connection was lost mid-turn '
                  '(${failure.text})'));
          waiter.future.then((_) {}, onError: (Object _) {});
        }
      }
    }
  }

  void _completeWaiters(String sid, String content) {
    final queue = _waiters[sid];
    if (queue != null) {
      // The FIFO head takes the reply; a dead entry (a send whose stream
      // already failed) is skipped over — the pairing rides.
      while (queue.isNotEmpty) {
        final waiter = queue.removeFirst();
        if (!waiter.isCompleted) {
          waiter.complete(content);
          return;
        }
      }
    }
    // The create race: a fold that landed while a pending create's sid was
    // unknown waits here (its send claims it at registration). No pending
    // create = a HISTORY fold — no stream, the message store already has it.
    if (_pendingCreates == 0) return;
    _unclaimed
        .putIfAbsent(sid, () => <(String, DateTime)>[])
        .add((content, DateTime.now()));
    _unclaimed[sid] = _unclaimed[sid]!
        .where(
          (entry) => DateTime.now().difference(entry.$2) <
              const Duration(seconds: 60),
        )
        .toList();
  }

  /// The waiter's registration claims its create race's fast fold (the same
  /// send's reply that landed between the prompt's response and here).
  void _drainUnclaimed(String sid, Completer<String> waiter) {
    final unclaimed = _unclaimed.remove(sid);
    if (unclaimed == null || unclaimed.isEmpty) return;
    if (!waiter.isCompleted) waiter.complete(unclaimed.first.$1);
  }
}