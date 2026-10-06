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
/// gap a focus change missed replays in on the way back. The TERMINAL
/// riders fold the send's close: a `turn.end` (any reason kind — the
/// daemon's turn closers, `lifecycle.h`'s reason union) is THE terminal for
/// this turn's send — a completed turn whose reply was `frame.report`ed
/// (the quiet-completion shape; no `msg.append` exists) completes its
/// waiter with the report's text as the [ChatDone], and an errored turn
/// completes it with a [ChatFailed] (the fold also writes the dimmed
/// assistant line the view renders). A `turn.end` that arrives after its
/// turn's assistant `msg.append` already fed the waiter settles nothing —
/// the fed turn's own closer passes. The remaining cell/control records
/// fold nothing — the chat view never sees them.
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

  /// The pending sends' completers, FIFO per sid — each turn's terminal
  /// (its assistant `msg.append`, or its `turn.end` rider when the turn
  /// closed quietly) completes the HEAD; the parallel-sends' pairing is the
  /// record order's.
  final Map<String, Queue<Completer<ChatEvent>>> _waiters =
      <String, Queue<Completer<ChatEvent>>>{};

  /// The create-race's fast-reply cache: a terminal (the assistant record,
  /// or the turn's `turn.end`) that folded while a pending create had no
  /// sid yet (the fold landed between the prompt's response and the
  /// waiter's registration) waits here for its send; entries die at 60 s
  /// (a fold nobody claimed was history's).
  final Map<String, List<(ChatEvent, DateTime)>> _unclaimed =
      <String, List<(ChatEvent, DateTime)>>{};

  /// The turn closers already owed: every assistant `msg.append` fold that
  /// fed a waiter leaves this turn's own `turn.end` unarrived — a terminal
  /// consuming one settles that fed turn (no waiter touch), so a report
  /// turn's closers never steal the next send.
  final Map<String, int> _fedTurns = <String, int>{};

  /// The quiet-completion texts per sid, FIFO: a `frame.report` (the
  /// payload's text — the cell's outcome wording) that landed before its
  /// `turn.end` rider; the terminal consumes one as its [ChatDone]'s
  /// content (the newest report wins is the log's convention — a turn
  /// closes with at most one).
  final Map<String, Queue<String>> _reports = <String, Queue<String>>{};

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

    final waiter = Completer<ChatEvent>();
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
        final queue =
            _waiters.putIfAbsent(sid, () => Queue<Completer<ChatEvent>>());
        queue.add(waiter);
        _drainUnclaimed(sid, waiter);
      } else if (prompt.queued) {
        final queue = _waiters.putIfAbsent(
            capturedId, () => Queue<Completer<ChatEvent>>());
        queue.add(waiter);
      } else {
        // The prompt failed loud: the send's stream errors into the view's
        // onError (the typing flag closes — the composer keeps its draft).
        throw StateError(
            'the daemon refused the prompt '
            '(status ${prompt.status})');
      }
      yield await waiter.future; // the turn's terminal: a done, or a failed
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
  void _removeWaiter(String sid, Completer<ChatEvent> waiter) {
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
    switch (decoded['type']) {
      case 'msg.append':
        _onMsgAppend(sid, record.seq, decoded['payload']);
        return;
      case 'frame.report':
        _onReport(sid, decoded['payload']);
        return;
      case 'turn.end':
        _onTurnEnd(sid, record.seq, decoded['payload']);
        return;
      default:
        return; // the cell/control records: the chat ignores them
    }
  }

  /// The `msg.append` fold: the bubble, then the waiter (an assistant's
  /// content is this turn's reply — its `turn.end` closer is now owed).
  void _onMsgAppend(String sid, int seq, dynamic payloadDyn) {
    if (payloadDyn is! Map<String, dynamic>) return;
    final role = payloadDyn['role'];
    final content = payloadDyn['content'];
    if (role is! String || content is! String) return;
    if (role != 'user' && role != 'assistant') {
      return; // some other role rides the same record type — fold nothing
    }
    final message = Message(
      // The deterministic (sid, seq) id — the fold's store-side dedupe
      // backstop (the same record's re-delivery never re-appends).
      id: 'm:$sid:$seq',
      role: role == 'user' ? MessageRole.user : MessageRole.assistant,
      content: content,
      timestamp: DateTime.now(),
    );
    _sessions.storeMessage(sid, message);
    if (role == 'assistant') {
      if (_feedHead(sid, ChatDone(content))) {
        _fedTurns[sid] = (_fedTurns[sid] ?? 0) + 1;
      }
    }
  }

  /// The `frame.report` fold: the quiet-completion text waits for its
  /// `turn.end` rider (the record order: report first, then the closer).
  /// The REPORT IS NOT A BUBBLE — the msg.append records stay the message
  /// source of truth; the report only feeds the terminal's content.
  void _onReport(String sid, dynamic payloadDyn) {
    if (payloadDyn is! Map<String, dynamic>) return;
    final text = payloadDyn['text'];
    if (text is! String) return;
    _reports.putIfAbsent(sid, () => Queue<String>()).add(text);
  }

  /// The `turn.end` fold — THE TERMINAL: the turn's send never hangs past
  /// its closer (the finding: a report-completed or errored turn left the
  /// typing flag up until dispose).
  ///
  /// The pairing: this turn's closer, or the NEXT unfed waiter's — an
  /// assistant fold that already fed its send leaves an owed closer here
  /// ([_fedTurns]); a terminal consuming one settles nothing. A waiter was
  /// never fed = the turn closed quietly (the report shape) or failed, so
  /// the terminal's text decides: an `error` kind fails the waiter with
  /// [ChatFailed] (the fold writes the dimmed assistant line), any other
  /// reason kind completes it with the report's text (or honestly empty).
  void _onTurnEnd(String sid, int seq, dynamic payloadDyn) {
    if (payloadDyn is! Map<String, dynamic>) return;
    final reason = payloadDyn['reason'];
    if (reason is! Map<String, dynamic>) return;
    final kind = reason['kind'];
    if (kind is! String) return;

    // The unread report text (FIFO — the record order's pairing), consumed
    // by THIS closer either way.
    String? reportText;
    final reports = _reports[sid];
    if (reports != null && reports.isNotEmpty) {
      reportText = reports.removeFirst();
      if (reports.isEmpty) _reports.remove(sid);
    }

    final fed = _fedTurns[sid] ?? 0;
    if (fed > 0) {
      _fedTurns[sid] = fed - 1; // the fed turn's own closer passes
      return;
    }

    final String controlText =
        reason['text'] is String ? reason['text'] as String : '';
    final bool failed = kind == 'error';
    final String text;
    if (failed) {
      // The failure's wording: the reason's control text, then the turn's
      // report; none readable = the honest fixed line (the bubble's text
      // never renders empty).
      final candidate = controlText.isNotEmpty
          ? controlText
          : (reportText ?? '');
      text = candidate.isNotEmpty ? candidate : 'the turn failed';
    } else {
      text = reportText ?? controlText;
    }
    if (failed) {
      _sessions.storeMessage(
        sid,
        Message(
          // The deterministic (sid, seq) id, `e:`-prefixed — the same
          // (sid, seq) dedupe backstop as the bubbles.
          id: 'e:$sid:$seq',
          role: MessageRole.assistant,
          content: text,
          timestamp: DateTime.now(),
          error: true,
        ),
      );
      _feedHead(sid, ChatFailed(text));
      return;
    }
    _feedHead(sid, ChatDone(text));
  }

  /// The queue's live head takes the folded terminal; a dead entry (a send
  /// whose stream already failed) is skipped over — the pairing rides. No
  /// live waiter: a turn that ran outside any send — a pending create's
  /// window caches the event for its send's registration (the create
  /// race's fast fold), otherwise it is a history fold nobody is on (the
  /// store already has what it shows).
  bool _feedHead(String sid, ChatEvent event) {
    final queue = _waiters[sid];
    if (queue != null) {
      while (queue.isNotEmpty) {
        final waiter = queue.removeFirst();
        if (!waiter.isCompleted) {
          waiter.complete(event);
          return true;
        }
      }
    }
    if (_pendingCreates == 0) return false;
    _unclaimed
        .putIfAbsent(sid, () => <(ChatEvent, DateTime)>[])
        .add((event, DateTime.now()));
    _unclaimed[sid] = _unclaimed[sid]!
        .where(
          (entry) => DateTime.now().difference(entry.$2) <
              const Duration(seconds: 60),
        )
        .toList();
    return true;
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

  /// The waiter's registration claims its create race's fast fold (the same
  /// send's reply that landed between the prompt's response and here —
  /// including its turn's terminal).
  void _drainUnclaimed(String sid, Completer<ChatEvent> waiter) {
    final unclaimed = _unclaimed.remove(sid);
    if (unclaimed == null || unclaimed.isEmpty) return;
    if (!waiter.isCompleted) waiter.complete(unclaimed.first.$1);
  }
}