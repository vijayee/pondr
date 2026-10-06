/// The sa_client binding (`SaClientNative`) — the FFI layer's logic half:
/// the op host (the blocking C ops never run on the main isolate), the
/// `NativeCallable.listener` wiring, the serial-pending map, and the
/// dispose-chain payload protocol.
///
/// THE OP ISOLATE (the flow): the wrapper posts serial'd `{op, args}`
/// commands to a long-lived op host — a background isolate in production
/// ([SaIsolateOpHost]; the blocking C calls are fine there — the UI never
/// sees them) or the same command loop in-process for tests and embedded
/// hosts ([SaInProcessOpHost], whose C calls never block). The op side
/// ([SaClientOpRunner]) owns the client pointer and runs the FIFO command
/// loop. The C's callbacks are `NativeCallable.listener`s created MAIN-side
/// and handed to the C as native function pointers (the config's error_cb;
/// the ops' callback args): they fire on ANY thread and post into the main
/// isolate's queue. An op's Future completes when its callback's post lands
/// — matched by serial (the serial rides through the C as the callback's
/// ctx) — with the op's own done-message as the refusal backstop. NOTE the
/// C contract: a request callback fires BEFORE the blocking call returns,
/// so the listener's post precedes the op-isolate's done message (both
/// hosts enforce that order) — the pending map dedups (its completer
/// resolves exactly once; the first resolution wins, the later arrival is
/// a drop).
///
/// THE DISPOSE-CHAIN PROTOCOL (the plan's pinned shape — see
/// `docs/ffi-binding-plan.md` Task 3's payload-registry bullet; follow it
/// exactly): sa_client HOLDS every payload until `sa_client_release_payload`
/// OR destroy, so a callback's pointer is legal for the main isolate to
/// deref LATER; the danger is only the destroy freeing a payload whose
/// listener post hasn't been read yet.
///
/// (a) every payload-carrying listener's closure on the main isolate: FIRST
///     copies the strings (`SaUtf8.read`), THEN posts a `{release, ptr}`
///     command to the op host (which owns the client and calls
///     `sa_client_release_payload`), THEN resolves ITS LINK in this class's
///     ordered completer chain (the events + error deliveries; a request
///     op's pending entry is its link);
/// (b) listener posts arrive on the main isolate FIFO, so every post queued
///     before the chain's tail resolves is read before the tail resolves —
///     no unread pointer survives the tail;
/// (c) `dispose()` = stop issuing ops → `unsubscribe_events` (awaited, up
///     to the terminal marker) → await the CHAIN's TAIL (all reads done,
///     all releases commanded) → the destroy command → the ack → close the
///     listeners. The C side's own reclaim-at-destroy covers any payload
///     whose release raced; the Dart side never touches a pointer past the
///     tail's resolution.
///
/// THE MANUAL DISPOSE CONTRACT: an `SaClientNative` MUST reach `dispose()` —
/// the five `NativeCallable.listener`s keep the main isolate alive until
/// closed, and the C's client (its reader thread + payloads) is torn down
/// only by destroy. There is no finalizer safety net: dispose IS the
/// safety net.
library;

import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:isolate';

import 'sa_ffi.dart';

// ── the user-facing value types ─────────────────────────────────────────────

/// The config as the app speaks it (the C's `sa_client_config_t`, plain
/// fields; the timeout defaults are the C's `sa_client_config_default`'s —
/// transport UNIX, 5000/10000/0). Strings are copied by the C at connect
/// (the api-key copy is scrubbed by length at destroy); TCP requires a
/// non-empty [apiKey], UNIX ignores it.
final class SaConfig {
  const SaConfig({
    this.transport = SaTransport.unix,
    this.socketPath,
    this.host,
    this.port = 0,
    this.apiKey,
    this.connectTimeoutMs = 5000,
    this.requestTimeoutMs = 10000,
    this.maxRetries = 0,
  });

  final SaTransport transport;
  final String? socketPath;
  final String? host;
  final int port;
  final String? apiKey;
  final int connectTimeoutMs;
  final int requestTimeoutMs;
  final int maxRetries;
}

/// One delivered event: a record (seq > 0, [recordJson] = the store
/// record's JSON verbatim) or a MARKER ([isMarker]: seq 0 + a null
/// recordJson — [op] echoes which transition, the wire's CA_EVENTS_*).
final class SaEventRecord {
  const SaEventRecord({
    required this.sid,
    required this.seq,
    required this.op,
    required this.recordJson,
  });

  final String sid;
  final int seq;
  final int op;
  final String? recordJson;

  bool get isMarker => seq == 0 && recordJson == null;
  bool get isLiveMarker => isMarker && op != saEventsOpUnsubscribe;
  bool get isUnsubscribeAck => isMarker && op == saEventsOpUnsubscribe;

  @override
  String toString() =>
      'SaEventRecord(sid: $sid, seq: $seq, op: $op,'
      ' record: ${recordJson == null ? '-' : 'json'})';
}

/// The failure channel's delivery (every request failure fires here — a
/// daemon refusal, a timeout, a lost connection, a local refusal).
final class SaFailure {
  const SaFailure({
    required this.reqId,
    required this.status,
    required this.text,
  });

  final int reqId;
  final int status;
  final String text;

  @override
  String toString() =>
      'SaFailure(reqId: $reqId, status: $status, text: $text)';
}

/// The callbacks the app registers on connect; both may be null (the C's
/// error_cb is never-null at the binding level — its deliveries still ride
/// the payload rules, they just go nowhere).
final class SaCallbacks {
  const SaCallbacks({this.onEvent, this.onError});

  /// Every events-channel record AND marker rides here.
  final void Function(SaEventRecord event)? onEvent;

  /// Every request failure rides here (the op's own callback ALSO
  /// completes with the failing status — the consumer picks either
  /// channel, neither hangs).
  final void Function(SaFailure failure)? onError;
}

/// The op results: connect/subscribe/unsubscribe/interrupt answer these;
/// prompt and listSessions answer their richer shapes below.
sealed class SaResult {
  const SaResult();
}

final class SaOk extends SaResult {
  const SaOk();

  @override
  String toString() => 'SaOk()';
}

final class SaErr extends SaResult {
  const SaErr(this.status, [this.text]);

  /// The delivered status, or `saClientStatusCallRefused` (-1) when the C
  /// refused the call outright (no callback fired at all, no status rode).
  final int status;
  final String? text;

  @override
  String toString() => 'SaErr(status: $status, text: $text)';
}

/// The prompt's outcome (the C's prompt callback, exactly one per call):
/// [started] (a new frame; [sid] = its "sessions/…" path), [queued] (the
/// steer; the "" sentinel), or [failed] (the status says which; "").
final class SaPromptResult {
  const SaPromptResult(this.status, this.sid);

  final int status;

  /// The new frame's sid; "" for the steer-queued sentinel or a failure.
  final String sid;

  bool get started => status == saClientStatusOk && sid.isNotEmpty;
  bool get queued => status == saClientStatusOk && sid.isEmpty;
  bool get failed => status != saClientStatusOk;

  @override
  String toString() => 'SaPromptResult(status: $status, sid: $sid)';
}

/// The config pair's outcome (the C's `sa_client_config_get`/`_set`
/// callback): status 0 = the three strings are the daemon's frame-config
/// template's POST-SET truth. A NULL member is the template's ABSENT
/// member (the wire's "" sentinel decoded) — a present-but-empty member
/// never exists. A failure delivery carries all-NULL.
final class SaConfigResult {
  const SaConfigResult(this.status, this.baseUrl, this.apiKey, this.model);

  final int status;
  final String? baseUrl;
  final String? apiKey;
  final String? model;

  bool get ok => status == saClientStatusOk;

  @override
  String toString() =>
      'SaConfigResult(status: $status, baseUrl: $baseUrl, '
      'model: $model)';
}

/// One sessions row (the wire's [sid, status, goal, created, depth] shape;
/// a NULL status means unknown, a NULL goal means absent).
final class SaSessionRecord {
  const SaSessionRecord({
    required this.sid,
    required this.status,
    required this.goal,
    required this.created,
    required this.depth,
  });

  final String sid;
  final String? status;
  final String? goal;
  final int created;
  final int depth;

  @override
  String toString() => 'SaSessionRecord(sid: $sid)';
}

/// The listSessions outcome ([rows] is empty on a failure; [status] says
/// which — a refusal surfaces `saClientStatusCallRefused`).
final class SaSessionsResult {
  const SaSessionsResult(this.status, this.rows);

  final int status;
  final List<SaSessionRecord> rows;

  bool get ok => status == saClientStatusOk;

  @override
  String toString() =>
      'SaSessionsResult(status: $status, rows: ${rows.length})';
}

// ── the op protocol (sendable both ways: plain fields only) ─────────────────

/// Main → op. The lifecycle commands carry serial 0 (no callback matches).
sealed class SaOpCommand {
  const SaOpCommand();

  /// The op's serial (0 for the lifecycle commands).
  int get serial;
}

final class SaConnectCmd extends SaOpCommand {
  const SaConnectCmd(this.config);

  @override
  int get serial => 0;

  final SaConfigFields config;
}

final class SaPromptCmd extends SaOpCommand {
  const SaPromptCmd(this.serial, this.sid, this.text, this.cb);

  @override
  final int serial;
  final String? sid;
  final String text;

  /// The prompt callback's native function pointer (an address).
  final int cb;
}

final class SaInterruptCmd extends SaOpCommand {
  const SaInterruptCmd(this.serial, this.sid, this.cb);

  @override
  final int serial;
  final String sid;
  final int cb;
}

final class SaListCmd extends SaOpCommand {
  const SaListCmd(this.serial, this.cb);

  @override
  final int serial;
  final int cb;
}

final class SaSubscribeCmd extends SaOpCommand {
  const SaSubscribeCmd(this.serial, this.sid, this.cb);

  @override
  final int serial;
  final String sid;
  final int cb;
}

final class SaUnsubscribeCmd extends SaOpCommand {
  const SaUnsubscribeCmd(this.serial);

  @override
  final int serial;
}

final class SaConfigGetCmd extends SaOpCommand {
  const SaConfigGetCmd(this.serial, this.cb);

  @override
  final int serial;

  /// The config callback's native function pointer (an address).
  final int cb;
}

final class SaConfigSetCmd extends SaOpCommand {
  const SaConfigSetCmd(this.serial, this.baseUrl, this.apiKey, this.model,
      this.cb);

  @override
  final int serial;

  /// The set members: null rides ABSENT (the template's member is kept; an
  /// empty string is ALSO the wire's absent sentinel — set to text, never
  /// to empty).
  final String? baseUrl;
  final String? apiKey;
  final String? model;

  /// The config callback's native function pointer (an address).
  final int cb;
}

final class SaReleaseCmd extends SaOpCommand {
  const SaReleaseCmd(this.payload);

  @override
  int get serial => 0;

  /// The held payload pointer (an address; never 0 — a NULL payload never
  /// reaches a callback per the C's ownership rule).
  final int payload;
}

final class SaDestroyCmd extends SaOpCommand {
  const SaDestroyCmd();

  @override
  int get serial => 0;
}

/// Stops the op host's command loop (the dispose flow's tail, after the
/// destroy's ack; the entry closes its port and the isolate exits).
final class SaOpExitCmd extends SaOpCommand {
  const SaOpExitCmd();

  @override
  int get serial => 0;
}

/// Op → main (the op → main handshake rides the host's own plumbing).
sealed class SaOpEvent {
  const SaOpEvent();
}

/// The connect refused (the C returned the null client).
final class SaOpRefused extends SaOpEvent {
  const SaOpRefused(this.reason);

  final String reason;

  @override
  String toString() => 'SaOpRefused(reason: $reason)';
}

/// The connect succeeded (the client pointer stays op-side — the main
/// isolate never holds it).
final class SaOpConnected extends SaOpEvent {
  const SaOpConnected();
}

/// An op's C return: 0 = its callback ran (or, for the done-driven ops, the
/// subscription's transition landed); -1 = the outright refusal (NO
/// callback fires at all). The callback-driven ops IGNORE a ret >= 0 done —
/// their callback's listener post resolves them (see the library doc's
/// dedup note).
final class SaOpDone extends SaOpEvent {
  const SaOpDone(this.serial, this.ret);

  final int serial;
  final int ret;

  @override
  String toString() => 'SaOpDone(serial: $serial, ret: $ret)';
}

/// The destroy acked (the reader joined, the payloads reclaimed).
final class SaOpDestroyed extends SaOpEvent {
  const SaOpDestroyed();
}

/// The op isolate's handshake carrying a load failure (the entry failed to
/// open the .so — surfaced as the connect's honest error instead of a
/// silent isolate death).
final class SaOpHandshakeFailed {
  SaOpHandshakeFailed(this.reason);

  final String reason;
}

// ── the op side ─────────────────────────────────────────────────────────────

/// The op-side command loop: one client pointer (+ its config alloc until
/// connect), the blocking C ops run HERE, commands handled FIFO, events
/// emitted per command. The op isolate's entry and the in-process host
/// both drive it. The client pointer NEVER leaves this object: the main
/// isolate has no need for it, and the payload releases ride back as
/// commands.
final class SaClientOpRunner {
  SaClientOpRunner(this.api);

  final SaFfiApi api;

  /// The client pointer — op-side ONLY.
  int? _client;

  /// An op issued while the client is absent (a leaky command order) is the
  /// C's outright refusal — 0 reads as the null client to every op.
  int get _clientOrBust => _client ?? 0;

  void handle(
    SaOpCommand command, {
    required void Function(SaOpEvent) emit,
  }) {
    switch (command) {
      case SaConnectCmd(:final config):
        _connect(config, emit);
      case SaPromptCmd(:final serial, :final sid, :final text, :final cb):
        emit(SaOpDone(
            serial, api.prompt(_clientOrBust, serial, sid, text, cb)));
      case SaInterruptCmd(:final serial, :final sid, :final cb):
        emit(SaOpDone(
            serial, api.interrupt(_clientOrBust, serial, sid, cb)));
      case SaListCmd(:final serial, :final cb):
        emit(SaOpDone(serial, api.listSessions(_clientOrBust, serial, cb)));
      case SaSubscribeCmd(:final serial, :final sid, :final cb):
        emit(SaOpDone(
            serial, api.subscribeEvents(_clientOrBust, serial, sid, cb)));
      case SaUnsubscribeCmd(:final serial):
        emit(SaOpDone(serial, api.unsubscribeEvents(_clientOrBust)));
      case SaConfigGetCmd(:final serial, :final cb):
        emit(SaOpDone(serial, api.configGet(_clientOrBust, serial, cb)));
      case SaConfigSetCmd(
            :final serial,
            :final baseUrl,
            :final apiKey,
            :final model,
            :final cb,
          ):
        emit(SaOpDone(
            serial, api.configSet(_clientOrBust, serial, baseUrl, apiKey, model, cb)));
      case SaReleaseCmd(:final payload):
        // After destroy this mirrors the C's release-after-destroy no-op.
        final client = _client;
        if (client != null) api.releasePayload(client, payload);
      case SaDestroyCmd():
        _destroy(emit);
      case SaOpExitCmd():
        break; // the host's own plumbing, never the runner's command
    }
  }

  void _connect(SaConfigFields config, void Function(SaOpEvent) emit) {
    if (_client != null) {
      emit(const SaOpRefused('the op runner is single-client'));
      return;
    }
    final config0 = api.buildConfig(config);
    final client = api.connect(config0);
    // The C copies every string at connect — the held struct's job is done
    // (every failure path returns BEFORE holding anything, and a mid-way
    // failure tears the client's own copies down with it).
    api.freeConfig(config0);
    if (client == 0) {
      emit(const SaOpRefused(
          'the connect refused (the C returned the null client)'));
      return;
    }
    _client = client;
    emit(const SaOpConnected());
  }

  void _destroy(void Function(SaOpEvent) emit) {
    final client = _client;
    _client = null;
    if (client != null) api.destroy(client);
    emit(const SaOpDestroyed());
  }
}

// ── the op hosts ────────────────────────────────────────────────────────────

/// The op host: where the blocking C ops run. `start` completes once the op
/// side is live (its command port FIFO-safe); `send` posts a command
/// WITHOUT blocking; `events` carries the op side's answers in order;
/// `stop` ends the host (awaiting its exit).
abstract interface class SaOpHost {
  bool get isRunning;

  Future<void> start(SaFfiApi api);

  /// The command sender (strict FIFO); commands posted after `stop` are
  /// dropped (the C's release-after-destroy no-op, mirrored).
  void send(SaOpCommand command);

  Stream<SaOpEvent> get events;

  Future<void> stop();
}

/// The REAL op host: a long-lived background isolate. The entry loads the
/// SAME .so (`DynamicLibrary.open` is process-wide — the same handle), runs
/// the same [SaClientOpRunner] loop, and exits on [SaOpExitCmd].
final class SaIsolateOpHost implements SaOpHost {
  final StreamController<SaOpEvent> _events = StreamController<SaOpEvent>();
  final Completer<SendPort> _commands = Completer<SendPort>();
  final Completer<void> _exited = Completer<void>();
  late final ReceivePort _handshakePort = ReceivePort();
  late final ReceivePort _eventsPort = ReceivePort();
  late final ReceivePort _exitPort = ReceivePort();
  bool _spawned = false;
  bool _handshakeDone = false;
  bool _closed = false;

  @override
  bool get isRunning => _spawned && _handshakeDone && !_closed;

  @override
  Future<void> start(SaFfiApi api) async {
    final libraryPath = api.libraryPath;
    if (libraryPath == null) {
      throw StateError(
        'the op host needs a dlopen-backed api (a fake api takes an '
        'explicit host — pass SaOpHost at the SaClientNative constructor)',
      );
    }
    _handshakePort.listen(_onHandshake);
    _eventsPort.listen((Object? event) => _events.add(event as SaOpEvent));
    _exitPort.listen((_) => _onExit());
    try {
      await Isolate.spawn(
        saClientOpIsolateEntry,
        <Object>[_handshakePort.sendPort, _eventsPort.sendPort, libraryPath],
        onExit: _exitPort.sendPort,
      );
    } catch (e) {
      // The spawn itself refused: the attached ports close before the
      // throw (nothing can answer their senders anymore).
      _handshakePort.close();
      _eventsPort.close();
      _exitPort.close();
      rethrow;
    }
    _spawned = true;
    await _commands.future;
  }

  void _onHandshake(Object? message) {
    if (message is SaOpHandshakeFailed) {
      // The entry's own load failed loud — surfaced as the connect's error.
      _commands.completeError(StateError(message.reason));
      return;
    }
    if (message is SendPort && !_handshakeDone) {
      _handshakeDone = true;
      _commands.complete(message);
    }
  }

  void _onExit() {
    if (!_commands.isCompleted) {
      _commands.completeError(
        StateError('the op isolate died before its handshake'),
      );
    }
    if (!_exited.isCompleted) _exited.complete();
    _events.close();
  }

  @override
  Stream<SaOpEvent> get events => _events.stream;

  @override
  void send(SaOpCommand command) {
    if (_closed) {
      // Post-stop strays: dropped (the C's release-after-destroy no-op).
      return;
    }
    if (!_handshakeDone) {
      throw StateError('the op isolate is not started');
    }
    _commands.future.then((port) => port.send(command));
  }

  @override
  Future<void> stop() async {
    if (!_spawned) return;
    _closed = true;
    if (!_handshakeDone) {
      // The handshake failed (the entry's load): the isolate exits by
      // itself — its exit port answers below; only the ports close.
    } else if (!_exited.isCompleted) {
      (await _commands.future).send(const SaOpExitCmd());
    }
    await _exited.future;
    _handshakePort.close();
    _eventsPort.close();
    _exitPort.close();
    await _events.close();
  }
}

/// The op isolate's entry — a top-level function (Isolate.spawn needs one).
/// Loads the .so itself (a load failure reports through the handshake, not
/// a silent isolate death), runs the FIFO loop, exits on [SaOpExitCmd].
@pragma('vm:entry-point')
void saClientOpIsolateEntry(List<Object> message) {
  final handshake = message[0] as SendPort;
  final events = message[1] as SendPort;
  final libraryPath = message[2] as String;
  final SaFfi api;
  try {
    api = SaFfi(libraryPath)..load();
  } catch (e) {
    handshake.send(SaOpHandshakeFailed(e.toString()));
    return;
  }
  final runner = SaClientOpRunner(api);
  final commands = ReceivePort();
  handshake.send(commands.sendPort);
  commands.listen((command) {
    if (command is SaOpExitCmd) {
      commands.close();
      return;
    }
    runner.handle(command as SaOpCommand, emit: events.send);
  });
}

/// The in-process op host: the SAME command loop ([SaClientOpRunner]) run
/// in-process — for the unit tests (which inject a fake api whose C calls
/// never block) and for an embedded host that makes the same guarantee.
///
/// The queue discipline mirrors the isolate's single message queue: a
/// command processes SYNCHRONOUSLY at its send (the fake's blocking call is
/// instant), but every emitted event is re-queued as a TASK — so anything
/// the C call fired along the way (the callback posts, queued during the
/// call) lands BEFORE that command's own done. The dispose flow rides a
/// task boundary before its chain drain, so both hosts guarantee the same
/// ordering.
final class SaInProcessOpHost implements SaOpHost {
  SaInProcessOpHost();

  SaClientOpRunner? _runner;
  final StreamController<SaOpEvent> _events = StreamController<SaOpEvent>();
  bool _running = false;

  @override
  bool get isRunning => _running;

  @override
  Future<void> start(SaFfiApi api) async {
    _runner = SaClientOpRunner(api);
    _running = true;
  }

  @override
  Stream<SaOpEvent> get events => _events.stream;

  /// The event's task-boundary delivery (the class doc's discipline).
  void _emit(SaOpEvent event) {
    Future<void>(() => _events.add(event));
  }

  final Completer<void> _stopped = Completer<void>();

  @override
  void send(SaOpCommand command) {
    if (command is SaOpExitCmd) {
      _running = false;
      _stopped.complete();
      _events.close();
      return;
    }
    if (!isRunning) {
      // Post-stop strays: dropped (the C's release-after-destroy no-op).
      return;
    }
    _runner!.handle(command, emit: _emit);
  }

  @override
  Future<void> stop() async {
    if (!isRunning) return;
    send(const SaOpExitCmd());
    await _stopped.future;
  }
}

// ── the wrapper ─────────────────────────────────────────────────────────────

enum _SaClientState { idle, connecting, ready, closing, closed }

/// The op kinds the pending map distinguishes: a `doneDriven` one (subscribe
/// / unsubscribe) resolves on its done message; the callback-driven ones
/// (prompt / interrupt / listSessions / the config pair) resolve on their
/// callback's listener post — and each kind names its OWN refusal shape (the
/// C's -1 return fires no callback and carries no status).
enum _SaOpKind { prompt, interrupt, list, subscribe, unsubscribe, configGet, configSet }

/// One op's pending completion: serial-keyed. The map's remove-on-complete
/// IS the dedup (the callback's post may land before or after the done;
/// the first resolution wins, the later arrival is a drop).
final class _PendingOp {
  _PendingOp({required this.kind, required this.serial});

  final Completer<Object?> completer = Completer<Object?>();
  final _SaOpKind kind;
  final int serial;

  bool get doneDriven =>
      kind == _SaOpKind.subscribe || kind == _SaOpKind.unsubscribe;

  /// The outright refusal's value (the C's -1 return; no callback ever).
  Object? refusal(int status) => switch (kind) {
        _SaOpKind.prompt => SaPromptResult(status, ''),
        _SaOpKind.interrupt => SaErr(status),
        _SaOpKind.list => SaSessionsResult(status, const <SaSessionRecord>[]),
        _SaOpKind.subscribe || _SaOpKind.unsubscribe => SaErr(status),
        _SaOpKind.configGet || _SaOpKind.configSet =>
          SaConfigResult(status, null, null, null),
      };

  Future<Object?> get future => completer.future;
}

/// The sa_client wrapper (the spec's SaClientNative): the app's one handle
/// to a C client — [connect] once, then the ops, then [dispose] (MANDATORY:
/// the library doc's manual dispose contract).
///
/// Concurrency: all commands run FIFO through the op host, one blocking C
/// op at a time — concurrent callers just queue (the C's one-in-flight-
/// request shape, made safe). Ops issued after `dispose()` started throw
/// StateError — the app's lifecycle owns the wrapper.
final class SaClientNative {
  SaClientNative({required this.api, SaOpHost? host})
      : host = host ?? SaIsolateOpHost();

  /// The injected ABI seam (the spec's SaFfi indirection).
  final SaFfiApi api;

  /// The op host (the .so-backed isolate one by default).
  final SaOpHost host;

  var _state = _SaClientState.idle;
  bool _subscribed = false;
  int _serial = 0;
  SaCallbacks? _callbacks;
  Future<void>? _closing;

  /// The connect's refusal fact, completed at the teardown's tail (the
  /// connect's caller resumes only after the teardown it can observe): the
  /// C's own refusal's reason, or the dispose/host-failure reason.
  String? _refusal;

  StreamSubscription<SaOpEvent>? _eventsSub;

  /// The six listeners (created on connect; closed at the dispose flow's
  /// ack or a failed connect — a NativeCallable.listener keeps the main
  /// isolate alive until closed).
  late final ffi.NativeCallable<SaPromptCbNative> _promptCb;
  late final ffi.NativeCallable<SaInterruptCbNative> _interruptCb;
  late final ffi.NativeCallable<SaSessionsCbNative> _sessionsCb;
  late final ffi.NativeCallable<SaEventsCbNative> _eventsCb;
  late final ffi.NativeCallable<SaErrorCbNative> _errorCb;
  late final ffi.NativeCallable<SaConfigCbNative> _configCb;
  bool _listenersOpen = false;

  final Map<int, _PendingOp> _pending = <int, _PendingOp>{};

  /// THE CHAIN: the events + error deliveries' ordered completer links (see
  /// the library doc's dispose-chain protocol).
  final List<Completer<void>> _chain = <Completer<void>>[];

  final Completer<SaResult> _connect = Completer<SaResult>();
  final Completer<void> _destroyed = Completer<void>();

  /// The in-flight subscribe's in-window error check: the C's subscribe can
  /// return 0 with the daemon's refusal having been delivered through the
  /// ERROR callback instead of a live marker (the refusal's callback always
  /// fires INSIDE the C call, so its post precedes the done's turn).
  bool _windowSaw = false;
  int _windowStatus = 0;
  int _subscribeWindow = 0;

  bool get isDisposed => _state == _SaClientState.closed;

  bool get isConnected => _state == _SaClientState.ready;

  /// The subscription's active state (false until a subscribe's SaOk).
  bool get isSubscribed => _subscribed;

  void _guardReady() {
    if (_state != _SaClientState.ready) {
      throw StateError(
        'the client is not available (state: ${_state.name}) — '
        'connect first, and never after dispose',
      );
    }
  }

  _PendingOp _beginOp(_SaOpKind kind) {
    final serial = ++_serial;
    final pending = _PendingOp(kind: kind, serial: serial);
    _pending[serial] = pending;
    return pending;
  }

  // ── connect ──

  /// Opens the C client: the connection + the callbacks' wiring. The ops'
  /// futures are only issuable once this completes.
  Future<SaResult> connect(SaConfig config, SaCallbacks callbacks) async {
    if (_state != _SaClientState.idle) {
      throw StateError('connect is one-shot (state: ${_state.name})');
    }
    _state = _SaClientState.connecting;
    _callbacks = callbacks;
    api.load();
    _createListeners();
    _eventsSub = host.events.listen(_onOpEvent);
    try {
      await host.start(api);
    } catch (e) {
      _refusal ??= 'the op host failed to start: $e';
      await dispose();
      return _connect.future;
    }
    if (_state != _SaClientState.connecting) {
      // A dispose raced the connect attempt: nothing connects after it
      // started (the tail's refusal completes the caller below).
      return _connect.future;
    }
    host.send(SaConnectCmd(_configFields(config)));
    return _connect.future;
  }

  SaConfigFields _configFields(SaConfig config) => SaConfigFields(
        transport: config.transport.valueOf,
        socketPath: config.socketPath,
        host: config.host,
        port: config.port,
        apiKey: config.apiKey,
        connectTimeoutMs: config.connectTimeoutMs,
        requestTimeoutMs: config.requestTimeoutMs,
        maxRetries: config.maxRetries,
        errorCb: _errorCb.nativeFunction.address,
      );

  // ── the ops ──

  /// The prompt: [sid] null creates + starts a top frame (the response's
  /// sid names it), a set sid is the steer.
  Future<SaPromptResult> prompt(String? sid, String text) {
    _guardReady();
    final pending = _beginOp(_SaOpKind.prompt);
    host.send(SaPromptCmd(
      pending.serial,
      sid,
      text,
      _promptCb.nativeFunction.address,
    ));
    return pending.future.then((value) => value! as SaPromptResult);
  }

  Future<SaResult> interrupt(String sid) {
    _guardReady();
    final pending = _beginOp(_SaOpKind.interrupt);
    host.send(SaInterruptCmd(
      pending.serial,
      sid,
      _interruptCb.nativeFunction.address,
    ));
    return pending.future.then((value) => value! as SaResult);
  }

  Future<SaSessionsResult> listSessions() {
    _guardReady();
    final pending = _beginOp(_SaOpKind.list);
    host.send(SaListCmd(pending.serial, _sessionsCb.nativeFunction.address));
    return pending.future.then((value) => value! as SaSessionsResult);
  }

  /// Subscribes ONE sid's events channel (replay-then-live from seq 0); the
  /// Future resolves when the subscription is live (the in-window error
  /// shapes the Err). The delivered records AND markers ride
  /// [SaCallbacks.onEvent] — they are the app's message source of truth
  /// (the spec's chat mapping).
  Future<SaResult> subscribeEvents(String sid) {
    _guardReady();
    final pending = _beginOp(_SaOpKind.subscribe);
    _subscribeWindow = pending.serial;
    _windowSaw = false;
    _windowStatus = 0;
    host.send(SaSubscribeCmd(
      pending.serial,
      sid,
      _eventsCb.nativeFunction.address,
    ));
    return pending.future.then((value) {
      final result = value! as SaResult;
      if (result is SaOk) _subscribed = true;
      return result;
    });
  }

  /// Ends the active subscription (blocks until the terminal marker was
  /// delivered — or the timeout). On SaErr the subscription may STILL be
  /// active (the C's timeout path) — the bookkeeping stays then.
  Future<SaResult> unsubscribeEvents() {
    _guardReady();
    return _unsubscribe();
  }

  /// The daemon's frame-config template (the CA_CONFIG get; the all-absent
  /// request shape). status 0 = the template's truth; the ABSENT members
  /// ride null.
  Future<SaConfigResult> configGet() {
    _guardReady();
    final pending = _beginOp(_SaOpKind.configGet);
    host.send(SaConfigGetCmd(
      pending.serial,
      _configCb.nativeFunction.address,
    ));
    return pending.future.then((value) => value! as SaConfigResult);
  }

  /// Mutates the daemon's frame-config template (the CA_CONFIG set): each
  /// NON-NULL member rides its set; null is absent (unchanged) — and an
  /// empty string is ALSO the wire's absent sentinel. The answer carries
  /// the template's post-set truth.
  Future<SaConfigResult> configSet({
    String? baseUrl,
    String? apiKey,
    String? model,
  }) {
    _guardReady();
    final pending = _beginOp(_SaOpKind.configSet);
    host.send(SaConfigSetCmd(
      pending.serial,
      baseUrl,
      apiKey,
      model,
      _configCb.nativeFunction.address,
    ));
    return pending.future.then((value) => value! as SaConfigResult);
  }

  /// The unsubscribe's body without the ready guard (the dispose flow's
  /// internal shape; the command carries the pending's serial).
  Future<SaResult> _unsubscribe() {
    final pending = _beginOp(_SaOpKind.unsubscribe);
    host.send(SaUnsubscribeCmd(pending.serial));
    return pending.future.then((value) {
      final result = value! as SaResult;
      if (result is SaOk) _subscribed = false;
      return result;
    });
  }

  // ── the listeners ──

  void _createListeners() {
    _promptCb = ffi.NativeCallable<SaPromptCbNative>.listener(_onPrompt);
    _interruptCb =
        ffi.NativeCallable<SaInterruptCbNative>.listener(_onInterrupt);
    _sessionsCb = ffi.NativeCallable<SaSessionsCbNative>.listener(_onSessions);
    _eventsCb = ffi.NativeCallable<SaEventsCbNative>.listener(_onEvent);
    _errorCb = ffi.NativeCallable<SaErrorCbNative>.listener(_onError);
    _configCb = ffi.NativeCallable<SaConfigCbNative>.listener(_onConfig);
    _listenersOpen = true;
  }

  void _closeListeners() {
    if (!_listenersOpen) return;
    _promptCb.close();
    _interruptCb.close();
    _sessionsCb.close();
    _eventsCb.close();
    _errorCb.close();
    _configCb.close();
    _listenersOpen = false;
  }

  /// THE DISPOSE-CHAIN, step (a): copy the held string FIRST, post its
  /// release to the op host SECOND (the op side owns the client and calls
  /// sa_client_release_payload), THEN the caller resolves its link.
  String? _readAndRelease(ffi.Pointer<ffi.Uint8> payload) {
    if (payload.address == 0) return null;
    final text = SaUtf8.read(payload);
    host.send(SaReleaseCmd(payload.address));
    return text;
  }

  void _onPrompt(
      ffi.Pointer<ffi.Void> ctx, int status, ffi.Pointer<ffi.Uint8> sid) {
    // (a) the copy → (b) the release post → (c) this op's resolution.
    final sidText = _readAndRelease(sid);
    _resolveOp(ctx.address, SaPromptResult(status, sidText ?? ''));
  }

  void _onInterrupt(ffi.Pointer<ffi.Void> ctx, int status) {
    _resolveOp(ctx.address,
        status == saClientStatusOk ? const SaOk() : SaErr(status));
  }

  void _onSessions(ffi.Pointer<ffi.Void> ctx, int status,
      ffi.Pointer<SaSessionRowFfi> rows, int nrows) {
    if (rows.address == 0) {
      // rows NULL = the request FAILED — nothing to release.
      _resolveOp(ctx.address, SaSessionsResult(status, <SaSessionRecord>[]));
      return;
    }
    // Copy every field of every row FIRST (NULL fields need no release;
    // the rows array itself is ONE held payload), THEN post the releases,
    // THEN resolve.
    final records = <SaSessionRecord>[];
    final releases = <int>[rows.address];
    for (var i = 0; i < nrows; i++) {
      final row = rows[i];
      final sidText = row.sid.address == 0 ? null : SaUtf8.read(row.sid);
      final statusText =
          row.status.address == 0 ? null : SaUtf8.read(row.status);
      final goalText = row.goal.address == 0 ? null : SaUtf8.read(row.goal);
      for (final field
          in <ffi.Pointer<ffi.Uint8>>[row.sid, row.status, row.goal]) {
        if (field.address != 0) releases.add(field.address);
      }
      records.add(SaSessionRecord(
        sid: sidText ?? '',
        status: statusText,
        goal: goalText,
        created: row.created,
        depth: row.depth,
      ));
    }
    for (final payload in releases) {
      host.send(SaReleaseCmd(payload));
    }
    _resolveOp(ctx.address, SaSessionsResult(status, records));
  }

  void _onConfig(ffi.Pointer<ffi.Void> ctx, int status,
      ffi.Pointer<ffi.Uint8> baseUrl, ffi.Pointer<ffi.Uint8> apiKey,
      ffi.Pointer<ffi.Uint8> model) {
    // The sessions callback's shape, three strings wide: the PRESENT
    // members are held payloads (copy → release → resolve); the absent
    // members ride NULL and need no release.
    final base = baseUrl.address == 0 ? null : _readAndRelease(baseUrl);
    final key = apiKey.address == 0 ? null : _readAndRelease(apiKey);
    final tag = model.address == 0 ? null : _readAndRelease(model);
    _resolveOp(ctx.address, SaConfigResult(status, base, key, tag));
  }

  void _onEvent(ffi.Pointer<ffi.Void> ctx, ffi.Pointer<ffi.Uint8> sid, int seq,
      int op, ffi.Pointer<ffi.Uint8> recordJson) {
    final link = Completer<void>();
    _chain.add(link);
    if (ctx.address == _subscribeWindow &&
        seq == 0 &&
        recordJson.address == 0 &&
        op != saEventsOpUnsubscribe) {
      // The subscription's live marker (the sub's ctx carries the
      // subscribe's serial): it resolves the subscribe ahead of its done.
      _closeSubscribeWindow();
      _resolveOp(ctx.address, const SaOk());
    }
    // (a) copy → (b) the release posts → the user callback → (c) the link
    // is the handler's LAST statement, so the chain's tail implies every
    // handler has done its own work.
    final sidText = _readAndRelease(sid);
    final recordText = _readAndRelease(recordJson);
    _callbacks?.onEvent?.call(SaEventRecord(
      sid: sidText ?? '',
      seq: seq,
      op: op,
      recordJson: recordText,
    ));
    link.complete();
  }

  void _onError(ffi.Pointer<ffi.Void> ctx, int reqId, int status,
      ffi.Pointer<ffi.Uint8> text) {
    final link = Completer<void>();
    _chain.add(link);
    if (_subscribeWindow != 0) {
      _windowSaw = true;
      _windowStatus = status;
    }
    final body = _readAndRelease(text);
    _callbacks?.onError
        ?.call(SaFailure(reqId: reqId, status: status, text: body ?? ''));
    link.complete();
  }

  // ── the op event handlers ──

  void _onOpEvent(SaOpEvent event) {
    switch (event) {
      case SaOpConnected():
        _state = _SaClientState.ready;
        _completeConnect(const SaOk());
      case SaOpRefused(:final reason):
        // The failed connect tears itself down in the flow that names the
        // refusal — the completer completes at the teardown's tail, so the
        // connect's caller resumes with the teardown observed.
        _refusal ??= reason;
        _closing ??= _disposeFlow();
      case SaOpDone(:final serial, :final ret):
        _onDone(serial, ret);
      case SaOpDestroyed():
        if (!_destroyed.isCompleted) _destroyed.complete();
    }
  }

  void _onDone(int serial, int ret) {
    final pending = _pending[serial];
    if (ret < 0) {
      // The outright refusal (NO callback fired at all): the done is the
      // backstop — and a subscribe's window closes with it.
      if (serial == _subscribeWindow) _closeSubscribeWindow();
      if (pending != null) {
        _pending.remove(serial);
        pending.completer.complete(pending.refusal(ret));
      }
      return;
    }
    if (pending == null) {
      // The callback/marker resolved it already (the dedup), or a foreign
      // serial; a subscribe's window closes either way.
      if (serial == _subscribeWindow) _closeSubscribeWindow();
      return;
    }
    if (pending.doneDriven) {
      _pending.remove(serial);
      pending.completer.complete(_doneDrivenValue(serial, pending));
      return;
    }
    // ret >= 0, callback-driven: the callback's post completes this op
    // (already queued or next turn — the C contract: ret 0 = the callback
    // ran).
  }

  /// The done-driven ops' value: the subscribe consults the window (an
  /// in-flight error = the daemon refused; a dead sub); the unsubscribe's
  /// done = the terminal marker was delivered.
  SaResult _doneDrivenValue(int serial, _PendingOp pending) {
    if (pending.kind == _SaOpKind.subscribe && serial == _subscribeWindow) {
      final refused = _windowSaw;
      final status = _windowStatus;
      _closeSubscribeWindow();
      if (refused) return SaErr(status);
    }
    return const SaOk();
  }

  void _closeSubscribeWindow() {
    _subscribeWindow = 0;
    _windowSaw = false;
    _windowStatus = 0;
  }

  void _resolveOp(int serial, Object? value) {
    final pending = _pending.remove(serial);
    if (pending != null) {
      pending.completer.complete(value);
    }
  }

  void _completeConnect(SaResult result) {
    if (!_connect.isCompleted) _connect.complete(result);
  }

  // ── the dispose chain ──

  /// THE DISPOSE-CHAIN PROTOCOL (c): stop issuing → unsubscribe (awaited,
  /// up to the terminal marker) → await the chain's TAIL → the destroy
  /// command → the ack → close the listeners. Double-dispose safe: the
  /// same flow, exactly one destroy. An unresolved connect completes at
  /// the tail (with the refusal's reason, or the mid-connect dispose's) —
  /// nobody's connect Future hangs.
  Future<void> dispose() => _closing ??= _disposeFlow();

  Future<void> _disposeFlow() async {
    if (_state == _SaClientState.connecting) {
      _refusal ??= 'disposed during connect';
    }
    final wasReady = _state == _SaClientState.ready;
    final wasSubscribed = wasReady && _subscribed;
    _state = _SaClientState.closing;
    if (wasSubscribed) {
      _subscribed = false;
      try {
        await _unsubscribe();
      } on StateError {
        // Closing raced the subscription's own end — destroy covers it.
      }
    }
    // Every delivery the C handed over before the unsubscribe's return (or
    // before dispose started) was QUEUED as a listener post — flush the
    // main isolate's queue at a task boundary so every queued post's
    // handler (copy → release post → link) runs BEFORE the tail's drain
    // (the protocol's (b): the posts are FIFO, so the tail resolves with
    // all of them read — no unread pointer survives it).
    await Future<void>.delayed(Duration.zero);
    // The chain's TAIL: every link's handler has run by the time the last
    // link resolves; links appended while draining ride the next round (a
    // delivery the reader was still handing over).
    while (_chain.isNotEmpty) {
      final links = List<Completer<void>>.of(_chain);
      _chain.clear();
      await Future.wait(links.map((link) => link.future));
    }
    if (host.isRunning) {
      host.send(const SaDestroyCmd());
      await _destroyed.future;
    }
    await host.stop();
    _closeListeners();
    await _eventsSub?.cancel();
    _state = _SaClientState.closed;
    // The connect's caller resumes LAST (with the teardown observed).
    _completeConnect(
        SaErr(saClientStatusCallRefused, _refusal ?? 'disposed'));
  }
}