/// The FFI layer's tests (the plan's Task 3 step 3): the fake-ABI seam's
/// in-process suite (the callback lifecycle + the dedup, the payload
/// copy→release→link order, THE DISPOSE CHAIN, the op host's
/// spawn/teardown), the mirror's non-live shape pin, and the LIVE
/// struct-drift tripwire (skip unless a `libsa_client.so` exists — set
/// `SA_LIBRARY_PATH` to the SecretAgent build's, e.g.
/// `SA_LIBRARY_PATH=$SECRETAGENT/cmake-build-debug/libsa_client.so`).
library;

import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pondr/ffi/sa_client_binding.dart';
import 'package:pondr/ffi/sa_ffi.dart';

// ── the pinned constants (the struct-drift tripwire, sa_ffi.dart) ───────────

const int kConfigSizeof = 72;
const List<int> kConfigOffsets = <int>[0, 8, 16, 24, 32, 40, 44, 48, 56, 64];

// ── the fake ABI ────────────────────────────────────────────────────────────

/// A one sessions row's fake shape (sid, status?, goal?, created, depth).
typedef FakeRow = ({
  String sid,
  String? status,
  String? goal,
  int created,
  int depth,
});

/// A one event's fake shape (sid, seq, op, recordJson?).
typedef FakeEvent = ({String sid, int seq, int op, String? recordJson});

/// The fake `SaFfiApi`: implements the C side in-process — allocates REAL
/// held payloads (malloc'd, tracked, freed on release), fires the callbacks
/// through the SAME native function pointers the binding handed it (the
/// NativeCallable posts, exactly the production mechanism), and records
/// every api-level call in [order] for the evidence assertions. No .so
/// involved.
final class FakeSaFfi implements SaFfiApi {
  /// The call order (the record's labels), including every payload release
  /// — the dispose-chain evidence's source.
  final List<String> order = <String>[];

  /// The held payloads: address → its order-log label.
  final Map<int, String> _labels = <int, String>{};

  // ── the scripts (per-test knobs, defaulted to the C's happy shape) ──

  /// The connect: false → the C's null client (the refusal).
  bool connectOk = true;

  /// The prompt's one callback ([promptFires] false → the -1 outright
  /// refusal, NO callback per the C); [deferCallbacks] moves the fire past
  /// the op's own done — the dedup's opposite order of the C's contract.
  bool promptFires = true;
  bool deferCallbacks = false;
  int promptStatus = 0;
  String? promptSid;

  /// The listSessions' rows; null = the FAILURE delivery (the callback
  /// fires with NULL rows + this failing status).
  List<FakeRow>? listed = const <FakeRow>[];
  int listedStatus = saClientStatusOk;

  /// The events channel: deliveries hold until [flushEvents] (or the
  /// unsubscribe's terminal ack, which flushes first — the reader's shape).
  bool holdEvents = false;
  final List<FakeEvent> queuedEvents = <FakeEvent>[];

  /// The subscribe's in-window refusal script (>= 0): the fake fires the
  /// ERROR callback INSIDE the subscribe call — the C's exact shape (the
  /// daemon's refusal rides the error channel, the call still returns 0).
  int subRefusalStatus = -1;
  int subRefusalReqId = 77;
  String subRefusalText = 'refused';

  /// The CONFIG pair's script: the template's truth the callback answers
  /// (the fixture daemon's default: base/key absent, the model the
  /// template's own default tag).
  int configStatus = 0;
  String? configBase;
  String? configKey;
  String? configModel;

  /// The CONFIG ops' outright refusal script (false = the -1 return, NO
  /// callback per the C's contract).
  bool configFires = true;

  /// The recorded SET calls (the members as they rode — absent = null).
  final List<(String?, String?, String?)> configSets = <(String?, String?, String?)>[];
  int configGetCalls = 0;
  int configSetCalls = 0;

  // ── the state the assertions read ──

  SaConfigFields? lastConfig;
  bool connected = false;
  String? subscribedTo;

  /// The held, unreleased payload count.
  int get outstandingPayloads => _outstanding;
  int _outstanding = 0;

  /// Captured at the destroy call: the payloads still held then.
  int outstandingAtDestroy = -1;
  int destroyCalls = 0;

  @override
  String? get libraryPath => null;

  @override
  bool get loaded => true;

  @override
  void load() {}

  @override
  int probeConfigSizeof() =>
      throw UnsupportedError('a fake has no honest answer to the C probes');

  @override
  int probeConfigOffset(int index) =>
      throw UnsupportedError('a fake has no honest answer to the C probes');

  @override
  SaConfigDefaults configDefaults() => const SaConfigDefaults(
        transport: 0,
        connectTimeoutMs: 5000,
        requestTimeoutMs: 10000,
        maxRetries: 0,
      );

  @override
  int buildConfig(SaConfigFields fields) {
    order.add('buildConfig');
    lastConfig = fields;
    _errorCb = fields.errorCb;
    return 0x2c0f1; // an opaque token
  }

  @override
  void freeConfig(int config) {
    order.add('freeConfig');
  }

  @override
  int connect(int config) {
    order.add('connect');
    if (!connectOk) return 0;
    connected = true;
    return 0x11; // the client token
  }

  @override
  void destroy(int client) {
    if (!connected) {
      return; // never reached from the runner (it refuses the null client)
    }
    outstandingAtDestroy = _outstanding;
    destroyCalls++;
    order.add('destroy');
    connected = false;
    // The C's destroy reclaims the unreleased payloads — mirror it: the
    // assertions read [outstandingAtDestroy] when it matters.
    _labels.clear();
    _outstanding = 0;
  }

  @override
  void releasePayload(int client, int payload) {
    order.add('release(${_labels[payload] ?? '?'})');
    final label = _labels.remove(payload);
    if (label != null) {
      _outstanding--;
      SaUtf8.freeAddress(payload); // the C's release frees
    }
  }

  /// A held payload: a malloc'd buffer the fake owns until released (the
  /// C's ownership rule, mirrored). The label names it in [order].
  int hold(String label, String text) {
    final pointer = SaUtf8.toNative(text);
    _labels[pointer.address] = label;
    _outstanding++;
    return pointer.address;
  }

  @override
  int prompt(int client, int serial, String? sid, String text, int promptCb) {
    order.add('prompt(sid=${sid ?? 'null'})');
    if (!promptFires) return -1; // the outright refusal: NO callback
    void fire() => _firePromptCb(promptCb, serial, promptStatus, promptSid);
    if (deferCallbacks) {
      Future<void>(fire);
    } else {
      fire();
    }
    return 0;
  }

  void _firePromptCb(int cbAddress, int serial, int status, String? sid) {
    final sidAddress = sid == null ? 0 : hold('prompt.sid', sid);
    (SaFfi.nativeFn<SaPromptCbNative>(cbAddress).asFunction<SaPromptCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(serial),
      status,
      ffi.Pointer<ffi.Uint8>.fromAddress(sidAddress),
    );
  }

  @override
  int interrupt(int client, int serial, String sid, int interruptCb) {
    order.add('interrupt(sid=$sid)');
    // The C's contract: the callback fires BEFORE the call returns.
    (SaFfi.nativeFn<SaInterruptCbNative>(interruptCb)
        .asFunction<SaInterruptCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(serial),
      0,
    );
    return 0;
  }

  @override
  int listSessions(int client, int serial, int sessionsCb) {
    order.add('listSessions');
    // The C fires the sessions callback BEFORE the call returns (its posts
    // queue during this call, before the done — the production order).
    if (listed == null) {
      // The failure delivery: the failing status + NULL rows (nothing to
      // release).
      (SaFfi.nativeFn<SaSessionsCbNative>(sessionsCb)
          .asFunction<SaSessionsCbDart>())(
        ffi.Pointer<ffi.Void>.fromAddress(serial),
        listedStatus,
        ffi.Pointer<SaSessionRowFfi>.fromAddress(0),
        0,
      );
      return 0;
    }
    final rows = listed!;
    final per = ffi.sizeOf<SaSessionRowFfi>();
    final array = SaUtf8.alloc(per * rows.length).cast<SaSessionRowFfi>();
    _labels[array.address] = 'rows(${rows.length})';
    _outstanding++;
    for (var i = 0; i < rows.length; i++) {
      final row = rows[i];
      array[i].sid = _holdRowString('rows[$i].sid', row.sid);
      array[i].status =
          row.status == null ? _null() : _holdRowString('rows[$i].status', row.status!);
      array[i].goal =
          row.goal == null ? _null() : _holdRowString('rows[$i].goal', row.goal!);
      array[i].created = row.created;
      array[i].depth = row.depth;
    }
    (SaFfi.nativeFn<SaSessionsCbNative>(sessionsCb)
        .asFunction<SaSessionsCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(serial),
      listedStatus,
      array,
      rows.length,
    );
    return 0;
  }

  ffi.Pointer<ffi.Uint8> _null() => ffi.Pointer<ffi.Uint8>.fromAddress(0);

  ffi.Pointer<ffi.Uint8> _holdRowString(String label, String text) {
    final pointer = SaUtf8.toNative(text);
    _labels[pointer.address] = label;
    _outstanding++;
    return pointer;
  }

  @override
  int subscribeEvents(int client, int serial, String sid, int eventsCb) {
    order.add('subscribe($sid)');
    _subCtx = serial;
    _subCb = eventsCb;
    subscribedTo = sid;
    if (subRefusalStatus >= 0) {
      // The C's refusal shape: the error callback fires INSIDE the call
      // (its post precedes the op's done), the call still returns 0.
      order.add('error(status=$subRefusalStatus)');
      final address = hold('error.text', subRefusalText);
      _fireError(address, subRefusalReqId, subRefusalStatus);
    }
    // The C's live transition: a replay-then-live subscription's marker is
    // DELIVERED when the sub goes live — but the refusal's sub was never
    // live (no marker). Held channel → the marker rides the queue (the
    // flush delivers it); otherwise it fires now (its post precedes the
    // subscribe's done — the production order).
    if (subRefusalStatus >= 0) return 0;
    if (holdEvents) {
      enqueue(sid, 0, saEventsOpReplayThenLive, null);
    } else {
      fire(sid, 0, saEventsOpReplayThenLive, null);
    }
    return 0;
  }

  void _fireError(int address, int reqId, int status) {
    (SaFfi.nativeFn<SaErrorCbNative>(_errorCb).asFunction<SaErrorCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(0),
      reqId,
      status,
      ffi.Pointer<ffi.Uint8>.fromAddress(address),
    );
  }

  @override
  int unsubscribeEvents(int client) {
    order.add('unsubscribe');
    // The reader's terminal-ack shape: the held deliveries flush first,
    // THEN the UNSUBSCRIBE marker (the sub's ctx rides it), then 0 — all
    // the posts queue during THIS C call, before its done.
    flushEvents();
    fire(subscribedTo ?? '', 0, saEventsOpUnsubscribe, null);
    return 0;
  }

  int _subCtx = 0;
  int _subCb = 0;
  int _errorCb = 0;

  @override
  int configGet(int client, int serial, int configCb) {
    order.add('configGet');
    configGetCalls++;
    if (!configFires) return -1; // the outright refusal: NO callback
    _fireConfigCb(configCb, serial);
    return 0;
  }

  @override
  int configSet(int client, int serial, String? baseUrl, String? apiKey,
      String? model, int configCb) {
    order.add('configSet');
    configSetCalls++;
    configSets.add((baseUrl, apiKey, model));
    if (!configFires) return -1; // the outright refusal: NO callback
    _fireConfigCb(configCb, serial);
    return 0;
  }

  void _fireConfigCb(int cbAddress, int serial) {
    final base = configBase == null ? 0 : hold('config.base', configBase!);
    final key = configKey == null ? 0 : hold('config.key', configKey!);
    final model = configModel == null ? 0 : hold('config.model', configModel!);
    (SaFfi.nativeFn<SaConfigCbNative>(cbAddress).asFunction<SaConfigCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(serial),
      configStatus,
      ffi.Pointer<ffi.Uint8>.fromAddress(base),
      ffi.Pointer<ffi.Uint8>.fromAddress(key),
      ffi.Pointer<ffi.Uint8>.fromAddress(model),
    );
  }

  /// Queues one event's delivery (the reader handing one over).
  void enqueue(String sid, int seq, int op, String? recordJson) =>
      queuedEvents.add((sid: sid, seq: seq, op: op, recordJson: recordJson));

  /// Fires the queued deliveries FIFO (the reader's flush — a replay's
  /// shape), each as a NativeCallable post.
  void flushEvents() {
    for (final event in queuedEvents) {
      fire(event.sid, event.seq, event.op, event.recordJson);
    }
    queuedEvents.clear();
  }

  void fire(String sid, int seq, int op, String? recordJson) {
    final sidAddress = hold('events(SEQ=$seq OP=$op).sid', sid);
    final recordAddress =
        recordJson == null ? 0 : hold('events(SEQ=$seq OP=$op).record', recordJson);
    (SaFfi.nativeFn<SaEventsCbNative>(_subCb).asFunction<SaEventsCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(_subCtx),
      ffi.Pointer<ffi.Uint8>.fromAddress(sidAddress),
      seq,
      op,
      ffi.Pointer<ffi.Uint8>.fromAddress(recordAddress),
    );
  }

  @override
  String toString() => 'FakeSaFfi(${order.join(', ')})';
}

// ── the harnesses ───────────────────────────────────────────────────────────

SaConfig theConfig({String socketPath = '/tmp/pondr-ffi-test.sock'}) => SaConfig(
      transport: SaTransport.unix,
      socketPath: socketPath,
      connectTimeoutMs: 250,
      requestTimeoutMs: 250,
    );

/// (wrapper, fake, events, errors)
typedef SaHarness = (
  SaClientNative,
  FakeSaFfi,
  List<SaEventRecord>,
  List<SaFailure>,
);

Future<SaHarness> connectHarness() async {
  final fake = FakeSaFfi();
  final events = <SaEventRecord>[];
  final errors = <SaFailure>[];
  final wrapper = SaClientNative(api: fake, host: SaInProcessOpHost());
  final result = await wrapper.connect(
    theConfig(),
    SaCallbacks(onEvent: events.add, onError: errors.add),
  );
  expect(result, isA<SaOk>(), reason: 'the fake connects by default');
  return (wrapper, fake, events, errors);
}

/// Lets every queued NativeCallable post land (a few task turns).
Future<void> pump() async {
  for (var i = 0; i < 4; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

// ── the tests ───────────────────────────────────────────────────────────────

void main() {
  group('the mirror shape (the NON-live pin)', () {
    test('the config mirror: 72 bytes, every C pinned offset lands', () {
      // The mirror's own byte shape, written through the fields and read
      // back raw — the C's committed offsets checked WITHOUT the .so (the
      // LIVE tripwire below asserts the C's own probes answer the same).
      expect(ffi.sizeOf<SaClientConfigFfi>(), kConfigSizeof);
      final base = SaUtf8.alloc(kConfigSizeof);
      final view = base.asTypedList(kConfigSizeof);
      view.fillRange(0, kConfigSizeof, 0); // the padding reads as zeros
      final cfg = base.cast<SaClientConfigFfi>();
      final socket = SaUtf8.toNative('/tmp/x');
      final host = SaUtf8.toNative('127.0.0.1');
      final key = SaUtf8.toNative('secret');
      try {
        cfg.ref
          ..transport = 1 // TCP (a non-zero enum value, the 4-byte int)
          ..socketPath = socket
          ..host = host
          ..port = 4711
          ..apiKey = key
          ..connectTimeoutMs = 0x11223344
          ..requestTimeoutMs = 0x55667788
          ..maxRetries = 0x99
          ..errorCb = ffi.Pointer<ffi.NativeFunction<SaErrorCbNative>>.fromAddress(0x1234)
          ..errorCtx = 0x7777;
        expect(view.sublist(0, 4), orderedEquals(<int>[1, 0, 0, 0]),
            reason: 'the transport (the 4-byte enum, little-endian)');
        _expectPadding(view, 4, 4);
        _expectAddress(view.sublist(8, 16), socket.address);
        _expectAddress(view.sublist(16, 24), host.address);
        expect(view.sublist(24, 26), orderedEquals(<int>[0x67, 0x12]),
            reason: 'the port (uint16 4711, little-endian)');
        _expectPadding(view, 26, 6);
        _expectAddress(view.sublist(32, 40), key.address);
        _expectU32(view, 40, 0x11223344);
        _expectU32(view, 44, 0x55667788);
        _expectU32(view, 48, 0x99);
        _expectPadding(view, 52, 4);
        _expectAddress(view.sublist(56, 64), 0x1234);
        _expectAddress(view.sublist(64, 72), 0x7777);
      } finally {
        SaUtf8.free(socket);
        SaUtf8.free(host);
        SaUtf8.free(key);
        SaUtf8.free(base);
      }
    });

    test('the rows mirror is 40 bytes (three pointers + created + depth)', () {
      expect(ffi.sizeOf<SaSessionRowFfi>(), 40);
    });

    test('SaUtf8 rejects a literal NUL (the wire never carries one)', () {
      expect(() => SaUtf8.toNative('a\x00b'), throwsArgumentError);
    });

    test('SaUtf8 round-trips a multi-byte string', () {
      final p = SaUtf8.toNative('héllø wörld');
      try {
        expect(SaUtf8.read(p), 'héllø wörld');
      } finally {
        SaUtf8.free(p);
      }
    });
  });

  group('the .so resolution', () {
    late Directory temp;

    setUp(() => temp = Directory.systemTemp.createTempSync('pondr-ffi-'));
    tearDown(() => temp.deleteSync(recursive: true));

    String existing([String name = 'a.so']) =>
        (File('${temp.path}/$name')..createSync()).path;
    String missing([String name = 'b.so']) => '${temp.path}/$name';

    test('the explicit sources: the define wins; a missing pair is loud',
        () {
      expect(
        resolveSaLibraryPath(define: existing('d.so'), env: existing('e.so')),
        existing('d.so'),
      );
      // Only the env exists → it wins (the first existing explicit source).
      expect(
        resolveSaLibraryPath(define: missing('m1.so'), env: existing('e.so')),
        existing('e.so'),
      );
      // All the explicit sources name missing files: no silent fallback.
      expect(
        () =>
            resolveSaLibraryPath(define: missing('m1.so'), env: missing('m2.so')),
        throwsStateError,
      );
      // A single explicit source missing is loud too.
      expect(
        () => resolveSaLibraryPath(define: null, env: missing('m2.so')),
        throwsStateError,
      );
    });

    test('with neither source set, the first existing candidate wins', () {
      final first = existing('c1.so');
      final second = existing('c2.so');
      expect(
        resolveSaLibraryPath(define: '', env: '', candidates: <String>[first, second]),
        first,
      );
      expect(
        resolveSaLibraryPath(
            define: '', env: '', candidates: <String>[second, missing()]),
        second,
      );
      expect(
        () => resolveSaLibraryPath(
            define: '', env: '', candidates: <String>[missing('x'), missing('y')]),
        throwsStateError,
      );
    });

    test('the exe-dir candidate is the bundle slot', () {
      expect(saLibraryExeCandidate(), endsWith('/lib/libsa_client.so'));
    });
  });

  group('SaClientNative (the in-process host + the fake ABI)', () {
    test('connect wires the config; the config is freed AFTER the connect',
        () async {
      final (wrapper, fake, _, _) = await connectHarness();
      expect(fake.connected, isTrue);
      expect(wrapper.isConnected, isTrue);
      final config = fake.lastConfig!;
      expect(config.transport, 0);
      expect(config.socketPath, '/tmp/pondr-ffi-test.sock');
      expect(config.connectTimeoutMs, 250);
      expect(config.requestTimeoutMs, 250);
      expect(config.errorCb, isNot(0), reason: 'the error listener rides');
      final order = fake.order;
      expect(order.indexOf('connect'), lessThan(order.indexOf('freeConfig')));
      await wrapper.dispose();
    });

    test("the connect's refusal shape (the C's null client)", () async {
      final fake = FakeSaFfi()..connectOk = false;
      final wrapper = SaClientNative(api: fake, host: SaInProcessOpHost());
      final result = await wrapper.connect(theConfig(), const SaCallbacks());
      final err = result as SaErr;
      expect(err.status, saClientStatusCallRefused);
      expect(err.text, contains('the null client'));
      expect(wrapper.isDisposed, isTrue);
      // An unusable wrapper refuses the ops loud.
      expect(() => wrapper.prompt(null, 'hi'), throwsStateError);
      // No client existed for the destroy.
      expect(fake.destroyCalls, 0);
      await wrapper.dispose(); // the double dispose is a no-op
      expect(fake.destroyCalls, 0);
    });

    test('ops before connect throw; dispose-before-connect settles', () async {
      final fake = FakeSaFfi();
      final wrapper = SaClientNative(api: fake, host: SaInProcessOpHost());
      expect(() => wrapper.prompt(null, 'hi'), throwsStateError);
      await wrapper.dispose();
      expect(wrapper.isDisposed, isTrue);
      expect(fake.destroyCalls, 0);
    });

    test('disposing mid-connect completes the connect loud + settles', () async {
      final fake = FakeSaFfi();
      final wrapper = SaClientNative(api: fake, host: SaInProcessOpHost());
      // The connect starts (the state rides connecting) but the dispose
      // lands first: the caller's Future must NOT hang.
      final connectFuture = wrapper.connect(theConfig(), const SaCallbacks());
      await wrapper.dispose();
      final result = await connectFuture;
      expect(result, isA<SaErr>());
      expect((result as SaErr).text, contains('disposed during connect'));
      expect(wrapper.isDisposed, isTrue);
      expect(() => wrapper.prompt(null, 'hi'), throwsStateError);
      expect(fake.destroyCalls, 0, reason: 'no client ever existed');
    });

    test('the default host refuses a non-dlopen api (the honest Err)',
        () async {
      final wrapper = SaClientNative(api: FakeSaFfi());
      final result = await wrapper.connect(theConfig(), const SaCallbacks());
      expect(result, isA<SaErr>());
      expect((result as SaErr).text, contains('dlopen-backed'));
      expect(wrapper.isDisposed, isTrue);
      expect(() => wrapper.prompt(null, 'hi'), throwsStateError);
    });

    test('the prompt: the callback lands; the sid copies then releases',
        () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.promptSid = 'sessions/frame-9';
      final result = await wrapper.prompt(null, 'the goal text');
      expect(result.started, isTrue);
      expect(result.sid, 'sessions/frame-9');
      final promptIndex = fake.order.indexOf('prompt(sid=null)');
      final releaseIndex = fake.order.indexOf('release(prompt.sid)');
      expect(promptIndex, greaterThanOrEqualTo(0));
      expect(releaseIndex, greaterThan(promptIndex),
          reason: 'the copy happened before the release was posted');
      await wrapper.dispose();
    });

    test('the dedup: the callback resolves once — the opposite order too',
        () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.promptSid = 'sessions/dedup';
      // deferCallbacks moves the fire's post past the op's own done — the
      // OPPOSITE of the C's contract (the C fires before the call
      // returns). The pending map's remove-on-complete makes both orders
      // one completion.
      fake.deferCallbacks = true;
      final result = await wrapper.prompt(null, 'text');
      expect(result.started, isTrue, reason: 'resolved via the deferred post');
      expect(result.sid, 'sessions/dedup');
      // The done that landed first did NOT resolve a duplicate.
      expect(fake.order.where((entry) => entry == 'prompt(sid=null)'), hasLength(1));
      await wrapper.dispose();
    });

    test('the outright-refused prompt (-1, no callback)', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.promptFires = false;
      final result = await wrapper.prompt(null, 'text');
      expect(result.failed, isTrue);
      expect(result.status, saClientStatusCallRefused);
      expect(fake.order, isNot(contains('release(prompt.sid)')));
      await wrapper.dispose();
    });

    test('the failing-delivery prompt (a non-zero status, NULL sid)', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.promptStatus = saClientStatusBusy;
      fake.promptSid = null;
      final result = await wrapper.prompt('sessions/1', 'a steer');
      expect(result.failed, isTrue);
      expect(result.status, saClientStatusBusy);
      await wrapper.dispose();
    });

    test('the steer sentinel: status 0 + the empty sid', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.promptStatus = 0;
      fake.promptSid = '';
      final result = await wrapper.prompt('sessions/1', 'a steer');
      expect(result.queued, isTrue);
      await wrapper.dispose();
    });

    test('the sessions list: copy→release→resolve; NULL fields not released',
        () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.listed = const <FakeRow>[
        (sid: 'sessions/a', status: 'running', goal: 'the goal', created: 12, depth: 2),
        (sid: 'sessions/b', status: null, goal: null, created: 9, depth: 0),
      ];
      final result = await wrapper.listSessions();
      expect(result.ok, isTrue);
      expect(result.rows, hasLength(2));
      expect(result.rows[0].sid, 'sessions/a');
      expect(result.rows[0].status, 'running');
      expect(result.rows[0].goal, 'the goal');
      expect(result.rows[0].created, 12);
      expect(result.rows[0].depth, 2);
      expect(result.rows[1].status, isNull);
      expect(result.rows[1].goal, isNull);
      // The rows ARRAY is one payload; each row's NON-NULL string is one
      // separately; a NULL field needs no release.
      expect(fake.order, containsAllInOrder(<String>[
        'release(rows(2))',
        'release(rows[0].sid)',
        'release(rows[0].status)',
        'release(rows[0].goal)',
        'release(rows[1].sid)',
      ]));
      // The listSessions' full release set, nothing more.
      expect(
        fake.order.where((entry) => entry.startsWith('release(')),
        unorderedMatches(<String>[
          'release(rows(2))',
          'release(rows[0].sid)',
          'release(rows[0].status)',
          'release(rows[0].goal)',
          'release(rows[1].sid)',
        ]),
        reason: 'the NULL status/goal need no release',
      );
      expect(fake.outstandingPayloads, 0);
      await wrapper.dispose();
    });

    test("the sessions list's failure delivery (NULL rows)", () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.listed = null;
      fake.listedStatus = saClientStatusTimeout;
      final result = await wrapper.listSessions();
      expect(result.ok, isFalse);
      expect(result.status, saClientStatusTimeout);
      expect(result.rows, isEmpty);
      expect(fake.order, isNot(contains('release(rows(?)')));
      await wrapper.dispose();
    });

    test('the events subscription: the records route in order; '
        'two releases per record, ONE for the marker', () async {
      final (wrapper, fake, events, _) = await connectHarness();
      final result = await wrapper.subscribeEvents('sessions/sub-1');
      expect(result, isA<SaOk>());
      expect(wrapper.isSubscribed, isTrue);
      expect(fake.subscribedTo, 'sessions/sub-1');

      // The subscribe's own live marker arrived with the subscribing C
      // call; the reader's deliveries: two records after it.
      fake.enqueue('sessions/sub-1', 1, 0, '{"kind":"msg.append"}');
      fake.enqueue('sessions/sub-1', 2, 0, '{"kind":"msg.append"}');
      fake.flushEvents();
      await pump();
      expect(events, hasLength(3));
      expect(events[0].isLiveMarker, isTrue);
      expect(events[0].recordJson, isNull);
      expect(events[0].sid, 'sessions/sub-1');
      expect(events[1].seq, 1);
      expect(events[1].recordJson, '{"kind":"msg.append"}');
      expect(events[1].isMarker, isFalse);
      expect(events[2].seq, 2);
      // Two releases per record (the sid + the record), ONE for the marker.
      expect(
        fake.order.where((entry) => entry.startsWith('release(')),
        unorderedMatches(<String>[
          'release(events(SEQ=1 OP=0).sid)',
          'release(events(SEQ=1 OP=0).record)',
          'release(events(SEQ=2 OP=0).sid)',
          'release(events(SEQ=2 OP=0).record)',
          'release(events(SEQ=0 OP=0).sid)',
        ]),
        reason: 'the marker delivers no record payload (one release only)',
      );
      expect(fake.outstandingPayloads, 0);
      await wrapper.dispose();
    });

    test('the terminal marker acks the unsubscribe (the awaited marker)',
        () async {
      final (wrapper, fake, events, _) = await connectHarness();
      await wrapper.subscribeEvents('sessions/sub-2');
      final result = await wrapper.unsubscribeEvents();
      expect(result, isA<SaOk>());
      expect(wrapper.isSubscribed, isFalse);
      // The terminal ack rode the events stream (the app's source of
      // truth) with the UNSUBSCRIBE op.
      final markers =
          events.where((event) => event.isUnsubscribeAck).toList();
      expect(markers, hasLength(1));
      expect(markers.single.op, saEventsOpUnsubscribe);
      // One release (the marker's sid; its record is NULL).
      expect(
        fake.order
.where((entry) => entry.startsWith('release(events(SEQ=0 OP=2'))
            .toList(),
        ['release(events(SEQ=0 OP=2).sid)'],
      );
      await wrapper.dispose();
    });

    test('the in-window error shapes the subscribe Err (the daemon refusal)',
        () async {
      final (wrapper, fake, _, errors) = await connectHarness();
      // The C's subscribe refusal path: ret 0 BUT the error callback fired
      // inside the call (before the done) — the window check reports it
      // instead of a dead-sub's SaOk.
      fake.subRefusalStatus = saClientStatusBusy;
      fake.subRefusalText = 'refused';
      final result = await wrapper.subscribeEvents('sessions/sub-4');
      expect(result, isA<SaErr>(), reason: 'the refusal rode the error channel');
      expect((result as SaErr).status, saClientStatusBusy);
      expect(wrapper.isSubscribed, isFalse);
      expect(errors.single.text, 'refused');
      // And the error text's payload was released (the chain rules).
      expect(fake.order, contains('release(error.text)'));
      await wrapper.dispose();
    });

    test('THE DISPOSE CHAIN: the in-flight deliveries drain before the destroy',
        () async {
      final (wrapper, fake, events, _) = await connectHarness();
      await wrapper.subscribeEvents('sessions/sub-3');
      // Two deliveries still queued (the reader handed them over; their
      // listener posts are in flight when dispose starts).
      fake.enqueue('sessions/sub-3', 5, 0, '{"kind":"one"}');
      fake.enqueue('sessions/sub-3', 6, 0, '{"kind":"two"}');

      expect(fake.order.lastIndexOf('destroy'), -1, reason: 'no destroy yet');
      await wrapper.dispose();

      // (1) The chain's tail was awaited: every in-flight delivery's
      // handler (copy → release post → user callback) RAN — the app saw
      // the live marker (the subscribe's), both records, AND the
      // unsubscribe's terminal marker, in order, before the teardown.
      expect(events.map((event) => event.recordJson).toList(), <String?>[
        null,
        '{"kind":"one"}',
        '{"kind":"two"}',
        null,
      ]);
      // (2) THE ORDER EVIDENCE: every payload's release precedes the
      // destroy command (the in-flight ones included); the destroy found
      // nothing unreleased (no survivor rode the C's own reclaim).
      final destroyIndex = fake.order.indexOf('destroy');
      expect(destroyIndex, greaterThan(-1));
      for (final entry in fake.order) {
        if (!entry.startsWith('release(')) continue;
        expect(fake.order.indexOf(entry), lessThan(destroyIndex),
            reason: '$entry must precede the destroy');
      }
      // (3) Exactly ONE destroy; double-dispose stays settled.
      expect(fake.order.lastIndexOf('destroy'), destroyIndex);
      expect(fake.destroyCalls, 1);
      // (4) Nothing unreleased survived the chain's drain.
      expect(fake.outstandingAtDestroy, 0);
      expect(fake.outstandingPayloads, 0);
    });

    test('dispose is idempotent (one destroy; the same settled flow)', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      await wrapper.dispose();
      expect(fake.destroyCalls, 1);
      final after = fake.order.length;
      await wrapper.dispose();
      await wrapper.dispose();
      expect(fake.destroyCalls, 1, reason: 'the double dispose is a no-op');
      expect(fake.order.length, after);
      expect(wrapper.isDisposed, isTrue);
    });

    test("the config get: the template's truth; the present members are "
        'held then released; the absent members ride null', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.configBase = 'http://127.0.0.1:11434';
      fake.configKey = 'sk-the-key';
      fake.configModel = 'deepseek-chat';
      final result = await wrapper.configGet();
      expect(result.ok, isTrue);
      expect(result.baseUrl, 'http://127.0.0.1:11434');
      expect(result.apiKey, 'sk-the-key');
      expect(result.model, 'deepseek-chat');
      expect(
        fake.order.where((entry) => entry.startsWith('release(')),
        unorderedMatches(<String>[
          'release(config.base)',
          'release(config.key)',
          'release(config.model)',
        ]),
        reason: 'the three present members released (the held-payload rules)',
      );
      expect(fake.outstandingPayloads, 0);

      // The absent members ride null (the "" sentinel decoded absent) and
      // need no release.
      fake.configBase = null;
      fake.configKey = null;
      final absent = await wrapper.configGet();
      expect(absent.baseUrl, isNull);
      expect(absent.apiKey, isNull);
      expect(absent.model, 'deepseek-chat');
      expect(fake.outstandingPayloads, 0);
      await wrapper.dispose();
    });

    test('the config set rides its members; the answer is the template\'s '
        'post-set truth', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      final result = await wrapper.configSet(
        baseUrl: 'http://127.0.0.1:11434',
        apiKey: 'sk-the-key',
        model: 'deepseek-chat',
      );
      expect(result.ok, isTrue);
      expect(fake.configSets.single, (
        'http://127.0.0.1:11434',
        'sk-the-key',
        'deepseek-chat',
      ));
      // A null member is the ABSENT member (never sent).
      fake.configSets.clear();
      await wrapper.configSet(model: 'gemma4:latest');
      expect(fake.configSets.single, (null, null, 'gemma4:latest'));
      expect(fake.outstandingPayloads, 0);
      await wrapper.dispose();
    });

    test('the outright-refused config op (-1, no callback)', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      fake.configFires = false;
      final getResult = await wrapper.configGet();
      expect(getResult.ok, isFalse);
      expect(getResult.status, saClientStatusCallRefused);
      expect(fake.order, isNot(contains('release(config.model)')));
      final setResult = await wrapper.configSet(model: 'x');
      expect(setResult.status, saClientStatusCallRefused);
      // The rides never lost the SET's record (the runner ran; no callback).
      expect(fake.configSetCalls, 1);
      await wrapper.dispose();
    });

    test('the in-process op host: started → stopped; post-stop drops', () async {
      final fake = FakeSaFfi();
      final host = SaInProcessOpHost();
      expect(host.isRunning, isFalse);
      await host.start(fake);
      expect(host.isRunning, isTrue);
      final stopped = host.stop();
      // A post-stop command is dropped, not thrown (the C's
      // release-after-destroy no-op mirrored).
      expect(() => host.send(const SaReleaseCmd(1)), returnsNormally);
      await stopped;
      expect(host.isRunning, isFalse);
    });

    test('the teardown: the destroy ack precedes the close; nothing outstanding',
        () async {
      final (wrapper, fake, _, _) = await connectHarness();
      await wrapper.dispose();
      expect(fake.destroyCalls, 1, reason: 'the destroy acked exactly once');
      expect(fake.connected, isFalse);
      expect(fake.outstandingPayloads, 0);
      final destroyIndex = fake.order.indexOf('destroy');
      expect(fake.order.length, greaterThan(destroyIndex),
          reason: 'the host stop + the listener closes follow the ack');
    });
  });

  // ── the LIVE ABI (skip unless the .so exists on this run) ────────────────

  final livePath = _liveLibraryPath();
  final liveSkip = livePath == null
      ? 'no libsa_client.so on this run — set SA_LIBRARY_PATH (e.g. the '
          'SecretAgent build\'s cmake-build-debug/libsa_client.so)'
      : false;

  group('the LIVE struct-drift tripwire', () {
    test('the C probes answer the pinned constants (the ABI never drifts)',
        skip: liveSkip, () {
      final api = SaFfi(livePath)..load();
      expect(api.probeConfigSizeof(), kConfigSizeof);
      expect(ffi.sizeOf<SaClientConfigFfi>(), api.probeConfigSizeof(),
          reason: 'the Dart mirror and the C agree on the size');
      expect(api.probeConfigOffset(-1), 0, reason: 'out of range answers 0');
      expect(api.probeConfigOffset(10), 0, reason: 'out of range answers 0');
      for (var i = 0; i < kConfigOffsets.length; i++) {
        expect(api.probeConfigOffset(i), kConfigOffsets[i],
            reason: 'the config member $i drifted (C: '
                '${api.probeConfigOffset(i)}, pinned: ${kConfigOffsets[i]})');
      }
    });

    test('the config defaults round-trip the C', skip: liveSkip, () {
      final api = SaFfi(livePath)..load();
      final defaults = api.configDefaults();
      expect(defaults.transport, 0, reason: 'SaTransport.unix');
      expect(defaults.connectTimeoutMs, 5000);
      expect(defaults.requestTimeoutMs, 10000);
      expect(defaults.maxRetries, 0, reason: 'the events channel retries forever');
    });

    test('the real op isolate spawns + tears down a failed connect',
        skip: liveSkip, () async {
      final wrapper = SaClientNative(api: SaFfi(livePath)); // the REAL isolate host
      final result = await wrapper.connect(
        theConfig(socketPath: '/tmp/pondr-ffi-no-such-socket.sock'),
        const SaCallbacks(),
      );
      // The unix connect against a dead socket path fails fast — the
      // honest Err through the FULL path: the main isolate's load, the op
      // isolate's spawn + handshake, the config build, the C's refusal,
      // the teardown's ack + the listener closes.
      expect(result, isA<SaErr>());
      expect(wrapper.isDisposed, isTrue);
      expect(() => wrapper.prompt(null, 'hi'), throwsStateError);
      await wrapper.dispose();
    });
  });
}

/// The live path: the resolution's first existing answer; null → skip.
String? _liveLibraryPath() {
  try {
    return resolveSaLibraryPath();
  } on StateError {
    return null;
  }
}

/// The padding's [offset, offset+length) bytes all zero.
void _expectPadding(Uint8List view, int offset, int length) {
  for (var i = offset; i < offset + length; i++) {
    expect(view[i], 0, reason: 'the padding byte at $offset..${offset + length}');
  }
}

/// The 8-byte LE word reads back the pinned value.
void _expectAddress(Uint8List bytes, int address) {
  for (var i = 0; i < 8; i++) {
    expect(bytes[i], (address >> (i * 8)) & 0xFF,
        reason: 'the address byte $i reads back at its offset');
  }
}

void _expectU32(Uint8List view, int offset, int value) {
  for (var i = 0; i < 4; i++) {
    expect(view[offset + i], (value >> (i * 8)) & 0xFF,
        reason: 'the uint32 at $offset reads back');
  }
}