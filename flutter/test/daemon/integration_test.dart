/// The daemon integration tests (the FFI-binding plan's Task 6 step 2):
/// the REAL `frame-demo serve` daemon + the REAL `libsa_client.so` under
/// the REAL service wiring — the same construction the app's bindings run.
///
/// TWO SHAPES:
/// - THE NO-MODEL SHAPE (non-live; runnable whenever the two binaries
///   exist): serve with an EMPTY model tag → a prompt starts its frame →
///   the frame's first turn hits `_frame_backend_get`'s refusal (an empty
///   model name is no creatable http backend — model.c) → the frame fails
///   loud with the `control` record `{kind: model-missing}` on the events
///   channel. What the APP'S SURFACE does with that (pin what EXISTS —
///   `FfiChatService`'s fold ignores non-`msg.append` records): the send
///   yields [ChatTyping] and completes with neither error nor reply — the
///   typing flag stays up for a frame that went down mid-turn. The control
///   record's UI mapping is the audit view's recorded follow-on; this test
///   pins the honest today shape so a later mapping change lands on
///   purpose.
/// - THE LIVE SHAPE (opt-in): the REAL Ollama end-to-end — the settings'
///   config adoption SETs `gemma4:latest` on the daemon, the send gets a
///   REAL assistant reply, and the session's records carry both bubbles.
///
/// THE LIVE GATE (the FFI tests' trio idiom, plus the run flag):
/// (1) `libsa_client.so` resolves (`SA_LIBRARY_PATH` env, the prepare
/// script's copy `./libsa_client.so`, the sibling SecretAgent build);
/// (2) the `frame-demo` daemon binary exists (`PONDR_DAEMON_BIN`, the
/// prepare script's copy `./daemon/frame-demo`, the SecretAgent build);
/// (3) the opt-in run flag: `--dart-define=pondr.live=true` or
/// `PONDR_LIVE=1`;
/// (4) the endpoint check: the live group skips when Ollama answers not or
/// `gemma4:latest` (or the `PONDR_LIVE_MODEL` override) is not pulled (the
/// flag is the INTENT, the endpoint is the truth).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pondr/data/bindings.dart' show defaultDaemonBinary;
import 'package:pondr/data/daemon/ffi_chat_service.dart';
import 'package:pondr/data/daemon/ffi_settings_service.dart';
import 'package:pondr/data/daemon/ffi_sessions_service.dart';
import 'package:pondr/data/models.dart' hide Provider;
import 'package:pondr/data/models.dart' as mockup show Provider;
import 'package:pondr/data/services.dart';
import 'package:pondr/daemon/app_config_store.dart';
import 'package:pondr/daemon/supervisor.dart';
import 'package:pondr/ffi/sa_client_binding.dart';

// ── the gates ───────────────────────────────────────────────────────────────

String? _resolveSo() {
  final envSet = Platform.environment['SA_LIBRARY_PATH'];
  final candidates = <String>[
    if (envSet != null && envSet.isNotEmpty) envSet,
    './libsa_client.so',
    '../../SecretAgent/cmake-build-debug/libsa_client.so',
  ];
  for (final candidate in candidates) {
    if (File(candidate).existsSync()) return File(candidate).absolute.path;
  }
  return null;
}

String? _resolveDaemonBinary() {
  final candidates = <String>[];
  final envSet = Platform.environment['PONDR_DAEMON_BIN'];
  if (envSet != null && envSet.isNotEmpty) candidates.add(envSet);
  candidates.add('./daemon/frame-demo');
  candidates.add('../../SecretAgent/cmake-build-debug/frame-demo');
  try {
    // The bindings' wiring is ALSO a candidate (the same resolution the app
    // runs; PATH-walkable bare names included).
    final wired = defaultDaemonBinary();
    if (FileSystemEntity.typeSync(wired) != FileSystemEntityType.notFound) {
      candidates.add(wired);
    }
  } on StateError {
    // (defaultDaemonBinary answers the bare name verbatim; harmless)
  }
  for (final candidate in candidates) {
    if (File(candidate).existsSync()) return File(candidate).absolute.path;
  }
  return null;
}

/// The run flag: `--dart-define=pondr.live=true`, or the env form.
bool get _liveWanted =>
    const bool.fromEnvironment('pondr.live') ||
    Platform.environment['PONDR_LIVE'] == '1';

/// The live model: the plan's pin (gemma4:latest), overridable per-machine
/// (a box whose memory cannot load a local gemma runs the gate against
/// another Ollama-served tag — the cloud models ride the same /api/chat).
String get _theLiveTag =>
    Platform.environment['PONDR_LIVE_MODEL'] ?? 'gemma4:latest';

/// Ollama's reach: the /api/tags listing carries the live tag; empty when
/// it does, the skip reason when it does not.
Future<String> _ollamaReach(String model) async {
  try {
    final request = await HttpClient()
        .getUrl(Uri.parse('http://127.0.0.1:11434/api/tags'))
        .timeout(const Duration(seconds: 3));
    final response = await request.close().timeout(
          const Duration(seconds: 3),
        );
    final body = await response
        .transform(utf8.decoder)
        .join()
        .timeout(const Duration(seconds: 2));
    final listing = jsonDecode(body) as Map<String, dynamic>;
    final models = listing['models'];
    if (models is! List) return 'the /api/tags listing read malformed';
    for (final m in models) {
      if (m is Map && m['name'] == model) return '';
    }
    return '$model is not pulled';
  } on Exception catch (e) {
    return 'Ollama answers not (http://127.0.0.1:11434): $e';
  }
}

// ── the harness ─────────────────────────────────────────────────────────────

/// One daemon + one connected client + the REAL service trio, exactly the
/// app's wiring (the bindings' construction with the settings store
/// pointed at a temp file).
final class DaemonHarness {
  DaemonHarness._(
    this.supervisor,
    this.client,
    this.sessions,
    this.chat,
    this.records,
    this.homePath,
  );

  /// Spawns `frame-demo serve` on a fresh temp store + socket and waits for
  /// the connect (the supervisor's whole flow).
  static Future<DaemonHarness> start({
    required String soPath,
    required String binaryPath,
    required String modelTag,
  }) async {
    final home = Directory.systemTemp.createTempSync('pondr-integration');
    final socketPath =
        '${home.path}${Platform.pathSeparator}pondr-${modelTag.isEmpty ? "nomodel" : modelTag}.sock';
    final records = <String>[];
    final supervisor = DaemonSupervisor(
      socketPath: socketPath,
      daemonBinary: binaryPath,
      // The store is a FRESH temp location per shape (the daemon's REAL
      // disk store; create-if-absent, like the app's run).
      argsTemplate: <String>[
        'serve',
        '--socket-path',
        socketPath,
        '--model',
        modelTag,
        '--location',
        '${home.path}${Platform.pathSeparator}sa-db',
      ],
      waitForSocket: const Duration(seconds: 20),
      connectTimeout: const Duration(seconds: 10),
      saLibraryPath: soPath,
    );
    final client = await supervisor.ensure();
    supervisor.events.listen((SaEventRecord r) {
      final json = r.recordJson;
      if (json != null) records.add('${r.sid}:$json');
    });
    final sessions = FfiSessionsService(
      clientFuture: Future<SaClientNative>.value(client),
    );
    final chat = FfiChatService(
      client: Future<SaClientNative>.value(client),
      sessions: sessions,
      events: supervisor.events,
      failures: supervisor.failures,
    );
    return DaemonHarness._(
      supervisor,
      client,
      sessions,
      chat,
      records,
      home.path,
    );
  }

  final DaemonSupervisor supervisor;
  final SaClientNative client;
  final FfiSessionsService sessions;
  final FfiChatService chat;

  /// The raw events records as the wire carried them (the daemon's frame
  /// log — the tests' evidence surface).
  List<String> records;

  /// The temp home (the store's parent; the settings file's parent too).
  final String homePath;

  /// The teardown: the chat's dispose, then the supervisor's stop (the
  /// client's dispose + the daemon's SIGTERM).
  Future<void> stop() async {
    chat.dispose();
    await supervisor.stop();
  }
}

/// One bounded wait on the harness's records: the first JSON body carrying
/// [needle]; the timeout fail names everything seen.
Future<String> _awaitRecord(
  DaemonHarness harness,
  String needle, {
  Duration limit = const Duration(seconds: 15),
}) async {
  final deadline = DateTime.now().add(limit);
  while (DateTime.now().isBefore(deadline)) {
    for (final line in harness.records) {
      if (line.contains(needle)) return line;
    }
    await Future<void>.delayed(const Duration(milliseconds: 25));
  }
  fail('the record never rode ("$needle") — the seen records:\n'
      '${harness.records.map((l) => '  $l').join('\n')}');
}

/// Lets the folds' microtask writes land (the broadcast deliveries ride
/// several turns).
Future<void> _settle([int turns = 16]) async {
  for (var i = 0; i < turns; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

void main() async {
  final soPath = _resolveSo();
  final binaryPath = _resolveDaemonBinary();
  final binariesSkip = (soPath == null || binaryPath == null)
      ? 'the binaries ride not — run flutter/daemon/prepare.sh (its copies '
          'are discoverable from the flutter project dir), or set '
          'SA_LIBRARY_PATH + PONDR_DAEMON_BIN'
      : false;

  // The endpoint truth, decided before the groups register (an async main
  // is legal — the groups register after the check).
  String liveEndpointReason = '';
  if (binariesSkip == false && _liveWanted) {
    liveEndpointReason = await _ollamaReach(_theLiveTag);
  }

  group(
    'the no-model shape (the binaries exist; the daemon template tag empty)',
    skip: binariesSkip,
    () {
    late DaemonHarness harness;

    setUpAll(() async {
      harness = await DaemonHarness.start(
        soPath: soPath!,
        binaryPath: binaryPath!,
        modelTag: '',
      );
      await harness.sessions.refresh().timeout(const Duration(seconds: 15));
    });

    tearDownAll(() async {
      await harness.stop();
    });

    test('the prompt starts its frame; the frame fails the control '
        'model-missing; the user bubble folds; the send keeps typing'
        ' (what exists)', timeout: const Timeout(Duration(minutes: 2)),
        () async {
      final draft = harness.sessions.create();
      expect(draft.id, '',
          reason: 'the draft sentinel: no daemon sid before the first send');
      expect(harness.sessions.active()?.id, '');

      // THE SEND: typing rides; the send's stream completes ONLY on an
      // assistant fold (the no-double-append pin's shape — the user bubble
      // is the record's arrival, never a local append).
      final List<ChatEvent> seen = <ChatEvent>[];
      Object? sendFailure;
      harness.chat
          .send('what is the mean free path', const <AttachedFile>[])
          .listen(
        seen.add,
        onError: (Object e, StackTrace _) => sendFailure = e,
      );

      // The user's msg.append record rides the send's commit.
      final userRecord = await _awaitRecord(
        harness,
        'mean free path',
        limit: const Duration(seconds: 30),
      );
      expect(
        userRecord.contains('"role":"user"'),
        isTrue,
        reason: 'the user record is what carries the bubble',
      );

      // The frame's failure: the daemon's control record, kind model-missing.
      final control = await _awaitRecord(
        harness,
        'model-missing',
        limit: const Duration(seconds: 60),
      );
      expect(
        control.contains('"type":"control"'),
        isTrue,
        reason: 'the failure rides as {type: control, payload: '
            '{kind: model-missing}} — the empty-tag template refuses its '
            'backend at the first turn',
      );

      // THE TODAY-SHAPE PIN: the fold ignores the control record — the send
      // did NOT error and did NOT complete; the session's records show the
      // user bubble alone.
      await _settle();
      expect(
        sendFailure,
        isNull,
        reason: 'the frame started; nothing refused the prompt itself',
      );
      expect(seen.whereType<ChatTyping>().length, 1,
          reason: 'the typing indicator rode the send');
      expect(seen.whereType<ChatDone>(), isEmpty,
          reason: 'no assistant record arrived — the send stays open');
      expect(harness.sessions.byId(''), isNull,
          reason: 'the draft adopted its daemon sid in place');
      final adopted = harness.sessions.active();
      expect(adopted, isNotNull);
      expect(adopted!.id, isNotEmpty);
      expect(adopted.messages, hasLength(1));
      expect(adopted.messages.single.role, MessageRole.user);
      expect(adopted.messages.single.content, 'what is the mean free path');
      // The stream is NOT cancelled here: it is still parked on the send's
      // reply wait — the teardown's dispose (the harness's stop below)
      // completes its waiter with the error, and THIS listener's onError is
      // the swallow that keeps the error out of the zone.
    });
  });

  group(
    'the live shape (the REAL Ollama end-to-end)',
    skip: binariesSkip,
    () {
    final liveSkip = _liveWanted
        ? (liveEndpointReason.isEmpty ? false : liveEndpointReason)
        : 'not wanted — pass --dart-define=pondr.live=true (or PONDR_LIVE=1)';

    late DaemonHarness harness;

    setUpAll(() async {
      harness = await DaemonHarness.start(
        soPath: soPath!,
        binaryPath: binaryPath!,
        modelTag: '',
      );
      await harness.sessions.refresh().timeout(const Duration(seconds: 15));
    });

    tearDownAll(() async {
      await harness.stop();
    });

    test('the config adoption lands the Ollama template; the send answers '
        'a REAL reply; the session carries both bubbles', skip: liveSkip,
        timeout: const Timeout(Duration(minutes: 5)), () async {
      // The settings' adoption: the temp JSON store carries one REAL
      // provider (Ollama local) + the live model row; the setter fires the
      // CA_CONFIG set through the SAME service the picker drives.
      final store = AppConfigStore(
        filePath: '${harness.homePath}${Platform.pathSeparator}'
            'app-config.json',
      );
      final provider = mockup.Provider(
        id: 'p-ollama',
        name: 'Ollama local',
        baseUrl: 'http://127.0.0.1:11434',
        apiKey: '',
        enabled: true,
        models: <ProviderModel>[
          ProviderModel(
            id: 'm-gemma',
            modelId: _theLiveTag,
            label: 'Gemma 4',
            enabled: true,
          ),
        ],
      );
      store.save(AppConfig(
        providers: <mockup.Provider>[provider],
        selectedModelKey: 'p-ollama::m-gemma',
      ));
      final settings = FfiSettingsService(
        store: store,
        client: () => Future<SaClientNative>.value(harness.client),
      );
      // The startup truth + the adoption (the absent tag rides the wire's
      // "" sentinel — decoded NULL: SaConfigResult.model null = absent).
      final truth = await settings.readDaemonTruth();
      expect(truth, isNotNull);
      expect(
        truth!.model ?? '',
        isEmpty,
        reason: 'the daemon was spawned with the empty tag — '
            'the adoption is what lands the live truth',
      );
      settings.selectedModelKey = 'p-ollama::m-gemma';
      final adopted = await settings.lastAdoption;
      expect(adopted, isNotNull, reason: 'the adoption answered');
      expect(adopted!.ok, isTrue, reason: 'the daemon accepted the template');
      expect(adopted.baseUrl, 'http://127.0.0.1:11434');
      expect(adopted.model, _theLiveTag);

      // THE SEND: a REAL reply, bounded by Ollama's own latency.
      final draft = harness.sessions.create();
      final List<ChatEvent> seen = <ChatEvent>[];
      Object? sendFailure;
      final StreamSubscription<ChatEvent> sub = harness.chat
          .send('Reply with exactly one short sentence about the sea.',
              const <AttachedFile>[])
          .listen(
        seen.add,
        onError: (Object e, StackTrace _) => sendFailure = e,
      );

      final userRecord = await _awaitRecord(
        harness,
        'one short sentence about the sea',
        limit: const Duration(seconds: 30),
      );
      expect(userRecord.contains('"role":"user"'), isTrue);

      // The REAL assistant record (the model's own words on the wire).
      final assistantRecord = await _awaitRecord(
        harness,
        '"role":"assistant"',
        limit: const Duration(seconds: 120),
      );
      expect(
        assistantRecord.contains(_theLiveTag) == false,
        isTrue,
        reason: 'the reply is the model\'s content, not the template echo',
      );
      await _settle();

      expect(sendFailure, isNull, reason: 'the send resolved clean');
      final done = seen.whereType<ChatDone>().toList();
      expect(done, hasLength(1));
      expect(done.single.content, isNotEmpty);
      final session = harness.sessions.byId(draft.id) ??
          harness.sessions.active()!;
      expect(session.messages, hasLength(2));
      expect(session.messages.first.role, MessageRole.user);
      expect(session.messages.last.role, MessageRole.assistant);
      expect(session.messages.last.content, done.single.content);
      await sub.cancel();
    });
  });
}
