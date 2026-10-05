/// The daemon supervisor's tests (the plan's Task 4 step 2): a REAL child
/// process against a fake long-running binary — the spawn + the socket wait
/// + the connect, the spawn-loop's budget, the respawn through
/// [DaemonSupervisor.onDaemonGone] (the bounded re-ensure), the SIGTERM's
/// clean stop, and the no-child shapes.
///
/// THE LIVE GATE: the ensure-flow's connect is the REAL `libsa_client.so`'s
/// (the honest wait — the C's unix connect needs only a LISTENING socket, so
/// the fake daemon passes it end-to-end). The .so resolves in order: the
/// `SA_LIBRARY_PATH` env, `./libsa_client.so` (the prepare script's copy),
/// the sibling SecretAgent build (`../../SecretAgent/cmake-build-debug/`);
/// when none exists the LIVE group SKIPs (the same idiom
/// `test/ffi/ffi_test.dart` pins — set `SA_LIBRARY_PATH` to the SecretAgent
/// build's to run it).
///
/// The child needs `dart` on PATH (the fake daemon is a dart script); a
/// dart-less machine skips the same group.
library;

import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pondr/daemon/supervisor.dart';

// ── the resolution plumbing ──────────────────────────────────────────────────

String? _resolveDart() {
  final paths = Platform.environment['PATH'] ?? '';
  for (final dir in paths.split(':')) {
    if (dir.isEmpty) continue;
    final candidate = '$dir${Platform.pathSeparator}dart';
    if (FileSystemEntity.typeSync(candidate) !=
        FileSystemEntityType.notFound) {
      return candidate;
    }
  }
  return null;
}

/// libsa_client.so's first-existing candidate, or null (the live group's
/// skip gate).
String? _resolveSo() {
  final envSet = Platform.environment['SA_LIBRARY_PATH'];
  final candidates = <String>[
    if (envSet != null && envSet.isNotEmpty) envSet,
    './libsa_client.so',
    '../../SecretAgent/cmake-build-debug/libsa_client.so',
  ];
  for (final candidate in candidates) {
    if (File(candidate).existsSync()) {
      return File(candidate).absolute.path;
    }
  }
  return null;
}

/// One test's plumbing: a fresh temp dir (the socket file's home), the fake
/// daemon's invocation shape, the supervisor with the test's budgets and the
/// captured log.
final class _Harness {
  _Harness({required this.dartPath, required this.soPath});

  final String dartPath;
  final String soPath;

  static const String _prefix = 'pondr-supervisor-test';

  /// A FRESH temp dir per test (flutter_test's tearDown is PER TEST: a
  /// shared temp would be deleted beneath the later tests' children).
  ({Directory dir, String socketPath}) fresh() {
    final created = Directory.systemTemp.createTempSync(_prefix);
    addTearDown(() {
      try {
        created.deleteSync(recursive: true);
      } catch (e) {
        stderr.writeln('the temp dir survived: $e');
      }
    });
    return (
      dir: created,
      socketPath: '${created.path}${Platform.pathSeparator}pondr.sock',
    );
  }

  /// The fake daemon's spawn template, the test's socket path (and the
  /// optional mode flag) baked in.
  List<String> serveArgs(String socketPath,
          {String? modeFlag, String? modeValue}) =>
      <String>[
        '${Directory.current.path}${Platform.pathSeparator}test'
        '${Platform.pathSeparator}daemon${Platform.pathSeparator}'
        'fake_daemon.dart',
        '--socket-path',
        socketPath,
        ?modeFlag,
        ?modeValue,
      ];

  DaemonSupervisor build(String socketPath, List<String> lines,
          List<String> args,
          {void Function()? onDaemonGone,
          int maxRespawns = 2,
          Duration waitForSocket = const Duration(seconds: 10),
          Duration backoff = const Duration(milliseconds: 20)}) =>
      DaemonSupervisor(
        socketPath: socketPath,
        daemonBinary: dartPath,
        argsTemplate: args,
        waitForSocket: waitForSocket,
        connectTimeout: const Duration(seconds: 2),
        maxRespawns: maxRespawns,
        spawnBackoffBase: backoff,
        saLibraryPath: soPath,
        log: lines.add,
      )..onDaemonGone = onDaemonGone;
}

void main() {
  /// A log line's ARRIVAL (the pipe's delivery rides the main isolate's
  /// event queue — it may land after the connect's own completion): the
  /// bounded poll, failing with the whole log's evidence.
  Future<void> awaitLogLine(List<String> lines, String needle,
      [Duration limit = const Duration(seconds: 3)]) async {
    final deadline = DateTime.now().add(limit);
    while (DateTime.now().isBefore(deadline)) {
      if (lines.any((line) => line.contains(needle))) return;
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    fail('the log line never rode ($needle) — the log:\n${lines.join('\n')}');
  }

  // The live gate's resolution, UP FRONT (the group's skip uses it; the
  // tests are sequential so the module level's facts hold for all of them).
  final dartPath = _resolveDart();
  final soPath = _resolveSo();
  final harness = (dartPath != null && soPath != null)
      ? _Harness(dartPath: dartPath, soPath: soPath)
      : null;
  final liveSkip = harness == null
      ? 'no dart on PATH or no libsa_client.so on this run — set '
          'SA_LIBRARY_PATH (e.g. the SecretAgent build) to run the live flow'
      : false;

  test('stop with no child is a no-op (fast, nothing signalled)', () async {
    final lines = <String>[];
    final dir = await Directory.systemTemp.createTemp('pondr-supervisor-');
    addTearDown(() => dir.deleteSync(recursive: true));
    final supervisor = DaemonSupervisor(
      socketPath: '${dir.path}/never.sock',
      argsTemplate: const <String>[],
      log: lines.add,
    );
    final watch = Stopwatch()..start();
    await supervisor.stop();
    expect(watch.elapsed, lessThan(const Duration(seconds: 3)));
    expect(lines, contains('stop: nothing to stop (no spawned daemon child)'));
  });

  test('the spawn-refused binary fails through the budget, loud', () async {
    final lines = <String>[];
    final dir = await Directory.systemTemp.createTemp('pondr-supervisor-');
    addTearDown(() => dir.deleteSync(recursive: true));
    final supervisor = DaemonSupervisor(
      socketPath: '${dir.path}/pondr.sock',
      argsTemplate: const <String>['serve'],
      daemonBinary: 'no-such-binary-pondr-test',
      waitForSocket: const Duration(milliseconds: 200),
      spawnBackoffBase: const Duration(milliseconds: 10),
      maxRespawns: 1,
      log: lines.add,
    );
    await expectLater(
        supervisor.ensure(),
        throwsA(isA<StateError>().having(
          (error) => error.message,
          'message',
          contains('the respawn budget exhausted after 2 attempt(s)'),
        )));
    // Every attempt refused; the backoff rode between them.
    expect(
        lines.where((line) => line.contains('the daemon spawn failed')),
        hasLength(2));
    expect(
        lines.where((line) => line.contains('backing off 10 ms')),
        hasLength(1));
  });

  test('a boot-crash child respawns through the budget, then throws loud',
      () async {
    final local = _resolveDart();
    if (local == null) {
      markTestSkipped('no dart on PATH — the test child cannot be run');
    }
    final lines = <String>[];
    final dir = await Directory.systemTemp.createTemp('pondr-supervisor-');
    addTearDown(() => dir.deleteSync(recursive: true));
    final socketPath = '${dir.path}/pondr.sock';
    final script =
        '${Directory.current.path}/test/daemon/fake_daemon.dart';
    final supervisor = DaemonSupervisor(
      socketPath: socketPath,
      daemonBinary: local!,
      argsTemplate: <String>[script, '--socket-path', socketPath, '--crash'],
      waitForSocket: const Duration(milliseconds: 300),
      spawnBackoffBase: const Duration(milliseconds: 10),
      maxRespawns: 2,
      log: lines.add,
    );
    await expectLater(
        supervisor.ensure(),
        throwsA(isA<StateError>().having(
            (error) => error.message,
            'message',
            contains('the respawn budget exhausted after 3 attempt(s)'))));
    // Three spawn attempts (the first spawn + the 2-budget), three boot
    // deaths, the backoff doubling between them (10 ms then 20 ms).
    expect(lines.where((line) => line.contains('spawning')), hasLength(3));
    expect(
        lines.where((line) => line.contains('the daemon child exited')),
        hasLength(3));
    expect(
        lines.where((line) => line.contains('backing off 10 ms')),
        hasLength(1));
    expect(
        lines.where((line) => line.contains('backing off 20 ms')),
        hasLength(1));
  });

  group('the live .so flow', skip: liveSkip, () {

    test('ensure spawns the daemon, waits the boot, and connects',
        () async {
      final lines = <String>[];
      final theHarness = harness!;
      final home = theHarness.fresh();
      final supervisor = theHarness.build(
          home.socketPath, lines, theHarness.serveArgs(home.socketPath));
      final client = await supervisor.ensure();
      expect(client.isConnected, isTrue);
      expect(client.isSubscribed, isFalse);
      expect(supervisor.isLive, isTrue);
      // The spawn's evidence + the fake's own greeting, drained through the
      // PIPE-logged stdout (its delivery rides the main isolate's event
      // queue — it may land after the connect's own completion, so the
      // arrival is awaited, bounded).
      expect(lines.any((line) => line.contains('spawning')), isTrue);
      await awaitLogLine(lines, 'out: READY');
      // The idempotence: a second ensure hands back the SAME generation (its
      // probe connects first; no second spawn).
      final again = await supervisor.ensure();
      expect(again, same(client));
      expect(lines.where((line) => line.contains('spawning')), hasLength(1));
      await supervisor.stop();
    });

    test('the daemon death fires onDaemonGone; the re-ensure respawns '
        '(bounded)', () async {
      final gone = Completer<void>();
      final lines = <String>[];
      final theHarness = harness!;
      final home = theHarness.fresh();
      final supervisor = theHarness.build(
        home.socketPath,
        lines,
        theHarness.serveArgs(home.socketPath,
            modeFlag: '--die-after', modeValue: '700'),
        onDaemonGone: gone.complete,
      );
      final first = await supervisor.ensure();
      expect(first.isConnected, isTrue);
      // The fake dies mid-life: the child's exit is the gone path's fact.
      await gone.future.timeout(const Duration(seconds: 5));
      expect(supervisor.isLive, isFalse);
      // The re-ensure: the gone generation's teardown joined first, then the
      // fresh spawn loop (a REAL second child, a dead sibling's stale socket
      // file cleared by the fresh daemon).
      final second = await supervisor.ensure();
      expect(second.isConnected, isTrue);
      expect(identical(second, first), isFalse);
      expect(
          lines.any((line) => line.contains('daemon gone: the daemon child '
              'exited')),
          isTrue);
      expect(lines.where((line) => line.contains('spawning')), hasLength(2));
      await supervisor.stop();
    });

    test('stop SIGTERMs the child down and awaits its exit; double-stop is '
        'a no-op', () async {
      final lines = <String>[];
      final theHarness = harness!;
      final home = theHarness.fresh();
      final supervisor = theHarness.build(
          home.socketPath, lines, theHarness.serveArgs(home.socketPath));
      await supervisor.ensure();
      final watch = Stopwatch()..start();
      await supervisor.stop();
      expect(watch.elapsed, lessThan(const Duration(seconds: 3)));
      // The exit awaited (the fake's graceful SIGTERM handler exits 0 when
      // it wins — with a CONNECTED client on its server socket the dart VM
      // sometimes loses the handler to the signal's default action (the
      // -15 kill): the supervisor's contract is the awaited exit either way.
      expect(
          lines.any((line) => line.contains('the daemon child exited')),
          isTrue);
      expect(supervisor.isLive, isFalse);
      // The double-stop: no child left, nothing to join, a fast no-op.
      await supervisor.stop();
      expect(
          lines
              .where((line) => line.contains('stop: nothing to stop'))
              .length,
          1);
      expect(watch.elapsed, lessThan(const Duration(seconds: 3)));
    });
  });
}