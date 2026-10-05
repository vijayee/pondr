/// The daemon supervisor (the spec §2's supervision block): who runs the
/// daemon. The app spawns + supervises; a manual serve is still legal —
/// [ensure] probes the connection FIRST and spawns `daemonBinary` only when
/// nothing answers the socket. [stop] is the app-exit hook.
///
/// THE SHAPE:
/// - [ensure]: idempotent (concurrent callers share the in-flight attempt).
///   One connection probe per attempt: the exists-serving check. It is the C
///   client's unix connect, a ONE blocking `connect()` syscall — honest
///   without a wire exchange (any listening socket answers). A connected
///   probe (a manual serve) ends the attempt with no spawn and no ownership:
///   a manual daemon is never [stop]'d or signalled, only connected to.
/// - the spawn loop: `Process.start(daemonBinary, argsTemplate)`; the boot
///   wait is the honest two-step poll, bounded by [waitForSocket] per
///   attempt — the socket FILE's existence, then the C client's CONNECT
///   itself, polled on its failure (the C's unix connect refuses INSTANTLY
///   against a missing or listener-less socket file — it never waits, so the
///   poll IS the wait). A dead-exit on boot respawns through the budget:
///   [maxRespawns] respawns after the first spawn (the client's
///   `max_retries` analog), each retry after a [spawnBackoffBase] doubling
///   sleep capped at 8 s — sa_client's own reconnect cadence. The budget's
///   exhaustion throws loud, naming the attempt count.
/// - the daemon's DEATH while live: [onDaemonGone], once per generation —
///   the spawned child's exit, or the client's DISCONNECTED failure (the
///   death of a manually-served daemon). The supervisor does NOT respawn:
///   the app layers its policy (a re-[ensure]). The dead client's teardown
///   starts at the signal (the C reconnect loop with `max_retries = 0`
///   would otherwise retry forever against the gone socket) and is awaited
///   by the next [ensure]/[stop].
/// - [events]/[failures]: the connect's callbacks, fanned out to broadcast
///   streams — the client belongs to the supervisor; the app's layers ride
///   these (the chat's message source of truth is the events stream; the
///   disconnect failures carry both channel and fact).
/// - [stop]: the client's dispose FIRST (its unsubscribe rides a
///   still-speaking daemon — the honest teardown order), then the child's
///   SIGTERM → [stopGrace] → SIGKILL, the exit awaited either way. No child
///   = a no-op. Safe to call twice. An in-flight [ensure] joins it (its
///   attempt aborts; a half-spawned child is put down by its own loop). The
///   supervisor's life ends at its stop: an [ensure] after it refuses loud
///   (a NEW daemon for a NEW app run is a NEW supervisor).
///
/// The .so: [saLibraryPath] pins libsa_client.so's path; null = the FFI
/// layer's own resolution order (the define, the env, the exe dir's `lib/`).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:pondr/ffi/sa_client_binding.dart';
import 'package:pondr/ffi/sa_ffi.dart';

final class DaemonSupervisor {
  DaemonSupervisor({
    required this.socketPath,
    required this.argsTemplate,
    this.daemonBinary = 'frame-demo',
    this.connectTimeout = const Duration(seconds: 10),
    this.waitForSocket = const Duration(seconds: 5),
    this.maxRespawns = 5,
    this.spawnBackoffBase = const Duration(seconds: 1),
    this.stopGrace = const Duration(seconds: 3),
    this.saLibraryPath,
    void Function(String line)? log,
  }) : _log = log ?? _stdoutLog;

  /// The daemon's AF_UNIX socket file.
  final String socketPath;

  /// The spawn's args, VERBATIM — the caller composes it from its settings
  /// (the socket path AND the model tag ride from there; the template is
  /// baked at construction: `['serve', '--socket-path', socketPath,
  /// '--model', modelTag]`). The model tag rides from settings — this layer
  /// adds nothing.
  final List<String> argsTemplate;

  /// The daemon's binary. A bare name walks the PATH's entries (the
  /// `frame-demo` default's resolution); a name with a path separator runs
  /// verbatim. FB6's runtime settings section will carry the real path —
  /// until then this is the constructor's default, on PATH resolved.
  final String daemonBinary;

  /// One connect probe's budget (the C client's connect timeout).
  final Duration connectTimeout;

  /// The boot budget: the socket file's wait AND the polled connect's wait —
  /// per spawn attempt (one attempt may overrun by up to [connectTimeout]:
  /// the C's own connect holds a wedge-socket's block).
  final Duration waitForSocket;

  /// The respawn budget (the client's `max_retries` analog): the respawns
  /// AFTER the first spawn — 5 means six attempts, all per [ensure].
  final int maxRespawns;

  /// The respawn backoff's base: doubled per attempt, capped at 8 s —
  /// sa_client's own reconnect cadence.
  final Duration spawnBackoffBase;

  /// The SIGTERM's grace before the SIGKILL (in [stop] and in the failed
  /// attempts' child cleanups).
  final Duration stopGrace;

  /// libsa_client.so's explicit path (null = the FFI layer's resolution).
  final String? saLibraryPath;

  /// The boot wait's step between the polls.
  static const Duration _pollStep = Duration(milliseconds: 25);

  /// The backoff's cap (sa_client's reconnect max).
  static const Duration _maxBackoff = Duration(seconds: 8);

  final void Function(String line) _log;

  static void _stdoutLog(String line) =>
      stdout.writeln('[daemon/supervisor] $line');

  // ── the live generation ───────────────────────────────────────────────────

  /// The generation's client — live, or dying mid-teardown (a gone mark's
  /// dispose, awaited by the next [ensure]/[stop] before anything new).
  SaClientNative? _client;

  /// The generation's child — ONLY ever a process this supervisor spawned
  /// (the manual serve's owner keeps the signal rights).
  Process? _child;

  /// A spawned child only: its exit is a daemon-gone once it went live.
  bool _childLive = false;

  /// The current generation's daemon is gone.
  bool _gone = false;

  /// The app-exit's stop ran (the in-flight ensure's abort signal).
  bool _stopping = false;

  Future<void>? _teardown;
  Future<SaClientNative>? _ensuring;

  // ── the deliveries (the connect callbacks' fan-out) ───────────────────────

  final StreamController<SaEventRecord> _events =
      StreamController<SaEventRecord>.broadcast();

  final StreamController<SaFailure> _failures =
      StreamController<SaFailure>.broadcast();

  /// Every events record AND marker the client delivers — the chat's
  /// message source of truth (the send's user bubble IS the record's
  /// arrival). A respawn's gap is the stream's shape (the re-[ensure]'s
  /// fresh subscription replays the gap from seq 0).
  Stream<SaEventRecord> get events => _events.stream;

  /// Every request failure (a daemon refusal, a timeout, a lost connection).
  Stream<SaFailure> get failures => _failures.stream;

  /// The daemon's loss, once per generation — settable (checked at the fire
  /// moment). The app's respawn policy hooks here.
  void Function()? onDaemonGone;

  /// The current generation's state: a client answers the socket.
  bool get isLive => _client != null && !_gone;

  // ── ensure ────────────────────────────────────────────────────────────────

  /// The live client: connected (the manual serve) or spawned + waited +
  /// connected. Concurrent callers share the in-flight attempt; a failed
  /// attempt throws to all of them — retry by calling again.
  Future<SaClientNative> ensure() {
    final running = _ensuring;
    if (running != null) return running;
    final future = _ensure();
    _ensuring = future;
    // The memo's own future carries the original's error — nobody else
    // awaits it: ignore() marks it handled (unawaited would NOT — the error
    // would double-deliver as an unhandled zone error).
    future.whenComplete(() => _ensuring = null).ignore();
    return future;
  }

  Future<SaClientNative> _ensure() async {
    // The stop's promise: nothing hands out a client under a stopping
    // supervisor (a post-stop ensure is the loud refusal, not a respawn —
    // the supervisor's life ends at its stop).
    if (_stopping) {
      throw StateError('the supervisor has stopped; ensure is refused');
    }
    final previous = _client;
    if (previous != null && !_gone) return previous;
    // The gone generation's teardown finishes before the next connect (its
    // C reader is mid-teardown; nobody reconnects under a destroying client).
    final retiring = _teardown;
    if (retiring != null) await retiring;
    // The generation's facts are past: a fresh cycle reads a quiet state.
    _client = null;
    _gone = false;

    // The attempt loop: one probe + one spawn + one boot wait each, the
    // backoff between; EVERY failure mode (a refusal to the probe, a dead
    // boot, a never-listening child, a spawn refusal) falls to the bottom's
    // budget check. maxRespawns + 1 attempts total, bounded and loud.
    var delay = spawnBackoffBase;
    for (var attempt = 0;; attempt++) {
      if (_stopping) {
        break;
      }
      // The exists-serving probe (a manual serve that showed up mid-way is
      // the same honesty as the first one).
      final probed = await _probe();
      if (probed != null) {
        // stop() started while the probe was in flight: its dispose owns the
        // client, never hand it out past the stop-promise (the review's gate).
        if (_stopping) {
          _log('the probe landed under stop — the client was probed into '
              'the teardown; refusing');
          throw StateError('daemon supervision: stopped mid-ensure');
        }
        _client = probed;
        return probed;
      }
      if (attempt > 0) {
        _log('backing off ${delay.inMilliseconds} ms before respawn '
            '$attempt of $maxRespawns');
        await _backoffSleep(delay);
        if (_stopping) break;
        delay = delay * 2 > _maxBackoff ? _maxBackoff : delay * 2;
      }
      _log('spawning $daemonBinary (attempt ${attempt + 1} of '
          '${maxRespawns + 1})');
      Process? child;
      try {
        child = await Process.start(_resolveBinary(daemonBinary), argsTemplate);
      } catch (e) {
        _log('the daemon spawn failed (attempt ${attempt + 1}): $e');
      }
      if (child != null) {
        _child = child;
        _childLive = false;
        _wireChild(child);
        final (client, died) = await _waitAndConnect(child);
        if (client != null) {
          // The client's and the child's death signals gate on these — set
          // them in this same turn (the first thing after the connect's
          // resolution), so a drop racing the return still finds its owner.
          _client = client;
          _childLive = true;
          if (_stopping) {
            // stop() ran mid-spawn: the generation is never handed out —
            // the loop puts BOTH of its products down itself (stop's own
            // kill pass already ran; it saw nothing here).
            _client = null;
            _childLive = false;
            await _dispose(client);
            await _killChild(child);
            throw StateError('the supervisor stopped mid-ensure');
          }
          return client;
        }
        // The attempt failed. A still-running child is ours to put down
        // before the next spawn (its boot produced no listening socket).
        _child = null;
        _childLive = false;
        if (!died) {
          _log('the daemon never came up on $socketPath; putting the child '
              'down');
          await _killChild(child);
        }
      }
      if (_stopping) break;
      if (attempt == maxRespawns) {
        throw StateError(
            'the daemon never came up on $socketPath ($daemonBinary) — '
            'the respawn budget exhausted after ${maxRespawns + 1} '
            'attempt(s)');
      }
    }
    throw StateError(
        'the supervisor stopped mid-ensure (the daemon never came up on '
        '$socketPath)');
  }

  /// The backoff's sleep in [stop]-slicing steps (the step is a coarse
  /// granularity — an app-exit never waits out a whole backoff period).
  Future<void> _backoffSleep(Duration delay) async {
    for (var remaining = delay;
        remaining > Duration.zero && !_stopping;) {
      final slice =
          remaining < _pollStep ? remaining : _pollStep;
      await Future<void>.delayed(slice);
      remaining -= slice;
    }
  }

  /// One connect probe: a fresh client against the socket. A refusal — a
  /// missing listener, a .so that never loads — answers null, and the failed
  /// probe tears itself down (the wrapper's own refusal flow; the wrapper's
  /// dispose is idempotent after it).
  Future<SaClientNative?> _probe() async {
    final client = _buildClient();
    try {
      final result = await client.connect(_config(), _callbacks(client));
      return result is SaOk ? client : null;
    } catch (e) {
      _log('the probe connect failed: $e');
      unawaited(client.dispose());
      return null;
    }
  }

  /// The boot wait, per attempt, bounded by [waitForSocket]: the socket
  /// FILE's existence poll, then the client's connect — polled on its
  /// failure (a stale file from a dead daemon passes the file gate; the
  /// connect itself answers the missing LISTENER, and the fresh daemon
  /// clears its own stale file before the bind). A dead-exit mid-wait ends
  /// the attempt right there. Returns the client (the attempt landed) or
  /// null (with the child's death as the second fact — a dead child needs
  /// no putting-down).
  Future<(SaClientNative?, bool)> _waitAndConnect(Process child) async {
    final deadline = DateTime.now().add(waitForSocket);
    var died = false;
    unawaited(child.exitCode.then((_) => died = true));
    for (;;) {
      if (died) {
        _log('the daemon child died during its boot');
        return (null, true);
      }
      if (_stopping) return (null, false);
      if (!DateTime.now().isBefore(deadline)) {
        _log("the daemon's boot wait elapsed "
            '(${waitForSocket.inMilliseconds} ms)');
        return (null, false);
      }
      if (FileSystemEntity.typeSync(socketPath) ==
          FileSystemEntityType.notFound) {
        await Future<void>.delayed(_pollStep);
        continue;
      }
      final client = _buildClient();
      try {
        final result = await client.connect(_config(), _callbacks(client));
        if (result is SaOk) return (client, false);
      } catch (e) {
        _log('the connect poll failed: $e');
        unawaited(client.dispose());
      }
      await Future<void>.delayed(_pollStep);
    }
  }

  // ── the death ─────────────────────────────────────────────────────────────

  /// The generation's connect callbacks: every delivery fans out; a
  /// DISCONNECTED failure is ALSO the (manual serve's) daemon-gone.
  SaCallbacks _callbacks(SaClientNative client) => SaCallbacks(
        onEvent: _events.add,
        onError: (failure) {
          _failures.add(failure);
          if (failure.status == saClientStatusDisconnected &&
              identical(client, _client) &&
              !_gone) {
            _markGone('the client reported the connection lost '
                '(${failure.text})');
          }
        },
      );

  /// The generation's child wiring: the exit's log + the gone path, the
  /// stdout/stderr's PIPE-drain (the logs; a full pipe would wedge the
  /// child's writes).
  void _wireChild(Process child) {
    unawaited(child.exitCode.then((code) {
      _log('the daemon child exited (code $code)');
      // Only the LIVE generation's death is a daemon-gone: a boot crash or
      // a stop()'s kill belongs to its own flow.
      if (!identical(child, _child) || !_childLive) return;
      _child = null;
      _markGone('the daemon child exited (code $code)');
    }));
    for (final (String label, Stream<List<int>> stream) in <(String,
        Stream<List<int>>)>[
      ('out', child.stdout),
      ('err', child.stderr),
    ]) {
      stream
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen((line) => _log('$label: $line'));
    }
  }

  /// Once per generation: announce, fire the app's policy hook, tear the
  /// client down (bounded — the C reconnect loop must not retry forever
  /// against a gone daemon). The next [ensure] awaits the teardown.
  void _markGone(String reason) {
    if (_gone) return;
    _gone = true;
    _log('daemon gone: $reason');
    final hook = onDaemonGone;
    if (hook != null) hook();
    final client = _client;
    if (client != null) {
      _teardown = _dispose(client);
    }
  }

  /// The client's full teardown (a gone generation's AT THE SIGNAL, or the
  /// stop's); at its tail the generation's facts clear (only when this
  /// client still IS the current one — stop() may have cleared them
  /// already).
  Future<void> _dispose(SaClientNative client) async {
    try {
      await client.dispose();
    } catch (e) {
      _log("the supervisor's client teardown failed: $e");
    }
    if (identical(_client, client)) {
      _client = null;
      _gone = false;
    }
  }

  // ── stop ──────────────────────────────────────────────────────────────────

  /// The app-exit hook: any in-flight [ensure] joins first (its attempt
  /// aborts), the gone generation's teardown joins, the live client's own
  /// dispose runs (its unsubscribe rides a still-speaking daemon), THEN the
  /// child's SIGTERM → [stopGrace] → SIGKILL, the exit awaited either way.
  /// No client and no child = logged and done. Safe to call twice.
  Future<void> stop() async {
    _stopping = true;
    final inFlight = _ensuring;
    if (inFlight != null) {
      try {
        await inFlight;
      } catch (_) {
        // The aborted attempt's own throw: not the stop's business.
      }
      if (identical(_ensuring, inFlight)) _ensuring = null;
    }
    final retiring = _teardown;
    if (retiring != null) await retiring;
    final client = _client;
    if (client != null) {
      _childLive = false;
      _client = null;
      _gone = false;
      await _dispose(client);
    }
    final child = _child;
    if (child == null) {
      _log('stop: nothing to stop (no spawned daemon child)');
      return;
    }
    _child = null;
    _childLive = false;
    await _killChild(child);
  }

  /// SIGTERM, the grace, then the SIGKILL; the exit awaited either way.
  Future<void> _killChild(Process child) async {
    _log('stopping the daemon child (pid ${child.pid})');
    child.kill(ProcessSignal.sigterm);
    try {
      await child.exitCode.timeout(stopGrace);
    } on TimeoutException {
      _log('the SIGTERM grace (${stopGrace.inMilliseconds} ms) elapsed; '
          'SIGKILL');
      child.kill(ProcessSignal.sigkill);
      await child.exitCode;
    }
  }

  // ── the pieces ────────────────────────────────────────────────────────────

  SaClientNative _buildClient() => SaClientNative(api: SaFfi(saLibraryPath));

  /// The connect's config: unix on [socketPath], [connectTimeout] as the C
  /// connect's budget; the request timeout and the reconnect cadence stay
  /// the binding's defaults (the events channel's `max_retries = 0`
  /// reconnects — the daemon's return reattaches a subscribed client).
  SaConfig _config() => SaConfig(
        transport: SaTransport.unix,
        socketPath: socketPath,
        connectTimeoutMs: connectTimeout.inMilliseconds,
      );

  /// A bare name walks the PATH's entries (the `frame-demo` default's
  /// resolution); anything with a separator runs verbatim. A bare name the
  /// PATH does not carry passes through anyway — the spawn's
  /// ProcessException is the loud refusal (the spawn loop's failed attempt).
  String _resolveBinary(String binary) {
    if (binary.contains(Platform.pathSeparator)) return binary;
    final paths = Platform.environment['PATH'] ?? '';
    for (final dir in paths.split(':')) {
      if (dir.isEmpty) continue;
      final candidate = '$dir${Platform.pathSeparator}$binary';
      if (FileSystemEntity.typeSync(candidate) !=
          FileSystemEntityType.notFound) {
        return candidate;
      }
    }
    return binary;
  }
}