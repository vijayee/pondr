// The fake daemon (the supervisor tests' REAL child): a `dart`-invoked
// helper that binds a LISTENING AF_UNIX socket at its --socket-path, prints
// READY, and sleeps — the honest end-to-end shape for the supervisor's
// spawn/wait/connect (the FFI client's unix connect needs only a LISTENING
// socket; the fake speaks no wire). Modes beyond the plain serve:
// --die-after <ms> — serve, then exit(3) once the delay passes (the
//     mid-life death's script; the onDaemonGone path's test).
// --crash — exit(9) immediately, the socket never appears (the boot crash's
//     script; the respawn budget's test).
library;

import 'dart:async';
import 'dart:io';

Future<void> main(List<String> args) async {
  var socketPath = '';
  var dieAfterMs = -1;
  var crash = false;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--socket-path':
        if (i + 1 >= args.length) {
          stderr.writeln('fake_daemon: --socket-path needs a value');
          exit(2);
        }
        socketPath = args[++i];
      case '--die-after':
        if (i + 1 >= args.length) {
          stderr.writeln('fake_daemon: --die-after needs a value');
          exit(2);
        }
        dieAfterMs = int.parse(args[++i]);
      case '--crash':
        crash = true;
      default:
        stderr.writeln('fake_daemon: unknown arg ${args[i]}');
        exit(2);
    }
  }
  if (crash) {
    stdout.writeln('CRASH');
    exit(9);
  }
  if (socketPath.isEmpty) {
    stderr.writeln('fake_daemon: --socket-path is required');
    exit(2);
  }
  // The daemon's own stale-file discipline (the unix transport unlinks its
  // bind path first): a leftover socket file from a dead sibling would kill
  // the bind with EADDRINUSE otherwise.
  final existing = FileSystemEntity.typeSync(socketPath);
  if (existing != FileSystemEntityType.notFound) {
    try {
      File(socketPath).deleteSync();
    } catch (e) {
      stderr.writeln("fake_daemon: the stale socket file didn't clear: $e");
      exit(4);
    }
  }
  // The LISTENER: the accepted sockets are kept open (the client's connect
  // rides the listening state alone; closing them would drop the client's
  // channel).
  final server =
      await ServerSocket.bind(InternetAddress(socketPath, type:
          InternetAddressType.unix), 0);
  server.listen((_) {});
  stdout.writeln('READY');
  if (dieAfterMs >= 0) {
    unawaited(
      Future<void>.delayed(
        Duration(milliseconds: dieAfterMs),
        () => exit(3),
      ),
    );
  }
  // SIGTERM's clean exit (the stop test's shape: the exit code rides 0).
  ProcessSignal.sigterm.watch().listen((_) => exit(0));
  await Completer<void>().future;
}