/// THE ONE SWAP (per the plan + spec): every interface is served through a
/// provider here; the later real bindings (the runtime FFI client, the
/// engine's subconscious surface) replace these lines and NOTHING else in
/// the app changes. Views import this file — never `mock/*` directly.
///
/// THE MODE FLAG (the FFI-binding plan's Task 5):
/// `--dart-define=pondr.mode=daemon` runs REAL — the supervised
/// `frame-demo serve` daemon + the direct dart:ffi on `libsa_client.so`
/// behind the [SessionsService]/[ChatService]/[SettingsService] shapes.
/// Anything else (including no define) is MOCK — the export's behaviours.
/// [SubconsciousService] STAYS MOCK in both modes (the spec's recorded
/// out-of-scope: the engine's serving surface is a later slice).
///
/// THE DAEMON WIRING (keepAlive-r providers — app-level lifetimes; the
/// container's teardown is the dispose hook):
/// - [appConfigStoreProvider] — the settings' JSON file.
/// - [daemonSupervisorProvider] — one [DaemonSupervisor]: the socket path +
///   the spawn's args (+ the model tag from the selected provider) composed
///   at construction; `stop()` rides the provider's dispose.
/// - [daemonClientProvider] — the FIRST ensure's client (the connect;
///   spawning happens here, lazily, when the app's first daemon-mode watch
///   lands).
/// - the three FFI services fold/subscribe through it — see
///   `lib/data/daemon/ffi_chat_service.dart` for the subscription + dedupe
///   policy, `ffi_settings_service.dart` for the daemon adoption, and
///   `lib/daemon/app_config_store.dart` for the store's shape.
///
/// THE ATTACHMENTS (recorded): a daemon-mode send carries TEXT ONLY — the
/// `msg.append` record has no file payload and no upload verb exists on the
/// wire yet (a recorded follow-on); the files ride the composer in mock
/// mode only.
library;

import 'dart:async';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:pondr/ffi/sa_client_binding.dart';

import '../daemon/app_config_store.dart';
import '../daemon/supervisor.dart';
import 'daemon/ffi_chat_service.dart';
import 'daemon/ffi_settings_service.dart';
import 'daemon/ffi_sessions_service.dart';
import 'mock/mock_services.dart';
import 'services.dart';

export 'mock/mock_data.dart';

/// The mode define (the build's one flag): `daemon` runs REAL; every other
/// value (mock included) is the export's mock.
const String pondrModeDefine =
    String.fromEnvironment('pondr.mode', defaultValue: 'mock');

/// The binding's mode (const — one app build, one mode): only the exact
/// 'daemon' define runs real.
const bool useDaemonMode = pondrModeDefine == 'daemon';

/// The mode's parse, testable (the define itself is a compile-time const —
/// a test cannot re-define it, so the parse is pinned instead).
bool useDaemonModeFromDefine(String define) => define == 'daemon';

// ── the mock wiring (the export's shapes, unchanged) ────────────────────────

/// The concrete mock, so [chatServiceProvider] can wire the SAME store the
/// views read (the export's handleSend writes through `setSessions`).
final _mockSessionsProvider = Provider<MockSessionsService>(
  (ref) => MockSessionsService(),
);

// ── the daemon wiring ───────────────────────────────────────────────────────

/// The settings' JSON file (the app-local store).
final appConfigStoreProvider = Provider<AppConfigStore>((ref) {
  ref.keepAlive();
  return AppConfigStore();
});

/// The daemon's AF_UNIX socket slot (the spec's `<cache>/pondr.sock`; the
/// XDG_CACHE_HOME convention, `~/.cache` otherwise) — LINUX v1.
String defaultDaemonSocketPath() {
  final cacheHome = Platform.environment['XDG_CACHE_HOME'];
  final base = cacheHome == null || cacheHome.isEmpty
      ? '${Platform.environment['HOME'] ?? '.'}/.cache'
      : cacheHome;
  return '$base/pondr/pondr.sock';
}

/// The daemon binary's default (the supervisor's spawn target): the
/// `prepare.sh` copy (`flutter/daemon/frame-demo`, resolved against the run's
/// working directory — a `flutter run` from the project dir), else the
/// `PONDR_DAEMON_BIN` env's explicit path, else the PATH's bare
/// `frame-demo` (the supervisor walks the PATH itself).
String defaultDaemonBinary() {
  final envSet = Platform.environment['PONDR_DAEMON_BIN'];
  if (envSet != null && envSet.isNotEmpty) return envSet;
  final local = '${Directory.current.path}'
      '${Platform.pathSeparator}daemon${Platform.pathSeparator}frame-demo';
  if (File(local).existsSync()) return local;
  return 'frame-demo';
}

/// One supervisor for the app run: the socket + the spawn's args composed
/// HERE (the model tag from the selected provider's `modelId` — the store's
/// truth read at construction; the picker's later SELECTION re-templates the
/// daemon through the CA_CONFIG set instead — the adoption rides
/// [daemonSettingsProvider]); `stop()` is the provider's dispose hook (the
/// app's exit teardown).
final daemonSupervisorProvider = Provider<DaemonSupervisor>((ref) {
  ref.keepAlive();
  final store = ref.watch(appConfigStoreProvider);
  final socketPath = defaultDaemonSocketPath();
  final config = store.load();
  final selected = resolveSelectedModel(config.providers, config.selectedModelKey);
  final supervisor = DaemonSupervisor(
    socketPath: socketPath,
    daemonBinary: defaultDaemonBinary(),
    argsTemplate: <String>[
      'serve',
      '--socket-path',
      socketPath,
      '--model',
      selected?.model.modelId ?? '',
    ],
  );
  ref.onDispose(() => unawaited(supervisor.stop()));
  return supervisor;
});

/// The FIRST ensure's client (the connect; a manual serve is probed first,
/// a spawn only when nothing answers). The supervisor's channels (its
/// `events`/`failures` fan-outs) belong to the same wiring — the chat's
/// record fold rides them.
final daemonClientProvider = FutureProvider<SaClientNative>((ref) async {
  ref.keepAlive();
  final supervisor = ref.watch(daemonSupervisorProvider);
  return supervisor.ensure();
});

/// The ensured client's future, read LAZY behind a closure — the daemon
/// settings' adoption reference (a construction cycle's breaker: the
/// supervisor's model tag rides the same settings).
final daemonSettingsProvider = Provider<FfiSettingsService>((ref) {
  ref.keepAlive();
  final store = ref.watch(appConfigStoreProvider);
  return FfiSettingsService(
    store: store,
    client: () => ref.read(daemonClientProvider.future),
  );
});

final _daemonSessionsProvider = Provider<FfiSessionsService>((ref) {
  ref.keepAlive();
  return FfiSessionsService(clientFuture: ref.watch(daemonClientProvider.future));
});

// ── the four interface providers (the swap) ─────────────────────────────────

/// The export's `sessions` state.
final sessionsProvider = Provider<SessionsService>(
  (ref) => useDaemonMode
      ? ref.watch(_daemonSessionsProvider)
      : ref.watch(_mockSessionsProvider),
);

/// The export's typing + reply flow.
final chatServiceProvider = Provider<ChatService>(
  (ref) {
    if (!useDaemonMode) {
      return MockChatService(ref.watch(_mockSessionsProvider));
    }
    final sessions = ref.watch(_daemonSessionsProvider);
    final supervisor = ref.watch(daemonSupervisorProvider);
    final chat = FfiChatService(
      client: ref.watch(daemonClientProvider.future),
      sessions: sessions,
      events: supervisor.events,
      failures: supervisor.failures,
    );
    ref.onDispose(chat.dispose);
    return chat;
  },
);

/// The export's settings state (profile / notifications / accent /
/// providers CRUD) — daemon mode: the JSON file's store + the daemon's
/// CA_CONFIG adoption.
final settingsProvider = Provider<SettingsService>(
  (ref) => useDaemonMode
      ? ref.watch(daemonSettingsProvider)
      : MockSettingsService(),
);

/// The export's BASE pair (STAYS mock: the engine's serving surface is a
/// later slice — the spec's recorded out-of-scope).
final subconsciousProvider = Provider<SubconsciousService>(
  (ref) => MockSubconsciousService(),
);