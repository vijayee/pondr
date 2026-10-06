/// The dart:ffi ABI on sa_client (`SecretAgent/src/ClientLibs/c/sa_client.h`)
/// — the .so resolution, the struct mirrors, the callback typedefs, and the
/// one seam (`SaFfiApi`) the binding (`sa_client_binding.dart`) and its unit
/// tests depend on.
///
/// The layer is deliberately DEPENDENCY-FREE: everything comes from dart:ffi
/// (`package:ffi`'s Utf8/calloc conveniences were probed and are NOT in
/// dart:ffi — `SaUtf8` below is the manual equivalent, backed by libc's
/// own malloc/free via `DynamicLibrary.process()`).
///
/// The ABI NEVER DRIFTS SILENTLY: the C side answers the struct's byte shape
/// through the exported probes (`sa_client_config_ffi_sizeof`,
/// `sa_client_config_ffi_offset`); `test/ffi/ffi_test.dart` pins the mirror
/// against them (and against the constants here) — see the constants block.
library;

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';

// ── the statuses handed to callbacks (sa_client.h's SA_CLIENT_STATUS_*) ─────

const int saClientStatusOk = 0;
const int saClientStatusTimeout = 10;
const int saClientStatusDisconnected = 11;
const int saClientStatusAlloc = 12;
const int saClientStatusBusy = 13;
const int saClientStatusLocal = 14;
const int saClientStatusReentrant = 15;

/// The C's `-1` op returns (the outright refusal that fires NO callback and
/// carries no status) surfaced as the failure status the Dart side sees.
const int saClientStatusCallRefused = -1;

/// The events channel's marker ops (the wire's CA_EVENTS_* request op,
/// echoed by every marker — see `client_api_wire.h` + sa_client.h's events
/// callback note). A marker = seq 0 + a NULL record_json.
const int saEventsOpReplayThenLive = 0;
const int saEventsOpLiveOnly = 1;
const int saEventsOpUnsubscribe = 2;

/// THE STRUCT-DRIFT TRIPWIRE, pinned once here: `sa_client_config_t`'s byte
/// shape as the C's exported probes answer it (member count 10, in the
/// header's declared order). The live ABI test asserts C == these constants
/// == the Dart mirror below; the non-live test asserts the mirror alone.
const int saClientConfigFfiSizeof = 72;
const List<int> saClientConfigFfiOffsets = <int>[
  0, // transport (sa_transport_e — a 4-byte C enum)
  8, // socket_path (pointer, after the enum's 4 padding bytes)
  16, // host
  24, // port (uint16; 6 padding bytes follow)
  32, // api_key
  40, // connect_timeout_ms
  44, // request_timeout_ms
  48, // max_retries
  56, // error_cb (pointer, after 4 padding bytes)
  64, // error_ctx
];

/// The transports (the C enum `sa_transport_e` rides the ABI as a 4-byte
/// int; the values are the enum's).
enum SaTransport {
  unix(0),
  tcp(1);

  const SaTransport(this.valueOf);

  final int valueOf;

  static SaTransport fromValue(int value) =>
      value == tcp.valueOf ? tcp : unix;
}

// ── the .so resolution ──────────────────────────────────────────────────────

/// The bundle's conventional slot: `Platform.resolvedExecutable`'s directory
/// + `/lib/` (what `flutter build linux` fills in production).
String saLibraryExeCandidate() {
  final exeDir = File(Platform.resolvedExecutable).parent.path;
  return '$exeDir/lib/libsa_client.so';
}

/// The two conventional (non-explicit) resolution steps, first-existing wins.
List<String> saLibraryCandidates() => <String>[
      saLibraryExeCandidate(),
      './libsa_client.so',
    ];

/// Resolves `libsa_client.so` — in order:
///
/// 1. the compile-time define (`--dart-define=SA_LIBRARY_PATH=…`);
/// 2. the runtime env var `SA_LIBRARY_PATH` (the same name — both sources are
///    supported on purpose: the define pins a BUILD, the env steers a DEV
///    run without a rebuild);
/// 3. `$ExeDir/lib/libsa_client.so`;
/// 4. `./libsa_client.so`.
///
/// The explicit sources are tried in that order — the first existing wins
/// (a missing define falls through to the env); when EVERY set explicit
/// source names a missing file, that's a misconfiguration — it FAILS LOUD
/// (no silent fallback onto the conventional candidates). With neither set,
/// the first existing conventional candidate wins or this throws a
/// [StateError] naming every candidate. [define]/[env]/[candidates] are the
/// test seams for the same branches (defaults read the real sources).
String resolveSaLibraryPath({String? define, String? env, List<String>? candidates}) {
  final explicit = <String>[];
  final dartDefined = define ?? const String.fromEnvironment('SA_LIBRARY_PATH');
  if (dartDefined.isNotEmpty) explicit.add(dartDefined);
  final envSet = env ?? Platform.environment['SA_LIBRARY_PATH'];
  if (envSet != null && envSet.isNotEmpty) explicit.add(envSet);
  for (final path in explicit) {
    if (File(path).existsSync()) return path;
  }
  if (explicit.isNotEmpty) {
    throw StateError(
      'SA_LIBRARY_PATH (${explicit.join(', ')}) names no existing '
      'libsa_client.so — the explicit source is authoritative, no fallback',
    );
  }
  for (final path in candidates ?? saLibraryCandidates()) {
    if (File(path).existsSync()) return path;
  }
  throw StateError(
    'no libsa_client.so found — tried '
    '${(candidates ?? saLibraryCandidates()).join(', ')} (pass '
    '--dart-define=SA_LIBRARY_PATH= or set the SA_LIBRARY_PATH env var)',
  );
}

// ── the native text mover (dart:ffi has no Utf8 helpers of its own) ─────────

/// NUL-terminated UTF-8 across the ABI, backed by libc's malloc/free (see
/// the library doc: dart:ffi carries no string or allocator conveniences).
final class SaUtf8 {
  SaUtf8._();

  /// Copies [text] into freshly malloc'd NUL-terminated UTF-8. Rejects
  /// strings containing U+0000 — the wire's bounded-string discipline never
  /// carries one, and a literal NUL would truncate the copy.
  static ffi.Pointer<ffi.Uint8> toNative(String text) {
    if (text.contains('\x00')) {
      throw ArgumentError.value(text, 'text', 'a NUL never crosses the wire');
    }
    final bytes = utf8.encode(text);
    final memory = _malloc(bytes.length + 1).cast<ffi.Uint8>();
    final view = memory.asTypedList(bytes.length + 1);
    view.setAll(0, bytes);
    view[bytes.length] = 0;
    return memory;
  }

  /// The pointer's held text (see sa_client.h's ownership rule: valid until
  /// released or destroyed — the caller copies FIRST, releases after).
  static String read(ffi.Pointer<ffi.Uint8> p) {
    var length = 0;
    while (p[length] != 0) {
      length++;
    }
    return utf8.decode(p.asTypedList(length), allowMalformed: true);
  }

  static void free(ffi.Pointer<ffi.Uint8> p) => _free(p.cast<ffi.Void>());

  static ffi.Pointer<ffi.Uint8> alloc(int bytes) =>
      _malloc(bytes).cast<ffi.Uint8>();

  static void freeAddress(int address) =>
      _free(ffi.Pointer<ffi.Uint8>.fromAddress(address).cast<ffi.Void>());
}

typedef _MallocNative = ffi.Pointer<ffi.Void> Function(ffi.Size);
typedef _MallocDart = ffi.Pointer<ffi.Void> Function(int);
typedef _FreeNative = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef _FreeDart = void Function(ffi.Pointer<ffi.Void>);

final _malloc = ffi.DynamicLibrary.process()
    .lookupFunction<_MallocNative, _MallocDart>('malloc');
final _free =
    ffi.DynamicLibrary.process().lookupFunction<_FreeNative, _FreeDart>('free');

// ── the struct mirrors ──────────────────────────────────────────────────────

/// `sa_client_config_t` (sa_client.h:134-161), mirrored field-for-field in
/// the header's declared order (dart:ffi lays structs out in declaration
/// order; the pointer fields 4-byte-align like the C's). The C members, in
/// index order: transport, socket_path, host, port, api_key,
/// connect_timeout_ms, request_timeout_ms, max_retries, error_cb, error_ctx.
final class SaClientConfigFfi extends ffi.Struct {
  @ffi.Int32()
  external int transport; // sa_transport_e — a C enum int

  external ffi.Pointer<ffi.Uint8> socketPath;
  external ffi.Pointer<ffi.Uint8> host;

  @ffi.Uint16()
  external int port;

  external ffi.Pointer<ffi.Uint8> apiKey;

  @ffi.Uint32()
  external int connectTimeoutMs;

  @ffi.Uint32()
  external int requestTimeoutMs;

  @ffi.Uint32()
  external int maxRetries;

  external ffi.Pointer<ffi.NativeFunction<SaErrorCbNative>> errorCb;

  @ffi.Uint64()
  external int errorCtx;
}

/// `sa_client_session_row_t` (sa_client.h:94-100), same discipline: a NULL
/// status means unknown, a NULL goal means absent (NULL fields need no
/// release — the whole array does).
final class SaSessionRowFfi extends ffi.Struct {
  external ffi.Pointer<ffi.Uint8> sid;
  external ffi.Pointer<ffi.Uint8> status;
  external ffi.Pointer<ffi.Uint8> goal;

  @ffi.Uint64()
  external int created;

  @ffi.Size()
  external int depth;
}

// ── the callback typedefs (sa_client.h:82-129) ──────────────────────────────

typedef SaPromptCbNative = ffi.Void Function(ffi.Pointer<ffi.Void> ctx,
    ffi.Uint8 status, ffi.Pointer<ffi.Uint8> sid);
typedef SaPromptCbDart = void Function(
    ffi.Pointer<ffi.Void> ctx, int status, ffi.Pointer<ffi.Uint8> sid);

typedef SaInterruptCbNative = ffi.Void Function(
    ffi.Pointer<ffi.Void> ctx, ffi.Uint8 status);
typedef SaInterruptCbDart = void Function(ffi.Pointer<ffi.Void> ctx, int status);

typedef SaSessionsCbNative = ffi.Void Function(ffi.Pointer<ffi.Void> ctx,
    ffi.Uint8 status, ffi.Pointer<SaSessionRowFfi> rows, ffi.Size nrows);
typedef SaSessionsCbDart = void Function(ffi.Pointer<ffi.Void> ctx, int status,
    ffi.Pointer<SaSessionRowFfi> rows, int nrows);

/// An event record (seq > 0, record_json set) or a MARKER (seq 0, record
/// NULL): sid and record_json are BOTH held — two releases per record, one
/// for a marker's sid only.
typedef SaEventsCbNative = ffi.Void Function(
    ffi.Pointer<ffi.Void> ctx,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Uint64 seq,
    ffi.Uint8 op,
    ffi.Pointer<ffi.Uint8> recordJson);
typedef SaEventsCbDart = void Function(
    ffi.Pointer<ffi.Void> ctx,
    ffi.Pointer<ffi.Uint8> sid,
    int seq,
    int op,
    ffi.Pointer<ffi.Uint8> recordJson);

typedef SaErrorCbNative = ffi.Void Function(ffi.Pointer<ffi.Void> ctx,
    ffi.Uint64 reqId, ffi.Uint8 status, ffi.Pointer<ffi.Uint8> text);
typedef SaErrorCbDart = void Function(ffi.Pointer<ffi.Void> ctx, int reqId,
    int status, ffi.Pointer<ffi.Uint8> text);

/* The CA_CONFIG pair's response callback (sa_client_config_get/set): the
   three strings are the frame-config template's truth or the failure
   delivery's all-NULL shape. The PRESENT members are HELD payloads (one
   release per non-NULL pointer; the absent member rides NULL per the
   wire's "" sentinel decode). */
typedef SaConfigCbNative = ffi.Void Function(ffi.Pointer<ffi.Void> ctx,
    ffi.Uint8 status, ffi.Pointer<ffi.Uint8> baseUrl,
    ffi.Pointer<ffi.Uint8> apiKey, ffi.Pointer<ffi.Uint8> model);
typedef SaConfigCbDart = void Function(ffi.Pointer<ffi.Void> ctx, int status,
    ffi.Pointer<ffi.Uint8> baseUrl, ffi.Pointer<ffi.Uint8> apiKey,
    ffi.Pointer<ffi.Uint8> model);

// ── the C surface's own signatures (sa_client.h:163-264) ────────────────────

typedef SaConfigDefaultFnNative = SaClientConfigFfi Function();
typedef SaConfigDefaultFnDart = SaClientConfigFfi Function();

typedef SaConnectFnNative = ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<SaClientConfigFfi> config);
typedef SaConnectFnDart = ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<SaClientConfigFfi> config);

typedef SaDestroyFnNative = ffi.Void Function(ffi.Pointer<ffi.Void> client);
typedef SaDestroyFnDart = void Function(ffi.Pointer<ffi.Void> client);

typedef SaReleasePayloadFnNative = ffi.Void Function(
    ffi.Pointer<ffi.Void> client, ffi.Pointer<ffi.Void> payload);
typedef SaReleasePayloadFnDart = void Function(
    ffi.Pointer<ffi.Void> client, ffi.Pointer<ffi.Void> payload);

typedef SaPromptFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.Uint8> text,
    ffi.Pointer<ffi.NativeFunction<SaPromptCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef SaPromptFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.Uint8> text,
    ffi.Pointer<ffi.NativeFunction<SaPromptCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

typedef SaInterruptFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.NativeFunction<SaInterruptCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef SaInterruptFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.NativeFunction<SaInterruptCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

typedef SaListSessionsFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.NativeFunction<SaSessionsCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef SaListSessionsFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.NativeFunction<SaSessionsCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

typedef SaSubscribeEventsFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.NativeFunction<SaEventsCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef SaSubscribeEventsFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.NativeFunction<SaEventsCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

typedef SaUnsubscribeEventsFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client);
typedef SaUnsubscribeEventsFnDart = int Function(ffi.Pointer<ffi.Void> client);

typedef _SaConfigGetFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.NativeFunction<SaConfigCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef _SaConfigGetFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.NativeFunction<SaConfigCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

typedef _SaConfigSetFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> baseUrl,
    ffi.Pointer<ffi.Uint8> apiKey,
    ffi.Pointer<ffi.Uint8> model,
    ffi.Pointer<ffi.NativeFunction<SaConfigCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef _SaConfigSetFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> baseUrl,
    ffi.Pointer<ffi.Uint8> apiKey,
    ffi.Pointer<ffi.Uint8> model,
    ffi.Pointer<ffi.NativeFunction<SaConfigCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

typedef SaProbeSizeofFnNative = ffi.Size Function();
typedef SaProbeSizeofFnDart = int Function();

typedef SaProbeOffsetFnNative = ffi.Size Function(ffi.Int32 index);
typedef SaProbeOffsetFnDart = int Function(int index);

// ── the seam ────────────────────────────────────────────────────────────────

/// The connect command's config payload: plain sendable fields, the op side
/// builds the C struct op-side via [SaFfiApi.buildConfig] (the callbacks'
/// native function pointer rides as an address).
final class SaConfigFields {
  const SaConfigFields({
    required this.transport,
    required this.socketPath,
    required this.host,
    required this.port,
    required this.apiKey,
    required this.connectTimeoutMs,
    required this.requestTimeoutMs,
    required this.maxRetries,
    required this.errorCb,
  });

  final int transport;
  final String? socketPath;
  final String? host;
  final int port;
  final String? apiKey;
  final int connectTimeoutMs;
  final int requestTimeoutMs;
  final int maxRetries;

  /// The error callback's native function pointer (0 = none): failures also
  /// route to their op's own callback, the error channel is additional.
  final int errorCb;
}

/// The C's `sa_client_config_default` as plain fields (the struct-by-value
/// return re-mirrored; the live ABI test asserts the round-trip against the
/// real call). The pinned values: transport UNIX, connect 5000 ms, request
/// 10000 ms, retries 0 (the events channel retries forever).
final class SaConfigDefaults {
  const SaConfigDefaults({
    required this.transport,
    required this.connectTimeoutMs,
    required this.requestTimeoutMs,
    required this.maxRetries,
  });

  final int transport;
  final int connectTimeoutMs;
  final int requestTimeoutMs;
  final int maxRetries;
}

/// THE SEAM: everything the binding's wrapper (and its unit tests) touch —
/// the wrapper depends on this abstraction, `SaFfi` implements it against
/// the real .so, a fake implements it in-process (its tests never dlopen
/// anything). Every client/payload/callback pointer value on the surface is
/// an OPAQUE process-wide int — the real implementation casts, a fake
/// invents tokens.
abstract interface class SaFfiApi {
  /// The `.so` path of [resolveSaLibraryPath]'s resolution — null when the
  /// implementation is NOT dlopen-backed (a fake); the default op host
  /// refuses a null path (its op side must load its own library).
  String? get libraryPath;

  /// Opens the library (idempotent). The binding loads MAIN-side first, so
  /// a broken .so fails loud before any op host spawns.
  void load();

  bool get loaded;

  /// The struct-drift tripwire pair (`sa_client_config_ffi_sizeof` /
  /// `_ffi_offset`, the header's ABI companion). A fake has no honest
  /// answer to an independent C truth — these refuse.
  int probeConfigSizeof();
  int probeConfigOffset(int index);

  /// The C's `sa_client_config_default` (loads the library first).
  SaConfigDefaults configDefaults();

  /// Builds + holds the C config (the strings copied native-side); the
  /// caller frees it after connect (the C copies every string at connect)
  /// via [freeConfig]. [fields.errorCb] may be 0 (no error channel).
  int buildConfig(SaConfigFields fields);

  void freeConfig(int config);

  /// Returns the client pointer (an opaque token; 0 = the C refused).
  int connect(int config);

  /// Final teardown (joins the reader thread; every unreleased payload is
  /// reclaimed and every callback pointer turns invalid).
  void destroy(int client);

  /// Releases one held payload exactly once (unknown/double-after-destroy
  /// releases are the C's safe no-op).
  void releasePayload(int client, int payload);

  /// The five ops. Each returns the C's int: 0 = the callback ran (or, for
  /// subscribe, the subscription is live / the failure was delivered
  /// through the callbacks); -1 = the outright refusal that fires NO
  /// callback (`saClientStatusCallRefused` Dart-side). The [serial] rides
  /// through as the callback's ctx (the C passes it through untouched).
  int prompt(int client, int serial, String? sid, String text, int promptCb);
  int interrupt(int client, int serial, String sid, int interruptCb);
  int listSessions(int client, int serial, int sessionsCb);
  int subscribeEvents(int client, int serial, String sid, int eventsCb);
  int unsubscribeEvents(int client);

  /// The CA_CONFIG pair's ops (the daemon's frame-config template; see
  /// sa_client.h). The GET is the all-absent request; the SET rides only
  /// the non-NULL members (an empty string is the wire's absent sentinel —
  /// a member is set to text, never to empty). The callback fires with
  /// status 0 + the template's post-set truth (the present members HELD —
  /// the absent ones NULL), or the all-NULL failure delivery.
  int configGet(int client, int serial, int configCb);
  int configSet(
      int client, int serial, String? baseUrl, String? apiKey, String? model,
      int configCb);
}

// ── the real implementation ─────────────────────────────────────────────────

/// The real [SaFfiApi]: loads `libsa_client.so` via [resolveSaLibraryPath]
/// and binds the eleven exports. The blocking ops are meant for the op
/// isolate's runner (`sa_client_binding.dart`) — NEVER the main isolate.
final class SaFfi implements SaFfiApi {
  SaFfi([String? path]) : _explicitPath = path;

  String? _explicitPath;
  ffi.DynamicLibrary? _lib;

  /// The resolved path, cached; [resolveSaLibraryPath]'s throw passes
  /// through to its callers ([libraryPath] swallows it to answer null).
  String get _resolvedPath => _explicitPath ??= resolveSaLibraryPath();

  @override
  String? get libraryPath {
    try {
      return _resolvedPath;
    } on StateError {
      return null;
    }
  }

  @override
  bool get loaded => _lib != null;

  @override
  void load() => _lib ??= ffi.DynamicLibrary.open(_resolvedPath);

  ffi.DynamicLibrary get _theLib => _lib ??= ffi.DynamicLibrary.open(_resolvedPath);

  // The lookups are lazy so a load() can never be missed.
  late final _configDefaultFn = _theLib
      .lookupFunction<SaConfigDefaultFnNative, SaConfigDefaultFnDart>(
          'sa_client_config_default');
  late final _connectFn =
      _theLib.lookupFunction<SaConnectFnNative, SaConnectFnDart>('sa_client_connect');
  late final _destroyFn =
      _theLib.lookupFunction<SaDestroyFnNative, SaDestroyFnDart>('sa_client_destroy');
  late final _releaseFn = _theLib
      .lookupFunction<SaReleasePayloadFnNative, SaReleasePayloadFnDart>(
          'sa_client_release_payload');
  late final _promptFn =
      _theLib.lookupFunction<SaPromptFnNative, SaPromptFnDart>('sa_client_prompt');
  late final _interruptFn = _theLib
      .lookupFunction<SaInterruptFnNative, SaInterruptFnDart>(
          'sa_client_interrupt');
  late final _listSessionsFn = _theLib
      .lookupFunction<SaListSessionsFnNative, SaListSessionsFnDart>(
          'sa_client_list_sessions');
  late final _subscribeFn = _theLib
      .lookupFunction<SaSubscribeEventsFnNative, SaSubscribeEventsFnDart>(
          'sa_client_subscribe_events');
  late final _unsubscribeFn = _theLib
      .lookupFunction<SaUnsubscribeEventsFnNative, SaUnsubscribeEventsFnDart>(
          'sa_client_unsubscribe_events');
  late final _configGetFn = _theLib
      .lookupFunction<_SaConfigGetFnNative, _SaConfigGetFnDart>(
          'sa_client_config_get');
  late final _configSetFn = _theLib
      .lookupFunction<_SaConfigSetFnNative, _SaConfigSetFnDart>(
          'sa_client_config_set');
  late final _probeSizeofFn = _theLib
      .lookupFunction<SaProbeSizeofFnNative, SaProbeSizeofFnDart>(
          'sa_client_config_ffi_sizeof');
  late final _probeOffsetFn = _theLib
      .lookupFunction<SaProbeOffsetFnNative, SaProbeOffsetFnDart>(
          'sa_client_config_ffi_offset');

  /// The config allocation's strings + the struct itself, per address:
  /// freed by [freeConfig] right after connect (the C copies at connect).
  final Map<int, (ffi.Pointer<ffi.Uint8>, List<ffi.Pointer<ffi.Uint8>>)> _configs =
      <int, (ffi.Pointer<ffi.Uint8>, List<ffi.Pointer<ffi.Uint8>>)>{};

  ffi.Pointer<ffi.Uint8> _copyField(String? text) {
    if (text == null) return ffi.Pointer<ffi.Uint8>.fromAddress(0);
    return SaUtf8.toNative(text);
  }

  @override
  int buildConfig(SaConfigFields fields) {
    final struct = _malloc(ffi.sizeOf<SaClientConfigFfi>())
        .cast<SaClientConfigFfi>();
    final strings = <ffi.Pointer<ffi.Uint8>>[
      _copyField(fields.socketPath),
      _copyField(fields.host),
      _copyField(fields.apiKey),
    ];
    struct.ref
      ..transport = fields.transport
      ..socketPath = strings[0]
      ..host = strings[1]
      ..port = fields.port
      ..apiKey = strings[2]
      ..connectTimeoutMs = fields.connectTimeoutMs
      ..requestTimeoutMs = fields.requestTimeoutMs
      ..maxRetries = fields.maxRetries
      ..errorCb = SaFfi.nativeFn<SaErrorCbNative>(fields.errorCb)
      ..errorCtx = 0;
    _configs[struct.address] = (struct.cast<ffi.Uint8>(), strings);
    return struct.address;
  }

  /// The held native function pointer behind an opaque address.
  static ffi.Pointer<ffi.NativeFunction<T>> nativeFn<T extends Function>(
          int address) =>
      ffi.Pointer<ffi.NativeFunction<T>>.fromAddress(address);

  @override
  void freeConfig(int config) {
    final held = _configs.remove(config);
    if (held == null) return;
    final (struct, strings) = held;
    for (final s in strings) {
      if (s.address != 0) SaUtf8.free(s);
    }
    _free(struct.cast<ffi.Void>());
  }

  @override
  int connect(int config) => _connectFn(
      ffi.Pointer<SaClientConfigFfi>.fromAddress(config)).address;

  @override
  void destroy(int client) =>
      _destroyFn(ffi.Pointer<ffi.Void>.fromAddress(client));

  @override
  void releasePayload(int client, int payload) => _releaseFn(
      ffi.Pointer<ffi.Void>.fromAddress(client),
      ffi.Pointer<ffi.Void>.fromAddress(payload));

  @override
  int prompt(int client, int serial, String? sid, String text, int promptCb) {
    final sidP = _copyField(sid);
    final textP = SaUtf8.toNative(text);
    try {
      return _promptFn(
        ffi.Pointer<ffi.Void>.fromAddress(client),
        sidP,
        textP,
        SaFfi.nativeFn<SaPromptCbNative>(promptCb),
        ffi.Pointer<ffi.Void>.fromAddress(serial),
      );
    } finally {
      // The C dups both strings at entry (sa_client.c's sa_client_prompt) —
      // the caller's buffers are consumed by the time the call returns.
      SaUtf8.free(textP);
      if (sidP.address != 0) SaUtf8.free(sidP);
    }
  }

  @override
  int interrupt(int client, int serial, String sid, int interruptCb) {
    final sidP = SaUtf8.toNative(sid);
    try {
      return _interruptFn(
        ffi.Pointer<ffi.Void>.fromAddress(client),
        sidP,
        SaFfi.nativeFn<SaInterruptCbNative>(interruptCb),
        ffi.Pointer<ffi.Void>.fromAddress(serial),
      );
    } finally {
      SaUtf8.free(sidP);
    }
  }

  @override
  int listSessions(int client, int serial, int sessionsCb) => _listSessionsFn(
      ffi.Pointer<ffi.Void>.fromAddress(client),
      SaFfi.nativeFn<SaSessionsCbNative>(sessionsCb),
      ffi.Pointer<ffi.Void>.fromAddress(serial));

  @override
  int subscribeEvents(int client, int serial, String sid, int eventsCb) {
    final sidP = SaUtf8.toNative(sid);
    try {
      return _subscribeFn(
        ffi.Pointer<ffi.Void>.fromAddress(client),
        sidP,
        SaFfi.nativeFn<SaEventsCbNative>(eventsCb),
        ffi.Pointer<ffi.Void>.fromAddress(serial),
      );
    } finally {
      SaUtf8.free(sidP);
    }
  }

  @override
  int unsubscribeEvents(int client) =>
      _unsubscribeFn(ffi.Pointer<ffi.Void>.fromAddress(client));

  @override
  int configGet(int client, int serial, int configCb) => _configGetFn(
      ffi.Pointer<ffi.Void>.fromAddress(client),
      SaFfi.nativeFn<SaConfigCbNative>(configCb),
      ffi.Pointer<ffi.Void>.fromAddress(serial));

  @override
  int configSet(int client, int serial, String? baseUrl, String? apiKey,
      String? model, int configCb) {
    final baseP = _copyField(baseUrl);
    final keyP = _copyField(apiKey);
    final modelP = _copyField(model);
    try {
      return _configSetFn(
        ffi.Pointer<ffi.Void>.fromAddress(client),
        baseP,
        keyP,
        modelP,
        SaFfi.nativeFn<SaConfigCbNative>(configCb),
        ffi.Pointer<ffi.Void>.fromAddress(serial),
      );
    } finally {
      // The C dups every string at entry (sa_client_config_set) — the
      // caller's buffers are consumed by the time the call returns.
      for (final p in <ffi.Pointer<ffi.Uint8>>[baseP, keyP, modelP]) {
        if (p.address != 0) SaUtf8.free(p);
      }
    }
  }

  @override
  int probeConfigSizeof() => _probeSizeofFn();

  @override
  int probeConfigOffset(int index) => _probeOffsetFn(index);

  @override
  SaConfigDefaults configDefaults() {
    final c = _configDefaultFn();
    return SaConfigDefaults(
      transport: c.transport,
      connectTimeoutMs: c.connectTimeoutMs,
      requestTimeoutMs: c.requestTimeoutMs,
      maxRetries: c.maxRetries,
    );
  }
}