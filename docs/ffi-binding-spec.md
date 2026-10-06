# The Pondr ↔ Runtime FFI Binding — Spec

**Date:** 2026-10-05
**Status:** owner-agreed shape settled in session (six decisions; the direct-FFI approach, no
shim). No open questions — every detail is pinned below.
**Goal:** the app's chat/sessions/settings views talk to REAL agent frames: "Pondr is the
first client" (msg 4218) becomes true.
**Shape (two-process):** the Flutter app links ONLY `libsa_client.so` and speaks the
client-API wire to a runtime DAEMON (`frame-demo serve`) that the app spawns + supervises.
CPython, the frames, the store live in the daemon; the app stays lean and crash-isolated.
**Repos touched:** SecretAgent (TWO small C items: the shared-lib target + the CA_CONFIG
wire pair) and pondr (everything else). The one-way arrow holds: pondr → agent-runtime.

## Decisions carried in (settled with the owner, 2026-10-05)

| Decision | Value | Authority |
|---|---|---|
| Process shape | TWO-PROCESS: the app = a sa_client consumer; the daemon = `frame-demo serve` (the co-process the app spawns + supervises) | owner |
| Settings reach the frames | a NEW **CA_CONFIG wire pair** (get/set the daemon's frame-config template — the session server holds it; new frames adopt; RUNNING frames keep theirs) | owner |
| Daemon lifecycle | the app SPAWNS + SUPERVISES (spawn when none is running; reconnect/respawn via sa_client's backoff; SIGTERM on app exit); manual serve still works (the app connects) | owner |
| FFI layer | DIRECT dart:ffi on sa_client's C API — NO C shim; callbacks via `NativeCallable.listener` (they fire on sa_client's reader thread); the payload-lifetime rule mirrored Dart-side | owner |
| Interface bindings | `SessionsService` + `ChatService` + `SettingsService` = REAL FFI implementations behind ONE binding swap (a provider flag); `SubconsciousService` STAYS MOCK (the engine's serving surface is a later slice) | owner |
| Out of scope | the engine's serving surface; the audit-view (cells/tree rendering); session DELETION on the daemon (recorded: deleting hides the row; the frame stays); a config file / per-prompt models; mobile targets' verification (Linux first, as always) | owner |

## 1. The C side (SecretAgent)

1. **`add_library(sa_client SHARED ...)`** in SecretAgent's root CMakeLists — exactly the client
   stack the FFI needs: `ClientLibs/c/sa_client.c` + `ClientApi/{client_api_wire, handlers}` +
   `ClientApi/{Unix,Tcp}` + `Network/stream_framer` + the needed closure (Platform/Util/Actor/
   Buffer/RefCounter). Compiles under the existing WDB+streams gates (the static target's
   discipline). `sa_client.h` itself UNCHANGED (already flat, versioned, FFI-ready).
2. **The CA_CONFIG pair** in the client API (the client-api spec's vocabulary family):
   `CA_CONFIG_REQUEST 14` / `CA_CONFIG_RESPONSE 15` (the pairing assert joins; adjacent to the
   AUTH pair's numbers). Get: `{req_id}` → the response carries the session server's
   frame-config template `{base_url, api_key, model}` (the api_key field follows the auth
   pair's by-length scrub discipline in its destroy). Set: `{req_id, base_url?, api_key?, tag?}`
   — absent fields unchanged; the template mutates for NEW frames; running frames untouched.
   The server holds the template already (the handlers' frame_create source) — the handler
   case + the wire's encode/decode/destroy + both suites' tests.

## 2. The Dart side (pondr/flutter)

**The FFI layer** (`lib/ffi/sa_ffi.dart` + `lib/ffi/sa_client_binding.dart`):
- `sa_ffi.dart`: the ABI mirrors — the config struct's layout, the five callback typedefs, the
  function lookups via `DynamicLibrary.open` (the path: `$APP/lib/libsa_client.so`, overridable)
  — resolved ONCE.
- `SaClientNative`: a `NativeCallable.listener` PER callback (C posts from the reader thread →
  the main isolate's queue); the held-payload registry (each delivered pointer stashed;
  `sa_client_release_payload` on release/dispose; double-release safe); the surface:
  `connect(config, callbacks)` / `prompt(sid, text)` / `interrupt(sid)` / `listSessions()` /
  `subscribeEvents(sid)` / `unsubscribeEvents(sid)` / `disconnect()` / `dispose()` —
  Future-returning; the request timeout rides the config.

**The REAL service implementations** (`lib/data/daemon/`):
- `FfiSessionsService`: `list()` = CA_SESSIONS; `create()` DEFERRED (a session exists at the
  first send — the daemon's spawn); active/select/delete = APP-LOCAL bookkeeping (delete
  hides the row; the daemon's frame stays — recorded).
- `FfiChatService.send`: PROMPT (no sid = create → the response's sid IS the session's id;
  the steer otherwise) — then the per-session EVENTS subscription DRIVES the message list:
  the daemon's records are the source of truth (the app does NOT double-append the steer;
  `msg.append` role=user → the user bubble; role=assistant → `done(content)`; the typing
  event rides the send's start; cell/lifecycle records pass as system events the chat ignores
  (the audit's view = a recorded follow-on)).
- `FfiSettingsService`: the LOCAL provider/model list persists app-side (a JSON file under
  the app's config dir); the DAEMON's adopted template = the SELECTED provider's
  base_url/key/model — applied via CA_CONFIG set on selection (and get() at startup for the
  truth display).

**The supervision** (`lib/daemon/supervisor.dart`): connect-fail → spawn
`frame-demo serve --socket-path <cache>/pondr.sock --model <the selected tag>` (the binary
path/args configurable — the settings' runtime section); wait-for-socket → connect; a dead
daemon = respawn via sa_client's reconnect cadence; app-exit = SIGTERM + await.
`bindings.dart` gains the mode flag (an AppSettings/define) choosing mock|daemon.

## 3. Testing

| Suite | Adds |
|---|---|
| SecretAgent `test_client_api_wire`/`handlers` | the CA_CONFIG pair's round-trips + the get/set's template mutations + the scrub on the key |
| SecretAgent CMake | `libsa_client.so` builds (a configure/build smoke; the OFF gate keeps refusing it loud) |
| pondr `test/ffi/` | the ABI mirrors' shapes (a fake-DynamicLibrary seam for the pure unit tests: the payload registry, the listener wiring, the released/never-released pointers); the service impls against a MOCKED SaClientNative (the stream/record mapping's semantics: the no-double-append rule, the user/assistant record routing, the create's sid flow); the supervision's spawn/respawn (a real child process against a fake long-running binary) |
| pondr integration (live-gate style, opt-in) | the REAL daemon: boot frame-demo serve on a temp store; the FFI services against it; a real model reply (or a scripted daemon?) — the live gate's evidence = the ctest/flutter-test run + a manual run |

**The verification bar**: `flutter analyze` zero + `flutter test` green (pondr);
`setarch -R ctest --test-dir cmake-build-debug --output-on-failure` green (SecretAgent);
ASan on the C-side's new shared-lib paths.

## Out of scope / recorded follow-ons

- The engine's serving surface (SubconsciousService's real data).
- The audit view (cells/tree rendering as a chat-adjacent surface); session deletion over the
  wire; mobile targets' verification; the persona/escalation layers (the NEXT runtime slice —
  its ask surface now has a real client to ask THROUGH).
## The binding's state (the Task 6 landing, 2026-10-05)

The slice LANDED: the daemon's supervised boot + the record-folded chat + the
CA_CONFIG adoption run live under `--dart-define=pondr.mode=daemon`
(`docs/flutter-app-run-evidence.md`'s BINDING section carries the run's
evidence). Landed beyond this spec's original surface, all gate-caught:

- **The create-prompt's user record** (SecretAgent `50e7894`): a create-path
  prompt commits its text as the frame's `msg.append` {role: user} BEFORE
  `frame_start` — without it the events channel never carried the first
  send's user bubble (the chat contract's §2 mapping broke for every
  session's FIRST send). Pinned by
  `TestPromptCreateCommitsTheUserMessageRecord`.
- **The dev's one command** `flutter/daemon/prepare.sh`: the sa_client +
  frame-demo build, the copies at the loader's candidates
  (`flutter/libsa_client.so`, the bundles' `lib/`), `frame-demo` at
  `flutter/daemon/frame-demo`.
- **The recorded follow-ons** (out of scope, unchanged): the engine's
  serving surface; the audit view — which also absorbs today's shape note:
  a no-model frame's `control {kind: model-missing}` record rides the
  events channel while the chat's send stays open (no error, no reply) —
  the control record's UI mapping lands with the audit surface; the idle
  daemon's ~2.2-core spin (recorded in the run evidence, unrepaired).
