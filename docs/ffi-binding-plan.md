# The FFI Binding Implementation Plan (Pondr ↔ Runtime)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Pondr chat app talks to REAL agent frames: `libsa_client.so` via dart:ffi, a supervised `frame-demo serve` daemon, one binding swap in `bindings.dart`.

**Architecture:** Two-process (the app = sa_client consumer; the daemon = CPython + frames + store). The C side (SecretAgent): the CA_CONFIG pair + the SHARED lib target. The Dart side (pondr): a thin FFI layer (`lib/ffi/`), the three REAL service implementations (`lib/data/daemon/`), the supervisor (`lib/daemon/`) — all behind the ONE binding swap.

**Tech Stack:** dart:ffi (NativeCallable.listener for the C callbacks; background isolates for the BLOCKING ops — calling sa_client's blocking ops on the main isolate would freeze the UI; pinned below); C11/libcbor (SecretAgent side).

**Spec:** `pondr/docs/ffi-binding-spec.md` — READ FIRST. **Repos:** Tasks 1-2 commit in SECRETAGENT; Tasks 3-6 commit in PONDR (its main; "flutter:"-prefixed conventional commits; no TODOs; de-wonk before completing; pondr/CLAUDE.md's bar). **Standing gates:** SecretAgent: `setarch -R ctest --test-dir cmake-build-debug --output-on-failure` (313/313 at 0820df4) + ASan; pondr: `cd pondr/flutter && export PATH="$HOME/flutter/bin:$PATH" && flutter analyze` (zero) + `flutter test` (119/119).

---

### Task 1: The CA_CONFIG wire pair (SecretAgent)

**Files:**
- Modify: `src/ClientApi/client_api_wire.h:18-60` (+ the payload types ~:100-150)
- Modify: `src/ClientApi/client_api_wire.c` (encode/decode/destroy)
- Modify: `src/ClientApi/handlers.h` / `handlers.c` (the handler case)
- Test: `test/test_client_api_wire.cpp`, `test/test_client_api_handlers.cpp`

- [ ] **Step 1: The failing tests** — in test_client_api_wire.cpp:

```cpp
TEST(TestClientApiWire, TestConfigRoundTrips) {
  /* GET: {req_id 9} → the response carries base_url/api_key/model strings
     (absent = the "" sentinel, per the wire's bounded-string discipline:
     CA_WIRE_CONFIG_* bounds — TEXT 512 for base_url, KEY 256 for api_key
     mirroring CA_WIRE_KEY_MAX, TAG 128 for the model tag). */
  /* SET: {req_id 9, base_url "http://127.0.0.1:11434", api_key "", tag
     "gemma4:latest"} = the absent-fields-unchanged shape ("" = absent
     everywhere in CONFIG's set) → decodes equal. */
}
```
  In test_client_api_handlers.cpp:
```cpp
TEST(TestClientApiHandlers, TestConfigGetAndSetTheTemplate) {
  /* GET → the response {status 0, base_url "", api_key "", model ""}
     (the daemon's template defaults empty); SET {tag "gemma4"} → status 0,
     GET → the tag rides; SET {base_url, api_key, tag} all → GET carries all;
     a RUNNING frame keeps its model (create a frame with the template's
     ORIGINAL config, then SET, then check the frame's config unchanged —
     via the frame's recorded model... read what the server holds and PIN
     the new-frames-only rule in the test). */
}
```
- [ ] **Step 2: The wire additions** — `CA_CONFIG_REQUEST 14` / `CA_CONFIG_RESPONSE 15` (BEFORE the ERROR's 11? NOTE: the enum's free numbers — CA_ERROR=11, the pair = 14/15 keeps the AUTH 12/13 family's adjacency discipline BUT 12/13 are AUTH and 11 is ERROR — 14/15 are the NEXT free pair (no collision); the pairing assert joins + the payload structs (the get's response carries the strings; the set's request the optional fields); the encode/decode/destroy + the SCRUB: the config response's api_key scrubs by length (the auth pair's idiom).
- [ ] **Step 3: The handler case** — the daemon template lives in the session server (its frame_config_t member); GET composes the response; SET mutates (absent fields unchanged — the "" sentinel reads as absent per the config wire's rule) + answers; the test pins the new-frames-adopt rule.
- [ ] **Step 4:** gates green (313 + n); valgrind/ASan on the new paths. **Commit (SecretAgent):** `git add -A src/ClientApi test && git commit -m "feat: the CA_CONFIG wire pair — the daemon's frame-config template travels the wire"`.

---

### Task 2: The shared lib target (SecretAgent)

**Files:**
- Modify: `CMakeLists.txt` (next to the static `secretagent` at :210)

- [ ] **Step 1: The target:**

```cmake
# The FFI client stack as ONE shared object: the C client + its wire/framer/
# transport closure — Pondr's dart:ffi loads this (the client-api spec §1).
# The gates mirror the static target's (a WDB/streams-less build has no
# client to link).
add_library(sa_client SHARED
  src/ClientLibs/c/sa_client.c
  src/ClientApi/client_api_wire.c
  src/ClientApi/handlers.c
  src/ClientApi/Unix/unix_transport.c src/ClientApi/Unix/unix_connection.c
  src/ClientApi/Tcp/tcp_transport.c src/ClientApi/Tcp/tcp_connection.c
  src/Network/stream_framer.c
)
target_compile_definitions(sa_client PRIVATE ...)  # the SAME gate defines the static target gets — MIRROR ITS target_compile_definitions/include-dirs blocks (read :210-270; share via a helper or repeat)
# Its link closure: Platform/Util/Actor/Buffer/RefCounter — whatever the static
# target's sources carry; the shared lib is SELF-CONTAINED (no libsecretagent.so
# dependency) so the consumer ships ONE file: list the exact .c files.
```
  ADAPT: read the static target's compile block and MIRROR (defines/links/include dirs); ensure the client stack's full closure of .c files is explicit (a self-contained .so — grep each client-stack file's includes to name the exact closure; the expected members: platform_*.c (socket/thread/time/process), allocator/log/atomic? (header-only)/json?/vec/bcrypt/refcounter/message_queue/actor/deque/pool/scheduler + the loop-thread... REPORT the closure you found; every member must link).
- [ ] **Step 2:** build smoke: `setarch -R cmake --build cmake-build-debug -j --target sa_client` → `libsa_client.so` exists; `nm -D --defined-only | grep sa_client_connect` exported. The OFF gate: the OFF configure refuses (its regex excludes the paths — check the OFF regex's current shape; if the OFF regex excludes `src/(Streams|...)` but the ClientApi sources were handled via the empty-TU idiom, the SHARED target must only add sources under the GATES TOO (a conditional target: `if(SA_ENABLE_WDB AND SA_ENABLE_STREAMS)`).
- [ ] **Step 3: Commit:** `git add CMakeLists.txt && git commit -m "build: the self-contained libsa_client shared target (the FFI consumer's one file)"`.

---

### Task 3: The FFI layer (pondr)

**Files:**
- Create: `lib/ffi/sa_ffi.dart` (the ABI), `lib/ffi/sa_client_binding.dart` (the wrapper)
- Test: `test/ffi/ffi_test.dart`

- [ ] **Step 1: The ABI** (`sa_ffi.dart`, a final class `SaFfi` with the lookups): the .so path resolution (`String resolveLibPath()` → `$Platform.resolvedExecutable`'s dir + `/lib/libsa_client.so`, overridable); the struct mirrors (ffi.Struct subclasses for `sa_client_config_t` — the field ORDER/LAYOUT exactly per sa_client.h:134-161 (enums = c int/4B; the uint16 port + padding? use @Uint16() and let dart:ffi compute; VERIFY the C struct's real layout with a C probe test? (a tiny C sizeof/offsetof check compiled as part of Task 2's smoke? — do it as a DART test: read the offsets from the .so via a C helper... SIMPLER: the C ABI's struct-passing risk is real; MITIGATE: the binding passes a CONFIG via a SMALL C-ALLOCATED mirror: read the offsets once at load time from... NO — dart:ffi structs with matched field order work; pin: `external class` with @Uint16/@Uint32/@Uint64/Pointer per the header; a Dart test asserts `sizeOf<SaClientConfigFfi>()` against a KNOWN constant (write the C-side's sizeof ONCE in Task 2 as an EXPORTED helper `sa_client_config_ffi_sizeof()` + `sa_client_config_ffi_layout_probe(int index) → offsetof` — a small Honest-to-ABI probe pair in sa_client.h guarded `#ifdef SA_FFI_PROBE`? NO — unguarded, tiny, documented as the FFI ABI's companion; the Dart test calls and asserts each offset — THE ABI NEVER DRIFTS SILENTLY).
- [ ] **Step 2: The wrapper** (`SaClientNative`):
  - callbacks: `NativeCallable.listener` per type (prompt/interrupt/sessions/events/error); the C side fires from its reader thread (the request ops fire on the CALLER thread — the caller = our op isolate!) → listener posts to our main isolate's queue: exactly right for both.
  - THE OP ISOLATE (the spec's pinned shape): the blocking ops CANNOT run on the main isolate. Shape: a long-lived BACKGROUND ISLATE (`_opIsolate`) owning a command queue (port receive) + the client's raw `Pointer` + its OWN pending map (the compleaters keyed by op serial): `connect(config)` = spawn the op isolate FIRST (it allocates + connects + stores the pointer, answers ok/dyn-lib error), then each op posts `{op, args, serial}` → the isolate calls the C op WITH the ISOLATE-LOCAL NativeCallable.receiver? — WAIT: the op's CALLBACK: sa_client_request fires the callback — if the callback is a NativeCallable.listener's nativeFunction (created on the MAIN isolate), it works from ANY thread — SO the op isolate calls the C op with the MAIN-isolate listener's function pointer; the op isolate's command handler AWAITS a RawReceivePort for the op's local completion? NO SIMPLER: the op isolate calls sa_client_prompt (BLOCKING — inside the op isolate, blocking is FINE — the UI never sees it), and the CALLER-thread callback is a NativeCallable.listener created by the MAIN isolate wrapping the Dart completion... BUT the listener posts async — the BLOCKING C call returns after firing; the listener's Dart-side landing is async. So the op's Future completes when the LISTENER's callback lands on the main isolate. THE FLOW: op-isolate calls C (blocking); the callback (a listener pointer) fires possibly before/after the op isolate returns its ok; the MAIN isolate's listener handler matches the serial → completes the Future → the wrapper awaits. WORKS. (The C contract: the callback fires before the call returns — listener delivery is queued — fine.)
  - the payload registry **(THE DISPOSE-CHAIN PROTOCOL — the decided shape; a subtle protocol,
    follow it exactly)**: sa_client HOLDS every payload until `release_payload` OR destroy —
    so a callback's pointer is legal for the main isolate to deref LATER; the danger is only
    the destroy freeing a payload whose listener post hasn't been read yet. THE PROTOCOL:
    (a) every payload-carrying callback = a `NativeCallable.listener` (main isolate); its
    Dart closure FIRST derefs + copies the strings (toDartString), THEN posts a
    `{release, ptr}` command to the op isolate (the op isolate owns the client handle and
    calls `sa_client_release_payload`), THEN resolves ITS LINK in the binding's ordered
    per-event completer chain;
    (b) listener posts arrive on the main isolate FIFO, so every post queued before the
    chain's tail resolves is read before the tail resolves — no unread pointer survives the
    tail;
    (c) `dispose()` = stop issuing ops → `unsubscribe_events` (awaited up to the marker) →
    await the CHAIN's TAIL (all reads done, all releases commanded) → the `destroy` command →
    the ack → close the listeners. The C side's own reclaim-at-destroy covers any payload
    whose release raced; the Dart side never touches a pointer past the tail's resolution.

```dart
// The dispose chain (THE RACE-CLOSURE PROTOCOL — the binding's contract):
// each events/error listener fires on the main isolate FIFO; the handler
// copies its payloads FIRST (toDartString), records the copy's facts, THEN
// resolves its link (a per-event completer in an ordered chain); the
// release command posts back to the op isolate from the handler AFTER the
// copy. dispose(): unsubscribe (an awaited marker) → the chain's tail
// awaited → the destroy command → the ack → close the listeners.
```
- [ ] **Step 3: tests** (a fake-ABI seam: `SaFfi`'s indirection — the wrapper takes an ABSTRACTION (`SaFfiApi` interface: the function bindings' methods) so unit tests inject a FAKE (no .so): the op-isolate + payload chain + the listener lifecycle's tests IN-PROC; the REAL load path: an integration test SKIP-unless the .so exists (an env/`File` check + a timeout — the live-gate idiom).
- [ ] **Step 4: Commit:** `git -C pondr add flutter/lib flutter/test && git commit -m "flutter: the sa_client FFI layer (native callbacks + the op isolate + the dispose chain)"`.

---

### Task 4: The daemon supervisor (pondr)

**Files:**
- Create: `lib/daemon/supervisor.dart`
- Test: `test/daemon/supervisor_test.dart`

- [ ] **Step 1:** the supervisor per the spec: `Supervisor.start({required socketPath, required daemonBinary, required modelTag, onExit})` → already-serving check (a quick sa_client probe connection: the FAKE-ABI's probe or a raw socket existence check — the honest check: try connect + a CA_SESSIONS request? the client's probe = the FFI's listSessions with a short timeout); none → `Process.start(daemonBinary, [serve, --socket-path, socketPath, --model, modelTag])` (stdout/stderr PIPE-logged? log a failure text via log util) + wait-for-the-socket (the file's appearance poll ≤ 5 s) + connect; a DEAD daemon (the client's DISCONNECTED + the onDaemonGone) → respawn (the backoff 1s→8s; max attempts config); `stop()` = the process SIGTERM + await, else SIGKILL; the dispose: app-exit hook.
- [ ] **Step 2: tests:** with a FAKE long-running child (a dart script/binary stub: `#!/usr/bin/env dart` script that touches the socket-path file + sleeps — WAIT the real wait-for-socket checks the file existence — the fake touches it: a REAL connect then fails... the supervisor's wait = the file + the client's connect; the test's fake = a tiny dart child that creates a LISTENING unix socket (dart:io ServerSocket bind on an abstract/domain path — dart supports domain sockets) so the supervisor's CONNECT succeeds end-to-end — the honest test); respawn (the fake dies after N seconds → respawned); the SIGTERM's clean stop. **Commit:** `flutter: the daemon supervisor (spawn, the socket-wait, the respawn, the stop)`.

---

### Task 5: The three FFI service implementations (pondr)

**Files:**
- Create: `lib/data/daemon/ffi_sessions_service.dart`, `lib/data/daemon/ffi_chat_service.dart`, `lib/data/daemon/ffi_settings_service.dart`
- Modify: `lib/data/bindings.dart` (the mode flag: a `--dart-define=pondr.mode=mock|daemon` + the local config file's shape)
- Create: `lib/daemon/app_config_store.dart` (the settings' app-local JSON persistence — an app-config-dir file)
- Test: `test/daemon/ffi_services_test.dart`

- [ ] **Step 1:** the three impls per the spec (READ the landed mock services + the interfaces FIRST — the behavior contracts they already pin (the session capture rule, the newest-first order rule, the selectedModelKey `<providerId>::<modelId>` format)):
  - `FfiSessionsService`: list() = `native.listSessions()` → the rows → the models; create() deferred/documented; active/select/delete = LOCAL bookkeeping (delete hides the row).
  - `FfiChatService.send`: the send's rule (the send-time session capture): the steer's ECHO — THE DAEMON'S events stream carries the user's msg.append → THE APP maps the stream as the message-source-of-truth: the user's local bubble = the RECORD's arrival (pin: no double-append; the send does NOT locally append — it waits the record's arrival: the UI's send→bubble latency = the daemon's store commit + the event flight — µs-ms, fine); the assistant record → done(content); the typing event on the send's start; the cell/lifecycle records → the ChatEvent.system (ignored by the chat view).
  - `FfiSettingsService`: the LOCAL providers (the app's JSON file) + the daemon's ADOPTION: the picker's selection → CA_CONFIG set (base_url/key/model) — get() at startup for the truth; the selectedModelKey format kept.
- [ ] **Step 2: tests:** against a MOCKED SaClientNative (the seam from Task 3's abstraction): the mapping rules' tests (the no-double-append pin, the record routing, the create's sid flow, the config adoption + the startup get); the bindings' swap (mode mock|daemon via the define/provider flag). **Commit:** `flutter: the FFI services behind the binding swap (chat/sessions/settings real)`.

---

### Task 6: The integration + the live gate + the run

- [ ] **Step 1:** the C side: `setarch -R ctest ...` green incl. the CA_CONFIG tests + `cmake --build cmake-build-debug --target sa_client` + copy the .so into `pondr/flutter/linux/app-bundle-placeholder`? NO — the loader's path: the app loads `$ExeDir/lib/libsa_client.so` — the Linux bundle's `flutter build linux` emits `build/linux/x64/debug/bundle/`; the DEV run's lib path = the flutter run's bundle? SIMPLIFY: the FFI loader tries, in order: (1) `SA_LIBRARY_PATH` define; (2) `$ExeDir/lib/libsa_client.so`; (3) `~/libsa_client.so`?? — pin: (1) the define; (2) the exe-dir/lib; documented. The dev flow: a `flutter/daemon/prepare.sh` script: builds SecretAgent's sa_client target + frame-demo + COPIES both into the flutter build's bundle dir (documented in the plan; the script = the dev's one command).
- [ ] **Step 2:** the INTEGRATION live test: `flutter/daemon/prepare.sh` then the daemon-boot test: the REAL frame-demo serve (a temp store; a REAL or scripted model — Ollama if the live gate's model is up; else the serve runs WITH a FAKE?? the daemon needs a model backend for REAL chat — the live gate = opt-in (an env flag) with Ollama gemma4:latest as the refine/live slices did): the FFI services end-to-end: create a session → the config adoption → send → the REAL reply → the chat view renders it. (The scripted-NO-MODEL variant: the daemon's frame fails loud → the app's error surface — test THAT shape with no model configured = the non-live test.)
- [ ] **Step 3:** `flutter analyze` 0 + `flutter test` green (119 + n); the run: `flutter run --dart-define=pondr.mode=daemon` + the manual evidence (screens; the supervisor's spawn evidence in the log). The de-wonk + the plan checkboxes. **Commit:** `flutter: the FFI binding live — the daemon supervised, the chat real` + the docs note (the spec's NEXT-section update: the binding's state).

The final report: the evidence + the recorded divergences + the NEXT seams (the engine's serving surface; the audit view; the persona/escalation slice ON THE RUNTIME side).