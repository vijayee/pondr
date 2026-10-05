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
  - the payload registry: on each callback's POINTER args: UTF-8 copy + `release_payload(client, ptr)` immediately (the binding OWNS the copy timing — the callback's data is copied BEFORE the listener's post? NO — the listener fires LATER on the main isolate; the C payload's pointer stays valid until release... the pointer crosses into the isolate's closure as an int; a LATER release via a posted message back to the op isolate = ASYNC complexity. SIMPLER HONEST SHAPE: the EVENTS/ERROR callbacks (the only ones with payloads on the reader thread) → the listener receives (ctx/sid/seq/op/record_json) as RAW VALUES?? — NativeCallable.listener callbacks can only receive VALUE-TYPE parameters? NO: listener callbacks CAN pass POINTERS (they're ints) — the pointer arrives as int; the UTF-8 COPY must run SOMEWHERE with access to the pointer: a listener callback's Dart handler CAN'T deref a foreign pointer?? IT CAN — Pointer.fromAddress + .toDartString() works from ANY isolate (memory reads are not isolate-restricted; the memory is alive UNTIL release). SO: the events listener (created on the MAIN isolate but fired on the reader thread → posts INTO the main isolate) — WAIT: NativeCallable.listener DELAYS delivery: by the time the Dart closure runs, the C payload could be... valid UNTIL release/destroy. THE DANGER: reader thread fires → the payload pointer valid; the listener POSTS (async); the main isolate handles LATER; release then = the pointer's data read AFTER the post?? The reader thread may REUSE... the payload is held by the C client until released — SO: read the strings in the LISTENER'S DART HANDLER on the main isolate, then POST BACK a release command to the op isolate (the op isolate holds the client pointer) OR keep a dedicated tiny RELEASE-only isolate... CLEANEST: give EVERY payload-callback a NativeCallable.listener whose Dart closure (main isolate) reads ALL strings via toDartString + then sends a `{release, serial of ptr}` to the op isolate which calls sa_client_release_payload(client, Pointer.fromAddress). The payload stays held until the release command lands. THE DESTROY must be preceded by a release-all sync... the C contract: unreleased payloads = reclaimed at destroy — SO the dispose flow: command the op isolate: destroy (the listener callbacks still in flight would fire on freed memory! the C destroy's "callbacks already delivered" rule — a listener pending in the main isolate while C destroy runs = UAF. THE DISPOSE PROTOCOL: quiesce first: no pending ops (await), then a final barrier: the op isolate's destroy command runs LAST-in-queue; any listener posted BEFORE the destroy is delivered BEFORE... NOT GUARANTEED (the listener's delivery order vs the op-isolate's message processing is async). SAFE ORDER: (1) the wrapper's dispose: stop issuing ops; (2) wait for a SENTINEL ack: post `destroy` to the op isolate; the op isolate calls sa_client_destroy and replies ack; (3) the main isolate, ON THE ACK, closes the listeners — any listener fired BEFORE the destroy was delivered before... STILL RACY in theory (a callback fired before destroy but its listener-post scheduled after). PRACTICAL HONESTY: the events/error callbacks fired pre-destroy with held payloads: their listeners' posts carry (ptr, data) — the main isolate reading FREED memory = the race. FIX THE SHAPE: the events listener's DART closure (main isolate) reads toDartString at that moment... the read may happen after the destroy = UAF. THE REAL FIX: read the strings INSIDE THE OP ISOLATE before the post: for the reader-thread callbacks the op isolate CAN'T intercept (the C fires the listener pointer directly from the reader thread) — SO: wrap the reader-thread-facing events/error callbacks in TINY C TRAMPOLINES? NO C. ALTERNATIVE: the listener fires on the main isolate LATE — instead of listener, use NativeCallable.ISOLATE-LOCAL with a dedicated thread?? Dart FFI: `@Native` callbacks from arbitrary threads: ONLY listener works cross-thread. PRAGMATIC RESOLUTION: the binding's events listener: the FIRST thing its Dart closure does on the main isolate: copy the strings — then send a release. The destroy race window: dispose() (1) unsubscribes (the events flow stops); (2) sends release-all + destroy to the op isolate + awaits its ack (all the reader-thread events delivered BEFORE the unsubscribe's terminal marker are... after unsubscription NO MORE events fire: the LAST event's listener post may still be in flight; the ack from destroy (which runs AFTER the unsubscribe's terminal marker) — the events listener's posts are FIFO per isolate — the terminal marker arrives BEFORE all... its listener post could ALSO be... the terminal marker = the last event → the ack arrives after the op-isolate processed destroy → the marker's listener-post lands BEFORE the main isolate sees the ack (FIFO within the isolate's own queue: listener posts queue on the MAIN isolate; the ack is ALSO a main-isolate queue message sent later → FIFO holds: all event-listener posts precede the ack. THE READS happened at the listener's handler (main isolate, FIFO-ordered) → all reads precede the ack → the destroy (which frees the payloads) happens in the op isolate possibly BEFORE the main isolate's reads?! THE C-SIDE's destroy frees the payloads — the pointer's memory frees the MOMENT the op isolate's destroy runs — but the main isolate's read of an ALREADY-POSTED event listener may happen AFTER. RACE REAL. THE RESOLUTION: BEFORE the destroy, the op isolate must WAIT until all the posted events have been READ... no mechanism. FINAL PRAGMATIC SHAPE (pin in the design): dispose = (1) unsubscribe + await the terminal marker (via an awaitable); (2) DRAIN-BARRIER: send a probe to the op isolate ("pending posts settled?") hmm... SIMPLEST TRUE FIX: the op isolate NEVER fires events via a NATIVE pointer post — instead, give sa_client's events callback a listener whose FUNCTION reads AND RELEASES IN C?? Can't. → ALTERNATIVE THAT ACTUALLY WORKS: the events callback function pointer = NativeCallable.listener built in the MAIN isolate; but the C fires it from the READER thread — NativeCallable.listener POSTS (queues the call + copies ARGUMENTS by value — pointers ARE values; the DATA isn't copied) — hmm. RESOLUTION: accept the tiny destroy race BUT make dispose = destroy-WITH-DELAY: after the terminal marker's ack (all events whose posts are queued BEFORE the ack = read before the ack per FIFO — the reads happen in the handler BEFORE... the handler runs; the reads complete in the handler; THEN the ack processed. The handler's toDartString reads happen when the main isolate processes the event post — the C payload is freed only at the op isolate's destroy — which happens after the ack send — the events' posts + their reads: an event post sent at time T; the destroy runs at T+Δ (op isolate); the main isolate's read of the event post could run AFTER T+Δ (if the main isolate was busy) → UAF. To truly close it: hold the pointer until the main isolate READS, then release via the op isolate → the READS are the guard: dispose = unsubscribe → await marker → send "release-all + destroy" → await ack → close listeners: the destroy's payload RECLAIM frees all the still-held ones; events posts in flight at that moment = the race REMAINS (the C side has the same race for the APP's own callbacks... the C contract says callbacks' payloads stay valid until release OR destroy — the C is FINE; the DART's late read is the violation. HONEST FINAL SHAPE: the binding releases payloads EAGERLY at each event handler (release via the op isolate AFTER the read — a queued command); the destroy waits for ALL RELEASE-COMMANDS to process (a completion count); THEN calls destroy. Every event: read (main) → queue release (op) → count. The disposal: stop-issuing → the release count's drain → (all reads done = all events processed? An UNREAD event post still queued while the destroy runs: the COUNT can't know. → Track: a generation counter: each event post increments pending on the op-side...

STOP. THIS IS THE PLAN'S AUTHOR DESIGNING FOR TOO LONG. THE HONEST SIMPLE SHAPE: THE EVENTS CALLBACK DOES THE COPY+RELEASE INSIDE... make the EVENTS callback a NativeCallable.listener whose DART closure runs ON THE READER-THEEAD? Not possible (listener always posts to its owning isolate).

FINAL DECISION (simple + real): the binding KEEPS the app's own in-memory copies EAGER: the binding's C-side... — NO. The REAL simple fix: **`sa_client`'s EVENTS callback payloads are small strings; the binding READS THEM EAGERLY by having the callback pointer BE a NativeCallable.listener POSTING THE RAW VALUES — impossible (pointer args are addresses)...

→ **THE TRULY SIMPLE ANSWER: the C side reads the strings and releases THEM on the READER thread — i.e., the binding wraps the callbacks with NativeCallable.listener FOR ARG VALUES ONLY: CHANGE SA_CLIENT'S C API: add callback-shape alternatives that pass LENGTH-PREFIXED COPIES? No...

OK the REAL engineering answer: this is exactly the case Dart's `@DartFfi` "async callbacks" solve with listener + the rule: **NEVER read a C payload pointer AFTER the post** — read it IMMEDIATELY IN THE LISTENER'S DART CLOSURE: the closure runs on the MAIN isolate LATER — CANNOT. The listener CANNOT deref immediately.

**THE ANSWER IS: `NativeCallable.listener` with arguments that are ALREADY COPIED — the C API's events callback gives us `const char*` strings; the binding instead passes to C a callback where the STRING ARGUMENTS have ALREADY been converted... impossible without C.

**ACTUALLY — RE-READ DART FFI: NativeCallable.listener's callbacks: "The callback will be called on the isolate that created it, ... arguments must be copied"? — Dart's docs: listener callbacks CAN ONLY HAVE ARGTYPES that are "statically known" (they pass-by-value: ints/doubles/pointers). The DEREF is legal: the C-side memory is valid until destroy/release. The standard pattern (e.g. sqlite bindings): the listener's closure DEREFERENCES the pointer immediately — the danger is ONLY when C side frees the memory while the post is in flight. Standard practice: keep the payload alive until the Dart side handles it = EXACTLY SA_CLIENT's OWN design (hold until release). SO: the binding's flow = the listener's closure (main isolate) derefs + copies + sends the release command to the op isolate; the destroy's ordering contract: THE OPIISOLATE'S DESTROY COMMAND RUNS AFTER EVERY RELEASED EVENT (the C's own FIFO of the events + the commands share...). The op isolate's queue: each event → its listener post (main) + the binding ALSO posts the release command from the MAIN isolate per event AFTER its read: the op isolate processes releases; the destroy command queued LAST at dispose: the dispose sends destroy ONLY AFTER (a) the terminal marker's read + its release done (an awaitable chain on the main isolate: the last event handler resolves a Completer), (b) the queue drained. THE RACE: any event post read completes BEFORE the dispose's completer chain resolves; the destroy command posts after ALL releases — so the C's destroy frees only ALREADY-RELEASED payloads + any UNREAD-LATE posts: posts arrive at the main isolate FIFO BEFORE the ack/complete resolution... the completer chain: each event's handler completes its link after reading + posting the release; dispose awaits THE MARKER's link (the last event); ALL PRE-MARKER posts were read by FIFO before the marker's link resolved. SO NO UNREAD POSTS REMAIN when the destroy is issued. THE UAF IS CLOSED BY THE FIFO ORDER + THE CHAIN. PIN THIS EXACT PROTOCOL IN THE DESIGN/PLAN (it is subtle and the implementer must follow it).

Whew. Write the plan with this protocol pinned precisely + tests for the dispose chain.
```dart
// The dispose chain (THE RACE-CLOSURE PROTOCOL — pin):
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