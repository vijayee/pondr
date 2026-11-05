# Ask UX Implementation Plan (the Pondr ask card + the FFI ask-reply op)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pending ask renders as an inline canvas card (question + plan markdown + option buttons + free-text + conditional reject); tapping posts `sa_client_ask_reply` through the FFI op isolate; the fold's terminal contract learns `blocked`.

**Architecture:** Ask records ride the EXISTING events stream into the record fold; a new riverpod `PendingAskStore` (keyed by sid) is the card's single source of truth; the reply is a new binding op mirroring `interrupt` byte-for-byte in shape (the interrupt-shaped cb: status 0 = delivered, 1 = not delivered, 10+ = the transport's own statuses). The answer's user-side `msg.append` record already renders as an ordinary bubble — the fold only clears the store.

**Tech Stack:** Flutter 3 + riverpod 3 Notifier stores + `dart:ffi` (`NativeCallable.listener` + the op isolate `SaOpHost`) + flutter_markdown (already a dep).

**Spec:** `docs/ask-ux-spec.md` (owner-approved 2026-10-13) — READ FIRST. The engine-side surface (CA_ASK_REPLY 16/17, the parked ask, `int sa_client_ask_reply(sa_client_t*, const char* sid, const char* ask_id, uint8_t decision, const char* value, void (*cb)(void* ctx, uint8_t status), void* ctx)`) LANDED in SecretAgent d7b855a: `-1` return = the outright refusal (no cb), `0` return = the cb fired; cb status `0` = delivered, `1` = delivered-false, `10..15` = the transport failures (TIMEOUT/DISCONNECTED/BUSY/LOCAL).

**Standing gates:** `flutter analyze` (zero) + `flutter test` from `flutter/` (188 baseline + the new). Conventional commits, NO Co-Authored-By, NO TODO comments. The lib is loaded by `resolveSaLibraryPath()` — the ffi tests skip when no lib/config is present (the standing `SA_LIBRARY_PATH` gating in `test/ffi/ffi_test.dart`).

---

### Task 1: the FFI ask-reply op (the interrupt shape verbatim)

**Files:** Modify `flutter/lib/ffi/sa_ffi.dart` (typedefs ~:237, lookup ~:538, `SaFfiApi` ~:476, `SaFfi` impl ~:646); Modify `flutter/lib/ffi/sa_client_binding.dart` (command ~:280, `_SaOpKind`/`_PendingOp.refusal` ~:726-749, runner `handle` ~:438, listeners ~:788, `SaClientNative` ~:893, close/unregister ~:1029, :1246); Test `flutter/test/ffi/ffi_test.dart`.

- [ ] **Step 1: the failing test** (the model = the PROMPT round-trip test at :622-635 — there is no interrupt round-trip test to copy; the fake fires the cb BEFORE returning, the interrupt fake's shape at :215-224). Add to `FakeSaFfi`:

```dart
  int? lastAskDecision;
  @override
  int askReply(int client, int serial, String sid, String askId, int decision,
      String value, int askCb) {
    order.add('askReply(sid=$sid,ask=$askId,d=$decision,v=$value)');
    lastAskDecision = decision;
    (SaFfi.nativeFn<SaInterruptCbNative>(askCb)
        .asFunction<SaInterruptCbDart>())(
      ffi.Pointer<ffi.Void>.fromAddress(serial),
      0, // delivered
    );
    return 0;
  }
```

  and the round-trip test after the prompt one:

```dart
    test('askReply round-trips delivered', () async {
      final (wrapper, fake, _, _) = await connectHarness();
      final delivered = await wrapper.askReply(
          'sessions/frame-9', 'a1b2c3d4', false, 'Approve');
      expect(delivered, isTrue);
      expect(fake.order,
          contains('askReply(sid=sessions/frame-9,ask=a1b2c3d4,d=0,v=Approve)'));
      expect(fake.lastAskDecision, 0);
      await wrapper.dispose();
      await pump();
    });

    test('askReply not-delivered answers false', () async {
      // the fake variant firing cb(status 1) — delivered-false
      final (wrapper, fake, _, _) = await connectHarness();
      fake.askDeliveredStatus = 1;
      expect(await wrapper.askReply('s', 'a1b2c3d4', false, 'x'), isFalse);
      await wrapper.dispose();
      await pump();
    });
```

  (Add `int askDeliveredStatus = 0;` to `FakeSaFfi` and use it in the fake's cb call.)
- [ ] **Step 2: `sa_ffi.dart`** — beside the `SaInterruptCb*` typedefs (:237-239) nothing new (the ask op REUSES `sa_client_interrupt_cb_t`); the function typedefs + lookup, the interface method, the impl:

```dart
typedef SaAskReplyFnNative = ffi.Int32 Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.Uint8> askId,
    ffi.Uint8 decision,
    ffi.Pointer<ffi.Uint8> value,
    ffi.Pointer<ffi.NativeFunction<SaInterruptCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);
typedef SaAskReplyFnDart = int Function(
    ffi.Pointer<ffi.Void> client,
    ffi.Pointer<ffi.Uint8> sid,
    ffi.Pointer<ffi.Uint8> askId,
    int decision,
    ffi.Pointer<ffi.Uint8> value,
    ffi.Pointer<ffi.NativeFunction<SaInterruptCbNative>> callback,
    ffi.Pointer<ffi.Void> ctx);

  // in SaFfiApi:
  /// The ask-reply op: 0 = the cb fired (ctx = serial), -1 = refused loud
  /// (no cb — the events-callback re-entry rule covers this op: a reply
  /// must never originate from inside a C events callback).
  int askReply(int client, int serial, String sid, String askId,
      int decision, String value, int askCb);

  // in SaFfi (the interrupt impl's try/finally idiom, :646-659 — THREE
  // string copies, each freed in finally):
  @override
  int askReply(int client, int serial, String sid, String askId,
      int decision, String value, int askCb) {
    final sidP = SaUtf8.toNative(sid);
    final askP = SaUtf8.toNative(askId);
    final valP = SaUtf8.toNative(value);
    try {
      return _askReplyFn(
        ffi.Pointer<ffi.Void>.fromAddress(client),
        sidP, askP, decision, valP,
        SaFfi.nativeFn<SaInterruptCbNative>(askCb),
        ffi.Pointer<ffi.Void>.fromAddress(serial),
      );
    } finally {
      SaUtf8.free(sidP);
      SaUtf8.free(askP);
      SaUtf8.free(valP);
    }
  }
```
  and `late final _askReplyFn = _theLib.lookupFunction<SaAskReplyFnNative, SaAskReplyFnDart>('sa_client_ask_reply');`
- [ ] **Step 3: the binding** — mirror `SaInterruptCmd`/`interrupt` exactly:

```dart
final class SaAskReplyCmd extends SaOpCommand {
  const SaAskReplyCmd(this.serial, this.sid, this.askId, this.decision,
      this.value, this.cb);
  @override
  final int serial;
  final String sid;
  final String askId;
  final bool decision;   // false = answer, true = reject
  final String value;
  final int cb;
}
```
  `_SaOpKind.askReply` enum entry + the `_PendingOp.refusal` case `_SaOpKind.askReply => SaErr(status),`; the runner case `case SaAskReplyCmd(:final serial, :final sid, :final askId, :final decision, :final value, :final cb): emit(SaOpDone(serial, api.askReply(_clientOrBust, serial, sid, askId, decision ? 1 : 0, value, cb)));` — note the cb fires with the raw byte status, so `_onAskReply` maps: status 0 → `SaOk()`, anything else → `SaErr(status)` — and `SaClientNative.askReply` interprets:

```dart
  /// Reply to a parked owner-surface ask. Returns true when DELIVERED
  /// (cb status 0). delivered-false (cb 1), the transport statuses (cb
  /// 10..15), and the outright refusal (-1, mapped SaErr(-1)) all answer
  /// false. NEVER call from inside an events callback (the C re-entry rule).
  Future<bool> askReply(String sid, String askId, bool reject, String value) {
    _guardReady();
    final pending = _beginOp(_SaOpKind.askReply);
    host.send(SaAskReplyCmd(pending.serial, sid, askId, reject, value,
        _askReplyCb.nativeFunction.address));
    return pending.future.then((value) => value! is SaOk);
  }
```
  The listener: `late final ffi.NativeCallable<SaInterruptCbNative> _askReplyCb;` created in `_createListeners()` (`_askReplyCb = ffi.NativeCallable<SaInterruptCbNative>.listener(_onAskReply);`), `_onAskReply` = the `_onInterrupt` body verbatim (`_resolveOp(ctx.address, status == saClientStatusOk ? const SaOk() : SaErr(status));`), closed in `_closeListeners()`, unregistered in the dispose chain at :1246.
- [ ] **Step 4:** `flutter analyze` zero + `flutter test test/ffi/ffi_test.dart` (32 baseline + the 2 new; the standing skip gating applies); the FULL `flutter test` green; commit
  `git add flutter/lib/ffi flutter/test/ffi/ffi_test.dart && git commit -m "ffi: the ask-reply op — the interrupt shape verbatim, delivered rides cb status 0"`.

---

### Task 2: the PendingAskStore + the fold's three cases

**Files:** Modify `flutter/lib/data/services.dart` (the `ChatService` interface gains `answerAsk`); Modify `flutter/lib/data/daemon/ffi_chat_service.dart` (the fold + the service surface); Create `flutter/lib/data/pending_ask.dart` (the model + the store); Modify `flutter/lib/data/bindings.dart` (the provider); Mock: `flutter/lib/data/mock/mock_services.dart` (Task 3 wires its implementation — the interface lands here so the mock compiles); Test `flutter/test/daemon/ffi_services_test.dart`.

- [ ] **Step 1: the failing tests** (modeling the no-double-append pin's arrange/act/assert at :346-406; `emit(sid, seq, recordJson)` + `drain()` exist):

```dart
    test('ask record parks the store and blocked keeps the turn open', () async {
      final h = await chatHarness();
      h.emit('s1', 5, '{"type":"msg.append","payload":{"role":"user","content":"make it fast"}}');
      h.emit('s1', 6, '{"type":"ask","payload":{"askId":"a1b2c3d4","question":"Approve this plan?","options":["Approve","Reject"],"plan":"1. measure"}}');
      h.emit('s1', 7, '{"type":"turn.end","payload":{"reason":{"kind":"blocked"}}}');
      await drain();
      expect(h.chats.state, contains(ChatTyping)); // the turn stays OPEN — no ChatDone/ChatFailed
      final ask = h.asks.state['s1'];
      expect(ask?.askId, 'a1b2c3d4');
      expect(ask?.options, ['Approve', 'Reject']);
      expect(ask?.plan, '1. measure');
      await h.dispose();
    });

    test('ask.reply clears the store and the answer bubble rides msg.append', () async {
      final h = await chatHarness();
      /* …ask setup as above… */
      h.emit('s1', 8, '{"type":"ask.reply","payload":{"askId":"a1b2c3d4","decision":"answer","value":"Approve"}}');
      h.emit('s1', 9, '{"type":"msg.append","payload":{"role":"user","content":"Approve"}}');
      await drain();
      expect(h.asks.state['s1'], isNull);
      /* the s1 session's messages contain a user bubble 'Approve' — via the
         existing msg.append path; find it on the sessions list. */
    });

    test('garbled ask record never mounts the store', () async {
      final h = await chatHarness();
      h.emit('s1', 10, '{"type":"ask","payload":{"question":"no id"}}');
      await drain();
      expect(h.asks.state, isEmpty);
      await h.dispose();
    });
```
  (The harness gains an `asks` PendingAskStore handle — Step 3 defines the construction.)
- [ ] **Step 2: the model + the store** (`flutter/lib/data/pending_ask.dart`):

```dart
/// One in-flight owner-surface ask (the escalation slice's parked ask).
/// The engine forbids two at once per frame — the store is simply
/// last-writer-wins per sid, cleared only by the reply.
final class PendingAsk {
  const PendingAsk({required this.askId, required this.question,
      required this.options, this.plan});
  final String askId;
  final String question;
  final List<String> options;
  final String? plan;
}

final class AskUpsert { const AskUpsert(this.sid, this.ask); final String sid; final PendingAsk ask; }
final class AskCleared { const AskCleared(this.sid); final String sid; }

/// The card's single source of truth, keyed by sid (the store holds the
/// IN-FLIGHT ask only — an answered ask is just the reply records).
class PendingAskStore extends Notifier<Map<String, PendingAsk>> {
  @override
  Map<String, PendingAsk> build() {
    final chat = ref.watch(chatServiceProvider);
    final sub = chat.asks().listen((event) {
      state = switch (event) {
        AskUpsert(:final sid, :final ask) => {...state, sid: ask},
        AskCleared(:final sid) => {...state..remove(sid)},
      };
    });
    ref.onDispose(sub.cancel);
    return const {};
  }

  PendingAsk? forSid(String sid) => state[sid];
}
final pendingAskProvider = NotifierProvider<PendingAskStore, Map<String, PendingAsk>>(PendingAskStore.new);
```
  `ChatService` gains the ask surface (services.dart):

```dart
  /// The ask stream: AskUpsert / AskCleared events (the fold's two cases;
  /// the store switches on the runtime type).
  Stream<Object> asks();
  /// Reply to the sid's parked ask. true = DELIVERED (the ack contract:
  /// bind/post only — the card clears on the ask.reply RECORD, never here).
  Future<bool> answerAsk({required String sid, required String askId,
      required bool reject, required String value});
```
- [ ] **Step 3: the service surface** (`FfiChatService`): a broadcast `StreamController<Object>` `_asks`; in the fold's `switch (decoded['type'])` — two cases BEFORE the default:

```dart
      case 'ask':
        _onAsk(sid, decoded['payload']);
        return;
      case 'ask.reply':
        _onAskReply(sid, decoded['payload']);
        return;
```
  with the parse (guards mirror `_onReport`'s shape — wrong-typed fields are silently ignored):

```dart
  void _onAsk(String sid, dynamic payloadDyn) {
    if (payloadDyn is! Map<String, dynamic>) return;
    final askId = payloadDyn['askId'];
    final question = payloadDyn['question'];
    if (askId is! String || question is! String) return;
    final options = switch (payloadDyn['options']) {
      final List<dynamic> l => [for (final o in l) if (o is String) o],
      _ => const <String>[],
    };
    final plan = switch (payloadDyn['plan']) { final String s => s, _ => null };
    _asks.add(AskUpsert(sid, PendingAsk(askId: askId, question: question,
        options: options, plan: plan)));
  }

  void _onAskReply(String sid, dynamic payloadDyn) {
    _asks.add(AskCleared(sid));
  }
```
  The ask stream accessor: `Stream<Object> asks() => _asks.stream;` (the broadcast controller created in the constructor, closed in `dispose` alongside `events`). The `ChatTyping` feed: NOTHING feeds in the blocked case — `_onTurnEnd` gains, right after `kind` is parsed (:392-428 region), `if (kind == 'blocked') return;` (the turn stays open — the parked ask card is its face; the typing indicator honestly rides until the reply's turn terminal).
  `answerAsk` on `FfiChatService`:

```dart
  @override
  Future<bool> answerAsk({required String sid, required String askId,
      required bool reject, required String value}) =>
      _client.then((c) => c.askReply(sid, askId, reject, value));
```
  (`_client` is the existing `Future<SaClientNative>` field — confirm the actual field name from :55's constructor and use it.) `MockChatService.answerAsk` — a `Future<bool>.value(true)` stub THIS task (Task 3 replaces it with the scripted behavior); `MockChatService.asks()` = a fresh empty broadcast stream field the same way.
- [ ] **Step 4:** the daemon harness (`chatHarness`) constructs the store's provider container? — the harness tests services DIRECTLY; the fold tests assert via a `PendingAskStore` seeded manually: simplest honest route — the harness exposes `h.asks` as a manually-driven store instance (`FakeAskSink` capturing the events the service emits, since the STORE's subscription logic is a provider-composition detail that view tests cover). Concretely: `FfiChatService` exposes `asks()`; the harness collects into a `Map<String,PendingAsk>` via a plain listener (`h.asks` = that map, updated by the test's subscription in `chatHarness()`; `dispose` cancels). Keep the STORE class pure-Dart-testable too: its upsert/clear mapping is 3 lines — the view tests cover it live.
- [ ] **Step 5:** `flutter analyze` zero; `flutter test test/daemon/ffi_services_test.dart` (27 baseline + 3); the fold's no-double-append pin and the terminal pins stay green; commit
  `git add flutter/lib/data flutter/test/daemon/ffi_services_test.dart && git commit -m "data: the ask fold + the pending-ask store — ask upserts, ask.reply clears, blocked keeps the turn open"`.

---

### Task 3: the mock ask

**Files:** Modify `flutter/lib/data/mock/mock_services.dart`; Test `flutter/test/daemon/` stays (the mock's own tests live in `test/data_layer_test.dart`'s mock family — read its current shape first; add there).

- [ ] **Step 1: the failing test** — in the mock family's style: send with text `'plan the refactor'` → the ask upserts (question "Approve this plan?", options `[Approve, Reject]`, plan non-empty); `answerAsk(sid, askId, reject: false, value: 'Approve')` → true + the mock's normal typing→reply flow resumes (the session gains its assistant reply message).
- [ ] **Step 2: the implementation** — in `MockChatService.send`'s reply stage: if the user message contains `'plan'` (case-insensitive), instead of writing the pool reply: write the plan text as an assistant message, emit the ask on the mock `_asks` controller, store the pending ask keyed by captured id (`_pendingMockAsk`), yield ChatTyping (the turn rides open) and RETURN (the send stream ends without ChatDone — the ask is the terminal face). `answerAsk`: if a pending mock ask matches askId → clear it, then run the existing typing-delay + pool-reply flow writing the assistant message (the same body as the pool-reply branch — factor the small `_deliverMockReply(capturedId)` private out of `send`'s tail if it reads cleaner, else duplicate the 6 lines — do NOT over-abstract). Unknown askId → false.
- [ ] **Step 3:** `flutter analyze` zero + `flutter test` full green; commit
  `git add flutter/lib/data/mock flutter/test && git commit -m "mock: the scripted plan ask — the card's full UX exercisable without the C side"`.

---

### Task 4: the AskCard widget

**Files:** Create `flutter/lib/views/chat/ask_card.dart`; Modify `flutter/lib/views/chat/chat_page.dart` (the canvas loop ~:203-244); Test `flutter/test/chat_view_test.dart`.

- [ ] **Step 1: the failing widget tests** (the `pumpChat` harness at :121-150; seed the mock service per Task 3's scripted ask):

```dart
    testWidgets('the pending ask renders as a card', (tester) async {
      /* pumpChat with the mock chat; send 'plan the refactor'; pumpAndSettle;
         expect(find.byKey(const Key('chat.ask-card')), findsOneWidget);
         expect(find.text('Approve this plan?'), findsOneWidget);
         expect(find.byKey(const Key('chat.ask-option-0')), findsOneWidget);
         expect(find.byKey(const Key('chat.ask-option-1')), findsOneWidget);
         expect(find.byKey(const Key('chat.ask-reject')), findsNothing); */
    });
    testWidgets('option tap posts the label and the card clears', ...) { /* tap option-0 ('Approve'); the delivered reply resolves and the ask.reply clears (the mock path); pumpAndSettle; find.byKey(ask-card) findsNothing; the mock's reply bubble landed. */ }
    testWidgets('free text posts the typed value', ...) { /* enterText on 'chat.ask-free' + tap 'chat.ask-send-free'. */ }
    testWidgets('a no-options ask shows reject', ...) { /* seed PendingAskStore manually with options: [] — testWidgets CAN override pendingAskProvider with a fixed map; expect chat.ask-reject present; tapping posts decision true. */ }
```
- [ ] **Step 2: the widget** (`ask_card.dart`) — the mockup's card idiom (`MessageBubble`'s container recipe: `Material` + `BorderRadius.circular(8)` + the `Color(0x…)` constants; the `_SendButton`'s `PressScale`/`InkWell` shape for the buttons):

```dart
final class AskCard extends ConsumerStatefulWidget { /* sid: String, ask: PendingAsk */ }
/* layout: Container(decoration: the assistant-bubble container color
   Color(0x1F888DDF) + border Color(0x4D888DDF), radius 8) >
   Column [ header row ('Pending ask' label + the question, bold),
   if plan != null: MarkdownBody(data: plan),
   for i, o in options: SizedBox(height 8) + full-width OutlinedButton-like
     InkWell(
       key: Key('chat.ask-option-$i'), onTap: busy ? null : () => _post(0, o))
   ,
   the free-text row: TextField(key: Key('chat.ask-free'),
       decoration: 'type a different answer', controller: _free),
     IconButton-like InkWell(key: Key('chat.ask-send-free'), onTap: ...)
   ,
   if ask.options.isEmpty: TextButton(key: Key('chat.ask-reject'),
       onTap: busy ? null : () => _post(1, ''))
   ,
   if (notDeliveredShown) the error line: Text('not delivered — the session
       may be gone', style: the muted style)
   ] */
```
  `_post(int decision, String value)`: set `busy` → `await ref.read(chatServiceProvider).answerAsk(sid: sid, askId: ask.askId, reject: decision == 1, value: value)` → true: nothing (the ask.reply record clears the store; the card unmounts when `pendingAskProvider` drops the sid) → false: `notDeliveredShown = true; busy = false` (the silent-drop corner: a DELIVERED ack also just releases `busy` — the card clears only via the record; if the record never arrives the card sits re-enabled, per the spec §3's pinned corner). Buttons disabled while busy with the `_SendButton`'s `AnimatedOpacity(enabled ? 1.0 : 0.4)` idiom.
- [ ] **Step 3: the canvas slot** (`chat_page.dart`): after the messages loop (beside the typing block), watch the pending ask for the ACTIVE sid:

```dart
final PendingAsk? pendingAsk = ref.watch(pendingAskProvider
    .select((m) => m[ref.watch(activeIdProvider) ?? '']));
/* if (pendingAsk != null) AskCard(key: Key('chat.ask-card'), sid: activeId!, ask: pendingAsk) —
   the same Column spacing (SizedBox(height: 24)) as the bubbles. */
```
- [ ] **Step 4:** `flutter analyze` zero + `flutter test` FULL green (119 view tests + the new; the existing 27 chat pins hold); commit
  `git add flutter/lib/views/chat flutter/test/chat_view_test.dart && git commit -m "views: the ask card — options, free text, reject, the not-delivered line"`.

---

### Task 5: gates + the close-out

**Files:** Modify `flutter/docs`→ repo-root `docs/ask-ux-spec.md` NO (spec stays); the memory is the CONTROLLER's write, not a repo file.

- [ ] **Step 1:** from `flutter/`: `flutter analyze` (zero — the standing bar) + `flutter test` (the full 188+new) + a MANUAL daemon-mode smoke if the environment offers it: `--dart-define=pondr.mode=daemon` with the real daemon (`frame-demo serve --escalation plan-ask-act` exists since SecretAgent a906cdf) — the plan-gate ask renders the card, tapping Approve resumes the turn. SKIPPED-WITH-REASON if no reachable daemon/model (the standing live-gate discipline) — record which.
- [ ] **Step 2:** the final sweep: no TODO/FIXME/HACK/XXX in touched files; the git log's conventional messages; report the counts + the divergence notes (the typing indicator riding a parked turn; the silent-drop corner) + the known-pending (non-active-session ask badges; the global interrupt button).

---

## Self-review notes (the controller's, 2026-10-13)

- **Spec coverage:** §1 flow → Tasks 1-2; §2 FFI → Task 1; §3 card → Task 4; §4 mock → Task 3; §5 tests → each task's step 1s + Task 5; §6 non-goals untouched. The harness `h.asks` shape (Task 2 Step 4) pins the fold-test strategy so the STORE's provider composition is view-test territory — matches the spec's "the store holds them keyed by sid" honesty.
- **Type consistency:** `PendingAsk{askId, question, options, plan}` used in Tasks 2/4; the op names `askReply`/`SaAskReplyCmd`/`SaAskReplyFnNative` consistent across Tasks 1's steps; `answerAsk` on the ChatService interface named once (Task 2) and consumed by Task 4's card.
- **Placeholders:** the widget's `/* ... */` bodies are LAYOUT COMMENTS in the house style (the code block IS the spec of the layout); every test step names its keys and assertions concretely.
