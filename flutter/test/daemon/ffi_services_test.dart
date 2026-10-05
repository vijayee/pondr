/// The FFI services' tests (the FFI-binding plan's Task 5): the three real
/// service implementations against a MOCKED `SaClientNative` (the landed
/// fake from `test/ffi/ffi_test.dart` — the seam's own fake ABI), pinning:
/// the session-replace flow (the first send creates: the local draft's id
/// becomes the daemon's sid), the NO-DOUBLE-APPEND pin (the records are the
/// message source of truth), the (sid, seq) record-dedupe (a replay + a
/// live duplicate deliver ONE message), the assistant-done mapping + the
/// typing timing, the config adoption (the picker's set → the native's
/// configSet with the right fields), the file's persistence round-trip, and
/// the mode flag's parse (the bindings' swap shape).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pondr/data/bindings.dart';
import 'package:pondr/daemon/app_config_store.dart';
import 'package:pondr/data/daemon/ffi_chat_service.dart';
import 'package:pondr/data/daemon/ffi_settings_service.dart';
import 'package:pondr/data/daemon/ffi_sessions_service.dart';
import 'package:pondr/data/mock/mock_services.dart';
import 'package:pondr/data/models.dart' hide Provider;
import 'package:pondr/data/models.dart' as mockup show Provider;
import 'package:pondr/data/services.dart';
import 'package:pondr/ffi/sa_client_binding.dart';
import 'package:pondr/ffi/sa_ffi.dart' show saClientStatusDisconnected;

import '../ffi/ffi_test.dart' show FakeRow, FakeSaFfi, theConfig;

// ── the harness pieces ──────────────────────────────────────────────────────

/// The record's JSON as the daemon composes the store's record (the
/// `{type, payload{role, content}}` shape the handlers' tests read).
String msgRecord(String role, String content) => jsonEncode(
      <String, dynamic>{
        'type': 'msg.append',
        'payload': <String, dynamic>{'role': role, 'content': content},
      },
    );

/// A control-family record (a non-msg.append record the fold must ignore —
/// the cell/lifecycle riders).
String controlRecord() => jsonEncode(<String, dynamic>{
      'type': 'control',
      'payload': <String, dynamic>{'reason': 'lifecycle'},
    });

/// The turn-closer rider (`lifecycle.h`'s `turn.end`): the reason's kind is
/// the daemon's union member; the text rides only when the closer carries
/// the control kind's wording.
String turnEndRecord(String kind, {String? text}) => jsonEncode(
      <String, dynamic>{
        'type': 'turn.end',
        'payload': <String, dynamic>{
          'turn': 1,
          'reason': <String, dynamic>{
            'kind': kind,
            'text': ?text,
          },
        },
      },
    );

/// The quiet-completion record (the frame's `frame.report`): {child_sid,
/// text} — its text is the turn.closer's done content when no assistant
/// bubble ever rode the turn.
String reportRecord(String text) => jsonEncode(<String, dynamic>{
      'type': 'frame.report',
      'payload': <String, dynamic>{'child_sid': 'sessions/e', 'text': text},
    });

/// The messages of a session's row (the tests' read shape).
List<Message> messagesOf(FfiSessionsService sessions, String sid) =>
    sessions.byId(sid)?.messages ?? const <Message>[];

/// Lets every scheduled microtask/completion land (the op host's
/// task-boundary deliveries and the fold's unawaited boot ride several
/// turns).
Future<void> drain() async {
  for (var i = 0; i < 8; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

/// A connected wrapper against a fresh fake, seeded with [rows].
Future<(SaClientNative, FakeSaFfi)> connectFake([
  List<FakeRow> rows = const <FakeRow>[],
]) async {
  final fake = FakeSaFfi()..listed = rows;
  final client = SaClientNative(api: fake, host: SaInProcessOpHost());
  final result = await client.connect(theConfig(), const SaCallbacks());
  expect(result, isA<SaOk>(), reason: 'the fake connects by default');
  return (client, fake);
}

/// The chat harness: the sessions + the chat service riding ONE connected
/// wrapper; the record fold feeds through [events] — the test's own
/// controller, the supervisor's `events` fan-out's role in production.
final class ChatHarness {
  ChatHarness._(this.client, this.fake, this.sessions, this.chat, this.events,
      this.failures);

  final SaClientNative client;
  final FakeSaFfi fake;
  final FfiSessionsService sessions;
  final FfiChatService chat;

  /// The raw events (the production supervisor's fan-out).
  final StreamController<SaEventRecord> events;

  /// The raw failures (the DISCONNECTED shape's producer in a test).
  final StreamController<SaFailure> failures;

  /// One record's fold (the daemon's store-notify shape).
  void emit(String sid, int seq, String? recordJson) => events
      .add(SaEventRecord(sid: sid, seq: seq, op: 0, recordJson: recordJson));

  void dispose() {
    chat.dispose();
    events.close();
    failures.close();
    client.dispose();
  }
}

Future<ChatHarness> chatHarness() async {
  final (client, fake) = await connectFake();
  final sessions = FfiSessionsService(clientFuture: Future.value(client));
  final events = StreamController<SaEventRecord>.broadcast();
  final failures = StreamController<SaFailure>.broadcast();
  final chat = FfiChatService(
    client: Future.value(client),
    sessions: sessions,
    events: events.stream,
    failures: failures.stream,
  );
  return ChatHarness._(client, fake, sessions, chat, events, failures);
}

/// A sessions-only harness (the fold's store reads).
Future<(SaClientNative, FakeSaFfi, FfiSessionsService)> sessionsHarness(
    [List<FakeRow> rows = const <FakeRow>[]]) async {
  final (client, fake) = await connectFake(rows);
  final sessions = FfiSessionsService(clientFuture: Future.value(client));
  return (client, fake, sessions);
}

void main() {
  group('FfiSessionsService (the rows fold + the store contracts)', () {
    test('list(): the rows fold in the NEWEST-FIRST built order (the '
        'created epoch-seconds are the sort; the goal becomes the name)',
        () async {
      final (_, _, sessions) =
          await sessionsHarness(<FakeRow>[
        (
          sid: 'sessions/old',
          status: 'running',
          goal: 'the old',
          created: 1000,
          depth: 3,
        ),
        (
          sid: 'sessions/new',
          status: 'done',
          goal: 'the new',
          created: 2500,
          depth: 1,
        ),
      ]);
      await drain(); // the boot listing's fold
      expect(sessions.list().map((s) => s.id).toList(), <String>[
        'sessions/new',
        'sessions/old',
      ], reason: 'newest-first by the rows\' created seconds (the '
          'listing\'s own order is the SID sort, never the app\'s)');
      final head = sessions.byId('sessions/new')!;
      expect(head.name, 'the new');
      // THE CREATED SEMANTICS: the wire's u64 UTC epoch seconds — the
      // app's display instant is that UTC instant, localized.
      expect(
        head.updatedAt,
        DateTime.fromMillisecondsSinceEpoch(2500 * 1000, isUtc: true)
            .toLocal(),
      );
    });

    test('create(): the LOCAL EMPTY DRAFT — no daemon sid, front-inserted, '
        'active; a second create returns THE SINGULAR draft', () async {
      final (_, _, sessions) = await sessionsHarness();
      await drain();
      final draft = sessions.create();
      expect(draft.id, '', reason: 'the draft sentinel — no daemon session '
          'exists before the first send');
      expect(draft.name, 'New conversation');
      expect(sessions.active()?.id, '');
      expect(sessions.list(), hasLength(1));
      final again = sessions.create();
      expect(again.id, '');
      expect(sessions.list(), hasLength(1),
          reason: 'the same draft stands — the store has no second empty '
              'session either');
      expect(identical(again, draft), isTrue);
    });

    test('delete(): the app-local hide — the refresh keeps it hidden (the '
        'daemon frame\'s staying is the recorded out-of-scope)', () async {
      final (_, fake, sessions) = await sessionsHarness(<FakeRow>[
        (
          sid: 'sessions/a',
          status: 'running',
          goal: '',
          created: 10,
          depth: 0,
        ),
      ]);
      await drain();
      expect(sessions.list(), isNotEmpty);
      sessions.delete('sessions/a');
      expect(sessions.active(), isNull, reason: 'nothing remains');
      fake.listed = <FakeRow>[
        (
          sid: 'sessions/a',
          status: 'running',
          goal: '',
          created: 10,
          depth: 0,
        ),
      ];
      await sessions.refresh();
      expect(sessions.list(), isEmpty,
          reason: 'a locally hidden sid stays hidden through the refresh');
      expect(sessions.hiddenSids, contains('sessions/a'));
    });

    test('the refresh preserves a known session\'s fold and its local '
        'name; the store\'s empty goal is never a rename', () async {
      final (_, _, sessions) = await sessionsHarness(<FakeRow>[
        (
          sid: 'sessions/a',
          status: 'running',
          goal: '',
          created: 50,
          depth: 1,
        ),
      ]);
      await drain();
      sessions.rename('sessions/a', 'the local name');
      sessions.storeMessage(
        'sessions/a',
        Message(
          id: 'm:sessions/a:1',
          role: MessageRole.user,
          content: 'kept',
          timestamp: DateTime.now(),
        ),
      );
      await sessions.refresh(); // the same row again (its goal still empty)
      final a = sessions.byId('sessions/a')!;
      expect(a.name, 'the local name');
      expect(a.messages.map((m) => m.content).toList(), <String>['kept']);
    });

    test('the active id: the refresh re-heads a vanished active', () async {
      final (_, fake, sessions) = await sessionsHarness(<FakeRow>[
        (
          sid: 'sessions/a',
          status: 'running',
          goal: '',
          created: 10,
          depth: 0,
        ),
      ]);
      await drain();
      expect(sessions.active()?.id, 'sessions/a');
      fake.listed = const <FakeRow>[]; // the daemon's row is gone
      await sessions.refresh();
      expect(sessions.active(), isNull);
    });
  });

  group('FfiChatService (the records are the message source of truth)', () {
    test('THE SESSION-REPLACE FLOW: the draft\'s first send creates the '
        'daemon session — the daemon\'s sid REPLACES the draft\'s empty id '
        'in place, and the active id follows', () async {
      final harness = await chatHarness();
      final chat = harness.chat;
      final sessions = harness.sessions;
      final fake = harness.fake;
      final rawEvents = harness.events;

      sessions.create(); // the draft ('', front, active)
      fake.promptSid = 'sessions/first';
      final events = <ChatEvent>[];
      final done = Completer<void>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat.send('the first goal', const <AttachedFile>[]).listen(
            events.add,
            onDone: () {
              sub.cancel();
              done.complete();
            },
            onError: done.completeError,
          );

      // The prompt's create went out with NO sid (the daemon owns the birth).
      await drain();
      expect(fake.order, contains('prompt(sid=null)'));
      // The daemon's records: the user's append, then the assistant's reply.
      rawEvents.add(SaEventRecord(
        sid: 'sessions/first',
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'the first goal'),
      ));
      await drain();
      rawEvents.add(SaEventRecord(
        sid: 'sessions/first',
        seq: 2,
        op: 0,
        recordJson: msgRecord('assistant', 'the reply'),
      ));
      await drain();
      await done.future.timeout(const Duration(seconds: 2));

      expect(events.first, isA<ChatTyping>(),
          reason: 'the typing rides the send\'s start');
      expect(events, hasLength(2), reason: 'the typing, then the done');
      expect((events.last as ChatDone).content, 'the reply');
      // THE REPLACE (pinned): the draft's empty id is gone — the daemon's
      // sid holds its position (the front), the local name rides, the
      // active id followed.
      final store = sessions.list();
      expect(store.map((s) => s.id).toList(), <String>['sessions/first']);
      expect(store.single.name, isNot('New conversation'));
      expect(store.single.name.startsWith('the first goal'), isTrue);
      expect(sessions.active()?.id, 'sessions/first');
      // The record fold wrote BOTH bubbles (the user one included).
      expect(messagesOf(sessions, 'sessions/first').map((m) => m.content),
          <String>['the first goal', 'the reply']);
      harness.dispose();
    });

    test('THE NO-DOUBLE-APPEND PIN: the send appends nothing locally — the '
        'user bubble is the record\'s arrival, exactly once', () async {
      final harness = await chatHarness();
      final rawEvents = harness.events;
      final fake = harness.fake;
      final chat = harness.chat;
      final sessions = harness.sessions;

      harness.sessions.create();
      fake.promptSid = 'sessions/dup';
      final collected = <ChatEvent>[];
      final done = Completer<void>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat
          .send('say the thing', const <AttachedFile>[])
          .listen(collected.add,
              onDone: () {
                sub.cancel();
                done.complete();
              },
              onError: done.completeError);
      await drain();
      const sid = 'sessions/dup';
      // BEFORE the record's arrival: no user message anywhere (the send
      // waits for the record; no local bubble was appended).
      expect(messagesOf(sessions, sid), isEmpty);
      rawEvents.add(SaEventRecord(
        sid: sid,
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'say the thing'),
      ));
      await drain();
      expect(messagesOf(sessions, sid), hasLength(1));
      // The SAME record delivered again (a replay + a live duplicate) —
      // still exactly one.
      rawEvents.add(SaEventRecord(
        sid: sid,
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'say the thing'),
      ));
      await drain();
      expect(messagesOf(sessions, sid), hasLength(1),
          reason: 'the (sid, seq) dedupe: a duplicate never re-appends');
      rawEvents.add(SaEventRecord(
        sid: sid,
        seq: 2,
        op: 0,
        recordJson: msgRecord('assistant', 'done text'),
      ));
      await drain();
      await done.future.timeout(const Duration(seconds: 2));
      expect(messagesOf(sessions, sid).map((m) => m.role).toList(),
          <MessageRole>[MessageRole.user, MessageRole.assistant]);
      // THE DETERMINISTIC IDS (the (sid, seq) shape).
      expect(messagesOf(sessions, sid).map((m) => m.id).toList(),
          <String>['m:sessions/dup:1', 'm:sessions/dup:2']);
      expect(collected, hasLength(2), reason: 'typing, done');
      harness.dispose();
    });

    test('THE REPLAY PIN: a re-subscribed session\'s whole log folds ONCE — '
        'the seen set swallows the replay, a new tail record still folds',
        () async {
      final harness = await chatHarness();
      final rawEvents = harness.events;
      final sessions = harness.sessions;

      // First subscription's view: two records fold.
      rawEvents.add(SaEventRecord(
        sid: 'sessions/r',
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'one'),
      ));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/r',
        seq: 2,
        op: 0,
        recordJson: msgRecord('assistant', 'two'),
      ));
      await drain();
      expect(messagesOf(sessions, 'sessions/r'), hasLength(2));

      // A REPLAY from seq 0 (the re-subscribe's whole log, redelivered):
      // every record is already seen — NOTHING re-folds.
      rawEvents.add(SaEventRecord(
        sid: 'sessions/r',
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'replayed one'),
      ));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/r',
        seq: 2,
        op: 0,
        recordJson: msgRecord('assistant', 'replayed two'),
      ));
      await drain();
      expect(messagesOf(sessions, 'sessions/r'), hasLength(2),
          reason: 'the replayed log folds once — the (sid, seq) dedupe');

      // The live tail after the replay folds as usual.
      rawEvents.add(SaEventRecord(
        sid: 'sessions/r',
        seq: 3,
        op: 0,
        recordJson: msgRecord('user', 'three'),
      ));
      await drain();
      expect(
        messagesOf(sessions, 'sessions/r').map((m) => m.content).toList(),
        <String>['one', 'two', 'three'],
      );
      harness.dispose();
    });

    test('the cell/lifecycle/control records and the non-chat roles fold '
        'nothing (the chat never sees them; the audit view is the recorded '
        'follow-on)', () async {
      final harness = await chatHarness();
      final rawEvents = harness.events;
      final sessions = harness.sessions;

      rawEvents.add(SaEventRecord(
        sid: 'sessions/c',
        seq: 1,
        op: 0,
        recordJson: controlRecord(),
      ));
      rawEvents.add(SaEventRecord(
          sid: 'sessions/c', seq: 2, op: 1, recordJson: null));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/c',
        seq: 3,
        op: 0,
        recordJson: jsonEncode(<String, dynamic>{
          'type': 'msg.append',
          'payload': <String, dynamic>{'role': 'system', 'content': 'nope'},
        }),
      ));
      await drain();
      expect(messagesOf(sessions, 'sessions/c'), isEmpty,
          reason: 'a control record, a marker, and a system-role '
              'msg.append all fold nothing');
      harness.dispose();
    });

    test('THE STEER: an existing sid\'s send prompts WITH the sid (no '
        'create) and its records close the stream', () async {
      final harness = await chatHarness();
      final chat = harness.chat;
      final fake = harness.fake;
      final sessions = harness.sessions;
      final rawEvents = harness.events;

      // A row folded in (the boot listing's shape) and selected.
      fake.listed = <FakeRow>[
        (
          sid: 'sessions/e',
          status: 'running',
          goal: '',
          created: 10,
          depth: 1,
        ),
      ];
      await sessions.refresh();
      sessions.select('sessions/e');
      fake.promptSid = ''; // the steer's queued sentinel
      final collected = <ChatEvent>[];
      final done = Completer<void>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat
          .send('the steer', const <AttachedFile>[])
          .listen(collected.add,
              onDone: () {
                sub.cancel();
                done.complete();
              },
              onError: done.completeError);
      await drain();
      expect(fake.order, contains('prompt(sid=sessions/e)'),
          reason: 'a non-empty captured id is the steer, not a create');
      rawEvents.add(SaEventRecord(
        sid: 'sessions/e',
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'the steer'),
      ));
      await drain();
      rawEvents.add(SaEventRecord(
        sid: 'sessions/e',
        seq: 2,
        op: 0,
        recordJson: msgRecord('assistant', 'steered reply'),
      ));
      await drain();
      await done.future.timeout(const Duration(seconds: 2));
      expect((collected.last as ChatDone).content, 'steered reply');
      expect(messagesOf(sessions, 'sessions/e').map((m) => m.content)
          .toList(), <String>['the steer', 'steered reply']);
      harness.dispose();
    });

    test('THE CREATE-RACE FAST REPLY: the assistant record that folds '
        'before the send\'s waiter registered still completes its send',
        () async {
      // The send's own subscribe (the create's sid is only known after the
      // prompt) leaves a window; the assistant fold waits in the unclaimed
      // cache and claims its send at the registration.
      final (client, fake) = await connectFake();
      final sessions = FfiSessionsService(clientFuture: Future.value(client));
      final rawEvents = StreamController<SaEventRecord>.broadcast();
      fake.promptSid = 'sessions/fast';
      final chat = FfiChatService(
        client: Future.value(client),
        sessions: sessions,
        events: rawEvents.stream,
        failures: const Stream<SaFailure>.empty(),
      );
      addTearDown(() {
        chat.dispose();
        rawEvents.close();
        unawaited(client.dispose());
      });
      sessions.create();
      final collected = <ChatEvent>[];
      final done = Completer<void>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat.send('fast', const <AttachedFile>[]).listen(collected.add,
          onDone: () {
            sub.cancel();
            done.complete();
          },
          onError: done.completeError);
      // The records land while the send sits between the prompt's response
      // and its registration (the fold's controller deliveries happen
      // during the send's awaits).
      await drain();
      rawEvents.add(SaEventRecord(
        sid: 'sessions/fast',
        seq: 1,
        op: 0,
        recordJson: msgRecord('user', 'fast'),
      ));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/fast',
        seq: 2,
        op: 0,
        recordJson: msgRecord('assistant', 'the fast reply'),
      ));
      await drain();
      await done.future.timeout(const Duration(seconds: 2));
      expect((collected.last as ChatDone).content, 'the fast reply',
          reason: 'the unclaimed create-race fold claims its send');
      expect(
        messagesOf(sessions, 'sessions/fast').map((m) => m.content).toList(),
        <String>['fast', 'the fast reply'],
      );
    });

    test('a refused prompt surfaces through the send\'s stream (the typing '
        'event, then the error; the typing flag\'s close is the view\'s '
        'onError)', () async {
      final harness = await chatHarness();
      harness.sessions.create();
      harness.fake.promptStatus = 2; // the daemon's refusal
      harness.fake.promptSid = null;
      final events = <ChatEvent>[];
      Object? error;
      final received = Completer<void>();
      final sub = harness.chat
          .send('fails', const <AttachedFile>[])
          .listen(events.add, onError: (Object e) {
        error = e;
        received.complete();
      });
      await received.future.timeout(const Duration(seconds: 2));
      expect(events, hasLength(1));
      expect(events.single, isA<ChatTyping>(),
          reason: 'the typing rode the send\'s start before the refusal');
      expect(error, isA<StateError>());
      await sub.cancel();
      harness.dispose();
    });

    test('the blank-send guard: nothing to send (the content is the text; '
        'the files ride no wire verb — the recorded follow-on)', () async {
      final harness = await chatHarness();
      harness.sessions.create();
      final events = <ChatEvent>[];
      await harness.chat
          .send('   ', const <AttachedFile>[])
          .forEach(events.add);
      expect(events, isEmpty, reason: 'a blank draft sends nothing');
      expect(harness.fake.order, isNot(contains('prompt(sid=null)')));
      expect(harness.sessions.active()?.id, '',
          reason: 'the draft stands — nothing was created');
      harness.dispose();
    });

    test('the DISCONNECTED failure settles the pending sends (the '
        'connection\'s loss errors their streams; nobody hangs)', () async {
      final harness = await chatHarness();
      final chat = harness.chat;
      final fake = harness.fake;
      final failures = harness.failures;
      final sessions = harness.sessions;

      sessions.create();
      fake.promptSid = 'sessions/gone';
      final done = Completer<Object?>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat.send('will lose the daemon', const <AttachedFile>[]).listen(
          (event) {},
          onDone: () {
            sub.cancel();
            done.complete('done');
          },
          onError: (Object e) {
            sub.cancel();
            done.complete(e);
          });
      await drain();
      // The connection dies mid-turn (the supervisor's failure channel).
      failures.add(SaFailure(
        reqId: 9,
        status: saClientStatusDisconnected,
        text: 'the connection was lost',
      ));
      await drain();
      final outcome = await done.future.timeout(const Duration(seconds: 2));
      expect(outcome, isA<StateError>(),
          reason: 'the pending send errored through the DISCONNECTED '
              'failure — nobody hangs');
      harness.dispose();
    });

    test('THE REPORT-COMPLETED TURN: a turn that closed via frame.report '
        '(no assistant msg.append exists) completes at its turn.end with '
        'the report\'s text as the done', () async {
      final harness = await chatHarness();
      final chat = harness.chat;
      final fake = harness.fake;
      final sessions = harness.sessions;
      final rawEvents = harness.events;

      fake.listed = <FakeRow>[
        (
          sid: 'sessions/e',
          status: 'running',
          goal: '',
          created: 10,
          depth: 1,
        ),
      ];
      await sessions.refresh();
      sessions.select('sessions/e');
      fake.promptSid = ''; // the steer's queued registration
      final collected = <ChatEvent>[];
      final done = Completer<void>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat.send('the quiet turn', const <AttachedFile>[]).listen(
          collected.add,
          onDone: () {
            sub.cancel();
            done.complete();
          }, onError: done.completeError);
      await drain();
      // The record order (the daemon's batch): the user's append, the
      // report, then the turn.end rider carrying {completed}.
      var seq = 0;
      rawEvents.add(SaEventRecord(
        sid: 'sessions/e',
        seq: ++seq,
        op: 0,
        recordJson: msgRecord('user', 'the quiet turn'),
      ));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/e',
        seq: ++seq,
        op: 0,
        recordJson: reportRecord('the report text'),
      ));
      await drain();
      expect(done.isCompleted, isFalse,
          reason: 'the report folds alone — the WAITER completes at the '
              'turn.end, the closers\' record');
      rawEvents.add(SaEventRecord(
        sid: 'sessions/e',
        seq: ++seq,
        op: 0,
        recordJson: turnEndRecord('completed'),
      ));
      await done.future.timeout(const Duration(seconds: 2));
      expect(collected, hasLength(2), reason: 'the typing, then ONE done');
      expect((collected.last as ChatDone).content, 'the report text',
          reason: 'the quiet-completed turn\'s done content is the '
              'report\'s text');
      expect(messagesOf(sessions, 'sessions/e').map((m) => m.content)
          .toList(), <String>['the quiet turn'],
          reason: 'the report is not a bubble — the msg.append records stay '
              'the message source of truth');
      harness.dispose();
    });

    test('THE FAILED TURN: a turn.end {error} completes the send with a '
        'ChatFailed and writes the dimmed failure line', () async {
      final harness = await chatHarness();
      final chat = harness.chat;
      final fake = harness.fake;
      final sessions = harness.sessions;
      final rawEvents = harness.events;

      fake.listed = <FakeRow>[
        (
          sid: 'sessions/f',
          status: 'running',
          goal: '',
          created: 10,
          depth: 1,
        ),
      ];
      await sessions.refresh();
      sessions.select('sessions/f');
      fake.promptSid = ''; // the steer's queued registration
      final collected = <ChatEvent>[];
      final done = Completer<void>();
      late final StreamSubscription<ChatEvent> sub;
      sub = chat.send('the doomed turn', const <AttachedFile>[]).listen(
          collected.add,
          onDone: () {
            sub.cancel();
            done.complete();
          }, onError: done.completeError);
      await drain();
      var seq = 0;
      rawEvents.add(SaEventRecord(
        sid: 'sessions/f',
        seq: ++seq,
        op: 0,
        recordJson: msgRecord('user', 'the doomed turn'),
      ));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/f',
        seq: ++seq,
        op: 0,
        recordJson: turnEndRecord('error', text: 'the model connection died'),
      ));
      await done.future.timeout(const Duration(seconds: 2));
      expect(collected, hasLength(2), reason: 'the typing, then the failed');
      expect((collected.last as ChatFailed).text, 'the model connection died',
          reason: 'the failed turn\'s terminal carries the closer\'s '
              'failure wording — the stream never hangs');
      // The fold wrote the failure line (the dimmed assistant bubble the
      // view renders), with its own deterministic id.
      final stored = messagesOf(sessions, 'sessions/f');
      expect(stored.map((m) => m.content).toList(),
          <String>['the doomed turn', 'the model connection died']);
      expect(stored.last.error, isTrue, reason: 'the failed-turn line');
      expect(stored.last.id, 'e:sessions/f:2',
          reason: 'the deterministic (sid, seq) backstop');
      harness.dispose();
    });

    test('THE FED TURN\'S CLOSER PASSES: a turn.end arriving after an '
        'assistant msg.append already fed the waiter settles NOTHING — the '
        'next send keeps its own pairing', () async {
      final harness = await chatHarness();
      final chat = harness.chat;
      final fake = harness.fake;
      final sessions = harness.sessions;
      final rawEvents = harness.events;

      fake.listed = <FakeRow>[
        (
          sid: 'sessions/g',
          status: 'running',
          goal: '',
          created: 10,
          depth: 1,
        ),
      ];
      await sessions.refresh();
      sessions.select('sessions/g');
      fake.promptSid = '';
      final first = <ChatEvent>[];
      final firstDone = Completer<void>();
      late final StreamSubscription<ChatEvent> firstSub;
      firstSub = chat.send('the fed turn', const <AttachedFile>[]).listen(
          first.add,
          onDone: () {
            firstSub.cancel();
            firstDone.complete();
          }, onError: firstDone.completeError);
      await drain();
      var seq = 0;
      rawEvents.add(SaEventRecord(
        sid: 'sessions/g',
        seq: ++seq,
        op: 0,
        recordJson: msgRecord('user', 'the fed turn'),
      ));
      rawEvents.add(SaEventRecord(
        sid: 'sessions/g',
        seq: ++seq,
        op: 0,
        recordJson: msgRecord('assistant', 'the fed reply'),
      ));
      await firstDone.future.timeout(const Duration(seconds: 2));
      expect((first.last as ChatDone).content, 'the fed reply');

      // The NEXT send registers; the fed turn's turn.end rider arrives.
      final second = <ChatEvent>[];
      final secondDone = Completer<void>();
      late final StreamSubscription<ChatEvent> secondSub;
      secondSub = chat.send('the next turn', const <AttachedFile>[]).listen(
          second.add,
          onDone: () {
            secondSub.cancel();
            secondDone.complete();
          }, onError: secondDone.completeError);
      await drain();
      rawEvents.add(SaEventRecord(
        sid: 'sessions/g',
        seq: ++seq,
        op: 0,
        recordJson: turnEndRecord('completed'),
      ));
      await drain();
      expect(second, hasLength(1),
          reason: 'the fed turn\'s own closer passed — the next send\'s '
              'waiter is untouched');
      rawEvents.add(SaEventRecord(
        sid: 'sessions/g',
        seq: ++seq,
        op: 0,
        recordJson: msgRecord('assistant', 'the next reply'),
      ));
      await secondDone.future.timeout(const Duration(seconds: 2));
      expect((second.last as ChatDone).content, 'the next reply');
      harness.dispose();
    });
  });

  group('FfiSettingsService (the store ride + the daemon adoption)', () {
    late Directory temp;

    setUp(() => temp = Directory.systemTemp.createTempSync('pondr-ffi-svc-'));
    tearDown(() => temp.deleteSync(recursive: true));

    test('THE CONFIG ADOPTION: the picker\'s set carries the selected '
        'provider\'s {baseUrl, apiKey, modelId} to the native configSet',
        () async {
      final (client, fake) = await connectFake();
      final service = FfiSettingsService(
        store: AppConfigStore(filePath: '${temp.path}/settings.json'),
        client: () => Future.value(client),
      );
      final providerId = service.saveProvider(
        name: 'The Vendor',
        baseUrl: 'http://127.0.0.1:11434',
        apiKey: 'sk-the-key',
      )!;
      final modelId = service.saveModel(
        providerId,
        modelId: 'deepseek-chat',
        label: 'DeepSeek Chat',
      )!;
      expect(fake.configSetCalls, 0, reason: 'no adoption before a '
          'selection');
      service.selectedModelKey = '$providerId::$modelId';
      await service.lastAdoption;
      expect(fake.configSetCalls, 1);
      expect(fake.configSets.single, (
        'http://127.0.0.1:11434',
        'sk-the-key',
        'deepseek-chat',
      ), reason: 'the WIRE name is the model\'s modelId — the label is '
          'display-only');
      // The key format is the interface's <providerId>::<modelId> (the
      // picker's internal ids) and it persisted app-side, too.
      expect(service.selectedModelKey, '$providerId::$modelId');
      expect(service.store.exists, isTrue);
      unawaited(client.dispose());
    });

    test('an unresolvable key adopts nothing; the startup GET reads the '
        'daemon\'s truth', () async {
      final (client, fake) = await connectFake();
      fake.configBase = 'http://other.example';
      fake.configModel = 'gemma4:latest';
      final service = FfiSettingsService(
        store: AppConfigStore(filePath: '${temp.path}/settings.json'),
        client: () => Future.value(client),
      );
      service.selectedModelKey = 'gone::stale';
      await service.lastAdoption;
      expect(fake.configSetCalls, 0,
          reason: 'an unresolvable key adopts nothing');
      expect(service.lastAdoptionError, isNull);
      final truth = await service.readDaemonTruth();
      expect(truth, isA<SaConfigResult>());
      expect(truth!.ok, isTrue);
      expect(truth.baseUrl, 'http://other.example');
      expect(truth.model, 'gemma4:latest');
      expect(truth.apiKey, isNull, reason: 'the template\'s absent member '
          'rides null');
      expect(fake.order, contains('configGet'));
      unawaited(client.dispose());
    });

    test('the adoption\'s failure records into lastAdoptionError (a dead '
        'client never leaks an unhandled zone error)', () async {
      final service = FfiSettingsService(
        store: AppConfigStore(filePath: '${temp.path}/settings.json'),
        // The ensure's failure shape: the client's future errors.
        client: () =>
            Future<SaClientNative>.error(StateError('the daemon never '
                'came up')),
      );
      final providerId =
          service.saveProvider(name: 'V', baseUrl: 'http://x', apiKey: 'k')!;
      final modelId = service.saveModel(providerId, modelId: 'm', label: 'M')!;
      service.selectedModelKey = '$providerId::$modelId';
      await service.lastAdoption;
      expect(service.lastAdoptionError, isA<StateError>());
    });

    test('the interface\'s shape rides the store (the export\'s defaults; '
        'a re-load reads the persisted truth)', () async {
      final path = '${temp.path}/settings.json';
      final service = FfiSettingsService(
        store: AppConfigStore(filePath: path),
      );
      expect(service.displayName, 'Ada Lovelace',
          reason: 'the export\'s fresh defaults');
      expect(service.username, 'ada_lovelace');
      expect(service.bio, '');
      expect(service.notifMessages, isTrue);
      expect(service.notifSounds, isFalse);
      expect(service.accentColor, '#888ddf');
      expect(service.selectedModelKey, isNull);
      expect(service.providers(), isEmpty,
          reason: 'the export\'s provider store starts EMPTY');

      service.displayName = 'Ada';
      service.accentColor = '#c3acda';
      service.notifSounds = true;
      final reloaded = FfiSettingsService(
          store: AppConfigStore(filePath: path)); // a SECOND load
      expect(reloaded.displayName, 'Ada');
      expect(reloaded.accentColor, '#c3acda');
      expect(reloaded.notifSounds, isTrue);
      expect(reloaded.providers(), isEmpty);
    });
  });

  group('AppConfigStore (the settings file)', () {
    late Directory temp;

    setUp(() => temp = Directory.systemTemp.createTempSync('pondr-ffi-store-'));
    tearDown(() => temp.deleteSync(recursive: true));

    test('the round-trip: the providers + the selection + every profile '
        'field', () {
      final store = AppConfigStore(filePath: '${temp.path}/settings.json');
      expect(store.exists, isFalse, reason: 'no file was ever written');
      final config = AppConfig(
        providers: <mockup.Provider>[
          mockup.Provider(
            id: 'p1',
            name: 'The Vendor',
            baseUrl: 'http://127.0.0.1:11434',
            apiKey: 'sk-secret',
            enabled: false,
            models: <ProviderModel>[
              const ProviderModel(
                id: 'm1',
                modelId: 'deepseek-chat',
                label: 'DeepSeek',
                enabled: true,
              ),
            ],
          ),
        ],
        selectedModelKey: 'p1::m1',
        displayName: 'Ada',
        username: 'ada',
        bio: 'the bio',
        notifMessages: false,
        notifSounds: true,
        accentColor: '#c3acda',
      );
      store.save(config);
      expect(store.exists, isTrue);
      expect(File('${temp.path}/settings.json.tmp').existsSync(), isFalse,
          reason: 'THE ATOMIC WRITE: the rename consumed the tmp — the '
              'target is the only committed shape');
      final loaded = store.load();
      expect(loaded.selectedModelKey, 'p1::m1');
      expect(loaded.displayName, 'Ada');
      expect(loaded.username, 'ada');
      expect(loaded.bio, 'the bio');
      expect(loaded.notifMessages, isFalse);
      expect(loaded.notifSounds, isTrue);
      expect(loaded.accentColor, '#c3acda');
      expect(loaded.providers, hasLength(1));
      final provider = loaded.providers.single;
      expect(provider.baseUrl, 'http://127.0.0.1:11434');
      expect(provider.apiKey, 'sk-secret');
      expect(provider.enabled, isFalse);
      expect(provider.models.single.modelId, 'deepseek-chat');
      expect(provider.models.single.label, 'DeepSeek');
    });

    test('a missing, corrupt, or schema-lying file repairs to the fresh '
        'defaults', () {
      final missing = AppConfigStore(filePath: '${temp.path}/missing.json');
      final fresh = missing.load();
      expect(fresh.providers, isEmpty);
      expect(fresh.displayName, 'Ada Lovelace');

      final corrupt = AppConfigStore(filePath: '${temp.path}/corrupt.json');
      File(corrupt.filePath).writeAsStringSync('{ not json');
      expect(corrupt.load().providers, isEmpty);

      final lied = AppConfigStore(filePath: '${temp.path}/lied.json');
      File(lied.filePath).writeAsStringSync('[]');
      expect(lied.load().displayName, 'Ada Lovelace',
          reason: 'a schema-lying shape repairs too');
    });

    test('the save creates the parent dirs (the first run)', () {
      final nested = '${temp.path}/deep/config/pondr/settings.json';
      final store = AppConfigStore(filePath: nested);
      store.save(const AppConfig());
      expect(File(nested).existsSync(), isTrue);
    });

    test('the default path is the XDG convention (Linux v1)', () {
      expect(defaultAppConfigPath().endsWith('pondr/settings.json'), isTrue,
          reason: 'the pinned convention: <config-home>/pondr/'
              'settings.json (XDG_CONFIG_HOME respected, ~/.config '
              'otherwise)');
    });
  });

  group('the bindings (the one swap, the mode flag)', () {
    test('the mode parse: only "daemon" runs real — the absence, the mock, '
        'and any other value ride the export\'s mock', () {
      expect(useDaemonMode, isFalse,
          reason: 'the suite runs with no define — the mock is the default');
      expect(useDaemonModeFromDefine('daemon'), isTrue);
      expect(useDaemonModeFromDefine('mock'), isFalse);
      expect(useDaemonModeFromDefine(''), isFalse);
      expect(useDaemonModeFromDefine('Daemon'), isFalse,
          reason: 'the define is exact (an unknown value cannot silently '
              'run real)');
    });

    test('the default bindings wire the MOCK shapes (the app views never '
        'change)', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);
      expect(container.read(sessionsProvider), isA<MockSessionsService>());
      expect(container.read(settingsProvider), isA<MockSettingsService>());
      expect(container.read(chatServiceProvider), isA<MockChatService>());
      expect(
        container.read(subconsciousProvider),
        isA<MockSubconsciousService>(),
        reason: 'the subconscious stays mock in BOTH modes (the spec\'s '
            'recorded out-of-scope)',
      );
    });
  });
}