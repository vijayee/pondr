import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pondr/data/bindings.dart';
import 'package:pondr/data/mock/mock_services.dart';
import 'package:pondr/data/models.dart';
import 'package:pondr/data/services.dart';

/// Task 2's contract pins: the send stream's shape, the date buckets, the
/// sessions + providers CRUD, the settings defaults — each rule traced to
/// the export's functions (`mockup_reference/app.tsx` is truth).

ChatSession sessionOn(DateTime at, {String id = 'x'}) => ChatSession(
  id: id,
  name: id,
  updatedAt: at,
  messages: const <Message>[],
);

DateTime calendar(int dayOffset) {
  final now = DateTime.now();
  return DateTime(now.year, now.month, now.day + dayOffset, 12);
}

void main() {
  group('groupSessions (the export app.tsx:101-123)', () {
    test('buckets by local-midnight comparison, empty buckets dropped', () {
      final groups = groupSessions(<ChatSession>[
        sessionOn(calendar(0), id: 'today'),
        sessionOn(calendar(-1), id: 'yesterday'),
        sessionOn(calendar(-4), id: 'week'),
        sessionOn(calendar(-8), id: 'earlier'),
      ]);
      expect(groups.map((g) => g.label).toList(), <String>[
        'Today',
        'Yesterday',
        'This Week',
        'Earlier',
      ]);
      expect(
        groups.map((g) => g.sessions.single.id).toList(),
        <String>['today', 'yesterday', 'week', 'earlier'],
      );
    });

    test('only non-empty buckets survive', () {
      final groups = groupSessions(<ChatSession>[
        sessionOn(calendar(0), id: 'today'),
        sessionOn(calendar(-8), id: 'earlier'),
      ]);
      expect(groups.map((g) => g.label).toList(), <String>['Today', 'Earlier']);
    });
  });

  group('MockSessionsService (the export handleNewChat/handleDeleteSession)', () {
    test('seeds the export order, s1 active', () {
      final sessions = MockSessionsService();
      expect(sessions.list().map((s) => s.id), <String>[
        's1',
        's2',
        's3',
        's4',
        's5',
      ]);
      expect(sessions.active()!.id, 's1');
    });

    test('create front-inserts "New conversation" and selects it', () {
      final sessions = MockSessionsService();
      final created = sessions.create();
      expect(created.name, 'New conversation');
      expect(created.messages, isEmpty);
      expect(sessions.list().first.id, created.id);
      expect(sessions.list().length, 6);
      expect(sessions.active()!.id, created.id);
    });

    test('delete reselects the head of the remainder', () {
      final sessions = MockSessionsService();
      sessions.select('s3');
      sessions.delete('s3');
      expect(sessions.active()!.id, 's1'); // next[0] of the remainder
      expect(sessions.list().map((s) => s.id), <String>['s1', 's2', 's4', 's5']);
    });

    test('deleting the last one leaves active dangling (null)', () {
      final sessions = MockSessionsService(seed: const <ChatSession>[]);
      sessions.create();
      sessions.delete(sessions.active()!.id);
      expect(sessions.list(), isEmpty);
      expect(sessions.active(), isNull);
    });

    test('changes() fires on create/select/delete', () async {
      final sessions = MockSessionsService(seed: const <ChatSession>[]);
      final fired = <ChatsChanged>[];
      sessions.changes().listen(fired.add);
      sessions.create();
      sessions.delete(sessions.active()!.id);
      await Future<void>.delayed(const Duration(milliseconds: 50));
      expect(fired.length, 2);
    });
  });

  group('MockChatService.send (the export handleSend)', () {
    test(
      'shape: typing (1100-1799 ms) → done with an AI_POOL reply, no deltas',
      () async {
        final sessions = MockSessionsService();
        final chat = MockChatService(sessions);
        final events = await chat
            .send('hello there', const <AttachedFile>[])
            .toList();
        expect(events, hasLength(2));
        final typing = events[0] as ChatTyping;
        final done = events[1] as ChatDone;
        expect(typing.duration.inMilliseconds, inInclusiveRange(1100, 1799));
        expect(aiPool, contains(done.content));
        expect(events.whereType<ChatDelta>(), isEmpty);
      },
      timeout: const Timeout(Duration(seconds: 10)),
    );

    test('the user message lands with the 45-char rename rule', () async {
      final sessions = MockSessionsService();
      final chat = MockChatService(sessions);
      sessions.create(); // the rename rule applies to "New conversation"
      final longText = 'x' * 50;
      await chat.send(longText, const <AttachedFile>[]).toList();
      var active = sessions.active()!;
      expect(
        active.name,
        'x' * 45 + '…', // slice(0,45) + ellipsis when longer
      );
      expect(active.messages.first.role, MessageRole.user);
      expect(active.messages.first.content, longText);
      expect(
        active.messages.last.role,
        MessageRole.assistant,
      );

      // A second send does NOT rename (only "New conversation" does).
      await chat.send('second', const <AttachedFile>[]).toList();
      active = sessions.active()!;
      expect(active.name, 'x' * 45 + '…');
      expect(active.messages, hasLength(4)); // user, reply, user, reply

      // The ellipsis gate reads the RAW input length (the export's
      // `inputText.length > 45`), not the trimmed one: a >45-char input
      // that trims to 44 still gets '…' appended.
      final spaced = '${'y' * 44}      ';
      final other = MockSessionsService(seed: const <ChatSession>[]);
      other.create();
      await MockChatService(other).send(spaced, const <AttachedFile>[]).toList();
      expect(other.active()!.name, 'y' * 44 + '…');
      // Both writes bump updatedAt (the export's `updatedAt: new Date()`).
      expect(
        active.updatedAt.isAfter(
          DateTime.now().subtract(const Duration(seconds: 5)),
        ),
        isTrue,
      );
    });

    test('attachments ride the user message', () async {
      final sessions = MockSessionsService();
      final chat = MockChatService(sessions);
      sessions.create(); // a fresh, empty session to append into
      const file = AttachedFile(
        id: 'f1',
        name: 'notes.txt',
        type: 'text/plain',
        size: 2048,
      );
      await chat.send('with a file', <AttachedFile>[file]).toList();
      expect(
        sessions.active()!.messages.first.files,
        <AttachedFile>[file],
      );
    });

    test('blank text with no files is a no-op (empty stream, no writes)', () async {
      final sessions = MockSessionsService(seed: const <ChatSession>[]);
      sessions.create();
      final before = sessions.active();
      final fired = <ChatsChanged>[];
      sessions.changes().listen(fired.add);
      final events = await MockChatService(
        sessions,
      ).send('   ', const <AttachedFile>[]).toList();
      expect(events, isEmpty);
      await Future<void>.delayed(const Duration(milliseconds: 50));
      expect(fired, isEmpty);
      expect(sessions.active(), before);
    });

    test('the send writes fire the session change stream twice', () async {
      final sessions = MockSessionsService();
      final fired = <ChatsChanged>[];
      sessions.changes().listen(fired.add);
      await MockChatService(sessions).send('hi', const <AttachedFile>[]).toList();
      await Future<void>.delayed(const Duration(milliseconds: 50));
      expect(fired.length, 2); // the user append + the assistant append
    });
  });

  group('MockSettingsService (the export state + saveProvider/saveModel)', () {
    test('defaults are the export initial state (app.tsx:822-829)', () {
      final settings = MockSettingsService();
      expect(settings.displayName, 'Ada Lovelace');
      expect(settings.username, 'ada_lovelace');
      expect(settings.bio, '');
      expect(settings.notifMessages, isTrue);
      expect(settings.notifSounds, isFalse);
      expect(settings.accentColor, '#888ddf');
      expect(settings.providers(), isEmpty);
      expect(settings.selectedModelKey, isNull);
    });

    test('saveProvider creates enabled + empty models at the end', () {
      final settings = MockSettingsService();
      final id = settings.saveProvider(
        name: 'DeepSeek',
        baseUrl: 'https://api.deepseek.com/v1',
        apiKey: 'sk-test',
      );
      expect(id, isNotNull);
      final providers = settings.providers();
      expect(providers, hasLength(1));
      expect(providers.single.id, id);
      expect(providers.single.enabled, isTrue);
      expect(providers.single.models, isEmpty);
      expect(providers.single.name, 'DeepSeek');
    });

    test('saveProvider edits in place and guards blank fields', () {
      final settings = MockSettingsService();
      final id = settings.saveProvider(
        name: 'DeepSeek',
        baseUrl: 'https://api.deepseek.com/v1',
        apiKey: 'sk-1',
      );
      // Blank guard: nothing is written, null comes back.
      expect(
        settings.saveProvider(editingId: id, name: ' ', baseUrl: 'x', apiKey: ''),
        isNull,
      );
      expect(settings.providers().single.name, 'DeepSeek');
      expect(settings.saveProvider(editingId: id, name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', apiKey: 'sk-2'), id);
      final edited = settings.providers().single;
      expect(edited.name, 'OpenAI');
      expect(edited.baseUrl, 'https://api.openai.com/v1');
      expect(edited.apiKey, 'sk-2');
    });

    test('toggleProvider + deleteProvider round-trip', () {
      final settings = MockSettingsService();
      final id = settings.saveProvider(name: 'P', baseUrl: 'https://x', apiKey: '');
      settings.toggleProvider(id!);
      expect(settings.providers().single.enabled, isFalse);
      settings.deleteProvider(id);
      expect(settings.providers(), isEmpty);
    });

    test('saveModel add/edit (enabled on create), delete, toggle, guard', () {
      final settings = MockSettingsService();
      final pid = settings.saveProvider(name: 'P', baseUrl: 'https://x', apiKey: '')!;
      // Blank modelId guard.
      expect(settings.saveModel(pid, modelId: ' ', label: ''), isNull);
      final mid = settings.saveModel(pid, modelId: 'deepseek-chat', label: 'DeepSeek Chat')!;
      var model = settings.providers().single.models.single;
      expect(model.id, mid);
      expect(model.enabled, isTrue);
      expect(model.modelId, 'deepseek-chat');

      expect(settings.saveModel(pid, editingModelId: mid, modelId: 'deepseek-reasoner', label: ''), mid);
      model = settings.providers().single.models.single;
      expect(model.modelId, 'deepseek-reasoner');
      expect(model.label, '');
      expect(model.enabled, isTrue);

      settings.toggleModel(pid, mid);
      expect(settings.providers().single.models.single.enabled, isFalse);

      settings.deleteModel(pid, mid);
      expect(settings.providers().single.models, isEmpty);
    });
  });

  group('MockSubconsciousService (the export BASE pair)', () {
    test('returns the export nodes + edges', () {
      final (nodes, edges) = MockSubconsciousService().graph();
      expect(nodes, hasLength(25));
      expect(edges, hasLength(29));
      expect(nodes.map((n) => n.id), containsAll(<String>['quantum-mechanics', 'ai-consciousness']));
      // The two nodes with their own colour, verbatim.
      expect(
        nodes.firstWhere((n) => n.id == 'quantum-consciousness').color,
        '#b8a4dc',
      );
      expect(
        nodes.firstWhere((n) => n.id == 'ai-consciousness').color,
        '#aab3e8',
      );
      expect(
        edges.first,
        const SimEdge(source: 'quantum-mechanics', target: 'superposition'),
      );
    });
  });

  group('the export helpers', () {
    test('formatFileSize (app.tsx:89-93)', () {
      expect(formatFileSize(512), '512 B');
      expect(formatFileSize(2048), '2.0 KB');
      expect(formatFileSize(1536), '1.5 KB');
      expect(formatFileSize(1024 * 1024 * 3 + 512 * 1024), '3.5 MB');
    });

    test('fileIconKind (getFileIcon, app.tsx:95-99)', () {
      expect(fileIconKind('image/png'), AttachedFileIcon.image);
      expect(fileIconKind('application/pdf'), AttachedFileIcon.document);
      expect(fileIconKind('text/plain'), AttachedFileIcon.document);
      expect(fileIconKind('application/vnd.ms-excel'), AttachedFileIcon.other);
    });

    test('entities compare by id (SimEdges by endpoints)', () {
      final now = DateTime.now();
      expect(
        Message(id: 'm1', role: MessageRole.user, content: 'a', timestamp: now),
        Message(id: 'm1', role: MessageRole.assistant, content: 'b', timestamp: now),
      );
      expect(
        const AttachedFile(id: 'f1', name: 'a', type: 'a', size: 1),
        const AttachedFile(id: 'f1', name: 'b', type: 'b', size: 2),
      );
      expect(
        ChatSession(id: 's1', name: 'a', updatedAt: now, messages: const <Message>[]),
        ChatSession(id: 's1', name: 'b', updatedAt: now, messages: const <Message>[]),
      );
      expect(
        const SimEdge(source: 'a', target: 'b'),
        const SimEdge(source: 'a', target: 'b'),
      );
      expect(
        const SimEdge(source: 'a', target: 'b'),
        isNot(const SimEdge(source: 'b', target: 'a')),
      );
    });

    test('seedSessions carries the five export sessions verbatim', () {
      final seeds = seedSessions();
      expect(seeds.map((s) => s.id), <String>['s1', 's2', 's3', 's4', 's5']);
      expect(seeds.map((s) => s.name), <String>[
        'Quantum Superposition & Qubits',
        'The Hard Problem of Consciousness',
        'Supervised vs Unsupervised Learning',
        'The Fermi Paradox',
        'What Makes Arguments Good?',
      ]);
      expect(seeds[0].messages.map((m) => m.id), <String>['m1a', 'm1b', 'm1c', 'm1d']);
      expect(seeds[1].messages.first.content, 'What is the hard problem of consciousness, and why is it hard?');
      expect(seeds[4].messages.single.id, 'm5a');
    });

    test('the suggestions bank verbatim', () {
      expect(suggestions, <String>[
        'Explain the Fermi paradox',
        'What is the hard problem of consciousness?',
        'How do transformer models work?',
        'What makes an argument convincing?',
      ]);
    });
  });

  group('the binding file', () {
    test('serves the four interfaces through the mock implementations', () {
      final container = ProviderContainer();
      addTearDown(container.dispose);
      expect(container.read(sessionsProvider), isA<MockSessionsService>());
      expect(container.read(chatServiceProvider), isA<MockChatService>());
      expect(container.read(settingsProvider), isA<MockSettingsService>());
      expect(container.read(subconsciousProvider), isA<MockSubconsciousService>());
    });

    test('the chat writes through the SAME session store the views read', () async {
      final container = ProviderContainer();
      addTearDown(container.dispose);
      final chat = container.read(chatServiceProvider);
      final userAppend = Completer<ChatEvent>();
      final sub = chat
          .send('store check', const <AttachedFile>[])
          .listen(userAppend.complete);
      await userAppend.future; // the user message is in before the first event
      await sub.cancel();
      final active = container.read(sessionsProvider).active()!;
      expect(active.messages.last.content, 'store check');
      expect(active.messages.last.role, MessageRole.user);
    });
  });
}