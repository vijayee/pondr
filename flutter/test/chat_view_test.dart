import 'package:flutter/gestures.dart' show PointerDeviceKind;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:pondr/app/shell.dart';
import 'package:pondr/data/bindings.dart';
import 'package:pondr/data/mock/mock_services.dart';
import 'package:pondr/app/router.dart'
    show AuthState, authStateProvider;
import 'package:pondr/data/models.dart' hide Provider;
import 'package:pondr/data/models.dart' as mockup show Provider;
import 'package:pondr/data/services.dart';
import 'package:pondr/main.dart';
import 'package:pondr/views/chat/chat_state.dart';
import 'package:pondr/views/chat/message_bubble.dart';
import 'package:pondr/views/chat/typing_indicator.dart';

/// Task 4's pins: the chat surface 1:1 with the export's chat regions —
/// the send flow through the REAL mocks, the typing indicator's lifecycle +
/// pulse, the bubble entrance, the sidebar's grouping/selection/delete, the
/// search toggle, the composer's send logic + the attachment sheet + the
/// model picker, the suggestion chips' empty-session rule, and both
/// responsive shapes. Every expectation traces to
/// `mockup_reference/app.tsx`.

/// A settings stub whose PROVIDER LIST is the test's fixture (the picker's
/// pool fixture) while everything else delegates to the real mock.
class _FixtureSettings extends MockSettingsService {
  _FixtureSettings(this._fixture);

  final List<mockup.Provider> _fixture;
  String? _modelKey;

  @override
  List<mockup.Provider> providers() => List.unmodifiable(_fixture);

  @override
  String? get selectedModelKey => _modelKey;

  @override
  set selectedModelKey(String? value) => _modelKey = value;
}

_FixtureSettings _pickerFixture() => _FixtureSettings(<mockup.Provider>[
      mockup.Provider(
        id: 'p1',
        name: 'Aurora',
        baseUrl: 'https://aurora.example',
        apiKey: 'k',
        enabled: true,
        models: <ProviderModel>[
          const ProviderModel(
            id: 'mA1',
            modelId: 'aurora-large',
            label: 'Aurora Large',
            enabled: true,
          ),
          const ProviderModel(
            id: 'mA2',
            modelId: 'aurora-mini',
            label: 'Aurora Mini',
            enabled: false,
          ),
        ],
      ),
      mockup.Provider(
        id: 'p2',
        name: 'Hush',
        baseUrl: 'https://hush.example',
        apiKey: 'k',
        enabled: true,
        models: <ProviderModel>[
          const ProviderModel(
            id: 'mH1',
            modelId: 'hush-pro',
            label: 'Hush Pro',
            enabled: true,
          ),
        ],
      ),
      mockup.Provider(
        id: 'p3',
        name: 'Nebula',
        baseUrl: 'https://nebula.example',
        apiKey: 'k',
        enabled: false,
        models: <ProviderModel>[
          const ProviderModel(
            id: 'mN1',
            modelId: 'nebula-x',
            label: 'Nebula X',
            enabled: true,
          ),
        ],
      ),
    ]);

/// The failed-turn chat stub (the daemon's `turn.end {error}` shape): the
/// mock never emits a failure (its replies always succeed), so the
/// composer's ChatFailed handling rides this stub.
class _FailedChatService implements ChatService {
  const _FailedChatService();

  @override
  Stream<ChatEvent> send(String text, List<AttachedFile> files) async* {
    yield const ChatTyping(duration: Duration(milliseconds: 50));
    // The failure rides in LATER (the turn ran first) — the typing window
    // is observable before the terminal lands.
    await Future<void>.delayed(const Duration(milliseconds: 100));
    yield const ChatFailed('the model refused');
  }
}

void main() {
  /// Pumps the app logged in at [size]; each test gets its own container and
  /// therefore its own Mock services. [chat] swaps the chat binding (the
  /// failed-turn shape's stub; the mock never fails).
  Future<ProviderContainer> pumpChat(
    WidgetTester tester,
    Size size, {
    MockSettingsService? settings,
    ChatService? chat,
  }) async {
    final AuthState authenticator = AuthState();
    final ProviderContainer container = ProviderContainer(
      overrides: [
        authStateProvider.overrideWith((_) => authenticator),
        if (settings != null) settingsProvider.overrideWith((_) => settings),
        if (chat != null) chatServiceProvider.overrideWith((_) => chat),
      ],
    );
    addTearDown(container.dispose);
    authenticator.signIn();
    await tester.binding.setSurfaceSize(size);
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(
      UncontrolledProviderScope(
        container: container,
        child: const PondrApp(),
      ),
    );
    await tester.pumpAndSettle();
    return container;
  }

  MockSessionsService sessionsOf(ProviderContainer container) =>
      container.read(sessionsProvider) as MockSessionsService;

  /// The rail boots collapsed — expand/collapse it with the plan's toggle.
  Future<void> toggleRail(WidgetTester tester) async {
    await tester.tap(find.byKey(const Key('sidebar.rail-toggle')));
    await tester.pumpAndSettle();
  }

  Future<void> sendDraft(WidgetTester tester, String text) async {
    await tester.enterText(find.byKey(const Key('composer.field')), text);
    await tester.pump();
    await tester.tap(find.byKey(const Key('composer.send')));
    await tester.pump();
  }

  group('the send flow through the real Mock services', () {
    testWidgets('send → user bubble → typing → reply bubble → the composer '
        'cleared + the auto-scroll + the header count', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));

      await sendDraft(tester, 'What is consciousness?');

      // The user message writes BEFORE the first event (the export's
      // handleSend write, app.tsx:907-913).
      expect(find.text('What is consciousness?'), findsOneWidget);
      // The composer cleared; the indicator rides the typing flag.
      final EditableText field =
          tester.widget<EditableText>(find.byType(EditableText).first);
      expect(field.controller.text, isEmpty);
      expect(find.byKey(const ValueKey('chat.typing')), findsOneWidget);
      // The typing window is 1100-1799 ms — still inside it at +500 ms.
      await tester.pump(const Duration(milliseconds: 500));
      expect(find.byKey(const ValueKey('chat.typing')), findsOneWidget);

      // Cross the whole window (worst case 1799 ms) in one frame.
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();
      expect(find.byKey(const ValueKey('chat.typing')), findsNothing);

      // The reply: a uniform pick from the pool onto the ACTIVE session s1.
      final MockSessionsService sessions = sessionsOf(container);
      final ChatSession s1 = sessions.byId('s1')!;
      expect(s1.messages.length, 6); // 4 seed + user + reply
      expect(s1.messages[4].role, MessageRole.user);
      expect(s1.messages[4].content, 'What is consciousness?');
      expect(s1.messages[5].role, MessageRole.assistant);
      expect(aiPool.contains(s1.messages[5].content), isTrue);

      // The header's readout (app.tsx:2134-2137): '6 messages'.
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('6 messages'),
        ),
        findsOneWidget,
      );

      // The auto-scroll rides the bottom anchor (the export's
      // scrollIntoView, app.tsx:851-853).
      final ScrollController controller =
          tester
              .widget<SingleChildScrollView>(find.byKey(
            const Key('chat.message-canvas'),
          ))
              .controller!;
      expect(
        controller.position.pixels,
        closeTo(controller.position.maxScrollExtent, 1.0),
      );
    });

    testWidgets("a fresh session takes the 45-character name + the export's "
        'ellipsis gate', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);
      await tester.tap(find.byKey(const Key('sidebar.new-chat')));
      await tester.pumpAndSettle();

      // RAW 60 (> 45) → the trimmed cut + '…' (app.tsx:909-911).
      await sendDraft(tester, 'a' * 60);
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();

      final MockSessionsService sessions = sessionsOf(container);
      final ChatSession fresh = sessions.active()!;
      expect(fresh.name, '${'a' * 45}…');
      expect(
        container.read(activeSessionHeaderProvider)?.$1,
        '${'a' * 45}…',
      );
    });

    testWidgets('a blank draft with nothing attached is a no-op — the '
        'composer keeps its draft (the export:897 guard)', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));

      await tester.enterText(find.byKey(const Key('composer.field')), '   ');
      final AnimatedOpacity send = tester.widget<AnimatedOpacity>(
        find.ancestor(
          of: find.byKey(const Key('composer.send')),
          matching: find.byType(AnimatedOpacity),
        ).first,
      );
      expect(send.opacity, 0.4); // the mock's cursor-not-allowed opacity
      await tester.tap(find.byKey(const Key('composer.send')));
      await tester.pumpAndSettle();

      expect(find.byKey(const ValueKey('chat.typing')), findsNothing);
      final EditableText field =
          tester.widget<EditableText>(find.byType(EditableText).first);
      expect(field.controller.text, '   '); // the draft stays
    });
  });

  group('the typing indicator', () {
    testWidgets('the dots pulse phase-shifted (delay 0/0.18/0.36 over the '
        '1.1 s period), then the whole indicator exits', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      await sendDraft(tester, 'probe');
      await tester.pump(); // mount at t≈0
      expect(find.byType(TypingIndicator), findsOneWidget);
      // The THREE dots.
      expect(find.byKey(const ValueKey('chat.typing')), findsOneWidget);

      List<double> dotScales() {
        final List<double> scales = <double>[];
        Iterable<Element> dots = find
            .descendant(
              of: find.byType(TypingIndicator),
              matching: find.byType(Transform),
            )
            .evaluate();
        for (final Element e in dots) {
          final double s =
              (e.widget as Transform).transform.getMaxScaleOnAxis();
          if ((s - 1).abs() > 0.001) {
            scales.add(s); // the entrance wrapper's is exactly 1.0
          }
        }
        return scales;
      }

      await tester.pump(const Duration(milliseconds: 550));
      final List<double> mid = dotScales();
      expect(mid.length, 3, reason: 'three pulsing dots');
      for (final double s in mid) {
        expect(s, greaterThan(1.0));
        expect(s, lessThanOrEqualTo(1.5));
      }
      expect(mid[0] != mid[1] && mid[1] != mid[2], isTrue,
          reason: 'the mocks delay: i * 0.18 phase-shifts each dot');

      await tester.pump(const Duration(milliseconds: 200));
      final List<double> later = dotScales();
      expect(later[0] != mid[0], isTrue, reason: 'the pulse animates');
      expect(later[1] != mid[1], isTrue);

      // Cross the typing window; the reply lands and the indicator exits.
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();
      expect(find.byType(TypingIndicator), findsNothing);
    });

    testWidgets('the indicator ENTERS rising (y: 8 + fade) and pins its '
        'period', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      expect(find.byType(TypingIndicator), findsNothing);
      await sendDraft(tester, 'probe');
      await tester.pump();
      // The entrance just mounted: opacity 0 at t=0, rising after.
      // The entrance's Opacity lives INSIDE the indicator's build — read
      // the outermost descendant.
      Opacity indicatorEntrance() => tester
          .widgetList<Opacity>(
            find.descendant(
              of: find.byKey(const ValueKey('chat.typing')),
              matching: find.byType(Opacity),
            ),
          )
          .first;
      expect(indicatorEntrance().opacity, 0.0);
      await tester.pump(const Duration(milliseconds: 110));
      expect(
        indicatorEntrance().opacity,
        inExclusiveRange(0.0, 1.0),
      );
      // The mock's pinned pulse period + inter-dot delay.
      expect(TypingIndicator.period, const Duration(milliseconds: 1100));
      expect(TypingIndicator.dotDelay, const Duration(milliseconds: 180));
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();
      expect(find.byType(TypingIndicator), findsNothing);
    });
  });

  group('the message canvas', () {
    testWidgets('the seeded bubbles: user plain + assistant markdown, the '
        'header count, no chips on a filled session', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));

      // ALL of s1's bubbles render through the export's markdown pipe
      // (MessageBubble's renderMarkdown, app.tsx:781) — 4 seed messages.
      final List<MarkdownBody> bodies = tester
          .widgetList<MarkdownBody>(find.byType(MarkdownBody))
          .toList();
      expect(bodies.length, 4);
      final String data = bodies.map((MarkdownBody b) => b.data).join();
      expect(data, contains('*multiple states simultaneously*'));
      expect(data, contains('**Why does this matter?**'));
      // The user messages render as markdown too (the export's
      // MessageBubble renders renderMarkdown for both roles).
      expect(bodies.map((MarkdownBody b) => b.data),
          contains('Can you explain quantum superposition in simple terms?'));
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('4 messages'),
        ),
        findsOneWidget,
      );
      expect(find.byKey(const Key('chat.suggestions')), findsNothing);
      expect(find.byKey(const Key('chat.message-canvas')), findsOneWidget);
    });

    testWidgets('a bubble ENTERS fading + rising exactly the export class '
        '(0.22 s, y: 10, easeOut)', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      // The duration pin (the plan's animation pin; the reference value).
      expect(
        MessageBubble.entranceDuration,
        const Duration(milliseconds: 220),
      );

      await sendDraft(tester, 'entrance probe');
      await tester.pump();
      final Opacity atZero = tester.widget<Opacity>(
        find
            .ancestor(
              of: find.text('entrance probe'),
              matching: find.byType(Opacity),
            )
            .first,
      );
      expect(atZero.opacity, 0.0);
      await tester.pump(const Duration(milliseconds: 110));
      final Opacity mid = tester.widget<Opacity>(
        find
            .ancestor(
              of: find.text('entrance probe'),
              matching: find.byType(Opacity),
            )
            .first,
      );
      expect(mid.opacity, inExclusiveRange(0.0, 1.0));
      await tester.pumpAndSettle();
      expect(
        tester
            .widget<Opacity>(
              find
                  .ancestor(
                    of: find.text('entrance probe'),
                    matching: find.byType(Opacity),
                  )
                  .first,
            )
            .opacity,
        1.0,
      );
    });

    testWidgets('an attachment-bearing send shows the export file chips '
        '(icon kinds + the formatted sizes)', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await tester.tap(find.byKey(const Key('composer.attach-button')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('attach.item-research-notes.pdf')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('composer.attach-button')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('attach.item-portrait.png')));
      await tester.pumpAndSettle();

      await sendDraft(tester, 'these two files');
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();

      // The mock clears the attach list at send; only the bubble's chips
      // carry the names now (app.tsx:762-778).
      expect(find.text('research-notes.pdf'), findsOneWidget);
      expect(find.text('1.9 MB'), findsOneWidget);
      expect(find.text('portrait.png'), findsOneWidget);
      expect(find.text('420.0 KB'), findsOneWidget);
      final ChatSession s1 = sessionsOf(container).byId('s1')!;
      expect(s1.messages.length, 6);
      expect(s1.messages[4].files!.length, 2);
      expect(s1.messages[4].files![0].name, 'research-notes.pdf');
      expect(s1.messages[4].files![1].name, 'portrait.png');
      expect(s1.messages[5].role, MessageRole.assistant);
      expect(aiPool.contains(s1.messages[5].content), isTrue);
    });

    testWidgets('a FAILED turn closes the typing flag (the ChatFailed '
        'terminal — the fold\'s turn.end {error} mapping reaches the view)',
        (WidgetTester tester) async {
      await pumpChat(
        tester,
        const Size(1100, 800),
        chat: const _FailedChatService(),
      );

      await sendDraft(tester, 'doomed');
      expect(find.byKey(const ValueKey('chat.typing')), findsOneWidget);
      await tester.pumpAndSettle();
      expect(find.byKey(const ValueKey('chat.typing')), findsNothing,
          reason: 'the failed terminal closes the indicator like a done');
    });

    testWidgets("the failed turn's line renders DIMMED (the error bubble's "
        'quieter inks; a reply keeps the mock’s card)',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            backgroundColor: const Color(0xFF141322),
            body: Center(
              child: Column(
                children: <Widget>[
                  MessageBubble(
                    key: const Key('probe.reply'),
                    message: Message(
                      id: 'm1',
                      role: MessageRole.assistant,
                      content: 'the reply',
                      timestamp: DateTime(2025),
                    ),
                  ),
                  MessageBubble(
                    key: const Key('probe.failed'),
                    message: Message(
                      id: 'm2',
                      role: MessageRole.assistant,
                      content: 'the turn failed',
                      timestamp: DateTime(2025),
                      error: true,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      // The cards' inks: the failed line is the quieter one, the reply the
      // mock's unchanged card.
      List<BoxDecoration> cardsOf(Key key) => tester
          .widgetList<Container>(find.descendant(
            of: find.byKey(key),
            matching: find.byType(Container),
          ))
          .map((Container c) => c.decoration)
          .whereType<BoxDecoration>()
          .where((BoxDecoration d) => d.color != null)
          .toList();
      expect(
        cardsOf(const Key('probe.reply')).map((d) => d.color),
        contains(const Color(0x1F888DDF)),
      );
      expect(
        cardsOf(const Key('probe.failed')).map((d) => d.color),
        contains(const Color(0x14FFFFFF)),
        reason: 'the failed bubble\'s card is dimmed, not the reply\'s card',
      );
      // The dimmed markdown ink (white/55).
      final MarkdownBody failedBody = tester.widgetList<MarkdownBody>(
          find.byType(MarkdownBody)).firstWhere((MarkdownBody b) =>
          b.data == 'the turn failed');
      expect(failedBody.styleSheet?.p?.color, const Color(0x8CFFFFFF),
          reason: 'the failure\'s words sit quieter than a reply\'s');
    });
  });

  group('the sessions sidebar', () {
    testWidgets('groupSessions buckets visible; the active highlight + the '
        'header follow the row taps', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);

      // The buckets depend on the wall clock (the seed's relative time)
      // — assert the labels the GROUPER produces, not the calendar.
      final List<String> labels = groupSessions(
        sessionsOf(container).list(),
      ).map((SessionGroup g) => g.label).toList();
      for (final String label in labels) {
        expect(find.text(label), findsOneWidget);
      }
      expect(find.byKey(const Key('sidebar.session-s1')), findsOneWidget);
      expect(find.byKey(const Key('sidebar.session-s5')), findsOneWidget);

      // Selection: tap s4 → its row paints the mock's active bg and the
      // header names it (app.tsx:2030-2040, 2131).
      await tester.tap(find.byKey(const Key('sidebar.session-s4')));
      await tester.pumpAndSettle();
      final BoxDecoration decoration = tester
          .widget<Container>(find
              .descendant(
                of: find.byKey(const Key('sidebar.session-s4')),
                matching: find.byType(Container),
              )
              .first)
          .decoration! as BoxDecoration;
      expect(decoration.color, const Color(0x2E888DDF));
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('The Fermi Paradox'),
        ),
        findsOneWidget,
      );

      // The count's singular rule (app.tsx:2136).
      await tester.tap(find.byKey(const Key('sidebar.session-s5')));
      await tester.pumpAndSettle();
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('1 message'),
        ),
        findsOneWidget,
      );
    });

    testWidgets("the delete's tap-target separation: deleting a NON-active "
        'row never selects it', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);

      await tester.tap(find.byKey(const Key('sidebar.delete-s2')));
      await tester.pumpAndSettle();

      final MockSessionsService sessions = sessionsOf(container);
      expect(sessions.byId('s2'), isNull);
      // s1 was active and STAYS active (app.tsx:880-884 only reselects when
      // the DELETED session was the active one).
      expect(sessions.active()!.id, 's1');
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('Quantum Superposition & Qubits'),
        ),
        findsOneWidget,
      );
    });

    testWidgets('deleting the ACTIVE session moves activity to the head of '
        'the remaining list', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);

      await tester.tap(find.byKey(const Key('sidebar.delete-s1')));
      await tester.pumpAndSettle();

      final MockSessionsService sessions = sessionsOf(container);
      expect(sessions.active()!.id, 's2');
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('The Hard Problem of Consciousness'),
        ),
        findsOneWidget,
      );
    });

    testWidgets('NEW chat: the front-insert, the focus, the composer clear, '
        'the suggestion chips', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);

      await tester.enterText(find.byKey(const Key('composer.field')), 'leftover');
      await tester.pump();
      await tester.tap(find.byKey(const Key('sidebar.new-chat')));
      await tester.pumpAndSettle();

      final MockSessionsService sessions = sessionsOf(container);
      final ChatSession head = sessions.list().first;
      expect(head.name, 'New conversation');
      expect(head.messages, isEmpty); // the export's front-insert shape
      expect(
        find.byKey(const Key('chat.suggestions')),
        findsOneWidget,
      );
      // The draft clears (app.tsx:875).
      final EditableText field =
          tester.widget<EditableText>(find.byType(EditableText).first);
      expect(field.controller.text, isEmpty);
      // The plan's addition: the composer's field takes focus.
      expect(field.focusNode.hasFocus, isTrue);
    });

    testWidgets('the rail toggle: the width animates 64 ↔ 288 (0.25 s) and '
        'the chevron rotates 0.18 s', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      expect(
        tester.getSize(find.byKey(shellStaticPaneKey)).width,
        kRailWidth,
      );
      AnimatedRotation chevron() => tester.widget<AnimatedRotation>(
            find.descendant(
              of: find.byKey(const Key('sidebar.rail-toggle')),
              matching: find.byType(AnimatedRotation),
            ),
          );
      expect(chevron().turns, 0);

      await toggleRail(tester);
      expect(
        tester.getSize(find.byKey(shellStaticPaneKey)).width,
        kPaneWidth,
      );
      expect(chevron().turns, 0.25);
      await toggleRail(tester);
      expect(
        tester.getSize(find.byKey(shellStaticPaneKey)).width,
        kRailWidth,
      );
      expect(chevron().turns, 0);
    });

    testWidgets('the MOCK hover also expands/collapses the rail '
        '(onMouseEnter/Leave, app.tsx:1892-1893)', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      final TestGesture mouse = await tester.createGesture(
        kind: PointerDeviceKind.mouse,
        pointer: 8,
      );
      addTearDown(mouse.removePointer);
      await mouse.addPointer(
        location: tester.getCenter(find.byKey(shellStaticPaneKey)),
      );
      await tester.pumpAndSettle();
      expect(
        tester.getSize(find.byKey(shellStaticPaneKey)).width,
        kPaneWidth,
      );
      await mouse.moveTo(const Offset(600, 300));
      await tester.pumpAndSettle();
      expect(
        tester.getSize(find.byKey(shellStaticPaneKey)).width,
        kRailWidth,
      );
    });

    testWidgets('the search toggle: the field filters BEFORE grouping, the '
        'X restores', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);

      await tester.tap(find.byKey(const Key('sidebar.search-open')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('sidebar.search-field')), findsOneWidget);

      await tester.enterText(
        find.byKey(const Key('sidebar.search-input')),
        'fermi',
      );
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('sidebar.session-s4')), findsOneWidget);
      expect(find.byKey(const Key('sidebar.session-s1')), findsNothing);

      await tester.tap(find.byKey(const Key('sidebar.search-close')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('sidebar.search-field')), findsNothing);
      expect(find.byKey(const Key('sidebar.session-s1')), findsOneWidget);
    });
  });

  group('the composer', () {
    testWidgets('SEND: disabled on empty draft, enabled on text only, and '
        'enabled on attachments alone', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));

      double sendOpacity() => tester
          .widget<AnimatedOpacity>(
            find
                .ancestor(
                  of: find.byKey(const Key('composer.send')),
                  matching: find.byType(AnimatedOpacity),
                )
                .first,
          )
          .opacity;

      expect(sendOpacity(), 0.4); // !inputText.trim() && attached.length === 0

      await tester.enterText(find.byKey(const Key('composer.field')), 'hi');
      await tester.pumpAndSettle();
      expect(sendOpacity(), 1.0);

      await tester.enterText(find.byKey(const Key('composer.field')), '');
      await tester.pumpAndSettle();
      expect(sendOpacity(), 0.4);

      await tester.tap(find.byKey(const Key('composer.attach-button')));
      await tester.pumpAndSettle();
      await tester.tap(
        find.byKey(const Key('attach.item-session-transcript.txt')),
      );
      await tester.pumpAndSettle();
      expect(sendOpacity(), 1.0); // attachment-only still sends

      await tester.tap(find.byKey(const Key('composer.send')));
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();

      final MockSessionsService sessions = sessionsOf(container);
      final ChatSession active = sessions.active()!;
      expect(active.messages.length, 6); // the seed pair + this send's pair
      expect(active.messages[4].files!.length, 1);
      expect(active.messages[4].content, isEmpty);
      expect(active.messages[5].role, MessageRole.assistant);
    });

    testWidgets('Enter sends; Shift+Enter does not (the export handleKeyDown, '
        'app.tsx:933-938)', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      final Finder fieldFinder = find.byKey(const Key('composer.field'));

      await tester.tap(fieldFinder);
      await tester.enterText(fieldFinder, 'via enter key');
      await tester.sendKeyEvent(LogicalKeyboardKey.enter);
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();
      expect(sessionsOf(container).active()!.messages.length, 6);

      await toggleRail(tester);
      await tester.tap(find.byKey(const Key('sidebar.new-chat')));
      await tester.pumpAndSettle();
      await tester.tap(fieldFinder);
      await tester.enterText(fieldFinder, 'draft holding');
      await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
      await tester.sendKeyEvent(LogicalKeyboardKey.enter);
      await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
      await tester.pumpAndSettle();
      // No send happened: the session still carries zero messages.
      expect(sessionsOf(container).active()!.messages, isEmpty);
    });

    testWidgets('the attachment sheet: four mock choices, the y:4 entrances, '
        'the chip preview with the remove', (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));

      await tester.tap(find.byKey(const Key('composer.attach-button')));
      await tester.pump(); // the postframe brings the portal up…
      await tester.pump(); // …and the next frame builds the sheet
      expect(find.byKey(const Key('attach.menu')), findsOneWidget);

      // Mid-stagger: at least one row's entrance still under way.
      await tester.pump(const Duration(milliseconds: 80));
      final List<Opacity> firstItemOpacities = find
          .ancestor(
            of: find.byKey(const Key('attach.item-research-notes.pdf')),
            matching: find.byType(Opacity),
          )
          .evaluate()
          .map((Element e) => e.widget as Opacity)
          .toList();
      expect(firstItemOpacities.any((Opacity o) => o.opacity < 1.0), isTrue,
          reason: 'the menu items animate in (y: 4 + fade, 0.18 s)');

      await tester.pumpAndSettle();
      expect(find.byKey(const Key('attach.item-research-notes.pdf')), findsOneWidget);
      expect(find.byKey(const Key('attach.item-portrait.png')), findsOneWidget);
      expect(find.byKey(const Key('attach.item-session-transcript.txt')), findsOneWidget);
      expect(find.byKey(const Key('attach.item-observations.csv')), findsOneWidget);

      // Pick + the chip preview row above the input (app.tsx:2190-2208).
      await tester.tap(find.byKey(const Key('attach.item-portrait.png')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('attach.chip-portrait.png')), findsOneWidget);
      expect(find.text('420.0 KB'), findsOneWidget);
      expect(find.byKey(const Key('attach.menu')), findsNothing);

      // The remove affordance.
      await tester.tap(find.byKey(const Key('attach.remove-portrait.png')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('attach.chip-portrait.png')), findsNothing);
    });
  });

  group('the model picker', () {
    testWidgets('no providers: the unpicked label, the tap is inert',
        (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));

      expect(find.text('No model selected'), findsOneWidget);
      await tester.tap(find.byKey(const Key('model.picker')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('model.menu')), findsNothing);
    });

    testWidgets('the pool: the ENABLED providers + the ENABLED models only, '
        'sections grouped, the selection persists via the settings service',
        (WidgetTester tester) async {
      final _FixtureSettings settings = _pickerFixture();
      await pumpChat(tester, const Size(1100, 800), settings: settings);

      // The first fallback (the export:2262 ?? availableModels[0] ?? null).
      expect(find.text('Aurora · Aurora Large'), findsOneWidget);

      await tester.tap(find.byKey(const Key('model.picker')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('model.menu')), findsOneWidget);
      // Sections: two enabled providers surface; the disabled Nebula does
      // not; aurora-mini (disabled model) does not.
      expect(find.text('Aurora'), findsOneWidget);
      expect(find.text('Hush'), findsOneWidget);
      expect(find.text('Nebula'), findsNothing);
      expect(find.text('aurora-mini'), findsNothing);
      expect(
        find.descendant(
          of: find.byKey(const Key('model.item-p2::mH1')),
          matching: find.text('Hush Pro'),
        ),
        findsOneWidget,
      );

      await tester.tap(find.byKey(const Key('model.item-p2::mH1')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('model.menu')), findsNothing);
      expect(find.text('Hush · Hush Pro'), findsOneWidget);
      expect(settings.selectedModelKey, 'p2::mH1');
    });
  });

  group('the suggestion chips + the empty state', () {
    testWidgets('the chips fill the composer (the export setInputText), so '
        'one tap + one send runs the whole flow on a fresh session',
        (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      await toggleRail(tester);
      await tester.tap(find.byKey(const Key('sidebar.new-chat')));
      await tester.pump();

      await tester.tap(
        find.byKey(const Key('chat.suggestion-0')),
      );
      await tester.pumpAndSettle();
      final EditableText field =
          tester.widget<EditableText>(find.byType(EditableText).first);
      expect(field.controller.text, suggestions.first);

      await tester.tap(find.byKey(const Key('composer.send')));
      await tester.pump(const Duration(milliseconds: 2000));
      await tester.pumpAndSettle();

      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('Explain the Fermi paradox'),
        ),
        findsOneWidget,
      );
      final MockSessionsService sessions = sessionsOf(container);
      final ChatSession fresh = sessions.active()!;
      expect(fresh.messages.length, 2);
      expect(fresh.name, 'Explain the Fermi paradox');
      // The suggestion chips are GONE now (the session is not empty).
      expect(find.byKey(const Key('chat.suggestions')), findsNothing);
    });

    testWidgets('the empty state (no sessions): the prompt copy + the '
        'chips + the header’s fallback string', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpChat(tester, const Size(1100, 800));
      for (final ChatSession s in sessionsOf(container).list()) {
        container.read(sessionsProvider).delete(s.id);
      }
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('chat.empty-canvas')), findsOneWidget);
      expect(find.text('New Conversation'), findsOneWidget);
      expect(
        find.text(
          'What would you like to ponder today? Ask anything — explanations, analysis, ideas, or creative exploration.',
        ),
        findsOneWidget,
      );
      expect(find.byKey(const Key('chat.suggestions')), findsOneWidget);
      expect(find.text('Earlier'), findsNothing);
    });
  });

  group('the responsive shapes', () {
    // The whole chat surface re-lays out cleanly at TIGHT sizes too — a
    // layout exception anywhere in the tree fails the test.
    for (final Size size in <Size>[
      const Size(500, 400),
      const Size(640, 480),
      const Size(750, 500),
    ]) {
      testWidgets('no layout exceptions at $size', (WidgetTester tester) async {
        await pumpChat(tester, size);
      });
    }

    testWidgets('1100 dp: the header in the page + the canvas + the composer '
        'bottom bar with the model picker and the disclaimer',
        (WidgetTester tester) async {
      await pumpChat(tester, const Size(1100, 800));
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('Quantum Superposition & Qubits'),
        ),
        findsOneWidget,
      );
      expect(find.byKey(const Key('chat.message-canvas')), findsOneWidget);
      expect(find.byKey(const Key('model.picker')), findsOneWidget);
      expect(
        find.text('Pondr may make mistakes — verify important information.'),
        findsOneWidget,
      );
      expect(find.byKey(shellMenuButtonKey), findsNothing);
    });

    testWidgets('750 dp: the header rides the shell’s top bar with the menu '
        'button; the drawer opens, a selection closes it',
        (WidgetTester tester) async {
      await pumpChat(tester, const Size(750, 800));
      expect(find.byKey(shellMenuButtonKey), findsOneWidget);
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('Quantum Superposition & Qubits'),
        ),
        findsOneWidget,
      );
      // The canvas + composer exist below the header.
      expect(find.byKey(const Key('composer.field')), findsOneWidget);

      await tester.tap(find.byKey(shellMenuButtonKey));
      await tester.pumpAndSettle();
      expect(tester.getTopLeft(find.byKey(shellDrawerPaneKey)).dx, 0.0);
      expect(
        find.descendant(
          of: find.byKey(shellDrawerPaneKey),
          matching: find.text('The Fermi Paradox'),
        ),
        findsOneWidget,
      );

      await tester.tap(find.byKey(const Key('sidebar.session-s4')));
      await tester.pumpAndSettle();
      // Selection closes the drawer (app.tsx:2030).
      expect(
        tester.getTopLeft(find.byKey(shellDrawerPaneKey)).dx,
        lessThan(0),
      );
      expect(
        find.descendant(
          of: find.byKey(const Key('chat.header')),
          matching: find.text('The Fermi Paradox'),
        ),
        findsOneWidget,
      );
    });
  });
}