import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

import 'package:pondr/app/router.dart';
import 'package:pondr/app/shell.dart';
import 'package:pondr/main.dart';
import 'package:pondr/views/auth/auth_background.dart';

void main() {
  // Pumps the app at a given logical size with a test-owned AuthState;
  // returns the container so tests drive the auth provider directly.
  Future<ProviderContainer> pumpApp(
    WidgetTester tester,
    Size size, {
    bool loggedIn = false,
  }) async {
    final AuthState auth = AuthState();
    final ProviderContainer container = ProviderContainer(
      overrides: [authStateProvider.overrideWith((_) => auth)],
    );
    addTearDown(container.dispose);
    if (loggedIn) {
      // Sign in BEFORE the pump, so the guard lands on /chat directly.
      auth.signIn();
    }
    await tester.binding.setSurfaceSize(size);
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(
      UncontrolledProviderScope(container: container, child: const PondrApp()),
    );
    await tester.pumpAndSettle();
    return container;
  }

  group('the redirect guard', () {
    testWidgets('not-logged-in on /chat snaps to /login', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800));

      expect(find.byKey(authLoginCardKey), findsOneWidget); // the login card
      expect(find.byKey(shellStaticPaneKey), findsNothing);
    });

    testWidgets('not-logged-in navigating to /settings snaps back to /login', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800));

      final BuildContext context = tester.element(
        find.byKey(authLoginCardKey),
      );
      context.go('/settings');
      await tester.pumpAndSettle();

      expect(find.text('settings — Task 6'), findsNothing);
      expect(find.byKey(authLoginCardKey), findsOneWidget); // still /login
    });

    testWidgets('signing in moves /login to /chat (the mock\'s '
        'handleLogin → setView("chat"))', (WidgetTester tester) async {
      final ProviderContainer container = await pumpApp(
        tester,
        const Size(1100, 800),
      );

      container.read(authStateProvider).signIn();
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('chat.header')), findsOneWidget);
      expect(find.byKey(authLoginCardKey), findsNothing); // left the login page
    });

    testWidgets('an authed viewer on an auth page redirects to /chat', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800), loggedIn: true);

      final BuildContext context = tester.element(
        find.byKey(const Key('chat.header')),
      );
      context.go('/register');
      await tester.pumpAndSettle();

      expect(find.text('register — Task 5'), findsNothing);
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
    });
  });

  group('the adaptive shell', () {
    testWidgets('1100 dp: the static pane + the body slot, no drawer '
        'affordance', (WidgetTester tester,) async {
      await pumpApp(tester, const Size(1100, 800), loggedIn: true);

      // The static sessions pane beside the body slot. The mock BOOTS with
      // the rail collapsed (`useState("sidebarCollapsed", true)`,
      // app.tsx:806) — a 64-dp icon rail.
      expect(find.byKey(shellStaticPaneKey), findsOneWidget);
      expect(tester.getSize(find.byKey(shellStaticPaneKey)).width, kRailWidth);
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
      // The drawer affordance is narrow-screen only (the mock's `lg:hidden`).
      expect(find.byKey(shellMenuButtonKey), findsNothing);
      expect(find.byKey(shellScrimKey), findsNothing);

      // Expanding: the width animates to the pane's 288 dp (the mock's
      // `sidebarCollapsed ? 64 : 288`).
      final BuildContext paneContext =
          tester.element(find.descendant(
        of: find.byKey(shellStaticPaneKey),
        matching: find.byType(Icon),
      ).first);
      ProviderScope.containerOf(paneContext)
          .read(sidebarCollapsedProvider.notifier)
          .toggle();
      await tester.pumpAndSettle();
      expect(tester.getSize(find.byKey(shellStaticPaneKey)).width, kPaneWidth);
    });

    testWidgets('750 dp: the drawer + scrim, hidden until the menu opens', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(750, 800), loggedIn: true);

      expect(find.byKey(shellMenuButtonKey), findsOneWidget);
      expect(find.byKey(shellStaticPaneKey), findsNothing);
      // The drawer pane is mounted but off-canvas (the mock's
      // `-translate-x-full`), and the scrim is inert while closed.
      expect(tester.getTopLeft(find.byKey(shellDrawerPaneKey)).dx, lessThan(0));
      final GestureDetector scrim =
          tester.widget(find.byKey(shellScrimKey)) as GestureDetector;
      final AnimatedOpacity fade =
          tester.widget(find.descendant(
            of: find.byKey(shellScrimKey),
            matching: find.byType(AnimatedOpacity),
          )) as AnimatedOpacity;
      expect(fade.opacity, 0.0);
      expect(scrim.onTap, isNotNull); // wired; IgnorePointer holds it shut

      // The menu opens the drawer AND un-collapses the pane (app.tsx:2120).
      await tester.tap(find.byKey(shellMenuButtonKey));
      await tester.pumpAndSettle();
      expect(
        tester.getTopLeft(find.byKey(shellDrawerPaneKey)).dx,
        0.0,
        reason: 'the mock\'s translate-x-0: the pane fully on-canvas',
      );
      final AnimatedOpacity openedFade =
          tester.widget(find.descendant(
            of: find.byKey(shellScrimKey),
            matching: find.byType(AnimatedOpacity),
          )) as AnimatedOpacity;
      expect(openedFade.opacity, 0.6);

      // Tapping the scrim closes it again (app.tsx:1885).
      await tester.tap(find.byKey(shellScrimKey), warnIfMissed: false);
      await tester.pumpAndSettle();
      expect(tester.getTopLeft(find.byKey(shellDrawerPaneKey)).dx, lessThan(0));
      expect(
        tester.widget<AnimatedOpacity>(find.descendant(
          of: find.byKey(shellScrimKey),
          matching: find.byType(AnimatedOpacity),
        )).opacity,
        0.0,
      );
    });
  });

  group('the route mapping', () {
    testWidgets('the subconscious page shares the chat shell (the sessions '
        'pane stays) and enters through the custom fade/scale transition', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800), loggedIn: true);

      final BuildContext context = tester.element(
        find.byKey(const Key('chat.header')),
      );
      context.go('/subconscious');
      await tester.pumpAndSettle();

      expect(find.text('subconscious — Task 7'), findsOneWidget);
      // The mock renders SubconsciousView inside the chat layout (app.tsx:
      // 2112-2114): the sessions pane stays beside it.
      expect(find.byKey(shellStaticPaneKey), findsOneWidget);
      // The plan's entrance: a custom transitioning page — the 220 ms pin
      // (the default pages transition at 300 ms), fade + scale present.
      final TransitionRoute<Object?> route =
          ModalRoute.of(tester.element(find.text('subconscious — Task 7')))!
              as TransitionRoute<Object?>;
      expect(route.transitionDuration, const Duration(milliseconds: 220));
      expect(
        find.ancestor(
          of: find.text('subconscious — Task 7'),
          matching: find.byType(FadeTransition),
        ),
        findsWidgets,
      );
      expect(
        find.ancestor(
          of: find.text('subconscious — Task 7'),
          matching: find.byType(ScaleTransition),
        ),
        findsWidgets,
      );
    });

    testWidgets('the settings page is OUTSIDE the shell (no sessions pane), '
        'and its back arrow returns to /chat', (WidgetTester tester) async {
      await pumpApp(tester, const Size(1100, 800), loggedIn: true);

      final BuildContext context = tester.element(
        find.byKey(const Key('chat.header')),
      );
      context.go('/settings');
      await tester.pumpAndSettle();

      expect(find.text('settings — Task 6'), findsOneWidget);
      expect(find.byKey(shellStaticPaneKey), findsNothing);

      // The mock's back-to-chat arrow (app.tsx:1254-1259).
      await tester.tap(find.byKey(const Key('settings.back')));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('chat.header')), findsOneWidget);
    });

    testWidgets('the shell\'s footer navigations: the avatar row to /settings '
        'and sign out back to /login', (WidgetTester tester) async {
      final ProviderContainer container =
          await pumpApp(tester, const Size(1100, 800), loggedIn: true);

      // The pane boots collapsed (icon rail) — expand it first.
      ProviderScope.containerOf(
        tester.element(find.byKey(shellStaticPaneKey)),
      ).read(sidebarCollapsedProvider.notifier).toggle();
      await tester.pumpAndSettle();

      await tester.tap(find.byKey(const Key('sidebar.footer-settings')));
      await tester.pumpAndSettle();
      expect(find.text('settings — Task 6'), findsOneWidget);

      // Back home, then sign out through the pane's footer (app.tsx:2089).
      await tester.tap(find.byKey(const Key('settings.back')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('sidebar.sign-out')));
      await tester.pumpAndSettle();

      expect(find.byKey(authLoginCardKey), findsOneWidget); // /login
      expect(container.read(authStateProvider).loggedIn, isFalse);
    });
  });
}