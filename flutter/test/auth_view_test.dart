import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_svg/flutter_svg.dart';

import 'package:pondr/app/router.dart' show authStateProvider;
import 'package:pondr/main.dart';
import 'package:pondr/views/auth/auth_background.dart';

/// Task 5's pins: the auth views 1:1 with the export's auth region
/// (`mockup_reference/app.tsx:947-1145`). The mock's forms carry NO
/// validation rules — `handleLogin`/`handleRegister` are bare
/// `setView("chat")` (app.tsx:855-863) — so the submit's success routing is
/// the ONLY submit behaviour, and the tests pin that absence: empty fields
/// sign in straight to /chat. The transition pins are the `AnimatePresence
/// mode="wait"` pair (app.tsx:969-974,1044-1049): out-up 0.3 s THEN in-from-
/// below 0.3 s.
void main() {
  Future<ProviderContainer> pumpApp(WidgetTester tester, Size size) async {
    final ProviderContainer container = ProviderContainer();
    addTearDown(container.dispose);
    await tester.binding.setSurfaceSize(size);
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(
      UncontrolledProviderScope(container: container, child: const PondrApp()),
    );
    await tester.pumpAndSettle();
    return container;
  }

  /// The shell's animated [Opacity] value (the card's enter/exit composite)
  /// read through the card key.
  double cardOpacity(WidgetTester tester, Key cardKey) {
    return tester
        .widget<Opacity>(
          find
              .ancestor(of: find.byKey(cardKey), matching: find.byType(Opacity))
              .first,
        )
        .opacity;
  }

  group('the login card', () {
    testWidgets('boots with the brand svg, the two fields and the heading', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800));

      expect(find.byKey(authLoginCardKey), findsOneWidget);
      expect(find.text('Welcome back'), findsOneWidget);
      expect(find.text('THE PONDER ENGINE'), findsOneWidget);
      expect(
        find.byWidgetPredicate(
          (Widget w) => w is SvgPicture, // the normalizeSvg'd wordmark
        ),
        findsOneWidget,
      );
      // The username + password inputs (the export's two `inputCls` fields).
      expect(find.byType(TextField), findsNWidgets(2));
      // The eye toggle boots HIDDEN (the export's password input
      // `type: "password"`, app.tsx:1012).
      expect(find.byKey(loginEyeKey), findsOneWidget);
      final EditableText pw = tester.widget(
        find.descendant(
          of: find.byKey(loginPasswordKey),
          matching: find.byType(EditableText),
        ),
      );
      expect(pw.obscureText, isTrue);
    });

    testWidgets('the eye toggle reveals and re-hides the password', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800));

      await tester.enterText(find.byKey(loginPasswordKey), 'pondr-secret');
      await tester.pump();

      EditableText pw() => tester.widget(
        find.descendant(
          of: find.byKey(loginPasswordKey),
          matching: find.byType(EditableText),
        ),
      );
      expect(pw().obscureText, isTrue);

      await tester.tap(find.byKey(loginEyeKey));
      await tester.pump();
      expect(pw().obscureText, isFalse); // the export's setShowLoginPw(true)

      await tester.tap(find.byKey(loginEyeKey));
      await tester.pump();
      expect(pw().obscureText, isTrue);
    });

    testWidgets('submit signs in EMPTY fields straight to /chat — the mock'
        ' carries no validation (app.tsx:855-858)', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpApp(
        tester,
        const Size(1100, 800),
      );

      await tester.tap(find.byKey(loginSubmitKey));
      await tester.pumpAndSettle();

      expect(container.read(authStateProvider).loggedIn, isTrue);
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
      expect(find.byKey(authLoginCardKey), findsNothing);
    });

    testWidgets('Enter in a field submits the form — the export\'s '
        '<form onSubmit> semantics (app.tsx:995)', (WidgetTester tester) async {
      final ProviderContainer container = await pumpApp(
        tester,
        const Size(1100, 800),
      );

      await tester.enterText(find.byKey(loginPasswordKey), 'anything');
      await tester.testTextInput.receiveAction(TextInputAction.done);
      await tester.pumpAndSettle();

      expect(container.read(authStateProvider).loggedIn, isTrue);
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
    });
  });

  group('the login ↔ register switch (AnimatePresence mode="wait")', () {
    testWidgets('the exit runs FIRST — the register card is absent at its '
        'midpoint — then the register card enters', (
      WidgetTester tester,
    ) async {
      await pumpApp(tester, const Size(1100, 800));

      // The export's setView("register") (app.tsx:1037): the login card
      // exits (opacity 1→0, y 0→-16) over 0.3 s BEFORE the register card
      // mounts its enter.
      await tester.tap(find.byKey(loginSwitchKey));
      await tester.pump(); // the frame that starts the exit
      await tester.pump(const Duration(milliseconds: 150)); // exit midway

      expect(find.byKey(authRegisterCardKey), findsNothing); // mode="wait"
      final double exitOpacity = cardOpacity(tester, authLoginCardKey);
      expect(exitOpacity, greaterThan(0.0));
      expect(exitOpacity, lessThan(1.0));

      // The exit completes → the route swap → the register card's enter.
      await tester.pumpAndSettle();
      expect(find.byKey(authLoginCardKey), findsNothing);
      expect(find.byKey(authRegisterCardKey), findsOneWidget);
      expect(cardOpacity(tester, authRegisterCardKey), 1.0); // entered
      expect(find.text('Create your account'), findsOneWidget);
    });

    testWidgets('…and back: the register card exits, then the login card '
        'enters', (WidgetTester tester) async {
      await pumpApp(tester, const Size(1100, 800));

      await tester.tap(find.byKey(loginSwitchKey));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(registerSwitchKey));
      await tester.pump(); // flush the tap's gesture frame
      await tester.pump(const Duration(milliseconds: 150));

      expect(find.byKey(authLoginCardKey), findsNothing); // mode="wait"
      final double exitOpacity = cardOpacity(tester, authRegisterCardKey);
      expect(exitOpacity, greaterThan(0.0));
      expect(exitOpacity, lessThan(1.0));

      await tester.pumpAndSettle();
      expect(find.byKey(authLoginCardKey), findsOneWidget);
      expect(find.text('Welcome back'), findsOneWidget);
    });

    testWidgets('the card shells re-run the enter at 0.3 s — the exit never '
        'stages the next view before completing', (WidgetTester tester) async {
      await pumpApp(tester, const Size(1100, 800));

      await tester.tap(find.byKey(loginSwitchKey));
      await tester.pump(); // flush the tap's gesture frame
      await tester.pump(const Duration(milliseconds: 299)); // exit still mid
      expect(find.byKey(authRegisterCardKey), findsNothing);

      await tester.pump(const Duration(milliseconds: 350));
      expect(find.byKey(authRegisterCardKey), findsOneWidget);
      // Mid-enter: past the exit's handoff, not yet settled.
      await tester.pump(const Duration(milliseconds: 150));
      final double entering = cardOpacity(tester, authRegisterCardKey);
      expect(entering, greaterThan(0.0));
      expect(entering, lessThan(1.0));
      await tester.pumpAndSettle();
      expect(cardOpacity(tester, authRegisterCardKey), 1.0);
    });
  });

  group('the register card', () {
    testWidgets('holds the four fields; the confirm is obscured with NO eye; '
        'only the password field carries the eye', (WidgetTester tester) async {
      await pumpApp(tester, const Size(1100, 800));
      await tester.tap(find.byKey(loginSwitchKey));
      await tester.pumpAndSettle();

      expect(find.byKey(authRegisterCardKey), findsOneWidget);
      expect(find.byType(TextField), findsNWidgets(4));
      expect(find.text('Full Name'), findsOneWidget);
      expect(find.text('Username'), findsOneWidget);
      expect(find.text('Password'), findsOneWidget);
      expect(find.text('Confirm Password'), findsOneWidget);
      expect(find.byKey(registerEyeKey), findsOneWidget);

      EditableText pw() => tester.widget(
        find.descendant(
          of: find.byKey(registerPasswordKey),
          matching: find.byType(EditableText),
        ),
      );
      EditableText confirm() => tester.widget(
        find.descendant(
          of: find.byKey(registerConfirmKey),
          matching: find.byType(EditableText),
        ),
      );
      expect(pw().obscureText, isTrue);
      expect(confirm().obscureText, isTrue);

      await tester.tap(find.byKey(registerEyeKey));
      await tester.pump();
      expect(pw().obscureText, isFalse);
      expect(confirm().obscureText, isTrue); // the confirm never toggles
    });

    testWidgets('submit signs in straight to /chat — the mock\'s bare'
        ' setView (app.tsx:860-863)', (WidgetTester tester) async {
      final ProviderContainer container = await pumpApp(
        tester,
        const Size(1100, 800),
      );
      await tester.tap(find.byKey(loginSwitchKey));
      await tester.pumpAndSettle();

      await tester.tap(find.byKey(registerSubmitKey));
      await tester.pumpAndSettle();

      expect(container.read(authStateProvider).loggedIn, isTrue);
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
      expect(find.byKey(authRegisterCardKey), findsNothing);
    });
  });
}
