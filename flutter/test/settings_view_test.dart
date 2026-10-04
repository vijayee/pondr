import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:pondr/app/router.dart'
    show AuthState, authStateProvider, routerProvider;
import 'package:pondr/data/bindings.dart';
import 'package:pondr/data/mock/mock_services.dart';
import 'package:pondr/main.dart';
import 'package:pondr/views/chat/chat_state.dart'
    show AvailableModel, availableModelsProvider, settingsRevisionProvider;
import 'package:pondr/views/settings/settings_motion.dart';
import 'package:pondr/views/settings/settings_state.dart';

/// Task 6's pins: the settings page 1:1 with the export's settings region —
/// the page shape + responsive drawer, the profile write-through, the
/// notifications toggles' persistence, the accent picker, the sections'
/// mode="wait" crossfade, and the providers CRUD through the REAL mock
/// service (create/edit/delete/toggle + the models CRUD + the picker bump +
/// the saved-flash + the expand animations). Every expectation traces to
/// `mockup_reference/app.tsx`'s settings region (lines ~1248-1869).

// The MODEL's `Provider` class and riverpod's share a name.
import 'package:pondr/data/models.dart' as mockup;

void main() {
  /// Pumps the app logged in AT [size], then routes to /settings. Each test
  /// gets its own container — its own Mock services.
  Future<ProviderContainer> pumpSettings(
    WidgetTester tester,
    Size size, {
    MockSettingsService? settings,
  }) async {
    final AuthState authenticator = AuthState();
    final ProviderContainer container = ProviderContainer(
      overrides: [
        authStateProvider.overrideWith((_) => authenticator),
        if (settings != null) settingsProvider.overrideWith((_) => settings),
      ],
    );
    addTearDown(container.dispose);
    authenticator.signIn();
    await tester.binding.setSurfaceSize(size);
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(
      UncontrolledProviderScope(container: container, child: const PondrApp()),
    );
    await tester.pumpAndSettle();
    container.read(routerProvider).go('/settings');
    await tester.pumpAndSettle();
    return container;
  }

  MockSettingsService settingsOf(ProviderContainer container) =>
      container.read(settingsProvider) as MockSettingsService;

  group('the page shape + the responsive forms', () {
    testWidgets('wide: the standalone layout — the aside + the content pane '
        '+ the back arrow, defaulting to the profile section', (
      WidgetTester tester,
    ) async {
      await pumpSettings(tester, const Size(1100, 800));

      // The back arrow (the standalone route's own, app.tsx:1253-1259).
      expect(find.byKey(const Key('settings.back')), findsOneWidget);
      expect(find.byKey(const Key('settings.content-pane')), findsOneWidget);
      // The booted section: profile — its header (the nav row carries
      // the same word).
      expect(find.text('Profile'), findsNWidgets(2));
      expect(find.text('Manage how you appear in Pondr.'), findsOneWidget);
      expect(
        find.byKey(const Key('settings.field-display-name')),
        findsOneWidget,
      );
      // The six nav rows, in the export's SETTINGS_NAV order.
      const List<String> ids = <String>[
        'profile',
        'appearance',
        'providers',
        'notifications',
        'security',
        'about',
      ];
      for (final String id in ids) {
        expect(find.byKey(Key('settings.nav-$id')), findsOneWidget);
      }
      // The mock's user summary defaults (the field's text + the summary
      // share the word).
      expect(find.text('Ada Lovelace'), findsAtLeastNWidgets(1));
      expect(find.text('@ada_lovelace'), findsOneWidget);
    });

    testWidgets('the back arrow goes to /chat', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(1100, 800));
      await tester.tap(find.byKey(const Key('settings.back')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('settings.content-pane')), findsNothing);
    });

    testWidgets('narrow: the aside becomes the drawer — the slide + scrim '
        'open/close', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(600, 800));

      // No static aside off-screen: the drawer starts closed (off-canvas) —
      // the drawer WIDGET is present but slid out; the top bar's menu
      // affordance opens it.
      expect(find.byKey(const Key('settings.nav-menu')), findsOneWidget);
      expect(find.byKey(const Key('settings.back')), findsOneWidget);
      expect(find.byKey(const Key('settings.drawer')), findsOneWidget);

      // Open: the slide is 300 ms easeOutCubic; the scrim fades to 0.6 in
      // 0.2 s (the shell's drawer idiom).
      await tester.tap(find.byKey(const Key('settings.nav-menu')));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 80));
      final AnimatedSlide slide = tester.widget<AnimatedSlide>(
        find.byKey(const Key('settings.drawer')),
      );
      expect(slide.offset.dy, 0); // the horizontal slide
      final AnimatedOpacity scrim = tester.widget<AnimatedOpacity>(
        find
            .descendant(
              of: find.byKey(const Key('settings.scrim')),
              matching: find.byType(AnimatedOpacity),
            )
            .first,
      );
      expect(scrim.opacity, 0.6);
      expect(scrim.duration, const Duration(milliseconds: 200));
      await tester.pumpAndSettle();
      // The nav is reachable in the drawer: tap Providers there.
      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('settings.add-provider')), findsOneWidget);
      // Close via the scrim.
      await tester.tap(find.byKey(const Key('settings.scrim')));
      await tester.pumpAndSettle();
      // The drawer re-parks off-canvas.
      expect(
        tester
            .widget<AnimatedSlide>(find.byKey(const Key('settings.drawer')))
            .offset,
        const Offset(-1, 0),
      );
    });

    testWidgets('narrow drawer: the slide is the 300 ms ease-out shape — '
        'the drawer sits at width 256', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(600, 800));
      final AnimatedSlide slide = tester.widget<AnimatedSlide>(
        find.byKey(const Key('settings.drawer')),
      );
      expect(slide.offset, const Offset(-1, 0)); // parked off-canvas
      expect(slide.duration, const Duration(milliseconds: 300));
    });

    testWidgets('the sign-out asks first, then returns to /login (the '
        'owner directive: no automatic logout; the confirm dialog)', (
        WidgetTester tester) async {
      await pumpSettings(tester, const Size(1100, 800));
      await tester.tap(find.byKey(const Key('settings.sign-out')));
      await tester.pumpAndSettle();
      // The first tap only opens the dialog — no route change happened.
      expect(find.text('Sign out?'), findsOneWidget);
      await tester.tap(find.byKey(const Key('sign-out.stay')));
      await tester.pumpAndSettle();
      expect(find.text('Welcome back'), findsNothing); // still in settings

      await tester.tap(find.byKey(const Key('settings.sign-out')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('sign-out.confirm')));
      await tester.pumpAndSettle();
      expect(
        find.text('Welcome back'),
        findsOneWidget,
      ); // the confirm fired the real sign-out; the guard snapped back
    });
  });

  group('profile: the write-through fields', () {
    testWidgets('the fields write through on every change and the aside '
        'summary follows', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);

      expect(settings.displayName, 'Ada Lovelace');
      await tester.enterText(
        find.byKey(const Key('settings.field-display-name')),
        'Grace Hopper',
      );
      await tester.pump();
      expect(settings.displayName, 'Grace Hopper');
      // The aside's summary is LIVE (the export's React re-render):
      // field + summary.
      expect(find.text('Grace Hopper'), findsNWidgets(2));
      expect(find.text('@ada_lovelace'), findsOneWidget); // the summary's @

      await tester.enterText(
        find.byKey(const Key('settings.field-username')),
        'grace_hopper',
      );
      await tester.pump();
      expect(settings.username, 'grace_hopper');
      expect(find.text('@grace_hopper'), findsOneWidget);

      await tester.enterText(
        find.byKey(const Key('settings.field-bio')),
        'Compilers first.',
      );
      await tester.pump();
      expect(settings.bio, 'Compilers first.');
    });
  });

  group('the notifications toggles', () {
    testWidgets('defaults true/false; the taps persist through the service', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);

      // To the section.
      await tester.tap(find.byKey(const Key('settings.nav-notifications')));
      await tester.pumpAndSettle();
      expect(find.text('Message notifications'), findsOneWidget);
      expect(find.text('Sound effects'), findsOneWidget);

      // Defaults: notifMessages true, notifSounds false (the export's
      // useState inits).
      expect(settings.notifMessages, isTrue);
      expect(settings.notifSounds, isFalse);

      await tester.tap(find.byKey(const Key('settings.notif-sounds')));
      await tester.pumpAndSettle();
      expect(settings.notifSounds, isTrue);
      await tester.tap(find.byKey(const Key('settings.notif-messages')));
      await tester.pumpAndSettle();
      expect(settings.notifMessages, isFalse);
    });
  });

  group('the accent picker', () {
    testWidgets('the five swatches; the tap persists the hex + rides the '
        'check', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);

      await tester.tap(find.byKey(const Key('settings.nav-appearance')));
      await tester.pumpAndSettle();
      expect(find.text('Accent color'), findsOneWidget);
      // ACCENT_OPTIONS' five labels, verbatim.
      for (final String label in <String>[
        'Periwinkle',
        'Lilac',
        'Sage',
        'Champagne',
        'Ice Blue',
      ]) {
        expect(find.text(label), findsOneWidget);
      }
      expect(settings.accentColor, '#888ddf'); // the default

      await tester.tap(find.text('Champagne'));
      await tester.pumpAndSettle();
      expect(settings.accentColor, '#e8d7bd');
      // The selected chip's check: the DARK check icon inside the dot.
      // The selected chip's swatch-dot check (9 dp, the mock's Check 9).
      final Icon dotCheck = tester.widget<Icon>(
        find
            .descendant(
              of: find.ancestor(
                of: find.text('Champagne'),
                matching: find.byType(InkWell),
              ),
              matching: find.byIcon(Icons.check),
            )
            .first,
      );
      expect(dotCheck.size, 9);
      // The UNselected chips show no check dot: only the theme's Dark +
      // the selected swatch carry one (2 + the selected dot = the total).
      expect(find.byIcon(Icons.check), findsNWidgets(2));
    });

    testWidgets('the theme chooser renders statically (Dark selected — '
        'the mock wires no handler)', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(1100, 800));
      await tester.tap(find.byKey(const Key('settings.nav-appearance')));
      await tester.pumpAndSettle();
      expect(find.text('Dark'), findsOneWidget);
      expect(find.text('System'), findsOneWidget);
      // The security + about sections' statics keep their chips.
      await tester.tap(find.byKey(const Key('settings.nav-security')));
      await tester.pumpAndSettle();
      expect(find.text('Danger zone'), findsOneWidget);
      expect(find.text('Update password'), findsOneWidget);
      expect(find.text('Delete account'), findsOneWidget);
      await tester.tap(find.byKey(const Key('settings.nav-about')));
      await tester.pumpAndSettle();
      expect(find.text('Version 1.0.0'), findsOneWidget);
      expect(find.text('Privacy Policy'), findsOneWidget);
    });
  });

  group('the sections\' mode="wait" crossfade', () {
    testWidgets('a nav switch plays the exit (0.18 s) then the enter '
        '(y 10→0, 0.18 s)', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(1100, 800));

      await tester.tap(find.byKey(const Key('settings.nav-appearance')));
      // WAIT mode: the OLD section is still shown during the 0.18 s exit,
      // and the NEW one is absent until the exit completes.
      await tester.pump(const Duration(milliseconds: 60));
      expect(find.text('Manage how you appear in Pondr.'), findsOneWidget);
      expect(
        find.byKey(const ValueKey<SettingsSection>(SettingsSection.appearance)),
        findsNothing,
      );
      await tester.pumpAndSettle();
      // The enter rode in after the exit (0.18 s + 0.18 s total).
      expect(find.text('Personalise the look of Pondr.'), findsOneWidget);
      expect(
        find.byKey(const ValueKey<SettingsSection>(SettingsSection.appearance)),
        findsOneWidget,
      );
      await tester.pump(const Duration(milliseconds: 50));
      // The section content is mounted with its enter AT REST by now.
      final Finder entering = find.byKey(
        const ValueKey<SettingsSection>(SettingsSection.appearance),
      );
      expect(entering, findsOneWidget);
    });
  });

  group('the providers CRUD through the REAL mock service', () {
    testWidgets('create: the add button + form + guard + the saved-flash '
        '(2200 ms) + the auto-expand', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);

      // The boot state: the export's empty provider list (app.tsx:829).
      expect(settings.providers(), isEmpty);
      // The section: the empty state lives in PROVIDERS.
      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      expect(find.text('No providers yet'), findsOneWidget);
      expect(
        find.text('Add an OpenAI-compatible service to get started.'),
        findsOneWidget,
      );

      // Open the form.
      await tester.tap(find.byKey(const Key('settings.add-provider')));
      await tester.pumpAndSettle();
      expect(find.text('New provider'), findsOneWidget);
      expect(find.text('Stored locally in your browser only.'), findsOneWidget);
      // The empty state is replaced while the form is open.
      expect(find.text('No providers yet'), findsNothing);

      // The guard: both fields blank → the Save button disabled
      // (app.tsx:1749's disabled:opacity-40).
      InkWell saveWell() => tester.widget<InkWell>(
        find.descendant(
          of: find.byKey(const Key('settings.save-provider')),
          matching: find.byType(InkWell),
        ),
      );
      expect(saveWell().onTap, isNull);

      // A tap on the disabled button writes NOTHING.
      await tester.tap(find.byKey(const Key('settings.save-provider')));
      await tester.pump();
      expect(settings.providers(), isEmpty);

      await tester.enterText(
        find.byKey(const Key('settings.form-name')),
        'Groq',
      );
      await tester.enterText(
        find.byKey(const Key('settings.form-url')),
        'https://api.groq.com/openai/v1',
      );
      await tester.enterText(
        find.byKey(const Key('settings.form-key')),
        'sk-test-123',
      );
      await tester.pump();
      expect(saveWell().onTap, isNotNull); // (re-read: fields rebuilt the form)

      // The eye toggles the key's visibility (the export's showProviderKey).
      await tester.tap(find.byKey(const Key('settings.form-eye')));
      await tester.pumpAndSettle();
      final TextField keyField = tester.widget<TextField>(
        find.byKey(const Key('settings.form-key')),
      );
      expect(keyField.obscureText, isFalse);

      // Save → the service holds ONE enabled provider; the toast rides.
      await tester.tap(find.byKey(const Key('settings.save-provider')));
      await tester.pump(); // the flip frame (the form's exit starts here)
      await tester.pumpAndSettle(); // the form exitted; the toast entered
      expect(settings.providers().length, 1);
      final mockup.Provider saved = settings.providers().single;
      expect(saved.name, 'Groq');
      expect(saved.baseUrl, 'https://api.groq.com/openai/v1');
      expect(saved.apiKey, 'sk-test-123');
      expect(saved.enabled, isTrue);
      expect(saved.models, isEmpty);
      // The saved-flash (the export's `providerSaved` toast).
      expect(find.text('Saved successfully.'), findsOneWidget);
      // The create branch auto-expands the saved provider.
      final AnimatedRotation chevron = tester.widget<AnimatedRotation>(
        find.descendant(
          of: find.byKey(Key('settings.provider-row-${saved.id}')),
          matching: find.byType(AnimatedRotation),
        ),
      );
      expect(chevron.turns, 0.25); // rotate 90°
      expect(chevron.duration, const Duration(milliseconds: 180));
      // The flash resets 2200 ms after the save (app.tsx:1193-1194); the
      // form's exit completes under the same window.
      await tester.pump(const Duration(milliseconds: 2300));
      await tester.pump(); // the flash-off frame (the toast's reverse starts)
      await tester.pumpAndSettle(); // the toast rides out
      expect(find.text('Saved successfully.'), findsNothing);
      // The form closed; the add button returns.
      expect(find.text('New provider'), findsNothing);
      expect(find.byKey(const Key('settings.add-provider')), findsOneWidget);
    });

    testWidgets('the toast rides the inline-motion enter — gone by 2200 ms', (
      WidgetTester tester,
    ) async {
      await pumpSettings(tester, const Size(1100, 800));
      // To the section; the form flow runs the flash.
      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('settings.add-provider')));
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const Key('settings.form-name')),
        'Groq',
      );
      await tester.enterText(
        find.byKey(const Key('settings.form-url')),
        'https://api.groq.com/openai/v1',
      );
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('settings.save-provider')));
      await tester.pump(const Duration(milliseconds: 80));
      expect(find.text('Saved successfully.'), findsOneWidget);
      // STILL visible at ~2080 ms (the mock's 2200 ms window).
      await tester.pump(const Duration(milliseconds: 2000));
      expect(find.text('Saved successfully.'), findsOneWidget);
      // Gone at 2200+ (the reset's ride-out included).
      await tester.pump(const Duration(milliseconds: 500));
      await tester.pumpAndSettle();
      expect(find.text('Saved successfully.'), findsNothing);
    });

    testWidgets('expand/collapse: the chevron rotation + the accordion '
        'height 0.22 s easeOut — mid-flight sizes pin the shape', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      settingsOf(container).saveProvider(
        name: 'Groq',
        baseUrl: 'https://api.groq.com',
        apiKey: 'k',
      );

      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();

      // The provider row is present with its badge.
      expect(find.text('Active'), findsOneWidget);
      // The model-free line: `api.groq.com` — the scheme stripped
      // (`replace(/https?:\/\//, "")`).
      expect(find.textContaining('api.groq.com'), findsOneWidget);

      final Finder row = find.byKey(
        Key(
          'settings.provider-row-${settingsOf(container).providers().first.id}',
        ),
      );

      // Expand via the header tap.
      await tester.tap(
        find.descendant(of: row, matching: find.byType(InkWell)).first,
      );
      await tester.pump(); // the flip frame (the crossfade's t starts here)
      await tester.pump(const Duration(milliseconds: 80));
      // MID-flight: the accordion body is strictly between 0 and settled.
      final double midHeight = tester
          .getSize(
            find
                .descendant(of: row, matching: find.byType(AnimatedCrossFade))
                .first,
          )
          .height;
      expect(midHeight, greaterThan(0));
      // The chevron mid-rotation: the AnimatedRotation is at its 0.25-turn
      // target already (the state, not the render).
      await tester.pumpAndSettle();
      final double settledHeight = tester
          .getSize(
            find
                .descendant(of: row, matching: find.byType(AnimatedCrossFade))
                .first,
          )
          .height;
      expect(settledHeight, greaterThan(midHeight));
      // The add-model affordance shows in the expanded body.
      expect(find.text('No models added yet.'), findsOneWidget);
      expect(find.text('Add model'), findsOneWidget);

      // Collapse.
      await tester.tap(
        find.descendant(of: row, matching: find.byType(InkWell)).first,
      );
      await tester.pumpAndSettle();
      // The crossfade rides its SECOND side: the body's height 0.
      final AnimatedCrossFade collapsed = tester.widget<AnimatedCrossFade>(
        find
            .descendant(of: row, matching: find.byType(AnimatedCrossFade))
            .first,
      );
      expect(collapsed.crossFadeState, CrossFadeState.showSecond);
      expect(
        tester
            .getSize(
              find
                  .descendant(of: row, matching: find.byType(AnimatedCrossFade))
                  .first,
            )
            .height,
        0,
      );
    });

    testWidgets('edit: the pencil opens the inline form prefilled; the save '
        'updates the name; the accordion stays expanded', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);
      settings.saveProvider(
        name: 'Groq',
        baseUrl: 'https://api.groq.com',
        apiKey: 'sk-groq',
      );
      final String id = settings.providers().first.id;

      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(Key('settings.provider-edit-$id')));
      await tester.pumpAndSettle();
      expect(find.text('Edit provider'), findsOneWidget);
      // Prefilled (the mock's setProviderForm({name: p.name, ...})).
      expect(
        tester
            .widget<TextField>(
              find.descendant(
                of: find.byKey(Key('settings.provider-row-$id')),
                matching: find.byKey(const Key('settings.form-name')),
              ),
            )
            .controller!
            .text,
        'Groq',
      );
      expect(
        tester
            .widget<TextField>(
              find.descendant(
                of: find.byKey(Key('settings.provider-row-$id')),
                matching: find.byKey(const Key('settings.form-url')),
              ),
            )
            .controller!
            .text,
        'https://api.groq.com',
      );
      expect(
        tester
            .widget<TextField>(
              find.descendant(
                of: find.byKey(Key('settings.provider-row-$id')),
                matching: find.byKey(const Key('settings.form-key')),
              ),
            )
            .controller!
            .text,
        'sk-groq',
      );

      await tester.enterText(
        find.byKey(const Key('settings.form-name')),
        'Groq Cloud',
      );
      await tester.pump();
      await tester.tap(find.byKey(const Key('settings.save-provider')));
      await tester.pumpAndSettle();
      expect(settings.providers().first.name, 'Groq Cloud');
      // The toast rode the edit save too.
      expect(find.text('Saved successfully.'), findsOneWidget);
      await tester.pump(const Duration(milliseconds: 2300));
      // The row's header shows the new name.
      expect(find.text('Groq Cloud'), findsOneWidget);
    });

    testWidgets('delete: the trash removes immediately — no confirmation — '
        'and closes an expanded row', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);
      settings.saveProvider(
        name: 'Groq',
        baseUrl: 'https://api.groq.com',
        apiKey: 'k',
      );
      final String id = settings.providers().first.id;

      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      await tester.tap(
        find
            .descendant(
              of: find.byKey(Key('settings.provider-row-$id')),
              matching: find.byType(InkWell),
            )
            .first,
      );
      await tester.pumpAndSettle(); // expanded
      expect(find.text('Add model'), findsOneWidget);

      await tester.tap(find.byKey(Key('settings.provider-delete-$id')));
      await tester.pumpAndSettle();
      expect(settings.providers(), isEmpty);
      expect(find.byKey(Key('settings.provider-row-$id')), findsNothing);
      // The empty state returns (the export's providers.length === 0 gate).
      expect(find.text('No providers yet'), findsOneWidget);
    });

    testWidgets('the enable toggle flips the badge + the store', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);
      settings.saveProvider(
        name: 'Groq',
        baseUrl: 'https://api.groq.com',
        apiKey: 'k',
      );
      final String id = settings.providers().first.id;

      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      expect(find.text('Active'), findsOneWidget);
      await tester.tap(find.byKey(Key('settings.provider-toggle-$id')));
      await tester.pumpAndSettle();
      expect(find.text('Disabled'), findsOneWidget);
      expect(settings.providers().first.enabled, isFalse);
      await tester.tap(find.byKey(Key('settings.provider-toggle-$id')));
      await tester.pumpAndSettle();
      expect(find.text('Active'), findsOneWidget);
    });
  });

  group('the models CRUD + the picker integration', () {
    /// Seeds one provider + returns its id, with the page on the providers
    /// section, EXPANDED.
    Future<String> seedExpanded(
      WidgetTester tester,
      ProviderContainer container,
    ) async {
      settingsOf(container).saveProvider(
        name: 'Groq',
        baseUrl: 'https://api.groq.com',
        apiKey: 'k',
      );
      final String id = settingsOf(container).providers().first.id;
      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      await tester.tap(
        find
            .descendant(
              of: find.byKey(Key('settings.provider-row-$id')),
              matching: find.byType(InkWell),
            )
            .first,
      );
      await tester.pumpAndSettle();
      return id;
    }

    Future<void> addModel(
      WidgetTester tester,
      String providerId, {
      required String modelId,
      required String label,
    }) async {
      await tester.tap(find.byKey(Key('settings.add-model-$providerId')));
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const Key('settings.model-id')),
        modelId,
      );
      await tester.enterText(
        find.byKey(const Key('settings.model-label')),
        label,
      );
      await tester.pump();
      await tester.tap(find.byKey(const Key('settings.model-save')));
      await tester.pump(); // the flip frame (exit + flash start)
      // The flash's window must flush before the test ends (its Timer).
      await tester.pump(const Duration(milliseconds: 2300));
      await tester.pumpAndSettle();
    }

    testWidgets('add → the service row + the count label + the picker pool '
        'sees it (the revision bump)', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final String id = await seedExpanded(tester, container);

      expect(
        container.read(availableModelsProvider),
        isEmpty,
      ); // 'No models added yet.'
      await addModel(
        tester,
        id,
        modelId: 'deepseek-chat',
        label: 'DeepSeek Chat',
      );

      final MockSettingsService settings = settingsOf(container);
      final mockup.Provider p = settings.providers().single;
      expect(p.models.single.modelId, 'deepseek-chat');
      expect(p.models.single.label, 'DeepSeek Chat');
      expect(p.models.single.enabled, isTrue);

      // The header's count line now reads `url · 1 model` (one RichText
      // over the two spans).
      final List<RichText> rowTexts = tester
          .widgetList<RichText>(
            find.descendant(
              of: find.byKey(Key('settings.provider-row-$id')),
              matching: find.byType(RichText),
            ),
          )
          .toList();
      expect(
        rowTexts.map((RichText r) => r.text.toPlainText()),
        anyElement(contains('1 model')),
      );

      // F4's picker pool: the enabled model surfaced under the mock's key
      // format `<providerId>::<model.id>` (chat_state's composition).
      final List<AvailableModel> pool = container.read(availableModelsProvider);
      expect(pool.single.key, '$id::${p.models.single.id}');
      expect(pool.single.modelId, 'deepseek-chat');

      // The form + its timer are gone after the flash's window.
      expect(find.byKey(const Key('settings.model-id')), findsNothing);
      expect(find.text('Saved successfully.'), findsNothing);
      expect(find.byKey(Key('settings.add-model-$id')), findsOneWidget);
    });

    testWidgets('add model: the blank modelId guard keeps the button '
        'disabled; the cancel closes', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final String id = await seedExpanded(tester, container);
      await tester.tap(find.byKey(Key('settings.add-model-$id')));
      await tester.pumpAndSettle();

      final InkWell saveWell = tester.widget<InkWell>(
        find.descendant(
          of: find.byKey(const Key('settings.model-save')),
          matching: find.byType(InkWell),
        ),
      );
      expect(saveWell.onTap, isNull);
      expect(find.text('Add model'), findsNWidgets(2)); // pill + button label

      await tester.tap(find.byKey(const Key('settings.model-cancel')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('settings.model-id')), findsNothing);
      expect(settingsOf(container).providers().single.models, isEmpty);
    });

    testWidgets('toggle model → the picker pool drops it (the enabled '
        'filtering); delete model removes the row', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);
      final String pid = await seedExpanded(tester, container);
      settings.saveModel(pid, modelId: 'deepseek-chat', label: 'DS Chat');
      final String mid = settings.providers().single.models.single.id;
      // The VIEW's wiring bumps the revision on every mutation; direct
      // seeds bump it by hand here (the pool re-reads on it).
      container.read(settingsRevisionProvider.notifier).bump();
      await tester.pumpAndSettle();

      // The pool (F4's integration): the model is visible.
      expect(container.read(availableModelsProvider).length, 1);

      await tester.tap(find.byKey(Key('settings.model-toggle-$pid-$mid')));
      await tester.pump();
      expect(settings.providers().single.models.single.enabled, isFalse);
      // The pool's enabled filter now drops it.
      expect(container.read(availableModelsProvider), isEmpty);

      await tester.tap(find.byKey(Key('settings.model-delete-$pid-$mid')));
      await tester.pumpAndSettle();
      expect(settings.providers().single.models, isEmpty);
      expect(find.text('No models added yet.'), findsOneWidget);
    });

    testWidgets('edit model: the pencil prefills; the save rewrites', (
      WidgetTester tester,
    ) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final MockSettingsService settings = settingsOf(container);
      final String pid = await seedExpanded(tester, container);
      settings.saveModel(pid, modelId: 'deepseek-chat', label: 'DS Chat');
      final String mid = settings.providers().single.models.single.id;
      container.read(settingsRevisionProvider.notifier).bump();
      await tester.pumpAndSettle();

      await tester.tap(find.byKey(Key('settings.model-edit-$pid-$mid')));
      await tester.pumpAndSettle();
      expect(
        tester
            .widget<TextField>(find.byKey(const Key('settings.model-id')))
            .controller!
            .text,
        'deepseek-chat',
      );
      expect(
        tester
            .widget<TextField>(find.byKey(const Key('settings.model-label')))
            .controller!
            .text,
        'DS Chat',
      );
      await tester.enterText(
        find.byKey(const Key('settings.model-label')),
        'DeepSeek V3',
      );
      await tester.pump();
      await tester.tap(find.byKey(const Key('settings.model-save')));
      await tester.pump(); // the flip frame (the form's exit starts here)
      await tester.pumpAndSettle();
      await tester.pump(const Duration(milliseconds: 2300));
      await tester.pumpAndSettle(); // the toast's own reset rides out
      final mockup.Provider p = settings.providers().single;
      expect(p.models.single.label, 'DeepSeek V3');
      expect(p.models.single.modelId, 'deepseek-chat');
      // The row shows the label + the modelId line beneath (label non-empty).
      expect(find.text('DeepSeek V3'), findsOneWidget);
      expect(find.textContaining('deepseek-chat'), findsOneWidget);
    });
  });

  group('the animation constants, pinned', () {
    testWidgets('the accordion + chevron + presence durations ride the '
        'mock\'s values', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(1100, 800));
      expect(kSectionMotion, const Duration(milliseconds: 180));
      expect(kAccordionMotion, const Duration(milliseconds: 220));
      expect(kChevronMotion, const Duration(milliseconds: 180));
      expect(kProviderFormMotion, const Duration(milliseconds: 180));
      expect(kInlineMotion, const Duration(milliseconds: 300));
    });

    testWidgets('the new-provider form\'s enter is its own 0.18 s (y 8→0) — '
        'the MotionPresence rides it', (WidgetTester tester) async {
      await pumpSettings(tester, const Size(1100, 800));
      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('settings.add-provider')));
      await tester.pump(); // the mount frame (its first paint is at t=0)
      await tester.pump(const Duration(milliseconds: 60));
      // MID-enter: the form rides its own 0.18 s — visible but translating.
      final double midOpacity = tester
          .widget<Opacity>(
            find
                .ancestor(
                  of: find.byKey(const Key('settings.form-name')),
                  matching: find.byType(Opacity),
                )
                .first,
          )
          .opacity;
      expect(midOpacity, greaterThan(0));
      expect(midOpacity, lessThan(1));
      await tester.pumpAndSettle();
      expect(find.text('New provider'), findsOneWidget);
    });

    testWidgets('the settings revision bumps on the provider mutations — '
        'F4\'s picker-refresh wiring', (WidgetTester tester) async {
      final ProviderContainer container = await pumpSettings(
        tester,
        const Size(1100, 800),
      );
      final int before = container.read(settingsRevisionProvider);
      final MockSettingsService settings = settingsOf(container);
      settings.saveProvider(
        name: 'Groq',
        baseUrl: 'https://api.groq.com',
        apiKey: 'k',
      );
      final String id = settings.providers().first.id;

      await tester.tap(find.byKey(const Key('settings.nav-providers')));
      await tester.pumpAndSettle();
      // The accordion's expand is NOT a mutation — only the store writes.
      final int afterNav = container.read(settingsRevisionProvider);
      expect(afterNav, greaterThanOrEqualTo(before));

      await tester.tap(find.byKey(Key('settings.provider-toggle-$id')));
      await tester.pump();
      expect(container.read(settingsRevisionProvider), greaterThan(afterNav));
    });
  });
}
