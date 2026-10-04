import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/shell.dart' show ShellMode, kPaneBreakpoint;
import '../../app/router.dart' show authStateProvider;
import '../../data/bindings.dart' show settingsProvider;
import '../../data/services.dart' show SettingsService;
import '../../theme/tokens.dart';
import '../chat/chat_state.dart' show settingsRevisionProvider;
import 'settings_motion.dart' show SectionPresence;
import 'settings_profile.dart' show ProfileSection;
import 'settings_appearance.dart' show AppearanceSection;
import 'settings_providers.dart' show ProvidersSection;
import 'settings_sections.dart';
import 'settings_state.dart';

/// The SETTINGS_NAV's lucide icons (`app.tsx:1157-1164`) — the Flutter pick
/// per nav row.
extension SettingsSectionIcons on SettingsSection {
  IconData get icon => switch (this) {
    SettingsSection.profile => Icons.person_outline,
    SettingsSection.appearance => Icons.palette_outlined,
    SettingsSection.providers => Icons.memory_outlined,
    SettingsSection.notifications => Icons.notifications_outlined,
    SettingsSection.security => Icons.shield_outlined,
    SettingsSection.about => Icons.info_outline,
  };
}

/// The export's SETTINGS region 1:1 (`app.tsx:1248-1869`): the standalone
/// full page — `flex h-screen` on `#0c0b1a` with the LEFT aside (w-64 =
/// 256: the back-to-chat arrow + the user summary + the six-section nav +
/// sign out) and the `overflow-y-auto px-8 py-8` content pane whose sections
/// crossfade in `AnimatePresence mode="wait"` (0.18 s; sections keyed).
///
/// RESPONSIVE (the plan's hard bar; the export's settings region has NO
/// breakpoint of its own — the aside would just squeeze): the page keeps the
/// mock's desktop shape at [kPaneBreakpoint] and up, and on NARROW screens
/// the aside becomes the off-canvas nav drawer with the shell's drawer
/// idiom — the slide `duration-300 ease-out` over a `bg-black/60` scrim
/// fading 0.2 s (lib/app/shell.dart's session-drawer port; the plan's
/// "use the drawer idiom for the nav on narrow").
class SettingsView extends ConsumerWidget {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Material(
      type: MaterialType.transparency,
      child: LayoutBuilder(
        builder: (BuildContext context, BoxConstraints constraints) {
          final bool wide = constraints.maxWidth >= kPaneBreakpoint;
          return ShellMode(
            wide: wide,
            child: ColoredBox(
              color: const Color(
                0xFF0C0B1A,
              ), // style={{ background: "#0c0b1a" }}
              child: wide
                  ? const Row(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: <Widget>[
                        SizedBox(
                          width: 256, // w-64 flex-shrink-0
                          child: SettingsAside(isDrawer: false),
                        ),
                        Expanded(child: SettingsContentPane()),
                      ],
                    )
                  : const _NarrowSettings(),
            ),
          );
        },
      ),
    );
  }
}

/// The narrow shape: the back/title bar on top (the mock's aside header,
/// hoisted so the surface stays reachable), the content pane, and the nav
/// aside as the open-able drawer. The menu affordance is this shape's
/// invention (the mock has none — its narrow settings just squeezes); the
/// drawer idiom is the plan's bar.
class _NarrowSettings extends ConsumerWidget {
  const _NarrowSettings();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool open = ref.watch(
      settingsUiProvider.select((SettingsUi ui) => ui.navOpen),
    );
    return Stack(
      children: <Widget>[
        Column(
          children: <Widget>[
            SizedBox(
              height: 56,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: Row(
                  children: <Widget>[
                    const SettingsBackButton(compact: true),
                    const SizedBox(width: 8),
                    const Text(
                      'Settings',
                      style: TextStyle(
                        fontFamily: PondrTokens.fontDisplay,
                        fontWeight: FontWeight.w700,
                        fontSize: 16, // text-base
                        color: Colors.white,
                      ),
                    ),
                    const Spacer(),
                    IconButton(
                      key: settingsNavMenuKey,
                      icon: const Icon(Icons.menu, size: 20),
                      color: Colors.white70,
                      tooltip: 'Settings menu',
                      onPressed: () =>
                          ref.read(settingsUiProvider.notifier).openNav(),
                    ),
                  ],
                ),
              ),
            ),
            const Expanded(child: SettingsContentPane()),
          ],
        ),
        // The scrim: the shell's `bg-black/60` fading 0.2 s, tap to close.
        Positioned.fill(
          child: IgnorePointer(
            ignoring: !open,
            child: GestureDetector(
              key: settingsScrimKey,
              onTap: () => ref.read(settingsUiProvider.notifier).closeNav(),
              child: AnimatedOpacity(
                opacity: open ? 0.6 : 0.0,
                duration: const Duration(milliseconds: 200),
                child: const ColoredBox(color: Colors.black),
              ),
            ),
          ),
        ),
        // The drawer: the shell's `duration-300 ease-out` slide over the
        // left edge.
        Positioned(
          left: 0,
          top: 0,
          bottom: 0,
          width: 256, // w-64
          child: AnimatedSlide(
            key: settingsDrawerKey,
            offset: open ? Offset.zero : const Offset(-1, 0),
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
            child: const SettingsAside(isDrawer: true),
          ),
        ),
      ],
    );
  }
}

/// The ASIDE (`app.tsx:1251-1320`): the back header, the user summary, the
/// six-section nav and the sign-out footer. [isDrawer] TRUE renders the same
/// widget as the narrow drawer (it must be a full standalone panel either
/// way — the export's aside is exactly that).
class SettingsAside extends ConsumerWidget {
  const SettingsAside({required this.isDrawer, super.key});

  final bool isDrawer;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    ref.watch(settingsRevisionProvider); // the summary re-reads on writes
    final SettingsService settings = ref.read(settingsProvider);
    final SettingsSection section = ref.watch(
      settingsUiProvider.select((SettingsUi ui) => ui.section),
    );

    return DecoratedBox(
      // The aside's own chrome: the #1a1929 fill + the `border-r` (and in the
      // desktop the summary/header blocks' `border-b` below) — the mock's
      // `borderColor: rgba(136,141,223,0.18)` — the sidebarBorder token.
      decoration: const BoxDecoration(
        color: PondrTokens.sidebar, // #1a1929
        border: Border(right: BorderSide(color: PondrTokens.sidebarBorder)),
      ),
      child: Column(
        children: <Widget>[
          // The header: back arrow + title (`app.tsx:1253-1263`). The DRAWER
          // shape drops the arrow — the bar above already carries it.
          if (!isDrawer)
            DecoratedBox(
              decoration: const BoxDecoration(
                border: Border(
                  bottom: BorderSide(color: PondrTokens.sidebarBorder),
                ),
              ),
              child: SizedBox(
                height: 64, // px-4 py-4 around the 32 dp button
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: <Widget>[
                      const SettingsBackButton(),
                      const SizedBox(width: 12), // gap-3
                      const Text(
                        'Settings',
                        style: TextStyle(
                          fontFamily: PondrTokens.fontDisplay,
                          fontWeight: FontWeight.w700,
                          fontSize: 16, // text-base
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          // The user summary (`app.tsx:1265-1282`).
          DecoratedBox(
            decoration: const BoxDecoration(
              border: Border(
                bottom: BorderSide(color: PondrTokens.sidebarBorder),
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(16), // px-4 py-4
              child: Row(
                children: <Widget>[
                  Container(
                    width: 40, // w-10 h-10
                    height: 40,
                    decoration: const BoxDecoration(
                      color: Color(0x38888DDF), // rgba(136,141,223,0.22)
                      shape: BoxShape.circle,
                    ),
                    alignment: Alignment.center,
                    child: const Icon(
                      Icons.person,
                      size: 16,
                      color: PondrTokens.primary, // text-primary
                    ),
                  ),
                  const SizedBox(width: 12), // gap-3
                  Flexible(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: <Widget>[
                        Text(
                          // `{displayName || "Ada Lovelace"}` — the export's
                          // display fallbacks.
                          settings.displayName.trim().isEmpty
                              ? 'Ada Lovelace'
                              : settings.displayName,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontWeight: FontWeight.w600, // font-semibold
                            fontSize: 14, // text-sm
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '@${settings.username.trim().isEmpty ? 'ada_lovelace' : settings.username}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 12, // text-xs
                            color: PondrTokens.primary, // #888ddf
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          // The nav (`app.tsx:1284-1300`).
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 12,
              ), // px-3 py-3
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  for (
                    int i = 0;
                    i < SettingsSection.values.length;
                    i++
                  ) ...<Widget>[
                    if (i > 0) const SizedBox(height: 2), // space-y-0.5
                    SettingsNavItem(
                      section: SettingsSection.values[i],
                      active: section == SettingsSection.values[i],
                    ),
                  ],
                ],
              ),
            ),
          ),
          // The sign-out footer (`app.tsx:1302-1318`).
          DecoratedBox(
            decoration: const BoxDecoration(
              border: Border(
                top: BorderSide(
                  color: Color(0x2E888DDF), // rgba(136,141,223,0.18)
                ),
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(12), // px-3 py-3
              child: _SignOutButton(),
            ),
          ),
        ],
      ),
    );
  }
}

/// The back-to-chat arrow — the export's `w-8 h-8 rounded-lg` button on
/// `rgba(136,141,223,0.15)` with the 15 dp ArrowLeft and the
/// `hover:scale-105` (`app.tsx:1253-1259`); the tap is `setView("chat")`.
class SettingsBackButton extends StatefulWidget {
  const SettingsBackButton({this.compact = false, super.key});

  /// TRUE in the narrow bar: the same arrow, unboxed (the top bar already
  /// owns the chrome) — still `context.go('/chat')`.
  final bool compact;

  @override
  State<SettingsBackButton> createState() => _SettingsBackButtonState();
}

class _SettingsBackButtonState extends State<SettingsBackButton> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    if (widget.compact) {
      return IconButton(
        key: settingsBackKey,
        icon: const Icon(Icons.arrow_back, size: 15),
        color: PondrTokens.primary,
        tooltip: 'Back to chat',
        onPressed: () => context.go('/chat'),
      );
    }
    // The `hover:scale-105` reads as a hover-scale affordance.
    return MouseRegion(
      onEnter: (PointerEvent _) => setState(() => _hover = true),
      onExit: (PointerEvent _) => setState(() => _hover = false),
      child: AnimatedScale(
        scale: _hover ? 1.05 : 1.0,
        duration: const Duration(milliseconds: 150), // transition-all
        curve: Curves.easeOut,
        child: Material(
          key: settingsBackKey,
          color: const Color(0x26888DDF), // rgba(136,141,223,0.15)
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(8)), // rounded-lg
          ),
          child: InkWell(
            customBorder: const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(Radius.circular(8)),
            ),
            splashColor: Colors.transparent,
            highlightColor: Colors.transparent,
            hoverColor: Colors.transparent,
            onTap: () => context.go('/chat'), // setView("chat")
            child: const SizedBox(
              width: 32, // w-8 h-8
              height: 32,
              child: Icon(
                Icons.arrow_back,
                size: 15,
                color: PondrTokens.primary,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// One nav row (`app.tsx:1284-1300`): the icon + label, the active state's
/// tinted fill + white text + trailing chevron, the hover shift on the
/// unselected rows.
class SettingsNavItem extends ConsumerStatefulWidget {
  const SettingsNavItem({
    required this.section,
    required this.active,
    super.key,
  });

  final SettingsSection section;
  final bool active;

  @override
  ConsumerState<SettingsNavItem> createState() => _SettingsNavItemState();
}

class _SettingsNavItemState extends ConsumerState<SettingsNavItem> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    final bool active = widget.active;
    final SettingsSection section = widget.section;
    return MouseRegion(
      onEnter: (PointerEvent _) => setState(() => _hover = true),
      onExit: (PointerEvent _) => setState(() => _hover = false),
      child: Material(
        color: active
            ? const Color(0x2E888DDF) // rgba(136,141,223,0.18)
            : _hover
            ? const Color(0x0DFFFFFF) // hover:bg-white/5
            : Colors.transparent,
        borderRadius: BorderRadius.circular(12), // rounded-xl
        child: InkWell(
          key: Key('settings.nav-${section.name}'),
          onTap: () =>
              ref.read(settingsUiProvider.notifier).setSection(section),
          splashColor: Colors.transparent,
          highlightColor: Colors.transparent,
          hoverColor: Colors.transparent,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: Row(
              children: <Widget>[
                Icon(
                  section.icon,
                  size: 14,
                  color: active
                      ? PondrTokens.primary
                      : _hover
                      ? const Color(0xCCFFFFFF) // hover:text-white/80
                      : const Color(0x8CFFFFFF), // text-white/55
                ),
                const SizedBox(width: 12), // gap-3
                Expanded(
                  child: Text(
                    section.label,
                    style: TextStyle(
                      fontWeight: FontWeight.w500, // font-medium
                      fontSize: 14, // text-sm
                      color: active
                          ? Colors.white
                          : _hover
                          ? const Color(0xCCFFFFFF) // hover:text-white/80
                          : const Color(0x8CFFFFFF), // text-white/55
                    ),
                  ),
                ),
                // The trailing chevron ONLY on the active row
                // (`app.tsx:1301`).
                if (active)
                  const Opacity(
                    opacity: 0.6,
                    child: Icon(
                      Icons.chevron_right,
                      size: 12,
                      color: Colors.white,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// The sign-out footer button (`app.tsx:1303-1317`): the export's
/// `setView("login")` — the router's auth guard picks the redirect up.
class _SignOutButton extends ConsumerWidget {
  const _SignOutButton();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Material(
      color: Colors.transparent,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        key: settingsSignOutKey,
        borderRadius: BorderRadius.circular(12),
        splashColor: Colors.transparent,
        highlightColor: Colors.transparent,
        hoverColor: Colors.transparent,
        onTap: () => ref.read(authStateProvider).signOut(), // setView("login")
        child: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Row(
            children: <Widget>[
              Icon(
                Icons.logout,
                size: 14,
                color: Color(0x80FFFFFF),
              ), // white/50
              SizedBox(width: 12), // gap-3
              Text(
                'Sign out',
                style: TextStyle(
                  fontWeight: FontWeight.w500,
                  fontSize: 14,
                  color: Color(0x80FFFFFF), // text-white/50
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// The CONTENT pane (`app.tsx:1321-1322`): `overflow-y-auto px-8 py-8`, the
/// `max-w-xl` (576) column and the section-presence switcher.
class SettingsContentPane extends ConsumerWidget {
  const SettingsContentPane({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final SettingsSection section = ref.watch(
      settingsUiProvider.select((SettingsUi ui) => ui.section),
    );
    return SingleChildScrollView(
      key: settingsContentPaneKey,
      padding: const EdgeInsets.all(32), // px-8 py-8
      child: Align(
        alignment: Alignment.topLeft,
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 576), // max-w-xl
          child: SectionPresence(
            childKey: ValueKey<SettingsSection>(section),
            child: switch (section) {
              SettingsSection.profile => const ProfileSection(),
              SettingsSection.appearance => const AppearanceSection(),
              SettingsSection.providers => const ProvidersSection(),
              SettingsSection.notifications => const NotificationsSection(),
              SettingsSection.security => const SecuritySection(),
              SettingsSection.about => const AboutSection(),
            },
          ),
        ),
      ),
    );
  }
}

/// Test keys — the page's chrome.
const Key settingsBackKey = Key('settings.back');
const Key settingsSignOutKey = Key('settings.sign-out');
const Key settingsContentPaneKey = Key('settings.content-pane');
const Key settingsNavMenuKey = Key('settings.nav-menu');
const Key settingsScrimKey = Key('settings.scrim');
const Key settingsDrawerKey = Key('settings.drawer');
