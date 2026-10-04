import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../theme/tokens.dart';
import 'router.dart' show authStateProvider;

/// The adaptive app shell — the chat branch's scaffolding (`lib/app/router.dart`
/// puts ONLY `/chat` + `/subconscious` under it; the mock's settings page is a
/// full page WITHOUT the sessions sidebar, app.tsx:1248-1869).
///
/// PINNED FROM THE REFERENCE (mockup_reference/app.tsx):
/// - The sessions sidebar is ONE widget in both modes: `fixed lg:static` —
///   static in the chat layout's Row on wide screens, sliding over an
///   off-canvas layer on narrow ones (app.tsx:1891-1908).
/// - The rail collapses to 64 dp and expands to 288 dp, animating the width
///   in 0.25 s easeOut (app.tsx:1894: `sidebarCollapsed ? 64 : 288`), and the
///   mock boots collapsed (`useState("sidebarCollapsed", true)`, app.tsx:806)
///   with ICON-ONLY content at the rail (app.tsx:2002-2010, 2098-2106).
/// - The narrow open is a slide (`transition-transform duration-300
///   ease-out`, app.tsx:1900-1902) over a `bg-black/60` scrim that fades in
///   0.2 s (app.tsx:1878-1888); tapping the scrim closes it. The mock's
///   narrow-screen affordance sets the sidebar open AND un-collapses the rail
///   (app.tsx:2120-2121).
/// - The breakpoint is the plan's 900 dp (the mock's Tailwind `lg:` is
///   1024 CSS px; the plan pins 900).
/// - Per the mock, the top bar is PER-VIEW composition (chat header
///   app.tsx:2118-2144, settings app.tsx:1253-1263, subconscious
///   app.tsx:514-535) — the shell renders one on narrow screens only, as the
///   drawer affordance's home; a branch replaces it wholesale through
///   [AdaptiveShell.topBarBuilder] (Task 4 hands the chat header with the
///   menu button wired in).
///
/// The pane CONTENT (the grouped sessions list, the search toggle, the hover
/// expansion) is Task 4 composition; until then [_PaneContent] hosts the
/// shell-scoped navigations the mock puts in the sidebar: View Subconscious
/// (app.tsx:1986-2011), the footer avatar row -> /settings (app.tsx:2063-2097)
/// and its sign out (app.tsx:2088-2093).

/// The mock's chat-layout breakpoint (the plan's 900 dp).
const double kPaneBreakpoint = 900.0;

/// The mock's collapsed rail / expanded pane / drawer widths (app.tsx:1894).
const double kRailWidth = 64.0;
const double kPaneWidth = 288.0;

/// Test keys, so the shell's shape is assertable at both sizes.
const Key shellStaticPaneKey = Key('shell.static-pane');
const Key shellDrawerPaneKey = Key('shell.drawer-pane');
const Key shellMenuButtonKey = Key('shell.menu-button');
const Key shellScrimKey = Key('shell.scrim');

/// The export's `sidebarOpen` state (app.tsx:805) — the narrow drawer.
class SidebarOpen extends Notifier<bool> {
  @override
  bool build() => false;

  void open() => state = true;
  void close() => state = false;
}

final sidebarOpenProvider =
    NotifierProvider<SidebarOpen, bool>(SidebarOpen.new);

/// The export's `sidebarCollapsed` state (app.tsx:806) — boots TRUE, the
/// collapsed rail; hovering the rail expands it (that hover lives with the
/// pane content in Task 4).
class SidebarCollapsed extends Notifier<bool> {
  @override
  bool build() => true;

  void toggle() => state = !state;

  /// The mock's narrow affordance un-collapses the rail as it opens the
  /// drawer (app.tsx:2121).
  void expand() => state = false;
}

final sidebarCollapsedProvider =
    NotifierProvider<SidebarCollapsed, bool>(SidebarCollapsed.new);

/// The shell around the chat branch's body slot. [topBarBuilder] replaces the
/// default narrow top bar; pass `null` (the default) and the stand-in bar —
/// carrying the [shellMenuButtonKey]-keyed button — shows.
class AdaptiveShell extends ConsumerWidget {
  const AdaptiveShell({
    required this.child,
    this.topBarBuilder,
    super.key,
  });

  /// The active branch's page (go_router's shell child).
  final Widget child;

  /// A branch's whole top bar for narrow screens; receives the menu button
  /// (pre-built, opens the drawer) so the composition stays 1:1 with the
  /// mock's header row.
  final Widget Function(BuildContext context, Widget menuButton)?
      topBarBuilder;

  static const double _kTopBarHeight = 56.0;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        final bool wide = constraints.maxWidth >= kPaneBreakpoint;
        if (wide) {
          return _WideShell(child: child);
        }
        return _NarrowShell(topBarBuilder: topBarBuilder, child: child);
      },
    );
  }
}

/// The mock's pane container: 64 dp collapsed / 288 dp, the sidebar bg, a
/// right border (`border-r border-sidebar-border`, app.tsx:1899), animating
/// the width 0.25 s easeOut. [drawer] forces it expanded (the mock's menu
/// button opens the sidebar AND un-collapses, app.tsx:2120-2121).
class _ShellPane extends ConsumerWidget {
  const _ShellPane({required this.drawer, required this.contentBuilder});

  final bool drawer;
  final Widget Function(bool expanded) contentBuilder;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool collapsed =
        drawer ? false : ref.watch(sidebarCollapsedProvider);
    final bool expanded = drawer || !collapsed;
    final double width = expanded ? kPaneWidth : kRailWidth;
    return AnimatedContainer(
      key: drawer ? shellDrawerPaneKey : shellStaticPaneKey,
      duration: const Duration(milliseconds: 250),
      curve: Curves.easeOut,
      width: width,
      decoration: BoxDecoration(
        color: PondrTokens.sidebar,
        border: Border(
          right: BorderSide(color: PondrTokens.sidebarBorder),
        ),
      ),
      child: contentBuilder(expanded),
    );
  }
}

/// The wide shell: rail + body slot — the mock's `flex h-screen` chat layout.
class _WideShell extends ConsumerWidget {
  const _WideShell({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        _ShellPane(drawer: false, contentBuilder: (bool expanded) {
          return _PaneContent(expanded: expanded);
        }),
        Expanded(child: child),
      ],
    );
  }
}

/// The narrow shell: the stand-in top bar with the menu affordance + the body
/// slot, with the sessions pane as the off-canvas drawer + scrim layer.
class _NarrowShell extends ConsumerWidget {
  const _NarrowShell({required this.child, required this.topBarBuilder});

  final Widget child;
  final Widget Function(BuildContext, Widget)? topBarBuilder;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool open = ref.watch(sidebarOpenProvider);
    final Widget body = ColoredBox(color: PondrTokens.background, child: child);

    final Widget menuButton = shellMenuButton(ref);
    final Widget topBar = topBarBuilder != null
        ? topBarBuilder!(context, menuButton)
        : SizedBox(
            height: AdaptiveShell._kTopBarHeight,
            child: Row(
              children: <Widget>[
                menuButton,
                const SizedBox(width: 12),
                const Text('Pondr'),
              ],
            ),
          );

    return Stack(
      children: <Widget>[
        Column(
          children: <Widget>[
            topBar,
            Expanded(child: body),
          ],
        ),
        // The scrim: `fixed inset-0 bg-black/60` fading 0.2 s, tap to close
        // (app.tsx:1878-1888).
        Positioned.fill(
          child: IgnorePointer(
            ignoring: !open,
            child: GestureDetector(
              key: shellScrimKey,
              onTap: ref.read(sidebarOpenProvider.notifier).close,
              child: AnimatedOpacity(
                opacity: open ? 0.6 : 0.0,
                duration: const Duration(milliseconds: 200),
                child: const ColoredBox(color: Colors.black),
              ),
            ),
          ),
        ),
        // The slide: `duration-300 ease-out` (app.tsx:1900-1902); Tailwind's
        // `ease-out` is cubic-bezier(0, 0, 0.2, 1) ≈ Curves.easeOutCubic.
        Positioned(
          left: 0,
          top: 0,
          bottom: 0,
          width: kPaneWidth,
          child: AnimatedSlide(
            offset: open ? Offset.zero : const Offset(-1, 0),
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOutCubic,
            child: _ShellPane(drawer: true, contentBuilder: (bool _) {
              return const _PaneContent(expanded: true);
            }),
          ),
        ),
      ],
    );
  }
}

/// The drawer's opening affordance. The mock's narrow-screen leading button
/// is the ThoughtSpark logo (app.tsx:2120-2128) — the Menu icon stands in
/// until the SVG work (Task 5). It opens the sidebar AND un-collapses the
/// rail, and is handed to the branch's top bar so Task 4's chat header
/// carries it too.
Widget shellMenuButton(WidgetRef ref) {
  return IconButton(
    key: shellMenuButtonKey,
    icon: const Icon(Icons.menu),
    onPressed: () {
      ref.read(sidebarCollapsedProvider.notifier).expand();
      ref.read(sidebarOpenProvider.notifier).open();
    },
  );
}

/// The pane content at BOTH shapes, mirroring the mock's expanded /
/// icon-only branches (app.tsx:1985-2011, 2063-2106). Task 4 swaps this
/// whole widget out for `session_sidebar.dart` — which must keep these
/// shell-scoped navigations: /subconscious, /settings on the avatar row and
/// the sign out.
class _PaneContent extends ConsumerWidget {
  const _PaneContent({required this.expanded});

  /// FALSE = the collapsed rail's icon-only variant (the mock's
  /// `sidebarCollapsed` branch); TRUE = the labeled pane / the drawer.
  final bool expanded;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // View Subconscious (app.tsx:1985-2011).
    final Widget subconsciousEntry = expanded
        ? TextButton.icon(
            onPressed: () => context.go('/subconscious'),
            icon: const Icon(Icons.psychology),
            label: const Text('View Subconscious'),
          )
        : IconButton(
            tooltip: 'View Subconscious',
            onPressed: () => context.go('/subconscious'),
            icon: const Icon(Icons.psychology),
          );

    final Widget newChatEntry = expanded
        ? TextButton.icon(
            onPressed: () => context.go('/chat'),
            icon: const Icon(Icons.add),
            label: const Text('New Chat'),
          )
        : IconButton(
            tooltip: 'New Chat',
            onPressed: () => context.go('/chat'),
            icon: const Icon(Icons.add),
          );

    return Material(
      type: MaterialType.transparency,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const SizedBox(height: 16),
          subconsciousEntry,
          newChatEntry,
          const Expanded(
            child: Center(child: Text('sessions — Task 4')),
          ),
          // The footer avatar row's navigations (app.tsx:2063-2097). The mock
          // builds these as simple flex rows whose labels collapse away —
          // mirrored here (a ListTile dies on the narrow animation widths).
          _PaneFooterButton(
            key: const Key('shell.footer-avatar'),
            expanded: expanded,
            icon: Icons.account_circle,
            label: 'Settings',
            onTap: () => context.go('/settings'),
          ),
          _PaneFooterButton(
            key: const Key('shell.sign-out'),
            expanded: expanded,
            icon: Icons.logout,
            label: 'Sign out',
            onTap: () {
              // The mock's footer sign-out goes back to the login view
              // (app.tsx:2089) — here: clear auth, the guard snaps to /login.
              ref.read(authStateProvider).signOut();
              context.go('/login');
            },
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}

/// One footer row: the icon always, the label only when the pane is expanded
/// (the mock's collapsed branch renders icon-only rows, app.tsx:2098-2106).
class _PaneFooterButton extends StatelessWidget {
  const _PaneFooterButton({
    required this.expanded,
    required this.icon,
    required this.label,
    required this.onTap,
    super.key,
  });

  final bool expanded;
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final Color fg = PondrTokens.primary;
    final Icon iconWidget = Icon(icon, color: fg);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        child: expanded
            ? Row(
                children: <Widget>[
                  iconWidget,
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.only(left: 12),
                      child: Text(
                        label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(color: fg),
                      ),
                    ),
                  ),
                ],
              )
            : Center(child: iconWidget),
      ),
    );
  }
}