import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../theme/tokens.dart';
import '../views/chat/session_sidebar.dart' show SessionPaneContent;

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
/// expansion) landed with Task 4: `SessionPaneContent` from
/// `lib/views/chat/session_sidebar.dart` hosts the shell-scoped navigations
/// the mock puts in the sidebar — View Subconscious (app.tsx:1986-2011), the
/// footer avatar row -> /settings (app.tsx:2063-2097) and its sign out
/// (app.tsx:2088-2093).

/// The mock's chat-layout breakpoint (the plan's 900 dp).
const double kPaneBreakpoint = 900.0;

/// The shell's wide/narrow decision, handed DOWN so a branch's page sees the
/// SAME decision the shell's [LayoutBuilder] made — a page can't re-measure
/// with its own MediaQuery (the test surface and the route constraints
/// disagree; the pane's own width animating makes body-slot width useless
/// too).
class ShellMode extends InheritedWidget {
  const ShellMode({required this.wide, required super.child, super.key});

  /// TRUE at [kPaneBreakpoint] and up.
  final bool wide;

  /// Null when no shell is above (a page pumped standalone falls back to
  /// MediaQuery).
  static bool? wideOf(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<ShellMode>()?.wide;

  @override
  bool updateShouldNotify(ShellMode oldWidget) => oldWidget.wide != wide;
}

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
/// collapsed rail; hovering the rail expands it (that hover lives in the
/// pane content, lib/views/chat/session_sidebar.dart), leaving it collapses
/// back.
class SidebarCollapsed extends Notifier<bool> {
  @override
  bool build() => true;

  void toggle() => state = !state;

  /// The mock's narrow affordance un-collapses the rail as it opens the
  /// drawer (app.tsx:2121).
  void expand() => state = false;

  /// The mock's onMouseLeave returns the rail to collapsed (app.tsx:1893).
  void collapse() => state = true;
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
    // The chat branch's Scaffold role: without a Material the texts fall to
    // the framework's "no-Material" fallback style (`family: monospace` +
    // the double yellow underline — THE debug run's broken face).
    return Material(
      type: MaterialType.transparency,
      child: LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        final bool wide = constraints.maxWidth >= kPaneBreakpoint;
        if (wide) {
          return ShellMode(wide: true, child: _WideShell(child: child));
        }
        return ShellMode(
          wide: false,
          child: _NarrowShell(topBarBuilder: topBarBuilder, child: child),
        );
      },
    ),
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
      clipBehavior: Clip.hardEdge,
      decoration: BoxDecoration(
        color: PondrTokens.sidebar,
        border: Border(
          right: BorderSide(color: PondrTokens.sidebarBorder),
        ),
      ),
      // The content's width NEVER tracks the animating width: it snaps to
      // the destination (like the export's instant React re-render) and the
      // pane CLIPS the interim stick-out — the browser's overflow painting
      // does the same for the export.
      child: OverflowBox(
        alignment: Alignment.topLeft,
        minWidth: width,
        maxWidth: width,
        child: contentBuilder(expanded),
      ),
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
          return SessionPaneContent(expanded: expanded, isDrawer: false);
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
              return SessionPaneContent(expanded: true, isDrawer: true);
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
