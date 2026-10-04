import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/sign_out.dart';
import '../../app/shell.dart'
    show sidebarCollapsedProvider, sidebarOpenProvider;
import '../../data/bindings.dart';
import '../../data/models.dart';
import '../../data/services.dart';
import '../../theme/tokens.dart';
import 'chat_state.dart';
import 'common.dart';

/// The sessions pane's content (the export's `<aside>`,
/// `app.tsx:1890-2108`) — the SHELL's pane content at both shapes:
/// [expanded] FALSE is the collapsed 64 dp icon rail, TRUE the 288 dp
/// labelled pane (the drawer boots forced-expanded, with the close X).
///
/// 1:1 notes:
/// - The hover expand/collapse of the rail lives HERE (the aside's
///   `onMouseEnter`/`onMouseLeave`, `app.tsx:1892-1893`) on top of the
///   shell's 0.25 s width animation.
/// - Labels fold away with the 0.2 s width/opacity animation
///   (`app.tsx:1913-1920`, `:1939-1946`).
/// - The rail toggle's chevron rotates 0.18 s — the plan's mandated explicit
///   toggle (the export's rail has none; the hover IS it).
/// - The search field toggles (`app.tsx:1949-1982`); rows are flex, not
///   `ListTile` (the shell's F3 note).
/// - The delete affordance reveals on the row's hover and is its own tap
///   target — the export's `e.stopPropagation()` is the nested-widget
///   property here.
/// Task 3's shell-scoped navigations survive: /subconscious, /settings on
/// the avatar row, and the sign out.
class SessionPaneContent extends ConsumerWidget {
  const SessionPaneContent({super.key, required this.expanded, required this.isDrawer});

  final bool expanded;
  final bool isDrawer;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final List<ChatSession> sessions = ref.watch(sessionsStoreProvider);
    final String? activeId = ref.watch(activeIdProvider);

    // The export's search filter BEFORE grouping (`app.tsx:940-943`).
    final String query = ref.watch(searchQueryProvider).toLowerCase();
    final List<ChatSession> filtered = query.isEmpty
        ? sessions
        : sessions
              .where(
                (ChatSession s) => s.name.toLowerCase().contains(query),
              )
              .toList();
    final List<SessionGroup> groups = groupSessions(filtered);

    return MouseRegion(
      onEnter: (_) => ref.read(sidebarCollapsedProvider.notifier).expand(),
      onExit: (_) => ref.read(sidebarCollapsedProvider.notifier).collapse(),
      child: Material(
        type: MaterialType.transparency,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _TopRow(expanded: expanded, isDrawer: isDrawer),
            _ActionRows(expanded: expanded, isDrawer: isDrawer),
            Expanded(
              child: _SessionsList(
                expanded: expanded,
                groups: groups,
                activeId: activeId,
              ),
            ),
            _SessionsFooter(expanded: expanded),
          ],
        ),
      ),
    );
  }
}

// ─── The top row ───────────────────────────────────────────────────────────

class _TopRow extends ConsumerWidget {
  const _TopRow({required this.expanded, required this.isDrawer});

  final bool expanded;
  final bool isDrawer;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: expanded ? 16 : 9, vertical: 16),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: PondrTokens.sidebarBorder)),
      ),
      child: Row(
        children: <Widget>[
          const SparkAvatar(size: 44, glow: true, bgAlpha: 0),
          const Spacer(),
          // The mock's logo folds away with the rail (the export's
          // `motion.div`, `app.tsx:1913-1920` — the 0.2 s width + opacity
          // animation). FLEXIBLE clamps the folding size against the
          // animating rail width; AnimatedSize clips the stick-out.
          Flexible(
            child: AnimatedSize(
              duration: const Duration(milliseconds: 200),
              curve: Curves.easeOut,
              alignment: Alignment.centerLeft,
              child: expanded
                  ? Padding(
                      padding: const EdgeInsets.only(left: 10),
                      // The export's wordmark `PondrLogo w-[174px] h-[40px]`
                      // (app.tsx:1921).
                      child: PondrWordmark(
                        key: const Key('sidebar.logo'),
                        width: 174,
                        height: 40,
                      ),
                    )
                  : const SizedBox.shrink(),
            ),
          ),
          if (isDrawer) ...<Widget>[
            const SizedBox(width: 8),
            IconButton(
              key: const Key('sidebar.close'),
              tooltip: 'Close',
              icon: const Icon(Icons.close, size: 17, color: Color(0x80FFFFFF)),
              onPressed: () => ref.read(sidebarOpenProvider.notifier).close(),
            ),
          ],
        ],
      ),
    );
  }
}

// ─── The action rows ───────────────────────────────────────────────────────

class _ActionRows extends ConsumerWidget {
  const _ActionRows({required this.expanded, required this.isDrawer});

  final bool expanded;
  final bool isDrawer;

  void _newChat(BuildContext context, WidgetRef ref) {
    // The export's handleNewChat (`app.tsx:865-876`): the front-insert +
    // activation; the draft clears (`:875`); narrow closes the drawer
    // (`:874`); the composer-focus tick is the plan's addition.
    ref.read(sessionsProvider).create();
    ref.read(composerStoreProvider.notifier).setText('');
    ref.read(composerFocusTickProvider.notifier).bump();
    if (isDrawer && ref.read(sidebarOpenProvider)) {
      ref.read(sidebarOpenProvider.notifier).close();
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          _NewChatButton(
            expanded: expanded,
            onTap: () => _newChat(context, ref),
          ),
          const SizedBox(height: 8),
          _SearchControl(expanded: expanded),
          const SizedBox(height: 8),
          _SubconsciousRow(expanded: expanded),
          const SizedBox(height: 8),
          _RailToggleButton(expanded: expanded),
        ],
      ),
    );
  }
}

class _NewChatButton extends StatelessWidget {
  const _NewChatButton({required this.expanded, required this.onTap});

  final bool expanded;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return PressScale(
      pressedScale: 0.97, // active:scale-[0.97]
      child: Material(
        type: MaterialType.button,
        color: PondrTokens.primary,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          key: const Key('sidebar.new-chat'),
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Container(
            padding: EdgeInsets.symmetric(
              horizontal: expanded ? 16 : 4,
              vertical: 10,
            ),
            alignment: Alignment.center,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: <Widget>[
                const Icon(
                  Icons.add,
                  size: 15,
                  color: PondrTokens.primaryForeground,
                ),
                // The label folds with the rail (the export's
                // `motion.span`, `app.tsx:1939-1946`); clamped as above.
                Flexible(
                  child: AnimatedSize(
                    duration: const Duration(milliseconds: 200),
                    curve: Curves.easeOut,
                    alignment: Alignment.centerLeft,
                    child: expanded
                        ? Padding(
                            padding: const EdgeInsets.only(left: 10),
                            child: Text(
                              'New Chat',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                color: PondrTokens.primaryForeground,
                              ),
                            ),
                          )
                        : const SizedBox.shrink(),
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

/// The search control (`app.tsx:1949-1982`).
class _SearchControl extends ConsumerWidget {
  const _SearchControl({required this.expanded});

  final bool expanded;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool showSearch = ref.watch(showSearchProvider);

    if (!expanded) {
      return Center(
        child: IconButton(
          key: const Key('sidebar.search-collapsed'),
          tooltip: 'Search conversations',
          icon: const Icon(Icons.search, size: 13, color: Color(0x80FFFFFF)),
          onPressed: () => ref.read(showSearchProvider.notifier).show(),
        ),
      );
    }
    if (!showSearch) {
      return Material(
        type: MaterialType.transparency,
        child: InkWell(
          key: const Key('sidebar.search-open'),
          onTap: () => ref.read(showSearchProvider.notifier).show(),
          borderRadius: BorderRadius.circular(12),
          hoverColor: PondrTokens.sidebarAccent,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: <Widget>[
                const Icon(Icons.search, size: 13, color: Color(0x80FFFFFF)),
                const SizedBox(width: 10),
                Flexible(
                  child: Text(
                    'Search conversations…',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 14, color: Color(0x80FFFFFF)),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }
    return Container(
      key: const Key('sidebar.search-field'),
      height: 36,
      decoration: BoxDecoration(
        color: const Color(0x1AFFFFFF), // rgba(255,255,255,0.1)
        border: Border.all(color: PondrTokens.border),
        borderRadius: BorderRadius.circular(12),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        children: <Widget>[
          const Icon(Icons.search, size: 13, color: Color(0x66FFFFFF)),
          const SizedBox(width: 8),
          Expanded(
            child: _SearchField(),
          ),
          GestureDetector(
            key: const Key('sidebar.search-close'),
            onTap: () {
              // The export's X (`app.tsx:1961-1964`): the toggle off AND
              // the query cleared.
              ref.read(showSearchProvider.notifier).hide();
              ref.read(searchQueryProvider.notifier).set('');
            },
            child: const Icon(Icons.close, size: 12, color: Color(0x80FFFFFF)),
          ),
        ],
      ),
    );
  }
}

/// The search field itself — a controller so typing filters the sessions
/// (the export's `onChange` filter, `app.tsx:940-943`).
class _SearchField extends ConsumerStatefulWidget {
  @override
  ConsumerState<_SearchField> createState() => _SearchFieldState();
}

class _SearchFieldState extends ConsumerState<_SearchField> {
  final TextEditingController _controller = TextEditingController();

  @override
  void initState() {
    super.initState();
    _controller.text = ref.read(searchQueryProvider);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    ref.listen<String>(searchQueryProvider, (
      String? previous,
      String next,
    ) {
      if (next != _controller.text) {
        // The X clears the query outside this field's onChanged path.
        _controller.text = next;
      }
    });
    return TextField(
      key: const Key('sidebar.search-input'),
      controller: _controller,
      onChanged: (String value) =>
          ref.read(searchQueryProvider.notifier).set(value),
      autofocus: true, // the export's autoFocus
      style: const TextStyle(fontSize: 14, color: Colors.white),
      cursorColor: PondrTokens.ring,
      decoration: const InputDecoration(
        isCollapsed: true,
        filled: false,
        border: InputBorder.none,
        contentPadding: EdgeInsets.zero,
        hintText: 'Search conversations…',
        hintStyle: TextStyle(fontSize: 14, color: Color(0x66FFFFFF)),
      ),
    );
  }
}

/// View Subconscious — the export's two variants (`app.tsx:1985-2011`).
class _SubconsciousRow extends StatelessWidget {
  const _SubconsciousRow({required this.expanded});

  final bool expanded;

  @override
  Widget build(BuildContext context) {
    if (expanded) {
      return Material(
        type: MaterialType.transparency,
        child: InkWell(
          key: const Key('sidebar.subconscious'),
          onTap: () => context.go('/subconscious'),
          borderRadius: BorderRadius.circular(12),
          child: Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0x1AC3ACDA), // rgba(195,172,218,0.1)
              border: Border.all(color: const Color(0x38C3ACDA)), // 0.22
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: <Widget>[
                const Icon(
                  Icons.psychology,
                  size: 12,
                  color: Color(0xFFE2E6FF),
                ),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'View Subconscious',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: Color(0xFFC3ACDA),
                    ),
                  ),
                ),
                // The node-count badge (`app.tsx:1997-2000`).
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0x2E888DDF),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    '${baseNodes.length}',
                    style: const TextStyle(
                      fontSize: 9,
                      fontWeight: FontWeight.w700,
                      color: PondrTokens.primary,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }
    return Center(
      child: IconButton(
        key: const Key('sidebar.subconscious'),
        tooltip: 'View Subconscious',
        onPressed: () => context.go('/subconscious'),
        icon: const Icon(Icons.psychology, size: 13, color: Color(0xFFC3ACDA)),
      ),
    );
  }
}

/// The plan's mandated rail toggle (0.18 s chevron rotation); the export's
/// rail is hover-only — the explicit button rides beside the other rows.
class _RailToggleButton extends ConsumerWidget {
  const _RailToggleButton({required this.expanded});

  final bool expanded;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final Widget chevron = AnimatedRotation(
      duration: const Duration(milliseconds: 180),
      curve: Curves.easeOut,
      turns: expanded ? 0.25 : 0,
      child: const Icon(Icons.chevron_right, size: 16, color: Color(0x80FFFFFF)),
    );
    void toggle() => ref.read(sidebarCollapsedProvider.notifier).toggle();

    if (expanded) {
      return Material(
        type: MaterialType.transparency,
        child: InkWell(
          key: const Key('sidebar.rail-toggle'),
          onTap: toggle,
          borderRadius: BorderRadius.circular(12),
          hoverColor: PondrTokens.sidebarAccent,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: <Widget>[
                chevron,
                const SizedBox(width: 8),
                const Text(
                  'Collapse rail',
                  style: TextStyle(fontSize: 14, color: Color(0x80FFFFFF)),
                ),
              ],
            ),
          ),
        ),
      );
    }
    return Center(
      child: IconButton(
        key: const Key('sidebar.rail-toggle'),
        tooltip: 'Expand rail',
        onPressed: toggle,
        icon: chevron,
      ),
    );
  }
}

// ─── The sessions list ─────────────────────────────────────────────────────

class _SessionsList extends StatelessWidget {
  const _SessionsList({
    required this.expanded,
    required this.groups,
    required this.activeId,
  });

  final bool expanded;
  final List<SessionGroup> groups;
  final String? activeId;

  @override
  Widget build(BuildContext context) {
    if (groups.isEmpty) {
      return expanded
          ? const Center(
              child: Padding(
                padding: EdgeInsets.only(bottom: 24),
                child: Text(
                  'No conversations yet',
                  style: TextStyle(fontSize: 12, color: Color(0x66FFFFFF)),
                ),
              ),
            )
          : const SizedBox.shrink();
    }
    return SingleChildScrollView(
      key: const Key('sidebar.sessions-scroll'),
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          for (int g = 0; g < groups.length; g++) ...<Widget>[
            if (g != 0) const SizedBox(height: 16),
            if (expanded)
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 0, 8, 6),
                child: Text(
                  groups[g].label,
                  style: const TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                    color: Color(0x66FFFFFF), // text-white/40
                  ),
                ),
              ),
            for (final ChatSession session in groups[g].sessions)
              _SessionRow(
                session: session,
                active: session.id == activeId,
                expanded: expanded,
              ),
          ],
        ],
      ),
    );
  }
}

/// One session row (`app.tsx:2027-2054`).
class _SessionRow extends ConsumerStatefulWidget {
  const _SessionRow({
    required this.session,
    required this.active,
    required this.expanded,
  });

  final ChatSession session;
  final bool active;
  final bool expanded;

  @override
  ConsumerState<_SessionRow> createState() => _SessionRowState();
}

class _SessionRowState extends ConsumerState<_SessionRow> {
  bool _hover = false;

  void _select() {
    ref.read(sessionsProvider).select(widget.session.id);
    ref.read(sidebarOpenProvider.notifier).close();
  }

  @override
  Widget build(BuildContext context) {
    final ChatSession session = widget.session;

    if (!widget.expanded) {
      // The icon-only rail row (`app.tsx:2033`), titled with the name.
      return Tooltip(
        message: session.name,
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            key: Key('sidebar.session-${session.id}'),
            onTap: _select,
            borderRadius: BorderRadius.circular(12),
            hoverColor: PondrTokens.sidebarAccent,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 10),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: widget.active ? const Color(0x2E888DDF) : null,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.chat_bubble_outline,
                size: 13,
                color: Color(0x80FFFFFF),
              ),
            ),
          ),
        ),
      );
    }

    return MouseRegion(
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      child: Material(
        type: MaterialType.transparency,
        child: InkWell(
          key: Key('sidebar.session-${session.id}'),
          // The export's row click (`app.tsx:2030`): select + the drawer
          // closes.
          onTap: () {
            ref.read(sessionsProvider).select(session.id);
            ref.read(sidebarOpenProvider.notifier).close();
          },
          borderRadius: BorderRadius.circular(12),
          hoverColor: PondrTokens.sidebarAccent,
          child: Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: widget.active
                  ? const Color(0x2E888DDF) // rgba(136,141,223,0.18)
                  : null,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: <Widget>[
                const Icon(
                  Icons.chat_bubble_outline,
                  size: 13,
                  color: Color(0x80FFFFFF), // opacity-50
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    session.name,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 14,
                      color: widget.active
                          ? Colors.white
                          : (_hover
                              ? const Color(0xE6FFFFFF) // white/90
                              : const Color(0xA6FFFFFF)), // white/65
                    ),
                  ),
                ),
                // The delete's hover reveal — a separate tap target inside
                // the row (the export's nested stopPropagation click).
                AnimatedOpacity(
                  duration: const Duration(milliseconds: 150),
                  opacity: _hover ? 1 : 0,
                  child: IconButton(
                    key: Key('sidebar.delete-${session.id}'),
                    tooltip: 'Delete',
                    visualDensity: VisualDensity.compact,
                    constraints: const BoxConstraints(
                      minWidth: 24,
                      minHeight: 24,
                    ),
                    padding: const EdgeInsets.all(2),
                    icon: const Icon(
                      Icons.delete_outline,
                      size: 13,
                      color: Color(0x66FFFFFF),
                    ),
                    onPressed: () {
                      // The export's handleDeleteSession
                      // (`app.tsx:878-885`): the service applies its
                      // reselect rule; no drawer side effects.
                      ref.read(sessionsProvider).delete(session.id);
                    },
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

// ─── The footer ────────────────────────────────────────────────────────────

/// The user footer (`app.tsx:2062-2107`): expanded, the avatar row opens
/// /settings with the hover-revealed settings/sign-out icons (those stop
/// propagation in the mock); collapsed, an icon-only settings row.
class _SessionsFooter extends ConsumerStatefulWidget {
  const _SessionsFooter({required this.expanded});

  final bool expanded;

  @override
  ConsumerState<_SessionsFooter> createState() => _SessionsFooterState();
}

class _SessionsFooterState extends ConsumerState<_SessionsFooter> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    ref.watch(settingsRevisionProvider); // re-read on Task 6's bumps
    final SettingsService settings = ref.read(settingsProvider);

    if (!widget.expanded) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: const BoxDecoration(
          border: Border(
            top: BorderSide(color: PondrTokens.sidebarBorder),
          ),
        ),
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            onTap: () => context.go('/settings'),
            borderRadius: BorderRadius.circular(12),
            hoverColor: PondrTokens.sidebarAccent,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 8),
              alignment: Alignment.center,
              child: const Icon(
                Icons.settings,
                size: 13,
                color: Color(0x66FFFFFF),
              ),
            ),
          ),
        ),
      );
    }

    return Container(
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: PondrTokens.sidebarBorder)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      child: MouseRegion(
        onEnter: (_) => setState(() => _hover = true),
        onExit: (_) => setState(() => _hover = false),
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            onTap: () => context.go('/settings'),
            borderRadius: BorderRadius.circular(12),
            hoverColor: PondrTokens.sidebarAccent,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
              child: Row(
                children: <Widget>[
                  Container(
                    width: 32,
                    height: 32,
                    decoration: const BoxDecoration(
                      color: Color(0x33888DDF), // bg-primary/20
                      shape: BoxShape.circle,
                    ),
                    alignment: Alignment.center,
                    child: const Icon(
                      Icons.person,
                      size: 13,
                      color: PondrTokens.primary,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Text(
                          settings.displayName.isEmpty
                              ? 'Ada Lovelace'
                              : settings.displayName,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                        Text(
                          '@${settings.username.isEmpty ? 'ada_lovelace' : settings.username}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 12,
                            color: Color(0x80FFFFFF),
                          ),
                        ),
                      ],
                    ),
                  ),
                  // The hover-revealed icons — separate tap targets; the
                  // mock stop-propagates them out of the row's click.
                  AnimatedOpacity(
                    duration: const Duration(milliseconds: 150),
                    opacity: _hover ? 1 : 0,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: <Widget>[
                        IconButton(
                          key: const Key('sidebar.footer-settings'),
                          tooltip: 'Settings',
                          visualDensity: VisualDensity.compact,
                          constraints: const BoxConstraints(
                            minWidth: 24,
                            minHeight: 24,
                          ),
                          padding: const EdgeInsets.all(2),
                          icon: const Icon(
                            Icons.settings,
                            size: 13,
                            color: Color(0x80FFFFFF),
                          ),
                          onPressed: () => context.go('/settings'),
                        ),
                        IconButton(
                          key: const Key('sidebar.sign-out'),
                          tooltip: 'Sign out',
                          visualDensity: VisualDensity.compact,
                          constraints: const BoxConstraints(
                            minWidth: 24,
                            minHeight: 24,
                          ),
                          padding: const EdgeInsets.all(2),
                          icon: const Icon(
                            Icons.logout,
                            size: 13,
                            color: Color(0x80FFFFFF),
                          ),
                          onPressed: () => signOutWithConfirm(context, ref),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}