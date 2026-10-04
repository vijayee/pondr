import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../theme/tokens.dart';
import 'shell.dart' show AdaptiveShell;

/// The route map.
///
/// PINNED MAPPING FROM THE REFERENCE (mockup_reference/app.tsx): the export's
/// `view` state is a GLOBAL machine over "login" | "register" | "chat" |
/// "settings" | "subconscious" (app.tsx:43,802) — a view SWITCH, not a stack.
/// So every view here is a PAGE route:
/// - `/login`, `/register` — the auth branch RETURNS EARLY before the chat
///   layout (app.tsx:951-1145): standalone full pages, sessions sidebar not
///   visible.
/// - `/settings` — the settings branch RETURNS EARLY too (app.tsx:1248-1869):
///   standalone full page with its own nav aside; the sessions sidebar is not
///   visible. Its back arrow just goes to /chat (app.tsx:1254-1259).
/// - `/chat` + `/subconscious` — the ONLY views under the chat layout's outer
///   flex (app.tsx:1873-2114): the export renders SubconsciousView INSIDE the
///   main area beside the sessions sidebar, and its container paints its own
///   opaque #07061A bg (app.tsx:504) — the chat canvas is NOT visible through
///   it. So the see-through-overlay question dissolves: an opaque page route
///   inside a ShellRoute shared with /chat is the honest 1:1 shape, with a
///   fade + slight scale as the plan's requested entrance (the plan, Task 3).
/// Placeholders carry the Task number that swaps them in.

/// The export's auth "state": there is no account system in the mock —
/// handleLogin/handleRegister just setView("chat") (app.tsx:855-863). This is
/// that boolean as a provider; the notifier doubles as go_router's
/// refreshListenable so flipping it re-evaluates the guard. (ChangeNotifier
/// rather than riverpod's Notifier because [GoRouter.refreshListenable]
/// wants a [Listenable].)
class AuthState extends ChangeNotifier {
  bool _loggedIn = false;
  bool get loggedIn => _loggedIn;

  void signIn() {
    _loggedIn = true;
    notifyListeners();
  }

  void signOut() {
    _loggedIn = false;
    notifyListeners();
  }
}

final authStateProvider = Provider<AuthState>((ref) {
  final AuthState notifier = AuthState();
  ref.onDispose(notifier.dispose);
  return notifier;
});

/// The app's one router: the guard + the five page routes.
final routerProvider = Provider<GoRouter>((ref) {
  final AuthState auth = ref.watch(authStateProvider);
  return GoRouter(
    initialLocation: '/login',
    refreshListenable: auth,
    redirect: (BuildContext context, GoRouterState state) {
      final String path = state.matchedLocation;
      if (!auth.loggedIn) {
        // The mock's only unauthed views are the two auth pages; everything
        // else snaps to /login (the plan's guard).
        final bool onAuthBranch = path == '/login' || path == '/register';
        return onAuthBranch ? null : '/login';
      }
      // The mock's handleLogin/handleRegister land on the chat view
      // (app.tsx:856,862) — an authed viewer sitting on an auth page moves
      // on to /chat.
      if (path == '/login' || path == '/register') return '/chat';
      return null;
    },
    routes: <RouteBase>[
      // Task 5 swaps in the real auth views (login/register_view.dart); they
      // call authStateProvider's signIn().
      GoRoute(
        path: '/login',
        builder: (BuildContext context, GoRouterState state) =>
            const LoginPlaceholder(),
      ),
      GoRoute(
        path: '/register',
        builder: (BuildContext context, GoRouterState state) =>
            const RegisterPlaceholder(),
      ),
      // The chat branch's SHELL (lib/app/shell.dart): the sessions pane + the
      // body slot. /settings is deliberately OUTSIDE it — the mock's settings
      // page has no sessions sidebar.
      ShellRoute(
        builder: (BuildContext context, GoRouterState state, Widget child) =>
            AdaptiveShell(child: child),
        routes: <RouteBase>[
          // Task 4 swaps the body slot for chat_page.dart (the header, the
          // message canvas, the composer). The mock's chat header is per-view
          // composition, not shell scaffolding (app.tsx:2118-2144).
          GoRoute(
            path: '/chat',
            builder: (BuildContext context, GoRouterState state) =>
                const ChatBodyPlaceholder(),
          ),
          // Task 7 swaps the body slot for subconscious_view.dart + the sim.
          GoRoute(
            path: '/subconscious',
            pageBuilder: (BuildContext context, GoRouterState state) =>
                CustomTransitionPage<Object?>(
                  key: state.pageKey,
                  child: const SubconsciousBodyPlaceholder(),
                  opaque: true,
                  transitionDuration: const Duration(milliseconds: 220),
                  transitionsBuilder: (
                    BuildContext context,
                    Animation<double> animation,
                    Animation<double> secondaryAnimation,
                    Widget child,
                  ) {
                    final Animation<double> curve = CurvedAnimation(
                      parent: animation,
                      curve: Curves.easeOut,
                    );
                    return FadeTransition(
                      opacity: curve,
                      child: ScaleTransition(
                        scale: Tween<double>(
                          begin: 0.98,
                          end: 1.0,
                        ).animate(curve),
                        child: child,
                      ),
                    );
                  },
                ),
          ),
        ],
      ),
      // Task 6 swaps in settings_view.dart.
      GoRoute(
        path: '/settings',
        builder: (BuildContext context, GoRouterState state) =>
            const SettingsBodyPlaceholder(),
      ),
    ],
  );
});

// ─── Branch placeholders (each swapped by its Task listed above) ──────────

/// Task 4 swaps this for the real chat canvas.
class ChatBodyPlaceholder extends StatelessWidget {
  const ChatBodyPlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    return const ColoredBox(
      color: PondrTokens.background,
      child: Center(child: Text('chat canvas — Task 4')),
    );
  }
}

/// Task 7 swaps this for the real subconscious sim. Its own opaque bg, per
/// the export's container (app.tsx:504).
class SubconsciousBodyPlaceholder extends StatelessWidget {
  const SubconsciousBodyPlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    return const ColoredBox(
      color: Color(0xFF07061A),
      child: Center(child: Text('subconscious — Task 7')),
    );
  }
}

/// Task 5 swaps this for the real login form.
class LoginPlaceholder extends StatelessWidget {
  const LoginPlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(child: Text('Pondr')),
    );
  }
}

/// Task 5 swaps this for the real register form.
class RegisterPlaceholder extends StatelessWidget {
  const RegisterPlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(body: Center(child: Text('register — Task 5')));
  }
}

/// Task 6 swaps this for the real settings page.
class SettingsBodyPlaceholder extends StatelessWidget {
  const SettingsBodyPlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            const Text('settings — Task 6'),
            const SizedBox(height: 8),
            IconButton(
              key: const Key('settings.back'),
              // The mock's back-to-chat arrow (app.tsx:1254-1259).
              onPressed: () => context.go('/chat'),
              icon: const Icon(Icons.arrow_back),
            ),
          ],
        ),
      ),
    );
  }
}

