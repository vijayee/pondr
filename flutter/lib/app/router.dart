import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../views/auth/login_view.dart' show LoginView;
import '../views/auth/register_view.dart' show RegisterView;
import '../views/chat/chat_page.dart' show ChatHeader, ChatPage;
import '../views/settings/settings_view.dart' show SettingsView;
import 'shell.dart' show AdaptiveShell;

/// The branch-swapping page — a ZERO-duration transition. The export's view
/// switches (`setView`) are instant EXCEPT the auth card pair's own 0.3 s
/// enter/exit, which lives in the views' [AuthCardShell] (mode="wait",
/// app.tsx:969) — so every whole-page swap here (login, register, the chat
/// SHELL, settings) must not paint its own transition over it.
CustomTransitionPage<void> _instantPage(GoRouterState state, Widget child) {
  return CustomTransitionPage<void>(
    key: state.pageKey,
    child: child,
    transitionDuration: Duration.zero,
    reverseTransitionDuration: Duration.zero,
    transitionsBuilder:
        (_, Animation<double> enter, Animation<double> exit, Widget child) =>
            child,
  );
}

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
  AuthState({bool startLoggedIn = false}) : _loggedIn = startLoggedIn;

  bool _loggedIn;
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
  // The dev boot seam: `flutter run --dart-define=pondr.bootChat=1` starts
  // AUTHED so a smoke run lands on the chat surface directly (production
  // leaves it unset — the guard routes /login first).
  final AuthState notifier = AuthState(
    startLoggedIn: const bool.fromEnvironment('pondr.bootChat'),
  );
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
      // The real auth views (lib/views/auth/): the submit calls
      // authStateProvider's signIn(); the cards' cross-auth switch rides
      // AuthCardShell's mode="wait" exit → context.go.
      GoRoute(
        path: '/login',
        pageBuilder: (BuildContext context, GoRouterState state) =>
            _instantPage(state, const LoginView()),
      ),
      GoRoute(
        path: '/register',
        pageBuilder: (BuildContext context, GoRouterState state) =>
            _instantPage(state, const RegisterView()),
      ),
      // The chat branch's SHELL (lib/app/shell.dart): the sessions pane + the
      // body slot. /settings is deliberately OUTSIDE it — the mock's settings
      // page has no sessions sidebar.
      ShellRoute(
        pageBuilder: (BuildContext context, GoRouterState state, Widget child) {
          if (state.matchedLocation == '/chat') {
            // The chat header is per-view composition (app.tsx:2118-2144):
            // narrow screens it lives in the shell's top-bar slot with the
            // menu button leading.
            return _instantPage(
              state,
              AdaptiveShell(
                topBarBuilder: (BuildContext context, Widget menuButton) =>
                    ChatHeader(menuButton: menuButton),
                child: child,
              ),
            );
          }
          return _instantPage(state, AdaptiveShell(child: child));
        },
        routes: <RouteBase>[
          // The chat canvas (lib/views/chat/chat_page.dart): the header
          // composition, the message canvas, the composer.
          GoRoute(
            path: '/chat',
            builder: (BuildContext context, GoRouterState state) =>
                const ChatPage(),
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
      // Task 6's settings page (lib/views/settings/settings_view.dart): the
      // standalone layout with its own nav aside (app.tsx:1248-1869).
      GoRoute(
        path: '/settings',
        pageBuilder: (BuildContext context, GoRouterState state) =>
            _instantPage(state, const SettingsView()),
      ),
    ],
  );
});

// ─── Branch placeholders (each swapped by its Task listed above) ──────────

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
