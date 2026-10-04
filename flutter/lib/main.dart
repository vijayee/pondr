import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'theme/app_theme.dart';

void main() {
  runApp(const PondrApp());
}

/// TEMPORARY router: Task 3 replaces this with `lib/app/router.dart` (the full
/// route set + the auth state as a provider). Kept here only so the theme can
/// be run: a single `/login` entry + a not-logged-in redirect stub.
final GoRouter _router = GoRouter(
  initialLocation: '/login',
  redirect: (BuildContext context, GoRouterState state) {
    // Auth stub: nobody is logged in yet, so every path snaps to /login.
    const bool loggedIn = false;
    final bool onLogin = state.matchedLocation == '/login';
    if (!loggedIn && !onLogin) return '/login';
    return null;
  },
  routes: <RouteBase>[
    GoRoute(
      path: '/login',
      builder: (BuildContext context, GoRouterState state) =>
          const _PlaceholderLogin(),
    ),
  ],
);

class PondrApp extends StatelessWidget {
  const PondrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Pondr',
      theme: buildPondrTheme(),
      routerConfig: _router,
      debugShowCheckedModeBanner: false,
    );
  }
}

class _PlaceholderLogin extends StatelessWidget {
  const _PlaceholderLogin();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(body: Center(child: Text('Pondr')));
  }
}