import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app/router.dart';
import 'theme/app_theme.dart';

void main() {
  runApp(const ProviderScope(child: PondrApp()));
}

/// The app widget: the theme + the router (the plan's Task 3 replaced the
/// temporary Task 1 stub — the route map lives in lib/app/router.dart).
class PondrApp extends ConsumerWidget {
  const PondrApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp.router(
      title: 'Pondr',
      theme: buildPondrTheme(),
      routerConfig: ref.watch(routerProvider),
      debugShowCheckedModeBanner: false,
    );
  }
}