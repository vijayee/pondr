import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:pondr/main.dart';
import 'package:pondr/theme/tokens.dart';
import 'package:pondr/views/auth/auth_background.dart';

void main() {
  // No font fetching at test time — the fallback face is fine here.

  testWidgets('PondrApp boots on /login with the token theme', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const ProviderScope(child: PondrApp()));
    await tester.pumpAndSettle();

    expect(find.byType(Scaffold), findsOneWidget);
    expect(find.byKey(authLoginCardKey), findsOneWidget);
    expect(find.byKey(loginSubmitKey), findsOneWidget);

    final BuildContext context = tester.element(find.byType(Scaffold).first);
    expect(Theme.of(context).scaffoldBackgroundColor, PondrTokens.background);
    expect(Theme.of(context).colorScheme.primary, PondrTokens.primary);
  });
}
