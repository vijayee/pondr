import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../theme/tokens.dart';
import 'router.dart' show authStateProvider;

/// Sign-out asks first. The mock's footers sign out on a bare tap
/// (`app.tsx:2089` `setView("login")`), but both sites sit in the UI's
/// footer rows — the collapsed rail's 24x24 Settings/LogOut icons are
/// adjacent — and an accidental tap there felt like a spontaneous logout.
/// The owner directive stands above the mockup here: no automatic logout.
Future<void> signOutWithConfirm(BuildContext context, WidgetRef ref) async {
  final bool confirmed = await showDialog<bool>(
        context: context,
        builder: (BuildContext dialogContext) => AlertDialog(
          backgroundColor: PondrTokens.card,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(PondrTokens.radius),
            side: const BorderSide(color: PondrTokens.border),
          ),
          title: const Text('Sign out?'),
          content: const Text(
            'Your sessions stay here; the login screen comes up.',
            style: TextStyle(color: PondrTokens.mutedForeground),
          ),
          actions: <Widget>[
            TextButton(
              key: const Key('sign-out.stay'),
              onPressed: () => dialogContext.pop(false),
              child: const Text('Stay'),
            ),
            TextButton(
              key: const Key('sign-out.confirm'),
              onPressed: () => dialogContext.pop(true),
              child: const Text(
                'Sign out',
                style: TextStyle(color: PondrTokens.destructive),
              ),
            ),
          ],
        ),
      ) ??
      false;
  if (!confirmed || !context.mounted) return;
  ref.read(authStateProvider).signOut();
  context.go('/login');
}