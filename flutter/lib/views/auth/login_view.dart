import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../app/router.dart' show authStateProvider;
import 'auth_background.dart';

/// The export's login card (`app.tsx:971-1044`): the brand lockup, the
/// `Welcome back` heading, the username + password fields (the eye toggle,
/// `app.tsx:1019-1024`) and the `Sign In` submit — whose press is the export's
/// `active:scale-[0.98]` (`app.tsx:1030`) and whose success is the export's
/// `handleLogin` — a bare `setView("chat")` with NO validation
/// (`app.tsx:855-858`): whatever the fields hold, the submit signs in and the
/// router's guard lands the app on /chat.
class LoginView extends ConsumerWidget {
  const LoginView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      body: AuthBackground(
        child: AuthCardShell(
          cardKey: authLoginCardKey,
          builder: (BuildContext context, SwitchTo switchTo) {
            return AuthCard(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  AuthBrand(gapBelow: 32), // mb-8
                  const AuthHeading(
                    title: 'Welcome back',
                    gapBelow: 24, // mb-6
                  ),
                  // `space-y-4` (app.tsx:995) — and the `<form onSubmit>`'s
                  // Enter semantics (app.tsx:995): every field's Enter runs
                  // the form's handler.
                  AuthField(
                    label: 'Username',
                    hintText: 'your_username',
                    fieldKey: loginUsernameKey,
                    autofillHints: const <String>[AutofillHints.username],
                    onSubmitted: (_) => ref.read(authStateProvider).signIn(),
                  ),
                  const SizedBox(height: 16),
                  AuthPasswordField(
                    label: 'Password',
                    hintText: '••••••••',
                    fieldKey: loginPasswordKey,
                    eyeKey: loginEyeKey,
                    autofillHints: const <String>[
                      AutofillHints.password, // autoComplete="current-password"
                    ],
                    onSubmitted: (_) => ref.read(authStateProvider).signIn(),
                    lastField: true,
                  ),
                  const SizedBox(height: 4), // mt-1
                  AuthSubmitButton(
                    label: 'Sign In',
                    key: loginSubmitKey,
                    onPressed: () =>
                        ref.read(authStateProvider).signIn(), // handleLogin
                  ),
                  AuthFooter(
                    premise: "Don't have an account? ",
                    link: 'Create one',
                    linkKey: loginSwitchKey,
                    // The mock's setView("register") crossfade — the exit
                    // plays, THEN the route swap (AnimatePresence mode="wait").
                    onLink: () => switchTo('/register'),
                    gapAbove: 24, // mt-6
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
