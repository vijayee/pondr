import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../app/router.dart' show authStateProvider;
import 'auth_background.dart';

/// The export's register card (`app.tsx:1045-1140`): Full Name, Username,
/// Password (the eye toggle) and Confirm Password — the confirm is ALWAYS
/// obscured, with no eye (`app.tsx:1117-1122`) — under the `Create your
/// account` heading. Like the login, the mock carries NO validation and the
/// submit is a bare `setView("chat")` (`app.tsx:860-863`): the router's guard
/// lands the app on /chat whatever the fields hold.
class RegisterView extends ConsumerWidget {
  const RegisterView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      body: AuthBackground(
        child: AuthCardShell(
          cardKey: authRegisterCardKey,
          builder: (BuildContext context, SwitchTo switchTo) {
            return AuthCard(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  AuthBrand(gapBelow: 24), // mb-6
                  const AuthHeading(
                    title: 'Create your account',
                    gapBelow: 20, // mb-5
                  ),
                  // `space-y-3.5` (app.tsx:1069) — and the `<form onSubmit>`'s
                  // Enter semantics (app.tsx:1069): every field's Enter runs
                  // the form's handler.
                  AuthField(
                    label: 'Full Name',
                    hintText: 'Ada Lovelace',
                    fieldKey: registerNameKey,
                    onSubmitted: (_) => ref.read(authStateProvider).signIn(),
                  ),
                  const SizedBox(height: 14),
                  AuthField(
                    label: 'Username',
                    hintText: 'ada_lovelace',
                    fieldKey: registerUsernameKey,
                    autofillHints: const <String>[AutofillHints.username],
                    onSubmitted: (_) => ref.read(authStateProvider).signIn(),
                  ),
                  const SizedBox(height: 14),
                  AuthPasswordField(
                    label: 'Password',
                    hintText: '••••••••',
                    fieldKey: registerPasswordKey,
                    eyeKey: registerEyeKey,
                    autofillHints: const <String>[
                      AutofillHints.newPassword, // autoComplete="new-password"
                    ],
                    onSubmitted: (_) => ref.read(authStateProvider).signIn(),
                  ),
                  const SizedBox(height: 14),
                  // The confirm: a plain obscured field, NO eye
                  // (app.tsx:1113-1122).
                  AuthField(
                    label: 'Confirm Password',
                    hintText: '••••••••',
                    fieldKey: registerConfirmKey,
                    obscureText: true,
                    autofillHints: const <String>[AutofillHints.newPassword],
                    onSubmitted: (_) => ref.read(authStateProvider).signIn(),
                    lastField: true,
                  ),
                  const SizedBox(height: 4), // mt-1
                  AuthSubmitButton(
                    label: 'Create Account',
                    key: registerSubmitKey,
                    onPressed: () =>
                        ref.read(authStateProvider).signIn(), // handleRegister
                  ),
                  AuthFooter(
                    premise: 'Already have an account? ',
                    link: 'Sign in',
                    linkKey: registerSwitchKey,
                    onLink: () => switchTo('/login'),
                    gapAbove: 20, // mt-5
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
