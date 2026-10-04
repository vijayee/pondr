import 'package:flutter/material.dart';

import 'tokens.dart';

/// The app's [ThemeData], built from [PondrTokens].
///
/// Inter is the body font and Nunito the display font (h1-h4), following the
/// export's `@layer base` rules in `mockup_reference/styles/globals.css`.
/// FONTS: the two faces are BUNDLED assets (the plan's Task 1 open decision
/// landed on offline determinism — the runtime fetch flaked in a real run);
/// `pubspec.yaml` carries the weights.
ThemeData buildPondrTheme() {
  final ColorScheme scheme = ColorScheme.dark(
    primary: PondrTokens.primary,
    onPrimary: PondrTokens.primaryForeground,
    secondary: PondrTokens.secondary,
    onSecondary: PondrTokens.secondaryForeground,
    surface: PondrTokens.card,
    onSurface: PondrTokens.cardForeground,
    surfaceContainerHighest: PondrTokens.secondary,
    error: PondrTokens.destructive,
    onError: PondrTokens.destructiveForeground,
    outline: PondrTokens.border,
    outlineVariant: PondrTokens.sidebarBorder,
    primaryContainer: PondrTokens.accent,
    onPrimaryContainer: PondrTokens.accentForeground,
    inversePrimary: PondrTokens.accent,
    tertiary: PondrTokens.mutedForeground,
    onTertiary: PondrTokens.muted,
    surfaceTint: PondrTokens.primary,
    surfaceContainerLowest: PondrTokens.background,
    brightness: Brightness.dark,
  );

  const TextTheme base = TextTheme(
    displayLarge: TextStyle(
      fontFamily: 'Inter',
      color: PondrTokens.foreground,
    ),
    displayMedium: TextStyle(
      fontFamily: 'Inter',
      color: PondrTokens.foreground,
    ),
    displaySmall: TextStyle(
      fontFamily: 'Inter',
      color: PondrTokens.foreground,
    ),
    headlineLarge: TextStyle(
      fontFamily: 'Nunito',
      fontWeight: FontWeight.w500, // the export's h1-h4: Nunito 500
      color: PondrTokens.foreground,
    ),
    headlineMedium: TextStyle(
      fontFamily: 'Nunito',
      fontWeight: FontWeight.w500,
      color: PondrTokens.foreground,
    ),
    headlineSmall: TextStyle(
      fontFamily: 'Nunito',
      fontWeight: FontWeight.w500,
      color: PondrTokens.foreground,
    ),
    titleLarge: TextStyle(
      fontFamily: 'Nunito',
      fontWeight: FontWeight.w500,
      color: PondrTokens.foreground,
    ),
    titleMedium: TextStyle(
      fontFamily: 'Nunito',
      fontWeight: FontWeight.w500,
      color: PondrTokens.foreground,
    ),
    titleSmall: TextStyle(
      fontFamily: 'Nunito',
      fontWeight: FontWeight.w500,
      color: PondrTokens.foreground,
    ),
    bodyLarge: TextStyle(fontFamily: 'Inter', color: PondrTokens.foreground),
    bodyMedium: TextStyle(fontFamily: 'Inter', color: PondrTokens.foreground),
    bodySmall: TextStyle(fontFamily: 'Inter', color: PondrTokens.foreground),
    labelLarge: TextStyle(fontFamily: 'Inter', color: PondrTokens.foreground),
    labelMedium: TextStyle(fontFamily: 'Inter', color: PondrTokens.foreground),
    labelSmall: TextStyle(fontFamily: 'Inter', color: PondrTokens.foreground),
  );

  return ThemeData.dark().copyWith(
    colorScheme: scheme,
    textTheme: base,
    primaryTextTheme: base,
    scaffoldBackgroundColor: PondrTokens.background,
    canvasColor: PondrTokens.background,
    cardColor: PondrTokens.card,
    dialogTheme: DialogThemeData(
      backgroundColor: PondrTokens.popover,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(PondrTokens.radius),
      ),
    ),
    dividerColor: PondrTokens.border,
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: PondrTokens.inputBackground,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(PondrTokens.radius),
        borderSide: BorderSide(color: PondrTokens.input),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(PondrTokens.radius),
        borderSide: BorderSide(color: PondrTokens.input),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(PondrTokens.radius),
        borderSide: BorderSide(color: PondrTokens.ring),
      ),
      focusColor: PondrTokens.ring,
      hoverColor: PondrTokens.border,
    ),
  );
}