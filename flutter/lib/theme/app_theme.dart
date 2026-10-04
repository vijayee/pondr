import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'tokens.dart';

/// The app's [ThemeData], built from [PondrTokens].
///
/// Inter is the body font and Nunito the display font (h1-h4), following the
/// export's `@layer base` rules in `mockup_reference/styles/globals.css`.
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

  final TextTheme base = GoogleFonts.interTextTheme(
    ThemeData.dark().textTheme.apply(
          bodyColor: PondrTokens.foreground,
          displayColor: PondrTokens.foreground,
        ),
  );
  // The export's h1-h4 use Nunito at medium (500) weight.
  final TextTheme textTheme = base.copyWith(
    headlineLarge: GoogleFonts.nunito(
      textStyle: base.headlineLarge,
      fontWeight: FontWeight.w500,
    ),
    headlineMedium: GoogleFonts.nunito(
      textStyle: base.headlineMedium,
      fontWeight: FontWeight.w500,
    ),
    headlineSmall: GoogleFonts.nunito(
      textStyle: base.headlineSmall,
      fontWeight: FontWeight.w500,
    ),
    titleLarge: GoogleFonts.nunito(
      textStyle: base.titleLarge,
      fontWeight: FontWeight.w500,
    ),
    titleMedium: GoogleFonts.nunito(
      textStyle: base.titleMedium,
      fontWeight: FontWeight.w500,
    ),
    titleSmall: GoogleFonts.nunito(
      textStyle: base.titleSmall,
      fontWeight: FontWeight.w500,
    ),
  );

  return ThemeData.dark().copyWith(
    colorScheme: scheme,
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
    textTheme: textTheme,
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