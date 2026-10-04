import 'package:flutter/material.dart';

/// The export's design tokens — verbatim values from the mockup's CSS.
///
/// Source of truth: `mockup_reference/styles/theme.css` (the `:root` /
/// `.dark` blocks — they are identical). Every colour here is pinned; the
/// chart colours were oklch in the CSS and are converted to hex below.
class PondrTokens {
  PondrTokens._();

  // --background
  static const Color background = Color(0xFF0C0B1A);
  // --foreground
  static const Color foreground = Color(0xFFECE9FF);
  // --card
  static const Color card = Color(0xFF2B293B);
  // --card-foreground
  static const Color cardForeground = Color(0xFFECE9FF);
  // --popover
  static const Color popover = Color(0xFF1A1632);
  // --popover-foreground
  static const Color popoverForeground = Color(0xFFECE9FF);
  // --primary
  static const Color primary = Color(0xFF888DDF);
  // --primary-foreground
  static const Color primaryForeground = Color(0xFF0C0B1A);
  // --secondary
  static const Color secondary = Color(0xFF1C1833);
  // --secondary-foreground
  static const Color secondaryForeground = Color(0xFFC4BDE8);
  // --muted
  static const Color muted = Color(0xFF1A1630);
  // --muted-foreground
  static const Color mutedForeground = Color(0xFF9B96C8);
  // --accent
  static const Color accent = Color(0xFFC3ACDA);
  // --accent-foreground
  static const Color accentForeground = Color(0xFF0C0B1A);
  // --destructive
  static const Color destructive = Color(0xFFD4183D);
  // --destructive-foreground
  static const Color destructiveForeground = Color(0xFFFFFFFF);
  // --border: rgba(136, 141, 223, 0.25) -> alpha round(0.25 * 255) = 64 = 0x40
  static const Color border = Color(0x40888DDF);
  // --input: rgba(136, 141, 223, 0.1) -> alpha round(0.1 * 255) = 26 = 0x1A
  static const Color input = Color(0x1A888DDF);
  // --input-background: rgba(136, 141, 223, 0.08) -> alpha round(20.4) = 20 = 0x14
  static const Color inputBackground = Color(0x14888DDF);
  // --switch-background
  static const Color switchBackground = Color(0xFF3A345E);
  // --ring
  static const Color ring = Color(0xFF888DDF);

  // --sidebar
  static const Color sidebar = Color(0xFF1A1929);
  // --sidebar-foreground
  static const Color sidebarForeground = Color(0xFFECE9FF);
  // --sidebar-primary
  static const Color sidebarPrimary = Color(0xFF888DDF);
  // --sidebar-primary-foreground
  static const Color sidebarPrimaryForeground = Color(0xFF0C0B1A);
  // --sidebar-accent
  static const Color sidebarAccent = Color(0xFF2D2B45);
  // --sidebar-accent-foreground
  static const Color sidebarAccentForeground = Color(0xFFECE9FF);
  // --sidebar-border: rgba(136, 141, 223, 0.18) -> alpha round(45.9) = 46 = 0x2E
  static const Color sidebarBorder = Color(0x2E888DDF);
  // --sidebar-ring
  static const Color sidebarRing = Color(0xFF5B67D8);

  /// --chart-1..5, converted from oklch to hex.
  ///
  /// Conversion method: OKLCH -> OKLab -> LMS cube-root -> linear sRGB ->
  /// gamma-encoded sRGB, channels clamped to [0,1] and rounded to 8 bits
  /// (computed with the standard colour-science matrix coefficients; the
  /// oklch() source values live in `mockup_reference/styles/theme.css`):
  ///
  /// - oklch(0.646 0.222 264) -> #4582FF
  /// - oklch(0.600 0.180 290) -> #8267E2
  /// - oklch(0.700 0.150 310) -> #BB82E3
  /// - oklch(0.650 0.200 240) -> #0099FA
  /// - oklch(0.750 0.120 200) -> #2AC4CC
  static const List<Color> chartPalette = <Color>[
    Color(0xFF4582FF), // --chart-1
    Color(0xFF8267E2), // --chart-2
    Color(0xFFBB82E3), // --chart-3
    Color(0xFF0099FA), // --chart-4
    Color(0xFF2AC4CC), // --chart-5
  ];

  /// --radius: 0.75rem at the export's 16px root font size = 12px.
  static const double radius = 12.0;

  /// Body font (inputs/buttons/text in the export's `@layer base`).
  static const String fontFamily = 'Inter';

  /// Display font (h1-h4 headings in the export's `@layer base`).
  static const String fontDisplay = 'Nunito';
}