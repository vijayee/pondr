import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:pondr/theme/app_theme.dart';
import 'package:pondr/theme/tokens.dart';

void main() {
  // No font fetching at test time — the fallback face is fine for pinning.

  group('PondrTokens — pinned against mockup_reference/styles/theme.css', () {
    test('the solid colours', () {
      expect(PondrTokens.background.toARGB32(), 0xFF0C0B1A);
      expect(PondrTokens.foreground.toARGB32(), 0xFFECE9FF);
      expect(PondrTokens.card.toARGB32(), 0xFF2B293B);
      expect(PondrTokens.popover.toARGB32(), 0xFF1A1632);
      expect(PondrTokens.primary.toARGB32(), 0xFF888DDF);
      expect(PondrTokens.secondary.toARGB32(), 0xFF1C1833);
      expect(PondrTokens.muted.toARGB32(), 0xFF1A1630);
      expect(PondrTokens.accent.toARGB32(), 0xFFC3ACDA);
      expect(PondrTokens.destructive.toARGB32(), 0xFFD4183D);
      expect(PondrTokens.ring.toARGB32(), 0xFF888DDF);
      expect(PondrTokens.sidebarAccent.toARGB32(), 0xFF2D2B45);
      expect(PondrTokens.sidebarRing.toARGB32(), 0xFF5B67D8);
    });

    test('the alpha colours (rgba over #888DDF, alpha = round(255 * a))', () {
      expect(PondrTokens.border.toARGB32(), 0x40888DDF);
      expect(PondrTokens.input.toARGB32(), 0x1A888DDF);
      expect(PondrTokens.inputBackground.toARGB32(), 0x14888DDF);
      expect(PondrTokens.sidebarBorder.toARGB32(), 0x2E888DDF);
    });

    test('the constants', () {
      expect(PondrTokens.radius, 12.0);
      expect(PondrTokens.fontFamily, 'Inter');
      expect(PondrTokens.fontDisplay, 'Nunito');
      expect(PondrTokens.chartPalette.length, 5);
    });

    test('the chart palette (oklch -> hex conversions)', () {
      expect(PondrTokens.chartPalette[0].toARGB32(), 0xFF4582FF); // .646 .222 264
      expect(PondrTokens.chartPalette[1].toARGB32(), 0xFF8267E2); // .600 .180 290
      expect(PondrTokens.chartPalette[2].toARGB32(), 0xFFBB82E3); // .700 .150 310
      expect(PondrTokens.chartPalette[3].toARGB32(), 0xFF0099FA); // .650 .200 240
      expect(PondrTokens.chartPalette[4].toARGB32(), 0xFF2AC4CC); // .750 .120 200
    });
  });

  group('the theme — built from the tokens', () {
    final ThemeData theme = buildPondrTheme();

    test('scaffold + scheme use the tokens', () {
      expect(theme.scaffoldBackgroundColor, PondrTokens.background);
      expect(theme.colorScheme.primary, PondrTokens.primary);
      expect(theme.colorScheme.surface, PondrTokens.card);
      expect(theme.colorScheme.onSurface, PondrTokens.cardForeground);
      expect(theme.colorScheme.error, PondrTokens.destructive);
      expect(theme.colorScheme.brightness, Brightness.dark);
      expect(theme.colorScheme.outline, PondrTokens.border);
    });

    test('the input decoration uses inputBackground + ring', () {
      expect(theme.inputDecorationTheme.fillColor, PondrTokens.inputBackground);
      expect(
        theme.inputDecorationTheme.focusedBorder!.borderSide.color,
        PondrTokens.ring,
      );
    });
  });

  // Golden-lite: a Container coloured with the token background paints, and
  // the rendered top-left pixel IS the token colour (no golden file diff).
  testWidgets('PondrTokens.background paints', (WidgetTester tester) async {
    await tester.pumpWidget(
      Directionality(
        textDirection: TextDirection.ltr,
        child: RepaintBoundary(
          child: Container(color: PondrTokens.background),
        ),
      ),
    );
    await tester.pumpAndSettle();

    final RenderRepaintBoundary boundary =
        tester.renderObject(find.byType(RepaintBoundary));
    // toImage/toByteData are real-async GPU work: they must run outside the
    // fake-async zone that widget tests use.
    final ByteData? bytesNullable = await tester.runAsync<ByteData?>(() async {
      final ui.Image image = await boundary.toImage(pixelRatio: 1.0);
      return image.toByteData(format: ui.ImageByteFormat.rawStraightRgba);
    });
    final ByteData bytes =
        bytesNullable ?? (throw StateError('no image bytes'));
    // Container fills the 800x600 test surface: the sampled pixel matches.
    expect(bytes.getUint8(0), 0x0C); // r
    expect(bytes.getUint8(1), 0x0B); // g
    expect(bytes.getUint8(2), 0x1A); // b
    expect(bytes.getUint8(3), 0xFF); // a
  });
}