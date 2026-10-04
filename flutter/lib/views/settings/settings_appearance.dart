import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/bindings.dart';
import '../../theme/tokens.dart';
import '../chat/chat_state.dart' show settingsRevisionProvider;
import 'settings_atoms.dart' show SectionHeader, SettingsChip;

/// The export's APPEARANCE section (`app.tsx:1389-1469`): the accent-color
/// swatch chips (the check rides INSIDE the swatch dot) and the static
/// Dark/System theme chooser.

/// The export's `ACCENT_OPTIONS` (`app.tsx:1148-1155`) — the five swatches,
/// verbatim.
const List<(String, String)> kAccentOptions = <(String, String)>[
  ('Periwinkle', '#888ddf'),
  ('Lilac', '#c3acda'),
  ('Sage', '#e0efe4'),
  ('Champagne', '#e8d7bd'),
  ('Ice Blue', '#e2e6ff'),
];

class AppearanceSection extends ConsumerWidget {
  const AppearanceSection({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // The service holds the accent; the revision re-reads after a pick.
    ref.watch(settingsRevisionProvider);
    final String accent = ref.read(settingsProvider).accentColor;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        const SectionHeader(
          title: 'Appearance', // app.tsx:1391
          subtitle: 'Personalise the look of Pondr.', // app.tsx:1392
        ),
        const SizedBox(height: 24), // space-y-6
        // The accent swatch row (`app.tsx:1395-1417`).
        const Text(
          'Accent color',
          style: TextStyle(
            fontFamily: PondrTokens.fontFamily,
            fontWeight: FontWeight.w700,
            fontSize: 12, // text-xs
            letterSpacing: 1.2, // tracking-widest
            color: Color(0xB3ECE9FF), // rgba(236,233,255,0.7)
          ),
        ),
        const SizedBox(height: 12), // mb-3
        Wrap(
          spacing: 12, // gap-3
          runSpacing: 12, // flex-wrap
          children: <Widget>[
            for (final (String label, String value) in kAccentOptions)
              _AccentSwatch(
                label: label,
                value: value,
                selected: accent == value,
                onTap: () {
                  ref.read(settingsProvider).accentColor = value;
                  ref.read(settingsRevisionProvider.notifier).bump();
                },
              ),
          ],
        ),
        const SizedBox(height: 24),
        // The theme chooser (`app.tsx:1419-1449`): the mock gives the two
        // buttons NO onClick — `Dark` is always selected, statically.
        const Text(
          'Theme',
          style: TextStyle(
            fontFamily: PondrTokens.fontFamily,
            fontWeight: FontWeight.w700,
            fontSize: 12,
            letterSpacing: 1.2,
            color: Color(0xB3ECE9FF),
          ),
        ),
        const SizedBox(height: 12), // mb-3
        const Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            _ThemeChip(label: 'Dark', selected: true),
            SizedBox(width: 12), // gap-3
            _ThemeChip(label: 'System', selected: false),
          ],
        ),
      ],
    );
  }
}

/// One accent chip (`app.tsx:1400-1414`): the tinted fill (`<value>22` —
/// the accent at alpha 0x22) with the accent-colored border + text when
/// selected, `white/5` on `white/10` otherwise; the 14 dp dot carries a 9 dp
/// check in `#0c0b1a` when selected.
class _AccentSwatch extends StatelessWidget {
  const _AccentSwatch({
    required this.label,
    required this.value,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final String value;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final Color color = Color(
      int.parse(value.substring(1), radix: 16) | 0xFF000000,
    ); // hexToColor
    return _ChipButton(
      onTap: onTap,
      background: selected
          ? color.withValues(alpha: 0x22 / 255)
          : const Color(0x0DFFFFFF),
      borderColor: selected ? color : const Color(0x1AFFFFFF),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Container(
            width: 14, // w-3.5 h-3.5
            height: 14,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            alignment: Alignment.center,
            child: selected
                ? const Icon(
                    Icons.check,
                    size: 9, // Check size={9}
                    color: Color(0xFF0C0B1A), // style color
                  )
                : null,
          ),
          const SizedBox(width: 10), // gap-2.5
          Text(
            label,
            style: TextStyle(
              fontFamily: PondrTokens.fontFamily,
              fontWeight: FontWeight.w500,
              fontSize: 14, // text-sm
              color: selected
                  ? color
                  : const Color(0x99FFFFFF), // text-[rgba(255,255,255,0.6)]
            ),
          ),
        ],
      ),
    );
  }
}

/// One theme chip (`app.tsx:1421-1436`): the DARK chip is the mock's
/// selected one (`rgba(136,141,223,0.18)` + the 0.45 border + the 12 dp
/// check); `System` reads the unselected `white/5` on `white/10`.
class _ThemeChip extends StatelessWidget {
  const _ThemeChip({required this.label, required this.selected});

  final String label;
  final bool selected;

  @override
  Widget build(BuildContext context) {
    return selected
        ? SettingsChip(
            background: const Color(0x2E888DDF),
            borderColor: const Color(0x73888DDF),
            child: const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Icon(Icons.check, size: 12, color: PondrTokens.primary),
                  SizedBox(width: 8), // gap-2
                  Text(
                    'Dark',
                    style: TextStyle(
                      fontFamily: PondrTokens.fontFamily,
                      fontWeight: FontWeight.w500,
                      fontSize: 14,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          )
        : SettingsChip(
            background: const Color(0x0DFFFFFF),
            borderColor: const Color(0x1AFFFFFF),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Text(
                label,
                style: const TextStyle(
                  fontFamily: PondrTokens.fontFamily,
                  fontWeight: FontWeight.w500,
                  fontSize: 14,
                  color: Color(0x80FFFFFF), // text-white/50
                ),
              ),
            ),
          );
  }
}

/// The tap-away button shell the interactive chips (the accent swatches)
/// build on: the rounded-xl box + the press affordance. The THEME chips'
/// static twin builds the same box itself.
class _ChipButton extends StatelessWidget {
  const _ChipButton({
    required this.onTap,
    required this.background,
    required this.borderColor,
    required this.child,
  });

  final VoidCallback onTap;
  final Color background;
  final Color borderColor;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: background,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12), // rounded-xl
        side: BorderSide(color: borderColor),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        splashColor: Colors.transparent,
        highlightColor: Colors.transparent,
        hoverColor: Colors.transparent,
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: child,
        ),
      ),
    );
  }
}
