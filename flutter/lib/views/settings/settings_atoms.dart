import 'package:flutter/material.dart';

import '../../theme/tokens.dart';

/// The settings pages' shared atoms — the export's repeated CSS classes as
/// widgets/style consts (`app.tsx:1248-1869`): the section header, the
/// uppercase field label, the field boxes, the primary CTA and the
/// notifications' toggle switch.

/// Test keys shared across the section files.
const Key settingsToggleMessagesKey = Key('settings.notif-messages');
const Key settingsToggleSoundsKey = Key('settings.notif-sounds');
const Key settingsProfileFieldKey = Key('settings.field-display-name');
const Key settingsUsernameFieldKey = Key('settings.field-username');
const Key settingsBioFieldKey = Key('settings.field-bio');
const Key settingsAddProviderKey = Key('settings.add-provider');
const Key settingsProviderNameFieldKey = Key('settings.form-name');
const Key settingsProviderUrlFieldKey = Key('settings.form-url');
const Key settingsProviderKeyFieldKey = Key('settings.form-key');
const Key settingsProviderEyeKey = Key('settings.form-eye');
const Key settingsProviderSaveKey = Key('settings.save-provider');
const Key settingsProviderCancelKey = Key('settings.cancel-provider');
const Key settingsSavedToastKey = Key(
  'settings.toast',
); // the saved-flash's pin
const Key settingsModelIdFieldKey = Key('settings.model-id');
const Key settingsModelLabelFieldKey = Key('settings.model-label');
const Key settingsModelSaveKey = Key('settings.model-save');
const Key settingsModelCancelKey = Key('settings.model-cancel');

/// A section's header: the `text-white font-nunito font-bold text-xl` h2
/// over the `text-sm` lead (`app.tsx:1324-1327` — every section opens with
/// the same pair).
class SectionHeader extends StatelessWidget {
  const SectionHeader({required this.title, required this.subtitle, super.key});

  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        Text(
          title,
          style: const TextStyle(
            fontFamily: PondrTokens.fontDisplay,
            fontWeight: FontWeight.w700,
            fontSize: 20, // text-xl
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 4), // mb-1
        Text(
          subtitle,
          style: const TextStyle(
            fontFamily: PondrTokens.fontFamily,
            fontSize: 14, // text-sm
            color: Color(0xFF9B96C8), // the mock's #9b96c8
          ),
        ),
      ],
    );
  }
}

/// The export's field label — `text-xs font-bold tracking-widest uppercase`
/// (`app.tsx:1337-1339`, repeated through the sections). Most labels sit on
/// `rgba(236,233,255,0.7)`; the inline forms dim to 0.6.
const TextStyle kFieldLabelStyle = TextStyle(
  fontFamily: PondrTokens.fontFamily,
  fontWeight: FontWeight.w700,
  fontSize: 12, // text-xs
  letterSpacing: 1.2, // tracking-widest (0.1em at 12px)
  color: Color(0xB3ECE9FF), // rgba(236,233,255,0.7)
);

/// The label + [gap]-below block the sections' fields open with.
class SettingsFieldLabel extends StatelessWidget {
  const SettingsFieldLabel(
    this.label, {
    this.color = const Color(0xB3ECE9FF),
    super.key,
  });

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Text(label, style: kFieldLabelStyle.copyWith(color: color));
  }
}

/// The export's settings input box — `rounded-xl px-4 py-3` on
/// `rgba(136,141,223,0.1)` with the `rgba(136,141,223,0.35)` border and the
/// `#888ddf` 2 px focus ring (`app.tsx:1343-1349`, profile + security);
/// [compact] (the inline forms) reads `rounded-lg px-3 py-2.5` on
/// `rgba(255,255,255,0.07)` with the 0.3 border (`app.tsx:1494-1499`).
InputDecoration settingsFieldDecoration({
  required String hint,
  Widget? prefixIcon,
  Widget? suffixIcon,
  bool compact = false,
  Color border = const Color(0x59888DDF),
}) {
  return InputDecoration(
    filled: true,
    fillColor: compact
        ? const Color(0x12FFFFFF) // rgba(255,255,255,0.07)
        : const Color(0x1A888DDF), // rgba(136,141,223,0.1)
    contentPadding: compact
        // px-3 py-2.5 (the API-key variant's pl-8 reads via [prefixIcon])
        ? const EdgeInsets.fromLTRB(12, 10, 12, 10)
        : const EdgeInsets.symmetric(horizontal: 16, vertical: 12), // px-4 py-3
    hintText: hint,
    hintStyle: TextStyle(
      fontFamily: PondrTokens.fontFamily,
      fontWeight: FontWeight.w500,
      fontSize: compact ? 14 : null,
      color: const Color(0x4DFFFFFF), // placeholder:text-white/30
    ),
    prefixIcon: prefixIcon,
    suffixIcon: suffixIcon,
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(
        compact ? 8 : 12,
      ), // rounded-lg / rounded-xl
      borderSide: BorderSide(color: border),
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(compact ? 8 : 12),
      borderSide: BorderSide(color: border),
    ),
    // focus:ring-2 focusRingColor ~ the mock's focus ring
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(compact ? 8 : 12),
      borderSide: const BorderSide(color: PondrTokens.primary, width: 2),
    ),
  );
}

/// The settings page's text style inside inputs:
/// `text-white ... text-sm` (`app.tsx:1347`).
const TextStyle kSettingsInputStyle = TextStyle(
  fontFamily: PondrTokens.fontFamily,
  fontWeight: FontWeight.w500,
  fontSize: 14, // text-sm
  color: Colors.white,
);

/// A section's pill-shaped row chip (the accent swatches' buttons,
/// the theme chooser, the about list): `rounded-xl px-4 py-2.5` — the exact
/// chrome varies per selection state; the caller supplies it.
class SettingsChip extends StatelessWidget {
  const SettingsChip({
    required this.background,
    required this.borderColor,
    required this.child,
    super.key,
  });

  final Color background;
  final Color borderColor;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(12), // rounded-xl
        border: Border.all(color: borderColor),
      ),
      child: child,
    );
  }
}

/// The notifications' toggle switch — the export's custom button, NOT a
/// material switch: `w-11 h-6 rounded-full` on `#888ddf` / `white/12`, the
/// `w-5 h-5` white knob with the `0 1px 4` shadow sliding across
/// (`app.tsx:1817-1836`); the `transition-all` reads as 150 ms [AnimatedContainer].
class PondrToggleSwitch extends StatelessWidget {
  const PondrToggleSwitch({
    required this.value,
    required this.onChanged,
    required this.switchKey,
    super.key,
  });

  final bool value;
  final ValueChanged<bool> onChanged;
  final Key switchKey;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      key: switchKey,
      onTap: () => onChanged(!value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150), // transition-all
        curve: Curves.easeOut,
        width: 44, // w-11
        height: 24, // h-6
        decoration: BoxDecoration(
          color: value
              ? PondrTokens.primary
              : const Color(0x1FFFFFFF), // white/12
          borderRadius: BorderRadius.circular(12),
        ),
        child: AnimatedAlign(
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
          alignment: value ? Alignment.centerRight : Alignment.centerLeft,
          child: Container(
            width: 20, // w-5 h-5
            height: 20,
            margin: const EdgeInsets.all(2), // top-0.5 / 0.125rem
            decoration: const BoxDecoration(
              color: Colors.white,
              shape: BoxShape.circle,
              boxShadow: <BoxShadow>[
                BoxShadow(
                  color: Color(0x4D000000), // rgba(0,0,0,0.3)
                  offset: Offset(0, 1),
                  blurRadius: 4,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
