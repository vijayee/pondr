import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../../data/bindings.dart';
import '../../data/services.dart' show SettingsService;
import '../../theme/tokens.dart';
import '../chat/chat_state.dart' show settingsRevisionProvider;
import '../chat/common.dart' show PressScale, kThoughtSparkAsset;
import 'settings_atoms.dart';

/// The export's three remaining sections — NOTIFICATIONS (`app.tsx:1765-1805`),
/// SECURITY (`app.tsx:1807-1868`) and ABOUT (`app.tsx:1870-1908`... the
/// mock's `about` block), each in its own widget; the file-focus rule puts
/// the three SMALL sections together while the big CRUD one
/// (`settings_providers.dart`) and the interactive two own their files.

// ─── Notifications ─────────────────────────────────────────────────────────

/// The export's NOTIFICATIONS section (`app.tsx:1765-1805`): the two rows
/// with the custom toggle switches, persisted through the settings service.
class NotificationsSection extends ConsumerWidget {
  const NotificationsSection({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    ref.watch(settingsRevisionProvider); // re-read after toggles
    final SettingsService settings = ref.read(settingsProvider);
    final bool messages = settings.notifMessages;
    final bool sounds = settings.notifSounds;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        const SectionHeader(
          title: 'Notifications', // app.tsx:1800
          subtitle: 'Control when and how Pondr alerts you.', // app.tsx:1801
        ),
        const SizedBox(height: 24), // space-y-6
        Column(
          children: <Widget>[
            _NotificationRow(
              switchKey: settingsToggleMessagesKey,
              label: 'Message notifications',
              desc: 'Get notified when Pondr finishes a response',
              value: messages,
              onToggle: (bool v) {
                settings.notifMessages = v;
                ref.read(settingsRevisionProvider.notifier).bump();
              },
            ),
            const SizedBox(height: 12), // space-y-3
            _NotificationRow(
              switchKey: settingsToggleSoundsKey,
              label: 'Sound effects',
              desc: 'Play a soft chime when responses arrive',
              value: sounds,
              onToggle: (bool v) {
                settings.notifSounds = v;
                ref.read(settingsRevisionProvider.notifier).bump();
              },
            ),
          ],
        ),
      ],
    );
  }
}

/// One notifications row (`app.tsx:1782-1811`): the boxed panel with the
/// label/desc pair and the custom switch.
class _NotificationRow extends StatelessWidget {
  const _NotificationRow({
    required this.switchKey,
    required this.label,
    required this.desc,
    required this.value,
    required this.onToggle,
  });

  final Key switchKey;
  final String label;
  final String desc;
  final bool value;
  final ValueChanged<bool> onToggle;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0x14888DDF), // rgba(136,141,223,0.08)
        borderRadius: BorderRadius.circular(12), // rounded-xl
        border: Border.all(
          color: const Color(0x33888DDF), // rgba(136,141,223,0.2)
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        child: Row(
          children: <Widget>[
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Text(
                    label,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 2), // mt-0.5
                  Text(
                    desc,
                    style: const TextStyle(
                      fontFamily: PondrTokens.fontFamily,
                      fontSize: 12,
                      color: Color(0xFF9B96C8), // #9b96c8
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            PondrToggleSwitch(
              value: value,
              onChanged: onToggle,
              switchKey: switchKey,
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Security ──────────────────────────────────────────────────────────────

/// The export's SECURITY section (`app.tsx:1807-1868`): the three password
/// fields (current/new/confirm — placeholders only, no state), the
/// `Update password` CTA and the DANGER zone. The mock wires NO handlers;
/// these are deliberately static here too.
class SecuritySection extends StatelessWidget {
  const SecuritySection({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        const SectionHeader(
          title: 'Security', // app.tsx:1813
          subtitle: 'Keep your account safe.', // app.tsx:1814
        ),
        const SizedBox(height: 24), // space-y-6 between the blocks
        _PasswordField('Current password'),
        const SizedBox(height: 16), // space-y-4
        _PasswordField('New password'),
        const SizedBox(height: 16),
        _PasswordField('Confirm new password'),
        const SizedBox(height: 24),
        // The `Update password` CTA (app.tsx:1855-1859) — no handler in the
        // mock; the press affordance is the only motion (as in the mock).
        PressScale(
          pressedScale: 0.98, // active:scale-[0.98]
          child: Material(
            color: PondrTokens.primary,
            borderRadius: BorderRadius.circular(12), // rounded-xl
            shadowColor: const Color(0x59888DDF), // rgba(136,141,223,0.35)
            elevation: 8,
            child: InkWell(
              borderRadius: BorderRadius.circular(12),
              splashColor: Colors.transparent,
              highlightColor: Colors.transparent,
              hoverColor: Colors.transparent,
              onTap: () {}, // the mock's button carries no onClick
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                child: Text(
                  'Update password',
                  style: TextStyle(
                    fontFamily: PondrTokens.fontDisplay, // font-nunito
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    color: Color(0xFF0C0B1A),
                  ),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 24),
        // The danger zone (`app.tsx:1861-1868`).
        const SizedBox(height: 16), // pt-4
        const Text(
          'Danger zone',
          style: TextStyle(
            fontFamily: PondrTokens.fontFamily,
            fontWeight: FontWeight.w700,
            fontSize: 12,
            letterSpacing: 1.2,
            color: Color(0x80ECE9FF), // rgba(236,233,255,0.5)
          ),
        ),
        const SizedBox(height: 12), // mb-3
        // The `Delete account` chip (app.tsx:1869-1873).
        SettingsChip(
          background: const Color(0x1FD4183D), // rgba(212,24,61,0.12)
          borderColor: const Color(0x4DD4183D), // rgba(212,24,61,0.3)
          child: const Padding(
            padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            child: Text(
              'Delete account',
              style: TextStyle(
                fontFamily: PondrTokens.fontFamily,
                fontWeight: FontWeight.w500,
                fontSize: 14,
                color: Color(0xFFF87171), // #f87171
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// The password field trio's box (`app.tsx:1824-1844`): the profile-field
/// chrome with `placeholder: ••••••••`.
class _PasswordField extends StatelessWidget {
  const _PasswordField(this.label);

  final String label;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        SettingsFieldLabel(label),
        const SizedBox(height: 6), // mb-1.5
        TextField(
          obscureText: true,
          enableSuggestions: false,
          autocorrect: false,
          cursorColor: Colors.white,
          style: kSettingsInputStyle,
          decoration: settingsFieldDecoration(hint: '••••••••'),
        ),
      ],
    );
  }
}

// ─── About ─────────────────────────────────────────────────────────────────

/// The three legal rows (`app.tsx:1878-1898`) — static chips with the
/// trailing chevron.
const List<String> kLegalLinks = <String>[
  'Privacy Policy',
  'Terms of Service',
  'Open Source Licenses',
];

/// The export's ABOUT section (`app.tsx:1870-1908`): the ThoughtSpark'd
/// Pondr card + the three static legal rows.
class AboutSection extends StatelessWidget {
  const AboutSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        const SectionHeader(
          title: 'About', // app.tsx:1876
          subtitle: 'Pondr version and legal information.', // app.tsx:1877
        ),
        const SizedBox(height: 24), // space-y-6
        // The version card (app.tsx:1879-1889).
        DecoratedBox(
          decoration: BoxDecoration(
            color: const Color(0x14888DDF), // rgba(136,141,223,0.08)
            borderRadius: BorderRadius.circular(12), // rounded-xl
            border: Border.all(
              color: const Color(0x33888DDF), // rgba(136,141,223,0.2)
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16), // px-4 py-4
            child: Row(
              children: <Widget>[
                SizedBox(
                  width: 48, // w-12 h-12
                  height: 48,
                  child: SvgPicture.asset(
                    kThoughtSparkAsset,
                    fit: BoxFit.contain,
                  ),
                ),
                const SizedBox(width: 16), // gap-4
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: const <Widget>[
                    Text(
                      'Pondr',
                      style: TextStyle(
                        fontFamily: PondrTokens.fontDisplay,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    SizedBox(height: 2), // mt-0.5
                    Text(
                      'The Ponder Engine',
                      style: TextStyle(
                        fontFamily: PondrTokens.fontFamily,
                        fontSize: 12,
                        color: PondrTokens.primary, // #888ddf
                      ),
                    ),
                    SizedBox(height: 2),
                    Text(
                      'Version 1.0.0',
                      style: TextStyle(
                        fontFamily: PondrTokens.fontFamily,
                        fontSize: 12,
                        color: Color(0xFF9B96C8), // #9b96c8
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16), // space-y-2's wrapper
        Column(
          children: <Widget>[
            for (int i = 0; i < kLegalLinks.length; i++) ...<Widget>[
              if (i > 0) const SizedBox(height: 8), // space-y-2
              _LegalRow(kLegalLinks[i]),
            ],
          ],
        ),
      ],
    );
  }
}

class _LegalRow extends StatelessWidget {
  const _LegalRow(this.label);

  final String label;

  @override
  Widget build(BuildContext context) {
    // The mock's rows are <button>s WITHOUT handlers — static chips.
    return SettingsChip(
      background: const Color(0x0AFFFFFF), // rgba(255,255,255,0.04)
      borderColor: const Color(0x26888DDF), // rgba(136,141,223,0.15)
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: <Widget>[
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  fontFamily: PondrTokens.fontFamily,
                  fontSize: 14,
                  color: Color(0xB3FFFFFF), // text-white/70
                ),
              ),
            ),
            const Icon(
              Icons.chevron_right,
              size: 14,
              color: Color(0x66FFFFFF), // opacity-40
            ),
          ],
        ),
      ),
    );
  }
}
