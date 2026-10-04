import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/bindings.dart';
import '../../data/services.dart' show SettingsService;
import '../../theme/tokens.dart';
import '../chat/chat_state.dart' show settingsRevisionProvider;
import '../chat/common.dart' show PressScale;
import 'settings_atoms.dart';

/// The export's PROFILE section (`app.tsx:1322-1387`): the title+lead, the
/// 64 dp avatar slot with its `Upload photo` chip, the Display Name /
/// Username / Bio fields and the `Save changes` CTA.

/// The section: owns the fields' [TextEditingController]s — the export's
/// `displayName`/`username`/`bio` state IS the settings service here (F2's
/// interface), and the fields WRITE THROUGH on every change (`onChange`
/// setters are the mock's persistence). The section also bumps the
/// settings revision, because the aside's user summary reads the service.
class ProfileSection extends ConsumerStatefulWidget {
  const ProfileSection({super.key});

  @override
  ConsumerState<ProfileSection> createState() => _ProfileSectionState();
}

class _ProfileSectionState extends ConsumerState<ProfileSection> {
  late final TextEditingController _name;
  late final TextEditingController _username;
  late final TextEditingController _bio;

  @override
  void initState() {
    super.initState();
    final SettingsService settings = ref.read(settingsProvider);
    _name = TextEditingController(text: settings.displayName);
    _username = TextEditingController(text: settings.username);
    _bio = TextEditingController(text: settings.bio);
  }

  @override
  void dispose() {
    _name.dispose();
    _username.dispose();
    _bio.dispose();
    super.dispose();
  }

  /// The write-through: the service setter + the revision bump (the aside's
  /// summary — and the chat surface's settings-backed widgets — re-read on
  /// the revision).
  void _write(String field, String value) {
    final SettingsService settings = ref.read(settingsProvider);
    switch (field) {
      case 'name':
        settings.displayName = value;
      case 'username':
        settings.username = value;
      case 'bio':
        settings.bio = value;
    }
    ref.read(settingsRevisionProvider.notifier).bump();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        const SectionHeader(
          title: 'Profile', // app.tsx:1325
          subtitle: 'Manage how you appear in Pondr.', // app.tsx:1326
        ),
        const SizedBox(height: 24), // space-y-6
        const _AvatarRow(), // app.tsx:1331-1347
        const SizedBox(height: 24),
        // The fields, `space-y-4` (app.tsx:1348-1382).
        _LabeledField(
          label: 'Display Name',
          hint: 'Ada Lovelace',
          fieldKey: settingsProfileFieldKey,
          controller: _name,
          onChanged: (String v) => _write('name', v),
        ),
        const SizedBox(height: 16),
        _LabeledField(
          label: 'Username',
          hint: 'ada_lovelace',
          fieldKey: settingsUsernameFieldKey,
          controller: _username,
          onChanged: (String v) => _write('username', v),
        ),
        const SizedBox(height: 16),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            const SettingsFieldLabel('Bio'),
            const SizedBox(height: 6), // mb-1.5
            TextField(
              key: settingsBioFieldKey,
              controller: _bio,
              onChanged: (String v) => _write('bio', v),
              maxLines: 3, // rows={3}
              cursorColor: Colors.white,
              style: kSettingsInputStyle,
              decoration: settingsFieldDecoration(
                hint: 'Tell us a little about yourself…',
              ),
            ),
          ],
        ),
        const SizedBox(height: 24),
        // The export's `Save changes` CTA (app.tsx:1384-1386) has NO onClick
        // — the mock's persistence is the fields' onChange writes (above),
        // so the button is a static affordance there. This port keeps the
        // button 1:1 and lets its press play the `active:scale-[0.98]`
        // affordance; the write-through above is the save semantics.
        const _SaveChangesButton(),
      ],
    );
  }
}

/// The label + field pair (`app.tsx:1337-1349`): the uppercase tracked label
/// over the boxed `px-4 py-3` input.
class _LabeledField extends StatelessWidget {
  const _LabeledField({
    required this.label,
    required this.hint,
    required this.fieldKey,
    required this.controller,
    required this.onChanged,
  });

  final String label;
  final String hint;
  final Key fieldKey;
  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        Text(label, style: kFieldLabelStyle),
        const SizedBox(height: 6), // mb-1.5
        TextField(
          key: fieldKey,
          controller: controller,
          cursorColor: Colors.white,
          style: kSettingsInputStyle,
          decoration: settingsFieldDecoration(hint: hint),
          onChanged: onChanged,
        ),
      ],
    );
  }
}

/// The avatar row (`app.tsx:1331-1347`): the 64 dp circle on
/// `rgba(136,141,223,0.22)` with its 2 px ring, then the caption pair and
/// the `Upload photo` chip.
class _AvatarRow extends StatelessWidget {
  const _AvatarRow();

  @override
  Widget build(BuildContext context) {
    return const Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: <Widget>[
        SizedBox(
          width: 64, // w-16 h-16
          height: 64,
          child: DecoratedBox(
            decoration: BoxDecoration(
              color: Color(0x38888DDF), // rgba(136,141,223,0.22)
              shape: BoxShape.circle,
              border: Border.fromBorderSide(
                BorderSide(color: Color(0x59888DDF), width: 2),
              ),
            ),
            child: Icon(Icons.person, size: 24, color: PondrTokens.primary),
          ),
        ),
        SizedBox(width: 16), // gap-4
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Text(
              'Profile picture',
              style: TextStyle(
                fontFamily: PondrTokens.fontFamily,
                fontWeight: FontWeight.w600, // font-semibold
                fontSize: 14, // text-sm
                color: Colors.white,
              ),
            ),
            SizedBox(height: 4), // mb-1
            Text(
              'JPG, PNG or GIF, max 2 MB',
              style: TextStyle(
                fontFamily: PondrTokens.fontFamily,
                fontSize: 12, // text-xs
                color: Color(0xFF9B96C8), // #9b96c8
              ),
            ),
            SizedBox(height: 8), // mb-2
            _UploadPhotoButton(),
          ],
        ),
      ],
    );
  }
}

/// The `Upload photo` chip button (`app.tsx:1341-1346`). The export gives it
/// NO onClick — a static chip there — so the press scale is the only
/// affordance (intentional 1:1 divergence note: wired NOTHING, as in the
/// mock).
class _UploadPhotoButton extends StatelessWidget {
  const _UploadPhotoButton();

  @override
  Widget build(BuildContext context) {
    return PressScale(
      pressedScale: 0.97,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: const Color(0x26888DDF), // rgba(136,141,223,0.15)
          borderRadius: BorderRadius.circular(8), // rounded-lg
          border: Border.all(color: const Color(0x4D888DDF)), // ...,0.3
        ),
        child: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          child: Text(
            'Upload photo',
            style: TextStyle(
              fontFamily: PondrTokens.fontFamily,
              fontWeight: FontWeight.w600,
              fontSize: 12, // text-xs
              color: PondrTokens.primary,
            ),
          ),
        ),
      ),
    );
  }
}

/// The `Save changes` CTA (`app.tsx:1384-1386`): `bg-primary text-[#0c0b1a]`
/// px-5 py-2.5 rounded-xl, the `0 4px 16` primary glow and the
/// `active:scale-[0.98]` press. See [ProfileSection.build]'s note on its
/// handler presence in the export.
class _SaveChangesButton extends StatelessWidget {
  const _SaveChangesButton();

  @override
  Widget build(BuildContext context) {
    return PressScale(
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
          // The export's button has NO handler (app.tsx:1384-1386) — the
          // fields' onChange writes ARE the persistence. Kept enabled for
          // the visual 1:1; the tap is a deliberate no-op.
          onTap: () {},
          child: const Padding(
            padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            child: Text(
              'Save changes',
              style: TextStyle(
                fontFamily: PondrTokens.fontDisplay, // font-nunito
                fontWeight: FontWeight.w700,
                fontSize: 14, // text-sm
                color: Color(0xFF0C0B1A), // text-[#0c0b1a]
              ),
            ),
          ),
        ),
      ),
    );
  }
}
