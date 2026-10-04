import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/bindings.dart';
import '../../data/services.dart' show SettingsService;
// The MODEL's `Provider` class and riverpod's `Provider` share a name —
// the data-side one comes in aliased (the chat_state.dart idiom).
import '../../data/models.dart' as mockup show Provider, ProviderModel;
import '../../theme/tokens.dart';
import '../chat/chat_state.dart' show settingsRevisionProvider;
import '../chat/common.dart' show PressScale;
import 'settings_atoms.dart';
import 'settings_motion.dart';
import 'settings_state.dart';

/// The export's PROVIDERS section (`app.tsx:1460-1860`): the accordion list
/// of OpenAI-compatible providers — each row with its rotate chevron, the
/// Active/Disabled badge, the enable/edit/remove affordances, the inline
/// edit form, the models CRUD (add/edit/delete/toggle) and the EMPTY state;
/// plus the new-provider form, the saved-flash toast and the add button.
class ProvidersSection extends ConsumerWidget {
  const ProvidersSection({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    ref.watch(settingsRevisionProvider); // re-read the service on mutations
    final SettingsService settings = ref.read(settingsProvider);
    final List<mockup.Provider> providers = settings.providers();
    return _ProvidersView(providers: providers);
  }
}

/// The section's column — factored so the tests can find the row-level parts
/// without the store's state leaking into the type signatures.
class _ProvidersView extends ConsumerWidget {
  const _ProvidersView({required this.providers});

  final List<mockup.Provider> providers;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final SettingsUi ui = ref.watch(settingsUiProvider);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        const SectionHeader(
          title: 'Providers', // app.tsx:1462
          subtitle:
              'Connect OpenAI-compatible endpoints and configure '
              'their models.', // app.tsx:1463
        ),
        const SizedBox(height: 20), // space-y-5
        if (providers.isNotEmpty)
          Column(
            children: <Widget>[
              for (final mockup.Provider p in providers)
                ProviderRow(provider: p),
            ],
          )
        // The empty state (`app.tsx:1710-1717`): the dashed Cpu box, only
        // while no form covers it (`!showProviderForm`).
        else if (!ui.providerFormOpen)
          const _ProvidersEmptyState(),
        const SizedBox(height: 16),
        const _SavedToast(), // app.tsx:1723-1733
        const _NewProviderForm(), // app.tsx:1736-1823
        const _AddProviderButton(), // app.tsx:1825-1832
      ],
    );
  }
}

/// The export's empty state (`app.tsx:1709-1717`): the `1px dashed`
/// `rgba(136,141,223,0.25)` box on `rgba(136,141,223,0.06)`, the 28 dp Cpu
/// glyph at 30% and the two caption lines.
class _ProvidersEmptyState extends StatelessWidget {
  const _ProvidersEmptyState();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0x0F888DDF), // rgba(136,141,223,0.06)
        borderRadius: BorderRadius.circular(12), // rounded-xl
        border: Border.all(
          color: const Color(0x40888DDF), // rgba(136,141,223,0.25)
        ),
      ),
      child: const Padding(
        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 40),
        child: SizedBox(
          width: double.infinity,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Opacity(
                opacity: 0.3, // opacity-30
                child: Icon(
                  Icons.memory_outlined, // the Cpu glyph
                  size: 28,
                  color: PondrTokens.primary,
                ),
              ),
              SizedBox(height: 12), // mb-3
              Text(
                'No providers yet',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 14, // text-sm
                  color: Color(0x99FFFFFF), // text-white/60
                ),
              ),
              SizedBox(height: 4), // mb-1
              Text(
                'Add an OpenAI-compatible service to get started.',
                style: TextStyle(
                  fontFamily: PondrTokens.fontFamily,
                  fontSize: 12, // text-xs
                  color: Color(0xFF9B96C8), // #9b96c8
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── The saved-flash toast ─────────────────────────────────────────────────

/// The export's saved toast (`app.tsx:1724-1733`): `Saved successfully.`
/// riding the `providerSaved` flag (MotionPresence's plain-fade enter y 4).
class _SavedToast extends ConsumerWidget {
  const _SavedToast();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool flash = ref.watch(
      settingsUiProvider.select((SettingsUi ui) => ui.savedFlash),
    );
    return MotionPresence(
      present: flash,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: const Color(0x1FE0EFE4), // rgba(224,239,228,0.12)
          borderRadius: BorderRadius.circular(12), // rounded-xl
          border: Border.all(color: const Color(0x4DE0EFE4)), // ...,0.3
        ),
        child: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Icon(Icons.check, size: 13, color: Color(0xFFE0EFE4)),
              SizedBox(width: 8), // gap-2
              Text(
                'Saved successfully.',
                style: TextStyle(
                  fontFamily: PondrTokens.fontFamily,
                  fontSize: 14, // text-sm
                  color: Color(0xFFE0EFE4),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── The new-provider form + the add button ────────────────────────────────

/// The export's new-provider form (`app.tsx:1735-1823`) — `showProviderForm
/// && !editingProviderId` gates it; the enter/exit is its OWN 0.18 s motion
/// (y 8 in / y 6 out). The fields seed from the store's draft (the mock's
/// `providerForm`) and write back on every change; `Add provider` disables
/// on the blank-field guard (`app.tsx:1749`).
class _NewProviderForm extends ConsumerWidget {
  const _NewProviderForm();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final SettingsUi ui = ref.watch(settingsUiProvider);
    final bool present = ui.providerFormOpen && ui.editingProviderId == null;
    return MotionPresence(
      present: present,
      dyIn: 8, // initial y: 8
      dyExit: 6, // exit y: 6
      durationIn: kProviderFormMotion, // 0.18
      durationExit: kProviderFormMotion,
      child: ProviderFormFields(
        key: const ValueKey<String>('provider-form-new'),
        isEdit: false,
      ),
    );
  }
}

/// The export's add-provider button (`app.tsx:1825-1832`) — rendered only
/// while NO provider form is open (`!showProviderForm`, app.tsx:1825; the
/// mock's conditional has no animation).
class _AddProviderButton extends ConsumerWidget {
  const _AddProviderButton();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool hidden = ref.watch(
      settingsUiProvider.select((SettingsUi ui) => ui.providerFormOpen),
    );
    if (hidden) {
      return const SizedBox.shrink();
    }
    return Align(
      alignment: Alignment.centerLeft,
      child: PressScale(
        pressedScale: 0.97, // active:scale-[0.97]
        child: Material(
          key: settingsAddProviderKey,
          color: const Color(0x26888DDF), // rgba(136,141,223,0.15)
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12), // rounded-xl
            side: const BorderSide(color: Color(0x59888DDF)), // ...,0.35
          ),
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            splashColor: Colors.transparent,
            highlightColor: Colors.transparent,
            hoverColor: Colors.transparent,
            onTap: () =>
                ref.read(settingsUiProvider.notifier).openNewProviderForm(),
            child: const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Icon(Icons.add, size: 14, color: PondrTokens.primary),
                  SizedBox(width: 8), // gap-2
                  Text(
                    'Add provider',
                    style: TextStyle(
                      fontFamily: PondrTokens.fontFamily,
                      fontWeight: FontWeight.w600,
                      fontSize: 14, // text-sm
                      color: PondrTokens.primary,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─── One provider row (the accordion) ──────────────────────────────────────

/// One provider's accordion row (`app.tsx:1470-1677`): the header (chevron +
/// name + badge + url/models line + the toggle/pencil/remove affordances)
/// and the height-animated body — the inline edit form, the model list, the
/// add-model flow.
class ProviderRow extends ConsumerWidget {
  const ProviderRow({required this.provider, super.key});

  final mockup.Provider provider;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final SettingsUi ui = ref.watch(settingsUiProvider);
    final bool isOpen = ui.expandedProviderId == provider.id;
    return Container(
      key: Key('settings.provider-row-${provider.id}'),
      clipBehavior: Clip.antiAlias, // rounded-xl overflow-hidden
      decoration: BoxDecoration(
        color: const Color(0x0F888DDF), // rgba(136,141,223,0.06)
        borderRadius: BorderRadius.circular(12), // rounded-xl
        border: Border.all(
          color: isOpen
              ? const Color(0x66888DDF) // rgba(136,141,223,0.4)
              : const Color(0x2E888DDF), // ...,0.18
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          _ProviderHeader(provider: provider, isOpen: isOpen),
          AccordionBody(
            open: isOpen,
            child: _ProviderBody(provider: provider),
          ),
        ],
      ),
    );
  }
}

/// The accordion header (`app.tsx:1400-1445`): the tap toggles the expansion
/// (the export's `setExpandedProviderId(isOpen ? null : p.id)`); the
/// affordances stop the propagation the way nested widgets win the hit test.
class _ProviderHeader extends ConsumerWidget {
  const _ProviderHeader({required this.provider, required this.isOpen});

  final mockup.Provider provider;
  final bool isOpen;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return InkWell(
      key: Key('settings.provider-header-${provider.id}'),
      onTap: () =>
          ref.read(settingsUiProvider.notifier).tapProvider(provider.id),
      splashColor: Colors.transparent,
      highlightColor: Colors.transparent,
      hoverColor: const Color(0x0DFFFFFF), // hover:bg-white/5
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: <Widget>[
            // The chevron's `motion.div animate={{ rotate: isOpen ? 90 : 0 }}
            // transition={{ duration: 0.18 }}` (app.tsx:1405-1408).
            AnimatedRotation(
              turns: isOpen ? 0.25 : 0, // 0° / 90°
              duration: kChevronMotion,
              curve: Curves.easeOut,
              child: const Icon(
                Icons.chevron_right,
                size: 14,
                color: PondrTokens.primary,
              ),
            ),
            const SizedBox(width: 12), // gap-3
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  // The name + the badge (`app.tsx:1407-1420`).
                  Row(
                    children: <Widget>[
                      Flexible(
                        child: Text(
                          provider.name,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8), // gap-2
                      _ProviderBadge(enabled: provider.enabled),
                    ],
                  ),
                  const SizedBox(height: 2), // mt-0.5
                  // `{p.baseUrl.replace(/https?:\/\//, "")} · N models`
                  // (app.tsx:1421-1424).
                  Builder(
                    builder: (BuildContext context) {
                      final int count = provider.models.length;
                      final String url = provider.baseUrl.replaceFirst(
                        RegExp(r'https?://'),
                        '',
                      ); // replace(/https?:\/\//)
                      return Text.rich(
                        TextSpan(
                          children: <TextSpan>[
                            TextSpan(
                              text: '$url${count > 0 ? ' · ' : ''}',
                              style: const TextStyle(
                                fontSize: 12,
                                color: Color(0xFF9B96C8), // #9b96c8
                              ),
                            ),
                            if (count > 0)
                              TextSpan(
                                text: '$count model${count != 1 ? 's' : ''}',
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: PondrTokens.primary,
                                ),
                              ),
                          ],
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      );
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(width: 4), // flex-shrink-0's breathing room
            _HeaderAffordances(provider: provider),
          ],
        ),
      ),
    );
  }
}

/// The Active/Disabled badge (`app.tsx:1407-1417`).
class _ProviderBadge extends StatelessWidget {
  const _ProviderBadge({required this.enabled});

  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: enabled
            ? const Color(0x26E0EFE4) // rgba(224,239,228,0.15)
            : const Color(0x0FFFFFFF), // rgba(255,255,255,0.06)
        borderRadius: BorderRadius.circular(999), // rounded-full
      ),
      child: Text(
        enabled ? 'Active' : 'Disabled',
        style: TextStyle(
          fontSize: 10, // text-[10px]
          fontWeight: FontWeight.w700,
          color: enabled
              ? const Color(0xFFE0EFE4)
              : const Color(0x59FFFFFF), // text-white/35
        ),
      ),
    );
  }
}

/// The header's three affordances (`app.tsx:1438-1445`): toggle
/// (ToggleRight/Left), edit (the pencil EXPANDS + opens the inline form,
/// app.tsx:1422-1424), remove (immediate — the export's `deleteProvider`
/// asks nothing).
class _HeaderAffordances extends ConsumerWidget {
  const _HeaderAffordances({required this.provider});

  final mockup.Provider provider;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        _RowIcon(
          key: Key('settings.provider-toggle-${provider.id}'),
          tooltip: provider.enabled ? 'Disable' : 'Enable',
          onTap: () =>
              ref.read(settingsUiProvider.notifier).toggleProvider(provider.id),
          child: provider.enabled
              ? const Icon(
                  Icons.toggle_on,
                  size: 16,
                  color: PondrTokens.primary,
                )
              : const Icon(
                  Icons.toggle_off,
                  size: 16,
                  color: Color(0x4DFFFFFF),
                ),
        ),
        _RowIcon(
          key: Key('settings.provider-edit-${provider.id}'),
          tooltip: 'Edit provider',
          onTap: () =>
              ref.read(settingsUiProvider.notifier).openEditProvider(provider),
          child: const Icon(
            Icons.edit_outlined,
            size: 12,
            color: Color(0x59FFFFFF),
          ),
        ),
        _RowIcon(
          key: Key('settings.provider-delete-${provider.id}'),
          tooltip: 'Remove provider',
          onTap: () =>
              ref.read(settingsUiProvider.notifier).deleteProvider(provider.id),
          child: const Icon(
            Icons.delete_outline,
            size: 12,
            color: Color(0x59FFFFFF),
          ),
        ),
      ],
    );
  }
}

/// The p-1.5 icon affordance over `hover:bg-white/10`.
class _RowIcon extends StatelessWidget {
  const _RowIcon({
    required this.child,
    required this.tooltip,
    required this.onTap,
    super.key,
  });

  final Widget child;
  final String tooltip;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      waitDuration: const Duration(milliseconds: 400),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8), // rounded-lg
        splashColor: Colors.transparent,
        highlightColor: Colors.transparent,
        hoverColor: const Color(0x1AFFFFFF), // hover:bg-white/10
        child: Padding(
          padding: const EdgeInsets.all(6), // p-1.5
          child: child,
        ),
      ),
    );
  }
}

// ─── The accordion body ────────────────────────────────────────────────────

/// The body's column (`app.tsx:1444-1675`): the inline edit-provider form,
/// then the models list / the empty state, then the add-model flow.
class _ProviderBody extends ConsumerWidget {
  const _ProviderBody({required this.provider});

  final mockup.Provider provider;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final SettingsUi ui = ref.watch(settingsUiProvider);
    final bool showEditForm =
        ui.providerFormOpen && ui.editingProviderId == provider.id;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16), // px-4 pb-4
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          const SizedBox(height: 8), // the border-t gap (space-y-2)
          const Divider(
            color: Color(0x26888DDF), // rgba(136,141,223,0.15)
            thickness: 1,
            height: 0,
          ),
          const SizedBox(height: 8),
          // The inline edit form (`app.tsx:1448-1522`).
          MotionPresence(
            present: showEditForm,
            child: ProviderFormFields(
              key: ValueKey<String>('provider-form-edit-${provider.id}'),
              isEdit: true,
            ),
          ),
          // The models list (`app.tsx:1525-1615`).
          if (provider.models.isNotEmpty) ...<Widget>[
            const SizedBox(height: 12), // mt-3
            const Text(
              'Models',
              style: TextStyle(
                fontSize: 10, // text-[10px]
                fontWeight: FontWeight.w700,
                letterSpacing: 1.0, // tracking-widest
                color: Color(0xB3888DDF), // rgba(136,141,223,0.7)
              ),
            ),
            const SizedBox(height: 8), // mb-2
            Column(
              children: <Widget>[
                for (final mockup.ProviderModel m in provider.models)
                  ModelRow(providerId: provider.id, model: m),
              ],
            ),
          ],
          // The empty state (`app.tsx:1640-1644`).
          if (provider.models.isEmpty && ui.addingModelToId != provider.id)
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 0, vertical: 12),
              child: Center(
                child: Text(
                  'No models added yet.',
                  style: TextStyle(
                    fontFamily: PondrTokens.fontFamily,
                    fontSize: 12,
                    color: Color(0xFF9B96C8),
                  ),
                ),
              ),
            ),
          const SizedBox(height: 8), // mt-2
          // The add-model form while open (app.tsx:1646-1706).
          MotionPresence(
            present:
                ui.addingModelToId == provider.id && ui.editingModelId == null,
            child: ModelFormFields(
              key: ValueKey<String>('model-form-add-${provider.id}'),
              providerId: provider.id,
              isEdit: false,
            ),
          ),
          // The add-model button, hidden while the form is open
          // (`addingModelToId !== p.id`, app.tsx:1709-1719).
          if (ui.addingModelToId != provider.id) ...<Widget>[
            const SizedBox(height: 4), // mt-1
            _AddModelButton(providerId: provider.id),
          ],
        ],
      ),
    );
  }
}

/// The export's `Add model` pill (`app.tsx:1709-1719`).
class _AddModelButton extends ConsumerWidget {
  const _AddModelButton({required this.providerId});

  final String providerId;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Material(
        key: Key('settings.add-model-$providerId'),
        color: const Color(0x1A888DDF), // rgba(136,141,223,0.1)
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8), // rounded-lg
          side: const BorderSide(color: Color(0x38888DDF)), // ...,0.22
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(8),
          splashColor: Colors.transparent,
          highlightColor: Colors.transparent,
          hoverColor: Colors.transparent,
          onTap: () => ref
              .read(settingsUiProvider.notifier)
              .openAddModelForm(providerId),
          child: const Padding(
            padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: <Widget>[
                Icon(Icons.add, size: 12, color: PondrTokens.primary),
                SizedBox(width: 6), // gap-1.5
                Text(
                  'Add model',
                  style: TextStyle(
                    fontFamily: PondrTokens.fontFamily,
                    fontWeight: FontWeight.w600,
                    fontSize: 12, // text-xs
                    color: PondrTokens.primary,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─── The provider forms (new + inline edit) ────────────────────────────────

/// The export's provider FORM — one field set for both gates: the NEW form
/// (`app.tsx:1735-1823`) and the inline EDIT form (`app.tsx:1448-1522`,
/// compact paddings + the `Edit provider` header). Controllers seed from the
/// store's draft and write back per change.
class ProviderFormFields extends ConsumerStatefulWidget {
  const ProviderFormFields({required this.isEdit, super.key});

  /// TRUE renders the inline EDIT variant (compact boxes + the Edit provider
  /// header; FALSE the New provider sheet).
  final bool isEdit;

  @override
  ConsumerState<ProviderFormFields> createState() => _ProviderFormFieldsState();
}

class _ProviderFormFieldsState extends ConsumerState<ProviderFormFields> {
  late final TextEditingController _name = TextEditingController(
    text: ref.read(settingsUiProvider).providerForm.name,
  );
  late final TextEditingController _url = TextEditingController(
    text: ref.read(settingsUiProvider).providerForm.baseUrl,
  );
  late final TextEditingController _key = TextEditingController(
    text: ref.read(settingsUiProvider).providerForm.apiKey,
  );

  @override
  void dispose() {
    _name.dispose();
    _url.dispose();
    _key.dispose();
    super.dispose();
  }

  // The guard (`app.tsx:1183`): the Save button disabled while the name or
  // the baseUrl is blank. Each keystroke DOUBLE-writes: the store's draft
  // (the mock's `providerForm` — the seed of a reopened form) and the local
  // `_enabled` (the disabled state re-renders without a riverpod round-trip).
  void _onField(String field, String value) {
    ref.read(settingsUiProvider.notifier).setProviderField(field, value);
    final bool wasEnabled = _enabled;
    _enabled = _name.text.trim().isNotEmpty && _url.text.trim().isNotEmpty;
    if (_enabled != wasEnabled) {
      setState(() {});
    }
  }

  bool _enabled = false;

  @override
  void initState() {
    super.initState();
    _enabled = _name.text.trim().isNotEmpty && _url.text.trim().isNotEmpty;
  }

  @override
  Widget build(BuildContext context) {
    final SettingsUi ui = ref.watch(settingsUiProvider);
    final bool keyVisible = ui.providerKeyVisible;
    final bool edit = widget.isEdit;

    final List<Widget> fields = <Widget>[
      _FormField(
        label: 'Provider name',
        placeholder: 'e.g. OpenAI, Groq, Ollama',
        fieldKey: settingsProviderNameFieldKey,
        controller: _name,
        onChanged: (String v) => _onField('name', v),
        compact: edit,
      ),
      _FormField(
        label: 'Base URL',
        placeholder: 'https://api.openai.com/v1',
        fieldKey: settingsProviderUrlFieldKey,
        controller: _url,
        onChanged: (String v) => _onField('baseUrl', v),
        compact: edit,
      ),
    ];

    return DecoratedBox(
      decoration: BoxDecoration(
        color: edit
            ? const Color(0x1A888DDF) // rgba(136,141,223,0.1)
            : const Color(0x14888DDF), // rgba(136,141,223,0.08)
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: edit
              ? const Color(0x4D888DDF) // rgba(136,141,223,0.3)
              : const Color(0x59888DDF), // ...,0.35
        ),
      ),
      child: Padding(
        padding: EdgeInsets.all(edit ? 16 : 20), // p-4 / p-5
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            // The two forms' titles: `Edit provider` (app.tsx:1451) /
            // `New provider` (app.tsx:1740).
            Text(
              edit ? 'Edit provider' : 'New provider',
              style: edit
                  ? const TextStyle(
                      fontFamily: PondrTokens.fontDisplay,
                      fontWeight: FontWeight.w700,
                      fontSize: 12, // text-xs
                      letterSpacing: 1.2,
                      color: Color(0xCCFFFFFF), // text-white/80
                    )
                  : const TextStyle(
                      fontFamily: PondrTokens.fontDisplay,
                      fontWeight: FontWeight.w700,
                      fontSize: 14, // text-sm
                      color: Colors.white,
                    ),
            ),
            SizedBox(height: edit ? 12 : 16),
            ..._spread(fields, edit ? 12 : 16), // space-y-3 / space-y-4
            // The API key field + its eye (`app.tsx:1474-1521`).
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: <Widget>[
                const SettingsFieldLabel('API key'),
                SizedBox(height: edit ? 4 : 6), // mb-1 / mb-1.5
                TextField(
                  key: settingsProviderKeyFieldKey,
                  controller: _key,
                  obscureText: !keyVisible, // type password / text
                  enableSuggestions: false,
                  autocorrect: false,
                  cursorColor: Colors.white,
                  style: kSettingsInputStyle,
                  onChanged: (String v) => _onField('apiKey', v),
                  decoration: settingsFieldDecoration(
                    hint: 'sk-••••••••••••••••',
                    compact: edit,
                    border: edit
                        ? const Color(0x47888DDF) // rgba(136,141,223,0.28)
                        : const Color(0x59888DDF),
                    prefixIcon: Icon(
                      Icons.key,
                      size: edit ? 13 : 14, // KeyRound 13 / 14
                      color: PondrTokens.primary,
                    ),
                    suffixIcon: IconButton(
                      key: settingsProviderEyeKey,
                      padding: EdgeInsets.only(right: edit ? 12 : 14),
                      constraints: const BoxConstraints(
                        minWidth: 30,
                        minHeight: 30,
                      ),
                      iconSize: edit ? 13 : 14,
                      color: const Color(0x59FFFFFF), // white/35-ish
                      icon: Icon(
                        keyVisible ? Icons.visibility_off : Icons.visibility,
                      ),
                      onPressed: () => ref
                          .read(settingsUiProvider.notifier)
                          .toggleProviderKey(),
                    ),
                  ),
                ),
                // The new form's storage note (`app.tsx:1620-1624`).
                if (!edit) ...<Widget>[
                  const SizedBox(height: 6), // mt-1.5
                  const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Icon(
                        Icons.error_outline,
                        size: 10,
                        color: Color(0xFF9B96C8),
                      ),
                      SizedBox(width: 6), // gap-1.5
                      Text(
                        'Stored locally in your browser only.',
                        style: TextStyle(
                          fontFamily: PondrTokens.fontFamily,
                          fontSize: 12,
                          color: Color(0xFF9B96C8),
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
            const SizedBox(height: 4), // pt-1
            // The Save/Cancel pair (app.tsx:1500-1512 / 1625-1636).
            Row(
              children: <Widget>[
                PressScale(
                  pressedScale: 0.97, // active:scale-[0.97]
                  child: Material(
                    key: settingsProviderSaveKey,
                    color: PondrTokens.primary,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(edit ? 8 : 12),
                    ),
                    child: InkWell(
                      onTap: _enabled
                          ? () {
                              ref
                                  .read(settingsUiProvider.notifier)
                                  .submitProvider();
                            }
                          : null,
                      child: Opacity(
                        opacity: _enabled ? 1 : 0.4, // disabled:opacity-40
                        child: Padding(
                          padding: EdgeInsets.symmetric(
                            horizontal: edit ? 16 : 20,
                            vertical: edit ? 8 : 10,
                          ),
                          child: Text(
                            edit ? 'Save' : 'Add provider',
                            style: TextStyle(
                              fontFamily: PondrTokens.fontDisplay,
                              fontWeight: FontWeight.w700,
                              fontSize: edit ? 14 : 14,
                              color: const Color(0xFF0C0B1A),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8), // gap-2
                Material(
                  key: settingsProviderCancelKey,
                  color: const Color(0x0FFFFFFF), // rgba(255,255,255,0.06)
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(edit ? 8 : 12),
                    side: const BorderSide(color: Color(0x1AFFFFFF)), // ...,0.1
                  ),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(edit ? 8 : 12),
                    splashColor: Colors.transparent,
                    highlightColor: Colors.transparent,
                    hoverColor: Colors.transparent,
                    onTap: () => ref
                        .read(settingsUiProvider.notifier)
                        .closeProviderForm(),
                    child: Padding(
                      padding: EdgeInsets.symmetric(
                        horizontal: edit ? 16 : 20,
                        vertical: edit ? 8 : 10,
                      ),
                      child: Text(
                        'Cancel',
                        style: TextStyle(
                          fontFamily: PondrTokens.fontFamily,
                          fontSize: 14,
                          fontWeight: edit ? FontWeight.w400 : FontWeight.w500,
                          color: Color(edit ? 0x8CFFFFFF : 0x99FFFFFF),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ─── The model rows + forms ────────────────────────────────────────────────

/// One model row (`app.tsx:1550-1584`) + its inline edit form
/// (`app.tsx:1586-1629`).
class ModelRow extends ConsumerWidget {
  const ModelRow({required this.providerId, required this.model, super.key});

  final String providerId;
  final mockup.ProviderModel model;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final SettingsUi ui = ref.watch(settingsUiProvider);
    final bool editingThis =
        ui.addingModelToId == providerId && ui.editingModelId == model.id;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        DecoratedBox(
          decoration: BoxDecoration(
            color: const Color(0x0AFFFFFF), // rgba(255,255,255,0.04)
            borderRadius: BorderRadius.circular(8), // rounded-lg
            border: Border.all(
              color: const Color(0x24888DDF), // rgba(136,141,223,0.14)
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: Row(
              children: <Widget>[
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Text(
                        // The picker's display truth: `m.label || m.modelId`
                        // (app.tsx:2260).
                        model.label.isEmpty ? model.modelId : model.label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontWeight: FontWeight.w500, // font-medium
                          fontSize: 14,
                          color: Colors.white,
                        ),
                      ),
                      // The modelId line ONLY when a label covers it
                      // (app.tsx:1555).
                      if (model.label.isNotEmpty)
                        Text(
                          model.modelId,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontFamily: PondrTokens.fontFamily,
                            fontSize: 12,
                            color: PondrTokens.primary,
                          ),
                        ),
                    ],
                  ),
                ),
                _RowIcon(
                  key: Key('settings.model-toggle-$providerId-${model.id}'),
                  tooltip: model.enabled ? 'Disable' : 'Enable',
                  onTap: () => ref
                      .read(settingsUiProvider.notifier)
                      .toggleModel(providerId, model.id),
                  child: model.enabled
                      ? const Icon(
                          Icons.toggle_on,
                          size: 15,
                          color: PondrTokens.primary,
                        )
                      : const Icon(
                          Icons.toggle_off,
                          size: 15,
                          color: Color(0x40FFFFFF), // text-white/25
                        ),
                ),
                _RowIcon(
                  key: Key('settings.model-edit-$providerId-${model.id}'),
                  tooltip: 'Edit model',
                  onTap: () => ref
                      .read(settingsUiProvider.notifier)
                      .openEditModelForm(providerId, model),
                  child: const Icon(
                    Icons.edit_outlined,
                    size: 11,
                    color: Color(0x4DFFFFFF), // text-white/30
                  ),
                ),
                _RowIcon(
                  key: Key('settings.model-delete-$providerId-${model.id}'),
                  tooltip: 'Remove model',
                  onTap: () => ref
                      .read(settingsUiProvider.notifier)
                      .deleteModel(providerId, model.id),
                  child: const Icon(
                    Icons.delete_outline,
                    size: 11,
                    color: Color(0x4DFFFFFF),
                  ),
                ),
              ],
            ),
          ),
        ),
        // The inline edit form — inside the model's row wrapper
        // (`app.tsx:1586-1629`).
        MotionPresence(
          present: editingThis,
          child: ModelFormFields(
            key: ValueKey<String>('model-form-edit-${model.id}'),
            providerId: providerId,
            isEdit: true,
          ),
        ),
      ],
    );
  }
}

/// The export's `Add model` form — folded into [ModelFormFields]'s
/// `isEdit: false` variant (the body renders it behind MotionPresence).

/// The model form — one field set for the ADD gate (`app.tsx:1645-1708`, the
/// `Add model` title + the blank-modelId disabled button) and the EDIT gate
/// (`app.tsx:1590-1629`). Controllers seed from the store's `modelForm`.
class ModelFormFields extends ConsumerStatefulWidget {
  const ModelFormFields({
    required this.providerId,
    required this.isEdit,
    super.key,
  });

  final String providerId;
  final bool isEdit;

  @override
  ConsumerState<ModelFormFields> createState() => _ModelFormFieldsState();
}

class _ModelFormFieldsState extends ConsumerState<ModelFormFields> {
  late final TextEditingController _modelId = TextEditingController(
    text: ref.read(settingsUiProvider).modelForm.modelId,
  );
  late final TextEditingController _label = TextEditingController(
    text: ref.read(settingsUiProvider).modelForm.label,
  );

  bool _enabled = false;

  @override
  void initState() {
    super.initState();
    _enabled = _modelId.text.trim().isNotEmpty;
  }

  @override
  void dispose() {
    _modelId.dispose();
    _label.dispose();
    super.dispose();
  }

  void _onField(String field, String value) {
    ref.read(settingsUiProvider.notifier).setModelField(field, value);
    final bool wasEnabled = _enabled;
    _enabled = _modelId.text.trim().isNotEmpty;
    if (_enabled != wasEnabled) {
      setState(() {});
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool edit = widget.isEdit;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0x1A888DDF), // rgba(136,141,223,0.1)
        borderRadius: BorderRadius.circular(8), // rounded-lg
        border: Border.all(
          color: edit
              ? const Color(0x47888DDF) // rgba(136,141,223,0.28)
              : const Color(0x47888DDF),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12), // p-3
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            if (!edit)
              Text(
                'Add model',
                style: TextStyle(
                  fontSize: 10, // text-[10px]
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                  color: const Color(0xCC888DDF), // rgba(136,141,223,0.8)
                ),
              ),
            const SizedBox(height: 10), // space-y-2.5
            // The grid: Model ID / Display label, two columns (grid-cols-2).
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Text(
                        'Model ID',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.0,
                          color: const Color(0x99ECE9FF), // ...,0.6
                        ),
                      ),
                      const SizedBox(height: 4), // mb-1
                      TextField(
                        key: settingsModelIdFieldKey,
                        controller: _modelId,
                        cursorColor: Colors.white,
                        style: kSettingsInputStyle,
                        decoration: settingsFieldDecoration(
                          hint: 'gpt-4o',
                          compact: true,
                          border: const Color(0x47888DDF),
                        ),
                        onChanged: (String v) => _onField('modelId', v),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8), // gap-2
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Text(
                        'Display label',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.0,
                          color: const Color(0x99ECE9FF),
                        ),
                      ),
                      const SizedBox(height: 4),
                      TextField(
                        key: settingsModelLabelFieldKey,
                        controller: _label,
                        cursorColor: Colors.white,
                        style: kSettingsInputStyle,
                        decoration: settingsFieldDecoration(
                          hint: edit ? 'GPT-4o' : 'GPT-4o (optional)',
                          compact: true,
                          border: const Color(0x47888DDF),
                        ),
                        onChanged: (String v) => _onField('label', v),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: <Widget>[
                Material(
                  key: settingsModelSaveKey,
                  color: PondrTokens.primary,
                  shape: const RoundedRectangleBorder(
                    borderRadius: BorderRadius.all(Radius.circular(8)),
                  ),
                  child: InkWell(
                    onTap: _enabled
                        ? () => ref
                              .read(settingsUiProvider.notifier)
                              .submitModel(widget.providerId)
                        : null,
                    child: Opacity(
                      opacity: _enabled ? 1 : 0.4, // disabled:opacity-40
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12, // px-3
                          vertical: 6, // py-1.5
                        ),
                        child: Text(
                          edit ? 'Save' : 'Add model',
                          style: const TextStyle(
                            fontFamily: PondrTokens.fontFamily,
                            fontWeight: FontWeight.w700,
                            fontSize: 12, // text-xs
                            color: Color(0xFF0C0B1A),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8), // gap-2
                Material(
                  key: settingsModelCancelKey,
                  color: const Color(0x0FFFFFFF),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                    side: const BorderSide(color: Color(0x1AFFFFFF)),
                  ),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(8),
                    splashColor: Colors.transparent,
                    highlightColor: Colors.transparent,
                    hoverColor: Colors.transparent,
                    onTap: () =>
                        ref.read(settingsUiProvider.notifier).closeModelForm(),
                    child: const Padding(
                      padding: EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      child: Text(
                        'Cancel',
                        style: TextStyle(
                          fontFamily: PondrTokens.fontFamily,
                          fontSize: 12,
                          color: Color(0x80FFFFFF), // text-white/50
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Small section utils ───────────────────────────────────────────────────

/// One provider-form field: the uppercase label over the input — the NEW
/// form's bigger box/label (`app.tsx:1762-1770`) or the EDIT form's compact
/// one (`app.tsx:1462-1470`).
class _FormField extends StatelessWidget {
  const _FormField({
    required this.label,
    required this.placeholder,
    required this.fieldKey,
    required this.controller,
    required this.onChanged,
    required this.compact,
  });

  final String label;
  final String placeholder;
  final Key fieldKey;
  final TextEditingController controller;
  final ValueChanged<String> onChanged;

  /// TRUE for the inline EDIT variant.
  final bool compact;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        SettingsFieldLabel(
          label,
          color: compact
              ? const Color(0x99ECE9FF) // rgba(236,233,255,0.6)
              : const Color(0xB3ECE9FF), // rgba(236,233,255,0.7)
        ),
        SizedBox(height: compact ? 4 : 6), // mb-1 / mb-1.5
        TextField(
          key: fieldKey,
          controller: controller,
          cursorColor: Colors.white,
          style: kSettingsInputStyle,
          decoration: settingsFieldDecoration(
            hint: placeholder,
            compact: compact,
          ),
          onChanged: onChanged,
        ),
      ],
    );
  }
}

/// `space-y-*` for a list of widgets: gaps BETWEEN the children.
List<Widget> _spread(List<Widget> children, double gap) {
  final List<Widget> out = <Widget>[];
  for (int i = 0; i < children.length; i++) {
    if (i > 0) {
      out.add(SizedBox(height: gap));
    }
    out.add(children[i]);
  }
  return out;
}
