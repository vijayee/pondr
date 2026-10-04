import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../theme/tokens.dart';
import 'chat_state.dart';
import 'common.dart';

/// The export's model picker (`app.tsx:2266-2329`): the bottom bar's button
/// + the dropdown above it, fed by the ENABLED providers' models (the
/// settings service via `chat_state.dart`'s stores), the selection
/// persisted through [SettingsService.selectedModelKey].
class ModelPicker extends ConsumerWidget {
  const ModelPicker({super.key});

  /// The dropdown's entrance (the export's `app.tsx:2293-2296`:
  /// `0.15 s`, `y: 6 → 0`, `scale 0.97 → 1`).
  static const Duration menuDuration = Duration(milliseconds: 150);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final List<AvailableModel> models = ref.watch(availableModelsProvider);
    final AvailableModel? selected = ref.watch(selectedModelProvider);
    final bool open = ref.watch(modelPickerOpenProvider);

    return AnchoredMenu(
      open: open,
      onDismiss: () =>
          ref.read(modelPickerOpenProvider.notifier).set(false),
      menuBuilder: (VoidCallback dismiss) => _ModelMenu(
        models: models,
        selectedKey: selected?.key,
        hasPersistedKey: ref.watch(modelSelectionProvider) != null,
        onSelect: (String key) {
          ref.read(modelSelectionProvider.notifier).select(key);
          ref.read(modelPickerOpenProvider.notifier).set(false);
        },
      ),
      anchor: _PickerButton(
        selected: selected,
        hasOptions: models.isNotEmpty,
        manyOptions: models.length > 1,
        onTap: models.isEmpty
            ? null
            : () => ref.read(modelPickerOpenProvider.notifier).toggle(),
      ),
    );
  }
}

/// The button (`app.tsx:2268-2285`): the Bot glyph, the
/// `provider · label` readout (or "No model selected"), the chevron only
/// when a choice exists. Unpicked state renders borderless + dimmed.
class _PickerButton extends StatelessWidget {
  const _PickerButton({
    required this.selected,
    required this.hasOptions,
    required this.manyOptions,
    required this.onTap,
  });

  final AvailableModel? selected;
  final bool hasOptions;
  final bool manyOptions;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final bool picked = selected != null;
    final Widget button = InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          key: const Key('model.picker'),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: picked ? const Color(0x1F888DDF) : Colors.transparent,
            border: picked
                ? Border.all(color: const Color(0x47888DDF))
                : Border.all(color: Colors.transparent),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Icon(
                Icons.smart_toy,
                size: 11,
                color: picked ? PondrTokens.primary : const Color(0x40FFFFFF),
              ),
              const SizedBox(width: 6),
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 160),
                child: Text(
                  picked
                      ? '${selected!.providerName} · ${selected!.label}'
                      : 'No model selected',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: picked
                        ? const Color(0xFFC3ACDA)
                        : const Color(0x40FFFFFF),
                  ),
                ),
              ),
              if (manyOptions) ...<Widget>[
                const SizedBox(width: 4),
                Icon(
                  Icons.keyboard_arrow_down,
                  size: 10,
                  color: const Color(0x99888DDF),
                ),
              ],
            ],
          ),
        ),
      );
    return hasOptions
        ? button
        : Tooltip(
            // The mock's hint at the empty pool (`app.tsx:2275`).
            message: 'Add a provider in Settings → Providers',
            child: button,
          );
  }
}

/// The dropdown (`app.tsx:2292-2325`): one section per provider carrying
/// enabled models, the active row checked, the panel animating in per the
/// export's 0.15 s `y: 6` + `0.97` scale entrance.
class _ModelMenu extends StatelessWidget {
  const _ModelMenu({
    required this.models,
    required this.selectedKey,
    required this.hasPersistedKey,
    required this.onSelect,
  });

  final List<AvailableModel> models;
  final String? selectedKey;

  /// FALSE = the export's unpersisted state, where the FIRST model is
  /// active by fallback (`app.tsx:2307`).
  final bool hasPersistedKey;
  final void Function(String key) onSelect;

  @override
  Widget build(BuildContext context) {
    return Material(
      type: MaterialType.transparency,
      child: MenuPanelEntrance(
        delay: const Duration(milliseconds: 6),
        dy: 6,
        initialScale: 0.97,
        duration: ModelPicker.menuDuration,
        child: Container(
          key: const Key('model.menu'),
          decoration: pondrMenuDecoration,
          constraints: const BoxConstraints(minWidth: 220),
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              // The export groups its dropdown by provider section headers
              // (`app.tsx:2300-2324`).
              for (int i = 0; i < models.length; i++) ...<Widget>[
                if (i == 0 ||
                    models[i].providerName != models[i - 1].providerName)
                  _ProviderHeader(models[i].providerName),
                _modelRow(models[i]),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _modelRow(AvailableModel model) {
    final bool isActive =
        model.key == selectedKey ||
        (!hasPersistedKey && models.isNotEmpty && model.key == models.first.key);
    return InkWell(
      key: Key('model.item-${model.key}'),
      onTap: () => onSelect(model.key),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        color: isActive ? const Color(0x1F888DDF) : null,
        child: Row(
          children: <Widget>[
            Expanded(
              child: Text(
                model.label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 14,
                  color: isActive ? Colors.white : const Color(0xA6FFFFFF),
                ),
              ),
            ),
            Text(
              model.modelId,
              style: const TextStyle(fontSize: 10, color: Color(0x99888DDF)),
            ),
            if (isActive)
              const Icon(Icons.check, size: 11, color: PondrTokens.primary),
          ],
        ),
      ),
    );
  }
}

/// The dropdown's section header (the export's `app.tsx:2302-2304`).
class _ProviderHeader extends StatelessWidget {
  const _ProviderHeader(this.name);

  final String name;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
      child: Text(
        name,
        style: const TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.4,
          color: Color(0xB3888DDF),
        ),
      ),
    );
  }
}