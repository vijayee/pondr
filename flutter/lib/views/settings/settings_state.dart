/// The settings view's state — the export's `App()` "Settings" state block
/// (`app.tsx:820-847`) as riverpod stores: the persisted half lives in the
/// [SettingsService] (profile / notifications / accent / providers), the
/// EPHEMERAL half (which section, which accordion is open, the add/edit
/// forms' presence + drafts, the saved-flash) is THIS store — the mock's
/// view-local `useState` bag in full.
///
/// Reactivity: the settings service has no change stream in v1, so every
/// mutation here bumps [settingsRevisionProvider] (chat_state.dart's
/// "Task 6's wiring point") — the page re-reads the service on it and the
/// chat's model picker refreshes through the same bump.
library;

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/bindings.dart' show settingsProvider;
import '../../data/services.dart';
// The MODEL's `Provider` class aliases the riverpod one (the
// chat_state.dart idiom).
import '../../data/models.dart' as mockup show Provider, ProviderModel;
import '../chat/chat_state.dart' show settingsRevisionProvider;

/// The export's `SETTINGS_NAV` identity (`app.tsx:1157-1164`) — the six
/// sections in their sidebar order.
enum SettingsSection {
  profile,
  appearance,
  providers,
  notifications,
  security,
  about,
}

extension SettingsSectionLabel on SettingsSection {
  /// The nav's label; the enum name is the export's id too.
  String get label => switch (this) {
    SettingsSection.profile => 'Profile',
    SettingsSection.appearance => 'Appearance',
    SettingsSection.providers => 'Providers',
    SettingsSection.notifications => 'Notifications',
    SettingsSection.security => 'Security',
    SettingsSection.about => 'About',
  };
}

/// The provider form's draft — the export's `providerForm` shape
/// (`app.tsx:826`). A record type over { name, baseUrl, apiKey }.
typedef ProviderForm = ({String name, String baseUrl, String apiKey});

/// The model form's draft — the export's `modelForm` shape
/// (`app.tsx:833`). A record type over { modelId, label }.
typedef ModelForm = ({String modelId, String label});

/// One immutable snapshot of the export's settings view-local state.
class SettingsUi {
  const SettingsUi({
    this.section = SettingsSection.profile,
    this.navOpen = false,
    this.expandedProviderId,
    this.providerFormOpen = false,
    this.editingProviderId,
    this.providerForm = (name: '', baseUrl: '', apiKey: ''),
    this.providerKeyVisible = false,
    this.addingModelToId,
    this.editingModelId,
    this.modelForm = (modelId: '', label: ''),
    this.savedFlash = false,
  });

  /// `settingsSection` (`app.tsx:820`); boots "profile".
  final SettingsSection section;

  /// The NARROW layout's nav drawer (the plan's drawer idiom; the export's
  /// only settings shape is the wide aside — see SettingsView's doc).
  final bool navOpen;

  /// `expandedProviderId` (`app.tsx:824`).
  final String? expandedProviderId;

  /// `showProviderForm` (`app.tsx:825`).
  final bool providerFormOpen;

  /// `editingProviderId` (`app.tsx:826`).
  final String? editingProviderId;

  /// `providerForm` (`app.tsx:827`).
  final ProviderForm providerForm;

  /// `showProviderKey` (`app.tsx:828`) — SHARED by the new-provider and
  /// the inline-edit forms, as in the export.
  final bool providerKeyVisible;

  /// `addingModelToId` (`app.tsx:829`).
  final String? addingModelToId;

  /// `editingModelId` (`app.tsx:830`).
  final String? editingModelId;

  /// `modelForm` (`app.tsx:831`).
  final ModelForm modelForm;

  /// `providerSaved` (`app.tsx:835`) — the saved-flash's presence.
  final bool savedFlash;

  SettingsUi copyWith({
    SettingsSection? section,
    bool? navOpen,
    Object? expandedProviderId = _sentinel,
    bool? providerFormOpen,
    Object? editingProviderId = _sentinel,
    ProviderForm? providerForm,
    bool? providerKeyVisible,
    Object? addingModelToId = _sentinel,
    Object? editingModelId = _sentinel,
    ModelForm? modelForm,
    bool? savedFlash,
  }) {
    return SettingsUi(
      section: section ?? this.section,
      navOpen: navOpen ?? this.navOpen,
      expandedProviderId: expandedProviderId == _sentinel
          ? this.expandedProviderId
          : expandedProviderId as String?,
      providerFormOpen: providerFormOpen ?? this.providerFormOpen,
      editingProviderId: editingProviderId == _sentinel
          ? this.editingProviderId
          : editingProviderId as String?,
      providerForm: providerForm ?? this.providerForm,
      providerKeyVisible: providerKeyVisible ?? this.providerKeyVisible,
      addingModelToId: addingModelToId == _sentinel
          ? this.addingModelToId
          : addingModelToId as String?,
      editingModelId: editingModelId == _sentinel
          ? this.editingModelId
          : editingModelId as String?,
      modelForm: modelForm ?? this.modelForm,
      savedFlash: savedFlash ?? this.savedFlash,
    );
  }
}

/// copyWith's nullable-field discriminator (fields legitimately go to null).
const Object _sentinel = _Sentinel();

class _Sentinel {
  const _Sentinel();
}

/// The export's settings view-local handlers (`app.tsx:1166-1246`) plus the
/// service mutations behind them. The saved-flash's 2200 ms reset
/// (`app.tsx:1194`) runs on a [Timer] here — cancelled on the store's
/// disposal (the export's bare `setTimeout` never cancels; ours must not
/// fire into a disposed provider).
class SettingsUiStore extends Notifier<SettingsUi> {
  Timer? _flashTimer;

  @override
  SettingsUi build() {
    ref.onDispose(() {
      _flashTimer?.cancel();
      _flashTimer = null;
    });
    return const SettingsUi();
  }

  SettingsService get _settings => ref.read(settingsProvider);

  void _bump() => ref.read(settingsRevisionProvider.notifier).bump();

  // ── Section + the narrow nav drawer ─────────────────────────────────────

  /// `setSettingsSection` (`app.tsx:1283`).
  void setSection(SettingsSection section) =>
      state = state.copyWith(section: section);

  void openNav() => state = state.copyWith(navOpen: true);

  void closeNav() => state = state.copyWith(navOpen: false);

  // ── The providers accordion + CRUD (app.tsx:1166-1246) ──────────────────

  /// The accordion header's click (`app.tsx:1402`): open or toggle.
  void tapProvider(String id) => state = state.copyWith(
    expandedProviderId: state.expandedProviderId == id ? null : id,
  );

  /// `openNewProviderForm` (`app.tsx:1166-1174`): reset the form, close the
  /// model forms, then show.
  void openNewProviderForm() {
    state = state.copyWith(
      providerForm: const (name: '', baseUrl: '', apiKey: ''),
      providerKeyVisible: false,
      addingModelToId: null,
      editingModelId: null,
      providerFormOpen: true,
      editingProviderId: null,
    );
  }

  /// `openEditProviderForm` (`app.tsx:1176-1181`) + the call-site's
  /// `setExpandedProviderId(p.id)` (`app.tsx:1423`).
  void openEditProvider(mockup.Provider p) {
    state = state.copyWith(
      providerForm: (name: p.name, baseUrl: p.baseUrl, apiKey: p.apiKey),
      providerKeyVisible: false,
      providerFormOpen: true,
      editingProviderId: p.id,
      expandedProviderId: p.id,
    );
  }

  /// The two cancels (`app.tsx:1667-1669`, `app.tsx:1717`).
  void closeProviderForm() {
    state = state.copyWith(providerFormOpen: false, editingProviderId: null);
  }

  void setProviderField(String field, String value) {
    final ProviderForm form = state.providerForm;
    final ProviderForm next = switch (field) {
      'name' => (name: value, baseUrl: form.baseUrl, apiKey: form.apiKey),
      'baseUrl' => (name: form.name, baseUrl: value, apiKey: form.apiKey),
      _ => (name: form.name, baseUrl: form.baseUrl, apiKey: value),
    };
    state = state.copyWith(providerForm: next);
  }

  /// The export's eye toggle (`app.tsx:1479` / `1609`).
  void toggleProviderKey() =>
      state = state.copyWith(providerKeyVisible: !state.providerKeyVisible);

  /// `saveProvider` (`app.tsx:1182-1196`): the guard + the service write,
  /// then close the form, EXPAND a created provider and flash the toast.
  ///
  /// Returns the saved id, or null on the blank-field guard (the disabled
  /// button already keeps it from most taps; the service guards again).
  String? submitProvider() {
    final String? id = _settings.saveProvider(
      editingId: state.editingProviderId,
      name: state.providerForm.name,
      baseUrl: state.providerForm.baseUrl,
      apiKey: state.providerForm.apiKey,
    );
    if (id == null) {
      return null;
    }
    final bool wasCreated = state.editingProviderId == null;
    state = state.copyWith(providerFormOpen: false, editingProviderId: null);
    // The create branch's `setExpandedProviderId(id)` (app.tsx:1190) — an
    // EDIT save leaves the expansion alone.
    if (wasCreated) {
      state = state.copyWith(expandedProviderId: id);
    }
    _flash();
    _bump();
    return id;
  }

  /// `deleteProvider` (`app.tsx:1198-1202`): immediate — no confirmation;
  /// a deleted provider's accordion and edit form close with it.
  void deleteProvider(String id) {
    _settings.deleteProvider(id);
    final bool wasExpanded = state.expandedProviderId == id;
    final bool wasEditing = state.editingProviderId == id;
    SettingsUi next = state;
    if (wasExpanded) {
      next = next.copyWith(expandedProviderId: null);
    }
    if (wasEditing) {
      next = next.copyWith(providerFormOpen: false, editingProviderId: null);
    }
    state = next;
    _bump();
  }

  /// `toggleProvider` (`app.tsx:1203-1205`).
  void toggleProvider(String id) {
    _settings.toggleProvider(id);
    _bump();
  }

  // ── The models CRUD (app.tsx:1208-1246) ─────────────────────────────────

  /// `openAddModelForm` (`app.tsx:1208-1212`).
  void openAddModelForm(String providerId) {
    state = state.copyWith(
      addingModelToId: providerId,
      editingModelId: null,
      modelForm: const (modelId: '', label: ''),
    );
  }

  /// `openEditModelForm` (`app.tsx:1214-1218`).
  void openEditModelForm(String providerId, mockup.ProviderModel m) {
    state = state.copyWith(
      addingModelToId: providerId,
      editingModelId: m.id,
      modelForm: (modelId: m.modelId, label: m.label),
    );
  }

  /// The two model-form cancels (`app.tsx:1655-1656`, `app.tsx:1690`).
  void closeModelForm() {
    state = state.copyWith(addingModelToId: null, editingModelId: null);
  }

  void setModelField(String field, String value) {
    final ModelForm form = state.modelForm;
    state = state.copyWith(
      modelForm: field == 'modelId'
          ? (modelId: value, label: form.label)
          : (modelId: form.modelId, label: value),
    );
  }

  /// `saveModel` (`app.tsx:1219-1233`): the guard + the service write, then
  /// close the form and flash the toast.
  String? submitModel(String providerId) {
    final String? id = _settings.saveModel(
      providerId,
      editingModelId: state.editingModelId,
      modelId: state.modelForm.modelId,
      label: state.modelForm.label,
    );
    if (id == null) {
      return null;
    }
    state = state.copyWith(addingModelToId: null, editingModelId: null);
    _flash();
    _bump();
    return id;
  }

  /// `deleteModel` (`app.tsx:1235-1240`): immediate, no flash.
  void deleteModel(String providerId, String modelId) {
    _settings.deleteModel(providerId, modelId);
    _bump();
  }

  /// `toggleModel` (`app.tsx:1241-1246`).
  void toggleModel(String providerId, String modelId) {
    _settings.toggleModel(providerId, modelId);
    _bump();
  }

  // ── The saved-flash (`app.tsx:1193-1194`) ────────────────────────────────

  /// `setProviderSaved(true)` + the 2200 ms reset.
  void _flash() {
    _flashTimer?.cancel();
    state = state.copyWith(savedFlash: true);
    // The store's disposal cancels the timer — no post-dispose `state`
    // callback (the export's bare setTimeout has no such hazard).
    _flashTimer = Timer(const Duration(milliseconds: 2200), () {
      if (_flashTimer != null) {
        state = state.copyWith(savedFlash: false);
        _flashTimer = null;
      }
    });
  }
}

/// The export's settings view-local state, as one provider. Plain (NOT
/// auto-dispose): like the export's App-level `useState`, the page re-mounts
/// (leaving and re-entering /settings) must NOT lose the draft — e.g. the
/// mock's edit form survives a collapse.
final settingsUiProvider = NotifierProvider<SettingsUiStore, SettingsUi>(
  SettingsUiStore.new,
);
