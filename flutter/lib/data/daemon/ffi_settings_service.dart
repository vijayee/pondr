/// The FFI-backed [SettingsService] (the FFI-binding plan's Task 5): the
/// app-local state rides [AppConfigStore] (the JSON file); the DAEMON's
/// ADOPTION is the picker's selection — the selected provider's
/// `{baseUrl, apiKey, modelId}` goes to the daemon's frame-config template
/// through the CA_CONFIG set ([SaClientNative.configSet]).
///
/// THE ADOPTION RULES:
/// - the SET is a live wire call — the selection's write is sync, the
///   adopt runs after it; [lastAdoption] is the latest adoption's future
///   (its error lands in [lastAdoptionError] — an unawaited fire never
///   leaks an unhandled zone error), never blocking the interface's sync
///   surface.
/// - a SET rides only when the key RESOLVES ([resolveSelectedModel]) — an
///   unknown/stale key is a local record only (the daemon's template keeps
///   its own truth until a real selection lands).
/// - [readDaemonTruth] is the startup GET (the spec's truth display) — the
///   daemon's template as it stands; the app's boot layer calls it after
///   the ensure.
///
/// The key format stays the interface's `<providerId>::<modelId>` (the
/// picker's rows); the model's WIRE name is [ProviderModel.modelId] — the
/// label is display-only.
library;

import 'dart:async';

import 'package:pondr/ffi/sa_client_binding.dart';

import '../../daemon/app_config_store.dart';
import '../models.dart';
import '../services.dart';

/// The resolved picker key: its provider and its model row. Null = the key
/// does not resolve (no selection, a stale model id after a delete, or no
/// provider at all).
({Provider provider, ProviderModel model})? resolveSelectedModel(
    List<Provider> providers, String? key) {
  if (key == null) return null;
  final sep = key.indexOf('::');
  if (sep <= 0) return null;
  final providerId = key.substring(0, sep);
  final modelId = key.substring(sep + 2);
  for (final p in providers) {
    if (p.id != providerId) continue;
    for (final m in p.models) {
      if (m.id == modelId) return (provider: p, model: m);
    }
    return null; // a provider hit with a stale model id: unresolvable
  }
  return null;
}

final class FfiSettingsService implements SettingsService {
  /// [client] is the ensured client's future — LAZY (a function, read at
  /// the adoption's call): the bindings' wiring (the daemon's spawned model
  /// tag rides the SETTINGS, the adoption rides the spawned client) cannot
  /// be a construction cycle.
  FfiSettingsService({
    required this.store,
    Future<SaClientNative> Function()? client,
  }) : _config = store.load() {
    _clientSource = client;
    _providers = List.of(_config.providers);
  }

  final AppConfigStore store;

  /// The lazy client source (the adoption/read calls).
  late final Future<SaClientNative> Function()? _clientSource;

  AppConfig _config;
  List<Provider> _providers = const <Provider>[];

  /// The latest adoption's future (its error is contained in
  /// [lastAdoptionError] — the interface's sync surface never blocks).
  Future<SaConfigResult?> get lastAdoption =>
      _lastAdoption ?? Future<SaConfigResult?>.value();
  Future<SaConfigResult?>? _lastAdoption;

  /// The latest daemon-interaction failure (the config set/get's refusal,
  /// the lost connection, the dead client); null after the last success or
  /// the no-op shapes.
  Object? lastAdoptionError;

  // ── the profile / notifications / appearance (the store's fields) ────────

  @override
  String get displayName => _config.displayName;
  @override
  set displayName(String value) => _mutate(_config.copyWith(displayName: value));
  @override
  String get username => _config.username;
  @override
  set username(String value) => _mutate(_config.copyWith(username: value));
  @override
  String get bio => _config.bio;
  @override
  set bio(String value) => _mutate(_config.copyWith(bio: value));
  @override
  bool get notifMessages => _config.notifMessages;
  @override
  set notifMessages(bool value) =>
      _mutate(_config.copyWith(notifMessages: value));
  @override
  bool get notifSounds => _config.notifSounds;
  @override
  set notifSounds(bool value) => _mutate(_config.copyWith(notifSounds: value));
  @override
  String get accentColor => _config.accentColor;
  @override
  set accentColor(String value) =>
      _mutate(_config.copyWith(accentColor: value));

  @override
  String? get selectedModelKey => _config.selectedModelKey;
  @override
  set selectedModelKey(String? value) {
    // The clear-setter shape (null = clear): copyWith's null-preserving
    // default needs the explicit clear flag.
    _mutate(_config.copyWith(
      selectedModelKey: value,
      clearSelectedModelKey: value == null,
    ));
    _adopt(value);
  }

  /// The write-through: the store's file is the truth's next load.
  void _mutate(AppConfig next) {
    _config = next;
    _providers = List.of(next.providers);
    store.save(_config);
  }

  // ── the daemon's adoption ────────────────────────────────────────────────

  /// The selection's adoption: the resolved provider's truth rides the
  /// CA_CONFIG set. Stale keys and a client-less ride (null) are the local
  /// record's shape; a refusal/dead client records into
  /// [lastAdoptionError] and answers null.
  void _adopt(String? key) {
    final target = resolveSelectedModel(_providers, key);
    if (target == null) {
      _lastAdoption = Future<SaConfigResult?>.value();
      return;
    }
    _lastAdoption = Future<SaConfigResult?>(() async {
      final native = await _clientSource?.call();
      if (native == null) return null;
      final result = await native.configSet(
        baseUrl: target.provider.baseUrl,
        apiKey: target.provider.apiKey,
        model: target.model.modelId,
      );
      if (!result.ok) {
        throw StateError(
            'the daemon refused the config adoption (status '
            '${result.status})');
      }
      return result;
    }).then((result) {
      lastAdoptionError = null;
      return result;
    }, onError: (Object e) {
      lastAdoptionError = e;
      return null;
    });
  }

  /// The startup GET (the spec's truth display): the daemon's template as
  /// it stands; null when no client rides. A failure records into
  /// [lastAdoptionError].
  Future<SaConfigResult?> readDaemonTruth() {
    return Future<SaConfigResult?>(() async {
      final native = await _clientSource?.call();
      if (native == null) return null;
      return native.configGet();
    }).then((result) {
      lastAdoptionError = null;
      return result;
    }, onError: (Object e) {
      lastAdoptionError = e;
      return null;
    });
  }

  // ── the provider CRUD (the store's provider list) ────────────────────────

  @override
  List<Provider> providers() => List.unmodifiable(_providers);

  @override
  String? saveProvider({
    String? editingId,
    required String name,
    required String baseUrl,
    required String apiKey,
  }) {
    // The export's guard (`app.tsx:1183`) — the form's disabled button.
    if (name.trim().isEmpty || baseUrl.trim().isEmpty) return null;
    if (editingId != null) {
      // The export's edit branch spreads the form over the provider.
      _providers = _providers
          .map(
            (p) => p.id == editingId
                ? p.copyWith(name: name, baseUrl: baseUrl, apiKey: apiKey)
                : p,
          )
          .toList();
      _mutate(_config.copyWith(providers: _providers));
      return editingId;
    }
    // The export's create branch: `p${Date.now()}`, enabled, zero models,
    // APPENDED to the end of the list.
    final provider = Provider(
      id: 'p${DateTime.now().millisecondsSinceEpoch}',
      name: name,
      baseUrl: baseUrl,
      apiKey: apiKey,
      enabled: true,
      models: const <ProviderModel>[],
    );
    _providers = [..._providers, provider];
    _mutate(_config.copyWith(providers: _providers));
    return provider.id;
  }

  @override
  void deleteProvider(String id) {
    _providers = _providers.where((p) => p.id != id).toList();
    _mutate(_config.copyWith(providers: _providers));
  }

  @override
  void toggleProvider(String id) {
    _providers = _providers
        .map((p) => p.id == id ? p.copyWith(enabled: !p.enabled) : p)
        .toList();
    _mutate(_config.copyWith(providers: _providers));
  }

  @override
  String? saveModel(
    String providerId, {
    String? editingModelId,
    required String modelId,
    required String label,
  }) {
    // The export's guard (`app.tsx:1220`).
    if (modelId.trim().isEmpty) return null;
    final out = editingModelId ?? 'm${DateTime.now().millisecondsSinceEpoch}';
    _providers = _providers.map((p) {
      if (p.id != providerId) return p;
      if (editingModelId != null) {
        return p.copyWith(
          models: p.models
              .map(
                (m) => m.id == editingModelId
                    ? m.copyWith(modelId: modelId, label: label)
                    : m,
              )
              .toList(),
        );
      }
      // Appended to the END of the provider's model list, enabled.
      return p.copyWith(
        models: <ProviderModel>[
          ...p.models,
          ProviderModel(id: out, modelId: modelId, label: label, enabled: true),
        ],
      );
    }).toList();
    _mutate(_config.copyWith(providers: _providers));
    return out;
  }

  @override
  void deleteModel(String providerId, String modelId) {
    _providers = _providers
        .map(
          (p) => p.id == providerId
              ? p.copyWith(
                  models: p.models.where((m) => m.id != modelId).toList())
              : p,
        )
        .toList();
    _mutate(_config.copyWith(providers: _providers));
  }

  @override
  void toggleModel(String providerId, String modelId) {
    _providers = _providers
        .map(
          (p) => p.id == providerId
              ? p.copyWith(
                  models: p.models
                      .map((m) =>
                          m.id == modelId ? m.copyWith(enabled: !m.enabled) : m)
                      .toList(),
                )
              : p,
        )
        .toList();
    _mutate(_config.copyWith(providers: _providers));
  }
}