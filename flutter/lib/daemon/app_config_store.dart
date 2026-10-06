/// The settings' app-local persistence (the FFI-binding plan's Task 5): a
/// JSON file at the platform's config conventions. LINUX v1 (recorded: the
/// other desktop targets' conventions are follow-ons): `XDG_CONFIG_HOME`
/// respected, `~/.config/pondr/settings.json` otherwise.
///
/// The store is DUMB ON PURPOSE: it carries no settings behaviour — a load
/// (missing/corrupt → the export's fresh defaults, a silent repair: a
/// settings file is not the daemon's store; a corrupt file's data is not
/// recoverable and blocks nothing) and a save (THE ATOMIC SHAPE: write the
/// sibling `.tmp`, rename over the target — a kill mid-write can never leave
/// a half-written settings read; the rename is the POSIX-visible commit).
library;

import 'dart:convert';
import 'dart:io';

import '../data/models.dart';

/// The config's shape: the provider store + the picker's selection + the
/// profile/notification/appearance fields. Defaults mirror the export's
/// initial state (`app.tsx:822-839`): Ada Lovelace / ada_lovelace / empty
/// bio, notifications on + sounds off, accent `#888ddf`, providers EMPTY.
final class AppConfig {
  const AppConfig({
    this.providers = const <Provider>[],
    this.selectedModelKey,
    this.displayName = 'Ada Lovelace',
    this.username = 'ada_lovelace',
    this.bio = '',
    this.notifMessages = true,
    this.notifSounds = false,
    this.accentColor = '#888ddf',
  });

  final List<Provider> providers;
  final String? selectedModelKey;
  final String displayName;
  final String username;
  final String bio;
  final bool notifMessages;
  final bool notifSounds;
  final String accentColor;

  /// The fresh defaults (the load's missing-file answer; a named alias to
  /// keep the intent legible at the call sites).
  const AppConfig.fresh() : this();

  /// The one-field-at-a-time shape the settings service's setters ride.
  AppConfig copyWith({
    List<Provider>? providers,
    String? selectedModelKey,
    bool clearSelectedModelKey = false,
    String? displayName,
    String? username,
    String? bio,
    bool? notifMessages,
    bool? notifSounds,
    String? accentColor,
  }) {
    return AppConfig(
      providers: providers ?? this.providers,
      selectedModelKey:
          clearSelectedModelKey ? null : (selectedModelKey ?? this.selectedModelKey),
      displayName: displayName ?? this.displayName,
      username: username ?? this.username,
      bio: bio ?? this.bio,
      notifMessages: notifMessages ?? this.notifMessages,
      notifSounds: notifSounds ?? this.notifSounds,
      accentColor: accentColor ?? this.accentColor,
    );
  }
}

/// The settings file's Linux v1 slot: `XDG_CONFIG_HOME/pondr/`
/// (`~/.config/pondr/` when the variable is unset or empty).
String defaultAppConfigPath() {
  final configHome = Platform.environment['XDG_CONFIG_HOME'];
  final base = configHome == null || configHome.isEmpty
      ? '${Platform.environment['HOME'] ?? '.'}/.config'
      : configHome;
  return '$base/pondr/settings.json';
}

final class AppConfigStore {
  /// [filePath] is the test's seam — production takes the default.
  AppConfigStore({String? filePath}) : filePath = filePath ?? defaultAppConfigPath();

  /// The settings file's resolved path.
  final String filePath;

  bool get exists => File(filePath).existsSync();

  /// The load: the fresh defaults when the file is missing; a corrupt or
  /// schema-lying file repairs to the SAME fresh defaults (a rename-over on
  /// the next save), never a throw into the app's boot.
  AppConfig load() {
    late final String raw;
    try {
      raw = File(filePath).readAsStringSync();
    } on FileSystemException {
      return const AppConfig.fresh();
    }
    try {
      return _fromJson(jsonDecode(raw) as Map<String, dynamic>);
    } on FormatException {
      return const AppConfig.fresh();
    } on TypeError {
      return const AppConfig.fresh();
    }
  }

  /// The save: the parent dir is created (the first run), the body lands in
  /// sibling `settings.json.tmp`, the rename commits (the shape's atomic
  /// write — a crash cannot leave a partial file at the target path).
  void save(AppConfig config) {
    final file = File(filePath);
    final dir = file.parent;
    if (!dir.existsSync()) dir.createSync(recursive: true);
    final tmp = '$filePath.tmp';
    File(tmp).writeAsStringSync(
      const JsonEncoder.withIndent('  ').convert(_toJson(config)),
      flush: true,
    );
    File(tmp).renameSync(filePath);
  }

  AppConfig _fromJson(Map<String, dynamic> json) => AppConfig(
        providers: <Provider>[
          for (final p in (json['providers'] as List<dynamic>? ?? const []))
            _providerFrom(p),
        ],
        selectedModelKey: json['selectedModelKey'] as String?,
        displayName: json['displayName'] as String? ?? 'Ada Lovelace',
        username: json['username'] as String? ?? 'ada_lovelace',
        bio: json['bio'] as String? ?? '',
        notifMessages: json['notifMessages'] as bool? ?? true,
        notifSounds: json['notifSounds'] as bool? ?? false,
        accentColor: json['accentColor'] as String? ?? '#888ddf',
      );

  Provider _providerFrom(dynamic raw) {
    final p = (raw as Map).cast<String, dynamic>();
    return Provider(
      id: p['id'] as String,
      name: p['name'] as String? ?? '',
      baseUrl: p['baseUrl'] as String? ?? '',
      apiKey: p['apiKey'] as String? ?? '',
      enabled: p['enabled'] as bool? ?? true,
      models: <ProviderModel>[
        for (final m in (p['models'] as List<dynamic>? ?? const []))
          _modelFrom(m),
      ],
    );
  }

  ProviderModel _modelFrom(dynamic raw) {
    final m = (raw as Map).cast<String, dynamic>();
    return ProviderModel(
      id: m['id'] as String,
      modelId: m['modelId'] as String? ?? '',
      label: m['label'] as String? ?? '',
      enabled: m['enabled'] as bool? ?? true,
    );
  }

  Map<String, dynamic> _toJson(AppConfig config) => <String, dynamic>{
        'providers': <Map<String, dynamic>>[
          for (final p in config.providers)
            <String, dynamic>{
              'id': p.id,
              'name': p.name,
              'baseUrl': p.baseUrl,
              'apiKey': p.apiKey,
              'enabled': p.enabled,
              'models': <Map<String, dynamic>>[
                for (final m in p.models)
                  <String, dynamic>{
                    'id': m.id,
                    'modelId': m.modelId,
                    'label': m.label,
                    'enabled': m.enabled,
                  },
              ],
            },
        ],
        'selectedModelKey': config.selectedModelKey,
        'displayName': config.displayName,
        'username': config.username,
        'bio': config.bio,
        'notifMessages': config.notifMessages,
        'notifSounds': config.notifSounds,
        'accentColor': config.accentColor,
      };
}