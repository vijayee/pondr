/// The chat view's state — the export's `App()` local state block
/// (`app.tsx:805-847`) expressed as riverpod stores fed by the services
/// (the plan's "no inline state" rule; the shell's own
/// `sidebarOpen`/`sidebarCollapsed` are the same idiom in `lib/app/shell.dart`).
///
/// Reactivity: the services are streams of CHANGE EVENTS, so the stores
/// re-read the service on every [ChatsChanged]. The ChatSession model keys
/// equality on id (`lib/data/models.dart`) — list stores therefore expose
/// fresh List identities (riverpod's == change test) instead of sessions.
library;

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/bindings.dart';
// The MODEL's `Provider` class and riverpod's `Provider` share a name —
// the data-side one comes in aliased.
import '../../data/models.dart' hide Provider;
import '../../data/models.dart' as mockup show Provider;
import '../../data/services.dart';

// ─── The sessions store ────────────────────────────────────────────────────

/// The full store, refreshed on every change event (the export's
/// `sessions` state + its `setSessions` re-renders).
class SessionsListStore extends Notifier<List<ChatSession>> {
  @override
  List<ChatSession> build() {
    final SessionsService service = ref.watch(sessionsProvider);
    final StreamSubscription<ChatsChanged> sub = service.changes().listen((_) {
      state = service.list();
    });
    ref.onDispose(sub.cancel);
    return service.list();
  }
}

final sessionsStoreProvider =
    NotifierProvider<SessionsListStore, List<ChatSession>>(
      SessionsListStore.new,
    );

/// The active session's id (the export's `activeId`), refreshed on every
/// change event — select/create/delete all flow through the store's events.
class ActiveIdStore extends Notifier<String?> {
  @override
  String? build() {
    final SessionsService service = ref.watch(sessionsProvider);
    final StreamSubscription<ChatsChanged> sub = service.changes()
        .listen((_) {
          state = service.active()?.id;
        });
    ref.onDispose(sub.cancel);
    return service.active()?.id;
  }
}

final activeIdProvider =
    NotifierProvider<ActiveIdStore, String?>(ActiveIdStore.new);

// Derived views of the ACTIVE session. The id-keyed == on ChatSession means
// a send REPLACING the session must not be seen as "unchanged" — the derived
// values below are List/int/record comparisons, which the mutations always
// change.
final activeMessagesProvider = Provider<List<Message>>((ref) {
  final String? id = ref.watch(activeIdProvider);
  final List<ChatSession> sessions = ref.watch(sessionsStoreProvider);
  for (final ChatSession s in sessions) {
    if (s.id == id) return List.unmodifiable(s.messages);
  }
  return const <Message>[];
});

/// (name, message count) — the header's two readouts (`app.tsx:2131-2137`).
final activeSessionHeaderProvider = Provider<(String, int)?>((ref) {
  final String? id = ref.watch(activeIdProvider);
  final List<ChatSession> sessions = ref.watch(sessionsStoreProvider);
  for (final ChatSession s in sessions) {
    if (s.id == id) return (s.name, s.messages.length);
  }
  return null;
});

// ─── The composer (the export's inputText / attached / isTyping) ──────────

class ComposerState {
  const ComposerState({
    this.text = '',
    this.attached = const <AttachedFile>[],
    this.isTyping = false,
  });

  final String text;
  final List<AttachedFile> attached;
  final bool isTyping;
}

/// The composer's draft + the send flow (the export's `handleSend`,
/// `app.tsx:896-931`). [send] subscribes the service's stream; the session
/// writes (the user message, then the reply) ride the store's change events
/// into the sidebar/canvas above.
class ComposerStore extends Notifier<ComposerState> {
  @override
  ComposerState build() {
    ref.onDispose(() => _disposed = true);
    return const ComposerState();
  }

  bool _disposed = false;

  void setText(String value) {
    state = ComposerState(
      text: value,
      attached: state.attached,
      isTyping: state.isTyping,
    );
  }

  void attach(AttachedFile file) {
    state = ComposerState(
      text: state.text,
      attached: <AttachedFile>[...state.attached, file],
      isTyping: state.isTyping,
    );
  }

  void removeFile(String id) {
    state = ComposerState(
      text: state.text,
      attached: state.attached.where((AttachedFile f) => f.id != id).toList(),
      isTyping: state.isTyping,
    );
  }

  /// The send: the mock's guard (a blank draft with nothing attached is a
  /// no-op — the composer keeps its draft), clear the inputs, the typing
  /// flag goes up immediately, and any terminal event of the reply's stream
  /// brings it down (the export's first `setTimeout` firing closes typing
  /// unconditionally — parallel sends behave the same).
  void send() {
    final String text = state.text;
    final List<AttachedFile> files = state.attached;
    if (text.trim().isEmpty && files.isEmpty) return;

    state = const ComposerState(isTyping: true);
    // Declared-first so the terminal-event closures can cancel their OWN
    // subscription (the parallel-send shape).
    StreamSubscription<ChatEvent>? sub;
    sub = ref.read(chatServiceProvider).send(text, files).listen(
      (ChatEvent event) {
        if (event is ChatDone) {
          _closeTyping(sub!);
        }
      },
      onDone: () => _closeTyping(sub!),
      onError: (Object _) => _closeTyping(sub!),
    );
    ref.onDispose(() => sub?.cancel());
  }

  /// Any terminal event of ANY send lands the typing flag — the export's
  /// first setTimeout closing it unconditionally (`app.tsx:929`).
  void _closeTyping(StreamSubscription<ChatEvent> sub) {
    sub.cancel();
    if (_disposed) return;
    state = ComposerState(
      text: state.text,
      attached: state.attached,
      isTyping: false,
    );
  }
}

final composerStoreProvider =
    NotifierProvider<ComposerStore, ComposerState>(ComposerStore.new);

/// A bump to focus the composer's field — the plan's "NEW chat focuses the
/// composer" (the export's handleNewChat clears the draft; the focus is the
/// plan's addition).
class ComposerFocusTickStore extends Notifier<int> {
  @override
  int build() => 0;

  void bump() => state++;
}

final composerFocusTickProvider =
    NotifierProvider<ComposerFocusTickStore, int>(ComposerFocusTickStore.new);

// ─── The sidebar's search (the export's showSearch / searchQuery) ─────────

class ShowSearchStore extends Notifier<bool> {
  @override
  bool build() => false;

  void show() => state = true;
  void hide() => state = false;
}

final showSearchProvider =
    NotifierProvider<ShowSearchStore, bool>(ShowSearchStore.new);

class SearchQueryStore extends Notifier<String> {
  @override
  String build() => '';

  void set(String value) => state = value;
}

final searchQueryProvider =
    NotifierProvider<SearchQueryStore, String>(SearchQueryStore.new);

// ─── The model picker (the export's selectedModelKey) ─────────────────────

/// One row of the export's `availableModels` flatMap (`app.tsx:2259-2261`).
class AvailableModel {
  const AvailableModel({
    required this.key,
    required this.providerName,
    required this.modelId,
    required this.label,
  });

  /// `<providerId>::<modelId>` — the export's key format.
  final String key;
  final String providerName;
  final String modelId;

  /// `m.label || m.modelId` (the export's fallback).
  final String label;
}

/// The export's picker list: the ENABLED providers' ENABLED models
/// (`app.tsx:2260`).
List<AvailableModel> availableModelsOf(List<mockup.Provider> providers) {
  return <AvailableModel>[
    for (final mockup.Provider p in providers)
      if (p.enabled)
        for (final ProviderModel m in p.models)
          if (m.enabled)
            AvailableModel(
              key: '${p.id}::${m.id}',
              providerName: p.name,
              modelId: m.modelId,
              label: m.label.isEmpty ? m.modelId : m.label,
            ),
  ];
}

/// Bumped after every settings mutation (Task 6's wiring point) so
/// settings-backed stores re-read the service — the settings interface has
/// no change stream in v1.
class SettingsRevisionStore extends Notifier<int> {
  @override
  int build() => 0;

  void bump() => state++;
}

final settingsRevisionProvider =
    NotifierProvider<SettingsRevisionStore, int>(SettingsRevisionStore.new);

/// The picker's pool, re-read when the revision (or the service identity)
/// changes.
final availableModelsProvider = Provider<List<AvailableModel>>((ref) {
  ref.watch(settingsRevisionProvider);
  final SettingsService settings = ref.watch(settingsProvider);
  return availableModelsOf(settings.providers());
});

/// The persisted picker selection (the export's `selectedModelKey`
/// view-local state — F2 pinned it onto the settings service; that
/// interface is landed, so the store reads/writes through it).
class ModelSelectionStore extends Notifier<String?> {
  @override
  String? build() {
    ref.watch(settingsRevisionProvider);
    return ref.read(settingsProvider).selectedModelKey;
  }

  /// The export's pick (`app.tsx:2311`): persist the key + close.
  void select(String key) {
    ref.read(settingsProvider).selectedModelKey = key;
    ref.read(settingsRevisionProvider.notifier).bump();
    state = key;
  }
}

final modelSelectionProvider =
    NotifierProvider<ModelSelectionStore, String?>(ModelSelectionStore.new);

/// The resolved selected row (the export's `?? availableModels[0] ?? null`
/// fallback, `app.tsx:2262`).
final selectedModelProvider = Provider<AvailableModel?>((ref) {
  final List<AvailableModel> models = ref.watch(availableModelsProvider);
  final String? key = ref.watch(modelSelectionProvider);
  for (final AvailableModel m in models) {
    if (m.key == key) return m;
  }
  return models.isEmpty ? null : models.first;
});

// ─── The popovers' open flags (the export's modelPickerOpen etc.) ─────────

class FlagStore extends Notifier<bool> {
  @override
  bool build() => false;

  void set(bool value) => state = value;
  void toggle() => state = !state;
}

/// The attachment picker sheet (the composer's `+`/paperclip) and the model
/// dropdown's open flags.
final attachmentsMenuOpenProvider =
    NotifierProvider<FlagStore, bool>(FlagStore.new);

final modelPickerOpenProvider =
    NotifierProvider<FlagStore, bool>(FlagStore.new);