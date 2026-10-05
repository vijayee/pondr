/// The MOCKUP's behaviours, ported function-for-function from
/// `mockup_reference/app.tsx` (`handleNewChat`/`handleDeleteSession`,
/// `handleSend`, `saveProvider`/`saveModel` + toggles, the settings
/// defaults, the BASE pair). Every doc-commented rule traces to the
/// referenced line range in the export.
library;

import 'dart:async';
import 'dart:math';

import '../models.dart';
import '../services.dart';
import 'mock_data.dart';

/// The sessions store in the mockup's shape: a list in INSERTION order
/// (the export never re-sorts; `handleNewChat` front-inserts so
/// newest-first comes from construction), plus the active id.
class MockSessionsService implements SessionsService {
  /// Seeds from the export's `SEED_SESSIONS` by default, active `s1` (the
  /// export's `useState(SEED_SESSIONS)` / `useState("s1")`).
  MockSessionsService({List<ChatSession>? seed, String? activeId})
    : _sessions = List.of(seed ?? seedSessions()),
      _activeId = activeId ?? 's1',
      _controller = StreamController<ChatsChanged>.broadcast();

  final List<ChatSession> _sessions;
  String? _activeId;
  final StreamController<ChatsChanged> _controller;

  @override
  List<ChatSession> list() => List.unmodifiable(_sessions);

  @override
  ChatSession? active() {
    for (final s in _sessions) {
      if (s.id == _activeId) return s;
    }
    return null; // the export's `find` — dangling after the last delete
  }

  @override
  void select(String id) {
    _activeId = id;
    _fire();
  }

  @override
  ChatSession create() {
    // The export's `s${Date.now()}`.
    final session = ChatSession(
      id: 's${DateTime.now().millisecondsSinceEpoch}',
      name: 'New conversation',
      updatedAt: DateTime.now(),
      messages: const <Message>[],
    );
    _sessions.insert(0, session);
    _activeId = session.id;
    _fire();
    return session;
  }

  @override
  void delete(String id) {
    // The export's filter + reselect rule (`app.tsx:878-885`); when nothing
    // remains the id stays dangling and [active] reports null — that IS the
    // mock's "no sessions left" state.
    _sessions.removeWhere((s) => s.id == id);
    if (_activeId == id) {
      _activeId = _sessions.isNotEmpty ? _sessions.first.id : id;
    }
    _fire();
  }

  /// The store's id lookup (the read side of the export's
  /// `prev.map(s => s.id === activeId)` matchers).
  ChatSession? byId(String id) {
    for (final s in _sessions) {
      if (s.id == id) return s;
    }
    return null;
  }

  @override
  Stream<ChatsChanged> changes() => _controller.stream;

  /// Plumbs a send into the store — the export's `setSessions(prev => prev.map(...))`
  /// step of `handleSend`. Mock-internal (used by [MockChatService]); swaps
  /// the stored session with the same id.
  void replaceSession(ChatSession updated) {
    for (var i = 0; i < _sessions.length; i++) {
      if (_sessions[i].id == updated.id) {
        _sessions[i] = updated;
        break;
      }
    }
    _fire();
  }

  void _fire() {
    if (!_controller.isClosed) _controller.add(const ChatsChanged());
  }
}

/// The export's typing + reply pick, behind the [ChatService] stream.
class MockChatService implements ChatService {
  /// [sessions] is the same store the views read, so the send's session
  /// writes are visible to the sidebar/list exactly like the export's
  /// `setSessions`. [random] is injectable for tests; production draws like
  /// `Math.random`.
  MockChatService(this._sessions, {Random? random})
    : _random = random ?? Random();

  final MockSessionsService _sessions;
  final Random _random;

  /// The message-id monotonic guard. The export's `m${Date.now()}` collides
  /// when two messages compose in the same millisecond: a test's fake-async
  /// pump lands the whole 1100-1799 ms typing delay inside one real
  /// millisecond (and a real user's instant double-send the same way), and
  /// the canvas's bubble key then trips the duplicate-key debug check. Each
  /// id keeps the export's `m<ms>` shape; a same-millisecond follower just
  /// ticks the counter forward one.
  int _lastMessageIdMs = 0;

  String _nextMessageId() {
    final int ms = DateTime.now().millisecondsSinceEpoch;
    _lastMessageIdMs = ms > _lastMessageIdMs ? ms : _lastMessageIdMs + 1;
    return 'm$_lastMessageIdMs';
  }

  @override
  Stream<ChatEvent> send(String text, List<AttachedFile> files) async* {
    // The export's guard (`app.tsx:897`): a blank text with nothing
    // attached sends nothing — the composer keeps its draft.
    final trimmed = text.trim();
    if (trimmed.isEmpty && files.isEmpty) return;

    // The send CAPTURES its session at entry (the export's `activeId` read
    // inside `handleSend`) — the reply targets THAT session, even if the
    // user switches away during the typing delay.
    // The send CAPTURES its session at entry (the export's `activeId` read
    // inside `handleSend`) — the reply later targets THAT id, not whatever
    // is active when the typing delay elapses.
    final captured = _sessions.active();
    final capturedId = captured?.id;
    final now = DateTime.now();
    if (captured != null) {
      final userMessage = Message(
        // The export's `m${Date.now()}` (see _nextMessageId's guard: the raw
        // ms collided on a same-millisecond append).
        id: _nextMessageId(),
        role: MessageRole.user,
        content: trimmed,
        timestamp: now,
        files: files.isEmpty ? null : List.of(files),
      );
      // The export's rename rule (`app.tsx:909-911`): only a session still
      // named "New conversation" takes its name from the first send —
      // the trimmed text's first 45 characters, plus '…' when the UNTRIMMED
      // input was longer (the export's length check is on `inputText`).
      var name = captured.name;
      if (name == 'New conversation' && trimmed.isNotEmpty) {
        final cut = trimmed.substring(0, trimmed.length.clamp(0, 45));
        name = text.length > 45 ? '$cut…' : cut;
      }
      _sessions.replaceSession(
        captured.copyWith(
          name: name,
          messages: [...captured.messages, userMessage],
          updatedAt: now,
        ),
      );
    }

    // The typing duration's distribution, verbatim from the export's
    // `setTimeout(..., 1100 + Math.random() * 700)` — 1100-1799 ms, floored.
    final delay = Duration(
      milliseconds: 1100 + (_random.nextDouble() * 700).floor(),
    );
    yield ChatTyping(duration: delay);
    await Future<void>.delayed(delay);

    // The pick rule: uniform random over the whole pool, every send
    // independent (`AI_POOL[Math.floor(Math.random() * AI_POOL.length)]`).
    final reply = aiPool[_random.nextInt(aiPool.length)];

    // The reply rides the CAPTURED id (the export's `setTimeout` closure
    // matches `s.id === activeId` from the send render): the stored session
    // with that id takes the reply; the view decides whether it is still the
    // active one. Absent (deleted mid-typing) → no write, like the export's
    // `map` missing.
    final repliedAt = DateTime.now();
    final target = capturedId == null ? null : _sessions.byId(capturedId);
    if (target != null) {
      _sessions.replaceSession(
        target.copyWith(
          messages: [
            ...target.messages,
            Message(
              id: _nextMessageId(),
              role: MessageRole.assistant,
              content: reply,
              timestamp: repliedAt,
            ),
          ],
          updatedAt: repliedAt,
        ),
      );
    }
    yield ChatDone(reply);
  }
}

/// In-memory settings defaulting to the export's initial state
/// (`app.tsx:822-839`): Ada Lovelace / ada_lovelace / empty bio,
/// notifications on + sounds off, accent `#888ddf`, providers EMPTY.
class MockSettingsService implements SettingsService {
  String _displayName = 'Ada Lovelace';
  String _username = 'ada_lovelace';
  String _bio = '';
  bool _notifMessages = true;
  bool _notifSounds = false;
  String _accentColor = '#888ddf';
  String? _selectedModelKey;
  List<Provider> _providers = const <Provider>[];

  @override
  String get displayName => _displayName;
  @override
  set displayName(String value) => _displayName = value;

  @override
  String get username => _username;
  @override
  set username(String value) => _username = value;

  @override
  String get bio => _bio;
  @override
  set bio(String value) => _bio = value;

  @override
  bool get notifMessages => _notifMessages;
  @override
  set notifMessages(bool value) => _notifMessages = value;

  @override
  bool get notifSounds => _notifSounds;
  @override
  set notifSounds(bool value) => _notifSounds = value;

  @override
  String get accentColor => _accentColor;
  @override
  set accentColor(String value) => _accentColor = value;

  @override
  String? get selectedModelKey => _selectedModelKey;
  @override
  set selectedModelKey(String? value) => _selectedModelKey = value;

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
    return provider.id;
  }

  @override
  void deleteProvider(String id) {
    _providers = _providers.where((p) => p.id != id).toList();
  }

  @override
  void toggleProvider(String id) {
    _providers = _providers
        .map((p) => p.id == id ? p.copyWith(enabled: !p.enabled) : p)
        .toList();
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
    final modelIdOut = editingModelId ?? 'm${DateTime.now().millisecondsSinceEpoch}';
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
        models: [
          ...p.models,
          ProviderModel(id: modelIdOut, modelId: modelId, label: label, enabled: true),
        ],
      );
    }).toList();
    return modelIdOut;
  }

  @override
  void deleteModel(String providerId, String modelId) {
    _providers = _providers
        .map(
          (p) => p.id == providerId
              ? p.copyWith(models: p.models.where((m) => m.id != modelId).toList())
              : p,
        )
        .toList();
  }

  @override
  void toggleModel(String providerId, String modelId) {
    _providers = _providers
        .map(
          (p) => p.id == providerId
              ? p.copyWith(
                  models: p.models
                      .map((m) => m.id == modelId ? m.copyWith(enabled: !m.enabled) : m)
                      .toList(),
                )
              : p,
        )
        .toList();
  }
}

/// The export's BASE pair.
class MockSubconsciousService implements SubconsciousService {
  @override
  (List<SimNode>, List<SimEdge>) graph() =>
      (List.unmodifiable(baseNodes), List.unmodifiable(baseEdges));
}