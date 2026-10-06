/// The app's four abstract services — the v1 seams (see
/// `docs/flutter-app-spec.md`'s "v1 data layer" row). The views only ever
/// touch these interfaces + the providers in `bindings.dart`; the real
/// runtime/engine bindings land BEHIND the same shapes later.
///
/// Behaviour contracts are pinned in the interface docs below; the mocks in
/// `mock/mock_services.dart` implement them exactly as the mockup's
/// `App.tsx` functions behave (`mockup_reference/app.tsx` is truth).
library;

import 'models.dart';

/// Emitted on any sessions-store mutation (create / select / delete / a send
/// writing into a session). Payload-free on purpose: listeners re-read
/// `list()` / `active()`.
class ChatsChanged {
  const ChatsChanged();
}

abstract class SessionsService {
  /// The full store in the mockup's order: the export never re-sorts —
  /// `handleNewChat` inserts at the front (`app.tsx:865-876`) and the seed
  /// order rules afterwards. Newest-first by construction, not by sort.
  List<ChatSession> list();

  /// The export's `sessions.find(s => s.id === activeId)` — null when the
  /// store is empty or the active id points at a deleted session
  /// (`app.tsx:849`, the mock's "no sessions left" state).
  ChatSession? active();

  void select(String id);

  /// The export's `handleNewChat`: id `s<now-millis>`, named
  /// "New conversation", zero messages, inserted at the FRONT, and
  /// immediately active. Returns the created session.
  ChatSession create();

  /// The export's `handleDeleteSession` (`app.tsx:878-885`): remove by id;
  /// if the deleted one was active AND sessions remain, the head of the
  /// remaining list becomes active; if none remain, active stays dangling
  /// (null from [active]).
  void delete(String id);

  /// Change events for the views (the export re-renders on the same
  /// mutations, including sends).
  Stream<ChatsChanged> changes();
}

/// One step of `ChatService.send`'s stream.
///
/// - `ChatTyping` — emitted immediately; [ChatTyping.duration] is the delay
///   the mock draws before the reply (the views use it to pace the
///   indicator's lifecycle).
/// - `ChatDelta` — reserved for streaming implementations; the MOCKUP never
///   emits deltas (its reply arrives whole — `app.tsx:922-930`).
/// - `ChatDone` — the full reply text; terminates the stream.
/// - `ChatFailed` — the daemon's turn ended with an error (the `turn.end`
///   error rider's control wording); the whole failed turn's text as an
///   assistant line. The MOCK never emits it (its replies always succeed —
///   the real daemon's failure shape), and the chat view renders it as the
///   dimmed assistant bubble the fold writes for it.
sealed class ChatEvent {
  const ChatEvent();
}

class ChatTyping extends ChatEvent {
  const ChatTyping({required this.duration});

  final Duration duration;
}

class ChatDelta extends ChatEvent {
  const ChatDelta(this.text);

  final String text;
}

class ChatDone extends ChatEvent {
  const ChatDone(this.content);

  final String content;
}

/// A failed turn's terminal (the daemon's `turn.end {error}` rider): the
/// stream's closing event where [ChatDone] rides a successful one. [text]
/// is the failure's wording verbatim (the reason's control text, or the
/// turn's report when the model said nothing) — the view renders it dimmed;
/// an empty [text] keeps the terminal (the typing flag still closes) with
/// nothing to show.
class ChatFailed extends ChatEvent {
  const ChatFailed(this.text);

  final String text;
}

abstract class ChatService {
  /// The export's `handleSend` (`app.tsx:896-931`), as a stream:
  ///
  /// 1. GUARD — blank text AND zero files is a no-op (empty stream, nothing
  ///    else happens); the composer keeps its draft.
  /// 2. The user's `Message` (id `m<now-millis>`, trimmed content) is
  ///    appended to the ACTIVE session before the first event, and the
  ///    session's `updatedAt` is bumped. Rename rule: a session still named
  ///    "New conversation" takes the first 45 characters of the TRIMMED
  ///    text plus '…' when the RAW input is longer than 45 (the export
  ///    checks `inputText.length`, not the trimmed length); textless
  ///    attachment-only sends leave the name alone.
  /// 3. `ChatTyping` fires immediately with the drawn delay — the mockup's
  ///    typing duration is `1100 + rand*700` ms, rounded down.
  /// 4. After the delay the assistant reply — a UNIFORM RANDOM pick from the
  ///    reply pool, every send independent (the mockup's
  ///    `AI_POOL[Math.floor(Math.random()*AI_POOL.length)]`) — is appended
  ///    to the (current) active session, `updatedAt` bumped, and `ChatDone`
  ///    carries the same text to close the stream.
  ///
  /// The session writes go through the session store, so its change stream
  /// fires on both steps.
  Stream<ChatEvent> send(String text, List<AttachedFile> files);
}

abstract class SettingsService {
  // ── Profile; the export's defaults: "Ada Lovelace" / "ada_lovelace" / "".
  String get displayName;
  set displayName(String value);
  String get username;
  set username(String value);
  String get bio;
  set bio(String value);

  // ── Notifications; defaults true / false.
  bool get notifMessages;
  set notifMessages(bool value);
  bool get notifSounds;
  set notifSounds(bool value);

  // ── Appearance: the accent as a hex string from the ACCENT_OPTIONS set
  // (the export's `accentColor` state, default `#888ddf`).
  String get accentColor;
  set accentColor(String value);

  /// The composer's model-picker selection, persisted via the settings
  /// service (the export holds it view-local as `selectedModelKey`,
  /// `app.tsx:839`; this plan persists it — the key format is
  /// `<providerId>::<modelId>`). Null = first available model wins
  /// (the export's picker fallback).
  String? get selectedModelKey;
  set selectedModelKey(String? value);

  /// The provider store; the export's initial state is EMPTY
  /// (`app.tsx:829`) — v1 ships with no providers configured.
  List<Provider> providers();

  /// The export's `saveProvider` (`app.tsx:1182-1195`) — one function for
  /// the add/edit flows: [editingId] null creates (`p<now-millis>`,
  /// enabled, zero models) and returns the new id; non-null edits that
  /// provider's name/baseUrl/apiKey in place and returns its id.
  ///
  /// Returns NULL on the export's blank-field guard (empty trimmed name or
  /// baseUrl — the form's disabled button): nothing is written.
  ///
  /// The SAVED FLASH (the export's `providerSaved` + its 2200 ms reset,
  /// `app.tsx:1193-1194`) is VIEW state — a successful save returns an id
  /// and the view animates the flash; no flash state lives here.
  String? saveProvider({
    String? editingId,
    required String name,
    required String baseUrl,
    required String apiKey,
  });

  void deleteProvider(String id);

  /// The export's `toggleProvider` (`app.tsx:1203-1205`).
  void toggleProvider(String id);

  /// The export's `saveModel` (`app.tsx:1219-1232`): [editingModelId] null
  /// appends (`m<now-millis>`, enabled) and returns the new model id;
  /// non-null rewrites that model's modelId/label. Returns NULL on the
  /// blank-modelId guard. Same saved-flash-split as [saveProvider].
  String? saveModel(
    String providerId, {
    String? editingModelId,
    required String modelId,
    required String label,
  });

  void deleteModel(String providerId, String modelId);

  /// The export's `toggleModel` (`app.tsx:1240-1246`).
  void toggleModel(String providerId, String modelId);
}

abstract class SubconsciousService {
  /// The export's BASE pair (nodes without the sim's live position fields,
  /// plus the edge list). Implementations return fresh unmodifiable lists;
  /// the force sim copies into its own mutable state.
  (List<SimNode>, List<SimEdge>) graph();
}