/// THE ONE SWAP (per the plan + spec): every interface is served through a
/// provider here; the later real bindings (the runtime FFI client, the
/// engine's subconscious surface) replace these lines and NOTHING else in
/// the app changes. Views import this file — never `mock/*` directly.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'mock/mock_services.dart';
import 'services.dart';

export 'mock/mock_data.dart';

/// The concrete mock, so [chatServiceProvider] can wire the SAME store the
/// views read (the export's handleSend writes through `setSessions`).
final _mockSessionsProvider = Provider<MockSessionsService>(
  (ref) => MockSessionsService(),
);

/// The export's `sessions` state.
final sessionsProvider = Provider<SessionsService>(
  (ref) => ref.watch(_mockSessionsProvider),
);

/// The export's typing + reply flow.
final chatServiceProvider = Provider<ChatService>(
  (ref) => MockChatService(ref.watch(_mockSessionsProvider)),
);

/// The export's settings state (profile / notifications / accent /
/// providers CRUD).
final settingsProvider = Provider<SettingsService>(
  (ref) => MockSettingsService(),
);

/// The export's BASE pair.
final subconsciousProvider = Provider<SubconsciousService>(
  (ref) => MockSubconsciousService(),
);