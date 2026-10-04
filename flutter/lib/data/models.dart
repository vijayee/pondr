/// The mockup's data types, ported verbatim from the reference
/// (`mockup_reference/app.tsx:44-99` and the subconscious block at
/// `app.tsx:307-320`). The export is truth: every field, default and helper
/// here reads straight from that file.
///
/// Colours stay hex strings (the export stores `#rrggbb` everywhere —
/// `CLUSTER_COLORS`, the accent options, the accent state) so the data layer
/// mirrors the export 1:1; the views convert via `hexToColor`.
library;

/// The export's `Message.role`.
enum MessageRole { user, assistant }

/// The export's `message.files` chip icon mapping (`getFileIcon`,
/// `app.tsx:95-99`): `image/*` → image, anything mentioning pdf/document/text
/// → document, everything else → a generic file.
enum AttachedFileIcon { image, document, other }

class AttachedFile {
  const AttachedFile({
    required this.id,
    required this.name,
    required this.type,
    required this.size,
  });

  final String id;
  final String name;
  final String type;
  final int size;

  AttachedFileIcon get icon => fileIconKind(type);

  /// Equality by id — the export's lookups key the chip list by identity.
  @override
  bool operator ==(Object other) => other is AttachedFile && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'AttachedFile($id, $name)';
}

class Message {
  const Message({
    required this.id,
    required this.role,
    required this.content,
    required this.timestamp,
    this.files,
  });

  final String id;
  final MessageRole role;
  final String content;
  final DateTime timestamp;
  final List<AttachedFile>? files;

  /// Equality by id — views key messages for the list animations by id.
  @override
  bool operator ==(Object other) => other is Message && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'Message($id, ${role.name})';
}

class ChatSession {
  const ChatSession({
    required this.id,
    required this.name,
    required this.updatedAt,
    required this.messages,
  });

  final String id;
  final String name;
  final DateTime updatedAt;
  final List<Message> messages;

  /// The export spreads the session object on every send
  /// (`app.tsx:907-913`); this is that spread.
  ChatSession copyWith({
    String? name,
    DateTime? updatedAt,
    List<Message>? messages,
  }) {
    return ChatSession(
      id: id,
      name: name ?? this.name,
      updatedAt: updatedAt ?? this.updatedAt,
      messages: messages ?? this.messages,
    );
  }

  /// Equality by id — the session sidebar highlights by id.
  @override
  bool operator ==(Object other) => other is ChatSession && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'ChatSession($id, $name)';
}

class ProviderModel {
  const ProviderModel({
    required this.id,
    required this.modelId,
    required this.label,
    required this.enabled,
  });

  final String id;

  /// The wire name, e.g. `deepseek-chat`.
  final String modelId;

  /// The display name; empty means fall back to `modelId`
  /// (the export's picker: `m.label || m.modelId`, `app.tsx:2260`).
  final String label;
  final bool enabled;

  ProviderModel copyWith({bool? enabled, String? modelId, String? label}) {
    return ProviderModel(
      id: id,
      modelId: modelId ?? this.modelId,
      label: label ?? this.label,
      enabled: enabled ?? this.enabled,
    );
  }

  /// Equality by id.
  @override
  bool operator ==(Object other) => other is ProviderModel && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'ProviderModel($id, $modelId)';
}

class Provider {
  const Provider({
    required this.id,
    required this.name,
    required this.baseUrl,
    required this.apiKey,
    required this.enabled,
    required this.models,
  });

  final String id;
  final String name;
  final String baseUrl;
  final String apiKey;
  final bool enabled;
  final List<ProviderModel> models;

  Provider copyWith({
    String? name,
    String? baseUrl,
    String? apiKey,
    bool? enabled,
    List<ProviderModel>? models,
  }) {
    return Provider(
      id: id,
      name: name ?? this.name,
      baseUrl: baseUrl ?? this.baseUrl,
      apiKey: apiKey ?? this.apiKey,
      enabled: enabled ?? this.enabled,
      models: models ?? this.models,
    );
  }

  /// Equality by id.
  @override
  bool operator ==(Object other) => other is Provider && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'Provider($id, $name)';
}

/// A subconscious-graph node without the sim's live position fields —
/// those are added by the force sim (`App.tsx`'s
/// `SubconsciousView` spreads the base nodes and seeds x/y/vx/vy; that is
/// Task 7's `force_sim.dart` territory, not the data layer's).
class SimNode {
  const SimNode({
    required this.id,
    required this.label,
    required this.cluster,
    required this.color,
    required this.r,
    required this.description,
  });

  final String id;
  final String label;
  final String cluster;
  final String color;
  final double r;
  final String description;

  /// Equality by id — the detail card and the dim-others set key on id.
  @override
  bool operator ==(Object other) => other is SimNode && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'SimNode($id)';
}

class SimEdge {
  const SimEdge({required this.source, required this.target});

  final String source;
  final String target;

  /// Equality by endpoints (edges have no id in the export).
  @override
  bool operator ==(Object other) =>
      other is SimEdge && other.source == source && other.target == target;

  @override
  int get hashCode => Object.hash(source, target);

  @override
  String toString() => 'SimEdge($source->$target)';
}

// ─── The subconscious CLUSTER_COLORS (export App.tsx:307-313) ─────────────

const Map<String, String> clusterColors = <String, String>{
  'Physics': '#888ddf',
  'Philosophy': '#c3acda',
  'AI': '#e2e6ff',
  'Astronomy': '#e8d7bd',
  'Logic': '#e0efe4',
};

// ─── Helpers (the export's `app.tsx:85-123`) ──────────────────────────────

/// The export's `formatFileSize` (`app.tsx:89-93`), verbatim thresholds and
/// rounding (`toFixed(1)` ↔ `toStringAsFixed(1)`).
String formatFileSize(num bytes) {
  if (bytes < 1024) return '$bytes B';
  if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
  return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
}

/// The export's `getFileIcon` decision (its `app.tsx:95-99` maps to lucide
/// icons; the views pick the Flutter icon from this kind).
AttachedFileIcon fileIconKind(String type) {
  if (type.startsWith('image/')) return AttachedFileIcon.image;
  if (type.contains('pdf') ||
      type.contains('document') ||
      type.contains('text')) {
    return AttachedFileIcon.document;
  }
  return AttachedFileIcon.other;
}

/// One date bucket of the sidebar — the export's `[string, ChatSession[]]`
/// pair from `groupSessions`.
typedef SessionGroup = ({String label, List<ChatSession> sessions});

/// The export's `groupSessions` (`app.tsx:101-123`), verbatim bucket rules:
/// the session's LOCAL midnight vs today / -1d / -7d midnights; empty
/// buckets are dropped.
List<SessionGroup> groupSessions(List<ChatSession> sessions) {
  final now = DateTime.now();
  final today = DateTime(now.year, now.month, now.day);
  final yesterday = today.subtract(const Duration(days: 1));
  final weekAgo = today.subtract(const Duration(days: 7));

  final groups = <SessionGroup>[
    (label: 'Today', sessions: <ChatSession>[]),
    (label: 'Yesterday', sessions: <ChatSession>[]),
    (label: 'This Week', sessions: <ChatSession>[]),
    (label: 'Earlier', sessions: <ChatSession>[]),
  ];

  for (final s in sessions) {
    final d = DateTime(
      s.updatedAt.year,
      s.updatedAt.month,
      s.updatedAt.day,
    );
    if (!d.isBefore(today)) {
      groups[0].sessions.add(s);
    } else if (!d.isBefore(yesterday)) {
      groups[1].sessions.add(s);
    } else if (!d.isBefore(weekAgo)) {
      groups[2].sessions.add(s);
    } else {
      groups[3].sessions.add(s);
    }
  }

  return groups.where((g) => g.sessions.isNotEmpty).toList();
}

/// Parses the export's `#rrggbb` colour strings (the accent options,
/// `CLUSTER_COLORS`) into Flutter colours for the views.
int hexToInt(String hex) =>
    int.parse(hex.substring(1), radix: 16) | 0xFF000000;