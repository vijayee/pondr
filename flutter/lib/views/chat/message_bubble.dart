import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../../data/models.dart';
import '../../theme/tokens.dart';
import 'common.dart' show kThoughtSparkAsset;

/// The export's MessageBubble (`app.tsx:732-788`) + its entrance motion.
///
/// - user: the primary-filled bubble `rgb(136,141,223)`, radius
///   `1rem 0.25rem 1rem 1rem`, glow shadow, laid out right (the row
///   reversed, `ml-auto`).
/// - assistant: the tinted `rgba(136,141,223,0.12)` card with the
///   `rgba(136,141,223,0.3)` border and the mirrored radius, laid out left.
/// - The entrance is the export's `initial { opacity: 0, y: 10 } →
///   animate { opacity: 1, y: 0 }` over 0.22 s easeOut (`app.tsx:736-739`),
///   played once per mount — a freshly appended bubble rises in.
class MessageBubble extends StatefulWidget {
  const MessageBubble({required this.message, super.key});

  final Message message;

  /// The export's bubble entrance (`app.tsx:739`: duration 0.22, easeOut).
  static const Duration entranceDuration = Duration(milliseconds: 220);

  /// The export's `initial: { y: 10 }`.
  static const double entranceDy = 10;

  @override
  State<MessageBubble> createState() => _MessageBubbleState();
}

class _MessageBubbleState extends State<MessageBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _entranceController = AnimationController(
    vsync: this,
    duration: MessageBubble.entranceDuration,
  )..forward();

  @override
  void dispose() {
    _entranceController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bool isUser = widget.message.role == MessageRole.user;
    return AnimatedBuilder(
      animation: _entranceController,
      builder: (BuildContext context, Widget? child) {
        final double t = Curves.easeOut.transform(_entranceController.value);
        return Opacity(
          opacity: t,
          child: Transform.translate(
            offset: Offset(0, (1 - t) * MessageBubble.entranceDy),
            child: child,
          ),
        );
      },
      child: _BubbleRow(message: widget.message, isUser: isUser),
    );
  }
}

/// The bubble group: avatar + content column, right-aligned when user
/// (`flex-row-reverse ml-auto`, `app.tsx:740`); capped at the mock's
/// `max-w-3xl` (768 px at 16 px root).
class _BubbleRow extends StatelessWidget {
  const _BubbleRow({required this.message, required this.isUser});

  final Message message;
  final bool isUser;

  @override
  Widget build(BuildContext context) {
    final List<Widget> row = <Widget>[
      _BubbleAvatar(isUser: isUser),
      Expanded(child: _BubbleColumn(message: message, isUser: isUser)),
    ];
    final Widget body = Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      textDirection: isUser ? TextDirection.rtl : TextDirection.ltr,
      children: row,
    );
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 768),
        child: body,
      ),
    );
  }
}

/// The 28 dp avatar (the export's `w-7 h-7`): user on `bg-primary/20` with
/// the 13 px user glyph (app.tsx:717); assistant on `rgba(195,172,218,0.18)`
/// with the 20 px (`w-5 h-5`) ThoughtSpark mark (app.tsx:713) — the
/// iridescent asset itself.
class _BubbleAvatar extends StatelessWidget {
  const _BubbleAvatar({required this.isUser});

  final bool isUser;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 28,
      height: 28,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: isUser ? const Color(0x33888DDF) : const Color(0x2EC3ACDA),
        shape: BoxShape.circle,
      ),
      child: isUser
          ? const Icon(Icons.person, size: 16, color: PondrTokens.primary)
          : SizedBox(
              width: 20, // w-5 h-5
              height: 20,
              child: SvgPicture.asset(kThoughtSparkAsset, fit: BoxFit.contain),
            ),
    );
  }
}

/// The content column: the bubble + its timestamp, end-aligned for user
/// (`items-end`, `app.tsx:754,784`).
class _BubbleColumn extends StatelessWidget {
  const _BubbleColumn({required this.message, required this.isUser});

  final Message message;
  final bool isUser;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 2),
      child: Column(
        crossAxisAlignment:
            isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: <Widget>[
          _BubbleBody(message: message, isUser: isUser),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Text(
              formatTime(message.timestamp),
              style: const TextStyle(
                fontSize: 11,
                color: Color(0x66FFFFFF), // text-white/40
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// The bubble itself: file chips above the markdown body (`app.tsx:755-783`).
class _BubbleBody extends StatelessWidget {
  const _BubbleBody({required this.message, required this.isUser});

  final Message message;
  final bool isUser;

  @override
  Widget build(BuildContext context) {
    final List<AttachedFile>? files = message.files;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: isUser ? PondrTokens.primary : const Color(0x1F888DDF),
        border: isUser
            ? null
            : Border.all(color: const Color(0x4D888DDF)),
        boxShadow: isUser
            ? const <BoxShadow>[
                BoxShadow(
                  color: Color(0x59888DDF), // rgba(136,141,223,0.35)
                  blurRadius: 16,
                  offset: Offset(0, 4),
                ),
              ]
            : null,
        borderRadius: isUser
            ? const BorderRadius.only(
                topLeft: Radius.circular(16),
                topRight: Radius.circular(4),
                bottomRight: Radius.circular(16),
                bottomLeft: Radius.circular(16),
              )
            : const BorderRadius.only(
                topLeft: Radius.circular(4),
                topRight: Radius.circular(16),
                bottomRight: Radius.circular(16),
                bottomLeft: Radius.circular(16),
              ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          if (files != null && files.isNotEmpty) ...<Widget>[
            for (int i = 0; i < files.length; i++) ...<Widget>[
              if (i != 0) const SizedBox(height: 6),
              _FileChip(file: files[i], isUser: isUser),
            ],
            const SizedBox(height: 10),
          ],
          if (message.content.isNotEmpty)
            MarkdownBody(
              data: message.content,
              // The export's `whitespace-pre-wrap` + its lite renderer
              // (`app.tsx:670-699`): single newlines are REAL line breaks,
              // full-line `**bold**` heads each render block.
              softLineBreak: false,
              styleSheet: _markdownStyle,
            ),
        ],
      ),
    );
  }

  // Not const-constructible in flutter_markdown 0.7.x — a static final holds
  // the same immutable style graph.
  static final MarkdownStyleSheet _markdownStyle = MarkdownStyleSheet(
    p: TextStyle(fontSize: 14, height: 1.625, color: Colors.white),
    pPadding: EdgeInsets.zero,
    strong: TextStyle(fontWeight: FontWeight.w600, color: Colors.white),
    em: TextStyle(fontStyle: FontStyle.italic, color: Colors.white),
    listBullet: TextStyle(fontSize: 14, height: 1.625, color: Colors.white),
    a: TextStyle(color: PondrTokens.primary, decoration: TextDecoration.underline),
  );
}

/// One attachment chip inside the bubble — the icon mapping
/// (`getFileIcon` → `AttachedFileIcon`), the truncated name, the formatted
/// size (`app.tsx:762-778`; the helpers live in `lib/data/models.dart`).
class _FileChip extends StatelessWidget {
  const _FileChip({required this.file, required this.isUser});

  final AttachedFile file;
  final bool isUser;

  @override
  Widget build(BuildContext context) {
    final IconData icon = switch (file.icon) {
      AttachedFileIcon.image => Icons.image,
      AttachedFileIcon.document => Icons.description,
      AttachedFileIcon.other => Icons.insert_drive_file,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color:
            isUser ? const Color(0x26FFFFFF) : const Color(0x1AFFFFFF), // 15/10
        border: isUser
            ? null
            : Border.all(color: const Color(0x26FFFFFF)), // white/15
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(icon, size: 12, color: const Color(0xB3FFFFFF)),
          const SizedBox(width: 8),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 180),
            child: Text(
              file.name,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 11, color: Colors.white),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            formatFileSize(file.size),
            style: const TextStyle(fontSize: 11, color: Color(0x80FFFFFF)),
          ),
        ],
      ),
    );
  }
}

/// The export's `formatTime` (`app.tsx:85-87`): a two-digit hour:minute
/// clock-time with the AM/PM suffix (the export's default `en-US` shape).
String formatTime(DateTime date) {
  final int hour = date.hour % 12 == 0 ? 12 : date.hour % 12;
  final String suffix = date.hour < 12 ? 'AM' : 'PM';
  return '$hour:${date.minute.toString().padLeft(2, '0')} $suffix';
}