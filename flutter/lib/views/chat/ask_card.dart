/// The pending ask's card (`docs/ask-ux-spec.md` §3): the escalation slice's
/// owner surface, rendered by [MessageCanvas] as the canvas's last row. The
/// card holds the question, the plan block, the ask's own option buttons,
/// the free-text escape, and — only when the ask's options LACK a
/// reject-shaped entry — the explicit reject. The store
/// (`lib/data/pending_ask.dart`) is the card's single source of truth: the
/// card never clears itself, whatever the reply op's future answers.
library;

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/bindings.dart' show chatServiceProvider;
import '../../data/pending_ask.dart';
import '../../theme/tokens.dart' show PondrTokens;
import 'common.dart' show PressScale;

/// The ask card (`docs/ask-ux-spec.md` §3), on the message bubble's
/// assistant container recipe (the tinted `rgba(136,141,223,0.12)` card,
/// the 0.3 border, radius 8).
///
/// The reply lifecycle: every control posts through [_post] — busy set, the
/// `answerAsk` op awaited (a thrown StateError counts as not-delivered,
/// `services.dart`'s contract), busy released with the not-delivered line
/// only on a false/failed op. THE SILENT-DROP CORNER: a DELIVERED ack
/// re-enables the controls immediately, but the card clears only when the
/// `ask.reply` RECORD drops the sid from the store — the unmount rides the
/// canvas's store watch, never this widget's future. No retry machinery.
class AskCard extends ConsumerStatefulWidget {
  const AskCard({required this.sid, required this.ask, super.key});

  /// The parked frame's session id — the reply op's target and the store's
  /// key.
  final String sid;

  /// The in-flight ask (the store row for [sid]).
  final PendingAsk ask;

  @override
  ConsumerState<AskCard> createState() => _AskCardState();
}

class _AskCardState extends ConsumerState<AskCard> {
  final TextEditingController _free = TextEditingController();
  bool _busy = false;
  bool _notDelivered = false;

  @override
  void dispose() {
    _free.dispose();
    super.dispose();
  }

  /// The typed-value post ([_post] with the free text's trimmed value);
  /// the field's onChanged rebuilds so the send affordance's gate
  /// (`_busy || trimmed empty`) reads fresh.
  void _onFreeChanged(String _) {
    if (!mounted) return;
    setState(() {});
  }

  /// The single reply path (the spec's `askReply(..., decision, value)`):
  /// [value] is the option LABEL for an option tap, the free text's
  /// trimmed content for the escape, empty for the reject (the engine's
  /// default refusal wording rides the empty value).
  Future<void> _post(int decision, String value) async {
    if (_busy) return;
    setState(() => _busy = true);
    bool delivered;
    try {
      delivered = await ref.read(chatServiceProvider).answerAsk(
            sid: widget.sid,
            askId: widget.ask.askId,
            reject: decision == 1,
            value: value,
          );
    } on StateError {
      // services.dart's contract: a not-ready or disposed client REJECTS
      // the future instead of answering false — the same not-delivered.
      delivered = false;
    }
    if (!mounted) return;
    // Delivered → no local clear (the silent-drop corner: the RECORD in
    // the store is the card's truth). Not delivered → the error line and
    // the controls re-enabled (retryable).
    setState(() {
      _busy = false;
      _notDelivered = !delivered;
    });
  }

  @override
  Widget build(BuildContext context) {
    final PendingAsk ask = widget.ask;
    final String trimmed = _free.text.trim();
    return Material(
      type: MaterialType.transparency,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0x1F888DDF), // rgba(136,141,223,0.12)
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: const Color(0x4D888DDF)), // rgba(..., 0.3)
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            // The header (the mock's timestamp-row ink: 11 px white/50).
            const Text(
              'the agent asks',
              style: TextStyle(fontSize: 11, color: Color(0x80FFFFFF)),
            ),
            const SizedBox(height: 6),
            Text(
              ask.question,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
            if (ask.plan != null) ...<Widget>[
              const SizedBox(height: 8),
              MarkdownBody(
                data: ask.plan!,
                softLineBreak: false,
                styleSheet: _planStyle,
              ),
            ],
            // The ask's own options — the LABEL rides verbatim as the
            // value (`askReply(..., 0, label)`).
            for (int i = 0; i < ask.options.length; i++)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: _AskButton(
                  inkKey: Key('chat.ask-option-$i'),
                  enabled: !_busy,
                  onTap: _busy ? null : () => _post(0, ask.options[i]),
                  child: Text(
                    ask.options[i],
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF0C0B1A),
                    ),
                  ),
                ),
              ),
            const SizedBox(height: 8),
            // The free-text escape: the one-line input + the send
            // affordance (the composer's send idiom).
            Row(
              children: <Widget>[
                Expanded(
                  child: TextField(
                    key: const Key('chat.ask-free'),
                    controller: _free,
                    onChanged: _onFreeChanged,
                    enabled: !_busy,
                    style: const TextStyle(fontSize: 13, color: Colors.white),
                    decoration: InputDecoration(
                      isDense: true,
                      hintText: 'type a different answer',
                      hintStyle: const TextStyle(
                        fontSize: 13,
                        color: Color(0x66FFFFFF),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 10,
                      ),
                      filled: true,
                      fillColor: const Color(0x14FFFFFF),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: const BorderSide(color: Color(0x26FFFFFF)),
                      ),
                      disabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: const BorderSide(color: Color(0x26FFFFFF)),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: const BorderSide(color: Color(0x4D888DDF)),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                _AskSendButton(
                  inkKey: const Key('chat.ask-send-free'),
                  enabled: !_busy && trimmed.isNotEmpty,
                  onTap:
                      (!_busy && trimmed.isNotEmpty)
                          ? () => _post(0, trimmed)
                          : null,
                ),
              ],
            ),
            // The explicit reject ONLY when the ask's own options lack a
            // reject-shaped entry (`askReply(..., 1, '')`).
            if (ask.options.isEmpty)
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton(
                  key: const Key('chat.ask-reject'),
                  onPressed: _busy ? null : () => _post(1, ''),
                  child: const Text(
                    'reject',
                    style: TextStyle(fontSize: 13, color: Color(0x8CFFFFFF)),
                  ),
                ),
              ),
            // The in-flight spinner line (the buttons disable above).
            if (_busy)
              const Padding(
                padding: EdgeInsets.only(top: 10),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    SizedBox(width: 8),
                    Text(
                      'sending…',
                      style: TextStyle(fontSize: 12, color: Color(0x8CFFFFFF)),
                    ),
                  ],
                ),
              ),
            if (_notDelivered)
              const Padding(
                padding: EdgeInsets.only(top: 10),
                child: Text(
                  'not delivered — the session may be gone',
                  style: TextStyle(fontSize: 12, color: Color(0x8CFFFFFF)),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // The bubble body's markdown ink, mirrored (`message_bubble.dart`'s
  // `_BubbleBody` graph is private there): the plan block reads as the
  // message it rides beneath.
  static final MarkdownStyleSheet _planStyle = MarkdownStyleSheet(
    p: const TextStyle(fontSize: 14, height: 1.625, color: Colors.white),
    pPadding: EdgeInsets.zero,
    strong: const TextStyle(fontWeight: FontWeight.w600, color: Colors.white),
    em: const TextStyle(fontStyle: FontStyle.italic, color: Colors.white),
    listBullet: const TextStyle(fontSize: 14, height: 1.625, color: Colors.white),
    a: TextStyle(color: PondrTokens.primary, decoration: TextDecoration.underline),
  );
}

/// The option/reject button on the SEND idiom (`composer.dart`'s
/// `_SendButton`): press-scaled, the 150 ms 0.4 dim while disabled, the
/// mint `#E0EFE4` material at radius 8 over the ink well.
class _AskButton extends StatelessWidget {
  const _AskButton({
    required this.inkKey,
    required this.enabled,
    required this.onTap,
    required this.child,
  });

  final Key inkKey;
  final bool enabled;
  final VoidCallback? onTap;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return PressScale(
      pressedScale: 0.9,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 150),
        opacity: enabled ? 1.0 : 0.4,
        child: Material(
          type: MaterialType.button,
          color: const Color(0xFFE0EFE4),
          borderRadius: BorderRadius.circular(8),
          shadowColor: const Color(0x80E0EFE4),
          elevation: 0,
          child: InkWell(
            key: inkKey,
            onTap: onTap,
            borderRadius: BorderRadius.circular(8),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              child: child,
            ),
          ),
        ),
      ),
    );
  }
}

/// The free-text send affordance — the composer's 32×32 send button
/// verbatim (`composer.dart`'s `_SendButton`: the send icon on `#E0EFE4`,
/// the glow only while sendable, disabled at 0.4 with onTap null).
class _AskSendButton extends StatelessWidget {
  const _AskSendButton({
    required this.inkKey,
    required this.enabled,
    required this.onTap,
  });

  final Key inkKey;
  final bool enabled;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return PressScale(
      pressedScale: 0.9,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 150),
        opacity: enabled ? 1.0 : 0.4,
        child: Material(
          type: MaterialType.button,
          color: const Color(0xFFE0EFE4),
          borderRadius: BorderRadius.circular(8),
          shadowColor: const Color(0x80E0EFE4),
          elevation: 0,
          child: InkWell(
            key: inkKey,
            onTap: onTap,
            borderRadius: BorderRadius.circular(8),
            child: Container(
              width: 32,
              height: 32,
              decoration: enabled
                  ? const BoxDecoration(
                      boxShadow: <BoxShadow>[
                        BoxShadow(
                          color: Color(0x80E0EFE4),
                          blurRadius: 14,
                          offset: Offset(0, 2),
                        ),
                      ],
                    )
                  : null,
              alignment: Alignment.center,
              child: const Icon(
                Icons.send,
                size: 15,
                color: Color(0xFF0C0B1A),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
