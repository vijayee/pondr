import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../app/shell.dart' show ShellMode, kPaneBreakpoint;
import '../../data/bindings.dart' show suggestions;
import '../../data/models.dart';
import '../../theme/tokens.dart';
import 'chat_state.dart';
import 'common.dart';
import 'composer.dart';
import 'message_bubble.dart';
import 'typing_indicator.dart';

/// The chat view's body (the export's chat-layout main area,
/// `app.tsx:2110-2340`): the header, the message canvas, the composer.
///
/// The header is the mock's PER-VIEW composition: wide screens render it
/// inside this page; narrow screens the SHELL hosts it through the route's
/// `topBarBuilder` (lib/app/router.dart hands `ChatHeader(menuButton:)`) —
/// driven by MediaQuery's width, so the rail's width animation can't
/// double-render it.
class ChatPage extends ConsumerWidget {
  const ChatPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool wide =
        ShellMode.wideOf(context) ??
        (MediaQuery.sizeOf(context).width >= kPaneBreakpoint);

    final Widget body = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Expanded(child: MessageCanvas()),
        const ChatComposer(),
      ],
    );
    if (!wide) {
      // The shell carries the narrow top bar (the mock's lg:hidden menu
      // button opens the drawer).
      return body;
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[const ChatHeader(), Expanded(child: body)],
    );
  }
}

/// The chat header (`app.tsx:2118-2144`): the active session's name + count
/// (“N message`s`” / “Start a new conversation”) and the “Pondr AI” status.
/// `menuButton` is the shell's drawer affordance on narrow screens.
class ChatHeader extends ConsumerWidget {
  const ChatHeader({this.menuButton, super.key});

  final Widget? menuButton;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final (String name, int messageCount)? meta =
        ref.watch(activeSessionHeaderProvider);

    return Container(
      key: const Key('chat.header'),
      decoration: const BoxDecoration(
        color: Color(0xFF1A1929),
        border: Border(bottom: BorderSide(color: PondrTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: <Widget>[
          if (menuButton != null) ...<Widget>[menuButton!, const SizedBox(width: 12)],
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  meta?.$1 ?? 'New Conversation',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700, // font-nunito bold
                    color: Colors.white,
                  ),
                ),
                Text(
                  meta == null
                      ? 'Start a new conversation'
                      : '${meta.$2} message${meta.$2 != 1 ? 's' : ''}',
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0x80FFFFFF), // text-white/50
                  ),
                ),
              ],
            ),
          ),
          // The mock's status badge (`app.tsx:2140-2143`).
          Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: Color(0xFF10B981), // emerald-500
                  shape: BoxShape.circle,
                  boxShadow: <BoxShadow>[
                    BoxShadow(
                      color: Color(0x8010B981),
                      blurRadius: 4,
                      spreadRadius: 1,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 6),
              const Text(
                'Pondr AI',
                style: TextStyle(fontSize: 12, color: Color(0x8CFFFFFF)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// The message canvas (`app.tsx:2146-2184`): the empty session's prompt
/// state (the spark + logo + copy + the suggestion chips) or the bubble
/// column with the typing indicator and the bottom-anchored scroll.
///
/// The auto-scroll is the export's `bottomRef.scrollIntoView({smooth})`
/// riding `messages.length` and `isTyping` (`app.tsx:851-853`): a
/// [ScrollController] animated to the bottom extent on both triggers.
class MessageCanvas extends ConsumerStatefulWidget {
  const MessageCanvas({super.key});

  @override
  ConsumerState<MessageCanvas> createState() => _MessageCanvasState();
}

class _MessageCanvasState extends ConsumerState<MessageCanvas> {
  final ScrollController _scroll = ScrollController();
  bool _scrollPending = false;

  @override
  void dispose() {
    _scroll.dispose();
    super.dispose();
  }

  void _scrollToBottomAfterFrame() {
    if (_scrollPending) {
      return;
    }
    _scrollPending = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollPending = false;
      // hasContentDimensions too: a canvas whose layout was aborted in this
      // same frame (a build exception mid-tree) answers hasClients while its
      // position holds no dimensions — the extent read would throw.
      if (!mounted ||
          !_scroll.hasClients ||
          !_scroll.position.hasContentDimensions) {
        return;
      }
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final List<Message> messages = ref.watch(activeMessagesProvider);
    final bool isTyping = ref.watch(
      composerStoreProvider.select((ComposerState s) => s.isTyping),
    );

    // The export's scroll effect triggers (`app.tsx:851-853`).
    ref.listen<List<Message>>(activeMessagesProvider, (
      List<Message>? previous,
      List<Message> next,
    ) {
      if ((previous?.length ?? 0) < next.length) {
        _scrollToBottomAfterFrame();
      }
    });
    ref.listen<bool>(
      composerStoreProvider.select((ComposerState s) => s.isTyping),
      (bool? previous, bool next) {
        if (next) {
          _scrollToBottomAfterFrame();
        }
      },
    );

    return ColoredBox(
      color: const Color(0x1AFFFFFF), // rgba(255,255,255,0.10)
      child: messages.isEmpty
          ? const EmptySessionState()
          : SingleChildScrollView(
              key: const Key('chat.message-canvas'),
              controller: _scroll,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 896), // max-w-4xl
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: <Widget>[
                      for (int i = 0; i < messages.length; i++) ...<Widget>[
                        if (i != 0) const SizedBox(height: 24), // space-y-6
                        MessageBubble(
                          key: Key('chat.bubble-${messages[i].id}'),
                          message: messages[i],
                        ),
                      ],
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        curve: Curves.easeOut,
                        padding: EdgeInsets.only(top: isTyping ? 24 : 0),
                        child: AnimatedSwitcher(
                          // The AnimatePresence pairing
                          // (`app.tsx:2178-2180`): the indicator fades out
                          // when the reply lands.
                          duration: const Duration(milliseconds: 300),
                          child: isTyping
                              ? const TypingIndicator(
                                  key: ValueKey('chat.typing'),
                                )
                              : const SizedBox.shrink(
                                  key: ValueKey('chat.not-typing'),
                                ),
                        ),
                      ),
                      // The `bottomRef` anchor (`app.tsx:2181`).
                      const SizedBox(height: 4),
                    ],
                  ),
                ),
              ),
            ),
    );
  }
}

/// The empty-session prompt state (`app.tsx:2149-2172`): the spark + the
/// copy + the suggestion chips — ONLY when the session carries no messages
/// (the export's `messages.length === 0` branch).
class EmptySessionState extends ConsumerWidget {
  const EmptySessionState({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // The mock's empty state fills the viewport and centers its column
    // (`flex flex-col items-center justify-center h-full`) — scrolling only
    // when the content grows past it.
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints viewport) {
        return SingleChildScrollView(
          key: const Key('chat.empty-canvas'),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 48),
          child: ConstrainedBox(
            constraints: BoxConstraints(minHeight: viewport.maxHeight - 96),
            child: Center(
              child: ConstrainedBox(
                constraints:
                    const BoxConstraints(maxWidth: 512), // max-w-lg
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: <Widget>[
              SizedBox(
                key: const Key('chat.empty-spark'),
                width: 261,
                height: 261,
                child: const SparkAvatar(size: 261, bgAlpha: 0, glow: true),
              ),
              const SizedBox(height: 24),
              // The export's `PondrLogo h-9 w-44` wordmark (app.tsx:2157).
              const PondrWordmark(
                key: Key('chat.empty-logo'),
                width: 176,
                height: 36,
              ),
              const SizedBox(height: 12),
              const Text(
                'What would you like to ponder today? Ask anything — explanations, analysis, ideas, or creative exploration.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  height: 1.6,
                  color: Color(0xA6FFFFFF), // text-white/65
                ),
              ),
              const SizedBox(height: 32),
              // The suggestion chips (`app.tsx:2160-2171`) — a tap fills the
              // composer's draft (the export's `setInputText(s)`).
              _SuggestionChips(),
                  ],
                ),
              ),
            ),
          ),
      );
      },
    );
  }
}

/// The four SUGGESTIONS chips (`app.tsx:2160-2171`), two-up when the canvas
/// fits them (the export's `sm:grid-cols-2`).
class _SuggestionChips extends StatelessWidget {
  const _SuggestionChips();

  @override
  Widget build(BuildContext context) {
    return Consumer(
      builder: (BuildContext context, WidgetRef ref, _) {
        return LayoutBuilder(
          builder: (BuildContext context, BoxConstraints constraints) {
            final bool twoUp = constraints.maxWidth >= 512;
            final int columns = twoUp ? 2 : 1;
            final double chipWidth =
                (constraints.maxWidth - (columns - 1) * 8) / columns;
            return Wrap(
              key: const Key('chat.suggestions'),
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                for (final String suggestion in suggestions)
                  SizedBox(
                    width: chipWidth,
                    child: Material(
                      type: MaterialType.transparency,
                      child: InkWell(
                        key: Key('chat.suggestion-${suggestions.indexOf(suggestion)}'),
                        onTap: () {
                          ref
                              .read(composerStoreProvider.notifier)
                              .setText(suggestion);
                        },
                        borderRadius: BorderRadius.circular(12),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 12,
                          ),
                          decoration: BoxDecoration(
                            color: const Color(0x1A888DDF),
                            border: Border.all(color: const Color(0x4D888DDF)),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            children: <Widget>[
                              const Icon(
                                Icons.auto_awesome, // the export's Sparkles
                                size: 13,
                                color: Color(0xFFE2E6FF),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  suggestion,
                                  style: const TextStyle(
                                    fontSize: 14,
                                    color: Color(0xB3FFFFFF), // text-white/70
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            );
          },
        );
      },
    );
  }
}