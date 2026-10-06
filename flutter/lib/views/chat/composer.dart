import 'package:flutter/material.dart';
import 'package:flutter/services.dart'
    show HardwareKeyboard, KeyDownEvent, LogicalKeyboardKey;
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../data/models.dart';
import '../../theme/tokens.dart';
import 'attachment_menu.dart';
import 'chat_state.dart';
import 'common.dart';
import 'model_picker.dart';

/// The export's input area (`app.tsx:2186-2337`): the attached preview, the
/// rounded input box on `rgba(136,141,223,0.10)` with the paperclip, the
/// auto-growing draft (Enter sends, Shift+Enter breaks, 160 px cap), the
/// SEND button's disabled logic, and the bottom bar's model picker +
/// disclaimer.
class ChatComposer extends ConsumerStatefulWidget {
  const ChatComposer({super.key});

  @override
  ConsumerState<ChatComposer> createState() => _ChatComposerState();
}

class _ChatComposerState extends ConsumerState<ChatComposer> {
  final TextEditingController _text = TextEditingController();
  final FocusNode _focus = FocusNode();

  @override
  void dispose() {
    _text.dispose();
    _focus.dispose();
    super.dispose();
  }

  void _send() => ref.read(composerStoreProvider.notifier).send();

  @override
  Widget build(BuildContext context) {
    final ComposerState composer = ref.watch(composerStoreProvider);

    // The store is the draft's source of truth (the suggestion chips and
    // New Chat write it); the controller mirrors external changes.
    ref.listen<ComposerState>(composerStoreProvider, (
      ComposerState? previous,
      ComposerState next,
    ) {
      if (next.text != _text.text) {
        _text.text = next.text;
        _text.selection = TextSelection.collapsed(offset: _text.text.length);
      }
    });

    return Material(
      type: MaterialType.transparency,
      child: Container(
      key: const Key('composer.root'),
      color: const Color(0xF51A1929), // rgba(26,25,41,0.96)
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          if (composer.attached.isNotEmpty) ...<Widget>[
            AttachedChipsRow(
              files: composer.attached,
              onRemove: (String id) =>
                  ref.read(composerStoreProvider.notifier).removeFile(id),
            ),
            const SizedBox(height: 8),
          ],
          _InputBox(
            composer: composer,
            controller: _text,
            focus: _focus,
            onChanged: (String value) =>
                ref.read(composerStoreProvider.notifier).setText(value),
            onSend: _send,
          ),
          const SizedBox(height: 8), // mt-2
          Row(
            children: <Widget>[
              const ModelPicker(),
              const Spacer(),
              Flexible(
                child: Text(
                  'Pondr may make mistakes — verify important information.',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.right,
                  style: const TextStyle(
                    fontSize: 11,
                    color: Color(0x4DFFFFFF), // text-white/30
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
      ),
    );
  }
}

/// The input box (`app.tsx:2212-2255`): paperclip (opening the mock picker
/// sheet), the draft field, the SEND button.
class _InputBox extends StatelessWidget {
  const _InputBox({
    required this.composer,
    required this.controller,
    required this.focus,
    required this.onChanged,
    required this.onSend,
  });

  final ComposerState composer;
  final TextEditingController controller;
  final FocusNode focus;
  final ValueChanged<String> onChanged;
  final VoidCallback onSend;

  bool get _canSend =>
      controller.text.trim().isNotEmpty || composer.attached.isNotEmpty;

  @override
  Widget build(BuildContext context) {
    // focus-within:border-primary/40 (app.tsx:2213) — the border brightens
    // while the draft field holds focus.
    return AnimatedBuilder(
      animation: focus,
      builder: (BuildContext context, Widget? child) {
        return Container(
          padding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: const Color(0x1A888DDF), // rgba(136,141,223,0.10)
            border: Border.all(
              color: focus.hasFocus
                  ? const Color(0x66888DDF) // border-primary/40
                  : const Color(0x59888DDF), // rgba(...,0.35)
            ),
            borderRadius: BorderRadius.circular(16), // rounded-2xl
          ),
          child: child,
        );
      },
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: <Widget>[
          _AttachButton(),
          const SizedBox(width: 8),
          Expanded(
            // The mock's cap: maxHeight 160 px ≈ 7 lines at the field's
            // 21 px line box; beyond it the field scrolls internally
            // (app.tsx:2230,2236). EDITABLETEXT cannot measure an unbounded
            // multiline body here, so the cap is expressed as a maxLines.
            child: Focus(
                // The export's handleKeyDown (`app.tsx:933-938`): Enter
                // sends, Shift+Enter inserts the line break.
                onKeyEvent: (FocusNode node, KeyEvent event) {
                  if (event is KeyDownEvent &&
                      event.logicalKey == LogicalKeyboardKey.enter &&
                      !HardwareKeyboard.instance.isShiftPressed) {
                    onSend();
                    return KeyEventResult.handled;
                  }
                  return KeyEventResult.ignored;
                },
                child: TextField(
                  key: const Key('composer.field'),
                  controller: controller,
                  focusNode: focus,
                  maxLines: controller.text.split('\n').length.clamp(1, 7),
                  keyboardType: TextInputType.multiline,
                  textInputAction: TextInputAction.newline,
                  decoration: const InputDecoration(
                    isCollapsed: true,
                    filled: false,
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(vertical: 3),
                    hintText: 'Ask Pondr anything… (Shift+Enter for new line)',
                    hintStyle: TextStyle(
                      fontSize: 14,
                      color: Color(0x59FFFFFF), // placeholder text-white/35
                    ),
                  ),
                  style: const TextStyle(
                    fontSize: 14,
                    height: 1.5,
                    color: Colors.white,
                  ),
                  cursorColor: PondrTokens.ring,
                  onChanged: onChanged,
                ),
            ),
          ),
          const SizedBox(width: 8),
          _SendButton(enabled: _canSend, onTap: _canSend ? onSend : null),
        ],
      ),
    );
  }
}

/// The paperclip; the mock's native file dialog is the spec's MOCK picker
/// sheet anchored above (`app.tsx:2216-2218`).
class _AttachButton extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool open = ref.watch(attachmentsMenuOpenProvider);
    return AnchoredMenu(
      open: open,
      onDismiss: () =>
          ref.read(attachmentsMenuOpenProvider.notifier).set(false),
      anchor: IconButton(
        key: const Key('composer.attach-button'),
        tooltip: 'Attach documents or images',
        icon: Icon(
          Icons.attach_file, // the export's Paperclip
          size: 17,
          color: open ? PondrTokens.primary : const Color(0x80FFFFFF),
        ),
        onPressed: () =>
            ref.read(attachmentsMenuOpenProvider.notifier).toggle(),
      ),
      menuBuilder: (VoidCallback dismiss) {
        return AttachmentMenu(
          onPick: (({String name, String type, int size}) choice) {
            final DateTime now = DateTime.now();
            ref.read(composerStoreProvider.notifier).attach(
                  AttachedFile(
                    id: 'f${now.microsecondsSinceEpoch}-${choice.name}',
                    name: choice.name,
                    type: choice.type,
                    size: choice.size,
                  ),
                );
          },
          dismiss: dismiss,
        );
      },
    );
  }
}

/// The SEND button (`app.tsx:2239-2254`): 32×32 rounded-lg on `#e0efe4`,
/// the glow only while sendable, dimmed at 0.4 when not, and press-scaled
/// (`active:scale-90`).
class _SendButton extends StatelessWidget {
  const _SendButton({required this.enabled, required this.onTap});

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
          shadowColor: const Color(0x80E0EFE4), // rgba(224,239,228,0.5)
          elevation: 0,
          child: InkWell(
            key: const Key('composer.send'),
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