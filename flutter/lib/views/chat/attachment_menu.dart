import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../theme/tokens.dart';
import 'common.dart';

/// The composer's attachment surface.
///
/// The export's composer uses a NATIVE file input behind the paperclip
/// (`app.tsx:2216-2222`) — desktop v1 pins the spec's swap: a MOCK picker
/// sheet ("docs/flutter-app-spec.md", the chat row: "the attachment
/// `+`-menu (a mock picker sheet — menu items animate in per the mockup's
/// `y:4 → 0` entrances, 0.18 s class)"). The chips row above the input and
/// the remove affordance are the export's, `app.tsx:2190-2208`.

/// The mock picker's offerings — a name/type/size mix covering the export's
/// accept list's icon kinds (image / document / other).
const List<({String name, String type, int size})> mockAttachmentChoices =
    <({String name, String type, int size})>[
      (name: 'research-notes.pdf', type: 'application/pdf', size: 1945600),
      (name: 'portrait.png', type: 'image/png', size: 430080),
      (name: 'session-transcript.txt', type: 'text/plain', size: 12600),
      (name: 'observations.csv', type: 'text/csv', size: 88000),
    ];

/// The mock picker's sheet. [onPick] hands the chosen mock file to the
/// composer's attachment store; [dismiss] closes via [widget]'s barrier.
class AttachmentMenu extends StatelessWidget {
  const AttachmentMenu({required this.onPick, required this.dismiss, super.key});

  final void Function(({String name, String type, int size})) onPick;
  final VoidCallback dismiss;

  @override
  Widget build(BuildContext context) {
    return Material(
      type: MaterialType.transparency,
      // The sheet's panel rides the mockup's 0.18 s / dy-4 profile class.
      child: MenuPanelEntrance(
        delay: const Duration(milliseconds: 4),
        dy: 4,
        duration: const Duration(milliseconds: 180),
        child: Container(
      key: const Key('attach.menu'),
      decoration: pondrMenuDecoration,
      constraints: const BoxConstraints(minWidth: 240),
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          for (int i = 0; i < mockAttachmentChoices.length; i++)
            MenuItemEntrance(
              delay: Duration(milliseconds: 60 * i),
              child: _ChoiceRow(
                choice: mockAttachmentChoices[i],
                onTap: () {
                  onPick(mockAttachmentChoices[i]);
                  dismiss();
                },
              ),
            ),
        ],
      ),
        ),
      ),
    );
  }
}

/// One row: the entrance's `y: 4 → 0` fade + the export-style row content
/// (icon by `fileIconKind`, the name, the formatted size).
class _ChoiceRow extends StatelessWidget {
  const _ChoiceRow({required this.choice, required this.onTap});

  final ({String name, String type, int size}) choice;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final IconData icon = switch (fileIconKind(choice.type)) {
      AttachedFileIcon.image => Icons.image,
      AttachedFileIcon.document => Icons.description,
      AttachedFileIcon.other => Icons.insert_drive_file,
    };
    return InkWell(
      key: Key('attach.item-${choice.name}'),
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: <Widget>[
            Icon(icon, size: 16, color: PondrTokens.primary),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                choice.name,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(color: Colors.white, fontSize: 13),
              ),
            ),
            Text(
              formatFileSize(choice.size),
              style: const TextStyle(fontSize: 11, color: Color(0x80FFFFFF)),
            ),
          ],
        ),
      ),
    );
  }
}

/// The export's `y: 4 → 0` + fade entrance, 0.18 s — the mockup's menu-item
/// class (staggered here by [delay] via a real timer, so `pumpAndSettle`
/// reaches the settled state).
class MenuItemEntrance extends StatefulWidget {
  const MenuItemEntrance({
    required this.child,
    required this.delay,
    super.key,
  });

  final Widget child;
  final Duration delay;

  @override
  State<MenuItemEntrance> createState() => _MenuItemEntranceState();
}

class _MenuItemEntranceState extends State<MenuItemEntrance>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 180),
  );

  @override
  void initState() {
    super.initState();
    Future<void>.delayed(widget.delay, () {
      if (mounted) {
        _controller.forward();
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (BuildContext context, Widget? child) {
        final double t = Curves.easeOut.transform(_controller.value);
        return Opacity(
          opacity: t,
          child: Transform.translate(
            offset: Offset(0, (1 - t) * 4),
            child: child,
          ),
        );
      },
      child: widget.child,
    );
  }
}

/// The attached preview row above the input box (`app.tsx:2190-2208`).
class AttachedChipsRow extends StatelessWidget {
  const AttachedChipsRow({required this.files, required this.onRemove, super.key});

  final List<AttachedFile> files;
  final void Function(String id) onRemove;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: <Widget>[
          for (final AttachedFile file in files)
            _RemoveableChip(file: file, onRemove: () => onRemove(file.id)),
        ],
      ),
    );
  }
}

class _RemoveableChip extends StatelessWidget {
  const _RemoveableChip({required this.file, required this.onRemove});

  final AttachedFile file;
  final VoidCallback onRemove;

  @override
  Widget build(BuildContext context) {
    final IconData icon = switch (file.icon) {
      AttachedFileIcon.image => Icons.image,
      AttachedFileIcon.document => Icons.description,
      AttachedFileIcon.other => Icons.insert_drive_file,
    };
    return Container(
      key: Key('attach.chip-${file.name}'),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0x1F888DDF),
        border: Border.all(color: const Color(0x59888DDF)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Icon(icon, size: 12, color: PondrTokens.primary),
          const SizedBox(width: 8),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 140),
            child: Text(
              file.name,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 12, color: Color(0xD9FFFFFF)),
            ),
          ),
          const SizedBox(width: 4),
          Text(
            formatFileSize(file.size),
            style: const TextStyle(fontSize: 12, color: Color(0x73FFFFFF)),
          ),
          const SizedBox(width: 2),
          GestureDetector(
            key: Key('attach.remove-${file.name}'),
            onTap: onRemove,
            child: const Icon(Icons.close, size: 11, color: Color(0x73FFFFFF)),
          ),
        ],
      ),
    );
  }
}