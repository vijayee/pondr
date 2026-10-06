/// The settings view's ANIMATION, ported from `mockup_reference/app.tsx`'s
/// settings region — the reference is truth; every duration/offset here is
/// read off that file:
///
/// - the SECTIONS' crossfade (`AnimatePresence mode="wait"`,
///   `app.tsx:1310-1320`): enter opacity 0→1 + y 10→0, exit opacity 1→0 +
///   y 0→-6, both `duration: 0.18`;
/// - the providers' accordion bodies (`app.tsx:1428-1441`): height 0→auto +
///   opacity, `duration: 0.22, ease: "easeOut"`;
/// - the accordion's chevron (`app.tsx:1405-1408`): rotate 0→90°,
///   `duration: 0.18`;
/// - the inline edit/model forms (`app.tsx:1449-1452`, `1587-1590`,
///   `1658-1662`) and the saved toast (`app.tsx:1725-1729`): enter opacity
///   0→1 + y 4→0, exit opacity→0 — no `transition` given, so framer's
///   DEFAULT duration (0.3 s) applies;
/// - the new-provider form (`app.tsx:1740-1745`): enter opacity 0→1 + y 8→0,
///   exit opacity→0 + y +6, `duration: 0.18`.
library;

import 'package:flutter/material.dart';

// ─── The constants, one place ──────────────────────────────────────────────

/// The section crossfade's 0.18 s (`app.tsx:1319`).
const Duration kSectionMotion = Duration(milliseconds: 180);

/// The accordion body's `0.22 / easeOut` (`app.tsx:1438`).
const Duration kAccordionMotion = Duration(milliseconds: 220);

/// The chevron's 0.18 s (`app.tsx:1406`).
const Duration kChevronMotion = Duration(milliseconds: 180);

/// The 0.18 s the NEW-PROVIDER form plays (`app.tsx:1744`).
const Duration kProviderFormMotion = Duration(milliseconds: 180);

/// Framer's default: the inline forms + the toast omit `transition`
/// (`app.tsx:1449-1452`, `1587-1590`, `1658-1662`, `1725-1729`).
const Duration kInlineMotion = Duration(milliseconds: 300);

/// `mode="wait"`'s sections switcher: the outgoing child plays its EXIT to
/// completion, THEN the incoming child plays the enter. The export's
/// `AnimatePresence mode="wait"` `initial` defaults to TRUE, so the first
/// mount plays the enter too (`app.tsx:1310-1320`).
class SectionPresence extends StatefulWidget {
  const SectionPresence({
    required this.child,
    required this.childKey,
    super.key,
  });

  /// The section's content, rebuilt fresh by the parent whenever [childKey]'s
  /// identity changes (the child itself should NOT change mid-exit — the
  /// parent keys each section).
  final Widget child;

  /// The child's section identity — a NEW key while an exit is playing is
  /// queued (mode="wait": at most one exit runs; the exit animates the
  /// child that was last fully shown).
  final Key childKey;

  @override
  State<SectionPresence> createState() => _SectionPresenceState();
}

class _SectionPresenceState extends State<SectionPresence>
    with TickerProviderStateMixin {
  /// The section to SHOW when the present exit (if any) completes.
  late Key _targetKey = widget.childKey;
  Widget? _exitingChild;
  bool _exitPending = false;

  late final AnimationController _enter = AnimationController(
    vsync: this,
    duration: kSectionMotion,
  );
  late final AnimationController _exit = AnimationController(
    vsync: this,
    duration: kSectionMotion,
  );

  @override
  void initState() {
    super.initState();
    _enter.forward();
  }

  @override
  void didUpdateWidget(SectionPresence oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.childKey != _targetKey) {
      _targetKey = widget.childKey;
      if (!_exitPending) {
        // Freeze the leaving child exactly as it last rendered (mode="wait"
        // — one child at a time) and play the exit FIRST.
        _exitPending = true;
        setState(() {
          _exitingChild = oldWidget.child;
        });
        _enter.reset();
        _exit
          ..reset()
          ..forward().whenComplete(() {
            if (!mounted) {
              return;
            }
            setState(() {
              _exitingChild = null;
              _exitPending = false;
            });
            _enter.forward();
          });
      }
    }
  }

  @override
  void dispose() {
    _enter.dispose();
    _exit.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge(<Listenable>[_enter, _exit]),
      builder: (BuildContext context, Widget? _) {
        if (_exitingChild != null) {
          final double e = Curves.easeOut.transform(_exit.value);
          return Opacity(
            opacity: 1 - e,
            child: Transform.translate(
              offset: Offset(0, -6 * e), // exit y: -6
              child: _exitingChild,
            ),
          );
        }
        final double t = Curves.easeOut.transform(_enter.value);
        return Opacity(
          opacity: t,
          child: Transform.translate(
            offset: Offset(0, (1 - t) * 10), // initial y: 10
            child: KeyedSubtree(key: _targetKey, child: widget.child),
          ),
        );
      },
    );
  }
}

/// The inline forms' + toast's presence — the export's `AnimatePresence`
/// with `initial={{ opacity: 0, y: dy }} animate={{ opacity: 1, y: 0 }}
/// exit={{ opacity: 0, y: dyExit }}`: mount-on-present with a play-in, and
/// a play-out BEFORE unmount (mode wait, one child at a time).
class MotionPresence extends StatefulWidget {
  const MotionPresence({
    required this.present,
    required this.child,
    this.dyIn = 4,
    this.dyExit = 0,
    this.durationIn = kInlineMotion,
    this.durationExit = kInlineMotion,
    super.key,
  });

  final bool present;
  final Widget child;

  /// The enter's rise (`initial y: 4` — 8 for the new-provider form).
  final double dyIn;

  /// The exit's drop (`exit y: 6` for the new-provider form; 0 = plain fade).
  final double dyExit;

  /// The enter's duration.
  final Duration durationIn;

  /// The exit's duration.
  final Duration durationExit;

  @override
  State<MotionPresence> createState() => _MotionPresenceState();
}

class _MotionPresenceState extends State<MotionPresence>
    with SingleTickerProviderStateMixin {
  // Created in initState — a NEVER-present form (the common mount) must
  // still dispose cleanly; a first-use creation inside dispose() looks up
  // the TickerMode of a DEACTIVATED element (the debug crash this guards).
  late final AnimationController _controller;
  bool _visible = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this);
    _visible = widget.present;
    if (_visible) {
      _controller
        ..duration = widget.durationIn
        ..forward();
    }
  }

  @override
  void didUpdateWidget(MotionPresence oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.present != oldWidget.present) {
      _controller.duration = widget.present
          ? widget.durationIn
          : widget.durationExit;
      if (widget.present) {
        setState(() => _visible = true);
        _controller.forward();
      } else {
        _controller.reverse().whenComplete(() {
          if (mounted) {
            // Do not clobber a re-enter that already became visible.
            if (!widget.present) {
              setState(() => _visible = false);
            }
          }
        });
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_visible) {
      return const SizedBox(width: double.infinity);
    }
    return AnimatedBuilder(
      animation: _controller,
      builder: (BuildContext context, Widget? _) {
        final double t = Curves.easeOut.transform(_controller.value);
        final bool reversing = _controller.status == AnimationStatus.reverse;
        final double dy = reversing
            ? (1 - t) * widget.dyExit
            : (1 - t) * widget.dyIn;
        return Opacity(
          opacity: t,
          child: Transform.translate(
            offset: Offset(0, dy),
            child: widget.child,
          ),
        );
      },
    );
  }
}

/// The providers' accordion body — the export's height 0→auto + opacity
/// animate with `0.22 / easeOut` (`app.tsx:1428-1441`). [AnimatedCrossFade]
/// is the two-child size+fade cross: the collapsed side is an empty box, and
/// a collapsed body's children stay mounted but pointer-dead (the export's
/// edit form SURVIVES a collapse — its `showProviderForm &&
/// editingProviderId` only clears on save/cancel/delete).
class AccordionBody extends StatelessWidget {
  const AccordionBody({required this.open, required this.child, super.key});

  final bool open;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return AnimatedCrossFade(
      duration: kAccordionMotion,
      sizeCurve: Curves.easeOut,
      firstCurve: Curves.easeOut,
      secondCurve: Curves.easeOut,
      crossFadeState: open
          ? CrossFadeState.showFirst
          : CrossFadeState.showSecond,
      firstChild: IgnorePointer(ignoring: !open, child: child),
      secondChild: const SizedBox(width: double.infinity),
    );
  }
}
