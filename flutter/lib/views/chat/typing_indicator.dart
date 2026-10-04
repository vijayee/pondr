import 'package:flutter/material.dart';

/// The export's TypingIndicator (`app.tsx:703-728`).
///
/// Motion, verbatim: the whole row fades in rising from y:8; three dots
/// pulse `scale [1, 1.5, 1]` / `opacity [0.4, 1, 0.4]` over a 1.1 s period,
/// each dot phase-delayed 0.18 s after the previous one, repeating forever.
/// The EXIT (the export's `exit={{ opacity: 0 }}`) is the message canvas'
/// AnimatedSwitcher's fade (chat_page.dart), per the AnimatePresence idiom.
class TypingIndicator extends StatefulWidget {
  const TypingIndicator({super.key});

  /// The export's pulse period (`app.tsx:722`: duration 1.1, repeat ∞).
  static const Duration period = Duration(milliseconds: 1100);

  /// The export's inter-dot delay (`transition.delay: i * 0.18`).
  static const Duration dotDelay = Duration(milliseconds: 180);

  /// The entrance's rise (the export's `initial: { opacity: 0, y: 8 }`).
  static const double entranceDy = 8;

  /// The entrance duration — the bubble class's value (`app.tsx:739` pins
  /// 0.22 s; the indicator's own entrance carries no explicit duration in
  /// the export, so it rides the same class).
  static const Duration entranceDuration = Duration(milliseconds: 220);

  @override
  State<TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<TypingIndicator>
    with TickerProviderStateMixin {
  /// The repeating dot pulse — one clock, three phase offsets.
  late final AnimationController _pulse = AnimationController(
    vsync: this,
    duration: TypingIndicator.period,
  )..repeat();

  late final AnimationController _entranceController = AnimationController(
    vsync: this,
    duration: TypingIndicator.entranceDuration,
  )..forward();

  @override
  void dispose() {
    _pulse.dispose();
    _entranceController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _entranceController,
      builder: (BuildContext context, Widget? child) {
        final double t = Curves.easeOut.transform(_entranceController.value);
        return Opacity(
          opacity: t,
          child: Transform.translate(
            offset: Offset(0, (1 - t) * TypingIndicator.entranceDy),
            child: child,
          ),
        );
      },
      child: _IndicatorRow(controller: _pulse),
    );
  }
}

/// The pulse row: the spark avatar + the bordered bubble of three dots
/// (`app.tsx:709-726`).
class _IndicatorRow extends StatelessWidget {
  const _IndicatorRow({required this.controller});

  final AnimationController controller;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        const _SparkAvatar(),
        const SizedBox(width: 12),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          decoration: BoxDecoration(
            color: const Color(0x1F888DDF),
            border: Border.all(color: const Color(0x4D888DDF)),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(4),
              topRight: Radius.circular(16),
              bottomRight: Radius.circular(16),
              bottomLeft: Radius.circular(16),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              for (int i = 0; i < 3; i++) ...<Widget>[
                if (i != 0) const SizedBox(width: 6),
                _PulseDot(controller: controller, index: i),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

/// One dot: a phase-shifted `1 → 1.5 → 1` scale / `0.4 → 1 → 0.4` opacity
/// triangle over the shared 1.1 s clock (the export's keyframe pair,
/// `app.tsx:721-722`).
class _PulseDot extends StatelessWidget {
  const _PulseDot({required this.controller, required this.index});

  final AnimationController controller;
  final int index;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (BuildContext context, Widget? child) {
        final double period =
            TypingIndicator.period.inMilliseconds.toDouble();
        final double phase =
            TypingIndicator.dotDelay.inMilliseconds * index / period;
        // The mock's `delay: i * 0.18` with `repeat: Infinity` is THIS clock
        // read with a wrapped offset — dot 2 at t picks up dot 0's phase
        // `0.36 s` earlier state, not its own start-later run.
        final double v = (controller.value - phase) % 1.0;
        final double triangle = v < 0.5 ? v * 2 : 2 - v * 2;
        return Transform.scale(
          scale: 1.0 + 0.5 * triangle,
          child: Opacity(
            opacity: 0.4 + 0.6 * triangle,
            child: child,
          ),
        );
      },
      child: Container(
        width: 6,
        height: 6,
        decoration: BoxDecoration(
          color: const Color(0xB3888DDF),
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}

/// The 28 dp spark avatar (the export's `w-7 h-7` circle on
/// `rgba(195,172,218,0.2)`); the ThoughtSpark SVG lands in Task 5's asset
/// work — the material stand-in holds the slot.
class _SparkAvatar extends StatelessWidget {
  const _SparkAvatar();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 28,
      height: 28,
      decoration: const BoxDecoration(
        color: Color(0x33C3ACDA),
        shape: BoxShape.circle,
      ),
      alignment: Alignment.center,
      child: const Icon(Icons.auto_awesome, size: 18, color: Color(0xFFE2E6FF)),
    );
  }
}