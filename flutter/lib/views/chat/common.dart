import 'package:flutter/material.dart';

/// Shared atoms of the chat surface: the spark avatar slot, the press-scale
/// affordance and the anchored-popover scaffold the attachment picker and
/// the model dropdown build on.

/// The ThoughtSpark avatar slot — the iridescent SVG's circle. The asset
/// itself lands with the auth views' SVG work (Task 5); the material
/// stand-in holds the circle + glow shape (`app.tsx:743` / `:1909-1911`).
class SparkAvatar extends StatelessWidget {
  const SparkAvatar({this.size = 28, this.bgAlpha = 0.18, this.glow = false, super.key});

  /// The circle's diameter (the export's `w-7 h-7` at 28).
  final double size;

  /// The circle fill's alpha on `#C3ACDA` (the export's `rgba(195,172,218,x)`).
  final double bgAlpha;

  /// The export's blurred halo (`absolute inset-0 rounded-full blur-sm`,
  /// `app.tsx:1910`) — the sidebar's 44 dp logo mark.
  final bool glow;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: Color.fromARGB((bgAlpha * 255).round(), 0xC3, 0xAC, 0xDA),
        shape: BoxShape.circle,
        boxShadow: glow
            ? const <BoxShadow>[
                BoxShadow(
                  color: Color(0x66888DDF), // rgba(136,141,223,0.4)
                  blurRadius: 6,
                ),
              ]
            : null,
      ),
      alignment: Alignment.center,
      child: Opacity(
        opacity: 0.9,
        child: Icon(
          Icons.auto_awesome,
          size: size * 0.6,
          color: const Color(0xFFE2E6FF),
        ),
      ),
    );
  }
}

/// The export's `active:scale-*` press affordance (`app.tsx:1932,2244`):
/// the wrapped child compresses while the pointer is down and eases back.
class PressScale extends StatefulWidget {
  const PressScale({
    required this.child,
    required this.pressedScale,
    super.key,
  });

  final Widget child;
  final double pressedScale;

  @override
  State<PressScale> createState() => _PressScaleState();
}

class _PressScaleState extends State<PressScale> {
  bool _down = false;

  @override
  Widget build(BuildContext context) {
    return Listener(
      onPointerDown: (_) => setState(() => _down = true),
      onPointerUp: (_) => setState(() => _down = false),
      onPointerCancel: (_) => setState(() => _down = false),
      child: AnimatedScale(
        scale: _down ? widget.pressedScale : 1.0,
        duration: const Duration(milliseconds: 150), // the export's `transition-all`
        curve: Curves.easeOut,
        child: widget.child,
      ),
    );
  }
}

/// The popovers' shared container (the export's model dropdown,
/// `app.tsx:2297-2299` — the mock's `#1a1929` panel with the primary-tinted
/// border and the `0 8px 32px` shadow).
const BoxDecoration pondrMenuDecoration = BoxDecoration(
  color: Color(0xFF1A1929),
  border: Border.fromBorderSide(BorderSide(color: Color(0x4D888DDF))),
  borderRadius: BorderRadius.all(Radius.circular(12)),
  boxShadow: <BoxShadow>[
    BoxShadow(
      color: Color(0x80000000), // rgba(0,0,0,0.5)
      blurRadius: 32,
      offset: Offset(0, 8),
    ),
  ],
);

/// The anchored popover scaffold: [anchor] is wrapped as the
/// [CompositedTransformTarget]; while [open], the root overlay carries a
/// full-screen tap-away barrier plus the menu positioned above-left of the
/// anchor (`bottom-full mb-2 left-0`, `app.tsx:2297`).
///
/// Closing is ALWAYS through [onDismiss] (the barrier, an item's callback,
/// the anchor's own toggle) — no imperative controller is exposed.
class AnchoredMenu extends StatefulWidget {
  const AnchoredMenu({
    required this.open,
    required this.onDismiss,
    required this.anchor,
    required this.menuBuilder,
    super.key,
  });

  final bool open;
  final VoidCallback onDismiss;
  final Widget anchor;
  final Widget Function(VoidCallback dismiss) menuBuilder;

  @override
  State<AnchoredMenu> createState() => _AnchoredMenuState();
}

class _AnchoredMenuState extends State<AnchoredMenu> {
  final OverlayPortalController _portal = OverlayPortalController();
  final LayerLink _link = LayerLink();

  @override
  void initState() {
    super.initState();
    if (widget.open) _portal.show();
  }

  @override
  void didUpdateWidget(AnchoredMenu oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.open != oldWidget.open) {
      // show()/hide() are not build-phase calls — the flag can flip during
      // a build; defer a frame.
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) {
          return;
        }
        if (widget.open) {
          if (_portal.isShowing) {
            _portal.hide();
          }
          _portal.show();
        } else if (_portal.isShowing) {
          _portal.hide();
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return OverlayPortal(
      controller: _portal,
      overlayChildBuilder: (BuildContext context) {
        return Stack(
          children: <Widget>[
            Positioned.fill(
              child: GestureDetector(
                behavior: HitTestBehavior.opaque,
                onTap: widget.onDismiss,
                child: const SizedBox.expand(),
              ),
            ),
            CompositedTransformFollower(
              link: _link,
              targetAnchor: Alignment.topLeft,
              followerAnchor: Alignment.bottomLeft,
              offset: const Offset(0, -8),
              showWhenUnlinked: false,
              child: widget.menuBuilder(widget.onDismiss),
            ),
          ],
        );
      },
      child: CompositedTransformTarget(link: _link, child: widget.anchor),
    );
  }
}

/// A popover panel's entrance: fade + rise + optional scale, once, after
/// [delay]. The mockup's profile classes:
/// - the model dropdown — 0.15 s, dy 6, scale 0.97 (`app.tsx:2293-2296`);
/// - the attachment sheet — 0.18 s, dy 4 (`docs/flutter-app-spec.md`'s
///   chat row).
class MenuPanelEntrance extends StatefulWidget {
  const MenuPanelEntrance({
    required this.child,
    required this.delay,
    required this.dy,
    required this.duration,
    this.initialScale = 1.0,
    super.key,
  });

  final Widget child;
  final Duration delay;
  final double dy;
  final double initialScale;
  final Duration duration;

  @override
  State<MenuPanelEntrance> createState() => _MenuPanelEntranceState();
}

class _MenuPanelEntranceState extends State<MenuPanelEntrance>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: widget.duration,
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
            offset: Offset(0, (1 - t) * widget.dy),
            child: Transform.scale(
              scale: widget.initialScale + (1.0 - widget.initialScale) * t,
              child: child,
            ),
          ),
        );
      },
      child: widget.child,
    );
  }
}