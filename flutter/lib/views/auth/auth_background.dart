import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';

import '../../theme/tokens.dart';
import '../chat/common.dart' show kPondrLogoAsset, PressScale;

/// The auth branch's shared chrome: the export's full-screen ambience, the
/// animated card shell, and the form atoms the two views compose. Every
/// value here is pinned to `mockup_reference/app.tsx`'s auth region
/// (`app.tsx:947-1145`).

/// Test keys — the card shells and the form controls, per view.
const Key authLoginCardKey = Key('login.card');
const Key authRegisterCardKey = Key('register.card');
const Key loginUsernameKey = Key('login.username');
const Key loginPasswordKey = Key('login.password');
const Key loginEyeKey = Key('login.eye');
const Key loginSubmitKey = Key('login.submit');
const Key loginSwitchKey = Key('login.switch');
const Key registerNameKey = Key('reg.name');
const Key registerUsernameKey = Key('reg.username');
const Key registerPasswordKey = Key('reg.password');
const Key registerConfirmKey = Key('reg.confirm');
const Key registerEyeKey = Key('reg.eye');
const Key registerSubmitKey = Key('reg.submit');
const Key registerSwitchKey = Key('reg.switch');

/// The export's input decoration (`app.tsx:948-950`): `rounded-xl px-4
/// py-3` on the primary-tinted fill `rgba(136,141,223,0.12)` + the 1 px
/// `rgba(136,141,223,0.45)` border, the 2 px `ring-primary/60` focus outline
/// and the `text-white/35` hint.
InputDecoration _fieldDecoration(String hint, Widget? suffixIcon) {
  return InputDecoration(
    filled: true,
    fillColor: const Color(0x1F888DDF), // rgba(136,141,223,0.12)
    contentPadding: const EdgeInsets.fromLTRB(16, 12, 16, 12), // px-4 py-3
    hintText: hint,
    hintStyle: const TextStyle(
      fontFamily: PondrTokens.fontFamily,
      fontWeight: FontWeight.w500,
      color: Color(0x59FFFFFF), // text-white/35
    ),
    border: const OutlineInputBorder(
      borderRadius: BorderRadius.all(Radius.circular(12)), // rounded-xl
      borderSide: BorderSide(color: Color(0x73888DDF)), // rgba(...,0.45)
    ),
    enabledBorder: const OutlineInputBorder(
      borderRadius: BorderRadius.all(Radius.circular(12)),
      borderSide: BorderSide(color: Color(0x73888DDF)),
    ),
    focusedBorder: const OutlineInputBorder(
      borderRadius: BorderRadius.all(Radius.circular(12)),
      borderSide: BorderSide(
        color: Color(0x99888DDF),
        width: 2,
      ), // ring-primary/60
    ),
    suffixIcon: suffixIcon,
  );
}

/// The auth screen's background (the export's auth container,
/// `app.tsx:952-968`): the `rgba(255,255,255,0.1)` wash over the body's
/// `--background`, the two ambient glow discs, and the 48 dp dot grid at
/// 15%. The CSS `blur` discs read as radial gradients — the Gaussian's
/// falloff, drawn cheap.
class AuthBackground extends StatelessWidget {
  const AuthBackground({required this.child, super.key});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(
        color: PondrTokens.background,
      ), // body bg-background
      child: Stack(
        children: <Widget>[
          const Positioned.fill(
            child: ColoredBox(
              color: Color(0x1AFFFFFF),
            ), // rgba(255,255,255,0.1)
          ),
          const Positioned.fill(
            child: ClipRect(
              child: CustomPaint(painter: _AuthAmbiencePainter()),
            ),
          ),
          // `min-h-screen flex items-center justify-center p-4` (app.tsx:951).
          // `min-h-screen` is a MINIMUM: a browser scrolls its VIEWPORT when
          // the card is taller than the window (the register card alone is
          // ~760 dp, which overflowed 246 px at a 500×500 window before this
          // scroll). The scroll view is that page scroll — the centered
          // `p-4` shape is preserved whenever the card fits.
          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: ConstrainedBox(
                constraints: const BoxConstraints(
                  maxWidth: 448, // max-w-md — the motion.div's box
                ),
                child: child,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// The two glow discs + the dot grid, one painter:
/// - `w-[500px]h-[500px] blur-[120px]` primary-22 at `top-1/4 left-1/3`
///   (`app.tsx:956-958`);
/// - `w-80h-80 blur-[100px]` accent-16 at `bottom-1/4 right-1/4`
///   (`app.tsx:959-960`);
/// - the `48px` tile dot grid (`radial-gradient ... 1px`) at 15% opacity
///   (`app.tsx:961-966`) — the composite the export's layer opacity makes.
class _AuthAmbiencePainter extends CustomPainter {
  const _AuthAmbiencePainter();

  void _glow(Canvas canvas, Offset center, double radius, Color color) {
    final Rect rect = Rect.fromCircle(center: center, radius: radius);
    final Paint paint = Paint()
      ..shader = RadialGradient(
        colors: <Color>[color, color.withValues(alpha: 0)],
      ).createShader(rect);
    canvas.drawCircle(center, radius, paint);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final double w = size.width;
    final double h = size.height;
    _glow(
      canvas,
      Offset(w / 3 + 250, h / 4 + 250),
      250,
      const Color(0x38888DDF), // rgba(136,141,223,0.22)
    );
    _glow(
      canvas,
      Offset(w - w / 4 - 320 + 160, h - h / 4 - 320 + 160),
      160,
      const Color(0x29C3ACDA), // rgba(195,172,218,0.16)
    );
    final Paint dot = Paint()
      ..color = const Color(
        0x13C8C3FF,
      ); // rgba(200,195,255,0.5) * the 15% layer
    for (
      double x = 24;
      x < w + 24;
      x += 48 // backgroundSize: 48px 48px
    ) {
      for (double y = 24; y < h + 24; y += 48) {
        canvas.drawCircle(Offset(x, y), 1, dot);
      }
    }
  }

  @override
  bool shouldRepaint(_AuthAmbiencePainter oldDelegate) => false;
}

/// `void Function(String route)` — [AuthCardShell]'s switch handle.
typedef SwitchTo = void Function(String route);

/// The export's `motion.div` card wrapper + the `AnimatePresence` pair
/// (`app.tsx:969-974,1044-1049`): every mount plays the ENTER — opacity
/// 0→1, y 20→0, 0.3 s easeOut — and every cross-auth SWITCH first plays the
/// EXIT — opacity 1→0, y 0→-16, 0.3 s easeOut — BEFORE the next view's
/// enter. That is `mode="wait"` (app.tsx:969), and it lives HERE rather
/// than in the route pages: go_router's page swaps replace the stack
/// wholesale, so this shell plays its exit in view and only then hands off
/// through `context.go(route)`; the destination's fresh mount plays the
/// enter. Every OTHER branch swap stays instant (the export's bare
/// `setView` — `app.tsx:855-863`), so the router's pages carry a
/// zero-duration transition (lib/app/router.dart).
class AuthCardShell extends StatefulWidget {
  const AuthCardShell({
    required this.builder,
    required this.cardKey,
    super.key,
  });

  /// The view's card content; [SwitchTo] plays the exit and then navigates.
  final Widget Function(BuildContext context, SwitchTo switchTo) builder;

  /// The card's test key — INSIDE the motion wrapper ([Opacity]), so the
  /// test pins read the shell's animated Opacity as the card's ancestor.
  final Key cardKey;

  /// The export's `transition={{ duration: 0.3, ease: "easeOut" }}`
  /// (app.tsx:973,1047).
  static const Duration kMotion = Duration(milliseconds: 300);

  @override
  State<AuthCardShell> createState() => _AuthCardShellState();
}

class _AuthCardShellState extends State<AuthCardShell>
    with TickerProviderStateMixin {
  late final AnimationController _enter = AnimationController(
    vsync: this,
    duration: AuthCardShell.kMotion,
    value: 0,
  );
  late final AnimationController _exit = AnimationController(
    vsync: this,
    duration: AuthCardShell.kMotion,
    value: 0,
  );

  @override
  void initState() {
    super.initState();
    _enter.forward();
  }

  @override
  void dispose() {
    _enter.dispose();
    _exit.dispose();
    super.dispose();
  }

  void _switchTo(String route) {
    if (_exit.isAnimating) {
      return;
    }
    _exit.forward().whenComplete(() {
      if (mounted) {
        context.go(route);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    // This builder runs the shell's OWN animation tick; the card content is
    // built fresh per frame like the export's re-render.
    return AnimatedBuilder(
      animation: Listenable.merge(<Listenable>[_enter, _exit]),
      builder: (BuildContext context, Widget? _) {
        final double t = Curves.easeOut.transform(_enter.value);
        final double e = Curves.easeOut.transform(_exit.value);
        return Opacity(
          opacity: t * (1 - e),
          child: Transform.translate(
            offset: Offset(
              0,
              (1 - t) * 20 - e * 16,
            ), // enter y:20↓ / exit y:-16↑
            // The card key INSIDE the motion wrapper: the test pins read the
            // shell's animated Opacity as the card's ancestor.
            child: KeyedSubtree(
              key: widget.cardKey,
              child: widget.builder(context, _switchTo),
            ),
          ),
        );
      },
    );
  }
}

/// The card's container: `rounded-2xl p-8` on `rgba(26,25,41,0.96)`
/// (app.tsx:975-981,1049-1055).
class AuthCard extends StatelessWidget {
  const AuthCard({required this.child, super.key});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(
        color: Color(0xF51A1929), // rgba(26,25,41,0.96)
        borderRadius: BorderRadius.all(Radius.circular(16)), // rounded-2xl
      ),
      child: Padding(
        padding: const EdgeInsets.all(32), // p-8
        child: child,
      ),
    );
  }
}

/// The brand lockup: the wordmark in the export's `h-[104px] max-w-sm`
/// overflow-hidden box (`app.tsx:984` — the normalizeSvg's `width 100%` fit
/// reads as `BoxFit.contain`) over `The Ponder Engine` (`text-[14px]
/// tracking-[0.25em] font-nunito font-bold`, `app.tsx:985`).
class AuthBrand extends StatelessWidget {
  const AuthBrand({required this.gapBelow, super.key});

  /// The brand block's bottom margin: `mb-8` (login, app.tsx:983) /
  /// `mb-6` (register, app.tsx:1059).
  final double gapBelow;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: <Widget>[
        SizedBox(
          height: 104,
          width: double.infinity,
          child: SvgPicture.asset(kPondrLogoAsset, fit: BoxFit.contain),
        ),
        const SizedBox(height: 8), // mt-2
        const Text(
          'THE PONDER ENGINE',
          style: TextStyle(
            fontFamily: PondrTokens.fontDisplay,
            fontWeight: FontWeight.w700,
            fontSize: 14,
            letterSpacing: 3.5, // tracking-[0.25em] at 14px
            color: Color(0xBFFFFFFF), // text-white/75
          ),
        ),
        SizedBox(height: gapBelow),
      ],
    );
  }
}

/// The auth heading: `text-center text-white font-nunito font-bold text-xl`
/// (app.tsx:990,1067) above the form. [gapBelow] is the export's `mb-6` /
/// `mb-5`.
class AuthHeading extends StatelessWidget {
  const AuthHeading({required this.title, required this.gapBelow, super.key});

  final String title;
  final double gapBelow;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Text(
          title,
          textAlign: TextAlign.center,
          style: const TextStyle(
            fontFamily: PondrTokens.fontDisplay,
            fontWeight: FontWeight.w700,
            fontSize: 20, // text-xl
            color: Colors.white,
          ),
        ),
        SizedBox(height: gapBelow),
      ],
    );
  }
}

/// The export's label class — `text-xs font-bold text-white/80
/// tracking-widest uppercase mb-1.5` (app.tsx:947).
const TextStyle _kLabelStyle = TextStyle(
  fontFamily: PondrTokens.fontFamily,
  fontWeight: FontWeight.w700,
  fontSize: 12, // text-xs
  letterSpacing: 1.2, // tracking-widest (0.1em at 12px)
  color: Color(0xCCFFFFFF), // text-white/80
);

/// One auth field: the [AuthField] label + `inputCls` input. [fieldKey] keys
/// the [TextField]; [autofillHints] carries the export's `autoComplete`;
/// [onSubmitted] is the web form's Enter submit (`<form onSubmit>` — every
/// field's Enter runs the form's handler in the browser) — `next` for a
/// non-last field, `done` for the last, per focus order.
class AuthField extends StatelessWidget {
  const AuthField({
    required this.label,
    required this.hintText,
    required this.fieldKey,
    this.obscureText = false,
    this.suffixIcon,
    this.autofillHints,
    this.onSubmitted,
    this.lastField = false,
    super.key,
  });

  final String label;
  final String hintText;
  final Key fieldKey;
  final bool obscureText;
  final Widget? suffixIcon;
  final List<String>? autofillHints;

  /// The field's Enter handler (the `<form onSubmit>` semantics); null just
  /// moves the action's default.
  final ValueChanged<String>? onSubmitted;

  /// TRUE for a form's LAST field — its Enter action reads `done`, the
  /// others `next` (the export's focus order over the two input types).
  final bool lastField;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Text(label, style: _kLabelStyle),
        const SizedBox(height: 6), // mb-1.5
        TextField(
          key: fieldKey,
          autofillHints: autofillHints,
          obscureText: obscureText,
          onSubmitted: onSubmitted,
          textInputAction: lastField
              ? TextInputAction.done
              : TextInputAction.next,
          cursorColor: Colors.white,
          style: const TextStyle(
            fontFamily: PondrTokens.fontFamily,
            fontWeight: FontWeight.w500, // font-medium
            fontSize: 16,
            color: Colors.white,
          ),
          decoration: _fieldDecoration(hintText, suffixIcon),
        ),
      ],
    );
  }
}

/// The password field: the export's eye toggle in the suffix slot
/// (`app.tsx:1019-1024,1101-1106` — `absolute right-3.5`, the 15 px
/// lucide Eye/EyeOff at white/50). The mock boots HIDDEN.
class AuthPasswordField extends StatefulWidget {
  const AuthPasswordField({
    required this.label,
    required this.hintText,
    required this.fieldKey,
    required this.eyeKey,
    this.autofillHints,
    this.onSubmitted,
    this.lastField = false,
    super.key,
  });

  final String label;
  final String hintText;
  final Key fieldKey;
  final Key eyeKey;
  final List<String>? autofillHints;
  final ValueChanged<String>? onSubmitted;
  final bool lastField;

  @override
  State<AuthPasswordField> createState() => _AuthPasswordFieldState();
}

class _AuthPasswordFieldState extends State<AuthPasswordField> {
  bool _shown = false;

  @override
  Widget build(BuildContext context) {
    return AuthField(
      label: widget.label,
      hintText: widget.hintText,
      fieldKey: widget.fieldKey,
      obscureText: !_shown,
      autofillHints: widget.autofillHints,
      onSubmitted: widget.onSubmitted,
      lastField: widget.lastField,
      suffixIcon: IconButton(
        key: widget.eyeKey,
        padding: const EdgeInsets.only(right: 14, left: 4), // right-3.5
        constraints: const BoxConstraints(minWidth: 30, minHeight: 30),
        iconSize: 15,
        color: const Color(0x80FFFFFF), // text-white/50
        icon: Icon(_shown ? Icons.visibility_off : Icons.visibility),
        onPressed: () => setState(() => _shown = !_shown),
      ),
    );
  }
}

/// The `w-full bg-primary py-3 rounded-xl active:scale-[0.98]` submit with
/// its `0 4px 24px` primary glow and the `hover:bg-primary/85` shift
/// (app.tsx:1027-1033,1122-1128).
class AuthSubmitButton extends StatefulWidget {
  const AuthSubmitButton({
    required this.label,
    required this.onPressed,
    super.key,
  });

  final String label;
  final VoidCallback onPressed;

  @override
  State<AuthSubmitButton> createState() => _AuthSubmitButtonState();
}

class _AuthSubmitButtonState extends State<AuthSubmitButton> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    return PressScale(
      pressedScale: 0.98, // active:scale-[0.98]
      child: DecoratedBox(
        decoration: const BoxDecoration(
          borderRadius: BorderRadius.all(Radius.circular(12)),
          boxShadow: <BoxShadow>[
            BoxShadow(
              color: Color(0x73888DDF), // rgba(136,141,223,0.45)
              offset: Offset(0, 4),
              blurRadius: 24,
            ),
          ],
        ),
        child: Material(
          color: _hover
              ? const Color(0xD9888DDF) // hover:bg-primary/85
              : PondrTokens.primary,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
          ),
          child: InkWell(
            customBorder: const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(Radius.circular(12)),
            ),
            splashColor: Colors.transparent,
            highlightColor: Colors.transparent,
            hoverColor: Colors.transparent,
            onHover: (bool hovering) => setState(() => _hover = hovering),
            onTap: widget.onPressed,
            child: SizedBox(
              height: 48, // py-3 around the 16px/1.5 line
              child: Center(
                child: Text(
                  widget.label,
                  style: const TextStyle(
                    fontFamily: PondrTokens.fontDisplay,
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// The card's footer — `text-white/55` premise + the primary `font-semibold`
/// link to the sibling view (`app.tsx:1035-1041,1132-1138`); the hover
/// shift `hover:text-primary/80` rides the text color.
class AuthFooter extends StatelessWidget {
  const AuthFooter({
    required this.premise,
    required this.link,
    required this.linkKey,
    required this.onLink,
    required this.gapAbove,
    super.key,
  });

  final String premise;
  final String link;
  final Key linkKey;
  final VoidCallback onLink;

  /// The export's `mt-6` (login) / `mt-5` (register).
  final double gapAbove;

  @override
  Widget build(BuildContext context) {
    // The export's footer is one inline <p> flow (premise + link) — a Wrap,
    // not a Row: it can't overflow, it wraps like the text would.
    return Padding(
      padding: EdgeInsets.only(top: gapAbove),
      child: Wrap(
        alignment: WrapAlignment.center,
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 2,
        children: <Widget>[
          Text(
            premise,
            style: const TextStyle(
              fontFamily: PondrTokens.fontFamily,
              fontSize: 14,
              color: Color(0x8CFFFFFF), // text-white/55
            ),
          ),
          TextButton(
            key: linkKey,
            style: const ButtonStyle(
              padding: WidgetStatePropertyAll<EdgeInsets>(EdgeInsets.zero),
              minimumSize: WidgetStatePropertyAll<Size>(Size.zero),
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
            onPressed: onLink,
            child: _FooterLink(label: link),
          ),
        ],
      ),
    );
  }
}

/// The link's text with the `text-primary hover:text-primary/80` shift
/// (app.tsx:1039,1136).
class _FooterLink extends StatefulWidget {
  const _FooterLink({required this.label});

  final String label;

  @override
  State<_FooterLink> createState() => _FooterLinkState();
}

class _FooterLinkState extends State<_FooterLink> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (PointerEvent _) => setState(() => _hover = true),
      onExit: (PointerEvent _) => setState(() => _hover = false),
      child: Text(
        widget.label,
        style: TextStyle(
          fontFamily: PondrTokens.fontFamily,
          fontWeight: FontWeight.w600, // font-semibold
          fontSize: 14,
          color: _hover
              ? const Color(0xCC888DDF) // hover:text-primary/80
              : PondrTokens.primary,
        ),
      ),
    );
  }
}
