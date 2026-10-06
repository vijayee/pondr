/// The subconscious view — the mock's `SubconsciousView` 1:1
/// (`mockup_reference/app.tsx:382-666`): the header row (514-535), the
/// ambient glows (507-511), the legend (538-546), the force-directed canvas
/// (548-613 — the painting runs through [GraphPainter], the sim lives in
/// `force_sim.dart`), and the node detail card (616-663).
///
/// Interactions pinned from the reference: wheel zoom (`onWheel`, 470-475 —
/// 1.12/0.89 per step, clamped 0.25..4), drag pan (477-490), the canvas
/// click → deselect and the node click → select/toggle (552,593), Escape →
/// close (457-461) and the header X → the chat view (529-533). The node's
/// settled positions are sim-space; the canvas transform —
/// `translate(cx + pan) scale(scale)` about the canvas centre (571, 495-496)
/// — is applied here for the hit-test and in the painter for the pixels.
library;

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../data/bindings.dart';
import '../../data/models.dart';
import '../chat/common.dart' show SparkAvatar;
import '../../theme/tokens.dart' show PondrTokens;
import 'force_sim.dart';

/// Test keys — the close button, the legend, the detail card, the canvas.
const Key scCloseKey = Key('subconscious.close');
const Key scLegendKey = Key('subconscious.legend');
const Key scCardKey = Key('subconscious.card');
const Key scCanvasKey = Key('subconscious.canvas');

class SubconsciousView extends ConsumerStatefulWidget {
  const SubconsciousView({super.key});

  @override
  SubconsciousState createState() => SubconsciousState();
}

/// The view's state — public so the tests drive the exact tap path
/// ([nodeAtScreen] / [globalFor] / the live [sim]).
class SubconsciousState extends ConsumerState<SubconsciousView>
    with SingleTickerProviderStateMixin {
  late final ForceSim sim;
  late final Ticker _ticker;
  late final List<SimEdge> _edges;
  final Map<String, SimNode> _nodeById = <String, SimNode>{};
  final GlobalKey _canvasKey = GlobalKey();

  /// The export's `selectedNode` state (app.tsx:394) — by id.
  String? _selectedId;

  /// The export's `pan` state (app.tsx:465).
  Offset _pan = Offset.zero;

  /// The export's `scale` state (app.tsx:466) — boots 1.
  double _zoom = 1;

  @override
  void initState() {
    super.initState();
    final (List<SimNode> nodes, List<SimEdge> edges) =
        ref.read(subconsciousProvider).graph();
    _edges = edges;
    _nodeById.clear();
    for (final SimNode node in nodes) {
      _nodeById[node.id] = node;
    }
    sim = ForceSim(nodes: nodes, edges: edges); // the export's Math.random()
    _ticker = createTicker(_onFrame);
    _ticker.start();
  }

  @override
  void dispose() {
    _ticker.dispose();
    sim.dispose();
    super.dispose();
  }

  /// The mock's raf loop (app.tsx:404-454): one [ForceSim.tick] per frame;
  /// the settle check's true return stops the loop — the export's raf
  /// chain dies there (app.tsx:414) and the layout freezes.
  void _onFrame(Duration elapsed) {
    if (sim.tick()) {
      _ticker.stop();
    }
  }

  /// The mock's onWheel (app.tsx:470-475): ×1.12 wheel-up / ×0.89
  /// wheel-down, clamped 0.25..4.
  void _onScroll(PointerSignalEvent event) {
    if (event is! PointerScrollEvent) {
      return;
    }
    final double dy = event.scrollDelta.dy;
    final double next = clampDouble(_zoom * (dy < 0 ? 1.12 : 0.89), 0.25, 4);
    if (next != _zoom) {
      setState(() => _zoom = next);
    }
  }

  /// The mock's drag pan (app.tsx:477-490): raw screen deltas, unscaled.
  void _onPan(DragUpdateDetails details) {
    setState(() => _pan += details.delta);
  }

  /// The mock's two click paths merged (app.tsx:552,593): a node hit —
  /// the svg's topmost disc — toggles it; the empty canvas clears. A tap on
  /// the ALREADY selected node reads the same as the mock's `sel ? null : n`.
  void _onTapUp(TapUpDetails details) {
    final SimNode? hit = nodeAtScreen(details.localPosition);
    setState(
      () => _selectedId =
          (hit == null || hit.id == _selectedId) ? null : hit.id,
    );
  }

  /// The mock's Escape handler (app.tsx:457-461).
  void _onKey(KeyEvent event) {
    if (event is KeyDownEvent &&
        event.logicalKey == LogicalKeyboardKey.escape) {
      context.go('/chat');
    }
  }

  /// The canvas [RenderBox] — shared by the hit-test and the tests'
  /// inverse mapping.
  RenderBox? get canvasRenderBox =>
      _canvasKey.currentContext?.findRenderObject() as RenderBox?;

  /// The mock's inverse viewport transform (app.tsx:571): a screen point
  /// (canvas-local) ↦ sim space, then the disc hit-test. The tap flows
  /// through exactly this.
  SimNode? nodeAtScreen(Offset screen) {
    final RenderBox? box = canvasRenderBox;
    if (box == null) {
      return null;
    }
    return sim.nodeAt((screen - box.size.center(Offset.zero) - _pan) / _zoom);
  }

  /// A sim node's screen (GLOBAL) point through the same viewport transform
  /// — the tap's exact geometry, so the tests hit-test honestly.
  Offset? globalFor(String id) {
    final LiveNode? live = sim.liveOf(id);
    final RenderBox? box = canvasRenderBox;
    if (live == null || box == null) {
      return null;
    }
    return box.localToGlobal(
      Offset(live.x, live.y) * _zoom + box.size.center(Offset.zero) + _pan,
    );
  }

  /// The export's `connectedIds` (app.tsx:498-500): every endpoint of the
  /// selected node's edges (the node's own id included).
  Set<String> _connectedIds() {
    final String? selectedId = _selectedId;
    if (selectedId == null) {
      return const <String>{};
    }
    return <String>{
      for (final SimEdge edge in _edges)
        if (edge.source == selectedId || edge.target == selectedId) ...<String>[
          edge.source,
          edge.target,
        ],
    };
  }

  @override
  Widget build(BuildContext context) {
    return KeyboardListener(
      focusNode: _focus,
      autofocus: true,
      onKeyEvent: _onKey,
      child: Stack(
        children: <Widget>[
          // The ambient glows sit BEHIND the whole view — they breathe into
          // the header through its 0.85 bg the same way (app.tsx:503-511).
          const Positioned.fill(
            child: IgnorePointer(child: CustomPaint(painter: _AmbientGlows())),
          ),
          Column(
            children: <Widget>[
              _header(),
              Expanded(
                child: Stack(
                  clipBehavior: Clip.none,
                  children: <Widget>[
                    Positioned.fill(
                      child: ClipRect(
                        child: Listener(
                          onPointerSignal: _onScroll,
                          child: MouseRegion(
                            // cursor-grab / active:cursor-grabbing
                            // (app.tsx:549).
                            cursor: SystemMouseCursors.grab,
                            child: GestureDetector(
                              key: scCanvasKey,
                              behavior: HitTestBehavior.opaque,
                              onTapUp: _onTapUp,
                              onPanUpdate: _onPan,
                              child: CustomPaint(
                                key: _canvasKey,
                                size: Size.infinite,
                                painter: GraphPainter(
                                  sim: sim,
                                  selectedId: _selectedId,
                                  connectedIds: _connectedIds(),
                                  pan: _pan,
                                  zoom: _zoom,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                    _legend(),
                    ..._detailCard(),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  final FocusNode _focus = FocusNode();

  /// The header row (app.tsx:514-535): the spark mark, the title + the
  /// concepts-connections-domains stats line, the (wide-only) hint and the
  /// close X.
  Widget _header() {
    final String stats =
        '${sim.live.length} concepts · ${_edges.length} connections · '
        '${clusterColors.keys.length} domains';
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xD907061A), // rgba(7,6,26,0.85)
        border: Border(
          bottom: BorderSide(color: Color(0x24888DDF)), // rgba(...,0.14)
        ),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      child: LayoutBuilder(
        builder: (BuildContext context, BoxConstraints constraints) {
          // The `hidden sm:block` hint (app.tsx:526) — the export gates on
          // the 640 px `sm` window width; this header sits in the body slot
          // whose width IS the window's under 900 (the narrow shell's drawer
          // overlays, it never squeezes).
          final Widget? hint = constraints.maxWidth >= 640
              ? const Text(
                  'Scroll to zoom · Drag to pan · Click a node to explore',
                  style: TextStyle(
                    fontFamily: PondrTokens.fontFamily,
                    fontSize: 12, // text-xs
                    color: Color(0x8C888DDF), // rgba(136,141,223,0.55)
                  ),
                )
              : null;
          return Row(
            children: <Widget>[
              const SparkAvatar(size: 36, bgAlpha: 0), // w-9 h-9, bare mark
              const SizedBox(width: 12), // gap-3
              Flexible(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    const Text(
                      'The Subconscious',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontFamily: PondrTokens.fontDisplay, // font-nunito
                        fontWeight: FontWeight.w700,
                        fontSize: 16, // text-base
                        letterSpacing: 0.4, // tracking-wide
                        color: Colors.white,
                        height: 1.25, // leading-tight
                      ),
                    ),
                    Text(
                      stats,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontFamily: PondrTokens.fontFamily,
                        fontSize: 11, // text-[11px]
                        color: Color(0xFF888DDF),
                      ),
                    ),
                  ],
                ),
              ),
              if (hint != null)
                Padding(
                  padding: const EdgeInsets.only(left: 8),
                  child: hint,
                ),
              const SizedBox(width: 16), // gap-4
              IconButton(
                key: scCloseKey,
                onPressed: () => context.go('/chat'),
                constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                iconSize: 16,
                color: const Color(0xFF888DDF),
                hoverColor: const Color(0x1AFFFFFF), // hover:bg-white/10
                icon: const Icon(Icons.close),
              ),
            ],
          );
        },
      ),
    );
  }

  /// The legend (app.tsx:538-546): `top-16 right-4` of the full view —
  /// 64 px down puts its top flush at the ~64 px header's bottom, i.e. the
  /// body stack's top; 16 px off the right edge.
  Widget _legend() {
    final List<Widget> rows = <Widget>[];
    for (final MapEntry<String, String> entry in clusterColors.entries) {
      final Color color = Color(
        int.parse(entry.value.substring(1), radix: 16) | 0xFF000000,
      );
      if (rows.isNotEmpty) {
        rows.add(const SizedBox(height: 6)); // gap-1.5
      }
      rows.add(
        Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Container(
              width: 8, // w-2 h-2
              height: 8,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
                boxShadow: <BoxShadow>[
                  BoxShadow(
                    color: color,
                    // `0 0 5px ${color}` — full-alpha, 5 px feather
                    blurRadius: 5,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8), // gap-2
            Text(
              entry.key,
              style: const TextStyle(
                fontFamily: PondrTokens.fontFamily,
                fontSize: 12, // text-xs
                color: Color(0x8CFFFFFF), // text-white/55
              ),
            ),
          ],
        ),
      );
    }
    return Positioned(
      top: 0,
      right: 16, // right-4
      child: Container(
        key: scLegendKey,
        padding: const EdgeInsets.all(12), // p-3
        decoration: BoxDecoration(
          color: const Color(0xCC07061A), // rgba(7,6,26,0.8)
          border: Border.all(color: const Color(0x29888DDF)), // rgba(...,0.16)
          borderRadius: const BorderRadius.all(Radius.circular(12)), // rounded-xl
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: rows,
        ),
      ),
    );
  }

  /// The node info panel (app.tsx:616-663): the `bottom-5 left-5` card under
  /// the mock's AnimatePresence — the enter plays opacity 0→1 / y 12→0, the
  /// exit 0→8, both 0.18 s; a SWITCH to another node re-enters (the panel is
  /// keyed by node id).
  List<Widget> _detailCard() {
    final String? id = _selectedId;
    final SimNode? node = id == null ? null : _nodeById[id];
    if (node == null) {
      return const <Widget>[];
    }
    // The peers in edge order (the export's BASE_EDGES.filter, app.tsx:646).
    final List<SimNode> peers = <SimNode>[];
    for (final SimEdge edge in _edges) {
      final String peerId =
          edge.source == id ? edge.target : (edge.target == id ? edge.source : '');
      if (peerId.isEmpty || peers.any((SimNode p) => p.id == peerId)) {
        continue;
      }
      final SimNode? peer = _nodeById[peerId];
      if (peer != null) {
        peers.add(peer);
      }
    }
    return <Widget>[
      Positioned(
        left: 20, // left-5
        bottom: 20, // bottom-5
        child: AnimatedSwitcher(
          // The card's entrance/exit (app.tsx:616-624): opacity 0→1 with
          // y 12→0, out y 8, over 0.18 s. The duration-omitted framer
          // transition uses its default `ease` — cubic-bezier(0.25,0.1,
          // 0.25,1) — which reads as [Curves.easeInOut] (the near-fit of
          // Flutter's presets).
          duration: const Duration(milliseconds: 180),
          switchInCurve: Curves.easeInOut,
          switchOutCurve: Curves.easeInOut,
          transitionBuilder: (Widget child, Animation<double> anim) {
            // Enter rises 12 px, exit sinks 8 px (app.tsx:620-622).
            final double from =
                anim.status == AnimationStatus.reverse ? 8 : 12;
            final double t = Curves.easeInOut.transform(anim.value);
            return Opacity(
              opacity: t,
              child: Transform.translate(
                offset: Offset(0, (1 - t) * from),
                child: child,
              ),
            );
          },
          child: _NodeCard(
            key: ValueKey<String>(node.id),
            node: node,
            peers: peers,
          ),
        ),
      ),
    ];
  }
}

/// The card's body (app.tsx:624-661): `max-w-[300px] rounded-2xl p-4` on
/// `rgba(14,12,32,0.96)`, the node-coloured border/glow, the dot + label +
/// cluster header, the description, and the footer's connection count +
/// peer chips.
class _NodeCard extends StatelessWidget {
  const _NodeCard({required this.node, required this.peers, super.key});

  final SimNode node;

  /// The connected peers in edge order — the export's `BASE_EDGES.filter`
  /// then `BASE_NODES.find` (app.tsx:638-651), as nodes.
  final List<SimNode> peers;

  @override
  Widget build(BuildContext context) {
    final Color color = Color(
      int.parse(node.color.substring(1), radix: 16) | 0xFF000000,
    );
    return ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 300), // max-w-[300px]
      child: Container(
        key: scCardKey,
        padding: const EdgeInsets.all(16), // p-4
        decoration: BoxDecoration(
          color: const Color(0xF50E0C20), // rgba(14,12,32,0.96)
          border: Border.all(color: color.withValues(alpha: 0x55 / 255)),
          borderRadius: const BorderRadius.all(Radius.circular(16)),
          boxShadow: <BoxShadow>[
            const BoxShadow(
              color: Color(0x80000000), // rgba(0,0,0,0.5)
              offset: Offset(0, 8),
              blurRadius: 32,
            ),
            BoxShadow(
              color: color.withValues(alpha: 0x22 / 255), // `0 0 0 1px …22`
              spreadRadius: 1,
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Container(
                  width: 12, // w-3 h-3
                  height: 12,
                  margin: const EdgeInsets.only(top: 2), // mt-0.5
                  decoration: BoxDecoration(
                    color: color,
                    shape: BoxShape.circle,
                    boxShadow: <BoxShadow>[
                      BoxShadow(color: color, blurRadius: 10), // 0 0 10px
                    ],
                  ),
                ),
                const SizedBox(width: 10), // gap-2.5
                Flexible(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Text(
                        node.label,
                        style: const TextStyle(
                          fontFamily: PondrTokens.fontFamily,
                          fontWeight: FontWeight.w600, // font-semibold
                          fontSize: 14, // text-sm
                          color: Colors.white,
                          height: 1.25, // leading-tight
                        ),
                      ),
                      const SizedBox(height: 2), // mt-0.5
                      Text(
                        node.cluster.toUpperCase(), // the `uppercase` class
                        style: TextStyle(
                          fontFamily: PondrTokens.fontFamily,
                          fontWeight: FontWeight.w700,
                          fontSize: 10, // text-[10px]
                          letterSpacing: 1.0, // tracking-widest
                          color: color,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10), // mb-2.5
            Text(
              node.description,
              style: const TextStyle(
                fontFamily: PondrTokens.fontFamily,
                fontSize: 12, // text-xs
                height: 1.625, // leading-relaxed
                color: Color(0xB3ECE9FF), // rgba(236,233,255,0.7)
              ),
            ),
            const SizedBox(height: 12), // mt-3
            Container(
              decoration: const BoxDecoration(
                border: Border(
                  top: BorderSide(color: Color(0x12FFFFFF)), // white/7% …
                ),
              ),
              padding: const EdgeInsets.only(top: 10), // pt-2.5
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    '${peers.length} connection'
                    '${peers.length != 1 ? 's' : ''}',
                    style: const TextStyle(
                      fontFamily: PondrTokens.fontFamily,
                      fontSize: 10, // text-[10px]
                      color: Color(0x8C888DDF), // rgba(136,141,223,0.55)
                    ),
                  ),
                  const SizedBox(width: 16), // gap-4
                  Expanded(
                    child: Wrap(
                      spacing: 4, // gap-1
                      runSpacing: 4,
                      children: <Widget>[
                        for (final SimNode peer in peers)
                          _PeerChip(key: ValueKey<String>(peer.id), node: peer),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// The peer chip (app.tsx:646-658): the peer's 9 px label on its colour.
class _PeerChip extends StatelessWidget {
  const _PeerChip({required this.node, super.key});

  final SimNode node;

  @override
  Widget build(BuildContext context) {
    final Color color = Color(
      int.parse(node.color.substring(1), radix: 16) | 0xFF000000,
    );
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0x20 / 255),
        border: Border.all(color: color.withValues(alpha: 0x40 / 255)),
        borderRadius: const BorderRadius.all(Radius.circular(999)),
      ),
      child: Text(
        node.label,
        style: TextStyle(
          fontFamily: PondrTokens.fontFamily,
          fontSize: 9, // text-[9px]
          color: color,
        ),
      ),
    );
  }
}

/// The graph painting — the mock's svg subtree (app.tsx:553-612) as one
/// painter. Edges first (573-585), then nodes (588-610) — the export's z
/// order; every stroke/fill alpha is pinned to the reference's values.
class GraphPainter extends CustomPainter {
  GraphPainter({
    required this.sim,
    required this.selectedId,
    required this.connectedIds,
    required this.pan,
    required this.zoom,
  }) : super(repaint: sim);

  final ForceSim sim;

  /// The export's `selectedNode?.id`.
  final String? selectedId;

  /// The export's `connectedIds` — the selected node's edge endpoints.
  final Set<String> connectedIds;

  final Offset pan;
  final double zoom;

  Color _color(String hex) =>
      Color(int.parse(hex.substring(1), radix: 16) | 0xFF000000);

  @override
  bool shouldRepaint(GraphPainter oldDelegate) =>
      selectedId != oldDelegate.selectedId ||
      pan != oldDelegate.pan ||
      zoom != oldDelegate.zoom;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    // The one <g> transform (app.tsx:571): the canvas centre + pan, scaled.
    final Offset origin = size.center(Offset.zero) + pan;
    canvas.translate(origin.dx, origin.dy);
    canvas.scale(zoom);

    // ── Edges (app.tsx:573-585) ──────────────────────────────────────────
    final Paint edge = Paint()
      ..strokeWidth = 0.75 // the dim stroke's width
      ..style = PaintingStyle.stroke;
    for (final EdgeIndex pair in sim.edgeIndices) {
      final LiveNode a = sim.live[pair.si];
      final LiveNode b = sim.live[pair.ti];
      final bool lit = selectedId != null &&
          (a.node.id == selectedId || b.node.id == selectedId);
      edge.strokeWidth = lit ? 1.5 : 0.75;
      edge.color = lit
          ? _color(a.node.color).withValues(alpha: 0.8)
          : const Color(0x21888DDF); // rgba(136,141,223,0.13)
      canvas.drawLine(Offset(a.x, a.y), Offset(b.x, b.y), edge);
    }

    // ── Nodes (app.tsx:588-610) ──────────────────────────────────────────
    for (final LiveNode node in sim.live) {
      final bool sel = node.node.id == selectedId;
      final bool dim =
          selectedId != null && !sel && !connectedIds.contains(node.node.id);
      final Color color = _color(node.node.color);
      // The dimming is the whole <g>'s opacity (app.tsx:600).
      final double opacity = dim ? 0.15 : 1;

      // The selected halo (app.tsx:594-597): r + 10 ring at 0.45, the
      // sc-glow-lg blur reads as an outer feather + its crisp stroke.
      if (sel) {
        final Paint halo = Paint()
          ..color = color.withValues(alpha: 0.45 * opacity)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5
          ..maskFilter = const MaskFilter.blur(BlurStyle.outer, 8);
        canvas.drawCircle(Offset(node.x, node.y), node.node.r + 10, halo);
        canvas.drawCircle(
          Offset(node.x, node.y),
          node.node.r + 10,
          Paint()
            ..color = color.withValues(alpha: 0.45 * opacity)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.5,
        );
      }

      // The sc-glow-sm/lg body glow (app.tsx:601): a coloured outer
      // feather under the fill — the filter's merge, read cheap.
      canvas.drawCircle(
        Offset(node.x, node.y),
        node.node.r,
        Paint()
          ..color = color.withValues(alpha: 0.6 * opacity)
          ..maskFilter = MaskFilter.blur(BlurStyle.outer, sel ? 8 : 3),
      );

      // The gradient fill (app.tsx:563-568): the radial `sc-g-<id>` —
      // 40%/35% centre, r 65%, the colour fading to 35% at the rim. The
      // svg's dim is the whole circle's opacity (app.tsx:600) — fill,
      // stroke and glow dim as one 0.15 whole, so the stops carry it too.
      final Paint fill = Paint()
        ..shader = RadialGradient(
          center: const Alignment(-0.2, -0.3), // cx 40% cy 35%
          radius: 0.65,
          colors: <Color>[
            color.withValues(alpha: opacity),
            color.withValues(alpha: 0.35 * opacity),
          ],
          stops: const <double>[0, 1],
        ).createShader(
          Rect.fromCircle(
            center: Offset(node.x, node.y),
            radius: node.node.r,
          ),
        );
      canvas.drawCircle(Offset(node.x, node.y), node.node.r, fill);
      // The stroke (app.tsx:599): the colour, 2 px selected / 0.8 px.
      canvas.drawCircle(
        Offset(node.x, node.y),
        node.node.r,
        Paint()
          ..color = color.withValues(alpha: opacity)
          ..style = PaintingStyle.stroke
          ..strokeWidth = sel ? 2 : 0.8,
      );

      // The label (app.tsx:602-607): Inter 9.5, centered UNDER the node,
      // baseline at r + 13 — the svg's y is the alphabetic baseline.
      final TextPainter label = TextPainter(
        text: TextSpan(
          text: node.node.label,
          style: TextStyle(
            fontFamily: PondrTokens.fontFamily,
            fontSize: 9.5,
            height: 1.0,
            fontWeight: sel ? FontWeight.w600 : FontWeight.w400,
            color: dim
                ? const Color(0x1FFFFFFF) // rgba(255,255,255,0.12)
                : sel
                    ? Colors.white
                    : const Color(0xA6FFFFFF), // white 0.65
          ),
        ),
        textDirection: TextDirection.ltr,
        textAlign: TextAlign.center,
      )..layout();
      // The svg's y = r + 13 IS the alphabetic baseline (app.tsx:602) —
      // subtract the ascent to land this painter's box on it. The svg's
      // text x is the node's x minus the anchor-middle width (app.tsx:603)
      // — the painter's box origin is the TOP-LEFT, so shift by the ascent
      // there; without the node's (x, y) every label would pile at the
      // canvas centre (the running-app finding).
      final double ascent = label.computeDistanceToActualBaseline(
        TextBaseline.alphabetic,
      );
      label.paint(
        canvas,
        Offset(
          node.x - label.width / 2,
          node.y + node.node.r + 13 - ascent,
        ),
      );
    }
    canvas.restore();
  }
}

/// The three ambient glow discs (app.tsx:507-511) — the CSS `blur-[…px]`
/// discs read as radial feathers, the auth background's idiom
/// (`auth_background.dart`'s painter) at the mock's own geometry.
class _AmbientGlows extends CustomPainter {
  const _AmbientGlows();

  void _glow(
    Canvas canvas,
    Offset center,
    double radius,
    Color color,
  ) {
    canvas.drawCircle(
      center,
      radius,
      Paint()
        ..shader = RadialGradient(
          colors: <Color>[color, color.withValues(alpha: 0)],
        ).createShader(Rect.fromCircle(center: center, radius: radius)),
    );
  }

  @override
  void paint(Canvas canvas, Size size) {
    final double w = size.width;
    final double h = size.height;
    _glow(
      canvas,
      Offset(w * 0.15 + 300, h * 0.10 + 300), // w-600 h-600, top 10% left 15%
      300,
      const Color(0x12888DDF), // rgba(136,141,223,0.07)
    );
    _glow(
      canvas,
      Offset(w * 0.80 - 250, h * 0.90 - 250), // 500², bottom 10% right 20%
      250,
      const Color(0x0FC3ACDA), // rgba(195,172,218,0.06)
    );
    _glow(
      canvas,
      Offset(w * 0.65 - 150, h * 0.40 + 150), // 300², top 40% right 35%
      150,
      const Color(0x0AE0EFE4), // rgba(224,239,228,0.04)
    );
  }

  @override
  bool shouldRepaint(_AmbientGlows oldDelegate) => false;
}