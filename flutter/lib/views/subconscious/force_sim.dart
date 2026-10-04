/// The subconscious force sim — the mock's tick loop VERBATIM
/// (`mockup_reference/app.tsx:404-454`, the placement init at 383-388).
///
/// Coordinate system: sim space is centre-anchored — the gravity term
/// (`n.vx -= GRAVITY * n.x`, app.tsx:443) pulls toward 0,0 exactly; the
/// viewport transform (`translate(${cx},${cy}) scale(${scale})`, app.tsx:571)
/// is the PAINTER's job (`subconscious_view.dart`), the same split the svg's
/// one `<g>` transform makes. Ticks integrate per frame with no dt — the
/// mock's raf cadence is the contract (`x += n.vx * alpha`, app.tsx:444).
///
/// Settle: the loop's entry check `alpha < 0.002` (app.tsx:414) stops the
/// raf chain — [tick] returns true there and the caller stops scheduling
/// frames; the layout then freezes forever (positions never move again —
/// the mock's `nodesRef` is read-only after that, app.tsx:390).
library;

import 'dart:math';
import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../../data/models.dart';

/// `CHARGE` (app.tsx:405).
const double kCharge = 2400;

/// `SPRING` (app.tsx:406).
const double kSpring = 0.07;

/// `REST` (app.tsx:407).
const double kRestLength = 160;

/// `GRAVITY` (app.tsx:408) — the centering toward sim-space 0,0.
const double kGravity = 0.025;

/// `DAMPING` (app.tsx:409), applied after integration (app.tsx:445-446).
const double kDamping = 0.82;

/// The per-tick decay: `alphaRef.current *= 0.988` (app.tsx:415).
const double kAlphaDecay = 0.988;

/// The settle threshold (app.tsx:414) — a tick that sees alpha below it
/// performs no motion; with the 0.988 decay that lands at tick 516
/// (0.988^514 = 0.002019 ≥ 0.002, 0.988^515 = 0.001995 < 0.002).
const double kAlphaSettle = 0.002;

/// The placement init (app.tsx:386): `rad = 180 + Math.random() * 120` —
/// a radius in [180, 300) on an evenly spaced ring.
const double kPlacementMinRadius = 180;
const double kPlacementRadiusRange = 120;

/// One node's live sim state — the export's `{...n, x, y, vx, vy}` spreads
/// (app.tsx:383-388). The base [SimNode] rides along so the painter and the
/// hit-test never need a second lookup.
class LiveNode {
  LiveNode(this.node);

  final SimNode node;

  double x = 0;
  double y = 0;
  double vx = 0;
  double vy = 0;
}

/// One edge as the mock's precomputed index pair
/// (`BASE_NODES.findIndex(...)`, app.tsx:397-402).
typedef EdgeIndex = ({int si, int ti});

/// The force sim proper. A [ChangeNotifier] so a `CustomPainter(
/// painter: GraphPainter(... : super(repaint: sim)))` repaints per tick.
///
/// [random] exists for the tests' determinism — the export uses
/// `Math.random()` (app.tsx:386); the app leaves it null (unseeded).
class ForceSim extends ChangeNotifier {
  ForceSim({
    required List<SimNode> nodes,
    required List<SimEdge> edges,
    Random? random,
  }) {
    // The placement init (app.tsx:383-388): node i sits on the ring at
    // angle (i / n) * 2π, radius 180 .. 300, zero velocities.
    final Random prng = random ?? Random();
    for (int i = 0; i < nodes.length; i++) {
      final double angle = (i / nodes.length) * 2 * pi;
      final double radius =
          kPlacementMinRadius + prng.nextDouble() * kPlacementRadiusRange;
      final LiveNode node = LiveNode(nodes[i])
        ..x = cos(angle) * radius
        ..y = sin(angle) * radius;
      live.add(node);
    }
    _count = nodes.length;

    // The mock's edge index pairs, precomputed once (app.tsx:397-402);
    // dangling endpoints are dropped — with the seeded pair none are.
    for (int i = 0; i < nodes.length; i++) {
      _index[nodes[i].id] = i;
    }
    edgeCount = edges.length;
    for (final SimEdge edge in edges) {
      final int? si = _index[edge.source];
      final int? ti = _index[edge.target];
      if (si == null || ti == null) continue;
      edgeIndices.add((si: si, ti: ti));
    }
  }

  final List<LiveNode> live = <LiveNode>[];

  /// The mock's `edgeIndices` (app.tsx:397) — the tick's spring pairs.
  final List<EdgeIndex> edgeIndices = <EdgeIndex>[];

  /// The mock's stats line counts `BASE_EDGES.length` raw (app.tsx:521).
  int edgeCount = 0;

  final Map<String, int> _index = <String, int>{};
  int _count = 0;

  double _alpha = 1.0; // alphaRef (app.tsx:391)

  double get alpha => _alpha;

  /// TRUE once a settle check has passed — the layout is frozen.
  bool get settled => _alpha < kAlphaSettle;

  /// The mock's tick (app.tsx:411-450), verbatim:
  ///
  /// 1. The settle check — alpha below 0.002: NOTHING moves this call; the
  ///    caller stops the frame loop. (The export's `alphaRef` only ever
  ///    decays here, so the check reads one frame behind integration.)
  /// 2. `alpha *= 0.988` — THEN the forces use the new alpha.
  /// 3. The repulsion: pairwise CHARGE/d² (both terms on the guarded d²),
  ///    mirrored onto both velocities; the `|| 1` guard substitutes 1 for
  ///    d² = 0 only (JS truthiness — a NaN d² passes through, as here).
  /// 4. The springs: `SPRING * (d - REST)` pulling a↔b to REST 160 apart,
  ///    a += f and b -= f on the source→target direction.
  /// 5. The gravity toward 0,0, then `x += v * alpha`, then velocities
  ///    damp 0.82.
  bool tick() {
    if (_alpha < kAlphaSettle) {
      return true;
    }
    _alpha *= kAlphaDecay;

    for (int i = 0; i < _count; i++) {
      for (int j = i + 1; j < _count; j++) {
        final double dx = live[i].x - live[j].x;
        final double dy = live[i].y - live[j].y;
        double d2 = dx * dx + dy * dy;
        if (d2 == 0) {
          d2 = 1; // the `|| 1` guard (app.tsx:422)
        }
        final double d = sqrt(d2);
        final double f = kCharge / d2;
        final double fx = f * dx / d;
        final double fy = f * dy / d;
        live[i].vx += fx;
        live[i].vy += fy;
        live[j].vx -= fx;
        live[j].vy -= fy;
      }
    }

    for (final EdgeIndex pair in edgeIndices) {
      final LiveNode a = live[pair.si];
      final LiveNode b = live[pair.ti];
      final double dx = b.x - a.x;
      final double dy = b.y - a.y;
      double d = sqrt(dx * dx + dy * dy);
      if (d == 0) {
        d = 1; // the `|| 1` guard (app.tsx:435)
      }
      final double f = kSpring * (d - kRestLength);
      final double fx = f * dx / d;
      final double fy = f * dy / d;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    for (final LiveNode node in live) {
      node.vx -= kGravity * node.x;
      node.vy -= kGravity * node.y;
      node.x += node.vx * _alpha;
      node.y += node.vy * _alpha;
      node.vx *= kDamping;
      node.vy *= kDamping;
    }

    notifyListeners();
    return false;
  }

  /// The mock's node hit-test: the click target is the circle itself
  /// (app.tsx:592-601 — the `<g>` wraps the disc, no text pointer events),
  /// and the LAST painted node wins an overlap, so the scan runs reversed.
  SimNode? nodeAt(Offset point) {
    for (final LiveNode node in live.reversed) {
      final double dx = node.x - point.dx;
      final double dy = node.y - point.dy;
      if (dx * dx + dy * dy <= node.node.r * node.node.r) {
        return node.node;
      }
    }
    return null;
  }

  LiveNode? liveOf(String id) => _index.containsKey(id)
      ? live[_index[id]!]
      : null;
}