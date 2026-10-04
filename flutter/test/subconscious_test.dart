import 'dart:math';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

import 'package:pondr/app/router.dart';
import 'package:pondr/app/shell.dart';
import 'package:pondr/data/mock/mock_data.dart';
import 'package:pondr/data/models.dart';
import 'package:pondr/main.dart';
import 'package:pondr/views/subconscious/force_sim.dart';
import 'package:pondr/views/subconscious/subconscious_view.dart';

/// The subconscious surface's tests: the sim's verbatim-port contract
/// (the placement, two ticks' numbers, the determinism, the settle), the
/// disc hit-test, and the view's chrome (header, legend, detail card, the
/// routes' close paths).
void main() {
  SimNode node(String id, double r) => SimNode(
        id: id,
        label: id,
        cluster: 'Physics',
        color: '#888ddf',
        r: r,
        description: 'description of $id',
      );

  group('ForceSim (the mock\'s tick, app.tsx:404-454)', () {
    test('the placement contract: the even ring, radius 180..300, zero '
        'velocities (app.tsx:383-388)', () {
      final ForceSim sim = ForceSim(
        nodes: baseNodes,
        edges: baseEdges,
        random: Random(101),
      );
      expect(sim.live.length, baseNodes.length);
      expect(sim.alpha, 1.0); // alphaRef boots 1 (app.tsx:391)
      final double twoPi = 2 * pi;
      for (int i = 0; i < sim.live.length; i++) {
        final LiveNode n = sim.live[i];
        final double angle = atan2(n.y, n.x) % twoPi;
        expect(angle, closeTo((i / 25) * twoPi % twoPi, 1e-9));
        final double radius = sqrt(n.x * n.x + n.y * n.y);
        expect(radius, inInclusiveRange(180, 300));
        expect(n.vx, 0);
        expect(n.vy, 0);
      }
      expect(sim.edgeCount, baseEdges.length);
      expect(sim.edgeIndices.length, baseEdges.length); // none dangling
    });

    test('two ticks on a 2-node/1-edge graph — the mock\'s numbers, '
        'recomputed from app.tsx:418-446 (the FIRST tick integrates at '
        'alpha 1.0; the second at 0.988)', () {
      final ForceSim sim = ForceSim(
        nodes: <SimNode>[node('a', 10), node('b', 12)],
        edges: const <SimEdge>[SimEdge(source: 'a', target: 'b')],
        random: Random(0),
      );
      // Replay the placement's two random draws in the same order, rebuild
      // the placed coordinates, and derive the tick's expected numbers — a
      // second reading of the reference's formulas, so the port cannot
      // drift a term.
      final Random placement = Random(0);
      final double ra = 180 + placement.nextDouble() * 120;
      final double rb = 180 + placement.nextDouble() * 120;
      final double ax0 = cos(0) * ra;
      final double ay0 = sin(0) * ra;
      final double bx0 = cos(pi) * rb;
      final double by0 = sin(pi) * rb;
      expect(sim.live[0].x, ax0);
      expect(sim.live[0].y, ay0);
      expect(sim.live[1].x, bx0);
      expect(sim.live[1].y, by0);

      final double dxA = ax0 - bx0;
      final double dyA = ay0 - by0;
      final double d2A = dxA * dxA + dyA * dyA;
      final double dA = sqrt(d2A);
      final double rep = 2400 / d2A;
      final double repFx = rep * dxA / dA;
      final double repFy = rep * dyA / dA;
      final double dxB = bx0 - ax0;
      final double dyB = by0 - ay0;
      final double dB = sqrt(dxB * dxB + dyB * dyB);
      final double spring = 0.07 * (dB - 160);
      final double springFx = spring * dxB / dB;
      final double springFy = spring * dyB / dB;

      final double aVx = repFx + springFx - 0.025 * ax0;
      final double aVy = repFy + springFy - 0.025 * ay0;
      final double bVx = -repFx + -springFx - 0.025 * bx0;
      final double bVy = -repFy + -springFy - 0.025 * by0;

      // Tick 1 — the SNAPSHOT precedes the decay (app.tsx:413-415): the
      // forces integrate at 1.0 and `alphaRef *= 0.988` only arms the next
      // tick, so sim.alpha reads 0.988 AFTER this tick's integration.
      expect(sim.tick(), isFalse);
      expect(sim.alpha, 0.988);
      expect(sim.live[0].x, closeTo(ax0 + aVx, 1e-9));
      expect(sim.live[0].y, closeTo(ay0 + aVy, 1e-9));
      expect(sim.live[1].x, closeTo(bx0 + bVx, 1e-9));
      expect(sim.live[1].y, closeTo(by0 + bVy, 1e-9));
      // The velocities damp AFTER integrating (app.tsx:445-446).
      expect(sim.live[0].vx, closeTo(aVx * 0.82, 1e-12));
      expect(sim.live[0].vy, closeTo(aVy * 0.82, 1e-12));
      expect(sim.live[1].vx, closeTo(bVx * 0.82, 1e-12));
      expect(sim.live[1].vy, closeTo(bVy * 0.82, 1e-12));

      // Tick 2 — the snapshot now reads 0.988: the forces re-run on the
      // tick-1 positions and the damped tick-1 velocities, then integrate
      // at 0.988.
      final double aX1 = ax0 + aVx;
      final double aY1 = ay0 + aVy;
      final double bX1 = bx0 + bVx;
      final double bY1 = by0 + bVy;
      final double dxA2 = aX1 - bX1;
      final double dyA2 = aY1 - bY1;
      final double d2A2 = dxA2 * dxA2 + dyA2 * dyA2;
      final double dA2 = sqrt(d2A2);
      final double rep2 = 2400 / d2A2;
      final double repFx2 = rep2 * dxA2 / dA2;
      final double repFy2 = rep2 * dyA2 / dA2;
      final double dxB2 = bX1 - aX1;
      final double dyB2 = bY1 - aY1;
      final double dB2 = sqrt(dxB2 * dxB2 + dyB2 * dyB2);
      final double spring2 = 0.07 * (dB2 - 160);
      final double springFx2 = spring2 * dxB2 / dB2;
      final double springFy2 = spring2 * dyB2 / dB2;

      final double aVx2 = aVx * 0.82 + repFx2 + springFx2 - 0.025 * aX1;
      final double aVy2 = aVy * 0.82 + repFy2 + springFy2 - 0.025 * aY1;
      final double bVx2 = bVx * 0.82 + -repFx2 + -springFx2 - 0.025 * bX1;
      final double bVy2 = bVy * 0.82 + -repFy2 + -springFy2 - 0.025 * bY1;

      expect(sim.tick(), isFalse);
      expect(sim.alpha, closeTo(0.988 * 0.988, 1e-12));
      expect(sim.live[0].x, closeTo(aX1 + aVx2 * 0.988, 1e-9));
      expect(sim.live[0].y, closeTo(aY1 + aVy2 * 0.988, 1e-9));
      expect(sim.live[1].x, closeTo(bX1 + bVx2 * 0.988, 1e-9));
      expect(sim.live[1].y, closeTo(bY1 + bVy2 * 0.988, 1e-9));
      expect(sim.live[0].vx, closeTo(aVx2 * 0.82, 1e-12));
      expect(sim.live[0].vy, closeTo(aVy2 * 0.82, 1e-12));
      expect(sim.live[1].vx, closeTo(bVx2 * 0.82, 1e-12));
      expect(sim.live[1].vy, closeTo(bVy2 * 0.82, 1e-12));
    });

    test('a seeded sim is deterministic — two runs, identical positions '
        'after 40 ticks', () {
      final ForceSim a = ForceSim(
        nodes: baseNodes,
        edges: baseEdges,
        random: Random(7),
      );
      final ForceSim b = ForceSim(
        nodes: baseNodes,
        edges: baseEdges,
        random: Random(7),
      );
      for (int i = 0; i < 40; i++) {
        a.tick();
        b.tick();
      }
      for (int i = 0; i < a.live.length; i++) {
        expect(a.live[i].x, b.live[i].x);
        expect(a.live[i].y, b.live[i].y);
        expect(a.live[i].vx, b.live[i].vx);
        expect(a.live[i].vy, b.live[i].vy);
      }
    });

    test('the settle: 515 moving ticks then the frozen frame; no NaN, no '
        'overlap (app.tsx:414)', () {
      final ForceSim sim = ForceSim(
        nodes: baseNodes,
        edges: baseEdges,
        random: Random(7),
      );
      int moving = 0;
      while (!sim.tick()) {
        moving++;
      }
      // alpha = 0.988^n drops below 0.002 first at n = 515
      // (0.988^514 = 0.002019; 0.988^515 = 0.001995); the settle call is
      // the 516th and moves nothing.
      expect(moving, 515);
      expect(sim.alpha, closeTo(pow(0.988, 515).toDouble(), 1e-12));
      expect(sim.settled, isTrue);
      final LiveNode first = sim.live[0];
      final double x0 = first.x;
      sim.tick();
      expect(first.x, x0); // frozen
      for (final LiveNode n in sim.live) {
        expect(n.x.isFinite, isTrue);
        expect(n.y.isFinite, isTrue);
        expect(n.vx.isFinite, isTrue);
        expect(n.vy.isFinite, isTrue);
      }
      for (int i = 0; i < sim.live.length; i++) {
        for (int j = i + 1; j < sim.live.length; j++) {
          final double dx = sim.live[i].x - sim.live[j].x;
          final double dy = sim.live[i].y - sim.live[j].y;
          expect(dx * dx + dy * dy, greaterThan(0));
        }
      }
    });

    test('the disc hit-test: the last painted node wins an overlap '
        '(app.tsx:588-601)', () {
      final ForceSim sim = ForceSim(
        nodes: <SimNode>[node('a', 10), node('b', 12), node('c', 14)],
        edges: const <SimEdge>[],
        random: Random(5),
      );
      expect(sim.nodeAt(Offset(10000, 10000)), isNull);
      final LiveNode a = sim.live[0];
      expect(sim.nodeAt(Offset(a.x, a.y)), same(a.node));
      expect(
        sim.nodeAt(Offset(a.x + a.node.r - 1, a.y)),
        same(a.node),
      ); // inside the disc
      // Overlapping discs: the topmost (last painted) answers.
      sim.live[1]
        ..x = a.x
        ..y = a.y;
      sim.live[2]
        ..x = a.x
        ..y = a.y;
      expect(sim.nodeAt(Offset(a.x, a.y)), same(sim.live[2].node));
    });
  });

  group('GraphPainter', () {
    GraphPainter painter(ForceSim sim, String? selected) => GraphPainter(
          sim: sim,
          selectedId: selected,
          connectedIds: const <String>{},
          pan: const Offset(3, 0),
          zoom: 1.2,
        );

    test('shouldRepaint: the selection, pan and zoom fields, not the sim '
        '(the sim rides `super(repaint:)`)', () {
      final ForceSim sim = ForceSim(
        nodes: <SimNode>[node('a', 12), node('b', 12)],
        edges: const <SimEdge>[SimEdge(source: 'a', target: 'b')],
        random: Random(0),
      );
      expect(
        painter(sim, 'a').shouldRepaint(painter(sim, 'b')),
        isTrue,
      );
      expect(
        painter(sim, 'a').shouldRepaint(painter(sim, 'a')),
        isFalse,
      );
    });

    test('the known nodes paint without throwing — settled, selected and '
        'zoomed (a 25/29 layout)', () {
      final ForceSim sim = ForceSim(
        nodes: baseNodes,
        edges: baseEdges,
        random: Random(7),
      );
      while (!sim.tick()) {}
      final ui.PictureRecorder recorder = ui.PictureRecorder();
      final Canvas canvas = Canvas(
        recorder,
        const Rect.fromLTWH(0, 0, 1024, 736),
      );
      for (final String? selected in const <String?>[null, 'consciousness']) {
        GraphPainter(
          sim: sim,
          selectedId: selected,
          connectedIds: const <String>{'qualia', 'hard-problem'},
          pan: const Offset(10, 0),
          zoom: 1.4,
        ).paint(canvas, const Size(1024, 736));
      }
      recorder.endRecording();
    });
  });

  group('the view (pumped through the router)', () {
    Future<void> pumpSubconscious(WidgetTester tester, Size size) async {
      final AuthState auth = AuthState(startLoggedIn: true);
      final ProviderContainer container = ProviderContainer(
        overrides: [authStateProvider.overrideWith((_) => auth)],
      );
      addTearDown(container.dispose);
      await tester.binding.setSurfaceSize(size);
      addTearDown(() => tester.binding.setSurfaceSize(null));
      await tester.pumpWidget(
        UncontrolledProviderScope(
          container: container,
          child: const PondrApp(),
        ),
      );
      await tester.pumpAndSettle();
      // The authed boot lands on /chat; navigate to the view the way the
      // sidebar's affordance does.
      final BuildContext context = tester.element(
        find.byKey(const Key('chat.header')),
      );
      context.go('/subconscious');
    }

    testWidgets('the header, the stats line and the legend render; the sim '
        'settles under pump', (WidgetTester tester) async {
      await pumpSubconscious(tester, const Size(1100, 800));
      await tester.pumpAndSettle();

      expect(find.text('The Subconscious'), findsOneWidget);
      expect(
        find.text('25 concepts · 29 connections · 5 domains'),
        findsOneWidget,
      );
      expect(
        find.text('Scroll to zoom · Drag to pan · Click a node to explore'),
        findsOneWidget,
      );
      // The legend: the 5 clusters, names and dots (Object.entries order).
      expect(find.byKey(scLegendKey), findsOneWidget);
      for (final String cluster in clusterColors.keys) {
        expect(
          find.descendant(
            of: find.byKey(scLegendKey),
            matching: find.text(cluster),
          ),
          findsOneWidget,
        );
      }
      // The painter painted the full sim.
      final CustomPaint canvas = tester.widget<CustomPaint>(
        find.byWidgetPredicate(
          (Widget w) => w is CustomPaint && w.painter is GraphPainter,
        ),
      );
      expect(
        (canvas.painter! as GraphPainter).sim.live.length,
        baseNodes.length,
      );
    });

    testWidgets('the header close X and Escape both return to /chat',
        (WidgetTester tester) async {
      await pumpSubconscious(tester, const Size(1100, 800));
      // The route swaps install on the first pump and build on the next.
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));

      await tester.tap(find.byKey(scCloseKey));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
      expect(find.text('The Subconscious'), findsNothing);

      // The Escape wiring (app.tsx:457-461) does the same.
      final BuildContext context = tester.element(
        find.byKey(const Key('chat.header')),
      );
      context.go('/subconscious');
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.sendKeyEvent(LogicalKeyboardKey.escape);
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('chat.header')), findsOneWidget);
    });

    testWidgets('a node tap opens the detail card — verbatim content, the '
        'connection count, the peer chips; the re-tap clears it',
        (WidgetTester tester) async {
      await pumpSubconscious(tester, const Size(1100, 800));
      await tester.pumpAndSettle();

      final SubconsciousState state =
          tester.state<SubconsciousState>(find.byType(SubconsciousView));
      final SimNode target = baseNodes[5]; // consciousness
      final Offset point = state.globalFor(target.id)!;

      await tester.tapAt(point);
      await tester.pumpAndSettle();

      expect(find.byKey(scCardKey), findsOneWidget);
      expect(find.text(target.label), findsOneWidget);
      expect(find.text(target.cluster.toUpperCase()), findsOneWidget);
      expect(find.text(target.description), findsOneWidget); // VERBATIM
      final List<SimNode> peers = <SimNode>[
        for (final SimEdge e in baseEdges)
          if (e.source == target.id) _nodeFor(e.target)
          else if (e.target == target.id) _nodeFor(e.source),
      ];
      expect(
        find.text('${peers.length} connection${peers.length != 1 ? 's' : ''}'),
        findsOneWidget,
      );
      for (final SimNode peer in peers) {
        expect(
          find.descendant(
            of: find.byKey(scCardKey),
            matching: find.text(peer.label),
          ),
          findsOneWidget,
        );
      }
      // The painter's selection moved; the connected set rides with it.
      final GraphPainter painter =
          tester
              .widget<CustomPaint>(
                find.byWidgetPredicate(
                  (Widget w) => w is CustomPaint && w.painter is GraphPainter,
                ),
              )
              .painter! as GraphPainter;
      expect(painter.selectedId, target.id);
      expect(painter.connectedIds, contains(target.id));
      // The card rides the mock's 0.18 s entrance.
      expect(find.byType(AnimatedSwitcher), findsOneWidget);
      expect(
        tester.widget<AnimatedSwitcher>(
          find.byType(AnimatedSwitcher),
        ).duration,
        const Duration(milliseconds: 180),
      );

      // The re-tap toggles off (app.tsx:593 — `sel ? null : n`).
      await tester.tapAt(point);
      await tester.pumpAndSettle();
      expect(find.byKey(scCardKey), findsNothing);
    });

    testWidgets('the canvas tap clears the selection (app.tsx:552)',
        (WidgetTester tester) async {
      await pumpSubconscious(tester, const Size(1100, 800));
      await tester.pumpAndSettle();

      final SubconsciousState state =
          tester.state<SubconsciousState>(find.byType(SubconsciousView));
      await tester.tapAt(state.globalFor(baseNodes[5].id)!);
      await tester.pumpAndSettle();
      expect(find.byKey(scCardKey), findsOneWidget);

      // The canvas' top-left corner: node-free after the settle (the
      // cluster hugs the centre; the card sits bottom-left).
      final Rect canvas = tester.getRect(find.byKey(scCanvasKey));
      await tester.tapAt(canvas.topLeft + const Offset(8, 8));
      await tester.pumpAndSettle();
      expect(find.byKey(scCardKey), findsNothing);
    });

    testWidgets("the tap's hit-test maps through the canvas transform "
        '(translate cx,cy + pan; scale)', (WidgetTester tester) async {
      await pumpSubconscious(tester, const Size(1100, 800));
      await tester.pumpAndSettle();

      final SubconsciousState state =
          tester.state<SubconsciousState>(find.byType(SubconsciousView));
      final LiveNode live = state.sim.live[0];
      // The node's own sim point, seen through the canvas-centre transform.
      expect(
        state.nodeAtScreen(
          state.canvasRenderBox!.size.center(Offset.zero) +
              Offset(live.x, live.y),
        ),
        same(live.node),
      );
      // A point 10000 px out: the disc never reaches it.
      expect(
        state.nodeAtScreen(
          state.canvasRenderBox!.size.center(Offset.zero) +
              const Offset(10000, 0),
        ),
        isNull,
      );
    });

    testWidgets('the narrow shell shows no stand-in top bar under this view '
        '(the mock ships no menu affordance here, app.tsx:2112-2144)',
        (WidgetTester tester) async {
      await pumpSubconscious(tester, const Size(500, 800));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 100));
      expect(find.byKey(shellMenuButtonKey), findsNothing);
      expect(find.text('The Subconscious'), findsOneWidget);
      // The `hidden sm:block` hint is gone under 640.
      expect(
        find.text('Scroll to zoom · Drag to pan · Click a node to explore'),
        findsNothing,
      );
    });
  });
}

SimNode _nodeFor(String id) =>
    baseNodes.firstWhere((SimNode n) => n.id == id);