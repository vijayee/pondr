# The Pondr Flutter App — Spec

**Date:** 2026-10-04
**Status:** owner-agreed shape settled in session 2026-10-04 (conversion of the Figma-Make
React export, `Pondr.zip`, into a Flutter app, **1:1 with the mockup's functionality
INCLUDING its animations**; the interface-based data layer with mock implementations only
in v1; the agent-runtime/engine bindings are LATER layers behind the same interfaces).
**Home:** `pondr/flutter/` (ONE codebase: desktop targets + mobile-capable; platform dirs are
Flutter's own — linux/ android/ ios/ folder scaffolds). **Targets: Linux desktop first;
mobile stays open on the same code.**
**Layers (from the design thread, msg 4218/4220):** Pondr = the application layer
(the semantic/"artificial subconscious" engine lives here in `src/`, the UI lives in
`flutter/`); the agent runtime (SecretAgent) is a LIBRARY Pondr embeds — the app consumes it
through its client API later; WaveDB is the shared floor. Hard boundaries: `flutter/` imports
NO Python; the engine's Python imports NO Dart.

## Decisions carried in (settled with the owner, 2026-10-04)

| Decision | Value | Authority |
|---|---|---|
| Where it lives | `pondr/flutter/` (one codebase; desktop first, mobile open) | owner |
| Responsiveness | HARD requirement: every view adaptive — the 3-pane desktop shape ≥ ~900 dp, drawer/sheet idiom below (the mockup's own patterns, breakpoint per view) | owner |
| State + routing | Riverpod + go_router | owner |
| v1 data layer | interface-per-view-needs (`ChatService`, `SessionsService`, `SettingsService`, `SubconsciousService`) + MOCK implementations seeded from the mockup's data; the later bindings = REAL implementations of the SAME interfaces, swapped at ONE wiring file | owner |
| Runtime binding (later) | `dart:ffi` wrapping SecretAgent's `sa_client` — default when the binding slice lands (one client binary-truth; the payload-lifetime machinery reusable). A native-Dart CBOR client stays the alternative | owner |
| Subconscious binding (later) | the ENGINE's serving surface (the semantic layer/episodes render) — decided when the engine grows one | owner |
| Animations | **FIRST-CLASS**: every mockup motion ported (the list below); a view is not done until its animations behave | owner |
| Engine in v1 | NONE (mock data only; no Python serving, no agent wire) | owner |

## The views + their behaviors (1:1 with `Pondr.zip/src/app/App.tsx`)

1. **login / register** — mock auth flows (the mockup's validation + error shapes); a switch/entrance transition between them; mock success → chat.
2. **chat** — the sessions sidebar (grouped by date — the mock's `groupSessions` idiom; collapsible rail; the + / new-session flow); the typing indicator (its dot pulse); markdown message bubbles (assistant/user + the attachments' file icons); the attachment `+`-menu (a mock picker sheet — menu items animate in per the mockup's `y:4 → 0` entrances, 0.18 s class); the model selector dropdown; suggestion chips; the send flow (mock reply from the `AI_POOL` bank after a typing delay); message entrance/exit (`opacity+y` transitions; the AnimatePresence idiom → `AnimatedSwitcher`/custom tweens).
3. **subconscious** — the full-screen overlay: a force-directed graph — the mock's EXACT parameters ported (`CHARGE 2400, SPRING 0.07, REST 160, GRAVITY 0.025, DAMPING 0.82, alpha decay *= 0.988`, seeded circular-placement + random radius), `CustomPainter` + a `Ticker` driving the tick until alpha-decay settles; node tap → the detail card (its entrance animation); the cluster colors/legend; the open/close overlay animation (`AnimatePresence` idiom ≈ scale/fade).
4. **settings** — providers (add/edit/enable + per-provider models; the pencil/chevron expand animations 0.18 s), appearance/security/info sections (mock persistence through the settings service).

**The animation pin list** (a view's animations verified in its tests/integration run — the
de-wonk bar): view-level transitions; the sidebar open/close + the rail collapse; the
subconscious overlay + its graph's settle; the typing indicator; message entrance/exit; the
attachment menu; the suggestion chips; the settings rows' expand/collapse + toggles;
button/tap responsivity (the mock's hover/press shapes → Material states, feel 1:1).

## The data layer (v1 = mocks)

```dart
abstract class ChatService { send(text) -> Stream<ChatEvent>; /* typing, delta, done */ }
abstract class SessionsService { list() -> List<ChatSession>; create(); ... }
abstract class SettingsService { providers/models get-set, appearance ... }
abstract class SubconsciousService { graph() -> (nodes, edges); nodeDetail(id); }
```
MOCK implementations seed from the export's data (SEED_SESSIONS / BASE_NODES+BASE_EDGES /
AI_POOL / SUGGESTIONS / CLUSTER_COLORS). ONE central binding file swaps an implementation per
interface — the later runtime (FFI of `sa_client`) and engine bindings NEVER touch the views.

## Theming

The mockup's design tokens (palette, radii, typography, the iridescent logo SVGs as assets —
rendered via the same normalization handling) extracted into a Dart token file; a dark-first
theme matching the export; visual parity is the review's test.

## Testing

- `flutter test`: widget tests per view (login, chat incl. the responsive shapes at two sizes,
  settings, subconscious at rest + after settle); the force-sim as a unit test (the tick's
  determinism: fixed seed + bounded settle); the services' mock contracts; the animation
  presence asserted (the de-wonk bar above is testable: e.g. the typing indicator's visibility
  through its lifecycle).
- The engine's Python suite stays untouched; `pondr/CLAUDE.md`'s de-wonk bar applies to the
  Flutter work too ("use the de-wonk skill before completing an implementation").

## Out of scope / recorded follow-ons

- The REAL bindings: the runtime's client-API wire (sa_client FFI) — a later slice here;
  the engine's serving surface — when Pondr grows one.
- Voice (the engine's SISO/fade stack), the community/ecosystem views, real auth/identity.
- macOS/Windows targets: buildable, untested (the runtime's Windows stance).