# The Pondr Flutter App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Figma-Make React export (`Pondr.zip`'s `src/app/App.tsx`, 2,341 lines) into the Flutter app at `pondr/flutter/` — 1:1 with the mockup's functionality including its animations.

**Architecture:** Riverpod + go_router; the data layer = four abstract services + mock implementations behind ONE binding file; the design as a token file (the export's CSS variables → Dart); the subconscious graph = a faithful force-sim port behind a `CustomPainter`. Every view converts from the REFERENCE (the export committed into `pondr/flutter/mockup_reference/`), never from memory.

**Tech Stack:** Flutter (SDK at ~/flutter), Riverpod ^2, go_router ^12, flutter_markdown; `flutter analyze` + `flutter test` are each task's gate; a real run at the end.

**Spec:** `pondr/docs/flutter-app-spec.md` — READ FIRST. **Conversion rules:** (1) the reference file is truth — read the relevant App.tsx region BEFORE each view; (2) NO inline state in widgets (the skill's rule) — all state flows through Riverpod providers backed by the services; (3) `const` constructors wherever possible; (4) dispose every controller/ticker/focus; (5) no network/state in `build()`; (6) animations are part of each view's done bar.

---

### Task 0: Scaffold + the reference

**Files:**
- Create: `pondr/flutter/` (`flutter create` scaffold + pubspec deps)
- Create: `pondr/flutter/mockup_reference/app.tsx` + the mockup's assets (`src/imports/*` svg/png copied)
- Modify: `pondr/.gitignore` (Flutter's `.dart_tool`, `build/` etc. — mirror `flutter create`'s generated ignores into the ROOT ignore so both projects ignore cleanly)

- [ ] **Step 1:** `export PATH="$HOME/flutter/bin:$PATH" && flutter --version` (the SDK is present; note the version in the report). `flutter create --platforms=linux,android,ios --org com.vijayee pondr/flutter` — confirm the dirs exist.
- [ ] **Step 2:** `flutter pub add flutter_riverpod go_router flutter_markdown`
- [ ] **Step 3:** Copy the mockup source as the reference (the export lives at
  `/home/victor/Workspace/src/github.com/vijayee/SecretAgent/Pondr.zip` — the repo root; flatten
  the paths after extraction so the reference reads flat):
```bash
cd pondr/flutter && mkdir -p mockup_reference/styles
unzip -o ../../SecretAgent/Pondr.zip -d /tmp/pondr_ref \
    'src/app/App.tsx' 'src/imports/*' 'src/styles/*'
mv /tmp/pondr_ref/src/app/App.tsx mockup_reference/app.tsx
mv /tmp/pondr_ref/src/imports/* mockup_reference/imports/
mv /tmp/pondr_ref/src/styles/* mockup_reference/styles/
rm -rf /tmp/pondr_ref
```
- [ ] **Step 4:** Build the app once (`flutter build linux` or at least `flutter pub get` + `flutter analyze`) — the empty scaffold must analyze clean BEFORE any task adds code. Commit:

```bash
git add pondr/flutter pondr/.gitignore && git commit -m "flutter: scaffold pondr/flutter + the mockup reference (App.tsx + assets)"
```

---

### Task 1: The token + theme file

**Files:**
- Create: `pondr/flutter/lib/theme/tokens.dart`, `pondr/flutter/lib/theme/app_theme.dart`
- Modify: `pondr/flutter/lib/main.dart`

- [ ] **Step 1: Port the export's CSS variables** (read `mockup_reference/.../theme.css` — wait, the theme file lives in the export's `src/styles/theme.css`; copy that + globals.css into `mockup_reference/styles/` in Task 0's step too) into `tokens.dart` — a sealed token map, EXACT values:

```dart
// The export's CSS variables (src/styles/theme.css:1-45), verbatim values.
class PondrTokens {
  static const background = Color(0xFF0C0B1A);
  static const foreground = Color(0xFFECE9FF);
  static const card = Color(0xFF2B293B);
  static const cardForeground = Color(0xFFECE9FF);
  static const popover = Color(0xFF1A1632);
  static const primary = Color(0xFF888DDF);
  static const primaryForeground = Color(0xFF0C0B1A);
  static const secondary = Color(0xFF1C1833);
  static const secondaryForeground = Color(0xFFC4BDE8);
  static const muted = Color(0xFF1A1630);
  static const mutedForeground = Color(0xFF9B96C8);
  static const accent = Color(0xFFC3ACDA);
  static const accentForeground = Color(0xFF0C0B1A);
  static const destructive = Color(0xFFD4183D);
  static const border = Color(0x3F888DDF);   // rgba(136,141,223,0.25)
  static const inputBg = Color(0x14888DDF);  // rgba(136,141,223,0.08)
  static const sidebar = Color(0xFF1A1929);
  static const sidebarAccent = Color(0xFF2D2B45);
  static const sidebarBorder = Color(0x2E888DDF);
  static const ring = Color(0xFF888DDF);
  static const radius = 12.0;                // --radius 0.75rem
  static const fontFamily = 'Inter';
}
```
- [ ] **Step 2: The fonts** — the export loads Nunito + Inter from Google Fonts; for a desktop app bundle them: `flutter pub add google_fonts` (a runtime fetch; if OFFLINE determinism is wanted, download once into `fonts/` — decide: google_fonts dependency, documented); `app_theme.dart` builds `ThemeData.dark` from the tokens (colorScheme: primary/background/surface/error + extensions: the chart colors 1-5 in oklch — PORT their hex approximations: read the CSS's fallbacks; if the file only has oklch, convert oklch→hex NOW with a converter and PIN the values in a comment).
- [ ] **Step 3:** main.dart = `MaterialApp.router` with the theme; router (Task 2) initially landing on `/login`. `flutter analyze` clean; a widget test pinning the tokens' colors (golden-lite: a Container test asserting `PondrTokens.background` paints).
- [ ] **Step 4: Commit** `flutter: the token + theme file — the export's design language verbatim`

---

### Task 2: The data layer — models, services, mocks, binding

**Files:**
- Create: `pondr/flutter/lib/data/models.dart`
- Create: `pondr/flutter/lib/data/services.dart` (the four abstract interfaces)
- Create: `pondr/flutter/lib/data/mock/mock_data.dart` + `mock/mock_services.dart`
- Create: `pondr/flutter/lib/data/bindings.dart` (the ONE swap file)

- [ ] **Step 1: The models** — port the export's types verbatim (the export's declarations at App.tsx:~44-90): `Message {id, role, content, timestamp, files?}`, `AttachedFile {id, name, type, size}`, `ChatSession {id, name, updatedAt, messages}`, `ProviderModel {id, modelId, label, enabled}`, `Provider {id, name, baseUrl, apiKey, enabled, models}`, `SimNode {id, label, cluster, color, r, description}`, `SimEdge {source, target}` — Dart records/classes with const constructors; `groupSessions` (the export's date-grouping logic — port the function + its edge rules: today/yesterday/earlier buckets).
- [ ] **Step 2: The seed data** — port SEED_SESSIONS (3 sessions with their full mockup message texts VERBATIM from the reference — they are the chat view's fixtures), BASE_NODES + BASE_EDGES + CLUSTER_COLORS, AI_POOL (the whole reply bank), SUGGESTIONS.
- [ ] **Step 3: The interfaces + mocks**:
```dart
abstract class SessionsService {
  List<ChatSession> list();             // newest-first, the mockup's order
  ChatSession? active(); void select(String id);
  ChatSession create(); void delete(String id);
  Stream<ChatsChanged> changes();       // provider-agnostic change events (renamed from the real one)
}
abstract class ChatService {
  Stream<ChatEvent> send(String text, List<AttachedFile> files);
  // ChatEvent: typing (duration ~1.2-3 s random from AI_POOL), delta, done(content)
}
abstract class SettingsService {
  String name; String username; String bio;
  List<Provider> providers();
  Provider save/create/delete per the export's CRUD; models add/edit/delete;
  bool notifMessages; bool notifSounds; Color accentColor;
}
abstract class SubconsciousService {
  (List<SimNode>, List<SimEdge>) graph();  // the export's BASE pair
}
```
Mock impls + the BINDING file:
```dart
// bindings.dart — THE ONE SWAP: every provider points at an implementation.
final sessionsProvider = Provider<SessionsService>((ref) => MockSessionsService());
final chatServiceProvider = Provider<ChatService>((ref) => MockChatService());
final settingsProvider = Provider<SettingsService>((ref) => MockSettingsService());
final subconsciousProvider = Provider<SubconsciousService>((ref) => MockSubconsciousService());
```
- [ ] **Step 4: Unit tests** — the services' contracts (the send stream's shape: typing→delta→done; groupSessions' buckets; the CRUD round-trips). Commit `flutter: the data layer — models, interfaces, mocks, the one binding file`

---

### Task 3: Router + the app shell + responsiveness

**Files:**
- Create: `pondr/flutter/lib/app/router.dart`, `pondr/flutter/lib/app/shell.dart`
- Modify: `pondr/flutter/lib/main.dart`

- [ ] **Step 1: Router** — go_router: `/login`, `/register`, `/chat`, `/settings`, `/subconscious` (the overlay as a route with a custom page-builder transition); auth redirect guard: not-logged-in → /login (the mock's auth state as a provider).
- [ ] **Step 2: The adaptive shell** — `LayoutBuilder`:
  - ≥ 900 dp: the mockup's chat layout = left rail (collapsed by default like the mock's `sidebarCollapsed=true`) + the sessions sidebar (~280 dp) + the chat canvas + the top bar's model picker/search/avatar.
  - < 900 dp: the sessions sidebar becomes a DRAWER (the mock's slide animation class: 0.24 s ease) + a top bar with the menu icon; the settings/subconscious as modal sheets/overlay (the mock's responsive patterns).
- [ ] **Step 3: A responsiveness test** — two widget tests (750 dp and 1100 dp) asserting the shape (drawer vs static sidebar presence). Commit `flutter: router + the adaptive shell (the 3-pane/drawer breakpoint)`

---

### Task 4: The chat view (the biggest)

**Files:**
- Create: `pondr/flutter/lib/views/chat/chat_page.dart`, `lib/views/chat/session_sidebar.dart`, `lib/views/chat/message_bubble.dart`, `lib/views/chat/typing_indicator.dart`, `lib/views/chat/composer.dart`, `lib/views/chat/attachment_menu.dart`, `lib/views/chat/model_picker.dart`
- Test: `pondr/flutter/test/chat_view_test.dart`

CONVERT FROM: the reference's chat region (read App.tsx's chat canvas + sidebar + composer sections). Checklist per region (the de-wonk list; each line = a behavior to make visible + its animation):
- [ ] The sidebar: sessions list (grouped), selection highlight (the mock's active shape on `sidebarAccent`), the hover/press states, the delete icon's hover reveal + its stopPropagation rule → tap-target separation, the NEW-chat button, the sidebar collapse toggle (the 160-dp rail + chevron rotate 0.18 s), the search field toggle.
- [ ] The message list: bubbles (assistant on card bg / user on primary-ish bg per the mock), markdown rendering in assistant bubbles, the file-attachment chips (the file-icon mapping + size formatting from the mock's helpers), the smooth auto-scroll on new content (the mock's `scrollIntoView` — a ScrollController anchored at bottom), the message ENTRANCE animation (0.18 s, y:4→0 + fade; exit via AnimatedSwitcher on the list).
- [ ] The typing indicator: the three-dot pulse (the mock's `motion.span` delays 0/0.15/0.3 — port as three `AnimationController`-offset dots or a pulse via Tween, period ~900 ms).
- [ ] The composer: the input shape (the mock's rounded input on `inputBg`), SEND button state, the attachment `+`-menu (menu items animate in/out per the mock's `y:4` entrances), the model picker (a dropdown reading the enabled providers' models; persists the selection via the settings service), the 4 suggestion chips (only when the session is empty — the mock's rule).
- [ ] The full send flow through ChatService: the message appends → typing → the reply streams/done → the scroll + the indicator's lifecycle; the empty-session prompt state.
- [ ] Tests: the send flow's event-shape (mock), the sidebar's selection/delete behavior, the typing indicator's presence during typing (THE animation pin), the entrance animation's duration pin, the two responsiveness sizes.

Commit `flutter: the chat view — 1:1 with the mockup's chat surface`

---

### Task 5: Login/register

**Files:**
- Create: `pondr/flutter/lib/views/auth/login_view.dart`, `lib/views/auth/register_view.dart`
- Test: `pondr/flutter/test/auth_view_test.dart`

- [ ] Convert from the reference's auth region: the two forms (username/pw + show/hide pw toggles + the register's confirm + validation error states per the mock's rules), the Pondr logo + iridescent ThoughtSpark assets (the SVG normalization from the mock — Flutter renders via the flutter_svg package: `flutter pub add flutter_svg`), the login↔register animated transition (the mock's crossfade/slide — the AnimatePresence pair), mock success → chat.
- [ ] Tests: validation shapes; the transition. Commit `flutter: auth views — the mock login/register flows`

---

### Task 6: Settings

**Files:**
- Create: `pondr/flutter/lib/views/settings/settings_view.dart` (+ `settings_providers.dart` section widget or a split — per the file-focus rule)
- Test: `pondr/flutter/test/settings_view_test.dart`

- [ ] Convert: the sections rail/tab (profile/security/appearance/info per the mock's list), profile fields, the NOTIFICATIONS toggles (the mock's Switch state via the settings service), the accent color picker (the swatch row — the mock's choices), the PROVIDERS CRUD with their EXPAND animations (the chevron rotate 0.18 s + the panel's `y:4→0` fades — the mock's exact motion), the saved-flash (the mock's `providerSaved` pulse), the add-model flow.
- [ ] Tests: the CRUD round-trip through the mock service; the expand animations' presence; the toggles' persistence. Commit `flutter: settings — providers CRUD + sections, animations 1:1`

---

### Task 7: The subconscious view (the sim)

**Files:**
- Create: `pondr/flutter/lib/views/subconscious/subconscious_view.dart`, `lib/views/subconscious/force_sim.dart`
- Test: `pondr/flutter/test/subconscious_test.dart`

- [ ] **The sim (the exact port of the mock's tick loop)** — `force_sim.dart`:
```dart
// The mock's constants (App.tsx SubconsciousView useEffect): verbatim.
const double kCharge = 2400;
const double kSpring = 0.07;
const double kRestLength = 160;
const double kGravity = 0.025;
const double kDamping = 0.82;
const double kAlphaDecay = 0.988;

class ForceSim extends ChangeNotifier {
  // tick(): repulsion (pairwise CHARGE/d2 springs, mirrored vx +/-),
  // edge springs (REST length, SPRING rate), the centering GRAVITY,
  // DAMPING on the velocities, x/y position integration, alpha decay.
  // Deterministic when seeded (the mock's circular placement + a RANDOM
  // radius: seed a fixed PRNG for the tests' determinism).
}
```
The view: a `Ticker`-driven `CustomPainter` (nodes = circles with the cluster colors + radius, labels; edges = lines) + the node tap hit-test → the detail card (its entrance animation) + the legend + the close. The overlay opens with the mock's scale/fade entrance.
- [ ] Tests: the sim's settle (alpha → ~0 after N ticks; no NaN); a node hit-test; the painter renders N known nodes. Commit `flutter: the subconscious view — the force sim, ported constants verbatim`

---

### Task 8: The final integration + de-wonk + run

- [ ] `flutter analyze` zero warnings; `flutter test` all green.
- [ ] **The de-wonk bar** (pondr/CLAUDE.md's rule): every view's animations list from the spec crossed off against the RUNNING app (`flutter run -d linux`): the transition shapes, the typing indicator, the graph settle, the entrance/exit motions — the manual visual run is the evidence (record what you saw in the report).
- [ ] Wire check: the binding file remains the ONLY place naming Mock* (grep: views import bindings.dart, never mock/*).
- [ ] Commit `flutter: the app complete — 1:1 with the mockup, the de-wonk pass`
- [ ] The final report: the run evidence + the mockup-parity notes (any intentional divergence listed for the owner's call) + the NEXT seams (the runtime FFI binding per the spec; the engine's subconscious data — the sim reads real episodes when the engine grows its serving surface).