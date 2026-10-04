# The Task 8 Run's Evidence (on-screen, archived)

**Date:** 2026-10-04. **Commit:** `f5e6aa9` — `flutter: the app complete — 1:1 with the mockup, the de-wonk pass` (main).
**Gates at the run:** `flutter test` 119/119 (115 + 4 overflow pins), `flutter analyze` zero issues, zero TODOs in the `flutter/` tree, wire check clean (views import `bindings.dart` only; no `Mock*` names outside the data layer).

An adversarial review over the same HEAD also found the archival gap this file closes: all nine animation-pin items had implementations + tests and a 6-sample verbatim diff passed, but the on-screen run's evidence lived only in the session transcript. This file is that transcript excerpt, structured.

## The Method

The 4096×2560 Xorg host is unreachable from this shell (`LockedHint=yes`), so per the F5/F7 notes the run rode a **Xephyr nested server**: `flutter run -d linux` (the toolchain's own run, twice) at **1280×800**, real **XTEST input**, **ImageMagick frame captures**. Run 2 additionally verified the `--dart-define=pondr.bootChat=true` seam live: with the define the app boots straight past the guard onto the chat surface (fresh timestamps, no login card).

**Presenter-stall caveat:** frame presentation stalls on discrete single-frame changes after the sim settles (libEGL **DRI3** warnings; frames do present while an animation stream is live — the wheel-zoom trick flushed the card). Escape and route changes presented; individual tap-after-idle frames sometimes didn't — the on-screen table below notes that filter.

## On-Screen Verified

Captured frames. "captured after an animation stream" marks frames whose presenter flush came from a live animation stream, per the caveat above.

| Behavior | Evidence (captured frames) |
|---|---|
| Boot → login card | 1:1 at 1280×800: brand, spark, fields, hints, eye toggle, glowing submit |
| Auth field/link hovers | hover states on the inputs and the links |
| Login↔register swap | the `mode="wait"` exchange — register absent at the exit's midpoint, then entering |
| Login submit → chat | the chat entrance after mock success |
| Chat layout | rail/56, sessions sidebar, canvas, composer, the disclaimers |
| Send flow | bubble appends, header count 4→5→6, composer clears, send-disable, the three-dot **typing indicator**, reply + auto-scroll |
| Search | "fermi" → only The Fermi Paradox, the × affordance |
| Rail collapse + hover-expand | the collapse toggle and the hover-expand on the rail |
| Attachment menu | the two mock items, open/close |
| Settings: profile + toggles | sections rail, profile page 1:1, the **notification toggles' animated travel** |
| Settings: providers CRUD live | Add provider form, the "✓ Saved successfully." **saved flash**, Active badge, Add model |
| Provider → model picker | the Ollama provider flowing into chat's **model picker** ("Ollama · lla") + dropdown open/select; the picker inert-by-design at zero enabled providers |
| Subconscious: header + legend | stats "25 concepts · 20 connections · 1 domain", the legend |
| Subconscious: settle + zoom | live settle motion → frozen settled layout; **wheel zoom** (captured after an animation stream) |
| Subconscious: node tap | **halo, lit edges, dim-to-15%** (captured after an animation stream) |
| Subconscious: detail card | 1:1 — verbatim description, "5 connections", five peer chips (captured after an animation stream) |
| Subconscious: Escape | Escape → /chat presented |
| Narrow shape | 800×600 → the drawer shape |

## Test-Pinned Only

Not resolvable on-screen through the stalling presenter — the exact tweens/cadences stay covered by the widget tests instead:

- the entrance's exact 0.18 s tween values
- the typing-pulse period
- the drawer's 0.24 s slide
- the chevron rotates
- the new-provider form's enter
- the saved-flash pulse shape
- the suggestion chips' only-when-empty rule
- the attachment-chip icon mapping
- drag-pan + the hit-test through pan/zoom (`test/subconscious_test.dart:455+`)
- the header × close

## Parity Divergences for the Owner's Record

**The 6-sample verbatim diff** vs `flutter/mockup_reference/app.tsx` — all PASS:

| # | Sample | Reference → Implementation | Verdict |
|---|---|---|---|
| 1 | User-bubble glow | app.tsx:758 `rgb(136,141,223)`, radius `1rem 0.25rem 1rem 1rem`, shadow `0 4px 16px rgba(136,141,223,0.35)` → `message_bubble.dart:173-187` matches exactly (16/4/16/16 radii, `Color(0x59888DDF)`, blur 16, offset (0,4)) | PASS |
| 2 | Sidebar group labels | app.tsx:108-111 Today/Yesterday/Earlier → `models.dart:289-292` identical labels | PASS |
| 3 | Settings aside width | app.tsx:1252 `w-64` → `settings_view.dart:65` `width: 256` | PASS |
| 4 | Chat header badge | app.tsx:2141 emerald-500 dot + "Pondr AI" → `chat_page.dart:107` `Color(0xFF10B981)` + `'Pondr AI'` | PASS |
| 5 | Detail card peer chips | app.tsx:646-654 (peer color, `color40` border, `color20` bg, edge order) → `subconscious_view.dart:400-412` ports the edge-order filter | PASS |
| 6 | Composer disclaimer | app.tsx:2340 `Pondr may make mistakes — verify important information.` → `composer.dart:87` verbatim, em-dash included | PASS |

**The deliberate divergences:**

- Fonts: the export's Google-Fonts runtime fetch → bundled Inter/Nunito ttf (offline determinism; the fetch flaked).
- "Upload photo" chip: wired NOTHING, as in the mock (no onClick there) — deliberate no-op.
- The legal rows (settings About) are static chips — the mock's handler-less `<button>`s.
- Auth carries NO validation — the mock's bare `setView("chat")`; the router guard holds that boolean.
- oklch chart colors → pinned hex approximations; CSS `blur` glow discs → radial-gradient feathers; framer's default ease → `Curves.easeInOut` near-fit.
- The subconscious overlay = an opaque page inside the chat's ShellRoute (the export renders it opaque inside the main area); see-through question dissolved by construction.

## NEXT Seams

- `sa_client` FFI binding behind the same four interfaces (one swap in `lib/data/bindings.dart`).
- The engine's serving surface for `graph()` — the sim reads real episodes when the engine grows its serving surface.
- Real auth/identity.
- Play release signing if Android ships.